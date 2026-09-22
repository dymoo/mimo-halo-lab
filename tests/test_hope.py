"""Tests for mimo_halo.pruning.hope (stdlib-only core).

Focus: analytic pathological fixtures vs the exhaustive oracle, an independent
product-space/dense oracle cross-check across E=8..16, count-safe validation
and tamper rejection, merge/resume additivity, capability partitions,
relaxation labeling and missing-dependency behavior, and the real CLI wiring.
Not field copies of internals.
"""

import contextlib
import io
import itertools
import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from mimo_halo.pruning import hope  # noqa: E402
from mimo_halo.pruning import reap_adapter  # noqa: E402  (import-safe: no torch/REAP)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "hope.json"


def _run_cli(argv):
    """Run hope.main capturing stdout; returns (exit_code, parsed_or_text)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = hope.main(argv)
    text = buffer.getvalue()
    try:
        return code, json.loads(text)
    except json.JSONDecodeError:
        return code, text


def _sample_stats() -> hope.HopeStats:
    """Two hand-checkable routed rows over 3 experts.

    row 1: experts [0,1], gates [0.25,0.75], norms [2.4] -> a = [0.5, 3.0]
    row 2: experts [1,2], gates [0.5,0.5],   norms [2,2] -> a = [1.0, 1.0]
    """
    stats = hope.HopeStats(3, 2)
    stats.add_batch([[0, 1]], [[0.25, 0.75]], [[2.0, 4.0]])
    stats.add_batch([[1, 2]], [[0.5, 0.5]], [[2.0, 2.0]])
    return stats


def _random_symmetric_f(e: int, rng: random.Random, high: int = 9):
    f = [[0] * e for _ in range(e)]
    for i in range(e):
        for j in range(i, e):
            value = rng.randint(0, high)
            f[i][j] = value
            f[j][i] = value
    return f


class StatsAccumulationTests(unittest.TestCase):
    def test_counts_and_derived_values_match_hand_computation(self):
        stats = _sample_stats()
        self.assertEqual(stats.total_rows, 2)
        self.assertEqual(stats.act_count, [1, 2, 1])
        self.assertEqual(stats.pair_count[0][0], 1)
        self.assertEqual(stats.pair_count[1][1], 2)
        self.assertEqual(stats.pair_count[2][2], 1)
        self.assertEqual(stats.pair_count[0][1], 1)
        self.assertEqual(stats.pair_count[1][2], 1)
        self.assertEqual(stats.pair_count[0][2], 0)
        self.assertEqual(stats.first_sum, [0.5, 4.0, 1.0])
        self.assertEqual(stats.gate_sum, [0.25, 1.25, 0.5])
        self.assertEqual(stats.norm_sum, [2.0, 6.0, 2.0])
        # First-order score = REAP formula: mean over ACTIVE rows of g*||f||.
        self.assertEqual(stats.first_order(), [0.5, 2.0, 1.0])
        self.assertEqual(stats.mean_gates(), [0.25, 0.625, 0.5])
        self.assertEqual(stats.mean_norms(), [2.0, 3.0, 2.0])
        self.assertEqual(stats.routing_frequency(), [0.5, 1.0, 0.5])
        # Conditional F (Eq. 3 / A.7): pair_sum / pair_count.
        f_matrix = stats.conditional_f()
        self.assertEqual(f_matrix[0][0], 0.25)  # E[a^2] on its single activation
        self.assertEqual(f_matrix[1][1], 5.0)  # (9 + 1) / 2
        self.assertEqual(f_matrix[2][2], 1.0)
        self.assertEqual(f_matrix[0][1], 1.5)  # 0.5 * 3.0 over one co-activation
        self.assertEqual(f_matrix[1][2], 1.0)  # 1.0 * 1.0
        self.assertEqual(f_matrix[0][2], 0.0)  # empty X_ij -> 0
        stats.validate()

    def test_capability_partition_rows_and_parent_bounded(self):
        stats = hope.HopeStats(3, 2)
        stats.add_batch([[0, 1]], [[0.5, 0.5]], [[1.0, 1.0]], capability="debugging")
        stats.add_batch([[1, 2]], [[0.5, 0.5]], [[1.0, 1.0]])
        self.assertEqual(sorted(stats.capabilities), ["debugging"])
        cap = stats.capabilities["debugging"]
        self.assertEqual(cap.total_rows, 1)
        self.assertEqual(cap.act_count, [1, 1, 0])
        self.assertEqual(stats.total_rows, 2)
        cap.validate()
        stats.validate()
        # An unlabeled remainder is allowed: sum(cap rows) <= total rows.
        self.assertLessEqual(
            sum(c.total_rows for c in stats.capabilities.values()), stats.total_rows
        )

    def test_rejects_invalid_rows_and_values(self):
        cases = {
            "expert id out of range": ([[0, 9]], [[0.5, 0.5]], [[1.0, 1.0]]),
            "duplicate expert id": ([[1, 1]], [[0.5, 0.5]], [[1.0, 1.0]]),
            "bool id": ([[True, 1]], [[0.5, 0.5]], [[1.0, 1.0]]),
            "float id": ([[0.0, 1.0]], [[0.5, 0.5]], [[1.0, 1.0]]),
            "negative gate": ([[0, 1]], [[-0.1, 1.1]], [[1.0, 1.0]]),
            "gate above 1": ([[0, 1]], [[1.5, 0.5]], [[1.0, 1.0]]),
            "nan gate": ([[0, 1]], [[float("nan"), 0.5]], [[1.0, 1.0]]),
            "inf gate": ([[0, 1]], [[float("inf"), 0.5]], [[1.0, 1.0]]),
            "negative norm": ([[0, 1]], [[0.5, 0.5]], [[1.0, -2.0]]),
            "inf norm": ([[0, 1]], [[0.5, 0.5]], [[1.0, float("inf")]]),
            "width mismatch": ([[0]], [[0.5, 0.5]], [[1.0, 1.0]]),
            "row count mismatch": ([[0, 1]], [[0.5, 0.5]], [[1.0, 1.0], [1.0, 1.0]]),
            "non-numeric gate": ([[0, 1]], [["0.5", 0.5]], [[1.0, 1.0]]),
        }
        for label, (selected, gates, norms) in cases.items():
            with self.subTest(label=label):
                stats = hope.HopeStats(3, 2)
                with self.assertRaises(hope.HopeValidationError):
                    stats.add_batch(selected, gates, norms)
                self.assertEqual(stats.total_rows, 0)

    def test_capability_label_must_be_non_empty_string(self):
        stats = hope.HopeStats(2, 1)
        with self.assertRaises(hope.HopeValidationError):
            stats.add_batch([[0]], [[1.0]], [[1.0]], capability="")

    def test_empty_batch_is_a_noop(self):
        stats = hope.HopeStats(4)
        stats.add_batch([], [], [])
        self.assertEqual(stats.total_rows, 0)
        self.assertIsNone(stats.top_k)
        stats.validate()

    def test_rejected_batch_leaves_statistics_untouched(self):
        stats = hope.HopeStats(3, 2)
        stats.add_batch([[0, 1]], [[0.5, 0.5]], [[1.0, 1.0]])
        before = stats.to_json()
        # Second row of the batch is invalid -> the FIRST row must not leak in.
        with self.assertRaises(hope.HopeValidationError):
            stats.add_batch(
                [[0, 2], [1, 9]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[1.0, 1.0], [1.0, 1.0]],
            )
        self.assertEqual(stats.to_json(), before)
        with self.assertRaises(hope.HopeValidationError):
            stats.add_batch([[0, 2]], [[0.5, 0.5]], [[1.0, -1.0]])
        self.assertEqual(stats.to_json(), before)


class CountSafeValidationTests(unittest.TestCase):
    def test_roundtrip_is_byte_deterministic(self):
        stats = _sample_stats()
        stats.add_batch([[2, 0]], [[0.5, 0.5]], [[3.0, 5.0]], capability="planning")
        payload = stats.to_json()
        restored = hope.HopeStats.from_json(payload)
        self.assertEqual(restored.to_json(), payload)
        self.assertEqual(hope.HopeStats.from_json(payload).to_json(), payload)

    def test_tampered_documents_fail_closed(self):
        base = _sample_stats()

        def doc():
            return json.loads(base.to_json())

        tampers = {}

        def register(name, mutate):
            tampers[name] = mutate

        register("schema", lambda d: d.update(schema_version=2))
        register("kind", lambda d: d.update(kind="something-else"))
        register("rows_vs_act", lambda d: d["act_count"].__setitem__(0, 99))
        register("diag_vs_act", lambda d: d["pair_count"][1].__setitem__(1, 5))
        register(
            "asymmetric_count",
            lambda d: (
                d["pair_count"][0].__setitem__(1, 1),
                d["pair_count"][1].__setitem__(0, 0),
            ),
        )
        register(
            "frechet_low",
            lambda d: (
                d["pair_count"][0].__setitem__(1, 0),
                d["pair_count"][1].__setitem__(0, 0),
            ),
        )
        register(
            "count_exceeds_min_act",
            lambda d: (
                d["pair_count"][0].__setitem__(1, 2),
                d["pair_count"][1].__setitem__(0, 2),
            ),
        )
        register("gate_bound", lambda d: d["gate_sum"].__setitem__(0, 7.0))
        register("negative_first_sum", lambda d: d["first_sum"].__setitem__(2, -1.0))
        register("nan_pair_sum", lambda d: d["pair_sum"][0].__setitem__(1, float("nan")))
        register("sum_with_zero_count", lambda d: d["pair_sum"][0].__setitem__(2, 5.0))
        register("float_count", lambda d: d["act_count"].__setitem__(1, 1.5))
        register("negative_count", lambda d: d["pair_count"][0].__setitem__(0, -1))
        register("missing_total_rows", lambda d: d.pop("total_rows"))
        for name, mutate in tampers.items():
            with self.subTest(tamper=name):
                bad = doc()
                mutate(bad)
                with self.assertRaises(hope.HopeValidationError):
                    hope.HopeStats.from_dict(bad)

    def test_capability_parent_bound_enforced(self):
        child = hope.HopeStats(3, 2)
        child.add_batch([[0, 1], [0, 1]], [[0.5, 0.5]] * 2, [[1.0, 1.0]] * 2)
        parent = _sample_stats()  # act_count[0] == 1
        self.assertGreater(child.act_count[0], parent.act_count[0])
        parent.capabilities["too_big"] = child
        with self.assertRaises(hope.HopeValidationError):
            parent.validate()

    def test_nested_capability_partitions_rejected(self):
        stats = _sample_stats()
        stats.add_batch([[0, 1]], [[0.5, 0.5]], [[1.0, 1.0]], capability="x")
        doc = stats.to_dict()
        doc["capabilities"]["x"]["capabilities"] = {"y": {}}
        with self.assertRaises(hope.HopeValidationError):
            hope.HopeStats.from_dict(doc)

    def test_zero_pair_count_requires_zero_pair_sum(self):
        stats = _sample_stats()
        doc = stats.to_dict()
        doc["pair_sum"][0][2] = 5.0
        doc["pair_sum"][2][0] = 5.0
        with self.assertRaises(hope.HopeValidationError):
            hope.HopeStats.from_dict(doc)


class MergeResumeTests(unittest.TestCase):
    @staticmethod
    def _dyadic_stats(rows):
        stats = hope.HopeStats(4, 2)
        spec = {
            "a": ([[0, 1]], [[0.5, 0.5]], [[2.0, 4.0]]),
            "b": ([[2, 3]], [[0.25, 0.75]], [[8.0, 4.0]]),
            "c": ([[0, 3]], [[0.25, 0.75]], [[4.0, 4.0]]),
        }
        for key in rows:
            selected, gates, norms = spec[key]
            stats.add_batch(selected, gates, norms, capability="cap")
        return stats

    def test_split_then_merge_equals_direct_accumulation(self):
        # All fixture values are dyadic rationals, so float addition is exact
        # regardless of association: equality must be byte-identical.
        direct = self._dyadic_stats("abc")
        part_ab = self._dyadic_stats("ab")
        part_c = self._dyadic_stats("c")
        merged = part_ab.merge(part_c)
        self.assertEqual(merged.to_json(), direct.to_json())

    def test_save_load_resume_equals_direct(self):
        direct = self._dyadic_stats("abc")
        with tempfile.TemporaryDirectory() as tmp:
            path_a = os.path.join(tmp, "part-a.json")
            self._dyadic_stats("ab").save(path_a)
            resumed = hope.HopeStats.load(path_a)
            resumed.merge(self._dyadic_stats("c"))
            self.assertEqual(resumed.to_json(), direct.to_json())

    def test_merge_rejects_shape_mismatch(self):
        with self.assertRaises(hope.HopeValidationError):
            hope.HopeStats(4, 2).merge(hope.HopeStats(5, 2))
        with self.assertRaises(hope.HopeValidationError):
            hope.HopeStats(4, 2).merge(hope.HopeStats(4, 3))

    def test_merge_capability_union_does_not_alias_source(self):
        left = hope.HopeStats(4, 2)
        left.add_batch([[0, 1]], [[0.5, 0.5]], [[1.0, 1.0]], capability="p")
        right = hope.HopeStats(4, 2)
        right.add_batch([[2, 3]], [[0.5, 0.5]], [[1.0, 1.0]], capability="q")
        merged = left.merge(right)
        self.assertEqual(sorted(merged.capabilities), ["p", "q"])
        right.capabilities["q"].add_batch([[0, 1]], [[0.5, 0.5]], [[1.0, 1.0]])
        # Mutating the source after merge must not leak into the merged doc.
        self.assertEqual(merged.capabilities["q"].total_rows, 1)


class ExhaustiveSelectorTests(unittest.TestCase):
    def test_matches_product_space_oracle_e8_to_e12(self):
        for e in range(8, 13):
            for seed in (0, 1):
                with self.subTest(e=e, seed=seed):
                    rng = random.Random(f"{e}:{seed}")
                    f_matrix = _random_symmetric_f(e, rng)
                    budget = e // 2
                    exact = hope.exhaustive_select(f_matrix, budget)
                    oracle = hope.oracle_exhaustive_select(f_matrix, budget)
                    self.assertEqual(exact["pruned"], oracle["pruned"])
                    self.assertEqual(exact["objective"], oracle["objective"])
                    self.assertEqual(exact["ties"], oracle["ties"])

    def test_enumerates_all_fixed_sets_e8_to_e16(self):
        for e in range(8, 17):
            with self.subTest(e=e):
                rng = random.Random(f"all-sets:{e}")
                f_matrix = _random_symmetric_f(e, rng)
                budget = e // 2
                exact = hope.exhaustive_select(f_matrix, budget)
                self.assertTrue(exact["exact"])
                self.assertTrue(exact["global_optimal"])
                self.assertEqual(
                    exact["subsets_evaluated"], math.comb(e, budget)
                )
                self.assertEqual(exact["expected_subsets"], math.comb(e, budget))
                # Independent dense recomputation of the delivered optimum.
                self.assertEqual(
                    exact["objective"],
                    hope.oracle_objective(f_matrix, exact["pruned"]),
                )
                # Independent tie recount on the tractable sizes.
                if e <= 14:
                    best = hope.oracle_objective(f_matrix, exact["pruned"])
                    ties = sum(
                        1
                        for combo in itertools.combinations(range(e), budget)
                        if hope.oracle_objective(f_matrix, combo) == best
                    )
                    self.assertEqual(exact["ties"], ties)

    def test_refuses_e17_with_pointer_to_relaxation(self):
        f_matrix = [[1.0] * 17 for _ in range(17)]
        with self.assertRaises(hope.HopeSelectionError) as ctx:
            hope.exhaustive_select(f_matrix, 3)
        message = str(ctx.exception)
        self.assertIn("E <= 16", message)
        self.assertIn(hope.RELAXATION_METHOD, message)

    def test_budget_boundaries(self):
        f_matrix = [[2.0, 1.0], [1.0, 2.0]]
        zero = hope.exhaustive_select(f_matrix, 0)
        self.assertEqual(zero["pruned"], [])
        self.assertEqual(zero["objective"], 0.0)
        full = hope.exhaustive_select(f_matrix, 2)
        self.assertEqual(full["pruned"], [0, 1])
        self.assertEqual(full["objective"], 6.0)
        with self.assertRaises(hope.HopeSelectionError):
            hope.exhaustive_select(f_matrix, -1)
        with self.assertRaises(hope.HopeSelectionError):
            hope.exhaustive_select(f_matrix, 3)

    def test_rejects_invalid_f(self):
        with self.assertRaises(hope.HopeValidationError):
            hope.exhaustive_select([[1.0, -1.0], [-1.0, 1.0]], 1)
        with self.assertRaises(hope.HopeValidationError):
            hope.exhaustive_select([[1.0, 0.5], [0.0, 1.0]], 1)
        with self.assertRaises(hope.HopeValidationError):
            hope.exhaustive_select([[float("nan")]], 0)
        with self.assertRaises(hope.HopeValidationError):
            hope.exhaustive_select([[1.0, 1.0]], 1)  # not square

    def test_uniform_ties_resolve_lexicographically(self):
        e, k = 9, 3
        f_matrix = [[1.0] * e for _ in range(e)]
        exact = hope.exhaustive_select(f_matrix, k)
        self.assertEqual(exact["pruned"], [0, 1, 2])
        self.assertEqual(exact["ties"], math.comb(e, k))

    def test_diag_topk_is_first_order_contrast(self):
        case = hope.FIXTURE_CASES["block_interaction"]
        self.assertEqual(
            hope.diag_topk_pruned(case.f, case.budget),
            list(case.first_order_pruned),
        )
        self.assertEqual(
            hope.quadratic_objective(case.f, case.first_order_pruned),
            case.first_order_objective,
        )


class PathologicalFixtureTests(unittest.TestCase):
    def test_all_cases_match_analytic_optima(self):
        self.assertEqual(
            sorted(hope.FIXTURE_CASES),
            [
                "block_interaction",
                "important_pair",
                "individually_strong_redundancy",
                "sparse",
                "ties",
                "uniform",
            ],
        )
        for name, case in hope.FIXTURE_CASES.items():
            with self.subTest(case=name):
                exact = hope.exhaustive_select(case.f, case.budget)
                self.assertEqual(tuple(exact["pruned"]), case.expected_pruned)
                self.assertEqual(exact["objective"], case.expected_objective)
                self.assertEqual(exact["ties"], case.expected_ties)
                self.assertEqual(
                    exact["objective"],
                    hope.oracle_objective(case.f, exact["pruned"]),
                )
                # First-order (zeroed off-diagonal) REAP-equivalent contrast.
                self.assertEqual(
                    hope.diag_topk_pruned(case.f, case.budget),
                    list(case.first_order_pruned),
                )
                self.assertEqual(
                    hope.oracle_objective(case.f, case.first_order_pruned),
                    case.first_order_objective,
                )
                self.assertLessEqual(
                    exact["objective"], case.first_order_objective
                )

    def test_pair_block_redundancy_first_order_is_strictly_worse(self):
        for name in (
            "important_pair",
            "individually_strong_redundancy",
            "block_interaction",
            "ties",
        ):
            case = hope.FIXTURE_CASES[name]
            with self.subTest(case=name):
                self.assertLess(
                    case.expected_objective, case.first_order_objective
                )

    def test_sparse_and_uniform_are_controls(self):
        for name in ("sparse", "uniform"):
            case = hope.FIXTURE_CASES[name]
            with self.subTest(case=name):
                self.assertEqual(
                    case.expected_objective, case.first_order_objective
                )

    def test_fixture_stats_build_and_validate(self):
        stats = hope.build_fixture_stats(
            "block_interaction",
            rows=64,
            top_k=2,
            seed=7,
            capabilities=["debugging", "planning"],
        )
        stats.validate()
        self.assertEqual(stats.total_rows, 64)
        self.assertEqual(stats.n_experts, 16)
        self.assertTrue(stats.capabilities)
        # Determinism of the seeded generator.
        again = hope.build_fixture_stats(
            "block_interaction",
            rows=64,
            top_k=2,
            seed=7,
            capabilities=["debugging", "planning"],
        )
        self.assertEqual(again.to_json(), stats.to_json())
        with self.assertRaises(hope.HopeValidationError):
            hope.build_fixture_stats("no-such-case", rows=4, top_k=2, seed=1)


class RelaxationTests(unittest.TestCase):
    def test_missing_scipy_raises_explicit_dependency_error(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(hope.HopeDependencyError) as ctx:
                hope.scipy_relax_select([[1.0]], 1)
        message = str(ctx.exception)
        self.assertIn("scipy", message)
        self.assertIn("pip install", message)

    def test_relaxation_behavior_when_scipy_available(self):
        import importlib.util

        if importlib.util.find_spec("scipy") is None:
            self.skipTest("scipy absent; missing-dependency branch covered above")
        case = hope.FIXTURE_CASES["block_interaction"]
        exact = hope.exhaustive_select(case.f, case.budget)
        relax = hope.scipy_relax_select(case.f, case.budget, seeds=(0, 1, 2))
        self.assertFalse(relax["exact"])
        self.assertFalse(relax["global_optimal"])
        self.assertFalse(relax["psd_assumed"])
        self.assertEqual(relax["method"], hope.RELAXATION_METHOD)
        # Heuristic may tie or lose to the optimum, never beat it.
        self.assertGreaterEqual(relax["objective"], exact["objective"] - 1e-9)
        # Deterministic: same seeds, same result.
        repeat = hope.scipy_relax_select(case.f, case.budget, seeds=(0, 1, 2))
        self.assertEqual(repeat["pruned"], relax["pruned"])
        self.assertEqual(repeat["objective"], relax["objective"])
        # Local swap only accepts strict improvements.
        swapped = hope.scipy_relax_select(
            case.f, case.budget, seeds=(0, 1, 2), local_swap=True
        )
        self.assertLessEqual(swapped["objective"], relax["objective"] + 1e-9)

    def test_auto_method_dispatches_by_size(self):
        small = hope.select([[1.0]], 0, method="auto")
        self.assertEqual(small["method"], hope.EXHAUSTIVE_METHOD)
        with self.assertRaises(hope.HopeSelectionError):
            hope.select([[1.0]], 0, method="nonsense")


class ToyrAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = hope.load_config(CONFIG_PATH)

    def test_run_toys_all_pass(self):
        report = hope.run_toys(self.config)
        self.assertTrue(report["pass"], report)
        self.assertEqual(len(report["cases"]), 6)
        for case in report["cases"]:
            self.assertTrue(case["pass"], case)
            self.assertTrue(all(case["checks"].values()), case["checks"])

    def test_observe_select_merge_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = os.path.join(tmp, "fixture")
            code, out = _run_cli(
                [
                    "observe-fixture",
                    "--config",
                    str(CONFIG_PATH),
                    "--out",
                    fixture_dir,
                ]
            )
            self.assertEqual(code, 0, out)
            stats_path = os.path.join(fixture_dir, "block_interaction.json")
            stats = hope.HopeStats.load(stats_path)
            self.assertEqual(stats.total_rows, self.config["fixtures"]["rows_per_case"])

            selection_path = os.path.join(tmp, "selection.json")
            code, out = _run_cli(
                [
                    "select",
                    "--config",
                    str(CONFIG_PATH),
                    "--stats",
                    stats_path,
                    "--budget",
                    "4",
                    "--out",
                    selection_path,
                ]
            )
            self.assertEqual(code, 0, out)
            with open(selection_path, "r", encoding="utf-8") as handle:
                selection = json.load(handle)
            cert = selection["certificate"]
            self.assertTrue(cert["exact"])
            self.assertEqual(cert["method"], hope.EXHAUSTIVE_METHOD)
            self.assertEqual(selection["dense_oracle_objective"], cert["objective"])
            self.assertEqual(
                sorted(selection["pruned"] + selection["retained"]),
                list(range(stats.n_experts)),
            )

            # Capability partition selection runs on the partition's stats.
            code, out = _run_cli(
                [
                    "select",
                    "--config",
                    str(CONFIG_PATH),
                    "--stats",
                    stats_path,
                    "--budget",
                    "4",
                    "--capability",
                    "implementation",
                ]
            )
            self.assertEqual(code, 0, out)
            self.assertLess(out["stats_total_rows"], stats.total_rows)

            merged_path = os.path.join(tmp, "merged.json")
            uniform = os.path.join(fixture_dir, "uniform.json")
            code, out = _run_cli(
                ["merge", "--out", merged_path, uniform, uniform]
            )
            self.assertEqual(code, 0, out)
            merged = hope.HopeStats.load(merged_path)
            self.assertEqual(
                merged.total_rows,
                2 * self.config["fixtures"]["rows_per_case"],
            )

    def test_cli_fails_closed_on_bad_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = os.path.join(tmp, "fixture")
            code, _ = _run_cli(
                ["observe-fixture", "--config", str(CONFIG_PATH), "--out", fixture_dir]
            )
            self.assertEqual(code, 0)
            block = os.path.join(fixture_dir, "block_interaction.json")
            uniform = os.path.join(fixture_dir, "uniform.json")

            code, out = _run_cli(
                ["select", "--config", str(CONFIG_PATH), "--stats", block,
                 "--budget", "4", "--capability", "nope"]
            )
            self.assertEqual(code, 2)
            self.assertIn("nope", out["error"])

            code, out = _run_cli(
                ["select", "--config", str(CONFIG_PATH), "--stats", block,
                 "--budget", "999"]
            )
            self.assertEqual(code, 2)
            self.assertIn("budget", out["error"])

            empty_path = os.path.join(tmp, "empty.json")
            hope.HopeStats(16, 2).save(empty_path)
            code, out = _run_cli(
                ["select", "--config", str(CONFIG_PATH), "--stats", empty_path,
                 "--budget", "4"]
            )
            self.assertEqual(code, 2)
            self.assertIn("no accumulated rows", out["error"])

            # Cross-shape merge must be rejected through the CLI too.
            code, out = _run_cli(
                ["merge", "--out", os.path.join(tmp, "bad.json"), uniform, block]
            )
            self.assertEqual(code, 2)
            self.assertIn("n_experts mismatch", out["error"])

    def test_cli_scipy_method_reports_missing_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = os.path.join(tmp, "fixture")
            _run_cli(
                ["observe-fixture", "--config", str(CONFIG_PATH), "--out", fixture_dir]
            )
            block = os.path.join(fixture_dir, "block_interaction.json")
            with mock.patch("importlib.util.find_spec", return_value=None):
                code, out = _run_cli(
                    ["select", "--config", str(CONFIG_PATH), "--stats", block,
                     "--budget", "4", "--method", "scipy"]
                )
            self.assertEqual(code, 2)
            self.assertIn("pip install", out["error"])

    def test_load_config_rejects_foreign_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad-config.json")
            with open(bad, "w", encoding="utf-8") as handle:
                json.dump({"kind": "not-hope", "schema_version": 1}, handle)
            with self.assertRaises(hope.HopeValidationError):
                hope.load_config(bad)


class ActualRoutingSeamTests(unittest.TestCase):
    """Shared-observation seam: correction-bias routing, one observation for
    REAP and HOPE, and fail-closed unsupported contracts.

    ``reap_adapter`` imports without torch or the pinned REAP clone, so these
    stay stdlib-only.
    """

    # E=4, top-2: raw-logit+bias and sigmoid+bias select different expert sets.
    LOGITS = [-2.0, 0.5, -1.5, 0.1]
    BIAS = (1.5, 0.0, 0.0, 0.0)
    UPSTREAM_IDS = [[1, 3]]  # pinned observer's raw-logit top-k proposal

    @staticmethod
    def _sigmoid(value):
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _raw_logit_bias_topk(logits, bias, k):
        """The REJECTED formula: topk(logits + bias)."""
        scores = [value + b for value, b in zip(logits, bias)]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return sorted(order[:k])

    @staticmethod
    def _unbiased_gates(logits, ids_row):
        scores = [ActualRoutingSeamTests._sigmoid(value) for value in logits]
        total = sum(scores[i] for i in ids_row)
        return [scores[i] / total for i in ids_row]

    @staticmethod
    def _upstream_active_weights(rows, ids_rows):
        """The pinned accumulator's gate math, replicated from
        upstreams/reap/src/reap/pruning_metrics.py: softmax over all experts,
        divide by the gathered selected-set sum (renormalize_router_weights)."""
        out = []
        for row, ids in zip(rows, ids_rows):
            maximum = max(row)
            exps = [math.exp(value - maximum) for value in row]
            total = sum(exps)
            weights = [value / total for value in exps]
            selected_sum = sum(weights[i] for i in ids)
            out.append([weights[i] / selected_sum for i in ids])
        return out

    @staticmethod
    def _sigmoid_spec(bias):
        return reap_adapter.RoutingSpec(
            mode=reap_adapter.SIGMOID_BIAS_TOPK_ROUTES, bias=bias
        )

    @staticmethod
    def _softmax_spec():
        return reap_adapter.RoutingSpec(mode=reap_adapter.SOFTMAX_TOPK_ROUTES)

    def test_contradictory_bias_case_selects_on_sigmoid_not_raw_logits(self):
        # The pre-fix (wrong) formula would select {1, 3}...
        self.assertEqual(
            self._raw_logit_bias_topk(self.LOGITS, self.BIAS, 2), [1, 3]
        )
        ids, gates, feed = reap_adapter.resolve_actual_routing(
            [list(self.LOGITS)], self.UPSTREAM_IDS, self._sigmoid_spec(self.BIAS)
        )
        # ...while the source-grounded selection topk(sigmoid(logits)+bias)
        # flips to {0, 1}; the upstream raw-logit proposal is not trusted.
        self.assertEqual(ids, [[0, 1]])
        self.assertNotEqual(ids, [sorted(self.UPSTREAM_IDS[0])])
        # Gates come from UNBIASED sigmoid(logits) renormalized over the
        # selected set — the biased gather is observably different.
        for got, want in zip(
            gates[0], self._unbiased_gates(self.LOGITS, ids[0])
        ):
            self.assertAlmostEqual(got, want, places=12)
        biased = [
            self._sigmoid(value + b) for value, b in zip(self.LOGITS, self.BIAS)
        ]
        biased_total = sum(biased[i] for i in ids[0])
        for got, wrong in zip(gates[0], [biased[i] / biased_total for i in ids[0]]):
            self.assertGreater(abs(got - wrong), 1e-3)
        # Accumulator feed is log(sigmoid(logits)), never raw logits.
        self.assertIsNotNone(feed)
        for logit, z_value in zip(self.LOGITS, feed[0]):
            self.assertAlmostEqual(
                z_value, math.log(self._sigmoid(logit)), places=12
            )

    def test_bias_affects_choice_but_not_gathered_weights(self):
        no_bias_ids, no_bias_gates, _ = reap_adapter.resolve_actual_routing(
            [list(self.LOGITS)], self.UPSTREAM_IDS, self._sigmoid_spec((0.0,) * 4)
        )
        ids_b, gates_b, _ = reap_adapter.resolve_actual_routing(
            [list(self.LOGITS)], self.UPSTREAM_IDS, self._sigmoid_spec(self.BIAS)
        )
        ids_c, gates_c, _ = reap_adapter.resolve_actual_routing(
            [list(self.LOGITS)], self.UPSTREAM_IDS, self._sigmoid_spec((1.7, 0.0, 0.0, 0.0))
        )
        # The bias changed the routed choice...
        self.assertEqual(no_bias_ids, [[1, 3]])
        self.assertEqual(ids_b, [[0, 1]])
        # ...but identical choices yield identical gates: the bias never
        # enters the gathered weights.
        self.assertEqual(ids_b, ids_c)
        for got, want in zip(gates_b[0], gates_c[0]):
            self.assertAlmostEqual(got, want, places=12)
        for ids_row, gates_row in ((no_bias_ids, no_bias_gates), (ids_b, gates_b)):
            expect = self._unbiased_gates(self.LOGITS, ids_row[0])
            for got, want in zip(gates_row[0], expect):
                self.assertAlmostEqual(got, want, places=12)

    def test_reap_and_hope_consume_one_observation(self):
        ids, gates, feed = reap_adapter.resolve_actual_routing(
            [list(self.LOGITS)], self.UPSTREAM_IDS, self._sigmoid_spec(self.BIAS)
        )
        # REAP side: the accumulator's own gate math on the injected feed
        # reproduces exactly the gates HOPE records — one observation.
        for got_row, want_row in zip(
            self._upstream_active_weights(feed, ids), gates
        ):
            for got, want in zip(got_row, want_row):
                self.assertAlmostEqual(got, want, places=12)
        # HOPE side: same ids/gates — HopeStats first_order equals REAP's
        # mean(g * ||f||) over active tokens for every expert.
        norms = [[3.0, 5.0]]
        stats = hope.HopeStats(4, 2)
        stats.add_batch(ids, gates, norms)
        contributions: dict[int, list[float]] = {expert: [] for expert in range(4)}
        for ids_row, gates_row, norms_row in zip(ids, gates, norms):
            for expert, gate, norm in zip(ids_row, gates_row, norms_row):
                contributions[expert].append(gate * norm)
        for expert, values in contributions.items():
            expected = sum(values) / len(values) if values else 0.0
            self.assertAlmostEqual(
                stats.first_order()[expert], expected, places=12
            )
        # Softmax mode: raw logits are the identity feed and agree as well.
        softmax_ids, softmax_gates, feed = reap_adapter.resolve_actual_routing(
            [list(self.LOGITS)], self.UPSTREAM_IDS, self._softmax_spec()
        )
        self.assertIsNone(feed)
        self.assertEqual(softmax_ids, self.UPSTREAM_IDS)
        for got_row, want_row in zip(
            self._upstream_active_weights([list(self.LOGITS)], softmax_ids),
            softmax_gates,
        ):
            for got, want in zip(got_row, want_row):
                self.assertAlmostEqual(got, want, places=12)

    def test_unverifiable_routing_fails_closed_not_approximated(self):
        cases = [
            (dict(mode=reap_adapter.SOFTMAX_TOPK_ROUTES, group_routing=True), "group"),
            (
                dict(
                    mode=reap_adapter.SIGMOID_BIAS_TOPK_ROUTES,
                    bias=self.BIAS,
                    group_routing=True,
                ),
                "group",
            ),
            (
                dict(mode=reap_adapter.SOFTMAX_TOPK_ROUTES, routed_scaling_factor=2.5),
                "scaling",
            ),
            (dict(mode=reap_adapter.SOFTMAX_TOPK_ROUTES, renormalize=False), "renorm"),
            (dict(mode=reap_adapter.SIGMOID_BIAS_TOPK_ROUTES), "bias"),
            (
                dict(mode=reap_adapter.SOFTMAX_TOPK_ROUTES, bias=(0.0,) * 4),
                "correction bias",
            ),
            (dict(mode="unknown-router"), "mode"),
        ]
        for kwargs, word in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(hope.HopeValidationError) as ctx:
                    reap_adapter.resolve_actual_routing(
                        [list(self.LOGITS)],
                        self.UPSTREAM_IDS,
                        reap_adapter.RoutingSpec(**kwargs),
                    )
                self.assertIn(word, str(ctx.exception))

    def test_invalid_inputs_reject_before_any_mutation(self):
        with self.assertRaises(hope.HopeValidationError) as ctx:
            reap_adapter.resolve_actual_routing(
                [[float("nan"), 0.0]], [[0]], self._softmax_spec()
            )
        self.assertIn("finite", str(ctx.exception))
        with self.assertRaises(hope.HopeValidationError) as ctx:
            reap_adapter.resolve_actual_routing(
                [[0.0, 1.0]], [[0, 5]], self._softmax_spec()
            )
        self.assertIn("expert id", str(ctx.exception))
        with self.assertRaises(hope.HopeValidationError) as ctx:
            reap_adapter.resolve_actual_routing(
                [[0.0, 1.0]], [[1, 1]], self._softmax_spec()
            )
        self.assertIn("duplicate", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
