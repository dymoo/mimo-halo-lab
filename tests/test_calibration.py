"""Tests for the calibration distribution and convergence sweep tools.

Framework: stdlib unittest (``python3 -m unittest tests.test_calibration``).
All fixtures are synthetic and hand-checkable:

- Calibration: exact-weight arithmetic (target mix vs realized) over four
  dyadic episodes, downweighted-event handling at spam_weight 0.25, golden
  exclusion, zero-target cells, unreachable targets, fail-closed contracts,
  double-run byte determinism.
- Convergence: three tiny deterministic HopeStats parts (one row per expert,
  gate 1.0, norm = a) where every expected Jaccard / Spearman value is
  hand-computed from mean-of-squares diagonals, the budget-schedule constant,
  fail-closed schedules, and double-run byte determinism.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimo_halo.pruning import convergence  # noqa: E402
from mimo_halo.pruning import hope  # noqa: E402
from mimo_halo.traces import calibration  # noqa: E402
from mimo_halo.traces.common import TraceError  # noqa: E402
from mimo_halo.traces.identity import _CAPABILITY_WEIGHTS  # noqa: E402
from mimo_halo.traces.partition import PARTITIONS  # noqa: E402


def _dataset_config() -> dict:
    return {
        "schema_version": "1.0.0",
        "partition": {
            "weights": {
                "pruning": 0.35,
                "quant": 0.20,
                "recovery": 0.20,
                "validation": 0.10,
                "golden": 0.10,
                "torture": 0.05,
            }
        },
        "episode": {"downweight": {"spam_weight": 0.25}},
    }


def _records() -> list[dict]:
    """Hand-check fixture (masses: 1.0, 0.3125, 2.0 [golden], 1.0, 0.0)."""
    return [
        {   # multi-label split: 1.0 -> 0.5 exploration + 0.5 planning
            "episode_id": "e1",
            "partition": "pruning",
            "capabilities": ["repo_exploration", "planning"],
            "primary_capability": "repo_exploration",
            "weight": 1.0,
            "event_seq": [0, 1, 2, 3],
            "downweighted_events": 0,
        },
        {   # downweighted: mass = 0.5 * (1 - 0.75 * 4/8) = 0.5 * 0.625 = 0.3125
            "episode_id": "e2",
            "partition": "pruning",
            "capabilities": ["implementation"],
            "weight": 0.5,
            "event_seq": [0, 1, 2, 3, 4, 5, 6, 7],
            "downweighted_events": 4,
        },
        {   # golden: excluded from the fold, reported (never calibrated)
            "episode_id": "e3",
            "partition": "golden",
            "capabilities": ["implementation"],
            "weight": 2.0,
            "event_seq": [0, 1],
            "downweighted_events": 0,
        },
        {   # known-but-unanchored capability: target 0.0, sampling weight 0.0
            "episode_id": "e4",
            "partition": "quant",
            "capabilities": ["architecture"],
            "weight": 1.0,
            "event_seq": [0, 1],
            "downweighted_events": 0,
        },
        {   # weight 0 is legal: counted, contributes zero mass
            "episode_id": "e5",
            "partition": "pruning",
            "capabilities": ["tool_use"],
            "weight": 0.0,
            "event_seq": [0, 1],
            "downweighted_events": 0,
        },
    ]


def _run_calibration_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = calibration.main(argv)
    return code, out.getvalue(), err.getvalue()


def _run_convergence_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = convergence.main(argv)
    return code, out.getvalue(), err.getvalue()


def _part(tmp: Path, name: str, norms: list[float]) -> Path:
    """One observation part: single-expert rows (gate 1.0, norm = a)."""
    stats = hope.HopeStats(4, 1)
    for expert, norm in enumerate(norms):
        stats.add_batch([[expert]], [[1.0]], [[float(norm)]])
    stats.validate()
    path = tmp / name
    stats.save(path)
    return path


def _group(doc: dict, partition: str, capability: str) -> dict:
    hits = [
        row
        for row in doc["groups"]
        if row["partition"] == partition and row["capability"] == capability
    ]
    if len(hits) != 1:
        raise AssertionError(f"expected exactly one row for {(partition, capability)}")
    return hits[0]


class CalibrationDistributionTests(unittest.TestCase):
    def test_exact_weight_arithmetic_target_vs_realized(self) -> None:
        doc = calibration.build_distribution(_records(), _dataset_config())

        # -- masses: extraction weight consumed as-is, spam_weight applied per
        # downweighted event (0.5 * (1 - 0.75 * 0.5) = 0.3125), golden excluded
        totals = doc["totals"]
        self.assertEqual(totals["records_included"], 4)
        self.assertEqual(totals["records_excluded_golden"], 1)
        self.assertEqual(totals["included_mass"], 0.5 + 0.5 + 0.3125 + 1.0)  # 37/16
        self.assertEqual(totals["excluded_golden_mass"], 2.0)

        # -- per-cell realized mass (hand: even multi-label split)
        self.assertEqual(_group(doc, "pruning", "repo_exploration")["realized_mass"], 0.5)
        self.assertEqual(_group(doc, "pruning", "planning")["realized_mass"], 0.5)
        pr_impl = _group(doc, "pruning", "implementation")
        self.assertEqual(pr_impl["realized_mass"], 0.3125)
        self.assertEqual(pr_impl["episodes"], 1)

        # -- target vs realized vs sampling weight, exact rational hand-check:
        # target = (0.35 / 0.9) * 0.20 = 7/18 * 1/5 = 7/90
        # realized = 0.3125 / 2.3125 = 5/37
        # sampling = (7/90) / (5/37) = 259/450
        self.assertAlmostEqual(pr_impl["target_share"], 7 / 90, places=12)
        self.assertAlmostEqual(pr_impl["realized_share"], 5 / 37, places=12)
        self.assertAlmostEqual(pr_impl["sampling_weight"], 259 / 450, places=12)

        # exploration cell: target = 7/18 * 0.15 = 7/120, realized = 0.5/2.3125
        # = 8/37, sampling = (7/120)/(8/37) = 259/960
        pr_expl = _group(doc, "pruning", "repo_exploration")
        self.assertAlmostEqual(pr_expl["target_share"], 7 / 120, places=12)
        self.assertAlmostEqual(pr_expl["realized_share"], 8 / 37, places=12)
        self.assertAlmostEqual(pr_expl["sampling_weight"], 259 / 960, places=12)

        # -- zero-target observed cell: known label, never sampled
        q_arch = _group(doc, "quant", "architecture")
        self.assertEqual(q_arch["target_share"], 0.0)
        self.assertEqual(q_arch["sampling_weight"], 0.0)
        self.assertNotIn("unreachable", q_arch)

        # -- zero-mass observed cell with a positive target: reported, not faked
        zero = _group(doc, "pruning", "tool_use")
        self.assertEqual(zero["episodes"], 1)
        self.assertEqual(zero["realized_mass"], 0.0)
        self.assertIsNone(zero["sampling_weight"])
        self.assertIs(zero["unreachable"], True)

        # -- positive target, no data at all: unreachable gap
        dbg = _group(doc, "pruning", "debugging")
        self.assertEqual(dbg["episodes"], 0)
        self.assertAlmostEqual(dbg["target_share"], 7 / 120, places=12)
        self.assertIsNone(dbg["sampling_weight"])
        self.assertIs(dbg["unreachable"], True)
        # 5 partitions x 9 anchored capabilities = 45 targeted cells; 3 carry
        # realized mass -> 42 unreachable (zero-mass observed cells still count)
        self.assertEqual(totals["unreachable_targets"], 42)
        self.assertEqual(len(doc["groups"]), 46)

        # -- targets: golden renormalized to 0, mixes sum to 1
        p_targets = doc["targets"]["partitions"]
        self.assertEqual(list(p_targets), list(PARTITIONS))
        self.assertEqual(p_targets["golden"], 0.0)
        self.assertAlmostEqual(p_targets["pruning"], 0.35 / 0.9, places=12)
        self.assertAlmostEqual(sum(p_targets.values()), 1.0, places=12)
        taxonomy = calibration._load_taxonomy()
        self.assertEqual(
            doc["targets"]["capabilities"],
            {tag: float(_CAPABILITY_WEIGHTS.get(tag, 0.0)) for tag in taxonomy},
        )
        self.assertAlmostEqual(sum(doc["targets"]["capabilities"].values()), 1.0, places=12)

        # -- methodology quote rides along in the artifact
        self.assertIn(
            "Jaccard/convergence across token budgets",
            doc["definitions"]["methodology"],
        )
        self.assertEqual(doc["schema_version"], "1.0.0")
        self.assertEqual(doc["kind"], "mimo-halo-calibration-distribution")
        self.assertEqual(len(doc["inputs"]["digest"]), 64)

    def test_known_but_unanchored_capability_allowed(self) -> None:
        doc = calibration.build_distribution(
            [
                {
                    "episode_id": "solo",
                    "partition": "recovery",
                    "capabilities": ["architecture"],
                    "weight": 1.0,
                    "event_seq": [0, 1],
                    "downweighted_events": 0,
                }
            ],
            _dataset_config(),
        )
        row = _group(doc, "recovery", "architecture")
        self.assertEqual(row["target_share"], 0.0)
        self.assertEqual(row["sampling_weight"], 0.0)
        self.assertEqual(row["realized_share"], 1.0)

    def test_fail_closed_contracts(self) -> None:
        config = _dataset_config()

        def record(**overrides) -> dict:
            base = {
                "episode_id": "x",
                "partition": "pruning",
                "capabilities": ["implementation"],
                "weight": 1.0,
                "event_seq": [0, 1, 2, 3],
                "downweighted_events": 0,
            }
            base.update(overrides)
            return base

        cases = {
            "unknown capability": (
                [record(capabilities=["sparkle_unicorn"])],
                config,
                "unknown capability",
            ),
            "unknown primary": (
                [record(primary_capability="sparkle_unicorn")],
                config,
                "primary_capability",
            ),
            "unknown partition": (
                [record(partition="vibes")], config, "partition must be one of"
            ),
            "missing partition": (
                [{k: v for k, v in record().items() if k != "partition"}],
                config,
                "partition must be one of",
            ),
            "negative weight": ([record(weight=-1.0)], config, "weight"),
            "missing weight": (
                [{k: v for k, v in record().items() if k != "weight"}],
                config,
                "weight",
            ),
            "downweighted exceeds events": (
                [record(downweighted_events=9)],
                config,
                "exceeds event count",
            ),
            "downweighted without event_seq": (
                [
                    {
                        k: v
                        for k, v in record(downweighted_events=2).items()
                        if k != "event_seq"
                    }
                ],
                config,
                "requires a non-empty event_seq",
            ),
            "duplicate capability": (
                [record(capabilities=["implementation", "implementation"])],
                config,
                "duplicate capability",
            ),
            "golden only (nothing eligible)": (
                [record(partition="golden")],
                config,
                "no calibration-eligible episodes",
            ),
            "bad partition weights in config": (
                [record()],
                {"partition": {"weights": {"pruning": 2.0}},
                 "episode": {"downweight": {"spam_weight": 0.25}}},
                "partition weights must cover exactly",
            ),
            "missing spam_weight in config": (
                [record()],
                {"partition": {"weights": config["partition"]["weights"]}},
                "episode.downweight.spam_weight",
            ),
            "empty records": ([], config, "no episode records"),
        }
        for name, (records, cfg, needle) in cases.items():
            with self.subTest(name):
                with self.assertRaises(TraceError) as ctx:
                    calibration.build_distribution(records, cfg)
                self.assertIn(needle, str(ctx.exception))

    def test_fail_closed_cli_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = root / "episodes.json"
            config_path = root / "dataset.json"
            out_path = root / "dist.json"
            episodes.write_text(
                json.dumps(
                    [
                        {
                            "episode_id": "bad",
                            "partition": "pruning",
                            "capabilities": ["sparkle_unicorn"],
                            "weight": 1.0,
                            "event_seq": [0, 1],
                            "downweighted_events": 0,
                        }
                    ]
                )
            )
            config_path.write_text(json.dumps(_dataset_config()))
            code, out, err = _run_calibration_cli(
                [
                    "--episodes", str(episodes),
                    "--config", str(config_path),
                    "--output", str(out_path),
                ]
            )
            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertIn("unknown capability", err)
            self.assertFalse(out_path.exists())  # no artifact on failure

    def test_deterministic_double_run_identical_bytes(self) -> None:
        first = calibration.build_distribution(_records(), _dataset_config())
        second = calibration.build_distribution(_records(), _dataset_config())
        render = lambda doc: json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False)  # noqa: E731
        self.assertEqual(render(first), render(second))
        self.assertEqual(
            json.dumps(first["inputs"]["digest"]),
            json.dumps(second["inputs"]["digest"]),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = root / "episodes.json"
            config_path = root / "dataset.json"
            episodes.write_text(json.dumps(_records()))
            config_path.write_text(json.dumps(_dataset_config()))
            argv = ["--episodes", str(episodes), "--config", str(config_path)]
            code_a, out_a, _ = _run_calibration_cli(argv + ["--output", str(root / "a.json")])
            code_b, out_b, _ = _run_calibration_cli(argv + ["--output", str(root / "b.json")])
            self.assertEqual((code_a, code_b), (0, 0))
            self.assertEqual(out_a, out_b)  # stdout bytes identical
            self.assertEqual((root / "a.json").read_bytes(), (root / "b.json").read_bytes())


class ConvergenceSweepTests(unittest.TestCase):
    """Three parts, one row per expert, norm sequence a.

    F diagonal after each budget = mean of squares of that expert's norms;
    selection prunes the 2 experts with the smallest diagonal sum (off-
    diagonals are 0: single-expert rows never co-activate).

        budget 4:  [1, 16, 81, 256]              -> prune {0,1} (1+16=17)
        budget 8:  [(1+1)/2, (16+81)/2, ...]     = [1, 48.5, 41, 328]
                   -> prune {0,2} (1+41=42)
        budget 12: [(1+1+49)/3, ...]             = [17, 98/3, 107/3, 219]
                   -> prune {0,1} (17+98/3 = 49.667)

    first_order = mean(norm): [1,4,9,16] -> [1,6.5,5,18] -> [3,14/3,5,37/3];
    ranks [4,3,2,1] -> [4,2,3,1] -> [4,3,2,1]; sum d^2 = 2 both steps, so
    Spearman rho = 1 - 6*2/(4*15) = 0.8; Jaccard({0,1},{0,2}) = 1/3.
    """

    PART_NORMS = [[1, 4, 9, 16], [1, 9, 1, 20], [7, 1, 5, 1]]

    def _parts(self, root: Path) -> list[Path]:
        return [
            _part(root, f"part-{k}.json", norms)
            for k, norms in enumerate(self.PART_NORMS)
        ]

    def test_convergence_metrics_hand_computed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = convergence.run_convergence(
                self._parts(Path(tmp)), prune_budget=2, budgets=(4, 8, 12)
            )
        steps = doc["steps"]
        self.assertEqual([s["budget"] for s in steps], [4, 8, 12])
        self.assertEqual([s["rows"] for s in steps], [4, 8, 12])
        self.assertEqual([s["parts_consumed"] for s in steps], [1, 2, 3])
        self.assertTrue(all(s["reached"] for s in steps))
        self.assertEqual([s["pruned"] for s in steps], [[0, 1], [0, 2], [0, 1]])
        self.assertEqual(steps[0]["objective"], 17.0)
        self.assertEqual(steps[1]["objective"], 42.0)
        self.assertAlmostEqual(steps[2]["objective"], 17 + 98 / 3, places=12)
        self.assertEqual(steps[0]["method_used"], "exhaustive")
        self.assertTrue(all(s["exact"] and s["global_optimal"] for s in steps))
        self.assertEqual(steps[1]["first_order_ranks"], [4, 2, 3, 1])

        pairs = doc["consecutive"]
        self.assertEqual(len(pairs), 2)
        for pair, (from_b, to_b) in zip(pairs, [(4, 8), (8, 12)]):
            self.assertEqual(pair["from_budget"], from_b)
            self.assertEqual(pair["to_budget"], to_b)
            self.assertAlmostEqual(pair["jaccard"], 1 / 3, places=12)
            self.assertAlmostEqual(pair["rank_stability"], 0.8, places=12)

        self.assertEqual(doc["prune_budget"], 2)
        self.assertEqual(doc["n_experts"], 4)
        self.assertEqual(doc["top_k"], 1)
        self.assertEqual(doc["parts_beyond_schedule"], 0)
        self.assertEqual(
            [p["rows"] for p in doc["inputs"]["parts"]], [4, 4, 4]
        )
        self.assertEqual(len(doc["inputs"]["digest"]), 64)
        self.assertIn(
            "Jaccard/convergence across token budgets",
            doc["definitions"]["methodology_row"],
        )
        self.assertIn("Spearman rho", doc["definitions"]["rank_stability"])
        self.assertIn("INTERSECT", doc["definitions"]["jaccard"])

    def test_deterministic_double_run_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = self._parts(root)
            render = lambda doc: json.dumps(  # noqa: E731
                doc, sort_keys=True, indent=1, allow_nan=False
            )
            first = convergence.run_convergence(parts, prune_budget=2, budgets=(4, 8, 12))
            second = convergence.run_convergence(parts, prune_budget=2, budgets=(4, 8, 12))
            self.assertEqual(render(first), render(second))

            argv = [str(p) for p in parts] + [
                "--prune-budget", "2",
                "--budgets", "4,8,12",
            ]
            code_a, out_a, _ = _run_convergence_cli(argv + ["--out", str(root / "a.json")])
            code_b, out_b, _ = _run_convergence_cli(argv + ["--out", str(root / "b.json")])
            self.assertEqual((code_a, code_b), (0, 0))
            self.assertEqual(out_a, out_b)
            self.assertEqual((root / "a.json").read_bytes(), (root / "b.json").read_bytes())

    def test_budget_schedule_constant_naming_and_default_run(self) -> None:
        # named constant: 50K/100K/250K/500K/1M, strictly increasing
        self.assertEqual(
            convergence.TOKEN_BUDGETS,
            (50_000, 100_000, 250_000, 500_000, 1_000_000),
        )
        self.assertEqual(
            list(convergence.TOKEN_BUDGETS),
            sorted(convergence.TOKEN_BUDGETS),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = self._parts(root)
            code, out, _ = _run_convergence_cli(
                [str(p) for p in parts] + ["--prune-budget", "2"]
            )
            self.assertEqual(code, 0)
            doc = json.loads(out)
            # tiny fixtures: default schedule runs past the 12 available rows
            self.assertEqual(doc["budgets"], list(convergence.TOKEN_BUDGETS))
            self.assertTrue(all(s["reached"] is False for s in doc["steps"]))
            self.assertEqual(doc["steps"][0]["rows"], 12)  # all parts merged
            self.assertEqual(doc["parts_beyond_schedule"], 0)
            # no new rows after the first budget -> identical selections
            self.assertTrue(
                all(p["jaccard"] == 1.0 for p in doc["consecutive"])
            )
            self.assertTrue(
                all(p["rank_stability"] == 1.0 for p in doc["consecutive"])
            )

    def test_fail_closed_schedules_and_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = self._parts(root)
            # genuine E=3 part: merge must reject the n_experts mismatch
            small = hope.HopeStats(3, 1)
            small.add_batch([[0], [1]], [[1.0], [1.0]], [[1.0], [2.0]])
            small_path = root / "part-e3.json"
            small.save(small_path)
            # zero-row part: selection on empty stats must be refused
            empty_path = root / "part-empty.json"
            hope.HopeStats(4, 1).save(empty_path)

            cases = {
                "no parts": lambda: convergence.run_convergence(
                    [], prune_budget=2
                ),
                "prune budget zero": lambda: convergence.run_convergence(
                    parts, prune_budget=0
                ),
                "non-increasing budgets": lambda: convergence.run_convergence(
                    parts, prune_budget=2, budgets=(8, 4)
                ),
                "empty schedule": lambda: convergence.run_convergence(
                    parts, prune_budget=2, budgets=()
                ),
                "n_experts mismatch": lambda: convergence.run_convergence(
                    [parts[0], small_path], prune_budget=2, budgets=(4, 8)
                ),
                "empty stats only": lambda: convergence.run_convergence(
                    [empty_path], prune_budget=1, budgets=(1,)
                ),
            }
            for name, thunk in cases.items():
                with self.subTest(name):
                    with self.assertRaises(hope.HopeError):
                        thunk()

            # CLI-level failure: unparsable --budgets -> exit 2, hope-error JSON
            code, out, _ = _run_convergence_cli(
                [str(parts[0]), "--prune-budget", "2", "--budgets", "x,y"]
            )
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out)["kind"], "hope-error")


class CliEndToEndTests(unittest.TestCase):
    def test_calibration_cli_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = root / "episodes.jsonl"
            episodes.write_text(
                "\n".join(json.dumps(record) for record in _records()) + "\n"
            )
            config_path = root / "dataset.json"
            config_path.write_text(json.dumps(_dataset_config()))
            out_path = root / "dist.json"
            code, out, err = _run_calibration_cli(
                [
                    "--episodes", str(episodes),
                    "--config", str(config_path),
                    "--output", str(out_path),
                ]
            )
            self.assertEqual(code, 0, err)
            doc = json.loads(out)
            self.assertEqual(doc["kind"], "mimo-halo-calibration-distribution")
            self.assertEqual(doc["totals"]["included_mass"], 2.3125)
            on_disk = json.loads(out_path.read_text())
            self.assertEqual(on_disk["inputs"]["digest"], doc["inputs"]["digest"])

    def test_convergence_cli_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = [
                _part(root, f"part-{k}.json", norms)
                for k, norms in enumerate(
                    ConvergenceSweepTests.PART_NORMS
                )
            ]
            out_path = root / "report.json"
            code, out, err = _run_convergence_cli(
                [str(p) for p in parts]
                + ["--prune-budget", "2", "--budgets", "4,8,12",
                   "--out", str(out_path)]
            )
            self.assertEqual(code, 0, err)
            doc = json.loads(out)
            self.assertEqual(doc["kind"], "mimo-halo-calibration-convergence")
            self.assertEqual(doc["budgets"], [4, 8, 12])
            self.assertAlmostEqual(
                doc["consecutive"][0]["jaccard"], 1 / 3, places=12
            )
            self.assertAlmostEqual(
                doc["consecutive"][1]["jaccard"], 1 / 3, places=12
            )
            self.assertTrue(out_path.is_file())


if __name__ == "__main__":
    unittest.main()
