"""Regression tests for mimo_halo.baselines.compare.

Focus areas mandated by the comparison contract:

- asymmetric removed-expert counters (baseline-only vs candidate-only),
  emitted BOTH as counts and as sorted original-expert-ID lists
  (removed_both_expert_ids / baseline_only_removed_expert_ids /
  candidate_only_removed_expert_ids), with role-swap identity checks,
- unequal retained counts between baseline and candidate,
- missing capability/pair evidence reported as unavailable (never zero,
  never inferred from selection overlap),
- fail-closed duplicate handling in optional files: repeated layer entries
  (which would silently drop stats) and duplicate pair/circuit identities
  within the SAME capability (which would double count) raise; identity is
  (layer, kind, expert set, capability), so the same pair observed under
  two capabilities is two legitimate distinct observations,
- pair semantics: candidate retained both endpoints while baseline removed
  at least one, labeled as evidence rather than proven circuit survival,
- fail-closed validation: duplicate/out-of-range expert IDs, mismatched
  layer universes, malformed JSON, non-finite importance values.

Run with::

    PYTHONPATH=src python -m unittest tests.test_baseline_compare -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mimo_halo.baselines.compare import ComparisonError, build_report, main

EXPERT_COUNT = 8


def make_map(layer: int, retained: list[int], expert_count: int = EXPERT_COUNT) -> dict:
    """A schema-complete layer entry: retained + pruned partition the universe."""
    pruned = sorted(set(range(expert_count)) - set(retained))
    return {
        "layer": layer,
        "original_expert_count": expert_count,
        "retained_expert_ids": list(retained),
        "pruned_expert_ids": pruned,
    }


def baseline_payload(layer_retained: dict[int, list[int]], projections=None) -> dict:
    layers = []
    for layer, retained in sorted(layer_retained.items()):
        entry = make_map(layer, retained)
        if projections is not None:
            entry["projections"] = projections.get(layer, {})
        layers.append(entry)
    return {"schema_version": 1, "layers": layers}


class ComparisonTestCase(unittest.TestCase):
    def report_for(self, baseline: dict, candidate: dict, **kwargs) -> dict:
        return build_report(
            baseline,
            candidate,
            kwargs.get("capabilities"),
            kwargs.get("pairs"),
            {"baseline": "baseline.json", "candidate": "candidate.json"},
        )


class TestSelectionMetrics(ComparisonTestCase):
    def test_identical_selections_give_perfect_jaccard_and_zero_removed(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3], 1: [0, 1, 2, 4]})
        report = self.report_for(baseline, json.loads(json.dumps(baseline)))
        self.assertEqual(report["comparison"]["layers_compared"], 2)
        for entry in report["per_layer"]:
            self.assertEqual(entry["retained_jaccard"], 1.0)
            self.assertEqual(entry["removed_both"], entry["removed_baseline"])
            self.assertEqual(entry["baseline_only_removed"], 0)
            self.assertEqual(entry["candidate_only_removed"], 0)
            self.assertEqual(entry["baseline_only_removed_expert_ids"], [])
            self.assertEqual(entry["candidate_only_removed_expert_ids"], [])
        # Layer 0 keeps {0,1,2,3}: the full pruned set {4,5,6,7} is removed
        # by both, named explicitly in the original-ID namespace.
        self.assertEqual(report["per_layer"][0]["removed_both_expert_ids"], [4, 5, 6, 7])
        self.assertEqual(report["agreement"]["layers_with_identical_retained_set"], [0, 1])
        self.assertEqual(report["agreement"]["layers_with_identical_pruned_sets"], [0, 1])

    def test_unequal_retained_counts_jaccard_and_removed_asymmetry(self) -> None:
        # Universe 0..7. Baseline keeps {0..5} (prunes {6,7}); candidate keeps
        # {0,1,2,3,6,7} (prunes {4,5}). Equal retained counts but disjoint
        # removals; a later test also covers unequal retained counts.
        baseline = baseline_payload({0: [0, 1, 2, 3, 4, 5]})
        candidate = baseline_payload({0: [0, 1, 2, 3, 6, 7]})
        entry = self.report_for(baseline, candidate)["per_layer"][0]
        self.assertEqual(entry["retained_baseline_count"], 6)
        self.assertEqual(entry["retained_candidate_count"], 6)
        self.assertEqual(entry["retained_intersection"], 4)
        self.assertEqual(entry["retained_union"], 8)
        self.assertAlmostEqual(entry["retained_jaccard"], 0.5)
        self.assertEqual(entry["removed_both"], 0)
        self.assertEqual(entry["baseline_only_removed"], 2)
        self.assertEqual(entry["candidate_only_removed"], 2)

    def test_unequal_retained_counts_are_reported_per_side(self) -> None:
        # Candidate retains only 4 of 8 experts (REAP-style deeper prune).
        baseline = baseline_payload({0: [0, 1, 2, 3, 4, 5]})
        candidate = baseline_payload({0: [0, 1, 2, 3]})
        entry = self.report_for(baseline, candidate)["per_layer"][0]
        self.assertEqual(entry["retained_baseline_count"], 6)
        self.assertEqual(entry["retained_candidate_count"], 4)
        self.assertEqual(entry["removed_baseline"], 2)
        self.assertEqual(entry["removed_candidate"], 4)
        # Baseline prunes {6,7}; candidate prunes {4,5,6,7}.
        self.assertEqual(entry["removed_both"], 2)
        self.assertEqual(entry["baseline_only_removed"], 0)
        self.assertEqual(entry["candidate_only_removed"], 2)
        # Jaccard over retained sets: 6 ∩ 4 = 4, union = 6.
        self.assertAlmostEqual(entry["retained_jaccard"], 4 / 6)

    def test_removed_counter_lists_name_exact_original_experts(self) -> None:
        # Universe 0..7. Baseline keeps {0..5} (prunes {6,7}); candidate
        # keeps {0,1,2,3,6,7} (prunes {4,5}) — disjoint removals.
        baseline = baseline_payload({0: [0, 1, 2, 3, 4, 5]})
        candidate = baseline_payload({0: [0, 1, 2, 3, 6, 7]})
        entry = self.report_for(baseline, candidate)["per_layer"][0]
        self.assertEqual(entry["removed_both_expert_ids"], [])
        self.assertEqual(entry["baseline_only_removed_expert_ids"], [6, 7])
        self.assertEqual(entry["candidate_only_removed_expert_ids"], [4, 5])
        # Candidate removes a superset: keeps {0,1,2,3} (prunes {4,5,6,7}).
        deeper = baseline_payload({0: [0, 1, 2, 3]})
        entry = self.report_for(baseline, deeper)["per_layer"][0]
        self.assertEqual(entry["removed_both_expert_ids"], [6, 7])
        self.assertEqual(entry["baseline_only_removed_expert_ids"], [])
        self.assertEqual(entry["candidate_only_removed_expert_ids"], [4, 5])
        # The named sets must be exactly the sets the counts summarize.
        self.assertEqual(len(entry["removed_both_expert_ids"]), entry["removed_both"])
        self.assertEqual(
            len(entry["baseline_only_removed_expert_ids"]), entry["baseline_only_removed"]
        )
        self.assertEqual(
            len(entry["candidate_only_removed_expert_ids"]), entry["candidate_only_removed"]
        )

    def test_removed_counters_are_asymmetric_under_role_swap(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3, 4, 5]})
        candidate = baseline_payload({0: [0, 1, 2, 3]})
        forward = self.report_for(baseline, candidate)["per_layer"][0]
        backward = self.report_for(candidate, baseline)["per_layer"][0]
        self.assertEqual(forward["baseline_only_removed"], backward["candidate_only_removed"])
        self.assertEqual(forward["candidate_only_removed"], backward["baseline_only_removed"])
        self.assertEqual(forward["removed_both"], backward["removed_both"])
        self.assertEqual(forward["removed_baseline"], backward["removed_candidate"])
        # The same swap must hold for the named original-ID sets themselves.
        self.assertEqual(
            forward["baseline_only_removed_expert_ids"],
            backward["candidate_only_removed_expert_ids"],
        )
        self.assertEqual(
            forward["candidate_only_removed_expert_ids"],
            backward["baseline_only_removed_expert_ids"],
        )
        self.assertEqual(
            forward["removed_both_expert_ids"], backward["removed_both_expert_ids"]
        )
        self.assertEqual(forward["candidate_only_removed_expert_ids"], [4, 5])

    def test_layer_patterns_flag_each_side_removal_dominance(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3]})
        candidate = baseline_payload(
            {
                0: [0, 1, 5],  # candidate prunes 5 experts vs baseline 4
                1: [0, 1, 2, 3],  # identical
                2: [0, 1, 2, 3, 4],  # baseline prunes 4 vs candidate 3
            }
        )
        asym = self.report_for(baseline, candidate)["agreement"]["asymmetry"]
        self.assertEqual(asym["layers_candidate_removed_more"], [0])
        self.assertEqual(asym["layers_baseline_removed_more"], [2])
        self.assertEqual(asym["baseline_only_removed_total"], 2)
        self.assertEqual(asym["candidate_only_removed_total"], 2)
        self.assertEqual(asym["removed_both_total"], 10)


class TestValidationFailClosed(unittest.TestCase):
    def test_layer_universe_mismatch_fails(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3], 1: [0, 1, 2, 3]})
        candidate = baseline_payload({0: [0, 1, 2, 3], 2: [0, 1, 2, 3]})
        with self.assertRaises(ComparisonError):
            build_report(baseline, candidate, None, None, {})

    def test_original_expert_count_mismatch_fails(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3]})
        candidate = {
            "schema_version": 1,
            "layers": [
                {"layer": 0, "original_expert_count": 4, "retained_expert_ids": [0, 1], "pruned_expert_ids": [2, 3]}
            ],
        }
        with self.assertRaises(ComparisonError):
            build_report(baseline, candidate, None, None, {})

    def test_duplicate_retained_id_fails(self) -> None:
        payload = baseline_payload({0: [0, 1, 2, 3]})
        payload["layers"][0]["retained_expert_ids"] = [0, 0, 1, 2]
        with self.assertRaises(ComparisonError):
            build_report(payload, payload, None, None, {})

    def test_out_of_range_id_fails(self) -> None:
        payload = baseline_payload({0: [0, 1, 2, 3]})
        payload["layers"][0]["pruned_expert_ids"][0] = EXPERT_COUNT + 5
        with self.assertRaises(ComparisonError):
            build_report(payload, payload, None, None, {})

    def test_incomplete_universe_fails(self) -> None:
        payload = baseline_payload({0: [0, 1, 2, 3]})
        payload["layers"][0]["retained_expert_ids"] = [0, 1, 2]
        payload["layers"][0]["pruned_expert_ids"] = [4, 5, 6, 7]
        with self.assertRaises(ComparisonError):
            build_report(payload, payload, None, None, {})

    def test_retained_and_pruned_overlap_fails(self) -> None:
        payload = baseline_payload({0: [0, 1, 2, 3]})
        payload["layers"][0]["pruned_expert_ids"] = [0, 4, 5, 6, 7]
        with self.assertRaises(ComparisonError):
            build_report(payload, payload, None, None, {})

    def test_duplicate_capability_layer_entry_fails_closed(self) -> None:
        # Repeating a layer entry would silently keep only the last stats.
        baseline = baseline_payload({0: [0, 1, 2, 3]})
        capabilities = {
            "schema_version": 1,
            "layers": [
                {"layer": 0, "capabilities": {"shell": {"expert_importance": {"0": 1.0, "1": 2.0}}}},
                {"layer": 0, "capabilities": {"debugging": {"expert_importance": {"6": 3.0}}}},
            ],
        }
        with self.assertRaises(ComparisonError) as ctx:
            build_report(baseline, baseline, capabilities, None, {})
        self.assertIn("duplicate layer", str(ctx.exception))

    def test_malformed_json_fails_via_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = tmp_path / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            ok = tmp_path / "ok.json"
            ok.write_text(json.dumps(baseline_payload({0: [0, 1, 2, 3]})), encoding="utf-8")
            exit_code = main(
                ["--baseline", str(broken), "--candidate", str(ok), "--output", str(tmp_path / "r.json")]
            )
            self.assertEqual(exit_code, 2)

    def test_nonfinite_importance_fails_closed(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3]})
        capabilities = {
            "schema_version": 1,
            "layers": [
                {
                    "layer": 0,
                    "capabilities": {
                        "debugging": {"expert_importance": {"0": 1.5, "1": float("nan")}}
                    },
                }
            ],
        }
        with self.assertRaises(ComparisonError):
            build_report(baseline, baseline, capabilities, None, {})

    def test_nonfinite_constant_rejected_at_parse(self) -> None:
        from mimo_halo.baselines.compare import _load_json

        nonfinite = (
            '{"schema_version": 1, "layers": [{"layer": 0, "capabilities": '
            '{"shell": {"expert_importance": {"0": Infinity}}}}]}'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "caps.json"
            path.write_text(nonfinite, encoding="utf-8")
            with self.assertRaises(ComparisonError):
                _load_json(path, "capabilities")

    def test_schema_version_mismatch_fails_via_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.json"
            payload = baseline_payload({0: [0, 1, 2, 3]})
            payload["schema_version"] = 2
            baseline_path.write_text(json.dumps(payload), encoding="utf-8")
            candidate_path = tmp_path / "candidate.json"
            candidate_path.write_text(
                json.dumps(baseline_payload({0: [0, 1, 2, 3]})), encoding="utf-8"
            )
            exit_code = main(
                ["--baseline", str(baseline_path), "--candidate", str(candidate_path), "--output", str(tmp_path / "r.json")]
            )
            self.assertEqual(exit_code, 2)


class TestMissingEvidenceUnavailable(ComparisonTestCase):
    def test_absent_capability_and_pair_files_report_unavailable_not_zero(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3]})
        report = self.report_for(baseline, json.loads(json.dumps(baseline)))
        self.assertEqual(report["capabilities"]["status"], "unavailable")
        self.assertEqual(report["pairs"]["status"], "unavailable")
        self.assertNotIn("counts", report["pairs"])
        self.assertNotIn("capabilities", report["capabilities"])
        self.assertIn("not inferred", report["capabilities"]["reason"])
        self.assertIn("not inferred", report["pairs"]["reason"])

    def test_partial_capability_layers_keep_missing_layers_unavailable(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3], 1: [0, 1, 2, 4]})
        capabilities = {
            "schema_version": 1,
            "layers": [
                {
                    "layer": 0,
                    "capabilities": {
                        "shell": {
                            "expert_importance": {
                                "0": 3.0, "1": 1.0, "2": 0.5, "3": 2.0,
                                "4": 0.25, "5": 0.125, "6": 4.0, "7": 0.75,
                            }
                        }
                    },
                }
            ],
        }
        report = self.report_for(
            baseline, json.loads(json.dumps(baseline)), capabilities=capabilities
        )
        capability = report["capabilities"]["capabilities"][0]
        self.assertEqual(capability["layers_available"], [0])
        self.assertEqual(capability["layers_missing"], [1])
        self.assertEqual(capability["per_layer"][1]["status"], "unavailable")
        # Identical selections remove identical importance mass.
        self.assertEqual(capability["delta_removed_importance_total"], 0.0)

    def test_capability_masses_reflect_each_side_removal(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3, 4, 5]})  # prunes {6,7}
        candidate = baseline_payload({0: [0, 1, 2, 3, 6, 7]})  # prunes {4,5}
        importance = {
            "0": 8.0, "1": 4.0, "2": 2.0, "3": 1.0,
            "4": 0.5, "5": 0.125, "6": 16.0, "7": 32.0,
        }
        capabilities = {
            "schema_version": 1,
            "layers": [{"layer": 0, "capabilities": {"git": {"expert_importance": importance}}}],
        }
        capability = self.report_for(
            baseline, candidate, capabilities=capabilities
        )["capabilities"]["capabilities"][0]
        self.assertAlmostEqual(capability["baseline_removed_importance_total"], 48.0)
        self.assertAlmostEqual(capability["candidate_removed_importance_total"], 0.625)
        self.assertAlmostEqual(capability["delta_removed_importance_total"], -47.375)
        self.assertAlmostEqual(capability["retained_importance_baseline_total"], 15.625)
        self.assertAlmostEqual(capability["retained_importance_candidate_total"], 63.0)


class TestPairSemantics(ComparisonTestCase):
    def pairs_file(self, layer: int, pairs: list[dict], circuits: list[dict] | None = None) -> dict:
        entry: dict = {"layer": layer, "pairs": pairs}
        if circuits is not None:
            entry["circuits"] = circuits
        return {"schema_version": 1, "layers": [entry]}

    def test_candidate_retained_both_baseline_removed_labeled_evidence(self) -> None:
        # Baseline prunes {4,5}; candidate keeps 4 and 5 but prunes {6,7}.
        baseline = baseline_payload({0: [0, 1, 2, 3, 6, 7]})
        candidate = baseline_payload({0: [0, 1, 2, 3, 4, 5]})
        pairs = self.pairs_file(
            0,
            [
                {"experts": [4, 5], "importance": 2.5, "capability": "debugging", "objective_source": "observer-run#42"},
                {"experts": [6, 7], "importance": 1.0},
                {"experts": [1, 6], "importance": 0.5},
            ],
        )
        pairs_section = self.report_for(baseline, candidate, pairs=pairs)["pairs"]
        self.assertEqual(pairs_section["status"], "available")
        counts = pairs_section["pairs"]["counts"]
        self.assertEqual(counts["total"], 3)
        # Only [4,5] has baseline removals ({4,5}), and both of them.
        self.assertEqual(counts["candidate_retained_both_baseline_removed"], 1)
        self.assertEqual(counts["baseline_removed_at_least_one"], 1)
        self.assertEqual(counts["baseline_removed_all"], 1)
        # Candidate removes {6,7}: [6,7] fully and [1,6] partially.
        self.assertEqual(counts["baseline_retained_both_candidate_removed"], 2)
        self.assertAlmostEqual(
            pairs_section["pairs"]["importance_sums"][
                "candidate_retained_both_baseline_removed_importance"
            ],
            2.5,
        )
        self.assertEqual(
            pairs_section["pairs"]["candidate_retained_both_baseline_removed_by_capability"],
            {"debugging": 1},
        )
        self.assertIn("NOT", pairs_section["evidence_label"].upper())
        self.assertIn("proven", pairs_section["evidence_label"])
        self.assertEqual(pairs_section["pairs_total"], 3)

    def test_pair_validation_requires_distinct_in_range_experts_and_importance(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3]})
        bad_cases = [
            self.pairs_file(0, [{"experts": [3, 3], "importance": 1.0}]),
            self.pairs_file(0, [{"experts": [0, EXPERT_COUNT], "importance": 1.0}]),
            self.pairs_file(0, [{"experts": [0], "importance": 1.0}]),
            self.pairs_file(0, [{"experts": [0, 1]}]),
            self.pairs_file(0, [{"experts": [0, 1], "importance": "high"}]),
            self.pairs_file(0, []),
        ]
        for pairs in bad_cases:
            with self.assertRaises(ComparisonError, msg=json.dumps(pairs)):
                build_report(baseline, baseline, None, pairs, {})

    def test_duplicate_pair_layer_entry_fails_closed(self) -> None:
        # Repeating a layer entry would silently drop the earlier pairs.
        baseline = baseline_payload({0: [0, 1, 2, 3]})
        first = self.pairs_file(0, [{"experts": [4, 5], "importance": 1.0}])
        second = self.pairs_file(0, [{"experts": [6, 7], "importance": 2.0}])
        duplicate = {"schema_version": 1, "layers": first["layers"] + second["layers"]}
        with self.assertRaises(ComparisonError) as ctx:
            build_report(baseline, baseline, None, duplicate, {})
        self.assertIn("duplicate layer", str(ctx.exception))

    def test_unmarked_duplicate_pair_identity_fails_closed(self) -> None:
        # Same expert set in either order is ONE identity per capability;
        # repeating it within the same capability would double count.
        baseline = baseline_payload({0: [0, 1, 2, 3]})
        bad_cases = [
            self.pairs_file(
                0,
                [
                    {"experts": [4, 5], "importance": 1.0},
                    {"experts": [5, 4], "importance": 2.0},
                ],
            ),
            self.pairs_file(
                0,
                [
                    {"experts": [4, 5], "importance": 1.0, "capability": "debugging"},
                    {"experts": [4, 5], "importance": 2.0, "capability": "debugging"},
                ],
            ),
        ]
        for pairs in bad_cases:
            with self.assertRaises(ComparisonError, msg=json.dumps(pairs)) as ctx:
                build_report(baseline, baseline, None, pairs, {})
            self.assertIn("double count", str(ctx.exception))

    def test_same_pair_across_capabilities_kept_in_by_capability_breakdowns(self) -> None:
        # Baseline prunes {2,3}; candidate keeps 2 and 3, so pair [2,3] is
        # candidate-retained-both / baseline-removed evidence under each of
        # its two observed capabilities — two distinct supplied
        # observations, not a duplicate.
        baseline = baseline_payload({0: [0, 1, 4, 5, 6, 7]})  # prunes {2,3}
        candidate = baseline_payload({0: [0, 1, 2, 3, 4, 5]})  # prunes {6,7}
        pairs = self.pairs_file(
            0,
            [
                {"experts": [2, 3], "importance": 1.0, "capability": "debugging"},
                {"experts": [2, 3], "importance": 0.5, "capability": "recovery"},
            ],
        )
        section = self.report_for(baseline, candidate, pairs=pairs)["pairs"]
        summary = section["pairs"]
        self.assertEqual(summary["counts"]["total"], 2)
        self.assertEqual(summary["counts"]["candidate_retained_both_baseline_removed"], 2)
        self.assertEqual(
            summary["candidate_retained_both_baseline_removed_by_capability"],
            {"debugging": 1, "recovery": 1},
        )
        self.assertAlmostEqual(
            summary["importance_sums"]["candidate_retained_both_baseline_removed_importance"],
            1.5,
        )
        # Totals are documented as supplied distinct capability observations.
        self.assertIn("distinct capability observations", section["totals_note"])
        self.assertIn("not unique proven functional circuits", section["totals_note"])

    def test_unmarked_duplicate_circuit_identity_fails_closed(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3]})
        pairs = self.pairs_file(
            0,
            [],
            circuits=[
                {"experts": [4, 5, 6], "objective_source": "toy-optimum"},
                {"experts": [6, 5, 4], "objective_source": "toy-replay"},
            ],
        )
        with self.assertRaises(ComparisonError) as ctx:
            build_report(baseline, baseline, None, pairs, {})
        self.assertIn("double count", str(ctx.exception))

    def test_circuit_expert_sets_are_scored_explicitly(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3]})  # prunes {4,5,6,7}
        candidate = baseline_payload({0: [0, 1, 4, 5]})  # prunes {2,3,6,7}
        pairs = self.pairs_file(
            0,
            [],
            circuits=[
                {"experts": [4, 5, 6], "capability": "recovery", "objective_source": "toy-optimum"},
            ],
        )
        circuits = self.report_for(baseline, candidate, pairs=pairs)["pairs"]["circuits"]
        self.assertEqual(circuits["counts"]["total"], 1)
        self.assertEqual(circuits["counts"]["candidate_retained_both_baseline_removed"], 0)
        self.assertEqual(circuits["counts"]["candidate_removed_at_least_one"], 1)
        self.assertEqual(circuits["counts"]["baseline_removed_at_least_one"], 1)


class TestPrecisionReference(ComparisonTestCase):
    def projections(self) -> dict:
        return {
            "gate": {"bits": 8, "group_size": 64, "mode": "affine", "source_key": "layers.0.mlp.gate_proj"},
            "up": {"bits": 3, "group_size": 128, "mode": "affine", "source_key": "layers.0.mlp.up_proj"},
            "down": {"bits": 2, "group_size": 128, "mode": "affine", "source_key": "layers.0.mlp.down_proj"},
        }

    def test_projection_format_histogram_and_higher_bits_listing(self) -> None:
        baseline = baseline_payload(
            {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]},
            projections={0: self.projections(), 1: self.projections()},
        )
        baseline["precision_summary"] = {
            "sensitive_projection_references": ["layers.0.mlp.gate_proj"]
        }
        reference = self.report_for(baseline, json.loads(json.dumps(baseline)))[
            "baseline_precision_reference"
        ]
        self.assertEqual(reference["status"], "available")
        self.assertEqual(
            reference["projection_format_histogram"],
            {
                "mode=affine bits=8 group_size=64": 2,
                "mode=affine bits=3 group_size=128": 2,
                "mode=affine bits=2 group_size=128": 2,
            },
        )
        higher = [
            (item["layer"], item["projection"])
            for item in reference["projections_allocated_higher_bits_than_modal"]
        ]
        # Modal bits = 2 (lowest of the tied counts); gate (8) and up (3) exceed it.
        self.assertEqual(higher, [(0, "gate"), (0, "up"), (1, "gate"), (1, "up")])
        self.assertIn("NOT measured sensitivity", reference["evidence_label"])
        self.assertEqual(
            reference["published_sensitive_projection_references"],
            ["layers.0.mlp.gate_proj"],
        )

    def test_precision_reference_unavailable_without_projections(self) -> None:
        baseline = baseline_payload({0: [0, 1, 2, 3]})
        reference = self.report_for(baseline, json.loads(json.dumps(baseline)))[
            "baseline_precision_reference"
        ]
        self.assertEqual(reference["status"], "unavailable")
        self.assertIn("no projection format data", reference["reason"])


class TestCliEndToEnd(unittest.TestCase):
    def test_cli_writes_report_and_unavailable_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.json"
            candidate_path = tmp_path / "candidate.json"
            output_path = tmp_path / "report.json"
            baseline_path.write_text(json.dumps(baseline_payload({0: [0, 1, 2, 3, 4, 5]})), encoding="utf-8")
            candidate_path.write_text(json.dumps(baseline_payload({0: [0, 1, 2, 3, 6, 7]})), encoding="utf-8")
            exit_code = main(
                [
                    "--baseline", str(baseline_path),
                    "--candidate", str(candidate_path),
                    "--output", str(output_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["comparison"]["layers_compared"], 1)
            self.assertAlmostEqual(report["per_layer"][0]["retained_jaccard"], 0.5)
            # The serialized report names the disagreeing experts, not only
            # their counts: baseline pruned {6,7}, candidate pruned {4,5}.
            self.assertEqual(report["per_layer"][0]["removed_both_expert_ids"], [])
            self.assertEqual(report["per_layer"][0]["baseline_only_removed_expert_ids"], [6, 7])
            self.assertEqual(report["per_layer"][0]["candidate_only_removed_expert_ids"], [4, 5])
            self.assertEqual(report["capabilities"]["status"], "unavailable")
            self.assertEqual(report["pairs"]["status"], "unavailable")

    def test_cli_validation_failure_does_not_write_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.json"
            baseline_path.write_text(json.dumps(baseline_payload({0: [0, 1]})), encoding="utf-8")
            candidate_path = tmp_path / "candidate.json"
            candidate_path.write_text(json.dumps(baseline_payload({1: [0, 1]})), encoding="utf-8")
            output_path = tmp_path / "report.json"
            exit_code = main(
                ["--baseline", str(baseline_path), "--candidate", str(candidate_path), "--output", str(output_path)]
            )
            self.assertEqual(exit_code, 2)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()