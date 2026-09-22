"""Regression tests for paired task-outcome evaluation."""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mimo_halo.evaluation.paired import (  # noqa: E402
    EvaluationError,
    analyze,
    exact_mcnemar_p,
    load_records,
    main,
)


def record(
    task_id,
    model_id,
    seed,
    success,
    *,
    turns=3,
    context_tokens=10000,
    language="python",
    domain="backend",
    catastrophic=False,
    failure_category=None,
    record_type=None,
    **extra,
):
    rec = {
        "schema_version": 1,
        "task_id": task_id,
        "model_id": model_id,
        "seed": seed,
        "success": success,
        "turns": turns,
        "context_tokens": context_tokens,
        "language": language,
        "domain": domain,
        "catastrophic": catastrophic,
        "failure_category": failure_category if failure_category is not None else ("none" if success else "implementation"),
    }
    if record_type is not None:
        rec["record_type"] = record_type
    rec.update(extra)
    return rec


def write_jsonl(records):
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    with handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    return handle.name


class PairedEvaluationTest(unittest.TestCase):
    def tearDown(self):
        for path in getattr(self, "_tempfiles", []):
            if os.path.exists(path):
                os.unlink(path)

    def keep(self, *paths):
        self._tempfiles = list(paths)
        return paths

    def analyze_files(self, ref_records, cand_records, **kwargs):
        ref_path = write_jsonl(ref_records)
        cand_path = write_jsonl(cand_records)
        self.keep(ref_path, cand_path)
        return analyze(
            load_records(ref_path),
            load_records(cand_path),
            reference_path=ref_path,
            candidate_path=cand_path,
            bootstrap_replicates=200,
            **kwargs,
        )

    # --- all win / all loss -------------------------------------------------

    def test_all_wins(self):
        ref = [record(f"t{i}", "ref-model", 0, False) for i in range(3)]
        cand = [record(f"t{i}", "cand-model", 0, True) for i in range(3)]
        report = self.analyze_files(ref, cand)
        self.assertEqual(report["task_comparison"], {"wins": 3, "losses": 0, "ties": 0})
        self.assertEqual(report["success"]["candidate"]["successes"], 3)
        self.assertEqual(report["success"]["reference"]["successes"], 0)
        # Zero-reference rule wins over a huge-looking ratio.
        self.assertIsNone(report["relative"]["ratio"])
        self.assertEqual(report["relative"]["ratio_status"], "not_applicable_reference_zero")
        # McNemar on 3 all-candidate-success pairs: p = 2 * C(3,0)/8 = 0.25.
        self.assertEqual(report["mcnemar"]["status"], "computed")
        self.assertEqual(report["mcnemar"]["discordant"]["reference_success_candidate_failure"], 0)
        self.assertEqual(report["mcnemar"]["discordant"]["reference_failure_candidate_success"], 3)
        self.assertAlmostEqual(report["mcnemar"]["p_exact_two_sided"], 0.25)

    def test_all_losses(self):
        ref = [record(f"t{i}", "ref-model", 0, True) for i in range(4)]
        cand = [record(f"t{i}", "cand-model", 0, False) for i in range(4)]
        report = self.analyze_files(ref, cand)
        self.assertEqual(report["task_comparison"], {"wins": 0, "losses": 4, "ties": 0})
        self.assertEqual(report["relative"]["ratio"], 0.0)
        self.assertEqual(report["relative"]["ratio_status"], "computed")
        self.assertEqual(report["catastrophes"]["candidate_increase"], False)

    # --- pairing failures ---------------------------------------------------

    def test_missing_pair_fails_without_report_missing(self):
        ref = [record("t1", "ref-model", 0, True)]
        cand = [
            record("t1", "cand-model", 0, True),
            record("t2", "cand-model", 0, False),
        ]
        with self.assertRaises(EvaluationError) as ctx:
            self.analyze_files(ref, cand)
        self.assertIn("t2", str(ctx.exception))
        self.assertIn("--report-missing", str(ctx.exception))

    def test_missing_pair_reported_with_flag(self):
        ref = [record("t1", "ref-model", 0, True)]
        cand = [
            record("t1", "cand-model", 0, True),
            record("t2", "cand-model", 0, False),
        ]
        report = self.analyze_files(ref, cand, report_missing=True)
        self.assertEqual(report["pairing"]["matched_pairs"], 1)
        self.assertEqual(report["pairing"]["missing_behavior"], "reported")
        # Reporting missing keys yields a matched-subset analysis: explicitly
        # incomplete evidence, not a substitute for the full paired run.
        self.assertEqual(report["pairing"]["evidence"], "incomplete_matched_subset_only")
        self.assertEqual(
            report["pairing"]["missing_pairs"],
            [{"task_id": "t2", "seed": 0, "missing_from": "reference"}],
        )
        # t2 must not silently enter any statistic.
        self.assertEqual(report["success"]["candidate"]["pairs"], 1)

    def test_duplicate_pair_fails(self):
        ref = [
            record("t1", "ref-model", 0, True),
            record("t1", "ref-model", 0, False),
        ]
        cand = [record("t1", "cand-model", 0, True)]
        with self.assertRaises(EvaluationError) as ctx:
            self.analyze_files(ref, cand)
        self.assertIn("duplicate", str(ctx.exception))

    def test_same_model_on_both_sides_fails(self):
        ref = [record("t1", "same-model", 0, True)]
        cand = [record("t1", "same-model", 0, False)]
        with self.assertRaises(EvaluationError):
            self.analyze_files(ref, cand)

    # --- seed clustering ----------------------------------------------------

    def test_seed_cluster_bootstrap_not_seed_independent(self):
        # One task measured at 10 seeds (5 successes each model). If seeds
        # were bootstrapped independently the CI would vary; clustering the
        # task makes every replicate the same cluster and the CI degenerate.
        ref = [record("t1", "ref-model", s, s % 2 == 0) for s in range(10)]
        cand = [record("t1", "cand-model", s, s % 2 == 0) for s in range(10)]
        report = self.analyze_files(ref, cand)
        self.assertEqual(report["pairing"]["matched_tasks"], 1)
        self.assertEqual(report["pairing"]["matched_pairs"], 10)
        self.assertEqual(report["task_comparison"], {"wins": 0, "losses": 0, "ties": 1})
        self.assertEqual(report["success"]["reference"]["task_mean_rate"], 0.5)
        self.assertEqual(report["success"]["reference"]["ci95"], [0.5, 0.5])
        self.assertEqual(report["success"]["candidate"]["ci95"], [0.5, 0.5])
        self.assertEqual(report["mcnemar"]["status"], "not_applicable")
        self.assertEqual(report["mcnemar"]["reason"], "repeated_seeds_per_task")

    def test_bootstrap_is_reproducible_and_seeded(self):
        ref = [record(f"t{i}", "ref-model", 0, i % 2 == 0) for i in range(6)]
        cand = [record(f"t{i}", "cand-model", 0, i % 3 != 0) for i in range(6)]
        first = self.analyze_files(ref, cand, bootstrap_seed=42)
        second = self.analyze_files(ref, cand, bootstrap_seed=42)
        self.assertEqual(first["success"]["reference"]["ci95"], second["success"]["reference"]["ci95"])
        self.assertEqual(first["relative"]["ci95"], second["relative"]["ci95"])
        self.assertEqual(first["provenance"]["bootstrap_seed"], 42)

    # --- tiny N / zero reference --------------------------------------------

    def test_tiny_n_zero_reference_ratio_null_not_inf(self):
        ref = [record("t1", "ref-model", 0, False), record("t2", "ref-model", 0, False)]
        cand = [record("t1", "cand-model", 0, True), record("t2", "cand-model", 0, True)]
        report = self.analyze_files(ref, cand)
        self.assertEqual(report["relative"]["ratio_status"], "not_applicable_reference_zero")
        self.assertIsNone(report["relative"]["ratio"])
        self.assertIsNone(report["relative"]["ci95"])
        # The fail-closed reason must reach the report consumer.
        self.assertTrue(report["relative"]["ci_note"])
        self.assertEqual(report["pairing"]["evidence"], "complete")
        self.assertNotEqual(math.isinf(report["relative"]["ratio"] or 0), True)

    def test_tiny_n_still_produces_ci(self):
        ref = [record("t1", "ref-model", 0, True), record("t2", "ref-model", 0, False)]
        cand = [record("t1", "cand-model", 0, True), record("t2", "cand-model", 0, True)]
        report = self.analyze_files(ref, cand)
        lo, hi = report["success"]["reference"]["ci95"]
        self.assertTrue(0.0 <= lo <= 0.75 <= hi <= 1.0)

    # --- exact McNemar ------------------------------------------------------

    def test_exact_mcnemar_values(self):
        self.assertAlmostEqual(exact_mcnemar_p(1, 3), 0.625)
        self.assertAlmostEqual(exact_mcnemar_p(0, 5), 0.0625)
        self.assertAlmostEqual(exact_mcnemar_p(1, 5), 0.21875)
        # Ties 2/2: two-sided exact tail exceeds 1 and is capped.
        self.assertEqual(exact_mcnemar_p(2, 2), 1.0)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(5, 0), 0.0625)

    def test_mcnemar_on_independent_pairs(self):
        # 4 tasks, single seed each: b=1 (ref pass, cand fail), c=3.
        ref = [
            record("t1", "ref-model", 0, True),
            record("t2", "ref-model", 0, False),
            record("t3", "ref-model", 0, False),
            record("t4", "ref-model", 0, False),
        ]
        cand = [
            record("t1", "cand-model", 0, False),
            record("t2", "cand-model", 0, True),
            record("t3", "cand-model", 0, True),
            record("t4", "cand-model", 0, True),
        ]
        report = self.analyze_files(ref, cand)
        self.assertEqual(report["mcnemar"]["status"], "computed")
        self.assertEqual(
            report["mcnemar"]["discordant"],
            {
                "reference_success_candidate_failure": 1,
                "reference_failure_candidate_success": 3,
            },
        )
        self.assertAlmostEqual(report["mcnemar"]["p_exact_two_sided"], 0.625)

    # --- catastrophes -------------------------------------------------------

    def test_catastrophe_increase_tracked(self):
        ref = [
            record("t1", "ref-model", 0, True),
            record("t2", "ref-model", 0, False, catastrophic=False),
        ]
        cand = [
            record("t1", "cand-model", 0, True, catastrophic=True, failure_category="runtime_corruption"),
            record("t2", "cand-model", 0, True),
        ]
        report = self.analyze_files(ref, cand)
        cats = report["catastrophes"]
        self.assertEqual(cats["reference"]["count"], 0)
        self.assertEqual(cats["candidate"]["count"], 1)
        self.assertAlmostEqual(cats["delta_rate"], 0.5)
        self.assertTrue(cats["candidate_increase"])

    # --- slices ---------------------------------------------------------------

    def test_slices_anchor_to_reference_population(self):
        ref = [
            record("t1", "ref-model", 0, True, language="python", turns=4, context_tokens=5000),
            record("t2", "ref-model", 0, False, language="rust", turns=6, context_tokens=200000),
        ]
        cand = [
            record("t1", "cand-model", 0, True, language="python", turns=7, context_tokens=90000),
            record("t2", "cand-model", 0, True, language="rust", turns=6, context_tokens=200000),
        ]
        report = self.analyze_files(ref, cand)
        py = report["slices"]["language"]["python"]
        self.assertEqual(py["reference"]["pairs"], 1)
        self.assertEqual(py["candidate"]["successes"], 1)
        # Behavioral regression: reference t1 took 4 turns, candidate t1 took
        # 7. Both sides land under the REFERENCE bucket 1-5, so every slice
        # row covers the identical paired population on both sides; the
        # candidate's own turns never move it into a different population.
        turns = report["slices"]["turns_bucket"]
        self.assertEqual(turns["1-5"]["reference"]["pairs"], 1)
        self.assertEqual(turns["1-5"]["candidate"]["pairs"], 1)
        self.assertEqual(turns["6-15"]["reference"]["pairs"], 1)
        self.assertEqual(turns["6-15"]["candidate"]["pairs"], 1)
        for stats in turns.values():
            self.assertEqual(stats["reference"]["pairs"], stats["candidate"]["pairs"])
        # The bucket disagreement is reported, not silently dropped.
        self.assertEqual(report["pairing"]["field_mismatches"]["turns_bucket"], 1)
        # Same anchoring for context buckets: reference t1 (5000 tokens) puts
        # both sides under <8K although the candidate used 90000 tokens.
        ctx = report["slices"]["context_bucket"]
        self.assertIn("<8K", ctx)
        self.assertEqual(ctx["<8K"]["reference"]["pairs"], 1)
        self.assertEqual(ctx["<8K"]["candidate"]["pairs"], 1)
        self.assertEqual(ctx["128K+"], {
            "reference": {"pairs": 1, "successes": 0, "catastrophes": 0, "rate": 0.0},
            "candidate": {"pairs": 1, "successes": 1, "catastrophes": 0, "rate": 1.0},
        })
        self.assertNotIn("64-128K", ctx)
        self.assertEqual(report["pairing"]["field_mismatches"]["context_bucket"], 1)
        self.assertEqual(report["provenance"]["slice_anchor"], "reference_record")

    def test_failed_zero_turn_record_kept_in_overall_stats_and_own_slice(self):
        ref = [
            record("t1", "ref-model", 0, False, turns=0),
            record("t2", "ref-model", 0, True, turns=3),
        ]
        cand = [
            record("t1", "cand-model", 0, False, turns=0),
            record("t2", "cand-model", 0, True, turns=3),
        ]
        report = self.analyze_files(ref, cand)
        # A zero-turn runtime failure is a real failure: it stays in the
        # overall paired population (dropping it would inflate pass rates).
        self.assertEqual(report["pairing"]["matched_pairs"], 2)
        self.assertEqual(report["success"]["reference"]["pairs"], 2)
        self.assertEqual(report["success"]["reference"]["successes"], 1)
        self.assertEqual(report["success"]["reference"]["pair_success_rate"], 0.5)
        self.assertEqual(report["task_comparison"], {"wins": 0, "losses": 0, "ties": 2})
        # ...and it gets its own explicit slice bucket.
        zero = report["slices"]["turns_bucket"]["0 (no agent turn)"]
        self.assertEqual(
            zero["reference"], {"pairs": 1, "successes": 0, "catastrophes": 0, "rate": 0.0}
        )
        self.assertEqual(zero["candidate"]["pairs"], 1)
        self.assertIn("1-5", report["slices"]["turns_bucket"])
        self.assertNotIn("0-5", report["slices"]["turns_bucket"])

    # --- negative controls --------------------------------------------------

    def test_negative_controls_excluded_from_main_stats(self):
        ref = [
            record("t1", "ref-model", 0, True),
            record("ctl1", "ref-model", 0, False, record_type="negative_control"),
        ]
        cand = [
            record("t1", "cand-model", 0, True),
            record("ctl1", "cand-model", 0, True, record_type="negative_control"),
        ]
        report = self.analyze_files(ref, cand)
        self.assertEqual(report["pairing"]["matched_pairs"], 1)
        self.assertEqual(report["negative_controls"]["reference"]["records"], 1)
        self.assertEqual(report["negative_controls"]["candidate"]["pass_rate"], 1.0)

    def test_duplicate_control_key_fails(self):
        ref = [record("t1", "ref-model", 0, True)]
        cand = [
            record("t1", "cand-model", 0, True),
            record("ctl", "cand-model", 0, False, record_type="negative_control"),
            record("ctl", "cand-model", 0, True, record_type="negative_control"),
        ]
        with self.assertRaises(EvaluationError) as ctx:
            self.analyze_files(ref, cand)
        self.assertIn("duplicate", str(ctx.exception))

    def test_control_colliding_with_task_key_fails(self):
        ref = [record("t1", "ref-model", 0, True)]
        cand = [
            record("t1", "cand-model", 0, True),
            record("t1", "cand-model", 0, False, record_type="negative_control"),
        ]
        with self.assertRaises(EvaluationError) as ctx:
            self.analyze_files(ref, cand)
        self.assertIn("duplicate", str(ctx.exception))

    def test_foreign_model_in_control_fails(self):
        ref = [record("t1", "ref-model", 0, True)]
        cand = [
            record("t1", "cand-model", 0, True),
            record("ctl", "other-model", 0, False, record_type="negative_control"),
        ]
        with self.assertRaises(EvaluationError) as ctx:
            self.analyze_files(ref, cand)
        self.assertIn("model_id", str(ctx.exception))

    # --- validation ---------------------------------------------------------

    def test_load_records_fail_closed(self):
        base = record("t1", "m", 0, True)
        cases = [
            ("missing field", {k: v for k, v in base.items() if k != "success"}),
            ("success not bool", {**base, "success": "yes"}),
            ("seed not int", {**base, "seed": "0"}),
            ("bool turns", {**base, "turns": True}),
            ("negative turns", {**base, "turns": -1}),
            ("zero-turn success", {**base, "turns": 0, "success": True}),
            ("string token metric", {**base, "tokens": {"generated_tokens": "900"}}),
            ("float token metric", {**base, "tokens": {"generated_tokens": 1.5}}),
            ("bool timing metric", {**base, "timings": {"wall_seconds": True}}),
            ("nonfinite timing metric", {**base, "timings": {"wall_seconds": float("nan")}}),
            ("negative timing metric", {**base, "timings": {"wall_seconds": -1.0}}),
            ("string tool metric", {**base, "toolmetrics": {"tool_calls": "4"}}),
            ("negative tool metric", {**base, "toolmetrics": {"tool_calls": -1}}),
            ("wrong schema_version", {**base, "schema_version": 2}),
            ("bad failure_category", {**base, "failure_category": "vibes"}),
            ("bad language", {**base, "language": "cobol"}),
            ("bad domain", {**base, "domain": "blockchain"}),
            ("bad record_type", {**base, "record_type": "golden"}),
            ("optional object", {**base, "tokens": [1, 2]}),
        ]
        for label, bad in cases:
            path = write_jsonl([bad])
            self.keep(path)
            with self.assertRaises(EvaluationError, msg=label):
                load_records(path)

    def test_optional_fields_accepted(self):
        path = write_jsonl(
            [record("t1", "m", 0, True, timings={"wall_seconds": 12.5}, tokens={"generated_tokens": 900}, toolmetrics={"tool_calls": 4, "bad_tool_calls": 0})]
        )
        self.keep(path)
        recs = load_records(path)
        self.assertEqual(recs[0]["toolmetrics"]["bad_tool_calls"], 0)

    def test_empty_file_fails(self):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        handle.close()
        self.keep(handle.name)
        with self.assertRaises(EvaluationError):
            load_records(handle.name)

    # --- CLI ---------------------------------------------------------------

    def test_cli_smoke(self):
        ref_path = write_jsonl(
            [record(f"t{i}", "ref-model", 0, i % 2 == 0) for i in range(5)]
        )
        cand_path = write_jsonl(
            [record(f"t{i}", "cand-model", 0, i % 2 == 1) for i in range(5)]
        )
        out_fd, out_path = tempfile.mkstemp(suffix=".json")
        os.close(out_fd)
        self.keep(ref_path, cand_path, out_path)
        rc = main(
            [
                "--reference",
                ref_path,
                "--candidate",
                cand_path,
                "--output",
                out_path,
                "--bootstrap-replicates",
                "100",
            ]
        )
        self.assertEqual(rc, 0)
        with open(out_path, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["analysis"], "paired_task_outcome")
        self.assertEqual(report["pairing"]["matched_pairs"], 5)
        self.assertEqual(
            report["task_comparison"],
            {"wins": 2, "losses": 3, "ties": 0},
        )
        self.assertEqual(report["mcnemar"]["status"], "computed")

    def test_cli_fails_closed_on_missing_pairs(self):
        ref_path = write_jsonl([record("t1", "ref-model", 0, True)])
        cand_path = write_jsonl(
            [record("t1", "cand-model", 0, True), record("t2", "cand-model", 0, True)]
        )
        self.keep(ref_path, cand_path)
        rc = main(["--reference", ref_path, "--candidate", cand_path, "--output", "-"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()