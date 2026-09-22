"""Tests for the fail-closed production quality gates."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from mimo_halo.evaluation.gates import (
    DEFAULT_THRESHOLDS,
    check_gates,
    main,
)


def passing_record() -> dict:
    return {
        "measured": True,
        "metrics": {
            "long_agent_retention": 0.94,
            "short_coding_retention": 0.95,
            "repo_tool_retention": 0.92,
            "resident_weight_bytes": 96_636_764_160,
            "c8_aggregate_tok_s": 68.5,
        },
        "structural_integrity_suite_passed": True,
        "no_known_numerical_correctness_bug": True,
        "tool_protocol_essentially_unchanged": True,
        "stable_overnight_c8": True,
        "catastrophic_failures_not_increased": True,
    }


def verdicts(result: dict) -> dict:
    return {g["gate"]: g["status"] for g in result["gates"]}


class PromotionTests(unittest.TestCase):
    def test_all_pass_promotes(self):
        result = check_gates(passing_record())
        self.assertTrue(result["promotable"])
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(set(verdicts(result).values()), {"pass"})

    def test_estimate_never_passes(self):
        for record in ({}, {"measured": False}, {"measured": None}):
            result = check_gates(record)
            self.assertFalse(result["promotable"])
            self.assertEqual(result["verdict"], "unmeasured")

    def test_fail_blocks_even_if_everything_else_passes(self):
        record = passing_record()
        record["metrics"]["long_agent_retention"] = 0.89
        result = check_gates(record)
        self.assertFalse(result["promotable"])
        self.assertEqual(result["verdict"], "fail")
        self.assertEqual(verdicts(result)["long_agent_retention"], "fail")

    def test_unmeasured_blocks_promotion(self):
        record = passing_record()
        del record["metrics"]["c8_aggregate_tok_s"]
        result = check_gates(record)
        self.assertFalse(result["promotable"])
        self.assertEqual(result["verdict"], "unmeasured")


class BoundaryTests(unittest.TestCase):
    def test_retention_floors_are_inclusive(self):
        record = passing_record()
        record["metrics"].update(
            long_agent_retention=0.90,
            short_coding_retention=0.93,
            repo_tool_retention=0.90,
        )
        self.assertEqual(check_gates(record)["verdict"], "pass")
        record["metrics"]["short_coding_retention"] = 0.929999
        self.assertEqual(check_gates(record)["verdict"], "fail")

    def test_weight_band_boundaries(self):
        record = passing_record()
        record["metrics"]["resident_weight_bytes"] = DEFAULT_THRESHOLDS["weight_bytes_min"]
        self.assertEqual(check_gates(record)["verdict"], "pass")
        record["metrics"]["resident_weight_bytes"] = DEFAULT_THRESHOLDS["weight_bytes_max"] + 1
        self.assertEqual(check_gates(record)["verdict"], "fail")

    def test_flag_false_is_fail_missing_is_unmeasured(self):
        record = passing_record()
        record["stable_overnight_c8"] = False
        self.assertEqual(verdicts(check_gates(record))["stable_overnight_c8"], "fail")
        record = passing_record()
        del record["stable_overnight_c8"]
        self.assertEqual(verdicts(check_gates(record))["stable_overnight_c8"], "unmeasured")

    def test_threshold_override_applies(self):
        record = passing_record()
        record["metrics"]["c8_aggregate_tok_s"] = 60.0
        self.assertEqual(check_gates(record)["verdict"], "fail")
        self.assertEqual(
            check_gates(record, {"c8_aggregate_tok_s_floor": 55.0})["verdict"], "pass"
        )


class CliTests(unittest.TestCase):
    def _run(self, record: dict) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "candidate.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(record, fh)
            return main(["--candidate", path])

    def test_exit_codes(self):
        self.assertEqual(self._run(passing_record()), 0)
        failing = passing_record()
        failing["metrics"]["long_agent_retention"] = 0.5
        self.assertEqual(self._run(failing), 1)
        unmeasured = passing_record()
        del unmeasured["metrics"]["c8_aggregate_tok_s"]
        self.assertEqual(self._run(unmeasured), 2)


if __name__ == "__main__":
    unittest.main()
