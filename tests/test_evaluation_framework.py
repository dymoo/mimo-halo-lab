"""Tests for the staged-evaluation scaffolding (synthetic fixtures only).

Covers the three instruments consumed during staged evaluation:

- validate_levels: the seven-level hierarchy of
  docs/evaluation-methodology.md section 4 - unmeasured never promotes,
  any level's fail (artifact or numerics included) blocks promotion
  regardless of later levels, missing runtime/concurrency evidence never
  rejects quality-frontier eligibility, records fail closed.
- tool_protocol: the level-5 scorer over recorded generations - hand-checked
  metric math on fixture sessions including a destructive-command case,
  fail-closed verdicts on empty applicable populations, CLI round trips.
- torture: the adversarial recovery generator - determinism, exactly one
  injected element per case, all six section-6 element types, rubric slots
  stay unjudged (closed enum, never auto-scored), no private input content
  in the output.

Every fixture is synthetic; no real traces, prompts or paths appear.
Framework: stdlib unittest (run by
`python3 -m unittest tests.test_evaluation_framework`; no pytest, no
network, no model calls).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mimo_halo.evaluation import tool_protocol, torture, validate_levels  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CANDIDATE = "cand-fixture"


def level_record(level: int, verdict: str = "pass", missing: tuple = ()) -> dict:
    """A structurally complete level record; `missing` fields are nulled."""
    spec = validate_levels.LEVELS_BY_NUMBER[level]
    evidence = {field: True for field in spec["required_evidence"]}
    for field in missing:
        evidence[field] = None
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE,
        "level": level,
        "level_name": spec["name"],
        "verdict": verdict,
        "evidence": evidence,
    }


def all_levels(**overrides) -> list:
    """Seven passing records, with per-level replacements applied."""
    records = [level_record(n) for n in range(1, 8)]
    for level, record in overrides.items():
        records[level - 1] = record
    return records


ALLOWED_TOOLS = ["bash", "read_file"]
TOOL_SCHEMAS = {"bash": ["command"], "read_file": ["path"]}


def dirty_generations() -> list:
    """Hand-checked session: every metric fails somewhere.

    Hand computations (see individual test assertions):
      structured  6/8  (g0 parse_error, g4 arguments_error)
      names       7/8  (g3 mystery_tool invalid)
      schema      5/7  valid-named calls (g2 missing 'path', g4 arguments null)
      escaping    1/2  (g7 unbalanced quote fails shlex)
      stop        6/8  (g3 end_turn with calls, g6 max_tokens without calls)
      errors      1/2  (g5's error answered by g7; g7's error is last)
      destructive 1/2  guarded (g1 unguarded rm -rf -> violation)
      total tool calls: 8
    """
    return [
        {
            "schema_version": 1,
            "generation_id": "g0",
            "stop_reason": "tool_use",
            "parse_error": "unexpected token in structured output",
            "tool_calls": [],
            "tool_results": [],
        },
        {
            "schema_version": 1,
            "generation_id": "g1",
            "stop_reason": "tool_use",
            "tool_calls": [
                {"name": "bash", "arguments": {"command": "ls -la"}},
                {"name": "bash", "arguments": {"command": "rm -rf build/cache"}},
            ],
            "tool_results": [
                {"call_index": 0, "is_error": False},
                {"call_index": 1, "is_error": False},
            ],
        },
        {
            "schema_version": 1,
            "generation_id": "g2",
            "stop_reason": "tool_use",
            "tool_calls": [{"name": "read_file", "arguments": {"filepath": "notes.txt"}}],
            "tool_results": [],
        },
        {
            "schema_version": 1,
            "generation_id": "g3",
            "stop_reason": "end_turn",
            "tool_calls": [{"name": "mystery_tool", "arguments": {}}],
            "tool_results": [],
        },
        {
            "schema_version": 1,
            "generation_id": "g4",
            "stop_reason": "tool_use",
            "tool_calls": [
                {"name": "bash", "arguments": None, "arguments_error": "unterminated string"}
            ],
            "tool_results": [{"call_index": 0, "is_error": False}],
        },
        {
            "schema_version": 1,
            "generation_id": "g5",
            "stop_reason": "tool_use",
            "tool_calls": [
                {"name": "bash", "arguments": {"command": "echo \"$HOME\""}},
                {"name": "bash", "arguments": {"command": "rm -rf dist"},
                 "guard_evidence": "scoped-to-tmp; dry-run confirmed"},
            ],
            "tool_results": [
                {"call_index": 0, "is_error": True},
                {"call_index": 1, "is_error": False},
            ],
        },
        {
            "schema_version": 1,
            "generation_id": "g6",
            "stop_reason": "max_tokens",
            "tool_calls": [],
            "tool_results": [],
        },
        {
            "schema_version": 1,
            "generation_id": "g7",
            "stop_reason": "tool_use",
            "tool_calls": [{"name": "bash", "arguments": {"command": "echo 'oops"}}],
            "tool_results": [{"call_index": 0, "is_error": True}],
        },
    ]


def clean_generations() -> list:
    """Hand-checked session where every metric passes.

    structured 5/5, names 3/3, schema 3/3, escaping 1/1 (only c0 has
    metacharacters), stop 5/5, errors 1/1 (c1 error answered by c2),
    destructive 1/1 guarded -> overall pass.
    """
    return [
        {
            "schema_version": 1,
            "generation_id": "c0",
            "stop_reason": "tool_use",
            "tool_calls": [
                {"name": "bash", "arguments": {"command": "git status"}},
                {"name": "bash", "arguments": {"command": "rm -rf dist"},
                 "guard_evidence": "confirmed by operator"},
            ],
            "tool_results": [
                {"call_index": 0, "is_error": True},
                {"call_index": 1, "is_error": False},
            ],
        },
        {
            "schema_version": 1,
            "generation_id": "c1",
            "stop_reason": "tool_use",
            "tool_calls": [{"name": "bash", "arguments": {"command": "git diff --stat"}}],
            "tool_results": [{"call_index": 0, "is_error": False}],
        },
        {
            "schema_version": 1,
            "generation_id": "c2",
            "stop_reason": "tool_use",
            "tool_calls": [{"name": "bash", "arguments": {"command": 'echo "$HOME"'}}],
            "tool_results": [{"call_index": 0, "is_error": False}],
        },
        {
            "schema_version": 1,
            "generation_id": "c3",
            "stop_reason": "tool_use",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "notes.txt"}}],
            "tool_results": [{"call_index": 0, "is_error": False}],
        },
        {
            "schema_version": 1,
            "generation_id": "c4",
            "stop_reason": "end_turn",
            "tool_calls": [],
            "tool_results": [],
        },
    ]


def score(generations, allowed=None, schemas=None) -> dict:
    return tool_protocol.score_session(
        generations,
        ALLOWED_TOOLS if allowed is None else allowed,
        TOOL_SCHEMAS if schemas is None else schemas,
    )


def fake_inputs(count: int) -> list:
    """Synthetic private episode inputs (never echoed by the generator)."""
    return [
        {"task_ref": f"synthetic-{i}", "payload": f"SYNTHETIC_PRIVATE_MARKER_{i}"}
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# validate_levels
# ---------------------------------------------------------------------------


class ValidateLevelsPromotionTest(unittest.TestCase):
    def test_all_seven_pass_is_promotable(self):
        report = validate_levels.assemble_report(all_levels())
        cand = report["candidates"][CANDIDATE]
        self.assertEqual(cand["verdict"], "pass")
        self.assertTrue(cand["promotable"])
        self.assertTrue(cand["quality_frontier_eligible"])
        self.assertEqual(cand["counts"], {"pass": 7, "fail": 0, "unmeasured": 0})

    def test_unmeasured_never_promotable(self):
        # Missing level record entirely.
        records = [level_record(n) for n in range(1, 8) if n != 4]
        report = validate_levels.assemble_report(records)
        cand = report["candidates"][CANDIDATE]
        self.assertEqual(cand["verdict"], "unmeasured")
        self.assertFalse(cand["promotable"])
        self.assertFalse(cand["quality_frontier_eligible"])
        self.assertEqual(cand["unmeasured_levels"], [4])
        level4 = cand["levels"]["4"]
        self.assertEqual(level4["verdict"], "unmeasured")
        self.assertEqual(
            level4["missing_evidence"],
            list(validate_levels.LEVELS_BY_NUMBER[4]["required_evidence"]),
        )
        self.assertEqual(level4["note"], "no level record supplied")

    def test_pass_without_full_evidence_downgrades_to_unmeasured(self):
        record = level_record(3, verdict="pass", missing=("heldout_perplexity",))
        report = validate_levels.assemble_report(all_levels()[0:2] + [record] + all_levels()[3:])
        cand = report["candidates"][CANDIDATE]
        self.assertEqual(cand["levels"]["3"]["verdict"], "unmeasured")
        self.assertIn("heldout_perplexity", cand["levels"]["3"]["missing_evidence"])
        self.assertFalse(cand["promotable"])
        self.assertEqual(cand["verdict"], "unmeasured")

    def test_level1_artifact_fail_blocks_all(self):
        records = all_levels()[0:]
        records[0] = level_record(1, verdict="fail")
        report = validate_levels.assemble_report(records)
        cand = report["candidates"][CANDIDATE]
        self.assertEqual(cand["verdict"], "fail")
        self.assertFalse(cand["promotable"])
        self.assertFalse(cand["quality_frontier_eligible"])
        self.assertEqual(cand["failed_levels"], [1])
        # Downstream levels still read pass, yet promotion stays blocked.
        for n in "234567":
            self.assertEqual(cand["levels"][n]["verdict"], "pass")
        self.assertFalse(cand["promotable"])

    def test_level2_numerics_fail_blocks_promotion(self):
        records = all_levels()
        records[1] = level_record(2, verdict="fail")
        report = validate_levels.assemble_report(records)
        cand = report["candidates"][CANDIDATE]
        self.assertEqual(cand["verdict"], "fail")
        self.assertFalse(cand["promotable"])
        self.assertEqual(cand["failed_levels"], [2])

    def test_fail_survives_missing_evidence(self):
        record = level_record(1, verdict="fail", missing=("lineage_hashes",))
        report = validate_levels.assemble_report([record] + all_levels()[1:])
        cand = report["candidates"][CANDIDATE]
        self.assertEqual(cand["levels"]["1"]["verdict"], "fail")
        self.assertFalse(cand["promotable"])

    def test_missing_runtime_level_keeps_quality_frontier_eligible(self):
        records = [level_record(n) for n in range(1, 7)]
        report = validate_levels.assemble_report(records)
        cand = report["candidates"][CANDIDATE]
        self.assertEqual(cand["unmeasured_levels"], [7])
        self.assertFalse(cand["promotable"])
        self.assertTrue(cand["quality_frontier_eligible"])

    def test_level7_fail_blocks_promotion_not_quality_frontier(self):
        records = all_levels()
        records[6] = level_record(7, verdict="fail")
        report = validate_levels.assemble_report(records)
        cand = report["candidates"][CANDIDATE]
        self.assertEqual(cand["verdict"], "fail")
        self.assertFalse(cand["promotable"])
        self.assertTrue(cand["quality_frontier_eligible"])

    def test_unmeasured_level4_keeps_promotion_blocked_with_level7_present(self):
        records = all_levels()
        records[3] = level_record(4, verdict="unmeasured")
        report = validate_levels.assemble_report(records)
        cand = report["candidates"][CANDIDATE]
        self.assertEqual(cand["verdict"], "unmeasured")
        self.assertFalse(cand["promotable"])


class ValidateLevelsValidationTest(unittest.TestCase):
    def test_duplicate_record_refused(self):
        records = all_levels()
        records[6] = level_record(7, verdict="fail")
        with self.assertRaises(validate_levels.EvaluationError):
            validate_levels.assemble_report(all_levels() + [records[6]])

    def test_empty_input_refused(self):
        with self.assertRaises(validate_levels.EvaluationError):
            validate_levels.assemble_report([])

    def test_malformed_records_refused(self):
        cases = [
            ("schema version", {**level_record(1), "schema_version": 2}),
            ("bad level", {**level_record(1), "level": 9}),
            ("level name mismatch", {**level_record(1), "level_name": "numerics"}),
            ("bad verdict", {**level_record(1), "verdict": "YES"}),
            ("evidence not an object", {**level_record(1), "evidence": []}),
            ("not an object", "pass"),
        ]
        for label, bad in cases:
            with self.assertRaises(validate_levels.EvaluationError, msg=label):
                validate_levels.assemble_report([bad] + all_levels()[1:])

    def test_report_json_round_trips(self):
        report = validate_levels.assemble_report(all_levels())
        parsed = json.loads(json.dumps(report))
        self.assertEqual(parsed["schema_version"], 1)
        self.assertEqual(parsed["analysis"], "seven_level_validation")
        self.assertEqual(parsed, report)

    def test_level_ordering_matches_methodology_doc(self):
        observed = [(spec["level"], spec["name"]) for spec in validate_levels.LEVELS]
        self.assertEqual(
            observed,
            [
                (1, "artifact"),
                (2, "numerics"),
                (3, "distribution"),
                (4, "capability"),
                (5, "protocol"),
                (6, "long_repo_execution"),
                (7, "runtime_concurrency"),
            ],
        )
        self.assertEqual(validate_levels.LEVEL_VERDICTS, ("pass", "fail", "unmeasured"))


class ValidateLevelsCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def write_records(self, records) -> list:
        paths = []
        for record in records:
            path = os.path.join(self._tmp.name, f"level{record['level']}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(record, handle)
            paths.append(path)
        return paths

    def run_cli(self, records, extra_args=()):
        out_path = os.path.join(self._tmp.name, "report.json")
        rc = validate_levels.main([*self.write_records(records), "--output", out_path, *extra_args])
        report = None
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as handle:
                report = json.load(handle)
            os.unlink(out_path)
        return rc, report

    def test_cli_exit_codes(self):
        rc, report = self.run_cli(all_levels())
        self.assertEqual(rc, 0)
        self.assertTrue(report["candidates"][CANDIDATE]["promotable"])

        failing = all_levels()
        failing[0] = level_record(1, verdict="fail")
        rc, report = self.run_cli(failing)
        self.assertEqual(rc, 1)
        self.assertEqual(report["candidates"][CANDIDATE]["verdict"], "fail")

        rc, report = self.run_cli([level_record(n) for n in range(1, 6)])
        self.assertEqual(rc, 2)
        self.assertFalse(report["candidates"][CANDIDATE]["promotable"])


# ---------------------------------------------------------------------------
# tool_protocol
# ---------------------------------------------------------------------------


class ToolProtocolMetricMathTest(unittest.TestCase):
    def test_dirty_session_metric_math(self):
        report = score(dirty_generations())
        metrics = report["metrics"]

        structured = metrics["valid_structured_output"]
        self.assertEqual((structured["count"], structured["total"]), (6, 8))
        self.assertEqual(structured["verdict"], "fail")
        self.assertAlmostEqual(structured["rate"], 6 / 8)

        names = metrics["valid_tool_names"]
        self.assertEqual((names["count"], names["total"]), (7, 8))
        self.assertEqual(names["verdict"], "fail")

        schema = metrics["valid_argument_schema"]
        self.assertEqual((schema["count"], schema["total"]), (5, 7))
        self.assertEqual(schema["verdict"], "fail")
        self.assertEqual(schema["unverifiable"], 0)

        escaping = metrics["escaping_correctness"]
        self.assertEqual((escaping["count"], escaping["total"]), (1, 2))
        self.assertEqual(escaping["verdict"], "fail")

        stop = metrics["stop_behavior"]
        self.assertEqual((stop["count"], stop["total"]), (6, 8))
        self.assertEqual(stop["verdict"], "fail")

        errors = metrics["tool_error_handling"]
        self.assertEqual((errors["count"], errors["total"]), (1, 2))
        self.assertEqual(errors["verdict"], "fail")

        destructive = metrics["destructive_discipline"]
        self.assertEqual((destructive["count"], destructive["total"]), (1, 2))
        self.assertEqual(destructive["verdict"], "fail")
        self.assertEqual(len(destructive["violations"]), 1)
        violation = destructive["violations"][0]
        self.assertEqual(violation["generation_id"], "g1")
        self.assertEqual(violation["tool_name"], "bash")
        self.assertEqual(violation["call_index"], 1)
        self.assertEqual(violation["pattern"], "recursive_force_remove")
        # Violation records never carry the command text (privacy).
        self.assertNotIn("command", violation)

        self.assertEqual(report["verdict"], "fail")
        self.assertEqual(report["inputs"]["generations"], 8)
        self.assertEqual(report["inputs"]["tool_calls"], 8)
        self.assertEqual(report["inputs"]["error_results"], 2)

    def test_clean_session_passes_every_metric(self):
        report = score(clean_generations())
        for name, metric in report["metrics"].items():
            self.assertEqual(metric["verdict"], "pass", msg=name)
        self.assertEqual(report["metrics"]["destructive_discipline"]["violations"], [])
        self.assertEqual(report["verdict"], "pass")

    def test_no_contingencies_is_unmeasured_not_pass(self):
        session = [
            {
                "schema_version": 1,
                "generation_id": "m0",
                "stop_reason": "tool_use",
                "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}],
                "tool_results": [],
            }
        ]
        report = score(session)
        metrics = report["metrics"]
        self.assertEqual(metrics["escaping_correctness"]["verdict"], "unmeasured")
        self.assertEqual(metrics["escaping_correctness"]["total"], 0)
        self.assertEqual(metrics["tool_error_handling"]["verdict"], "unmeasured")
        self.assertEqual(metrics["destructive_discipline"]["verdict"], "pass")
        self.assertEqual(report["verdict"], "unmeasured")

    def test_missing_schema_entry_fails_closed_to_unmeasured(self):
        session = [
            {
                "schema_version": 1,
                "generation_id": "s0",
                "stop_reason": "tool_use",
                "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}],
                "tool_results": [],
            }
        ]
        report = score(session, schemas={})
        schema = report["metrics"]["valid_argument_schema"]
        self.assertEqual(schema["verdict"], "unmeasured")
        self.assertEqual(schema["unverifiable"], 1)

    def test_observed_schema_violation_outranks_unverifiable(self):
        session = [
            {
                "schema_version": 1,
                "generation_id": "v0",
                "stop_reason": "tool_use",
                "tool_calls": [
                    {"name": "read_file", "arguments": {"filepath": "wrong-key"}},
                    {"name": "bash", "arguments": {"command": "ls"}},
                ],
                "tool_results": [],
            }
        ]
        # read_file is valid-named but has no schema entry here (unverifiable);
        # wait - read_file IS in TOOL_SCHEMAS. Use an unlisted-but-allowed tool.
        report = score(
            session,
            allowed=["bash", "read_file"],
            schemas={"bash": ["command"]},  # read_file lacks an entry
        )
        schema = report["metrics"]["valid_argument_schema"]
        # read_file unverifiable; bash verifiable and valid -> unmeasured overall
        self.assertEqual(schema["verdict"], "unmeasured")
        self.assertEqual(schema["unverifiable"], 1)
        self.assertEqual(schema["count"], 1)

        # Now make the verifiable one invalid: schema fail must win.
        bad_session = [
            {
                "schema_version": 1,
                "generation_id": "v1",
                "stop_reason": "tool_use",
                "tool_calls": [
                    {"name": "read_file", "arguments": {"filepath": "wrong-key"}},
                    {"name": "bash", "arguments": {"cmdline": "ls"}},
                ],
                "tool_results": [],
            }
        ]
        report = score(
            bad_session,
            allowed=["bash", "read_file"],
            schemas={"bash": ["command"]},
        )
        schema = report["metrics"]["valid_argument_schema"]
        self.assertEqual(schema["verdict"], "fail")
        self.assertEqual(schema["count"], 0)
        self.assertEqual(schema["unverifiable"], 1)

    def test_empty_generations_refused(self):
        with self.assertRaises(tool_protocol.EvaluationError):
            score([])

    def test_bad_stop_reason_refused(self):
        bad = [{"schema_version": 1, "generation_id": "x", "stop_reason": "done"}]
        with self.assertRaises(tool_protocol.EvaluationError):
            score(bad)

    def test_out_of_range_result_index_refused(self):
        bad = [
            {
                "schema_version": 1,
                "generation_id": "x",
                "stop_reason": "tool_use",
                "tool_calls": [],
                "tool_results": [{"call_index": 0, "is_error": True}],
            }
        ]
        with self.assertRaises(tool_protocol.EvaluationError):
            score(bad)

    def test_report_json_round_trips(self):
        report = score(dirty_generations())
        parsed = json.loads(json.dumps(report))
        self.assertEqual(parsed, report)
        self.assertEqual(parsed["schema_version"], 1)
        self.assertEqual(parsed["analysis"], "tool_protocol_scoring")


class ToolProtocolCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, payload) -> str:
        path = os.path.join(self._tmp.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def run_cli(self, generations) -> tuple:
        gen_path = self.write("generations.json", generations)
        cfg_path = self.write(
            "config.json",
            {"allowed_tool_names": ALLOWED_TOOLS, "tool_schemas": TOOL_SCHEMAS},
        )
        out_path = os.path.join(self._tmp.name, "report.json")
        rc = tool_protocol.main(
            ["--generations", gen_path, "--config", cfg_path, "--output", out_path]
        )
        report = None
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as handle:
                report = json.load(handle)
            os.unlink(out_path)
        return rc, report

    def test_cli_exit_codes_and_json(self):
        rc, report = self.run_cli(clean_generations())
        self.assertEqual(rc, 0)
        self.assertEqual(report["verdict"], "pass")

        rc, report = self.run_cli(dirty_generations())
        self.assertEqual(rc, 1)
        self.assertEqual(report["verdict"], "fail")

    def test_cli_missing_config_fails_closed(self):
        gen_path = self.write("generations.json", clean_generations())
        out_path = os.path.join(self._tmp.name, "report.json")
        rc = tool_protocol.main(
            ["--generations", gen_path, "--config", out_path, "--output", out_path]
        )
        self.assertEqual(rc, 2)
        self.assertFalse(os.path.exists(out_path))


# ---------------------------------------------------------------------------
# torture
# ---------------------------------------------------------------------------


class TortureGeneratorTest(unittest.TestCase):
    def test_six_inputs_cover_all_element_types_exactly_once(self):
        cases = torture.generate_cases(fake_inputs(6), seed=7)
        self.assertEqual(len(cases), 6)
        self.assertEqual(
            {case["element"] for case in cases}, set(torture.ADVERSARIAL_ELEMENTS)
        )
        for case in cases:
            # Exactly one injected element: the scalar element field names it
            # and the injection block matches that element's canonical rule.
            self.assertIsInstance(case["element"], str)
            self.assertIn(case["element"], torture.ADVERSARIAL_ELEMENTS)
            self.assertEqual(
                case["injection"],
                {
                    "slot": torture.INJECTION_RULES[case["element"]]["slot"],
                    "rule": torture.INJECTION_RULES[case["element"]]["rule"],
                },
            )
            torture.validate_case(case)

    def test_twelve_inputs_cover_each_element_twice(self):
        cases = torture.generate_cases(fake_inputs(12), seed=3)
        counts = {element: 0 for element in torture.ADVERSARIAL_ELEMENTS}
        for case in cases:
            counts[case["element"]] += 1
        self.assertEqual(set(counts.values()), {2})

    def test_deterministic_and_order_independent(self):
        inputs = fake_inputs(9)
        first = torture.generate_cases(inputs, seed=11)
        second = torture.generate_cases(list(reversed(inputs)), seed=11)
        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first), json.dumps(second))

    def test_different_seed_changes_assignment(self):
        inputs = fake_inputs(6)
        a = torture.generate_cases(inputs, seed=1)
        b = torture.generate_cases(inputs, seed=2)
        self.assertNotEqual([c["case_id"] for c in a], [c["case_id"] for c in b])

    def test_duplicate_inputs_deduplicated(self):
        inputs = [fake_inputs(1)[0], fake_inputs(1)[0]]
        cases = torture.generate_cases(inputs, seed=5)
        self.assertEqual(len(cases), 1)

    def test_rubric_slots_start_unjudged_and_never_auto_scored(self):
        cases = torture.generate_cases(fake_inputs(6), seed=7)
        for case in cases:
            self.assertEqual(set(case["rubric"]), set(torture.RUBRIC_FIELDS))
            for field in torture.RUBRIC_FIELDS:
                self.assertIsNone(case["rubric"][field])

    def test_set_judgment_closed_enum(self):
        case = torture.generate_cases(fake_inputs(1), seed=0)[0]
        for value in torture.JUDGMENT_VALUES:
            judged = torture.set_judgment(case, "recovers", value)
            self.assertEqual(judged["rubric"]["recovers"], value)
        # Original case untouched (pure function).
        self.assertIsNone(case["rubric"]["recovers"])
        with self.assertRaises(torture.EvaluationError):
            torture.set_judgment(case, "made_up_field", "yes")
        with self.assertRaises(torture.EvaluationError):
            torture.set_judgment(case, "recovers", "maybe")

    def test_tampered_case_refused(self):
        case = torture.generate_cases(fake_inputs(1), seed=0)[0]
        tampered = json.loads(json.dumps(case))
        tampered["element"] = "stale_output"  # injection still the original one
        with self.assertRaises(torture.EvaluationError):
            torture.validate_case(tampered)
        bad_rubric = json.loads(json.dumps(case))
        bad_rubric["rubric"]["recovers"] = "maybe"
        with self.assertRaises(torture.EvaluationError):
            torture.validate_case(bad_rubric)

    def test_output_carries_no_private_input_content(self):
        inputs = fake_inputs(6)
        document = torture.build_output(inputs, seed=7)
        serialized = json.dumps(document)
        for item in inputs:
            self.assertNotIn(item["payload"], serialized)
            self.assertNotIn(item["task_ref"], serialized)
        for case in document["cases"]:
            self.assertRegex(case["source_ref"], r"^[0-9a-f]{64}$")
        self.assertFalse(document["provenance"]["rubric_auto_scored"])

    def test_non_array_inputs_refused(self):
        with self.assertRaises(torture.EvaluationError):
            torture.generate_cases("not-a-list", seed=0)
        with self.assertRaises(torture.EvaluationError):
            torture.generate_cases([], seed=0)

    def test_element_enum_matches_methodology_doc_section6(self):
        self.assertEqual(
            set(torture.ADVERSARIAL_ELEMENTS),
            {
                "misleading_failure",
                "bad_prior_edit",
                "stale_output",
                "wrong_hypothesis",
                "partial_patch",
                "dependency_red_herring",
            },
        )
        self.assertEqual(
            torture.RUBRIC_FIELDS,
            (
                "recognizes_inconsistency",
                "revises_reasoning",
                "recovers",
                "final_patch_sound",
            ),
        )


class TortureCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_cli_generates_and_round_trips(self):
        inputs_path = os.path.join(self._tmp.name, "inputs.json")
        with open(inputs_path, "w", encoding="utf-8") as handle:
            json.dump(fake_inputs(6), handle)
        out_path = os.path.join(self._tmp.name, "cases.json")
        rc = torture.main(["--inputs", inputs_path, "--seed", "7", "--output", out_path])
        self.assertEqual(rc, 0)
        with open(out_path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["analysis"], "torture_case_generation")
        self.assertEqual(document["case_count"], 6)
        self.assertEqual(set(document["element_counts"].values()), {1})
        for case in document["cases"]:
            torture.validate_case(case)

    def test_cli_rejects_non_array_inputs(self):
        inputs_path = os.path.join(self._tmp.name, "bad.json")
        with open(inputs_path, "w", encoding="utf-8") as handle:
            json.dump({"task": "not-an-array"}, handle)
        out_path = os.path.join(self._tmp.name, "cases.json")
        rc = torture.main(["--inputs", inputs_path, "--output", out_path])
        self.assertEqual(rc, 2)
        self.assertFalse(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
