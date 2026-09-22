"""Seven-level validation hierarchy records and fail-closed promotion checks.

Each measurement stage of the compression campaign records one level record
(schema_version=1) per candidate against the seven evaluation levels of
docs/evaluation-methodology.md section 4. Numbering follows that document
exactly: 1 artifact, 2 numerics, 3 distribution, 4 capability, 5 protocol,
6 long repo execution, 7 runtime concurrency.

This module validates those records and assembles a per-candidate report.
Fail-closed promotion rules (no evidence, no pass):

- A level record claiming "pass" while any required evidence field is
  missing or null is downgraded to "unmeasured". A level with no record at
  all is "unmeasured" and lists every required field as missing.
- `promotable` is DEPLOYMENT readiness (all seven levels "pass"), never
  quality-frontier eligibility. Any level's "fail" blocks promotion
  regardless of the other six: a level-1 artifact failure (or a level-2
  numerical failure) ends promotion no matter how well level 7 reads,
  because downstream levels never mask upstream fails.
- `quality_frontier_eligible` (quality selection) is the distinct verdict,
  decided over quality-level evidence (levels 1-6) only. Missing or failed
  runtime/concurrency evidence (level 7) withholds deployment readiness but
  never rejects a candidate from the quality frontier: performance is
  excluded from quality ranking (see docs/experiments/compression-sweep.md),
  so quality-selection eligibility and deployment readiness stay distinct.
- A level with no record is reported "unmeasured" listing its required
  fields as missing; the report never marks an unrun level as evaluated or
  applicable, in either verdict.

CLI: PYTHONPATH=src python3 -m mimo_halo.evaluation.validate_levels \
     RECORD.json [RECORD.json ...] [--output REPORT.json]
(Positional arguments may also be directories; every *.json inside is read.)
Exit codes: 0 every candidate promotable; 1 at least one candidate has a
fail; 2 no fails but at least one candidate is not promotable, or a load
error (reported on stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

SCHEMA_VERSION = 1
ANALYSIS_ID = "seven_level_validation"

LEVEL_VERDICTS = ("pass", "fail", "unmeasured")

# docs/evaluation-methodology.md section 4 table, in documented order.
# required_evidence lists the "Observed metrics" of each row; a level reads
# "pass" only when every field is present and non-null.
LEVELS: tuple[dict[str, Any], ...] = (
    {
        "level": 1,
        "name": "artifact",
        "measures": "Checkpoint integrity and provenance",
        "required_evidence": (
            "param_tensor_inventory",
            "missing_tensors",
            "dtype_changes",
            "nan_inf",
            "reload_round_trip",
            "top8_preservation",
            "expert_maps",
            "lineage_hashes",
        ),
    },
    {
        "level": 2,
        "name": "numerics",
        "measures": "Quantized vs parent numeric damage",
        "required_evidence": (
            "kl_divergence",
            "js_divergence",
            "top1_agreement",
            "top5_agreement",
            "logit_cosine",
            "entropy_delta",
            "per_slice_edge_deltas",
        ),
    },
    {
        "level": 3,
        "name": "distribution",
        "measures": "Routing and calibration distribution",
        "required_evidence": (
            "routing_top8_overlap",
            "gate_correlation",
            "expert_frequency",
            "jaccard_convergence",
            "heldout_perplexity",
        ),
    },
    {
        "level": 4,
        "name": "capability",
        "measures": "Short structured tasks",
        "required_evidence": (
            "short_coding_success",
            "short_debug_success",
            "short_tool_success",
            "short_repo_success",
            "reference_short_success",
            "paired_same_seeds",
        ),
    },
    {
        "level": 5,
        "name": "protocol",
        "measures": "Agent protocol behavior",
        "required_evidence": (
            "valid_structured_output",
            "valid_tool_names",
            "valid_argument_schema",
            "escaping_correctness",
            "stop_behavior",
            "tool_error_handling",
            "destructive_discipline",
            "negative_control_pass_rate",
        ),
    },
    {
        "level": 6,
        "name": "long_repo_execution",
        "measures": "Golden heldout real-repo tasks",
        "required_evidence": (
            "paired_task_success",
            "turns",
            "generated_tokens",
            "tool_calls",
            "bad_tool_calls",
            "wrong_hypotheses",
            "recovery_events",
            "wall_time",
            "patch_quality",
            "catastrophes",
        ),
    },
    {
        "level": 7,
        "name": "runtime_concurrency",
        "measures": "Serving behavior under load",
        "required_evidence": (
            "c1_aggregate_tok_s",
            "c2_aggregate_tok_s",
            "c4_aggregate_tok_s",
            "c8_aggregate_tok_s",
            "memory",
            "throttle_decay",
            "canary_stability",
            "bursty_trace_replay_throughput",
        ),
    },
)

LEVELS_BY_NUMBER = {spec["level"]: spec for spec in LEVELS}
LEVELS_BY_NAME = {spec["name"]: spec for spec in LEVELS}

QUALITY_FRONTIER_LEVELS = (1, 2, 3, 4, 5, 6)


class EvaluationError(Exception):
    """Fail-closed error for malformed level records or contract violations."""


def required_evidence(level: int) -> tuple[str, ...]:
    """Required evidence field names for a level (1-7)."""
    spec = LEVELS_BY_NUMBER.get(level)
    if spec is None:
        raise EvaluationError(f"level {level!r} is outside the seven-level hierarchy (1-7)")
    return spec["required_evidence"]


def validate_record(doc: Any, source: str = "<record>") -> dict[str, Any]:
    """Validate one level record and return its normalized report entry.

    Normalization is fail-closed: a "pass" verdict with any missing or null
    required evidence field is downgraded to "unmeasured"; "fail" is always
    retained (a recorded failure blocks regardless of evidence completeness).
    """
    if not isinstance(doc, dict):
        raise EvaluationError(f"{source}: level record is not a JSON object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise EvaluationError(
            f"{source}: schema_version must be {SCHEMA_VERSION}, got {doc.get('schema_version')!r}"
        )
    candidate_id = doc.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise EvaluationError(f"{source}: candidate_id must be a non-empty string")
    level = doc.get("level")
    if isinstance(level, bool) or not isinstance(level, int) or level not in LEVELS_BY_NUMBER:
        raise EvaluationError(f"{source}: level must be an integer in 1-7, got {level!r}")
    spec = LEVELS_BY_NUMBER[level]
    level_name = doc.get("level_name")
    if level_name is not None and level_name != spec["name"]:
        raise EvaluationError(
            f"{source}: level_name {level_name!r} does not match level {level} "
            f"({spec['name']!r}) under docs/evaluation-methodology.md section 4"
        )
    verdict = doc.get("verdict")
    if verdict not in LEVEL_VERDICTS:
        raise EvaluationError(f"{source}: verdict {verdict!r} not in {LEVEL_VERDICTS}")
    evidence = doc.get("evidence")
    if not isinstance(evidence, dict):
        raise EvaluationError(f"{source}: evidence must be an object of required fields")
    note = doc.get("notes")
    if note is not None and not isinstance(note, str):
        raise EvaluationError(f"{source}: notes must be a string or null")

    missing = [field for field in spec["required_evidence"] if evidence.get(field) is None]
    normalized = verdict
    if verdict == "pass" and missing:
        normalized = "unmeasured"
        downgrade = (
            "pass downgraded to unmeasured; missing required evidence: " + ", ".join(missing)
        )
        note = f"{note}; {downgrade}" if note else downgrade

    return {
        "level": level,
        "level_name": spec["name"],
        "verdict": normalized,
        "missing_evidence": missing,
        "note": note,
        "candidate_id": candidate_id,
    }


def load_records(path: str) -> list[Any]:
    """Load level record JSON from a file (a single object or a list)."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError as exc:
        raise EvaluationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{path}: invalid JSON ({exc})") from exc
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    raise EvaluationError(f"{path}: expected a level record object or a list of records")


def _expand_path(path: str) -> list[str]:
    import os

    if os.path.isdir(path):
        entries = sorted(
            os.path.join(path, name) for name in os.listdir(path) if name.endswith(".json")
        )
        if not entries:
            raise EvaluationError(f"{path}: directory contains no .json level records")
        return entries
    if os.path.isfile(path):
        return [path]
    raise EvaluationError(f"{path}: not a file or directory")


def assemble_report(records: list[Any], record_files: list[str] | None = None) -> dict[str, Any]:
    """Assemble the per-candidate seven-level report. Fails closed."""
    if not isinstance(records, list) or not records:
        raise EvaluationError("no level records supplied; nothing to report")

    by_candidate: dict[str, dict[int, dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for index, doc in enumerate(records):
        entry = validate_record(doc, source=f"record[{index}]")
        key = (entry["candidate_id"], entry["level"])
        if key in seen:
            raise EvaluationError(
                f"duplicate level record for candidate {entry['candidate_id']!r} "
                f"level {entry['level']}; conflicting evidence is refused"
            )
        seen.add(key)
        by_candidate.setdefault(entry["candidate_id"], {})[entry["level"]] = entry

    candidates: dict[str, Any] = {}
    for candidate_id in sorted(by_candidate):
        levels: dict[str, Any] = {}
        counts = {"pass": 0, "fail": 0, "unmeasured": 0}
        failed_levels: list[int] = []
        unmeasured_levels: list[int] = []
        for spec in LEVELS:
            number = spec["level"]
            entry = by_candidate[candidate_id].get(number)
            if entry is None:
                entry = {
                    "level": number,
                    "level_name": spec["name"],
                    "verdict": "unmeasured",
                    "missing_evidence": list(spec["required_evidence"]),
                    "note": "no level record supplied",
                }
            levels[str(number)] = {
                "level": number,
                "level_name": entry["level_name"],
                "verdict": entry["verdict"],
                "missing_evidence": entry["missing_evidence"],
                "note": entry["note"],
            }
            counts[entry["verdict"]] += 1
            if entry["verdict"] == "fail":
                failed_levels.append(number)
            elif entry["verdict"] == "unmeasured":
                unmeasured_levels.append(number)

        if failed_levels:
            verdict = "fail"
        elif unmeasured_levels:
            verdict = "unmeasured"
        else:
            verdict = "pass"

        quality_eligible = all(
            by_candidate[candidate_id].get(number, {}).get("verdict") == "pass"
            for number in QUALITY_FRONTIER_LEVELS
        )
        candidates[candidate_id] = {
            "levels": levels,
            "counts": counts,
            "failed_levels": failed_levels,
            "unmeasured_levels": unmeasured_levels,
            "verdict": verdict,
            "promotable": verdict == "pass",
            "quality_frontier_eligible": quality_eligible,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_ID,
        "inputs": {"record_files": list(record_files or []), "records": len(records)},
        "levels": [
            {"level": spec["level"], "name": spec["name"], "measures": spec["measures"]}
            for spec in LEVELS
        ],
        "candidates": candidates,
        "provenance": {
            "level_definitions": "docs/evaluation-methodology.md section 4",
            "promotion_rule": (
                "promotable is deployment readiness: it requires all seven levels "
                "pass; any level's fail blocks promotion regardless of the other "
                "levels; unmeasured never promotes"
            ),
            "quality_frontier_rule": (
                "quality_frontier_eligible is decided on levels 1-6 only; runtime/"
                "concurrency evidence (level 7) affects deployment readiness only, "
                "never quality-selection eligibility; unrun levels are reported "
                "unmeasured, never asserted evaluated"
            ),
            "inference_performed": False,
        },
    }


def _write_json(report: dict[str, Any], output: str) -> None:
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output == "-":
        sys.stdout.write(text)
    else:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.evaluation.validate_levels",
        description=(
            "Assemble the seven-level validation report for compression candidates "
            "from per-level evidence records. Fail-closed: missing evidence is "
            "unmeasured, unmeasured never promotes, any fail blocks promotion."
        ),
    )
    parser.add_argument(
        "records",
        nargs="+",
        metavar="PATH",
        help="level record JSON file or directory of *.json level records",
    )
    parser.add_argument("--output", default="-", metavar="PATH", help="report path ('-' for stdout)")
    args = parser.parse_args(argv)

    try:
        files: list[str] = []
        docs: list[Any] = []
        for path in args.records:
            for file_path in _expand_path(path):
                files.append(file_path)
                docs.extend(load_records(file_path))
        report = assemble_report(docs, record_files=files)
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _write_json(report, args.output)
    verdicts = [candidate["verdict"] for candidate in report["candidates"].values()]
    if all(verdict == "pass" for verdict in verdicts):
        return 0
    if any(verdict == "fail" for verdict in verdicts):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
