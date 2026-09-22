"""Deployment-readiness gates: fail-closed evidence checks over supplied inputs.

Consumes a candidate record and refuses promotion when any hard floor from
docs/quality-expectations.md is missed. Rules:

- Scope: these gates check the evidence a record supplies; `measured=True`
  is a record assertion, never authentication that a measurement happened.
- Estimates never pass a gate. A record without measured provenance is
  'unmeasured', and unmeasured never promotes.
- `promotable` here is DEPLOYMENT readiness, not quality-frontier
  eligibility: throughput/stability floors gate deployment only, while
  quality-frontier eligibility (validate_levels.quality_frontier_eligible)
  is decided from quality-level evidence alone.
- Invalid evidence or configuration is rejected, not scored: a non-object
  metrics container, or a metric/threshold value that is boolean, not a
  number, non-finite (inf/nan) or negative, raises GateError. Retention is
  a ratio and may exceed 1 for genuine improvement; it is never capped.
- Any 'fail' blocks promotion outright; 'unmeasured' blocks too (exit code 2
  distinguishes them for operators).
- Thresholds live here as named constants with their source documented in
  docs/quality-expectations.md; override via a config dict, never silently
  (unknown keys and nonsensical bounds raise GateError).

CLI: python3 -m mimo_halo.evaluation.gates --candidate CANDIDATE.json \
     [--config GATES.json] -> verdict JSON;
exit 0 = all gates pass, 1 = at least one fail, 2 = no fails but unmeasured
(rejected input/configuration also exits 2 with an error on stderr).
"""

from __future__ import annotations

import json
import math
import sys

GIB = 1024**3

# Hard floors: docs/quality-expectations.md ("Production quality gates").
DEFAULT_THRESHOLDS = {
    # Retention of full-MiMo capability (fractions, measured on the golden bank).
    "long_agent_retention_floor": 0.90,  # target 0.93, stretch 0.95 (report-only)
    "short_coding_retention_floor": 0.93,
    "repo_tool_retention_floor": 0.90,
    # Resident weight bytes: the matched sweep budget band (90 GiB +/-2%),
    # widened to the production ceiling. The older spec figure (100-105 GiB)
    # is superseded by the quality-first sweep directive.
    "weight_bytes_min": 94_704_028_877,
    "weight_bytes_max": 98_569_499_443,
    "weight_bytes_production_ceiling": 105 * GIB,
    # Diagnostics floors that still gate production (measured, PRE-OPTIMISATION
    # labeling does not excuse missing them).
    "c8_aggregate_tok_s_floor": 65.0,
}

# Explicit flags a record must carry as true (never inferred).
REQUIRED_TRUE_FLAGS = (
    "structural_integrity_suite_passed",  # 100% of the integrity suite
    "no_known_numerical_correctness_bug",
    "tool_protocol_essentially_unchanged",
    "stable_overnight_c8",
    "catastrophic_failures_not_increased",  # paired evidence, not vibes
)

# Metric keys gates read; any other metrics keys are ignored diagnostics.
METRIC_KEYS = (
    "long_agent_retention",
    "short_coding_retention",
    "repo_tool_retention",
    "resident_weight_bytes",
    "c8_aggregate_tok_s",
)


class GateError(Exception):
    """Fail-closed rejection of malformed records or threshold configuration."""


def _checked_number(value: object, where: str) -> float:
    """Return a finite, non-boolean, non-negative number or raise GateError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{where}: expected a number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise GateError(f"{where}: expected a finite number, got {value!r}")
    if value < 0:
        raise GateError(f"{where}: expected a non-negative number, got {value!r}")
    return float(value)


def _validate_metrics(metrics: object) -> dict:
    """Reject a malformed metrics container or any invalid gate metric."""
    if metrics is None:
        return {}
    if not isinstance(metrics, dict):
        raise GateError(f"metrics must be an object or null, got {type(metrics).__name__}")
    for key in METRIC_KEYS:
        if metrics.get(key) is not None:
            _checked_number(metrics[key], f"metrics.{key}")
    return metrics


def _validate_thresholds(thresholds: object) -> dict:
    """Merge threshold overrides over the defaults, rejecting bad configs."""
    if thresholds is None:
        return dict(DEFAULT_THRESHOLDS)
    if not isinstance(thresholds, dict):
        raise GateError(f"threshold config must be an object, got {type(thresholds).__name__}")
    unknown = sorted(set(thresholds) - set(DEFAULT_THRESHOLDS))
    if unknown:
        raise GateError(f"unknown threshold key(s): {', '.join(unknown)}")
    merged = dict(DEFAULT_THRESHOLDS)
    for key, value in thresholds.items():
        merged[key] = _checked_number(value, f"threshold {key}")
    if merged["weight_bytes_min"] > merged["weight_bytes_max"]:
        raise GateError("threshold weight_bytes_min must be <= weight_bytes_max")
    if merged["weight_bytes_max"] > merged["weight_bytes_production_ceiling"]:
        raise GateError(
            "threshold weight_bytes_max must be <= weight_bytes_production_ceiling"
        )
    return merged


def _gate(name: str, status: str, observed=None, required=None, note: str = "") -> dict:
    return {
        "gate": name,
        "status": status,  # pass | fail | unmeasured
        "observed": observed,
        "required": required,
        "note": note,
    }


def check_gates(record: dict, thresholds: dict | None = None) -> dict:
    """Return per-gate verdicts plus an overall promotion verdict.

    Fail-closed: rejects (GateError) malformed metrics containers, invalid
    metric values (boolean, non-numeric, non-finite, negative) and invalid
    threshold configurations instead of scoring them.
    """
    th = _validate_thresholds(thresholds)

    if not isinstance(record, dict) or record.get("measured") is not True:
        gates = [
            _gate(
                "measured_provenance",
                "unmeasured",
                observed=record.get("measured") if isinstance(record, dict) else None,
                required=True,
                note="records must declare measured=True; estimates never pass",
            )
        ]
        return {"promotable": False, "verdict": "unmeasured", "gates": gates}

    gates = [_gate("measured_provenance", "pass", observed=True, required=True)]
    metrics = _validate_metrics(record.get("metrics"))

    # Retention floors (golden-bank measured fractions).
    for key, floor, label in (
        ("long_agent_retention", th["long_agent_retention_floor"], "long_agent_retention"),
        ("short_coding_retention", th["short_coding_retention_floor"], "short_coding_retention"),
        ("repo_tool_retention", th["repo_tool_retention_floor"], "repo_tool_retention"),
    ):
        val = metrics.get(key)
        if val is None:
            gates.append(_gate(label, "unmeasured", required=f">= {floor}"))
        elif val >= floor:
            gates.append(_gate(label, "pass", observed=val, required=f">= {floor}"))
        else:
            gates.append(_gate(label, "fail", observed=val, required=f">= {floor}"))

    # Resident weight band.
    wb = metrics.get("resident_weight_bytes")
    if wb is None:
        gates.append(_gate("weight_budget", "unmeasured", required="band"))
    else:
        in_band = th["weight_bytes_min"] <= wb <= th["weight_bytes_max"]
        under_ceiling = wb <= th["weight_bytes_production_ceiling"]
        if in_band and under_ceiling:
            gates.append(_gate("weight_budget", "pass", observed=wb, required="90GiB band"))
        else:
            gates.append(
                _gate(
                    "weight_budget",
                    "fail",
                    observed=wb,
                    required=f"[{th['weight_bytes_min']}, {th['weight_bytes_max']}]",
                    note="out of matched band or over production ceiling",
                )
            )

    # C8 throughput floor (measured; PRE-OPTIMISATION label allowed).
    c8 = metrics.get("c8_aggregate_tok_s")
    if c8 is None:
        gates.append(_gate("c8_floor", "unmeasured", required=f">= {th['c8_aggregate_tok_s_floor']}"))
    elif c8 >= th["c8_aggregate_tok_s_floor"]:
        gates.append(
            _gate("c8_floor", "pass", observed=c8, required=f">= {th['c8_aggregate_tok_s_floor']}")
        )
    else:
        gates.append(
            _gate("c8_floor", "fail", observed=c8, required=f">= {th['c8_aggregate_tok_s_floor']}")
        )

    # Explicit boolean flags.
    for flag in REQUIRED_TRUE_FLAGS:
        val = record.get(flag)
        if val is True:
            gates.append(_gate(flag, "pass", observed=True, required=True))
        elif val is None:
            gates.append(_gate(flag, "unmeasured", required=True))
        else:
            gates.append(_gate(flag, "fail", observed=val, required=True))

    statuses = [g["status"] for g in gates]
    if any(s == "fail" for s in statuses):
        verdict = "fail"
    elif any(s == "unmeasured" for s in statuses):
        verdict = "unmeasured"
    else:
        verdict = "pass"
    return {"promotable": verdict == "pass", "verdict": verdict, "gates": gates}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Production quality gates (fail-closed).")
    parser.add_argument("--candidate", required=True, help="measured candidate record JSON")
    parser.add_argument("--config", default=None, help="threshold overrides JSON")
    args = parser.parse_args(argv)

    with open(args.candidate, encoding="utf-8") as fh:
        record = json.load(fh)
    thresholds = None
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            thresholds = json.load(fh)

    try:
        result = check_gates(record, thresholds)
    except GateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return {"pass": 0, "fail": 1, "unmeasured": 2}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
