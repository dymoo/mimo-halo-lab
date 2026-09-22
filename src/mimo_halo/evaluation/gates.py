"""Production quality gates: fail-closed promotion checks.

Consumes MEASURED candidate records and refuses promotion when any hard floor
from docs/quality-expectations.md is missed. Rules:

- Estimates never pass a gate. A record without measured provenance is
  'unmeasured', and unmeasured never promotes.
- Any 'fail' blocks promotion outright; 'unmeasured' blocks too (exit code 2
  distinguishes them for operators).
- Thresholds live here as named constants with their source documented in
  docs/quality-expectations.md; override via a config dict, never silently.

CLI: python3 -m mimo_halo.evaluation.gates --candidate CANDIDATE.json \
     [--outcomes OUTCOMES.json] [--config GATES.json] -> verdict JSON;
exit 0 = all gates pass, 1 = at least one fail, 2 = no fails but unmeasured.
"""

from __future__ import annotations

import json
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


def _gate(name: str, status: str, observed=None, required=None, note: str = "") -> dict:
    return {
        "gate": name,
        "status": status,  # pass | fail | unmeasured
        "observed": observed,
        "required": required,
        "note": note,
    }


def check_gates(record: dict, thresholds: dict | None = None) -> dict:
    """Return per-gate verdicts plus an overall promotion verdict."""
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

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
    metrics = record.get("metrics") or {}

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

    result = check_gates(record, thresholds)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return {"pass": 0, "fail": 1, "unmeasured": 2}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
