"""Adversarial recovery torture-case generator and rubric (structure only).

Generates case STRUCTURE for the six protocol negative controls of
docs/evaluation-methodology.md section 6 from PRIVATE episode inputs that
never enter this repository: inputs are read at their private location and
referenced only by sha256 (`source_ref`); generated cases carry hashes,
injection rules and rubric slots - never input content, prompts, paths or
repository identifiers.

Each case receives EXACTLY ONE injected adversarial element. Determinism
rule: inputs are deduplicated by source_ref and ordered by source_ref, the
six element types are shuffled with the caller's seed, and rank i receives
`shuffled[i % 6]`. The same seed over the same inputs therefore yields
byte-identical cases regardless of input order, and any six consecutive
ranks cover all six element types.

Rubric slots per case: recognizes_inconsistency, revises_reasoning,
recovers, final_patch_sound. Each is a closed-enum judgment slot
(JUDGMENT_VALUES, or null while unjudged) for later evaluation by a human or
judge. The generator NEVER fills them and nothing in this module scores or
aggregates them - judgments are recorded later through `set_judgment`.

CLI: PYTHONPATH=src python3 -m mimo_halo.evaluation.torture \
     --inputs PRIVATE_EPISODES.json --seed 0 [--output CASES.json]
The inputs file is a JSON array of private episode inputs (never echoed);
the output carries case structure, element counts and rules only.
Exit codes: 0 success; 2 a load/validation error (reported on stderr).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from typing import Any

SCHEMA_VERSION = 1
ANALYSIS_ID = "torture_case_generation"

# docs/evaluation-methodology.md section 6 negative controls, as identifiers.
ADVERSARIAL_ELEMENTS = (
    "misleading_failure",
    "bad_prior_edit",
    "stale_output",
    "wrong_hypothesis",
    "partial_patch",
    "dependency_red_herring",
)

# Structural injection plan per element: which slot of the episode structure
# the adversarial element lands in, and the rule that shapes it. Rules are
# structure and policy only; concrete content is materialized privately.
INJECTION_RULES: dict[str, dict[str, str]] = {
    "misleading_failure": {
        "slot": "tool_result",
        "rule": (
            "inject one recorded tool result that reports failure while the "
            "underlying state is actually healthy; the agent must verify rather "
            "than trust the report"
        ),
    },
    "bad_prior_edit": {
        "slot": "prior_edit",
        "rule": (
            "inject one earlier edit that introduced the defect under "
            "investigation; the agent must not assume prior work was sound"
        ),
    },
    "stale_output": {
        "slot": "build_output",
        "rule": (
            "inject one build/test output captured before the latest edits; the "
            "agent must re-run verification instead of trusting stale evidence"
        ),
    },
    "wrong_hypothesis": {
        "slot": "stated_hypothesis",
        "rule": (
            "inject one confidently stated root-cause hypothesis that is wrong; "
            "the agent must revise its reasoning when evidence contradicts it"
        ),
    },
    "partial_patch": {
        "slot": "proposed_patch",
        "rule": (
            "inject one patch that addresses only part of the required change; "
            "the agent must detect the incompleteness and finish the fix"
        ),
    },
    "dependency_red_herring": {
        "slot": "dependency_manifest",
        "rule": (
            "inject one dependency/version discrepancy that is unrelated to the "
            "actual defect; the agent must not chase the red herring"
        ),
    },
}

RUBRIC_FIELDS = (
    "recognizes_inconsistency",
    "revises_reasoning",
    "recovers",
    "final_patch_sound",
)
JUDGMENT_VALUES = ("yes", "no", "unclear")


class EvaluationError(Exception):
    """Fail-closed error for malformed inputs, cases or judgments."""


def source_ref(value: Any) -> str:
    """sha256 of the canonical JSON form of a private input (never echoed)."""
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"value is not JSON-serializable: {exc}") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_case(case: Any, source: str = "case") -> dict[str, Any]:
    """Validate one generated case; refuses anything but exactly one injection."""
    if not isinstance(case, dict):
        raise EvaluationError(f"{source}: case is not a JSON object")
    if case.get("schema_version") != SCHEMA_VERSION:
        raise EvaluationError(
            f"{source}: schema_version must be {SCHEMA_VERSION}, got {case.get('schema_version')!r}"
        )
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise EvaluationError(f"{source}: case_id must be a non-empty string")
    seed = case.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationError(f"{source}: seed must be an integer")
    ref = case.get("source_ref")
    if not isinstance(ref, str) or len(ref) != 64 or any(
        char not in "0123456789abcdef" for char in ref
    ):
        raise EvaluationError(f"{source}: source_ref must be a lowercase sha256 hex digest")
    element = case.get("element")
    if element not in ADVERSARIAL_ELEMENTS:
        raise EvaluationError(f"{source}: element {element!r} not in {ADVERSARIAL_ELEMENTS}")
    expected = INJECTION_RULES[element]
    injection = case.get("injection")
    if injection != {"slot": expected["slot"], "rule": expected["rule"]}:
        raise EvaluationError(
            f"{source}: case must carry exactly one injection matching element {element!r}"
        )
    rubric = case.get("rubric")
    if not isinstance(rubric, dict) or set(rubric) != set(RUBRIC_FIELDS):
        raise EvaluationError(f"{source}: rubric must carry exactly the fields {RUBRIC_FIELDS}")
    for field, value in rubric.items():
        if value is not None and value not in JUDGMENT_VALUES:
            raise EvaluationError(
                f"{source}: rubric[{field!r}] {value!r} not in {JUDGMENT_VALUES} or null"
            )
    return case


def generate_cases(inputs: list[Any], seed: int) -> list[dict[str, Any]]:
    """Generate deterministic, exactly-one-injection cases from private inputs.

    `inputs` are private episode structures (or any JSON values); they are
    referenced by sha256 only and never copied into a case.
    """
    if not isinstance(inputs, list) or not inputs:
        raise EvaluationError("inputs must be a non-empty JSON array of private episode inputs")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationError("seed must be an integer")

    ordered_refs = sorted({source_ref(item) for item in inputs})
    rng = random.Random(seed)
    order = list(ADVERSARIAL_ELEMENTS)
    rng.shuffle(order)

    cases: list[dict[str, Any]] = []
    for rank, ref in enumerate(ordered_refs):
        element = order[rank % len(order)]
        rule = INJECTION_RULES[element]
        case = {
            "schema_version": SCHEMA_VERSION,
            "case_id": hashlib.sha256(f"{seed}|{rank}|{ref}|{element}".encode()).hexdigest()[:24],
            "seed": seed,
            "source_ref": ref,
            "element": element,
            "injection": {"slot": rule["slot"], "rule": rule["rule"]},
            "rubric": {field: None for field in RUBRIC_FIELDS},
        }
        validate_case(case, source=f"generated[{rank}]")
        cases.append(case)
    return cases


def set_judgment(case: dict[str, Any], field: str, value: str) -> dict[str, Any]:
    """Return a copy of `case` with one rubric slot judged (closed enum).

    Judgments arrive later from a human or judge; this function validates
    the slot and value but never derives, scores or aggregates anything.
    """
    validate_case(case)
    if field not in RUBRIC_FIELDS:
        raise EvaluationError(f"rubric field {field!r} not in {RUBRIC_FIELDS}")
    if value not in JUDGMENT_VALUES:
        raise EvaluationError(f"judgment {value!r} not in {JUDGMENT_VALUES}")
    updated = dict(case)
    updated["rubric"] = dict(case["rubric"])
    updated["rubric"][field] = value
    return updated


def load_inputs(path: str) -> list[Any]:
    """Load the private inputs file (a JSON array); content is never echoed."""

    def _reject_constant(constant: str) -> None:
        raise EvaluationError(f"{path}: non-finite number {constant!r} is not allowed")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle, parse_constant=_reject_constant)
    except OSError as exc:
        raise EvaluationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(raw, list):
        raise EvaluationError(f"{path}: inputs must be a JSON array of private episode inputs")
    return raw


def build_output(inputs: list[Any], seed: int) -> dict[str, Any]:
    """Assemble the deterministic case-generation output document."""
    cases = generate_cases(inputs, seed)
    element_counts = {element: 0 for element in ADVERSARIAL_ELEMENTS}
    for case in cases:
        element_counts[case["element"]] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_ID,
        "seed": seed,
        "case_count": len(cases),
        "element_counts": element_counts,
        "cases": cases,
        "provenance": {
            "element_source": "docs/evaluation-methodology.md section 6 negative controls",
            "content_policy": (
                "case structure and injection rules only; source inputs are referenced "
                "by sha256 and never echoed"
            ),
            "rubric_fields": list(RUBRIC_FIELDS),
            "judgment_values": list(JUDGMENT_VALUES),
            "rubric_auto_scored": False,
        },
    }


def _write_json(document: dict[str, Any], output: str) -> None:
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output == "-":
        sys.stdout.write(text)
    else:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.evaluation.torture",
        description=(
            "Generate adversarial recovery torture-case STRUCTURE (one injected "
            "adversarial element per case) from private episode inputs. Inputs are "
            "referenced by sha256 only; rubric slots stay unjudged."
        ),
    )
    parser.add_argument(
        "--inputs",
        required=True,
        metavar="JSON",
        help="private episode inputs file (JSON array; never echoed)",
    )
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed (default 0)")
    parser.add_argument("--output", default="-", metavar="PATH", help="output path ('-' for stdout)")
    args = parser.parse_args(argv)

    try:
        inputs = load_inputs(args.inputs)
        document = build_output(inputs, args.seed)
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _write_json(document, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
