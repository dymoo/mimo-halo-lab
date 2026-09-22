"""Paired task-outcome statistical analysis for mimo_halo.

Consumes JSONL task-outcome records conforming to
``schemas/task-outcome.schema.json`` (schema_version=1), as produced by
external evaluation runners. This module is statistics only: it never runs
inference, never starts a sandbox, and never infers task success from
token or distribution agreement.

Pairing is by ``(task_id, seed)``. Bootstrap confidence intervals resample
*task clusters* (all seeds of a task move together), never individual
seed-level records. McNemar's exact two-sided binomial test is computed
only over independent single-seed task pairs; with repeated seeds per task
it is reported as ``not_applicable`` with an explicit reason.

Baseline-zero reference rates yield ratio status
``not_applicable_reference_zero``: the ratio is null, never ``inf``, and
never silently treated as passing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from typing import Any

SCHEMA_VERSION = 1
ANALYSIS_ID = "paired_task_outcome"

FAILURE_CATEGORIES = (
    "none",
    "misunderstanding",
    "search",
    "architecture",
    "implementation",
    "tool",
    "compiler",
    "test",
    "recovery",
    "forgotten_constraint",
    "context_loss",
    "runtime_corruption",
    "timeout",
    "other",
)

LANGUAGES = (
    "typescript",
    "javascript",
    "python",
    "rust",
    "go",
    "cpp",
    "sql",
    "shell",
    "other",
)

DOMAINS = (
    "frontend",
    "backend",
    "databases",
    "distributed",
    "concurrency",
    "network",
    "native",
    "build",
    "ci",
    "infra",
    "api",
    "testing",
)

RECORD_TYPES = ("task", "negative_control")

REQUIRED_STRING_FIELDS = ("task_id", "model_id", "language", "domain", "failure_category")
REQUIRED_BOOL_FIELDS = ("success", "catastrophic")
REQUIRED_INT_FIELDS = ("seed", "turns", "context_tokens")
OPTIONAL_OBJECT_FIELDS = ("timings", "tokens", "toolmetrics")

NO_TURN_BUCKET = "0 (no agent turn)"
TURNS_BUCKETS = (NO_TURN_BUCKET, "1-5", "6-15", "16-30", "31-60", "60+")
CONTEXT_BUCKET_K = 1024
CONTEXT_BUCKETS = ("<8K", "8-16K", "16-32K", "32-64K", "64-128K", "128K+")


class EvaluationError(Exception):
    """Fail-closed error for malformed inputs or contract violations."""


def turns_bucket(turns: int) -> str:
    if turns == 0:
        return NO_TURN_BUCKET
    for upper, label in ((5, "1-5"), (15, "6-15"), (30, "16-30"), (60, "31-60")):
        if turns <= upper:
            return label
    return "60+"


def context_bucket(context_tokens: int) -> str:
    k = CONTEXT_BUCKET_K
    if context_tokens < 8 * k:
        return "<8K"
    if context_tokens < 16 * k:
        return "8-16K"
    if context_tokens < 32 * k:
        return "16-32K"
    if context_tokens < 64 * k:
        return "32-64K"
    if context_tokens < 128 * k:
        return "64-128K"
    return "128K+"


def _validate_record(record: Any, source: str, line_number: int) -> dict[str, Any]:
    where = f"{source}:{line_number}"
    if not isinstance(record, dict):
        raise EvaluationError(f"{where}: record is not a JSON object")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise EvaluationError(
            f"{where}: schema_version must be {SCHEMA_VERSION}, got {record.get('schema_version')!r}"
        )
    for field in REQUIRED_STRING_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise EvaluationError(f"{where}: field {field!r} must be a non-empty string")
    for field in REQUIRED_BOOL_FIELDS:
        if not isinstance(record.get(field), bool):
            raise EvaluationError(f"{where}: field {field!r} must be a boolean")
    for field in REQUIRED_INT_FIELDS:
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise EvaluationError(f"{where}: field {field!r} must be an integer")
        if field != "seed" and value < 0:
            raise EvaluationError(f"{where}: field {field!r} must be >= 0")
    if record["turns"] == 0 and record["success"]:
        raise EvaluationError(
            f"{where}: turns=0 (no agent turn) requires success=false; "
            "zero-turn records are runtime failures, never successes"
        )
    if record["failure_category"] not in FAILURE_CATEGORIES:
        raise EvaluationError(
            f"{where}: failure_category {record['failure_category']!r} not in taxonomy"
        )
    if record["language"] not in LANGUAGES:
        raise EvaluationError(f"{where}: language {record['language']!r} not in taxonomy")
    if record["domain"] not in DOMAINS:
        raise EvaluationError(f"{where}: domain {record['domain']!r} not in taxonomy")
    record_type = record.get("record_type", "task")
    if record_type not in RECORD_TYPES:
        raise EvaluationError(f"{where}: record_type {record_type!r} not in {RECORD_TYPES}")
    for field in OPTIONAL_OBJECT_FIELDS:
        value = record.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise EvaluationError(f"{where}: optional field {field!r} must be an object")
        for key, item in value.items():
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise EvaluationError(
                    f"{where}: {field}.{key} must be a non-boolean number, got {item!r}"
                )
            if field != "timings" and not isinstance(item, int):
                raise EvaluationError(f"{where}: {field}.{key} must be an integer, got {item!r}")
            if not math.isfinite(item):
                raise EvaluationError(f"{where}: {field}.{key} must be finite, got {item!r}")
            if item < 0:
                raise EvaluationError(f"{where}: {field}.{key} must be >= 0, got {item!r}")
    return record


def load_records(path: str) -> list[dict[str, Any]]:
    """Load and validate JSONL task-outcome records. Fails closed."""
    records: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
                records.append(_validate_record(parsed, path, line_number))
    except OSError as exc:
        raise EvaluationError(f"cannot read {path}: {exc}") from exc
    if not records:
        raise EvaluationError(f"{path}: no records found")
    return records


def _index(records: list[dict[str, Any]], label: str) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    for record in records:
        key = (record["task_id"], record["seed"])
        if key in index:
            duplicates.append(key)
        else:
            index[key] = record
    if duplicates:
        listed = ", ".join(f"{task}/seed {seed}" for task, seed in sorted(duplicates))
        raise EvaluationError(f"{label}: duplicate (task_id, seed) pairs: {listed}")
    model_ids = sorted({record["model_id"] for record in records})
    if len(model_ids) != 1:
        raise EvaluationError(f"{label}: records must carry exactly one model_id, found {model_ids}")
    return index


def _pair(
    reference_index: dict[tuple[str, int], dict[str, Any]],
    candidate_index: dict[tuple[str, int], dict[str, Any]],
    *,
    report_missing: bool,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    ref_keys = set(reference_index)
    cand_keys = set(candidate_index)
    missing = sorted(ref_keys ^ cand_keys)
    if missing and not report_missing:
        listed = ", ".join(
            f"{task}/seed {seed} (missing from "
            f"{'reference' if (task, seed) in cand_keys else 'candidate'})"
            for task, seed in missing
        )
        raise EvaluationError(
            "unpaired (task_id, seed) keys present in only one input; "
            f"re-run with --report-missing to analyze the matched subset: {listed}"
        )
    matched = [
        (reference_index[key], candidate_index[key]) for key in sorted(ref_keys & cand_keys)
    ]
    missing_report = [
        {
            "task_id": task,
            "seed": seed,
            "missing_from": "candidate" if (task, seed) in ref_keys else "reference",
        }
        for task, seed in missing
    ]
    return matched, missing_report


def _task_cluster_stats(
    matched: list[tuple[dict[str, Any], dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Aggregate matched pairs into one entry per task (the seed cluster)."""
    by_task: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for ref, cand in matched:
        by_task.setdefault(ref["task_id"], []).append((ref, cand))
    clusters: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        pairs = by_task[task_id]
        n = len(pairs)
        ref_successes = sum(1 for ref, _ in pairs if ref["success"])
        cand_successes = sum(1 for _, cand in pairs if cand["success"])
        clusters.append(
            {
                "task_id": task_id,
                "seeds": n,
                "reference_successes": ref_successes,
                "candidate_successes": cand_successes,
                "reference_mean": ref_successes / n,
                "candidate_mean": cand_successes / n,
            }
        )
    return clusters


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _bootstrap(
    clusters: list[dict[str, Any]], replicates: int, seed: int
) -> tuple[list[float], list[float], list[float], int]:
    """Task-cluster bootstrap: resample tasks with replacement; every seed of
    a resampled task is carried along, so repeated seeds are never treated
    as independent observations. The seed controls replicate sampling
    exactly, so the interval is reproducible."""
    rng = random.Random(seed)
    n = len(clusters)
    ref_samples: list[float] = []
    cand_samples: list[float] = []
    ratio_samples: list[float] = []
    zero_reference_replicates = 0
    for _ in range(replicates):
        ref_sum = 0.0
        cand_sum = 0.0
        for _ in range(n):
            cluster = clusters[rng.randrange(n)]
            ref_sum += cluster["reference_mean"]
            cand_sum += cluster["candidate_mean"]
        ref_mean = ref_sum / n
        cand_mean = cand_sum / n
        ref_samples.append(ref_mean)
        cand_samples.append(cand_mean)
        if ref_mean > 0:
            ratio_samples.append(cand_mean / ref_mean)
        else:
            zero_reference_replicates += 1
    return ref_samples, cand_samples, ratio_samples, zero_reference_replicates


def _rate_summary(
    pairs: int, successes: int, task_rate: float, ci: list[float | None] | None
) -> dict[str, Any]:
    return {
        "pairs": pairs,
        "successes": successes,
        "pair_success_rate": successes / pairs if pairs else None,
        "task_mean_rate": task_rate,
        "ci95": ci,
    }


def exact_mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar p-value over discordant pairs.

    ``b`` = reference success & candidate failure, ``c`` = reference failure
    & candidate success. p = 2 * sum_{k<=min(b,c)} C(b+c, k) / 2^(b+c).
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) * 0.5**n
    return min(1.0, 2.0 * tail)


def _mcnemar(
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    seeds_per_task = {cluster["task_id"]: cluster["seeds"] for cluster in clusters}
    repeated = sorted(task for task, seeds in seeds_per_task.items() if seeds > 1)
    if repeated:
        return {
            "status": "not_applicable",
            "reason": "repeated_seeds_per_task",
            "tasks_with_repeated_seeds": repeated,
            "n_task_pairs": len(matched),
            "discordant": None,
            "p_exact_two_sided": None,
        }
    b = sum(1 for ref, cand in matched if ref["success"] and not cand["success"])
    c = sum(1 for ref, cand in matched if not ref["success"] and cand["success"])
    return {
        "status": "computed",
        "reason": None,
        "n_task_pairs": len(matched),
        "discordant": {
            "reference_success_candidate_failure": b,
            "reference_failure_candidate_success": c,
        },
        "p_exact_two_sided": exact_mcnemar_p(b, c),
    }


def _bucket_slices(
    matched: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Per-slice per-model observed rates.

    Every dimension is anchored to the reference record of each matched
    pair: ``language``/``domain`` directly, ``turns_bucket``/
    ``context_bucket`` by bucketing the reference value for both sides.
    Both models are therefore always counted in the same bucket, so the
    two sides of every slice row cover the identical paired population;
    pairs whose candidate value would land in a different bucket are
    counted in ``pairing.field_mismatches``. Observed input slices only;
    no gate logic is applied here.
    """
    slices: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "language": {},
        "domain": {},
        "turns_bucket": {},
        "context_bucket": {},
    }

    def bump(key: str, bucket: str, model: str, record: dict[str, Any]) -> None:
        stats = slices[key].setdefault(bucket, {})
        side = stats.setdefault(model, {"pairs": 0, "successes": 0, "catastrophes": 0})
        side["pairs"] += 1
        if record["success"]:
            side["successes"] += 1
        if record["catastrophic"]:
            side["catastrophes"] += 1

    for ref, cand in matched:
        for key, bucket in (
            ("language", ref["language"]),
            ("domain", ref["domain"]),
            ("turns_bucket", turns_bucket(ref["turns"])),
            ("context_bucket", context_bucket(ref["context_tokens"])),
        ):
            bump(key, bucket, "reference", ref)
            bump(key, bucket, "candidate", cand)
    for buckets in slices.values():
        for stats in buckets.values():
            for side in stats.values():
                side["rate"] = side["successes"] / side["pairs"] if side["pairs"] else None
    return slices


def _control_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    successes = sum(1 for r in records if r["success"])
    catastrophes = sum(1 for r in records if r["catastrophic"])
    return {
        "records": n,
        "successes": successes,
        "pass_rate": successes / n if n else None,
        "catastrophes": catastrophes,
    }


def analyze(
    reference_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    report_missing: bool = False,
    bootstrap_seed: int = 0,
    bootstrap_replicates: int = 10000,
    reference_path: str | None = None,
    candidate_path: str | None = None,
) -> dict[str, Any]:
    """Produce the paired-outcome analysis report (deterministic output)."""
    # Identity/duplicate fail-closed checks run over EVERY record in each
    # file, negative controls included: a duplicated (task_id, seed) or a
    # foreign model_id in a control row corrupts the control summary too.
    reference_keys = _index(reference_records, "reference")
    candidate_keys = _index(candidate_records, "candidate")
    ref_tasks = [r for r in reference_records if r.get("record_type", "task") == "task"]
    cand_tasks = [r for r in candidate_records if r.get("record_type", "task") == "task"]
    reference_index = {
        key: r for key, r in reference_keys.items() if r.get("record_type", "task") == "task"
    }
    candidate_index = {
        key: r for key, r in candidate_keys.items() if r.get("record_type", "task") == "task"
    }
    ref_model = reference_records[0]["model_id"]
    cand_model = candidate_records[0]["model_id"]
    if ref_model == cand_model:
        raise EvaluationError(
            f"reference and candidate must be two different models; both are {ref_model!r}"
        )

    matched, missing_report = _pair(reference_index, candidate_index, report_missing=report_missing)
    if not matched:
        raise EvaluationError(
            "no matched (task_id, seed) pairs between inputs; nothing to analyze"
        )

    clusters = _task_cluster_stats(matched)
    n_tasks = len(clusters)
    ref_successes = sum(c["reference_successes"] for c in clusters)
    cand_successes = sum(c["candidate_successes"] for c in clusters)
    ref_task_rate = sum(c["reference_mean"] for c in clusters) / n_tasks
    cand_task_rate = sum(c["candidate_mean"] for c in clusters) / n_tasks

    ref_samples, cand_samples, ratio_samples, zero_ref_reps = _bootstrap(
        clusters, bootstrap_replicates, bootstrap_seed
    )
    ref_ci = [_percentile(sorted(ref_samples), 0.025), _percentile(sorted(ref_samples), 0.975)]
    cand_ci = [
        _percentile(sorted(cand_samples), 0.025),
        _percentile(sorted(cand_samples), 0.975),
    ]

    wins = sum(1 for c in clusters if c["candidate_mean"] > c["reference_mean"])
    losses = sum(1 for c in clusters if c["candidate_mean"] < c["reference_mean"])
    ties = n_tasks - wins - losses

    if ref_task_rate > 0:
        ratio = cand_task_rate / ref_task_rate
        ratio_status = "computed"
        ratio_ci = [
            _percentile(sorted(ratio_samples), 0.025),
            _percentile(sorted(ratio_samples), 0.975),
        ]
        ci_note = None
        if zero_ref_reps:
            ci_note = (
                f"{zero_ref_reps} of {bootstrap_replicates} bootstrap replicates had a zero "
                "reference task mean and are excluded from the ratio CI"
            )
    else:
        ratio = None
        ratio_ci = None
        ratio_status = "not_applicable_reference_zero"
        ci_note = (
            "reference task-mean success rate is zero; the success ratio is undefined "
            "and reported as null (never inf, never treated as passing)"
        )

    ref_cat = sum(1 for ref, _ in matched if ref["catastrophic"])
    cand_cat = sum(1 for _, cand in matched if cand["catastrophic"])
    ref_cat_rate = ref_cat / len(matched)
    cand_cat_rate = cand_cat / len(matched)

    language_mismatches = sum(1 for ref, cand in matched if ref["language"] != cand["language"])
    domain_mismatches = sum(1 for ref, cand in matched if ref["domain"] != cand["domain"])
    turns_bucket_mismatches = sum(
        1 for ref, cand in matched if turns_bucket(ref["turns"]) != turns_bucket(cand["turns"])
    )
    context_bucket_mismatches = sum(
        1
        for ref, cand in matched
        if context_bucket(ref["context_tokens"]) != context_bucket(cand["context_tokens"])
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_ID,
        "inputs": {
            "reference": {
                "path": reference_path,
                "model_id": ref_model,
                "records": len(ref_tasks),
            },
            "candidate": {
                "path": candidate_path,
                "model_id": cand_model,
                "records": len(cand_tasks),
            },
        },
        "pairing": {
            "key": "(task_id, seed)",
            "matched_pairs": len(matched),
            "matched_tasks": n_tasks,
            "missing_pairs": missing_report,
            "missing_behavior": "reported" if report_missing else "failed",
            "evidence": "complete" if not missing_report else "incomplete_matched_subset_only",
            "field_mismatches": {
                "language": language_mismatches,
                "domain": domain_mismatches,
                "turns_bucket": turns_bucket_mismatches,
                "context_bucket": context_bucket_mismatches,
            },
        },
        "success": {
            "reference": _rate_summary(len(matched), ref_successes, ref_task_rate, ref_ci),
            "candidate": _rate_summary(len(matched), cand_successes, cand_task_rate, cand_ci),
        },
        "relative": {
            "ratio": ratio,
            "ratio_status": ratio_status,
            "ci95": ratio_ci,
            "ci_note": ci_note,
        },
        "task_comparison": {"wins": wins, "losses": losses, "ties": ties},
        "mcnemar": _mcnemar(matched, clusters),
        "catastrophes": {
            "reference": {"count": ref_cat, "rate": ref_cat_rate},
            "candidate": {"count": cand_cat, "rate": cand_cat_rate},
            "delta_rate": cand_cat_rate - ref_cat_rate,
            "candidate_increase": cand_cat_rate > ref_cat_rate,
        },
        "slices": {
            key: {bucket: buckets[bucket] for bucket in sorted(buckets)}
            for key, buckets in _bucket_slices(matched).items()
        },
        "provenance": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_unit": "task_cluster",
            "mcnemar_test": "exact_two_sided_binomial",
            "slice_anchor": "reference_record",
            "input_schema": "schemas/task-outcome.schema.json",
            "inference_performed": False,
        },
    }

    controls_ref = [r for r in reference_records if r.get("record_type") == "negative_control"]
    controls_cand = [r for r in candidate_records if r.get("record_type") == "negative_control"]
    if controls_ref or controls_cand:
        report["negative_controls"] = {
            "note": (
                "observed record_type=negative_control outcomes; excluded from the main "
                "task-success statistics above; rates are observed inputs, not gate verdicts"
            ),
            "reference": _control_summary(controls_ref),
            "candidate": _control_summary(controls_cand),
        }
    return report


def _write_json(report: dict[str, Any], output: str) -> None:
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output == "-":
        sys.stdout.write(text)
    else:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.evaluation.paired",
        description=(
            "Paired task-outcome analysis between exactly two models. "
            "Statistics over JSONL task-outcome records only; performs no inference."
        ),
    )
    parser.add_argument(
        "--reference", required=True, metavar="JSONL", help="reference/baseline model JSONL"
    )
    parser.add_argument(
        "--candidate", required=True, metavar="JSONL", help="candidate model JSONL"
    )
    parser.add_argument("--output", default="-", metavar="PATH", help="report output path ('-' for stdout)")
    parser.add_argument(
        "--report-missing",
        action="store_true",
        help=(
            "analyze the matched subset when (task_id, seed) pairs are missing from one "
            "side, listing them in the report instead of failing"
        ),
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=0, help="seed for the task-cluster bootstrap (default 0)"
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10000,
        help="task-cluster bootstrap replicates (default 10000)",
    )
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be >= 1")

    try:
        reference_records = load_records(args.reference)
        candidate_records = load_records(args.candidate)
        report = analyze(
            reference_records,
            candidate_records,
            report_missing=args.report_missing,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            reference_path=args.reference,
            candidate_path=args.candidate,
        )
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _write_json(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())