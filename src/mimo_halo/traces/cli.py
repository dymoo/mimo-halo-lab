"""Trace tooling CLI.

Commands:
- discover  : read-only aggregate discovery; public summary to the repo,
              private session index to the output root.
- normalize : parse sessions into normalized episode sets (hash-light by
              default; redacted text only with --include-text).
- partition : deterministic grouped split with frozen-assignment immutability;
              golden is an eligibility-independent hash reservation.
- freeze    : golden eligibility evaluation and bank manifest (freezes only
              reserved+eligible tasks; never fabricates oracles; never imports
              a training group when an oracle appears later).
- migrate-reservations : audited pre-calibration migration that restores
              hash-reserved golden groups wrongly routed to training; refuses
              while frozen golden tasks or consumption evidence exist, backs up
              the agent-created assignment files with digests and records an
              audit reason.

The output root is EXPLICIT and must live outside this Git repository (private
workstation storage, e.g. the MIMO_LAB workspace). Raw transcripts are read at
their origin and never copied.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import (
    TraceError,
    atomic_write_json,
    ensure_dir,
    file_sha256,
    free_bytes,
    is_relative_to,
)
from .discovery import discover
from .events import SessionTrace
from .golden import evaluate_golden_eligibility, summarize_golden_candidates
from .identity import DEFAULT_ISSUE_PATTERN, build_task_groups
from .partition import (
    assign_partition,
    guard_pre_calibration_migration,
    partition_tasks,
    regenerated_reservations,
)
from .session import normalize_sessions

REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_SUMMARY = REPO_ROOT / "manifests" / "datasets" / "discovery-summary.json"
CONFIG_PATH = REPO_ROOT / "configs" / "dataset.json"
SCHEMA_PATH = REPO_ROOT / "schemas" / "episode.schema.json"


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise TraceError(f"missing config: {path}")
    return json.loads(path.read_text())


def resolve_output_root(raw: str) -> Path:
    root = Path(raw).expanduser().resolve()
    if is_relative_to(root, REPO_ROOT):
        raise TraceError(
            f"output root {root} is inside the Git repository; use a private "
            "workspace location outside the repo"
        )
    ensure_dir(root)
    free = free_bytes(root)
    if free < (1 << 29):  # 512 MiB headroom guard
        raise TraceError(
            f"output volume for {root} has only {free} bytes free; "
            "refusing to write (no deletions performed)"
        )
    return root


def _write_private(path: Path, payload) -> None:
    ensure_dir(path.parent)
    # compact encoding for private machine artifacts (bounded output volume)
    atomic_write_json(path, payload, indent=None)


def _update_public_summary(section: str, payload: dict) -> None:
    """Merge an aggregate (counts/status-only) section into the public summary."""
    summary: dict = {}
    if PUBLIC_SUMMARY.is_file():
        try:
            summary = json.loads(PUBLIC_SUMMARY.read_text())
        except json.JSONDecodeError:
            summary = {}
    summary["schema_version"] = "1.0.0"
    summary["kind"] = "mimo-halo-discovery-summary"
    summary[section] = payload
    atomic_write_json(PUBLIC_SUMMARY, summary)


def cmd_discover(args) -> int:
    config = load_config(Path(args.config))
    out_root = resolve_output_root(args.output_root)
    result = discover(config, max_sessions=args.max_sessions)

    # Private session index: hashed identities only, written to the output root.
    private_index = {
        "schema_version": "1.0.0",
        "kind": "mimo-halo-discovery-private-index",
        "note": "hashed session identity facts; no raw paths, payloads or secrets",
        "output_root_free_bytes": free_bytes(out_root),
        "sources": {
            src: {
                "files": result["per_source"][src]["files"],
                "total_bytes": result["per_source"][src]["total_bytes"],
            }
            for src in result["per_source"]
        },
        "sessions": result["sessions"],
    }
    _write_private(out_root / "traces" / "discovered" / "session-index.json", private_index)

    # Public summary: aggregate counts, bytes, schema facts, eligibility signals.
    public = {
        "schema_version": "1.0.0",
        "kind": "mimo-halo-discovery-summary",
        "sources": {
            src: {k: v for k, v in stats.items() if k != "root_configured"}
            for src, stats in result["per_source"].items()
        },
        "extra_adapters_needed": {
            name: {"format": meta.get("format"), "status": meta.get("status")}
            for name, meta in result["extra_adapters_needed"].items()
        },
        "schema_facts": result["schema_facts"],
        "privacy": {
            "raw_payloads_retained": False,
            "paths_published": False,
            "public_fields": "counts/bytes/hashes/schema facts only",
            "redaction_claim": "credential-pattern redaction only; not complete anonymization",
        },
        "repo_head_at_discovery": _repo_head(),
        "config_sha256": file_sha256(Path(args.config)),
    }
    atomic_write_json(PUBLIC_SUMMARY, public)

    print(f"discovery complete: public summary -> {PUBLIC_SUMMARY}")
    print(f"private session index -> {out_root / 'traces' / 'discovered' / 'session-index.json'}")
    for src, stats in result["per_source"].items():
        print(f"  {src}: files={stats['files']} parsed={stats['parsed_sessions']} "
              f"tool_calls={stats['tool_call_count']} unmatched_calls={stats['unmatched_tool_calls']} "
              f"repo_identity={stats['sessions_with_repo_identity']} "
              f"start_revision={stats['sessions_with_start_revision']}")
    return 0


def cmd_normalize(args) -> int:
    config = load_config(Path(args.config))
    out_root = resolve_output_root(args.output_root)
    sources = config["traces"]["sources"]

    from .discovery import expand_root, list_transcripts

    pairs: list[tuple[str, Path]] = []
    for source, cfg in sources.items():
        root = expand_root(cfg["root"])
        transcripts, _records = list_transcripts(source, root, cfg.get("transcript_glob", "**/*.jsonl"))
        for p in transcripts:
            pairs.append((source, p))
    pairs.sort()
    if args.max_sessions is not None:
        pairs = pairs[: args.max_sessions]

    traces: dict[str, SessionTrace] = {}
    for source, path in pairs:
        from .session import normalize_session

        traces[str(path)] = normalize_session(path, source)

    include_text = bool(args.include_text)
    written = 0
    total_episodes = 0
    for source, path in pairs:
        trace = traces[str(path)]
        doc = _session_document(trace, include_text)
        digest = file_sha256(path)
        doc["session"]["content_sha256"] = digest
        out_name = f"{digest[:24]}-{trace.source_path_sha256[:8]}.json"
        out_path = out_root / "traces" / "normalized" / source / out_name
        _write_private(out_path, doc)
        written += 1
        total_episodes += len(doc["episodes"])
    print(f"normalized {written} sessions, {total_episodes} episodes -> "
          f"{out_root / 'traces' / 'normalized'}")
    print(f"text included: {include_text} (default is hash-light; redaction is not full anonymization)")
    return 0


def _session_document(trace: SessionTrace, include_text: bool) -> dict:
    from .session import segment_episodes

    episodes = segment_episodes(trace)
    return {
        "schema_version": "1.0.0",
        "kind": "mimo-halo-normalized-trace",
        "session": {
            "source": trace.source,
            "session_id": trace.session_id,
            "source_path_sha256": trace.source_path_sha256,
            "cwd_sha256": trace.cwd_sha256,
            "repo_identifier": trace.repo_identifier,
            "branch": trace.branch,
            "start_revision": trace.start_revision,
            "started_at": trace.started_at,
            "ended_at": trace.ended_at,
            "models": trace.models,
            "parent_session_ids": trace.parent_session_ids,
            "has_branching": trace.has_branching,
            "sidechain_sessions": trace.sidechain_sessions,
            "event_count": len(trace.events),
            "tool_call_count": sum(1 for e in trace.events if e.type == "tool_call"),
            "tool_result_count": sum(1 for e in trace.events if e.type == "tool_result"),
            "unmatched_tool_calls": trace.unmatched_tool_calls,
            "unmatched_tool_results": trace.unmatched_tool_results,
            "redaction_hits": trace.redaction_hits,
        },
        "events": [e.to_public(include_text) for e in trace.events],
        "episodes": [ep.to_public() for ep in episodes],
    }


def cmd_partition(args) -> int:
    config = load_config(Path(args.config))
    out_root = resolve_output_root(args.output_root)
    private_index_path = out_root / "traces" / "discovered" / "session-index.json"
    if not private_index_path.is_file():
        raise TraceError(f"run 'discover' first; missing {private_index_path}")

    # Re-parse enough to rebuild identity groups (identity needs prompts; kept private).
    groups_doc = _build_groups_from_index(config, args)
    frozen_path = Path(args.frozen_assignments) if args.frozen_assignments else (
        out_root / "traces" / "partition" / "frozen-assignments.json")
    frozen = {}
    if Path(frozen_path).is_file():
        frozen = json.loads(Path(frozen_path).read_text()).get("assignments", {})

    eligible_ids = set()
    if Path(args.golden_eligible).is_file():
        eligible_doc = json.loads(Path(args.golden_eligible).read_text())
        eligible_ids = {c["task_id"] for c in eligible_doc.get("candidates", []) if c["eligible"]}
    elif args.golden_eligible:
        raise TraceError(f"golden eligibility file not found: {args.golden_eligible}")

    weights = config["partition"]["weights"]
    salt = config["partition"]["salt"]
    result = partition_tasks(groups_doc["groups"], salt, frozen, eligible_ids)
    if groups_doc["quarantined"]:
        result["quarantined_groups"] = groups_doc["quarantined"]

    _write_private(out_root / "traces" / "partition" / "task-assignments.json", result)
    if frozen_path and not Path(frozen_path).is_file():
        _write_private(Path(frozen_path), {
            "schema_version": "1.0.0",
            "note": "frozen assignments are immutable; golden entries are "
                    "eligibility-independent reservations (reserved+ineligible "
                    "stays not_ready, never training); later runs must reuse this file",
            "salt": salt,
            "assignments": result["assignments"],
        })
    print(f"partitioned {len(result['assignments'])} task groups; counts: {result['counts']}")
    print(f"reserved golden groups: {result['reserved_golden_count']} "
          f"(ready: {len(result['reserved_ready'])}, "
          f"not_ready: {len(result['reserved_not_ready'])})")
    print(f"quarantined identity failures: {len(groups_doc['quarantined'])}")
    print(f"assignments -> {out_root / 'traces' / 'partition' / 'task-assignments.json'}")
    _update_public_summary("partition", {
        "task_groups_assigned": len(result["assignments"]),
        "counts": result["counts"],
        "quarantined_identity_failures": len(groups_doc["quarantined"]),
        "frozen_respected": result["frozen_respected"],
        "reserved_golden_groups": result["reserved_golden_count"],
        "reserved_golden_ready": len(result["reserved_ready"]),
        "reserved_golden_not_ready": len(result["reserved_not_ready"]),
    })
    return 0


def _build_groups_from_index(config: dict, args) -> dict:
    """Rebuild task identity groups by re-parsing narrowly (prompts stay private)."""
    from .discovery import expand_root, list_transcripts
    from .session import normalize_session

    traces = []
    for source, cfg in config["traces"]["sources"].items():
        root = expand_root(cfg["root"])
        transcripts, _records = list_transcripts(source, root, cfg.get("transcript_glob", "**/*.jsonl"))
        paths = transcripts
        if args.max_sessions is not None:
            paths = paths[: args.max_sessions]
        for p in paths:
            try:
                traces.append(normalize_session(p, source))
            except Exception:
                continue
    return build_task_groups(traces, DEFAULT_ISSUE_PATTERN)


def cmd_freeze(args) -> int:
    config = load_config(Path(args.config))
    out_root = resolve_output_root(args.output_root)
    groups_doc = _build_groups_from_index(config, args)

    oracle_index = {}
    if args.oracle_index:
        oracle_path = Path(args.oracle_index)
        if not oracle_path.is_file():
            raise TraceError(f"oracle index not found: {oracle_path}")
        oracle_index = json.loads(oracle_path.read_text())

    salt = config["partition"]["salt"]
    reserved_ids = {
        g["group_id"] for g in groups_doc["groups"]
        if assign_partition(salt, g["group_id"]) == "golden"
    }
    evaluation = evaluate_golden_eligibility(
        groups_doc["groups"],
        oracle_index=oracle_index,
        min_tasks=config["partition"].get("golden_min_tasks", 50),
        max_tasks=config["partition"].get("golden_max_tasks", 200),
        reserved_ids=reserved_ids,
    )

    manifest = {
        "schema_version": "1.0.0",
        "kind": "mimo-halo-golden-freeze",
        "bank_status": evaluation["bank_status"],
        "eligible_count": evaluation["eligible_count"],
        "ineligible_count": evaluation["ineligible_count"],
        "reserved_count": evaluation["reserved_count"],
        "reserved_eligible_count": evaluation["reserved_eligible_count"],
        "non_reserved_eligible_count": evaluation["non_reserved_eligible_count"],
        "missing_field_counts": evaluation["missing_field_counts"],
        "blocker": evaluation["blocker"],
        "frozen_tasks": [
            {"task_id": c["task_id"], "repo_identifier": c["repo_identifier"],
             "starting_revision": c["starting_revision"], "issue_ref": c["issue_ref"]}
            for c in evaluation["candidates"] if c["reserved"] and c["eligible"]
        ],
        "note": "frozen tasks are reserved+eligible groups only; a training group "
                "is never imported into golden when an oracle appears. Frozen tasks "
                "reference privately held repos/rows/oracles; no raw prompt or "
                "oracle content is published here",
    }
    _write_private(out_root / "traces" / "golden" / "eligibility.json", evaluation)
    _write_private(out_root / "traces" / "golden" / "freeze-manifest.json", manifest)

    print(f"golden bank status: {evaluation['bank_status']}")
    print(f"reserved golden groups: {evaluation['reserved_count']} "
          f"(reserved+eligible: {evaluation['reserved_eligible_count']}, "
          f"eligible but unreserved: {evaluation['non_reserved_eligible_count']})")
    print(f"eligible: {evaluation['eligible_count']} / candidates: {len(evaluation['candidates'])}")
    if evaluation["missing_field_counts"]:
        print(f"missing fields: {evaluation['missing_field_counts']}")
    print(f"artifacts -> {out_root / 'traces' / 'golden'}")
    _update_public_summary("golden", {
        "bank_status": evaluation["bank_status"],
        "eligible_count": evaluation["eligible_count"],
        "ineligible_count": evaluation["ineligible_count"],
        "reserved_count": evaluation["reserved_count"],
        "reserved_eligible_count": evaluation["reserved_eligible_count"],
        "non_reserved_eligible_count": evaluation["non_reserved_eligible_count"],
        "missing_field_counts": evaluation["missing_field_counts"],
        "bank_bounds": evaluation["bank_bounds"],
        "blocker": evaluation["blocker"],
    })
    return 0


_GOLDEN_PRODUCTION_FILES = frozenset({"eligibility.json", "freeze-manifest.json"})

MIGRATION_AUTHORIZED_BY = "Main"
MIGRATION_REASON = (
    "partition.py redistributed hash-reserved golden groups that lacked oracle "
    "eligibility into training partitions, leaking the holdout into training; "
    "reservation must be independent of eligibility. Audited pre-calibration "
    "migration: no model calibration ran."
)


def golden_consumption_evidence(out_root: Path) -> list[str]:
    """Artifacts showing the golden holdout was frozen for execution or consumed.

    Any file beside the two golden producers (execution results, evaluation
    reports, ...) under traces/golden, and any golden-named metric, counts as
    evidence. AppleDouble sidecar files are not evidence.
    """
    evidence: list[str] = []
    golden_dir = out_root / "traces" / "golden"
    if golden_dir.is_dir():
        for path in sorted(golden_dir.rglob("*")):
            if (path.is_file() and not path.name.startswith("._")
                    and path.name not in _GOLDEN_PRODUCTION_FILES):
                evidence.append(str(path))
    metrics_dir = out_root / "metrics"
    if metrics_dir.is_dir():
        for path in sorted(metrics_dir.rglob("*")):
            if (path.is_file() and not path.name.startswith("._")
                    and "golden" in path.name.lower()):
                evidence.append(str(path))
    return evidence


def cmd_migrate_reservations(args) -> int:
    """Audited pre-calibration migration of frozen assignments.

    Restores hash-reserved golden groups that the old redistribution defect
    routed into training. Fail-closed: refuses while frozen golden tasks or
    golden-consumption evidence exist, or on any assignment divergence outside
    the reserved golden slice. Backs up the agent-created assignment files with
    sha256 digests, regenerates correct reservations, and records an audit
    reason before reporting the new reserved counts.
    """
    config = load_config(Path(args.config))
    out_root = resolve_output_root(args.output_root)
    salt = config["partition"]["salt"]
    partition_dir = out_root / "traces" / "partition"
    golden_dir = out_root / "traces" / "golden"
    frozen_path = Path(args.frozen_assignments) if args.frozen_assignments else (
        partition_dir / "frozen-assignments.json")
    if not frozen_path.is_file():
        raise TraceError(f"nothing to migrate; missing {frozen_path}")
    frozen_doc = json.loads(frozen_path.read_text())
    old_assignments = frozen_doc.get("assignments")
    if not isinstance(old_assignments, dict) or not old_assignments:
        raise TraceError(f"frozen assignments file has no assignments: {frozen_path}")

    # Pre-calibration preconditions: no frozen golden, no consumption evidence.
    freeze_manifest_path = golden_dir / "freeze-manifest.json"
    freeze_manifest = (
        json.loads(freeze_manifest_path.read_text())
        if freeze_manifest_path.is_file() else None
    )
    guard_pre_calibration_migration(freeze_manifest, golden_consumption_evidence(out_root))

    frozen_salt = frozen_doc.get("salt")
    if frozen_salt is not None and frozen_salt != salt:
        raise TraceError(
            f"refusing migration: frozen salt {frozen_salt!r} does not match "
            f"config salt {salt!r}"
        )

    # Pure recomputation first; nothing is written until every check passes.
    regenerated, changed = regenerated_reservations(salt, old_assignments)
    reserved_ids = {
        gid for gid, part in regenerated.items() if part == "golden"
    }

    eligibility_path = golden_dir / "eligibility.json"
    eligibility_doc = (
        json.loads(eligibility_path.read_text())
        if eligibility_path.is_file() else None
    )
    eligible_ids: set[str] = set()
    if eligibility_doc:
        eligible_ids = {
            c["task_id"] for c in eligibility_doc.get("candidates", []) if c.get("eligible")
        }

    summary = None
    if eligibility_doc is not None:
        summary = summarize_golden_candidates(
            eligibility_doc["candidates"],
            min_tasks=config["partition"].get("golden_min_tasks", 50),
            max_tasks=config["partition"].get("golden_max_tasks", 200),
            reserved_ids=reserved_ids,
        )
        would_freeze = [
            c for c in summary["candidates"] if c["reserved"] and c["eligible"]
        ]
        if would_freeze:
            raise TraceError(
                f"refusing migration: {len(would_freeze)} reserved+eligible task(s) "
                "would freeze golden before the migration"
            )

    # Backup agent-created assignment files with digests before any rewrite.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = partition_dir / "audit" / f"{stamp}-golden-reservation-migration"
    backup_dir = audit_dir / "pre-migration"
    ensure_dir(backup_dir)
    backups = []
    task_path = partition_dir / "task-assignments.json"
    for path in (frozen_path, task_path):
        if path.is_file():
            digest = file_sha256(path)
            copy = backup_dir / path.name
            copy.write_bytes(path.read_bytes())
            if file_sha256(copy) != digest:
                raise TraceError(f"backup digest mismatch for {path.name}")
            backups.append({
                "file": path.name, "source": str(path), "sha256": digest,
                "backup": str(copy), "backup_sha256": digest,
            })
    (audit_dir / "SHA256SUMS").write_text(
        "".join(f"{b['sha256']}  pre-migration/{b['file']}\n" for b in backups)
    )

    atomic_write_json(frozen_path, {
        "schema_version": "1.0.0",
        "note": "frozen assignments are immutable; golden entries are "
                "eligibility-independent reservations (reserved+ineligible "
                "stays not_ready, never training); later runs must reuse this file",
        "salt": salt,
        "assignments": regenerated,
        "migration": {
            "kind": "golden-reservation-migration",
            "authorized_by": MIGRATION_AUTHORIZED_BY,
            "reason": MIGRATION_REASON,
            "audit": str(audit_dir / "migration-audit.json"),
        },
    })

    old_task_doc = json.loads(task_path.read_text()) if task_path.is_file() else {}
    result = partition_tasks(
        [{"group_id": gid} for gid in regenerated],
        salt,
        frozen_assignments=regenerated,
        golden_eligible_ids=eligible_ids,
    )
    if old_task_doc.get("quarantined_groups"):
        result["quarantined_groups"] = old_task_doc["quarantined_groups"]
    _write_private(task_path, result)

    written = [frozen_path, task_path]
    golden_public = None
    if summary is not None:
        new_manifest = {
            "schema_version": "1.0.0",
            "kind": "mimo-halo-golden-freeze",
            "bank_status": summary["bank_status"],
            "eligible_count": summary["eligible_count"],
            "ineligible_count": summary["ineligible_count"],
            "reserved_count": summary["reserved_count"],
            "reserved_eligible_count": summary["reserved_eligible_count"],
            "non_reserved_eligible_count": summary["non_reserved_eligible_count"],
            "missing_field_counts": summary["missing_field_counts"],
            "blocker": summary["blocker"],
            "frozen_tasks": [
                {"task_id": c["task_id"], "repo_identifier": c["repo_identifier"],
                 "starting_revision": c["starting_revision"], "issue_ref": c["issue_ref"]}
                for c in summary["candidates"] if c["reserved"] and c["eligible"]
            ],
            "note": "frozen tasks are reserved+eligible groups only; a training group "
                    "is never imported into golden when an oracle appears. Frozen tasks "
                    "reference privately held repos/rows/oracles; no raw prompt or "
                    "oracle content is published here",
        }
        _write_private(eligibility_path, summary)
        _write_private(freeze_manifest_path, new_manifest)
        written += [eligibility_path, freeze_manifest_path]
        golden_public = {
            "bank_status": summary["bank_status"],
            "eligible_count": summary["eligible_count"],
            "ineligible_count": summary["ineligible_count"],
            "reserved_count": summary["reserved_count"],
            "reserved_eligible_count": summary["reserved_eligible_count"],
            "non_reserved_eligible_count": summary["non_reserved_eligible_count"],
            "missing_field_counts": summary["missing_field_counts"],
            "bank_bounds": summary["bank_bounds"],
            "blocker": summary["blocker"],
        }

    _update_public_summary("partition", {
        "task_groups_assigned": len(regenerated),
        "counts": result["counts"],
        "quarantined_identity_failures": len(result.get("quarantined_groups", [])),
        "frozen_respected": result["frozen_respected"],
        "reserved_golden_groups": result["reserved_golden_count"],
        "reserved_golden_ready": len(result["reserved_ready"]),
        "reserved_golden_not_ready": len(result["reserved_not_ready"]),
    })
    if golden_public is not None:
        _update_public_summary("golden", golden_public)

    counts_before: dict[str, int] = {}
    for part in old_assignments.values():
        counts_before[part] = counts_before.get(part, 0) + 1
    audit = {
        "schema_version": "1.0.0",
        "kind": "mimo-halo-golden-reservation-migration-audit",
        "timestamp_utc": stamp,
        "authorized_by": MIGRATION_AUTHORIZED_BY,
        "reason": MIGRATION_REASON,
        "preconditions": {
            "no_model_calibration_ran": True,
            "frozen_golden_tasks": 0,
            "golden_consumption_evidence": [],
            "frozen_salt_matches_config": True,
        },
        "salt": salt,
        "groups_migrated": len(regenerated),
        "assignments_changed": len(changed),
        "changed_groups_all_reserved_golden": True,
        "counts_before": dict(sorted(counts_before.items())),
        "counts_after": result["counts"],
        "reserved_golden_count": result["reserved_golden_count"],
        "reserved_golden_ready": len(result["reserved_ready"]),
        "reserved_golden_not_ready": len(result["reserved_not_ready"]),
        "eligible_count": summary["eligible_count"] if summary else None,
        "bank_status": summary["bank_status"] if summary else None,
        "backups": backups,
        "files_written": [
            {"path": str(path), "sha256": file_sha256(path)} for path in written
        ],
    }
    atomic_write_json(audit_dir / "migration-audit.json", audit)

    print(f"migrated {len(regenerated)} frozen assignments; "
          f"changed {len(changed)} (all reserved golden)")
    print(f"counts: {result['counts']}")
    print(f"reserved golden groups: {result['reserved_golden_count']} "
          f"(ready: {len(result['reserved_ready'])}, "
          f"not_ready: {len(result['reserved_not_ready'])})")
    if summary is not None:
        print(f"golden bank status: {summary['bank_status']} "
              f"(eligible: {summary['eligible_count']})")
        if summary["blocker"]:
            print(f"blocker: {summary['blocker']}")
    print(f"audit -> {audit_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m mimo_halo.traces",
        description="Read-only harness trace discovery, normalization, "
                    "grouped partitioning and golden eligibility.",
    )
    parser.add_argument("--config", default=str(CONFIG_PATH), help="dataset config path")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--output-root", required=True,
                       help="private output root OUTSIDE the Git repository (e.g. MIMO_LAB workspace)")
        p.add_argument("--max-sessions", type=int, default=None,
                       help="cap sessions parsed per run (light runs for low-capacity volumes)")

    p = sub.add_parser("discover", help="aggregate read-only discovery")
    add_common(p)
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("normalize", help="normalize sessions to the episode schema")
    add_common(p)
    p.add_argument("--include-text", action="store_true",
                   help="include credential-redacted bounded text (private output only)")
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("partition", help="deterministic grouped split")
    add_common(p)
    p.add_argument("--frozen-assignments", default=None,
                   help="path to frozen assignments JSON (immutable once written)")
    p.add_argument("--golden-eligible", default="",
                   help="path to golden eligibility JSON from 'freeze'")
    p.set_defaults(func=cmd_partition)

    p = sub.add_parser("freeze", help="golden eligibility and bank freeze manifest")
    add_common(p)
    p.add_argument("--oracle-index", default=None,
                   help="private JSON mapping task_id -> test oracle descriptor")
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser(
        "migrate-reservations",
        help="audited pre-calibration migration restoring eligibility-independent golden reservations")
    add_common(p)
    p.add_argument("--frozen-assignments", default=None,
                   help="path to the frozen assignments JSON to migrate "
                        "(default: <output-root>/traces/partition/frozen-assignments.json)")
    p.set_defaults(func=cmd_migrate_reservations)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return args.func(args)
    except TraceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _repo_head() -> str | None:
    git_head = REPO_ROOT / ".git" / "HEAD"
    if not git_head.is_file():
        return None
    return git_head.read_text().strip()


if __name__ == "__main__":
    raise SystemExit(main())
