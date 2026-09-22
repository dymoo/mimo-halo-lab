"""Read-only discovery over the four mandatory harness roots.

Emits aggregate facts only: per-source file counts, byte sizes, hashed paths
(sha256 of the absolute path string, never the path), per-session identity
facts (session id, hashed cwd, repo identifier/revision when the harness
records them) and schema-fact text. Raw payloads stay at origin and are never
copied, uploaded or rewritten.
"""

from __future__ import annotations

import os
from pathlib import Path

from .session import normalize_session

SOURCES = ("claude", "codex", "pi", "omp")

SCHEMA_FACTS = {
    "claude": "records {type:user|assistant|system|attachment|mode|ai-title|...}; "
              "tool_use blocks in assistant.message.content; tool_result blocks in "
              "user.message.content keyed by tool_use_id; parentUuid uuid chains "
              "(repeated parent = fork); isSidechain marks subagent threads; "
              "cwd/gitBranch per record",
    "codex": "lines {timestamp,type,payload}; session_meta payload has id/cwd/"
             "git{branch,commit_hash,repository_url}/parent_thread_id|forked_from_id; "
             "function_call{call_id,name,arguments} <-> function_call_output{call_id,output}; "
             "custom_tool_call/custom_tool_call_output share the call_id scheme; "
             "message/user_message/agent_message/reasoning text records",
    "pi": "entries {type,id,parentId,timestamp}; session entry has cwd; "
          "message entries with message.role user|assistant|toolResult; assistant "
          "content blocks {text|thinking|toolCall{id,name,arguments}}; toolResult "
          "messages carry toolCallId/toolName/isError; parentId chains mark forks; "
          "custom entries (tool_execution_start, goal-state, session_exit, ...)",
    "omp": "same entry schema as pi plus custom_message entries and "
           "credential_pin/compaction/title_change; custom types include "
           "tool_execution_start, goal-state, subagents:record, session_exit",
}


def expand_root(root: str) -> Path:
    return Path(os.path.expanduser(root))


def list_transcripts(source: str, root: Path, glob_pattern: str) -> tuple[list[Path], list[Path]]:
    """Return (session_transcripts, non_transcript_records).

    Claude Code writes operational `journal.jsonl` files alongside session
    transcripts; they are a distinct record class and are counted but not
    parsed as sessions.
    """
    if not root.is_dir():
        return [], []
    files = sorted(p for p in root.glob(glob_pattern) if p.is_file())
    if source == "claude":
        transcripts = [p for p in files if p.name != "journal.jsonl"]
        records = [p for p in files if p.name == "journal.jsonl"]
        return transcripts, records
    return files, []


def discover(config: dict, max_sessions: int | None = None) -> dict:
    """Aggregate discovery across configured sources. Parses narrowly, emits
    hashes/counts/identity facts only."""
    traces_cfg = config["traces"]["sources"]
    per_source: dict[str, dict] = {}
    session_rows: list[dict] = []
    for source in SOURCES:
        cfg = traces_cfg[source]
        root = expand_root(cfg["root"])
        files, non_transcript = list_transcripts(source, root, cfg.get("transcript_glob", "**/*.jsonl"))
        total_bytes = sum(p.stat().st_size for p in files)
        parse_targets = files if max_sessions is None else files[:max_sessions]
        sessions = []
        parse_failures = 0
        for p in parse_targets:
            try:
                t = normalize_session(p, source)
            except Exception:
                parse_failures += 1
                continue
            sessions.append(t)
            session_rows.append(_session_row(t))
        per_source[source] = {
            "root_configured": cfg["root"],
            "files": len(files),
            "non_transcript_records": len(non_transcript),
            "total_bytes": total_bytes,
            "parsed_sessions": len(sessions),
            "parse_failures": parse_failures,
            "empty_sessions": sum(1 for t in sessions if not t.events),
            "event_count": sum(len(t.events) for t in sessions),
            "tool_call_count": sum(sum(1 for e in t.events if e.type == "tool_call") for t in sessions),
            "tool_result_count": sum(sum(1 for e in t.events if e.type == "tool_result") for t in sessions),
            "unmatched_tool_calls": sum(t.unmatched_tool_calls for t in sessions),
            "unmatched_tool_results": sum(t.unmatched_tool_results for t in sessions),
            "sessions_with_repo_identity": sum(1 for t in sessions if t.repo_identifier),
            "sessions_with_start_revision": sum(1 for t in sessions if t.start_revision),
            "sessions_with_branching": sum(1 for t in sessions if t.has_branching),
            "sessions_with_sidechains": sum(1 for t in sessions if t.sidechain_sessions),
            "redaction_hits": sum(t.redaction_hits for t in sessions),
        }
    return {
        "per_source": per_source,
        "sessions": session_rows,
        "extra_adapters_needed": config["traces"].get("extra_adapters_needed", {}),
        "schema_facts": SCHEMA_FACTS,
    }


def _session_row(t) -> dict:
    return {
        "source": t.source,
        "session_id": t.session_id,
        "source_path_sha256": t.source_path_sha256,
        "cwd_sha256": t.cwd_sha256,
        "repo_identifier": t.repo_identifier,
        "branch": t.branch,
        "start_revision": t.start_revision,
        "started_at": t.started_at,
        "ended_at": t.ended_at,
        "models": t.models,
        "parent_session_ids": t.parent_session_ids,
        "has_branching": t.has_branching,
        "event_count": len(t.events),
        "tool_call_count": sum(1 for e in t.events if e.type == "tool_call"),
        "tool_result_count": sum(1 for e in t.events if e.type == "tool_result"),
    }
