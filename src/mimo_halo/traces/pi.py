"""Pi and OMP session parsers.

Both harnesses use the same JSONL entry schema (evidence from local sessions):
- session: {"type":"session","id","timestamp","cwd","version"}
- message: {"type":"message","id","parentId","timestamp",
            "message":{"role","content":[...],"provider","model","usage",
            "stopReason","timestamp","toolCallId","toolName","isError",...}}
- custom:  {"type":"custom","customType","data","id","parentId","timestamp"}
- model_change / thinking_level_change / session_info / compaction / ...

Roles observed: user, assistant, toolResult (with toolCallId/toolName/isError),
developer. Assistant content blocks: {"type":"text"|"thinking"|"toolCall",
"id","name","arguments"}. Custom entries include tool_execution_start,
goal-state, session_exit, etc. OMP adds custom_message entries and
credential_pin/compaction entries.

Branching: entries chain by parentId; a parent that is not the immediately
preceding entry id marks a fork/retry branch. OMP subagent sessions live in
sibling directories and parse as independent sessions.
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import iter_jsonl, repo_key_for_cwd, sha256_path
from .events import NormalizedEvent, SessionTrace, bounded_redacted_text, inspect_tool_arguments





def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                parts.append(b["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def parse_pi_style_session(path: Path, source: str) -> SessionTrace:
    trace = SessionTrace(source=source, session_id=None, source_path_sha256=sha256_path(path))
    events: list[NormalizedEvent] = []
    last_entry_id: str | None = None
    seq = 0

    def add(ev: NormalizedEvent, hits: int) -> None:
        nonlocal seq
        ev.seq = seq
        seq += 1
        events.append(ev)
        trace.redaction_hits += hits

    for _lineno, d in iter_jsonl(path):
        etype = d.get("type")
        ts = d.get("timestamp")
        eid = d.get("id")
        parent = d.get("parentId")
        if last_entry_id is not None and parent is not None and parent != last_entry_id:
            trace.has_branching = True
        last_entry_id = eid

        if etype == "session":
            trace.session_id = d.get("id")
            trace.cwd_sha256 = repo_key_for_cwd(d.get("cwd"))
            trace.started_at = ts if trace.started_at is None else trace.started_at
            continue
        if etype == "model_change":
            model = d.get("modelId")
            if model and model not in trace.models:
                trace.models.append(model)
            continue
        if etype != "message":
            # custom / custom_message / compaction / ... stay meta
            continue

        msg = d.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        model = msg.get("model")
        if model and model not in trace.models:
            trace.models.append(model)
        content = msg.get("content")

        if role == "toolResult":
            text, digest, n, hits = bounded_redacted_text(_content_text(content))
            details = msg.get("details")
            is_error = msg.get("isError")
            if is_error is None and isinstance(details, dict):
                is_error = details.get("isError")
            add(NormalizedEvent(seq=0, type="tool_result", timestamp=ts,
                                tool_call_id=msg.get("toolCallId"),
                                tool_name=msg.get("toolName"), text=text,
                                raw_text_sha256=digest, raw_text_chars=n,
                                result_is_error=bool(is_error) if is_error is not None else None,
                                source_type=f"message:{role}", tool_kind="result"), hits)
            continue

        if not isinstance(content, list):
            if isinstance(content, str) and content:
                text, digest, n, hits = bounded_redacted_text(content)
                if role == "user" and trace.first_user_prompt is None:
                    trace.first_user_prompt = text
                add(NormalizedEvent(seq=0, type="user" if role == "user" else "assistant",
                                    timestamp=ts, text=text, raw_text_sha256=digest,
                                    raw_text_chars=n, source_type=f"message:{role}",
                                    model=model), hits)
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "toolCall":
                digest, nchars, verif, spam = inspect_tool_arguments(block.get("name"), block.get("arguments"))
                add(NormalizedEvent(seq=0, type="tool_call", timestamp=ts,
                                    tool_call_id=block.get("id"), tool_name=block.get("name"),
                                    arguments_sha256=digest, arguments_chars=nchars,
                                    verification_hint=verif, spam_hint=spam,
                                    source_type=f"message:{btype}", tool_kind="call",
                                    model=model), 0)
            elif btype == "text":
                text, digest, n, hits = bounded_redacted_text(block.get("text") or "")
                if role == "user" and trace.first_user_prompt is None:
                    trace.first_user_prompt = text
                add(NormalizedEvent(seq=0, type="user" if role == "user" else "assistant",
                                    timestamp=ts, text=text, raw_text_sha256=digest,
                                    raw_text_chars=n, source_type=f"message:{btype}",
                                    model=model), hits)
            elif btype == "thinking":
                text, digest, n, hits = bounded_redacted_text(block.get("thinking") or "")
                add(NormalizedEvent(seq=0, type="reasoning", timestamp=ts, text=text,
                                    raw_text_sha256=digest, raw_text_chars=n,
                                    source_type=f"message:{btype}", model=model), hits)
            # image and unknown blocks intentionally not retained

    trace.events = events
    trace.sidechain_sessions = 0
    if events:
        trace.started_at = trace.started_at or next(
            (e.timestamp for e in events if e.timestamp), None)
        trace.ended_at = next((e.timestamp for e in reversed(events) if e.timestamp), None)
    return trace


def parse_pi_session(path: Path) -> SessionTrace:
    return parse_pi_style_session(path, source="pi")


def parse_omp_session(path: Path) -> SessionTrace:
    return parse_pi_style_session(path, source="omp")
