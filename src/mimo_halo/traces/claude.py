"""Claude Code transcript parser (~/.claude/projects/**/*.jsonl).

Observed real event shapes (evidence from local sessions, 2026):
- user:  {"type":"user","uuid","parentUuid","sessionId","timestamp","cwd",
          "gitBranch","isSidechain","message":{"role":"user","content":[...]}}
- assistant: {"type":"assistant", "message":{"role":"assistant","content":[
      {"type":"text","text"}, {"type":"thinking","thinking"},
      {"type":"tool_use","id","name","input"}], "model", "usage", ...}}
- tool results arrive as user messages with content blocks
  {"type":"tool_result","tool_use_id","content","is_error"}.
- Non-message record types (last-prompt, mode, ai-title, queue-operation,
  system, bridge-session, file-history-delta, attachment, ...) are meta.
Branching: parentUuid chains; repeated parents mark forks. isSidechain marks
subagent threads; sidechain events are excluded from main-chain episodes.
"""

from __future__ import annotations

from pathlib import Path

from .common import iter_jsonl, repo_key_for_cwd, sha256_path
from .events import NormalizedEvent, SessionTrace, bounded_redacted_text, inspect_tool_arguments


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def parse_claude_session(path: Path) -> SessionTrace:
    trace = SessionTrace(source="claude", session_id=None, source_path_sha256=sha256_path(path))
    events: list[NormalizedEvent] = []
    parents: dict[str, str | None] = {}
    seen_cwds: set[str] = set()
    seq = 0

    def add(ev: NormalizedEvent, hits: int) -> None:
        nonlocal seq
        ev.seq = seq
        seq += 1
        events.append(ev)
        trace.redaction_hits += hits

    for _lineno, d in iter_jsonl(path):
        rtype = d.get("type")
        ts = d.get("timestamp")
        if trace.session_id is None and d.get("sessionId"):
            trace.session_id = d["sessionId"]
        if d.get("cwd"):
            seen_cwds.add(d["cwd"])
        if d.get("gitBranch") and trace.branch is None:
            trace.branch = d["gitBranch"]
        if rtype not in ("user", "assistant"):
            continue
        uuid = d.get("uuid")
        parent = d.get("parentUuid")
        if uuid:
            if parent in parents and parents.get(parent) != uuid:
                trace.has_branching = True
            parents[uuid] = parent
        msg = d.get("message") or {}
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        sidechain = bool(d.get("isSidechain"))
        model = msg.get("model")
        if model and model not in trace.models:
            trace.models.append(model)

        if isinstance(content, str) or content is None:
            text_raw = content if isinstance(content, str) else ""
            if not text_raw:
                continue
            text, digest, n, hits = bounded_redacted_text(text_raw)
            if rtype == "user" and trace.first_user_prompt is None:
                trace.first_user_prompt = text
            add(NormalizedEvent(seq=0, type=rtype, timestamp=ts, text=text,
                                raw_text_sha256=digest, raw_text_chars=n, sidechain=sidechain,
                                source_type=rtype, parent_event_id=parent, model=model), hits)
            continue

        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                digest, nchars, verif, spam = inspect_tool_arguments(block.get("name"), block.get("input"))
                add(NormalizedEvent(seq=0, type="tool_call", timestamp=ts,
                                    tool_call_id=block.get("id"), tool_name=block.get("name"),
                                    arguments_sha256=digest, arguments_chars=nchars,
                                    verification_hint=verif, spam_hint=spam,
                                    sidechain=sidechain, source_type=f"{rtype}:tool_use",
                                    parent_event_id=parent, tool_kind="call", model=model), 0)
            elif btype == "tool_result":
                text, digest, n, hits = bounded_redacted_text(_result_text(block.get("content")))
                add(NormalizedEvent(seq=0, type="tool_result", timestamp=ts,
                                    tool_call_id=block.get("tool_use_id"), text=text,
                                    raw_text_sha256=digest, raw_text_chars=n,
                                    result_is_error=bool(block.get("is_error")),
                                    sidechain=sidechain, source_type=f"{rtype}:tool_result",
                                    parent_event_id=parent, tool_kind="result"), hits)
            elif btype == "text":
                text, digest, n, hits = bounded_redacted_text(block.get("text") or "")
                if not text and not digest:
                    continue
                if rtype == "user" and trace.first_user_prompt is None:
                    trace.first_user_prompt = text
                add(NormalizedEvent(seq=0, type=rtype, timestamp=ts, text=text,
                                    raw_text_sha256=digest, raw_text_chars=n, sidechain=sidechain,
                                    source_type=f"{rtype}:text", parent_event_id=parent,
                                    model=model), hits)
            elif btype == "thinking":
                text, digest, n, hits = bounded_redacted_text(block.get("thinking") or "")
                add(NormalizedEvent(seq=0, type="reasoning", timestamp=ts, text=text,
                                    raw_text_sha256=digest, raw_text_chars=n, sidechain=sidechain,
                                    source_type=f"{rtype}:thinking", parent_event_id=parent,
                                    model=model), hits)
            # other block types (image, ...) intentionally not retained

    trace.events = events
    trace.cwd_sha256 = repo_key_for_cwd(next(iter(sorted(seen_cwds)), None))
    trace.sidechain_sessions = 1 if any(e.sidechain for e in events) else 0
    if events:
        trace.started_at = next((e.timestamp for e in events if e.timestamp), None)
        trace.ended_at = next((e.timestamp for e in reversed(events) if e.timestamp), None)
    return trace
