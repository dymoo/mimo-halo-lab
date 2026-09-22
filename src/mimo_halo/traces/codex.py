"""Codex session parser (~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl).

Observed real event shapes (evidence from local sessions, 2025-2026):
- Every line: {"timestamp", "type", "payload"}.
- First record: payload session_meta with keys id, timestamp, cwd, cli_version,
  originator, source, model_provider, instructions, git{branch, commit_hash,
  repository_url}, and on resumed threads parent_thread_id / forked_from_id.
- Tool calls: payload type "function_call" {name, arguments, call_id} and
  "custom_tool_call" {name, input, call_id}; results: "function_call_output" /
  "custom_tool_call_output" {call_id, output}.
- Reasoning: "reasoning" {summary, encrypted_content} and "agent_reasoning"
  {text}. User/assistant text: "message" {role, content} (older files:
  "user_message"/"agent_message"). Turn lifecycle: task_started/task_complete/
  turn_aborted. Other observed: token_count, patch_apply_end, mcp_tool_call_end,
  web_search_call/end, item_completed, context_compacted, thread_goal_updated,
  sub_agent_activity, error.
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
            if isinstance(b, dict):
                if isinstance(b.get("text"), str):
                    parts.append(b["text"])
                elif isinstance(b.get("input_text"), str):
                    parts.append(b["input_text"])
                elif isinstance(b.get("output_text"), str):
                    parts.append(b["output_text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


_CALL_TYPES = {"function_call": "arguments", "custom_tool_call": "input",
               "local_shell_call": "action"}
_RESULT_TYPES = {"function_call_output", "custom_tool_call_output", "local_shell_call_output"}


def parse_codex_session(path: Path) -> SessionTrace:
    trace = SessionTrace(source="codex", session_id=None, source_path_sha256=sha256_path(path))
    events: list[NormalizedEvent] = []
    seq = 0

    def add(ev: NormalizedEvent, hits: int) -> None:
        nonlocal seq
        ev.seq = seq
        seq += 1
        events.append(ev)
        trace.redaction_hits += hits

    for _lineno, d in iter_jsonl(path):
        ts = d.get("timestamp")
        payload = d.get("payload")
        if not isinstance(payload, dict):
            continue
        ptype = payload.get("type")

        if "cli_version" in payload and "id" in payload and ptype is None:
            # session_meta
            trace.session_id = payload.get("id")
            trace.cwd_sha256 = repo_key_for_cwd(payload.get("cwd"))
            for key in ("parent_thread_id", "forked_from_id"):
                if payload.get(key):
                    trace.parent_session_ids.append(payload[key])
            git = payload.get("git")
            if isinstance(git, dict):
                trace.branch = git.get("branch")
                commit = git.get("commit_hash")
                if commit:
                    trace.start_revision = commit
                repo_url = git.get("repository_url")
                if repo_url:
                    trace.repo_identifier = repo_url
            started = payload.get("timestamp")
            if started and trace.started_at is None:
                trace.started_at = started
            continue

        if ptype in _CALL_TYPES:
            name = payload.get("name")
            raw_args = payload.get(_CALL_TYPES[ptype])
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    pass
            digest, nchars, verif, spam = inspect_tool_arguments(name, raw_args)
            add(NormalizedEvent(seq=0, type="tool_call", timestamp=ts,
                                tool_call_id=payload.get("call_id") or payload.get("id"),
                                tool_name=name, arguments_sha256=digest, arguments_chars=nchars,
                                verification_hint=verif, spam_hint=spam,
                                source_type=ptype, tool_kind="call"), 0)
        elif ptype in _RESULT_TYPES:
            text, digest, n, hits = bounded_redacted_text(str(payload.get("output") or ""))
            add(NormalizedEvent(seq=0, type="tool_result", timestamp=ts,
                                tool_call_id=payload.get("call_id"), text=text,
                                raw_text_sha256=digest, raw_text_chars=n,
                                result_is_error=_output_is_error(payload.get("output")),
                                source_type=ptype, tool_kind="result"), hits)
        elif ptype == "message":
            role = payload.get("role")
            text, digest, n, hits = bounded_redacted_text(_content_text(payload.get("content")))
            if role == "user" and trace.first_user_prompt is None:
                trace.first_user_prompt = text
            add(NormalizedEvent(seq=0, type="user" if role == "user" else "assistant",
                                timestamp=ts, text=text, raw_text_sha256=digest, raw_text_chars=n,
                                source_type=ptype), hits)
        elif ptype == "user_message":
            text, digest, n, hits = bounded_redacted_text(str(payload.get("message") or ""))
            if trace.first_user_prompt is None:
                trace.first_user_prompt = text
            add(NormalizedEvent(seq=0, type="user", timestamp=ts, text=text,
                                raw_text_sha256=digest, raw_text_chars=n, source_type=ptype), hits)
        elif ptype == "agent_message":
            text, digest, n, hits = bounded_redacted_text(str(payload.get("message") or ""))
            add(NormalizedEvent(seq=0, type="assistant", timestamp=ts, text=text,
                                raw_text_sha256=digest, raw_text_chars=n, source_type=ptype), hits)
        elif ptype in ("agent_reasoning", "reasoning"):
            if ptype == "reasoning":
                text_raw = _content_text(payload.get("summary")) if payload.get("summary") else ""
            else:
                text_raw = str(payload.get("text") or "")
            text, digest, n, hits = bounded_redacted_text(text_raw)
            add(NormalizedEvent(seq=0, type="reasoning", timestamp=ts, text=text,
                                raw_text_sha256=digest, raw_text_chars=n, source_type=ptype), hits)
        elif ptype == "task_started" and trace.started_at is None:
            trace.started_at = payload.get("started_at") or ts
        elif ptype in ("task_complete", "turn_aborted"):
            ended = payload.get("completed_at") or ts
            if ended:
                trace.ended_at = ended
        # everything else (token_count, web_search_*, item_completed, ...) is meta

    trace.events = events
    trace.sidechain_sessions = 0
    trace.has_branching = bool(trace.parent_session_ids)
    return trace


def _output_is_error(output) -> bool | None:
    if isinstance(output, str):
        low = output[:200].lower()
        if low.startswith("error") or low.startswith("{\"error"):
            return True
        return False
    if isinstance(output, dict):
        return "error" in output
    return None
