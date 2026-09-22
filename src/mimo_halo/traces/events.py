"""Normalized event model shared by all harness parsers.

Design invariants:
- Raw provider payloads are never retained; text is credential-redacted and
  bounded, arguments are reduced to hashes.
- Tool calls and results carry the harness' own correlation id so pairing is
  evidence-based, never positional guessing alone.
- Sidechain (subagent) events are marked and excluded from main-chain episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import sha256_text
from .redact import redact

MAX_TEXT_CHARS = 8192

EVENT_TYPES = ("user", "assistant", "tool_call", "tool_result", "reasoning", "system", "custom", "meta")


@dataclass
class NormalizedEvent:
    seq: int
    type: str  # one of EVENT_TYPES
    timestamp: str | None = None
    text: str | None = None  # redacted, bounded
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_sha256: str | None = None
    arguments_chars: int | None = None
    result_is_error: bool | None = None
    verification_hint: bool | None = None  # tool call likely runs tests/build/lint
    spam_hint: bool | None = None  # cheap exploration or build/install spam call
    parent_event_id: str | None = None
    model: str | None = None
    usage: dict | None = None
    sidechain: bool = False
    source_type: str = ""  # original harness event key
    correlated: bool | None = None
    # internal (not serialized)
    raw_text_sha256: str | None = None
    raw_text_chars: int | None = None
    tool_kind: str | None = None  # "call" | "result"

    def to_public(self, include_text: bool) -> dict:
        out: dict = {
            "seq": self.seq,
            "type": self.type,
            "timestamp": self.timestamp,
            "text_sha256": self.raw_text_sha256,
            "text_chars": self.raw_text_chars,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments_sha256": self.arguments_sha256,
            "arguments_chars": self.arguments_chars,
            "result_is_error": self.result_is_error,
            "verification_hint": self.verification_hint,
            "spam_hint": self.spam_hint,
            "parent_event_id": self.parent_event_id,
            "model": self.model,
            "usage": self.usage,
            "sidechain": self.sidechain,
            "source_type": self.source_type,
            "correlated": self.correlated,
        }
        if include_text and self.text is not None:
            out["text"] = self.text
        # omit-null encoding: absent facts are keys with no value stored
        return {k: v for k, v in out.items() if v is not None}


def bounded_redacted_text(text: str) -> tuple[str, str, int, int]:
    """Return (redacted_bounded_text, full_text_sha256, full_text_chars, redaction_hits)."""
    digest = sha256_text(text)
    n = len(text)
    result = redact(text[:MAX_TEXT_CHARS])
    return result.text, digest, n, result.hits


_SPAM_TOOLS_BASE = {"grep", "glob", "ls", "find", "read", "view", "cat", "search", "list"}
_SHELL_TOOLS = {"bash", "shell", "terminal", "exec", "run_command", "command"}
_VERIFICATION_PREFIXES = (
    "pytest", "python -m pytest", "vitest", "jest", "npm test", "npx vitest",
    "cargo test", "go test", "make test", "make check", "ruff", "mypy",
    "tsc", "gradle test", "mvn test", "tox", "unittest",
)
_SPAM_PREFIXES = ("npm install", "pip install", "cargo build", "make ", "yarn install",
                  "pnpm install", "brew install", "apt-get install")


def inspect_tool_arguments(tool_name: str | None, args) -> tuple[str, int, bool | None, bool | None]:
    """Hash tool arguments without retaining content; derive structural hints.

    Returns (arguments_sha256, arguments_chars, verification_hint, spam_hint).
    Hints are inferred from shell command prefixes / tool names only; no
    argument text is retained.
    """
    import json as _json

    try:
        raw = _json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        raw = str(args)
    digest = sha256_text(raw)

    command: str | None = None
    if isinstance(args, dict):
        for key in ("command", "cmd", "script", "input"):
            value = args.get(key)
            if isinstance(value, str):
                command = value.strip()
                break
        if command is None and isinstance(args.get("args"), list) and args["args"]:
            command = " ".join(str(a) for a in args["args"])
    elif isinstance(args, str):
        command = args.strip()

    name = (tool_name or "").lower()
    verification: bool | None = None
    spam: bool | None = None
    if name in _SPAM_TOOLS_BASE:
        spam = True
    elif command is not None:
        low = command.lower()
        verification = any(low.startswith(p) or f" {p}" in low for p in _VERIFICATION_PREFIXES)
        spam = any(low.startswith(p) for p in _SPAM_PREFIXES)
    elif name in _SHELL_TOOLS:
        verification = None  # shell call with unparseable args: unknown
    return digest, len(raw), verification, spam


@dataclass
class Episode:
    episode_id: str
    turn_index: int
    sidechain: bool
    started_at: str | None
    ended_at: str | None
    event_seq: list[int]
    tool_calls: int
    has_decision: bool
    verification_events: int
    recovered: bool
    downweighted_events: int
    capabilities: list[str]
    weight: float

    def to_public(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "turn_index": self.turn_index,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "sidechain": self.sidechain,
            "event_seq": self.event_seq,
            "tool_calls": self.tool_calls,
            "has_decision": self.has_decision,
            "verification_events": self.verification_events,
            "recovered": self.recovered,
            "downweighted_events": self.downweighted_events,
            "capabilities": self.capabilities,
            "weight": round(self.weight, 4),
        }


@dataclass
class SessionTrace:
    """One parsed harness session plus its derived normalized structures."""

    source: str
    session_id: str | None
    source_path_sha256: str
    content_sha256: str | None = None
    cwd_sha256: str | None = None
    repo_identifier: str | None = None
    branch: str | None = None
    start_revision: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    models: list[str] = field(default_factory=list)
    parent_session_ids: list[str] = field(default_factory=list)
    has_branching: bool = False
    sidechain_sessions: int = 0
    events: list[NormalizedEvent] = field(default_factory=list)
    unmatched_tool_calls: int = 0
    unmatched_tool_results: int = 0
    redaction_hits: int = 0
    redaction_kinds: set[str] = field(default_factory=set)
    parse_errors: int = 0
    first_user_prompt: str | None = None  # redacted, bounded; private use only
    verification_commands: list[str] = field(default_factory=list)

    # ---- correlation -------------------------------------------------

    def correlate(self) -> None:
        """Match tool calls to results by the harness' own id.

        Pi/OMP duplicate the id on both blocks; Claude/Codex use explicit
        tool_use_id/call_id. Positional fallback pairs a remaining call with the
        next remaining result of the same tool name ONLY when ids are absent
        on both sides; otherwise items stay unmatched (fail closed, no guessing).
        """
        calls: dict[str, NormalizedEvent] = {}
        results: dict[str, NormalizedEvent] = {}
        anon_calls: list[NormalizedEvent] = []
        anon_results: list[NormalizedEvent] = []
        for ev in self.events:
            if ev.tool_call_id:
                if ev.tool_kind == "call":
                    calls[ev.tool_call_id] = ev
                else:
                    results[ev.tool_call_id] = ev
            elif ev.tool_kind == "call":
                anon_calls.append(ev)
            elif ev.tool_kind == "result":
                anon_results.append(ev)
        matched: set[int] = set()
        for cid, call in calls.items():
            res = results.pop(cid, None)
            call.correlated = res is not None
            if res is not None:
                res.correlated = True
                matched.add(res.seq)
        for cid, res in results.items():
            res.correlated = False
        for res in anon_results:
            match = next((c for c in anon_calls if c.tool_name == res.tool_name and c.seq not in matched), None)
            if match is not None:
                match.correlated = True
                res.correlated = True
                matched.add(match.seq)
            else:
                res.correlated = False
        for c in anon_calls:
            if c.seq not in matched:
                c.correlated = False
        self.unmatched_tool_calls = sum(
            1 for e in self.events if e.tool_kind == "call" and e.correlated is False
        )
        self.unmatched_tool_results = sum(
            1 for e in self.events if e.tool_kind == "result" and e.correlated is False
        )


def episode_weight(tool_spam_events: int, total_events: int, has_decision: bool, verification: int) -> float:
    """Contract episode weight: bounded decision episodes outrank giant tool spam.

    Calibration anchors: 15% exploration, 15% planning, 20% implementation,
    15% debugging, 15% compiler/tests, 10% shell/git/tools, 10% late recovery
    are applied downstream per capability; here we produce the per-episode base.
    """
    if total_events == 0:
        return 0.0
    spam_ratio = tool_spam_events / total_events
    base = 1.0 - 0.5 * spam_ratio
    if has_decision:
        base += 0.2
    base += min(verification, 3) * 0.1
    return max(0.05, base)
