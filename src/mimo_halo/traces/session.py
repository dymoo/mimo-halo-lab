"""Session normalization pipeline: parse -> correlate -> episode segmentation.

Episodes are bounded per-turn segments of the main chain (sidechain events
excluded): a user turn (or a tool_result triggering retry) opens an episode;
assistant decision text/thinking, tool calls/results and verification gather
inside it. Tool-spam-heavy episodes are downweighted per the dataset contract.
"""

from __future__ import annotations

from pathlib import Path

from .common import TraceError, stable_hash
from .events import Episode, NormalizedEvent, SessionTrace, episode_weight

_RECOVERY_WINDOW = 12

_CAPABILITY_TOOL_MAP = {
    "repo_exploration": {"grep", "glob", "read", "ls", "find", "search", "view", "cat", "list", "web_search", "fetch"},
    "planning": {"todo", "plan", "todowrite", "task"},
    "implementation": {"edit", "write", "multiedit", "notebookedit", "apply_patch", "patch", "str_replace", "create"},
    "debugging": {"diagnose", "debug"},
    "compiler_interpretation": {"cargo", "tsc", "rustc", "gcc", "clang"},
    "test_interpretation": {"pytest", "vitest", "jest", "test"},
    "tool_use": {"mcp", "tool"},
    "shell": {"bash", "shell", "terminal", "exec", "command", "run"},
    "git": {"git"},
    "verification": {"pytest", "vitest", "jest", "test", "make", "lint", "ruff", "mypy", "tsc", "check"},
}


def segment_episodes(trace: SessionTrace) -> list[Episode]:
    """Bounded per-turn episodes over main-chain (non-sidechain) events."""
    episodes: list[Episode] = []
    current: list[NormalizedEvent] = []
    turn_index = 0

    def close() -> None:
        nonlocal current
        if not current:
            return
        tool_calls = sum(1 for e in current if e.type == "tool_call")
        spam = sum(1 for e in current if _is_spam(e))
        verification = sum(1 for e in current if _is_verification(e))
        has_decision = any(
            e.type in ("assistant", "reasoning") and (e.text or "").strip() for e in current
        )
        recovered = _episode_recovered(current)
        caps = _capabilities(current)
        weight = episode_weight(spam, len(current), has_decision, verification)
        if caps:
            weight *= 1.0 + 0.05 * ("debugging" in caps)
        first_ts = current[0].timestamp
        last_ts = next((e.timestamp for e in reversed(current) if e.timestamp), None)
        episodes.append(Episode(
            episode_id=stable_hash({
                "session": trace.source_path_sha256, "turn": turn_index,
                "start": current[0].seq, "end": current[-1].seq,
            }),
            turn_index=turn_index,
            sidechain=False,
            started_at=first_ts,
            ended_at=last_ts,
            event_seq=[e.seq for e in current],
            tool_calls=tool_calls,
            has_decision=has_decision,
            verification_events=verification,
            recovered=recovered,
            downweighted_events=spam,
            capabilities=caps,
            weight=weight,
        ))
        current = []

    for ev in trace.events:
        if ev.sidechain:
            continue
        if ev.type == "user":
            close()
            turn_index += 1
            current.append(ev)
        elif ev.type == "tool_result" and ev.result_is_error and not current:
            # retry after error without a new user turn still opens a bounded episode
            turn_index += 1
            current.append(ev)
        else:
            current.append(ev)
            # cap runaway turns: giant tool spam becomes its own bounded episode
            if len(current) >= 64:
                close()
    close()
    return episodes


def _is_spam(ev: NormalizedEvent) -> bool:
    if ev.spam_hint is not None:
        return ev.spam_hint
    if ev.type == "tool_result" and (ev.raw_text_chars or 0) > 16384:
        return True
    return False


def _is_verification(ev: NormalizedEvent) -> bool:
    if ev.verification_hint is not None:
        return bool(ev.verification_hint)
    if ev.type != "tool_call" or not ev.tool_name:
        return False
    name = ev.tool_name.lower()
    return any(h in name for h in ("test", "check", "lint"))


def _capabilities(events: list[NormalizedEvent]) -> list[str]:
    caps: set[str] = set()
    for ev in events:
        if ev.type == "tool_call" and ev.tool_name:
            name = ev.tool_name.lower()
            for cap, names in _CAPABILITY_TOOL_MAP.items():
                if name in names or any(n in name for n in names if len(n) > 3):
                    caps.add(cap)
    if _is_verification(events[-1]) if events else False:
        caps.add("verification")
    return sorted(caps)


def _episode_recovered(events: list[NormalizedEvent]) -> bool:
    for idx, ev in enumerate(events):
        if ev.type == "tool_result" and ev.result_is_error:
            window = events[idx + 1: idx + 1 + _RECOVERY_WINDOW]
            if any(
                (w.type == "tool_result" and w.result_is_error is False)
                or (w.type == "tool_call" and _is_verification(w))
                for w in window
            ):
                return True
    return False


_PARSERS = {}


def register_parser(source: str, parser):
    _PARSERS[source] = parser


from .claude import parse_claude_session  # noqa: E402
from .codex import parse_codex_session  # noqa: E402
from .pi import parse_omp_session, parse_pi_session  # noqa: E402

register_parser("claude", parse_claude_session)
register_parser("codex", parse_codex_session)
register_parser("pi", parse_pi_session)
register_parser("omp", parse_omp_session)


def normalize_session(path: Path, source: str) -> SessionTrace:
    parser = _PARSERS.get(source)
    if parser is None:
        raise TraceError(f"no parser registered for source {source!r}")
    trace = parser(Path(path))
    trace.correlate()
    return trace


def normalize_sessions(paths: list[tuple[str, Path]], max_sessions: int | None = None) -> list[SessionTrace]:
    """Parse (source, path) pairs deterministically, optionally capped."""
    ordered = sorted(paths, key=lambda sp: (sp[0], str(sp[1])))
    if max_sessions is not None:
        ordered = ordered[:max_sessions]
    return [normalize_session(path, source) for source, path in ordered]
