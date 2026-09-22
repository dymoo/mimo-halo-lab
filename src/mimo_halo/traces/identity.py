"""Task identity: group session attempts that worked the same underlying task.

Rules (dataset contract):
- An underlying task requires explicit linkage: a repo working directory
  (hashed cwd / repo identifier) plus an explicit issue/ticket reference found
  in the user prompt, OR same-harness retry dedup by first-user-prompt content
  hash within one repo.
- Cross-harness grouping happens ONLY through the explicit issue reference;
  prompt-hash-only groups from different harnesses are quarantined as
  ambiguous, never merged and never randomly split.
- Dedup: identical first-prompt content hash inside the same group collapses
  retry attempts into one candidate.
"""

from __future__ import annotations

import re

from .common import sha256_text
from .events import SessionTrace

DEFAULT_ISSUE_PATTERN = (
    r"\b[A-Z][A-Z0-9]{1,14}-\d{1,7}\b"          # Jira-style ABC-123
    r"|\b(?:issue|ticket|bug)\s*#?\d{1,6}\b"     # issue #123 / ticket 42
    r"|\bGH-\d{1,6}\b"
)
_PLAIN_HASH_PATTERN = r"#\d{1,6}\b"

_CAPABILITY_WEIGHTS = {  # contract calibration weights, applied at aggregation time
    "repo_exploration": 0.15, "planning": 0.15, "implementation": 0.20,
    "debugging": 0.15, "compiler_interpretation": 0.075, "test_interpretation": 0.075,
    "tool_use": 0.10, "shell": 0.05, "git": 0.05, "verification": 0.0,
}


def find_issue_refs(text: str | None, pattern: str = DEFAULT_ISSUE_PATTERN) -> list[str]:
    if not text:
        return []
    return sorted(set(re.findall(pattern, text)))


def _plain_hash_refs(text: str | None) -> list[str]:
    if not text:
        return []
    return sorted(set(re.findall(_PLAIN_HASH_PATTERN, text)))


def build_task_groups(
    traces: list[SessionTrace],
    issue_pattern: str = DEFAULT_ISSUE_PATTERN,
) -> dict:
    """Group sessions into task candidates.

    Returns groups with evidence and a quarantine list. Session identity is the
    source_path_sha256, so no raw paths ever leave the private index.
    """
    explicit: dict[str, list] = {}      # (repo_key, issue_ref) -> sessions
    prompt_hash: dict[str, list] = {}   # (repo_key, prompt_hash) -> sessions (cross-harness aware)
    quarantine: list[dict] = []

    for t in traces:
        repo_key = t.cwd_sha256
        if not repo_key:
            quarantine.append({
                "reason": "no_resolvable_cwd",
                "session_ref": t.source_path_sha256[:16],
            })
            continue
        refs = find_issue_refs(t.first_user_prompt, issue_pattern)
        if refs:
            for ref in refs:
                explicit.setdefault((repo_key, ref), []).append(t)
        prompt_digest = sha256_text(t.first_user_prompt or "")
        if t.first_user_prompt:
            prompt_hash.setdefault((repo_key, prompt_hash_key(prompt_digest)), []).append(t)
        elif not refs:
            quarantine.append({
                "reason": "no_user_prompt_no_issue_ref",
                "session_ref": t.source_path_sha256[:16],
            })

    groups: list[dict] = []
    for (repo_key, ref), sessions in sorted(explicit.items()):
        uniq = _dedup_sessions(sessions)
        sources = sorted({s.source for s in uniq})
        groups.append({
            "group_id": sha256_text(f"explicit|{repo_key}|{ref}")[:24],
            "identity": "explicit_issue_ref",
            "issue_ref": ref,
            "repo_key": repo_key,
            "sources": sources,
            "session_count": len(uniq),
            "session_refs": [s.source_path_sha256 for s in uniq],
            "repo_identifier": _first_value(uniq, "repo_identifier"),
            "start_revision": _first_value(uniq, "start_revision"),
            "branch": _first_value(uniq, "branch"),
            "task_prompt": _first_value(uniq, "first_user_prompt"),
            "ambiguous": False,
        })

    for (repo_key, pdigest), sessions in sorted(prompt_hash.items()):
        # skip sessions already covered by an explicit group
        covered_refs = {s.source_path_sha256 for sessions_list in explicit.values() for s in sessions_list}
        remaining = [s for s in sessions if s.source_path_sha256 not in covered_refs]
        if not remaining:
            continue
        sources = sorted({s.source for s in remaining})
        if len(sources) > 1:
            # cross-harness prompt match without explicit linkage: quarantine
            # the whole ambiguous cluster (never merged, never split per harness)
            quarantine.append({
                "reason": "cross_harness_prompt_match_without_explicit_linkage",
                "repo_key": repo_key,
                "prompt_sha256": pdigest,
                "sources": sources,
            })
            continue
        uniq = _dedup_sessions(remaining)
        groups.append({
            "group_id": sha256_text(f"prompt|{sources[0]}|{repo_key}|{pdigest}")[:24],
            "identity": "same_harness_prompt_hash",
            "issue_ref": None,
            "repo_key": repo_key,
            "sources": sources,
            "session_count": len(uniq),
            "session_refs": [s.source_path_sha256 for s in uniq],
            "repo_identifier": _first_value(uniq, "repo_identifier"),
            "start_revision": _first_value(uniq, "start_revision"),
            "branch": _first_value(uniq, "branch"),
            "task_prompt": _first_value(uniq, "first_user_prompt"),
            "ambiguous": False,
        })

    return {"groups": groups, "quarantined": quarantine}


def prompt_hash_key(digest: str) -> str:
    return digest[:16]


def _dedup_sessions(sessions: list) -> list:
    """Collapse retry attempts with identical first-prompt content hash."""
    seen: dict[str, object] = {}
    for s in sorted(sessions, key=lambda x: x.source_path_sha256):
        digest = sha256_text(s.first_user_prompt or "")
        seen.setdefault(digest, s)
    return sorted(seen.values(), key=lambda x: x.source_path_sha256)


def _first_value(sessions: list, attr: str):
    for s in sessions:
        value = getattr(s, attr, None)
        if value:
            return value
    return None
