"""Deterministic extractor-side taxonomy labeling for segmented episodes.

Applies the closed label sets of docs/datasets.md ("Taxonomies", "Tagging
rules and boundary cases") to already-segmented episodes as automatically
inferred FEATURES, never truth:

- ``capabilities`` refinement: segmentation's ``_CAPABILITY_TOOL_MAP`` output
  is extended here only where the doc's boundary cases demand it (failed test
  vs confirm run, compiler diagnostic vs hypothesis, recovery, long_context,
  mcp tool_use, verification hints).
- ``primary_capability`` (22-tag enum): exactly one tag per the primary-tag
  rule below.
- ``languages`` (9-tag enum): subject-language evidence from file extensions
  and unambiguous syntax markers in retained (redacted) event text.
- ``difficulty`` (D0..D4): structural heuristic, table below.
- ``downweight_classes`` (5-class enum): per-event attribution of exactly the
  events segmentation already counted in ``downweighted_events``.

Determinism: pure functions over the episode's own normalized events — no
randomness, no network, no model calls. The same trace segments and labels to
byte-identical output on every run.

Omission rules (schemas/episode.schema.json allows every label to be absent;
the labeler omits rather than guesses):
- ``primary_capability``: omitted when ``capabilities`` is empty, or when
  every eligible candidate is the never-primary ``long_context``.
- ``languages``: omitted when the episode shows no subject-code evidence
  (file extension or unambiguous syntax marker). A language *name* alone
  never counts (incidental mentions are never tagged). ``other`` is emitted
  when subject code is evidenced but falls outside the eight named languages
  (C, Ruby, Java, ...); non-subject formats (json, md, lock files) never
  count as code evidence.
- ``difficulty``: always emitted — a pure structural read of the episode
  (span, tool count, decision, recovery, verification, long_context), and D0
  honestly covers the no-decision floor.
- ``downweight_classes``: omitted when the episode downweights nothing, or
  when any downweighted event maps to none of the five observed classes
  (never padded, never guessed). When emitted,
  ``len(downweight_classes) == downweighted_events`` holds by construction.

Primary-tag rule (docs/datasets.md, evaluated in order over capabilities
excluding the cross-cutting ``long_context``):

1. a compiler/typecheck diagnostic was observed in an error result
   -> ``compiler_interpretation`` (the message itself names the cause).
2. an error was corrected by a later edit (fix chain) -> ``debugging``
   (hypothesis across code fixed an unnamed cause; failing-test fixes keep
   ``test_interpretation`` as the secondary tag).
3. an action (edit/apply) with no error context -> ``implementation``.
4. no action at all and a failed test/build ran -> ``test_interpretation``.
5. restoring broken state is the episode's entire point: ``recovery`` and
   nothing but ``shell``/``tool_use`` vehicle tags beside it, no action
   -> ``recovery``.
6. otherwise (no action): the purpose capability of the first tool call the
   decision reasons about, dropping ``shell``/``tool_use`` vehicle tags when
   a purpose tag remains (a clean confirm run lands on ``verification``).
7. any remaining tie -> the lexicographically smallest tag.

Difficulty rule table (first match wins; every feature is monotone, so more
span/tool calls/recovery/verification never lowers the level):

| feature evidence                                                  | level |
|-------------------------------------------------------------------|-------|
| ``long_context`` and ``recovered`` and ``verification_events >= 1`` | D4   |
| ``recovered`` or ``long_context``                                  | D3   |
| ``has_decision`` and (span >= 16 or ``tool_calls`` >= 8)           | D2   |
| ``has_decision``                                                    | D1   |
| otherwise (mechanical / no decision)                                | D0   |

``long_context`` itself fires at span >= 40 events with a decision
(``_LONG_CONTEXT_SPAN``), the structural proxy for the doc's "evidence spread
across many files or earlier turns". Level anchors from docs/datasets.md:
D0 mechanical, D1 bounded, D2 normal (read a few modules to find the seam),
D3 hard/ambiguous (chase failures, cross-module contracts), D4 long-horizon
with verification checkpoints. Levels are categorical: never interpolated,
never averaged.

Downweight attribution (docs/datasets.md "Downweight classes"), walked over
exactly the events segmentation flagged with ``session.is_spam``:

- ``lockfile``: exploration result whose retained text quotes lockfile
  structure or filenames (lockfileVersion, Cargo.lock, go.sum, ...). Paths
  are hashed away in normalized events, so content markers are the
  observable proxy for the doc's path-pattern rule.
- ``duplicate_file_content``: same content hash (result text_sha256) or the
  same read arguments hash already seen earlier in the episode.
- ``enormous_grep_output``: search-family result (grep/rg/find/glob/ls/
  search/list) exceeding the 8192-char output bound.
- ``generated_bundle``: oversized result whose longest retained line is
  >= 2000 chars (minified output); treated as opaque, never quoted.
- ``repeated_build_spam``: build/install/test run events (spam from command
  prefixes) when the episode shows at least two such flagged runs.

If any flagged event matches none of the five classes, ``downweight_classes``
is omitted for the episode (see omission rules).
"""

from __future__ import annotations

import re
from typing import NamedTuple

from .events import Episode, NormalizedEvent, SessionTrace
from .session import (
    _CAPABILITY_TOOL_MAP,
    _episode_recovered,
    _is_spam,
    _is_verification,
)

# Closed enums — kept byte-identical with schemas/episode.schema.json
# (tests/test_labeling.py asserts the equality against the schema file).
CAPABILITIES = (
    "repo_exploration", "planning", "architecture", "implementation",
    "debugging", "compiler_interpretation", "test_interpretation", "tool_use",
    "shell", "git", "refactoring", "code_review", "verification", "recovery",
    "long_context", "dependency_reasoning", "concurrency", "database",
    "frontend", "backend", "systems", "build_tooling",
)
LANGUAGES = (
    "typescript", "javascript", "python", "rust", "go", "cpp", "sql",
    "shell", "other",
)
DIFFICULTY_LEVELS = ("D0", "D1", "D2", "D3", "D4")
DOWNWEIGHT_CLASSES = (
    "lockfile", "enormous_grep_output", "repeated_build_spam",
    "generated_bundle", "duplicate_file_content",
)

_LONG_CONTEXT_SPAN = 40  # events with a decision -> long_context (cross-cutting)
_DIFFICULTY_SPAN = 16    # decision span at/above which a turn reads as D2
_DIFFICULTY_TOOL_CALLS = 8
_GREP_BOUND_CHARS = 8192        # doc: search-tool output bound
_OVERSIZED_RESULT_CHARS = 16384  # session.is_spam result threshold
_MINIFIED_LINE_CHARS = 2000

_VEHICLE_CAPS = frozenset({"shell", "tool_use"})

# Subject-language evidence (docs/datasets.md Languages): extensions of the
# eight named languages; generated artifacts (*.min.js, *.bundle.js, *.d.ts)
# are stripped first because they are opaque, never subject evidence.
_LANGUAGE_EXTENSIONS = {
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".rs": "rust",
    ".go": "go",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".sql": "sql",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ksh": "shell", ".fish": "shell",
}
# Subject code outside the eight named languages -> `other` when evidenced.
_OTHER_CODE_EXTENSIONS = frozenset({
    ".c", ".h", ".rb", ".java", ".php", ".lua", ".kt", ".kts", ".swift",
    ".scala", ".cs", ".m", ".r", ".pl", ".ex", ".exs", ".erl", ".hs",
    ".ml", ".dart", ".jl", ".nim", ".zig", ".vim",
})
# C/C++-family evidence that only implies `other` when cpp is not otherwise
# evidenced (a `.h` next to `.hpp` reads as C++ subject code).
_C_FAMILY_EXTENSIONS = frozenset({".c", ".h"})

_GENERATED_SUFFIX_RE = re.compile(r"\.(?:min|bundle|d)\.[A-Za-z][A-Za-z0-9]*")
_EXTENSION_RE = re.compile(r"\.([A-Za-z][A-Za-z0-9]{0,7})\b(?!\.map)")

_SYNTAX_MARKERS: tuple[tuple[str | None, re.Pattern[str]], ...] = (
    # (language or None -> C-family/other, unambiguous marker)
    ("python", re.compile(r"(?m)^\s*def\s+\w+\([^)]*\)\s*:")),
    ("rust", re.compile(r"\blet\s+mut\b|#\[derive\(")),
    ("go", re.compile(r"(?m)^func\s+\w+\(|\bpackage\s+main\b")),
    ("cpp", re.compile(r"\bstd::")),
    ("sql", re.compile(r"(?i)\bcreate\s+table\b|\binsert\s+into\b")),
    # shebang interpreter line: works for "#!/bin/bash" and the common
    # "#!/usr/bin/env bash" form; \b keeps "shuttle"-style false hits out
    ("shell", re.compile(r"(?m)^#!.*(?:ba|z|k)?sh\b")),
    (None, re.compile(r"(?m)^#include\s*<")),  # C/C++ header include alone
)

_LOCKFILE_RE = re.compile(
    r"lockfileVersion|package-lock\.json|yarn\.lock|Cargo\.lock|go\.sum"
    r"|pnpm-lock\.yaml|poetry\.lock|Gemfile\.lock|composer\.lock"
    r"|\[\[package\]\]|__metadata:",
)
_GENERATED_SUFFIX_MARKERS = re.compile(r"\.(?:min|bundle)\.[a-z]+\b|sourceMappingURL=")

_SEARCH_FAMILY = frozenset({"grep", "rg", "find", "glob", "ls", "search", "list"})


class EpisodeFacts(NamedTuple):
    """Single-pass structural facts shared by capability refinement and the
    primary-tag rule (one traversal, no re-derivation drift)."""

    error_present: bool
    compiler_diagnostic: bool
    test_run_present: bool
    edit_present: bool
    edit_after_error: bool
    first_tool_call: NormalizedEvent | None


def episode_facts(events: list[NormalizedEvent]) -> EpisodeFacts:
    error_present = compiler_diagnostic = test_run = edit = edit_after_error = False
    error_seq: int | None = None
    first_call: NormalizedEvent | None = None
    for ev in events:
        if ev.type == "tool_call" and ev.tool_name:
            if first_call is None:
                first_call = ev
            if _is_tool_in(ev.tool_name, _CAPABILITY_TOOL_MAP["implementation"]):
                edit = True
                if error_seq is not None and ev.seq > error_seq:
                    edit_after_error = True
            if _is_verification(ev):
                test_run = True
        elif ev.type == "tool_result" and ev.result_is_error:
            error_present = True
            if error_seq is None:
                error_seq = ev.seq
            if _COMPILER_DIAGNOSTIC_RE.search(ev.text or ""):
                compiler_diagnostic = True
    return EpisodeFacts(
        error_present=error_present,
        compiler_diagnostic=compiler_diagnostic,
        test_run_present=test_run,
        edit_present=edit,
        edit_after_error=edit_after_error,
        first_tool_call=first_call,
    )


def _is_tool_in(tool_name: str, names: set[str]) -> bool:
    name = tool_name.lower()
    return name in names or any(n in name for n in names if len(n) > 3)


def tool_capabilities(tool_name: str | None, verification_hint: bool | None = None) -> set[str]:
    """Capability tags one tool call evidences via the `_CAPABILITY_TOOL_MAP`
    path, plus the doc's `tool_use` boundary for MCP-prefixed non-shell tools
    (the map's substring rule skips the short literal "mcp")."""
    caps: set[str] = set()
    name = (tool_name or "").lower()
    for cap, names in _CAPABILITY_TOOL_MAP.items():
        if name in names or any(n in name for n in names if len(n) > 3):
            caps.add(cap)
    if "mcp" in name:
        caps.add("tool_use")
    if verification_hint:
        caps.add("verification")
    return caps


_COMPILER_DIAGNOSTIC_RE = re.compile(
    r"error\[E\d+\]"                       # rustc
    r"|error:\s*TS\d+"                     # tsc phrasing
    r"|\bTS\d{4}\b"                        # tsc diagnostic codes
    r"|^[^\n]*:\d+:\s*(?:fatal\s+)?error:"  # gcc/clang file:line: error:
    r"|^error:"                            # cargo top-level error
    r"|cannot find (?:symbol|type|value)"
    r"|undefined (?:identifier|symbol|variable)"
    r"|mismatched types"
    r"|(?:SyntaxError|IndentationError):"
    r"|\bLNK\d{4}\b",                      # MSVC linker
    re.IGNORECASE | re.MULTILINE,
)


def refine_capabilities(base: list[str], events: list[NormalizedEvent],
                        facts: EpisodeFacts) -> list[str]:
    """Extend segmentation's `_CAPABILITY_TOOL_MAP` output exactly where
    docs/datasets.md's boundary cases demand it. Idempotent."""
    caps = set(base)
    for ev in events:
        if ev.type == "tool_call" and ev.tool_name:
            caps |= tool_capabilities(ev.tool_name, ev.verification_hint)
            if _is_verification(ev):
                caps.add("verification")
    # debugging vs test_interpretation vs verification boundary: reading a
    # failed test run is test_interpretation (and debugging when fixed);
    # a confirm run is verification, neither of the two.
    if facts.test_run_present and facts.error_present:
        caps.add("test_interpretation")
    else:
        caps.discard("test_interpretation")
    # compiler_interpretation vs debugging: a diagnostic message that names
    # the cause replaces the hypothesis; an unnamed cause fixed later is debugging.
    if facts.compiler_diagnostic:
        caps.add("compiler_interpretation")
        caps.discard("debugging")
    elif facts.edit_after_error:
        caps.add("debugging")
    # recovery is attached only when an error is followed by a correction
    # within the 12-event window (secondary by default; primary eligibility
    # is enforced in choose_primary).
    if _episode_recovered(events):
        caps.add("recovery")
    # long_context rides along when the decision spans many events.
    if len(events) >= _LONG_CONTEXT_SPAN and _has_decision(events):
        caps.add("long_context")
    return sorted(caps)


def choose_primary(capabilities: list[str], facts: EpisodeFacts) -> str | None:
    """Primary-tag rule from docs/datasets.md. Returns None when no eligible
    tag exists (capabilities empty, or only the never-primary long_context)."""
    candidates = [c for c in capabilities if c != "long_context"]
    if not candidates:
        return None
    # 1. The diagnostic resolved the cause.
    if facts.compiler_diagnostic and "compiler_interpretation" in candidates:
        return "compiler_interpretation"
    # 2. The action's decision was a fix hypothesis (failing test -> fix keeps
    #    test_interpretation secondary; secondary tags never win here).
    if facts.edit_after_error and "debugging" in candidates:
        return "debugging"
    # 3. An action with no error context: rule 1 (decision that produced it).
    if facts.edit_present and "implementation" in candidates:
        return "implementation"
    # 4. No action; a failed test/build is being read to infer behavior.
    if facts.error_present and facts.test_run_present and "test_interpretation" in candidates:
        return "test_interpretation"
    # 5. Restoring broken state is the entire point: only vehicle tags besides
    #    recovery remain and there is no action.
    others = [c for c in candidates if c != "recovery"]
    if ("recovery" in candidates and not facts.edit_present
            and all(c in _VEHICLE_CAPS for c in others)):
        return "recovery"
    # 6. Rule 2: no action -> capability of the first tool call; vehicle tags
    #    (shell/tool_use) yield to purpose tags.
    if facts.first_tool_call is not None and facts.first_tool_call.tool_name:
        call_caps = tool_capabilities(
            facts.first_tool_call.tool_name,
            facts.first_tool_call.verification_hint,
        )
        pool = [c for c in candidates if c in call_caps]
        purpose = [c for c in pool if c not in _VEHICLE_CAPS]
        if purpose:
            return min(purpose)
        if pool:
            return min(pool)
    # 7. Rule 3: lexicographically smallest eligible tag.
    return min(candidates)


def detect_languages(events: list[NormalizedEvent]) -> list[str] | None:
    """Subject-language tags (9-set) from file extensions and unambiguous
    syntax markers in retained text. Returns None when no code is evidenced
    (omission rule); bare language-name mentions never count."""
    named: set[str] = set()
    c_family = False
    foreign = False
    for ev in events:
        text = ev.text or ""
        if not text:
            continue
        cleaned = _GENERATED_SUFFIX_RE.sub("", text)
        for match in _EXTENSION_RE.finditer(cleaned):
            ext = "." + match.group(1).lower()
            if ext in _LANGUAGE_EXTENSIONS:
                named.add(_LANGUAGE_EXTENSIONS[ext])
            elif ext in _C_FAMILY_EXTENSIONS:
                c_family = True
            elif ext in _OTHER_CODE_EXTENSIONS:
                foreign = True
        for language, marker in _SYNTAX_MARKERS:
            if marker.search(cleaned):
                if language is None:
                    c_family = True
                else:
                    named.add(language)
    if "cpp" in named:
        c_family = False  # .h next to .hpp reads as the same C++ subject
    if foreign or c_family:
        named.add("other")
    ordered = [lang for lang in LANGUAGES if lang in named]
    return ordered or None


def assign_difficulty(ep: Episode) -> str:
    """Structural D0..D4 label; see the module docstring for the rule table.
    Always emitted: every input feature is observable structure."""
    span = len(ep.event_seq)
    long_context = "long_context" in (ep.capabilities or ())
    if long_context and ep.recovered and ep.verification_events >= 1:
        return "D4"
    if ep.recovered or long_context:
        return "D3"
    if ep.has_decision and (span >= _DIFFICULTY_SPAN or ep.tool_calls >= _DIFFICULTY_TOOL_CALLS):
        return "D2"
    if ep.has_decision:
        return "D1"
    return "D0"


def downweight_classes(events: list[NormalizedEvent],
                       downweighted_events: int) -> list[str] | None:
    """Attribute each event session.is_spam flagged to one observed class.

    Returns None (field omitted) when nothing was downweighted or when any
    flagged event has no class evidence; when a list is returned its length
    equals `downweighted_events` by construction.
    """
    if downweighted_events <= 0:
        return None
    flagged = [ev for ev in events if _is_spam(ev)]
    if len(flagged) != downweighted_events:
        return None  # defensive: segmentation counted the same predicate
    call_by_id = {
        ev.tool_call_id: ev for ev in events
        if ev.type == "tool_call" and ev.tool_call_id
    }
    result_by_id = {
        ev.tool_call_id: ev for ev in events
        if ev.type == "tool_result" and ev.tool_call_id
    }

    def family(ev: NormalizedEvent) -> str | None:
        name = ev.tool_name
        if not name and ev.tool_call_id:
            call = call_by_id.get(ev.tool_call_id)
            name = call.tool_name if call else None
        if not name:
            return None
        lowered = name.lower()
        if any(n in lowered for n in _CAPABILITY_TOOL_MAP["repo_exploration"]):
            return "exploration"
        return "run"

    run_flagged = sum(1 for ev in flagged if family(ev) == "run")

    def minified(text: str, chars: int | None) -> bool:
        if (chars or 0) <= _OVERSIZED_RESULT_CHARS:
            return False
        longest = max((len(line) for line in text.splitlines()), default=0)
        return longest >= _MINIFIED_LINE_CHARS

    def attribute(ev: NormalizedEvent) -> str | None:
        fam = family(ev)
        text = ev.text or ""
        if ev.type == "tool_result":
            if _LOCKFILE_RE.search(text):
                return "lockfile"
            if ev.raw_text_sha256 and ev.raw_text_sha256 in seen_text:
                return "duplicate_file_content"
            if fam == "exploration" and _is_search_tool(ev, call_by_id) and (
                    (ev.raw_text_chars or 0) > _GREP_BOUND_CHARS):
                return "enormous_grep_output"
            if _GENERATED_SUFFIX_MARKERS.search(text) or minified(text, ev.raw_text_chars):
                return "generated_bundle"
            if fam == "run" and run_flagged >= 2:
                return "repeated_build_spam"
            return None
        # tool_call
        if fam == "exploration":
            paired = result_by_id.get(ev.tool_call_id or "")
            if paired is not None and _LOCKFILE_RE.search(paired.text or ""):
                return "lockfile"
            if ev.arguments_sha256 and ev.arguments_sha256 in seen_args:
                return "duplicate_file_content"
            if paired is not None and (
                    (paired.raw_text_chars or 0) > _GREP_BOUND_CHARS
                    and _is_search_tool(paired, call_by_id)):
                return "enormous_grep_output"
            if paired is not None and (
                    _GENERATED_SUFFIX_MARKERS.search(paired.text or "")
                    or minified(paired.text or "", paired.raw_text_chars)):
                return "generated_bundle"
            return None
        if fam == "run" and run_flagged >= 2:
            return "repeated_build_spam"
        return None

    classes: list[str] = []
    seen_text: set[str] = set()
    seen_args: set[str] = set()
    for ev in events:
        if _is_spam(ev):
            cls = attribute(ev)
            if cls is None:
                return None  # unattributable: omit rather than guess
            classes.append(cls)
        if ev.raw_text_sha256:
            seen_text.add(ev.raw_text_sha256)
        if ev.type == "tool_call" and ev.arguments_sha256:
            seen_args.add(ev.arguments_sha256)
    return classes


def _is_search_tool(ev: NormalizedEvent, call_by_id: dict) -> bool:
    name = ev.tool_name
    if not name and ev.tool_call_id:
        call = call_by_id.get(ev.tool_call_id)
        name = call.tool_name if call else None
    lowered = (name or "").lower()
    return any(n in lowered for n in _SEARCH_FAMILY)


def _has_decision(events: list[NormalizedEvent]) -> bool:
    return any(
        e.type in ("assistant", "reasoning") and (e.text or "").strip()
        for e in events
    )


def label_episodes(trace: SessionTrace, episodes: list[Episode]) -> list[Episode]:
    """Label every episode in place (and return them). Deterministic and
    idempotent: same trace -> same labels, byte for byte."""
    by_seq = {ev.seq: ev for ev in trace.events}
    for ep in episodes:
        events = [by_seq[s] for s in ep.event_seq if s in by_seq]
        facts = episode_facts(events)
        ep.capabilities = refine_capabilities(ep.capabilities, events, facts)
        ep.primary_capability = choose_primary(ep.capabilities, facts)
        ep.languages = detect_languages(events)
        ep.difficulty = assign_difficulty(ep)
        ep.downweight_classes = downweight_classes(events, ep.downweighted_events)
    return episodes
