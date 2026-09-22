#!/usr/bin/env python3
"""Round-2 evidence mining for the reserved golden bank (hypotheses H1-H5).

Exhausts secondary evidence sources for reserved groups that round 1 left
incomplete, strictly through the documented freeze CLI contracts
(--oracle-index for test oracles, --evidence-index for repo/revision fills).
golden.py, partition and identity code are never touched.

Hypotheses under test (each counted honestly: candidates found / registered /
refused with reasons):
  H1  cross-session: later/other sessions of the same task group (linked by
      explicit issue ref across cwd drift, same cwd with prompt drift, or
      repo+revision identity) contain verification runs with pass markers.
  H1b same linkage, sessions assigned to pruning/quant/validation/golden
      partitions (H4 rows below get recovery/torture ones).
  H2  command-output evidence: tool outputs in the group's OWN sessions with
      explicit zero-exit evidence + pass markers whose commands round 1's
      verification classifier did not recognize (widened to all tool outputs;
      inspection commands are refused, never registered).
  H3  PR/issue/CI evidence: tool outputs carrying CI outcome text for the
      same task (chat mentions are refused: chat never makes an oracle).
  H4  recovery/torture cross-links: linked sessions whose own group sits in
      the recovery/torture partition and that contain the verification
      moment (attributed separately from H1).
  H5  git-history: read-only git HEAD records (log/show/rev-parse/status,
      incl. `git -C` forms round 1 missed) in the group's own and linked
      sessions, plus cross-cwd linked session_meta, supply
      starting_revision where the harness recorded none.

Fail-closed everywhere: a linkage with a recorded-revision mismatch, a
conflicting value across candidates, an inspection-shaped command, an
errored/absent result, fail markers, or chat-only evidence REFUSES the
registration and is counted with its reason. Nothing is fabricated.

Writes (private, under $MIMO_LAB/traces/oracles/, never traces/golden/):
  oracle-registry.json     updated in place (round-1 fields preserved,
                           round-2 registrations added with pointers)
  oracle-index.json        union of round-1 + round-2 oracles (flat)
  evidence-index.json      union of round-1 + round-2 field fills (flat,
                           every field carries a pointer)
  round2-yield-report.json per-source yield table, refusals, new blockers
Every write is re-read and re-parsed (write-corruption guard).
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIMO_RAW = os.environ.get("MIMO_LAB")
if not MIMO_RAW:
    raise SystemExit("MIMO_LAB environment variable is required")
MIMO = Path(MIMO_RAW).expanduser()
OUT_DIR = MIMO / "traces" / "oracles"
sys.path.insert(0, str(REPO / "src"))

from mimo_halo.traces.cli import CONFIG_PATH, load_config  # noqa: E402
from mimo_halo.traces.common import (  # noqa: E402
    atomic_write_json, ensure_dir, sha256_path, sha256_text,
)
from mimo_halo.traces.discovery import expand_root, list_transcripts  # noqa: E402
from mimo_halo.traces.events import SessionTrace  # noqa: E402
from mimo_halo.traces.identity import (  # noqa: E402
    DEFAULT_ISSUE_PATTERN, _dedup_sessions, build_task_groups, find_issue_refs,
    prompt_hash_key,
)
from mimo_halo.traces.partition import assign_partition  # noqa: E402
from mimo_halo.traces.redact import redact  # noqa: E402
from mimo_halo.traces.session import normalize_session  # noqa: E402

# ------------------------------------------------- round-1 contract patterns
# (copied verbatim from the round-1 builder so both rounds classify identically)

EXTRA_TEST_RE = re.compile(
    r"(?:\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b|\bnode\s+--test\b"
    r"|\bdeno\s+(?:task\s+)?test\b|\bpython3?\s+-m\s+(?:pytest|unittest)\b|\bpytest\b"
    r"|\bvitest\b|\bjest\b|\bcargo\s+test\b|\bgo\s+test\b"
    r"|\bmake\s+(?:test|check)\b|\btox\b|\bphpunit\b|\brspec\b|\bunittest\b"
    r"|\bgradle\s+(?:test|check)\b|\bmvn\s+(?:test|verify)\b|\btsc\b"
    r"|\bmypy\b|\bruff\s+(?:check|format)\b|\beslint\b|\bpylint\b"
    r"|\btypecheck\b|\bsvelte-check\b"
    r"|\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:lint|check)\b"
    r"|\bterraform\s+(?:test|validate|fmt\s+-check)"
    r"|\btofu\s+(?:test|validate)\b"
    r"|\b(?:black|isort)\s+--check\b|\bprettier\s+--check\b"
    r"|\bpyright\b"
    r"|\bgradlew?\s+(?:test|check)\b"
    r"|\b(?:just|task)\s+(?:test|check|ci|lint)\b"
    r"|\b(?:dotnet|swift|sbt|mix|lein)\s+test\b|\bzig\s+build\s+test\b"
    r"|\bcypress\s+run\b|\bplaywright\s+test\b|\bmocha\b|\bkarma\s+start\b)",
    re.IGNORECASE,
)

SELF_RUNNERS = ("pytest", "tox", "mypy", "ruff", "eslint", "pylint", "tsc",
                "vitest", "jest", "phpunit", "rspec", "unittest",
                "svelte-check", "pyright", "prettier", "black", "isort",
                "cypress", "playwright", "mocha", "karma", "tap", "ava")
WORD_RUNNERS = ("npx", "pnpm", "npm", "yarn", "bun", "node", "deno",
                "python3", "python", "py", "cargo", "go", "make", "gradle",
                "gradlew", "./gradlew", "mvn", "php", "bash", "sh", "zsh",
                "terraform", "tofu", "uv", "poetry", "pipenv", "pdm", "hatch",
                "bundle", "just", "task", "sbt", "dotnet", "swift", "zig",
                "mix", "lein", "xargs", "nix")
SEARCH_PREFIXES = frozenset({
    "grep", "rg", "find", "fd", "ag", "ls", "cat", "head", "tail", "awk",
    "sed", "which", "whereis", "man", "jq", "strings", "nm", "objdump",
    "file", "stat", "wc", "xxd", "dig", "sort", "uniq", "cut", "tr", "diff",
    "echo", "printf", "pwd", "date", "sleep", "wait", "true", "false", "cd",
    "export", "set", "source", ".", "env", "git", "hg", "svn", "less",
    "column", "basename", "dirname", "realpath", "readlink", "tree", "nl",
    "more", "bat", "open", "curl", "wget", "ssh", "scp", "rsync", "cp",
    "mv", "rm", "mkdir", "touch", "chmod", "chown", "ln", "tar", "tee",
    "kill", "ps", "df", "du", "free", "uname", "whoami", "id", "top",
    "crontab", "screen", "tmux", "nohup", "docker", "kubectl", "helm",
    "brew", "apt", "apt-get", "yum", "dnf", "pacman", "mount", "df",
})
WRAPPER_PREFIXES = ("sudo", "env", "command", "time", "nohup", "nice")
TESTISH_RE = re.compile(
    r"\b(?:test|tests|spec|specs|lint|check|typecheck|validate)(?:s|ing)?\b"
    r"|\btype-?check\b", re.IGNORECASE)

PASS_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("count_summary_passed", re.compile(r"\b[1-9]\d*\s+passed\b", re.M)),
    ("count_summary_passing", re.compile(r"\b[1-9]\d*\s+passing\b", re.M)),
    ("go_ok", re.compile(r"^ok\s+\S+", re.M)),
    ("go_pass_line", re.compile(r"^PASS$", re.M)),
    ("cargo_test_result_ok", re.compile(r"test result:\s*ok", re.M)),
    ("node_test_pass", re.compile(r"#\s*pass\s+[1-9]", re.M)),
    ("ruff_all_checks_passed", re.compile(r"All checks passed", re.M)),
    ("mypy_success", re.compile(r"Success: no issues found", re.M)),
    ("eslint_zero_problems", re.compile(r"\b0 problems\b", re.M)),
    ("unittest_ok", re.compile(r"^OK$", re.M)),
    ("all_tests_passed", re.compile(r"(?i)all tests passed")),
    ("check_mark", re.compile(r"[✔✓]")),
    ("build_success", re.compile(r"BUILD SUCCESS(?:FUL)?", re.M)),
    ("bun_pass_count", re.compile(r"\b[1-9]\d*\s+pass\b", re.M)),
    ("terraform_run_pass", re.compile(r"run\s+\"[^\"]*\"\.\.\.\s*pass")),
    ("terraform_valid", re.compile(r"The configuration is valid")),
    ("exit_code_zero", re.compile(r"Process exited with code 0", re.M)),
]

FAIL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("zero_passed", re.compile(r"\b0\s+passed\b", re.M)),
    ("bare_fail_count", re.compile(r"\b[1-9]\d*\s+fail\b", re.M)),
    ("build_failed", re.compile(r"BUILD FAILED|FAILURES!!!", re.M)),
    ("nonzero_failed", re.compile(r"\b[1-9]\d*\s+(?:failed|failing|errors?)\b", re.M)),
    ("failed_upper", re.compile(r"\bFAILED\b", re.M)),
    ("go_fail_line", re.compile(r"^\s*FAIL\b", re.M)),
    ("found_errors", re.compile(r"Found\s+[1-9]\d*\s+errors?", re.M)),
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("tsc_error_ts", re.compile(r"\berror TS\d+")),
    ("rust_panic", re.compile(r"\bpanicked at\b")),
    ("would_reformat", re.compile(r"Would reformat")),
    ("npm_err", re.compile(r"npm ERR!")),
    ("terraform_run_fail", re.compile(r"run\s+\"[^\"]*\"\.\.\.\s*fail")),
    ("block_error", re.compile(r"│\s*Error:")),
    ("make_failed", re.compile(r"\*\*\* \[.*\] (?:Error|Stop)")),
    ("nonzero_exit_text", re.compile(
        r"(?:Command exited with code|exited with (?:code|status)"
        r"|exit code)[:\s]+[1-9]\d*", re.M)),
    ("cross_mark", re.compile(r"[✖✗]")),
]

# H2: explicit zero-exit evidence in the output itself (codex/omp wrappers).
EXIT0_RE = re.compile(
    r"(?:Process exited with code|Command exited with code"
    r"|exited with (?:code|status)|exit(?:ed)? code)[:\s]+0\b", re.M)

# H3: CI outcome text (pass and fail families).
CI_PASS_RE = re.compile(
    r"(?i)(?:all checks (?:have )?passed|checks? (?:have )?passed"
    r"|conclusion:\s*success|\"conclusion\"\s*:\s*\"SUCCESS\""
    r"|status:\s*(?:success|completed)|build (?:passed|succeeded)"
    r"|all jobs? (?:have )?passed|workflow run .{0,60}completed: success"
    r"|✓\s*checks|tests?\s+passed\s+in\s+ci)")
CI_FAIL_RE = re.compile(
    r"(?i)(?:checks? failed|conclusion:\s*failure"
    r"|\"conclusion\"\s*:\s*\"FAILURE\"|status:\s*failure"
    r"|build failed|all jobs? failed|workflow run .{0,60}completed: failure"
    r"|×\s*checks)")
CI_CMD_RE = re.compile(
    r"(?:\b(?:gh|hub)\b[^\n]*(?:\brun\b|\bpr\b|\bchecks?\b)|circleci"
    r"|buildkite|woodpecker|drone\b|\bci\b.*\brun\b)", re.IGNORECASE)

GIT_STATE_OP_RE = re.compile(
    r"\bgit\b[^\n|;&]*\b(?:commit|merge|rebase|reset|checkout|switch|pull"
    r"|cherry-pick|revert|stash|am)\b", re.IGNORECASE)
GIT_SUB_RE = re.compile(
    r"^(?:(?:--no-pager|-C\s+\S+|-c\s+\S+|--\S+)\s+)*"
    r"(?:rev-parse|log|remote|config|fetch|pull|push|status|diff|show)\b",
    re.IGNORECASE)
SHELL_TOOLS = frozenset({"bash", "sh", "zsh", "shell", "terminal", "exec",
                         "run_command", "command", "exec_command"})
GIT_WRAPPER_TOOLS = frozenset({"hub", "git", "git_tool", "git_cli"})
# round-1 matcher + round-2 widening: global flags (`git -C <path> …`) allowed
REV_PARSE_RE = re.compile(
    r"\bgit\b(?:\s+(?:-[A-Za-z]+\s+\S+|--\S+(?:=\S+)?))*\s+rev-parse"
    r"\b[^\n|;&]*\bHEAD\b(?![\^~:{])"
    r"|\bgit\b(?:\s+(?:-[A-Za-z]+\s+\S+|--\S+(?:=\S+)?))*\s+rev-parse"
    r"\s+--verify\b[^\n|;&]*\bHEAD\b(?![\^~:{])",
    re.IGNORECASE)
GIT_LOG_RE = re.compile(r"\bgit\b(?:\s+(?:-[A-Za-z]+\s+\S+|--\S+(?:=\S+)?))*\s+log\b",
                        re.IGNORECASE)
# round-2: `git show` with no positional rev = HEAD tip read
GIT_SHOW_RE = re.compile(r"\bgit\b(?:\s+(?:-[A-Za-z]+\s+\S+|--\S+(?:=\S+)?))*\s+show\b",
                         re.IGNORECASE)
GIT_LOG_REV_RANGE_RE = re.compile(r"\.\.|@\{|HEAD[\^~]", re.IGNORECASE)

URL_RE = re.compile(r"(?:https?|git|ssh)://\S+|git@\S+:\S+")
FULL_HASH_RE = re.compile(r"\b([0-9a-f]{40})\b")
COMMIT_LINE_HASH_RE = re.compile(r"^\s*commit\s+([0-9a-f]{40})", re.M)
STATUS_OID_RE = re.compile(r"^#\s*branch\.oid\s+([0-9a-f]{7,40})\s*$", re.M)


# ------------------------------------------------------------- small helpers

def repo_norm(url: str) -> str:
    u = url.strip().rstrip("/")
    u = re.sub(r"^(?:https?://|git://|ssh://[^/]+/)", "", u)
    u = re.sub(r"^git@[^:]+:", "", u)
    u = re.sub(r"^([\w.-]+\.[A-Za-z]{2,}):", r"\1/", u)
    if u.endswith(".git"):
        u = u[:-4]
    return u.lower()


def command_of(raw_args) -> str | None:
    if isinstance(raw_args, str):
        return raw_args.strip() or None
    if isinstance(raw_args, dict):
        for key in ("command", "cmd", "script", "input"):
            value = raw_args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                return " ; ".join(str(v) for v in value)
        if isinstance(raw_args.get("args"), list) and raw_args["args"]:
            return " ".join(str(a) for a in raw_args["args"])
    if isinstance(raw_args, list) and raw_args:
        return " ; ".join(str(a) for a in raw_args)
    return None


def parse_json(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def raw_artifacts(path: Path, source: str):
    """(commands, results, tool_names) keyed by tool_call_id, read at origin."""
    cmds: dict[str, str] = {}
    results: dict[str, str] = {}
    tool_names: dict[str, str] = {}

    def put_result(cid, out):
        if cid is None:
            return
        if isinstance(out, list):
            out = "\n".join(b.get("text", "") for b in out
                            if isinstance(b, dict))
        elif not isinstance(out, str):
            out = str(out)
        results[str(cid)] = out

    def put_call(cid, cmd, name):
        if not cid:
            return
        if cmd:
            cmds[str(cid)] = cmd
        if name:
            tool_names[str(cid)] = str(name)

    for d in parse_json(path):
        if source == "claude":
            rtype = d.get("type")
            msg = d.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            if rtype == "assistant":
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        put_call(blk.get("id"), command_of(blk.get("input")),
                                 blk.get("name"))
            elif rtype == "user":
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        put_result(blk.get("tool_use_id"), blk.get("content"))
        elif source == "codex":
            payload = d.get("payload")
            if not isinstance(payload, dict):
                continue
            ptype = payload.get("type")
            key = {"function_call": "arguments", "custom_tool_call": "input",
                   "local_shell_call": "action"}.get(ptype)
            if key:
                raw = payload.get(key)
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        pass
                put_call(payload.get("call_id") or payload.get("id"),
                         command_of(raw), payload.get("name"))
            elif ptype in ("function_call_output", "custom_tool_call_output",
                           "local_shell_call_output"):
                put_result(payload.get("call_id"), payload.get("output"))
        else:  # pi / omp
            if d.get("type") != "message":
                continue
            msg = d.get("message") or {}
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            if role == "assistant":
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "toolCall":
                        put_call(blk.get("id"), command_of(blk.get("arguments")),
                                 blk.get("name"))
            elif role == "toolResult":
                put_result(msg.get("toolCallId"), content)
    return cmds, results, tool_names


def _ts_key(value):
    """Comparable timestamp key across harness families (ISO strings vs
    epoch numbers): never mixes str and int in one comparison slot."""
    if not value:
        return (0, 0.0, "")
    if isinstance(value, (int, float)):
        return (1, float(value), "")
    return (2, 0.0, str(value))


def end_key(t: SessionTrace):
    return (1 if (t.ended_at or t.started_at) else 0,
            *_ts_key(t.ended_at or t.started_at),
            t.source_path_sha256)


def start_key(t: SessionTrace):
    return (1 if (t.started_at or t.ended_at) else 0,
            *_ts_key(t.started_at or t.ended_at),
            t.source_path_sha256)


def first_user_event_seq(t: SessionTrace) -> int | None:
    for ev in t.events:
        if ev.type == "user":
            return ev.seq
    return None


def eval_pass(text: str | None, is_error: bool | None, cmd: str | None = None):
    if is_error is True:
        return False, "result_is_error"
    if not text or not text.strip():
        if is_error is False and cmd is not None and EXTRA_TEST_RE.search(cmd):
            return True, "no_error_no_output"
        return False, "empty_output_unknown_status"
    for label, pat in FAIL_PATTERNS:
        if pat.search(text):
            return False, f"fail_marker:{label}"
    for label, pat in PASS_PATTERNS:
        if pat.search(text):
            return True, label
    return False, "no_pass_marker"


def kind_of(cmd: str) -> str:
    c = cmd.lower()
    if re.search(r"\btsc\b|\bmypy\b", c):
        return "typecheck"
    if re.search(r"ruff|eslint|pylint|flake8|lint|black --check|isort --check", c):
        return "lint"
    if re.search(r"pytest|vitest|jest|cargo test|go test|node --test"
                 r"|(?:npm|pnpm|yarn|bun) (?:run )?test|make test|make check"
                 r"|tox|phpunit|rspec|unittest|deno test|-m pytest"
                 r"|(?:terraform|tofu) test", c):
        return "test"
    return "verification"


def _strip_wrappers(segment: str) -> str:
    core = segment.strip()
    while True:
        m = re.match(r"^([A-Za-z_][\w.]*)=\S+\s+", core)
        first = core.split()[0].rsplit("/", 1)[-1] if core.split() else ""
        if m:
            core = core[m.end():]
        elif first in WRAPPER_PREFIXES:
            core = core.split(None, 1)[1] if " " in core else ""
        else:
            return core


def looks_like_test_run(cmd: str) -> bool:
    """Round-1 verification classifier (verbatim): True only for RUN segments."""
    for segment in re.split(r"\s*(?:&&|\|\||;|\|)\s*", cmd):
        core = _strip_wrappers(segment)
        if not core:
            continue
        tokens = core.split()
        first = tokens[0].rsplit("/", 1)[-1].lower()
        if first in SEARCH_PREFIXES:
            continue
        low = core.lower()
        if first in SELF_RUNNERS:
            return True
        if any(low.startswith(r + " ") or low.startswith(r + "-")
               for r in SELF_RUNNERS):
            return True
        runner_word = any(
            re.search(rf"\b{re.escape(r)}\b", low) for r in SELF_RUNNERS)
        if first in WORD_RUNNERS or any(
                low.startswith(w + " ") for w in WORD_RUNNERS):
            if runner_word or TESTISH_RE.search(low):
                return True
        if re.match(r"^\./[\w./-]*tests?[\w./-]*", low) and \
                TESTISH_RE.search(low):
            return True
    return False


def round1_classified(ev, cmd: str) -> bool:
    return bool(ev.verification_hint is True or EXTRA_TEST_RE.search(cmd)
                or looks_like_test_run(cmd))


def has_runner_segment(cmd: str) -> bool:
    """H2: some pipeline/&& segment is neither search nor inspection."""
    for segment in re.split(r"\s*(?:&&|\|\||;|\|)\s*", cmd):
        core = _strip_wrappers(segment)
        if not core:
            continue
        first = core.split()[0].rsplit("/", 1)[-1].lower()
        if first not in SEARCH_PREFIXES:
            return True
    return False


def ptr(session: SessionTrace, event_index: int, source: str) -> dict:
    return {"session_id_sha256": session.source_path_sha256,
            "event_index": event_index, "source": source}


def redacted_bound(text: str, limit: int = 400) -> str:
    cleaned = " ".join(text.split())
    r = redact(cleaned)
    if len(r.text) > limit:
        return f"{r.text[:300]}… (len={len(r.text)})"
    return r.text


def git_log_reads_tip(cmd: str) -> bool:
    tokens = cmd.split()
    try:
        i = next(k for k, t in enumerate(tokens) if t == "log"
                 or t.endswith("/log"))
    except StopIteration:
        return False
    filter_flags = ("--grep", "--author", "--committer", "--since",
                    "--until", "--after", "--before", "--skip")
    value_flags = {"-n", "--max-count", "--format", "--pretty", "--date",
                   "-S", "-G"}
    skip_next = False
    for t in tokens[i + 1:]:
        if skip_next:
            skip_next = False
            continue
        if t.startswith(filter_flags) or t in ("-S", "-G"):
            return False
        if t in value_flags:
            skip_next = t in ("-S", "-G") or t in value_flags
            continue
        if t.startswith("-"):
            continue
        if t == "HEAD":
            continue
        return False
    return True


def show_reads_tip(cmd: str) -> bool:
    """`git show` prints HEAD only when no positional rev/ref is given."""
    tokens = cmd.split()
    try:
        i = next(k for k, t in enumerate(tokens) if t == "show"
                 or t.endswith("/show"))
    except StopIteration:
        return False
    value_flags = {"--format", "--pretty", "--abbrev", "--date", "-U",
                   "--diff-filter", "-m", "-M", "-C", "--find-renames"}
    skip_next = False
    for t in tokens[i + 1:]:
        if skip_next:
            skip_next = False
            continue
        if t in value_flags:
            skip_next = True
            continue
        if t.startswith("-"):
            continue
        return False  # positional rev/branch/ref: tip not guaranteed
    return True


def head_hash_from(cmd: str, text: str):
    """HEAD hash from a read-only git probe's recorded output (round-1 rules
    + round-2 widening: global-flag forms and `git show` tip reads)."""
    if not text:
        return None
    if text.lstrip().startswith("fatal:") or "\nfatal:" in text[:200]:
        return None
    if REV_PARSE_RE.search(cmd):
        m = re.search(r"^\s*([0-9a-f]{7,40})\s*$", text, re.M)
        return m.group(1) if m else None
    if GIT_LOG_RE.search(cmd) and not GIT_LOG_REV_RANGE_RE.search(cmd) \
            and git_log_reads_tip(cmd):
        m = COMMIT_LINE_HASH_RE.search(text)
        if m:
            return m.group(1)
        m2 = re.search(r"^\s*([0-9a-f]{7,40})(?:\s|$)", text, re.M)
        return m2.group(1) if m2 else None
    if GIT_SHOW_RE.search(cmd) and not GIT_LOG_REV_RANGE_RE.search(cmd) \
            and show_reads_tip(cmd):
        m = COMMIT_LINE_HASH_RE.search(text)
        if m:
            return m.group(1)
        m2 = re.search(r"^\s*([0-9a-f]{7,40})\s*$", text, re.M)
        return m2.group(1) if m2 else None
    return None


def status_head_hash(cmd: str, text: str):
    if re.search(r"\bgit\b(?:\s+(?:-\S+))*\s+status\b", cmd) \
            and "--porcelain=v2" in cmd:
        m = STATUS_OID_RE.search(text)
        return m.group(1) if m else None
    return None


# ---------------------------------------------------------------- load phase

print("loading config…", flush=True)
config = load_config(Path(str(CONFIG_PATH)))
salt = config["partition"]["salt"]

traces: list[SessionTrace] = []
path_by_ref: dict[str, Path] = {}
for source, cfg in config["traces"]["sources"].items():
    root = expand_root(cfg["root"])
    transcripts, _nont = list_transcripts(
        source, root, cfg.get("transcript_glob", "**/*.jsonl"))
    for p in transcripts:
        path_by_ref[sha256_path(p)] = p
        try:
            traces.append(normalize_session(p, source))
        except Exception:
            continue
print(f"parsed {len(traces)} sessions", flush=True)

groups = build_task_groups(traces, DEFAULT_ISSUE_PATTERN)["groups"]
reserved = [g for g in groups if assign_partition(salt, g["group_id"]) == "golden"]
print(f"groups={len(groups)} reserved={len(reserved)}", flush=True)

# session -> its group + partition (for H4 attribution)
sess_group: dict[str, dict] = {}
for g in groups:
    for ref in g["session_refs"]:
        sess_group[ref] = g
part_of: dict[str, str] = {g["group_id"]: assign_partition(salt, g["group_id"])
                           for g in groups}

raw_cache: dict[str, dict[str, str | None]] = {}


def raw_for(t: SessionTrace):
    if t.source_path_sha256 not in raw_cache:
        p = path_by_ref.get(t.source_path_sha256)
        if p:
            cmds, results, names = raw_artifacts(p, t.source)
        else:
            cmds, results, names = {}, {}, {}
        raw_cache[t.source_path_sha256] = (cmds, results, names)
    return raw_cache[t.source_path_sha256]


def member_sessions(g: dict):
    """Round-1 membership + the group's own session_refs (union)."""
    repo_key = g["repo_key"]
    if g["identity"] == "explicit_issue_ref":
        ref = g["issue_ref"]
        members = [t for t in traces
                   if t.cwd_sha256 == repo_key
                   and ref in find_issue_refs(t.first_user_prompt,
                                              DEFAULT_ISSUE_PATTERN)]
    else:
        pdigest = prompt_hash_key(sha256_text(g["task_prompt"] or ""))
        members = [t for t in traces
                   if t.cwd_sha256 == repo_key
                   and not find_issue_refs(t.first_user_prompt,
                                           DEFAULT_ISSUE_PATTERN)
                   and prompt_hash_key(sha256_text(t.first_user_prompt or ""))
                   == pdigest]
    refs = {t.source_path_sha256 for t in members}
    for ref in g["session_refs"]:
        if ref not in refs:
            by_ref = next((t for t in traces
                           if t.source_path_sha256 == ref), None)
            if by_ref is not None:
                members.append(by_ref)
    return _dedup_sessions(members), members


def linked_sessions(g: dict, members_set: set[str],
                    g_repo: str | None, g_rev: str | None,
                    refusals: dict[str, int]):
    """H1/H4/H5 session widening: same task, excluded from round-1 members."""
    g_repo_n = repo_norm(g_repo) if g_repo else None
    out: list[tuple[SessionTrace, str]] = []
    for t in traces:
        if t.source_path_sha256 in members_set:
            continue
        link = None
        if g.get("issue_ref") and g["issue_ref"] in find_issue_refs(
                t.first_user_prompt):
            link = "issue_ref_cross_cwd"
        elif t.cwd_sha256 == g["repo_key"]:
            link = "same_cwd_prompt_drift"
        elif (g_repo_n and g_rev and t.repo_identifier and t.start_revision
              and repo_norm(t.repo_identifier) == g_repo_n
              and t.start_revision == g_rev):
            link = "repo_revision"
        if link is None:
            continue
        # fail-closed: a linked session recording a DIFFERENT revision than
        # the group's resolved starting revision is not provably the same task
        if link != "repo_revision" and g_rev and t.start_revision \
                and t.start_revision != g_rev:
            refusals["linkage_revision_mismatch"] = refusals.get(
                "linkage_revision_mismatch", 0) + 1
            continue
        out.append((t, link))
    return out


# ------------------------------------------------------ verification scanner

def outcome_of(t: SessionTrace, call_ev):
    """Paired result for a call, following async-shell exit records."""
    res = next((e for e in t.events
                if e.type == "tool_result"
                and e.tool_call_id == call_ev.tool_call_id), None)
    if res is None:
        return None, "final_verification_unpaired"
    if res.text and "Process running with session ID" in res.text:
        follow = next((e for e in t.events
                       if e.type == "tool_result" and e.seq > res.seq
                       and e.text
                       and "Process exited with code" in e.text), None)
        if follow is None:
            return None, "async_verification_outcome_missing"
        res = follow
    return res, None


def scan_classified(sessions: list[SessionTrace]) -> tuple[list, list]:
    """Round-1-classified verification runs: (passes, failures)."""
    passes, failures = [], []
    for t in sessions:
        cmds, results, _names = raw_for(t)
        for ev in t.events:
            if ev.type != "tool_call" or not ev.tool_call_id:
                continue
            cmd = cmds.get(ev.tool_call_id)
            if not cmd or not round1_classified(ev, cmd):
                continue
            res, err = outcome_of(t, ev)
            if res is None:
                failures.append((t, ev, cmd, None, err))
                continue
            text = results.get(res.tool_call_id or "") or (res.text or "")
            ok, label = eval_pass(text, res.result_is_error, cmd)
            if ok:
                passes.append((t, ev, cmd, res, label))
            else:
                failures.append((t, ev, cmd, res,
                                 f"final_verification_not_passing:{label}"))
    return passes, failures


def scan_outputs(sessions: list[SessionTrace]) -> tuple[list, list]:
    """H2: unclassified tool outputs with zero-exit evidence + pass markers.

    found  = runner-shaped (non-inspection) command whose recorded output
             carries exit-0 evidence AND a pass marker
    refused = found but fail markers / error status / async outcome missing;
             inspection commands are counted separately (never registered)
    """
    found, refused = [], []
    for t in sessions:
        cmds, results, _names = raw_for(t)
        for ev in t.events:
            if ev.type != "tool_call" or not ev.tool_call_id:
                continue
            cmd = cmds.get(ev.tool_call_id)
            if not cmd:
                continue
            res, err = outcome_of(t, ev)
            if res is None:
                continue  # no outcome text to inspect at all
            text = results.get(res.tool_call_id or "") or (res.text or "")
            if not text:
                continue
            exit0 = bool(EXIT0_RE.search(text)) or res.result_is_error is False
            if not exit0:
                continue
            pass_hit = next(((label, pat) for label, pat in PASS_PATTERNS
                             if pat.search(text)), None)
            if pass_hit is None:
                continue
            if round1_classified(ev, cmd):
                continue  # round 1's contract, already evaluated there
            if not has_runner_segment(cmd):
                refused.append((t, ev, cmd, res, "inspection_command"))
                continue
            if res.result_is_error is True:
                refused.append((t, ev, cmd, res, "result_is_error"))
                continue
            fail = next(((label, pat) for label, pat in FAIL_PATTERNS
                         if pat.search(text)), None)
            if fail is not None:
                refused.append((t, ev, cmd, res,
                                f"fail_marker:{fail[0]}"))
                continue
            found.append((t, ev, cmd, res, pass_hit[0]))
    return found, refused


def scan_ci(sessions: list[SessionTrace]) -> tuple[list, list, list]:
    """H3: CI outcome text in tool outputs; chat mentions counted refused."""
    found, refused = [], []
    chat_keys = []
    for t in sessions:
        cmds, results, _names = raw_for(t)
        for ev in t.events:
            if ev.type != "tool_call" or not ev.tool_call_id:
                continue
            cmd = cmds.get(ev.tool_call_id)
            res, _err = outcome_of(t, ev)
            if res is None:
                continue
            text = results.get(res.tool_call_id or "") or (res.text or "")
            if not text:
                continue
            has_ci = bool(CI_PASS_RE.search(text) or CI_FAIL_RE.search(text))
            if not has_ci:
                continue
            ref_in_text = bool(find_issue_refs(text[:2000]))
            ci_cmd = bool(cmd and CI_CMD_RE.search(cmd))
            if not (ci_cmd or ref_in_text):
                continue
            if cmd and not has_runner_segment(cmd) and not ci_cmd:
                refused.append((t, ev, cmd, res, "inspection_command"))
                continue
            if CI_FAIL_RE.search(text):
                refused.append((t, ev, cmd, res, "ci_outcome_failed"))
                continue
            if res.result_is_error is True:
                refused.append((t, ev, cmd, res, "result_is_error"))
                continue
            if not CI_PASS_RE.search(text):
                continue
            found.append((t, ev, cmd or "", res,
                          "gh_ci" if ci_cmd else "ci_text_in_output"))
        # chat-only CI mentions: never an executable oracle
        for ev in t.events:
            if ev.type in ("user", "assistant", "text") and ev.text \
                    and CI_PASS_RE.search(ev.text):
                chat_keys.append((t.source_path_sha256, ev.seq))
    return found, refused, chat_keys


def make_record(t, ev, cmd, res, label, source_kind, extra=None) -> dict:
    record = {
        "value": (f"{redacted_bound(cmd)} :: pass={label}"
                  f" kind={kind_of(cmd)}"
                  f" ev={t.source_path_sha256[:16]}#{ev.seq}"),
        "command": redacted_bound(cmd),
        "success_condition": (
            "harness result_is_error != true; output matches success "
            f"evidence '{label}'; no failure marker present"),
        "verification_kind": kind_of(cmd),
        "pass_marker": label,
        "evidence": ptr(t, ev.seq, "tool_call:verification"),
        "result_evidence": ptr(t, res.seq, "tool_result"),
        "round2_source": source_kind,
    }
    if extra:
        record.update(extra)
    return record


# ------------------------------------------------------------- H5: revision

def git_head_observations(sessions: list[SessionTrace]):
    """First read-only HEAD record per session, before any git state op
    (round-1 rules + round-2 `git -C` / `git show` widening)."""
    obs = []
    for t in sessions:
        cmds, results, _names = raw_for(t)
        seen_state_op = False
        for ev in sorted(t.events, key=lambda e: e.seq):
            if ev.type == "tool_call" and ev.tool_call_id:
                cmd = cmds.get(ev.tool_call_id)
                if cmd and GIT_STATE_OP_RE.search(cmd):
                    seen_state_op = True
            elif ev.type == "tool_result" and ev.tool_call_id \
                    and ev.text and not seen_state_op:
                cmd = cmds.get(ev.tool_call_id)
                # wrapper-tool reconstruction like round 1
                if cmd and not re.match(r"^\s*git\b", cmd) \
                        and not re.search(r"[;&|]\s*git\s+\S", cmd):
                    name = (_names.get(ev.tool_call_id) or "").lower()
                    if name in GIT_WRAPPER_TOOLS and GIT_SUB_RE.match(cmd):
                        cmd = "git " + cmd
                    elif name in SHELL_TOOLS:
                        cmd = cmd if GIT_SUB_RE.match(cmd) else None
                    else:
                        cmd = None
                if not cmd or GIT_STATE_OP_RE.search(cmd):
                    continue
                text = results.get(ev.tool_call_id or "") or (ev.text or "")
                h = head_hash_from(cmd, text) or status_head_hash(cmd, text)
                if h:
                    obs.append((t, ev.seq, h))
                    break
    return obs


# ------------------------------------------------------------ session caches
# Every scan below is a pure function of one session; caching per session id
# turns the per-target rescans into set unions (semantics unchanged).

_CLASS_CACHE: dict[str, tuple] = {}
_OUT_CACHE: dict[str, tuple] = {}
_CI_CACHE: dict[str, tuple] = {}
_GITHEAD_CACHE: dict[str, list] = {}


def _uniq_sessions(sessions):
    seen, out = set(), []
    for t in sessions:
        if t.source_path_sha256 not in seen:
            seen.add(t.source_path_sha256)
            out.append(t)
    return out


def scan_classified_cached(sessions):
    passes, failures = [], []
    for t in _uniq_sessions(sessions):
        got = _CLASS_CACHE.get(t.source_path_sha256)
        if got is None:
            got = scan_classified([t])
            _CLASS_CACHE[t.source_path_sha256] = got
        passes.extend(got[0])
        failures.extend(got[1])
    return passes, failures


def scan_outputs_cached(sessions):
    found, refused = [], []
    for t in _uniq_sessions(sessions):
        got = _OUT_CACHE.get(t.source_path_sha256)
        if got is None:
            got = scan_outputs([t])
            _OUT_CACHE[t.source_path_sha256] = got
        found.extend(got[0])
        refused.extend(got[1])
    return found, refused


def scan_ci_cached(sessions):
    found, refused, chat_keys = [], [], []
    for t in _uniq_sessions(sessions):
        got = _CI_CACHE.get(t.source_path_sha256)
        if got is None:
            got = scan_ci([t])
            _CI_CACHE[t.source_path_sha256] = got
        found.extend(got[0])
        refused.extend(got[1])
        chat_keys.extend(got[2])
    return found, refused, chat_keys


def git_head_observations_cached(sessions):
    obs = []
    for t in _uniq_sessions(sessions):
        got = _GITHEAD_CACHE.get(t.source_path_sha256)
        if got is None:
            got = git_head_observations([t])
            _GITHEAD_CACHE[t.source_path_sha256] = got
        obs.extend(got)
    return obs


# --------------------------------------------------------------- main loop

registry = json.loads((OUT_DIR / "oracle-registry.json").read_text())

# Self-reset of round-2 provenance (idempotent re-runs): fields a previous
# round-2 pass registered are cleared back to missing before re-mining, so the
# funnel precedence (H1/H4 classified runs > H2 command outputs > H3 CI) is
# applied to a clean base and no stale index entry can survive. Round-1
# fields (no round-2 provenance) are never touched.
ROUND2_REV_TIERS = {"cross_cwd_linked_session_meta", "git_head_output_widened"}
REQUIRED_FIELDS = ("task_id", "repo_identifier", "starting_revision",
                   "task_prompt", "test_oracle")
cleared = {"test_oracle": 0, "starting_revision": 0}
for _rec in registry["tasks"].values():
    _orc = _rec["fields"].get("test_oracle")
    if _orc and _orc.get("round2_source"):
        del _rec["fields"]["test_oracle"]
        cleared["test_oracle"] += 1
    _rev = _rec["fields"].get("starting_revision")
    if _rev and _rev.get("recovery_tier") in ROUND2_REV_TIERS:
        del _rec["fields"]["starting_revision"]
        cleared["starting_revision"] += 1
for _rec in registry["tasks"].values():
    _f = _rec["fields"]
    _rec["missing_fields"] = [f for f in REQUIRED_FIELDS
                              if not _f.get(f) or not _f[f].get("evidence")]
    _rec["complete"] = not _rec["missing_fields"]
print(f"self-reset round-2 fields: {cleared}", flush=True)

reg_ids = set(registry["tasks"])
res_ids = {g["group_id"] for g in reserved}
if reg_ids != res_ids:
    raise SystemExit("FAIL-CLOSED: registry reserved set != recomputed "
                     f"golden partition ({len(reg_ids)} vs {len(res_ids)})")

targets = [g for g in sorted(reserved, key=lambda x: x["group_id"])
           if registry["tasks"][g["group_id"]]["missing_fields"]]
print(f"incomplete reserved targets: {len(targets)}", flush=True)

# per-source yield accounting (funnel: a group is scanned by each source
# only while it still lacks the field the source can supply)
sources = ["H1_cross_session", "H4_recovery_torture", "H2_command_output",
           "H3_pr_issue_ci"]
# units (honest counting): groups_scanned = groups whose funnel reached this
# source; found/refused = DISTINCT candidate events (session hash + event
# index, deduped within a source even when groups share sessions);
# registered = groups filled (at most one fill per group per source).
_ROW_KEYS = ("groups_scanned", "found", "registered", "refused")


def _row() -> dict:
    return {**{k: 0 for k in _ROW_KEYS}, "refusal_reasons": {},
            "seen": set()}


yields: dict[str, dict] = {s: _row() for s in sources}
field_yields = {"H5_starting_revision": _row()}
linkage_refusals: dict[str, int] = {}
refusal_examples: list[dict] = []


def count_found(source: str, t: SessionTrace, ev) -> bool:
    row = yields.get(source) or field_yields[source]
    key = (t.source_path_sha256, ev.seq)
    if key in row["seen"]:
        return False
    row["seen"].add(key)
    row["found"] += 1
    return True


def refuse(source: str, gid: str, reason: str, detail: dict | None = None,
           key=None):
    row = yields.get(source) or field_yields[source]
    if key is not None:
        if key in row["seen"]:
            return
        row["seen"].add(key)
    row["refused"] += 1
    row["refusal_reasons"][reason] = row["refusal_reasons"].get(reason, 0) + 1
    if len(refusal_examples) < 40:
        entry = {"source": source, "task_id": gid, "reason": reason}
        if detail:
            entry.update(detail)
        refusal_examples.append(entry)


registered = {"repo_identifier": [], "starting_revision": [], "test_oracle": []}
link_stats = {"linked_sessions_considered": 0, "by_link": {}}

internal_errors: list[dict] = []


def process(g: dict) -> None:
    gid = g["group_id"]
    rec = registry["tasks"][gid]
    fields = rec["fields"]
    need_oracle = "test_oracle" in rec["missing_fields"]
    need_rev = "starting_revision" in rec["missing_fields"]
    uniq, members = member_sessions(g)
    members_set = {t.source_path_sha256 for t in members}
    g_repo = fields.get("repo_identifier", {}).get("value") or \
        g.get("repo_identifier")
    g_rev = fields.get("starting_revision", {}).get("value") or \
        g.get("start_revision")

    linked = linked_sessions(g, members_set, g_repo, g_rev, linkage_refusals)
    link_stats["linked_sessions_considered"] += len(linked)
    for _t, lk in linked:
        link_stats["by_link"][lk] = link_stats["by_link"].get(lk, 0) + 1

    # ---- H5: starting_revision ------------------------------------------------
    if need_rev:
        field_yields["H5_starting_revision"]["groups_scanned"] += 1
        filled = False
        conflicted = False
        # tier 1: cross-cwd linked session_meta (same_cwd meta is round-1 tier 4)
        meta_hits = sorted(
            [(t, link) for t, link in linked
             if t.start_revision and link == "issue_ref_cross_cwd"],
            key=lambda row: start_key(row[0]))
        metas = {t.start_revision for t, _lk in meta_hits}
        row5 = field_yields["H5_starting_revision"]
        if len(metas) > 1:
            conflicted = True
            refuse("H5_starting_revision", gid, "linked_session_meta_conflict",
                   {"distinct_values": len(metas)}, key=(gid, "meta_conflict"))
        elif len(metas) == 1:
            t = meta_hits[0][0]
            row5["found"] += 1
            row5["registered"] += 1
            fields["starting_revision"] = {
                "value": t.start_revision,
                "evidence": ptr(t, 0, "session_meta"),
                "recovery_tier": "cross_cwd_linked_session_meta",
            }
            registered["starting_revision"].append(gid)
            g_rev = t.start_revision
            filled = True
        # tier 2: git HEAD records in own + linked sessions (fail-closed: a
        # conflict at tier 1 blocks every lower tier for this field)
        if not filled and not conflicted:
            scope = uniq + [t for t, _lk in linked]
            obs = git_head_observations_cached(scope)
            if obs:
                row5["found"] += 1
                # round-1 semantics: earliest session's first record wins
                t, seq, h = sorted(obs, key=lambda row: start_key(row[0]))[0]
                row5["registered"] += 1
                fields["starting_revision"] = {
                    "value": h,
                    "evidence": ptr(t, seq, "tool_result:git_head"),
                    "recovery_tier": "git_head_output_widened",
                }
                registered["starting_revision"].append(gid)
                g_rev = h
                filled = True
        if not filled and linked:
            refuse("H5_starting_revision", gid,
                   "no_head_record_in_linked_sessions",
                   {"linked_sessions": len(linked)},
                   key=(gid, "no_head_record"))

    # ---- oracle funnel: H1/H4 -> H2 -> H3 -------------------------------------
    if need_oracle and linked:
        # H1 and H4 share this one scan over linked sessions (attribution by
        # the source session's own group partition); both rows count the scan.
        yields["H1_cross_session"]["groups_scanned"] += 1
        yields["H4_recovery_torture"]["groups_scanned"] += 1

        # classification of linked sessions by their own group's partition
        def attributed(t: SessionTrace) -> str:
            sg = sess_group.get(t.source_path_sha256)
            part = part_of.get(sg["group_id"]) if sg else "quarantined"
            return "H4_recovery_torture" if part in ("recovery", "torture") \
                else "H1_cross_session"

        passes, failures = scan_classified_cached([t for t, _lk in linked])
        # distinct candidate events
        for t, ev, cmd, res, why in failures:
            src = attributed(t)
            refuse(src, gid, why, {"session": t.source_path_sha256[:16],
                                   "event_index": ev.seq},
                   key=(t.source_path_sha256, ev.seq))
        for t, ev, cmd, res, label in passes:
            count_found(attributed(t), t, ev)
        if passes:
            t, ev, cmd, res, label = max(
                passes, key=lambda row: (end_key(row[0]), row[1].seq))
            src = attributed(t)
            link = next((lk for tt, lk in linked if tt is t), None)
            yields[src]["registered"] += 1
            fields["test_oracle"] = make_record(
                t, ev, cmd, res, label, src, {"linkage": link})
            registered["test_oracle"].append(gid)
            need_oracle = False

    if need_oracle:
        # H2: widened command-output rule over the group's OWN sessions and
        # linked sessions (unclassified commands only; round-1-classified runs
        # belong to the H1 scan and are never re-counted here)
        yields["H2_command_output"]["groups_scanned"] += 1
        found, refused_rows = scan_outputs_cached(
            members + [t for t, _lk in linked])
        for t, ev, cmd, res, why in refused_rows:
            refuse("H2_command_output", gid, why,
                   {"session": t.source_path_sha256[:16],
                    "event_index": ev.seq},
                   key=(t.source_path_sha256, ev.seq))
        for t, ev, cmd, res, label in found:
            count_found("H2_command_output", t, ev)
        if found:
            t, ev, cmd, res, label = max(
                found, key=lambda row: (end_key(row[0]), row[1].seq))
            yields["H2_command_output"]["registered"] += 1
            link = next((lk for tt, lk in linked if tt is t), None)
            fields["test_oracle"] = make_record(
                t, ev, cmd, res, label, "H2_command_output",
                {"linkage": link or "own_sessions"})
            registered["test_oracle"].append(gid)
            need_oracle = False

    if need_oracle:
        # H3: CI evidence over own + linked sessions
        yields["H3_pr_issue_ci"]["groups_scanned"] += 1
        scope = members + [t for t, _lk in linked]
        found, refused_rows, chat_keys = scan_ci_cached(scope)
        for t, ev, cmd, res, why in refused_rows:
            refuse("H3_pr_issue_ci", gid, why,
                   {"session": t.source_path_sha256[:16],
                    "event_index": ev.seq},
                   key=(t.source_path_sha256, ev.seq))
        for ck in chat_keys:
            refuse("H3_pr_issue_ci", gid, "chat_only_ci_mention", key=ck)
        for t, ev, cmd, res, label in found:
            count_found("H3_pr_issue_ci", t, ev)
        if found:
            t, ev, cmd, res, label = max(
                found, key=lambda row: (end_key(row[0]), row[1].seq))
            yields["H3_pr_issue_ci"]["registered"] += 1
            fields["test_oracle"] = make_record(
                t, ev, cmd, res, label, "H3_pr_issue_ci", {"variant": label})
            registered["test_oracle"].append(gid)
            need_oracle = False

    # refresh completeness
    required = ("task_id", "repo_identifier", "starting_revision",
                "task_prompt", "test_oracle")
    rec["missing_fields"] = [f for f in required
                             if not fields.get(f)
                             or not fields[f].get("evidence")]
    rec["complete"] = not rec["missing_fields"]
    if rec["complete"]:
        rec["oracle_absence"] = None

for idx, g in enumerate(targets, 1):
    if idx % 50 == 0:
        print(f"  scanned {idx}/{len(targets)} targets", flush=True)
    try:
        process(g)
    except Exception as exc:  # fail closed: the group keeps every missing field
        internal_errors.append({"task_id": g["group_id"],
                                "error": f"{type(exc).__name__}: {exc}"})
        print(f"  INTERNAL ERROR on {g['group_id']}: {exc}", flush=True)

print("mining done", flush=True)

# ------------------------------------------------------ registry histograms
report = registry["histograms"]
report["field_presence"] = {f: 0 for f in
                            ("task_id", "task_prompt", "repo_identifier",
                             "starting_revision", "test_oracle")}
report["rev_sources"] = {}
report["oracle_kinds"] = {}
report["blockers"] = {}
report["blocker_field_counts"] = {}
report["round2_source_yields"] = {
    **{s: {k: v for k, v in yields[s].items()
           if k not in ("refusal_reasons", "seen")}
       for s in sources},
    **{s: {k: v for k, v in field_yields[s].items()
           if k not in ("refusal_reasons", "seen")}
       for s in field_yields},
}
report["round2_refusal_reasons"] = {
    **{s: yields[s]["refusal_reasons"] for s in sources if yields[s]["refusal_reasons"]},
    **{s: field_yields[s]["refusal_reasons"] for s in field_yields
       if field_yields[s]["refusal_reasons"]},
}
report["round2_linkage_refusals"] = linkage_refusals
report["round2_link_stats"] = link_stats

complete_ids = []
for gid in sorted(registry["tasks"]):
    rec = registry["tasks"][gid]
    fields = rec["fields"]
    for f in ("task_id", "task_prompt", "repo_identifier", "starting_revision",
              "test_oracle"):
        if fields.get(f) and fields[f].get("evidence"):
            report["field_presence"][f] += 1
    rev = fields.get("starting_revision")
    if rev and rev.get("evidence"):
        tier = rev.get("recovery_tier", "unknown")
        report["rev_sources"][tier] = report["rev_sources"].get(tier, 0) + 1
    orc = fields.get("test_oracle")
    if orc and orc.get("evidence"):
        k = orc.get("verification_kind", "unknown")
        report["oracle_kinds"][k] = report["oracle_kinds"].get(k, 0) + 1
    missing = rec["missing_fields"]
    key = "+".join(missing) if missing else "<complete>"
    report["blockers"][key] = report["blockers"].get(key, 0) + 1
    for f in missing:
        report["blocker_field_counts"][f] = \
            report["blocker_field_counts"].get(f, 0) + 1
    if not missing:
        complete_ids.append(gid)

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
registry["generated_utc"] = now
registry["complete_count"] = len(complete_ids)
registry["round2"] = {
    "generated_utc": now,
    "hypotheses": ["H1_cross_session", "H2_command_output", "H3_pr_issue_ci",
                   "H4_recovery_torture", "H5_starting_revision"],
    "new_oracles": len(registered["test_oracle"]),
    "new_revisions": len(registered["starting_revision"]),
    "new_repos": len(registered["repo_identifier"]),
    "yield_table": report["round2_source_yields"],
    "refusal_reasons": report["round2_refusal_reasons"],
    "linkage_refusals": linkage_refusals,
    "link_stats": link_stats,
    "internal_errors": internal_errors,
    "complete_after": len(complete_ids),
}

# rebuild flat indexes FROM the registry (source of truth; no stale entries)
oracle_index: dict[str, str] = {}
evidence_index: dict[str, dict] = {}
for gid, rec2 in registry["tasks"].items():
    orc = rec2["fields"].get("test_oracle")
    if orc and orc.get("value"):
        oracle_index[gid] = orc["value"]
    entry: dict = {"evidence": {}}
    for field in ("repo_identifier", "starting_revision"):
        f = rec2["fields"].get(field)
        if f and f.get("value") and f.get("evidence") \
                and f.get("recovery_tier") != "native_session_meta":
            entry[field] = f["value"]
            entry["evidence"][field] = f["evidence"]
    if entry["evidence"]:
        evidence_index[gid] = entry

ensure_dir(OUT_DIR)
outputs = {
    "oracle-registry.json": registry,
    "oracle-index.json": dict(sorted(oracle_index.items())),
    "evidence-index.json": dict(sorted(evidence_index.items())),
    "round2-yield-report.json": {
        "schema_version": "1.0.0",
        "kind": "mimo-halo-oracle-round2-yield-report",
        "generated_utc": now,
        "note": ("PRIVATE: per-source funnel yields over the incomplete "
                 "reserved groups; pointers are session-id-hash + event "
                 "index; refusal examples carry one hashed session ref only"),
        "targets_incomplete_at_start": len(targets),
        "yield_table": report["round2_source_yields"],
        "refusal_reasons": report["round2_refusal_reasons"],
        "linkage_refusals": linkage_refusals,
        "link_stats": link_stats,
        "refusal_examples": refusal_examples,
        "internal_errors": internal_errors,
        "registered_task_ids": {k: sorted(v) for k, v in registered.items()},
        "complete_after": len(complete_ids),
        "blockers_after": dict(sorted(
            report["blockers"].items(), key=lambda kv: -kv[1])),
        "blocker_field_counts_after": dict(sorted(
            report["blocker_field_counts"].items(), key=lambda kv: -kv[1])),
    },
}
for name, payload in outputs.items():
    atomic_write_json(OUT_DIR / name, payload, indent=None)

# write-corruption guard: re-read + parse every written file
for name, payload in outputs.items():
    back = json.loads((OUT_DIR / name).read_text())
    if back != payload:
        raise SystemExit(f"round-trip mismatch (write corruption?): {name}")
    print(f"verified parse: {name}", flush=True)

print("\n=== round-2 yield table ===")
for s, row in {**yields, **field_yields}.items():
    print(f"{s:26s} scanned={row['groups_scanned']:4d} "
          f"found={row['found']:4d} "
          f"registered={row['registered']:4d} refused={row['refused']:4d}")
print(f"linkage refusals: {linkage_refusals}")
print(f"internal errors (fail-closed, untouched): {len(internal_errors)}")
print(f"new oracles={len(registered['test_oracle'])} "
      f"new revisions={len(registered['starting_revision'])} "
      f"new repos={len(registered['repo_identifier'])}")
print(f"complete reserved groups after round 2: {len(complete_ids)} / "
      f"{len(reserved)}")
print("blockers after:", json.dumps(report["blocker_field_counts"]))
print("done", flush=True)
