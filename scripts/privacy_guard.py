#!/usr/bin/env python3
"""Versioned privacy guard for the public dymoo/mimo-halo-lab repository.

Standard library only (Python >= 3.11). The default mode inspects *staged
blobs* (never the working tree) against the empty tree or HEAD, so what
leaves the workstation is exactly what was scanned. An explicit ``--paths``
mode scans working-tree files for CI or manual runs.

Scanning contract
-----------------
- Path policy (protected dirs/extensions/env files) runs BEFORE any read in
  both modes; blocked payload blobs are never read.
- Accepted blobs are bounded at 64 MiB: a larger blob fails closed (exit 2)
  via ``git cat-file -s`` / ``os.lstat`` BEFORE it is read. Within the bound
  the WHOLE blob is scanned -- there are no prefix windows anywhere.
- Content decodes strictly as UTF-8; NUL bytes or undecodable bytes
  anywhere in the blob block as ``binary-payload`` (no silent
  ``errors="replace"``).
- Parsing is NUL-safe end to end (``-z`` diff/index output, argv-separated
  git invocations), so spaces, quotes, unicode and newlines in filenames
  are handled safely.

Blocked
-------
- protected directories (models, raw/normalized-private traces, caches,
  scratch, upstreams, datasets/private, artifacts/models, calibration/cache)
  matched over casefolded path components -- TRACES/RAW is still protected.
- protected model/weight extensions and ``*.parquet.private``, casefolded.
- env files: ``.env``, ``.env.*``, ``*.env`` and ``.envrc`` (casefolded).
  ``.env.example`` is allowed by PATH ONLY -- its contents are still fully
  credential-scanned and a real credential in it still blocks.
- binary payloads: NUL or non-UTF-8 anywhere in the blob.
- long base64/hex runs (>=4096 / >=2048 contiguous characters).
- structural conversation-trace JSONL at ANY path: at least 2 lines, every
  non-blank line a JSON object, and an object carrying a chat record
  (``role`` in user/assistant/system/tool/developer plus ``content``, or
  that record nested under ``messages`` / ``message`` / ``item``). Line
  length alone is NOT a trace signal: minified one-line metadata, ~tens-of-
  MB official metadata, pretty-printed JSON documents and non-chat JSONL
  (schema/config/header/tensor-inventory/metrics records) are accepted.
- credential signatures: AWS/GitHub/OpenAI/Anthropic/Slack/Google/HF/npm
  tokens, Stripe SECRET and RESTRICTED keys (publishable ``pk_`` keys are
  public by design and allowed), private key blocks, JWTs, bearer headers,
  and generic credential assignments whose identifier TAIL is a credential
  alias -- ``db_password``, ``AWS_SECRET_ACCESS_KEY``, ``client_secret``,
  ``access_token`` all match; tokenizer/metadata identifiers (``tokenizer``,
  ``tokenizer_class``, ``max_tokens``, ``token_encoding``) do not. Quoted
  values may contain common punctuation; unquoted values must be a single
  whitespace-free token.
- symlinks (staged and ``--paths``): never followed. The stored target
  string itself is checked -- absolute, worktree-escaping, or (by root and
  link-parent normalization, casefolded) protected destinations block.

Documented heuristic limits (conservative by design -- there is NO broad
filename or value allowlist that could let a real credential out):
- ``encoded-payload``/``hex-payload`` fire on >=4096 base64 / >=2048 hex
  runs; a base64 data-URI image embedded in public docs WILL block. Link the
  asset instead of embedding it; the rule is not relaxed by filename.
- ``jwt`` matches the documented public example token pair (``eyJ...`` dot
  ``eyJ...``) used in OAuth docs and fixtures. Same resolution: annotate or
  rewrite the doc; no override list exists.
- Only clearly non-secret VALUE shapes are skipped, based on the value alone
  in every file (never because of the filename): ``${VAR}``, ``$VAR``,
  ``<placeholder>``, long ``x``/``X``/``*`` runs, changeme-style words.
- Unquoted credential values containing whitespace are only caught via their
  quoted form; non-UTF-8 text (latin-1 prose etc.) blocks as
  ``binary-payload``; a chat record inside a single-line (non-JSONL) JSON
  document is out of scope, as is a JSONL file with a malformed line.

Fail closed: any git/I/O error, or any accepted blob over the 64 MiB cap,
exits 2 and blocks the commit. Offending secret text is NEVER printed --
only repr()-quoted paths, rule ids and line numbers.

Exit codes: 0 clean, 1 policy violations, 2 guard error (fail closed).
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

SCANNER_VERSION = "1.1.0"
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
DIFF_FILTER = "ACMRT"  # added, copied, modified, renamed, typechanged
MAX_SCAN_BYTES = 64 * 1024 * 1024  # accepted blobs over this fail closed pre-read
ZERO_SHA_RE = re.compile(r"^0+$")


class GuardError(RuntimeError):
    """Any failure that must block the commit (fail closed)."""


# Root-relative directory prefixes that never carry committable content.
# Top-level only: src/mimo_halo/{models,traces,artifacts,...} are legitimate
# source packages; the raw-payload locations are not. Compared over
# casefolded path components (see check_path).
PROTECTED_DIR_PREFIXES = (
    ("models",),
    ("upstreams",),
    ("datasets", "private"),
    ("traces", "raw"),
    ("traces", "normalized-private"),
    ("calibration", "cache"),
    ("artifacts", "models"),
    ("scratch",),
    (".scratch",),
    (".cache",),
)

BLOCKED_EXTENSIONS = frozenset(
    {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".ggml", ".npy", ".npz"}
)

# Path-allowed example env file. Contents are still fully credential-scanned.
ENV_EXEMPT_NAMES = frozenset({".env.example"})

SECRET_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("github-fine-grained-token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[bpars]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("npm-token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    # Secret and restricted Stripe keys only; pk_ publishable keys are
    # designed to be public in browser code and are allowed.
    ("stripe-secret-key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}\b")),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\b")),
    (
        "bearer-header",
        re.compile(r"(?i)\bauthorization[\"']?\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    ),
]

# Generic credential assignments. The credential alias must TERMINATE the
# identifier (so prefixed names like db_password / AWS_SECRET_ACCESS_KEY /
# client_secret / access_token match via the [\w-]* prefix, while
# tokenizer/metadata identifiers such as tokenizer_class or max_tokens do
# not). Values: any quoted literal >= 20 chars (interior punctuation
# allowed), or a whitespace-free unquoted token >= 20 chars.
GENERIC_CREDENTIAL_RE = re.compile(
    r"(?i)[\w-]*(?:password|passwd|secret[_-]?key|client[_-]?secret|secret"
    r"|api[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token"
    r"|credential|token)\b"
    r"[\"']?\s*[:=]\s*"
    r"(?:\"(?P<dquot>[^\"\n]{20,})\"|'(?P<squot>[^'\n]{20,})'"
    r"|(?P<bare>[A-Za-z0-9+/_\-.@]{20,}))"
)

# Clearly non-secret VALUE shapes, matched against the value (or whole token)
# alone -- never against a filename. This is a narrow shape exemption, not a
# credential bypass: anything that does not fully match these shapes blocks.
PLACEHOLDER_VALUE_RE = re.compile(
    r"\$\{[A-Za-z_][A-Za-z0-9_]*\}"
    r"|\$[A-Za-z_][A-Za-z0-9_]*"
    r"|<[A-Za-z0-9 _.:/\-]+>"
    r"|[A-Za-z0-9_.+/\-]*[xX*]{8,}[A-Za-z0-9_.+/\-]*"
    r"|(?:change|replace|pick)[_\-]?me"
    r"|your[_\-]?(?:token|key|secret|password|credential)[_\-]?(?:here|name|goes[_\-]?here)?"
    r"|placeholder[_\-]?(?:value|token)?"
    r"|not[_\-]?a[_\-]?secret"
    r"|redacted",
    re.IGNORECASE,
)

ENCODED_PAYLOAD_RE = re.compile(r"[A-Za-z0-9+/=]{4096,}")
HEX_PAYLOAD_RE = re.compile(r"[0-9a-fA-F]{2048,}")

TRACE_ROLES = frozenset({"user", "assistant", "system", "tool", "developer"})
_WINDOWS_ABS_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def check_path(path: str) -> str | None:
    """Return a rule id if the path itself is protected, else None.

    Every component is casefolded before comparison so case variants
    (TRACES/RAW, PROD.ENV, MODEL.GGUF) are protected identically.
    """
    parts = PurePosixPath(path).parts
    if not parts:
        return "empty-path"
    parts_cf = tuple(part.casefold() for part in parts)
    name = parts_cf[-1]
    if "." in name:
        suffix = name.rsplit(".", 1)[1]
        if f".{suffix}" in BLOCKED_EXTENSIONS:
            return f"protected-extension:{suffix}"
    if name not in ENV_EXEMPT_NAMES and (
        name == ".envrc" or name.endswith(".env") or name.startswith(".env.")
    ):
        return "env-file"
    if name.endswith(".parquet.private"):
        return "private-parquet"
    for prefix in PROTECTED_DIR_PREFIXES:
        if len(parts_cf) > len(prefix) and parts_cf[: len(prefix)] == prefix:
            return "protected-dir:" + "/".join(prefix)
    return None


def check_symlink_target(target: str, link_path: str) -> str | None:
    """Policy for a symlink's stored target string. The link is never
    followed -- only this string commits, so absolute/worktree-escaping/
    protected destinations all block. Both interpretations are checked:
    repo-root-relative (how symlink farms name things) and parent-relative
    (real POSIX symlink semantics), with the parent derived from the link's
    own location and every check casefolded via check_path.
    """
    target = target.strip()
    if not target or "\x00" in target:
        return "symlink-invalid-target"
    if target.startswith(("/", "\\")) or _WINDOWS_ABS_RE.match(target):
        return "symlink-absolute-target"
    root_rel = posixpath.normpath(target)
    if root_rel != ".." and not root_rel.startswith("../") and check_path(root_rel):
        return "symlink-protected-target"
    parent_rel = posixpath.normpath(posixpath.join(posixpath.dirname(link_path), target))
    if parent_rel == ".." or parent_rel.startswith("../") or posixpath.isabs(parent_rel):
        return "symlink-escaping-target"
    if check_path(parent_rel):
        return "symlink-protected-target"
    return None


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _is_chat_record(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False

    def is_msg(value: object) -> bool:
        return (
            isinstance(value, dict)
            and isinstance(value.get("role"), str)
            and value["role"].casefold() in TRACE_ROLES
            and "content" in value
        )

    if is_msg(obj):
        return True
    messages = obj.get("messages")
    if isinstance(messages, list) and any(is_msg(m) for m in messages):
        return True
    for key in ("message", "item"):
        if is_msg(obj.get(key)):
            return True
    return False


def find_trace_jsonl(text: str) -> int | None:
    """Return the 1-based line of the first chat record in structural
    conversation-trace JSONL, else None.

    Structural, not size-based: at least 2 lines, every non-blank line a
    JSON object that parses. Any other shape (prose, pretty-printed JSON,
    minified one-line metadata, mixed logs) is not a trace here -- safe JSON
    schema/config/header/tensor-inventory documents are accepted, and line
    length alone never fires. A malformed line aborts classification
    (documented conservative limit).
    """
    lines = [(n, line) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
    if len(lines) < 2:
        return None
    if not all(line.lstrip().startswith("{") for _, line in lines):
        return None
    for n, line in lines:
        try:
            obj = json.loads(line)
        except ValueError:
            return None
        if _is_chat_record(obj):
            return n
    return None


def check_text(text: str) -> list[tuple[str, int]]:
    """Content rules on decoded text. Returns (rule_id, line_number) pairs."""
    findings: list[tuple[str, int]] = []
    for rule_id, pattern in SECRET_RULES:
        for match in pattern.finditer(text):
            if PLACEHOLDER_VALUE_RE.fullmatch(match.group(0)):
                continue
            findings.append((rule_id, _line_of(text, match.start())))
            break
    for match in GENERIC_CREDENTIAL_RE.finditer(text):
        value = match.group("dquot") or match.group("squot") or match.group("bare") or ""
        if PLACEHOLDER_VALUE_RE.fullmatch(value):
            continue
        findings.append(("generic-credential-assignment", _line_of(text, match.start())))
        break
    for rule_id, pattern in (
        ("encoded-payload", ENCODED_PAYLOAD_RE),
        ("hex-payload", HEX_PAYLOAD_RE),
    ):
        match = pattern.search(text)
        if match:
            findings.append((rule_id, _line_of(text, match.start())))
    trace_line = find_trace_jsonl(text)
    if trace_line is not None:
        findings.append(("trace-jsonl", trace_line))
    return findings


def scan_blob(raw: bytes, dst_mode: str, path: str) -> list[tuple[str, int]]:
    """Scan a whole accepted blob (already size-checked by the caller)."""
    if dst_mode.startswith("160000"):
        return []  # gitlink/submodule pointer: no content (object may be absent)
    if dst_mode.startswith("120000"):
        try:
            target = raw.decode("utf-8")
        except UnicodeDecodeError:
            return [("symlink-invalid-target", 1)]
        rule = check_symlink_target(target, path)
        return [(rule, 1)] if rule else []
    if b"\x00" in raw:
        return [("binary-payload", 1)]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [("binary-payload", 1)]
    return check_text(text)


def git_bytes(cwd: str, *args: str) -> bytes:
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    except OSError as exc:  # git missing/unreadable -> fail closed
        raise GuardError(f"git {args[0]} could not be executed: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise GuardError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout


def head_or_empty_tree(cwd: str) -> str:
    """Diff base: HEAD when it exists, else the empty tree (unborn branch)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=cwd, capture_output=True
        )
    except OSError as exc:
        raise GuardError(f"git rev-parse could not be executed: {exc}") from exc
    sha = proc.stdout.strip().decode("ascii", errors="replace")
    if proc.returncode == 0 and sha:
        return sha
    return EMPTY_TREE_SHA


def parse_raw_diff(out: bytes) -> list[tuple[str, str, str]]:
    """Parse ``git diff-index --cached -z`` raw output.

    Returns (destination_path, destination_blob_sha, destination_mode) triples.
    NUL-delimited parsing is safe for spaces, quotes, unicode and newlines in
    filenames. Copy/rename entries carry a source path before the
    destination path; only the destination commits and is scanned.
    """
    entries: list[tuple[str, str, str]] = []
    tokens = out.split(b"\x00")
    i = 0
    while i < len(tokens):
        header = tokens[i]
        i += 1
        if not header.startswith(b":"):
            continue
        fields = header.decode("ascii", errors="replace").split(" ")
        if len(fields) != 5:
            raise GuardError(f"unparsable diff-index header: {header!r}")
        dst_mode, dst_sha, status = fields[1], fields[3], fields[4]
        if status[:1] in ("C", "R"):
            i += 1  # source path; the destination follows
        if i >= len(tokens):
            raise GuardError("diff-index output ended mid-entry")
        entries.append((os.fsdecode(tokens[i]), dst_sha, dst_mode))
        i += 1
    return entries


def parse_ls_files(out: bytes) -> dict[str, str]:
    """Stage-0 path -> blob sha map from ``git ls-files -s -z``."""
    index: dict[str, str] = {}
    for record in out.split(b"\x00"):
        if not record:
            continue
        meta, _, raw_path = record.partition(b"\t")
        fields = meta.split()
        if len(fields) >= 3 and fields[2] == b"0":
            index[os.fsdecode(raw_path)] = fields[1].decode("ascii", errors="replace")
    return index


def staged_rev(dst_sha: str, path: str, index: dict[str, str]) -> str:
    """Object rev for the stage-0 content of ``path`` (safe for any name)."""
    sha = dst_sha
    if ZERO_SHA_RE.match(sha):
        sha = index.get(path, "")
    if not sha or ZERO_SHA_RE.match(sha):
        # Last resort (e.g. intent-to-add edge cases); stage 0 rev syntax is
        # safe for any path because the rev starts with ':'.
        return f":0:{path}"
    return sha


def blob_size(toplevel: str, rev: str, path: str) -> int:
    """Preflight size via ``git cat-file -s`` BEFORE any content read."""
    raw = git_bytes(toplevel, "cat-file", "-s", rev).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise GuardError(f"unparsable size for {path!r}: {raw!r}") from exc


def scan_staged(cwd: str) -> list[tuple[str, str, int]]:
    toplevel = git_bytes(cwd, "rev-parse", "--show-toplevel").decode("utf-8", errors="replace").strip()
    base = head_or_empty_tree(cwd)
    entries = parse_raw_diff(
        git_bytes(
            cwd, "diff-index", "--cached", "-r", "-z", "-M",
            f"--diff-filter={DIFF_FILTER}", base,
        )
    )
    index = parse_ls_files(git_bytes(cwd, "ls-files", "-s", "-z"))
    violations: list[tuple[str, str, int]] = []
    for path, dst_sha, dst_mode in entries:
        rule = check_path(path)
        if rule:
            violations.append((path, rule, 1))
            continue  # never read a blocked payload blob
        if dst_mode.startswith("160000"):
            continue  # submodule pointer: nothing to read; object may be absent
        rev = staged_rev(dst_sha, path, index)
        size = blob_size(toplevel, rev, path)
        if size > MAX_SCAN_BYTES:
            raise GuardError(
                f"staged {path!r} is {size} bytes, over the {MAX_SCAN_BYTES}-byte "
                "scan cap; fail closed before reading"
            )
        raw = git_bytes(toplevel, "cat-file", "blob", rev)
        if len(raw) > MAX_SCAN_BYTES:
            raise GuardError(f"staged {path!r} changed size during the scan; fail closed")
        for content_rule, line_no in scan_blob(raw, dst_mode, path):
            violations.append((path, content_rule, line_no))
    return violations


def scan_paths(paths: list[str]) -> list[tuple[str, str, int]]:
    """Explicit working-tree scan (CI / manual). Missing files fail closed.

    Entries resolve against the current directory; entries outside it fail
    closed. The path policy runs before the lstat size check and before any
    read. Symlinks are NEVER followed -- only the stored target string is
    checked -- so private file content is never reached through a link.
    """
    violations: list[tuple[str, str, int]] = []
    cwd = os.getcwd()
    for raw_path in paths:
        absolute = os.path.abspath(raw_path)
        relative = os.path.relpath(absolute, cwd)
        if relative == os.pardir or relative.startswith(os.pardir + os.sep):
            raise GuardError(f"--paths entry {raw_path!r} resolves outside {cwd!r}; fail closed")
        path = relative.replace(os.sep, "/")
        rule = check_path(path)
        if rule:
            violations.append((path, rule, 1))
            continue  # blocked path: never lstat-size or read the payload
        try:
            info = os.lstat(absolute)
        except OSError as exc:
            raise GuardError(f"cannot read metadata of {raw_path!r}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            try:
                target = os.readlink(absolute)
            except OSError as exc:
                raise GuardError(f"cannot read symlink {raw_path!r}: {exc}") from exc
            link_rule = check_symlink_target(target, path)
            if link_rule:
                violations.append((path, link_rule, 1))
            continue  # never follow the link into another file
        if not stat.S_ISREG(info.st_mode):
            raise GuardError(f"cannot scan non-regular file {raw_path!r}; fail closed")
        if info.st_size > MAX_SCAN_BYTES:
            raise GuardError(
                f"{path!r} is {info.st_size} bytes, over the {MAX_SCAN_BYTES}-byte "
                "scan cap; fail closed before reading"
            )
        try:
            data = Path(absolute).read_bytes()
        except OSError as exc:
            raise GuardError(f"cannot read {raw_path!r}: {exc}") from exc
        if len(data) > MAX_SCAN_BYTES:
            raise GuardError(f"{path!r} changed size during the scan; fail closed")
        for content_rule, line_no in scan_blob(data, "100644", path):
            violations.append((path, content_rule, line_no))
    return violations


def report(violations: list[tuple[str, str, int]]) -> None:
    print(f"privacy-guard v{SCANNER_VERSION}: blocked {len(violations)} violation(s):")
    for path, rule, line_no in violations:
        print(f"  BLOCKED {path!r} rule={rule} line={line_no}")
    print(
        "privacy-guard: commit aborted. Remove or redact the flagged staged "
        "content and re-stage. Offending secret text is never echoed by design."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Versioned staged-blob privacy guard for dymoo/mimo-halo-lab."
    )
    parser.add_argument(
        "--version", action="version", version=f"privacy-guard {SCANNER_VERSION}"
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="scan staged blobs against HEAD (or the empty tree); the default",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        metavar="PATH",
        help="explicit working-tree scan mode for CI or manual runs",
    )
    args = parser.parse_args(argv)
    if args.paths is not None and args.staged:
        parser.error("--paths and --staged are mutually exclusive")
    try:
        if args.paths is not None:
            violations = scan_paths(args.paths)
        else:
            violations = scan_staged(os.getcwd())
    except GuardError as exc:
        print(f"privacy-guard v{SCANNER_VERSION}: ERROR (fail closed): {exc}", file=sys.stderr)
        return 2
    if violations:
        report(violations)
        return 1
    print(f"privacy-guard v{SCANNER_VERSION}: clean. No privacy problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
