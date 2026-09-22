"""Shared primitives for trace tooling. Standard library only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator


class TraceError(RuntimeError):
    """Fail-closed error: missing required input, ambiguous identity, or unsafe output location."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    """Hash of the absolute path string itself (never the path is emitted)."""
    return sha256_text(str(path.resolve()))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return sha256_text(stable_json(value))


def iter_jsonl(path: Path, limit: int | None = None) -> Iterator[tuple[int, dict]]:
    """Yield (line_number, parsed) for each valid JSON object line. Invalid lines are skipped (counted by caller via parse errors hook if needed)."""
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            yield lineno, obj
            count += 1
            if limit is not None and count >= limit:
                return


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path: Path, payload: Any, indent: int | None = 2) -> None:
    """Write JSON atomically; refuses when the destination volume lacks free headroom."""
    data = json.dumps(payload, indent=indent, ensure_ascii=False).encode("utf-8")
    directory = (path.parent)
    directory.mkdir(parents=True, exist_ok=True)
    check_capacity(path, len(data) + (1 << 20))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def check_capacity(path: Path, required_bytes: int) -> int:
    """Return free bytes on the destination volume; raise when below required."""
    directory = path if path.is_dir() else path.parent
    st = os.statvfs(directory)
    free = st.f_bavail * st.f_frsize
    if free < required_bytes:
        raise TraceError(
            f"insufficient free capacity on output volume for {path}: "
            f"{free} bytes free, {required_bytes} required"
        )
    return free


def free_bytes(path: Path) -> int:
    st = os.statvfs(path if path.is_dir() else path.parent)
    return st.f_bavail * st.f_frsize


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def repo_key_for_cwd(cwd: str | None) -> str | None:
    """Stable key for the session working directory (hashed, never the path)."""
    if not cwd:
        return None
    resolved = Path(os.path.expanduser(cwd)).resolve()
    return sha256_text(str(resolved))
