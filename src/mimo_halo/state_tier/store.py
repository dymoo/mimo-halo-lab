"""Blocking opaque state store for the NVMe state tier.

Consumer seam from local://nvme-implementation-contract.md:

    PYTHONPATH=src python3 -m mimo_halo.state_tier.store put \\
        --root DIR --session OPAQUE_ID --generation N --state-file INPUT \\
        --identity IDENTITY.json --token-count N --token-sha256 SHA256 \\
        --output RECEIPT.json
    PYTHONPATH=src python3 -m mimo_halo.state_tier.store get \\
        --root DIR --snapshot-id ID --expected-identity IDENTITY.json \\
        --output RESTORED.bin
    PYTHONPATH=src python3 -m mimo_halo.state_tier.store inspect \\
        --root DIR --snapshot-id ID

Standard library only (Python >= 3.11), blocking I/O, POSIX advisory
locking. Full behaviour, schema, locking guarantees and limitations are
documented in docs/state-store.md; the receipt schema is
schemas/state-snapshot.schema.json.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
KIND = "target_state"

_CHUNK_BYTES = 1 << 20
_HEX64 = re.compile("[0-9a-f]{64}")
_HEX40 = re.compile("[0-9a-f]{40}")

_BLOBS_DIR = "blobs"
_ACTIVE_INDEX_DIR = "active-index"
_INDEX_FILE = "index.sqlite"
_LOCK_FILE = "store.lock"

_IDENTITY_KEYS = (
    "schema_version",
    "model_sha256",
    "quant_revision",
    "tokenizer_sha256",
    "chat_template_sha256",
    "runtime_commit",
    "runtime_patch_sha256",
    "cache_format_version",
    "kv_config_sha256",
    "namespace",
)

_HEX64_IDENTITY_KEYS = (
    "model_sha256",
    "tokenizer_sha256",
    "chat_template_sha256",
    "runtime_patch_sha256",
    "kv_config_sha256",
)

_SNAPSHOT_TABLE = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id      TEXT PRIMARY KEY,
    namespace        TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    generation       INTEGER NOT NULL,
    state_sha256     TEXT NOT NULL,
    serialized_bytes INTEGER NOT NULL,
    token_count      INTEGER NOT NULL,
    token_sha256     TEXT NOT NULL,
    identity_json    TEXT NOT NULL,
    blob_path        TEXT NOT NULL,
    kind             TEXT NOT NULL,
    created_at       INTEGER NOT NULL,
    last_accessed    INTEGER NOT NULL,
    UNIQUE (namespace, session_id, generation)
)
"""


class StoreRefusal(Exception):
    """Named fail-closed refusal: reported on stderr with exit status 1."""


# ---------------------------------------------------------------------------
# Small filesystem helpers
# ---------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _discard(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _check_outside_git(root: Path) -> None:
    """Refuse a store root inside any Git working tree (private-state rule)."""
    resolved = Path(os.path.realpath(root))
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            raise StoreRefusal(
                f"store root must be outside a Git working tree: "
                f"{resolved} lies within {candidate}"
            )


def _write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise StoreRefusal(f"output directory does not exist: {parent}")
    data = (json.dumps(record, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    fd, name = tempfile.mkstemp(dir=parent, prefix=".state-output-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        _discard(name)
        raise
    _fsync_dir(parent)


# ---------------------------------------------------------------------------
# Identity validation (structure and equality only, never attestation)
# ---------------------------------------------------------------------------


def _load_identity(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StoreRefusal(f"identity unreadable: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise StoreRefusal("identity must be a JSON object")
    _validate_identity(raw)
    return raw


def _validate_identity(identity: dict[str, Any]) -> None:
    if set(identity) != set(_IDENTITY_KEYS):
        missing = sorted(set(_IDENTITY_KEYS) - set(identity))
        extra = sorted(set(identity) - set(_IDENTITY_KEYS))
        raise StoreRefusal(
            f"identity schema mismatch: missing={missing} extra={extra}"
        )
    if identity["schema_version"] != 1:
        raise StoreRefusal("identity schema_version must be 1")
    for key in _HEX64_IDENTITY_KEYS:
        value = identity[key]
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            raise StoreRefusal(f"identity {key} must be lowercase 64-hex sha256")
    commit = identity["runtime_commit"]
    if not isinstance(commit, str) or _HEX40.fullmatch(commit) is None:
        raise StoreRefusal("identity runtime_commit must be lowercase 40-hex")
    for key in ("quant_revision", "cache_format_version", "namespace"):
        value = identity[key]
        if not isinstance(value, str) or not value:
            raise StoreRefusal(f"identity {key} must be a nonempty string")


# ---------------------------------------------------------------------------
# Store layout, locking, index
# ---------------------------------------------------------------------------


def _snapshot_id(namespace: str, session_id: str, generation: int) -> str:
    payload = json.dumps(
        ["mimo-halo-state-snapshot-v1", namespace, session_id, generation],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextlib.contextmanager
def _store_lock(active_dir: Path) -> Iterator[None]:
    """Exclusive advisory lock held for the whole command.

    Serializing every command is the documented concurrency contract: two
    commands can never interleave blob and index mutation, and the kernel
    releases the lock if the process dies.
    """
    fd = os.open(active_dir / _LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _open_index(active_dir: Path) -> sqlite3.Connection:
    index_path = active_dir / _INDEX_FILE
    conn = sqlite3.connect(index_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    # SQLite creates the database with default umask permissions; the store
    # contract requires private files, so tighten to 0600 on every open.
    os.chmod(index_path, 0o600)
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute(_SNAPSHOT_TABLE)
    conn.commit()
    return conn


def _record_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": row["snapshot_id"],
        "session_id": row["session_id"],
        "generation": row["generation"],
        "token_count": row["token_count"],
        "token_sha256": row["token_sha256"],
        "identity": json.loads(row["identity_json"]),
        "state_sha256": row["state_sha256"],
        "serialized_bytes": row["serialized_bytes"],
        "created_at": row["created_at"],
        "last_accessed": row["last_accessed"],
        "blob_path": row["blob_path"],
        "kind": row["kind"],
    }


def _confined_blob_path(root: Path, blob_path: str) -> Path:
    """Resolve a stored relative blob path, refusing any escape."""
    if not blob_path or PurePosixPath(blob_path).is_absolute():
        raise StoreRefusal(f"index blob path is not store-relative: {blob_path!r}")
    parts = PurePosixPath(blob_path).parts
    if not parts or parts[0] != _BLOBS_DIR or ".." in parts:
        raise StoreRefusal(f"index blob path escapes the store layout: {blob_path!r}")
    candidate = root.joinpath(*parts)
    blobs_root = Path(os.path.realpath(root / _BLOBS_DIR))
    parent = Path(os.path.realpath(candidate.parent))
    if not parent.is_relative_to(blobs_root):
        raise StoreRefusal(
            f"index blob path escapes the blobs directory: {blob_path!r}"
        )
    return candidate


def _blob_matches(path: Path, want_sha256: str, want_size: int) -> bool:
    """True only for an existing regular file (never a symlink) whose bytes
    hash exactly to the recorded content."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return False
    hasher = hashlib.sha256()
    total = 0
    try:
        while True:
            chunk = os.read(fd, _CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
            total += len(chunk)
    finally:
        os.close(fd)
    return total == want_size and hasher.hexdigest() == want_sha256


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_put(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    session_id = args.session
    if not session_id:
        raise StoreRefusal("--session must be a nonempty opaque id")
    generation = args.generation
    if generation < 0:
        raise StoreRefusal("--generation must be >= 0")
    token_count = args.token_count
    if token_count < 0:
        raise StoreRefusal("--token-count must be >= 0")
    if _HEX64.fullmatch(args.token_sha256) is None:
        raise StoreRefusal("--token-sha256 must be lowercase 64-hex sha256")
    state_file = Path(args.state_file)
    if not state_file.is_file():
        raise StoreRefusal(f"--state-file is not a regular file: {state_file}")
    identity = _load_identity(Path(args.identity))
    output = Path(args.output)

    _check_outside_git(root)
    blobs_dir = root / _BLOBS_DIR
    active_dir = root / _ACTIVE_INDEX_DIR
    ns_shard = hashlib.sha256(identity["namespace"].encode("utf-8")).hexdigest()[:16]

    _ensure_private_dir(root)
    _ensure_private_dir(blobs_dir)
    _ensure_private_dir(active_dir)
    _fsync_dir(root)
    ns_dir = blobs_dir / ns_shard
    _ensure_private_dir(ns_dir)
    _fsync_dir(blobs_dir)

    with _store_lock(active_dir):
        conn = _open_index(active_dir)
        try:
            _fsync_dir(active_dir)  # durable dir entries for lock + index file

            # Prehash pass: stream the input read-only while hashing every
            # payload byte. No payload temp file is created yet, so the
            # generation check below refuses a collision without any payload
            # write having happened.
            hasher = hashlib.sha256()
            total = 0
            try:
                with state_file.open("rb") as source:
                    while True:
                        chunk = source.read(_CHUNK_BYTES)
                        if not chunk:
                            break
                        hasher.update(chunk)
                        total += len(chunk)
            except OSError as exc:
                raise StoreRefusal(f"failed to read --state-file: {exc}") from exc
            state_sha256 = hasher.hexdigest()

            snapshot_id = _snapshot_id(identity["namespace"], session_id, generation)
            blob_rel = f"{_BLOBS_DIR}/{ns_shard}/{state_sha256}.bin"
            blob_abs = root.joinpath(*PurePosixPath(blob_rel).parts)
            now = int(time.time())

            row = conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
            if row is not None:
                stored_identity = json.loads(row["identity_json"])
                unchanged = (
                    row["state_sha256"] == state_sha256
                    and stored_identity == identity
                    and row["token_count"] == token_count
                    and row["token_sha256"] == args.token_sha256
                )
                if not unchanged:
                    raise StoreRefusal(
                        "generation collision: session "
                        f"{session_id!r} generation {generation} already stores "
                        "different content, identity or token metadata"
                    )
                created_at = row["created_at"]
            else:
                created_at = now

            # An existing valid blob (the record's own, or another record's
            # content-addressed blob with identical bytes) is reused as-is:
            # no payload temp file is created and zero payload bytes are
            # written. Otherwise stream-copy the input into a private
            # exclusive temp file, re-hashing the bytes actually written,
            # and verify that copy against the prehash (TOCTOU guard: a
            # --state-file that changed after the prehash pass must fail
            # before anything is published). Publish the immutable blob
            # before any index row can reference it.
            payload_bytes_written = 0
            if _blob_matches(blob_abs, state_sha256, total):
                pass  # already durable and byte-identical: nothing to write
            else:
                staging_fd, staging_name = tempfile.mkstemp(
                    dir=ns_dir, prefix=".staging-"
                )
                copy_hasher = hashlib.sha256()
                copy_total = 0
                try:
                    with os.fdopen(staging_fd, "wb") as staging, state_file.open(
                        "rb"
                    ) as source:
                        while True:
                            chunk = source.read(_CHUNK_BYTES)
                            if not chunk:
                                break
                            staging.write(chunk)
                            copy_hasher.update(chunk)
                            copy_total += len(chunk)
                        staging.flush()
                        os.fsync(staging.fileno())
                except OSError as exc:
                    _discard(staging_name)
                    raise StoreRefusal(f"failed to read --state-file: {exc}") from exc
                if copy_total != total or copy_hasher.hexdigest() != state_sha256:
                    _discard(staging_name)
                    raise StoreRefusal(
                        "--state-file changed while putting (prehash/stream "
                        "mismatch); nothing published, no receipt written"
                    )
                os.replace(staging_name, blob_abs)
                _fsync_dir(ns_dir)
                payload_bytes_written = copy_total
            os.chmod(blob_abs, 0o600)

            if row is None:
                try:
                    conn.execute(
                        "INSERT INTO snapshots (snapshot_id, namespace, session_id,"
                        " generation, state_sha256, serialized_bytes, token_count,"
                        " token_sha256, identity_json, blob_path, kind, created_at,"
                        " last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            snapshot_id,
                            identity["namespace"],
                            session_id,
                            generation,
                            state_sha256,
                            total,
                            token_count,
                            args.token_sha256,
                            json.dumps(
                                identity,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            ),
                            blob_rel,
                            KIND,
                            created_at,
                            now,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise StoreRefusal(
                        "generation collision: session "
                        f"{session_id!r} generation {generation} already exists"
                    ) from exc
                conn.commit()
            else:
                conn.execute(
                    "UPDATE snapshots SET last_accessed = ? WHERE snapshot_id = ?",
                    (now, snapshot_id),
                )
                conn.commit()

            record = {
                "schema_version": SCHEMA_VERSION,
                "snapshot_id": snapshot_id,
                "session_id": session_id,
                "generation": generation,
                "token_count": token_count,
                "token_sha256": args.token_sha256,
                "identity": identity,
                "state_sha256": state_sha256,
                "serialized_bytes": total,
                "created_at": created_at,
                "last_accessed": now,
                "blob_path": blob_rel,
                "kind": KIND,
                # Honest per-operation accounting: payload blob bytes this
                # put actually wrote (0 on blob reuse/dedup). Excludes
                # SQLite/index writes, filesystem metadata and device-level
                # write amplification; not persisted in the index, so
                # inspect output omits it.
                "io_metrics": {"payload_bytes_written": payload_bytes_written},
            }
        finally:
            conn.close()

    _write_json_atomic(output, record)
    return record


def _cmd_get(args: argparse.Namespace) -> None:
    root = Path(args.root)
    expected_identity = _load_identity(Path(args.expected_identity))
    dest = Path(args.output)

    _check_outside_git(root)
    active_dir = root / _ACTIVE_INDEX_DIR
    if not active_dir.is_dir():
        raise StoreRefusal(f"store root not found: {root}")

    with _store_lock(active_dir):
        conn = _open_index(active_dir)
        try:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (args.snapshot_id,)
            ).fetchone()
            if row is None:
                raise StoreRefusal(f"unknown snapshot id: {args.snapshot_id}")

            stored_identity = json.loads(row["identity_json"])
            if stored_identity != expected_identity:
                raise StoreRefusal(
                    "identity/namespace mismatch for snapshot "
                    f"{args.snapshot_id}; destination untouched"
                )

            blob_abs = _confined_blob_path(root, row["blob_path"])
            dest_dir = dest.parent
            if not dest_dir.is_dir():
                raise StoreRefusal(f"output directory does not exist: {dest_dir}")

            # Stream the blob to a private temp file beside the destination,
            # hashing and counting, then verify BEFORE publishing anything at
            # the destination path. Any refusal leaves an existing
            # destination exactly as it was.
            staging_fd, staging_name = tempfile.mkstemp(
                dir=dest_dir, prefix=".restore-"
            )
            try:
                with os.fdopen(staging_fd, "wb") as staging:
                    try:
                        blob_fd = os.open(
                            blob_abs, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                        )
                    except OSError as exc:
                        raise StoreRefusal(
                            "blob unreadable or not a regular file: "
                            f"{row['blob_path']}: {exc}"
                        ) from exc
                    hasher = hashlib.sha256()
                    total = 0
                    with os.fdopen(blob_fd, "rb") as source:
                        while True:
                            chunk = source.read(_CHUNK_BYTES)
                            if not chunk:
                                break
                            staging.write(chunk)
                            hasher.update(chunk)
                            total += len(chunk)
                    staging.flush()
                    os.fsync(staging.fileno())

                if (
                    total != row["serialized_bytes"]
                    or hasher.hexdigest() != row["state_sha256"]
                ):
                    raise StoreRefusal(
                        "blob integrity check failed (checksum/size mismatch); "
                        "destination untouched"
                    )

                os.replace(staging_name, dest)
                _fsync_dir(dest_dir)
            except BaseException:
                _discard(staging_name)
                raise

            conn.execute(
                "UPDATE snapshots SET last_accessed = ? WHERE snapshot_id = ?",
                (int(time.time()), args.snapshot_id),
            )
            conn.commit()
        finally:
            conn.close()


def _cmd_inspect(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    _check_outside_git(root)
    active_dir = root / _ACTIVE_INDEX_DIR
    if not active_dir.is_dir():
        raise StoreRefusal(f"store root not found: {root}")

    with _store_lock(active_dir):
        conn = _open_index(active_dir)
        try:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (args.snapshot_id,)
            ).fetchone()
            if row is None:
                raise StoreRefusal(f"unknown snapshot id: {args.snapshot_id}")
            return _record_from_row(row)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m mimo_halo.state_tier.store",
        description=(
            "Blocking opaque state store: durable put, verified get and "
            "metadata inspect for target-state snapshots."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    put = sub.add_parser("put", help="durable, atomic snapshot commit")
    put.add_argument("--root", required=True, help="explicit store root, outside Git")
    put.add_argument("--session", required=True, help="opaque logical-session id")
    put.add_argument("--generation", required=True, type=int)
    put.add_argument("--state-file", required=True)
    put.add_argument(
        "--identity", required=True, help="identity JSON, schema_version 1"
    )
    put.add_argument("--token-count", required=True, type=int)
    put.add_argument("--token-sha256", required=True)
    put.add_argument("--output", required=True, help="receipt JSON destination")

    get = sub.add_parser("get", help="verified restore to a destination file")
    get.add_argument("--root", required=True)
    get.add_argument("--snapshot-id", required=True)
    get.add_argument("--expected-identity", required=True)
    get.add_argument("--output", required=True)

    inspect = sub.add_parser("inspect", help="print the canonical record as JSON")
    inspect.add_argument("--root", required=True)
    inspect.add_argument("--snapshot-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "put":
            record = _cmd_put(args)
            print(json.dumps(record, indent=2, ensure_ascii=True))
        elif args.command == "get":
            _cmd_get(args)
        else:
            record = _cmd_inspect(args)
            print(json.dumps(record, indent=2, ensure_ascii=True))
    except StoreRefusal as exc:
        print(f"store: {exc}", file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error) as exc:
        print(f"store: I/O failure: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
