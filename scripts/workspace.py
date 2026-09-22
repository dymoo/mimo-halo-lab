#!/usr/bin/env python3
"""Workspace provisioning for the MiMo Halo lab (stdlib only, Python >= 3.11).

The lab workspace root must be supplied explicitly via the MIMO_LAB environment
variable; the tool never guesses. It creates the directory layout only:

- mkdir only; no formatting, no renames, no writes outside the workspace
  root. Only `init` creates directories (including the root itself);
  `status` is strictly read-only and never creates an absent root.
- Every workspace path is checked component-wise first: a symlink anywhere
  under the root fails the operation closed, so creates and writes can never
  escape the workspace through a symlinked directory or file.
- Free-space accounting uses the filesystem the workspace lives on; when free
  bytes fall below the model-staging floor, model directories are NOT created
  and staging is reported as blocked. Metadata directories are always created.
- A hash round-trip smoke exclusively creates (O_CREAT|O_EXCL) a uniquely
  named probe file under `caches/`, hashes it, reads it back and verifies the
  digest, then removes only the probe it created. Pre-existing files —
  including anything at the old fixed probe name, symlink or not — are never
  opened, followed or overwritten.

Output written into the workspace records facts only (free bytes, counts,
statuses); it never records absolute paths. Public stdout summarizes without
paths; operators run it locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

GIB = 2**30
LAYOUT = {
    "source-models": [],
    "pruned-models": [],
    "recovered-models": [],
    "quantized-models": [],
    "traces": ["discovered", "normalized", "private"],
    "datasets": [],
    "calibration": [],
    "teacher-cache": [],
    "eval-workspaces": [],
    "manifests": [],
    "metrics": [],
    "scratch": [],
    "caches": [],
}
MODEL_DIRS = {"source-models", "pruned-models", "recovered-models", "quantized-models"}
PROBE_DIR = "caches"
PROBE_PREFIX = ".workspace-hash-probe-"
DEFAULT_MIN_MODEL_FREE_BYTES = 120 * GIB


def _fail(message: str) -> None:
    print(f"workspace: {message}", file=sys.stderr)
    raise SystemExit(2)


def _checked_join(root: str, rel: str) -> str:
    """Join root/rel, failing closed if any existing component is a symlink."""
    path = root
    parts = [p for p in rel.split(os.sep) if p]
    for i, part in enumerate(parts):
        path = os.path.join(path, part)
        if os.path.islink(path):
            _fail(f"refusing to proceed: {'/'.join(parts[:i + 1])} is a symlink")
    return path


def _ensure_root(root: str, *, create: bool) -> None:
    """Reject a symlinked/non-dir root; create an absent root only for init."""
    if os.path.islink(root):
        _fail("MIMO_LAB names a symlink; refusing to use it as the workspace root")
    if os.path.isdir(root):
        return
    if os.path.lexists(root):
        _fail("MIMO_LAB exists and is not a directory")
    if not create:
        _fail("workspace root does not exist; only init creates it")
    try:
        os.makedirs(root)
    except OSError as exc:
        _fail(f"cannot create workspace root: {exc}")


def free_bytes(root: str) -> int:
    """Free bytes available to unprivileged users on the filesystem holding root."""
    try:
        st = os.statvfs(root)
    except (AttributeError, OSError) as exc:  # pragma: no cover - platform dependent
        _fail(f"cannot stat filesystem for {root!r}: {exc}")
    return st.f_bavail * st.f_frsize


def probe_roundtrip(root: str) -> dict:
    """Exclusive unique probe: write, hash, read back, verify; own file only."""
    caches = _checked_join(root, PROBE_DIR)
    try:
        os.makedirs(caches, exist_ok=True)
    except OSError as exc:
        _fail(f"cannot create {PROBE_DIR}: {exc}")
    if os.path.islink(caches):
        _fail(f"refusing to proceed: {PROBE_DIR} is a symlink")
    name = f"{PROBE_PREFIX}{os.getpid()}-{os.urandom(6).hex()}.tmp"
    path = os.path.join(caches, name)
    data = os.urandom(4096)
    expected = hashlib.sha256(data).hexdigest()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    except OSError as exc:
        _fail(f"cannot exclusively create workspace probe: {exc}")
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 8192)
            if not chunk:
                break
            chunks.append(chunk)
        got = hashlib.sha256(b"".join(chunks)).hexdigest()
    finally:
        os.close(fd)
        try:
            os.unlink(path)  # only the probe this call exclusively created
        except FileNotFoundError:
            pass
    if got != expected:
        _fail(f"hash round-trip mismatch on workspace probe ({expected} != {got})")
    return {"probe": f"{PROBE_DIR}/{name}", "sha256": expected, "bytes": len(data), "status": "verified"}


def plan_paths(min_model_free_bytes: int, root_bytes_free: int) -> tuple[list[str], list[str]]:
    """Split the layout into creatable and blocked directories."""
    staging_blocked = root_bytes_free < min_model_free_bytes
    ok, blocked = [], []
    for parent, children in LAYOUT.items():
        rels = [parent] if not children else [os.path.join(parent, c) for c in children]
        for rel in rels:
            if staging_blocked and parent in MODEL_DIRS:
                blocked.append(rel)
            else:
                ok.append(rel)
    return ok, blocked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-model-free-bytes", type=int, default=DEFAULT_MIN_MODEL_FREE_BYTES,
                    help="free-byte floor below which model staging is blocked (default 120 GiB)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in (("init", "create the layout (mkdir only) and run the hash round-trip smoke"),
                           ("status", "report layout completeness, free bytes and staging gate"),
                           ("smoke", "run the hash round-trip smoke only")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--min-model-free-bytes", type=int, default=None, dest="cmd_min_model_free_bytes",
                        help="overrides the top-level floor when given after the subcommand")

    args = ap.parse_args()
    root = os.environ.get("MIMO_LAB")
    if not root:
        _fail("MIMO_LAB environment variable is required and must name the workspace root")
    floor = args.cmd_min_model_free_bytes if args.cmd_min_model_free_bytes is not None \
        else args.min_model_free_bytes

    print(f"workspace root: {root}", file=sys.stderr)  # operator-only line; never written to files
    _ensure_root(root, create=args.cmd == "init")
    free = free_bytes(root)

    staging_blocked = free < floor
    created, blocked, existed = [], [], []
    if args.cmd == "init":
        ok_paths, blocked_paths = plan_paths(floor, free)
        for rel in ok_paths:
            path = _checked_join(root, rel)  # any symlink component fails closed
            if os.path.isfile(path):
                _fail(f"refusing to proceed: {rel} exists as a regular file")
            if os.path.isdir(path):
                existed.append(rel)
            else:
                try:
                    os.makedirs(path)
                except OSError as exc:
                    _fail(f"cannot create {rel}: {exc}")
                created.append(rel)
        if blocked_paths:
            print(
                "workspace: MODEL STAGING BLOCKED: "
                f"{free} free bytes < {floor} required; "
                f"{len(blocked_paths)} model directories not created; metadata unaffected",
                file=sys.stderr,
            )
    else:
        ok_paths, blocked_paths = [], []
        for parent, children in LAYOUT.items():
            base = [parent] if not children else [os.path.join(parent, c) for c in children]
            for rel in base:
                (existed if os.path.isdir(_checked_join(root, rel)) else blocked).append(rel)

    smoke = None
    if args.cmd in ("init", "smoke"):
        smoke = probe_roundtrip(root)

    record = {
        "schema_version": 1,
        "command": args.cmd,
        "root_recorded": False,
        "filesystem_free_bytes": free,
        "filesystem_free_gib": round(free / GIB, 3),
        "min_model_free_bytes": floor,
        "model_staging": "blocked" if staging_blocked else "allowed",
        "dirs_created": sorted(created),
        "dirs_existing": sorted(existed),
        "dirs_blocked": sorted(blocked_paths),
        "hash_smoke": smoke,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.cmd == "init":
        manifests_dir = _checked_join(root, "manifests")
        os.makedirs(manifests_dir, exist_ok=True)
        record_path = os.path.join(manifests_dir, "workspace.json")
        if os.path.islink(record_path):
            _fail("refusing to overwrite symlinked workspace record")
        with open(record_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, sort_keys=True, indent=2)
            fh.write("\n")
    # Public summary: no paths, no identifiers.
    public = {k: record[k] for k in ("schema_version", "command", "filesystem_free_bytes", "filesystem_free_gib",
                                     "min_model_free_bytes", "model_staging")}
    if smoke:
        public["hash_smoke"] = {"probe": smoke["probe"], "status": smoke["status"]}
    public["dirs_created"] = len(created)
    public["dirs_existing"] = len(existed)
    public["dirs_blocked"] = len(blocked_paths)
    print(json.dumps(public, sort_keys=True, indent=2))
    if args.cmd == "init" and blocked_paths:
        raise SystemExit(1)  # explicit nonzero: staging was blocked


if __name__ == "__main__":
    main()
