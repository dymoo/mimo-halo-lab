"""Artifact sidecar creation and verification for mimo-halo-lab.

Every large artifact lives in its own directory (the *artifact root*) and
carries seven sidecars plus its payload files:

    manifest.json  checksums.txt  commands.txt  source_commits.json
    dataset_manifest.json  metrics.json  environment.json

Everything is caller-supplied: this module never scans the private
filesystem, never collects the process environment, and never invents
provenance.  ``create`` preflights every explicit input before writing
anything, refuses unexplained existing root content, writes sidecars
atomically with ``manifest.json`` last, and fails closed when a
stage-required input is missing.  ``verify`` recomputes every digest
and rejects tampered payloads, sidecars, manifests, malformed parent
digests, non-immutable source revisions, unexplained files, and path
escapes.

Determinism: manifests carry ``schema_version`` and a canonical
``manifest_sha256`` computed over the manifest JSON with that field
excluded (canonical form: sorted keys, compact separators, UTF-8).
Identical inputs produce byte-identical sidecars in any artifact root.

CLI::

    PYTHONPATH=src python -m mimo_halo.artifacts create --root DIR --kind KIND ...
    PYTHONPATH=src python -m mimo_halo.artifacts verify --root DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

SCHEMA_VERSION = 1

ARTIFACT_KINDS = (
    "dataset",
    "calibration",
    "pruned",
    "recovered",
    "quantized",
    "evaluation",
)

MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "checksums.txt"
COMMANDS_NAME = "commands.txt"
SOURCE_COMMITS_NAME = "source_commits.json"
DATASET_MANIFEST_NAME = "dataset_manifest.json"
METRICS_NAME = "metrics.json"
ENVIRONMENT_NAME = "environment.json"

#: Sidecars whose content is derived from the manifest itself.
RESERVED_NAMES = frozenset({MANIFEST_NAME, CHECKSUMS_NAME})
#: Sidecars written by ``create`` from explicit CLI flags.
TOOL_SIDECARS = frozenset({COMMANDS_NAME, SOURCE_COMMITS_NAME})
#: Sidecars copied verbatim from caller-supplied files.
CALLER_SIDECARS = frozenset({DATASET_MANIFEST_NAME, METRICS_NAME, ENVIRONMENT_NAME})
ALL_SIDECARS = RESERVED_NAMES | TOOL_SIDECARS | CALLER_SIDECARS

#: Structured quant/calibration inputs, copied into the artifact root under
#: fixed names and referenced from ``manifest.inputs``.
INPUT_FILENAMES = {
    "calibration_config": "calibration_config.json",
    "expert_map": "expert_map.json",
    "quant_assignment": "quant_assignment.json",
}

#: Parent roles with an expected parent artifact kind. ``None`` means the
#: role accepts any kind.
PARENT_ROLE_KINDS = {
    "dataset": "dataset",
    "pruned": "pruned",
    "recovered": "recovered",
    "calibration-post-recovery": "calibration",
}

#: Per-stage fail-closed requirements. Unquantized stages never require
#: quant fields; ``quantized`` requires full source/recovery lineage plus a
#: post-recovery calibration parent digest.
STAGE_REQUIREMENTS = {
    "dataset": {
        "parents": frozenset(),
        "inputs": frozenset(),
        "seed": False,
        "metrics": False,
    },
    "calibration": {
        "parents": frozenset(),
        "inputs": frozenset({"calibration_config"}),
        "seed": True,
        "metrics": False,
    },
    "pruned": {
        "parents": frozenset(),
        "inputs": frozenset({"expert_map"}),
        "seed": True,
        "metrics": False,
    },
    "recovered": {
        "parents": frozenset({"pruned"}),
        "inputs": frozenset(),
        "seed": True,
        "metrics": False,
    },
    "quantized": {
        "parents": frozenset({"pruned", "recovered", "calibration-post-recovery"}),
        "inputs": frozenset({"quant_assignment"}),
        "seed": False,
        "metrics": False,
    },
    "evaluation": {
        "parents": frozenset({"candidate"}),
        "inputs": frozenset(),
        "seed": False,
        "metrics": True,
    },
}

PRODUCTION_KINDS = frozenset({"pruned", "recovered", "quantized"})

#: Production allowance is recorded exactly as declared by the caller; no
#: signature verification is performed and none is claimed.
PRODUCTION_POLICY = (
    "Production allowance is declaration-based: the official Xiaomi source "
    "and signed-source declarations are recorded as caller-supplied "
    "provenance. No cryptographic signature verification is performed and "
    "none is claimed; promotion additionally requires separately declared "
    "signed sources."
)

_SECRET_KEY_MARKERS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "private_key",
    "authorization",
    "credential",
)
_READ_CHUNK = 1 << 20


class ArtifactError(Exception):
    """Raised for any creation or verification failure. Fails closed."""


# ---------------------------------------------------------------------------
# Digest and path primitives
# ---------------------------------------------------------------------------


def sha256_file(path):
    """Stream a file through SHA-256; never loads whole files."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(obj):
    """Canonical JSON: sorted keys, compact separators, UTF-8, ASCII-safe."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_manifest_digest(manifest):
    """Digest of the manifest with its self-reference excluded."""
    trimmed = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(trimmed)).hexdigest()


def _safe_relative_path(path):
    """Validate a relative POSIX path inside an artifact root; return it."""
    if not isinstance(path, str):
        raise ArtifactError(f"path must be a string: {path!r}")
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        raise ArtifactError(f"path is not relative POSIX inside the artifact root: {path!r}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ArtifactError(f"path must not traverse outside the artifact root: {path!r}")
    return path


def _check_inside_root(root_real, candidate_real, label):
    if candidate_real != root_real and not candidate_real.startswith(root_real + os.sep):
        raise ArtifactError(f"{label} escapes the artifact root: {candidate_real}")


def scan_artifact_files(root):
    """List payload/sidecar files under ``root`` deterministically.

    Returns sorted ``(relative_posix_path, absolute_path)`` pairs. Rejects
    symlinks, non-regular files, and any path that escapes the root.
    """
    root_real = os.path.realpath(root)
    found = []

    def walk(dir_path, rel_prefix):
        with os.scandir(dir_path) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            rel = rel_prefix + entry.name
            if entry.is_symlink():
                raise ArtifactError(f"symlink is not allowed in an artifact: {rel}")
            entry_real = os.path.realpath(entry.path)
            _check_inside_root(root_real, entry_real, "path")
            if entry.is_dir(follow_symlinks=False):
                walk(entry.path, rel + "/")
            elif entry.is_file(follow_symlinks=False):
                found.append((rel, entry.path))
            else:
                raise ArtifactError(f"only regular files are allowed in an artifact: {rel}")

    walk(root, "")
    return found


def _is_hex_of_length(value, length):
    return isinstance(value, str) and len(value) == length and all(
        c in "0123456789abcdef" for c in value
    )


def _is_sha256_hex(value):
    return _is_hex_of_length(value, 64)


# ---------------------------------------------------------------------------
# Caller input validation
# ---------------------------------------------------------------------------


def _require_nonempty_str(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ArtifactError(f"{label} must be a non-empty, trimmed string")
    return value


def _validate_revision(revision, label):
    """Accept only immutable full revisions.

    A source revision must be a full 40-hex git/HF commit SHA or a
    64-hex content digest. Branch names, tags, abbreviated SHAs, and any
    other moving or ambiguous ref fail closed; there is no blocklist to
    bypass.
    """
    if not _is_hex_of_length(revision, 40) and not _is_sha256_hex(revision):
        raise ArtifactError(
            f"{label} must be an immutable full revision (40-hex git/HF commit SHA "
            f"or 64-hex content digest); moving refs and abbreviated SHAs are "
            f"rejected: {revision!r}"
        )
    return revision


def _validate_commit_sha(commit, label):
    """Conversion/runtime commits are full 40-hex commit SHAs or null."""
    if commit is None:
        return None
    if not _is_hex_of_length(commit, 40):
        raise ArtifactError(f"{label} commit must be a full 40-hex commit SHA: {commit!r}")
    return commit


def _require_regular_file(path, label):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ArtifactError(f"{label} must be a regular file: {path}")
    return path


def parse_source_spec(spec):
    """Parse ``name=url@FULL_SHA`` (immutable full revision) into a source record."""
    _require_nonempty_str(spec, "source spec")
    name, sep, rest = spec.partition("=")
    if not sep:
        raise ArtifactError(f"source must be name=url@FULL_SHA: {spec!r}")
    _require_nonempty_str(name, "source name")
    url, sep, revision = rest.rpartition("@")
    if not sep:
        raise ArtifactError(f"source must be name=url@FULL_SHA: {spec!r}")
    _require_nonempty_str(url, f"source url for {name!r}")
    _require_nonempty_str(revision, f"source revision for {name!r}")
    if not url.startswith("https://"):
        raise ArtifactError(f"source url must be https: {url!r}")
    _validate_revision(revision, "source revision")
    return {"name": name, "url": url, "revision": revision, "declared_signed": False}


def parse_parent_spec(spec):
    """Parse ``role=kind:sha256`` (kind optional: ``role=sha256``)."""
    _require_nonempty_str(spec, "parent spec")
    role, sep, rest = spec.partition("=")
    if not sep:
        raise ArtifactError(f"parent must be role=kind:sha256 or role=sha256: {spec!r}")
    _require_nonempty_str(role, "parent role")
    kind, sep, digest = rest.partition(":")
    if sep:
        if kind not in ARTIFACT_KINDS:
            raise ArtifactError(f"parent kind must be one of {sorted(ARTIFACT_KINDS)}: {kind!r}")
    else:
        digest, kind = rest, None
    if not _is_sha256_hex(digest):
        raise ArtifactError(f"parent digest must be 64 lowercase hex chars: {rest!r}")
    expected = PARENT_ROLE_KINDS.get(role)
    if expected is not None and kind is not None and kind != expected:
        raise ArtifactError(
            f"parent role {role!r} requires kind {expected!r}, got {kind!r}"
        )
    parent = {"role": role, "sha256": digest}
    if kind is not None:
        parent["kind"] = kind
    return parent


def _validate_parents(parents):
    seen = {}
    for parent in parents:
        role = parent["role"]
        if role in seen:
            raise ArtifactError(f"duplicate parent role: {role!r}")
        seen[role] = parent
        kind = parent.get("kind")
        expected = PARENT_ROLE_KINDS.get(role)
        if expected is not None and kind is not None and kind != expected:
            raise ArtifactError(
                f"parent role {role!r} requires kind {expected!r}, got {kind!r}"
            )
    return seen


def _check_stage_requirements(kind, parent_roles, input_roles, seed, has_metrics):
    req = STAGE_REQUIREMENTS[kind]
    missing_parents = sorted(req["parents"] - parent_roles)
    if missing_parents:
        raise ArtifactError(
            f"stage {kind!r} requires parent artifacts with roles {missing_parents}"
        )
    missing_inputs = sorted(req["inputs"] - input_roles)
    if missing_inputs:
        raise ArtifactError(
            f"stage {kind!r} requires structured inputs {missing_inputs}"
        )
    if req["seed"] and seed is None:
        raise ArtifactError(f"stage {kind!r} requires an explicit --seed")
    if req["metrics"] and not has_metrics:
        raise ArtifactError(f"stage {kind!r} requires a metrics.json sidecar")


def _reject_secret_keys(obj, where):
    """Fail closed if sanitized input still looks like a secret carrier."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in _SECRET_KEY_MARKERS):
                raise ArtifactError(
                    f"{where}: key {key!r} looks like a secret; environment "
                    "sidecars must be sanitized before creation"
                )
            _reject_secret_keys(value, where)
    elif isinstance(obj, list):
        for item in obj:
            _reject_secret_keys(item, where)


# ---------------------------------------------------------------------------
# Sidecar writers
# ---------------------------------------------------------------------------


def _remove_quietly(path):
    """Remove one of our own temp files; never touches existing content."""
    try:
        os.remove(path)
    except OSError:
        pass


def _json_sidecar_bytes(obj):
    """Exact on-disk bytes for a JSON sidecar (deterministic across roots)."""
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_bytes(path, data):
    """Write ``data`` via temp file + rename so no reader ever sees a partial file."""
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp",
        dir=os.path.dirname(path) or ".",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        _remove_quietly(tmp_path)
        raise


def _copy_into_root(root, src, dest_name):
    """Stream ``src`` into the root under ``dest_name`` atomically.

    A source that already *is* the destination file (a declared payload or
    input already in place) is adopted without copying, so an existing
    large payload is never re-read or clobbered.
    """
    dest = os.path.join(root, dest_name)
    try:
        already_in_place = (
            os.path.isfile(dest)
            and not os.path.islink(dest)
            and os.path.samefile(src, dest)
        )
    except OSError:
        already_in_place = False
    if already_in_place:
        return dest
    fd, tmp_path = tempfile.mkstemp(prefix=f".{dest_name}.", suffix=".tmp", dir=root)
    try:
        with open(src, "rb") as reader, os.fdopen(fd, "wb") as writer:
            while True:
                chunk = reader.read(_READ_CHUNK)
                if not chunk:
                    break
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(tmp_path, dest)
    except BaseException as exc:
        _remove_quietly(tmp_path)
        if isinstance(exc, OSError):
            raise ArtifactError(f"cannot copy {src} into the artifact root: {exc}") from exc
        raise
    return dest


def _write_json_sidecar(root, name, obj):
    path = os.path.join(root, name)
    _atomic_write_bytes(path, _json_sidecar_bytes(obj))
    return path


def _write_commands(root, commands):
    """Write ``commands.txt`` (commands are preflighted by the caller)."""
    path = os.path.join(root, COMMANDS_NAME)
    _atomic_write_bytes(path, ("\n".join(commands) + "\n").encode("utf-8"))
    return path


def _validate_source_commits(obj):
    """Structural and immutability checks shared by ``create`` and ``verify``."""
    if not isinstance(obj, dict):
        raise ArtifactError("source_commits.json must be a JSON object")
    sources = obj.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ArtifactError("source_commits.json must list at least one source")
    seen = set()
    for record in sources:
        if not isinstance(record, dict) or set(record) != {
            "name", "url", "revision", "declared_signed"
        }:
            raise ArtifactError(
                "source records must have exactly name, url, revision, declared_signed"
            )
        name = _require_nonempty_str(record.get("name"), "source name")
        if name in seen:
            raise ArtifactError(f"duplicate source name: {name!r}")
        seen.add(name)
        url = _require_nonempty_str(record.get("url"), f"source url for {name!r}")
        if not url.startswith("https://"):
            raise ArtifactError(f"source url must be https: {url!r}")
        _validate_revision(record.get("revision"), f"source revision for {name!r}")
        if not isinstance(record.get("declared_signed"), bool):
            raise ArtifactError(f"declared_signed must be a boolean for source {name!r}")
    _validate_commit_sha(obj.get("conversion_commit"), "conversion")
    _validate_commit_sha(obj.get("runtime_commit"), "runtime")


def _source_commits_object(sources, signed_sources, conversion_commit, runtime_commit):
    """Validate caller sources/commits and build the ``source_commits.json`` object."""
    if not sources:
        raise ArtifactError("at least one --source name=url@FULL_SHA is required")
    records = []
    for source in sources:
        if not isinstance(source, dict):
            raise ArtifactError(f"source record must be an object: {source!r}")
        records.append(dict(source))
    obj = {
        "schema_version": SCHEMA_VERSION,
        "sources": records,
        "conversion_commit": conversion_commit,
        "runtime_commit": runtime_commit,
    }
    _validate_source_commits(obj)
    signed = {_require_nonempty_str(name, "signed source name") for name in signed_sources}
    unknown = sorted(signed - {record["name"] for record in records})
    if unknown:
        raise ArtifactError(f"--signed-source names not among sources: {unknown}")
    for record in records:
        record["declared_signed"] = record["name"] in signed
    obj["sources"] = sorted(records, key=lambda record: record["name"])
    return obj


def _read_environment_file(path):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ArtifactError(f"environment input must be a regular file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        try:
            obj = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"environment file must be valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ArtifactError("environment file must be a JSON object (sanitized map)")
    _reject_secret_keys(obj, ENVIRONMENT_NAME)
    return obj


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def create_artifact(
    root,
    kind,
    *,
    sources,
    commands,
    environment_path,
    dataset_manifest_path=None,
    dataset_hashes=(),
    metrics_path=None,
    expert_map_path=None,
    calibration_config_path=None,
    quant_assignment_path=None,
    payload_paths=(),
    parents=(),
    seed=None,
    conversion_commit=None,
    runtime_commit=None,
    signed_sources=(),
    production=False,
    official_source=None,
    created_at=None,
):
    """Create the full sidecar set in ``root`` from explicit caller inputs.

    Returns the manifest dict. Every explicit input (stage requirements,
    sources and immutable revisions, commands, dataset provenance,
    environment, structured inputs, payload names and collisions) is
    validated in a write-free preflight, so a refused creation leaves no
    files behind. Existing root content is refused unless it is a
    declared file already at its destination, which is adopted in place.
    Sidecars are written atomically and ``manifest.json`` last. Fails
    closed on missing stage-required data, symlinks, path escapes, and
    duplicate/forbidden names.
    """
    # --- Preflight: validate every explicit input before any write. ---
    if kind not in ARTIFACT_KINDS:
        raise ArtifactError(f"kind must be one of {sorted(ARTIFACT_KINDS)}: {kind!r}")

    if os.path.islink(root):
        raise ArtifactError(f"artifact root must not be a symlink: {root}")
    if os.path.exists(root) and not os.path.isdir(root):
        raise ArtifactError(f"artifact root is not a directory: {root}")
    if os.path.exists(os.path.join(root, MANIFEST_NAME)):
        raise ArtifactError(f"refusing to overwrite an existing artifact: {MANIFEST_NAME} present")

    parent_map = _validate_parents(parents)
    input_sources = {
        "calibration_config": calibration_config_path,
        "expert_map": expert_map_path,
        "quant_assignment": quant_assignment_path,
    }
    input_paths = {}
    for role, src in input_sources.items():
        if src is not None:
            _require_regular_file(src, f"input file for role {role!r}")
            input_paths[role] = INPUT_FILENAMES[role]
    _check_stage_requirements(kind, set(parent_map), set(input_paths), seed, metrics_path is not None)

    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ArtifactError("seed must be an integer or null")
    if created_at is not None:
        _require_nonempty_str(created_at, "created_at")

    if production:
        if kind not in PRODUCTION_KINDS:
            raise ArtifactError(f"--production is not valid for stage {kind!r}")
        if official_source is None:
            raise ArtifactError(
                "--production requires --official-source URL@FULL_SHA declaring the "
                "official Xiaomi source"
            )
        official = parse_source_spec(f"official_source={official_source}")
        production_block = {
            "requested": True,
            "official_source": {
                "url": official["url"],
                "revision": official["revision"],
            },
            "policy": PRODUCTION_POLICY,
        }
    else:
        production_block = {"requested": False, "official_source": None, "policy": PRODUCTION_POLICY}

    source_commits_obj = _source_commits_object(
        list(sources), list(signed_sources), conversion_commit, runtime_commit
    )

    if not commands:
        raise ArtifactError("at least one --command is required for reproducibility")
    cleaned_commands = [_require_nonempty_str(command, "command") for command in commands]

    if dataset_manifest_path is not None and dataset_hashes:
        raise ArtifactError("pass either --dataset-manifest or --dataset, not both")
    if dataset_manifest_path is not None:
        _require_regular_file(dataset_manifest_path, "dataset manifest")
        datasets = None
    else:
        if not dataset_hashes:
            raise ArtifactError(
                "dataset provenance is required: pass --dataset-manifest PATH or "
                "--dataset NAME=SHA256"
            )
        datasets = []
        seen_datasets = set()
        for spec in dataset_hashes:
            name, sep, digest = spec.partition("=")
            if not sep:
                raise ArtifactError(f"dataset must be NAME=SHA256: {spec!r}")
            _require_nonempty_str(name, "dataset name")
            if name in seen_datasets:
                raise ArtifactError(f"duplicate dataset name: {name!r}")
            seen_datasets.add(name)
            if not _is_sha256_hex(digest):
                raise ArtifactError(f"dataset hash must be 64 lowercase hex chars: {digest!r}")
            datasets.append({"name": name, "sha256": digest})
        datasets.sort(key=lambda entry: entry["name"])

    if environment_path is None:
        raise ArtifactError("environment input is required: pass --environment PATH")
    _read_environment_file(environment_path)
    if metrics_path is not None:
        _require_regular_file(metrics_path, "metrics")

    # Explicit payload files, validated under their destination basename.
    payload_copies = []
    payload_names = set()
    for payload in payload_paths:
        _require_regular_file(payload, "payload")
        name = os.path.basename(payload)
        _safe_relative_path(name)
        if name in ALL_SIDECARS or name in INPUT_FILENAMES.values():
            raise ArtifactError(f"payload name collides with a reserved sidecar: {name!r}")
        if name in payload_names:
            raise ArtifactError(f"duplicate payload name: {name!r}")
        payload_names.add(name)
        payload_copies.append((payload, name))

    # --- Refuse unexplained existing root content; adopt declared files in place. ---
    planned_copies = list(payload_copies)
    if dataset_manifest_path is not None:
        planned_copies.append((dataset_manifest_path, DATASET_MANIFEST_NAME))
    if metrics_path is not None:
        planned_copies.append((metrics_path, METRICS_NAME))
    planned_copies.append((environment_path, ENVIRONMENT_NAME))
    for role, src in input_sources.items():
        if src is not None:
            planned_copies.append((src, INPUT_FILENAMES[role]))
    if os.path.isdir(root):
        existing = {rel for rel, _ in scan_artifact_files(root)}
        adopted = set()
        for src, dest_name in planned_copies:
            dest = os.path.join(root, dest_name)
            try:
                in_place = (
                    os.path.isfile(dest)
                    and not os.path.islink(dest)
                    and os.path.samefile(src, dest)
                )
            except OSError:
                in_place = False
            if in_place:
                adopted.add(dest_name)
        unexplained = sorted(existing - adopted)
        if unexplained:
            raise ArtifactError(
                "refusing unexplained existing content in the artifact root (declare "
                "each file as a payload already at its destination, or start from a "
                f"clean root): {unexplained}"
            )
    os.makedirs(root, exist_ok=True)

    # --- Write phase: sidecars atomically, manifest.json last. ---
    if dataset_manifest_path is not None:
        _copy_into_root(root, dataset_manifest_path, DATASET_MANIFEST_NAME)
    else:
        _write_json_sidecar(
            root, DATASET_MANIFEST_NAME, {"schema_version": SCHEMA_VERSION, "datasets": datasets}
        )
    if metrics_path is not None:
        _copy_into_root(root, metrics_path, METRICS_NAME)
    else:
        _write_json_sidecar(root, METRICS_NAME, {"schema_version": SCHEMA_VERSION, "metrics": {}})
    _copy_into_root(root, environment_path, ENVIRONMENT_NAME)
    for role, src in input_sources.items():
        if src is not None:
            _copy_into_root(root, src, INPUT_FILENAMES[role])
    _write_commands(root, cleaned_commands)
    _write_json_sidecar(root, SOURCE_COMMITS_NAME, source_commits_obj)
    for payload, name in payload_copies:
        _copy_into_root(root, payload, name)

    # Digest everything except the two manifest-derived sidecars.
    files = []
    for rel, abs_path in scan_artifact_files(root):
        if rel in RESERVED_NAMES:
            continue
        files.append(
            {
                "path": rel,
                "sha256": sha256_file(abs_path),
                "size_bytes": os.path.getsize(abs_path),
            }
        )
    files.sort(key=lambda entry: entry["path"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "seed": seed,
        "parents": [parent_map[role] for role in sorted(parent_map)],
        "production": production_block,
        "inputs": {role: input_paths[role] for role in sorted(input_paths)},
        "files": files,
    }
    if created_at is not None:
        manifest["created_at"] = created_at
    manifest["manifest_sha256"] = canonical_manifest_digest(manifest)

    # checksums.txt explains manifest.json, so it is computed from the
    # exact manifest bytes and written first; manifest.json lands last and
    # is the commit point of the artifact.
    manifest_bytes = _json_sidecar_bytes(manifest)
    _write_checksums(root, files, hashlib.sha256(manifest_bytes).hexdigest())
    _atomic_write_bytes(os.path.join(root, MANIFEST_NAME), manifest_bytes)
    return manifest


def _checksums_text(file_entries, manifest_sha256):
    lines = []
    for entry in file_entries:
        lines.append(f"{entry['sha256']}  {entry['path']}")
    lines.append(f"{manifest_sha256}  {MANIFEST_NAME}")
    lines.sort()
    return "\n".join(lines) + "\n"


def _write_checksums(root, file_entries, manifest_sha256):
    path = os.path.join(root, CHECKSUMS_NAME)
    _atomic_write_bytes(path, _checksums_text(file_entries, manifest_sha256).encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_artifact(root):
    """Recompute every digest and structural rule; return a report dict.

    Catches modified payloads or sidecars, a modified or malformed
    manifest, malformed parent digests, non-immutable source revisions
    in ``source_commits.json`` and the declared official source,
    unexplained files, secret-bearing environment sidecars, stage
    precondition violations, and path escapes.
    """
    if os.path.islink(root) or not os.path.isdir(root):
        raise ArtifactError(f"artifact root must be an existing directory: {root}")

    manifest_path = os.path.join(root, MANIFEST_NAME)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError as exc:
        raise ArtifactError("missing manifest.json") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactError("manifest.json must be a JSON object")

    recorded = manifest.get("manifest_sha256")
    if not _is_sha256_hex(recorded):
        raise ArtifactError("manifest_sha256 is missing or not 64 lowercase hex chars")
    recomputed = canonical_manifest_digest(manifest)
    if recomputed != recorded:
        raise ArtifactError(
            "canonical manifest digest mismatch: manifest.json was modified or is malformed"
        )

    kind = manifest.get("kind")
    if kind not in ARTIFACT_KINDS:
        raise ArtifactError(f"manifest kind is not a known stage: {kind!r}")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ArtifactError("manifest.files must be a non-empty list")
    listed = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ArtifactError("manifest.files entries must be objects")
        path = _safe_relative_path(entry.get("path"))
        if path in RESERVED_NAMES:
            raise ArtifactError(f"manifest.files must not list {path}")
        if path in listed:
            raise ArtifactError(f"duplicate path in manifest.files: {path}")
        if not _is_sha256_hex(entry.get("sha256")):
            raise ArtifactError(f"invalid sha256 for {path}")
        size = entry.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArtifactError(f"size_bytes must be a non-negative integer for {path}")
        listed[path] = entry
    for required in (COMMANDS_NAME, SOURCE_COMMITS_NAME, DATASET_MANIFEST_NAME, ENVIRONMENT_NAME):
        if required not in listed:
            raise ArtifactError(f"manifest.files is missing required sidecar {required}")

    parents = manifest.get("parents")
    if not isinstance(parents, list):
        raise ArtifactError("manifest.parents must be a list")
    parent_roles = set()
    for parent in parents:
        if not isinstance(parent, dict):
            raise ArtifactError("manifest.parents entries must be objects")
        role = _require_nonempty_str(parent.get("role"), "parent role")
        if role in parent_roles:
            raise ArtifactError(f"duplicate parent role: {role!r}")
        parent_roles.add(role)
        digest = parent.get("sha256")
        if not _is_sha256_hex(digest):
            raise ArtifactError(f"malformed parent digest for role {role!r}")
        parent_kind = parent.get("kind")
        if parent_kind is not None:
            if parent_kind not in ARTIFACT_KINDS:
                raise ArtifactError(f"malformed parent kind for role {role!r}: {parent_kind!r}")
            expected = PARENT_ROLE_KINDS.get(role)
            if expected is not None and parent_kind != expected:
                raise ArtifactError(
                    f"parent role {role!r} requires kind {expected!r}, got {parent_kind!r}"
                )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ArtifactError("manifest.inputs must be an object")
    input_files = set()
    for role, path in inputs.items():
        if role not in INPUT_FILENAMES or INPUT_FILENAMES[role] != path:
            raise ArtifactError(f"manifest.inputs has unknown or mismatched role {role!r} -> {path!r}")
        if path not in listed:
            raise ArtifactError(f"manifest.inputs references missing file {path!r}")
        input_files.add(path)

    seed = manifest.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ArtifactError("manifest.seed must be an integer or null")

    production = manifest.get("production")
    if not isinstance(production, dict) or set(production) != {"requested", "official_source", "policy"}:
        raise ArtifactError("manifest.production is missing or malformed")
    if production.get("requested") and kind not in PRODUCTION_KINDS:
        raise ArtifactError(f"production is not valid for stage {kind!r}")
    official = production.get("official_source")
    if official is not None:
        if not isinstance(official, dict) or set(official) != {"url", "revision"}:
            raise ArtifactError("manifest.production.official_source is missing or malformed")
        url = _require_nonempty_str(official.get("url"), "official source url")
        if not url.startswith("https://"):
            raise ArtifactError(f"official source url must be https: {url!r}")
        _validate_revision(official.get("revision"), "official source revision")

    # Stage preconditions hold for the stored manifest, not just at creation.
    _check_stage_requirements(
        kind, parent_roles, set(inputs), seed, METRICS_NAME in listed
    )

    # Every on-disk file must be explained by the manifest, and every
    # listed file must exist on disk.
    on_disk = {rel for rel, _ in scan_artifact_files(root)}
    unexplained = sorted(on_disk - set(listed) - {MANIFEST_NAME, CHECKSUMS_NAME})
    if unexplained:
        raise ArtifactError(f"unexplained files present in the artifact: {unexplained}")
    missing = sorted(set(listed) - on_disk)
    if missing:
        raise ArtifactError(f"files listed in the manifest are missing from disk: {missing}")

    # Source lineage keeps the same immutable-revision policy as create:
    # the stored source_commits.json must record full, immutably-identified
    # revisions and commits, never moving refs.
    try:
        with open(os.path.join(root, SOURCE_COMMITS_NAME), "r", encoding="utf-8") as handle:
            source_commits = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"source_commits.json is not valid JSON: {exc}") from exc
    _validate_source_commits(source_commits)

    # Recompute payload and sidecar digests.
    for rel, abs_path in scan_artifact_files(root):
        if rel in (CHECKSUMS_NAME, MANIFEST_NAME):
            continue
        entry = listed[rel]
        actual_size = os.path.getsize(abs_path)
        if actual_size != entry["size_bytes"]:
            raise ArtifactError(
                f"size mismatch for {rel}: manifest says {entry['size_bytes']}, disk has {actual_size}"
            )
        actual = sha256_file(abs_path)
        if actual != entry["sha256"]:
            raise ArtifactError(f"content digest mismatch for {rel}: payload or sidecar was modified")

    # Environment sidecar must remain a sanitized object with no secret keys.
    try:
        with open(os.path.join(root, ENVIRONMENT_NAME), "r", encoding="utf-8") as handle:
            environment = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"environment.json is not valid JSON: {exc}") from exc
    if not isinstance(environment, dict):
        raise ArtifactError("environment.json must be a JSON object")
    _reject_secret_keys(environment, ENVIRONMENT_NAME)

    # checksums.txt must explain exactly the payload files and sidecars.
    checksums_path = os.path.join(root, CHECKSUMS_NAME)
    try:
        with open(checksums_path, "r", encoding="utf-8") as handle:
            on_disk_checksums = handle.read()
    except FileNotFoundError as exc:
        raise ArtifactError("missing checksums.txt") from exc
    entries = [listed[rel] for rel in sorted(listed)]
    expected = _checksums_text(entries, sha256_file(manifest_path))
    if on_disk_checksums != expected:
        raise ArtifactError(
            "checksums.txt does not match recomputed digests: unexplained or tampered checksum sidecar"
        )

    return {
        "status": "verified",
        "kind": kind,
        "manifest_sha256": recorded,
        "files": len(listed),
        "parents": len(parents),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.artifacts",
        description="Create and verify artifact provenance sidecars.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    create = sub.add_parser("create", help="create artifact sidecars in --root")
    create.add_argument("--root", required=True, help="artifact root directory")
    create.add_argument(
        "--kind", required=True, choices=ARTIFACT_KINDS, help="artifact stage"
    )
    create.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=URL@FULL_SHA",
        help="explicit immutable source revision: 40-hex git/HF commit SHA or "
        "64-hex content digest (repeatable)",
    )
    create.add_argument(
        "--signed-source",
        action="append",
        default=[],
        metavar="NAME",
        help="mark a source as declared-signed (repeatable)",
    )
    create.add_argument(
        "--parent",
        action="append",
        default=[],
        metavar="ROLE=KIND:SHA256",
        help="parent artifact digest (KIND optional; repeatable)",
    )
    create.add_argument("--seed", type=int, default=None, help="deterministic seed")
    create.add_argument(
        "--command",
        action="append",
        default=[],
        required=True,
        metavar="CMD",
        help="reproduction command line (repeatable)",
    )
    create.add_argument(
        "--environment",
        required=True,
        metavar="PATH",
        help="sanitized environment JSON file (never a raw env dump)",
    )
    create.add_argument(
        "--dataset-manifest",
        default=None,
        metavar="PATH",
        help="dataset manifest JSON to copy as dataset_manifest.json",
    )
    create.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="dataset content digest (repeatable; builds the dataset sidecar)",
    )
    create.add_argument("--metrics", default=None, metavar="PATH", help="metrics JSON sidecar")
    create.add_argument("--expert-map", default=None, metavar="PATH", help="per-layer expert map JSON")
    create.add_argument(
        "--calibration-config", default=None, metavar="PATH", help="calibration config JSON"
    )
    create.add_argument(
        "--quant-assignment", default=None, metavar="PATH", help="per-tensor quant assignment JSON"
    )
    create.add_argument(
        "--payload", action="append", default=[], metavar="PATH", help="payload file (repeatable)"
    )
    create.add_argument(
        "--conversion-commit",
        default=None,
        metavar="SHA",
        help="conversion commit (full 40-hex commit SHA)",
    )
    create.add_argument(
        "--runtime-commit", default=None, metavar="SHA", help="runtime commit (full 40-hex commit SHA)"
    )
    create.add_argument(
        "--production",
        action="store_true",
        help="declare production lineage (requires --official-source)",
    )
    create.add_argument(
        "--official-source",
        default=None,
        metavar="URL@FULL_SHA",
        help="declared official Xiaomi source, immutable full revision "
        "(recorded, not verified)",
    )
    create.add_argument(
        "--created-at",
        default=None,
        metavar="TIMESTAMP",
        help="caller-supplied creation timestamp (kept out otherwise for determinism)",
    )

    verify = sub.add_parser("verify", help="verify artifact sidecars in --root")
    verify.add_argument("--root", required=True, help="artifact root directory")
    return parser


def main(argv=None):
    """CLI entry point. Returns 0 on success, 1 on any artifact error."""
    args = _build_parser().parse_args(argv)
    try:
        if args.mode == "create":
            sources = [parse_source_spec(spec) for spec in args.source]
            parents = [parse_parent_spec(spec) for spec in args.parent]
            manifest = create_artifact(
                args.root,
                args.kind,
                sources=sources,
                commands=list(args.command),
                environment_path=args.environment,
                dataset_manifest_path=args.dataset_manifest,
                dataset_hashes=list(args.dataset),
                metrics_path=args.metrics,
                expert_map_path=args.expert_map,
                calibration_config_path=args.calibration_config,
                quant_assignment_path=args.quant_assignment,
                payload_paths=list(args.payload),
                parents=parents,
                seed=args.seed,
                conversion_commit=args.conversion_commit,
                runtime_commit=args.runtime_commit,
                signed_sources=list(args.signed_source),
                production=args.production,
                official_source=args.official_source,
                created_at=args.created_at,
            )
            report = {
                "status": "created",
                "kind": manifest["kind"],
                "manifest_sha256": manifest["manifest_sha256"],
                "files": len(manifest["files"]),
            }
        else:
            report = verify_artifact(args.root)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
