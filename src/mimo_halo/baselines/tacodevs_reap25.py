"""Importer for the tacodevs REAP25 mixed-3bit GPTQ MTP baseline.

Archives small public metadata from the immutable Hugging Face revision
``95d80053eb22112f2acc4244330eb67b4f534c94`` of
``tacodevs/MiMo-V2.6-Flash-RL-MLX-REAP25-mixed3bit-GPTQ-MTP`` and
normalizes it to the ``schema_version=1`` baseline contract consumed by
``mimo_halo.baselines.compare``.

Strictness rules enforced here:

- Never fetches ``.safetensors``/``.pt``/``.bin`` (or any file outside the
  explicit allowlist). Weight byte counts come from the API tree listing
  and the safetensors index, never from weight downloads.
- Every archived file: extension allowlisted, actual byte count must equal
  the tree-declared size and stay under the global limit, sha256 and git
  blob sha1 recorded, fetch timestamp recorded.
- Normalization parses the actual archived metadata (compression_alloc,
  config, index, README tables); contradictory dimensions, duplicate or
  out-of-range expert IDs, missing metadata fail closed.
- The checkpoint stores experts as packed ``switch_mlp`` tensors without
  per-expert tensor names, so retained original IDs are derived as the
  complement of the published per-layer pruned-ID lists. A packed index
  is never reported as if it were an original expert ID.
- ``import`` re-hashes and size-checks every consumed raw file against
  ``evidence/files.json`` before writing output, so provenance stamps
  describe the bytes actually parsed; a refusal never overwrites an
  existing ``normalized.json``. Both ``import`` and ``verify`` require
  the record set to be unique, allowlisted, confined under ``raw/``
  (no ``..``, absolute paths, or symlinks), URL-bound to the pinned
  revision, exactly equal to the on-disk raw files, and reconciled with
  ``manifest.json`` ``file_count``/``total_archived_bytes``.
- The recorded sha256/git-blob-sha1 checksums are unsigned integrity
  evidence against the recorded fetch, not cryptographic authenticity;
  the provenance claim is the pinned revision fetched over HTTPS
  (revision API sha confirmation plus tree-oid binding).
- Published card values (held-out PPL, top-1 agreement, expert
  bits/weight) are parsed from the archived README rather than
  hardcoded, and every agreement label is computed from the actual
  values.

CLI (run with ``PYTHONPATH=src``)::

    python -m mimo_halo.baselines.tacodevs_reap25 archive  --out manifests/baselines/tacodevs-reap25
    python -m mimo_halo.baselines.tacodevs_reap25 import   --manifest manifests/baselines/tacodevs-reap25 --output manifests/baselines/tacodevs-reap25/normalized.json
    python -m mimo_halo.baselines.tacodevs_reap25 verify   --manifest manifests/baselines/tacodevs-reap25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASELINE_ID = "tacodevs-reap25-mixed3bit-gptq-mtp"
REPO_ID = "tacodevs/MiMo-V2.6-Flash-RL-MLX-REAP25-mixed3bit-GPTQ-MTP"
REVISION = "95d80053eb22112f2acc4244330eb67b4f534c94"
BASE_MODEL = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
PIPELINE_SOURCE = "https://github.com/irvollo/mimo-mlx-compress"

RESOLVE_BASE = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}"
TREE_URL = f"https://huggingface.co/api/models/{REPO_ID}/tree/{REVISION}?recursive=true"
REVISION_URL = f"https://huggingface.co/api/models/{REPO_ID}/revision/{REVISION}"

GLOBAL_FILE_LIMIT = 131072  # bytes; every archived file is small public metadata
FETCH_TIMEOUT = 60.0

# Small text metadata allowlist: extension-allowlisted paths under the
# pinned revision. Weight files (.safetensors/.pt/.bin) are NEVER archived.
ARCHIVE_FILES: tuple[str, ...] = (
    "README.md",
    "config.json",
    "compression_alloc.json",
    "compression_eval.json",
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "audio_tokenizer/chat_template.jinja",
    "audio_tokenizer/config.json",
    "audio_tokenizer/generation_config.json",
    "audio_tokenizer/tokenizer_config.json",
    "dflash/config.json",
    "dflash/dflash.py",
    "dflash/model.safetensors.index.json",
    "mtp/config.json",
    "omnimodal/config.json",
    "omnimodal/manifest.json",
)
ALLOWED_EXTENSIONS = frozenset({".json", ".md", ".jinja", ".py"})
FORBIDDEN_EXTENSIONS = frozenset({".safetensors", ".pt", ".bin", ".gguf", ".ckpt"})

TEXT_SHARD_PREFIX = "model-"
TEXT_SHARD_SUFFIX = ".safetensors"
AUX_WEIGHT_DIRS = ("mtp/", "dflash/", "omnimodal/", "audio_tokenizer/")
WEIGHT_EXTENSIONS = (".safetensors", ".pt", ".bin")


class BaselineError(RuntimeError):
    """Fail-closed error for archive/import/verify of this baseline."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _json_bytes(obj: object) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _http_get_bytes(url: str, limit: int) -> bytes:
    """Fetch a URL, aborting if the response exceeds ``limit`` bytes.

    Content-Length alone is never trusted; the body is streamed and
    truncated-abort happens as soon as the limit is exceeded.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "mimo-halo-lab/0.1 (baseline-archive)"})
    chunks: list[bytes] = []
    total = 0
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise BaselineError(f"{url}: response exceeded byte limit {limit}; aborting")
            chunks.append(chunk)
    return b"".join(chunks)


def _http_get_json(url: str, limit: int = 4_194_304) -> object:
    return json.loads(_http_get_bytes(url, limit).decode("utf-8"))


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------


def _validate_allowlist(path: str, declared_size: int) -> None:
    if path not in ARCHIVE_FILES:
        raise BaselineError(f"refusing to archive non-allowlisted path: {path}")
    suffix = Path(path).suffix
    if suffix not in ALLOWED_EXTENSIONS or suffix in FORBIDDEN_EXTENSIONS:
        raise BaselineError(f"extension {suffix!r} of {path} not in allowlist {sorted(ALLOWED_EXTENSIONS)}")
    if declared_size > GLOBAL_FILE_LIMIT:
        raise BaselineError(f"{path}: declared size {declared_size} exceeds global limit {GLOBAL_FILE_LIMIT}")


def archive(out_dir: Path) -> dict:
    """Archive small metadata from the pinned revision into ``out_dir``.

    Layout::

        out_dir/
          raw/<original repo paths>
          evidence/revision.json      API revision response
          evidence/tree.json          recursive API tree response (sizes, oids)
          evidence/files.json         per-file sha256/blob-sha1/url/timestamps
          evidence/checksums.sha256
          manifest.json               summary
    """
    raw_dir = out_dir / "raw"
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    revision = _http_get_json(REVISION_URL)
    if not isinstance(revision, dict) or revision.get("sha") != REVISION:
        raise BaselineError(f"revision API did not confirm pinned sha {REVISION}")
    tree = _http_get_json(TREE_URL)
    if not isinstance(tree, list):
        raise BaselineError("tree API returned unexpected payload type")
    tree_by_path = {entry["path"]: entry for entry in tree if entry.get("type") == "file"}
    if not tree_by_path:
        raise BaselineError("tree API returned no file entries")

    (evidence_dir / "revision.json").write_bytes(_json_bytes(revision))
    (evidence_dir / "tree.json").write_bytes(_json_bytes(tree))

    files_records: list[dict] = []
    checksum_lines: list[str] = []
    for path in ARCHIVE_FILES:
        entry = tree_by_path.get(path)
        if entry is None:
            raise BaselineError(f"allowlisted path {path} missing from tree at pinned revision")
        declared = entry["size"]
        _validate_allowlist(path, declared)
        url = f"{RESOLVE_BASE}/{path}"
        data = _http_get_bytes(url, GLOBAL_FILE_LIMIT)
        if len(data) != declared:
            raise BaselineError(
                f"{path}: fetched {len(data)} bytes but tree declares {declared}; aborting"
            )
        if Path(path).suffix == ".json":
            json.loads(data)  # parse check; malformed JSON is never archived silently
        digest = _sha256(data)
        blob_sha1 = _git_blob_sha1(data)
        if entry.get("oid") and not entry.get("lfs") and blob_sha1 != entry["oid"]:
            raise BaselineError(
                f"{path}: computed git blob sha1 {blob_sha1} != tree oid {entry['oid']}"
            )
        target = raw_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files_records.append(
            {
                "path": path,
                "size_bytes": declared,
                "sha256": digest,
                "git_blob_sha1": blob_sha1,
                "url": url,
                "tree_oid": entry.get("oid"),
                "lfs_oid": (entry.get("lfs") or {}).get("oid"),
                "fetched_at": _utcnow(),
            }
        )
        checksum_lines.append(f"{digest}  raw/{path}\n")

    (evidence_dir / "files.json").write_bytes(_json_bytes(files_records))
    (evidence_dir / "checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "base_model": BASE_MODEL,
        "retrieved_at": _utcnow(),
        "file_count": len(files_records),
        "total_archived_bytes": sum(r["size_bytes"] for r in files_records),
        "excluded": {
            "tokenizer.json": "11.4 MB tokenizer vocabulary exceeds metadata size budget; sizes recorded in tree.json",
            "weights": "all .safetensors/.pt/.bin files never fetched; byte counts recorded in tree.json",
        },
    }
    manifest_path = evidence_dir.parent / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return manifest


# ---------------------------------------------------------------------------
# verification (tamper detection)
# ---------------------------------------------------------------------------


_REQUIRED_RECORD_KEYS = ("path", "sha256", "size_bytes", "url")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _confined_raw_path(raw_dir: Path, rel: str) -> Path:
    """Resolve a record path, refusing anything not confined under ``raw/``."""
    if not isinstance(rel, str) or not rel:
        raise BaselineError(f"record path must be a non-empty string, got {rel!r}")
    parts = rel.split("/")
    if "\\" in rel or "\x00" in rel or any(part in ("", ".", "..") for part in parts):
        raise BaselineError(
            f"record path {rel!r} is not confined under raw/ "
            "(absolute paths and empty, '.', or '..' segments are refused)"
        )
    probe = raw_dir
    for part in parts:
        probe = probe / part
        if probe.is_symlink():
            raise BaselineError(f"raw/{rel}: symlinked archive paths are not allowed")
    return raw_dir.joinpath(*parts)


def _validate_record(record: object, raw_dir: Path) -> str:
    """Structurally validate one evidence/files.json record; returns its path."""
    if not isinstance(record, dict):
        raise BaselineError(f"evidence/files.json: record must be an object, got {type(record).__name__}")
    missing_keys = [key for key in _REQUIRED_RECORD_KEYS if key not in record]
    if missing_keys:
        raise BaselineError(f"evidence/files.json record {record.get('path', '<no path>')!r}: missing keys {missing_keys}")
    rel = record["path"]
    _confined_raw_path(raw_dir, rel)
    if rel not in ARCHIVE_FILES:
        raise BaselineError(f"{rel}: record path not in the archive allowlist")
    sha = record["sha256"]
    if not isinstance(sha, str) or _SHA256_RE.match(sha) is None:
        raise BaselineError(f"{rel}: sha256 {sha!r} is not a lowercase 64-hex digest")
    size = record["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise BaselineError(f"{rel}: size_bytes must be a non-negative integer, got {size!r}")
    revision = record.get("revision")
    if revision is not None:
        if revision != REVISION:
            raise BaselineError(f"{rel}: recorded revision {revision!r} != pinned revision {REVISION}")
    else:
        url = record.get("url")
        if not isinstance(url, str) or f"/resolve/{REVISION}/" not in url:
            raise BaselineError(f"{rel}: url {record.get('url')!r} is not bound to pinned revision {REVISION}")
    return rel


def _validated_records(manifest_dir: Path) -> tuple[list[dict], dict, bytes]:
    """Load evidence/files.json and reconcile it with manifest.json.

    Returns ``(records, manifest_summary, manifest_bytes)``. Fails closed on
    missing or malformed inputs, malformed records, forged record paths
    (absolute, ``..``-escaping, or symlinked), non-allowlisted paths,
    duplicate records, records not bound to the pinned revision, and any
    mismatch between the record set and ``manifest.json`` ``file_count`` /
    ``total_archived_bytes``.
    """
    files_path = manifest_dir / "evidence" / "files.json"
    if not files_path.is_file():
        raise BaselineError(f"missing {files_path}")
    try:
        records = json.loads(files_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BaselineError(f"evidence/files.json: invalid JSON: {exc}") from None
    if not isinstance(records, list):
        raise BaselineError("evidence/files.json: expected a list of records")

    raw_dir = manifest_dir / "raw"
    validated: list[dict] = []
    seen: set[str] = set()
    for record in records:
        rel = _validate_record(record, raw_dir)
        if rel in seen:
            raise BaselineError(f"evidence/files.json: duplicate record for {rel}")
        seen.add(rel)
        validated.append(record)

    manifest_path = manifest_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BaselineError(f"missing {manifest_path}; cannot reconcile archive completeness")
    manifest_bytes = manifest_path.read_bytes()
    try:
        summary = json.loads(manifest_bytes)
    except ValueError as exc:
        raise BaselineError(f"manifest.json: invalid JSON: {exc}") from None
    if not isinstance(summary, dict):
        raise BaselineError("manifest.json: expected an object")
    file_count = summary.get("file_count")
    if isinstance(file_count, bool) or not isinstance(file_count, int):
        raise BaselineError(f"manifest.json: file_count missing or not an integer, got {file_count!r}")
    total_bytes = summary.get("total_archived_bytes")
    if isinstance(total_bytes, bool) or not isinstance(total_bytes, int):
        raise BaselineError(f"manifest.json: total_archived_bytes missing or not an integer, got {total_bytes!r}")
    if len(validated) != file_count:
        raise BaselineError(
            f"evidence/files.json has {len(validated)} unique records but manifest.json file_count is {file_count}"
        )
    record_total = sum(r["size_bytes"] for r in validated)
    if record_total != total_bytes:
        raise BaselineError(
            f"evidence/files.json records {record_total} total bytes but manifest.json "
            f"total_archived_bytes is {total_bytes}"
        )
    return validated, summary, manifest_bytes


def _raw_tree_files(raw_dir: Path) -> set[str]:
    """Every regular file under ``raw/`` (posix-relative); symlinks fail."""
    if not raw_dir.is_dir():
        raise BaselineError(f"missing raw directory {raw_dir}")
    found: set[str] = set()
    stack = [raw_dir]
    while stack:
        directory = stack.pop()
        for entry in sorted(directory.iterdir()):
            rel = entry.relative_to(raw_dir).as_posix()
            if entry.is_symlink():
                raise BaselineError(f"raw/{rel}: symlinks are not allowed in the archive")
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                found.add(rel)
            else:
                raise BaselineError(f"raw/{rel}: not a regular file")
    return found


def _verified_raw_bytes(records: list[dict], raw_dir: Path) -> dict[str, bytes]:
    """Hash and size-check every listed raw file; refuse missing/extra content.

    ``records`` must come from ``_validated_records``. Returns exactly the
    bytes that matched the recorded sha256/size, so callers parse only
    verified content; any mismatch raises before the map is returned.
    """
    on_disk = _raw_tree_files(raw_dir)
    listed = {record["path"] for record in records}
    failures: list[str] = [f"{rel}: missing raw file" for rel in sorted(listed - on_disk)]
    failures.extend(
        f"raw/{rel}: present on disk but not listed in evidence/files.json"
        for rel in sorted(on_disk - listed)
    )
    out: dict[str, bytes] = {}
    for record in records:
        rel = record["path"]
        if rel not in on_disk:
            continue  # already recorded as missing above
        data = (raw_dir / rel).read_bytes()
        digest = _sha256(data)
        if digest != record["sha256"]:
            failures.append(f"{rel}: sha256 {digest} != recorded {record['sha256']}")
        if len(data) != record["size_bytes"]:
            failures.append(f"{rel}: size {len(data)} != recorded {record['size_bytes']}")
        out[rel] = data
    if failures:
        raise BaselineError("verify failed:\n" + "\n".join(failures))
    return out


def verify(manifest_dir: Path) -> dict:
    """Reconcile the record set and rehash every archived raw file.

    Fails closed on duplicate/forged/non-allowlisted/unbound records,
    record-set vs ``manifest.json`` mismatches (``file_count``,
    ``total_archived_bytes``), missing or unlisted raw files, symlinks, and
    any sha256/size mismatch. The checksums are integrity evidence against
    the recorded fetch, not cryptographic signatures of the publisher.
    """
    manifest_dir = Path(manifest_dir)
    records, _summary, _manifest_bytes = _validated_records(manifest_dir)
    _verified_raw_bytes(records, manifest_dir / "raw")
    return {"schema_version": 1, "baseline_id": BASELINE_ID, "checked_files": len(records), "ok": True}


# ---------------------------------------------------------------------------
# README table parsing (published metrics, source-cited)
# ---------------------------------------------------------------------------


def _markdown_tables(text: str) -> list[tuple[list[str], list[dict]]]:
    """Extract every markdown table as (header cells, row dicts)."""
    tables: list[tuple[list[str], list[dict]]] = []
    columns: list[str] | None = None
    rows: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if columns is not None:
                tables.append((columns, rows))
                columns, rows = None, []
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= {"-", ":", " "} for c in cells):  # separator row
            continue
        if columns is None:
            columns, rows = cells, []
        else:
            rows.append(dict(zip(columns, cells)))
    if columns is not None:
        tables.append((columns, rows))
    return tables


def parse_published_tables(readme_text: str) -> dict:
    """Parse the held-out and on-policy markdown tables from the model card.

    Values are kept exactly as published (strings); nothing is invented, and
    missing tables raise.
    """
    quality = next(
        (s for s in re.split(r"^## ", readme_text, flags=re.MULTILINE) if s.startswith("Quality")),
        None,
    )
    if quality is None:
        raise BaselineError("README: no 'Quality' section found for published metrics")

    heldout = None
    on_policy = None
    for columns, rows in _markdown_tables(quality):
        if len(columns) == 3 and columns[0].startswith("Metric") and heldout is None:
            heldout = rows
        if len(columns) == 7 and columns[0] == "group" and columns[1] == "tokens" and on_policy is None:
            on_policy = rows
    if not heldout or not on_policy:
        raise BaselineError("README: published metric tables not found or headers drifted")
    return {"heldout": heldout, "on_policy": on_policy}


def _published_bpw_claim(readme_text: str) -> tuple[str, float]:
    """Extract the published expert bits/weight claim from the archived card."""
    match = re.search(r"experts average ([0-9]+(?:\.[0-9]+)?) bits/weight", readme_text)
    if match is None:
        raise BaselineError("README: published 'experts average <n> bits/weight' claim not found")
    text = match.group(1)
    return text, float(text)


def _heldout_card_values(heldout_rows: list[dict]) -> dict:
    """Published held-out PPL / top-1 agreement parsed from the card table.

    Values come from the "This model" column of the archived README table —
    never from hardcoded literals.
    """

    def _this_model_value(prefix: str) -> str:
        for row in heldout_rows:
            cells = list(row.values())
            if cells and isinstance(cells[0], str) and cells[0].startswith(prefix):
                return cells[-1]
        raise BaselineError(f"README held-out table: no '{prefix}...' row found")

    ppl_text = _this_model_value("Perplexity")
    top1_text = _this_model_value("Top-1")
    try:
        ppl = float(ppl_text)
    except ValueError:
        raise BaselineError(f"README held-out table: Perplexity value {ppl_text!r} is not numeric") from None
    if not top1_text.endswith("%"):
        raise BaselineError(f"README held-out table: Top-1 agreement value {top1_text!r} is not a percentage")
    try:
        top1 = float(top1_text[:-1])
    except ValueError:
        raise BaselineError(f"README held-out table: Top-1 agreement value {top1_text!r} is not a percentage") from None
    return {"ppl": ppl, "top1_agree_percent": top1}


# ---------------------------------------------------------------------------
# normalization core (pure functions over parsed metadata; unit-testable)
# ---------------------------------------------------------------------------


VALID_MODES = frozenset({"affine", "mxfp4"})
VALID_BITS = frozenset({2, 3, 4})
VALID_GROUP_SIZES = frozenset({32, 64, 128})
PROJECTION_NAMES = ("gate", "up", "down")


def _require_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselineError(f"{what}: expected integer, got {value!r}")
    return value


def _validate_projection_spec(spec: object, where: str) -> dict:
    if not isinstance(spec, dict) or set(spec) != {"bits", "group_size", "mode"}:
        raise BaselineError(f"{where}: projection spec keys must be exactly bits/group_size/mode, got {spec!r}")
    bits, group, mode = spec["bits"], spec["group_size"], spec["mode"]
    _require_int(bits, f"{where} bits")
    _require_int(group, f"{where} group_size")
    if mode not in VALID_MODES:
        raise BaselineError(f"{where}: unknown mode {mode!r}")
    if bits not in VALID_BITS:
        raise BaselineError(f"{where}: bits {bits} outside validated set {sorted(VALID_BITS)}")
    if group not in VALID_GROUP_SIZES:
        raise BaselineError(f"{where}: group_size {group} outside validated set {sorted(VALID_GROUP_SIZES)}")
    if mode == "mxfp4" and (bits != 4 or group != 32):
        raise BaselineError(f"{where}: mxfp4 must be bits=4 group=32, got bits={bits} group={group}")
    if mode == "affine" and bits == 4:
        raise BaselineError(f"{where}: affine bits=4 is not part of the published allocation vocabulary")
    return {"bits": bits, "group_size": group, "mode": mode}


def _derive_original_expert_count(pruned_lists: list[list[int]], n_routed_experts: object, prune_k: object) -> tuple[int, dict]:
    """Derive the original expert count and confirm it from independent sources.

    Original count must be derivable from the pruned-ID universe (max+1)
    AND equal retained(=config n_routed_experts) + pruned(=prune_k); any
    contradiction fails.
    """
    flat = [i for lst in pruned_lists for i in lst]
    if not flat:
        raise BaselineError("compression_alloc: no pruned IDs published; cannot derive original expert count")
    original_from_ids = max(flat) + 1
    retained = _require_int(n_routed_experts, "config.n_routed_experts")
    pruned_k = _require_int(prune_k, "compression_alloc.prune_k")
    original_from_counts = retained + pruned_k
    if original_from_ids != original_from_counts:
        raise BaselineError(
            f"contradictory original expert count: max pruned ID + 1 = {original_from_ids} "
            f"but n_routed_experts({retained}) + prune_k({pruned_k}) = {original_from_counts}"
        )
    status = {
        "max_pruned_id_plus_1": original_from_ids,
        "n_routed_experts_plus_prune_k": original_from_counts,
        "published_card_statement": "README: '192 of 256 experts remain in every MoE layer'",
    }
    return original_from_ids, status


def _layer_projection_specs(alloc: dict, layer: int) -> dict:
    experts = alloc["experts"]
    key = str(layer)
    entry = experts.get(key)
    if entry is None:
        raise BaselineError(f"compression_alloc: no projection metadata for MoE layer {layer}")
    if set(entry) != {"gate_proj", "up_proj", "down_proj"}:
        raise BaselineError(f"compression_alloc layer {layer}: projection keys must be gate/up/down_proj, got {sorted(entry)}")
    source_key = f"experts.{key}"
    return {
        name: {**_validate_projection_spec(entry[f"{name}_proj"], f"{source_key}.{name}_proj"), "source_key": f"{source_key}.{name}_proj"}
        for name in PROJECTION_NAMES
    }


def tree_weight_bytes(tree: list[dict] | None, *, prefix: str | None, suffix: str, dirs: tuple[str, ...] = ()) -> int | None:
    """Sum declared sizes from the archived tree listing (never weights fetched)."""
    if tree is None:
        return None
    total = 0
    for entry in tree:
        if entry.get("type") != "file":
            continue
        path = entry["path"]
        if not path.endswith(suffix):
            continue
        if prefix is not None and not Path(path).name.startswith(prefix):
            continue
        if dirs and not path.startswith(dirs):
            continue
        total += _require_int(entry["size"], f"tree size of {path}")
    return total


def normalize(
    config: dict,
    alloc: dict,
    compression_eval: dict,
    index: dict | None,
    readme_text: str,
    source_files: list[dict],
    tree: list[dict] | None,
    retrieved_at: str,
) -> dict:
    """Build the normalized ``schema_version=1`` document from parsed metadata.

    Pure function: all inputs are parsed JSON/bytes; raises ``BaselineError``
    on any contract violation. Unknown published fields stay null, never zero.
    """
    # -- config ------------------------------------------------------------
    num_layers = _require_int(config.get("num_hidden_layers"), "config.num_hidden_layers")
    moe_freq = config.get("moe_layer_freq")
    if not isinstance(moe_freq, list) or len(moe_freq) != num_layers:
        raise BaselineError("config.moe_layer_freq: missing or length != num_hidden_layers")
    if not moe_freq or moe_freq[0] != 0:
        raise BaselineError("config.moe_layer_freq: first entry must be 0 (dense lead layer) for this baseline")
    moe_layers = [i for i, flag in enumerate(moe_freq) if flag == 1]
    top_k = _require_int(config.get("num_experts_per_tok"), "config.num_experts_per_tok")
    n_routed = config.get("n_routed_experts")
    hidden_size = _require_int(config.get("hidden_size"), "config.hidden_size")
    moe_inter = _require_int(config.get("moe_intermediate_size"), "config.moe_intermediate_size")
    if config.get("n_shared_experts") is not None:
        raise BaselineError(f"config.n_shared_experts: expected none, got {config['n_shared_experts']!r}")

    # -- allocation --------------------------------------------------------
    if not isinstance(alloc, dict) or not isinstance(alloc.get("pruned"), dict):
        raise BaselineError("compression_alloc: 'pruned' per-layer map missing")
    prune_k = _require_int(alloc.get("prune_k"), "compression_alloc.prune_k")
    pruned_map = alloc["pruned"]
    pruned_lists: list[list[int]] = []
    pruned_by_layer: dict[int, list[int]] = {}
    for layer in moe_layers:
        raw_list = pruned_map.get(str(layer))
        if raw_list is None:
            raise BaselineError(f"compression_alloc: missing pruned list for MoE layer {layer}")
        if not isinstance(raw_list, list):
            raise BaselineError(f"compression_alloc layer {layer}: pruned list must be a list, got {type(raw_list).__name__}")
        ids: list[int] = []
        for value in raw_list:
            identifier = _require_int(value, f"compression_alloc layer {layer} pruned ID")
            if identifier < 0 or identifier >= 256:
                raise BaselineError(
                    f"compression_alloc layer {layer}: pruned original ID {identifier} out of range [0, 255] "
                    "(packed 192-expert indices are NOT original IDs)"
                )
            if identifier in ids:
                raise BaselineError(f"compression_alloc layer {layer}: duplicate pruned original ID {identifier}")
            ids.append(identifier)
        if ids != sorted(ids):
            raise BaselineError(f"compression_alloc layer {layer}: pruned IDs not sorted; refusing to guess order")
        if len(ids) != prune_k:
            raise BaselineError(f"compression_alloc layer {layer}: {len(ids)} pruned IDs != prune_k {prune_k}")
        pruned_lists.append(ids)
        pruned_by_layer[layer] = ids

    original_experts, original_status = _derive_original_expert_count(pruned_lists, n_routed, prune_k)
    if original_experts != 256:
        raise BaselineError(f"derived original expert count {original_experts} != 256 for this baseline")
    retained_count = _require_int(n_routed, "config.n_routed_experts")
    if len(moe_layers) != 47:
        raise BaselineError(f"derived MoE layer count {len(moe_layers)} != 47 for this baseline")

    # -- index (packed-ambiguity gate) -------------------------------------
    packed_note = None
    index_total = None
    if index is not None:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise BaselineError("model.safetensors.index.json: weight_map missing")
        index_total = index.get("metadata", {}).get("total_size")
        per_expert_named = [t for t in weight_map if re.search(r"experts\.\d+\.", t)]
        if per_expert_named:
            raise BaselineError(
                "packed-index ambiguity: index names individual experts "
                f"(e.g. {per_expert_named[0]!r}) but the source provides no packed->original "
                "mapping; retained original IDs cannot be derived safely"
            )
        packed_note = (
            "checkpoint stores experts as packed switch_mlp tensors "
            "(model.layers.N.mlp.switch_mlp.{gate,up,down}_proj) with no per-expert tensor names; "
            "retained_expert_ids are the ascending complement of the published pruned lists, "
            "not packed tensor indices"
        )

    # -- layers ------------------------------------------------------------
    layers_out: list[dict] = []
    projection_counts: dict[str, int] = {}
    for layer in moe_layers:
        specs = _layer_projection_specs(alloc, layer)
        pruned = pruned_by_layer[layer]
        retained = sorted(set(range(original_experts)) - set(pruned))
        if len(retained) != retained_count:
            raise BaselineError(
                f"layer {layer}: derived retained count {len(retained)} != n_routed_experts {retained_count}"
            )
        for name in PROJECTION_NAMES:
            spec = specs[name]
            key = f"{spec['mode']}_b{spec['bits']}_g{spec['group_size']}"
            projection_counts[key] = projection_counts.get(key, 0) + 1
        layers_out.append(
            {
                "layer": layer,
                "original_expert_count": original_experts,
                "retained_expert_ids": retained,
                "pruned_expert_ids": pruned,
                "projections": specs,
                "retained_ids_derivation": "ascending complement of compression_alloc.pruned (source order unavailable: packed tensors)",
            }
        )

    # -- precision summary ---------------------------------------------------
    projection_counts = dict(sorted(projection_counts.items()))
    expert_param_count = len(moe_layers) * retained_count * 3 * hidden_size * moe_inter
    expert_bytes_total = alloc.get("expert_bytes")
    computed_bpw = None
    if isinstance(expert_bytes_total, (int, float)) and expert_param_count > 0:
        computed_bpw = round(expert_bytes_total * 8.0 / expert_param_count, 4)
    published_bpw_text, published_bpw = _published_bpw_claim(readme_text)
    computed_bpw_source = (
        "compression_alloc.expert_bytes over derived expert parameter count "
        f"({len(moe_layers)} layers × {retained_count} experts × 3 × {hidden_size} × "
        f"{config.get('moe_intermediate_size')})"
    )
    if computed_bpw is None:
        computed_bpw_source += "; computed value unavailable (compression_alloc.expert_bytes missing or invalid)"
    elif round(computed_bpw, 2) == published_bpw:
        computed_bpw_source += f"; agrees with published {published_bpw_text}"
    else:
        computed_bpw_source += f"; does NOT agree with published {published_bpw_text} (computed {computed_bpw})"

    # -- size ---------------------------------------------------------------
    text_bytes = tree_weight_bytes(tree, prefix=TEXT_SHARD_PREFIX, suffix=TEXT_SHARD_SUFFIX)
    aux_bytes = 0
    for suffix in WEIGHT_EXTENSIONS:
        part = tree_weight_bytes(tree, prefix=None, suffix=suffix, dirs=AUX_WEIGHT_DIRS)
        aux_bytes = None if (aux_bytes is None or part is None) else aux_bytes + part
    if aux_bytes == 0:
        aux_bytes = None

    published = parse_published_tables(readme_text)
    card_values = _heldout_card_values(published["heldout"])
    published_text_gb = 100.4
    published_aux_gb = 7.3
    if "100.4 GB" not in readme_text:
        raise BaselineError("README: published '100.4 GB' text-model size claim not found")
    if "7.3 GB" not in readme_text:
        raise BaselineError("README: published '7.3 GB' auxiliary size claim not found")
    if "192 of 256 experts remain in every MoE layer" not in readme_text:
        raise BaselineError("README: published '192 of 256 experts remain in every MoE layer' card statement not found")

    # -- document -----------------------------------------------------------
    doc = {
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "files": [
                {"path": r["path"], "sha256": r["sha256"], "size_bytes": r["size_bytes"], "url": r["url"]}
                for r in source_files
            ],
            "retrieved_at": retrieved_at,
        },
        "architecture": {
            "original_experts_per_layer": original_experts,
            "retained_experts_per_layer": retained_count,
            "top_k": top_k,
            "moe_layer_count": len(moe_layers),
            "total_retained_expert_instances": len(moe_layers) * retained_count,
            "field_status": {
                "original_experts_per_layer": original_status,
                "retained_experts_per_layer": "config.n_routed_experts, cross-checked with per-layer pruned complement",
                "top_k": "config.num_experts_per_tok",
                "moe_layer_count": f"config.moe_layer_freq ({num_layers} entries, first dense); "
                f"compression_alloc experts/pruned cover exactly layers {moe_layers[0]}..{moe_layers[-1]}",
                "total_retained_expert_instances": "moe_layer_count × retained_experts_per_layer",
            },
        },
        "layers": layers_out,
        "precision_summary": {
            "projection_counts": projection_counts,
            "expert_bpw": {
                "value": published_bpw,
                "kind": "published",
                "includes_overhead": None,
                "source": f"archived README.md: 'experts average {published_bpw_text} bits/weight'",
            },
            "computed_expert_bpw": {
                "value": computed_bpw,
                "kind": "computed",
                "includes_overhead": True,
                "source": computed_bpw_source,
            },
            "sensitive_projection_references": [
                "compression_alloc.json: per-projection bits/group_size/mode allocation",
                "compression_alloc.json:cost (published allocation cost scalar)",
                "archived README.md 'What was done' step 2: Hessian-weighted output error, MILP precision allocation",
            ],
        },
        "size": {
            "text_weight_bytes": text_bytes,
            "text_weight_gb": round(text_bytes / 1e9, 6) if text_bytes is not None else None,
            "text_weight_gib": round(text_bytes / 2**30, 6) if text_bytes is not None else None,
            "published_text_gb": published_text_gb,
            "index_total_size_bytes": index_total,
            "auxiliary_weight_bytes": aux_bytes,
            "published_auxiliary_gb": published_aux_gb,
            "runtime_memory_measured": False,
            "unit_note": "decimal GB = bytes/1e9; GiB = bytes/2**30; published 100.4 GB matches the "
            "safetensors index total_size (100431282304 bytes = 100.431 decimal GB = 93.534 GiB); "
            "text_weight_bytes is the tree-summed shard size and includes safetensors headers",
        },
        "published_evaluation": {
            "distribution": {
                "compression_eval": compression_eval,
                "heldout_table": published["heldout"],
                "cross_check": {
                    "eval_ppl_rounded3": round(compression_eval.get("ppl"), 3),
                    "card_ppl": card_values["ppl"],
                    "agrees": round(compression_eval.get("ppl"), 3) == card_values["ppl"],
                    "eval_top1_agree": compression_eval.get("top1_agree"),
                    "card_top1_agree_percent": card_values["top1_agree_percent"],
                },
                "source": "compression_eval.json (archived); README.md 'Quality' held-out table",
            },
            "on_policy": {
                "table": published["on_policy"],
                "source": "archived README.md 'On-policy' section: 40 responses sampled from the "
                "real MiMo-V2.6-Flash via API; published distribution metrics",
                "task_success": None,
            },
            "task_success": None,
            "independently_reproduced": False,
        },
        "provenance": {
            "official_parent": f"https://huggingface.co/{REPO_ID} (base_model: {BASE_MODEL})",
            "pipeline_source": PIPELINE_SOURCE,
            "production_source_allowed": False,
            "role": "comparison_only",
            "metadata_sha256": None,
            "notes": [
                "published metrics are distribution-level, not task success",
                packed_note,
            ],
        },
    }
    return doc


# ---------------------------------------------------------------------------
# import CLI entry
# ---------------------------------------------------------------------------


def import_manifest(manifest_dir: Path, output: Path | None = None) -> dict:
    """Read archived raw metadata, verify the consumed bytes, emit normalized.json.

    The record set is reconciled with ``manifest.json`` and the raw directory,
    and every listed raw file is hashed/size-checked, before anything is
    parsed — so ``source.files`` and ``metadata_sha256`` always describe the
    bytes actually consumed. Any refusal happens before the output write and
    never overwrites an existing ``normalized.json``.
    """
    manifest_dir = Path(manifest_dir)
    raw_dir = manifest_dir / "raw"
    evidence_dir = manifest_dir / "evidence"

    files_records, manifest_summary, manifest_bytes = _validated_records(manifest_dir)
    by_path = {r["path"]: r for r in files_records}
    required = ("config.json", "compression_alloc.json", "compression_eval.json", "README.md")
    missing = [p for p in required if p not in by_path or not (raw_dir / p).is_file()]
    if missing:
        raise BaselineError(f"manifest missing required archived files: {sorted(set(missing))}")

    raw_bytes = _verified_raw_bytes(files_records, raw_dir)

    def _read_json(name: str) -> dict:
        return json.loads(raw_bytes[name].decode("utf-8"))

    config = _read_json("config.json")
    alloc = _read_json("compression_alloc.json")
    compression_eval = _read_json("compression_eval.json")
    readme_text = raw_bytes["README.md"].decode("utf-8")
    index = None
    if "model.safetensors.index.json" in by_path:
        index = _read_json("model.safetensors.index.json")
    tree = None
    tree_path = evidence_dir / "tree.json"
    if tree_path.is_file():
        tree = json.loads(tree_path.read_text(encoding="utf-8"))

    retrieved_at = manifest_summary.get("retrieved_at") or _utcnow()

    doc = normalize(
        config=config,
        alloc=alloc,
        compression_eval=compression_eval,
        index=index,
        readme_text=readme_text,
        source_files=files_records,
        tree=tree,
        retrieved_at=retrieved_at,
    )

    doc["provenance"]["metadata_sha256"] = _sha256(manifest_bytes)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_json_bytes(doc))
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.baselines.tacodevs_reap25",
        description="Archive, verify and normalize the tacodevs REAP25 baseline metadata",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_archive = sub.add_parser("archive", help="fetch small public metadata from the pinned revision")
    p_archive.add_argument("--out", type=Path, default=Path("manifests/baselines/tacodevs-reap25"))

    p_import = sub.add_parser("import", help="normalize archived metadata")
    p_import.add_argument("--manifest", type=Path, default=Path("manifests/baselines/tacodevs-reap25"))
    p_import.add_argument("--output", type=Path, default=None)

    p_verify = sub.add_parser("verify", help="recompute checksums of archived raw files")
    p_verify.add_argument("--manifest", type=Path, default=Path("manifests/baselines/tacodevs-reap25"))

    args = parser.parse_args(argv)
    try:
        if args.command == "archive":
            manifest = archive(args.out)
            print(f"archived {manifest['file_count']} files, {manifest['total_archived_bytes']} bytes -> {args.out}")
        elif args.command == "import":
            out = args.output or (args.manifest / "normalized.json")
            doc = import_manifest(args.manifest, out)
            arch = doc["architecture"]
            print(
                f"normalized {BASELINE_ID}: {arch['moe_layer_count']} MoE layers x "
                f"{arch['retained_experts_per_layer']}/{arch['original_experts_per_layer']} experts, "
                f"top_k={arch['top_k']} -> {out}"
            )
        else:
            result = verify(args.manifest)
            print(f"verified {result['checked_files']} archived files: checksums match")
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())