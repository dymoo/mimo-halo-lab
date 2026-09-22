"""Exact safetensors inventory for XiaomiMiMo/MiMo-V2.6-Flash-RL.

Bounded HTTP-Range metadata reader (8-byte length prefix + JSON header only,
never weight payloads), header validation, exact logical-parameter counting
from official shapes/packing, and prune-candidate memory arithmetic.

Everything is stdlib-only (Python >= 3.11).  Design rules enforced here:

* Every safetensors access is a ranged HTTPS GET.  A response is accepted
  only when status == 206 and Content-Range matches the requested window
  exactly; a 200 (server ignored Range) aborts immediately without reading
  the body.  A full response is never consumed.
* Metadata bytes are capped; anything larger aborts.
* Logical parameter counts are derived from official dtype/shape evidence.
  Unknown packing raises instead of guessing.
* Scales and other quantization aux tensors are overhead: role
  "quantization_auxiliary" with logical_parameters == 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ID = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
REVISION = "5711b268169967567844e1e560e8a3966da959b1"
DEFAULT_OUT = Path("manifests/models/mimo-v2.6-flash-rl")
RAW_URL = f"https://huggingface.co/{REPO_ID}/raw/{REVISION}/{{path}}"
RESOLVE_URL = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{{path}}"
API_URL = f"https://huggingface.co/api/models/{REPO_ID}?blobs=true"

# Metadata safety caps.  Shard headers here are O(0.1-1 MB); the cap exists
# so a misbehaving server can never push unbounded bytes at us.
MAX_HEADER_BYTES = 64 << 20  # 64 MiB per safetensors header
MAX_METADATA_BYTES = 64 << 20  # 64 MiB per text/JSON metadata file
MAX_REDIRECTS = 10

# Host suffixes we are willing to follow HTTPS redirects into (HF + HF CDN).
ALLOWED_HOST_SUFFIXES = ("huggingface.co", "hf.co")

# safetensors dtype -> itemsize (stored bytes per element).
DTYPE_ITEMSIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

SCHEMA_VERSION = 1

# Prune candidates (retained routed experts out of 256).
PRUNE_CANDIDATES = (256, 192, 176, 160, 144)


class InventoryError(Exception):
    """Base class for inventory failures."""


class RangeNotRespectedError(InventoryError):
    """Server answered a ranged request without an exact 206."""


class HeaderError(InventoryError):
    """Safetensors header failed structural validation."""


class PackingError(InventoryError):
    """Logical packing of a tensor cannot be determined from evidence."""


# --------------------------------------------------------------------------
# Bounded HTTPS Range reader
# --------------------------------------------------------------------------


def _https_allowed(url: str) -> bool:
    if not url.startswith("https://"):
        return False
    host = urllib.request.urlsplit(url).hostname or ""
    return any(
        host == s or host.endswith("." + s) for s in ALLOWED_HOST_SUFFIXES
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Suppress automatic redirects so we can validate each hop ourselves."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


@dataclass
class RangeResponse:
    data: bytes
    total_size: int  # from Content-Range "*/total"; -1 when absent


class HttpsRangeClient:
    """Minimal stdlib HTTPS client with validated, bounded Range reads."""

    def __init__(
        self,
        timeout: float = 60.0,
        max_header_bytes: int = MAX_HEADER_BYTES,
        max_metadata_bytes: int = MAX_METADATA_BYTES,
    ) -> None:
        self.timeout = timeout
        self.max_header_bytes = max_header_bytes
        self.max_metadata_bytes = max_metadata_bytes
        self.opener = urllib.request.build_opener(_NoRedirect)
        self.opener.addheaders = [("User-Agent", "mimo-halo-inventory/1.0")]

    # -- low level ---------------------------------------------------------

    def _open(self, url: str, rng: tuple[int, int] | None) -> tuple[int, int, int, object]:
        """Request url with optional closed byte range, following redirects
        manually.  Returns (status, start, length, response-like fp)."""
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            if not _https_allowed(current):
                raise InventoryError(f"refusing non-allowlisted URL: {current}")
            req = urllib.request.Request(current, method="GET")
            if rng is not None:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            try:
                resp = self.opener.open(req, timeout=self.timeout)
            except urllib.error.HTTPError as err:
                if err.code in (301, 302, 303, 307, 308):
                    loc = err.headers.get("Location")
                    if not loc:
                        raise InventoryError(
                            f"redirect without Location at {current}"
                        ) from err
                    # 303 mandates GET; others preserve GET here.
                    err.close()
                    current = urllib.request.urljoin(current, loc)
                    continue
                err.close()
                raise InventoryError(f"HTTP {err.code} for {current}") from err
            return resp.status, rng[0] if rng else 0, (
                rng[1] - rng[0] + 1 if rng else -1
            ), resp
        raise InventoryError(f"too many redirects fetching {url}")

    def _read_exact(self, resp, length: int, cap: int, what: str) -> bytes:
        if length > cap:
            raise InventoryError(
                f"{what}: requested {length} bytes exceeds cap {cap}"
            )
        content_length = resp.headers.get("Content-Length")
        if content_length is not None and int(content_length) != length:
            resp.close()
            raise InventoryError(
                f"{what}: Content-Length {content_length} != expected {length}"
            )
        data = resp.read(length + 1)
        resp.close()
        if len(data) < length:
            raise InventoryError(f"{what}: truncated response ({len(data)} < {length})")
        if len(data) > length:
            raise InventoryError(
                f"{what}: server sent more than requested "
                f"({len(data)} > {length}); aborting full-response read"
            )
        return data

    # -- public API --------------------------------------------------------

    def read_range(self, url: str, start: int, length: int, what: str = "range") -> RangeResponse:
        """Read exactly `length` bytes at `start`, requiring an exact 206."""
        if length <= 0:
            raise InventoryError(f"{what}: non-positive range length {length}")
        status, req_start, req_len, resp = self._open(url, (start, start + length - 1))
        if status == 200:
            resp.close()
            raise RangeNotRespectedError(
                f"{what}: server ignored Range and returned the full response "
                f"for {url}; aborting (no fallback download)"
            )
        if status != 206:
            resp.close()
            raise RangeNotRespectedError(
                f"{what}: expected 206 Partial Content, got HTTP {status} for {url}"
            )
        content_range = resp.headers.get("Content-Range")
        if content_range is None:
            resp.close()
            raise RangeNotRespectedError(
                f"{what}: 206 without Content-Range for {url}"
            )
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\*|(\d+))", content_range.strip())
        if match is None:
            resp.close()
            raise RangeNotRespectedError(
                f"{what}: unparsable Content-Range {content_range!r} for {url}"
            )
        c_start, c_end = int(match.group(1)), int(match.group(2))
        c_total = -1 if match.group(3) == "*" else int(match.group(4))
        if c_start != req_start or c_end != req_start + length - 1:
            resp.close()
            raise RangeNotRespectedError(
                f"{what}: Content-Range {c_start}-{c_end} does not match "
                f"requested {req_start}-+{length} for {url}"
            )
        data = self._read_exact(resp, length, self.max_header_bytes, what)
        return RangeResponse(data=data, total_size=c_total)

    def read_text(self, url: str, what: str = "metadata") -> str:
        """Fetch a small text/JSON artifact with a hard byte cap."""
        status, _, _, resp = self._open(url, None)
        if status != 200:
            resp.close()
            raise InventoryError(f"{what}: expected 200, got HTTP {status} for {url}")
        content_length = resp.headers.get("Content-Length")
        if content_length is not None and int(content_length) > self.max_metadata_bytes:
            resp.close()
            raise InventoryError(f"{what}: {content_length} bytes exceeds metadata cap")
        chunks = []
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > self.max_metadata_bytes:
                resp.close()
                raise InventoryError(f"{what}: exceeds metadata cap {self.max_metadata_bytes}")
            chunks.append(chunk)
        resp.close()
        return b"".join(chunks).decode("utf-8")


# --------------------------------------------------------------------------
# safetensors header reading + validation
# --------------------------------------------------------------------------


def _json_no_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise HeaderError(f"duplicate JSON key in header: {key!r}")
        obj[key] = value
    return obj


@dataclass
class SafetensorsHeader:
    name: str
    url: str
    file_size: int
    header_length: int  # N, including alignment padding
    tensors: dict  # name -> {"dtype", "shape", "data_offsets"}
    metadata: dict | None = None


def parse_safetensors_header(
    name: str,
    url: str,
    first8: bytes,
    header_bytes: bytes,
    file_size: int,
) -> SafetensorsHeader:
    """Validate the 8-byte prefix + JSON header block of a safetensors file."""
    if len(first8) != 8:
        raise HeaderError(f"{name}: expected 8 prefix bytes, got {len(first8)}")
    (n,) = struct.unpack("<Q", first8)
    if n < 8:
        raise HeaderError(f"{name}: header length {n} < 8 (truncated/corrupt)")
    if n > MAX_HEADER_BYTES:
        raise HeaderError(f"{name}: header length {n} exceeds cap {MAX_HEADER_BYTES}")
    if file_size >= 0 and 8 + n > file_size:
        raise HeaderError(
            f"{name}: header needs {8 + n} bytes but file is {file_size} bytes "
            "(truncated header)"
        )
    if len(header_bytes) != n:
        raise HeaderError(
            f"{name}: header block {len(header_bytes)} bytes != N {n}"
        )
    try:
        header = json.loads(header_bytes.decode("utf-8"), object_pairs_hook=_json_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise HeaderError(f"{name}: header is not valid JSON: {err}") from err
    if not isinstance(header, dict):
        raise HeaderError(f"{name}: header is not a JSON object")
    metadata = header.pop("__metadata__", None)

    data_start = 8 + n
    prev_end = 0
    tensors = {}
    for tname, entry in header.items():
        if not isinstance(entry, dict):
            raise HeaderError(f"{name}: tensor {tname!r} entry is not an object")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in DTYPE_ITEMSIZE:
            raise HeaderError(f"{name}: tensor {tname!r} unknown dtype {dtype!r}")
        if (
            not isinstance(shape, list)
            or not all(isinstance(d, int) and d >= 0 for d in shape)
        ):
            raise HeaderError(f"{name}: tensor {tname!r} invalid shape {shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) and o >= 0 for o in offsets)
        ):
            raise HeaderError(f"{name}: tensor {tname!r} invalid data_offsets {offsets!r}")
        start, end = offsets
        if end < start:
            raise HeaderError(f"{name}: tensor {tname!r} end < start")
        numel = 1
        for d in shape:
            numel *= d
        expected = numel * DTYPE_ITEMSIZE[dtype]
        if end - start != expected:
            raise HeaderError(
                f"{name}: tensor {tname!r} span {end - start} != "
                f"{numel} x {DTYPE_ITEMSIZE[dtype]} ({expected})"
            )
        if start < prev_end:
            raise HeaderError(
                f"{name}: tensor {tname!r} offsets [{start},{end}) overlap/precede "
                f"previous end {prev_end}"
            )
        prev_end = end
        tensors[tname] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
    if tensors and prev_end != 0 and tensors:
        first = min(v["data_offsets"][0] for v in tensors.values())
        if first != 0:
            raise HeaderError(f"{name}: data does not start at offset 0 (got {first})")
    if file_size >= 0:
        payload_end = data_start + prev_end
        if payload_end > file_size:
            raise HeaderError(
                f"{name}: payload ends at {payload_end} beyond file size {file_size}"
            )
    return SafetensorsHeader(
        name=name,
        url=url,
        file_size=file_size,
        header_length=n,
        tensors=tensors,
        metadata=metadata,
    )


def fetch_safetensors_header(client: HttpsRangeClient, name: str, url: str) -> SafetensorsHeader:
    """Two bounded ranged GETs: bytes 0-7 then the JSON header.  Never the payload."""
    prefix = client.read_range(url, 0, 8, what=f"{name}:prefix")
    file_size = prefix.total_size
    (n,) = struct.unpack("<Q", prefix.data)
    if n < 8:
        raise HeaderError(f"{name}: header length {n} < 8 (truncated/corrupt)")
    if n > client.max_header_bytes:
        raise HeaderError(f"{name}: header length {n} exceeds cap {client.max_header_bytes}")
    if file_size >= 0 and 8 + n > file_size:
        raise HeaderError(
            f"{name}: header needs {8 + n} bytes but file is {file_size} bytes"
        )
    body = client.read_range(url, 8, n, what=f"{name}:header")
    if file_size < 0:
        file_size = body.total_size  # cannot happen with a matched Content-Range
    return parse_safetensors_header(name, url, prefix.data, body.data, file_size)


# --------------------------------------------------------------------------
# Architecture / packing rules (evidence-driven)
# --------------------------------------------------------------------------

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_EXPERT_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")
_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")
_SCALE_SUFFIXES = ("weight_scale", "weight_scale_inv")


def tensor_role(name: str) -> str:
    if name.endswith(_SCALE_SUFFIXES):
        return "quantization_auxiliary"
    return "parameter"


def classify_component(name: str, file_name: str) -> str:
    if file_name.startswith("dflash/"):
        return "dflash"
    if name.startswith("model.mtp.") or file_name == "model_mtp.safetensors":
        return "mtp"
    if name.startswith("visual."):
        return "vision"
    if name.startswith("audio_encoder.") or name.startswith("speech_embeddings."):
        return "audio"
    return "text"


def quantization_record(name: str, dtype: str, config: dict, index_meta: dict) -> dict | None:
    """Build the per-tensor quantization record from official config evidence."""
    role = tensor_role(name)
    quant = config.get("quantization_config") or {}
    store_dtype = quant.get("store_dtype")
    save_format = (index_meta or {}).get("save_format")
    if role == "quantization_auxiliary":
        kind = "mxfp4_scale" if name.endswith("weight_scale") else "fp8_scale"
        rec = {"kind": kind}
        if kind == "fp8_scale":
            rec["block_size"] = quant.get("weight_block_size")
        else:
            rec["block_size"] = quant.get("mxfp4_block_size")
        return rec
    if name.endswith(".weight"):
        if dtype == "U8" and store_dtype == "mxfp4" and save_format == "mxfp4":
            return {
                "kind": "mxfp4_packed",
                "stored_dtype": dtype,
                "block_size": quant.get("mxfp4_block_size"),
                "scale_tensor": name + "_scale",
            }
        if dtype == "F8_E4M3":
            return {
                "kind": "fp8_e4m3",
                "stored_dtype": dtype,
                "block_size": quant.get("weight_block_size"),
                "scale_tensor": name + "_inv",
            }
        return None
    return None


def logical_view(
    name: str,
    dtype: str,
    shape: list[int],
    role: str,
    header: SafetensorsHeader,
    config: dict,
    index_meta: dict,
) -> tuple[list[int], int]:
    """Return (logical_shape, logical_parameters) from official packing evidence.

    Raises PackingError when the layout is not fully determined by evidence —
    exact counting must fail, never guess.
    """
    numel = 1
    for d in shape:
        numel *= d
    if role == "quantization_auxiliary":
        return list(shape), 0
    quant = config.get("quantization_config") or {}
    store_dtype = quant.get("store_dtype")
    save_format = (index_meta or {}).get("save_format")
    if dtype == "U8":
        mxfp4 = store_dtype == "mxfp4" and save_format == "mxfp4"
        sibling = name + "_scale" in header.tensors
        if not (mxfp4 and sibling and name.endswith(".weight")):
            raise PackingError(
                f"{header.name}: {name!r} stored U8 without confirmed mxfp4 "
                f"packing evidence (store_dtype={store_dtype!r}, "
                f"save_format={save_format!r}, scale_sibling={sibling})"
            )
        # two 4-bit nibbles per stored byte; last dimension doubles
        logical = list(shape)
        logical[-1] = logical[-1] * 2
        return logical, numel * 2
    if dtype in ("F8_E4M3", "F8_E5M2"):
        # one byte per logical parameter (fp8 storage)
        return list(shape), numel
    if dtype in DTYPE_ITEMSIZE:
        # BF16/F16/F32/I64/... stored 1:1 with logical elements
        return list(shape), numel
    raise PackingError(f"{header.name}: {name!r} unsupported dtype {dtype!r}")


def build_tensor_record(
    name: str,
    entry: dict,
    header: SafetensorsHeader,
    file_name: str,
    config: dict,
    index_meta: dict,
    duplicate_names: set[str] | None = None,
) -> dict:
    dtype = entry["dtype"]
    shape = entry["shape"]
    stored_bytes = entry["data_offsets"][1] - entry["data_offsets"][0]
    role = tensor_role(name)
    logical_shape, logical_parameters = logical_view(
        name, dtype, shape, role, header, config, index_meta
    )
    return {
        "name": name,
        "shape": list(shape),
        "dtype": dtype,
        "stored_bytes": stored_bytes,
        "logical_shape": logical_shape,
        "logical_parameters": logical_parameters,
        "role": role,
        "layer": parse_layer(name),
        "expert": parse_expert(name),
        "component": classify_component(name, file_name),
        "shard": file_name,
        "quantization": quantization_record(name, dtype, config, index_meta),
    }


def parse_layer(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    if match:
        return int(match.group(1))
    match = _BLOCK_RE.search(name)
    if match:
        return int(match.group(1))
    return None


def parse_expert(name: str) -> int | None:
    match = _EXPERT_RE.search(name)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------
# Inventory assembly
# --------------------------------------------------------------------------


def moe_layers_from_config(config: dict) -> list[int]:
    freq = config.get("moe_layer_freq") or []
    return [i for i, flag in enumerate(freq) if flag]


def check_index_header_agreement(
    index_weight_map: dict[str, str],
    headers: dict[str, SafetensorsHeader],
) -> None:
    """Exact bijection between index weight_map entries and header tensors."""
    # every header tensor of an index-referenced shard must be mapped by
    # the index to that same shard (separately indexed files, like dflash,
    # are validated against their own index by the caller)
    index_files = set(index_weight_map.values())
    for shard_name, header in headers.items():
        if shard_name not in index_files:
            continue
        expected = {
            t for t, f in index_weight_map.items() if f == shard_name
        }
        actual = set(header.tensors)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise HeaderError(
                f"{shard_name}: index/header mismatch "
                f"(missing from header: {len(missing)}, e.g. {sorted(missing)[:3]}; "
                f"in header but not index shard: {len(extra)}, "
                f"e.g. {sorted(extra)[:3]})"
            )
    # files referenced by the index that we have no header for
    mapped_files = set(index_weight_map.values())
    header_files = set(headers)
    unheadered = mapped_files - header_files
    if unheadered:
        raise HeaderError(
            f"index references files without headers: {sorted(unheadered)[:5]}"
        )


def build_inventory(
    config: dict,
    index: dict,
    dflash_config: dict,
    dflash_index: dict,
    headers: dict[str, SafetensorsHeader],
    source_files: list[dict],
) -> dict:
    """Assemble the schema_version-1 inventory from archived evidence."""
    weight_map = index["weight_map"]
    index_meta = index.get("metadata", {})
    config["quantization_config"] = config.get("quantization_config") or {}

    # global duplicate detection across the whole weight map
    seen: dict[str, str] = {}
    for tensor, shard in weight_map.items():
        if tensor in seen:
            raise HeaderError(
                f"tensor {tensor!r} mapped to both {seen[tensor]} and {shard}"
            )
        seen[tensor] = shard

    check_index_header_agreement(weight_map, headers)

    records: list[dict] = []
    main_records: list[dict] = []
    for tensor_name, shard_name in sorted(weight_map.items()):
        header = headers[shard_name]
        entry = header.tensors[tensor_name]
        record = build_tensor_record(
            tensor_name, entry, header, shard_name, config, index_meta
        )
        records.append(record)
        main_records.append(record)

    # dflash drafter is indexed separately: same missing/extra/declared-total
    # guarantees as the main index, keyed by index membership (which header
    # the dflash weight_map actually claims), not by path alone.
    dflash_map = dflash_index["weight_map"]
    dflash_meta = dflash_index.get("metadata", {})
    dflash_records: list[dict] = []
    dflash_mapped: dict[str, set[str]] = {}
    seen_d: dict[str, str] = {}
    for tensor_name, shard_name in sorted(dflash_map.items()):
        if tensor_name in seen_d:
            raise HeaderError(
                f"dflash tensor {tensor_name!r} mapped twice: "
                f"{seen_d[tensor_name]} and {shard_name}"
            )
        seen_d[tensor_name] = shard_name
        header = headers.get(shard_name) or headers.get("dflash/" + shard_name)
        if header is None:
            raise HeaderError(f"dflash header missing for {shard_name}")
        if tensor_name not in header.tensors:
            raise HeaderError(
                f"dflash tensor {tensor_name!r} absent from {header.name} header"
            )
        entry = header.tensors[tensor_name]
        record = build_tensor_record(
            tensor_name, entry, header, header.name, dflash_config, dflash_meta
        )
        records.append(record)
        dflash_records.append(record)
        dflash_mapped.setdefault(header.name, set()).add(tensor_name)

    # header tensors never claimed by the dflash index would be silently
    # dropped; refuse them (participation is index membership plus any
    # dflash/-prefixed header, mirroring the main index's bijection check)
    for header in headers.values():
        participates = header.name in dflash_mapped or header.name.startswith(
            "dflash/"
        )
        if not participates:
            continue
        extra = set(header.tensors) - dflash_mapped.get(header.name, set())
        if extra:
            raise HeaderError(
                f"{header.name}: tensors present in dflash header but absent "
                f"from dflash index weight_map: {sorted(extra)[:5]}"
            )

    # declared totals must equal observed stored bytes (both indexes)
    declared_total = (index.get("metadata") or {}).get("total_size")
    observed_total = sum(r["stored_bytes"] for r in main_records)
    if declared_total is not None and declared_total != observed_total:
        raise HeaderError(
            f"index total_size {declared_total} != sum of tensor bytes {observed_total}"
        )
    dflash_declared_total = (dflash_index.get("metadata") or {}).get("total_size")
    dflash_observed_total = sum(r["stored_bytes"] for r in dflash_records)
    if (
        dflash_declared_total is not None
        and dflash_declared_total != dflash_observed_total
    ):
        raise HeaderError(
            f"dflash index total_size {dflash_declared_total} != sum of "
            f"dflash tensor bytes {dflash_observed_total}"
        )

    totals = summarize_totals(records)

    # whole-artifact accounting: file size vs header vs payload vs padding
    artifact_layout = []
    for shard_name, header in sorted(headers.items()):
        payload = sum(
            e["data_offsets"][1] - e["data_offsets"][0]
            for e in header.tensors.values()
        )
        header_bytes = 8 + header.header_length
        artifact_layout.append(
            {
                "file": shard_name,
                "file_size": header.file_size,
                "header_bytes": header_bytes,
                "tensor_payload_bytes": payload,
                "padding_bytes": (
                    header.file_size - header_bytes - payload
                    if header.file_size >= 0
                    else None
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "files": source_files,
        },
        "architecture": {
            "original_experts_per_layer": config.get("n_routed_experts"),
            "top_k": config.get("num_experts_per_tok"),
            "moe_layers": moe_layers_from_config(config),
        },
        "tensors": records,
        "totals": totals,
        "artifact_layout": artifact_layout,
    }


def summarize_totals(records: list[dict]) -> dict:
    by_component: dict[str, dict] = {}
    for rec in records:
        comp = rec["component"]
        bucket = by_component.setdefault(
            comp,
            {
                "tensors": 0,
                "parameters": 0,
                "stored_bytes": 0,
                "logical_parameters": 0,
                "quantization_auxiliary_bytes": 0,
            },
        )
        bucket["tensors"] += 1
        bucket["stored_bytes"] += rec["stored_bytes"]
        bucket["logical_parameters"] += rec["logical_parameters"]
        if rec["role"] == "quantization_auxiliary":
            bucket["quantization_auxiliary_bytes"] += rec["stored_bytes"]
        elif rec["quantization"]:
            bucket["parameters"] += 1
        else:
            bucket["parameters"] += 1
    grand = {
        "tensors": sum(b["tensors"] for b in by_component.values()),
        "stored_bytes": sum(b["stored_bytes"] for b in by_component.values()),
        "logical_parameters": sum(b["logical_parameters"] for b in by_component.values()),
        "quantization_auxiliary_bytes": sum(
            b["quantization_auxiliary_bytes"] for b in by_component.values()
        ),
    }
    return {"by_component": by_component, "grand": grand}


# --------------------------------------------------------------------------
# Memory / prune-candidate report
# --------------------------------------------------------------------------


def memory_report(
    inventory: dict, candidate_counts: list[int] | None = None
) -> dict:
    """Exact native counts + prune-candidate byte math from header evidence.

    Prune candidates subtract BOTH the removed expert tensors (weight +
    sibling scale, uniform across MoE layers) and the dense router rows those
    removals drop: pruning/maps slices every MoE router from
    ``original_experts_per_layer`` rows down to the retained count, so the
    candidate totals match the planned post-prune layout exactly.  Router
    tensors are dense with ``quantization: null``; per-row cost is exact byte
    arithmetic with a divisibility check — a packed or quantized router is
    refused, never guessed.

    ``candidate_counts`` overrides the default candidate list (the original
    expert count plus PRUNE_CANDIDATES filtered to ``<= original``); every
    count must satisfy ``0 < count <= original_experts_per_layer``.
    Precision alternatives are computed only for layouts the evidence
    supports (mxfp4 packed U8, bf16); anything else would be an estimate and
    is excluded.
    """
    records = inventory["tensors"]
    arch = inventory["architecture"]
    n_experts = arch["original_experts_per_layer"]

    comp_totals = inventory["totals"]["by_component"]
    text = comp_totals["text"]

    expert_records = [
        r
        for r in records
        if r["component"] == "text" and r["expert"] is not None
    ]
    non_expert_text = [
        r for r in records if r["component"] == "text" and r["expert"] is None
    ]
    non_expert_stored = sum(r["stored_bytes"] for r in non_expert_text)
    non_expert_logical = sum(r["logical_parameters"] for r in non_expert_text)

    # Exact per-expert-index totals (one expert index summed over ALL MoE
    # layers) with divisibility checks.
    expert_total_stored = sum(r["stored_bytes"] for r in expert_records)
    expert_total_logical = sum(r["logical_parameters"] for r in expert_records)
    if expert_total_stored % n_experts:
        raise InventoryError("expert stored bytes not divisible by expert count")
    if expert_total_logical % n_experts:
        raise InventoryError(
            "expert logical parameters not divisible by expert count"
        )
    if len(expert_records) % n_experts:
        raise InventoryError("expert tensor records not divisible by expert count")
    per_index_stored = expert_total_stored // n_experts
    per_index_logical = expert_total_logical // n_experts
    records_per_index = len(expert_records) // n_experts

    # sanity: every expert owns the same tensor set (uniform prune scaling)
    def expert_signature(e: int) -> frozenset:
        return frozenset(
            (r["name"].split(f".experts.{e}.", 1)[-1], r["stored_bytes"])
            for r in expert_records
            if r["expert"] == e
        )

    reference = expert_signature(0)
    for e in range(n_experts):
        if expert_signature(e) != reference:
            raise InventoryError(f"expert {e} tensor set differs from expert 0")

    # True per-layer expert layout: one expert index inside one MoE layer.
    moe_layers = arch["moe_layers"]
    n_moe_layers = len(moe_layers)
    if n_moe_layers == 0:
        raise InventoryError("architecture declares no MoE layers")
    if (
        per_index_stored % n_moe_layers
        or per_index_logical % n_moe_layers
        or records_per_index % n_moe_layers
    ):
        raise InventoryError(
            "per-expert-index totals do not divide evenly across MoE layers"
        )
    instance_stored = per_index_stored // n_moe_layers
    instance_logical = per_index_logical // n_moe_layers
    tensors_per_instance = records_per_index // n_moe_layers

    # Dense router accounting: pruning/maps slices every MoE router from
    # original_experts_per_layer rows down to the retained count, so each
    # dropped expert also drops one router row per MoE layer.  Routers are
    # dense (quantization null, row-uniform); bytes are divided by the
    # original row count only after exact divisibility is proven.
    router_records = [
        r
        for r in non_expert_text
        if r.get("layer") in moe_layers
        and (
            r["name"].endswith(".mlp.gate.weight")
            or r["name"].endswith(".mlp.gate.e_score_correction_bias")
        )
    ]
    router_per_row_stored = 0
    router_per_row_logical = 0
    for rec in router_records:
        quant = rec.get("quantization")
        if quant is not None:
            raise InventoryError(
                f"{rec['name']}: router tensor is quantized "
                f"({quant.get('format')}); dense row scaling does not apply"
            )
        rows = rec["shape"][0]
        if rows != n_experts:
            raise InventoryError(
                f"{rec['name']}: router rows {rows} != "
                f"original expert count {n_experts}"
            )
        if rec["stored_bytes"] % rows or rec["logical_parameters"] % rows:
            raise InventoryError(
                f"{rec['name']}: router bytes not divisible by {rows} rows; "
                "cannot row-scale exactly"
            )
        router_per_row_stored += rec["stored_bytes"] // rows
        router_per_row_logical += rec["logical_parameters"] // rows
    router_total_stored = router_per_row_stored * n_experts
    router_total_logical = router_per_row_logical * n_experts

    # Candidate list: default is the original expert count plus
    # PRUNE_CANDIDATES filtered to <= original (no fabricated counts for
    # small fixtures); explicit counts are honored as given.
    if candidate_counts is None:
        requested = {n_experts}
        requested.update(c for c in PRUNE_CANDIDATES if c <= n_experts)
        candidate_list = sorted(requested, reverse=True)
    else:
        candidate_list = sorted(set(candidate_counts), reverse=True)
    for retained in candidate_list:
        if retained <= 0 or retained > n_experts:
            raise InventoryError(
                f"candidate count {retained} outside (0, {n_experts}]"
            )

    candidates = []
    for retained in candidate_list:
        dropped = n_experts - retained
        router_stored_removed = dropped * router_per_row_stored
        router_logical_removed = dropped * router_per_row_logical
        candidates.append(
            {
                "retained_experts": retained,
                "dropped_experts": dropped,
                "native_text_payload_bytes": (
                    non_expert_stored
                    - router_stored_removed
                    + retained * per_index_stored
                ),
                "logical_parameters": (
                    non_expert_logical
                    - router_logical_removed
                    + retained * per_index_logical
                ),
                "router_stored_bytes_removed": router_stored_removed,
                "router_logical_parameters_removed": router_logical_removed,
            }
        )

    # precision alternatives for one expert projection set, evidence-backed:
    # native mxfp4 (U8 nibble-packed, E8M0-style scale, block 32) and bf16
    # dequant (logical_parameters x 2 bytes).  NOTE: native MXFP4 is a
    # distinct format from GGUF Q4_0_ROCMFP4 / _FAST despite coincidentally
    # equal effective bpw; no lossless cross-format conversion is implied.
    precision = {
        "mxfp4_native_u8_packed": {
            "bytes_per_retained_expert_across_moe_layers": per_index_stored,
            "bits_per_logical_parameter_exact": (
                per_index_stored * 8 / per_index_logical
            ),
            "note": (
                "official native format: U8 nibble-packed weights + per-32 "
                "scale bytes; not interchangeable with ROCm FP4 GGUF formats"
            ),
        },
        "bf16": {
            "bytes_per_retained_expert_across_moe_layers": (
                per_index_logical * 2
            ),
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "repo_id": inventory["source"]["repo_id"],
        "revision": inventory["source"]["revision"],
        "native": {
            "retained_experts": n_experts,
            "moe_layers": n_moe_layers,
            "text_stored_bytes": text["stored_bytes"],
            "text_logical_parameters": text["logical_parameters"],
            "non_expert_stored_bytes": non_expert_stored,
            "per_retained_expert_across_moe_layers_stored_bytes": per_index_stored,
            "per_retained_expert_across_moe_layers_logical_parameters": (
                per_index_logical
            ),
            "tensor_records_per_retained_expert_across_moe_layers": (
                records_per_index
            ),
            "per_layer_expert_layout": {
                "stored_bytes": instance_stored,
                "logical_parameters": instance_logical,
                "tensors_per_expert_instance": tensors_per_instance,
            },
            "total_expert_instances": n_experts * n_moe_layers,
            "total_expert_tensor_records": len(expert_records),
            "router_stored_bytes": router_total_stored,
            "router_logical_parameters": router_total_logical,
            "router_stored_bytes_per_retained_expert_across_moe_layers": (
                router_per_row_stored
            ),
            "router_logical_parameters_per_retained_expert_across_moe_layers": (
                router_per_row_logical
            ),
            "per_retained_expert_note": (
                "per_retained_expert_across_moe_layers_* is one expert index "
                "summed over every MoE layer, not one layer-instance; the "
                "single-instance cost is per_layer_expert_layout"
            ),
            "router_note": (
                "dense BF16 gate weight + F32 correction bias per MoE layer; "
                "row-uniform, quantization null; pruning/maps slices these "
                "rows to the retained expert count and prune_candidates "
                "subtracts router_*_removed accordingly"
            ),
        },
        "prune_candidates": candidates,
        "precision_costs": precision,
        "components": {
            name: {
                "stored_bytes": bucket["stored_bytes"],
                "logical_parameters": bucket["logical_parameters"],
                "tensors": bucket["tensors"],
            }
            for name, bucket in comp_totals.items()
        },
    }


# --------------------------------------------------------------------------
# Fetch / archive CLI
# --------------------------------------------------------------------------


def default_client() -> HttpsRangeClient:
    return HttpsRangeClient()


METADATA_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "dflash/config.json",
    "dflash/model.safetensors.index.json",
    "README.md",
    "generation_config.json",
)
WEIGHT_FILES = (
    tuple(f"model_pp0_ep{i}_shard0.safetensors" for i in range(64))
    + ("model_mtp.safetensors", "dflash/dflash_draft_model.safetensors")
)


def fetch_source(out_dir: Path, client: HttpsRangeClient | None = None) -> dict:
    """Archive immutable-URL metadata: file list with checksums, then headers."""
    client = client or default_client()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "headers").mkdir(exist_ok=True)

    api_doc = json.loads(client.read_text(API_URL, what="api"))
    source = {
        "repo_id": REPO_ID,
        "revision": api_doc["sha"],
        "api_url": f"https://huggingface.co/api/models/{REPO_ID}",
        "files": [],
    }
    for sib in api_doc.get("siblings", []):
        entry = {
            "path": sib["rfilename"],
            "url": RESOLVE_URL.format(path=sib["rfilename"]),
            "raw_url": RAW_URL.format(path=sib["rfilename"]),
            "size": sib.get("size"),
            "lfs_sha256": (sib.get("lfs") or {}).get("sha256"),
            "lfs_size": (sib.get("lfs") or {}).get("size"),
        }
        source["files"].append(entry)
    (out_dir / "source.json").write_text(json.dumps(source, indent=2) + "\n")

    for path in METADATA_FILES:
        target = out_dir / "source-metadata" / (path.replace("/", "_"))
        if not (target.exists() and target.stat().st_size > 0):
            text = client.read_text(RAW_URL.format(path=path), what=path)
            target.write_text(text)
    # The HF API only gives lfs_sha256 (null for the non-LFS metadata files
    # the inventory is derived from).  Add a verifiable checksum for those:
    # metadata_sha256 = sha256 over the exact bytes of the archived
    # source-metadata/<path with '/' replaced by '_'> file, computed locally
    # after download.  Rewritten in place so already-archived runs gain the
    # hashes on their next `fetch`.
    by_path = {f["path"]: f for f in source["files"]}
    for path in METADATA_FILES:
        target = out_dir / "source-metadata" / (path.replace("/", "_"))
        entry = by_path.get(path)
        if entry is not None and target.exists():
            entry["metadata_sha256"] = hashlib.sha256(
                target.read_bytes()
            ).hexdigest()
    (out_dir / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    return source


def fetch_headers(
    out_dir: Path,
    client: HttpsRangeClient | None = None,
    only: list[str] | None = None,
    force: bool = False,
) -> list[str]:
    """Archive safetensors headers (8-byte prefix + JSON header only)."""
    client = client or default_client()
    headers_dir = out_dir / "headers"
    headers_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for path in WEIGHT_FILES:
        archive_name = path.replace("/", "_") + ".header.json"
        target = headers_dir / archive_name
        if only and path not in only:
            continue
        if target.exists() and not force:
            continue
        url = RESOLVE_URL.format(path=path)
        header = fetch_safetensors_header(client, path, url)
        header_doc = (
            {**header.tensors, "__metadata__": header.metadata}
            if header.metadata
            else dict(header.tensors)
        )
        payload = {
            "file": path,
            "url": url,
            "revision": REVISION,
            "file_size": header.file_size,
            "header_length": header.header_length,
            "header_bytes_on_wire": 8 + header.header_length,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            # sha256 over json.dumps(header, sort_keys=True,
            # separators=(",", ":")) utf-8 bytes; load_headers re-checks it
            # when present.
            "header_sha256": hashlib.sha256(
                json.dumps(header_doc, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "header": header_doc,
        }
        target.write_text(json.dumps(payload, indent=1) + "\n")
        fetched.append(path)
    return fetched


def load_headers(out_dir: Path) -> dict[str, SafetensorsHeader]:
    headers: dict[str, SafetensorsHeader] = {}
    # cross-check envelopes against the pinned revision and the archived
    # source file list (declared size) when source.json is present
    source_path = out_dir / "source.json"
    files_by_path: dict[str, dict] = {}
    if source_path.exists():
        source = json.loads(source_path.read_text())
        if source.get("revision") != REVISION:
            raise InventoryError(
                f"source.json revision {source.get('revision')!r} != pinned {REVISION!r}"
            )
        files_by_path = {f["path"]: f for f in source.get("files", [])}
    for path in sorted((out_dir / "headers").glob("*.header.json")):
        doc = json.loads(path.read_text())
        if doc.get("revision") != REVISION:
            raise InventoryError(
                f"{doc.get('file')}: envelope revision "
                f"{doc.get('revision')!r} != pinned {REVISION!r}"
            )
        expected_url = RESOLVE_URL.format(path=doc["file"])
        if doc.get("url") != expected_url:
            raise InventoryError(
                f"{doc['file']}: envelope URL is not the pinned immutable URL"
            )
        entry = files_by_path.get(doc["file"])
        if (
            entry is not None
            and entry.get("size") is not None
            and entry["size"] != doc["file_size"]
        ):
            raise InventoryError(
                f"{doc['file']}: envelope file_size {doc['file_size']} != "
                f"source.json declared size {entry['size']}"
            )
        stored_sha = doc.get("header_sha256")
        if stored_sha is not None:
            # same method as fetch_headers: sha256 over
            # json.dumps(header, sort_keys=True, separators=(",", ":")) utf-8
            canonical = json.dumps(
                doc["header"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() != stored_sha:
                raise InventoryError(
                    f"{doc['file']}: archived header_sha256 mismatch"
                )
        raw = json.dumps(doc["header"], separators=(",", ":")).encode("utf-8")
        n = doc["header_length"]
        if len(raw) > n:
            raise HeaderError(
                f"{doc['file']}: re-serialized header exceeds archived N {n}"
            )
        raw = raw + b" " * (n - len(raw))
        header = parse_safetensors_header(
            doc["file"],
            doc["url"],
            struct.pack("<Q", n),
            raw,
            doc["file_size"],
        )
        header.metadata = doc["header"].get("__metadata__")
        headers[header.name] = header
    return headers


def load_source_files(out_dir: Path) -> list[dict]:
    source = json.loads((out_dir / "source.json").read_text())
    if source.get("revision") != REVISION:
        raise InventoryError(
            f"source.json revision {source.get('revision')!r} != pinned {REVISION!r}"
        )
    return source["files"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _cli_fetch(args, out_dir: Path) -> int:
    client = default_client()
    fetch_source(out_dir, client)
    fetched = fetch_headers(out_dir, client, only=args.only, force=args.force)
    total = len(WEIGHT_FILES)
    print(f"fetch: source metadata archived; {len(fetched)}/{total} headers fetched")
    return 0


def _cli_inventory(args, out_dir: Path) -> int:
    config = json.loads((out_dir / "source-metadata" / "config.json").read_text())
    index = json.loads(
        (out_dir / "source-metadata" / "model.safetensors.index.json").read_text()
    )
    dflash_config = json.loads(
        (out_dir / "source-metadata" / "dflash_config.json").read_text()
    )
    dflash_index = json.loads(
        (out_dir / "source-metadata" / "dflash_model.safetensors.index.json").read_text()
    )
    headers = load_headers(out_dir)
    source_files = load_source_files(out_dir)
    inventory = build_inventory(
        config, index, dflash_config, dflash_index, headers, source_files
    )
    target = out_dir / "inventory.json"
    target.write_text(json.dumps(inventory, indent=1) + "\n")
    grand = inventory["totals"]["grand"]
    print(
        f"inventory: {grand['tensors']} tensors, "
        f"{grand['stored_bytes']} stored bytes, "
        f"{grand['logical_parameters']} logical parameters -> {target}"
    )
    return 0


def _cli_memory(args, out_dir: Path) -> int:
    inventory = json.loads((out_dir / "inventory.json").read_text())
    report = memory_report(
        inventory, candidate_counts=args.candidates or None
    )
    target = out_dir / "memory-report.json"
    target.write_text(json.dumps(report, indent=1) + "\n")
    for cand in report["prune_candidates"]:
        print(
            f"memory: {cand['retained_experts']} experts -> "
            f"{cand['native_text_payload_bytes']} text payload bytes "
            f"({cand['router_stored_bytes_removed']} router bytes removed)"
        )
    print(f"memory: full report -> {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mimo-halo-inventory",
        description="Bounded safetensors header inventory for MiMo-V2.6-Flash-RL",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="manifest directory (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="archive source metadata + safetensors headers")
    p_fetch.add_argument("--only", nargs="*", default=None, help="restrict to these files")
    p_fetch.add_argument("--force", action="store_true", help="refetch existing headers")
    p_fetch.set_defaults(func=_cli_fetch)

    p_inv = sub.add_parser("inventory", help="build inventory.json from archived headers")
    p_inv.set_defaults(func=_cli_inventory)

    p_mem = sub.add_parser("memory", help="build memory/prune-candidate report")
    p_mem.add_argument(
        "--candidates",
        type=int,
        nargs="*",
        default=None,
        help=(
            "retained-expert counts to report (each must be in "
            "(0, original_experts_per_layer]); default: the original count "
            "plus the built-in candidate list filtered to <= original"
        ),
    )
    p_mem.set_defaults(func=_cli_memory)

    args = parser.parse_args(argv)
    out_dir: Path = args.out
    return args.func(args, out_dir)


if __name__ == "__main__":
    sys.exit(main())
