"""Candidate construction: original Xiaomi weights + prune map + recipe
-> sharded safetensors checkpoint with full artifact sidecars.

Pipeline (all phases fail closed, :mod:`mimo_halo.build.errors`):

1. **Verify inputs** - inventory revision must be the pinned source
   revision; the source root must be the verified revision directory and
   every input path must resolve inside it; the prune map must be an
   external-selection map (shape-only placeholder maps are refused) with
   old->new expert id maps consistent with the inventory.
2. **Validate the recipe** against the sweep contract
   (configs/experiments/compression-sweep.json): allocation-table rows are
   internally exact (``class_bytes == units * bytes_per_unit``), format /
   ``second_gen`` / ``bit_exact`` must agree, the ``second_gen_applied``
   ledger must name exactly the classes that re-encode, and the per-layer
   Q3 distribution invariant must hold.
3. **Plan every tensor**: text tensors dispatch by allocation class
   (bit-exact copy, exact dense router row slice, or second-gen affine
   encode), non-text components copy bit-exact, auxiliaries under source
   subdirectories are excluded and recorded byte-exact. Static byte
   accounting must close exactly against ``allocation_table`` before a
   single byte is written.
4. **Write** output shards streamed with bounded memory, then re-read every
   written tensor (independent read-back): bit-exact tensors must reproduce
   the source digest exactly; second-gen tensors carry per-tensor
   quant-dequant error statistics measured against the BF16-decoded source
   values (compute-only).
5. **Sidecars**: quant assignment, build report and the schema-shaped
   candidate record are written, then ``artifacts.create_artifact`` adds
   the provenance sidecars (pinned source revision, expert map, quant
   assignment, dataset/calibration refs, honest conversion-commit field)
   and ``verify_artifact`` must report ``verified``.

The candidate artifact is FLAT (artifacts.py payload names are basename-
flat): source files under subdirectories (dflash/, audio_tokenizer/,
assets/) are excluded from the artifact and recorded byte-exact in the
build report; no text tensor lives in a subdir shard (verified against the
pinned inventory, enforced fail-closed here).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .. import artifacts
from ..models.inventory import (
    DTYPE_ITEMSIZE,
    MAX_HEADER_BYTES,
    REPO_ID,
    REVISION,
    HeaderError,
    SafetensorsHeader,
    parse_safetensors_header,
)
from ..pruning.maps import PruneMapError, load_inventory
from .errors import (
    AccountingError,
    BuildError,
    PackingEvidenceError,
    PassthroughQuantError,
    RequantError,
    SourcePathError,
)
from .quantize import (
    AFFINE_CHUNK_BYTES,
    COPY_CHUNK_BYTES,
    FORMATS,
    affine_expected_bytes,
    decode_bf16,
    decode_mxfp4,
    encode_affine,
    require_numpy,
)

# Schema contract (schemas/compression-candidate.schema.json); tests assert
# these tuples against the schema file so drift fails loudly.
RECIPE_KEYS = (
    "reap_percent_pruned",
    "rounding_rule",
    "original_experts_per_layer",
    "retained_experts_per_layer",
    "retained_total_experts",
    "pruned_total_experts",
    "moe_layers",
    "allocation_table",
    "expert_precision",
    "second_gen_applied",
)
SCHEMA_CLASSES = (
    "experts_native",
    "experts_second_gen",
    "router",
    "attention_qkv",
    "attention_qkv_scales",
    "attention_o_proj",
    "attention_sinks",
    "dense_ffn",
    "dense_ffn_scales",
    "embeddings_lm_head",
    "norms",
)
ALLOCATION_ROW_KEYS = (
    "class",
    "format",
    "units",
    "unit_label",
    "bytes_per_unit",
    "class_bytes",
    "second_gen",
    "bit_exact",
    "source_field",
)
LEDGER_ENTRY_KEYS = (
    "class",
    "format",
    "scope",
    "bytes_before",
    "bytes_after",
    "reason",
)
EXPERT_PRECISION_KEYS = (
    "native_mxfp4_instances",
    "q3_instances",
    "instance_bytes_native",
    "instance_bytes_q3",
    "q3_percent",
    "per_layer_distribution",
    "assignment_rule",
)
DISTRIBUTION_KEYS = ("base_q3_per_layer", "extra_q3_layers", "extra_q3_per_layer")
UNIT_LABELS = ("expert_instance", "retained_index", "tensor")
REAP_PERCENTS = (25, 30, 35, 40, 45)
TARGET_RESIDENT_BYTES = 90 * (1 << 30)  # 96636764160
TOLERANCE_BYTES = 1932735283  # floor(2% of target); window matches the config
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
#: The single routed-expert-count scalar of the pinned config (only field
#: that contradicts row-sliced routers when left at 256).
ROUTED_EXPERT_COUNT_KEYS = ("n_routed_experts",)
#: Replacement for a store_dtype claim of "mxfp4" on a build whose expert
#: tensors are second-gen affine codes: never implies native MXFP4.
SECOND_GEN_STORE_DTYPE = "second_gen_affine"
QUANT_ASSIGNMENT_NAME = "quant_assignment.json"
BUILD_REPORT_NAME = "build_report.json"
CANDIDATE_RECORD_NAME = "candidate.json"

_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)"
    r"\.(weight|weight_scale)$"
)
_ROUTER_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias)$"
)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_digest(obj) -> str:
    return hashlib.sha256(artifacts.canonical_json_bytes(obj)).hexdigest()


def _read_exact(handle, length: int, what: str) -> bytes:
    chunks = []
    remaining = length
    while remaining > 0:
        chunk = handle.read(remaining)
        if not chunk:
            raise BuildError(f"truncated read for {what}: wanted {length} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_safetensors_header(path: str | Path) -> SafetensorsHeader:
    """Parse a local safetensors header with the inventory's validator."""
    path = str(path)
    file_size = os.path.getsize(path)
    with open(path, "rb") as handle:
        first8 = _read_exact(handle, 8, f"{path}: prefix")
        (n,) = struct.unpack("<Q", first8)
        if n < 8 or n > MAX_HEADER_BYTES:
            raise BuildError(f"{path}: implausible safetensors header length {n}")
        body = _read_exact(handle, n, f"{path}: header")
    try:
        return parse_safetensors_header(os.path.basename(path), path, first8, body, file_size)
    except HeaderError as exc:
        raise BuildError(f"safetensors header rejected: {exc}") from exc


def _resolve_inside(root_real: Path, rel: str, label: str) -> Path:
    """Resolve ``rel`` under the verified source root; refuse any escape."""
    if not isinstance(rel, str) or not rel:
        raise SourcePathError(f"{label}: empty input path")
    if os.path.isabs(rel) or rel.startswith("/") or "\\" in rel:
        raise SourcePathError(
            f"{label}: input path must be relative to the verified source "
            f"root: {rel!r}"
        )
    parts = Path(rel).parts
    if ".." in parts or parts == ():
        raise SourcePathError(
            f"{label}: input path escapes the verified source root: {rel!r}"
        )
    candidate = root_real / rel
    resolved = Path(os.path.realpath(candidate))
    if resolved != root_real and root_real not in resolved.parents:
        raise SourcePathError(
            f"{label}: input path escapes the verified source root: {rel!r}"
        )
    if not resolved.is_file():
        raise SourcePathError(f"{label}: input file missing on disk: {rel!r}")
    if os.path.islink(candidate):
        raise SourcePathError(f"{label}: input file must not be a symlink: {rel!r}")
    return resolved


def _classify_text_tensor(name: str) -> str:
    """Map a text-component tensor name to its allocation class.

    Grounded against the pinned inventory: these patterns reproduce every
    class-row unit count of configs/experiments/compression-sweep.json with
    zero unclassified text tensors.
    """
    if _EXPERT_RE.match(name) or _ROUTER_RE.match(name):
        raise BuildError(
            f"internal: expert/router tensor {name!r} routed to text classification"
        )
    if name.endswith(".self_attn.qkv_proj.weight"):
        return "attention_qkv"
    if name.endswith(".self_attn.qkv_proj.weight_scale_inv"):
        return "attention_qkv_scales"
    if name.endswith(".self_attn.o_proj.weight"):
        return "attention_o_proj"
    if "attention_sink" in name:
        return "attention_sinks"
    if name.endswith(".weight_scale_inv") and ".mlp." in name:
        return "dense_ffn_scales"
    if name.endswith(".weight") and ".mlp." in name:
        return "dense_ffn"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "embeddings_lm_head"
    if "norm" in name.lower() and name.endswith(".weight"):
        return "norms"
    raise BuildError(
        f"text tensor matches no allocation class (fail closed, never "
        f"silently dropped): {name!r}"
    )


# ---------------------------------------------------------------------------
# Recipe validation (static, before any write)
# ---------------------------------------------------------------------------


def _require_keys(obj, keys, where, exact=False):
    if not isinstance(obj, dict):
        raise BuildError(f"{where} must be a JSON object")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise BuildError(f"{where} is missing keys {missing}")
    if exact:
        extra = sorted(set(obj) - set(keys))
        if extra:
            raise BuildError(f"{where} has unknown keys {extra}")


# The schema pins original_experts_per_layer to a const (256); sweep-config
# candidate entries may omit it, so it defaults to the const and is ALWAYS
# cross-checked against the inventory architecture below (a non-256 source
# without the explicit key fails closed).
RECIPE_REQUIRED_KEYS = tuple(k for k in RECIPE_KEYS if k != "original_experts_per_layer")


def _validate_recipe(recipe: dict, inv: dict) -> dict:
    """Validate the candidate entry against the sweep contract.

    Returns the parsed rows/ledger/expert-precision blocks. Refusals:
    structural contradictions -> :class:`BuildError`; passthrough/second-gen
    contradictions -> :class:`PassthroughQuantError`; byte arithmetic that
    cannot close -> :class:`AccountingError`.
    """
    _require_keys(recipe, RECIPE_REQUIRED_KEYS, "recipe")
    if recipe["reap_percent_pruned"] not in REAP_PERCENTS:
        raise BuildError(
            f"recipe reap_percent_pruned {recipe['reap_percent_pruned']!r} not in {REAP_PERCENTS}"
        )
    arch = inv["architecture"]
    original = arch["original_experts_per_layer"]
    moe_layers = arch["moe_layers"]
    declared_original = recipe.get("original_experts_per_layer", 256)
    if not _is_int(declared_original) or declared_original != original:
        raise BuildError(
            f"recipe original_experts_per_layer {declared_original!r} "
            f"!= inventory architecture {original}"
        )
    retained = recipe["retained_experts_per_layer"]
    if not _is_int(retained) or not 1 <= retained <= original:
        raise BuildError(f"recipe retained_experts_per_layer out of range: {retained!r}")
    # round_half_up(original * pct / 100) == floor(original*pct/100 + 0.5)
    expected_retained = original - math.floor(
        original * recipe["reap_percent_pruned"] / 100 + 0.5
    )
    if retained != expected_retained:
        raise BuildError(
            f"recipe retained count {retained} contradicts the rounding rule "
            f"({original} - round_half_up({original} * {recipe['reap_percent_pruned']} / 100) "
            f"= {expected_retained})"
        )
    if not isinstance(recipe["rounding_rule"], str) or not recipe["rounding_rule"]:
        raise BuildError("recipe rounding_rule must be a non-empty string")
    if not _is_int(recipe["moe_layers"]) or recipe["moe_layers"] != len(moe_layers):
        raise BuildError(
            f"recipe moe_layers {recipe['moe_layers']!r} != inventory MoE layer count "
            f"{len(moe_layers)}"
        )
    if recipe["retained_total_experts"] != retained * len(moe_layers):
        raise BuildError(
            f"recipe retained_total_experts {recipe['retained_total_experts']} != "
            f"{retained} x {len(moe_layers)}"
        )
    expected_pruned = (original - retained) * len(moe_layers)
    if recipe["pruned_total_experts"] != expected_pruned:
        raise BuildError(
            f"recipe pruned_total_experts {recipe['pruned_total_experts']} != "
            f"{expected_pruned}"
        )

    # Allocation table.
    table = recipe["allocation_table"]
    if not isinstance(table, list) or not table:
        raise BuildError("recipe allocation_table must be a non-empty list")
    rows: dict[str, dict] = {}
    for raw in table:
        _require_keys(raw, ALLOCATION_ROW_KEYS, "allocation row", exact=True)
        cls = raw["class"]
        if cls not in SCHEMA_CLASSES:
            raise BuildError(f"allocation class {cls!r} not in the schema enum")
        if cls in rows:
            raise BuildError(f"duplicate allocation row for class {cls!r}")
        fmt = raw["format"]
        if fmt not in FORMATS:
            raise BuildError(f"unknown allocation format {fmt!r}")
        for key in ("units", "bytes_per_unit", "class_bytes"):
            if not _is_int(raw[key]) or raw[key] < 0:
                raise BuildError(f"allocation row {cls}.{key} must be a non-negative int")
        if raw["unit_label"] not in UNIT_LABELS:
            raise BuildError(f"allocation row {cls}: unknown unit_label {raw['unit_label']!r}")
        if not isinstance(raw["second_gen"], bool) or not isinstance(raw["bit_exact"], bool):
            raise BuildError(f"allocation row {cls}: second_gen/bit_exact must be booleans")
        if not isinstance(raw["source_field"], str) or not raw["source_field"]:
            raise BuildError(f"allocation row {cls}: source_field must be a non-empty string")
        if raw["class_bytes"] != raw["units"] * raw["bytes_per_unit"]:
            raise AccountingError(
                f"byte accounting does not close for class {cls!r}: class_bytes "
                f"{raw['class_bytes']} != units {raw['units']} x bytes_per_unit "
                f"{raw['bytes_per_unit']}"
            )
        # Format decides what dispatch does; marks must agree with it.
        is_affine = FORMATS[fmt]["mode"] == "affine"
        if is_affine and (raw["second_gen"] is not True or raw["bit_exact"] is not False):
            raise PassthroughQuantError(
                f"second-gen quant routed at recipe class {cls!r} marked passthrough: "
                f"format {fmt!r} is affine but the row records "
                f"second_gen={raw['second_gen']}, bit_exact={raw['bit_exact']}"
            )
        if not is_affine and (raw["second_gen"] is not False or raw["bit_exact"] is not True):
            raise PassthroughQuantError(
                f"recipe class {cls!r} claims second_gen={raw['second_gen']} on the "
                f"passthrough format {fmt!r} (re-encoding a passthrough class is refused)"
            )
        rows[cls] = raw

    # second_gen_applied ledger: exhaustive, consistent with the table.
    ledger = recipe["second_gen_applied"]
    if not isinstance(ledger, list):
        raise BuildError("recipe second_gen_applied must be a list")
    ledger_by_class: dict[str, dict] = {}
    for entry in ledger:
        _require_keys(entry, LEDGER_ENTRY_KEYS, "second_gen_applied entry", exact=True)
        cls = entry["class"]
        if cls not in rows:
            raise BuildError(
                f"second_gen_applied entry names unknown allocation class {cls!r}"
            )
        if cls in ledger_by_class:
            raise BuildError(f"duplicate second_gen_applied entry for {cls!r}")
        row = rows[cls]
        if row["second_gen"] is not True:
            raise PassthroughQuantError(
                f"second_gen_applied ledger names recipe class {cls!r} marked "
                f"passthrough (second_gen=false): second-gen quant on a passthrough "
                "class is refused"
            )
        if entry["format"] != row["format"]:
            raise BuildError(
                f"second_gen_applied entry for {cls!r} records format "
                f"{entry['format']!r} but the allocation row says {row['format']!r}"
            )
        for key in ("bytes_before", "bytes_after"):
            if not _is_int(entry[key]) or entry[key] < 0:
                raise BuildError(f"second_gen_applied {cls}.{key} must be a non-negative int")
        if entry["bytes_after"] != row["class_bytes"]:
            raise AccountingError(
                f"byte accounting does not close for class {cls!r}: ledger "
                f"bytes_after {entry['bytes_after']} != allocation class_bytes "
                f"{row['class_bytes']}"
            )
        for key in ("scope", "reason"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise BuildError(f"second_gen_applied {cls}.{key} must be a non-empty string")
        ledger_by_class[cls] = entry
    # Presence of a ledger entry for every affine dispatch is checked at
    # dispatch time (candidate.py require_ledger), where the source kind is
    # known and the refusal wording can name the mxfp4 requantization case
    # exactly; a ledger entry naming a passthrough class is refused above.

    # Expert precision block.
    precision = recipe["expert_precision"]
    _require_keys(precision, EXPERT_PRECISION_KEYS, "recipe expert_precision", exact=True)
    native_n = precision["native_mxfp4_instances"]
    q3_n = precision["q3_instances"]
    if not _is_int(native_n) or not _is_int(q3_n) or native_n < 0 or q3_n < 0:
        raise BuildError("expert_precision instance counts must be non-negative ints")
    if native_n + q3_n != recipe["retained_total_experts"]:
        raise BuildError(
            f"expert_precision native {native_n} + q3 {q3_n} != retained_total "
            f"{recipe['retained_total_experts']}"
        )
    for key in ("instance_bytes_native", "instance_bytes_q3"):
        if not _is_int(precision[key]) or precision[key] <= 0:
            raise BuildError(f"expert_precision {key} must be a positive int")
    if not isinstance(precision["q3_percent"], (int, float)) or isinstance(
        precision["q3_percent"], bool
    ):
        raise BuildError("expert_precision q3_percent must be a number")
    if not 0 <= precision["q3_percent"] <= 100:
        raise BuildError(f"expert_precision q3_percent out of range: {precision['q3_percent']}")
    if not isinstance(precision["assignment_rule"], str) or not precision["assignment_rule"]:
        raise BuildError("expert_precision assignment_rule must be a non-empty string")

    distribution = precision["per_layer_distribution"]
    _require_keys(distribution, DISTRIBUTION_KEYS, "per_layer_distribution", exact=True)
    base = distribution["base_q3_per_layer"]
    extra_layers = distribution["extra_q3_layers"]
    extra_per = distribution["extra_q3_per_layer"]
    if not _is_int(base) or base < 0:
        raise BuildError("per_layer_distribution base_q3_per_layer must be a non-negative int")
    if not isinstance(extra_layers, list) or not all(
        _is_int(layer) for layer in extra_layers
    ):
        raise BuildError("per_layer_distribution extra_q3_layers must be a list of ints")
    if len(set(extra_layers)) != len(extra_layers):
        raise BuildError("per_layer_distribution extra_q3_layers has duplicates")
    if not _is_int(extra_per) or extra_per < 0:
        raise BuildError("per_layer_distribution extra_q3_per_layer must be a non-negative int")
    moe_set = set(moe_layers)
    for layer in extra_layers:
        if layer not in moe_set:
            raise BuildError(
                f"per_layer_distribution extra layer {layer} is not a MoE layer"
            )
    if extra_layers and extra_per != base + 1:
        raise BuildError(
            f"per_layer_distribution invariant: extra_q3_per_layer {extra_per} "
            f"must be base+1 ({base + 1}) when extra layers are listed"
        )
    if not extra_layers and extra_per not in (0, base):
        raise BuildError(
            f"per_layer_distribution invariant: extra_q3_per_layer {extra_per} "
            "with an empty layer list must be 0 or the base"
        )
    derived_q3 = base * len(moe_layers) + len(extra_layers) * (extra_per - base)
    if derived_q3 != q3_n:
        raise BuildError(
            f"per_layer_distribution produces {derived_q3} Q3 instances but "
            f"expert_precision.q3_instances says {q3_n}"
        )
    if q3_n > 0 and "experts_second_gen" not in rows:
        raise BuildError(
            "recipe plans Q3 instances but has no experts_second_gen allocation row"
        )
    if native_n > 0 and "experts_native" not in rows:
        raise BuildError(
            "recipe plans native instances but has no experts_native allocation row"
        )
    return {"rows": rows, "ledger": ledger_by_class, "precision": precision,
            "distribution": distribution}


def _load_prune_map(path: str | Path, inv: dict) -> dict:
    path = Path(path)
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except OSError as exc:
        raise BuildError(f"cannot read prune map {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"prune map {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BuildError("prune map must be a JSON object")
    if data.get("schema_version") != 1 or data.get("kind") != "prune_map":
        raise BuildError("input is not a schema_version=1 prune_map document")
    source = data.get("source") or {}
    if source.get("revision") != REVISION:
        raise BuildError(
            f"prune map source revision {source.get('revision')!r} is not the "
            f"pinned source revision {REVISION}"
        )
    mode = data.get("mode")
    if mode == "shape_only":
        raise BuildError(
            "shape-only placeholder selection is not a quality map; construction "
            "requires an external selection (--selection of mimo_halo.pruning.maps)"
        )
    if not isinstance(data.get("layers"), list) or not data["layers"]:
        raise BuildError("prune map has no layers")
    arch = inv["architecture"]
    original = arch["original_experts_per_layer"]
    moe_set = set(arch["moe_layers"])
    retained_by_layer: dict[int, list[int]] = {}
    old_to_new_by_layer: dict[int, dict[int, int | None]] = {}
    seen: set[int] = set()
    retained_count = data.get("retained_count")
    if not _is_int(retained_count):
        raise BuildError("prune map retained_count missing or not an int")
    for entry in data["layers"]:
        if not isinstance(entry, dict):
            raise BuildError("prune map layer entries must be objects")
        layer = entry.get("layer")
        if not _is_int(layer) or layer not in moe_set or layer in seen:
            raise BuildError(f"prune map has a bad or duplicate layer {layer!r}")
        seen.add(layer)
        retained = entry.get("retained")
        old_to_new = entry.get("old_to_new")
        pruned = entry.get("pruned")
        if not isinstance(retained, list) or not retained:
            raise BuildError(f"prune map layer {layer}: retained must be a non-empty list")
        if len(retained) != retained_count:
            raise BuildError(
                f"prune map layer {layer} retains {len(retained)} experts, map "
                f"retained_count is {retained_count}"
            )
        if not isinstance(old_to_new, dict) or len(old_to_new) != original:
            raise BuildError(
                f"prune map layer {layer}: old_to_new must cover all {original} ids"
            )
        expected = {str(i): None for i in range(original)}
        for new_id, old_id in enumerate(retained):
            if not _is_int(old_id) or not 0 <= old_id < original:
                raise BuildError(f"prune map layer {layer}: bad retained id {old_id!r}")
            if old_to_new.get(str(old_id)) != new_id:
                raise BuildError(
                    f"prune map layer {layer}: old_to_new[{old_id}] must equal the "
                    "retained position (new-id order)"
                )
            expected[str(old_id)] = new_id
        if old_to_new != expected:
            raise BuildError(
                f"prune map layer {layer}: old_to_new is not the exact "
                "retained->0..N-1 / pruned->null mapping"
            )
        if not isinstance(pruned, list) or sorted(pruned) != sorted(
            i for i in range(original) if i not in set(retained)
        ):
            raise BuildError(f"prune map layer {layer}: pruned list is inconsistent")
        retained_by_layer[layer] = list(retained)
        old_to_new_by_layer[layer] = {
            int(k): v for k, v in old_to_new.items()
        }
    if seen != moe_set:
        raise BuildError(f"prune map is missing MoE layers {sorted(moe_set - seen)}")
    provenance = data.get("provenance") or {}
    return {
        "document": data,
        "digest": _sha256_bytes(raw),
        "mode": mode,
        "quality_map": bool(provenance.get("quality_map", False)),
        "provenance": provenance,
        "retained_count": retained_count,
        "retained_by_layer": retained_by_layer,
        "old_to_new_by_layer": old_to_new_by_layer,
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass
class PlannedTensor:
    out_name: str
    out_dtype: str
    out_shape: list
    out_bytes: int
    action: str  # copy | row_slice | affine | aux_copy
    cls: str  # allocation class ("" for aux-unbudgeted copies)
    fmt: str
    component: str
    src_shard: str
    src_name: str
    src_off: int
    src_len: int
    rows: list | None = None
    row_bytes: int | None = None
    src_shape: list | None = None
    src_scale_off: int | None = None
    src_scale_len: int | None = None
    params: int | None = None
    error: dict | None = None
    input_sha256: str | None = None
    output_sha256: str | None = None

    @property
    def bit_exact(self) -> bool:
        return self.action in ("copy", "row_slice", "aux_copy")


@dataclass
class Plan:
    shards: dict[str, list[PlannedTensor]] = field(default_factory=dict)
    shard_payloads: dict[str, int] = field(default_factory=dict)
    shard_data_start: dict[str, int] = field(default_factory=dict)
    shard_source_path: dict[str, str] = field(default_factory=dict)
    classes: dict[str, dict] = field(default_factory=dict)
    aux_tensors: int = 0
    aux_bytes: int = 0
    metadata: list = field(default_factory=list)
    excluded_files: list = field(default_factory=list)
    excluded_components: dict = field(default_factory=dict)
    index_doc: dict | None = None
    index_note: str | None = None
    router_units: int = 0
    floor_bytes: int = 0
    dispatch: dict = field(default_factory=dict)
    config_adaptation: dict = field(default_factory=dict)
    expert_second_gen: bool = False
    total_text_bytes: int = 0
    out_names_seen: set = field(default_factory=set)


def _plan_output_shape_affine(logical_shape: list, bits: int, group: int):
    rows_cols = logical_shape if len(logical_shape) == 2 else [1] + list(logical_shape)
    if len(logical_shape) > 2:
        raise BuildError(
            f"affine dispatch supports rank-1/2 tensors, got shape {logical_shape}"
        )
    rows, cols = rows_cols[0], rows_cols[1]
    codes_shape = [cols * bits // 8] if len(logical_shape) == 1 else [rows, cols * bits // 8]
    groups_per_row = cols // group
    scale_shape = (
        [groups_per_row, 2]
        if len(logical_shape) == 1
        else [rows, groups_per_row, 2]
    )
    return rows, cols, codes_shape, scale_shape


def _validate_expert_order(
    expert_order: dict | None, retained_by_layer: dict[int, list[int]], original: int
) -> None:
    """Require an exact original-ID permutation for each retained MoE layer."""
    if expert_order is None:
        return
    if not isinstance(expert_order, dict):
        raise BuildError("expert_order must be a layer-to-list object")
    expected = set(retained_by_layer)
    for layer in expert_order:
        if not _is_int(layer):
            raise BuildError(f"expert_order layer key {layer!r} must be an integer")
    missing = sorted(expected - expert_order.keys())
    unexpected = sorted(expert_order.keys() - expected)
    if missing:
        raise BuildError(f"expert_order is missing MoE layer {missing[0]}")
    if unexpected:
        raise BuildError(f"expert_order has unexpected MoE layer {unexpected[0]}")
    for layer, order in expert_order.items():
        if not isinstance(order, list):
            raise BuildError(f"expert_order[{layer}] must be a list of original expert ids")
        if any(not _is_int(expert) or not 0 <= expert < original for expert in order):
            raise BuildError(
                f"expert_order[{layer}] contains an invalid original expert id "
                f"(expected integers in [0, {original}))"
            )
        if sorted(order) != sorted(retained_by_layer[layer]):
            raise BuildError(
                f"expert_order[{layer}] must be a permutation of the layer's "
                "retained expert ids"
            )


def _plan_build(
    inv: dict,
    pmap: dict,
    recipe: dict,
    parsed: dict,
    root_real: Path,
    expert_order: dict | None,
) -> Plan:
    rows = parsed["rows"]
    ledger = parsed["ledger"]
    precision = parsed["precision"]
    distribution = parsed["distribution"]
    plan = Plan()

    # Mixed recipes take the first experts in the supplied precision order.
    # When every retained expert is Q3, relative order cannot affect
    # precision; without an order, record original IDs without implying salience.
    retained_by_layer = pmap["retained_by_layer"]
    _validate_expert_order(
        expert_order, retained_by_layer, inv["architecture"]["original_experts_per_layer"]
    )
    if (
        0 < precision["q3_instances"] < recipe["retained_total_experts"]
        and expert_order is None
    ):
        raise BuildError("expert_order is required for a mixed native/Q3 recipe")
    base = distribution["base_q3_per_layer"]
    extra_layers = set(distribution["extra_q3_layers"])
    extra_per = distribution["extra_q3_per_layer"]
    q3_by_layer: dict[int, list[int]] = {}
    if precision["q3_instances"] == 0:
        order_basis = "no Q3 expert instances; all survivors retain native MXFP4"
    elif precision["native_mxfp4_instances"] == 0:
        order_basis = (
            "all retained experts Q3; caller-supplied order recorded "
            "(precision order irrelevant)"
            if expert_order is not None
            else "all retained experts Q3; ascending original IDs "
                 "(precision order irrelevant)"
        )
    else:
        order_basis = (
            "caller-supplied precision order (ascending REAP salience, when used, "
            "is a PROXY, not measured quantization sensitivity or a quality winner)"
        )
    for layer in sorted(retained_by_layer):
        n_q3 = extra_per if layer in extra_layers else base
        order = (
            expert_order[layer]
            if expert_order is not None
            else sorted(retained_by_layer[layer])
        )
        if n_q3 > len(order):
            raise BuildError(
                f"layer {layer}: distribution asks for {n_q3} Q3 instances but "
                f"only {len(order)} experts are retained"
            )
        q3_by_layer[layer] = order[:n_q3]
    total_q3 = sum(len(v) for v in q3_by_layer.values())
    if total_q3 != precision["q3_instances"]:
        raise BuildError(
            f"per-layer dispatch produced {total_q3} Q3 instances, recipe says "
            f"{precision['q3_instances']}"
        )
    total_retained = sum(len(v) for v in retained_by_layer.values())
    if total_retained - total_q3 != precision["native_mxfp4_instances"]:
        raise BuildError(
            f"per-layer dispatch produced {total_retained - total_q3} native "
            f"instances, recipe says {precision['native_mxfp4_instances']}"
        )
    q3_sets = {layer: set(ids) for layer, ids in q3_by_layer.items()}

    # Source files: every file path verified against the source root.
    files_by_path = {}
    for entry in inv["source"]["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise BuildError("inventory source.files entries must carry a path")
        files_by_path[entry["path"]] = entry
        _resolve_inside(root_real, entry["path"], "source file")

    # Group inventory tensors by shard; open every root shard header once.
    shard_tensors: dict[str, list[dict]] = {}
    for tensor in inv["tensors"]:
        shard_tensors.setdefault(tensor["shard"], []).append(tensor)

    root_shards: dict[str, SafetensorsHeader] = {}
    for shard_rel in sorted(shard_tensors):
        if "/" in shard_rel:
            # Auxiliary subdir shard: excluded with its file record.
            entry = files_by_path.get(shard_rel)
            if entry is None:
                raise SourcePathError(
                    f"inventory shard {shard_rel!r} has no source.files record"
                )
            bad = [t["name"] for t in shard_tensors[shard_rel] if t.get("component") == "text"]
            if bad:
                raise BuildError(
                    f"text tensors live in excluded subdir shard {shard_rel!r}: {bad[:3]}"
                )
            plan.excluded_components.setdefault(
                shard_rel, {"tensors": 0, "stored_bytes": 0, "components": set()}
            )
            bucket = plan.excluded_components[shard_rel]
            for tensor in shard_tensors[shard_rel]:
                bucket["tensors"] += 1
                bucket["stored_bytes"] += tensor["stored_bytes"]
                bucket["components"].add(tensor.get("component") or "(no component)")
            plan.excluded_files.append(
                {
                    "path": shard_rel,
                    "size_bytes": entry.get("size"),
                    "lfs_sha256": entry.get("lfs_sha256"),
                }
            )
            continue
        path = _resolve_inside(root_real, shard_rel, "source shard")
        header = read_safetensors_header(path)
        declared = files_by_path.get(shard_rel, {})
        if "size" in declared and declared["size"] != os.path.getsize(path):
            raise SourcePathError(
                f"source shard {shard_rel!r} size {os.path.getsize(path)} != "
                f"pinned inventory size {declared['size']}"
            )
        root_shards[shard_rel] = header
        plan.shard_data_start[shard_rel] = 8 + header.header_length
        plan.shard_source_path[shard_rel] = str(path)

    # Source files classification: shards / index / subdir / root metadata.
    for rel, entry in sorted(files_by_path.items()):
        if rel in shard_tensors:
            continue
        if rel == INDEX_NAME:
            continue  # regenerated from the build plan, never copied stale
        if "/" in rel:
            plan.excluded_files.append(
                {
                    "path": rel,
                    "size_bytes": entry.get("size"),
                    "lfs_sha256": entry.get("lfs_sha256"),
                }
            )
            continue
        if rel.endswith(".safetensors"):
            raise BuildError(
                f"root safetensors file {rel!r} is not covered by the inventory "
                "(every root weight byte must be accounted for)"
            )
        plan.metadata.append(
            {
                "path": rel,
                "declared_size": entry.get("size"),
                "declared_lfs_sha256": entry.get("lfs_sha256"),
                # config.json is adapted to the built shapes (kept counts +
                # honest quant metadata); everything else is verbatim.
                "adapt": rel == CONFIG_NAME,
            }
        )

    def add_tensor(tensor: PlannedTensor) -> None:
        if tensor.out_name in plan.out_names_seen:
            raise BuildError(f"duplicate output tensor name {tensor.out_name!r}")
        plan.out_names_seen.add(tensor.out_name)
        plan.shards.setdefault(tensor.src_shard, []).append(tensor)
        plan.shard_payloads[tensor.src_shard] = (
            plan.shard_payloads.get(tensor.src_shard, 0) + tensor.out_bytes
        )

    def bump(cls: str, units: int, nbytes: int) -> None:
        bucket = plan.classes.setdefault(
            cls, {"planned_units": 0, "planned_bytes": 0, "tensor_count": 0}
        )
        bucket["planned_units"] += units
        bucket["planned_bytes"] += nbytes

    def require_ledger(cls: str, source_is_mxfp4: bool) -> None:
        if cls in ledger:
            return
        if source_is_mxfp4:
            raise RequantError(
                f"requantization attempt on an mxfp4 tensor without an explicit "
                f"recorded override: class {cls!r} dispatches an affine format but "
                "second_gen_applied does not record it"
            )
        raise RequantError(
            f"second-generation quantisation of class {cls!r} without an explicit "
            "recorded second_gen_applied entry"
        )

    def planned_copy(
        *, out_name, dtype, shape, nbytes, cls, fmt, component, shard, name, off, length
    ) -> PlannedTensor:
        return PlannedTensor(
            out_name=out_name,
            out_dtype=dtype,
            out_shape=list(shape),
            out_bytes=nbytes,
            action="copy",
            cls=cls,
            fmt=fmt,
            component=component,
            src_shard=shard,
            src_name=name,
            src_off=off,
            src_len=length,
        )

    # Index weight_map shard membership (for the regenerated index).
    index_membership_values: set[str] = set()
    index_path = root_real / INDEX_NAME
    index_doc_in = None
    if index_path.is_file():
        try:
            index_doc_in = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BuildError(f"cannot parse source {INDEX_NAME}: {exc}") from exc
        weight_map = index_doc_in.get("weight_map")
        if not isinstance(weight_map, dict):
            raise BuildError(f"source {INDEX_NAME} has no weight_map")
        index_membership_values = set(weight_map.values())

    consumed_scales: set[str] = set()
    instance_acc: dict[tuple[int, int], int] = {}
    instance_class: dict[tuple[int, int], str] = {}

    for shard_rel in sorted(root_shards):
        header = root_shards[shard_rel]
        data_start = plan.shard_data_start[shard_rel]
        tensors = sorted(shard_tensors[shard_rel], key=lambda t: t["name"])
        for tensor in tensors:
            name = tensor["name"]
            component = tensor.get("component") or "(no component)"
            entry = header.tensors.get(name)
            if entry is None:
                raise BuildError(f"shard {shard_rel!r} lacks inventoried tensor {name!r}")
            if tensor["stored_bytes"] != entry["data_offsets"][1] - entry["data_offsets"][0]:
                raise AccountingError(
                    f"inventory/header byte disagreement for {name!r}: inventory "
                    f"{tensor['stored_bytes']} vs header "
                    f"{entry['data_offsets'][1] - entry['data_offsets'][0]}"
                )
            off, end = entry["data_offsets"]
            length = end - off

            expert_match = _EXPERT_RE.match(name)
            router_match = _ROUTER_RE.match(name)
            if expert_match:
                layer = int(expert_match.group(1))
                expert = int(expert_match.group(2))
                is_scale = expert_match.group(4) == "weight_scale"
                if is_scale:
                    # Scales are dispatched with their weight (packing evidence
                    # for native/decode; replaced by group scale/zero for affine).
                    weight_name = name[: -len("_scale")]
                    if weight_name not in header.tensors:
                        raise PackingEvidenceError(
                            f"packing-evidence sibling {name!r} has no weight tensor "
                            f"{weight_name!r} in shard {shard_rel!r}"
                        )
                    consumed_scales.add(name)
                    continue
                old_to_new = pmap["old_to_new_by_layer"].get(layer)
                if old_to_new is None:
                    raise BuildError(
                        f"expert tensor {name!r} is in layer {layer}, which is not a "
                        "MoE layer of the prune map"
                    )
                new_id = old_to_new.get(expert)
                scale_name = name + "_scale"
                scale_present = scale_name in header.tensors
                if new_id is None:
                    # Pruned expert: both weight and scale are dropped.
                    continue
                cls = (
                    "experts_second_gen"
                    if expert in q3_sets.get(layer, set())
                    else "experts_native"
                )
                if cls not in rows:
                    raise BuildError(
                        f"text tensor class {cls!r} has no allocation row"
                    )
                row = rows[cls]
                fmt = row["format"]
                mode = FORMATS[fmt]["mode"]
                out_name = (
                    f"model.layers.{layer}.mlp.experts.{new_id}."
                    f"{expert_match.group(3)}.weight"
                )
                logical = tensor.get("logical_shape") or tensor["shape"]
                params = tensor.get("logical_parameters")
                if not _is_int(params):
                    raise BuildError(f"expert weight {name!r} lacks logical_parameters")
                if mode == "mxfp4":
                    if tensor["quantization"] is None or tensor["quantization"].get(
                        "kind"
                    ) != "mxfp4_packed":
                        raise BuildError(
                            f"mxfp4_native format applied to non-mxfp4 tensor {name!r}"
                        )
                    if not scale_present:
                        raise PackingEvidenceError(
                            f"missing packing-evidence sibling {scale_name!r} for "
                            f"mxfp4 tensor {name!r} (required for bit-exact copy)"
                        )
                    scale = header.tensors[scale_name]
                    if scale["dtype"] != "U8":
                        raise BuildError(
                            f"packing-evidence sibling {scale_name!r} has dtype "
                            f"{scale['dtype']!r}, expected U8"
                        )
                    weight_t = planned_copy(
                        out_name=out_name,
                        dtype=entry["dtype"],
                        shape=entry["shape"],
                        nbytes=length,
                        cls=cls,
                        fmt=fmt,
                        component=component,
                        shard=shard_rel,
                        name=name,
                        off=off,
                        length=length,
                    )
                    add_tensor(weight_t)
                    scale_t = planned_copy(
                        out_name=out_name + "_scale",
                        dtype=scale["dtype"],
                        shape=scale["shape"],
                        nbytes=scale["data_offsets"][1] - scale["data_offsets"][0],
                        cls=cls,
                        fmt=fmt,
                        component=component,
                        shard=shard_rel,
                        name=scale_name,
                        off=scale["data_offsets"][0],
                        length=scale["data_offsets"][1] - scale["data_offsets"][0],
                    )
                    add_tensor(scale_t)
                    instance_bytes_count = weight_t.out_bytes + scale_t.out_bytes
                    bump(cls, 0, instance_bytes_count)  # units counted per instance
                elif mode == "affine":
                    if not scale_present:
                        raise PackingEvidenceError(
                            f"missing packing-evidence sibling {scale_name!r} for "
                            f"mxfp4 tensor {name!r} (required for decode)"
                        )
                    require_ledger(cls, source_is_mxfp4=True)
                    scale = header.tensors[scale_name]
                    bits = FORMATS[fmt]["bits"]
                    group = FORMATS[fmt]["group_size"]
                    expected = affine_expected_bytes(params, bits, group)
                    rows_a, cols_a, codes_shape, scale_shape = _plan_output_shape_affine(
                        logical, bits, group
                    )
                    if rows_a * cols_a != params:
                        raise BuildError(
                            f"expert weight {name!r}: logical_parameters {params} != "
                            f"{rows_a} x {cols_a}"
                        )
                    add_tensor(
                        PlannedTensor(
                            out_name=out_name,
                            out_dtype="U8",
                            out_shape=codes_shape,
                            out_bytes=(params * bits) // 8,
                            action="affine",
                            cls=cls,
                            fmt=fmt,
                            component=component,
                            src_shard=shard_rel,
                            src_name=name,
                            src_off=off,
                            src_len=length,
                            src_shape=list(logical),
                            src_scale_off=scale["data_offsets"][0],
                            src_scale_len=scale["data_offsets"][1]
                            - scale["data_offsets"][0],
                            params=params,
                        )
                    )
                    add_tensor(
                        PlannedTensor(
                            out_name=out_name + "_scale",
                            out_dtype="I8",
                            out_shape=scale_shape,
                            out_bytes=(params // group) * 2,
                            action="affine",
                            cls=cls,
                            fmt=fmt,
                            component=component,
                            src_shard=shard_rel,
                            src_name=name,
                            src_off=off,
                            src_len=length,
                            params=params,
                        )
                    )
                    instance_bytes_count = expected
                    bump(cls, 0, instance_bytes_count)  # units counted per instance
                else:
                    raise BuildError(
                        f"format {fmt!r} (mode {mode}) is not valid for expert "
                        f"instance dispatch: {name!r}"
                    )
                inst_key = (layer, expert)
                instance_acc[inst_key] = instance_acc.get(inst_key, 0) + instance_bytes_count
                instance_class[inst_key] = cls
                continue

            if router_match:
                layer = int(router_match.group(1))
                row = rows.get("router")
                if row is None:
                    raise BuildError("text tensor class 'router' has no allocation row")
                fmt = row["format"]
                if FORMATS[fmt]["mode"] != "row_slice":
                    raise BuildError(
                        "router must use the native_dense_row_sliced format "
                        f"(docs/pruning-integrity.md), got {fmt!r}"
                    )
                retained = pmap["retained_by_layer"][layer]
                dims = tensor.get("logical_shape") or tensor["shape"]
                item = DTYPE_ITEMSIZE.get(tensor["dtype"])
                if tensor.get("quantization") is not None or item is None:
                    raise AccountingError(
                        f"router tensor {name!r} is not a dense row-uniform "
                        f"{tensor['dtype']} tensor; exact row-slicing refused"
                    )
                old_rows = dims[0]
                rest = 1
                for d in dims[1:]:
                    rest *= d
                row_bytes = rest * item
                if old_rows <= 0 or length % old_rows or length // old_rows != row_bytes:
                    raise AccountingError(
                        f"router {name!r} stored_bytes {length} is not row-uniform "
                        f"over {old_rows} rows ({row_bytes} bytes/row expected)"
                    )
                out_bytes = len(retained) * row_bytes
                map_key = "weight" if router_match.group(2) == "weight" else "e_score_correction_bias"
                map_entry = None
                for layer_entry in pmap["document"]["layers"]:
                    if layer_entry["layer"] == layer:
                        router_block = layer_entry.get("router") or {}
                        candidate_entry = router_block.get(map_key)
                        if isinstance(candidate_entry, dict):
                            map_entry = candidate_entry
                        break
                if map_entry is None or map_entry.get("new_stored_bytes") != out_bytes:
                    raise AccountingError(
                        f"router {name!r}: planned {out_bytes} bytes disagrees with "
                        f"prune-map accounting "
                        f"{None if map_entry is None else map_entry.get('new_stored_bytes')}"
                    )
                add_tensor(
                    PlannedTensor(
                        out_name=name,
                        out_dtype=tensor["dtype"],
                        out_shape=[len(retained)] + list(dims[1:]),
                        out_bytes=out_bytes,
                        action="row_slice",
                        cls="router",
                        fmt=fmt,
                        component=component,
                        src_shard=shard_rel,
                        src_name=name,
                        src_off=off,
                        src_len=length,
                        rows=list(retained),
                        row_bytes=row_bytes,
                    )
                )
                bump("router", 0, out_bytes)  # units counted once (retained index)
                continue

            if component != "text":
                add_tensor(
                    planned_copy(
                        out_name=name,
                        dtype=entry["dtype"],
                        shape=entry["shape"],
                        nbytes=length,
                        cls="",
                        fmt="bit_exact_copy",
                        component=component,
                        shard=shard_rel,
                        name=name,
                        off=off,
                        length=length,
                    )
                )
                plan.aux_tensors += 1
                plan.aux_bytes += length
                continue

            cls = _classify_text_tensor(name)
            if cls == "dense_ffn" and tensor.get("layer") != 0:
                raise BuildError(
                    f"text tensor {name!r} looks like a dense FFN weight but is not "
                    "in layer 0"
                )
            if cls == "dense_ffn_scales" and tensor.get("layer") != 0:
                raise BuildError(
                    f"text tensor {name!r} looks like a dense FFN scale but is not "
                    "in layer 0"
                )
            row = rows.get(cls)
            if row is None:
                raise BuildError(f"text tensor class {cls!r} has no allocation row")
            fmt = row["format"]
            mode = FORMATS[fmt]["mode"]
            if mode == "row_slice":
                raise BuildError(
                    f"native_dense_row_sliced format is reserved for routers, "
                    f"applied to {name!r}"
                )
            if mode == "mxfp4":
                raise BuildError(
                    f"mxfp4_native format applies only to expert instances, "
                    f"applied to {name!r}"
                )
            if mode == "passthrough":
                allowed = FORMATS[fmt]["source_dtypes"]
                if tensor["dtype"] not in allowed:
                    raise BuildError(
                        f"source dtype {tensor['dtype']!r} of {name!r} does not "
                        f"match native format {fmt!r} (expected one of {allowed})"
                    )
                add_tensor(
                    planned_copy(
                        out_name=name,
                        dtype=entry["dtype"],
                        shape=entry["shape"],
                        nbytes=length,
                        cls=cls,
                        fmt=fmt,
                        component=component,
                        shard=shard_rel,
                        name=name,
                        off=off,
                        length=length,
                    )
                )
                bump(cls, 1, length)
                continue
            # mode == "affine"
            quant = tensor.get("quantization")
            source_is_mxfp4 = bool(quant and quant.get("kind") == "mxfp4_packed")
            if source_is_mxfp4 and tensor["dtype"] != "U8":
                raise BuildError(f"mxfp4 record with dtype {tensor['dtype']!r}: {name!r}")
            if source_is_mxfp4:
                if name + "_scale" not in header.tensors:
                    raise PackingEvidenceError(
                        f"missing packing-evidence sibling {name + '_scale'!r} for "
                        f"mxfp4 tensor {name!r} (required for decode)"
                    )
                require_ledger(cls, source_is_mxfp4=True)
            else:
                if tensor["dtype"] == "F8_E4M3":
                    raise BuildError(
                        f"refusing second-generation quantisation of an already-"
                        f"quantized FP8 tensor (no re-encode): {name!r}"
                    )
                if tensor["dtype"] != "BF16":
                    raise BuildError(
                        f"affine dispatch supports mxfp4 U8 or native BF16 sources, "
                        f"got {tensor['dtype']!r} for {name!r}"
                    )
                require_ledger(cls, source_is_mxfp4=False)
            bits = FORMATS[fmt]["bits"]
            group = FORMATS[fmt]["group_size"]
            logical = tensor.get("logical_shape") or tensor["shape"]
            params = tensor.get("logical_parameters")
            if not _is_int(params):
                raise BuildError(f"tensor {name!r} lacks an exact logical_parameters")
            expected = affine_expected_bytes(params, bits, group)
            rows_a, cols_a, codes_shape, scale_shape = _plan_output_shape_affine(
                logical, bits, group
            )
            if rows_a * cols_a != params:
                raise BuildError(
                    f"tensor {name!r}: logical_parameters {params} != {rows_a} x {cols_a}"
                )
            weight_t = PlannedTensor(
                out_name=name,
                out_dtype="U8",
                out_shape=codes_shape,
                out_bytes=(params * bits) // 8,
                action="affine",
                cls=cls,
                fmt=fmt,
                component=component,
                src_shard=shard_rel,
                src_name=name,
                src_off=off,
                src_len=length,
                src_shape=list(logical),
                params=params,
            )
            if source_is_mxfp4:
                scale = header.tensors[name + "_scale"]
                weight_t.src_scale_off = scale["data_offsets"][0]
                weight_t.src_scale_len = scale["data_offsets"][1] - scale["data_offsets"][0]
                consumed_scales.add(name + "_scale")
            add_tensor(weight_t)
            add_tensor(
                PlannedTensor(
                    out_name=name + "_scale",
                    out_dtype="I8",
                    out_shape=scale_shape,
                    out_bytes=(params // group) * 2,
                    action="affine",
                    cls=cls,
                    fmt=fmt,
                    component=component,
                    src_shard=shard_rel,
                    src_name=name,
                    src_off=off,
                    src_len=length,
                    params=params,
                )
            )
            bump(cls, 1, expected)

    # Every inventory scale record must have been consumed by its weight.
    for shard_rel, tensors in shard_tensors.items():
        if "/" in shard_rel:
            continue
        for tensor in tensors:
            name = tensor["name"]
            if not name.endswith("weight_scale"):
                continue
            match = _EXPERT_RE.match(name)
            if match is None:
                continue
            if name in consumed_scales:
                continue
            # Not consumed: the weight was pruned, and the scale is dropped
            # together with it. Anything else means a retained weight record
            # was never dispatched (missing weight record).
            layer, expert = int(match.group(1)), int(match.group(2))
            new_id = pmap["old_to_new_by_layer"][layer].get(expert)
            if new_id is not None:
                raise BuildError(
                    f"retained expert scale {name!r} was never dispatched "
                    "(missing weight record?)"
                )

    # Non-expert, non-router text tensors are all dispatched; now close the
    # static accounting against the allocation table.
    plan.router_units = pmap["retained_count"]
    for cls, row in rows.items():
        bucket = plan.classes.get(cls, {"planned_units": 0, "planned_bytes": 0, "tensor_count": 0})
        plan.classes[cls] = bucket
        planned_units = bucket["planned_units"]
        if row["unit_label"] == "retained_index":
            planned_units = plan.router_units
        elif row["unit_label"] == "expert_instance":
            planned_units = sum(
                1 for assigned in instance_class.values() if assigned == cls
            )
        if planned_units != row["units"]:
            raise AccountingError(
                f"byte accounting does not close for class {cls!r}: planned "
                f"{planned_units} units vs recipe {row['units']}"
            )
        bucket["planned_units"] = planned_units
        if bucket["planned_bytes"] != row["class_bytes"]:
            raise AccountingError(
                f"byte accounting does not close for class {cls!r}: planned "
                f"{bucket['planned_bytes']} bytes vs recipe class_bytes "
                f"{row['class_bytes']}"
            )
    plan.total_text_bytes = sum(b["planned_bytes"] for b in plan.classes.values())
    declared_sum = sum(row["class_bytes"] for row in rows.values())
    if declared_sum != plan.total_text_bytes:  # pragma: no cover - implied above
        raise AccountingError("allocation_table sum disagrees with planned bytes")
    planned_total = recipe.get("planned_resident_weight_bytes")
    if planned_total is not None and planned_total != plan.total_text_bytes:
        raise AccountingError(
            f"byte accounting does not close: recipe planned_resident_weight_bytes "
            f"{planned_total} != planned output {plan.total_text_bytes}"
        )
    # Pure-MXFP4 floor: every second-gen class at its recorded source bytes.
    floor = 0
    for cls, row in rows.items():
        if row["second_gen"]:
            floor += ledger[cls]["bytes_before"]
        else:
            floor += row["class_bytes"]
    declared_floor = recipe.get("pure_mxfp4_floor_bytes")
    if declared_floor is not None and declared_floor != floor:
        raise AccountingError(
            f"byte accounting does not close: recipe pure_mxfp4_floor_bytes "
            f"{declared_floor} != inventory-derived {floor}"
        )
    plan.floor_bytes = floor

    # Per-instance uniformity: every retained instance must cost exactly the
    # allocation row's bytes_per_unit.
    for (layer, expert), nbytes in sorted(instance_acc.items()):
        cls = (
            "experts_second_gen"
            if expert in q3_sets.get(layer, set())
            else "experts_native"
        )
        row = rows[cls]
        if nbytes != row["bytes_per_unit"]:
            raise AccountingError(
                f"expert instance ({layer}, {expert}) plans {nbytes} bytes but "
                f"class {cls!r} declares bytes_per_unit {row['bytes_per_unit']}"
            )
    expected_instances = precision["native_mxfp4_instances"] + precision["q3_instances"]
    if len(instance_acc) != expected_instances:
        raise AccountingError(
            f"planned {len(instance_acc)} retained expert instances, recipe "
            f"implies {expected_instances}"
        )
    if rows.get("experts_native") and precision.get("instance_bytes_native") != rows[
        "experts_native"
    ]["bytes_per_unit"]:
        raise AccountingError(
            "expert_precision.instance_bytes_native disagrees with the allocation "
            "row bytes_per_unit"
        )
    if rows.get("experts_second_gen") and precision.get("instance_bytes_q3") != rows[
        "experts_second_gen"
    ]["bytes_per_unit"]:
        raise AccountingError(
            "expert_precision.instance_bytes_q3 disagrees with the allocation row "
            "bytes_per_unit"
        )

    # Regenerated index membership: the source index lists exactly the shards
    # it covers (values of weight_map; the inventory's index/header bijection
    # guarantees every tensor of those shards is indexed), so the output
    # weight_map covers every output tensor of an indexed shard.
    if index_doc_in is not None:
        weight_map_out: dict[str, str] = {}
        payload_total = 0
        index_shards = set(
            value for value in index_membership_values if isinstance(value, str)
        )
        for shard_rel, tensors in sorted(plan.shards.items()):
            if shard_rel not in index_shards:
                continue
            for tensor in sorted(tensors, key=lambda t: t.out_name):
                weight_map_out[tensor.out_name] = shard_rel
                payload_total += tensor.out_bytes
        plan.index_doc = {
            "metadata": {"total_size": payload_total},
            "weight_map": dict(sorted(weight_map_out.items())),
        }
        plan.index_note = (
            f"regenerated from the build plan; metadata.total_size {payload_total} "
            "is the exact output payload sum of index-covered shards (save_format "
            "dropped: the candidate mixes formats)"
        )
    plan.dispatch = {
        "order_basis": order_basis,
        "q3_old_ids_by_layer": {str(k): v for k, v in sorted(q3_by_layer.items())},
        "retained_by_layer": {str(k): v for k, v in sorted(retained_by_layer.items())},
        "old_to_new_by_layer": {
            str(k): {str(o): n for o, n in sorted(v.items())}
            for k, v in sorted(pmap["old_to_new_by_layer"].items())
        },
    }
    plan.expert_second_gen = any(
        tensor.action == "affine" and tensor.cls.startswith("experts")
        for shard_tensors_out in plan.shards.values()
        for tensor in shard_tensors_out
    )
    return plan


def _adapt_config(
    config: dict,
    counts_by_layer: dict[int, int],
    *,
    expert_second_gen: bool,
    original_top_k: int,
) -> tuple[dict, dict]:
    """Adapt source config.json to the shapes the build actually writes.

    * ``n_routed_experts`` is a single scalar: when the prune map keeps
      varying counts across layers (no uniform scalar exists), the build
      refuses explicitly instead of writing an inconsistent config.
    * ``num_experts_per_tok`` (top-k routing) is never changed and must
      agree with the pinned inventory architecture.
    * On a build with second-gen expert tensors, a ``store_dtype: mxfp4``
      claim would imply the Q3 U8 codes are native MXFP4: the claim is
      replaced with an explicit non-mxfp4 marker and recorded. All-native
      builds keep the quant metadata verbatim (it is true).
    """
    if not counts_by_layer:
        raise BuildError("config adaptation: no per-layer kept counts available")
    distinct = sorted(set(counts_by_layer.values()))
    if len(distinct) != 1:
        raise BuildError(
            "config n_routed_experts is a single scalar but the prune map keeps "
            f"varying per-layer expert counts "
            f"{dict(sorted(counts_by_layer.items()))}; refusing an inconsistent config"
        )
    kept = distinct[0]
    if not _is_int(original_top_k) or original_top_k <= 0:
        raise BuildError(
            f"config adaptation: pinned architecture top_k is invalid: {original_top_k!r}"
        )
    adapted = json.loads(json.dumps(config))  # deep copy of JSON data
    changes: dict = {
        "n_routed_experts": None,
        "store_dtype": None,
        "num_experts_per_tok": original_top_k,
        "kept_experts_per_layer": kept,
    }
    found = [key for key in ROUTED_EXPERT_COUNT_KEYS if key in adapted]
    if not found:
        raise BuildError(
            "config.json has no recognized routed-expert count field "
            f"({', '.join(ROUTED_EXPERT_COUNT_KEYS)}); refusing a config whose "
            "router consistency cannot be verified"
        )
    for key in found:
        before = adapted[key]
        if not _is_int(before):
            raise BuildError(f"config {key} must be an int, got {before!r}")
        if before != kept:
            adapted[key] = kept
            changes["n_routed_experts"] = {"from": before, "to": kept}
    top_k = adapted.get("num_experts_per_tok")
    if top_k is not None and top_k != original_top_k:
        raise BuildError(
            f"config num_experts_per_tok {top_k!r} contradicts the pinned "
            f"architecture top_k {original_top_k!r}"
        )
    quant = adapted.get("quantization_config")
    if expert_second_gen and isinstance(quant, dict):
        # The Q3 U8 codes are NOT native MXFP4: strip every key that
        # describes the native expert layout. store_dtype "mxfp4" -> an
        # explicit marker that matches no known quant-vocabulary word (the
        # quant_method key is never edited), and mxfp4_block_size, which
        # only has meaning under the native layout, is neutralized to null.
        if quant.get("store_dtype") == "mxfp4":
            quant["store_dtype"] = SECOND_GEN_STORE_DTYPE
            changes["store_dtype"] = {"from": "mxfp4", "to": SECOND_GEN_STORE_DTYPE}
        if quant.get("mxfp4_block_size") is not None:
            changes["mxfp4_block_size"] = {
                "from": quant["mxfp4_block_size"],
                "to": None,
            }
            quant["mxfp4_block_size"] = None
    return adapted, changes


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_json_checked(path: Path, obj) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    back = json.loads(path.read_text(encoding="utf-8"))
    if back != obj:  # pragma: no cover - only on write corruption
        raise BuildError(f"written JSON failed read-back verification: {path.name}")


def _fill_copy(src, data_start: int, tensor: PlannedTensor, out) -> int:
    src.seek(data_start + tensor.src_off)
    digest = hashlib.sha256()
    written = 0
    remaining = tensor.src_len
    while remaining > 0:
        chunk = src.read(min(COPY_CHUNK_BYTES, remaining))
        if not chunk:
            raise BuildError(f"unexpected EOF reading {tensor.src_name!r}")
        digest.update(chunk)
        out.write(chunk)
        written += len(chunk)
        remaining -= len(chunk)
    tensor.input_sha256 = digest.hexdigest()
    return written


def _fill_slice(src, data_start: int, tensor: PlannedTensor, out) -> int:
    digest = hashlib.sha256()
    written = 0
    for row in tensor.rows:
        src.seek(data_start + tensor.src_off + row * tensor.row_bytes)
        remaining = tensor.row_bytes
        while remaining > 0:
            chunk = src.read(min(COPY_CHUNK_BYTES, remaining))
            if not chunk:
                raise BuildError(f"unexpected EOF reading row {row} of {tensor.src_name!r}")
            digest.update(chunk)
            out.write(chunk)
            written += len(chunk)
            remaining -= len(chunk)
    tensor.input_sha256 = digest.hexdigest()
    return written


def _fill_affine(src, data_start: int, tensor: PlannedTensor, out) -> tuple[int, bytes]:
    """Encode one affine weight tensor: writes packed codes, RETURNS the
    interleaved scale/zero blob for the sibling ``<name>_scale`` tensor
    (same shard; the caller stashes it until the sibling is filled)."""
    fmt = FORMATS[tensor.fmt]
    bits = fmt["bits"]
    group = fmt["group_size"]
    logical = tensor.src_shape
    rows_total, cols = (1, logical[0]) if len(logical) == 1 else (logical[0], logical[1])
    source_is_mxfp4 = tensor.src_scale_off is not None
    weight_row_bytes = tensor.src_len // rows_total
    if source_is_mxfp4:
        scale_row_bytes = tensor.src_scale_len // rows_total
        per_row = weight_row_bytes + scale_row_bytes
    else:
        per_row = weight_row_bytes
    rows_chunk = max(1, AFFINE_CHUNK_BYTES // max(1, per_row))
    digest = hashlib.sha256()
    written = 0
    scale_zero = bytearray()
    err_max = 0.0
    err_sum = 0.0
    elements = 0
    for start in range(0, rows_total, rows_chunk):
        count = min(rows_chunk, rows_total - start)
        src.seek(data_start + tensor.src_off + start * weight_row_bytes)
        weight = _read_exact(src, count * weight_row_bytes, tensor.src_name)
        scale = b""
        if source_is_mxfp4:
            src.seek(data_start + tensor.src_scale_off + start * scale_row_bytes)
            scale = _read_exact(src, count * scale_row_bytes, tensor.src_name + "_scale")
        digest.update(weight)
        if scale:
            digest.update(scale)
        if source_is_mxfp4:
            values = decode_mxfp4(weight, scale, count, cols)
        else:
            values = decode_bf16(weight, count, cols)
        encoded = encode_affine(values, bits, group)
        out.write(encoded.codes)
        scale_zero += encoded.scale_zero
        written += len(encoded.codes)
        stats = encoded.stats
        err_max = max(err_max, stats["max_abs_err"])
        err_sum += stats["mean_abs_err"] * stats["elements"]
        elements += stats["elements"]
    tensor.input_sha256 = digest.hexdigest()
    tensor.error = {
        "elements": elements,
        "max_abs_err": err_max,
        "mean_abs_err": (err_sum / elements) if elements else 0.0,
        "bits": bits,
        "group_size": group,
        "reference": (
            "BF16-decoded source values (compute-only; MXFP4 decode for packed "
            "experts) - not a pre-QAT full-precision claim"
        ),
    }
    return written, bytes(scale_zero)


def _write_shards(plan: Plan, out_dir: Path) -> dict:
    """Write every output shard streamed; returns per-shard file facts."""
    facts = {}
    stashes: dict[str, tuple[bytes, str | None]] = {}
    for shard_rel in sorted(plan.shards):
        tensors = sorted(plan.shards[shard_rel], key=lambda t: t.out_name)
        header_doc = {}
        offset = 0
        for tensor in tensors:
            header_doc[tensor.out_name] = {
                "dtype": tensor.out_dtype,
                "shape": list(tensor.out_shape),
                "data_offsets": [offset, offset + tensor.out_bytes],
            }
            offset += tensor.out_bytes
        raw = json.dumps(header_doc, separators=(",", ":")).encode("utf-8")
        raw += b" " * ((8 - len(raw) % 8) % 8)
        payload_total = offset
        out_path = out_dir / shard_rel
        with open(out_path, "wb") as out, open(
            plan.shard_source_path[shard_rel], "rb"
        ) as src:
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            data_start = plan.shard_data_start[shard_rel]
            for tensor in tensors:
                if tensor.action == "affine" and tensor.out_dtype == "I8":
                    # Sibling scale/zero blob produced while the codes tensor
                    # (same shard, sorted immediately before) was filled.
                    stash = stashes.pop(tensor.out_name, None)
                    if stash is None:
                        raise BuildError(
                            f"affine scale/zero output {tensor.out_name!r} has no "
                            "encoded blob from its codes tensor (fill order broken)"
                        )
                    blob, codes_input_sha = stash
                    if tensor.input_sha256 is None:
                        tensor.input_sha256 = codes_input_sha
                    out.write(blob)
                    written = len(blob)
                elif tensor.action == "affine":
                    written, scale_zero = _fill_affine(src, data_start, tensor, out)
                    stashes[tensor.out_name + "_scale"] = (scale_zero, tensor.input_sha256)
                elif tensor.action == "row_slice":
                    written = _fill_slice(src, data_start, tensor, out)
                else:
                    written = _fill_copy(src, data_start, tensor, out)
                if written != tensor.out_bytes:
                    raise AccountingError(
                        f"byte accounting does not close for {tensor.out_name!r}: "
                        f"wrote {written}, recipe predicted {tensor.out_bytes}"
                    )
        if stashes:
            raise BuildError(
                f"unconsumed affine scale/zero blobs after writing {shard_rel!r}: "
                f"{sorted(stashes)}"
            )
        expected_size = 8 + len(raw) + payload_total
        actual_size = os.path.getsize(out_path)
        if actual_size != expected_size:
            raise AccountingError(
                f"output shard {shard_rel!r} is {actual_size} bytes, predicted "
                f"{expected_size} (header {8 + len(raw)} + payload {payload_total})"
            )
        facts[shard_rel] = {
            "file": shard_rel,
            "size_bytes": actual_size,
            "payload_bytes": payload_total,
            "tensors": len(tensors),
        }
    return facts


def _readback_verify(plan: Plan, out_dir: Path) -> None:
    """Re-read every written tensor; enforce bit-exact digest equality."""
    for shard_rel in sorted(plan.shards):
        out_path = out_dir / shard_rel
        header = read_safetensors_header(out_path)
        tensors = sorted(plan.shards[shard_rel], key=lambda t: t.out_name)
        if set(header.tensors) != {t.out_name for t in tensors}:
            raise BuildError(
                f"written header of {shard_rel!r} does not match the plan "
                "(names differ)"
            )
        data_start = 8 + header.header_length
        with open(out_path, "rb") as handle:
            for tensor in tensors:
                entry = header.tensors.get(tensor.out_name)
                if (
                    entry is None
                    or entry["dtype"] != tensor.out_dtype
                    or entry["shape"] != list(tensor.out_shape)
                ):
                    raise BuildError(
                        f"written header entry for {tensor.out_name!r} does not "
                        "match the plan"
                    )
                off, end = entry["data_offsets"]
                if end - off != tensor.out_bytes:
                    raise AccountingError(
                        f"written span of {tensor.out_name!r} is {end - off}, "
                        f"predicted {tensor.out_bytes}"
                    )
                handle.seek(data_start + off)
                digest = hashlib.sha256()
                remaining = tensor.out_bytes
                while remaining > 0:
                    chunk = handle.read(min(COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise BuildError(
                            f"truncated read-back of {tensor.out_name!r}"
                        )
                    digest.update(chunk)
                    remaining -= len(chunk)
                tensor.output_sha256 = digest.hexdigest()
                if tensor.bit_exact and tensor.output_sha256 != tensor.input_sha256:
                    raise BuildError(
                        f"bit-exact copy digest mismatch for {tensor.out_name!r}: "
                        f"input {tensor.input_sha256} != output {tensor.output_sha256}"
                    )


def _copy_metadata(plan: Plan, root_real: Path, out_dir: Path, *, original_top_k: int) -> None:
    for entry in plan.metadata:
        src_path = _resolve_inside(root_real, entry["path"], "metadata file")
        dst_path = out_dir / entry["path"]
        source_bytes = src_path.read_bytes()
        source_sha = _sha256_bytes(source_bytes)
        size = len(source_bytes)
        if entry["declared_size"] is not None and entry["declared_size"] != size:
            raise SourcePathError(
                f"metadata file {entry['path']!r} size {size} != pinned inventory "
                f"size {entry['declared_size']}"
            )
        declared_sha = entry["declared_lfs_sha256"]
        if declared_sha is not None and declared_sha != source_sha:
            raise SourcePathError(
                f"metadata file {entry['path']!r} fails digest verification against "
                "the pinned inventory"
            )
        entry["source_sha256"] = source_sha
        if entry.get("adapt"):
            # config.json: verified against the pinned source FIRST, then
            # adapted to the shapes this build writes (kept counts + honest
            # quant metadata); the source tree itself is never touched.
            try:
                source_config = json.loads(source_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BuildError(f"source config.json is not valid JSON: {exc}") from exc
            if not isinstance(source_config, dict):
                raise BuildError("source config.json must be a JSON object")
            counts = {
                int(layer): len(retained)
                for layer, retained in plan.dispatch["retained_by_layer"].items()
            }
            adapted, changes = _adapt_config(
                source_config,
                counts,
                expert_second_gen=plan.expert_second_gen,
                original_top_k=original_top_k,
            )
            text = json.dumps(adapted, indent=2, ensure_ascii=False) + "\n"
            dst_path.write_text(text, encoding="utf-8")
            back = json.loads(dst_path.read_text(encoding="utf-8"))
            if back != adapted:  # pragma: no cover - only on write corruption
                raise BuildError("adapted config.json failed read-back verification")
            entry["sha256"] = _sha256_bytes(text.encode("utf-8"))
            entry["verified"] = (
                "source sha256 verified against the pinned inventory before "
                "adaptation; output bytes adapted (see build_report "
                "config_adaptation)"
            )
            entry["adapted"] = True
            plan.config_adaptation = {
                "source_sha256": source_sha,
                "output_sha256": entry["sha256"],
                "changes": changes,
                "note": (
                    "config.json is a NEW file in the artifact root; the source "
                    "tree is never modified. n_routed_experts follows the prune "
                    "map's uniform kept count; num_experts_per_tok (top-k) and "
                    "all other keys are retained verbatim; a store_dtype 'mxfp4' "
                    "claim is replaced only when expert tensors are second-gen "
                    "affine codes (never implies native MXFP4)."
                ),
            }
            continue
        dst_path.write_bytes(source_bytes)
        if _sha256_bytes(dst_path.read_bytes()) != source_sha:
            raise BuildError(
                f"copied metadata digest mismatch for {entry['path']!r}"
            )
        entry["sha256"] = source_sha
        entry["verified"] = (
            "sha256 vs pinned inventory lfs_sha256"
            if declared_sha is not None
            else "sha256 recorded from bytes (no pinned digest for this file)"
        )


def _verify_config_router_consistency(out_dir: Path, plan: Plan) -> None:
    """Fail closed unless the written config matches the written tensors.

    Re-reads the adapted ``config.json`` from disk and cross-checks every
    shape the config implies (routed-expert scalar -> router rows, expert id
    space, top-k over the retained set) against the planned tensors that the
    read-back pass already proved byte-identical to the shard headers.
    """
    if not any(entry.get("adapt") for entry in plan.metadata):
        return  # source has no config.json; nothing claims router shapes
    config_path = out_dir / CONFIG_NAME
    if not config_path.is_file():
        raise BuildError("config.json was planned but is missing from the output")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"written config.json is not valid JSON: {exc}") from exc
    routed = config.get("n_routed_experts")
    if not _is_int(routed) or routed <= 0:
        raise BuildError(f"written config n_routed_experts is invalid: {routed!r}")
    counts = {
        int(layer): len(retained)
        for layer, retained in plan.dispatch["retained_by_layer"].items()
    }
    if set(counts.values()) != {routed}:
        raise BuildError(
            f"written config n_routed_experts {routed} disagrees with the built "
            f"kept counts {dict(sorted(counts.items()))}"
        )
    top_k = config.get("num_experts_per_tok")
    if _is_int(top_k) and top_k > routed:
        raise BuildError(
            f"written config num_experts_per_tok {top_k} exceeds the retained "
            f"expert count {routed} (routing stays top-k over the retained set)"
        )
    out_names = set(plan.out_names_seen)
    for tensor in plan_shard_tensors(plan):
        if tensor.action == "row_slice" and tensor.out_shape[0] != routed:
            raise BuildError(
                f"router tensor {tensor.out_name!r} has {tensor.out_shape[0]} rows "
                f"but the written config claims n_routed_experts={routed}"
            )
    expert_re = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    ids_by_layer_proj: dict[tuple[int, str], set[int]] = {}
    for name in out_names:
        match = expert_re.match(name)
        if match:
            ids_by_layer_proj.setdefault(
                (int(match.group(1)), match.group(3)), set()
            ).add(int(match.group(2)))
    expected_ids = set(range(routed))
    for layer in sorted(counts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            found = ids_by_layer_proj.get((layer, proj), set())
            if found != expected_ids:
                raise BuildError(
                    f"layer {layer} {proj} expert ids {sorted(found)} do not match "
                    f"the written config's n_routed_experts={routed} "
                    f"(expected {sorted(expected_ids)})"
                )


# ---------------------------------------------------------------------------
# Candidate record (schemas/compression-candidate.schema.json)
# ---------------------------------------------------------------------------


def _empty_metrics() -> dict:
    null_slice = {"delta_perplexity": None, "agreement_fraction": None, "notes": None}
    return {
        "kill": {
            "executed": False,
            "stage_reached": None,
            "killed": False,
            "criterion": None,
        },
        "calibration": {
            "perplexity": None,
            "delta_perplexity_vs_original": None,
            "kl_divergence_nats": None,
            "corpus": None,
        },
        "on_policy_nll_nats": None,
        "agreement": {
            "top1_fraction": None,
            "code_fraction": None,
            "reasoning_fraction": None,
            "agent_tooluse_fraction": None,
            "corpus": None,
        },
        "long_context_degradation": {
            key: dict(null_slice)
            for key in ("8k", "16k-32k", "32k-64k", "64k-128k", "128k_plus")
        },
        "pathological_generations": [],
        "benchmarks": [],
    }


def _empty_performance() -> dict:
    point = {"value": None, "unit": "tokens/s", "method": None}
    return {
        "label": "PRE-OPTIMISATION",
        "ranking_use": "excluded-from-quality-ranking",
        "pp": dict(point),
        "c1_decode": dict(point),
        "c4_decode": dict(point),
        "c8_decode": dict(point),
        "theoretical_bytes_per_token": None,
        "active_weight_bandwidth": {"value": None, "unit": "bytes/s", "note": None},
    }


def candidate_record(recipe: dict, source_info: dict, *, build_report=None, notes=None) -> dict:
    """Build a candidate record shaped exactly like the sweep schema.

    ``build_report=None`` produces a ``planned`` record from the recipe's own
    planned byte fields; a build report (from :func:`build_candidate`)
    produces a ``built`` record whose ``size`` figures are the measured,
    accounting-closed output bytes.
    """
    _require_keys(recipe, RECIPE_REQUIRED_KEYS, "recipe")
    if "original_experts_per_layer" not in recipe:
        # Schema const 256: fill the required key from the pinned constant.
        recipe = {**recipe, "original_experts_per_layer": 256}
    _require_keys(
        source_info,
        ("repo_id", "revision", "file_count", "total_bytes", "digest_basis"),
        "source_info",
    )
    if not isinstance(source_info["digest_basis"], list) or not source_info["digest_basis"]:
        raise BuildError("source_info.digest_basis must be a non-empty list")
    if build_report is None:
        _require_keys(
            recipe,
            ("candidate_id", "planned_resident_weight_bytes", "pure_mxfp4_floor_bytes"),
            "planned recipe",
        )
        resident = recipe["planned_resident_weight_bytes"]
        floor = recipe["pure_mxfp4_floor_bytes"]
        status = "planned"
    else:
        _require_keys(build_report, ("resident_weight_bytes", "pure_mxfp4_floor_bytes"), "build report")
        resident = build_report["resident_weight_bytes"]
        floor = build_report["pure_mxfp4_floor_bytes"]
        status = "built"
    candidate_id = recipe.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise BuildError("recipe candidate_id must be a non-empty string")
    residual = resident - TARGET_RESIDENT_BYTES
    record = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "status": status,
        "source_weights": {
            "repo_id": source_info["repo_id"],
            "revision": source_info["revision"],
            "file_count": source_info["file_count"],
            "total_bytes": source_info["total_bytes"],
            "digest_basis": list(source_info["digest_basis"]),
        },
        "recipe": {key: recipe[key] for key in RECIPE_KEYS},
        "size": {
            "resident_weight_bytes": resident,
            "target_bytes": TARGET_RESIDENT_BYTES,
            "gib": resident / (1 << 30),
            "decimal_gb": resident / 1_000_000_000,
            "residual_bytes": residual,
            "residual_percent": residual / TARGET_RESIDENT_BYTES * 100.0,
            "tolerance_percent": 2,
            "within_tolerance": abs(residual) <= TOLERANCE_BYTES,
            "pure_mxfp4_floor_bytes": floor,
            "pure_mxfp4_floor_within_tolerance": abs(floor - TARGET_RESIDENT_BYTES)
            <= TOLERANCE_BYTES,
        },
        "metrics": _empty_metrics(),
        "performance_pre_optimisation": _empty_performance(),
        "notes": notes,
    }
    return record


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_candidate(
    *,
    inventory_path: str | Path,
    prune_map_path: str | Path,
    recipe: dict,
    source_root: str | Path,
    out_dir: str | Path,
    seed: int,
    environment_path: str | Path,
    dataset_hashes,
    digest_basis,
    calibration_config_path: str | Path | None = None,
    commands=None,
    parents=(),
    expert_order: dict | None = None,
    kind: str = "pruned",
    signed_sources=(),
    conversion_commit: str | None = None,
    runtime_commit: str | None = None,
    production: bool = False,
    official_source: str | None = None,
    created_at: str | None = None,
    notes: str | None = None,
) -> dict:
    """Construct one candidate checkpoint with full artifact sidecars.

    All inputs are explicit; every refusal fails closed before or during the
    write phase, and sidecars are created only after the byte accounting has
    closed and the read-back digest verification has passed.
    """
    # --- verified source root -------------------------------------------
    try:
        inv = load_inventory(inventory_path)
    except PruneMapError as exc:
        raise BuildError(f"inventory rejected: {exc}") from exc
    if inv["source"].get("repo_id") != REPO_ID:
        raise SourcePathError(
            f"inventory repo_id {inv['source'].get('repo_id')!r} is not the "
            f"original Xiaomi source {REPO_ID!r}"
        )
    if inv["source"].get("revision") != REVISION:
        raise SourcePathError(
            f"inventory revision {inv['source'].get('revision')!r} is not the "
            f"pinned source revision {REVISION}"
        )
    source_root_real = Path(os.path.realpath(source_root))
    if not source_root_real.is_dir():
        raise SourcePathError(f"source root is not a directory: {source_root!r}")
    if source_root_real.name != REVISION:
        raise SourcePathError(
            f"source root is not the verified pinned revision directory "
            f"(expected a directory named {REVISION}, got {source_root_real.name!r})"
        )
    out_path = Path(out_dir)
    # realpath resolves symlinks in the existing prefix even when the leaf
    # does not exist yet (macOS /var -> /private/var), so the containment
    # check below sees one canonical namespace.
    out_real = Path(os.path.realpath(out_dir))
    if out_real == source_root_real or source_root_real in out_real.parents:
        raise SourcePathError(
            "output directory must be outside the verified source root "
            "(the source tree stays pristine)"
        )
    if out_path.exists():
        if not out_path.is_dir():
            raise BuildError(f"output path exists and is not a directory: {out_dir!r}")
        if any(os.scandir(out_path)):
            raise BuildError(
                f"output directory must be absent or empty, found existing "
                f"content: {out_dir!r}"
            )

    # --- recipe + prune map ---------------------------------------------
    pmap = _load_prune_map(prune_map_path, inv)
    if pmap["retained_count"] != recipe.get("retained_experts_per_layer"):
        raise BuildError(
            f"recipe retained_experts_per_layer "
            f"{recipe.get('retained_experts_per_layer')!r} != prune map "
            f"retained_count {pmap['retained_count']}"
        )
    parsed = _validate_recipe(recipe, inv)

    # --- plan (static accounting must close before any write) -----------
    plan = _plan_build(inv, pmap, recipe, parsed, source_root_real, expert_order)
    if any(tensor.action == "affine" for tensor in plan_shard_tensors(plan)):
        require_numpy()  # fail closed before creating any output

    # --- write phase ----------------------------------------------------
    out_path.mkdir(parents=True, exist_ok=True)
    shard_facts = _write_shards(plan, out_path)
    _readback_verify(plan, out_path)
    _copy_metadata(
        plan, source_root_real, out_path,
        original_top_k=inv["architecture"]["top_k"],
    )
    if plan.index_doc is not None:
        _write_json_checked(out_path / INDEX_NAME, plan.index_doc)
    _verify_config_router_consistency(out_path, plan)

    # --- records --------------------------------------------------------
    file_entries = inv["source"]["files"]
    source_info = {
        "repo_id": inv["source"]["repo_id"],
        "revision": inv["source"]["revision"],
        "file_count": len(file_entries),
        "total_bytes": sum(f.get("size") or 0 for f in file_entries),
        "digest_basis": list(digest_basis),
    }
    candidate_id = recipe.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise BuildError("recipe candidate_id must be a non-empty string")

    errors = [t.error for t in plan_shard_tensors(plan) if t.error is not None]
    second_gen_tensors = [t for t in plan_shard_tensors(plan) if t.action == "affine"]
    bit_exact_tensors = [t for t in plan_shard_tensors(plan) if t.bit_exact]
    quant_assignment = {
        "schema_version": 1,
        "kind": "quant_assignment",
        "candidate_id": candidate_id,
        "source_revision": REVISION,
        "conventions": {
            "affine": (
                "scale slot = int8 power-of-two exponent e, zero slot = int8 "
                "zero-point code z; dequant value = (code - z) * 2**e; codes "
                "packed bits-wide LSB-first, rows byte-aligned"
            ),
            "mxfp4": (
                "bit-exact copy of packed U8 weights (low nibble = even column) "
                "and U8 E8M0 scale siblings (2**(byte - 127))"
            ),
        },
        "formats": {
            fmt: {
                **{k: v for k, v in FORMATS[fmt].items() if k != "source_dtypes"},
                "source_dtypes": list(FORMATS[fmt]["source_dtypes"]),
            }
            for fmt in sorted({row["format"] for row in parsed["rows"].values()})
        },
        "classes": [
            {
                "class": cls,
                "format": parsed["rows"][cls]["format"],
                "unit_label": parsed["rows"][cls]["unit_label"],
                "units": parsed["rows"][cls]["units"],
                "bytes_per_unit": parsed["rows"][cls]["bytes_per_unit"],
                "class_bytes": parsed["rows"][cls]["class_bytes"],
                "planned_units": plan.classes[cls]["planned_units"]
                if cls != "router"
                else plan.router_units,
                "planned_bytes": plan.classes[cls]["planned_bytes"],
                "second_gen": parsed["rows"][cls]["second_gen"],
                "bit_exact": parsed["rows"][cls]["bit_exact"],
            }
            for cls in sorted(parsed["rows"])
        ],
        "second_gen_applied": recipe["second_gen_applied"],
        "expert_dispatch": plan.dispatch,
        "accounting": {
            "allocation_table_sum": sum(
                row["class_bytes"] for row in parsed["rows"].values()
            ),
            "predicted_text_bytes": plan.total_text_bytes,
            "closed": True,
        },
        "tensors": {
            t.out_name: {
                "class": t.cls,
                "format": t.fmt,
                "component": t.component,
                "action": t.action,
                "out_dtype": t.out_dtype,
                "out_shape": list(t.out_shape),
                "out_bytes": t.out_bytes,
                "source_shard": t.src_shard,
                "source_tensor": t.src_name,
                "input_sha256": t.input_sha256,
                "output_sha256": t.output_sha256,
                **({"error": t.error} if t.error else {}),
            }
            for t in sorted(plan_shard_tensors(plan), key=lambda t: t.out_name)
        },
        "second_gen_error_summary": {
            "quantized_tensors": len(second_gen_tensors),
            "max_abs_err": max((entry["max_abs_err"] for entry in errors), default=0.0),
            "reference": (
                "BF16-decoded source values (compute-only); never a pre-QAT "
                "full-precision claim"
            ),
        },
    }
    quant_path = out_path / QUANT_ASSIGNMENT_NAME
    _write_json_checked(quant_path, quant_assignment)

    build_report = {
        "schema_version": 1,
        "kind": "build_report",
        "candidate_id": candidate_id,
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "verified_source_root_basename": source_root_real.name,
            "source_shard_digest_verification": (
                "shard digests are not re-hashed by the builder; per-tensor input "
                "sha256 values are recorded in quant_assignment.json, metadata "
                "files are verified against pinned lfs_sha256, and the source-tree "
                "digest basis is the recipe's source_weights.digest_basis"
            ),
        },
        "inputs": {
            "inventory_path": str(inventory_path),
            "prune_map_path": str(prune_map_path),
            "prune_map_sha256": pmap["digest"],
            "prune_map_mode": pmap["mode"],
            "prune_map_quality_map": pmap["quality_map"],
            "recipe_sha256": _json_digest(recipe),
            "seed": seed,
        },
        "resident_weight_bytes": plan.total_text_bytes,
        "pure_mxfp4_floor_bytes": plan.floor_bytes,
        "accounting": {
            "allocation_table_sum": sum(
                row["class_bytes"] for row in parsed["rows"].values()
            ),
            "planned_text_bytes": plan.total_text_bytes,
            "written_text_bytes": sum(
                t.out_bytes
                for shard in plan.shards.values()
                for t in shard
                if t.cls
            ),
            "closed": True,
            "aux_written_tensors": plan.aux_tensors,
            "aux_written_bytes": plan.aux_bytes,
            "static_then_dynamic": (
                "static plan closed against allocation_table before writing; "
                "per-tensor and per-shard byte checks re-closed after writing"
            ),
        },
        "bit_exact": {
            "tensors_checked": len(bit_exact_tensors),
            "method": (
                "input sha256 computed while reading the source range; output "
                "sha256 recomputed by an independent read-back of the written "
                "range; equality enforced"
            ),
        },
        "second_gen": {
            "quantized_tensors": len(second_gen_tensors),
            "max_abs_err": quant_assignment["second_gen_error_summary"]["max_abs_err"],
        },
        "output_shards": [shard_facts[name] for name in sorted(shard_facts)],
        "metadata_files": plan.metadata,
        "config_adaptation": plan.config_adaptation,
        "index_regenerated": plan.index_doc is not None,
        "excluded_files": plan.excluded_files,
        "excluded_components": {
            path: {
                "tensors": bucket["tensors"],
                "stored_bytes": bucket["stored_bytes"],
                "components": sorted(bucket["components"]),
            }
            for path, bucket in sorted(plan.excluded_components.items())
        },
        "expert_dispatch": plan.dispatch,
        "conversion_commit": conversion_commit,
        "runtime_commit": runtime_commit,
    }
    report_path = out_path / BUILD_REPORT_NAME
    _write_json_checked(report_path, build_report)

    record_notes = (
        f"constructed by mimo_halo.build.candidate from the pinned source "
        f"revision; prune map mode={pmap['mode']}, "
        f"quality_map={str(pmap['quality_map']).lower()}; Q3 order basis: "
        f"{plan.dispatch['order_basis']}"
    )
    if plan.excluded_files or plan.excluded_components:
        record_notes += (
            f"; {len(plan.excluded_files)} auxiliary source files and "
            f"{len(plan.excluded_components)} subdir shards excluded from the flat "
            "artifact, recorded byte-exact in build_report.json"
        )
    if notes:
        record_notes = record_notes + "; " + notes
    record = candidate_record(recipe, source_info, build_report=build_report, notes=record_notes)
    record_path = out_path / CANDIDATE_RECORD_NAME
    _write_json_checked(record_path, record)

    # --- artifact sidecars ---------------------------------------------
    payload_paths = (
        [out_path / name for name in sorted(shard_facts)]
        + [out_path / entry["path"] for entry in plan.metadata]
        + ([out_path / INDEX_NAME] if plan.index_doc is not None else [])
        + [record_path, report_path]
    )
    if commands is None:
        commands = [
            "mimo_halo.build.candidate (library call; exact arguments recorded in "
            "build_report.json inputs)"
        ]
    try:
        manifest = artifacts.create_artifact(
            str(out_path),
            kind,
            sources=[
                artifacts.parse_source_spec(
                    f"model=https://huggingface.co/{REPO_ID}@{REVISION}"
                )
            ],
            commands=list(commands),
            environment_path=str(environment_path),
            dataset_hashes=list(dataset_hashes),
            expert_map_path=str(prune_map_path),
            calibration_config_path=(
                str(calibration_config_path) if calibration_config_path else None
            ),
            quant_assignment_path=str(quant_path),
            payload_paths=[str(p) for p in payload_paths],
            parents=[artifacts.parse_parent_spec(spec) for spec in parents],
            seed=seed,
            conversion_commit=conversion_commit,
            runtime_commit=runtime_commit,
            signed_sources=list(signed_sources),
            production=production,
            official_source=official_source,
            created_at=created_at,
        )
        verify = artifacts.verify_artifact(str(out_path))
    except artifacts.ArtifactError as exc:
        raise BuildError(f"artifact sidecar creation/verification failed: {exc}") from exc
    if verify.get("status") != "verified":  # pragma: no cover - verify raises otherwise
        raise BuildError(f"artifact verification did not pass: {verify!r}")

    return {
        "out_dir": str(out_path),
        "manifest": manifest,
        "verify": verify,
        "record": record,
        "build_report": build_report,
        "quant_assignment_path": str(quant_path),
        "record_path": str(record_path),
        "report_path": str(report_path),
    }


def plan_shard_tensors(plan: Plan):
    for shard_rel in sorted(plan.shards):
        yield from plan.shards[shard_rel]
