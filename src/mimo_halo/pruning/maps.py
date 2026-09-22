"""Prune-map planning for MiMo-V2.6-Flash-RL MoE experts.

Consumes an inventory JSON (``schema_version=1``, tensors with exact
``logical_parameters`` / ``stored_bytes`` derived from official shapes and
quantization packing) and produces a *planning-only* prune map:

* per-layer old->new expert id maps, retained/pruned ids,
* router gating weight + e-score correction bias remap plan,
* expert weight + weight_scale keep/drop tensor lists,
* exact logical parameter and native (stored) byte totals taken from the
  inventory, never recounted or guessed,
* exact post-prune payload/logical totals per component (text/main vs
  MTP/DFlash/vision/audio), with dense BF16/F32 router bytes row-scaled
  exactly and packed/unknown formats kept explicitly null — never guessed.

Two selection modes:

* ``--retained-count N --seed S`` generates a deterministic **shape-only**
  selection. It is explicitly NOT a REAP/HOPE quality map: no routing
  statistics, activations, or quality signals are used, and the generated
  candidate carries that disclaimer in its provenance.
* ``--selection candidate.json`` consumes an external actual selection that
  uses original expert ids and preserves the given order (order defines new
  ids).

Nothing here slices weights or reloads a checkpoint: output always records
``"applied": false``. Unknown or ambiguous logical packing fails with an
exact-count requirement rather than being guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

SCHEMA_VERSION = 1
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PROJECTION_WEIGHT_KINDS = ("weight", "weight_scale")

_TEXT_MAIN_COMPONENTS = {"text/main", "text_main", "text-main", "text", "main"}
# Normalized once so _text_main has a single authority for accepted spellings.
_TEXT_MAIN_COMPONENT_NORM = {
    c.lower().replace("-", "_").replace("/", "_") for c in _TEXT_MAIN_COMPONENTS
}
_EXPERT_TENSOR_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)"
    r"\.(weight|weight_scale)$"
)
_ROUTER_TENSOR_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias)$"
)
# Dense row-uniform router dtypes: dtype -> exact itemsize. Only these exact
# types are row-scaled to exact post-remap stored bytes; any other dtype —
# packed, quantized, or unknown — keeps an explicit null figure whose cost is
# never guessed. No dtype is inferred from byte counts.
_DENSE_ROW_UNIFORM_DTYPES = {"BF16": 2, "F32": 4}


class PruneMapError(Exception):
    """A validation failure that must abort the dry-run."""


# --------------------------------------------------------------------------
# Inventory loading (schema consumption only; no inventory-module imports)
# --------------------------------------------------------------------------

def load_inventory(path: str | Path) -> dict:
    """Load and structurally validate an inventory JSON document."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PruneMapError(f"cannot read inventory {path}: {exc}") from exc
    try:
        inv = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PruneMapError(f"inventory {path} is not valid JSON: {exc}") from exc
    if not isinstance(inv, dict):
        raise PruneMapError("inventory must be a JSON object")
    if inv.get("schema_version") != SCHEMA_VERSION:
        raise PruneMapError(
            f"unsupported inventory schema_version {inv.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    source = inv.get("source")
    if not isinstance(source, dict) or "repo_id" not in source or "revision" not in source:
        raise PruneMapError("inventory.source must include repo_id and revision")
    arch = inv.get("architecture")
    if not isinstance(arch, dict):
        raise PruneMapError("inventory.architecture missing")
    experts = arch.get("original_experts_per_layer")
    top_k = arch.get("top_k")
    moe_layers = arch.get("moe_layers")
    if not isinstance(experts, int) or isinstance(experts, bool) or experts <= 0:
        raise PruneMapError("architecture.original_experts_per_layer must be a positive int")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise PruneMapError("architecture.top_k must be a positive int")
    if not isinstance(moe_layers, list) or not moe_layers:
        raise PruneMapError("architecture.moe_layers must be a non-empty list of ints")
    if any(not isinstance(x, int) or isinstance(x, bool) for x in moe_layers):
        raise PruneMapError("architecture.moe_layers must be a non-empty list of ints")
    if len(set(moe_layers)) != len(moe_layers):
        raise PruneMapError("architecture.moe_layers contains duplicates")
    tensors = inv.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise PruneMapError("inventory.tensors must be a non-empty list")
    for t in tensors:
        _validate_tensor_entry(t)
    return inv


def _validate_tensor_entry(t: object) -> None:
    if not isinstance(t, dict):
        raise PruneMapError("inventory.tensors entries must be objects")
    name = t.get("name")
    if not isinstance(name, str) or not name:
        raise PruneMapError(f"tensor entry missing name: {t!r}")
    where = f"tensor {name}"
    shape = t.get("shape")
    if (
        not isinstance(shape, list)
        or any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in shape)
    ):
        raise PruneMapError(f"{where}: shape must be a list of non-negative ints")
    if not isinstance(t.get("dtype"), str) or not t["dtype"]:
        raise PruneMapError(f"{where}: dtype must be a non-empty string")
    for field in ("stored_bytes",):
        value = t.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PruneMapError(f"{where}: {field} must be a non-negative int")
    role = t.get("role")
    if role not in ("parameter", "quantization_auxiliary", "buffer"):
        raise PruneMapError(f"{where}: unknown role {role!r}")
    logical = t.get("logical_parameters")
    if role == "parameter":
        if not isinstance(logical, int) or isinstance(logical, bool) or logical < 0:
            raise PruneMapError(
                f"{where}: role=parameter requires an exact non-negative int "
                "logical_parameters (unknown logical packing must fail, not be guessed)"
            )
    else:
        if logical not in (0, None):
            raise PruneMapError(f"{where}: {role} tensors must carry logical_parameters=0")
    logical_shape = t.get("logical_shape", shape)
    if (
        not isinstance(logical_shape, list)
        or any(not isinstance(d, int) or isinstance(d, bool) for d in logical_shape)
    ):
        raise PruneMapError(f"{where}: logical_shape must be a list of ints")


def _text_main(tensor: dict) -> bool:
    """True when the tensor belongs to the main text stack (the prunable one)."""
    component = tensor.get("component")
    if isinstance(component, str) and component.strip():
        normalized = component.strip().lower().replace("-", "_").replace("/", "_")
        return normalized in _TEXT_MAIN_COMPONENT_NORM
    # Component missing: fall back to official tensor-name prefixes.
    return tensor.get("name", "").startswith("model.layers.")


def _component_key(tensor: dict) -> str:
    """Bucket name for post-prune by-component totals.

    Every accepted text-main spelling collapses to the canonical
    ``"text/main"`` so pruned-stack buckets compare across inventories;
    other components keep their raw (trimmed) value, and a tensor with no
    component at all uses the same official-prefix rule as :func:`_text_main`.
    """
    component = tensor.get("component")
    if isinstance(component, str) and component.strip():
        stripped = component.strip()
        normalized = stripped.lower().replace("-", "_").replace("/", "_")
        if normalized in _TEXT_MAIN_COMPONENT_NORM:
            return "text/main"
        return stripped
    if tensor.get("name", "").startswith("model.layers."):
        return "text/main"
    return "(no component)"


# --------------------------------------------------------------------------
# Expert/router structure extraction
# --------------------------------------------------------------------------

def _dims_of(tensor: dict) -> list[int]:
    logical_shape = tensor.get("logical_shape")
    return logical_shape if isinstance(logical_shape, list) else tensor["shape"]


def collect_moe_structure(inv: dict) -> dict:
    """Extract per-layer expert tensors and router tensors from the inventory.

    Returns ``{"architecture":…, "layers": {L: {"experts": {E: {"gate_proj":
    {"weight": t, "weight_scale": t}, …}, "router": {"weight": t, "bias": t}}},
    "untouched": [tensors]}`` for main-text tensors only.
    """
    arch = inv["architecture"]
    n_experts = arch["original_experts_per_layer"]
    moe_layers = set(arch["moe_layers"])
    layers: dict[int, dict] = {L: {"experts": {}, "router": {}} for L in arch["moe_layers"]}
    untouched: list[dict] = []
    for tensor in inv["tensors"]:
        if not _text_main(tensor):
            untouched.append(tensor)
            continue
        name = tensor["name"]
        expert_match = _EXPERT_TENSOR_RE.match(name)
        if expert_match:
            layer, expert, proj, kind = expert_match.groups()
            layer, expert = int(layer), int(expert)
            if layer not in moe_layers:
                raise PruneMapError(
                    f"unexpected expert tensor {name} in non-MoE layer {layer}"
                )
            if expert >= n_experts:
                raise PruneMapError(
                    f"bad expert id {expert} in {name}: out of range [0, {n_experts})"
                )
            bucket = layers[layer]["experts"].setdefault(expert, {})
            if proj in bucket and kind in bucket[proj]:
                raise PruneMapError(f"duplicate inventory tensor {name}")
            bucket.setdefault(proj, {})[kind] = tensor
            continue
        router_match = _ROUTER_TENSOR_RE.match(name)
        if router_match:
            layer, kind = router_match.groups()
            layer = int(layer)
            if layer not in moe_layers:
                raise PruneMapError(
                    f"unexpected router tensor {name} in non-MoE layer {layer}"
                )
            if kind in layers[layer]["router"]:
                raise PruneMapError(f"duplicate inventory tensor {name}")
            layers[layer]["router"][kind] = tensor
            continue
        untouched.append(tensor)

    # Completeness: every MoE layer, every expert, every projection.
    for layer in sorted(layers):
        router = layers[layer]["router"]
        if "weight" not in router:
            raise PruneMapError(f"layer {layer}: missing router tensor *.mlp.gate.weight")
        if "e_score_correction_bias" not in router:
            raise PruneMapError(
                f"layer {layer}: missing router tensor *.mlp.gate.e_score_correction_bias"
            )
        expected = set(range(n_experts))
        found = set(layers[layer]["experts"])
        if found != expected:
            missing = sorted(expected - found)[:8]
            raise PruneMapError(
                f"layer {layer}: missing experts {missing}"
                + ("…" if len(expected - found) > 8 else "")
            )
        for expert in sorted(found):
            for proj in EXPERT_PROJECTIONS:
                entries = layers[layer]["experts"][expert].get(proj, {})
                for kind in PROJECTION_WEIGHT_KINDS:
                    if kind not in entries:
                        raise PruneMapError(
                            f"layer {layer} expert {expert}: missing {proj}.{kind}"
                        )

    # Router dimensions must match the routed expert count.
    for layer, bucket in sorted(layers.items()):
        for kind, router in bucket["router"].items():
            dims = _dims_of(router)
            if not dims or dims[0] != n_experts:
                raise PruneMapError(
                    f"layer {layer}: router {kind} leading dimension "
                    f"{dims[0] if dims else None} != original_experts_per_layer {n_experts}"
                )

    # dtype uniformity across the whole expert set (dry-run byte math needs it).
    for kind in PROJECTION_WEIGHT_KINDS:
        for proj in EXPERT_PROJECTIONS:
            dtypes: set[str] = set()
            for bucket in layers.values():
                for expert in bucket["experts"].values():
                    dtypes.add(expert[proj][kind]["dtype"])
            if len(dtypes) > 1:
                raise PruneMapError(
                    f"dtype change: {proj}.{kind} dtypes differ across layers/experts: "
                    + ", ".join(sorted(dtypes))
                )
    for kind in ("weight", "e_score_correction_bias"):
        dtypes = {bucket["router"][kind]["dtype"] for bucket in layers.values()}
        if len(dtypes) > 1:
            raise PruneMapError(
                f"dtype change: router {kind} dtypes differ across layers: "
                + ", ".join(sorted(dtypes))
            )

    return {"architecture": arch, "layers": layers, "untouched": untouched}


# --------------------------------------------------------------------------
# Selection generation and validation
# --------------------------------------------------------------------------

def generate_selection(arch: dict, retained_count: int, seed: int) -> dict[int, list[int]]:
    """Deterministic shape-only selection, per layer, in generation order.

    Uses a per-layer seeded RNG so the result depends only on
    ``(seed, layer, expert count, retained_count)`` — reproducible across
    runs, layer iteration order, and platforms. This is a shape-only
    placeholder, NOT a REAP/HOPE quality map.
    """
    n_experts = arch["original_experts_per_layer"]
    _validate_retained_count(retained_count, n_experts)
    selection: dict[int, list[int]] = {}
    for layer in sorted(arch["moe_layers"]):
        rng = random.Random(f"{seed}:{layer}")
        selection[layer] = rng.sample(range(n_experts), retained_count)
    return selection


def _validate_retained_count(retained_count: int, n_experts: int) -> None:
    if not isinstance(retained_count, int) or isinstance(retained_count, bool):
        raise PruneMapError("--retained-count must be an integer")
    if not 1 <= retained_count <= n_experts:
        raise PruneMapError(
            f"retained count {retained_count} out of range [1, {n_experts}]"
        )


def parse_external_selection(data: object, arch: dict) -> dict[int, list[int]]:
    """Validate an external candidate selection (original ids, order kept).

    New expert ids are assigned by position in each layer's ``retained`` list;
    the given order is authoritative and never re-sorted.
    """
    n_experts = arch["original_experts_per_layer"]
    top_k = arch["top_k"]
    if not isinstance(data, dict):
        raise PruneMapError("selection must be a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise PruneMapError(
            f"unsupported selection schema_version {data.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    declared_top_k = data.get("top_k")
    if declared_top_k is not None and declared_top_k != top_k:
        raise PruneMapError(
            f"top_k change rejected: selection declares {declared_top_k}, "
            f"architecture routing stays top-{top_k}"
        )
    entries = data.get("layers")
    if not isinstance(entries, list) or not entries:
        raise PruneMapError("selection.layers must be a non-empty list")
    moe_layers = set(arch["moe_layers"])
    selection: dict[int, list[int]] = {}
    seen_layers: set[int] = set()
    uniform_count: int | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            raise PruneMapError("selection.layers entries must be objects")
        layer = entry.get("layer")
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise PruneMapError(f"selection layer {layer!r}: layer must be an int")
        if layer not in moe_layers:
            raise PruneMapError(
                f"selection layer {layer} is not a MoE layer of the architecture"
            )
        if layer in seen_layers:
            raise PruneMapError(f"selection has duplicate entries for layer {layer}")
        seen_layers.add(layer)
        declared_original = entry.get("original_expert_count")
        if declared_original is not None and declared_original != n_experts:
            raise PruneMapError(
                f"layer {layer}: original_expert_count {declared_original} != "
                f"architecture original_experts_per_layer {n_experts}"
            )
        retained = entry.get("retained_expert_ids")
        if not isinstance(retained, list) or not retained:
            raise PruneMapError(
                f"layer {layer}: retained_expert_ids must be a non-empty list"
            )
        if len(retained) != len(set(retained)):
            raise PruneMapError(f"layer {layer}: duplicate expert ids in retained list")
        for expert in retained:
            if isinstance(expert, bool) or not isinstance(expert, int):
                raise PruneMapError(
                    f"layer {layer}: bad expert id {expert!r} (must be an int)"
                )
            if not 0 <= expert < n_experts:
                raise PruneMapError(
                    f"layer {layer}: expert id {expert} out of range [0, {n_experts})"
                )
        if uniform_count is None:
            uniform_count = len(retained)
        elif len(retained) != uniform_count:
            raise PruneMapError(
                f"unequal count: layer {layer} retains {len(retained)} experts, "
                f"layer with first entry retains {uniform_count}"
            )
        selection[layer] = list(retained)  # order preserved exactly as given
        declared_pruned = entry.get("pruned_expert_ids")
        if declared_pruned is not None:
            if not isinstance(declared_pruned, list):
                raise PruneMapError(f"layer {layer}: pruned_expert_ids must be a list")
            if sorted(declared_pruned) != sorted(set(range(n_experts)) - set(retained)):
                raise PruneMapError(
                    f"layer {layer}: pruned_expert_ids do not match the retained set"
                )
    missing = sorted(moe_layers - seen_layers)
    if missing:
        raise PruneMapError(
            f"selection is missing layers {missing[:8]}"
            + ("…" if len(missing) > 8 else "")
        )
    return selection


def load_selection_file(path: str | Path, arch: dict) -> tuple[dict[int, list[int]], dict]:
    """Load a candidate selection file and return (selection, provenance)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PruneMapError(f"cannot read selection {path}: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PruneMapError(f"selection {path} is not valid JSON: {exc}") from exc
    selection = parse_external_selection(data, arch)
    provenance = {
        "mode": "external_selection",
        "selection_path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "order_preserved": True,
        "quality_map": bool(data.get("quality_map", False)),
    }
    return selection, provenance


# --------------------------------------------------------------------------
# Map construction
# --------------------------------------------------------------------------

def _sum(tensors: list[dict], field: str) -> int:
    # Non-parameter roles may carry logical_parameters=null; they count as 0.
    return sum(t[field] for t in tensors if t[field] is not None)


def _dense_post_remap_bytes(
    tensor: dict, dims: list[int], new_rows: int, where: str
) -> int | None:
    """Exact post-remap stored bytes for a dense row-uniform router tensor.

    Dense (``quantization: null``) BF16/F32 tensors store fixed-width rows, so
    the retained payload scales exactly:
    ``stored_bytes // original_rows * new_rows``. The layout is validated, not
    assumed: ``stored_bytes`` must be divisible by the original row count and
    each row must equal ``prod(dims[1:]) * itemsize`` bytes — a contradiction
    fails closed rather than being estimated. Packed/quantized tensors and any
    other dtype have an unknown packing cost and return ``None``: explicitly
    unknown, never guessed.
    """
    if tensor.get("quantization") is not None:
        return None
    itemsize = _DENSE_ROW_UNIFORM_DTYPES.get(tensor["dtype"].upper())
    if itemsize is None:
        return None
    old_rows = dims[0]
    row_items = 1
    for d in dims[1:]:
        row_items *= d
    row_bytes = row_items * itemsize
    old_bytes = tensor["stored_bytes"]
    if old_rows <= 0 or old_bytes % old_rows != 0 or old_bytes // old_rows != row_bytes:
        raise PruneMapError(
            f"{where}: dense {tensor['dtype']} stored_bytes {old_bytes} is not "
            f"row-uniform over {old_rows} original rows ({row_bytes} bytes/row "
            "expected); exact row-scaling refused rather than guessed"
        )
    return old_bytes // old_rows * new_rows


def build_prune_map(
    inv: dict,
    selection: dict[int, list[int]],
    mode: str,
    provenance: dict,
    inventory_path: str,
) -> dict:
    """Build the complete dry-run prune-map document from a validated selection."""
    struct = collect_moe_structure(inv)
    arch = struct["architecture"]
    n_experts = arch["original_experts_per_layer"]
    top_k = arch["top_k"]
    layers_out: list[dict] = []
    totals = {
        "expert_tensors_kept": 0,
        "expert_tensors_dropped": 0,
        "logical_parameters_kept": 0,
        "logical_parameters_dropped": 0,
        "logical_parameters_router_remapped": 0,
        "stored_bytes_kept": 0,
        "stored_bytes_dropped": 0,
        "router_tensors": 0,
        "router_stored_bytes": 0,
        "router_logical_parameters": 0,
    }
    # Exact dense post-remap router payload bytes, accumulated only while every
    # router tensor is provably row-uniform dense; one unknown format makes the
    # whole figure null (unknown, never guessed).
    router_post_bytes_sum = 0
    router_post_known = True
    for layer in sorted(struct["layers"]):
        bucket = struct["layers"][layer]
        retained = selection[layer]
        if len(retained) != len(set(retained)):
            raise PruneMapError(f"layer {layer}: duplicate expert ids in selection")
        if any(e < 0 or e >= n_experts for e in retained):
            raise PruneMapError(f"layer {layer}: expert id out of range")
        pruned = sorted(set(range(n_experts)) - set(retained))
        old_to_new: dict[str, int | None] = {}
        for new_id, old_id in enumerate(retained):
            old_to_new[str(old_id)] = new_id
        for old_id in pruned:
            old_to_new[str(old_id)] = None

        keep: list[str] = []
        drop: list[str] = []
        keep_tensors: list[dict] = []
        drop_tensors: list[dict] = []
        for expert, entries in sorted(bucket["experts"].items()):
            for proj in EXPERT_PROJECTIONS:
                for kind in PROJECTION_WEIGHT_KINDS:
                    tensor = entries[proj][kind]
                    if expert in set(retained):
                        keep.append(tensor["name"])
                        keep_tensors.append(tensor)
                    else:
                        drop.append(tensor["name"])
                        drop_tensors.append(tensor)
        keep.sort()
        drop.sort()

        router_w = bucket["router"]["weight"]
        router_b = bucket["router"]["e_score_correction_bias"]
        w_dims = _dims_of(router_w)
        b_dims = _dims_of(router_b)
        # Exact post-remap router size: leading dim retained_count, rest unchanged.
        w_new_dims = [len(retained)] + w_dims[1:]
        b_new_dims = [len(retained)] + b_dims[1:]
        w_rest = 1
        for d in w_dims[1:]:
            w_rest *= d
        b_rest = 1
        for d in b_dims[1:]:
            b_rest *= d
        w_new_params = len(retained) * w_rest
        b_new_params = len(retained) * b_rest
        if w_new_params > router_w["logical_parameters"] or b_new_params > router_b["logical_parameters"]:
            raise PruneMapError(
                f"layer {layer}: router remap grows parameters; refusing to guess"
            )
        router_original_bytes = router_w["stored_bytes"] + router_b["stored_bytes"]
        router_original_params = _sum([router_w, router_b], "logical_parameters")
        w_post_bytes = _dense_post_remap_bytes(
            router_w, w_dims, len(retained), f"layer {layer} router weight"
        )
        b_post_bytes = _dense_post_remap_bytes(
            router_b,
            b_dims,
            len(retained),
            f"layer {layer} router e_score_correction_bias",
        )
        layer_post_bytes = (
            w_post_bytes + b_post_bytes
            if w_post_bytes is not None and b_post_bytes is not None
            else None
        )
        if layer_post_bytes is None:
            router_post_known = False
        else:
            router_post_bytes_sum += layer_post_bytes
        totals["logical_parameters_router_remapped"] += w_new_params + b_new_params
        totals["router_tensors"] += 2
        totals["router_stored_bytes"] += router_original_bytes
        totals["router_logical_parameters"] += router_original_params

        kept_params = _sum(keep_tensors, "logical_parameters")
        kept_bytes = _sum(keep_tensors, "stored_bytes")
        dropped_params = _sum(drop_tensors, "logical_parameters")
        dropped_bytes = _sum(drop_tensors, "stored_bytes")
        totals["expert_tensors_kept"] += len(keep)
        totals["expert_tensors_dropped"] += len(drop)
        totals["logical_parameters_kept"] += kept_params
        totals["logical_parameters_dropped"] += dropped_params
        totals["stored_bytes_kept"] += kept_bytes
        totals["stored_bytes_dropped"] += dropped_bytes

        layers_out.append(
            {
                "layer": layer,
                "component": "text/main",
                "retained": list(retained),
                "pruned": pruned,
                "old_to_new": old_to_new,
                "router": {
                    "weight": {
                        "name": router_w["name"],
                        "logical_shape": w_dims,
                        "new_logical_shape": w_new_dims,
                        "rows_kept": list(retained),
                        "new_logical_parameters": w_new_params,
                        "new_stored_bytes": w_post_bytes,
                    },
                    "e_score_correction_bias": {
                        "name": router_b["name"],
                        "logical_shape": b_dims,
                        "new_logical_shape": b_new_dims,
                        "entries_kept": list(retained),
                        "new_logical_parameters": b_new_params,
                        "new_stored_bytes": b_post_bytes,
                    },
                    "plan": (
                        "slice gating-weight rows and correction-bias entries for "
                        "retained experts in the new-id order given by old_to_new; "
                        f"routing stays top-{top_k} over the retained set"
                    ),
                    "accounting": {
                        "tensor_count": 2,
                        "original_stored_bytes": router_original_bytes,
                        "original_logical_parameters": router_original_params,
                        "post_remap_logical_parameters": w_new_params + b_new_params,
                        "post_remap_stored_bytes": layer_post_bytes,
                        "post_remap_stored_bytes_known": layer_post_bytes is not None,
                        "post_remap_stored_bytes_note": (
                            "exact dense row-scaling of native payload bytes "
                            "(quantization null, BF16/F32, row-uniform); no dtype "
                            "guess, no format or metadata overhead"
                            if layer_post_bytes is not None
                            else "post-remap stored bytes unknown: packed/quantized "
                            "or non-BF16/F32 router format; packing cost never guessed"
                        ),
                    },
                },
                "expert_tensors": {"keep": keep, "drop": drop},
                "tensor_counts": {
                    "keep": len(keep),
                    "drop": len(drop),
                    "total": len(keep) + len(drop),
                },
                "logical_parameters": {
                    "keep": kept_params,
                    "drop": dropped_params,
                    "total": kept_params + dropped_params,
                },
                "stored_bytes": {
                    "keep": kept_bytes,
                    "drop": dropped_bytes,
                    "total": kept_bytes + dropped_bytes,
                },
            }
        )

    untouched = struct["untouched"]
    totals["unchanged_tensors"] = len(untouched)
    totals["unchanged_stored_bytes"] = _sum(untouched, "stored_bytes")
    totals["unchanged_logical_parameters"] = _sum(untouched, "logical_parameters")

    # Exact closure: expert keep/drop + routers + unchanged must account for
    # every inventory tensor's count, stored bytes, and logical parameters.
    # Original router values are used here; post-remap router parameters live
    # in logical_parameters_router_remapped and exact dense post-remap router
    # bytes — null only while some router format keeps the cost unknowable —
    # in router_stored_bytes_post_remap. The closure identity itself never moves.
    inventory_tensors = len(inv["tensors"])
    inventory_stored_bytes = _sum(inv["tensors"], "stored_bytes")
    inventory_logical_parameters = _sum(inv["tensors"], "logical_parameters")
    accounted_tensors = (
        totals["expert_tensors_kept"]
        + totals["expert_tensors_dropped"]
        + totals["router_tensors"]
        + totals["unchanged_tensors"]
    )
    accounted_stored_bytes = (
        totals["stored_bytes_kept"]
        + totals["stored_bytes_dropped"]
        + totals["router_stored_bytes"]
        + totals["unchanged_stored_bytes"]
    )
    accounted_logical_parameters = (
        totals["logical_parameters_kept"]
        + totals["logical_parameters_dropped"]
        + totals["router_logical_parameters"]
        + totals["unchanged_logical_parameters"]
    )
    if (
        accounted_tensors != inventory_tensors
        or accounted_stored_bytes != inventory_stored_bytes
        or accounted_logical_parameters != inventory_logical_parameters
    ):
        raise PruneMapError(
            "accounting closure failed: keep+drop+router+unchanged does not "
            f"equal inventory totals (tensors {accounted_tensors} != "
            f"{inventory_tensors}, stored_bytes {accounted_stored_bytes} != "
            f"{inventory_stored_bytes}, logical_parameters "
            f"{accounted_logical_parameters} != {inventory_logical_parameters})"
        )
    totals["router_stored_bytes_post_remap"] = (
        router_post_bytes_sum if router_post_known else None
    )
    totals["router_stored_bytes_post_remap_known"] = router_post_known
    totals["router_stored_bytes_post_remap_note"] = (
        "exact dense row-scaled native payload bytes for every router tensor "
        "(quantization null, BF16/F32, row-uniform); null only when a router "
        "is packed/quantized or of unknown format — that cost is never guessed"
        if router_post_known
        else "post-remap router stored bytes unknown: at least one router is "
        "packed/quantized or of non-BF16/F32 format; packing cost never guessed"
    )
    totals["accounting_closure"] = {
        "inventory_tensors": inventory_tensors,
        "accounted_tensors": accounted_tensors,
        "inventory_stored_bytes": inventory_stored_bytes,
        "accounted_stored_bytes": accounted_stored_bytes,
        "inventory_logical_parameters": inventory_logical_parameters,
        "accounted_logical_parameters": accounted_logical_parameters,
        "closes": True,
    }

    # Exact post-prune payload/logical totals by component. The text/main
    # bucket is the pruned stack — kept experts + sliced routers + untouched
    # text tensors — so a consumer can compare against a memory report's text
    # candidate totals without subtracting MTP/DFlash/vision/audio by hand.
    # stored_bytes are native tensor payload bytes: no dtype guess, no
    # GGUF/container format or metadata overhead is claimed pre-export.
    component_totals: dict[str, dict] = {}
    for tensor in untouched:
        slot = component_totals.setdefault(
            _component_key(tensor),
            {
                "tensor_count": 0,
                "stored_bytes": 0,
                "logical_parameters": 0,
                "stored_bytes_known": True,
            },
        )
        slot["tensor_count"] += 1
        slot["stored_bytes"] += tensor["stored_bytes"]
        slot["logical_parameters"] += tensor["logical_parameters"] or 0
    text_slot = component_totals.setdefault(
        "text/main",
        {
            "tensor_count": 0,
            "stored_bytes": 0,
            "logical_parameters": 0,
            "stored_bytes_known": True,
        },
    )
    text_slot["tensor_count"] += totals["expert_tensors_kept"] + totals["router_tensors"]
    text_slot["stored_bytes"] += totals["stored_bytes_kept"]
    text_slot["logical_parameters"] += (
        totals["logical_parameters_kept"]
        + totals["logical_parameters_router_remapped"]
    )
    text_slot["stored_bytes_known"] = router_post_known
    if router_post_known:
        text_slot["stored_bytes"] += router_post_bytes_sum
    else:
        text_slot["stored_bytes"] = None
    totals["post_prune_by_component"] = dict(sorted(component_totals.items()))

    retained_count = len(next(iter(selection.values())))
    warnings = [
        "Planning-only dry run: no weights were read, sliced, or written; "
        "'applied' is false and no checkpoint reload is claimed.",
        "Router gating rows are sliced without retraining or calibration; "
        "sigmoid/top-k routing renormalizes over the retained expert set.",
    ]
    if mode == "shape_only":
        warnings.insert(
            0,
            "Shape-only deterministic selection: NOT a REAP/HOPE quality map; "
            "no routing statistics or activations were used.",
        )
    mtp = arch.get("mtp_layer_count")
    if mtp is None:
        warnings.append(
            "MTP layer count unresolved by this inventory (config says 3, model card "
            "prose says 5): expert counts here cover the main text stack only."
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": "prune_map",
        "mode": mode,
        "applied": False,
        "quality_map": False,
        "source": {
            "inventory_path": str(inventory_path),
            "repo_id": inv["source"]["repo_id"],
            "revision": inv["source"]["revision"],
            "files": inv["source"].get("files"),
        },
        "architecture": {
            "original_experts_per_layer": n_experts,
            "top_k": top_k,
            "moe_layers": sorted(arch["moe_layers"]),
        },
        "retained_count": retained_count,
        "provenance": provenance,
        "warnings": warnings,
        "candidate": _candidate_block(arch, selection, mode, provenance),
        "layers": layers_out,
        "totals": totals,
    }
    return document


def _candidate_block(
    arch: dict,
    selection: dict[int, list[int]],
    mode: str,
    provenance: dict,
) -> dict:
    """Candidate JSON compatible with the baseline comparison layer schema.

    Authoritative contract (local://baseline-contract.md): top-level
    ``schema_version=1`` and ``layers`` of
    ``{layer, original_expert_count, retained_expert_ids (ORIGINAL ids in
    selection order), pruned_expert_ids (sorted original ids)}``. Extra
    top-level selection/provenance context is permitted by that contract.
    """
    n_experts = arch["original_experts_per_layer"]
    return {
        "schema_version": SCHEMA_VERSION,
        "layers": [
            {
                "layer": layer,
                "original_expert_count": n_experts,
                "retained_expert_ids": list(selection[layer]),
                "pruned_expert_ids": sorted(set(range(n_experts)) - set(selection[layer])),
            }
            for layer in sorted(selection)
        ],
        "selection": {"mode": mode, "retained_experts_per_layer": len(next(iter(selection.values())))},
        "provenance": provenance,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.pruning.maps",
        description=(
            "Dry-run prune-map planning for MiMo-V2.6-Flash-RL MoE experts. "
            "Planning only: no weights are read or written."
        ),
    )
    parser.add_argument("--inventory", required=True, help="inventory JSON path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--retained-count",
        type=int,
        metavar="N",
        help="generate a shape-only selection retaining N experts per MoE layer",
    )
    group.add_argument(
        "--selection",
        metavar="CANDIDATE_JSON",
        help="external candidate selection with original expert ids (order preserved)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for --retained-count generation (default 0)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the prune-map JSON here (default: stdout)",
    )
    parser.add_argument(
        "--candidate-out",
        metavar="PATH",
        help="also write the embedded candidate block alone to PATH as a "
        "baseline-comparison-compatible candidate.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        inv = load_inventory(args.inventory)
        arch = inv["architecture"]
        if args.selection is not None:
            selection, provenance = load_selection_file(args.selection, arch)
            mode = "external_selection"
        else:
            retained_count = args.retained_count
            _validate_retained_count(retained_count, arch["original_experts_per_layer"])
            selection = generate_selection(arch, retained_count, args.seed)
            mode = "shape_only"
            provenance = {
                "mode": "shape_only",
                "seed": args.seed,
                "generator": "mimo_halo.pruning.maps.generate_selection",
                "rule": (
                    "per layer L: random.Random(f'{seed}:{L}').sample("
                    "range(original_experts_per_layer), retained_count)"
                ),
                "quality_map": False,
                "disclaimer": (
                    "Shape-only deterministic selection. NOT a REAP/HOPE quality map; "
                    "no routing statistics, activations, or quality signals were used."
                ),
            }
        document = build_prune_map(inv, selection, mode, provenance, args.inventory)
    except PruneMapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(document, indent=2, sort_keys=False) + "\n"
    if args.output:
        out_path = Path(args.output)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"wrote {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(payload)
    if args.candidate_out:
        candidate_path = Path(args.candidate_out)
        if candidate_path.parent and not candidate_path.parent.exists():
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            json.dumps(document["candidate"], indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {candidate_path}", file=sys.stderr)
    totals = document["totals"]
    text_slot = totals["post_prune_by_component"]["text/main"]
    print(
        f"mode={document['mode']} layers={len(document['layers'])} "
        f"retained_count={document['retained_count']} "
        f"expert_tensors kept={totals['expert_tensors_kept']} "
        f"dropped={totals['expert_tensors_dropped']} "
        f"logical_params dropped={totals['logical_parameters_dropped']} "
        f"native_bytes kept={totals['stored_bytes_kept']} "
        f"freed={totals['stored_bytes_dropped']} "
        f"router_bytes={totals['router_stored_bytes']} "
        f"router_bytes_post_remap="
        f"{totals['router_stored_bytes_post_remap'] if totals['router_stored_bytes_post_remap_known'] else 'unknown'} "
        f"text_bytes_post="
        f"{text_slot['stored_bytes'] if text_slot['stored_bytes_known'] else 'unknown'} "
        f"text_logical_post={text_slot['logical_parameters']} "
        f"accounting={'closed' if totals['accounting_closure']['closes'] else 'open'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
