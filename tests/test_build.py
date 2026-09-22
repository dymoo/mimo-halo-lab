"""Tests for src/mimo_halo/build: candidate construction.

Synthetic tiny source tree (2 MoE layers x 8 experts, MXFP4-packed experts
with per-32 U8 scale siblings, BF16/F32/F8-E4M3 natives, mtp + vision aux,
one excluded dflash/ subdir shard), an external-selection prune map from
``mimo_halo.pruning.maps``, and a sweep-shaped recipe. All offline, stdlib
+ numpy only.

Covers the acceptance surface:

* bit-exact passthrough proof (input/output sha256 of a tensor pair),
* Q3 roundtrip error bound against the compute-only BF16 reference,
* exact size accounting: builder output bytes == recipe prediction,
* prune-map original expert-id preservation,
* every refusal path (the refusal table, one test per row),
* the candidate record against schemas/compression-candidate.schema.json,
* a tiny end-to-end build whose artifact dir verifies loadable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import numpy as np  # noqa: E402

from mimo_halo.build import (  # noqa: E402
    AccountingError,
    BuildError,
    PackingEvidenceError,
    PassthroughQuantError,
    RequantError,
    SourcePathError,
    affine_expected_bytes,
    build_candidate,
    candidate_record,
    decode_mxfp4,
    dequant_affine,
    encode_affine,
    read_safetensors_header,
    unpack_codes,
    RECIPE_KEYS,
    SCHEMA_CLASSES,
)
from mimo_halo.models.inventory import DTYPE_ITEMSIZE, REPO_ID, REVISION  # noqa: E402
from mimo_halo.pruning.maps import (  # noqa: E402
    build_prune_map,
    load_selection_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_CONFIG = REPO_ROOT / "configs" / "experiments" / "compression-sweep.json"
CANDIDATE_SCHEMA = REPO_ROOT / "schemas" / "compression-candidate.schema.json"

HIDDEN = 128
EXPERTS = 8
MOE_LAYERS = (1, 2)
RETAINED = {1: [6, 4, 2, 0, 1, 3], 2: [7, 5, 1, 3, 0, 2]}
# Default dispatch order is descending original expert id (byte totals are
# order-independent; the basis is recorded in the quant assignment).
Q3_OLD_IDS = {1: [6, 4, 3], 2: [7, 5]}
INST_NATIVE = 26112   # 3 projections x (8192 B packed weight + 512 B scale)
INST_Q3 = 19200       # 3 x (128*128*3/8 + (16384/128)*2) = 3 x 6400
RESIDENT = 313664     # exact planned text payload (hand sum below)
FLOOR = 348224        # pure-native floor (all experts native, no second-gen)
AUX_BYTES = 640       # vision 128 + mtp 512: copied, never budgeted
DATASET_HASH = "fixture-corpus=" + "ab" * 32
DIGEST_BASIS = ["synthetic fixture: digests recomputed by tests/test_build.py"]


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj) -> Path:
    """Write JSON and read it back (the write-corruption guard of this repo)."""
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    back = json.loads(path.read_text(encoding="utf-8"))
    assert back == obj, f"write verification failed for {path}"
    return path


def prod(dims) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def make_mxfp4(rows: int, cols: int, seed: int):
    """Exactly-representable MXFP4 fixture: E2M1 codes x E8M0 exponents.

    Returns (packed weight bytes [rows, cols/2], scale bytes [rows, cols/32],
    the exact float32 values the pair represents).
    """
    rng = np.random.default_rng(seed)
    e2m1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
    codes = rng.integers(0, 8, size=(rows, cols)).astype(np.uint8)
    sign = rng.integers(0, 2, size=(rows, cols)).astype(np.uint8)
    codes = codes | (sign << np.uint8(3))
    exps = rng.integers(116, 136, size=(rows, cols // 32)).astype(np.uint8)
    magnitudes = e2m1[codes & np.uint8(7)]
    values = np.where(sign == 1, -magnitudes, magnitudes).astype(np.float32)
    values *= np.exp2(exps.astype(np.float32) - np.float32(127.0)).repeat(32, axis=1)
    packed = ((codes[:, 1::2] << np.uint8(4)) | codes[:, 0::2]).astype(np.uint8)
    return np.ascontiguousarray(packed).tobytes(), exps.tobytes(), values


def bf16_bytes(arr) -> bytes:
    f = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f.view(np.uint32) >> np.uint32(16)
    return u32.astype("<u2").tobytes()


def f32_bytes(arr) -> bytes:
    return np.ascontiguousarray(arr, dtype="<f4").tobytes()


def write_safetensors(path: Path, tensors: dict) -> int:
    """Write a safetensors file (header order == byte order); returns payload bytes."""
    doc = {}
    payload = b""
    offset = 0
    for name, t in tensors.items():
        data = t["data"]
        expected = prod(t["shape"]) * DTYPE_ITEMSIZE[t["dtype"]]
        assert len(data) == expected, (name, len(data), expected)
        doc[name] = {
            "dtype": t["dtype"],
            "shape": list(t["shape"]),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload += data
    raw = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return offset


def rec(
    name,
    shape,
    dtype,
    *,
    logical_shape,
    logical_parameters,
    role="parameter",
    layer=None,
    expert=None,
    component="text",
    shard="model.safetensors",
    quantization=None,
):
    return {
        "name": name,
        "shape": list(shape),
        "dtype": dtype,
        "stored_bytes": prod(shape) * DTYPE_ITEMSIZE[dtype],
        "logical_shape": list(logical_shape),
        "logical_parameters": logical_parameters,
        "role": role,
        "layer": layer,
        "expert": expert,
        "component": component,
        "shard": shard,
        "quantization": quantization,
    }


def allocation_rows() -> list[dict]:
    """Hand-computed fixture allocation table (exact integers only)."""
    rows = [
        ("experts_native", "mxfp4_native", 7, "expert_instance", INST_NATIVE, False, True),
        ("experts_second_gen", "q3_affine_g128", 5, "expert_instance", INST_Q3, True, False),
        ("router", "native_dense_row_sliced", 6, "retained_index", 520, False, True),
        ("attention_qkv", "native_fp8_e4m3", 1, "tensor", 4096, False, True),
        ("attention_qkv_scales", "native_f32", 1, "tensor", 4, False, True),
        ("attention_o_proj", "native_bf16", 1, "tensor", 8192, False, True),
        ("attention_sinks", "native_bf16", 1, "tensor", 128, False, True),
        ("dense_ffn", "native_fp8_e4m3", 3, "tensor", 2048, False, True),
        ("dense_ffn_scales", "native_f32", 3, "tensor", 4, False, True),
        ("embeddings_lm_head", "native_bf16", 2, "tensor", 6144, False, True),
        ("norms", "native_bf16", 7, "tensor", 128, False, True),
    ]
    return [
        {
            "class": cls,
            "format": fmt,
            "units": units,
            "unit_label": label,
            "bytes_per_unit": bpu,
            "class_bytes": units * bpu,
            "second_gen": second_gen,
            "bit_exact": bit_exact,
            "source_field": "fixture inventory tensors[]",
        }
        for cls, fmt, units, label, bpu, second_gen, bit_exact in rows
    ]


def make_recipe() -> dict:
    return {
        "candidate_id": "fixture-reap25-mixed-q3",
        "description": "synthetic fixture: 6-of-8 retention, mixed native/Q3 experts",
        "reap_percent_pruned": 25,
        "rounding_rule": (
            "retained = 8 - round_half_up(8 * 25 / 100) = 6 "
            "(fixture scale; the pinned 256-expert rule is schema-const)"
        ),
        "original_experts_per_layer": 8,
        "retained_experts_per_layer": 6,
        "retained_total_experts": 12,
        "pruned_total_experts": 4,
        "moe_layers": 2,
        "allocation_table": allocation_rows(),
        "expert_precision": {
            "native_mxfp4_instances": 7,
            "q3_instances": 5,
            "instance_bytes_native": INST_NATIVE,
            "instance_bytes_q3": INST_Q3,
            "q3_percent": 41.67,
            "per_layer_distribution": {
                "base_q3_per_layer": 2,
                "extra_q3_layers": [1],
                "extra_q3_per_layer": 3,
            },
            "assignment_rule": (
                "fixture: Q3 goes to the highest original expert ids first "
                "(deterministic placeholder; byte totals are order-independent "
                "and the exact assignment is recorded per layer)"
            ),
        },
        "second_gen_applied": [
            {
                "class": "experts_second_gen",
                "format": "q3_affine_g128",
                "scope": "5 of 12 retained expert instances (3 in layer 1, 2 in layer 2)",
                "bytes_before": 5 * INST_NATIVE,
                "bytes_after": 5 * INST_Q3,
                "reason": (
                    "matched-budget fixture: the native-only total "
                    f"{FLOOR} B is outside any realistic band; the class-by-class "
                    "reduction is recorded here"
                ),
            }
        ],
        "planned_resident_weight_bytes": RESIDENT,
        "pure_mxfp4_floor_bytes": FLOOR,
    }


def make_native_recipe() -> dict:
    """All-native variant: no second-gen anywhere, no ledger entries."""
    recipe = make_recipe()
    recipe["candidate_id"] = "fixture-reap25-all-native"
    rows = [row for row in recipe["allocation_table"] if row["class"] != "experts_second_gen"]
    for row in rows:
        if row["class"] == "experts_native":
            row["units"] = 12
            row["class_bytes"] = 12 * INST_NATIVE
    recipe["allocation_table"] = rows
    recipe["expert_precision"] = {
        "native_mxfp4_instances": 12,
        "q3_instances": 0,
        "instance_bytes_native": INST_NATIVE,
        "instance_bytes_q3": INST_Q3,
        "q3_percent": 0.0,
        "per_layer_distribution": {
            "base_q3_per_layer": 0,
            "extra_q3_layers": [],
            "extra_q3_per_layer": 0,
        },
        "assignment_rule": "no Q3 anywhere: every survivor keeps bit-exact native MXFP4",
    }
    recipe["second_gen_applied"] = []
    recipe["planned_resident_weight_bytes"] = FLOOR
    recipe["pure_mxfp4_floor_bytes"] = FLOOR
    return recipe


def make_fixture(base: Path, *, drop_scale: str | None = None) -> dict:
    """Build the synthetic verified source tree + inventory + map + refs."""
    root = base / REVISION
    (root / "dflash").mkdir(parents=True, exist_ok=True)
    (root / "audio_tokenizer").mkdir(parents=True, exist_ok=True)

    shard0: dict = {}
    mtp_shard: dict = {}
    dflash_shard: dict = {}
    records: list[dict] = []

    def add(bucket, name, dtype, shape, data):
        bucket[name] = {"dtype": dtype, "shape": list(shape), "data": data}

    # Experts: 2 MoE layers x 8 experts x 3 projections (mxfp4 + scale).
    for layer in MOE_LAYERS:
        for expert in range(EXPERTS):
            for pi, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
                wname = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                weight, scale, _ = make_mxfp4(
                    HIDDEN, HIDDEN, seed=layer * 10000 + expert * 100 + pi
                )
                add(shard0, wname, "U8", [HIDDEN, HIDDEN // 2], weight)
                records.append(
                    rec(
                        wname,
                        [HIDDEN, HIDDEN // 2],
                        "U8",
                        logical_shape=[HIDDEN, HIDDEN],
                        logical_parameters=HIDDEN * HIDDEN,
                        layer=layer,
                        expert=expert,
                        quantization={
                            "kind": "mxfp4_packed",
                            "stored_dtype": "U8",
                            "block_size": 32,
                            "scale_tensor": wname + "_scale",
                        },
                    )
                )
                sname = wname + "_scale"
                add(shard0, sname, "U8", [HIDDEN, HIDDEN // 32], scale)
                records.append(
                    rec(
                        sname,
                        [HIDDEN, HIDDEN // 32],
                        "U8",
                        logical_shape=[HIDDEN, HIDDEN // 32],
                        logical_parameters=0,
                        role="quantization_auxiliary",
                        layer=layer,
                        expert=expert,
                        quantization={"kind": "mxfp4_scale", "block_size": 32},
                    )
                )

    # Routers: dense BF16 gate + F32 correction bias per MoE layer.
    rng = np.random.default_rng(99)
    for layer in MOE_LAYERS:
        gname = f"model.layers.{layer}.mlp.gate.weight"
        gate = (rng.standard_normal((EXPERTS, HIDDEN)) * 0.02).astype(np.float32)
        add(shard0, gname, "BF16", [EXPERTS, HIDDEN], bf16_bytes(gate))
        records.append(
            rec(
                gname,
                [EXPERTS, HIDDEN],
                "BF16",
                logical_shape=[EXPERTS, HIDDEN],
                logical_parameters=EXPERTS * HIDDEN,
                layer=layer,
                quantization=None,
            )
        )
        bname = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
        bias = rng.standard_normal(EXPERTS).astype(np.float32)
        add(shard0, bname, "F32", [EXPERTS], f32_bytes(bias))
        records.append(
            rec(
                bname,
                [EXPERTS],
                "F32",
                logical_shape=[EXPERTS],
                logical_parameters=EXPERTS,
                layer=layer,
                quantization=None,
            )
        )

    # Dense layer-0 FFN (FP8 + F32 scales).
    for proj in ("gate_proj", "up_proj", "down_proj"):
        name = f"model.layers.0.mlp.{proj}.weight"
        raw = rng.integers(0, 255, size=(32, 64), dtype=np.uint8).tobytes()
        add(shard0, name, "F8_E4M3", [32, 64], raw)
        records.append(
            rec(
                name,
                [32, 64],
                "F8_E4M3",
                logical_shape=[32, 64],
                logical_parameters=32 * 64,
                layer=0,
                quantization={
                    "kind": "fp8_e4m3",
                    "block_size": [128, 128],
                    "scale_tensor": f"model.layers.0.mlp.{proj}.weight_scale_inv",
                },
            )
        )
        sname = f"model.layers.0.mlp.{proj}.weight_scale_inv"
        add(shard0, sname, "F32", [1, 1], f32_bytes(np.zeros((1, 1), np.float32)))
        records.append(
            rec(
                sname,
                [1, 1],
                "F32",
                logical_shape=[1, 1],
                logical_parameters=0,
                role="quantization_auxiliary",
                layer=0,
                quantization={"kind": "fp8_scale", "block_size": [128, 128]},
            )
        )

    # Attention classes: qkv FP8 + F32 scale, o_proj BF16, one sink BF16[64].
    qkv = "model.layers.0.self_attn.qkv_proj.weight"
    add(shard0, qkv, "F8_E4M3", [64, 64], rng.integers(0, 255, size=(64, 64), dtype=np.uint8).tobytes())
    records.append(
        rec(qkv, [64, 64], "F8_E4M3", logical_shape=[64, 64], logical_parameters=4096, layer=0,
            quantization={"kind": "fp8_e4m3", "block_size": [128, 128],
                          "scale_tensor": qkv + "_inv"})
    )
    qkvs = "model.layers.0.self_attn.qkv_proj.weight_scale_inv"
    add(shard0, qkvs, "F32", [1, 1], f32_bytes(np.zeros((1, 1), np.float32)))
    records.append(
        rec(qkvs, [1, 1], "F32", logical_shape=[1, 1], logical_parameters=0,
            role="quantization_auxiliary", layer=0,
            quantization={"kind": "fp8_scale", "block_size": [128, 128]})
    )
    oproj = "model.layers.0.self_attn.o_proj.weight"
    add(shard0, oproj, "BF16", [64, 64],
        bf16_bytes((rng.standard_normal((64, 64)) * 0.02).astype(np.float32)))
    records.append(
        rec(oproj, [64, 64], "BF16", logical_shape=[64, 64], logical_parameters=4096,
            layer=0, quantization=None)
    )
    sink = "model.layers.1.self_attn.attention_sink_bias"
    add(shard0, sink, "BF16", [64], bf16_bytes(rng.standard_normal(64).astype(np.float32)))
    records.append(
        rec(sink, [64], "BF16", logical_shape=[64], logical_parameters=64, layer=1,
            quantization=None)
    )

    # Embeddings + norms (BF16).
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        add(shard0, name, "BF16", [48, 64],
            bf16_bytes((rng.standard_normal((48, 64)) * 0.05).astype(np.float32)))
        records.append(
            rec(name, [48, 64], "BF16", logical_shape=[48, 64], logical_parameters=48 * 64,
                layer=None, quantization=None)
        )
    norm_names = []
    for layer in (0, 1, 2):
        for kind in ("input_layernorm", "post_attention_layernorm"):
            norm_names.append((f"model.layers.{layer}.{kind}.weight", layer))
    norm_names.append(("model.norm.weight", None))
    for name, layer in norm_names:
        add(shard0, name, "BF16", [64], bf16_bytes(rng.standard_normal(64).astype(np.float32)))
        records.append(
            rec(name, [64], "BF16", logical_shape=[64], logical_parameters=64, layer=layer,
                quantization=None)
        )

    # Aux components inside the root shard (copied, never budgeted).
    vision = "visual.patch_embed.proj.weight"
    add(shard0, vision, "BF16", [8, 8],
        bf16_bytes((rng.standard_normal((8, 8))).astype(np.float32)))
    records.append(
        rec(vision, [8, 8], "BF16", logical_shape=[8, 8], logical_parameters=64,
            layer=None, component="vision", quantization=None)
    )
    mtp = "model.mtp.layers.0.norm.weight"
    add(mtp_shard, mtp, "BF16", [16, 16],
        bf16_bytes((rng.standard_normal((16, 16))).astype(np.float32)))
    records.append(
        rec(mtp, [16, 16], "BF16", logical_shape=[16, 16], logical_parameters=256,
            layer=None, component="mtp", shard="model_mtp.safetensors", quantization=None)
    )
    # Excluded subdir shard (dflash lives under dflash/ in the real tree).
    draft = "dflash.draft_norm.weight"
    add(dflash_shard, draft, "BF16", [16, 16],
        bf16_bytes((rng.standard_normal((16, 16))).astype(np.float32)))
    records.append(
        rec(draft, [16, 16], "BF16", logical_shape=[16, 16], logical_parameters=256,
            layer=None, component="dflash",
            shard="dflash/dflash_draft_model.safetensors", quantization=None)
    )

    full_records = list(records)

    # Prune map from the FULL inventory (external selection, order preserved).
    inventory_full = {
        "schema_version": 1,
        "source": {"repo_id": REPO_ID, "revision": REVISION, "files": []},
        "architecture": {
            "original_experts_per_layer": EXPERTS,
            "top_k": 2,
            "moe_layers": list(MOE_LAYERS),
        },
        "tensors": full_records,
    }
    selection_doc = {
        "schema_version": 1,
        "top_k": 2,
        "layers": [
            {
                "layer": layer,
                "original_expert_count": EXPERTS,
                "retained_expert_ids": list(RETAINED[layer]),
                "pruned_expert_ids": sorted(set(range(EXPERTS)) - set(RETAINED[layer])),
            }
            for layer in MOE_LAYERS
        ],
    }
    selection_path = write_json(base / "selection.json", selection_doc)
    selection, provenance = load_selection_file(
        selection_path, inventory_full["architecture"]
    )
    map_doc = build_prune_map(
        inventory_full, selection, "external_selection", provenance, str(base / "inventory.json")
    )
    map_path = write_json(base / "prune_map.json", map_doc)

    # Drop-scale variants remove the sibling from the shard AND the written
    # inventory after the map exists (maps itself demands completeness).
    if drop_scale is not None:
        records = [t for t in records if t["name"] != drop_scale]
        shard0.pop(drop_scale, None)

    shard_payload = write_safetensors(root / "model.safetensors", shard0)
    mtp_payload = write_safetensors(root / "model_mtp.safetensors", mtp_shard)
    write_safetensors(root / "dflash" / "dflash_draft_model.safetensors", dflash_shard)
    write_json(root / "config.json", {"fixture": True, "hidden_size": HIDDEN})
    write_json(root / "audio_tokenizer" / "config.json", {"fixture": True})
    write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": shard_payload + mtp_payload, "save_format": "mxfp4"},
            "weight_map": {
                **{name: "model.safetensors" for name in shard0},
                **{name: "model_mtp.safetensors" for name in mtp_shard},
            },
        },
    )

    # source.files with pinned digests, recorded over the real bytes on disk.
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        data = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": len(data),
                "lfs_sha256": sha(data),
            }
        )
    inventory = {
        "schema_version": 1,
        "source": {"repo_id": REPO_ID, "revision": REVISION, "files": files},
        "architecture": inventory_full["architecture"],
        "tensors": records,
    }
    inventory_path = write_json(base / "inventory.json", inventory)
    environment_path = write_json(base / "environment.json",
                                  {"platform": "darwin-test", "gpu_count": 0})
    calibration_path = write_json(
        base / "calibration_config.json", {"kind": "fixture", "sequences": 2, "tokens": 256}
    )
    return {
        "root": root,
        "inventory": inventory_path,
        "prune_map": map_path,
        "environment": environment_path,
        "calibration": calibration_path,
        "base": base,
        "inventory_doc": inventory,
        "map_sha256_before": sha(map_path.read_bytes()),
    }


def build_kwargs(fx: dict, out: Path, **over) -> dict:
    kwargs = dict(
        inventory_path=fx["inventory"],
        prune_map_path=fx["prune_map"],
        recipe=make_recipe(),
        source_root=fx["root"],
        out_dir=out,
        seed=7,
        environment_path=fx["environment"],
        dataset_hashes=[DATASET_HASH],
        digest_basis=DIGEST_BASIS,
        calibration_config_path=fx["calibration"],
    )
    kwargs.update(over)
    return kwargs


# ---------------------------------------------------------------------------
# minimal JSON-Schema subset validator (jsonschema is not a project dep)
# ---------------------------------------------------------------------------

_SCHEMA_KEYS = {
    "$schema", "$id", "$ref", "title", "description", "$defs", "type", "required",
    "properties", "additionalProperties", "items", "enum", "const", "pattern",
    "minLength", "minItems", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "allOf",
}


def assert_schema(instance, schema, root=None, path="$"):
    """Assert ``instance`` against the schema subset this record exercises.

    Fails loudly when the schema grows a construct this validator does not
    implement, so conformance never silently degrades.
    """
    root = schema if root is None else root
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return assert_schema(instance, node, root, path)
    unknown = set(schema) - _SCHEMA_KEYS
    assert not unknown, (
        f"schema uses constructs the test validator does not implement: "
        f"{sorted(unknown)} at {path}"
    )
    for sub in schema.get("allOf", []):
        assert_schema(instance, sub, root, path)
    if "enum" in schema:
        assert any(
            type(instance) is type(opt) and instance == opt for opt in schema["enum"]
        ), f"{path}: {instance!r} not in enum {schema['enum']}"
    if "const" in schema:
        assert (
            type(instance) is type(schema["const"]) and instance == schema["const"]
        ), f"{path}: {instance!r} != const {schema['const']!r}"
    if "pattern" in schema and isinstance(instance, str):
        assert re.search(schema["pattern"], instance), (path, instance)
    types = schema.get("type")
    if types is not None:
        wanted = [types] if isinstance(types, str) else types
        ok = False
        for t in wanted:
            if t == "object" and isinstance(instance, dict):
                ok = True
            elif t == "array" and isinstance(instance, list):
                ok = True
            elif t == "string" and isinstance(instance, str):
                ok = True
            elif t == "integer" and isinstance(instance, int) and not isinstance(instance, bool):
                ok = True
            elif t == "number" and isinstance(instance, (int, float)) and not isinstance(
                instance, bool
            ):
                ok = True
            elif t == "boolean" and isinstance(instance, bool):
                ok = True
            elif t == "null" and instance is None:
                ok = True
        assert ok, f"{path}: {instance!r} does not match type {types}"
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            assert key in instance, f"{path}: missing required key {key!r}"
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(props)
            assert not extra, f"{path}: additionalProperties violation {sorted(extra)}"
        for key, sub in props.items():
            if key in instance:
                assert_schema(instance[key], sub, root, f"{path}.{key}")
    if isinstance(instance, list):
        assert len(instance) >= schema.get("minItems", 0), path
        if "items" in schema:
            for i, item in enumerate(instance):
                assert_schema(item, schema["items"], root, f"{path}[{i}]")
    if isinstance(instance, str) and "minLength" in schema:
        assert len(instance) >= schema["minLength"], (path, instance)
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema:
            assert instance >= schema["minimum"], (path, instance)
        if "exclusiveMinimum" in schema:
            assert instance > schema["exclusiveMinimum"], (path, instance)
        if "maximum" in schema:
            assert instance <= schema["maximum"], (path, instance)
        if "exclusiveMaximum" in schema:
            assert instance < schema["exclusiveMaximum"], (path, instance)
    return True


# ---------------------------------------------------------------------------
# kernel-level tests
# ---------------------------------------------------------------------------


class QuantKernelTest(unittest.TestCase):
    def test_byte_rule_matches_pinned_sweep_constants(self):
        # Exact config constants, reproduced from the pinned integer rule.
        self.assertEqual(affine_expected_bytes(8388608, 3, 128), 3276800)
        self.assertEqual(3 * affine_expected_bytes(8388608, 3, 128), 9830400)
        self.assertEqual(3 * (4194304 + 262144), 13369344)
        self.assertEqual(affine_expected_bytes(33554432, 8, 64), 34603008)
        self.assertEqual(affine_expected_bytes(152576 * 4096, 6, 128), 478478336)
        self.assertEqual(affine_expected_bytes(64, 8, 64), 66)
        self.assertEqual(47 * 192 * (4096 * 2 + 4), 73960704)
        self.assertEqual(47 * (4096 * 2 + 4), 385212)

        cfg = json.loads(SWEEP_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(len(cfg["candidates"]), 5)
        for cand in cfg["candidates"]:
            rows = cand["allocation_table"]
            floor = 0
            for row in rows:
                self.assertEqual(
                    row["class_bytes"], row["units"] * row["bytes_per_unit"],
                    f"{cand['candidate_id']}/{row['class']}",
                )
                if row["second_gen"]:
                    entry = next(
                        e for e in cand["second_gen_applied"] if e["class"] == row["class"]
                    )
                    self.assertEqual(entry["bytes_after"], row["class_bytes"])
                    floor += entry["bytes_before"]
                else:
                    floor += row["class_bytes"]
            self.assertEqual(
                sum(r["class_bytes"] for r in rows),
                cand["planned_resident_weight_bytes"],
                cand["candidate_id"],
            )
            self.assertEqual(floor, cand["pure_mxfp4_floor_bytes"], cand["candidate_id"])
            retained = 256 - int(256 * cand["reap_percent_pruned"] / 100 + 0.5)
            self.assertEqual(cand["retained_experts_per_layer"], retained)

    def test_q3_roundtrip_error_bound(self):
        rng = np.random.default_rng(7)
        rows, cols = 4, 128
        e2m1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
        idx = rng.integers(0, 8, size=(rows, cols))
        sign = rng.integers(0, 2, size=(rows, cols))
        values = e2m1[idx] * np.where(sign == 1, -1.0, 1.0)
        exps = rng.integers(120, 132, size=(rows, cols // 32))
        values = (values * np.exp2((exps - 127).astype(np.float32)).repeat(32, axis=1)).astype(
            np.float32
        )
        encoded = encode_affine(values, 3, 128)
        self.assertEqual(len(encoded.codes), rows * cols * 3 // 8)  # 192
        self.assertEqual(len(encoded.scale_zero), rows * (cols // 128) * 2)  # 8
        deq = dequant_affine(encoded.codes, encoded.scale_zero, rows, cols, 3, 128)
        err = np.abs(deq - values)
        step = 2.0 ** encoded.stats["max_group_scale_exp"]
        # Bound: max per-element error never exceeds the group step, and the
        # recorded statistic equals the recomputed roundtrip error.
        self.assertLessEqual(float(err.max()), step + 1e-9)
        self.assertAlmostEqual(encoded.stats["max_abs_err"], float(err.max()), places=12)
        self.assertLessEqual(encoded.stats["mean_abs_err"], float(err.max()))
        codes = unpack_codes(encoded.codes, rows, cols, 3)
        self.assertEqual(codes.shape, (rows, cols))
        self.assertLessEqual(int(codes.max()), 7)
        # Zero groups dequantize exactly; constant groups stay in bound.
        zero = encode_affine(np.zeros((1, 128), np.float32), 3, 128)
        dz = dequant_affine(zero.codes, zero.scale_zero, 1, 128, 3, 128)
        self.assertTrue(np.array_equal(dz, np.zeros((1, 128), np.float32)))
        const = encode_affine(np.full((2, 128), 0.35, np.float32), 3, 128)
        dc = dequant_affine(const.codes, const.scale_zero, 2, 128, 3, 128)
        self.assertLessEqual(
            float(np.abs(dc - 0.35).max()),
            2.0 ** const.stats["max_group_scale_exp"] + 1e-9,
        )
        # q6/q8 pack roundtrip (bits 6 and 8 exceed the tacodevs test vocab).
        wide = (rng.standard_normal((3, 512)) * 0.1).astype(np.float32)
        for bits, group in ((6, 128), (8, 64)):
            enc = encode_affine(wide, bits, group)
            self.assertEqual(len(enc.codes), 3 * 512 * bits // 8)
            dq = dequant_affine(enc.codes, enc.scale_zero, 3, 512, bits, group)
            bound = 2.0 ** enc.stats["max_group_scale_exp"] + 1e-6
            self.assertLessEqual(float(np.abs(dq - wide).max()), bound)

    def test_mxfp4_decode_is_bit_exact(self):
        weight, scale, values = make_mxfp4(4, 128, seed=1234)
        decoded = decode_mxfp4(weight, scale, 4, 128)
        self.assertTrue(np.array_equal(decoded, values))

    def test_affine_byte_rule_fails_closed_on_non_exact_inputs(self):
        with self.assertRaisesRegex(AccountingError, "not a whole number of bytes"):
            affine_expected_bytes(130, 3, 128)
        with self.assertRaisesRegex(AccountingError, "not divisible"):
            affine_expected_bytes(200, 3, 128)
        with self.assertRaisesRegex(AccountingError, "does not divide row width"):
            encode_affine(np.zeros((2, 33), np.float32), 3, 128)
        with self.assertRaisesRegex(BuildError, "supports bits"):
            encode_affine(np.zeros((2, 256), np.float32), 5, 128)


# ---------------------------------------------------------------------------
# record-schema conformance (against the real sweep config)
# ---------------------------------------------------------------------------


class CandidateRecordSchemaTest(unittest.TestCase):
    def test_planned_records_from_sweep_config_conform_to_schema(self):
        schema = json.loads(CANDIDATE_SCHEMA.read_text(encoding="utf-8"))
        cfg = json.loads(SWEEP_CONFIG.read_text(encoding="utf-8"))
        source_info = dict(cfg["source_weights"])
        for candidate in cfg["candidates"]:
            record = candidate_record(candidate, source_info, notes="planned conformance")
            self.assertTrue(
                assert_schema(record, schema),
                f"{candidate['candidate_id']} does not conform",
            )
            self.assertEqual(record["status"], "planned")
            self.assertEqual(record["metrics"]["calibration"]["perplexity"], None)
            self.assertEqual(record["performance_pre_optimisation"]["label"], "PRE-OPTIMISATION")
            self.assertEqual(record["size"]["resident_weight_bytes"],
                             candidate["planned_resident_weight_bytes"])
        # Schema drift guard: the builder's vocabulary mirrors the schema.
        classes = set(schema["$defs"]["classRow"]["properties"]["class"]["enum"])
        self.assertEqual(classes, set(SCHEMA_CLASSES))
        recipe_required = schema["$defs"]["recipe"]["required"]
        self.assertEqual(set(RECIPE_KEYS), set(recipe_required))
        row_required = schema["$defs"]["classRow"]["required"]
        for row in cfg["candidates"][0]["allocation_table"]:
            self.assertEqual(set(row), set(row_required))

    def test_built_record_shape(self):
        # The full schema pins the real 47x256 shape (moe_layers const 47,
        # instance-byte consts), so the fixture record is checked structurally
        # while still reusing the same emitter.
        record = BUILT["record"]
        self.assertEqual(record["status"], "built")
        self.assertEqual(set(record["recipe"]), set(RECIPE_KEYS))
        self.assertEqual(record["size"]["resident_weight_bytes"], RESIDENT)
        self.assertEqual(record["size"]["pure_mxfp4_floor_bytes"], FLOOR)
        self.assertEqual(record["size"]["target_bytes"], 90 * (1 << 30))
        self.assertEqual(record["size"]["within_tolerance"], False)
        self.assertEqual(record["source_weights"]["revision"], REVISION)
        self.assertIsNone(record["metrics"]["on_policy_nll_nats"])
        self.assertEqual(record["metrics"]["pathological_generations"], [])
        self.assertEqual(record["metrics"]["benchmarks"], [])
        self.assertEqual(record["metrics"]["long_context_degradation"]["128k_plus"],
                         {"delta_perplexity": None, "agreement_fraction": None, "notes": None})
        self.assertIsNone(record["performance_pre_optimisation"]["theoretical_bytes_per_token"])


# ---------------------------------------------------------------------------
# end-to-end build (one shared fixture; read-only assertions)
# ---------------------------------------------------------------------------

_BUILD_TMP = tempfile.TemporaryDirectory()
_BUILD_BASE = Path(_BUILD_TMP.name)
FIXTURE = make_fixture(_BUILD_BASE)
OUT_MAIN = _BUILD_BASE / "out-main"
BUILT = build_candidate(**build_kwargs(FIXTURE, OUT_MAIN))


class EndToEndBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(BUILT["out_dir"])
        cls.header = read_safetensors_header(cls.out / "model.safetensors")

    @classmethod
    def tearDownClass(cls):
        _BUILD_TMP.cleanup()

    # -- loadable artifact ------------------------------------------------
    def test_end_to_end_build_is_a_loadable_artifact(self):
        self.assertEqual(BUILT["verify"]["status"], "verified")
        manifest = BUILT["manifest"]
        self.assertEqual(manifest["kind"], "pruned")
        self.assertEqual(manifest["seed"], 7)
        inputs = set(manifest["inputs"])
        self.assertEqual(inputs, {"calibration_config", "expert_map", "quant_assignment"})
        listed = {entry["path"] for entry in manifest["files"]}
        for name in (
            "model.safetensors",
            "model_mtp.safetensors",
            "config.json",
            "model.safetensors.index.json",
            "candidate.json",
            "build_report.json",
            "quant_assignment.json",
            "expert_map.json",
            "calibration_config.json",
            "commands.txt",
            "source_commits.json",
            "dataset_manifest.json",
            "environment.json",
            "metrics.json",
        ):
            self.assertIn(name, listed)
        source_commits = json.loads((self.out / "source_commits.json").read_text())
        self.assertEqual(source_commits["sources"][0]["revision"], REVISION)
        self.assertEqual(
            source_commits["sources"][0]["url"],
            f"https://huggingface.co/{REPO_ID}",
        )
        # Honest conversion-commit field: null when the caller never supplied one.
        self.assertIsNone(source_commits["conversion_commit"])
        self.assertIsNone(source_commits["runtime_commit"])
        # Re-verify through the public tool (the acceptance "loadable" gate).
        report = artifacts_verify(self.out)
        self.assertEqual(report["status"], "verified")
        # Output structure: contiguous new expert ids, sliced routers.
        names = set(self.header.tensors)
        for layer in MOE_LAYERS:
            for new_id in range(6):
                self.assertIn(
                    f"model.layers.{layer}.mlp.experts.{new_id}.gate_proj.weight", names
                )
            for old_id in (6, 7):
                self.assertNotIn(
                    f"model.layers.{layer}.mlp.experts.{old_id}.gate_proj.weight", names
                )
            gate = self.header.tensors[f"model.layers.{layer}.mlp.gate.weight"]
            self.assertEqual(gate["shape"], [6, HIDDEN])
            bias = self.header.tensors[
                f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
            ]
            self.assertEqual(bias["shape"], [6])
        # Q3 expert tensors: packed codes + I8 [rows, groups, 2] scale/zero.
        codes = self.header.tensors["model.layers.1.mlp.experts.0.gate_proj.weight"]
        self.assertEqual(codes["dtype"], "U8")
        self.assertEqual(codes["shape"], [HIDDEN, HIDDEN * 3 // 8])
        scale_zero = self.header.tensors[
            "model.layers.1.mlp.experts.0.gate_proj.weight_scale"
        ]
        self.assertEqual(scale_zero["dtype"], "I8")
        self.assertEqual(scale_zero["shape"], [HIDDEN, 1, 2])
        # Regenerated index: exact weight_map over output names, honest totals.
        index = json.loads((self.out / "model.safetensors.index.json").read_text())
        mtp_header = read_safetensors_header(self.out / "model_mtp.safetensors")
        self.assertEqual(
            set(index["weight_map"]), set(self.header.tensors) | set(mtp_header.tensors)
        )
        self.assertNotIn("save_format", index["metadata"])
        payload = sum(
            e["data_offsets"][1] - e["data_offsets"][0] for e in self.header.tensors.values()
        ) + sum(
            e["data_offsets"][1] - e["data_offsets"][0]
            for e in mtp_header.tensors.values()
        )
        self.assertEqual(index["metadata"]["total_size"], payload)
        # Quant assignment sidecar: closed accounting + dispatch record.
        qa = json.loads((self.out / "quant_assignment.json").read_text())
        self.assertTrue(qa["accounting"]["closed"])
        self.assertEqual(qa["accounting"]["allocation_table_sum"], RESIDENT)
        self.assertEqual(qa["accounting"]["predicted_text_bytes"], RESIDENT)
        classes = {c["class"]: c for c in qa["classes"]}
        self.assertEqual(classes["experts_native"]["planned_units"], 7)
        self.assertEqual(classes["experts_second_gen"]["planned_units"], 5)
        self.assertEqual(classes["router"]["planned_bytes"], 3120)
        self.assertEqual(qa["expert_dispatch"]["q3_old_ids_by_layer"],
                         {"1": Q3_OLD_IDS[1], "2": Q3_OLD_IDS[2]})
        self.assertEqual(qa["expert_dispatch"]["retained_by_layer"],
                         {"1": RETAINED[1], "2": RETAINED[2]})
        self.assertEqual(qa["second_gen_applied"][0]["bytes_after"], 5 * INST_Q3)
        # Report: exclusions recorded byte-exact, aux separated.
        report_doc = json.loads((self.out / "build_report.json").read_text())
        self.assertEqual(report_doc["resident_weight_bytes"], RESIDENT)
        self.assertEqual(report_doc["accounting"]["written_text_bytes"], RESIDENT)
        self.assertEqual(report_doc["accounting"]["aux_written_bytes"], AUX_BYTES)
        excluded_paths = {e["path"] for e in report_doc["excluded_files"]}
        self.assertEqual(
            excluded_paths,
            {"dflash/dflash_draft_model.safetensors", "audio_tokenizer/config.json"},
        )
        dflash = report_doc["excluded_components"]["dflash/dflash_draft_model.safetensors"]
        self.assertEqual(dflash["stored_bytes"], 512)
        self.assertEqual(dflash["components"], ["dflash"])

    def test_bit_exact_passthrough_digest_evidence(self):
        """Input/output sha256 of a passthrough tensor pair (the proof)."""
        qa = json.loads((self.out / "quant_assignment.json").read_text())
        # Native survivor: old expert 0 -> new id 3 in layer 1.
        out_name = "model.layers.1.mlp.experts.3.gate_proj.weight"
        src_name = "model.layers.1.mlp.experts.0.gate_proj.weight"
        entry = qa["tensors"][out_name]
        self.assertEqual(entry["action"], "copy")
        self.assertEqual(entry["input_sha256"], entry["output_sha256"])

        source_header = read_safetensors_header(FIXTURE["root"] / "model.safetensors")
        src = source_header.tensors[src_name]
        with open(FIXTURE["root"] / "model.safetensors", "rb") as handle:
            handle.seek(8 + source_header.header_length + src["data_offsets"][0])
            src_bytes = handle.read(src["data_offsets"][1] - src["data_offsets"][0])
        out = self.header.tensors[out_name]
        with open(self.out / "model.safetensors", "rb") as handle:
            handle.seek(8 + self.header.header_length + out["data_offsets"][0])
            out_bytes = handle.read(out["data_offsets"][1] - out["data_offsets"][0])
        self.assertEqual(sha(src_bytes), entry["input_sha256"])
        self.assertEqual(sha(out_bytes), entry["output_sha256"])
        self.assertEqual(sha(src_bytes), sha(out_bytes))

        # Scale sibling of the same native instance.
        scale_out = qa["tensors"][out_name + "_scale"]
        self.assertEqual(scale_out["action"], "copy")
        self.assertEqual(scale_out["input_sha256"], scale_out["output_sha256"])

        # Router row-slice: exact dense slice, digest of the concatenated
        # source rows equals the written tensor.
        router_out = "model.layers.1.mlp.gate.weight"
        router_entry = qa["tensors"][router_out]
        self.assertEqual(router_entry["action"], "row_slice")
        self.assertEqual(router_entry["input_sha256"], router_entry["output_sha256"])
        row_bytes = HIDDEN * 2
        hasher = hashlib.sha256()
        src_router = source_header.tensors[router_out]
        with open(FIXTURE["root"] / "model.safetensors", "rb") as handle:
            for row in RETAINED[1]:
                handle.seek(
                    8
                    + source_header.header_length
                    + src_router["data_offsets"][0]
                    + row * row_bytes
                )
                hasher.update(handle.read(row_bytes))
        self.assertEqual(hasher.hexdigest(), router_entry["output_sha256"])

        print(
            "BIT-EXACT EVIDENCE (passthrough tensor pair)\n"
            f"  tensor     : {out_name}\n"
            f"  input sha  : {entry['input_sha256']}\n"
            f"  output sha : {entry['output_sha256']}\n"
            f"  scale in   : {scale_out['input_sha256']}\n"
            f"  scale out  : {scale_out['output_sha256']}\n"
            f"  router row-slice sha (in=out): {router_entry['output_sha256']}",
            file=sys.stderr,
        )

    def test_size_accounting_closes_exactly_on_fixture(self):
        rows = allocation_rows()
        hand_total = sum(row["units"] * row["bytes_per_unit"] for row in rows)
        # Hand sum, spelled out: 182784 + 96000 + 3120 + 4096 + 4 + 8192 +
        # 128 + 6144 + 12 + 12288 + 896
        self.assertEqual(hand_total, 313664)
        self.assertEqual(hand_total, RESIDENT)

        record = BUILT["record"]
        self.assertEqual(record["size"]["resident_weight_bytes"], RESIDENT)
        report = json.loads((self.out / "build_report.json").read_text())
        self.assertEqual(report["resident_weight_bytes"], RESIDENT)
        self.assertEqual(report["accounting"]["planned_text_bytes"], RESIDENT)
        self.assertEqual(report["accounting"]["written_text_bytes"], RESIDENT)
        self.assertTrue(report["accounting"]["closed"])
        qa = json.loads((self.out / "quant_assignment.json").read_text())
        self.assertEqual(qa["accounting"]["allocation_table_sum"], RESIDENT)

        # Independent recount from the written safetensors headers, split by
        # the class each tensor was dispatched under.
        mtp_header = read_safetensors_header(self.out / "model_mtp.safetensors")
        text_sum = 0
        aux_sum = 0
        for header in (self.header, mtp_header):
            for name, entry in header.tensors.items():
                span = entry["data_offsets"][1] - entry["data_offsets"][0]
                if qa["tensors"][name]["class"]:
                    text_sum += span
                else:
                    aux_sum += span
        self.assertEqual(text_sum, RESIDENT)
        self.assertEqual(aux_sum, AUX_BYTES)
        # Per-class recount against the recipe prediction, exactly.
        per_class: dict[str, int] = {}
        for name, entry in self.header.tensors.items():
            cls = qa["tensors"][name]["class"]
            if cls:
                per_class[cls] = per_class.get(cls, 0) + (
                    entry["data_offsets"][1] - entry["data_offsets"][0]
                )
        for row in rows:
            if row["class"] in per_class:
                self.assertEqual(per_class[row["class"]], row["class_bytes"], row["class"])
        # Shard container bytes close exactly: 8-byte prefix + header + payload.
        shard_facts = {f["file"]: f for f in report["output_shards"]}
        for shard in ("model.safetensors", "model_mtp.safetensors"):
            parsed = read_safetensors_header(self.out / shard)
            payload = sum(
                e["data_offsets"][1] - e["data_offsets"][0]
                for e in parsed.tensors.values()
            )
            expected_size = 8 + parsed.header_length + payload
            self.assertEqual(os.path.getsize(self.out / shard), expected_size)
            self.assertEqual(shard_facts[shard]["payload_bytes"], payload)

    def test_prune_map_original_ids_preserved(self):
        map_bytes_sha = sha(Path(FIXTURE["prune_map"]).read_bytes())
        self.assertEqual(map_bytes_sha, FIXTURE["map_sha256_before"])
        map_doc = json.loads(Path(FIXTURE["prune_map"]).read_text())
        by_layer = {entry["layer"]: entry for entry in map_doc["layers"]}
        self.assertEqual(by_layer[1]["retained"], RETAINED[1])
        self.assertEqual(by_layer[2]["retained"], RETAINED[2])
        self.assertEqual(
            by_layer[1]["old_to_new"],
            {str(old): new for new, old in enumerate(RETAINED[1])}
            | {str(old): None for old in (5, 7)},
        )
        # Output uses new ids; dispatch records the original ids it consumed.
        qa = json.loads((self.out / "quant_assignment.json").read_text())
        self.assertEqual(qa["expert_dispatch"]["q3_old_ids_by_layer"],
                         {"1": [6, 4, 3], "2": [7, 5]})
        expert_map = json.loads((self.out / "expert_map.json").read_text())
        self.assertEqual(
            {e["layer"]: e["retained"] for e in expert_map["layers"]},
            {1: RETAINED[1], 2: RETAINED[2]},
        )


def artifacts_verify(root):
    from mimo_halo import artifacts as _artifacts

    return _artifacts.verify_artifact(str(root))


# ---------------------------------------------------------------------------
# refusal table: one test per row
# ---------------------------------------------------------------------------


class RefusalTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.n = 0

    def tearDown(self):
        self._tmp.cleanup()

    def fresh(self, **kw):
        self.n += 1
        return make_fixture(self.base / f"fx{self.n}", **kw)

    def assert_refused(self, kwargs, exception, pattern, *, expect_no_outdir=True):
        out = kwargs["out_dir"]
        with self.assertRaisesRegex(exception, pattern):
            build_candidate(**kwargs)
        if expect_no_outdir:
            self.assertFalse(
                Path(out).exists(),
                f"refused build left output behind at {out}",
            )

    # -- refusal: input paths outside the verified source root ------------
    def test_refusal_input_path_outside_verified_source_root(self):
        fx = self.fresh()
        inv = json.loads(Path(fx["inventory"]).read_text())
        # A source file whose relative path escapes the verified root.
        inv["source"]["files"].append(
            {"path": "../evil.safetensors", "size": 4, "lfs_sha256": None}
        )
        write_json(Path(fx["inventory"]), inv)
        self.assert_refused(
            build_kwargs(fx, self.base / "out-escape"),
            SourcePathError,
            "escapes the verified source root",
        )

    def test_refusal_source_root_not_the_pinned_revision(self):
        fx = self.fresh()
        # Parent directory instead of the revision directory itself.
        self.assert_refused(
            build_kwargs(fx, self.base / "out-root", source_root=fx["base"]),
            SourcePathError,
            "not the verified pinned revision directory",
        )
        # A directory that merely contains the revision name suffix.
        wrong = self.base / "elsewhere"
        wrong.mkdir()
        self.assert_refused(
            build_kwargs(fx, self.base / "out-root2", source_root=wrong),
            SourcePathError,
            "not the verified pinned revision directory",
        )

    def test_refusal_output_inside_source_root(self):
        fx = self.fresh()
        self.assert_refused(
            build_kwargs(fx, fx["root"] / "out"),
            SourcePathError,
            "outside the verified source root",
        )

    # -- refusal: unrecorded mxfp4 requantization -------------------------
    def test_refusal_unrecorded_mxfp4_requantization(self):
        fx = self.fresh()
        recipe = make_recipe()
        recipe["second_gen_applied"] = []  # the ledger forgets the Q3 class
        self.assert_refused(
            build_kwargs(fx, self.base / "out-requant", recipe=recipe),
            RequantError,
            "requantization attempt on an mxfp4 tensor without an explicit "
            "recorded override",
        )

    # -- refusal: second-gen quant on a class marked passthrough ----------
    def test_refusal_second_gen_on_passthrough_class(self):
        fx = self.fresh()
        recipe = make_recipe()
        for row in recipe["allocation_table"]:
            if row["class"] == "experts_second_gen":
                # Format says affine, row marks say passthrough: dispatch
                # would silently re-encode a passthrough class.
                row["second_gen"] = False
                row["bit_exact"] = True
        self.assert_refused(
            build_kwargs(fx, self.base / "out-pass1", recipe=recipe),
            PassthroughQuantError,
            "marked passthrough",
        )

        fx2 = self.fresh()
        recipe2 = make_recipe()
        recipe2["second_gen_applied"].append(
            {
                "class": "norms",
                "format": "native_bf16",
                "scope": "all 7 norm tensors",
                "bytes_before": 896,
                "bytes_after": 896,
                "reason": "ledger names a passthrough class (refused)",
            }
        )
        self.assert_refused(
            build_kwargs(fx2, self.base / "out-pass2", recipe=recipe2),
            PassthroughQuantError,
            "ledger names recipe class 'norms' marked passthrough",
        )

    # -- refusal: byte accounting that does not close exactly -------------
    def test_refusal_accounting_does_not_close(self):
        # (a) bytes_per_unit/class_bytes moved together but away from the
        # predicted tensor bytes: static per-class closure must fail.
        fx = self.fresh()
        recipe = make_recipe()
        for row in recipe["allocation_table"]:
            if row["class"] == "norms":
                row["bytes_per_unit"] = 129
                row["class_bytes"] = 7 * 129
        self.assert_refused(
            build_kwargs(fx, self.base / "out-acct1", recipe=recipe),
            AccountingError,
            "class_bytes 903",
        )
        # (b) recipe's declared planned total disagrees with the prediction.
        fx2 = self.fresh()
        recipe2 = make_recipe()
        recipe2["planned_resident_weight_bytes"] = RESIDENT + 1
        self.assert_refused(
            build_kwargs(fx2, self.base / "out-acct2", recipe=recipe2),
            AccountingError,
            f"planned_resident_weight_bytes {RESIDENT + 1}",
        )
        # (c) class_bytes inconsistent with units x bytes_per_unit at validation.
        fx3 = self.fresh()
        recipe3 = make_recipe()
        for row in recipe3["allocation_table"]:
            if row["class"] == "norms":
                row["class_bytes"] = 904
        self.assert_refused(
            build_kwargs(fx3, self.base / "out-acct3", recipe=recipe3),
            AccountingError,
            "class_bytes 904 != units 7 x bytes_per_unit 128",
        )

    # -- refusal: missing packing-evidence sibling ------------------------
    def test_refusal_missing_packing_evidence_sibling(self):
        # Native MXFP4 survivor (layer 1, expert 0) loses its weight_scale.
        fx = self.fresh(
            drop_scale="model.layers.1.mlp.experts.0.gate_proj.weight_scale"
        )
        self.assert_refused(
            build_kwargs(fx, self.base / "out-pack"),
            PackingEvidenceError,
            "missing packing-evidence sibling",
        )

    # -- additional fail-closed guards ------------------------------------
    def test_refusal_shape_only_placeholder_selection(self):
        fx = self.fresh()
        map_doc = json.loads(Path(fx["prune_map"]).read_text())
        map_doc["mode"] = "shape_only"
        shape_map = write_json(self.base / "shape_map.json", map_doc)
        self.assert_refused(
            build_kwargs(fx, self.base / "out-shape", prune_map_path=shape_map),
            BuildError,
            "shape-only placeholder selection is not a quality map",
        )

    def test_refusal_quantized_stage_without_lineage(self):
        fx = self.fresh()
        self.assert_refused(
            build_kwargs(fx, self.base / "out-quantized", kind="quantized"),
            BuildError,
            "requires parent artifacts",
            expect_no_outdir=False,  # sidecar preflight runs after the write
        )


# ---------------------------------------------------------------------------
# all-native build: honest conversion commit + no numpy dependency
# ---------------------------------------------------------------------------


class AllNativeBuildTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.fx = make_fixture(self.base)

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_native_build_needs_no_numpy_and_records_conversion_commit(self):
        commit = "d" * 40
        with mock.patch.dict(sys.modules, {"numpy": None}):
            result = build_candidate(
                **build_kwargs(
                    self.fx,
                    self.base / "out-native",
                    recipe=make_native_recipe(),
                    conversion_commit=commit,
                )
            )
        self.assertEqual(result["verify"]["status"], "verified")
        source_commits = json.loads(
            (Path(result["out_dir"]) / "source_commits.json").read_text()
        )
        self.assertEqual(source_commits["conversion_commit"], commit)
        report = json.loads((Path(result["out_dir"]) / "build_report.json").read_text())
        self.assertEqual(report["resident_weight_bytes"], FLOOR)
        self.assertEqual(report["pure_mxfp4_floor_bytes"], FLOOR)
        qa = json.loads((Path(result["out_dir"]) / "quant_assignment.json").read_text())
        self.assertEqual(qa["second_gen_error_summary"]["quantized_tensors"], 0)
        self.assertEqual(qa["second_gen_applied"], [])
        # Every expert survivor tensor is a bit-exact native copy:
        # 12 instances x 3 projections x (weight + scale) = 72 tensors.
        expert_entries = [
            t for name, t in qa["tensors"].items() if ".mlp.experts." in name
        ]
        self.assertEqual(len(expert_entries), 12 * 3 * 2)
        for entry in expert_entries:
            self.assertEqual(entry["action"], "copy")
            self.assertEqual(entry["input_sha256"], entry["output_sha256"])

    def test_second_gen_build_fails_closed_without_numpy(self):
        with mock.patch.dict(sys.modules, {"numpy": None}):
            with self.assertRaisesRegex(BuildError, "numpy is required"):
                build_candidate(**build_kwargs(self.fx, self.base / "out-none"))
        self.assertFalse((self.base / "out-none").exists())


if __name__ == "__main__":
    unittest.main()
