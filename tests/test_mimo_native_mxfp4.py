"""Focused regression for patches/strix-mimo-native-mxfp4.patch (NativeGgufBridge).

The patch fixes conversion/mimo.py of the pinned strix llama.cpp converter:
routed-expert scale siblings (`...weight_scale`) used to count towards the
``n_experts * 3`` merge threshold (merge fired early -> KeyError), and packed
native MXFP4 experts were never repacked into ggml `block_mxfp4` tensors.

The tests copy the pinned converter tree to a throwaway directory (the
original upstreams/strix-llama.cpp checkout stays untouched), verify the patch
applies cleanly with `git apply`, and run `convert_hf_to_gguf.py` on tiny
synthetic MiMo-V2 checkpoints covering the three expert formats. The pristine
conversion/mimo.py under test is read from the pinned git revision (blob), so
the suite works whether or not the local checkout already carries the patch.

  a. original native - packed MXFP4 experts (U8 codes + U8 E8M0 scale
     siblings) convert; every output expert tensor decoded from ggml
     block_mxfp4 matches the source represented values exactly; file type is
     MOSTLY_MXFP4_MOE; the TP-aware FP8 qkv path output is preserved;
  b. uniformly pruned native - retained expert count and top-k land in the
     GGUF metadata and in the expert/router tensor shapes, values exact;
  c. mixed second-gen affine (the builder's q3_affine_g128 expert codes) -
     conversion refuses with an explicit unsupported-format diagnostic, both
     when config.json records store_dtype "second_gen_affine" and when the
     config lies "mxfp4" but the tensors are affine;
  (red) the pristine converter cannot convert (a): scale siblings trigger the
     n_experts * 3 merge early (KeyError near mimo.py line 211) - the exact
     bug class this patch fixes.

Model fixtures (config.json + model.safetensors) are generated here from
fixed-seed exactly-representable draws (E2M1 x E8M0 for MXFP4, an E4M3 grid
for FP8), torch-free, stdlib + numpy + mimo_halo only.

Tokenizer note: the converter's gpt2 vocab path hash-checks the pre-tokenizer
against a table of known reference tokenizers (conversion/base.py
get_vocab_base_pre), which synthetic tokenizers cannot pass - an unknown hash
is a hard failure. The success tests therefore copy the converter-input
tokenizer aux files from $MIMO_HALO_TOKENIZER_SRC (a checkpoint dir) or from
scratch/loadability/tiny_a and skip with a precise message when neither is
available; the mixed-affine refusal and the red test need no tokenizer (the
failure fires in MimoV2Model.__init__/prepare_tensors, before set_vocab).

Run:  python3 -m unittest discover -s tests -p 'test_mimo_native_mxfp4.py' -v
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # noqa: E402

import numpy as np  # noqa: E402

from mimo_halo.build import encode_affine  # noqa: E402
from mimo_halo.models.inventory import DTYPE_ITEMSIZE  # noqa: E402

STRIX = REPO_ROOT / "upstreams" / "strix-llama.cpp"
PATCH = REPO_ROOT / "patches" / "strix-mimo-native-mxfp4.patch"
TOKENIZER_SRC_ENV = "MIMO_HALO_TOKENIZER_SRC"
FALLBACK_TOKENIZER_DIR = REPO_ROOT / "scratch" / "loadability" / "tiny_a"
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "added_tokens.json",
)

# Tiny MiMo-V2 shape (probe-faithful, scaled down; vocab kept at the payload
# size so the real tokenizer's ids stay in range).
H = 128
L = 4
VOCAB = 152576
HEADS, KV, HD = 4, 4, 16
MOE_INT = 128
QKV_ROWS = HEADS * HD + KV * HD + KV * HD   # 192 rows = 4 TP ranks x 48

# ggml kvalues_mxfp4: e2m1 values doubled, sign in bit 3 of the code
# (ggml-common.h kvalues_fp4; ggml-cpu quants.c uses them at half scale).
KVALUES_MXFP4 = np.array(
    (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), np.float32)
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
# exactly representable float8_e4m3fn values (bias 7, no infinities)
E4M3_GRID = np.array(
    [-4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0,
     0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0], np.float32)

GGML_TYPE_MXFP4 = 39
GGUF_FILE_TYPE_MOSTLY_MXFP4_MOE = 38
FFN_EXP_SUFFIX = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


# ---------------------------------------------------------------------------
# fixture helpers (same layout rules as tests/test_build.py artifacts)
# ---------------------------------------------------------------------------

def prod(dims) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def make_mxfp4(rows: int, cols: int, seed: int):
    """Exactly-representable MXFP4: E2M1 codes x E8M0 exponents.

    Returns (packed bytes, scale bytes, represented float32 values) for the
    compressed-tensors "mxfp4-pack-quantized" layout: packed uint8
    [rows, cols/2] (element 2i low nibble, 2i+1 high) + scale uint8
    [rows, cols/32] (E8M0 biased exponent per 32-element group).
    """
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 8, size=(rows, cols)).astype(np.uint8)
    sign = rng.integers(0, 2, size=(rows, cols)).astype(np.uint8)
    codes = codes | (sign << np.uint8(3))
    exps = rng.integers(116, 136, size=(rows, cols // 32)).astype(np.uint8)
    magnitudes = E2M1[codes & np.uint8(7)]
    values = np.where(sign == 1, -magnitudes, magnitudes).astype(np.float32)
    values *= np.exp2(exps.astype(np.float32) - np.float32(127.0)).repeat(32, axis=1)
    packed = ((codes[:, 1::2] << np.uint8(4)) | codes[:, 0::2]).astype(np.uint8)
    return (np.ascontiguousarray(packed).tobytes(), exps.tobytes(),
            np.ascontiguousarray(values))


def e4m3_bytes(vals: np.ndarray) -> bytes:
    """Pack an exactly representable E4M3 grid array to float8_e4m3fn bytes."""
    f = np.ascontiguousarray(vals, np.float32)
    bits = f.view(np.uint32)
    sign = ((bits >> np.uint32(24)) & np.uint32(0x80)).astype(np.uint8)
    exp = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32) - 127
    man = ((bits >> np.uint32(20)) & np.uint32(0x7)).astype(np.uint8)
    out = np.where(f == 0, np.uint8(0),
                   (sign | ((exp + 7).astype(np.uint8) << np.uint8(3)) | man))
    out = np.ascontiguousarray(out, np.uint8)
    assert np.array_equal(e4m3_decode(out), f), "e4m3 pack is not exact"
    return out.tobytes()


def e4m3_decode(raw: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(raw, np.uint8).astype(np.uint32)
    sign = np.where(b & np.uint32(0x80), np.float32(-1.0), np.float32(1.0))
    exp = ((b >> np.uint32(3)) & np.uint32(0xF)).astype(np.int32)
    man = (b & np.uint32(0x7)).astype(np.float32)
    mag = np.where(exp == 0, man * np.float32(2.0 ** -9),
                   (np.float32(1.0) + man * np.float32(0.125))
                   * np.exp2((exp - 7).astype(np.float32)))
    return (sign * mag).astype(np.float32)


def bf16_bytes(arr) -> bytes:
    f = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f.view(np.uint32) >> np.uint32(16)
    return u32.astype("<u2").tobytes()


def bf16_decode(u16: np.ndarray) -> np.ndarray:
    return ((u16.astype(np.uint32) << np.uint32(16)).view(np.float32))


def f32_bytes(arr) -> bytes:
    return np.ascontiguousarray(arr, dtype="<f4").tobytes()


def write_safetensors(path: Path, tensors: dict) -> int:
    doc = {}
    payload = b""
    offset = 0
    for name, t in tensors.items():
        data = t["data"]
        expected = prod(t["shape"]) * DTYPE_ITEMSIZE[t["dtype"]]
        assert len(data) == expected, (name, len(data), expected)
        doc[name] = {"dtype": t["dtype"], "shape": list(t["shape"]),
                     "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        payload += data
    raw = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return offset


def decode_mxfp4_ggml(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Independent ggml block_mxfp4 decode, per pinned ggml ABI.

    ggml-common.h: block_mxfp4 = { uint8_t e; uint8_t qs[QK_MXFP4/2] } -
    17 bytes per 32 elements. ggml-cpu quants.c dot product: element j comes
    from `qs[j] & 0xf`, element j+16 from `qs[j] >> 4`, value =
    kvalues_mxfp4[nibble] * e8m0_to_fp32_half(e). kvalues_mxfp4 holds the
    e2m1 levels doubled and the scale is halved (2**(e-128)), so the value
    equals e2m1 * 2**(e-127) - the represented source value.
    """
    assert cols % 32 == 0
    n_blocks = cols // 32
    blocks = np.ascontiguousarray(raw, np.uint8).reshape(rows, n_blocks, 17)
    d = np.exp2(blocks[:, :, 0].astype(np.float32) - np.float32(128.0))
    qs = blocks[:, :, 1:]
    codes = np.empty((rows, n_blocks, 32), np.uint8)
    codes[:, :, :16] = qs & np.uint8(0x0F)          # elements 0..15
    codes[:, :, 16:] = (qs >> np.uint8(4)) & np.uint8(0x0F)  # elements 16..31
    return (KVALUES_MXFP4[codes] * d[:, :, None]).reshape(rows, cols)


# ---------------------------------------------------------------------------
# tiny synthetic MiMo-V2 checkpoint
# ---------------------------------------------------------------------------

NATIVE_QCONF = {
    "activation_scheme": "dynamic", "fmt": "e4m3", "ignored_layers": [],
    "mxfp4_block_size": 32, "quant_method": "fp8", "store_dtype": "mxfp4",
    "weight_block_size": [128, 128],
}
AFFINE_QCONF = dict(NATIVE_QCONF, store_dtype="second_gen_affine",
                    mxfp4_block_size=None)


def tiny_config(n_routed_experts: int, quant_config: dict) -> dict:
    return {
        "model_type": "mimo_v2",
        "architectures": ["MiMoV2ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_mimo_v2.MiMoV2Config",
            "AutoModel": "modeling_mimo_v2.MiMoV2Model",
            "AutoModelForCausalLM": "modeling_mimo_v2.MiMoV2ForCausalLM",
        },
        "vocab_size": VOCAB,
        "hidden_size": H,
        "intermediate_size": H,
        "num_hidden_layers": L,
        "hybrid_layer_pattern": [0, 1, 1, 1],
        "moe_layer_freq": [0, 1, 1, 1],
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "head_dim": HD,
        "v_head_dim": HD,
        "swa_num_attention_heads": HEADS,
        "swa_num_key_value_heads": KV,
        "swa_head_dim": HD,
        "swa_v_head_dim": HD,
        "swa_rope_theta": 10000.0,
        "rope_theta": 10000.0,
        "rope_parameters": {"rope_type": "default", "type": "default",
                            "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        "partial_rotary_factor": 0.5,
        "sliding_window": 8,
        "sliding_window_size": 8,
        "add_swa_attention_sink_bias": True,
        "add_full_attention_sink_bias": False,
        "attention_projection_layout": "fused_qkv",
        "attention_bias": False,
        "attention_value_scale": 0.707,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "layernorm_epsilon": 1e-06,
        "max_position_embeddings": 4096,
        "initializer_range": 0.02,
        "use_cache": True,
        "tie_word_embeddings": False,
        "n_routed_experts": n_routed_experts,
        "moe_intermediate_size": MOE_INT,
        "num_experts_per_tok": 2,
        "routed_scaling_factor": None,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "moe_router_dtype": "bfloat16",
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "num_nextn_predict_layers": 0,
        "eos_token_id": 2,
        "bos_token_id": 1,
        "pad_token_id": 0,
        "vision_config": None,
        "audio_config": None,
        "processor_config": None,
        "quantization_config": quant_config,
        "transformers_version": "4.57.6",
    }


def add(bucket, name, dtype, shape, data):
    bucket[name] = {"dtype": dtype, "shape": list(shape), "data": data}


def build_tiny(out_dir: Path, kind: str, tokenizer_src: Path | None) -> dict:
    """kind: 'a' native original | 'b' pruned native | 'c_mixed' mixed
    affine/native (config lies "mxfp4") | 'c_config' affine declared in
    config. Returns expected values for later comparison."""
    if kind == "a":
        experts, qconf = 6, NATIVE_QCONF
    elif kind == "b":
        experts, qconf = 5, NATIVE_QCONF
    elif kind == "b_dense":
        experts, qconf = 5, NATIVE_QCONF
    elif kind == "c_mixed":
        experts, qconf = 6, NATIVE_QCONF
    elif kind == "c_config":
        experts, qconf = 6, AFFINE_QCONF
    else:
        raise ValueError(kind)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(
        json.dumps(tiny_config(experts, qconf), indent=2, sort_keys=True) + "\n")
    if tokenizer_src is not None:
        for f in TOKENIZER_FILES:
            src = tokenizer_src / f
            if src.is_file():
                shutil.copyfile(src, out_dir / f)

    tensors: dict = {}
    expected = {"kind": kind, "experts": experts, "mxfp4": {}, "dense": {},
                "qkv": {}}

    # -- attention: fused qkv FP8 (TP-stacked rows, 4 block-row scales) ------
    for layer in range(L):
        base = f"model.layers.{layer}.self_attn"
        q = E4M3_GRID[np.random.default_rng(1000 + layer).integers(
            0, len(E4M3_GRID), size=(HEADS * HD, H))]
        k = E4M3_GRID[np.random.default_rng(2000 + layer).integers(
            0, len(E4M3_GRID), size=(KV * HD, H))]
        v = E4M3_GRID[np.random.default_rng(3000 + layer).integers(
            0, len(E4M3_GRID), size=(KV * HD, H))]
        # stack per TP rank [q_per | k_per | v_per]; all-ones block-row scales
        stacked = np.concatenate([
            np.concatenate([q[r * HD:(r + 1) * HD],
                            k[r * HD:(r + 1) * HD],
                            v[r * HD:(r + 1) * HD]]) for r in range(KV)
        ])
        assert stacked.shape == (QKV_ROWS, H)
        add(tensors, f"{base}.qkv_proj.weight", "F8_E4M3", [QKV_ROWS, H],
            e4m3_bytes(stacked))
        add(tensors, f"{base}.qkv_proj.weight_scale_inv", "F32", [KV, 1],
            f32_bytes(np.ones((KV, 1), np.float32)))
        # expected merged output of the TP-aware dequant path: [q | k | v]
        expected["qkv"][layer] = np.concatenate([q, k, v], axis=0)

        o = np.random.default_rng(4000 + layer).standard_normal((H, HEADS * HD)) * 0.05
        add(tensors, f"{base}.o_proj.weight", "BF16", [H, HEADS * HD], bf16_bytes(o))
        if layer > 0:  # hybrid_layer_pattern == 1 -> SWA sink bias
            sink = np.random.default_rng(5000 + layer).standard_normal(HEADS) * 0.01
            add(tensors, f"{base}.attention_sink_bias", "BF16", [HEADS], bf16_bytes(sink))
        for norm in ("input_layernorm", "post_attention_layernorm"):
            nv = np.random.default_rng(6000 + layer).standard_normal(H) * 0.02 + 1.0
            add(tensors, f"model.layers.{layer}.{norm}.weight", "BF16", [H], bf16_bytes(nv))

    # -- layer 0 dense mlp: FP8 + per-block weight_scale_inv ----------------
    for pi, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
        vals = E4M3_GRID[np.random.default_rng(7000 + pi).integers(
            0, len(E4M3_GRID), size=(H, H))]
        add(tensors, f"model.layers.0.mlp.{proj}.weight", "F8_E4M3", [H, H],
            e4m3_bytes(vals))
        sc = np.abs(np.random.default_rng(8000).standard_normal((1, 1))) + 0.5
        add(tensors, f"model.layers.0.mlp.{proj}.weight_scale_inv", "F32",
            [1, 1], f32_bytes(sc.astype(np.float32)))

    # -- MoE layers 1..L-1 --------------------------------------------------
    for layer in range(1, L):
        for e in range(experts):
            for pi, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
                wname = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
                seed = layer * 100000 + e * 1000 + pi
                if kind == "c_mixed" and (layer + e) % 2 == 1:
                    # builder's second-gen affine codes (q3_affine_g128):
                    # U8 LSB-first codes + I8 scale/zero pairs [rows, g, 2]
                    rng = np.random.default_rng(seed)
                    values = rng.standard_normal((MOE_INT, MOE_INT)).astype(np.float32)
                    enc = encode_affine(values, 3, 128)
                    add(tensors, wname, "U8", [MOE_INT, MOE_INT * 3 // 8],
                        np.frombuffer(enc.codes, dtype=np.uint8).copy().tobytes())
                    add(tensors, wname + "_scale", "I8", [MOE_INT, MOE_INT // 128, 2],
                        np.frombuffer(enc.scale_zero, dtype=np.int8).copy().tobytes())
                elif kind == "b_dense":
                    # REAP bf16 save: dense expert weights, no scale siblings
                    _, _, values = make_mxfp4(MOE_INT, MOE_INT, seed)
                    add(tensors, wname, "BF16", [MOE_INT, MOE_INT], bf16_bytes(values))
                    expected["dense"][(layer, e, proj)] = values
                else:
                    packed, scale, values = make_mxfp4(MOE_INT, MOE_INT, seed)
                    add(tensors, wname, "U8", [MOE_INT, MOE_INT // 2], packed)
                    add(tensors, wname + "_scale", "U8", [MOE_INT, MOE_INT // 32], scale)
                    expected["mxfp4"][(layer, e, proj)] = values
        g = np.random.default_rng(9000 + layer).standard_normal((experts, H)) * 0.02
        b = np.random.default_rng(9500 + layer).standard_normal(experts)
        add(tensors, f"model.layers.{layer}.mlp.gate.weight", "BF16",
            [experts, H], bf16_bytes(g))
        add(tensors, f"model.layers.{layer}.mlp.gate.e_score_correction_bias",
            "F32", [experts], f32_bytes(b))

    # -- embeddings / head / final norm -------------------------------------
    emb = np.random.default_rng(111).standard_normal((VOCAB, H)) * 0.05
    add(tensors, "model.embed_tokens.weight", "BF16", [VOCAB, H], bf16_bytes(emb))
    add(tensors, "lm_head.weight", "BF16", [VOCAB, H], bf16_bytes(emb * 0.5))
    add(tensors, "model.norm.weight", "BF16", [H],
        bf16_bytes(np.ones(H, np.float32)))

    write_safetensors(out_dir / "model.safetensors", tensors)
    expected["tensor_bytes"] = sum(len(t["data"]) for t in tensors.values())
    return expected


# ---------------------------------------------------------------------------
# converter driver
# ---------------------------------------------------------------------------

def _copy_converter_tree(dst: Path, pristine_mimo: bytes) -> None:
    """Throwaway copy of the pinned converter: entry point + package + gguf-py.

    conversion/mimo.py is replaced with the file recorded at the pinned
    revision (git blob), never the mutable working tree: the local checkout
    may or may not carry the patch and both states must work.
    """
    dst.mkdir(parents=True)
    shutil.copyfile(STRIX / "convert_hf_to_gguf.py", dst / "convert_hf_to_gguf.py")
    shutil.copytree(STRIX / "conversion", dst / "conversion")
    shutil.copytree(STRIX / "gguf-py", dst / "gguf-py")
    (dst / "conversion" / "mimo.py").write_bytes(pristine_mimo)


def _pinned_mimo_blob() -> bytes:
    p = subprocess.run(["git", "show", "HEAD:conversion/mimo.py"],
                       cwd=str(STRIX), capture_output=True)
    if p.returncode != 0:
        raise unittest.SkipTest(
            "cannot read pinned conversion/mimo.py from git: "
            f"{p.stderr.decode()!r}")
    return p.stdout


def _git_apply(tree: Path, check_only: bool = False) -> subprocess.CompletedProcess:
    cmd = ["git", "apply", "-p1"]
    if check_only:
        cmd.append("--check")
    return subprocess.run(cmd, input=PATCH.read_bytes(), capture_output=True,
                          cwd=str(tree))


def run_converter(tree: Path, model_dir: Path, outfile: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(tree / "convert_hf_to_gguf.py"), str(model_dir),
           "--outfile", str(outfile), "--outtype", "f16"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1200,
                          cwd=str(REPO_ROOT))


def read_gguf(path: Path):
    """GGUFReader over the throwaway gguf-py copy (independent of repo code)."""
    gguf_py = str(TMP_ROOT / "patched" / "gguf-py")
    if gguf_py not in sys.path:
        sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    return GGUFReader(str(path))


def tensor_by_name(reader, name: str):
    for t in reader.tensors:
        if t.name == name:
            return t
    raise AssertionError(f"tensor {name!r} missing; have "
                         f"{sorted(t.name for t in reader.tensors)}")


def kv_int(reader, key: str) -> int:
    for k in reader.fields:
        if k == key or k.endswith("." + key):
            return int(reader.fields[k].contents())
    raise AssertionError(f"KV {key!r} missing; have {sorted(reader.fields)}")


def raw_bytes(t) -> np.ndarray:
    return np.asarray(t.data).view(np.uint8).reshape(-1)


TMP_ROOT: Path


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None,
    "pinned converter needs torch + transformers in the running interpreter",
)
class MimoNativeMxfp4Test(unittest.TestCase):
    """Conversion outcomes per expert format, against patched + pristine trees."""

    @classmethod
    def setUpClass(cls):
        global TMP_ROOT
        if not PATCH.is_file():
            raise unittest.SkipTest(f"missing {PATCH.relative_to(REPO_ROOT)}")
        if not (STRIX / "conversion" / "mimo.py").is_file():
            raise unittest.SkipTest("pinned upstreams/strix-llama.cpp checkout absent")

        TMP_ROOT = Path(tempfile.mkdtemp(prefix="mimo-native-mxfp4-"))
        cls.tmp = TMP_ROOT
        cls.pristine = TMP_ROOT / "pristine"
        cls.patched = TMP_ROOT / "patched"
        pristine_mimo = _pinned_mimo_blob()
        _copy_converter_tree(cls.pristine, pristine_mimo)
        _copy_converter_tree(cls.patched, pristine_mimo)

        # the patch must apply reproducibly to the pristine pinned file
        chk = _git_apply(cls.pristine, check_only=True)
        if chk.returncode != 0:
            raise AssertionError(
                "patches/strix-mimo-native-mxfp4.patch does not apply to the "
                f"pinned conversion/mimo.py: {chk.stderr.decode()!r}")
        ap = _git_apply(cls.patched)
        if ap.returncode != 0:
            raise AssertionError(f"git apply failed: {ap.stderr.decode()!r}")

        tokenizer_src = cls._tokenizer_src()
        cls.fixtures = {}
        cls.expect = {}
        for kind in ("a", "b", "b_dense", "c_mixed", "c_config"):
            d = TMP_ROOT / ("tiny_" + kind)
            cls.expect[kind] = build_tiny(d, kind, tokenizer_src)
            cls.fixtures[kind] = d

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "tmp", None) is not None:
            shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _tokenizer_src(cls) -> Path | None:
        for cand in (os.environ.get(TOKENIZER_SRC_ENV), FALLBACK_TOKENIZER_DIR):
            if cand and (Path(cand) / "tokenizer.json").is_file():
                return Path(cand)
        return None

    def _skip_without_tokenizer(self):
        if self._tokenizer_src() is None:
            self.skipTest(
                "converter-input tokenizer aux not available: set "
                f"{TOKENIZER_SRC_ENV} to a checkpoint dir (or keep "
                "scratch/loadability/tiny_a); synthetic tokenizers cannot "
                "pass get_vocab_base_pre's pre-tokenizer hash gate")

    def _assert_mxfp4_exact(self, gguf_path: Path, kind: str):
        reader = read_gguf(gguf_path)
        self.assertEqual(kv_int(reader, "general.file_type"),
                         GGUF_FILE_TYPE_MOSTLY_MXFP4_MOE)
        checked = 0
        for (layer, e, proj), values in self.expect[kind]["mxfp4"].items():
            name = f"blk.{layer}.ffn_{FFN_EXP_SUFFIX[proj]}_exps.weight"
            t = tensor_by_name(reader, name)
            self.assertEqual(int(t.tensor_type), GGML_TYPE_MXFP4)
            rows, cols = values.shape
            n_experts = self.expect[kind]["experts"]
            decoded = decode_mxfp4_ggml(raw_bytes(t), n_experts * rows, cols)
            got = decoded.reshape(n_experts, rows, cols)[e]
            self.assertTrue(np.array_equal(got, values),
                            f"{name} expert {e}: decoded values differ from source")
            checked += 1
        self.assertGreater(checked, 0)

    def test_original_native_converts_with_exact_mxfp4_values(self):
        self._skip_without_tokenizer()
        out = self.tmp / "a.gguf"
        proc = run_converter(self.patched, self.fixtures["a"], out)
        self.assertEqual(proc.returncode, 0, proc.stdout[-4000:] + proc.stderr[-4000:])
        self.assertTrue(out.is_file())
        self._assert_mxfp4_exact(out, "a")

        # source TP-aware FP8 qkv path preserved: merged [q | k | v], F16-exact
        reader = read_gguf(out)
        for layer in range(L):
            t = tensor_by_name(reader, f"blk.{layer}.attn_qkv.weight")
            expected = self.expect["a"]["qkv"][layer].astype(np.float16)
            got = raw_bytes(t).view(np.float16).reshape(QKV_ROWS, H)
            self.assertTrue(np.array_equal(got, expected),
                            f"blk.{layer}.attn_qkv.weight: TP-aware qkv values changed")

    def test_pruned_native_converts_with_retained_expert_count_and_topk(self):
        self._skip_without_tokenizer()
        out = self.tmp / "b.gguf"
        proc = run_converter(self.patched, self.fixtures["b"], out)
        self.assertEqual(proc.returncode, 0, proc.stdout[-4000:] + proc.stderr[-4000:])
        self.assertTrue(out.is_file())

        reader = read_gguf(out)
        self.assertEqual(kv_int(reader, "expert_count"), 5)
        self.assertEqual(kv_int(reader, "expert_used_count"), 2)
        for proj in ("gate", "up", "down"):
            t = tensor_by_name(reader, f"blk.1.ffn_{proj}_exps.weight")
            self.assertEqual(int(t.tensor_type), GGML_TYPE_MXFP4)
            self.assertEqual(int(t.shape[2]), 5)  # gguf dims: [cols rows experts]
        router = tensor_by_name(reader, "blk.1.ffn_gate_inp.weight")
        self.assertEqual(int(router.shape[1]), 5)
        self._assert_mxfp4_exact(out, "b")

    def test_mixed_affine_refuses_with_unsupported_format_diagnostic(self):
        for kind in ("c_mixed", "c_config"):
            out = self.tmp / f"{kind}.gguf"
            proc = run_converter(self.patched, self.fixtures[kind], out)
            log = proc.stdout + proc.stderr
            self.assertNotEqual(proc.returncode, 0, f"{kind}: conversion must refuse")
            self.assertIn("unsupported expert format", log, kind)
            self.assertIn("second_gen_affine", log, kind)
            self.assertFalse(out.is_file(), f"{kind}: no GGUF may be written")

    def test_dense_bf16_experts_keep_the_stacked_f16_path(self):
        self._skip_without_tokenizer()
        out = self.tmp / "b_dense.gguf"
        proc = run_converter(self.patched, self.fixtures["b_dense"], out)
        self.assertEqual(proc.returncode, 0, proc.stdout[-4000:] + proc.stderr[-4000:])
        self.assertTrue(out.is_file())

        reader = read_gguf(out)
        self.assertEqual(kv_int(reader, "general.file_type"), 1)  # MOSTLY_F16
        n_experts = self.expect["b_dense"]["experts"]
        for proj in ("gate", "up", "down"):
            t = tensor_by_name(reader, f"blk.1.ffn_{proj}_exps.weight")
            self.assertEqual(int(t.tensor_type), 1)  # F16, not MXFP4
            self.assertEqual(tuple(int(x) for x in t.shape), (MOE_INT, MOE_INT, n_experts))
        # dense values survive the bf16 store -> f16 write exactly
        for (layer, e, proj), values in self.expect["b_dense"]["dense"].items():
            name = f"blk.{layer}.ffn_{FFN_EXP_SUFFIX[proj]}_exps.weight"
            t = tensor_by_name(reader, name)
            got = raw_bytes(t).view(np.float16).reshape(n_experts, MOE_INT, MOE_INT)[e]
            stored = bf16_decode(
                np.frombuffer(bf16_bytes(values), "<u2")).reshape(MOE_INT, MOE_INT)
            self.assertTrue(np.array_equal(got, stored.astype(np.float16)),
                            f"{name} expert {e}: dense values changed")

    def test_pristine_converter_rejects_native_expert_scale_siblings(self):
        out = self.tmp / "a-pristine.gguf"
        proc = run_converter(self.pristine, self.fixtures["a"], out)
        log = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0, "pristine converter unexpectedly succeeded")
        self.assertFalse(out.is_file())
        # documented bug signature: n_experts*3 merge fires before all payloads
        self.assertIn("KeyError", log)


if __name__ == "__main__":
    unittest.main()
