# MiMo-V2.6-Flash-RL — Exact Artifact Manifest

Source of record for the exact safetensors inventory of
`XiaomiMiMo/MiMo-V2.6-Flash-RL` at pinned revision
`5711b268169967567844e1e560e8a3966da959b1`.

Everything documented here is derived from primary artifacts (official
config/index JSON and real safetensors headers fetched over bounded
HTTP-Range reads). No weight payload bytes are ever downloaded; no numbers
on this page are estimates.

## Files this manifest owns

```
manifests/models/mimo-v2.6-flash-rl/
  source.json                 # immutable per-file URLs + sha256 (from HF API at pinned revision)
  source-metadata/            # verbatim copies of official text metadata
    config.json
    model.safetensors.index.json
    dflash_config.json
    dflash_model.safetensors.index.json
    README.md
    generation_config.json
  headers/                    # archived safetensors headers (8-byte prefix + JSON header only)
    <shard>.safetensors.header.json
  inventory.json              # generated: schema_version 1 exact tensor inventory
  memory-report.json          # generated: native counts + prune-candidate byte math
```

## CLI

Stdlib-only, Python >= 3.11:

```bash
PYTHONPATH=src python3 -m mimo_halo.models.inventory \
  --out manifests/models/mimo-v2.6-flash-rl fetch      # archive source.json, metadata, all 66 headers
PYTHONPATH=src python3 -m mimo_halo.models.inventory \
  --out manifests/models/mimo-v2.6-flash-rl inventory  # validate + emit inventory.json
PYTHONPATH=src python3 -m mimo_halo.models.inventory \
  --out manifests/models/mimo-v2.6-flash-rl memory     # emit memory-report.json
```

`fetch --only <file> [<file> ...]` restricts the header pass;
`--force` refetches. Fetch is idempotent: existing artifacts are skipped.

## Bounded reader contract

- Every safetensors access is a ranged HTTPS GET: first `bytes=0-7`
  (little-endian uint64 header length `N`), then `bytes=8-(8+N-1)` for the
  JSON header. Payload bytes are never requested.
- A response is accepted only with status `206` and a `Content-Range` that
  exactly matches the requested window. A `200` (server ignored Range)
  aborts immediately — there is no full-download fallback. Any response
  longer than requested aborts before consuming it.
- Redirects are followed manually, HTTPS-only, into an allowlist of HF
  hosts (`huggingface.co`, `hf.co` and subdomains — the real redirect target
  is `us.aws.cdn.hf.co`). Metadata reads are capped at 64 MiB.

## Storage layout facts (verified from real headers)

- All 64 MoE shards `model_pp0_ep{0..63}_shard0.safetensors` store expert
  tensors packed MXFP4: `dtype U8`, two 4-bit nibbles per stored byte.
  Example (every expert, every shard): `mlp.experts.N.gate_proj.weight`
  stored `[2048, 2048] U8` → logical `[2048, 4096]` (8,388,608 parameters),
  sibling `weight_scale` stored `[2048, 128]` (one scale byte per 32 logical
  values, `mxfp4_block_size: 32`). Exact cost including scales: 4.25 bits per
  logical parameter. This native format is distinct from GGUF
  `Q4_0_ROCMFP4`/`_FAST` despite coincidentally equal bpw; no lossless
  cross-format conversion is implied.
- Dense (non-expert) quantized weights store `F8_E4M3` with `F32`
  `weight_scale_inv` block scales (`weight_block_size: [128, 128]`).
- Router `mlp.gate.weight`, all norms, `qkv_proj`/attention weights on
  ignored layers, embeddings and the vision/audio towers are dense BF16;
  `mlp.gate.e_score_correction_bias` is F32. These carry
  `quantization: null` in the inventory and their byte cost is exactly
  `logical_parameters × itemsize`.
- Per expert instance (uniform across all 256 experts and 47 MoE layers —
  enforced at build time): `gate/up/down_proj.weight` each `[2048, 2048]`
  U8 plus `weight_scale` each `[2048, 128]` U8.
- Index `metadata.total_size` (172,923,364,096 B) is the exact sum of
  stored tensor bytes; the on-disk files are larger (headers + alignment
  padding). `inventory.json:artifact_layout` separates file size,
  header bytes, payload and padding per file.
- `model_mtp.safetensors` and `dflash/dflash_draft_model.safetensors`
  headers are archived alongside the 64 shards (66 files total).

## Architecture facts (from official config/index at the pinned revision)

- 48 layers; layer 0 dense FFN, layers 1–47 MoE (`moe_layer_freq`).
- 256 routed experts per MoE layer, top-8 (`n_routed_experts`,
  `num_experts_per_tok`), no shared experts.
- 39 SWA layers / 9 global attention layers (`hybrid_layer_pattern`),
  SWA window 128; SWA layers carry `attention_sink_bias`.
- MTP resolution: the main-model MTP head has **3** layers
  (`model.mtp.layers.{0,1,2}` inside `model_mtp.safetensors`, matching
  `num_nextn_predict_layers: 3`). The **5**-layer drafter in the model card
  is DFlash (`dflash/dflash_draft_model.safetensors`, qwen3-type, 5 SWA
  layers, window 1024). These are two different components; the apparent
  3-vs-5 contradiction is dissolved by the headers.
- Vision tower: 28 blocks (`visual.blocks.*`, window 128, full attention on
  blocks 0/9/18/27 → 24 `attn.sinks` tensors). Audio encoder:
  `audio_encoder.*` (6 local transformer layers) + `speech_embeddings.*`.
  The separate `audio_tokenizer/` codec artifact is listed in
  `source.json` with checksums but is not part of the text-model inventory.
- Index metadata says `tp_size: 4` while the weight map has 64 shard
  files — recorded as an observation; the header-level mapping is the
  authority for tensor placement.

## Inventory JSON (schema_version 1)

Top level: `schema_version`, `source {repo_id, revision, files}`,
`architecture {original_experts_per_layer, top_k, moe_layers}`,
`tensors`, `totals`, `artifact_layout`.

Each entry in `tensors`:

| field | meaning |
|---|---|
| `name` | full tensor name |
| `shape` / `dtype` / `stored_bytes` | as stored in the shard header |
| `logical_shape` / `logical_parameters` | unpacked view; scales count 0 |
| `role` | `parameter` or `quantization_auxiliary` |
| `layer` / `expert` | parsed indices, `null` when not applicable |
| `component` | `text` \| `mtp` \| `dflash` \| `vision` \| `audio` |
| `shard` | owning safetensors file |
| `quantization` | format record or `null` (dense/uncompressed) |

Validation performed before `inventory.json` is written: header `N`
bounds and truncation, dtype/span consistency, offset contiguity,
duplicate JSON keys, duplicate tensor names across the weight map,
exact index↔header bijection per shard, declared index `total_size`
equal to the sum of stored bytes, and expert-tensor uniformity.

Counting rules: unknown packing raises (`PackingError`) — exact counts
are never guessed. Scales and `_inv` tensors are overhead
(`quantization_auxiliary`, `logical_parameters = 0`); genuine biases and
norms are parameters.

## Memory report

`memory-report.json` contains exact native counts, prune candidates at
256/192/176/160/144 retained experts (native MXFP4 stored bytes and
logical parameters scale exactly because every expert instance owns an
identical tensor set; each candidate also subtracts the dense router rows
`pruning/maps` slices away — routers are row-uniform, `quantization: null`,
so the row arithmetic is exact and reported under `router_*_removed`), and
precision costs computed only from evidence-backed layouts (native mxfp4
U8-packed and BF16 dequant). Per-retained-expert-index figures live under
`native.per_retained_expert_across_moe_layers_*`; the single-layer
instance cost is `native.per_layer_expert_layout`.

## Tests

`tests/test_inventory.py` (offline, unittest): range refusal (200 /
non-206 / Content-Range mismatch / extra bytes), truncated header, packed
logical counts, scale roles, U8-without-evidence failures, duplicate
JSON keys and cross-shard duplicates, index↔header bijection, component
boundaries, and exact prune-candidate arithmetic.
