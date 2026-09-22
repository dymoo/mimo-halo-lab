# Prune-map integrity contract (src/mimo_halo/pruning)

Planning-only dry-run mapping for MiMo-V2.6-Flash-RL MoE expert pruning at
160/176 retained experts. Nothing here reads weights, slices tensors, or
claims checkpoint reload; every artifact records `"applied": false`.

## What it consumes

An inventory JSON (owner: ExactInventory) conforming to the shared reader
contract:

- `schema_version: 1`, `source {repo_id, revision, files}`, `architecture
  {original_experts_per_layer, top_k, moe_layers: [ints]}`.
- `tensors: [{name, shape, dtype, stored_bytes, logical_shape,
  logical_parameters, role: parameter|quantization_auxiliary|buffer, layer,
  expert, component, shard, quantization}]`.

The module reads that schema directly and imports **no**
`mimo_halo.inventory` internals. A tensor whose `role` is `parameter` must
carry an exact non-negative integer `logical_parameters`; unknown logical
packing fails the run — it is never guessed. `quantization_auxiliary` and
`buffer` tensors must carry `logical_parameters = 0` and are counted only as
native (stored) bytes.

Main-text tensors are matched by `component` (accepted spellings:
`text/main`, `text_main`, `text-main`, `text`, `main`, case-insensitive);
when `component` is absent the official name prefix `model.layers.` decides.
MTP/DFlash/vision/audio tensors are never remapped and appear only in the
`unchanged_*` totals.

## Official tensor names (pinned revision 5711b268169967567844e1e560e8a3966da959b1)

Confirmed against the official `model.safetensors.index.json` weight map,
not inferred:

- Router outer gating weight: `model.layers.{L}.mlp.gate.weight` (47, one per
  MoE layer 1–47; layer 0 is dense).
- Router correction bias: `model.layers.{L}.mlp.gate.e_score_correction_bias`
  (47). `noaux_tc` sigmoid routing relies on it, so it is remapped row-wise
  together with the gate weight, never dropped.
- Expert weights: `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight`
  with sibling per-projection `weight_scale` (MXFP4 packed, block 32;
  256 experts per MoE layer, top-8 routing).
- MTP: `model.mtp.layers.{0..2}` — dense FFN (`weight_scale_inv`, fp8), no
  experts, no router. The config's 3 nextn layers vs. model-card prose "5
  MTP SWA layers" stays unresolved here; the map reports it as unresolved
  unless the inventory's architecture carries `mtp_layer_count`.

Reference dims: hidden 4096, moe_intermediate 2048, `num_experts_per_tok: 8`.

## CLI

    PYTHONPATH=src python -m mimo_halo.pruning.maps \
        --inventory INVENTORY.json --retained-count 160 --seed 7 \
        --output map.json --candidate-out candidate.json

    PYTHONPATH=src python -m mimo_halo.pruning.maps \
        --inventory INVENTORY.json --selection candidate.json \
        --output map.json

`--retained-count` accepts any integer in `[1, original_experts_per_layer]`
(160 and 176 are the project targets; 144/192 are tested generically).
`--output` defaults to stdout; `--candidate-out` extracts the candidate block
alone as a `compare.py`-consumable file. Exit codes: 0 success, 1 validation
failure (message on stderr), 2 usage error. A one-line summary always goes to
stderr, including `router_bytes=` (original router stored bytes),
`router_bytes_post_remap=` (exact dense figure or `unknown`),
`text_bytes_post=`/`text_logical_post=` (text/main component post-prune
totals, or `unknown` for bytes while a router format is unknowable), and
`accounting=closed`.

## Selection modes

1. **Shape-only generated** (`--retained-count` + `--seed`): per layer
   `random.Random(f"{seed}:{L}").sample(range(E), retained_count)`. Fully
   deterministic given `(seed, layer, E, count)` — no wall clock, no hash
   randomization, no dict order. Provenance records the seed, the exact rule,
   and the disclaimer that this is **NOT a REAP/HOPE quality map**: no
   routing statistics, activations, or quality signals were used. It is a
   structural placeholder until a real selection lands.
2. **External actual selection** (`--selection`): candidate JSON in the
   authoritative baseline comparison layer schema — `schema_version: 1`,
   `layers: [{layer, original_expert_count, retained_expert_ids (ORIGINAL ids
   in selection order), pruned_expert_ids (sorted original ids)}]`, optional
   `top_k`. Original ids are used verbatim and the given order defines the
   new ids (`new_id = position in retained_expert_ids`); the order is never
   re-sorted. Provenance records the file path and its sha256.

## Output (prune map)

Top level: `schema_version 1, kind "prune_map", mode, applied: false,
quality_map: false, source (inventory path + repo_id/revision/files),
architecture, retained_count, provenance, warnings, candidate, layers,
totals`.

Per MoE layer:

- `retained` (new-id order), `pruned` (sorted), complete `old_to_new` map —
  every original id appears exactly once, retained ids map to `0..N-1`,
  pruned ids map to `null`.
- `router`: gate weight + e-score correction bias remap plan — logical shapes
  before/after (`[256, 4096] -> [160, 4096]`, `[256] -> [160]`), kept
  rows/entries, exact post-remap `new_logical_parameters` and
  `new_stored_bytes`, and the plan text (slice rows in new-id order; routing
  stays top-8 over the retained set). The `accounting` block reports the two
  router tensors explicitly: original inventory `stored_bytes`,
  `logical_parameters`, and `tensor_count`, plus post-remap logical
  parameters, `post_remap_stored_bytes`, and `post_remap_stored_bytes_known`.
  For dense routers (`quantization: null`) with dtype exactly BF16/F32 the
  byte figure is exact — native payload rows scale uniformly,
  `stored_bytes // original_rows * retained_rows` — validated by byte-count
  divisibility and the dense row layout (`prod(shape[1:]) * itemsize` bytes
  per row); a contradiction fails the run closed, no dtype is ever guessed
  from byte counts, and no format or metadata overhead is claimed. Packed,
  quantized, or any other router format stays explicitly `null` with
  `known: false` and a note: that packing cost is never guessed.
- `expert_tensors`: keep/drop lists covering each retained/pruned expert's 3
  weights **and** their 3 `weight_scale` siblings (scales travel with their
  weight; they are overhead bytes, never logical parameters).
- `tensor_counts` (keep/drop/total), `logical_parameters` and `stored_bytes`
  (keep/drop/total) — summed exactly from inventory fields, never recounted
  or estimated.

Totals roll per-layer sums up, add `unchanged_*` counts/bytes for non-expert,
non-router main-text tensors plus other components, and account for routers
explicitly: `router_tensors`, `router_stored_bytes`,
`router_logical_parameters` (all original inventory values),
`logical_parameters_router_remapped` (post-remap, preserved for downstream
consumers), and `router_stored_bytes_post_remap` with
`router_stored_bytes_post_remap_known`: the exact dense row-scaled payload
figure when every router is provably dense BF16/F32, otherwise `null` with
`known: false` — either way a known-vs-unknown note spells out which.

`post_prune_by_component` gives exact post-prune payload/logical totals per
component bucket: `tensor_count`, `stored_bytes` (native tensor payload
bytes — no dtype guess, no GGUF/container format or metadata overhead, which
are deliberately not claimed before export), `logical_parameters`, and
`stored_bytes_known`. Every accepted text spelling collapses to the
canonical `text/main` key; other components keep their raw names. The
`text/main` bucket is kept experts + sliced routers + untouched text
tensors, so it compares directly with a memory report's text candidate
totals without subtracting MTP/DFlash/vision/audio by hand — those
components appear as their own buckets, summed unchanged.

`accounting_closure` proves the exact identity: expert keep + expert drop +
router + unchanged equals the inventory's total tensor count, stored bytes,
and logical parameters — a mismatch fails the run closed. The closure stays
on original router values and never moves. Future memory budgets must not
use kept + unchanged while omitting the router: parameter budgets use kept +
unchanged + `logical_parameters_router_remapped`; byte budgets add exact
`router_stored_bytes_post_remap` when `known`, and fall back to original
router bytes (closure values) only while it is unknown — a packed router's
cost is never invented.

Warnings always state: (a) planning-only, no weights read/written, no
checkpoint reload claimed; (b) gate rows are sliced without retraining or
calibration, so routing renormalizes over the retained set; (c) for
shape-only mode, that it is not a quality map; (d) if MTP layer count is
unresolved. No quality claims of any kind are made.

`candidate` (and `--candidate-out`) emits the exact comparison-layer schema
consumed by `mimo_halo.baselines.compare`, plus permitted extra
`selection`/`provenance` context.

## Fail-closed validation

Usage and inventory errors exit 1 with a named reason (2 for CLI usage):

- inventory: schema_version, missing source/architecture/tensors, unknown
  tensor role, parameter tensor without exact `logical_parameters`, negative
  or malformed shapes.
- structure: expert tensor in a non-MoE layer, expert id out of range
  (`bad expert`), duplicate tensor, missing expert, missing projection or
  scale (`missing gate_proj.weight_scale`), missing router tensor, router
  leading dimension != `original_experts_per_layer`, dtype change across
  layers/experts (expert weights/scales and router tensors must be uniform),
  dense quantization-null BF16/F32 router whose `stored_bytes` violate
  row-uniformity (`row-uniform`; exact row-scaling refused rather than
  guessed).
- accounting: keep + drop + router + unchanged failing to equal the
  inventory's total tensor count, stored bytes, or logical parameters
  (`accounting closure failed`).
- selection: duplicates, out-of-range or non-int ids, missing layers,
  non-MoE layer, duplicate layer entries, unequal count across layers,
  `top_k` change, `original_expert_count`/`pruned_expert_ids` contradictions.

## Verification performed

`tests/test_prune_maps.py` (36 cases, stdlib `unittest`, no network):
end-to-end small-fixture dry-runs with exact byte/parameter arithmetic,
determinism (same seed identical, different seed different), external
selection order preservation + sha256 provenance, every malformed case
above failing closed, generic counts 144/160/176/192/256 on a 256-expert
architecture, and a full 47-layer x 256-expert synthetic dry-run at 160 and
176 (per-layer maps complete, totals equal 47 x per-layer sums, 256-entry
old-to-new per layer). Router accounting regressions: an 8-expert fixture
with known router bytes checks the full sum closure (keep + drop + router +
unchanged == inventory counts/bytes/parameters), the candidate parameter
count including the remapped router, the exact dense post-remap
bytes (`4 * HIDDEN * 2 + 4 * 4` per layer, `old_bytes // rows * new_rows`),
the by-component text/vision totals, and the inventory-text-minus-dropped-
minus-router-delta identity; a packed router weight and an unrecognized
dense router dtype both keep `post_remap_stored_bytes: null` (never
guessed), a non-row-uniform dense router fails closed (`row-uniform`), and
the 47-layer run asserts router totals, exact dense row-scaling,
by-component text identity, and closure at both 160 and 176. CLI smoke at
real dims (hidden 4096 / inter 2048, 72,293 tensors): 160/176 runs complete
in ~0.5 s with exact totals (e.g. 176: kept native bytes 130,107,310,080 =
47 x 176 x 15,728,640 on the synthetic layout). Offline smoke on the
archived inventory (73,144 tensors, all 94 routers dense BF16/F32):
`text/main` post-prune at 160 = 109,395,059,968 native bytes / 195,212,097,312
logical parameters (`router_stored_bytes_post_remap` 61,633,920) and at 176 =
119,454,970,048 / 214,139,877,904 (67,797,312), with accounting closure exact
at both counts (73,144 tensors, 175,859,478,400 stored bytes, 312,224,379,840
logical parameters).

Integration expectations: real manifest comes from ExactInventory; Main runs
end-to-end comparison after landing. Real selections may carry quality
provenance (REAP/HOPE); the CLI preserves it via `provenance.quality_map`
and never generates one itself.
