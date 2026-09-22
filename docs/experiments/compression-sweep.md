# Compression sweep at ~90 GiB: where is MiMo's expert-pruning quality cliff?

Status: **design** (non-goals of this slice: any code, any model compute). Nothing here
has been built or measured; every measurement slot is null until measured. Supersedes
`docs/experiments/memory-matched.md` / `configs/experiments/memory-matched.json` **by
reference** (those files are unchanged on disk). Machine-readable form:
`configs/experiments/compression-sweep.json`. Per-candidate record contract:
`schemas/compression-candidate.schema.json`.

## The question

At the same ~90 GiB resident weight budget, is MiMo-V2.6-Flash better served by
**retaining more experts quantized harder** (REAP25 + aggressive Q3) or by **pruning
more low-salience experts while keeping survivors near their native QAT MXFP4
representation** (REAP40/45 + MXFP4)? Five candidates, all constructed from
Xiaomi's original weights, all landed inside one exact-byte window. The deliverable
is evidence of where the quality cliff actually is — not an assumption that REAP25
(the published checkpoint's choice) is optimal, nor that REAP40/45 (avoiding
requantization) is viable.

## Unit discipline and budget

All fit decisions are exact integer bytes; decimal GB is 10^9 bytes, GiB is 2^30
bytes, the two are never mixed.

| Quantity | Bytes | GiB |
| --- | --- | --- |
| **Matched-budget target (90 GiB exactly)** | **96,636,764,160** | 90.000000 |
| Tolerance ±2% | ±1,932,735,283 | ±1.80 |
| Window (inclusive) | [94,704,028,877 … 98,569,499,443] | [88.20 … 91.80] |

The budget covers the **text-model resident weight payload**: retained expert
instances + non-expert text classes with the router row-sliced to the retained
count. Auxiliary components (`memory-report.json components.*` except `text`:
mtp 1,189,400,448 B; vision 1,457,188,864 B; dflash 2,936,114,304 B; audio
522,254,336 B) are recorded separately and never netted in. KV cache and runtime
headroom are excluded (`runtime_memory_measured: false`).

A candidate whose planned total falls outside the window **may not be built as
specified**: its recipe is re-planned class-by-class (recorded), or it is reported
as an explicit budget delta. No fudging.

## Source pin

| Field | Value |
| --- | --- |
| `repo_id` | `XiaomiMiMo/MiMo-V2.6-Flash-RL` |
| `revision` | `5711b268169967567844e1e560e8a3966da959b1` |
| Files / bytes | 90 files, 177,767,644,228 B |
| Digest basis | `inventory.json source.files[].lfs_sha256` (71 LFS digests) + recorded sizes for all 90 files, recomputed against the on-disk tree at `$MIMO_LAB/source-models/XiaomiMiMo--MiMo-V2.6-Flash-RL/5711b268169967567844e1e560e8a3966da959b1`; digests verified = recorded sha256 matches the bytes on disk |
| Construction rule | Every candidate is constructed **only** from these original weights. The tacodevs MLX checkpoint and its published tensor-sensitivity work are **prior information only** and are never a construction source. |

## REAP ratio → retained counts (rounding rule)

`REAPxx` is defined as **the percent of experts PRUNED per layer**. Rounding rule,
stated once and applied uniformly to all 47 MoE layers:

```
retained = 256 − round_half_up(256 × reap_percent_pruned / 100)
round_half_up(x) = floor(x + 0.5)
```

| REAP | raw pruned (256 × %) | pruned (round-half-up) | **retained/layer** | retained total (×47) | pruned total (×47) |
| --- | --- | --- | --- | --- | --- |
| 25% | 64.0 | 64 | **192 of 256** | 9,024 | 3,008 |
| 30% | 76.8 | 77 | **179 of 256** | 8,413 | 3,619 |
| 35% | 89.6 | 90 | **166 of 256** | 7,802 | 4,230 |
| 40% | 102.4 | 102 | **154 of 256** | 7,238 | 4,794 |
| 45% | 115.2 | 115 | **141 of 256** | 6,627 | 5,405 |

Router rows (gate weight + `e_score_correction_bias`) are sliced row-wise to the
retained count with exact dense row scaling (`docs/pruning-integrity.md`,
`router_stored_bytes_post_remap_known = true`); routing stays top-8 over the
retained set.

## Byte constants — every number traceable to a source field

From `manifests/models/mimo-v2.6-flash-rl/memory-report.json` (unless noted):

| Constant | Bytes | Source field |
| --- | --- | --- |
| Expert instance, native MXFP4 (3 weights + 3 scales) | 13,369,344 | `native.per_layer_expert_layout.stored_bytes` |
| Expert instance logical parameters | 25,165,824 | `native.per_layer_expert_layout.logical_parameters` |
| Per retained index across 47 layers, native | 628,359,168 | `native.per_retained_expert_across_moe_layers_stored_bytes` |
| Per retained index, logical parameters | 1,182,793,728 | `native.per_retained_expert_across_moe_layers_logical_parameters` |
| Router bytes per retained index (row-slice) | 385,212 | `native.router_stored_bytes_per_retained_expert_across_moe_layers` (= 47 × (4096×2 B BF16 gate row + 4 B F32 bias entry)) |
| Router total at 256 | 98,614,272 | `native.router_stored_bytes` |
| Non-expert text stored (incl. router) | 8,894,573,440 | `native.non_expert_stored_bytes` |
| Non-expert text minus router | 8,795,959,168 | 8,894,573,440 − 98,614,272; equals the exact class-row sum below |
| MXFP4 exact rate | 4.25 bpw | `precision_costs.mxfp4_native_u8_packed.bits_per_logical_parameter_exact` |
| Text payload at 256 experts | 169,754,520,448 | `native.text_stored_bytes` |
| Pure-MXFP4 resident(R) formula | — | `R×628,359,168 + R×385,212 + 8,795,959,168`; validated against **all five** `prune_candidates[].native_text_payload_bytes` rows (256/192/176/160/144) |

Derived format bytes (exact integer formulas; every parameter count is divisible by
its group size, so no rounding occurs anywhere):

| Format | Rule | Per expert instance | bpw |
| --- | --- | --- | --- |
| `mxfp4_native` | params×4/8 + params/32×1 (U8 scales) | 13,369,344 | 4.25 |
| `q3_affine_g128` | params×3/8 + (params/128)×2 (int8 scale + int8 zero) | 9,830,400 (= 3 × 3,276,800) | 3.125 |
| `q8_affine_g64` | params×8/8 + (params/64)×2 | o_proj 34,603,008/tensor; sink 66/tensor | 8.25 |
| `q6_affine_g128` | params×6/8 + (params/128)×2 | embed 478,478,336/tensor (= 152,576 × (3,072+64)) | 6.125 |
| `native_fp8_e4m3` / `native_bf16` / `native_f32` | stored bytes, bit-exact copy | see class table | — |
| `native_dense_row_sliced` | router: 385,212 × R | — | — |

Per index across 47 layers: native 628,359,168 B; all-Q3 462,028,800 B
(47×3×3,276,800); per instance saving Q3 vs native = 3,538,944 B.

## Non-expert class table (exact inventory sums, closure proof)

Exact sums over `inventory.json tensors[]` (`component = "text"`, non-expert):

| Class | Units | B/unit | Class bytes | Native format |
| --- | --- | --- | --- | --- |
| `attention_qkv` (`.self_attn.qkv_proj.weight`) | 48 | 59,834,368 | 2,872,049,664 | FP8-E4M3 |
| `attention_qkv_scales` (`weight_scale_inv`) | 48 | 14,656 | 703,488 | F32 (aux) |
| `attention_o_proj` | 48 | 67,108,864 | 3,221,225,472 | BF16 |
| `attention_sinks` (`attention_sink_bias`) | 39 | 128 | 4,992 | BF16 |
| `dense_ffn` (layer 0) | 3 | 67,108,864 | 201,326,592 | FP8-E4M3 |
| `dense_ffn_scales` | 3 | 16,384 | 49,152 | F32 (aux) |
| `embeddings_lm_head` (embed_tokens + lm_head, [152576, 4096]) | 2 | 1,249,902,592 | 2,499,805,184 | BF16 |
| `norms` (97 layernorms) | 97 | 8,192 | 794,624 | BF16 |
| **Non-router sum** | | | **8,795,959,168** | |
| `router` (47 BF16 gates + 47 F32 biases) | 256 index | 385,212 | 98,614,272 | dense, row-sliced per recipe |
| **Total** | | | **8,894,573,440** = `native.non_expert_stored_bytes` (exact) | |

## Protected-class table (reproduced from the tacodevs prior, adapted to this budget)

Source pointers (prior information, allocation evidence only — **not** measured
sensitivity, **not** task-success evidence, **not** their allocation copied):

- `docs/baselines/tacodevs-reap25.md` § “Published precision reference (allocation
  evidence, not measured sensitivity)” and § Caveats.
- `manifests/baselines/tacodevs-reap25/normalized.json`:
  `precision_summary.projection_counts` (141 projections = 93× affine-b3-g128,
  22× mxfp4-b4-g32, 18× affine-b2-g128, 8× affine-b3-g64),
  `precision_summary.sensitive_projection_references` (per-projection
  bits/group_size/mode; Hessian-weighted output error + MILP allocation),
  `published_evaluation.distribution.compression_eval`
  (`attn_bits: 8`, `embed_bits: 16`, `lm_bits: 8`).
- `docs/experiments/memory-matched.json` `baseline_precision_reference`
  (“other text tensors: 8-bit affine g64”).
- Quant-spec vocabulary `bits/group_size/mode` incl. `mxfp4`:
  `tests/test_tacodevs_reap25.py::_spec_cycle()`.

| Class | Tacodevs prior | Our rule (adapted) | Applied in this sweep |
| --- | --- | --- | --- |
| router | inside “other text tensors 8-bit affine g64” | **highest precision**: native dense BF16 gate + F32 bias, exact row-slice (raised vs prior) | all 5 candidates, bit-exact row-slice |
| norms | 8-bit affine g64 | **FP16-class**: bit-exact native BF16 copy (raised vs prior) | all 5 candidates |
| embeddings / lm_head | `embed_bits 16` / `lm_bits 8` | **Q6-class ceiling** (`q6_affine_g128`): bit-exact native BF16 while budget allows; Q6 only where budget requires | native in (a)(b)(d)(e); **Q6 in (c)** |
| attention qkv | `attn_bits 8` | native FP8-E4M3 + F32 scales (already 8-bit class): bit-exact, **never re-encoded** | all 5 candidates |
| attention o_proj + sinks | `attn_bits 8` — “attention higher than bulk” | **Q8-class ceiling** (`q8_affine_g64`), far above any bulk format: native BF16 while budget allows; Q8 only where budget requires | native in (a)(b)(d)(e); **Q8 in (c)** |
| dense layer-0 FFN | other text tensors 8-bit | native FP8-E4M3 (already ~8-bit), bit-exact | all 5 candidates |
| expert projections | mixed b3/b2 affine + 22 native-MXFP4 slots, 3.29 bpw **their** allocation | native MXFP4 (4.25 bpw) for highest-salience survivors; `q3_affine_g128` (3.125 bpw) for the rest; **no 2-bit tier anywhere** | per candidate table |

Adaptation rule: class-table precisions are **protection ceilings** — a class never
drops below its ceiling and stays bit-exact native while the budget allows.
Second-generation quantization is applied **class-by-class only where the matched
budget requires it**, and every application is recorded (class, exact scope, bytes
before/after, reason) in that candidate's `second_gen_applied` list.

## The five candidates — exact allocation tables

Byte arithmetic per class; every row is `units × bytes_per_unit = class_bytes`
(exact integers); rows sum to the planned resident total; residual vs
96,636,764,160 B stated. Shared non-expert rows are byte-identical across (a),
(b), (d), (e) (all native, bit-exact): `attention_qkv` 2,872,049,664 +
`attention_qkv_scales` 703,488 + `attention_o_proj` 3,221,225,472 +
`attention_sinks` 4,992 + `dense_ffn` 201,326,592 + `dense_ffn_scales` 49,152 +
`embeddings_lm_head` 2,499,805,184 + `norms` 794,624 = **8,795,959,168**.
Only (c) replaces three of those rows (marked `Q8`/`Q6` below).

Within each layer, Q3 goes to the **lowest-REAP-salience retained experts first**;
the highest-salience survivors keep bit-exact native MXFP4. Per layer the Q3 count
is `base` (or `base+1` in the first `q mod 47` layers by layer index). Byte totals
are independent of which experts are chosen — every instance costs identically.

### (a) `reap25-aggressive-q3` — REAP25 (192/256) + aggressive Q3, 100% coverage

| Class | Format | Units × B/unit | Class bytes | 2nd-gen | Bit-exact |
| --- | --- | --- | --- | --- | --- |
| experts (all) | q3_affine_g128 | 9,024 × 9,830,400 | 88,709,529,600 | **yes** | no |
| router | native_dense_row_sliced | 192 × 385,212 | 73,960,704 | no | yes |
| shared non-expert rows (above) | native | — | 8,795,959,168 | no | yes |
| **Total** | | | **97,579,449,472** | | |
| vs target | | | **+942,685,312 (+0.9755%) — inside ±2%** | | |

Q3 9,024/9,024 instances (100%), native MXFP4 0. Per-layer: 192 Q3 everywhere.
Second-gen applied: expert payloads only — all 9,024 instances
(120,644,960,256 → 88,709,529,600 B). The Q3-only floor is +0.9755%; going lower
would require a sub-Q3 tier (out of recipe identity) and is not done because the
budget already accepts the floor.

### (b) `reap30-mixed-mxfp4-q3` — REAP30 (179/256) + mixed MXFP4/Q3

| Class | Format | Units × B/unit | Class bytes | 2nd-gen | Bit-exact |
| --- | --- | --- | --- | --- | --- |
| experts, native survivors | mxfp4_native | 1,432 × 13,369,344 | 19,144,900,608 | no | **yes** |
| experts, Q3 | q3_affine_g128 | 6,981 × 9,830,400 | 68,626,022,400 | **yes** | no |
| router | native_dense_row_sliced | 179 × 385,212 | 68,952,948 | no | yes |
| shared non-expert rows | native | — | 8,795,959,168 | no | yes |
| **Total** | | | **96,635,835,124** | | |
| vs target | | | **−929,036 (−0.0010%) — inside ±2%** | | |

Q3 6,981/8,413 instances (82.98%), native 1,432 (17.02%). Per-layer: Q3 = 149
(layers 1–25) / 148 (layers 26–47); native = 30 / 31. The highest-salience 30–31
survivors per layer keep native MXFP4. Second-gen: expert payloads only
(93,331,390,464 → 68,626,022,400 B for the converted instances). No non-expert
class was cut.

### (c) `reap35-mostly-mxfp4` — REAP35 (166/256) + mostly MXFP4, Q3 on low-salience survivors

| Class | Format | Units × B/unit | Class bytes | 2nd-gen | Bit-exact |
| --- | --- | --- | --- | --- | --- |
| experts, native survivors | mxfp4_native | 4,008 × 13,369,344 | 53,584,330,752 | no | **yes** |
| experts, Q3 | q3_affine_g128 | 3,794 × 9,830,400 | 37,296,537,600 | **yes** | no |
| router | native_dense_row_sliced | 166 × 385,212 | 63,945,192 | no | yes |
| attention_qkv (+ F32 scales) | native FP8 (2,872,049,664 + 703,488) | 48×59,834,368 + 48×14,656 | 2,872,753,152 | no | yes |
| attention_o_proj | **q8_affine_g64** | 48 × 34,603,008 | 1,660,944,384 | **yes** | no |
| attention_sinks | **q8_affine_g64** | 39 × 66 | 2,574 | **yes** | no |
| dense_ffn (+ scales), native | FP8 / F32 | — | 201,375,744 | no | yes |
| embeddings_lm_head | **q6_affine_g128** | 2 × 478,478,336 | 956,956,672 | **yes** | no |
| norms | native BF16 | 97 × 8,192 | 794,624 | no | yes |
| **Total** | | | **96,637,640,694** | | |
| vs target | | | **+876,534 (+0.0009%) — inside ±2%** | | |

Q3 3,794/7,802 instances (48.63%), native 4,008 (**51.37% — majority MXFP4**).
Per-layer: Q3 = 81 (layers 1–34) / 80 (layers 35–47); native = 85 / 86.

**Why this candidate also converts two protected classes (recorded exactly):**
with every non-expert class native, *any* allocation that keeps a native-MXFP4
majority (`q ≤ 3,900`) lands at **≥ 99,365,644,648 B (+2.824%)** — outside the
±2% window. To honor the recipe's identity (“mostly MXFP4”) inside the matched
budget, the budget pressure is absorbed by exactly two class-table conversions:
`embeddings_lm_head` → Q6 (2,499,805,184 → 956,956,672 B) and
`attention_o_proj`+`attention_sinks` → Q8 (3,221,230,464 → 1,660,946,958 B),
total −3,103,132,018 B. The already-quantized qkv FP8 tensors are **not**
re-encoded; norms and router stay untouched. Without these two cuts the
alternative would be to break “mostly MXFP4” (q ≥ 4,126 → ≤ 47% native), which
would misrepresent the candidate — both options are stated; this is the chosen,
recorded one.

### (d) `reap40-mxfp4-minimal-second-gen` — REAP40 (154/256) + minimal second-gen

| Class | Format | Units × B/unit | Class bytes | 2nd-gen | Bit-exact |
| --- | --- | --- | --- | --- | --- |
| experts, native survivors | mxfp4_native | 4,699 × 13,369,344 | 62,822,547,456 | no | **yes** |
| experts, Q3 | q3_affine_g128 | 2,539 × 9,830,400 | 24,959,385,600 | **yes** | no |
| router | native_dense_row_sliced | 154 × 385,212 | 59,322,648 | no | yes |
| shared non-expert rows (all classes untouched) | native | — | 8,795,959,168 | no | yes |
| **Total** | | | **96,637,214,872** | | |
| vs target | | | **+450,712 (+0.0005%) — inside ±2%** | | |

Q3 2,539/7,238 instances (35.08%), native 4,699 (64.92%). Per-layer: Q3 = 55
(layer 1) / 54 (layers 2–47); native = 99 / 100. Second-gen: expert payloads only
(33,944,764,416 → 24,959,385,600 B), the minimal cut that reaches the window;
**zero** protected-class conversion.

### (e) `reap45-mxfp4` — REAP45 (141/256) + pure MXFP4, no second-gen at all

| Class | Format | Units × B/unit | Class bytes | 2nd-gen | Bit-exact |
| --- | --- | --- | --- | --- | --- |
| experts, all native | mxfp4_native | 6,627 × 13,369,344 | 88,598,642,688 | no | **yes** |
| router | native_dense_row_sliced | 141 × 385,212 | 54,314,892 | no | yes |
| shared non-expert rows | native | — | 8,795,959,168 | no | yes |
| **Total** | | | **97,448,916,748** | | |
| vs target | | | **+812,152,588 (+0.8404%) — inside ±2%** | | |

Q3 0/6,627 (0%), native 6,627 (100%). `second_gen_applied: []` — the budget
requires no reduction, so **no class is re-encoded**: every tensor is a bit-exact
copy (or exact row-slice) of the pinned source.

## Pure-MXFP4 feasibility — the honest math

Floor = all survivors native MXFP4 + non-expert classes all native + router
row-sliced (`R×628,359,168 + R×385,212 + 8,795,959,168`, the formula validated
against all five `prune_candidates` rows):

| Candidate | Retained | Pure-MXFP4 floor (bytes) | vs target | In ±2% band? | Consequence |
| --- | --- | --- | --- | --- | --- |
| (a) REAP25 | 192 | 129,514,880,128 (120.62 GiB) | +32,878,115,968 (+34.02%) | **no** | 100% expert Q3 applied (recipe identity); lands +0.98% |
| (b) REAP30 | 179 | 121,341,203,188 (113.01 GiB) | +24,704,439,028 (+25.56%) | **no** | Q3 on 6,981 instances; non-experts untouched |
| (c) REAP35 | 166 | 113,167,526,248 (105.40 GiB) | +16,530,762,088 (+17.11%) | **no** | majority-MXFP4 impossible without the two recorded protected-class cuts (see (c) above) |
| (d) REAP40 | 154 | 105,622,593,688 (98.37 GiB) | +8,985,829,528 (+9.30%) | **no** | even applying the *full* class-table cut (embeddings Q6 + attention Q8, −3,103,132,018 B) only reaches 102,519,461,670 B (+6.09%) — still out; expert-side second-gen is unavoidable. **Alternative stated, not fudged:** build REAP40 pure-MXFP4 as an out-of-band frontier point at 105,622,593,688 B = 98.37 GiB, explicit budget delta **+8,985,829,528 B (+9.30%)** over target. The default recipe above instead uses the minimal Q3 cut and stays in band. |
| (e) REAP45 | 141 | 97,448,916,748 (90.76 GiB) | +812,152,588 (+0.84%) | **yes** | zero second-gen needed; only (e) fits pure |

So: **pure MXFP4 can hold ~90 GiB only at REAP45 (141/256)**. REAP40 gets close
but not inside ±2% by pruning alone; every less-pruned recipe needs
second-generation quantization of expert payloads, applied class-by-class and
recorded exactly as tabulated above.

## Prior information used / not copied

**Used (ordering priors and vocabulary only):**

- The tacodevs published precision allocation as a *sensitivity prior*: 141
  projections split 93× b3-g128 / 22× native-MXFP4 / 18× b2-g128 / 8× b3-g64;
  other text tensors 8-bit affine g64; `attn_bits 8`, `embed_bits 16`,
  `lm_bits 8` (sources above). This tells us which classes a published plan chose
  to protect — it is **not** measured sensitivity and **not** task-success evidence.
- Quant-spec vocabulary `bits/group_size/mode` incl. `mxfp4` from
  `tests/test_tacodevs_reap25.py::_spec_cycle()`.
- Their published calibration/held-out/on-policy metrics (PPL 9.271 → 9.529,
  KL 0.8315, top-1 79.5%; on-policy ALL 90.4% / code 90.8% / agent 88.5% /
  reasoning 94.2%) as **context only** — never reused as our measurements, never
  claimed reproduced.
- `docs/experiments/memory-matched.md` feasibility arithmetic and the
  `prune_candidates` byte cross-checks (superseded by reference, files untouched).

**Not copied:**

- No tacodevs/MLX tensor values, packed `switch_mlp` checkpoints, GPTQ artifacts,
  or their per-layer/per-projection precision map enter any candidate or any byte
  figure. No MLX requantization happens, ever.
- Their per-class numbers are not our per-class numbers where the budget demands
  difference: we quantize embeddings at Q6 (vs their embed-16), keep router and
  norms *above* their 8-bit tier, and never use their 2-bit slots.
- Every byte figure in this document derives from **our inventory** of the pinned
  original Xiaomi revision (`memory-report.json` fields + `inventory.json`
  tensor sums — closure shown in the class table).
- The published REAP25 checkpoint proves nothing about REAP25 being optimal, and
  avoiding requantization proves nothing about REAP40/45 viability: this sweep
  exists to find the actual cliff.

Construction rules (bit-exact MXFP4 passthrough; no MLX requantization ever;
BF16/FP16 intermediates compute-only and never treated as restored pre-QAT
weights; second-gen only where budget requires, recorded class-by-class; exact
affine byte math; router row-slice and accounting closure per
`docs/pruning-integrity.md`; real REAP selections via the external `--selection`
schema, never shape-only seeds as quality maps) live in
`configs/experiments/compression-sweep.json` `construction_rules`.

## Staged evaluation (design now, run later)

Mechanics (corpora, harnesses) are specified in the sibling eval-protocol slice
(`docs/experiments/eval-protocol.md`, `configs/experiments/eval-protocol.json`).
This sweep owns the stage order, the kill structure, and the decision rule:

| Stage | Name | Measures | Kill criteria |
| --- | --- | --- | --- |
| 1 | calibration distribution | perplexity, delta-ppl vs original, KL (nats), on-policy NLL (nats/token), top-1 + code/reasoning/agent agreement (fractions) | **PLACEHOLDER — thresholds null; must be calibrated against the observed stage-1 spread across all five candidates + original-MiMo control before anything runs. No fixed numbers are invented here.** |
| 2 | representative trajectories | code / reasoning / agent-trajectory agreement, long-context slices (8K, 16–32K, 32–64K, 64–128K, 128K+), pathological-generation scan | **PLACEHOLDER — same rule; set only after stage 1 exists** |
| 3 | reduced benchmark suite | reduced benchmark scores within practical limits | **PLACEHOLDER — set only after stage 2 exists** |
| 4 | full benchmarks | full suite — **only for Pareto-frontier candidates** | none; this is the selection stage |

Bad candidates die at the cheapest stage that can kill them. Every threshold stays
`null` + `status: placeholder_needs_calibration_against_stage1_spread` in the
config until the spread is observed; stages are `executed: false` with empty
`killed_candidates` until run.

**Decision rule:** rank survivors by **quality at matched size** (~90 GiB, ±2%);
report the quality-vs-resident-bytes Pareto frontier. **Throughput is excluded
from ranking** — all throughput numbers are PRE-OPTIMISATION diagnostics (kernel
asymmetry between datatypes is an implementation artifact, not a compression
property): a high-quality candidate is never rejected for poor current
throughput, a worse candidate never selected for a lucky kernel. The **top 2–3**
quality candidates proceed to runtime optimisation afterwards.

## Record contract and validation example

`schemas/compression-candidate.schema.json` (draft 2020-12, `schema_version` 1)
pins every required per-candidate field: exact REAP ratio + rounding rule,
retained counts per layer, the tensor-level allocation table, second-gen ledger,
resident/target bytes + residual + tolerance, calibration perplexity,
delta-perplexity, KL, on-policy NLL, top-1/code/reasoning/agent agreement,
all five long-context slices, pathological-generation findings, benchmark scores,
and the PRE-OPTIMISATION performance block (PP, C1, C4, C8, theoretical
bytes/token, active-weight bandwidth). Labels/units are explicit in every field
description; unmeasured values are `null`/`[]`, never zero.

The following validated example is a design-phase record for candidate (c);
its byte figures are identical to the config's `reap35-mostly-mxfp4` block:

```json
{
  "schema_version": 1,
  "candidate_id": "reap35-mostly-mxfp4",
  "status": "planned",
  "source_weights": {
    "repo_id": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
    "revision": "5711b268169967567844e1e560e8a3966da959b1",
    "file_count": 90,
    "total_bytes": 177767644228,
    "digest_basis": [
      "inventory.json source.files[].lfs_sha256 + recorded sizes recomputed against $MIMO_LAB/source-models/XiaomiMiMo--MiMo-V2.6-Flash-RL/5711b268169967567844e1e560e8a3966da959b1 (90 files, 177767644228 B)"
    ]
  },
  "recipe": {
    "reap_percent_pruned": 35,
    "rounding_rule": "retained = 256 - round_half_up(256 * reap_percent_pruned / 100)",
    "original_experts_per_layer": 256,
    "retained_experts_per_layer": 166,
    "retained_total_experts": 7802,
    "pruned_total_experts": 4230,
    "moe_layers": 47,
    "allocation_table": [
      { "class": "experts_native", "format": "mxfp4_native", "units": 4008, "unit_label": "expert_instance", "bytes_per_unit": 13369344, "class_bytes": 53584330752, "second_gen": false, "bit_exact": true, "source_field": "memory-report.json native.per_layer_expert_layout.stored_bytes" },
      { "class": "experts_second_gen", "format": "q3_affine_g128", "units": 3794, "unit_label": "expert_instance", "bytes_per_unit": 9830400, "class_bytes": 37296537600, "second_gen": true, "bit_exact": false, "source_field": "formula: 3794 * 9830400" },
      { "class": "router", "format": "native_dense_row_sliced", "units": 166, "unit_label": "retained_index", "bytes_per_unit": 385212, "class_bytes": 63945192, "second_gen": false, "bit_exact": true, "source_field": "memory-report.json native.router_stored_bytes_per_retained_expert_across_moe_layers" },
      { "class": "attention_qkv", "format": "native_fp8_e4m3", "units": 48, "unit_label": "tensor", "bytes_per_unit": 59834368, "class_bytes": 2872049664, "second_gen": false, "bit_exact": true, "source_field": "inventory.json tensors[] sum .self_attn.qkv_proj.weight" },
      { "class": "attention_qkv_scales", "format": "native_f32", "units": 48, "unit_label": "tensor", "bytes_per_unit": 14656, "class_bytes": 703488, "second_gen": false, "bit_exact": true, "source_field": "inventory.json tensors[] sum .self_attn.qkv_proj.weight_scale_inv" },
      { "class": "attention_o_proj", "format": "q8_affine_g64", "units": 48, "unit_label": "tensor", "bytes_per_unit": 34603008, "class_bytes": 1660944384, "second_gen": true, "bit_exact": false, "source_field": "formula: (67108864 * 8 // 8) + (33554432 // 64) * 2 per tensor = 34603008" },
      { "class": "attention_sinks", "format": "q8_affine_g64", "units": 39, "unit_label": "tensor", "bytes_per_unit": 66, "class_bytes": 2574, "second_gen": true, "bit_exact": false, "source_field": "formula: 64 params * 1 + (64 // 64) * 2 = 66 per tensor" },
      { "class": "dense_ffn", "format": "native_fp8_e4m3", "units": 3, "unit_label": "tensor", "bytes_per_unit": 67108864, "class_bytes": 201326592, "second_gen": false, "bit_exact": true, "source_field": "inventory.json layer-0 mlp weights" },
      { "class": "dense_ffn_scales", "format": "native_f32", "units": 3, "unit_label": "tensor", "bytes_per_unit": 16384, "class_bytes": 49152, "second_gen": false, "bit_exact": true, "source_field": "inventory.json layer-0 weight_scale_inv" },
      { "class": "embeddings_lm_head", "format": "q6_affine_g128", "units": 2, "unit_label": "tensor", "bytes_per_unit": 478478336, "class_bytes": 956956672, "second_gen": true, "bit_exact": false, "source_field": "formula: 152576 rows * (4096 * 6 // 8 + (4096 // 128) * 2) = 152576 * 3136 per tensor" },
      { "class": "norms", "format": "native_bf16", "units": 97, "unit_label": "tensor", "bytes_per_unit": 8192, "class_bytes": 794624, "second_gen": false, "bit_exact": true, "source_field": "inventory.json layernorm tensors" }
    ],
    "expert_precision": {
      "native_mxfp4_instances": 4008,
      "q3_instances": 3794,
      "instance_bytes_native": 13369344,
      "instance_bytes_q3": 9830400,
      "q3_percent": 48.63,
      "per_layer_distribution": { "base_q3_per_layer": 80, "extra_q3_layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34], "extra_q3_per_layer": 81 },
      "assignment_rule": "Q3 only on the lowest-REAP-salience survivors: per-layer Q3 count = 81 (layers 1-34) or 80 (layers 35-47); the highest-salience 85/86 survivors per layer keep bit-exact native MXFP4 (51.37% of instances native overall); totals independent of ordering (uniform instance bytes)"
    },
    "second_gen_applied": [
      { "class": "experts_second_gen", "format": "q3_affine_g128", "scope": "3794 of 7802 retained expert instances (lowest REAP salience first; 81 per layer in layers 1-34, 80 in layers 35-47)", "bytes_before": 50723291136, "bytes_after": 37296537600, "reason": "matched budget while keeping a native-MXFP4 majority: with all non-expert classes native, every majority-MXFP4 allocation lands at >= 99365644648 B (+2.82%), outside the +/-2% band" },
      { "class": "embeddings_lm_head", "format": "q6_affine_g128", "scope": "both [152576, 4096] tensors (embed_tokens + lm_head)", "bytes_before": 2499805184, "bytes_after": 956956672, "reason": "budget-required class-by-class reduction at the class table's Q6 ceiling so the majority of survivors can stay native MXFP4; norms/router untouched" },
      { "class": "attention_o_proj", "format": "q8_affine_g64", "scope": "all 48 o_proj tensors", "bytes_before": 3221225472, "bytes_after": 1660944384, "reason": "budget-required class-by-class reduction at the class table's Q8 ceiling (attention stays far above bulk precision); qkv stays native FP8 - never re-encoded" },
      { "class": "attention_sinks", "format": "q8_affine_g64", "scope": "all 39 attention_sink_bias tensors", "bytes_before": 4992, "bytes_after": 2574, "reason": "same budget-required attention-class conversion as o_proj" }
    ]
  },
  "size": {
    "resident_weight_bytes": 96637640694,
    "target_bytes": 96636764160,
    "gib": 90.0008,
    "decimal_gb": 96.637641,
    "residual_bytes": 876534,
    "residual_percent": 0.0009,
    "tolerance_percent": 2,
    "within_tolerance": true,
    "pure_mxfp4_floor_bytes": 113167526248,
    "pure_mxfp4_floor_within_tolerance": false
  },
  "metrics": {
    "kill": { "executed": false, "stage_reached": null, "killed": false, "criterion": null },
    "calibration": { "perplexity": null, "delta_perplexity_vs_original": null, "kl_divergence_nats": null, "corpus": null },
    "on_policy_nll_nats": null,
    "agreement": { "top1_fraction": null, "code_fraction": null, "reasoning_fraction": null, "agent_tooluse_fraction": null, "corpus": null },
    "long_context_degradation": {
      "8k": { "delta_perplexity": null, "agreement_fraction": null, "notes": null },
      "16k-32k": { "delta_perplexity": null, "agreement_fraction": null, "notes": null },
      "32k-64k": { "delta_perplexity": null, "agreement_fraction": null, "notes": null },
      "64k-128k": { "delta_perplexity": null, "agreement_fraction": null, "notes": null },
      "128k_plus": { "delta_perplexity": null, "agreement_fraction": null, "notes": null }
    },
    "pathological_generations": [],
    "benchmarks": []
  },
  "performance_pre_optimisation": {
    "label": "PRE-OPTIMISATION",
    "ranking_use": "excluded-from-quality-ranking",
    "pp": { "value": null, "unit": "tokens/s", "method": null },
    "c1_decode": { "value": null, "unit": "tokens/s", "method": null },
    "c4_decode": { "value": null, "unit": "tokens/s", "method": null },
    "c8_decode": { "value": null, "unit": "tokens/s", "method": null },
    "theoretical_bytes_per_token": null,
    "active_weight_bandwidth": { "value": null, "unit": "bytes/s", "note": null }
  },
  "notes": "Design-phase record: no compute has run; every measurement slot is null/empty by contract."
}
```

Validation actually performed (jsonschema 4.26.0, draft 2020-12):

```python
import json
from jsonschema import Draft202012Validator
schema = json.load(open("schemas/compression-candidate.schema.json"))
Draft202012Validator.check_schema(schema)          # passes: valid draft 2020-12
record = json.loads(<the JSON block above>)         # parses; validates: PASS
```

Negative controls (each rejected by the schema): `agreement.top1_fraction = 1.2`
(fraction > 1), deleting the `8k` long-context slice (required key),
`size.target_bytes = 1` (const 96636764160), `performance.label = "POST"`
(const `PRE-OPTIMISATION`), an unknown `format` value (enum), and a record
missing any required measurement key (e.g. `metrics.on_policy_nll_nats`).

## Performance diagnostics (PRE-OPTIMISATION)

Recorded per candidate, **excluded from quality ranking**:

- `pp` — prompt-processing throughput (llama-bench batched prompt processing;
  batch-size dependent, **never** labeled concurrency), tokens/s.
- `c1_decode` / `c4_decode` / `c8_decode` — decode tokens/s at 1/4/8 actually
  concurrent streams (aggregate over the concurrent wave), tokens/s.
- `theoretical_bytes_per_token` — exact active-weight streaming per decoded token
  (integer bytes):

```
bytes/token = embed_row + lm_head_row
            + Σ_{l=0..47} ( attention_bytes(l) + norms_bytes(l) )
            + dense_ffn_bytes(l=0)
            + Σ_{l=1..47} ( router_bytes(l) + Σ_{e ∈ top-8(l)} expert_instance_bytes(e) )
```

  where rows are one 4096-param row at the class format (native BF16: 8,192 B;
  Q6-g128: 3,136 B), `expert_instance_bytes(e)` = 13,369,344 native or 9,830,400
  Q3, and KV-cache reads are excluded (active weights only).
- `active_weight_bandwidth` = `theoretical_bytes_per_token ÷ measured decode
  seconds per token`, bytes/s.

## Later phase: concurrency-scaling sweep (SHELVED)

**Gating decision (recorded):** concurrency/runtime tests are **shelved until a
quality baseline exists** (original-MiMo control plus surviving candidates
measured through stages 1–3). Runtime optimisation — and this sweep with it —
stays gated behind the quality frontier: no concurrency number may be produced,
compared, or acted on before then. When unshelved, it runs **only for serious
candidates** = those surviving to the Pareto / top 2–3 quality selection.

- **Levels (measure all):** C1, C2, C4, C6, C8, C12, C16.
- **Per point record:** aggregate decode tok/s; per-stream tok/s; PP throughput;
  TTFT and inter-token latency (ms); RAM usage (bytes); obvious
  kernel/dispatch cliffs (notes); and if practical, expert overlap / unique
  experts touched as concurrency rises (availability recorded honestly when not
  practical).
- **Goal (a) — hardware throughput sweet spot:** where aggregate TPS stops
  improving materially.
- **Goal (b) — interactive sweet spot:** aggregate high without per-stream speed
  becoming unpleasant.
- **Rules:** do **not** assume powers-of-two or higher concurrency is better; if
  throughput drops at one level, profile it as a possible kernel/dispatch
  artifact **before** treating it as a hardware limit; all numbers are
  **PRE-RUNTIME-OPTIMISATION**, labeled as such — used to understand scaling and
  spot suspicious cliffs, **never** to pick the compression-quality winner;
  quality-first rules unchanged.
- **Harness vocabulary (reused):**
  `scripts/benchmark_matrix.py` serve mode — concurrency = actually parallel
  HTTP requests against one server, `aggregate_tokens_per_s` over the wave,
  statuses `ok/timeout/failed/parse_error`, model recorded by basename only;
  extend `concurrency.levels` to `[1,2,4,6,8,12,16]`. Its bench-mode PP points
  are batch-size knobs and are **never** reported as concurrency.
  `scripts/burst_replay.py` — schedule JSON `schema_version 1`,
  `events[{t_s, kind, prompt_tokens, gen_tokens}]`, per-event
  `latency_s`/`start_lag_s`/`predicted_tokens` → TTFT and inter-token latency
  for the interactive sweet spot; “successful requests per replay wall-hour”
  is schedule-specific observed throughput.

## Verification performed (this design slice)

- `schemas/compression-candidate.schema.json` — written, read back (21,713 B),
  re-parsed with `json.load`, and checked with `Draft202012Validator.check_schema`
  (valid draft 2020-12). The embedded example validates; six negative controls
  are rejected (listed above).
- `configs/experiments/compression-sweep.json` — written, read back, re-parsed
  with `json.load`; internal consistency assertions pass for all five candidates:
  every allocation row satisfies `units × bytes_per_unit = class_bytes`; row sums
  equal the planned totals and the pure-floor formula
  `R×628,359,168 + R×385,212 + 8,795,959,168`; residuals equal
  `total − 96,636,764,160` and all five sit inside ±1,932,735,283 B; the
  rounding rule reproduces 192/179/166/154/141 from its raw values; per-layer
  Q3 distributions sum to each `q3_instances`; native+Q3 instances equal the
  retained totals; the non-expert class closure equals
  `native.non_expert_stored_bytes` (8,894,573,440 B); all three stage-1..3 kill
  thresholds are `null` with placeholder status; the concurrency block is
  `shelved` with levels `[1,2,4,6,8,12,16]`.
- This document — read back after write; its embedded example block re-extracted
  and re-validated against the schema; every figure cross-checked against the
  config (double-write verification: write → read-back → parse/validate →
  confirm; no corruption observed in any of the three files).
- Privacy: no absolute local paths (placeholders `$MIMO_LAB/...` only), no trace
  content, no prompts, no secrets in any written file.
