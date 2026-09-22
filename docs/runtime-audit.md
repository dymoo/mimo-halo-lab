# Runtime and upstream source audit

Audited at the exact pinned revisions in `manifests/upstreams.json`, from the
local source-only clones (no builds run, no patches applied). Every finding
below cites the exact file and line in the pinned checkout. Nothing was
"fixed" by blind patching: findings record behavior and seams only.

Pins:

| Upstream | Revision | Role |
| --- | --- | --- |
| CerebrasResearch/reap | `1970473c51ca3caeb98c10392f15b3a08a672974` | REAP observation/pruning plumbing |
| halo-box/strix-llama.cpp | `8c1c282ecb194e8f02613defcc4a07c22b6d1c08` | Strix Halo runtime target |
| charlie12345/rocmfp4 | `ddd08770a7f063380c35353bdd7c44d7604106a6` | ROCmFP4 quant format |
| ggml-org/llama.cpp | `38a5b42d9a3e82e0a586bcd1caed121f36c87a73` | Official correctness reference |

The llama.cpp pin is the merge-base (fork point) of the pinned strix tree
against `ggml-org/llama.cpp` master, resolved via the GitHub compare API
(commit "HIP: Enable AllReduce for ROCm (#27825)", GPG-verified, 2026-09-15).

## 1. REAP (observation/plumbing seams for pruning)

### 1.1 Expert-count assumptions: none hardcoded; 160/176/192/256 safe by construction

- `src/reap/prune.py:82-146` — per-layer selection is
  `torch.topk(saliency, n_experts_to_prune, largest=False)` over a saliency
  vector whose length comes from `observer_data[layer]["expert_frequency"].shape[0]`
  (line 83): expert count is read from observed tensors, not assumed. Slicing
  reindexes the expert `ModuleList`, then rows of `router.weight`, then
  `router.bias` when present, then patches the config attribute
  (`model_attrs["num_experts"]`). All index-list based; no power-of-two or
  fixed-count guards. 256→192, 192→176, 176→160 all flow through identical code.
- `src/reap/pruning_metrics.py:22-64` — `initialize_pruning_state(num_experts)`
  allocates `expert_frequency`, `pairwise_expert_frequency` (E×E), `ean_*`,
  `reap` from the actual count; no shape constants.
- Only modulo/index arithmetic in clustering (`cluster.py:63`, `605`,
  `restricted_cluster.py:44-47`) is pair-index unpacking, not count-dependent.
- **Gap (must be built, not patched in place): no MiMo entry in
  `MODEL_ATTRS`.** `src/reap/model_util.py:7-118` registers Qwen3Moe,
  Qwen3-Coder, NonUniformQwen3, Llama4, Mixtral, DeepseekV2, Ernie4_5 (two
  variants), gpt-oss-20b, Glm4Moe. MiMo-V2.6-Flash-RL needs a new
  `MODEL_ATTRS` row (attribute names: `moe_block`, `up_proj`, `down_proj`,
  `gate_proj`, `fused`, `router`, `num_experts`, `num_experts_per_tok`) plus a
  matching `OBSERVER_CONFIG_REGISTRY` row in `src/reap/observer.py`
  (`num_experts_attr_name` / `top_k_attr_name` must name the real MiMo config
  attributes; the observer validates them at `observer.py:318-327` and fails
  closed). Closest precedents: Glm4Moe (sigmoid gate +
  `e_score_correction_bias`, `models/glm/modeling_glm4_moe.py:224-268`) and
  Ernie (correction bias sliced at `prune.py:115-123`, gated on exact class
  names `Ernie4_5_MoEForCausalLM`/`Ernie4_5_MoeForCausalLM`). MiMo's actual
  normalized-sigmoid + correction-bias router will need the same correction
  bias handling — modeled on the Ernie branch, verified against real metadata
  first, never assumed.

### 1.2 Saliency math (contract cross-check)

- `src/reap/pruning_metrics.py:188-198` — per active expert `i`:
  `ean_norm = ‖activations‖₂` over tokens where the expert is selected;
  `reap[i] = (ean_norm * active_router_weights).mean()` — the **conditional
  mean of `g_k·‖f_k‖`**, exactly the contract's saliency definition.
  `ean_sum`/`weighted_ean_sum` (`sum` over active tokens) are retained as
  separate unconditional accumulators — diagnostic only; nothing substitutes
  them for the conditional score.
- Router weight provenance: the observer computes routing weights from
  recorded logits (`observer.py:350-355` fused path returns `(router_scores)`
  from the module output; loop path extracts logits then top-k at
  `layerwise_observer.py:664-665`). `renormalize_router_weights` maps the
  `norm_topk_prob` renormalization path (`main.py:121-130`,
  `args.py:144-148`). For MiMo the gates recorded must be the model's own
  sigmoid+bias gates — the observer reads `selected_experts` and
  `routing_weights` from the actual forward, so the transfer hypothesis
  (paper softmax assumptions vs MiMo normalized sigmoid) stays at the
  gate-source seam, not in the metric code. Record this per contract;
  never alter the router.
- `prune.py:136` comment: fused-expert slicing "only tested for llama-4". The
  MiMo HF layout is a non-fused `ModuleList` (assumption to verify from the
  real metadata), so the unfused path applies; the fused branch is not a
  blocker.

### 1.3 What is missing upstream (recorded, deliberately not patched here)

- No MiMo `MODEL_ATTRS` / `OBSERVER_CONFIG_REGISTRY` entry (above).
- `patched_model_map` (`model_util.py:168-186`) hard-paths a few local
  artifact models; MiMo needs its own entry if remote-code loading is
  required.
- No HOPE/QP code exists upstream (expected; HOPE official code is
  unreleased). Extension lives in the observation/plumbing seams only.

## 2. strix-llama.cpp runtime (MiMo, expert counts, backends)

### 2.1 MiMo architecture support is native at this pin

- `src/llama-models/models/mimo2.cpp:3-23` — `LLM_ARCH_MIMO2`
  (`"mimo2"`, `src/llama-arch.cpp:148`); `swa_type = LLAMA_SWA_TYPE_STANDARD`;
  per-layer SWA flags read from the `attention.sliding_window.pattern` GGUF
  array (`load_arch_hparams`, and `ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,
  hparams.is_swa_impl)`), plus `attention.value_scale` and SWA RoPE base —
  the 39-SWA/9-global split is metadata-driven, not hardcoded.
- `mimo2.cpp:26-27` — `n_layer == 48` recognized as
  `LLM_TYPE_310B_A15B` (`src/llama-model.h:150`); other layer counts fall
  back to `LLM_TYPE_UNKNOWN` (non-fatal label only).
- `mimo2.cpp:57-64` — MoE tensors: `ffn_gate_inp` (router), fused expert
  `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` with dims `{n_embd, n_ff_exp,
  n_expert}` straight from GGUF metadata, and `ffn_exp_probs_b` (the router
  correction bias). A pruned checkpoint with a changed expert count converts
  and loads if config→GGUF metadata and tensor dims stay consistent; the
  runtime imposes no fixed expert count.
- `mimo2.cpp` graph — `build_moe_ffn(..., n_expert, n_expert_used,
  LLM_FFN_SILU, /*norm weights=*/true, hparams.expert_weights_scale,
  LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il)`: fused top-k routing with
  sigmoid gating, correction-bias add, weighted (renormalized) expert
  reduction — matches the contract's fused top8/no-shared/weighted-reduction
  shape. `n_expert`/`n_expert_used` come from GGUF `expert_count` /
  `expert_used_count` (converter `tools/convert_hf_to_gguf.py`
  `mimo.py` `set_gguf_parameters`; converter class registered for
  `MiMoV2FlashForCausalLM` and `MiMoV2ForCausalLM`).
- MTP: `graph_mtp` (`mimo2.cpp:261+`) requires `nextn.eh_proj/enorm/hnorm`
  and asserts `n_layer_nextn > 0`; the tensor loader probes
  `blk.<n_layer>.nextn.eh_proj.weight` and skips MTP tensors when absent
  (`mimo2.cpp:28-33`), so a trunk-only checkpoint loads cleanly.
- Attention: fused `wqkv` with distinct Q/K/V head dims plus optional
  attention sinks (`mimo2.cpp:57-80`); CUDA FA tile dispatch special-cases
  head dim 192 with `gqa_ratio` 8 (SWA) or 16 (full)
  (`ggml/src/ggml-cuda/fattn.cu:297-301`, `fattn-tile.cuh:1403-1407`).

### 2.2 Expert-count guards and backend fallbacks (160/176/192/256, powers of two)

- **CUDA routed MMQ** `ggml/src/ggml-cuda/mmq.cuh`:
  - range guard `mmq_rdna3_5_id_n_experts_ok`: `8 <= n_experts <= MMQ_ROUTED_MAX_EXPERTS` (`1024`) — `mmq.cuh:1354-1361, 1802`;
  - tile width `J` is selected from `rows_per_expert = ceil(ncols_dst /
    nchannels_y)` — computed from the *actual* expert count (`mmq.cuh:1803`),
    bucketed by measured tables (`mmq.cuh:1608-1621`). The comment
    (`mmq.cuh:1601-1605`) explains tuning was done for 256-expert shapes on
    gfx1151; **correctness is guarded for any count in [8, 1024], but the
    J-choice measurements do not cover 192/176/160 rows-per-expert values —
    a performance unknown, recorded as such, not patched**;
  - the special `j48_128e` case fires only for exactly 128 experts
    (`mmq.cuh:1631-1634`) — inert for 192/176/160/256.
  - dispatch chain with explicit fallbacks (`ggml-cuda.cu:1922-1971`):
    MMVQ (per-arch batch caps, `mmvq.cu:152-292` `get_mmvq_mmid_max_batch`)
    → MMB batched path (`mmb.cu:845+`, gated by
    `ggml_cuda_mmb_supported_mmid`) → MMQ → MMF → a fallback that requires
    stream synchronization (`GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync)`
    at `ggml-cuda.cu:1968`); CUDA-graph capture never reaches the sync path.
  - the RDNA3.5 fused weighted-reduction kernels
    (`mmvq.cu:3096-3217`, guard `mmvq.cu:3168-3175`) support only
    `GGML_TYPE_IQ4_NL` and `GGML_TYPE_Q8_0`; other weight types (including
    ROCmFP4 `Q4_0_ROCMFP4*`) take the generic dispatch. Fallback, not a bug.
- **Vulkan** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:
  - fused top-k MoE pipeline picked by `ceil(log2(n_experts))` with
    power-of-two spec constants; non-power-of-two counts (192/176/160) take
    the push-constant variant — explicit non-pow2 support
    (`ggml-vulkan.cpp:13739-13743`, pipelines built at `7104-7108`);
  - hard cap `n_expert > (1 << (num_topk_moe_pipelines-1))` = 512 refuses
    the fusion (`ggml-vulkan.cpp:19784-19787`) and falls back to the
    unfused op sequence — a real guard, visible as a perf cliff only;
  - `topk_moe.comp` handles `n_experts % WARP_SIZE != 0` with a lane guard
    (`vulkan-shaders/topk_moe.comp:111-113`) — 176/192/160 are all fine;
  - expert row-id handling: `count_experts.comp` builds per-expert counts +
    hoisted row-id lists (`hoist_row_ids`); the matmul shaders choose a
    power-of-two fast path only for `nei0` (tokens per slot), with a generic
    ballot/scan fallback (`mul_mm.comp:222-231`,
    `mul_mm_id_funcs.glsl:8-11`) — expert count itself needs no pow2;
  - sigmoid+bias gating (`GATING_FUNC_SIGMOID` with `BiasProbs` buffer,
    `topk_moe.comp:118-124`) covers the MiMo router on Vulkan.
- **CPU**: `MUL_MAT_ID` has an IQ-panel fast path gated by a per-expert
  batch threshold (`ggml-cpu/iqp.cpp:25-27, 1115-1125`), generic chunked
  path otherwise (`ggml-cpu.c:1662-1920`) — no count assumptions.
- **Backend fallback ladder** (observed, unchanged): Vulkan is the strix
  tree's default recommendation on the RDNA3.5 iGPU (`README.md:130`),
  ROCm/HIP is supported with the known control below, CPU is the AVX-512
  fallback (`README.md:132`). No silent provider substitution exists; the
  backend is chosen by device availability and `--device`.

### 2.3 The HIP correctness control (record as control, never as performance)

- `README.md:76-82` — on gfx1151, HIP async execution returns badly wrong
  output for batched inference (perplexity ~88 vs ~9.4); setting
  `HIP_LAUNCH_BLOCKING=1` serializes kernel launches and restores
  correctness at a performance cost. Upstream words: "a workaround for a
  ROCm/HIP issue, not a fix".
- `CONTRIBUTING.md:127` — issue reports are expected to state
  `ROCm <version>, HIP_LAUNCH_BLOCKING=<0|1>`.
- `.github/workflows/build-self-hosted.yml:117-122` — the pinned tree's own
  gfx1151 ROCm CI sets `HIP_LAUNCH_BLOCKING: "1"` with an explanatory
  comment.
- Policy: all HIP-path benchmarks in this project run with
  `HIP_LAUNCH_BLOCKING=1` and the value is recorded in the benchmark config
  (`configs/hardware-benchmarks.json` → `runtime_controls.hip`). Vulkan runs
  are the perf lane; HIP-with-control runs are the correctness-verified
  comparison lane. Neither lane's numbers are claimed as the other's.

### 2.4 Other recorded runtime knobs (upstream-owned, not patched)

- Vulkan batched mat-vec chunking: batches of 3/5/6 columns are slow on
  RADV; the fork splits them (`README.md:84-88`). `GGML_VK_MMV_NO_SPLIT=1`
  restores the single upstream dispatch for comparison.
- `GGML_VK_PERF_LOGGER=1` gives per-op Vulkan timings (profiling only).
- UMA/GTT sizing (`amdgpu.gttsize`, `ttm.pages_limit`, BIOS UMA split)
  gates whether a model that fits RAM actually allocates on the iGPU
  (`README.md:72-75`).

### 2.5 Portability note (local tracked patch, Mac compile fix only)

- The only local source patch against this pin is
  `patches/strix-macos-prefetch.patch` (base
  `8c1c282ecb194e8f02613defcc4a07c22b6d1c08`, SHA-256
  `c199b3b31c1b2e5c0037824ff119fae2748d95f3455a454d50dcd7a5777abe7e`),
  recorded under the strix entry's `patches` array in
  `manifests/upstreams.json`. It guards `llama_ple_disk::prefetch`'s
  per-row file-read advice in `src/llama-ple-disk.cpp` by platform: Linux
  keeps the original `posix_fadvise(..., POSIX_FADV_WILLNEED)` call
  unchanged, Darwin uses `fcntl(fd, F_RDADVISE, &radvisory)` with `ra_count`
  clamped to `INT_MAX`, and other non-Windows platforms compile the loop
  as a no-op — the `#ifdef __linux__` guard precedent already used for the
  `posix_fadvise` call in `src/llama-mmap.cpp:474-479`. `<fcntl.h>` was
  already included via the file's existing `#if !defined(_WIN32)` block;
  only `<climits>` was added.
- **Platform compile fix, not a Qwen/MiMo architecture port.** The Darwin
  branch failed to compile solely because `posix_fadvise` is undeclared on
  macOS (`llama-ple-disk.cpp:386`, failure artifact://338). No pread path,
  RAII, error handling, prefetch/advice semantics, caching behavior, or
  Linux/gfx1151 behavior changed; no model-architecture, graph, or backend
  code is touched. A wrong prefetch still costs readahead and nothing else
  (`llama-context.cpp:1743-1747`).
- Pin verification remains valid: `git -C upstreams/strix-llama.cpp
  rev-parse HEAD` still equals
  `8c1c282ecb194e8f02613defcc4a07c22b6d1c08` (single dirty worktree file),
  and `scripts/build_runtime.py` pin-checks HEAD against the manifest
  revision, not the worktree — no driver change was needed.
- Build status: **not compiled here** — Main reruns the actual incremental
  Mac build after this patch; this note is not a green-build claim.

## 3. rocmfp4 (quant format)

- `docs/ROCmFP4-SPEC.md:1-93` — `Q4_0_ROCMFP4`: 32-weight blocks, E2M1-style
  4-bit codebook with the largest magnitude retuned (`{0,±1,±2,±3,±4,±6,±8,±10}`
  half-scale integers), dual unsigned-E4M3 scales (one per 16-weight half),
  18 bytes/block = **4.50 bpw** (verified at `ROCmFP4-SPEC.md:54`).
- `README.md:16` — `Q4_0_ROCMFP4_FAST`: single-scale speed layout at
  **4.25 bpw**; `_LEAN/_COHERENT/_STRIX` recipe variants documented in
  `docs/IMPLEMENTATION-NOTES.md:22-25`.
- NOT MXFP4/NVFP4: a distinct format; MLX checkpoints are never requantized
  through it (project contract).
- Upstream-owned scripts (used as-is, referenced from the bringup doc):
  `scripts/apply-rocmfp4.sh`, `scripts/build-strix-rocmfp4-mtp.sh`,
  `scripts/check-rocmfp4-all-regression.sh`,
  `scripts/check-rocmfp4-{quant,rocm-runtime,vulkan-runtime,rocm-cpy,
  rocm-fattn,qwen-mtp,qwen35-a3b-mtp}-regression.sh`, quickstart
  `docs/STRIX-HALO-QUICKSTART.md` (quantize + run command forms cited
  verbatim there).

## 4. llama.cpp reference (tool behavior we rely on)

- `tools/llama-bench/llama-bench.cpp:226-249, 434` — output formats
  `csv|json|jsonl|md|sql`; `-p/-n/-pg` define test shapes (`test` values like
  `pp8192`, `tg512`, `pp8192+tg512`, `:2050-2056`), JSON rows carry
  `avg_ts`/`stddev_ts`/`avg_ns` (`:1583-1603`).
- **No parallel-request flag exists** at this pin (arg surface
  `:550-1058`): only `-b/--batch-size`, `-ub/--ubatch-size`. Therefore the
  C1/C2/C4/C8 concurrency numbers are measured by
  `scripts/benchmark_matrix.py --mode serve` against real concurrent HTTP
  requests, and llama-bench batch-size knobs are never reported as
  concurrency.

## 5. Transfer-hypothesis ledger (to verify with real metadata before any use)

1. MiMo HF layout: `ModuleList` experts (non-fused) — assumed; verify from
   safetensors header/index before REAP slicing.
2. Router: normalized sigmoid gates + correction bias — matches
   `build_moe_ffn(... SIGMOID)` + `ffn_exp_probs_b` on the runtime side and
   the GLM/ERNIE precedent on the REAP side; verify gate semantics from the
   HF config and observe with the model's own gates.
3. top8 → `n_expert_used = 8`; 48 layers → `310B_A15B` label; 39 SWA + 9
   global via the pattern array — verify all three from the real GGUF
   conversion before benchmark claims.
4. 4.25/4.50 bpw hypotheses confirmed by the ROCmFP4 spec (§3) — byte
   budgets stay measured, not nominal.
