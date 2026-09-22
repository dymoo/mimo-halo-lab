# HOPE objective: implementation notes, sources, and exact commands

Scope: `src/mimo_halo/pruning/hope.py` (stdlib core), `src/mimo_halo/pruning/reap_adapter.py`
(REAP integration), `tests/test_hope.py`, `configs/hope.json`, `patches/reap-hope.patch`.

## 1. Sources

- **Paper**: Tseng, Kaul, Zancato, Xia, Soatto, *Higher-order pruning of experts in
  mixture-of-experts language models*, **arXiv:2609.18916v1** (2026-09-16), CC-BY-4.0.
  Sections implemented: §2.2 (REAP baseline score), §3.1–3.4 (objective, error
  decomposition, Theorem 1), Eq. 3 / Appendix A.7 (conditional-normalized F),
  §3.5/Fig. S4 (zeroed off-diagonals recover REAP), Theorem 2 (prune-set QP),
  §3.6 (continuous relaxation + top-|P| rounding), §4.1 (calibration setup).
- **REAP upstream**: `CerebrasResearch/reap` pinned at
  `1970473c51ca3caeb98c10392f15b3a08a672974`, cloned at `upstreams/reap`
  (override with `$REAP_SRC`). Read-only anchor: `src/reap/pruning_metrics.py`,
  `src/reap/observer.py`, `src/reap/prune.py`, `src/reap/model_util.py`.
- **HOPE official code is NOT released** ("code … will be released upon
  publication"); the authors' QP solver is unnamed and Appendix D solver settings
  are not retrievable from the v1 HTML. Our solver therefore reproduces the
  *recipe* of §3.6, not the authors' exact solver. This is stated in every
  relaxation certificate (`solver_note`).

## 2. Formulas as implemented

Per MoE layer, E experts, top-K routing; `a_k(x) = g_k(x) * ||f_k(x)||_2` for the
actually-routed set `T(x)`:

| Quantity | Formula | Code |
|---|---|---|
| First-order (REAP) score | `S_k = (1/\|X_k\|) Σ_{x∈X_k} a_k(x)` | `HopeStats.first_order()` — matches pinned REAP's per-expert `reap` metric (`mean(ean_norm * active_router_weight)` over active tokens) |
| Interaction (conditional, Eq. 3/A.7) | `F_ij = (1/\|X_ij\|) Σ_{x∈X_ij} a_i a_j`, `X_ij = ∅ ⇒ 0` | `HopeStats.conditional_f()` |
| Diagonal | `F_kk = E[a_k²]` (contribution variance kept; differs from REAP's `E[a_k]²` — paper §3.5) | pair diagonal accumulator |
| QP (Theorem 2) | `min_{p∈{0,1}^E, Σp=\|P\|} pᵀF p`, `p_k=1 ⇔ k pruned` | `exhaustive_select` / `scipy_relax_select` |
| Routing frequency | `\|X_k\| / N` (REAP `expert_frequency/total_tokens`) | `HopeStats.routing_frequency()` |

**`F` is entrywise non-negative but not necessarily PSD.** The exhaustive
selector never needs PSD; the relaxation uses gradient `(F+Fᵀ)p`, also PSD-free.
Nothing in this codebase claims convexity.

**Never substitute REAP's `pairwise_expert_frequency` for `X_ij` counts**: upstream
stores the marginal outer-sum `e_i + e_j`, not co-activations. HOPE statistics
accumulate true per-pair co-activation counts (`pair_count[i][j]`).

### Count-safe statistics contract

One `HopeStats` per layer. `add_batch(selected, gates, norms)` is atomic
(validate whole batch, then commit) and enforces: fixed `top_k` width, integer
unique in-range ids, gates finite in `[0, 1+slack]`, norms finite ≥ 0.
`validate()` (run on every load/merge/export) checks: `Σ act_count ==
total_rows * top_k`, pair-count symmetry, `pair_count diagonal == act_count`,
Fréchet bounds `max(0, act_i + act_j − N) ≤ |X_ij| ≤ min(act_i, act_j)`,
`gate_sum ≤ act_count`, non-negativity/finiteness of all sums, zero pair-sum on
zero counts, and per-capability parent monotonicity. Serialization is
deterministic (sorted keys, atomic `os.replace`); `merge()` implements the
streaming/resume path and rejects shape mismatches.

### Capability partitions

`add_batch(..., capability=name)` accumulates a full sub-statistics per label
(rows are labeled round-robin by the caller). Unlabeled rows exist only at the
top level; capability counts are validated as subsets of the parent. Selection
can run on one partition: `select --capability NAME`.

## 3. Selectors

- **`exhaustive`** (default for `E ≤ 16`): enumerates all `C(E,|P|)` fixed-size
  sets. Returns the exact global optimum with a certificate
  (`subsets_evaluated == C(E,|P|)`, tie count, `exact=true`,
  `global_optimal=true`). Ties resolve to the lexicographically smallest id set
  (strict-improvement replacement in combinations order). `E > 16` is **refused**
  with a pointer to the relaxation — never a silent downgrade.
- **`scipy-slsqp-relaxation-top-round`** (required for `E > 16`):
  SLSQP on `p ∈ [0,1]^E, Σp = |P|`, deterministic linspace start + seeded
  restarts (config `relaxation.seeds`), then top-|P| rounding (§3.6 recipe),
  optional deterministic greedy binary 1-swap refinement (config
  `local_swap`, reported as `local_swap_applied` + move count). Certificates
  always carry `exact=false`, `global_optimal=false`, `psd_assumed=false`, the
  relaxed objective, and the rounding gap. **Missing scipy raises
  `HopeDependencyError` with the install command — there is never a silent
  fallback between objectives.**

### Pathological toy cases (closed-form, `hope toys`)

| Case | E, \|P\| | Structure | HOPE optimum | First-order (REAP diag) |
|---|---|---|---|---|
| `uniform` | 12, 4 | all pairs equal | `{0,1,2,3}`, ties = C(12,4)=495 | identical (control) |
| `important_pair` | 10, 3 | one pair punished by 2·1000 | `{0,2,3}`, obj 3 | `{0,1,2}`, obj 2003 |
| `individually_strong_redundancy` | 8, 2 | cheap-but-redundant pair | `{0,2}`, obj 55 | `{0,1}`, obj 90 |
| `block_interaction` | 16, 4 | two 8-cliques, within-w=30 | `{0,1,8,9}`, obj 124 | `{0,1,2,3}`, obj 364 |
| `ties` | 9, 4 | three linked pairs | `{0,2,4,6}`, obj 4, ties = 66 (inclusion–exclusion) | `{0,1,2,3}`, obj 44 |
| `sparse` | 16, 6 | one weak pair, cheap diags | `{0,1,2,3,4,8}`, obj 50 | identical (control) |

`tests/test_hope.py` cross-checks every selector against an independent
product-space oracle (E = 8…12), an independent dense quadratic-form
recomputation over **all** fixed sets (E = 8…16, `subsets_evaluated ==
C(E,k)`), and analytic expectations for all six cases.

## 4. One-pass REAP integration

`reap_adapter.OnePassListener` wraps `reap.observer.update_pruning_state` (the
module-level name the observer hook calls). Each call first resolves the
model's **actual** routed ids/gates at the shared seam
(`reap_adapter.resolve_actual_routing`) — *before* the pinned accumulator
touches layer state — injects those ids as `selected_experts` so REAP metrics
accumulate over the same experts, then decodes the returned
`PreparedPruningBatch` (filtered activations `(E, tokens, d)`, selected ids,
router logits) into HOPE statistics with the same ids/gates and a post-call
cross-check that the batch carries exactly the injected ids. If actual routing
cannot be resolved, the seam raises **before any metric mutation** — there is
no approximation branch. One routed pass produces:

- REAP side: `expert_frequency`, `ean_*`, `reap` scores (pinned upstream accumulators — code untouched, fed the seam's actual ids/gates);
- HOPE side: `first_sum`, `pair_sum`, true `pair_count`, gates, norms, capability partitions (decoded from the same observation).

No second model execution, no upstream edit for observation, no second
checkpoint framework. Layer attribution is by observer-state identity and fails
closed if a state disappears. `install()`/`uninstall()` are symmetric.

**Actual routed ids / gates (shared observation seam).** The upstream hook
re-selects with `topk` on **raw** router logits (`observer.py`:
`_, selected_experts = torch.topk(router_logits, top_k)`); REAP and HOPE both
consume the seam's single resolved observation instead of post-hoc relabeling.
Contracts verified against the installed transformers **4.57.6** source:

| Block / router | Model routing (source read) | Seam behavior |
|---|---|---|
| `MixtralSparseMoeBlock` | `softmax → topk → /= selected sum`, unconditional (`models/mixtral/modeling_mixtral.py`, `MixtralSparseMoeBlock.forward`) | upstream raw-logit top-k **is** the model's top-k (softmax strictly increasing); gates = selected-set-renormalized softmax; raw logits pass through unchanged (identity feed) |
| `Qwen3MoeSparseMoeBlock` | same, renormalization gated on `norm_topk_prob` (`models/qwen3_moe/modeling_qwen3_moe.py`) | accepted only when `norm_topk_prob=True`; otherwise explicit unsupported error |
| sigmoid + `e_score_correction_bias` (e.g. `Glm4MoeTopkRouter`: selection `topk(sigmoid(logits)+bias)` with group masking, weights = **unbiased** `sigmoid(logits)` gather, optional `norm_topk_prob`, `× routed_scaling_factor` — `models/glm4_moe/modeling_glm4_moe.py`) | see source | the seam implements exactly that selection/gather (`resolve_actual_routing`), and feeds the accumulator `log(sigmoid(logits))` so its internal softmax + renormalization reproduces the same gates — **unit-tested** (`tests/test_hope.py::ActualRoutingSeamTests`, including a case where raw-logit+bias and sigmoid+bias pick different experts) |
| everything else — DeepSeek-V2 (`softmax` scores + `topk_method` group selection + `routed_scaling_factor`), Ernie, gpt-oss, fused Llama4, `Glm4MoeMoE`/`DeepseekV2MoE` (block `forward` returns a bare tensor, so the observer's logits capture is not the router's logits), **MiMo** (no `MODEL_ATTRS` entry) | unverified / group-scaled | **fail closed at `attach_model`** with an explicit unsupported error before any forward or metric mutation — no mirroring, no approximation |

Consequences: (a) the previously implemented bias formula
`topk(logits + bias)` + gathered `sigmoid(logits + bias)` was **wrong** for
normalized-sigmoid/noaux_tc routers and is removed — selection is
`topk(sigmoid(logits) + bias)` while output weights gather **unbiased**
`sigmoid(logits)` and renormalize (the correction bias affects the choice, never
the gathered weight values); (b) no claim of support survives for architectures
with unverified group-routing or routed scaling — those raise instead of
recording diverging REAP/HOPE stats; (c) **MiMo observer integration remains
open in #21**. The paper assumes softmax top-K (§3.1); sigmoid+correction-bias
routing violates that assumption, so even a future verified bias integration
would be a documented modeling choice, not paper ground truth. Recording
raw-logit top-k and calling it "actual routing" on a bias router is rejected by
design.

**Slicing/selection use the real upstream path.** `reap.prune.prune` performs
expert `ModuleList` reindexing, router row slicing, config patching and
`save_pretrained`. REAP selection runs natively inside upstream code
(`topk(reap, largest=False)`). HOPE selection pre-solves the exhaustive QP and
injects the set as an exact 0/1 saliency (`encode_exact_prune_set`): exactly
`|P|` zeros, next value 1.0, so upstream top-k returns precisely that set —
upstream slicing code byte-identical for both methods.

## 5. Tiny integration smoke (INTEGRATION evidence only)

Builds `MixtralForCausalLM` locally from `MixtralConfig` — 2 layers, 8 experts,
d=64, top-2, 451,904 params, **random init, no weight download** — runs the
pinned REAP observer for 8 seeded forwards (512 rows/layer), one-pass
statistics, exact HOPE selection (E=8 within the 8–16 oracle range) plus native
REAP selection, slices two copies through upstream `reap.prune.prune`, saves,
reloads both checkpoints, and forwards again. Report checks (all must be true):
rows match, count-safe validation, one-pass both metrics, hope objective ≤ the
REAP set's objective under F, exact certificates, capability partitions
populated, both reloads valid (config/gate = 6 experts, finite logits), tiny
size bounds (100 KB–512 MB proves local tiny model), no remote download.

This is **not** agentic-quality, model-quality, or benchmark evidence. Toy
fixtures and the tiny smoke must never be reported as agent-quality proof.

## 6. Upstream patch (`patches/reap-hope.patch`)

At the pinned revision, `reap.prune` imports `reap.eval`, which imports
`lm_eval`, `evalplus`, `vllm`, `uvloop` at module scope — vllm is not
installable on macOS, which blocks even importing the slicing path. The patch
(2 files, import statements only) moves:

- `eval.py`: `lm_eval`/`make_table` into the `run_lm_eval` branch,
  `evalplus_evaluator` into the `run_evalplus` branch (the deferred pattern the
  file already uses for lcb_runner/helm/evalscope); removes the three
  never-used imports (`run_server`, `AsyncEngineArgs`, `uvloop` — the server is
  spawned via `subprocess ["vllm", "serve", …]`);
- `data.py`: `TokensPrompt` into `TYPE_CHECKING` + its two runtime branches.

No behavior changes; `git apply --check` passes against the clean pin.
`reap_adapter.ensure_reap()` refuses a clone whose `HEAD` differs from the pin.

## 7. Exact optional dependencies and install command

Verified on this workstation (macOS arm64, Python 3.14.7). The stdlib HOPE core
(`hope.py`, its CLI, and `tests/test_hope.py`) needs **none** of this — only the
integration path does.

```bash
python3 -m venv --system-site-packages .venv     # inherits system torch/transformers
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  "packaging>=24" "numpy>=2.3" "accelerate>=1.7.0" \
  "datasets>=3.6.0,<4.0.0" "python-dotenv>=1.1.0" \
  "scipy>=1.14" "scikit-learn>=1.5" "matplotlib>=3.10" "seaborn>=0.13"
# If transformers still fails with "Unable to compare versions ... found=None"
# (broken dist-info in the system site-packages), force fresh metadata into the venv:
.venv/bin/python -m pip install --force-reinstall --no-deps "packaging>=24" "numpy>=2.3"
```

Exact versions proven to work together (`.venv` resolution):

| Package | Version | Role |
|---|---|---|
| packaging | 26.3 | transformers import-version metadata (venv shadows broken system copy) |
| numpy | 2.5.3 | REAP metrics / relaxation (same shadow fix) |
| accelerate | 1.15.0 | `reap.prune` module import (`set_seed`, hooks) |
| datasets | 3.6.0 | `reap.data` module import |
| python-dotenv | 1.2.3 | `reap.args` calls `dotenv.load_dotenv()` |
| scipy | 1.18.1 | `reap.cluster` import + optional HOPE relaxation |
| scikit-learn | 1.9.1 | `reap.permute` (via `reap.merge` ← `reap.main` ← `reap.prune`) |
| matplotlib | 3.11.2 | `reap.cluster_plots` (via `reap.main`) |
| seaborn | 0.13.2 | `reap.cluster_plots` |
| torch | 2.11.0 | system, inherited |
| transformers | 4.57.6 | system, inherited (REAP pins 4.55; 4.57.6 verified working for Mixtral observer+slicing) |
| safetensors / tokenizers / tqdm / PyYAML / requests | 0.7.0 / 0.22.2 / 4.67.3 / 6.0.3 / 2.33.1 | system, inherited |

Upstream pyproject pins `torch==2.7.1`, `transformers==4.55.0`, `vllm==0.10.0`,
`lm-eval`, `evalplus` for the full eval-serving stack; those heavy eval deps are
**not** needed for observation/slicing after the patch and are not installed
here. No weights are downloaded; no API spend; telemetry disabled via config.

## 8. Commands (for Main)

```bash
# 1) Toy suite — closed-form pathological cases vs analytic optima (stdlib only)
PYTHONPATH=src python3 -m mimo_halo.pruning.hope toys --config configs/hope.json

# 2) Fixture statistics -> exact selection -> merge (stdlib only)
PYTHONPATH=src python3 -m mimo_halo.pruning.hope observe-fixture \
  --config configs/hope.json --out scratch/hope-fixture
PYTHONPATH=src python3 -m mimo_halo.pruning.hope select \
  --config configs/hope.json --stats scratch/hope-fixture/block_interaction.json \
  --budget 4 --out scratch/hope-selection.json
PYTHONPATH=src python3 -m mimo_halo.pruning.hope select \
  --config configs/hope.json --stats scratch/hope-fixture/block_interaction.json \
  --budget 4 --capability implementation

# 3) Unit tests (stdlib core; relaxation test skips without scipy)
python3 -m unittest tests.test_hope -v          # no scipy: missing-dep branch
.venv/bin/python -m unittest tests.test_hope -v # with scipy: relaxation branch

# 4) Actual tiny Transformers Mixtral integration through pinned REAP
git -C upstreams/reap rev-parse HEAD   # must print 1970473c51ca3caeb98c10392f15b3a08a672974
git -C upstreams/reap apply "$PWD/patches/reap-hope.patch"
PYTHONPATH=src .venv/bin/python -m mimo_halo.pruning.reap_adapter tiny-mixtral-smoke \
  --config configs/hope.json --out scratch/hope-tiny-smoke
git -C upstreams/reap checkout -- src/reap       # restore clone
git -C upstreams/reap status --porcelain         # must be empty; HEAD must still be the pin
```

Smoke output lands in `scratch/hope-tiny-smoke/report.json` (plus per-layer
`stats/hope_layer*.json` and the two sliced checkpoints under `models/`).

## 9. Explicitly unsupported / blocked (not completed claims)

- **Full MiMo-V2.6 integration**: pinned REAP has **no `MODEL_ATTRS` entry** for
  MiMo (`model_util.MODEL_ATTRS` registers Qwen3MoE, Llama4, Mixtral,
  DeepseekV2, Ernie4_5, gpt-oss-20b, Glm4Moe only). Adding one requires verified
  checkpoint metadata (expert/layer/top-k counts, gate formula) plus local
  weights; the adapter fails closed with the registered list instead of
  approximating. The sigmoid+correction-bias caveat in §4 applies to MiMo:
  the corrected seam formula is unit-tested, but **MiMo (and every other
  correction-bias router) observer integration remains open in #21** — no
  model-level bias observation runs until a verified registration lands.
- **Full-model / gpt-oss observation on this Mac**: blocked on hardware/backend,
  not on HOPE code — verified that transformers v4.56.1 MXFP4 quantization
  dequantizes to BF16 (~42 GB) on CPU/MPS and REAP fully loads the model before
  layer streaming, which exceeds the 48 GiB RAM + 26.6 GiB external budget.
  No weights were downloaded.
- **Paper fidelity limits**: authors' solver and Appendix D hyperparameters are
  unreleased; the relaxation here matches the published recipe only. Calibration
  at paper scale (N ≥ 10k prompts, Evol-CodeAlpaca/SWE-Bench) and any
  agentic-quality evaluation are separate, gated work items.
- **Quality claims**: none. Nothing in this deliverable measures KL,
  reconstruction, or task success.
