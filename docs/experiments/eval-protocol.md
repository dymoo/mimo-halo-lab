# Evaluation protocol for the expert-pruning / precision compression sweep

Status: **design**. No model runs are claimed by this document. Machine-readable
form: `configs/experiments/eval-protocol.json`.

Two jobs:

1. **Comparability axis** — reproduce the tacodevs REAP25 evaluation
   methodology so our REAP25 candidate is directly comparable to their
   published REAP25 mixed-3bit result, with every silent spot in their public
   metadata annotated "unspecified there -> our choice".
2. **Our axis** — a coding-agent-heavy evaluation built from our own
   partitioned trace material (partition names and counts only, never
   content), plus a four-stage design so bad candidates die at the cheapest
   stage that can kill them.

This document **extends by reference** the statistical contract in
[`docs/evaluation-methodology.md`](../evaluation-methodology.md) (data
contract §1, statistics §3, evaluation-levels ladder §4, golden §5, negative
controls and release gates §6) and the partitions/closed taxonomies in
[`docs/datasets.md`](../datasets.md). Neither document is modified by this
protocol; their closed enums are used as-is (re-implementations are bugs).
Task-outcome statistics reuse
[`src/mimo_halo/evaluation/paired.py`](../../src/mimo_halo/evaluation/paired.py)
against [`schemas/task-outcome.schema.json`](../../schemas/task-outcome.schema.json).

**Non-goals:** no model execution in this design phase, no private content in
any output of this protocol, no code in this slice.

---

## 1. Candidates under evaluation

Budget question: at the same ~90 GiB resident weight budget, retain more
experts quantized harder, or prune more low-salience experts while keeping
survivors near their native QAT MXFP4 representation. All candidates are
built from Xiaomi's **original** MiMo-V2.6-Flash weights (revision
`5711b268169967567844e1e560e8a3966da959b1`); the tacodevs MLX checkpoint and
its published sensitivity work are **prior information only**, never a build
source.

| Candidate | REAPxx = % experts pruned per layer | Retained / 256 per layer |
| --- | --- | --- |
| REAP25 | 25% | 192 |
| REAP30 | 30% | 179 |
| REAP35 | 35% | 166 |
| REAP40 | 40% | 154 |
| REAP45 | 45% | 141 |

Exact tensor recipes, byte totals and budget matching live in the sweep
design/build artifacts; this protocol only says how each built candidate is
**measured**.

---

## 2. Tacodevs methodology transcription (metric by metric)

Sources (all under the pinned archive
`manifests/baselines/tacodevs-reap25/`, revision
`95d80053eb22112f2acc4244330eb67b4f534c94`):

- **[CARD]** `manifests/baselines/tacodevs-reap25/raw/README.md`
  (model card: "What was done", "Calibration", "Quality" sections)
- **[EVAL]** `manifests/baselines/tacodevs-reap25/raw/compression_eval.json`
- **[NORM]** `manifests/baselines/tacodevs-reap25/normalized.json` ->
  `published_evaluation` (deterministic parse of the archived bytes)
- **[DOC]** `docs/baselines/tacodevs-reap25.md`

### 2.1 Corpus and calibration protocol

| Item | As published by tacodevs | Source | Unspecified there -> our choice |
| --- | --- | --- | --- |
| Calibration corpus | 256 sequences x 2048 tokens (= 524,288 tokens) from a five-family mix: evol-codealpaca, Mixture-of-Thoughts, SWE-smith trajectories, glaive function calling, UltraChat; rendered with the model's chat template | [CARD] Calibration sentence | Per-family sample counts, proportions, and sample identities are **unspecified** -> **our choice:** fixed recipe at the same sizes (256x2048), uniform per-family sampling, every sampled item hashed into a private manifest, family composition recorded as counts |
| Held-out (eval) corpus | "disjoint samples from the same mix"; 31 x 2048 nominal; `tokens: 63457` in the eval JSON | [CARD] Calibration sentence; [EVAL] `tokens` | The 31-token gap between 31x2048 = 63,488 nominal and 63,457 scored is **unexplained** -> **our choice:** record the exact scored-token count; never round it to the nominal size |
| Held-out domain label | "held-out agentic/coding mix" | [CARD] Quality table header | Domain tagging method **unspecified** -> **our choice:** every held-out item carries its source family tag; we report per-family metrics alongside the aggregate |
| Disjointness | "disjoint samples from the same mix" (claim, not evidence) | [CARD] | Verification method **unspecified** -> **our choice:** disjointness is proven by manifest hash: the eval holdout must be disjoint from every candidate's build-calibration corpus, checked per candidate |
| Chat template / tokenizer | archived alongside (`chat_template.jinja`, `tokenizer_config.json`) | archive listing [DOC] | Version pinning of template vs model revision **unspecified** -> **our choice:** template and tokenizer pinned to official revision `5711b268169967567844e1e560e8a3966da959b1` and recorded by hash in the run manifest |
| Reference model | "Original (MXFP4/FP8)" = official Xiaomi checkpoint | [CARD] Quality table | Logit computation precision/hardware **unspecified** -> **our choice:** original Xiaomi weights at stored precision, FP32 logit accumulation, hardware + runtime recorded per run |
| Seeds / batching | not mentioned | - | **our choice:** deterministic seeds recorded in the run manifest; scoring is teacher-forced and order-independent |

### 2.2 Held-out distribution metrics

| Metric | Published value | Unit | Published definition | Source | Unspecified there -> our choice |
| --- | --- | --- | --- | --- | --- |
| Perplexity, original | 9.271 | perplexity (dimensionless) | held-out mix, 31x2048 | [CARD] table | Aggregation (global token-mean vs per-sequence mean) **unspecified** -> **our choice:** primary = exp(global mean NLL in nats over all scored tokens); per-sequence mean recorded as a secondary number; mean NLL (nats/token) always reported next to PPL |
| Perplexity, candidate | 9.529 card / 9.528511047363281 exact | same | same | [CARD]; [EVAL] `ppl`; cross-checked in [NORM] `distribution.cross_check.agrees = true` | same as above |
| KL(original \|\| candidate), mean per token | 0.8315 card / 0.8314559955519677 exact | nats/token (presumed) | "KL(original ‖ this), mean per token" | [CARD]; [EVAL] `kl_base_to_quant` | Log base, vocabulary scope, temperature **unspecified** -> **our choice:** natural log, full vocabulary, both models at temperature 1, direction KL(original ‖ candidate), mean over all scored tokens |
| Top-1 next-token agreement | 79.5% card / 0.7949004837921743 exact | percent of scored positions | "Top-1 next-token agreement with original" | [CARD]; [EVAL] `top1_agree` | Tie handling, position set **unspecified** -> **our choice:** argmax over the full vocabulary with lowest-index tie-break; every position scored (labels shifted by one, no prompt/loss masking differences between models); both models score exactly the same positions |
| Eval-config record | `alloc: alloc_k64rel.json`, `attn_bits: 8`, `embed_bits: 16`, `lm_bits: 8`, `gptq: true` | config fields | eval JSON header | [EVAL] | **Inconsistency, recorded:** card step 4 says embeddings are 8-bit affine g64, the eval JSON says `embed_bits: 16`; also the eval JSON names `alloc_k64rel.json` while the archived allocation is `compression_alloc.json`. Published tensor-recipe metadata is internally inconsistent -> **our choice:** we take **no** tensor recipe from tacodevs metadata; our recipes come only from our own builder, and we record this inconsistency rather than resolving it |

The card also claims: "Any perturbation of this MoE (even 8-bit attention)
sits at KL≈0.4 on foreign text because top-8 routing flips, so only
on-policy deltas are comparable across variants." Transcribed as a **publisher
claim** ([CARD] prose), not a measured fact of ours; it motivates weighting
the on-policy axis at least as heavily as held-out KL.

### 2.3 On-policy metrics (40 API responses from the real MiMo-V2.6-Flash; assistant tokens only)

Transcribed from [CARD] On-policy table (normalized in [NORM]
`on_policy.table`):

| group | tokens | original NLL | candidate NLL | Δ | KL | top-1 agree |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | 24136 | 0.530 | 0.609 | +0.079 | 0.105 | 90.4% |
| code | 10627 | 0.511 | 0.589 | +0.078 | 0.106 | 90.8% |
| agent | 4786 | 0.695 | 0.779 | +0.084 | 0.126 | 88.5% |
| reasoning | 3815 | 0.313 | 0.358 | +0.045 | 0.052 | 94.2% |
| general | 4908 | 0.580 | 0.682 | +0.102 | 0.125 | 88.5% |

(Consistency check we performed: 10627 + 4786 + 3815 + 4908 = 24136 = ALL.)

| Item | As published | Source | Unspecified there -> our choice |
| --- | --- | --- | --- |
| Response sourcing | 40 responses sampled from the real MiMo-V2.6-Flash via API | [CARD] | Prompt set, sampling parameters, length cap **unspecified** -> **our choice:** fixed prompt manifest, generation from the **original official weights** (not an API of unknown revision), temperature / top-p / seed / max-tokens recorded per run; at least 40 responses to match their count, exact counts published |
| Scored tokens | assistant tokens only | [CARD] | Exact masking rule **unspecified** -> **our choice:** score only assistant-span tokens; prompt, tool results and system text never scored; mask identical for both models |
| NLL unit | 0.530 / 0.609 style values | [CARD] | Log base **unspecified** -> **our choice:** nats per assistant token (natural log); Δ = candidate − original, nats/token |
| KL column | 0.105 … 0.052 | [CARD] | Same as held-out KL: log base/vocab/temp unspecified -> **our choice:** KL(original ‖ candidate) in nats/token, full vocab, T=1 |
| Top-1 agree column | 90.4% … 94.2% | [CARD] | Same as held-out top-1 -> **our choice:** identical definition on both axes so the two are cross-checkable |
| Slice membership (code / agent / reasoning / general) | group names only; no labeling method | [CARD] | **Unspecified** -> **our choice:** taxonomy-derived groups from `docs/datasets.md` (mapping in §3.3); "general" = residual scored assistant tokens not covered by any labeled episode. Direct numeric comparability of the slices is therefore **in-spirit, not definition-identical**; the ALL row is the definition-identical comparison |
| Task success | `task_success: null`; `independently_reproduced: false` | [NORM]; [DOC] | They published **no** benchmark or task-success scores -> **our choice:** stages 3–4 of our protocol have no tacodevs counterpart; we never present their distribution numbers as capability evidence and never compare our benchmark scores against a number they did not publish |

---

## 3. Our second axis: coding-agent-heavy evaluation

All corpora here are **names and counts only**; raw trace content, prompts,
repo identifiers and paths never enter this document or any Git-tracked
output. Partition counts are from
`manifests/datasets/discovery-summary.json` (2525 task groups assigned;
58 quarantined identity failures).

### 3.1 Corpora slices

| Slice | Partition / source | Counts | Role |
| --- | --- | --- | --- |
| `stage1-holdout-public-mix` | public five-family mix (same families as tacodevs, see §2.1) | target 31 x 2048 sequences/tokens; exact scored count recorded at build | Stage-1 held-out PPL/KL/top-1, size-comparable to the tacodevs protocol |
| `stage2-validation-trajectories` | `validation` partition | 258 task groups | Stage-2 representative code / reasoning / agent trajectories; supplies contexts and slice labels |
| `stage2-torture-longcontext` | `torture` partition | 117 task groups | Long-context degradation slices (§3.4); buckets with no material are reported empty, never invented |
| `stage3-validation-shorttasks` | derived from `validation` partition | built by the stage-3 runner; counts recorded then | Short structured tasks (levels ladder level 4) and reduced benchmark prompts |
| `stage4-golden` | `golden` partition (hash-reserved) | reserved 235, eligible 0, `bank_status: not_ready`, bank bounds 50–200 | Final-confirmation paired task success only; **currently unavailable** until reserved groups gain registered oracles |
| excluded | `pruning` 922, `quant` 486, `recovery` 507, quarantined 58 | - | Training-side or quarantined; never evaluation input |

Golden structure facts (naming only): reserved groups stay reserved and never
redistribute; only hash-reserved **and** fully eligible groups freeze into the
executable bank; golden never participates in calibration or method selection
(`docs/datasets.md`; `docs/evaluation-methodology.md` §5).

### 3.2 On-policy NLL over agent trajectories

Definition (mirrors the tacodevs on-policy shape, fixed to our material):

1. Take contexts from `validation`-partition trajectory prefixes (torture
   partition for the long-context slices).
2. Generate continuations from the **original official MiMo weights** under a
   recorded sampling config; these reference-generated assistant spans are the
   scored targets. (Trace-supplied assistant text comes from other
   models/harnesses — it is **off-policy** and is never used as a scored
   target; it supplies context and labels only.)
3. Score every assistant token under original and candidate with identical
   masks; report per group and per context bucket: original NLL, candidate
   NLL, Δ (nats/token), KL(original ‖ candidate) (nats/token), top-1 agreement
   (%).

### 3.3 Agreement slice groups (taxonomy-derived)

Group membership is a deterministic function of the episode
`primary_capability` label (closed 22-tag set from `docs/datasets.md`;
all 22 tags are assigned to exactly one group):

| Group | `primary_capability` values |
| --- | --- |
| **code** (code-specific agreement) | `implementation`, `refactoring`, `code_review`, `debugging`, `compiler_interpretation`, `test_interpretation`, `frontend`, `backend`, `systems`, `database` |
| **reasoning** (reasoning-specific agreement) | `planning`, `architecture`, `dependency_reasoning`, `concurrency`, `long_context` |
| **agent** (agent/tool-use agreement) | `repo_exploration`, `tool_use`, `shell`, `git`, `verification`, `recovery`, `build_tooling` |
| **general** (residual) | scored assistant tokens not covered by any labeled episode |

The **ALL** row (union of the four groups) is the tacodevs-comparable row;
the three named groups are our-axis slices that happen to share their group
names. Secondary slices recorded alongside: difficulty (`D0`–`D4`) and
language enums, both closed sets from `docs/datasets.md`.

### 3.4 Long-context degradation slices

Buckets are the closed `context_bucket` enum of
`src/mimo_halo/evaluation/paired.py` (K = 1024), reused verbatim:

`<8K`, `8-16K`, `16-32K`, `32-64K`, `64-128K`, `128K+`

Per bucket, over segments whose prefix length lands in the bucket:
ΔNLL (nats/token) and Δtop-1 agreement (pp) vs original, plus a degradation
slope ΔNLL(128K+) − ΔNLL(<8K). Every bucket publishes its scored-token
count; buckets below a readable token count are marked small-N/inconclusive,
never zero-filled. The `<8K` bucket is the "8K-and-below" bucket of the enum
(the assignment's "8K" slice).

### 3.5 Pathological-generation detection protocol

**What counts** (recorded enum; category names align with the
`catastrophic` semantics and `failure_category` vocabulary of
`docs/evaluation-methodology.md` §1 where an alias exists):

1. `degenerate_repetition` — looping n-grams or a tool-call loop that fails
   the recovery window (alias: `runtime_corruption` when output collapses).
2. `malformed_tool_invocation` — unparseable tool-call arguments or a tool
   name outside the allowed set (alias: `tool`).
3. `goal_loss` — off-task drift, work contradicting the stated task
   (alias: `context_loss`).
4. `fake_verification` — claiming tests/build passed while the recorded tool
   output says otherwise (alias: `test`).
5. `destructive_shell` — malformed/destructive command outside task scope
   (alias: `tool`).
6. `uncontrolled_rewrite` — wholesale replacement unrelated to the request
   (alias: `implementation`).
7. `context_confusion` — citing files, edits or results that do not exist
   (alias: `context_loss`).
8. `termination_pathology` — empty output, immediate EOS against a
   non-trivial prompt, or runaway to the token cap with no stop (alias:
   `timeout` / `runtime_corruption`).

**Who judges:** two tiers. (a) Deterministic automated detectors
(structural rules: repetition factor, tool-call parse failures,
claim-vs-tool-output contradiction, scope checks) flag candidates; nothing is
judged by a language model. (b) A human adjudicator reviews flags **blind to
`model_id`**, confirming or refuting per incident. The counted pathology rate
is adjudicated positives; refuted flags are still counted as "flagged" for
detector calibration.

**How recorded:** per-incident records (category, bucket, group, detector,
adjudication) live in private run artifacts under `$MIMO_LAB`
(`$MIMO_LAB/eval/pathology/…`) — never in Git. Public outputs carry counts
and rates only (incidents per 100 scored generations, by category). Stage-4
runner outcomes additionally flow into the task-outcome JSONL
(`catastrophic: true` + `failure_category`) so the paired tool's catastrophe
section sees them.

---

## 4. Staged evaluation design

Mapping to the levels ladder of `docs/evaluation-methodology.md` §4 is by
reference: stage 1 = levels 1–3 controls; stage 2 = levels 2–3 over agent
material plus level-5-style observation; stage 3 = levels 4–5 reduced;
stage 4 = levels 6–7 plus the golden decision metric (§5 of that document).

**Kill criteria are placeholders.** Every `δ` below is marked
**CALIBRATED AFTER STAGE-1 SPREAD**: after all candidates produce stage-1
numbers, the inter-candidate spread sets each threshold, thresholds are
written back here and into the JSON config, and only then does culling become
binding. The one exception is stage 1's artifact gate, which is absolute and
needs no calibration.

| Stage | Inputs | Metrics | Kill criterion | Expected compute class |
| --- | --- | --- | --- | --- |
| **1 — distribution controls** | artifact inventory of the built candidate; `stage1-holdout-public-mix` (31×2048 target, manifest-hashed, disjoint from build calibration); ≥40 reference-sampled responses (on-policy set) | held-out PPL + ΔPPL vs original; KL(original‖cand) nats/token; top-1 agreement %; on-policy ALL: NLL, ΔNLL, KL, top-1 %; artifact gate: tensor inventory, dtype changes, NaN/Inf, reload round-trip | **Absolute (not calibrated):** artifact-gate failure, NaN/Inf, or unloadable weights → immediate kill. **Spread-calibrated:** `δ_ppl`, `δ_kl`, `δ_top1` on (ΔPPL, KL, top-1) vs the best candidate → **CALIBRATED AFTER STAGE-1 SPREAD**; provisional rule: kill candidates strictly dominated on all three by another candidate at equal-or-larger budget | `scoring`: teacher-forced passes over ~0.6M + 63K-token budgets and the on-policy set; single GPU pass, hours per candidate |
| **2 — representative trajectories** | `stage2-validation-trajectories` (258 groups), `stage2-torture-longcontext` (117 groups); stage-1 survivors | on-policy NLL/ΔNLL/KL/top-1 per slice group (code / reasoning / agent / general / ALL); long-context ΔNLL + Δagreement per `<8K…128K+` bucket and slope; pathology flags + adjudicated rate by category; difficulty/language secondary slices | `δ_slice`, `δ_ctx`, `δ_path` → **CALIBRATED AFTER STAGE-1 SPREAD**; provisional rule: kill on (a) any primary slice (code/reasoning/agent) dominated by another candidate on both ΔNLL and top-1, (b) degradation slope worse than best candidate + `δ_ctx`, or (c) adjudicated pathology rate above reference + `δ_path` | `scoring + bounded rollout`: reference generation over trajectory prefixes plus teacher-forced scoring; GPU-hours per candidate |
| **3 — reduced benchmark suite** | stage-2 survivors; `stage3-validation-shorttasks`; candidate suites (our choice, all small/standard): **evalplus HumanEval+ and MBPP+** (pass@1), **LiveCodeBench via `lcb_runner`** at a pinned release window (time-sliced subset), one **`lm-eval`** small reasoning subset as the reasoning control, plus level-5 protocol negative controls (`docs/evaluation-methodology.md` §6). Suite names come from the upstream REAP pipeline's own eval branches recorded in `docs/hope-objective.md` §6 (`lm_eval`, `evalplus`, `lcb_runner`) — they published no benchmark scores ([DOC]) | pass@1 (and suite-native metrics) per benchmark with paired seeds; negative-control pass rate; short structured task success paired vs reference (task-outcome JSONL → `paired.py`: task-cluster bootstrap CIs, wins/losses/ties, exact McNemar where seeds allow) | `δ_bench` → **CALIBRATED AFTER STAGE-1 SPREAD** (spread of stage-1 metrics informs the provisional band until stage-3 numbers exist); provisional rule: kill a candidate whose reduced-suite retention sits below the sweep's provisional floor or whose paired CI is strictly below parity vs the reference; release-gate shapes (§6 of the methodology: e.g. short coding retention ≥93%) apply at release time, not as stage-3 auto-pass | `benchmark_reduced`: hours to ~1 day per candidate on one machine; still cheap relative to stage 4 |
| **4 — full benchmarks, Pareto-frontier only** | Pareto-frontier candidates from stages 1–3 (quality vs bytes, throughput excluded from the frontier); `stage4-golden` frozen bank (**currently `not_ready`: 0 eligible of 235 reserved — reported unavailable, never substituted**); full-window agentic suites (e.g. SWE-bench-class public-repo agent suite — our choice; no tacodevs counterpart exists); full LiveCodeBench window; extended negative controls; serving diagnostics | golden paired task success (absolute rates, task-cluster 95% CIs, ratio with CI, wins/losses/ties, McNemar where applicable — methodology §5, via `paired.py`); catastrophes per pair; full benchmark scores; **PRE-OPTIMISATION** diagnostics: PP8K/PP32K/PP64K/PP128K prompt-processing, C1/C4/C8 decode concurrency (commands: `docs/hardware-bringup.md` §G), theoretical bytes/token, active-weight bandwidth | **No automatic kill.** Decisions are Pareto-frontier + golden paired outcome; CIs straddling parity are inconclusive by construction (methodology §5); no promotion rests on ±1 task; throughput never removes a candidate from the quality frontier | `benchmark_full + serving`: multi-day, frontier candidates only |

Bad candidates die at stage 1 for artifact/numerical damage, at stage 2 for
slice collapse or pathology, at stage 3 for capability loss; only credible
candidates ever pay stage-4 prices.

---

## 5. Comparability matrix

Axis labels: `tacodevs-comparable` (direct numeric comparison against their
published REAP25 numbers is meaningful under §2) | `our-axis` (no published
counterpart).

| Metric (unit) | Axis | Note |
| --- | --- | --- |
| Held-out PPL / ΔPPL (dimensionless) | tacodevs-comparable | replicate corpus sizes; their aggregation method unspecified → our fixed choice (§2.2) |
| Held-out KL(original‖cand) (nats/token) | tacodevs-comparable | log base unspecified there; comparison valid if we state ours |
| Held-out top-1 agreement (%) | tacodevs-comparable | definition fixed by us (§2.2); ALL positions |
| On-policy ALL: NLL, ΔNLL, KL, top-1 | tacodevs-comparable | assistant tokens only; nats/token; ≥40 responses |
| On-policy code/agent/reasoning/general slices (%) | tacodevs-comparable (in spirit) | their labeling method unpublished → our taxonomy groups differ; ALL row is the definition-identical anchor |
| Slice code/reasoning/agent agreement — ours (%) | our-axis | taxonomy-derived (§3.3) |
| On-policy NLL over agent trajectories (nats/token) | our-axis | their on-policy set is API-sampled, not trajectory-derived |
| Long-context ΔNLL/Δagreement by `<8K…128K+` (nats/token, pp) | our-axis | tacodevs published no context analysis |
| Pathological-generation rate (incidents/100) | our-axis | tacodevs published none |
| Short-task / reduced-suite retention (%) | our-axis | they published no benchmark scores; never compare against a number they didn't publish |
| Golden paired task success (rate, CI, ratio) | our-axis | their `task_success: null`; decisive metric per methodology §5 |
| Negative-control pass rate (%) | our-axis | methodology §6 |
| PP8K–PP128K, C1/C4/C8 decode (tok/s) | our-axis — **PRE-OPTIMISATION** | implementation artifact, never a quality signal |
| Theoretical bytes/token (bytes); active-weight bandwidth (GB/s) | our-axis | arithmetic from the exact inventory (`manifests/models/mimo-v2.6-flash-rl/memory-report.json`), not a measurement |

---

## 6. Claims and prohibitions

- Distribution metrics (PPL, KL, agreement, NLL) are **controls**, never task
  success; a high-agreement candidate with a worse golden paired outcome
  loses (methodology §5).
- **Quality decides the winner.** All throughput numbers are
  PRE-OPTIMISATION diagnostics: a high-quality candidate is never rejected
  for poor current throughput, and a worse candidate is never selected for a
  lucky kernel.
- Published tacodevs numbers are publisher-reported and
  `independently_reproduced: false`; reproducing their *method* is our
  comparability work, not an endorsement of their values.
- Golden never enters calibration or method selection; the golden bank is
  reported unavailable (`not_ready`) until reserved+eligible ≥ 50.
- No raw trace content, prompts, repo identifiers, secrets, or absolute local
  paths in any Git-tracked output of this protocol; private artifacts stay
  under `$MIMO_LAB`; public outputs carry counts, hashes and rates only.
- Missing measurements stay `null`/unavailable — never zero, never inferred.
- `δ` thresholds marked **CALIBRATED AFTER STAGE-1 SPREAD** are non-binding
  until calibrated and written back into
  `configs/experiments/eval-protocol.json`.
