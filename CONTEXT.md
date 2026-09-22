# CONTEXT — domain glossary

Single context for `mimo-halo-lab`. This file is **vocabulary only**: canonical terms and their meanings, no implementation details. Design and decisions live in `docs/`; use these words verbatim in issue titles, hypotheses, and test names — do not drift to synonyms.

## NVMe state tier

Source of truth: [`docs/nvme-state-tier.md`](docs/nvme-state-tier.md) and the approved contract behind it.

- **Logical session** — one coding agent's durable conversation identity, independent of any server slot.
- **Hot slot** — a physical inference slot with resident target-model sequence state; a logical session owns it while hot.
- **Target snapshot** — a versioned, checksummed complete continuation state for one logical-session generation. No claim that sampler/speculative state is required for basic target continuation.
- **Prefix snapshot** — immutable reusable inference state for an exact token prefix under a compatible model/runtime configuration.
- **HOT** — UMA-resident state, active or idle.
- **WARM** — a recent durable snapshot on NVMe with a preferred retention window.
- **COLD** — an older still-restorable NVMe snapshot. COLD is not a cache miss.
- **Cold miss** — no compatible valid reusable state remains; an explicit cold-prefill path is required and its true physical token cost must be reported.
- **Logical input tokens** — the complete context tokens logically presented for a measured request (denominator always stated).
- **Physical prefill tokens** — tokens actually sent through target-model prefill computation; never tokenized/counted/restored tokens.
- **Snapshot generation** — a monotonically identified semantic state revision, used to fence late asynchronous work and avoid saving unchanged state.
- **Warm/cold retention class** — a metadata class over the *same* immutable blob, orthogonal to lifecycle state; promotion/demotion never rewrites payload.

Invariant to keep in mind everywhere: restoring a context must preserve actual prefix reuse — a 100K history plus a 4K suffix physically prefills ~4K, never ~104K. An HTTP 200 restore is not acceptance evidence.

## Compression

Source of truth: [`docs/hope-objective.md`](docs/hope-objective.md), [`docs/pruning-integrity.md`](docs/pruning-integrity.md), [`docs/evaluation-methodology.md`](docs/evaluation-methodology.md), [`docs/artifact-provenance.md`](docs/artifact-provenance.md).

- **REAP** — first-order expert-importance scoring and pruning: per-expert saliency from routed-token activations times router weights; the upstream pruning baseline (`CerebrasResearch/reap`, pinned).
- **HOPE** — higher-order (pairwise interaction) pruning objective from arXiv:2609.18916; selects experts by a quadratic form over conditional co-activation statistics, not first-order scores alone.
- **Expert map / prune map** — the deterministic per-layer record of which experts are retained and which are pruned (`kind: "prune_map"`); may be a structural placeholder or a REAP/HOPE quality map, always labelled as which.
- **Quality map** — a prune map derived from observed routing/activation statistics; a structural (shape-only) map is explicitly *not* a quality map.
- **Survivor** — an expert retained after pruning (e.g. 176 or 160 of 256).
- **Calibration** — a representative token pass that measures activations/distributions for observation, imatrix, or quant planning; never run on the golden held-out set.
- **Recovery** — post-prune repair (router-first, then evidence-backed adapters) to restore lost capability before final evaluation.
- **Imatrix** — importance matrix calibration data used to guide quantization.
- **Quant plan** — the measured per-tensor precision assignment under an exact byte budget; compared in exact bytes (DecimalGB ≠ GiB).
- **Golden held-out set** — 50–200 frozen executable real-repo tasks used only for final confirmation, never for calibration, training, or model selection.
- **Retention (quality)** — candidate task-success divided by full-reference task-success on the same frozen population, with absolute rates and paired uncertainty also reported.
- **Catastrophe** — a category of destructive/dishonest task failure tracked separately from mean pass rate.

## Workflow vocabulary

- **Parent spec** — a `ready-for-human` epic excluded from AFK implementation selection; work happens in its child Issues.
- **Tracer** — a smallest end-to-end Issue slice that proves a seam before optimization.
- **Seam** — the highest practical boundary at which behavior is tested (server continuation, store commit/load, policy decisions, workload benchmark).
- **P0 / P1 / P2** — priority tiers: executable before hardware / hardware-arrival frontier / selected-candidate-to-release.
