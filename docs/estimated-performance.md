# Estimated performance (PRE-HARDWARE — these are estimates, not measurements)

Status: **no Strix Halo measurement exists yet.** Every number on this page is a
design-time estimate from the project spec and the memory-matched design
(`docs/experiments/memory-matched.md`). They exist to size the target and to be
falsified. When the hardware arrives, measured values replace them and every
measured number carries a `PRE-OPTIMISATION` or post-optimisation label.

## Target decode estimates (spec planning numbers)

These are TARGETS to design against, not benchmark results:

| Regime | Estimate (tok/s) | Note |
| --- | --- | --- |
| C1 target-only | 23–28 | single interactive agent stream |
| C1 + good MTP | 30–40 | only if speculative decoding earns its RAM |
| C4 aggregate | 45–60 | four concurrent agent streams |
| C8 aggregate | 65–80 | the production planning regime |

Planning budget: **70 tok/s at C8** until measured. Also targeted for the final
product: ~98–105 GiB resident model weights on the 128 GB Halo.

## Theoretical bytes/token method (kernel-independent ceiling)

Decode is bandwidth-bound at these sizes. The ceiling is computed, not measured:

1. Active weight bytes per decode token = sum over 47 MoE layers of
   (top-k 8 × per-expert stored bytes for that candidate's recipe) + per-layer
   router/dense contributions + the non-expert weights amortized as documented
   in `manifests/models/mimo-v2.6-flash-rl/memory-report.json`.
2. Exact per-class byte counts for every candidate recipe live in
   `configs/experiments/compression-sweep.json` (each candidate records
   `active_weight_bandwidth` and per-class `bytes_per_unit` arithmetic).
3. Ceiling tok/s ≈ sustainable memory bandwidth ÷ active bytes/token. The
   Strix Halo's LPDDR5X-8000 quad-channel theoretical peak is ~256 GB/s;
   SUSTAINABLE bandwidth is always lower and is itself a measurement to make.

This calculation is deliberately independent of kernel quality: it bounds what
any runtime could achieve on the data movement alone. The gap between this
ceiling and measured throughput is exactly what the runtime optimisation
campaign (post-quality-selection) tries to close.

## Rules that govern these numbers

- Throughput NEVER selects the compression-quality winner. Kernel/datatype
  performance artifacts are implementation details; the winner is chosen on
  quality at matched ~90 GiB resident size (ticket #88 is the only gate).
- A throughput drop at one concurrency level is profiled as a possible
  kernel/dispatch artifact before being called a hardware limit.
- Concurrency scaling (C1/C2/C4/C6/C8/C12/C16) stays shelved until the quality
  baseline exists, then runs per serious candidate as PRE-OPTIMISATION
  diagnostics with the two sweet spots (hardware throughput plateau; interactive
  per-stream comfort) as its outputs.

## What replaces this page

Per-candidate measured records (schema:
`schemas/compression-candidate.schema.json`) carry PP, C1 decode, C4, C8,
TTFT/inter-token latency, RAM, plus theoretical bytes/token and
active-weight bandwidth — with labels separating pre-optimisation diagnostics
from post-optimisation results. Nothing on this page is ever promoted to
evidence; evidence comes only from recorded runs.
