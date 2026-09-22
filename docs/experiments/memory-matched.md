# Memory-matched precision comparison: tacodevs REAP25 reference vs REAP160 / HOPE160 / HOPE176

Status: **design**. Nothing here has been run on hardware; no inference
functionality is claimed by this document. Every measurement slot is either a
published number with a source pointer or explicitly unavailable.

Source issues: [#50 baseline archive/importer](https://github.com/dymoo/mimo-halo-lab/issues/50),
[#51 selection disagreement analysis](https://github.com/dymoo/mimo-halo-lab/issues/51),
[#52 this experiment design](https://github.com/dymoo/mimo-halo-lab/issues/52).
Machine-readable form: `configs/experiments/memory-matched.json`.

## Purpose

Decide, under one 128 GB Strix Halo memory, whether the published tacodevs
REAP25 baseline (arm A) or our own candidates (arms B/C/D) better serves
long-horizon coding agent quality at equal weight-byte budgets — and record
the published precision allocation of A as the sensitivity-prioritization
reference for our quant planner.

## Unit discipline (mandatory)

All budget comparisons happen on **exact integer bytes**. Decimal GB is
10^9 bytes; GiB is 2^30 bytes. The units are never mixed.

| Quantity | Bytes | Decimal GB | GiB |
| --- | --- | --- | --- |
| **A exact text weights (archived tree shard sum)** | **100,431,428,728** | **100.431428728** | **93.53405677527189** |
| A published card label "100.4 GB" (rounded; informational only) | 100,400,000,000 | 100.4 | 93.504786 |
| User's approximate 100 GB control | 100,000,000,000 | 100.0 | 93.132257 |
| Project target lower bound (100 GiB) | 107,374,182,400 | 107.374182 | 100.0 |
| Original production target (103 GiB) | 110,595,407,872 | 110.595408 | 103.0 |
| A published auxiliary weights | 7,300,000,000 | 7.3 | 6.798655 |

Consequences, stated once and not negotiable:

- **The exact A control cap is 100,431,428,728 bytes =
  100.431428728 decimal GB = 93.53405677527189 GiB (~93.53 GiB)** — the
  sum of the 22 top-level `model-*-of-00022.safetensors` text shards in the
  archived tree (`manifests/baselines/tacodevs-reap25/evidence/tree.json`,
  revision `95d80053eb22112f2acc4244330eb67b4f534c94`). All fit
  comparisons use this figure.
- **The model-card "100.4 GB" is a rounded label only** (100,400,000,000
  bytes read as decimal GB = 93.504786 GiB). It differs from the exact
  archived tree sum by 31,428,728 bytes. Never call the rounded card cap
  the exact baseline size.
- **100.4 GB is not 100.4 GiB.** 100.4 GiB would be 107,803,679,130 bytes
  (107.803679 decimal GB).
- **The original 103 GiB production target is a distinct cap** from A's
  exact archived tree sum (100,431,428,728 bytes = 93.53405677527189 GiB).
  103 GiB = 110.6 decimal GB.
- The existing project weight target 100–105 GiB (107.37–112.74 decimal GB)
  sits **above** A's reported size, so A does not fill the project budget.
- Arms are called "memory matched" only under the exact-byte fit rule below,
  never by prose-unit similarity.

## Arms

| Arm | Definition | Experts | bpw | Text size | Role |
| --- | --- | --- | --- | --- | --- |
| A | tacodevs-reap25-mixed3bit-gptq-mtp | 192/256 | 3.29 (published) | 100,431,428,728 B archived tree sum ("100.4 GB" card label rounded) | comparison only; not a production source |
| B | MiMo-V2.6-Code-REAP160 | 160/256 | 4–5 planned | measured at build | equal-budget candidate |
| C | MiMo-V2.6-Code-HOPE160 | 160/256 | 4–5 planned | measured at build | equal-budget candidate |
| D | MiMo-V2.6-Code-HOPE176 | 176/256 | 4–5 planned | measured at build | quality-first frontier point, outside equal budget |

- B and C come from the same observation run (REAP and paper-faithful HOPE
  selections), so method and survivor count are compared fairly.
- D (HOPE176) is a quality-first point: it is reported as a separate
  frontier entry and only ships if it safely fits the production budget. It
  is never blended into equal-budget claims.
- A is comparison-only. The official Xiaomi source remains the production
  lineage; its stored MXFP4 expert quantization is independently confirmed
  and documented, not concealed — the MXFP4 QAT characterization (and the
  "no BF16 expert release" statement) is a tacodevs card claim, not an
  officially confirmed fact. We never convert already-quantized MLX
  checkpoints into production ROCmFP4.

## Fit rule and headroom

An arm fits a cap iff `measured_text_weight_bytes <= cap.weight_bytes`, with
the cap's exact bytes from the table above. Auxiliary weights, KV cache and
runtime headroom are recorded **separately** and never netted into text
weight bytes. Nothing runtime-related is measured yet
(`runtime_memory_measured: false` everywhere).

Caps compared: A's exact archived tree text-shard sum (100,431,428,728
bytes = 93.53405677527189 GiB, from
`manifests/baselines/tacodevs-reap25/evidence/tree.json`; the rounded card
label 100.4 GB is informational only), the
user's approximate 100 GB control (100 decimal GB = 93.13 GiB), the project
target lower bound (100 GiB) and the original production target (103 GiB).

## Feasibility math gate (official inventory)

Exact per-retained-expert-index figures from the official Xiaomi memory
report (`manifests/models/mimo-v2.6-flash-rl/memory-report.json`, derived
from the safetensors-header inventory
`manifests/models/mimo-v2.6-flash-rl/inventory.json`;
`XiaomiMiMo/MiMo-V2.6-Flash-RL` revision
`5711b268169967567844e1e560e8a3966da959b1`):

| Quantity | Exact value |
| --- | --- |
| Logical expert parameters per retained index, summed over the 47 MoE layers | 1,182,793,728 |
| FAST/native packing rate | 4.25 bpw exact = 628,359,168 B per retained index (any ROCmFP4/FAST format at ≥ 4.25 bpw costs at least this much) |
| 160 retained indices, EXPERTS ONLY | 160 × 628,359,168 = **100,537,466,880 B** |
| Protected non-expert text weights (kept as stored) | 8,894,573,440 B |
| Full 160-expert text model at the native rate | 109,432,040,320 B |

The arithmetic gates the experiment and is recorded, not smoothed over:

- **At A's exact control cap (100,431,428,728 B) the verdict is
  `infeasible_with_only_ROCmFP4_FAST_or_higher`.** 160 expert indices at
  4.25 bpw already total 100,537,466,880 B — EXPERTS ONLY, over the cap by
  106,038,152 B — before a single protected non-expert byte. Pure
  ROCmFP4/FAST-160 plus protected non-experts cannot fit A's cap.
- **A stays the mandatory static control.** Candidates that do not fit are
  reported as not fitting; unequal weight budgets are never silently called
  "memory matched" — the exact-byte fit rule above decides.
- **B/C at the ~103 GiB production cap are feasible with a dynamic
  precision mix, to be planned:** 109,432,040,320 B ≤ 110,595,407,872 B at
  the native 4.25 rate with protected non-experts (1,163,367,552 B
  headroom). The exact mix is quant-planner work, not asserted here.
- **A fair common-cap comparison against A (192 experts) requires either:**
  a cap-matched **192-expert reconstruction from the OFFICIAL Xiaomi
  source** (production lineage), **or** an explicitly labeled exploration of
  formats **below 4.25 bpw** and/or a **144-expert** arm. The choice is
  stated explicitly whenever made; silent substitution is prohibited.
- No production MLX requantization: already-quantized MLX checkpoints are
  never converted into production ROCmFP4 (restated under Arms).
- This gate changes neither the user's primary 160-expert goal nor any
  original byte budget; it records the arithmetic so the goal is pursued
  with eyes open.

No runtime/hardware memory measurement exists yet; when one does it is
cited here (`runtime_memory_measured: false` remains in force).

## Evaluation protocol

- Distribution controls: held-out perplexity delta, KL, top-token
  agreement, on-policy response agreement slices. **These are controls, not
  capabilities.** Published agreement figures (A: ALL 90.4%, code 90.8%,
  agent 88.5%, reasoning 94.2%) are distribution metrics, NOT measured task
  success.
- The decisive metric is held-out long-horizon golden repository task
  success on the frozen golden set. Golden tasks are frozen before
  calibration and never enter calibration, imatrix, QLoRA, QAT or quant
  search.
- Official source only for production; unofficial checkpoints stay
  comparison-only.

## Full validation matrix

Final validation (issue 46) includes: full MiMo, full-model low-bit Q2,
community REAP50, Ling, Qwen, arm A, and our arms B/C/D. Alternatives enter
the matrix only after verified availability; missing alternatives are
reported unavailable, never invented.

## Published precision reference (allocation evidence, not measured sensitivity)

From A's published allocation (archived under
`manifests/baselines/tacodevs-reap25/`, normalized by the importer):

- 141 projections total: 93 affine 3-bit g128, 22 native MXFP4,
  18 affine 2-bit g128, 8 affine 3-bit g64.
- Other text tensors: 8-bit affine g64; `attention_value_scale` folded into
  `v_proj`.
- Independently confirmed: the official Xiaomi release stores its experts
  natively as MXFP4. The tacodevs card additionally claims MXFP4 QAT
  training and that no BF16 expert release exists — those are card claims,
  not officially confirmed facts.
- Calibration 256×2048 tokens; held-out 31×2048: PPL 9.271 → 9.529,
  KL 0.8315, top1 agreement 79.5%.

**Sensitive layers/tensor types from allocated higher bits:** the tensors
the published plan protects with more bits — non-expert text tensors at
8-bit affine g64 (attention, norms, router and other non-expert weights),
native MXFP4 experts, and 3-bit vs 2-bit expert projections — are the plan's
implicit sensitivity ordering.

This is **published allocation evidence** for sensitivity prioritization
only. It is NOT measured sensitivity, NOT our allocation, and NOT
task-success evidence. We use it to prioritize, never to copy blindly and
never to claim we measured sensitivity.

## Selection comparison tooling

The comparison CLI (`src/mimo_halo/baselines/compare.py`) reports per-layer
retained Jaccard and each removed category BOTH as counts and as sorted
original-expert-ID lists (`removed_both_expert_ids`,
`baseline_only_removed_expert_ids`, `candidate_only_removed_expert_ids`), so
the report names exactly which experts disagree per layer in the original
256-ID namespace (packed or retained-set indices are never used), plus layer
pattern aggregates, optional capability importance masses and optional pair
retention evidence. Missing evidence is reported unavailable, never inferred
from overlap and never zero; the optional files also fail closed (exit 2) on
repeated layer entries and on duplicate pair/circuit identities within the
same capability rather than silently dropping or double-counting stats.
Identity is (layer, kind, expert set, capability): the same pair observed
under two capabilities is two distinct supplied observations, and pair/circuit
totals count these supplied distinct capability observations — not unique
proven functional circuits. Pair retention (candidate retained both endpoints
while baseline removed at least one) is labeled evidence, not proven
functional circuit survival.

## Smoke commands

```bash
# Runnable now: the REAL archived baseline vs a SHAPE-ONLY 160-expert
# placeholder (schema exercise only — NOT a real selection; no selection,
# agreement or quality claim may be read from this candidate):
python - <<'PY'
import json

with open("manifests/baselines/tacodevs-reap25/normalized.json") as fh:
    baseline = json.load(fh)
shape_only = {
    "schema_version": 1,
    "candidate_id": "shape-only-160-placeholder-not-a-real-selection",
    "layers": [
        {
            "layer": entry["layer"],
            "original_expert_count": entry["original_expert_count"],
            "retained_expert_ids": list(range(160)),
            "pruned_expert_ids": list(range(160, entry["original_expert_count"])),
        }
        for entry in baseline["layers"]
    ],
}
with open("/tmp/shape-only-160.json", "w") as fh:
    json.dump(shape_only, fh)
PY
PYTHONPATH=src python -m mimo_halo.baselines.compare \
  --baseline manifests/baselines/tacodevs-reap25/normalized.json \
  --candidate /tmp/shape-only-160.json \
  --output /tmp/shape-only-160-vs-tacodevs.json

# Compare against a future real candidate map (same schema):
PYTHONPATH=src python -m mimo_halo.baselines.compare \
  --baseline manifests/baselines/tacodevs-reap25/normalized.json \
  --candidate manifests/baselines/candidates/reap160.json \
  --output manifests/comparisons/reap160-vs-tacodevs.json

# With observed capability importance and pair evidence:
PYTHONPATH=src python -m mimo_halo.baselines.compare \
  --baseline manifests/baselines/tacodevs-reap25/normalized.json \
  --candidate manifests/baselines/candidates/hope160.json \
  --capabilities manifests/observations/capabilities.json \
  --pairs manifests/observations/pairs.json \
  --output manifests/comparisons/hope160-vs-tacodevs.json

# Regression tests for the comparison tool:
PYTHONPATH=src python -m unittest tests.test_baseline_compare -v
```

The candidate file is any future selection map with the same layer schema
(`schema_version`, `layers[].{layer, original_expert_count,
retained_expert_ids, pruned_expert_ids}`). The tool accepts future candidate
maps without fabricating data: mismatched universes, duplicates (including
repeated layer entries in the optional capability/pair files and duplicate
pair/circuit identities within the same capability), out-of-range IDs,
malformed JSON and non-finite importance values fail closed with exit code 2.