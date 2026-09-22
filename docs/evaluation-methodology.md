# Evaluation methodology

Scope of this document: the statistical evaluation contract for paired
task-outcome analysis, implemented by `src/mimo_halo/evaluation/paired.py`
(CLI: `python -m mimo_halo.evaluation.paired`). The tool consumes observed
task outcomes produced by external evaluation runners. It performs
**statistics only** — it runs no inference, starts no sandbox, invents no
runner, and never derives task success from token or distribution
agreement.

---

## 1. Data contract

Input is JSONL, one record per line, conforming to
[`schemas/task-outcome.schema.json`](../schemas/task-outcome.schema.json)
(`schema_version=1`). Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | `1` | Record schema version. |
| `task_id` | string | Underlying task identity. All retries/harness attempts on the same repo issue share one `task_id`. |
| `model_id` | string | Evaluated model. Exactly one distinct `model_id` per file; the two input files must be two different models. |
| `seed` | integer | Inference seed. Repeats of the same `task_id` at different seeds are repeated measurements of one task, never independent tasks. |
| `success` | boolean | Observed task success from the runner. Never inferred from token agreement or distribution metrics. |
| `turns` | integer ≥ 0 | Agent turns. `0` means no agent turn ran (a runtime failure) and is only valid with `success=false`; a `success=true` record must have `turns ≥ 1`. Zero-turn failures stay in the overall statistics under their own slice bucket. |
| `context_tokens` | integer ≥ 0 | Context length in tokens. |
| `language` | enum | `typescript, javascript, python, rust, go, cpp, sql, shell, other`. |
| `domain` | enum | `frontend, backend, databases, distributed, concurrency, network, native, build, ci, infra, api, testing`. |
| `catastrophic` | boolean | Observed catastrophe (unrelated destructive edits, loops, fake test success, goal loss, malformed destructive shell, uncontrolled rewrite, severe context confusion). |
| `failure_category` | enum | `none, misunderstanding, search, architecture, implementation, tool, compiler, test, recovery, forgotten_constraint, context_loss, runtime_corruption, timeout, other`; `"none"` when `success` is true. |

Optional per-record objects (observed runner outputs, never required):
`timings` (number-valued, e.g. `wall_seconds`), `tokens` (integer-valued,
e.g. `generated_tokens`), `toolmetrics` (integer-valued, e.g. `tool_calls`,
`bad_tool_calls`). Every value inside them must be a non-boolean number,
finite, and ≥ 0 — stringified metrics, booleans, NaN/Inf, and negatives all
abort the analysis, so the loader enforces exactly what the schema
declares. Optional `record_type`: `"task"` (default) or
`"negative_control"`.

Loading is fail-closed: malformed JSON, missing required fields, wrong
types, out-of-taxonomy values, or empty files abort the analysis with a
non-zero exit code. Unknown optional fields are passed through untouched;
unknown never means zero.

## 2. Pairing rules

- Exactly **two** inputs: `--reference` and `--candidate`, each a
  single-model JSONL file with **different** `model_id` values. Fewer or
  more models fail.
- Records pair on the key **`(task_id, seed)`**.
- A duplicate `(task_id, seed)` key within one file fails the run. The
  duplicate-key and single-`model_id` checks apply to **every record in
  the file, negative controls included**: a control row with a foreign
  `model_id`, or any two records sharing a key (task/task, control/control,
  control/task), fails the run.
- A key present in only one file is an unpaired outcome: the run **fails**
  unless `--report-missing` is passed, in which case the matched subset is
  analyzed and every missing key is listed in `pairing.missing_pairs` with
  the side it is missing from. Outcomes are never silently dropped. Such a
  report is **incomplete evidence** — it covers only the matched subset,
  not the full paired population — and the report says so explicitly:
  `pairing.evidence` is `"incomplete_matched_subset_only"` when any key is
  missing and `"complete"` on a fully paired run.
- Negative-control records are excluded from the main paired statistics
  and summarized separately (see §6).

## 3. Statistics

### 3.1 Absolute success rate

Two complementary views, both reported:

- **Pair success rate**: successes / matched pairs (raw observed rate).
- **Task-mean rate** (primary): mean over tasks of each task's per-seed
  success fraction. With one seed per task this equals the pair rate.

### 3.2 Bootstrap confidence intervals (task-cluster)

95% percentile bootstrap CIs over the **task cluster**: resampling draws
whole tasks (with replacement), and all seeds of a drawn task move
together. Repeated seeds are therefore never treated as independent
observations — a 10-seed task contributes the variance of one cluster, not
ten. Parameters `--bootstrap-seed` (default `0`) and
`--bootstrap-replicates` (default `10000`) make the interval exactly
reproducible; the report records both under `provenance`.

### 3.3 Relative success ratio

`ratio = candidate task-mean rate / reference task-mean rate`, with a
bootstrap CI from the same replicates. Fail-closed rule: when the
reference task-mean rate is **zero**, the ratio is reported as `null` with
`ratio_status: "not_applicable_reference_zero"` — never `inf`, never
implicitly passing. Ratio CIs exclude replicates whose reference task mean
is zero, and the count of excluded replicates is reported in `ci_note`
whenever any occurred (`ci_note` is `null` when nothing was excluded;
in the reference-zero case it carries the fail-closed explanation above).

### 3.4 Wins / losses / ties

Per task (cluster level): candidate wins when its per-task seed-mean
success exceeds the reference mean, loses when below, ties when equal.
Reported globally as `task_comparison`.

### 3.5 McNemar (exact, two-sided binomial)

McNemar's test is computed **only over independent task pairs**: every
task must have exactly one matched seed on each side. Discordant counts
`b` (reference success, candidate failure) and `c` (reference failure,
candidate success) give the exact two-sided binomial p-value
`2 · Σ_{k≤min(b,c)} C(b+c, k) / 2^(b+c)`, capped at 1. When any task has
repeated seeds, McNemar is reported as `"status": "not_applicable"` with
`"reason": "repeated_seeds_per_task"` and the affected task list — it is
never approximated on clustered data.

### 3.6 Catastrophes

Per-model catastrophe count and rate over matched pairs, the rate delta,
and an explicit `candidate_increase` flag. Catastrophe rates are observed
inputs; release gates about "no unacceptable catastrophic increase" are
applied by humans/release process reading these numbers, not by this tool.

### 3.7 Slices

Observed per-slice per-model rates (pairs, successes, catastrophes, rate)
for `language`, `domain`, `turns_bucket` (`0 (no agent turn), 1-5, 6-15,
16-30, 31-60, 60+`) and `context_bucket` (`<8K, 8-16K, 16-32K, 32-64K,
64-128K, 128K+`, K = 1024 tokens). **Every dimension is anchored to the
reference record of each matched pair** (recorded as
`provenance.slice_anchor = "reference_record"`): `language`/`domain` are
task-intrinsic and taken from the reference record, and `turns_bucket`/
`context_bucket` are computed from the reference side's `turns`/
`context_tokens` for **both** models. Both sides of a pair therefore
always land in the same bucket, so every slice row compares the two models
over an identical paired population — a reference-4-turn /
candidate-7-turn pair counts on both sides under reference `1-5`. Pairs
whose candidate value would fall in a different bucket are not moved:
they are counted in `pairing.field_mismatches` (`language`, `domain`,
`turns_bucket`, `context_bucket`). The `0 (no agent turn)` bucket holds
failed zero-turn records; they remain in the overall statistics, never
dropped (dropping failures would inflate pass rates). No gate logic runs
inside slices.

## 4. Seven evaluation levels

The overall campaign evaluates seven levels; levels 1–3 are distribution
controls, levels 4–7 decide. The paired tool consumes the outputs of
levels 4–7 (task outcomes); levels 1–3 must never be converted into task
success.

| Level | What is measured | Observed metrics |
| --- | --- | --- |
| 1. Artifact | Checkpoint integrity and provenance | Param/tensor inventory, missing tensors, dtype changes, NaN/Inf, reload round-trip, top8 preservation, expert maps, lineage hashes. |
| 2. Numerics | Quantized vs parent numeric damage | KL/JS divergence, top1/top5 agreement, logit cosine, entropy delta, per-slice edge deltas. |
| 3. Distribution | Routing and calibration distribution | Routing top8 overlap, gate correlation, expert frequency, Jaccard/convergence across token budgets, heldout perplexity. |
| 4. Capability | Short structured tasks | Short coding/debug/tool/repo task success, paired with the reference model, same seeds, sliceable by capability/language/domain. |
| 5. Protocol | Agent protocol behavior | Structured output validity, tool name/arg correctness, escaping, stop behavior, error recovery, destructive discipline, negative-control pass rate. |
| 6. Long repo execution | Golden heldout real-repo tasks | Paired task success (§5), turns, generated tokens, tool calls/bad calls, wrong hypotheses, recovery events, wall time, patch quality, catastrophes. |
| 7. Runtime concurrency | Serving behavior under load | C1/C2/C4/C8 aggregate tok/s, memory, throttle/decay, canary stability, bursty-trace replay throughput. |

## 5. Long golden metric and baseline-required comparison

The **long golden metric** is the paired task success rate on the frozen
golden heldout set (50–200 executable real repo tasks, frozen before
calibration): candidate vs reference on the identical task set and seeds,
reported as absolute rates with task-cluster 95% CIs, per-task
wins/losses/ties, and the relative ratio with its CI. Retention is the
paired candidate/reference task-success ratio; absolute rates and
denominator uncertainty are always reported alongside. **Undefined
baseline-zero ratios fail closed.** No promotion decision rests on ±1
task noise; differences whose CIs straddle parity are inconclusive by
construction.

**Every candidate number is baseline-required**: a candidate measurement
is interpretable only against a reference (baseline) run over the same
tasks, seeds, and harness, producing paired outcomes as in §1. A
candidate run without a paired baseline is not evidence. Levels 1–3
metrics (KL, top1 agreement, perplexity) are *controls only* — token
agreement never becomes task success, and a high-agreement candidate with
a worse golden paired outcome loses.

**Golden is never quant search**: golden tasks are a final-confirmation
heldout metric. They are never used for calibration, imatrix, QLoRA, QAT,
quantization search, or method selection; using them so invalidates the
confirmation.

## 6. Negative controls and promotion quality gates

Protocol negative controls (misleading failure, bad prior edit, stale
output, wrong hypothesis, partial patch, dependency red herring) are
carried as `record_type: "negative_control"` records — an **observed
input type**, produced by the real runner. This tool reports their counts
and pass rates per model in a separate `negative_controls` section and
excludes them from main task-success statistics. Release gates (negative
controls ~99%+, tool syntax ≥99.5%, short coding retention ≥93%, repo/tool
≥90%, long agent ≥90%, no unacceptable catastrophic increase) are
thresholds applied to these observed rates by the release process; the
tool computes and reports the rates, it does not implement or fake a
runner or auto-pass anything.

## 7. CLI

```sh
PYTHONPATH=src python -m mimo_halo.evaluation.paired \
  --reference runs/reference.jsonl \
  --candidate runs/candidate.jsonl \
  --output runs/report.json \
  [--report-missing] [--bootstrap-seed 0] [--bootstrap-replicates 10000]
```

Exit code `0` on success; `2` on any fail-closed condition (unpaired keys
without `--report-missing`, duplicates, schema violations, zero matched
pairs, identical models on both sides). The report is deterministic JSON
(`schema_version=1`): fixed key ordering, seeded bootstrap, no timestamps.
Sections: `inputs`, `pairing` (incl. `missing_pairs`, `evidence`,
`field_mismatches`), `success`
(per-model absolute rates + CI), `relative` (ratio + status + `ci_note`), 
`task_comparison` (wins/losses/ties), `mcnemar`, `catastrophes`, `slices`
(reference-anchored, see §3.7),
optional `negative_controls`, `provenance` (incl. `slice_anchor`).

## 8. Limitations

- This tool performs no inference and claims no sandbox/runner
  implementation; runners produce the JSONL records.
- Bootstrap CIs on very small task counts are wide and are reported, not
  hidden; tiny-N reports should be read as inconclusive when CIs straddle
  parity.
- McNemar is unavailable (explicitly `not_applicable`) whenever tasks have
  repeated seeds; use the clustered bootstrap and wins/losses there.