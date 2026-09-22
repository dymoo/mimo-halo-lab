# Quality expectations (pre-measurement — not claims of losslessness)

Status: no compressed MiMo candidate has been built or evaluated yet. These are
the EXPECTATIONS a well-executed pipeline (HOPE-calibrated pruning to ~160
experts + healthy quantization + recovery) is aiming for. They are targets to
hit or miss honestly — never claims. Every miss is reported; nothing here is
rounded up.

## Expected capability retention (of full MiMo V2.6 Flash)

| Capability slice | Expectation |
| --- | --- |
| Short coding | 93–98% |
| Reasoning / debugging | 90–96% |
| Tool use | 90–96% |
| Normal repository agents | 90–95% |
| Long autonomous agents | 85–93% (pre-recovery) |
| Long autonomous agents after successful recovery | 90–95% target; >=93% target line; >=95% stretch |

## The decisive metric

**Long-horizon held-out task success is the decisive metric.** It is measured on
the frozen golden bank — currently 102 reserved frozen records (hashed ids in
`manifests/datasets/golden-freeze-summary.json`, never used for calibration,
imatrix, QAT, or quant search), of which only 36 carry supported
member-session oracle provenance and 66 are provisional (borrowed or
unrecorded linkage), and none has yet demonstrated executable-task oracle
validation, so the registry is **not** a validated ≥50-task golden bank —
measured with the paired statistical machinery
(`src/mimo_halo/evaluation/paired.py`: exact McNemar, bootstrap CIs, win/loss
tables). Perplexity, KL and agreement are diagnostic screens — they rank cheap
candidates during staged evaluation but they do not crown the winner.

A model that only retains short-task capability has failed, regardless of its
perplexity.

## Production quality gates (hard floors)

- Long-agent retention >= 90% of full MiMo (target >= 93%, stretch >= 95%).
- Short-coding retention >= 93%; repo/tool retention >= 90%.
- Structural integrity suite 100% (inventory, missing tensors, dtype changes,
  NaN/Inf, reload round-trip, top-8 preservation, expert maps, lineage hashes).
- No known numerical correctness bug; no unacceptable increase in catastrophic
  failures (catastrophes tracked separately from mean scores).
- Tool-protocol behavior essentially unchanged; stable overnight C8 operation.
- Weight budget: the matched sweep band (~90 GiB, 96,636,764,160 B +/-2%)
  with a 105 GiB production ceiling (the earlier spec figure of 100–105 GiB is
  superseded by the quality-first ~90 GiB sweep directive); C8 floor >= 65
  aggregate tok/s (estimates page); useful RAM margin retained.
  Enforcement machinery: `src/mimo_halo/evaluation/gates.py` (fail-closed:
  estimates never pass; unmeasured never promotes).

## Decision rules already fixed

- If a full-model Q2 quant wins real tasks, ship Q2.
- If REAP beats HOPE on held-out tasks, ship REAP.
- If 176 experts materially beats 160, spend the RAM.
- The methodology is subordinate to the objective: maximize the probability an
  autonomous coding agent correctly completes a difficult repository task.

## What replaces this page

The full validation matrix rows per candidate (per-language, per-domain,
trajectory-length slices, long-context slices 8K..128K+, failure taxonomy,
catastrophic-failure counts, patch-quality scores) recorded during the staged
evaluation — with paired deltas vs original MiMo and vs the tacodevs REAP25
mixed-3bit control. Expectations here get marked HIT or MISSED against those
records; they never silently convert into claims.
