# Trace-labeling pilot: typed judgments vs the deterministic rule labeler

Status: **complete — run executed 2026-09-22.** Verdict: **supported** (under the
pre-registered decision rule below). Machine-readable form:
`manifests/experiments/labeling-pilot.json`. Helper:
`scripts/labeling_pilot_sample.py`. Private evidence (states, judgments,
comparison, inspection records) lives under `$MIMO_LAB/scratch/labeling-pilot/`
and never leaves this workstation.

## Hypothesis

Typed judgments over redacted episodes provide labels that measurably improve
on (or reliably flag errors in) the deterministic rule labeler
(`src/mimo_halo/traces/labeling.py`).

Non-goals held: no external API call was made by pilot code; `labeling.py` and
the schemas were not modified; the golden partition never entered the corpus.

## Pre-registered decision rule

Recorded in `frame-summary.json` (written **before** any judgment was
collected; file mtime precedes `judgments.json`):

| Verdict | Criteria over 10 hand-inspected disagreements |
| --- | --- |
| supported | `judge_better >= 6` and `rule_better <= 2` |
| falsified | `rule_better >= 6` and `judge_better <= 2` |
| inconclusive | anything else, including fewer than 10 inspectable disagreements |

Difficulty/primary agreement rates are a **construct check, reported but not
gating**.

## Method

### Sampling frame and stratification

Single full pass over the raw local transcripts (read-only at origin) on
2026-09-22: **4,140 sessions parsed (0 parse failures), 40,381 episodes**,
203.9 s parse / 279.8 s total wall.

Exclusions before the frame:

| Exclusion | Count |
| --- | --- |
| golden-partition episodes (contamination rule) | 6,229 episodes |
| sessions with no assigned group | 80 sessions |
| sessions in groups mapping to >1 partition | 27 sessions |
| groups unresolvable: no cwd / no prompt+ref / cross-harness prompt quarantine | 32 / 9 / 39 sessions |

**Frame: 34,152 episodes** across the five non-golden partitions
(`pruning 12,188 · recovery 12,026 · quant 5,288 · validation 3,731 ·
torture 919`).

Strata = (partition × rule difficulty) cells; quota = 2-per-cell floor +
largest-remainder proportional fill to **200**; within a cell, episodes are
picked capability-round-robin in `sha256(seed|episode_id)` order with
`seed = labeling-pilot-v1`. All 25 cells non-empty and represented
(per-cell allocation in `frame-summary.json`). Sample ids are stable sha256
episode ids.

### Rule labels

Produced by the checked-in deterministic labeler, **recomputed in memory from
the raw transcripts**: the hash-light normalized corpus predates the labeler
and carries no labels. `tests/test_labeling.py` run at pilot time: **31
passed, 14 subtests passed**. Frame-level observation worth recording: across
34,152 frame episodes the rule emitted only **11 of the 22 capability tags**
(`git, refactoring, code_review, architecture, dependency_reasoning,
concurrency, database, frontend, backend, systems, build_tooling` never
fire — the tool-name capability map has no path to them).

### Typed judgments (the local judgment path)

Tooling: the harness's in-context `judge_batch` (typed questions; one state =
one blinded redacted episode). **Model reported by batch status:
`openrouter/typesafe/jev-1.13-20260917`** — TypeSafe/Jev credentials were
present in this environment, so the judgments did run through that model.
Batch: 200/200 states answered, 0 failed, 11 questions each = **2,200
answers, 3.2 s wall**.

States are **blinded**: raw facts only (event sequence/types, tool names,
error flags, credential-redacted bounded text, first-prompt excerpt). No rule
label, no labeler input hint (`verification_hint`/`spam_hint`/weights), no
partition. States exist only under `$MIMO_LAB`.

Question set (verbatim spec: `$MIMO_LAB/scratch/labeling-pilot/questions.json`,
sha256 `dab0612f…77f1e4`):

| id | type | spec source |
| --- | --- | --- |
| `difficulty` | Score over `D0..D4` | datasets.md level anchors, verbatim in instructions |
| `primary_capability` | Choice over the 22 tags | datasets.md tag table + primary-tag rule + boundary cases |
| `verification` | Noul (bool) | datasets.md verification definition + boundary |
| `tag_<x>` × 8 | Noul (bool) | five boundary-critical tags (`debugging, compiler_interpretation, test_interpretation, verification, recovery`) + top-3 frame-frequency fill (`long_context, shell, repo_exploration`) |

Answer decoding: difficulty = argmax of the returned probability vector;
bools = P(yes) ≥ 0.5; choice = returned label. **Twin-question self-check**:
the two verification-framed questions (Noul vs capability tag) agreed on
163/200 = **0.815** — the judge's own consistency ceiling for that construct.

### Hand inspection

Selection rule fixed before inspection: **one hash-ordered disagreement per
family across 10 families** (difficulty, primary_capability, verification,
and the 7 tag families; the verification twin tag excluded), distinct
episodes. Inspector: **this pilot's worker agent, session model
`openrouter/xiaomi/mimo-v2.6-flash`** — independent of the judging model.
Single inspector, no second rater (recorded limitation).

## Results

### Agreement headline

| Question | Metric | Result |
| --- | --- | --- |
| difficulty | exact | **50/200 = 0.250** |
| difficulty | within one level | **118/200 = 0.590** |
| primary capability | exact | **66/200 = 0.330** (66/172 = 0.384 on rule-labeled rows; 28 rows rule-unlabeled) |
| verification Noul | agreement | **150/200 = 0.750** |
| per-tag Nouls (8) | agreement range | **0.405 … 0.865** (table below) |

### Difficulty confusion matrix (rows = rule, cols = judge)

|  | D0 | D1 | D2 | D3 | D4 |
| --- | --- | --- | --- | --- | --- |
| **D0** | 9 | 0 | 12 | 18 | 13 |
| **D1** | 9 | 2 | 7 | 14 | 11 |
| **D2** | 0 | 0 | 6 | 8 | 8 |
| **D3** | 1 | 0 | 13 | 25 | 28 |
| **D4** | 0 | 0 | 5 | 3 | 8 |

Marginals: rule `D0 52 · D1 43 · D2 22 · D3 67 · D4 16`; judge
`D0 19 · D1 2 · D2 43 · D3 68 · D4 68`. Mean level (0–4 index): rule
**1.76**, judge **2.82**. Split by the rule's own decision flag:

| subset | n | rule mean | judge mean |
| --- | --- | --- | --- |
| `has_decision = false` | 52 | 0.00 | 2.50 |
| `has_decision = true` | 148 | 2.38 | 2.93 |

Two systematic judge behaviors, stated plainly: the judge **effectively never
selects D1** (argmax D1 in 2/200; mean P(D1) = 0.038) and rates episodes
harder overall. All 43 rule-D0 disagreements sit on the `has_decision =
false` floor.

### Primary capability

Top disagreement pairs (rule → judge):

| n | rule | judge |
| --- | --- | --- |
| 7 | debugging | implementation |
| 7 | implementation | shell |
| 7 | repo_exploration | tool_use |
| 6 | implementation | planning |
| 5 | (rule omitted) | shell |
| 5 | (rule omitted) | tool_use |
| 5 | implementation | debugging |
| 4 | (rule omitted) | repo_exploration |
| 4 | compiler_interpretation | debugging |
| 4 | debugging | shell |
| 4 | planning | tool_use |
| 4 | shell | repo_exploration |
| 4 | test_interpretation | repo_exploration |
| 4 | verification | shell |

Disagreement patterns (134 total): rule omitted / judge picked **28** ·
rule purpose → judge vehicle **39** · rule vehicle → judge purpose **7** ·
other **60**. Judge used 15 distinct tags; the rule's frame marginals are
dominated by `implementation/repo_exploration/planning/shell/debugging`.

### Verification Noul and per-tag Nouls

Verification: 150 agree / 50 disagree — directions: rule=false & judge=true
**36**, rule=true & judge=false **14**.

| tag | both | rule-only | judge-only | neither | agree | rate |
| --- | --- | --- | --- | --- | --- | --- |
| debugging | 13 | 2 | 117 | 68 | 81 | 0.405 |
| compiler_interpretation | 10 | 4 | 30 | 156 | 166 | 0.830 |
| test_interpretation | 15 | 6 | 64 | 115 | 130 | 0.650 |
| verification | 45 | 3 | 62 | 90 | 135 | 0.675 |
| recovery | 45 | 5 | 62 | 88 | 133 | 0.665 |
| long_context | 73 | 3 | 92 | 32 | 105 | 0.525 |
| shell | 119 | 0 | 27 | 54 | 173 | 0.865 |
| repo_exploration | 84 | 8 | 65 | 43 | 127 | 0.635 |

Every tag family is dominated by judge-only positives: the rule labeler omits
rather than guesses (by design), the judge tags liberally. Content inspection
below shows **both** genuine rule misses and judge over-reads in that mass.

### Disagreement census (816 total)

| family | n | structural pattern |
| --- | --- | --- |
| primary_capability | 134 | 28 rule-omission rows; vehicle↔purpose swaps dominate the rest |
| tag:debugging | 119 | judge-only 117 vs rule-only 2 — judge reads any failure-hypothesis work as debugging |
| tag:long_context | 95 | 92 judge-only; rule gate needs `has_decision` |
| tag:repo_exploration | 73 | 65 judge-only; rule maps capabilities by tool name only |
| tag:test_interpretation | 70 | 64 judge-only; rule fires only on error+test-run co-occurrence |
| tag:recovery | 67 | 62 judge-only; rule requires a later success, doc boundary says "correction" |
| tag:verification (twin) | 65 | mirrors the verification family with different framing |
| difficulty | 82 | 43 no-decision-floor (all rule-D0), 33 judge-harder-with-decision, 6 rule-structurally-harder; judge harder in 76, easier in 6 |
| verification | 50 | 36 rule-miss direction / 14 judge-miss direction |
| tag:compiler_interpretation | 34 | 30 judge-only; rule requires a diagnostic in an error result |
| tag:shell | 27 | 27 judge-only; includes narration-in-window cases |

### Ten hand-inspected disagreements

| # | family (hash-ordered pick) | rule | judge | verdict | cause |
| --- | --- | --- | --- | --- | --- |
| 1 | difficulty `04268b3077a17e70…18403` | D0 | D3 | **judge_better** | rule error: `has_decision=false` (no retained decision text) pins a 64-event multi-system infrastructure episode to the no-decision floor; the content requires cross-subsystem diagnosis |
| 2 | primary `04968bac4b188b5d…9d09` | test_interpretation | repo_exploration | **judge_better** | rule error: no-action rule 4 fired on an error from a verification-prefixed shell command, but no test code/output was interpreted; the episode is repository/web research (judge split repo_exploration vs shell) |
| 3 | verification `125663597f48daea…df69` | false | true | **judge_better** | rule error: staged fail-closed smoke tests of a deployed service run via shell commands outside the detector's test/build/lint prefix family (`verification_events = 0`) |
| 4 | tag:debugging `0286ca3b114db0b9…0099` | false | true | **ambiguous** | genuinely ambiguous: hypothesis-driven diagnosis of failure causes with no defect-fixing edit inside the episode; the taxonomy requires forming *and* fixing a defect |
| 5 | tag:long_context `04d79841c44f6ed6…f9d4` | false | true | **judge_better** | rule error: cross-module contract fixes spanning source, adapter and two test files, but the `long_context ≥40 events AND has_decision` gate fails on the same missing-decision-text gap |
| 6 | tag:shell `27728c1ef4d5a032…01ae` | false | true | **rule_better** | judge error: the episode window contains only reasoning/assistant narration (`tool_calls = 0`); the contract tags what *events* evidence, and no shell event exists in this episode |
| 7 | tag:repo_exploration `0d4493bf0faf4051…3b99` | false | true | **judge_better** | rule error: a repository source read executed through the generic exec tool; the tool-name capability map cannot see shell-disguised exploration |
| 8 | tag:test_interpretation `061250bf6e2dc712…66a4` | false | true | **judge_better** | rule error: episode reads test fixture diffs to infer expected behavior — the taxonomy's own definition — but runs no failing test, and the rule only fires on error+test-run co-occurrence |
| 9 | tag:recovery `226e1f93477d0a72…5a99` | false | true | **judge_better** | rule vs doc boundary: failed run → corrective edit within the window, no re-run before the episode closed. datasets.md's boundary sentence says "error followed by a **correction**"; the contract table says "followed by **success**"; the implementation follows the latter, the judge the former |
| 10 | tag:compiler_interpretation `1b7a0b4fb2dcc839…a2a8` | false | true | **rule_better** | judge error: no compiler/type-checker diagnostic appears in any error event of the episode — only narration mentions type-check status; judge confidence was the lowest of the ten (0.58) |

**Tally: judge_better 7 · rule_better 2 · ambiguous 1.**

## Verdict

**SUPPORTED** — `judge_better = 7 ≥ 6` and `rule_better = 2 ≤ 2` under the
pre-registered rule. On this sample, blinded typed judgments of redacted
episodes **measurably improve on and reliably flag** deterministic rule
labels:7 of 10 content-inspected disagreements are rule errors (decision-text
proxy gaps, verification-prefix blindness, tool-name-only capability mapping,
test-run-only observables, a doc-boundary conflict), 2 are judge errors
(both: capability inferred from narration rather than events), 1 is a genuine
taxonomy boundary ambiguity.

What this **does not** establish:

- The judge is **not a drop-in relabeler**: difficulty exact agreement is
  0.250 (within-one 0.590), the judge skips D1 entirely and runs ~1.1 levels
  harder than the structural rule; `debugging` per-tag agreement (0.405) shows
  heavy liberal over-tagging. The construct check says the two sources share
  signal but not calibration.
- The rule labeler is **conservative by construction** (omits rather than
  guesses), so judge-only positives structurally outnumber rule-only ones;
  the value demonstrated is in **disagreement triage** — judge-rule
  disagreement is a high-yield flag queue for rule blind spots — not in
  replacing the rule.
- Single inspector for the10 verdicts; no second rater.

## Honest labeling (who produced what)

| Role | Producer |
| --- | --- |
| Rule labels | deterministic Python, `src/mimo_halo/traces/labeling.py` — no model |
| Typed judgments (2,200 answers) | harness `judge_batch` tooling, model **`openrouter/typesafe/jev-1.13-20260917`** exactly as reported by the batch status field; TypeSafe/Jev credentials were present, no other API was called |
| Hand inspection of the 10 + verdicts | pilot worker agent of this harness session, session model **`openrouter/xiaomi/mimo-v2.6-flash`** — independent of the judging model |
| Sampling/comparison/statistics | local Python code only (`scripts/labeling_pilot_sample.py` + in-session analysis); pure functions, no network |

## Privacy

Repo files (this doc + the manifest) contain **hashed episode ids and
aggregates only** — no episode content, prompts, paths, repo identifiers, or
secrets; privacy self-grep recorded in the run evidence. Private states are
credential-redacted bounded text (repo `redact.py` at parse time) stored only
under `$MIMO_LAB/scratch/labeling-pilot/`. The golden partition was excluded
from the frame entirely (6,229 episodes).

## Reproduce (runnable spec)

```bash
# 1) sample + blinded states (full pass, ~4.6 min on this workstation)
PYTHONPATH=src python3 scripts/labeling_pilot_sample.py \
    --config configs/dataset.json \
    --output-root "$MIMO_LAB" \
    --target 200
# -> $MIMO_LAB/scratch/labeling-pilot/{frame-summary.json,sample-index.json,sample-states.jsonl}
#    frame-summary.json records seed, quotas, tag subset and the pre-registered criteria.

# 2) judgments: harness in-context judge_batch, states = sample-states.jsonl
#    "state" values, questions verbatim from questions.json (sha256 above),
#    decode: score argmax / bool >= 0.5 / choice label. -> judgments.json

# 3) comparison (definitions, exactly as run):
#    difficulty: exact = judge argmax level == rule; within-one = |Δlevel| <= 1;
#    confusion rows=rule cols=judge over D0..D4.
#    primary: exact on the 22-label choice (rule None never equals).
#    verification/tag nouls: rule side = tag in rule capabilities;
#    judge side = P(yes) >= 0.5; agreement = matches / 200.
#    census = every mismatch row {q, episode_id, rule, judge, stratum, structure}.
```

## Evidence pointers

- Manifest: `manifests/experiments/labeling-pilot.json`
- Helper: `scripts/labeling_pilot_sample.py`
- Private (read-only in practice): `$MIMO_LAB/scratch/labeling-pilot/` —
  `frame-summary.json` (pre-registered criteria, written pre-judging),
  `sample-index.json`, `sample-states.jsonl`, `questions.json`,
  `judgments.json` (model string + 2,200 answers), `comparison.json`,
  `handinspect.json` (the10 states + both labels)
- Rule-labeler tests: `tests/test_labeling.py` (31 passed at pilot time)
