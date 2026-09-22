# Datasets and traces

Read-only ingestion of local coding-agent harness transcripts into normalized,
task-linked episodes, plus deterministic partitioning and golden-bank
eligibility. Nothing here copies, rewrites or uploads raw traces: source files
are read at their origin, and all private artifacts stay in an explicit local
output root outside this Git repository.

## Sources and formats

The four mandatory harnesses (formats verified against real local sessions):

| Source  | Root (config: `configs/dataset.json`)       | Layout |
|---------|---------------------------------------------|--------|
| claude  | `~/.claude/projects`                        | one JSONL per session under munged-cwd dirs; nested files are subagent/sidechain sessions |
| codex   | `~/.codex/sessions`                         | JSONL per session under `YYYY/MM/DD`; first record is `session_meta` |
| pi      | `~/.pi/agent/sessions`                      | JSONL per session under munged-cwd dirs; entries chained by `parentId` |
| omp     | `~/.omp/agent/sessions`                     | same entry schema as pi; subagent sessions in sibling dirs |

Claude Code also writes operational `journal.jsonl` files; these are counted
as non-transcript records and are not parsed as sessions. A small number of
Codex rollouts contain only wrapper records with no payload; they are reported
as `empty_sessions` rather than guessed into events.

Discovered extra (adapter **needed**, not claimed as supported):
`opencode` stores sessions in `~/.local/share/opencode/opencode.db` (SQLite).
It is reported in the discovery summary as an extra adapter; it is not one of
the four mandatory parsers.

## Normalized episode schema

`schemas/episode.schema.json` is the consumer contract (`schema_version 1.0.0`).
One document per parsed session:

- `session`: hashed source identity (`source_path_sha256`), hashed cwd,
  repo identifier / starting revision **when the harness records them** (Codex
  records `git.branch`, `git.commit_hash`, `git.repository_url`; Claude records
  `gitBranch` only; Pi/OMP record the working directory only), model list,
  resume lineage (`parent_session_ids`), branching flag, sidechain count.
- `events`: normalized records with `type` ∈ user/assistant/tool_call/
  tool_result/reasoning/system/custom/meta, the harness' own correlation ids,
  text hashes (text itself only with `--include-text`, credential-redacted and
  bounded to 8192 chars), argument hashes, error flags, and
  verification/spam structural hints derived from command prefixes.
- `episodes`: bounded per-turn decision segments (not giant tool-spam runs)
  with capability labels (`capabilities` plus one `primary_capability`),
  language and difficulty labels, downweight counts and weights — specified
  under "Taxonomies" and "Semantic episode contract" below.

## Taxonomies (closed label sets)

Three closed vocabularies drive automatic labeling. Each is enforced as an
`enum` in `schemas/episode.schema.json`; the lists here and those enums must
stay byte-identical. A label outside a set is a bug, not an extension point.

### Capability tags (22, multi-label)

`repo_exploration`, `planning`, `architecture`, `implementation`,
`debugging`, `compiler_interpretation`, `test_interpretation`, `tool_use`,
`shell`, `git`, `refactoring`, `code_review`, `verification`, `recovery`,
`long_context`, `dependency_reasoning`, `concurrency`, `database`,
`frontend`, `backend`, `systems`, `build_tooling`.

| Tag | Definition |
|-----|------------|
| `repo_exploration` | Reading or searching the repository to build context; no state change. |
| `planning` | Explicit decomposition of the work (todo lists, step plans) before or while executing. |
| `architecture` | Decisions about module boundaries, interfaces, data flow, or structural trade-offs. |
| `implementation` | Writing or changing production code to add behavior. |
| `debugging` | Forming and testing a hypothesis to locate and fix a defect. |
| `compiler_interpretation` | Reading compiler, linker, or type-checker diagnostics to interpret program state. |
| `test_interpretation` | Reading test code or test output to infer expected behavior. |
| `tool_use` | Invoking non-shell tools (MCP servers, fetch, harness-specialized tools). |
| `shell` | Executing shell commands (bash/terminal/exec), regardless of purpose. |
| `git` | History and index operations: commit, branch, rebase, diff, blame. |
| `refactoring` | Restructuring code without changing observable behavior. |
| `code_review` | Reviewing diffs for correctness or style; commenting, not (yet) editing. |
| `verification` | Running tests, build, or lint to confirm a change still holds. |
| `recovery` | Repairing a broken state after an error: retry, revert, cleanup. |
| `long_context` | Decision depends on evidence spread across many files or earlier turns (cross-cutting). |
| `dependency_reasoning` | Reasoning across package manifests, version constraints, vendored APIs. |
| `concurrency` | Threads, async tasks, locks, race conditions. |
| `database` | Schemas, queries, migrations, ORMs. |
| `frontend` | UI, DOM, styles, client-side frameworks. |
| `backend` | Servers, APIs, services, routes. |
| `systems` | Low-level code: memory, syscalls, networking internals, OS interfaces. |
| `build_tooling` | Build systems, CI configuration, task runners, toolchain setup. |

Multi-label: tag every capability an event in the episode evidences. Exactly
one tag is primary, chosen by the **primary-tag rule**:

1. The capability attached to the model decision that produced the episode's
   `action` (edit/apply) is primary.
2. No `action` in the episode → the capability of the first tool call the
   decision reasons about is primary.
3. Any remaining tie → the lexicographically smallest tag wins (deterministic).

`capabilities` stores all tags; `primary_capability` stores exactly one.

### Languages (9)

`typescript`, `javascript`, `python`, `rust`, `go`, `cpp`, `sql`, `shell`,
`other`.

A language is tagged only when it is the **subject** of the episode's
decision/action (code being read with intent, edited, compiled, or executed as
the object of the work). Multi-language episodes tag each subject language.
`other` covers any subject language outside the eight named entries; incidental
mentions and non-subject file formats are never tagged.

### Difficulty (five Score levels)

`D0` mechanical, `D1` bounded, `D2` normal, `D3` hard/ambiguous, `D4`
long-horizon/high-reasoning.

These are **Score levels for automatic labeling**: categorical tags used as
such for calibration weighting and evaluation slicing. No numeric interpolation
is implied — "D2.5" is meaningless, levels are never averaged, and the
distance D0→D1 is not asserted equal to D3→D4. Each level is defined by
self-standing situations:

- **`D0` mechanical**
  - Apply a rename or file move that the request specifies exactly, updating
    every reference the tool lists.
  - Set a version pin or config value to the literal value given in the
    request.
  - Add the import, export registration, or route entry that the error message
    names precisely.
- **`D1` bounded**
  - Fix a lint or type error whose location and cause the diagnostic already
    states.
  - Write a small function from a spec that fixes inputs, outputs, and edge
    cases.
  - Update every call site after a signature change, with all callers visible
    in one search.
- **`D2` normal**
  - Implement a feature that requires reading two or three modules to find the
    right seam.
  - Diagnose a failing test whose output names the symptom but not the cause.
  - Refactor a unit while keeping its existing callers' behavior intact.
- **`D3` hard/ambiguous**
  - Chase an intermittent failure with no reproduction steps while several
    causes remain plausible.
  - Resolve a request that conflicts with, or underspecifies, existing
    behavior; choose and state a defensible interpretation.
  - Change a cross-module contract whose correct shape can only be inferred
    from current usage.
- **`D4` long-horizon/high-reasoning**
  - Run a repo-wide migration with dependent steps, verification checkpoints,
    and resumption across sessions.
  - Track a concurrency defect through hypothesis, instrumentation, and a
    verified fix.
  - Redesign a subsystem spanning build configuration, public API, and tests.

## Semantic episode contract

Decision-bearing extraction walks this spine, in order, once per episode:

`task` → relevant current state → `tool_call` → bounded relevant `output` →
`decision` → `action` (edit/apply) → `verification` → `recovery`

| Field | Type | Required? | Bounded-size rule |
|-------|------|-----------|-------------------|
| `task` | string (stable task id) | required | Reference only, ≤ 256 chars; never raw prompt text. |
| `current_state` | string (excerpt) | optional | ≤ 2048 chars; only what the model actually saw at decision time; omitted when the decision needed no external state. |
| `tool_call` | object `{tool_name, arguments}` | required, ≥ 1 per decision-bearing episode | `tool_name` ≤ 64 chars; `arguments` ≤ 2048 chars, otherwise sha256 + char count only. |
| `output` | string (tool result) | required when `tool_call` present | ≤ 8192 chars (`max_bounded_output_chars`); head+tail truncation with a marker; the full payload is never stored. |
| `decision` | string (the model's stated choice) | required for decision-bearing episodes | ≤ 4096 chars; without it the turn is not decision-bearing (`has_decision=false`) and is excluded. |
| `action` | object (edit/apply performed) | optional | file key + content hash + char delta only; full pre/post contents never stored; absent when the decision was "no change". |
| `verification` | object `{kind, passed}` | optional | `kind` names the command family (test/build/lint/typecheck/manual); present only when verification ran inside the episode. |
| `recovery` | object `{error_event, window}` | optional | present when an error result is followed by success within `recovery_window_events` = 12. |

The spine order and these bounds are **extraction rules**, not stored columns.
The normalized episode record in `schemas/episode.schema.json` keeps the
derived facts: `event_seq` (the episode's ordered events), `has_decision` ↔
`decision`, `verification_events` ↔ `verification`, `recovered` ↔ `recovery`,
`downweighted_events` (+ optional `downweight_classes`) ↔ downweighting, and
`capabilities` / `primary_capability` / `languages` / `difficulty` ↔ labels.
`task` is linkage: it is resolved downstream by task grouping (see "Task
identity, dedup, quarantine") and is not stored inside the episode record.

### Downweight classes (extraction rules)

Five explicit classes reduce an event's contribution; each observed class is
recorded in `downweight_classes`, and the count of reduced events lands in
`downweighted_events`:

| Class (schema enum) | Trigger rule |
|---------------------|--------------|
| `lockfile` | Read/edit of dependency lock artifacts — path matches `*.lock`, `*.sum`, or `*-lock.json` (package-lock.json, yarn.lock, Cargo.lock, go.sum, …). Content is never quoted; every such event counts as downweighted. |
| `enormous_grep_output` | Search-tool result (grep/rg/find/glob listing) exceeding the 8192-char output bound. Head+tail and counts only; each overflowing result counts once. |
| `repeated_build_spam` | Consecutive build/install/test runs (`bash:build`/`bash:install` families, `spam_hint`) whose normalized output repeats. Only the first and the final run carry weight; intermediate runs count as downweighted. |
| `generated_bundle` | Generated artifacts — `*.min.js`, `*.bundle.js`, `*.map` and friends. Treated as opaque; never quoted as evidence. |
| `duplicate_file_content` | The same content hash (`text_sha256`) read more than once in an episode; occurrences after the first count as downweighted. |

## Tagging rules and boundary cases

- **debugging vs test_interpretation**: reading test code or test output to
  learn what behavior is expected = `test_interpretation`. Using a failing
  test to locate and fix a defect = `debugging` primary, `test_interpretation`
  secondary. Running tests only to confirm an already-made change =
  `verification`, neither of the two.
- **compiler_interpretation vs debugging**: acting on a diagnostic whose
  message itself resolves the cause = `compiler_interpretation`. Forming a
  hypothesis across code to fix an unnamed cause = `debugging`.
- **verification vs test_interpretation**: confirming a change holds by
  executing tests/build/lint = `verification`; inferring intended behavior
  from test code or output = `test_interpretation`.
- **tool_use vs shell**: `shell` applies when the tool is a shell/terminal/
  exec invocation, whatever its purpose (a pipeline of `grep|sort|uniq` is
  `shell`, not `tool_use`). `tool_use` applies only to non-shell tools (MCP
  servers, fetch, harness-specialized tools). A generic "run" wrapper around a
  shell command is still `shell`.
- **recovery as secondary tag**: `recovery` is attached only when an error or
  failed command is followed by a correction within the 12-event recovery
  window, and it is a **secondary** tag by default. It may become primary only
  when restoring a broken state is the episode's entire point (revert, cleanup,
  crash recovery) and no other decision-bearing capability applies.
- **long_context is cross-cutting**: it rides along as an extra tag and is
  never primary; if the primary rules would select it, fall through to the
  remaining tags.

## Labels are features, not truth

Capability, language, and difficulty labels are automatically inferred
**features**: they feed calibration weighting and evaluation slicing, and
nothing else. They are never evidence of task success, never move a group
between partitions — splits are by underlying task, so every episode of one
task lands in one split regardless of its labels (see "Task identity, dedup,
quarantine" and "Partitioning") — and they never decide golden-bank
membership: golden stays **human-confirmed and execution-grounded** (a
reserved group with a registered test oracle, see "Golden bank freeze").

## Correlation and resume/branch handling

- Claude: `tool_use.id` ↔ `tool_result.tool_use_id`; `parentUuid` uuid chains
  (repeated parent = fork); `isSidechain` subagent events excluded from
  main-chain episodes.
- Codex: `function_call`/`custom_tool_call` `call_id` ↔ corresponding outputs;
  resume lineage via `parent_thread_id`/`forked_from_id`.
- Pi/OMP: `toolCall.id` ↔ `toolResult.toolCallId`; `parentId` chains (a parent
  that is not the previous entry = fork/retry branch).

Unmatched calls/results are reported, never guessed into pairs.

## Task identity, dedup, quarantine

An underlying task requires explicit linkage:

- a repo working directory (hashed), plus
- an explicit issue/ticket reference (e.g. `PROJ-42`, `issue #99`) found in the
  first user prompt.

Cross-harness grouping happens only through the explicit issue reference.
Same-prompt retries inside one harness dedup by first-prompt content hash.
Cross-harness prompt lookalikes without an explicit linkage are **quarantined**
(never merged, never randomly split). Sessions with no resolvable working
directory or no prompt are quarantined as well.

## Partitioning

Deterministic grouped split (all episodes of a task in one partition):
`pruning 35%`, `quant 20%`, `recovery 20%`, `validation 10%`, `golden 10%`,
`torture 5%`, assigned by a stable salted hash. Once written, frozen
assignments are immutable; later runs reuse them and only place new groups.

The golden 10% is a **reservation drawn purely from the salted hash**,
independent of oracle eligibility: a reserved group always keeps the golden
assignment. A reserved group that lacks prerequisites stays reserved and is
reported `not_ready` (`reserved_golden_missing_eligibility`) — it is never
redistributed into the five non-golden partitions, so the holdout can never
leak into training. Eligibility only marks readiness of reserved groups; it
can never move a training group into golden when an oracle appears later.

The single exception to frozen immutability is the audited pre-calibration
`migrate-reservations` command (explicitly authorized for the one-time fix of
the old redistribution defect): it refuses while frozen golden tasks or
golden-consumption evidence exist, backs up the agent-created assignment files
with sha256 digests, regenerates reservations, and records an audit reason.

## Golden bank freeze

A golden task is eligible only with ALL of, evaluated from real history:

- `task_id` (deterministic group id),
- `repo_identifier` (remote origin URL recorded by the harness),
- `starting_revision` (commit hash captured at session start),
- `task_prompt` (first user prompt),
- `test_oracle` (explicitly registered privately; chat transcripts are never
  treated as executable oracles).

The `freeze` command returns the exact missing fields per candidate. Only
groups that are **both hash-reserved and fully eligible** freeze into the
executable bank — an eligible training group is reported but never imported.
If fewer than 50 reserved candidates are fully eligible, the bank status is
`not_ready` with the exact counts and missing prerequisites — no fake tasks
are selected, and golden never participates in calibration/quant search.

## Privacy boundary

- Raw transcripts are read-only at origin; nothing is copied, rewritten or
  committed.
- Public artifacts (this repo) carry hashes, counts, bytes and schema facts
  only — never raw paths or payloads.
- Private artifacts go to the explicit `--output-root` (e.g. the MIMO_LAB
  workspace). Default normalization is hash-light; `--include-text` adds
  credential-redacted, bounded text.
- Redaction removes well-known secret shapes (API keys, tokens, private key
  blocks, bearer headers, password assignments). This is redaction, **not** a
  claim of complete anonymization; free-form secrets may remain.

## Commands

```bash
# read-only discovery: public summary in repo, private index in output root
PYTHONPATH=src python3 -m mimo_halo.traces discover \
  --output-root "$MIMO_LAB"

# normalized episode sets (hash-light by default)
PYTHONPATH=src python3 -m mimo_halo.traces normalize \
  --output-root "$MIMO_LAB" [--include-text] [--max-sessions N]

# deterministic grouped split with frozen-assignment immutability
PYTHONPATH=src python3 -m mimo_halo.traces partition \
  --output-root "$MIMO_LAB" \
  --frozen-assignments "$MIMO_LAB/traces/partition/frozen-assignments.json" \
  --golden-eligible "$MIMO_LAB/traces/golden/eligibility.json"

# golden eligibility + freeze manifest (reserved + eligible only)
PYTHONPATH=src python3 -m mimo_halo.traces freeze \
  --output-root "$MIMO_LAB" [--oracle-index path/to/private-oracles.json]

# audited pre-calibration migration: restore hash-reserved golden groups that
# the old redistribution defect routed into training. Refuses while frozen
# golden tasks or golden-consumption evidence exist; writes backups + digests
# and an audit record under <output-root>/traces/partition/audit/.
PYTHONPATH=src python3 -m mimo_halo.traces migrate-reservations \
  --output-root "$MIMO_LAB"
```

The output root must be outside this Git repository and is capacity-checked
before any write; the tool refuses to write and never deletes when headroom is
insufficient. `--max-sessions` bounds a run for low-capacity volumes.
