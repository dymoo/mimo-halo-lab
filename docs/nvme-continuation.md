# NVMe / SWA whole-context continuation tests

Regression seam for the whole-context continuation bug: after
`POST /slots/{id}?action=restore`, the next continuation request returns
HTTP 200 but physically re-prefills the **entire** prompt instead of
replaying only the new suffix. Everything below is written against the
pinned runtime (revision from `manifests/upstreams.json`), source anchors
in `upstreams/strix-llama.cpp/…` refer to that checkout. No runtime patch
is applied by this test set; the harness exists to capture the RED proof
first.

## Mechanism under test

- Restore clears the slot prompt and drops the previous task, keeping only
  logical bookkeeping
  (`tools/server/server-context.cpp:2874-2903`,
  `SLOT_STATE_RESTORE → prompt_clear(), task_prev = ptask->task, prev_id = -1`).
- A restored slot therefore enters the continuation path with
  `slot.prompt.checkpoints` empty
  (`tools/server/server-context.cpp:3617` finds `it == rend()`).
- The SWA gate `pos_min_thold = pos_next - n_swa - (has_new ? 0 : 1)`
  (`tools/server/server-context.cpp:3533-3538`) is satisfiable for long
  prompts, so the checkpoint search runs, misses, `do_reset` forces
  `n_past = 0`, and stats report full reprocess
  (`n_prompt_cached = n_past = 0`,
  `n_prompt_processed = prompt_n = full length`,
  `tools/server/server-context.cpp:3617,3648-3651`).
- The documented forced last-token replay is exactly one token
  (`[TAG_PROMPT_LOGITS] if (n_past == n_prompt_in) n_past--; n_batch = 1`,
  `tools/server/server-context.cpp:3641-3644`) — the only allowed
  out-of-suffix physical prefill on the correct path.
- Physical token accounting: per-batch `metrics_queue_prompt` →
  `metrics_flush_prompt` → `llamacpp:prompt_tokens_total`;
  cached tokens mirror as `llamacpp:prompt_tokens_cached_total`
  (`tools/server/server-context.cpp:4308,4357`,
  `tools/server/server-common.cpp:86-88` for the per-request
  `timings.cache_n` / `timings.prompt_n` keys).

## Fixture

Pinned, bounded, never a moving HF branch:

| Field | Value |
| --- | --- |
| Repo | `ggml-org/tinygemma3-GGUF` |
| Revision | `c287502cd9e278dac8eed805c112cce5d0081e0b` |
| File | `tinygemma3-Q8_0.gguf` |
| Size | 47,227,552 bytes (hard download cap, exact match required) |
| SHA-256 | `7566ae7219c93ea2ecc692a931ee122d30c55261d0e2c3347acb8b939d2e9abd` |
| License | WTFPL **as declared upstream** — recorded as declared metadata, not a legal approval |
| Resolve URL | `https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/c287502cd9e278dac8eed805c112cce5d0081e0b/tinygemma3-Q8_0.gguf` |
| Local cache | `.cache/state-tests/tinygemma3-Q8_0.gguf` (git-ignored: `/.cache/`, `*.gguf`) |

Fetch (streams with size cap + sha256 verify, atomic rename, no-op when
already verified):

```bash
python3 scripts/state_restore_harness.py fetch-fixture
```

The model is then served **offline from the local file** — the launch
recipe asserts `--model` + `--offline` and rejects `-hf`/`--hf-repo`.

## Server launch (hub-ready)

Snapshot directory first:

```bash
mkdir -p .cache/state-tests/snapshots
```

Small default (ctx 8192 / history 4096 / suffix 256), CPU first, RAM
prompt cache **disabled** so no false warm hit can mask the bug, and
**never** `--swa-full` (forcing it would mask the pre-patch behavior):

```bash
upstreams/strix-llama.cpp/build-metal/bin/llama-server \
  --model .cache/state-tests/tinygemma3-Q8_0.gguf \
  --offline \
  --ctx-size 8192 \
  --parallel 1 \
  --batch-size 512 \
  --ubatch-size 512 \
  --threads 1 \
  --n-gpu-layers 0 \
  --seed 42 \
  --temp 0 \
  --n-predict 8 \
  --metrics \
  --slots \
  --slot-save-path .cache/state-tests/snapshots \
  --cache-ram 0 \
  --no-cache-idle-slots \
  --host 127.0.0.1 \
  --port 8080
```

Slow large variant (100K history + 4K suffix, ctx 131072) — same command
with `--ctx-size 131072 --batch-size 1024 --ubatch-size 1024 --port 8081`.

Hub start form (readiness by port; the harness also polls `/health`):

```
hub op:"start" name:"nvme-continuation-server" \
  application:"upstreams/strix-llama.cpp/build-metal/bin/llama-server" \
  args:["--model",".cache/state-tests/tinygemma3-Q8_0.gguf","--offline","--ctx-size","8192","--parallel","1","--batch-size","512","--ubatch-size","512","--threads","1","--n-gpu-layers","0","--seed","42","--temp","0","--n-predict","8","--metrics","--slots","--slot-save-path",".cache/state-tests/snapshots","--cache-ram","0","--no-cache-idle-slots","--host","127.0.0.1","--port","8080"] \
  ready:{port:8080, timeout:300}
```

The exact argument vectors live in `configs/state-restore-fixture.json`
(`variants.<name>.launch_args`) and the harness *asserts* the recipe
(`recipe.*` checks): `--cache-ram 0`, `--no-cache-idle-slots`,
`--metrics`, `--slots`, `--slot-save-path`, `--model`+`--offline`,
`--threads 1` + `--n-gpu-layers 0`, `--parallel 1`, matching
`--ctx-size`/`--batch-size`, and **no** `--swa-full`.

## Harness

```bash
# small (default)
python3 scripts/state_restore_harness.py run \
  --base-url http://127.0.0.1:8080 --variant small --report /tmp/nvme-small.json

# slow large CLI variant (102400 + 4096, ctx 131072)
python3 scripts/state_restore_harness.py run \
  --base-url http://127.0.0.1:8081 --variant slow --report /tmp/nvme-slow.json
```

Stdlib only; connects to an externally managed server, never spawns one.
Exit codes: `0` all assertions pass, `1` assertion failure (report lists
the failures), `2` transport/fixture failure (report carries `error`).

Flow (identical continuation request bytes in both arms; prompts are token
id arrays obtained from `/tokenize` with `add_special: false`, so BPE
never re-tokenizes history):

1. recipe + fixture + `/health` + `/props` (slot `n_ctx` must equal the
   variant) + `/slots` preflight; erase slot for a fresh slate;
2. **warm**: prefill `history` (4096 / 102400 tokens), generate `G0`
   (`n_predict` short, temperature 0, seed 42) — asserts fresh
   `cache_n == 0`, `prompt_n == history`;
3. **save**: asserts `n_saved == len(history)+len(G0) -
   replay_boundary_tokens` (prefix + generated history saved correctly;
   each processing pass persists prompt+generated minus the forced
   last-token replay boundary, `server-context.cpp:3641-3644`) and
   `n_written > 0`;
4. **HOT arm** (uninterrupted): continuation with
   `history + G0 + suffix` → per-request `timings.prompt_n/cache_n`,
   per-request **and** global counter deltas, positions, top logprobs;
5. **erase**: asserts `n_erased == len(slot prompt) -
   replay_boundary_tokens` exactly (slot bookkeeping gone), slot
   `is_processing == false`, zero physical delta;
6. **cold probe**: resend the saved state with the RAM cache disabled —
   asserts `cache_n == 0` and full physical `prompt_n` (no false warm hit
   proving `--cache-ram 0` took effect at runtime);
7. **restore**: asserts `n_restored == n_saved` and `n_read == n_written`;
8. **RESTORED arm**: byte-identical continuation request → same
   instrumentation, then the comparison;
9. cumulative `llamacpp:prompt_tokens_total` / `…_cached_total` deltas must
   equal the sum of per-request `prompt_n` / `cache_n`;
10. final erase (exact count, zero delta) to leave the slot clean.

### Physical prefill bound (named constants)

Continuation prompt = saved state (LCP) + suffix. Correct replay is:

| Term | Named in config | Value (small / slow) | Why |
| --- | --- | --- | --- |
| new tokens | `suffix_tokens` | 256 / 4096 | must be evaluated |
| final-fill gap | `batch_size` | 512 / 1024 | the near-end checkpoint is created *before* the last decode batch (`server-context.cpp:3550-3564`), so it trails the prompt end by at most one batch |
| generated tail | `n_predict` | 8 / 8 | `G0` was generated after the prefill, past the checkpoint |
| boundary | `replay_boundary_tokens` | 1 / 1 | documented forced last-token replay (`server-context.cpp:3641-3644`) |

`bounds.prompt_n_max = suffix + batch + n_predict + 1` → 777 (small),
5129 (slow). Full re-prefill is `history + G0 + suffix` (up to 4360 /
106504) — far above the bound. Asserted per arm (`*.prefill_suffix_bound`), plus
`*.position_identity` (`prompt_n + cache_n == len(prompt array)`,
`usage.prompt_tokens` / `prompt_tokens_details.cached_tokens` when
reported), `compare.prompt_n_parity` (restored == hot), token-id and
logprob equality (`determinism.logprob_abs_tol`), and
`*.metrics_delta_physical` (global counter delta == `prompt_n`).

## Test

Skips **only** when `NVME_CONTINUATION_BASE_URL` is unset — no other skip
condition exists:

```bash
NVME_CONTINUATION_BASE_URL=http://127.0.0.1:8080 \
  python3 -m unittest discover -s tests -p 'test_nvme_continuation.py' -v
```

`NVME_CONTINUATION_VARIANT=slow` selects the 100K+4K run.

## Expected RED (pre-patch)

Exactly three assertions are expected to fail on the pinned runtime; the
save/erase bookkeeping assertions use `replay_boundary_tokens`-derived
expectations and stay **green** in RED (they verify bookkeeping, not the
bug). Captured RED values from the live endpoint (variant small):

| Failing assertion | Observed | Mechanism |
| --- | --- | --- |
| `restored.prefill_suffix_bound` | `prompt_n` 4360 (full `history+G0+suffix`) vs bound 777 | restore leaves `slot.prompt.checkpoints` empty → SWA-gate checkpoint search misses → `do_reset` → `n_past = 0` full reprocess (`server-context.cpp:3617,3648-3651`), HTTP 200 |
| `compare.prompt_n_parity` | restored 4360 vs hot 257 | same request bytes under identical state; only the restored arm falls back to full re-prefill |
| `compare.logprob_within_tol` | max \|d logprob\| ≈ 0.0266 > tol 0.001 | full re-prefill changes the decode batch shape vs the hot arm's suffix-only batch; the pinned README documents batch-shape logit nondeterminism — genuine bug damage, must pass post-fix when both arms share the suffix batch shape |

The hot arm stays green, and the cumulative
`llamacpp:prompt_tokens_total` delta shows the full physical prefill
too. That is the proof to capture — the harness does not apply or fake a
fix.

## Results (post-patch, known gap)

State of the full acceptance command on the patched runtime
(`patches/strix-slot-restore-checkpoints.patch`, byte-exact with the
upstream working tree, HEAD == manifest pin `8c1c282e…`), small variant,
after the restore handler gained `seq_rm(-1,-1)` before state load plus
a synthesized end-of-prefix checkpoint:

```bash
NVME_CONTINUATION_BASE_URL=http://127.0.0.1:8080 \
  python3 -m unittest discover -s tests -p 'test_nvme_continuation.py' -v
# Ran 1 test in 8.2s — FAILED (failures=1), RC=1
# summary={'total': 48, 'failed_count': 1,
#          'failed': ['compare.logprob_within_tol'], 'ok': False}
```

| Assertion | State | Measured |
| --- | --- | --- |
| `restored.prefill_suffix_bound` | GREEN | `prompt_n` 257 ∈ [256, 777] (pre-fix RED: 4360 = full re-prefill) |
| `compare.prompt_n_parity` | GREEN | hot 257 == restored 257, `cache_n` 4103/4103 (pre-fix RED: 4360 vs 257) |
| `compare.generated_tokens_equal` | GREEN | 8× token 80632 on both arms (temperature 0, seed 42) |
| `save`/`restore`/`erase` bookkeeping, metrics, recipe (44 more) | GREEN | `n_saved == n_restored == 4103`, `n_written == n_read == 67238252`, cumulative counter deltas exact |
| `compare.logprob_within_tol` | **KNOWN GAP** | max \|d logprob\| `0.01680135726928711` > tol `0.001` (tolerance is seam-fixed; pre-fix value was 0.0266) |

Determinism facts underpinning the gap (live-vs-live experiment, plus
the reproduction driver below): hot-vs-hot and restored-vs-restored are
**bit-identical** across runs (H1==H2, R1==R2), while hot-vs-restored
differs **deterministically** by the same vector every time —
`[0.013054, 0.001345, 0.007094, 0.016801, 0.000877, 0.003625, 0.001237, 0.009002]`
over the 8 generated positions. Save→save also round-trips
byte-identically (F1==F2 sha256), so the state file itself is a fixed
point.

### Batch-split instrumentation (BSP): the leading hypothesis is falsified

Temporary WRN-level probes (checkpoint-search inputs, checkpoint
hit/miss, final `n_past`, every prompt-fill split, every
`llama_decode` submit, restore synthesis) were added to
`server-context.cpp`, built, and run once via the reproduction driver;
they were then stripped — the shipped tree contains none of them.

Continuation split tables from the probe logs — **identical on both
arms**, so the "hot replays the forced boundary as its own `n_batch=1`
pass while restored folds it into the suffix batch" hypothesis is
falsified (no `n_batch=1` pass exists on either arm):

| pass | hot arm | restored arm |
| --- | --- | --- |
| suffix batch | fill 253 tokens @ pos 4103 → decode `n_view=253`, `has_output=0` | fill 253 tokens @ pos 4103 → decode `n_view=253`, `has_output=0` |
| final-fill gap | fill 4 tokens @ pos 4356 → decode `n_view=4`, `has_output=1` | fill 4 tokens @ pos 4356 → decode `n_view=4`, `has_output=1` |
| generation | 8 × decode `n_view=1`, `has_output=1` | 8 × decode `n_view=1`, `has_output=1` |

(The shared warm prefill splits as 7×512 + 508 + 4 on both runs too.)

What the probes *did* reveal is a search-entry asymmetry:

| arm | memory `pos_min` | `pos_min_thold` | checkpoint search | `n_past` |
| --- | --- | --- | --- | --- |
| hot | 0 | 7 (`pos_next 4103 − n_swa 4096`) | **skipped** (`0 < 7`) — no checkpoint code runs at all | 4103 from the common prefix |
| restored | **7** | 7 | **entered** (`7 ≥ 7`) — synthesized `ckpt_hit` runs `load_tgt`/`load_dft`/spec-state | 4103 (unchanged, `pos_next`/`n_past` both no-op to 4103) |

Why `pos_min` differs: the state save skips SWA-masked cells, so after
`llama_state_seq_load_file` the SWA sub-cache min sits at
`pos_next − n_swa` (7) while the hot slot's live SWA cache still
physically holds the dead cells 0–6. The snapshot file's base section
keeps all 4103 cells (`pos [0..4102]`, no gaps, `v_trans 0` —
`/tmp/parse_state.py` output); only the masked cells are absent from
the post-load state. `BSP restore` shows the synthesized checkpoint
`n_tokens=4103 pos_min=0 pos_max=4103`.

### T1 (approved minimal fix attempt): negative result

T1 made the entered path a true no-op: the restore handler's
synthesized checkpoint keeps only `update_pos` — no
`update_tgt`/`update_dft`/`get_state` capture — so `load_tgt`/
`load_dft` return early on empty data and
`common_speculative_set_state` returns on the null spec (no draft
model). Rationale: if the restored arm must enter the checkpoint
search while the hot arm skips it, the entered path must not rewrite
any state the hot arm never touches; rollback of cells past the prefix
is done by the continuation's `seq_rm(p0,-1)` regardless of any
snapshot.

Measured after T1: max \|d logprob\| **unchanged, bit-identical
`0.016801`**, per-position vector identical, BSP logs confirm the
`ckpt_hit` still fires (with the load now a no-op) and splits stay
identical. Conclusion: the redundant load is **exonerated** as the
numeric cause in this experiment. A restored-versus-live cache occupancy or
layout difference remains a hypothesis, not a demonstrated root cause.
Identical batch sizes and the negative no-op experiment do not prove all
internal state equivalent. The seam's `logprob_abs_tol = 0.001` therefore
remains unmet. T1 was reverted; the shipped tree equals the committed patch
byte-for-byte. This investigation is deferred behind the quality baseline;
the patch is not evidence of complete save/restore numerical equivalence.

### Reproduction driver

Essential logic of `/tmp/e8_bsp.py` (stdlib-only consumer of
`scripts/state_restore_harness.py`, same fixtures and request bytes as
the seam):

```python
corpus = srh.build_corpus(BASE, 4352, timeout, cfg['corpus']['line'])
history, suffix = corpus['tokens'][:4096], corpus['tokens'][4096:4352]
# per arm: erase -> warm(prompt=history) -> G0 = generated_ids(warm)
#          -> save -> continuation(history+G0+suffix)  [hot]
op('POST', f'/slots/{slot}?action=restore', {'filename': snap})
#          -> byte-identical continuation             [restored]
# prints per-position logprobs and per-position |d|
```

This snippet describes the experiment; it is not a standalone runnable script.
Use the full acceptance command above to reproduce the remaining failure.
The worker reported three consecutive runs with identical per-position
differences on the instrumented and shipped builds.

## Observation semantics

- **Hot-state freed** = sequence-cache ownership/cells are reusable:
  the cold probe re-populates the erased slot and restore succeeds over
  it. The preallocated KV pool's resident size need not fall; **no RSS
  drop is claimed or asserted**.
- All timings in the report are verbatim server `timings` plus
  client-observed `wall_ms`; no GPU time or p95 is fabricated.
- Logprobs are compared under greedy temperature 0; the pinned README
  documents batch-shape logit nondeterminism for cached prompts, hence
  `determinism.logprob_abs_tol` (token-id equality is asserted strictly).
- Fixture license is recorded as declared upstream metadata (WTFPL), not
  a legal approval.
