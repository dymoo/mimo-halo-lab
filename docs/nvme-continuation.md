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
