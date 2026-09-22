# State retention policy

Pure decision module for the NVMe state tier: explicit observation in,
decision JSON out. One deterministic pass, no scheduler coupling, no
disk/runtime mutation, no wall-clock reads.

```bash
PYTHONPATH=src python3 -m mimo_halo.state_tier.retention \
  --config configs/state-retention.json \
  --input OBSERVATIONS.json --output DECISIONS.json
```

- Source: `src/mimo_halo/state_tier/retention.py` (stdlib, Python >= 3.11)
- Config: `configs/state-retention.json`
- Public JSON contract (config, observation, decision):
  `schemas/state-retention.schema.json`
- Consumer tests: `tests/test_state_retention.py` (invokes the literal CLI)

Contract: `local://nvme-implementation-contract.md`; spec:
`local://nvme-contract.md`. This module is deliberately separate from the
hot-slot scheduler (Issue: configurable hot/warm/cold retention policy):
the scheduler applies decisions with fencing and admission accounting; the
policy only maps observations to decisions. No timers, async machinery,
plugin registry, or future-proof abstractions live here.

## Decision document (schema_version 1)

| Field | Meaning |
| --- | --- |
| `hot_evictions` | Session IDs to yield from HOT, in selection order |
| `retention_actions` | Exactly one `{session_id, action, reason}` per observed session, input order; `action` ∈ `keep` / `demote_to_cold` / `evict_snapshot` |
| `prefix_evictions` | Prefix IDs to evict, selection order |
| `rationale` | Human-readable accounting, including any explicit shortfall |
| `effective_config` | The validated config actually applied |

All values are decisions; consumers (the scheduler, the store) perform any
real mutation. `demote_to_cold` changes class metadata only — warm and cold
are classes over the same immutable blobs, never duplicate payload copies;
the warm window and warm-quota overflow both select it.

## Policy rules

1. **No pressure, no eviction.** With `needs_hot_slot`, `uma_pressure` and
   `disk_pressure` all false, nothing leaves HOT no matter how long it has
   been idle: a 20-minute (or 20-day) idle HOT session stays HOT. Every
   window below is a soft target; no age value ever triggers a deletion.
2. **Hot victims.** Only `HOT` + `WAITING` sessions are ever victims.
   `RUNNING`, `SUSPENDING` and `RESTORING` are never eviction victims under
   any pressure, priority, or grace value.
3. **Ranking.** Victims are ranked **lowest priority first, then longest
   idle** (priority 3 background before priority 0 interactive; older
   `last_accessed` before newer). Ties fall back to input order.
4. **Grace with soft priority override.** A HOT waiter becomes eligible
   once idle past `hot_idle_grace_seconds` (default 180 s), **or** when an
   incoming request strictly outranks it (`incoming_priority < session
   priority`) while admission pressure demands a slot, **or** when actual
   `uma_pressure` carries a positive `required_free_bytes` — real UMA byte
   demand overrides grace even with no incoming priority. An interactive
   admission can therefore select a 20-second-idle background waiter;
   ordinary short waits (10–30 s compiler waits) do not churn when there
   is no demand, and a background admission cannot eject an equal-or-
   better-priority short wait.
5. **Sizing.** `needs_hot_slot` is a boolean and is sized as exactly one
   incoming slot. `uma_pressure` and `disk_pressure` are sized by
   `required_free_bytes` against their own pools (hot sessions for UMA;
   prefix entries first, then NVMe snapshots, for disk). Freed numbers in
   the rationale are always sums of observed `state_bytes`. When the
   observation cannot size the decision — boolean demand with no eligible
   victim, `required_free_bytes = 0`, or an unmet byte target — the
   rationale states it explicitly (`unserved`, `no byte target`,
   `shortfall N bytes unmet and not claimed as freed`) instead of
   inventing freed memory. Class demotions free zero disk bytes and are
   never counted in any freed-bytes claim.
6. **Warm window → class demotion only.** A WARM snapshot idle past
   `warm_nvme.preferred_retention_seconds` (default 3600 s) gets
   `demote_to_cold`: a metadata-only class change, bytes untouched. The
   cold window (default 86400 s) triggers **no** automatic action; older
   COLD state stays until justified by quota/LRU/pressure — never a TTL.
7. **Quota accounting: warm demotes, only cold/disk delete.** A warm pool
   over its `max_size_bytes` (512 GB by default) first demotes
   least-recently-accessed warm sessions to cold — metadata only. That
   frees **warm budget, never disk bytes**, and the demoted bytes then
   consume the cold quota. `evict_snapshot` is selected only when the
   cold pool exceeds its `max_size_bytes` (1 TB by default) or when
   `disk_pressure` sets a real byte target; both pick
   least-recently-accessed snapshots first.
8. **Prefixes are independent.** `prefix_cache.preferred_retention_seconds`
   is always `null` by validation: prefix entries never expire by age and
   never inherit conversation windows. Incompatible entries are always
   evicted (identity can never match again); compatible entries only via
   the prefix quota (default 500 GB) or an explicit disk free-bytes
   target, LRU in both cases.

## Configuration (exact defaults, decimal bytes)

| Key | Default |
| --- | --- |
| `hot_idle_grace_seconds` | `180` |
| `warm_nvme.preferred_retention_seconds` | `3600` |
| `warm_nvme.max_size_bytes` | `512000000000` |
| `cold_nvme.preferred_retention_seconds` | `86400` |
| `cold_nvme.max_size_bytes` | `1000000000000` |
| `prefix_cache.preferred_retention_seconds` | `null` (must stay null) |
| `prefix_cache.eviction` | `"LRU"` |
| `prefix_cache.max_size_bytes` | `500000000000` |
| `quick_reactivation_window_seconds` | `30` |

Every key is required and no unknown key is accepted: a typo fails closed
instead of silently falling back to a default. Byte caps must be finite
non-negative integers; timestamps must be finite and no later than the
explicit `now`; priorities must be integers 0–3; session/prefix IDs must be
non-empty and unique; lifecycle and tier must be consistent (`HOT` ⇔
in-UMA lifecycle, `WARM`/`COLD` ⇔ `NVME_RESIDENT`). Any violation exits
nonzero with a message and writes no output file.

`quick_reactivation_window_seconds` is a telemetry observation horizon (see
below), not an eviction trigger.

## Metric denominator conventions (specification, not emitted telemetry)

This CLI emits **no metrics** and contains **no workload numbers**. The
definitions below fix the denominators for when real counters exist;
until then these rates are undefined (`null`), never zero and never
estimated from invented workload data. All windows use one explicit
observation clock.

- **Wake event** (the denominator for hit rates): a logical session
  resuming execution from `WAITING` or `NVME_RESIDENT`. Every wake is
  classified exactly once as hot hit (resumed with resident state), warm
  NVMe hit (restored from a WARM-class snapshot), cold NVMe hit (restored
  from a COLD-class snapshot), or cold miss (no compatible state; explicit
  cold prefill with its true physical token cost). The four classes
  partition wake events, so `hot_hit_rate + warm_nvme_hit_rate +
  cold_nvme_hit_rate + cold_miss_rate = 1` over any window — each rate is
  its class count divided by total wake events in the window.
- **`recompute_ratio`**: `sum(physically_prefilled_tokens) /
  sum(logical_input_tokens)` over paired requests in the window — a
  ratio of sums (token-weighted), not a mean of per-request ratios. The
  denominators come from measured counters; restored token counts alone
  do not imply reuse.
- **`unnecessary_eviction_rate`**: evictions followed by a reactivation of
  the same session within `quick_reactivation_window_seconds` (default
  30 s), divided by evictions whose full observation window has already
  elapsed (**matured** evictions). Evictions near the window boundary are
  **right-censored**: their next use has not been observed yet, so they
  are excluded from the denominator and reported as a separate censored
  count — counting them as "no quick reactivation" would make fresh
  evictions look efficient by construction. If next-use observation is
  unavailable, the rate is `null`, not a guess.
- **Idle-before-next-use / prefill-avoided** distributions use the same
  censoring rule: an event contributes only once its observation horizon
  has passed.

## Smoke

```bash
FIX=$(mktemp -d)
printf '{"schema_version":1,"now":1700000000,"pressure":{"needs_hot_slot":true,"uma_pressure":false,"disk_pressure":false,"required_free_bytes":0},"incoming_priority":0,"sessions":[],"prefix_entries":[]}\n' > "$FIX/obs.json"
PYTHONPATH=src python3 -m mimo_halo.state_tier.retention \
  --config configs/state-retention.json --input "$FIX/obs.json" --output "$FIX/decision.json"
cat "$FIX/decision.json"   # exit 0; rationale says the boolean demand is unserved
```
