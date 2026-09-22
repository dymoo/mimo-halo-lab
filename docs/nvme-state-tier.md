# NVMe-backed inference state tier

Design document for the state-tier work tracked under epic [#59](https://github.com/dymoo/mimo-halo-lab/issues/59). Vocabulary is canonical in the root [`CONTEXT.md`](../CONTEXT.md); this document carries the design, the seams, and the links to the tracked Issues. The full approved spec lives in the epic body (published verbatim from the spec contract); the dependency graph is recorded natively on GitHub and mirrored in [`manifests/nvme-issues.json`](../manifests/nvme-issues.json).

## Problem and invariant

A 128 GB UMA machine must serve more long-running logical coding agents than its C1–C8 physically resident decoding slots. Agents alternate between generation and tool waits; the 4 TB Samsung 990 Pro is a first-class context/state tier, not just artifact storage. Its 7–8 GB/s is a planning estimate until measured — disk bandwidth is not UMA capacity or latency.

**Critical invariant:** restoring a context must preserve actual prefix reuse. 100K history + 4K new tokens must physically prefill ≈4K, not 104K. An HTTP 200 restore is not acceptance evidence; every bounded boundary-token/tail replay is measured and explained.

## Design

### Lifecycle

`RUNNING → WAITING → SUSPENDING → NVME_RESIDENT → RESTORING → RUNNING`. Short waits can return directly to `RUNNING`. RUNNING slots are never eviction victims. A logical-session generation cannot change during snapshot capture; a slot frees only after complete state plus metadata are durably committed. Restore validates compatibility/checksum before admitting suffix computation. Failure never silently substitutes a full prefill — cold reconstruction is reported explicitly as a **cold miss** with its true physical token cost. Warm/cold retention class is orthogonal to `NVME_RESIDENT`; class changes never rewrite payload.

### Modules and seams (tested at the highest practical boundary)

| Module | Responsibility | Seam |
|---|---|---|
| Runtime continuation | Real suspend/restore/continue on the pinned server + physical-token counters | Actual HTTP/native server continuation: output equivalence **and** physical-prefill count **and** hot-state release |
| State store | Commit/load/inspect immutable snapshots with typed compatibility identity + checksums | Atomic publication, checksum/incompatibility refusal, crash windows, namespace/path safety |
| Retention policy | Pure deterministic keep/suspend/promote/demote/expire decisions from observations + pressure | Deterministic pressure/priority/time scenarios (no disk or scheduler side effects) |
| Hot-slot scheduler | Logical sessions ↔ finite physical slots; generation fencing, priority, admission-memory accounting | Real workload behavior: wake latency, recomputation, decode throughput |

First implementation is blocking and test-first. No plugin registry, no interchangeable-storage framework, no token-level SSD KV paging, no MTP dependency.

### Storage and privacy

Target layout under `/mnt/ai/`: `slots/blobs/`, `slots/active-index/`, `prefix-cache/`, plus the existing model/calibration/eval trees. Slot snapshots and prefix blobs contain private conversation state and **never enter Git**. Opaque IDs, path confinement, private permissions, security-principal namespaces; cross-principal prefix sharing only for explicitly public immutable prefixes. Commit order: temp blob → fsync → atomic rename → fsync parent → durable index reference. Last-access metadata lives in the index, never in payload mutation.

### Compatibility identity

Strong content/config identity: model bytes/hash + quant revision, tokenizer + chat template, runtime commit + local patch/cache-format identity, token sequence, semantic cache config (context/KV precision/RoPE/SWA/etc.), security namespace. Incompatible state is rejected, never reused by assumption. A retention-policy change alone must not invalidate valid numerical state.

### Retention policy (dedicated Issue — not scheduler timers)

Configurable **soft** defaults, decimal bytes: hot idle grace 180s (tuning target 3–5 min); warm preferred 1h / 512 GB cap; cold preferred 24h / 1 TB cap; immutable prefix cache no time expiry, LRU / 500 GB cap; quick-reactivation window 30s. Decisions weigh actual UMA pressure, hot-slot admission demand, disk pressure, priority, expected wakeup, state size/restore cost, prefix reuse value. Priority order: interactive Dylan agents > high-priority local jobs > normal users > background. Grace never overrides admission priority or active-decode protection.

### Metrics

First-class: `logical_input_tokens`, `physically_prefilled_tokens`, `prefix_cache_hit_tokens`, `restored_state_tokens`, `state_save_bytes`, `state_restore_bytes`, `state_save_ms`, `state_restore_ms`, `suffix_prefill_ms`, `cold_reprefill_ms_equivalent`, `slot_evictions`, `slot_restores`, `failed_restores`, `restore_checksum_failures`, `nvme_state_bytes_written_total`.

Derived: `recompute_ratio = physical_prefill / logical_input`, `hot_hit_rate`, `warm_nvme_hit_rate`, `unnecessary_eviction_rate` — each with a stated denominator and explicit treatment of right-censored (still-unobserved) evictions.

### Async evolution and endurance

Blocking and correct first. Async capture/serialization/prefetch, preallocated buffers, io_uring only if measured useful. Admission accounts for transient buffers; stale completions fenced by generation. Write only meaningful changed state — unchanged generation/content reuses the durable snapshot; count actual host bytes written (distinct from NAND amplification); delete stale blobs only after references are safely removed.

## Source evidence (pinned runtime `8c1c282ecb194e8f02613defcc4a07c22b6d1c08`)

Audit artifacts: `agent://NvmeServerAudit` (tools/server), `agent://NvmeStateAudit` (core state serialization), `agent://NvmeTestModel` (CI fixtures; payload is a JSON-encoded string — decode before use). Pin: [`manifests/upstreams.json`](../manifests/upstreams.json). Key facts, with actual source paths:

- **The full-reprefill defect:** `SLOT_RESTORE` clears `slot->prompt` including `prompt.checkpoints` (`upstreams/strix-llama.cpp/tools/server/server-context.cpp:2874`), and the file carries no checkpoint payload. The SWA prefill gate then finds an empty checkpoint list and runs `do_reset` → `n_past = 0` (`server-context.cpp:3617-3621`), so a restored slot on default-windowed SWA reprocesses ~104K tokens while returning HTTP 200. The RAM cache path keeps checkpoints (asymmetric with disk); the existing test skips this case (`tools/server/tests/unit/test_slot_save.py:217`).
- **Physical counters exist:** per-slot `n_prompt_processed` / `timings.prompt_n` (`server-context.cpp:4353`), Prometheus `llamacpp:prompt_tokens_total` (`tools/server/server-task.cpp:1538-1585`), `GET /slots` processed/cache fields (`server-context.cpp:733-735`).
- **Format/compat gates:** `GGSQ` v3 / `GGSN` v10 (`upstreams/strix-llama.cpp/include/llama.h:48-49`), exact-match flags, **no checksum in the format** — the store must supply its own digest. Full-context durable snapshots require `LLAMA_STATE_SEQ_FLAGS_NONE` (0); `PARTIAL_ONLY` skips global KV (`src/llama-kv-cache-iswa.cpp:259-273`); `ON_DEVICE` is per-context and must never back NVMe blobs (`llama.h:925-937`).
- **Identity is absent from blobs:** load validates an arch string only (`src/llama-context.cpp:3309-3352`); model/quant/tokenizer/template/RoPE/SWA/tenant metadata must be enforced server/store-side.
- **Durability gap:** state files are written with plain `fwrite`, no fsync/rename (`src/llama-mmap.cpp:358-366`) — atomic publication is the store's job.
- **Thread safety:** all slot/KV mutation runs on the single scheduler thread; save/restore are declined while yielding (`tools/server/server-queue.cpp:222-249`). Async work needs per-slot epoch/generation fencing (`server-context.cpp:2780-2922` defer semantics; `pos_min == -1` abort guard at `:3546-3553`).
- **Native reuse points:** RAM `server_prompt_cache` (`tools/server/server-task.h:617-640`) and `common_prompt_checkpoint` (`common/common.h:1205-1256`) — extend these seams, don't invent parallel wrappers.

## Issue graph (epic #59 and 16 tracer children)

Native sub-issue + blocked-by edges live on GitHub; this table mirrors them (plan IDs from the approved ticket plan → discovered GitHub numbers).

| Plan | Issue | Title | Blocked by |
|---|---|---|---|
| 1 | [#60](https://github.com/dymoo/mimo-halo-lab/issues/60) | Audit: llama.cpp slot serialization for MiMo/SWA | existing #25 |
| 16 | [#61](https://github.com/dymoo/mimo-halo-lab/issues/61) | Configurable hot/warm/cold retention policy (dedicated module) | 60 |
| 4 | [#62](https://github.com/dymoo/mimo-halo-lab/issues/62) | Slot-store module — atomic opaque snapshot commit/load/inspect | 60 |
| 2 | [#63](https://github.com/dymoo/mimo-halo-lab/issues/63) | Save/restore correctness harness at the real-server seam | 60 |
| 5 | [#64](https://github.com/dymoo/mimo-halo-lab/issues/64) | Snapshot metadata, versioning and checksum gates | 62 |
| 3 | [#65](https://github.com/dymoo/mimo-halo-lab/issues/65) | Persist complete MiMo target slot state (checkpoints + SWA) | 63 |
| 6 | [#66](https://github.com/dymoo/mimo-halo-lab/issues/66) | Basic blocking save/restore end-to-end with first-class metrics | 64, 65 |
| 7 | [#67](https://github.com/dymoo/mimo-halo-lab/issues/67) | Benchmark restore versus cold/hot prefill (8K–256K × A–F) | 63, 66 · `hardware-needed` |
| 8 | [#68](https://github.com/dymoo/mimo-halo-lab/issues/68) | Persistent prefix cache with namespace identity | 64, 66 |
| 9 | [#69](https://github.com/dymoo/mimo-halo-lab/issues/69) | Priority-aware hot/warm slot scheduler (policy delegated) | **61**, 66 |
| 10 | [#70](https://github.com/dymoo/mimo-halo-lab/issues/70) | Asynchronous save pipeline with generation fencing | 66, 67 |
| 14 | [#71](https://github.com/dymoo/mimo-halo-lab/issues/71) | Endurance and write-amplification metrics | 62 |
| 13 | [#72](https://github.com/dymoo/mimo-halo-lab/issues/72) | Restart and crash recovery for sessions/index | 64, 66, 69 |
| 11 | [#73](https://github.com/dymoo/mimo-halo-lab/issues/73) | Asynchronous restore/prefetch | 67, 69 |
| 12 | [#74](https://github.com/dymoo/mimo-halo-lab/issues/74) | C8 + 32-dormant-agent stress benchmark | 67, 69, 70, 73 · `hardware-needed` |
| 15 | [#75](https://github.com/dymoo/mimo-halo-lab/issues/75) | Optimize measured tiering bottlenecks | 67, 71, 74 · `hardware-needed` |

The retention policy (#61) is deliberately its **own Issue** and a native prerequisite of the scheduler (#69) — no ad hoc TTL timers inside the scheduler. `hardware-needed` applies only to the physical 990 Pro/Strix measurement Issues (67, 74, 75); software contracts and small-model regressions proceed on the Mac.

## Testing seams

- **Server continuation:** uninterrupted hot vs save → destroy/free → restore → continue; deterministic outputs; measured physical prefill for 100K+4K must be suffix-scale; hot memory verifiably freed. Tiny-model CI first (`ggml-org/test-model-stories260K`, `stories15M_MOE`, `tinygemma3-GGUF`, zero-download `tests/test-llama-archs.cpp` fixtures), MiMo/SWA regression once available.
- **Store:** atomic publication, checksum/incompatibility refusal, crash windows, namespace/path safety.
- **Policy:** no-pressure 20-min keep-hot; pressure/high-priority 20s eligible eviction; 10–30s waits don't churn; unlimited-time prefix retention; budgets/LRU and restart-clock behavior.
- **Benchmark:** 8K/32K/64K/128K/practical 256K × A cold / B hot / C restore+0 / D restore+1K / E restore+8K / F restart+restore. Unobservable GPU time reported as null, never wall time renamed.

## Milestones (epic checklists)

- **First:** 128K+ context durably suspended to the 990 Pro, hot state demonstrably freed, restored with suffix-only physical prefill, deterministic continuation, measured wake advantage.
- **Final:** C8 physical slots serve 32+ logical long-running agents under priority/pressure-aware tiering, with measured recomputation reduction and no unacceptable quality or active-throughput regression.

## Links

- Epic/spec: [#59](https://github.com/dymoo/mimo-halo-lab/issues/59) (full approved spec, `ready-for-human`, excluded from AFK execution)
- Issue graph manifest: [`manifests/nvme-issues.json`](../manifests/nvme-issues.json)
- Glossary: [`CONTEXT.md`](../CONTEXT.md)
- Runtime audit: [`docs/runtime-audit.md`](runtime-audit.md); runtime pin: [`manifests/upstreams.json`](../manifests/upstreams.json)
- Existing P0 roadmap: parent spec [#1](https://github.com/dymoo/mimo-halo-lab/issues/1); runtime prerequisite [#25](https://github.com/dymoo/mimo-halo-lab/issues/25)
- Public-release intent: [#54](https://github.com/dymoo/mimo-halo-lab/issues/54) (aggregate NVMe metrics only, never private state blobs)
