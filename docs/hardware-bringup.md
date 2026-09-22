# Hardware bringup runbook — command-complete

Every step is a command or a gate; no step invents a measurement. Paths are
environment variables only:

- `$MIMO_LAB` — the lab workspace root (operator sets it; never hardcoded).
- `$HALO` — the Strix Halo Linux host (operator-provided SSH target).
- `$REPO` — a checkout of this repository.
- `$STRIX_SRC` — pinned strix-llama.cpp source (`upstreams/strix-llama.cpp`).
- `$ROCMFP4_SRC` — pinned rocmfp4 source (`upstreams/rocmfp4`).
- `$MODEL` — the candidate GGUF under evaluation (operator sets it).
- `$MODEL_REF` — a well-supported reference-architecture MoE GGUF under
  `$MIMO_LAB/source-models/` (operator sets it).

Numbered 1–49. Gates are explicit stop conditions; a gate that cannot be
passed honestly blocks the pipeline — it is never waved through.

## A. Mac workstation preparation

1. `export MIMO_LAB=/path/to/lab` — required; every later step fails closed
   without it.
2. `python3 scripts/workspace.py init --min-model-free-bytes 128849018880` (120 GiB floor) → creates
   the layout (mkdir only), reports free bytes, and **blocks model staging**
   when free bytes are short while still creating metadata dirs (exit 1 =
   staging blocked — an expected state on a nearly-full volume, not an error).
3. `python3 scripts/workspace.py status` — read-only; confirm `model_staging`
   and `filesystem_free_bytes` before any staging decision (it never creates
   an absent root).
4. `python3 scripts/workspace.py smoke` — hash round-trip probe: a uniquely
   named file under `caches/` is created exclusively, written, hashed, read
   back, verified and removed again; pre-existing files are never touched.
5. Inspect the external disk read-only:
   `diskutil list && diskutil info "$MIMO_LAB" && df -h "$MIMO_LAB" && mount | grep -i "$(basename "$MIMO_LAB")"`.
   `df -h` is human-readable only — never record it as a byte count. Exact
   free bytes come from `python3 scripts/workspace.py status`
   (`filesystem_free_bytes`, measured via `statvfs`) or, equivalently,
   `diskutil info -plist "$MIMO_LAB" | plutil -extract FreeSpace raw -o - -`.
6. Record the exact free bytes (never `df`'s rounded output) from step 5 into
   a private local note under `$MIMO_LAB/traces/private/hardware-evidence/`.
   Never commit it.
7. **No reformat, no repartition, no rename of an existing volume, ever.**
   If the volume is not writable or too small, stop and re-plan; do not
   delete anything.

## B. Clone/verify pinned sources (already done once; re-verify per checkout)

8. `git -C "$REPO/upstreams/reap" rev-parse HEAD` → must equal
   `1970473c51ca3caeb98c10392f15b3a08a672974`.
9. `git -C "$REPO/upstreams/strix-llama.cpp" rev-parse HEAD` →
   `8c1c282ecb194e8f02613defcc4a07c22b6d1c08`.
10. `git -C "$REPO/upstreams/rocmfp4" rev-parse HEAD` →
    `ddd08770a7f063380c35353bdd7c44d7604106a6`.
11. `git -C "$REPO/upstreams/llama.cpp" rev-parse HEAD` →
    `38a5b42d9a3e82e0a586bcd1caed121f36c87a73` (official reference pin).
12. Cross-check the pins against `manifests/upstreams.json`; any mismatch
    blocks everything downstream.

## C. Mac correctness build (deferred to Main; exact commands)

13. `cd "$STRIX_SRC" && cmake -B build-metal -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_METAL=ON -DGGML_VULKAN=OFF -DGGML_HIP=OFF`
14. `cmake --build build-metal --config Release -j 16`
15. `ctest --test-dir build-metal --output-on-failure`
16. Equivalently, via the driver:
    `python3 scripts/build_runtime.py --source "$STRIX_SRC" --backend metal --run --yes`
    (plan-only by default: `python3 scripts/build_runtime.py --source "$STRIX_SRC" --backend metal`;
    the plan prints each step as an argv list plus env requirements, defaults
    to the per-backend `build-metal` directory, and installs no tools).
17. Mac-side purpose: CPU/Metal correctness sanity of codecs and tooling.
    gfx1151 performance truth lives on the Halo box only; HIP is not
    buildable on macOS.

## D. Halo Linux box — inventory capture (read-only)

18. Write the command list once, e.g. `$REPO/configs/hw-commands.txt`:

    ```text
    uname -a
    lscpu
    free -b
    lsblk
    lspci -nn -k
    dmidecode -t bios            # optional, sudo-marked
    dmidecode -t memory          # optional, sudo-marked
    smartctl -a /dev/nvme0n1     # optional, sudo-marked
    ```

19. `python3 scripts/hardware_capture.py --commands "$REPO/configs/hw-commands.txt"`
    — raw output lands in `$MIMO_LAB/traces/private/hardware-evidence/`
    (identifiers intact, stays local); a sanitized public summary is printed
    (MACs/IPs/UUIDs/serials/hostnames/home-paths replaced by
    `__KIND_n__` tokens).
20. sudo lines are run as written by the operator when they choose to
    authorize them; the tool never injects or prompts for sudo
    automatically — failed sudo lines are recorded as `failed`, explicitly.
21. **Gate D (BIOS/firmware before change):** record UMA split, GTT sizing
    (`amdgpu.gttsize`, `ttm.pages_limit`), power profile, IOMMU, SecureBoot
    state. One change at a time, each with before/after/benchmark.
22. Disk safety on the Halo box: benchmark SSDs with file-based fio reads
    only —
    `fio --name=ssd-read --filename="$MIMO_LAB/scratch/fio-probe.bin" --rw=read --bs=1M --direct=1 --size=1G --allow_file_create=0`
    (create the probe file first by copying an existing payload; never a raw
    device target; never a write workload against read-only-safe media).
23. **Checksum sync, no deletion:** replicate a payload directory Mac↔Halo —
    `rsync -a --checksum "$MIMO_LAB/datasets/" "$HALO:$MIMO_LAB/datasets/"`
    (explicitly NOT `--delete`; repeat per payload directory) — then verify
    on the receiving side with
    `cd "$MIMO_LAB/datasets" && sha256sum -c checksums.txt`.
    Mac copy is retained as the master until a promotion decision says
    otherwise.
24. Standard layout on the Halo box mirrors step 2: same
    `python3 scripts/workspace.py init` against the Halo-side `$MIMO_LAB`.

## E. Backend builds on the Halo box

25. Vulkan (default recommendation): `cd "$STRIX_SRC" &&
    cmake -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_VULKAN=ON -DGGML_METAL=OFF -DGGML_HIP=OFF &&
    cmake --build build-vulkan --config Release -j 16`.
26. `ctest --test-dir build-vulkan --output-on-failure`
    — or `python3 scripts/build_runtime.py --source "$STRIX_SRC" --backend vulkan --run --yes`.
27. HIP (control lane): `cd "$STRIX_SRC" && HIPCXX="$(hipconfig -l)/clang"
    HIP_PATH="$(hipconfig -R)" cmake -B build-hip -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_HIP=ON
    -DGPU_TARGETS=gfx1151 -DGGML_METAL=OFF -DGGML_VULKAN=OFF && cmake --build build-hip --config Release -j 16`.
28. Every HIP-path run sets `HIP_LAUNCH_BLOCKING=1` (and
    `HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1` where
    the ROCmFP4 quickstart says so). This is a documented correctness
    control (README:76-82, CI config), not a performance claim.
29. CPU reference build: `cd "$STRIX_SRC" && cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_HIP=OFF &&
    cmake --build build-cpu --config Release -j 16` (AVX-512 Zen 5 baseline).

## F. Reference-model sanity ladder (bare-metal, before any virtualization)

30. Tiny GGUF sanity run on the CPU build:
    `"$STRIX_SRC/build-cpu/bin/llama-cli" -m "$MIMO_LAB/source-models/tiny.gguf" -p "sanity" -n 8`.
31. Same tiny GGUF on Vulkan:
    `"$STRIX_SRC/build-vulkan/bin/llama-cli" -m "$MIMO_LAB/source-models/tiny.gguf" -p "sanity" -n 8 -ngl 999`.
32. Same tiny GGUF on HIP with the control env from step 28:
    `HIP_LAUNCH_BLOCKING=1 HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1 "$STRIX_SRC/build-hip/bin/llama-cli" -m "$MIMO_LAB/source-models/tiny.gguf" -p "sanity" -n 8 -ngl 999`.
33. Reference-architecture performance point (`$MODEL_REF`) on Vulkan, then
    HIP-with-control; record both, never mix lanes:
    `"$STRIX_SRC/build-vulkan/bin/llama-bench" -m "$MODEL_REF" -p 8192`,
    then
    `HIP_LAUNCH_BLOCKING=1 HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1 "$STRIX_SRC/build-hip/bin/llama-bench" -m "$MODEL_REF" -p 8192`.
34. MiMo-architecture smoke: convert the real trunk checkpoint
    (`MiMoV2FlashForCausalLM` → `mimo2`, upstream conversion flow) into
    `"$MIMO_LAB/source-models/mimo2-trunk.gguf"`, then run
    `"$STRIX_SRC/build-vulkan/bin/llama-cli" -m "$MIMO_LAB/source-models/mimo2-trunk.gguf" -p "sanity" -n 16 -ngl 999`
    and
    `"$STRIX_SRC/build-vulkan/bin/llama-perplexity" -m "$MIMO_LAB/source-models/mimo2-trunk.gguf" --file "$MIMO_LAB/datasets/perplexity-slice.txt" -ngl 999`
    on a small slice; record actual expert count/layer count/top-k from the
    GGUF metadata and compare with the hypothesis ledger in
    `docs/runtime-audit.md` §5.
35. Gate E (1% calibration gate): run the observation pipeline on ~1% of the
    planned calibration stream end-to-end; verify hidden-state persistence,
    resumability (`--resume`), seed control, and that per-layer statistics
    are complete and finite before scaling up.
36. Gate F (10% pass): same pipeline at 10% with
    `--sample-fraction 0.1`; check memory ceiling, wall time projection, and
    statistical stability of routing frequencies against the 1% run.
37. Both gates pass → proceed; either fails → stop, diagnose, re-run that
    gate. Never skip a gate.

## G. Benchmarks, burst replay, soak

38. Prompt-processing matrix (Vulkan lane):
    `python3 scripts/benchmark_matrix.py --mode bench --tool "$STRIX_SRC/build-vulkan/bin/llama-bench" --model "$MODEL" --output "$MIMO_LAB/metrics/pp-vulkan.json"`.
    These are in-process batched prompt-processing numbers (PP8K/32K/64K/128K)
    — never labelled as concurrency.
39. Concurrency matrix (Vulkan lane, real parallel requests):
    `python3 scripts/benchmark_matrix.py --mode serve --server "$STRIX_SRC/build-vulkan/bin/llama-server" --model "$MODEL" --output "$MIMO_LAB/metrics/c-vulkan.json"`.
    C1/C2/C4/C8 are actual concurrent HTTP requests against one server;
    aggregate tok/s comes from server-reported token counts over the wave
    wall time.
40. Repeat steps 38–39 on the HIP lane (tools from
    `$STRIX_SRC/build-hip/bin/`, outputs `$MIMO_LAB/metrics/pp-hip.json` and
    `$MIMO_LAB/metrics/c-hip.json`) with `HIP_LAUNCH_BLOCKING=1` in the
    environment (the runner sets the config's runtime controls
    automatically); report HIP numbers with the control explicitly on.
41. Any failed/timed-out point is explicit in the report (`status` field) and
    makes the runner exit nonzero — a partial matrix is never presented as
    complete.
42. Bursty replay — plan first (no model calls):
    `python3 scripts/burst_replay.py --schedule timings.json`
43. Bursty replay — live:
    `python3 scripts/burst_replay.py --schedule timings.json --server-url http://127.0.0.1:18080/completion`.
    The schedule must come from real sanitized timing input or local
    trace-derived timings; the tool invents no performance and echoes the
    schedule into the report.
44. Soak ladder (Vulkan primary): 1 h → 6 h → overnight C8 under the bursty
    schedule; watch crashes, GPU resets, memory growth, throttling, slot
    leakage. **Release blockers: C1 correctness canary and C8 stability
    canary.**
45. MTP controls: C1 aggressive, C2 adaptive, C4 measured, C8 off initially —
    recorded per config, never all-at-once.
46. Full candidate matrix (once pruned/quantized artifacts exist): each
    candidate gets its own size, C1, C8, and long-task quality rows; Pareto
    annotation is over C8 vs long-task success with size labels.

## H. Hygiene

47. Everything public (docs, manifests, benchmark JSONs) carries no personal
    paths, hostnames, serials, MACs, or network identifiers; capture raw
    output stays in `$MIMO_LAB/traces/private/`.
48. Storage claims are measured: exact free bytes come from the workspace
    status (`filesystem_free_bytes`) or equivalent byte-level tools — never
    `df`'s rounded display and never nominal bpw math; model-byte budgets
    from real byte sums.
49. Any change to the hardware stack (BIOS, kernel params, driver, backend
    build) re-runs steps 21 → 27 → 33 before new numbers are accepted.
