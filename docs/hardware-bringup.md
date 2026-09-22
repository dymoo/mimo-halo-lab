# Hardware bringup runbook — command-complete

Every step is a command or a gate; no step invents a measurement. Paths are
environment variables only:

- `$MIMO_LAB` — the lab workspace root (operator sets it; never hardcoded).
- `$HALO` — the Strix Halo Linux host (operator-provided SSH target).
- `$REPO` — a checkout of this repository.
- `$STRIX_SRC` — pinned strix-llama.cpp source (`upstreams/strix-llama.cpp`).
- `$ROCMFP4_SRC` — pinned rocmfp4 source (`upstreams/rocmfp4`).
- `$CANDIDATE_DIR` — the selected candidate checkpoint directory, derived from
  the pinned original Xiaomi weights (operator sets it; it contains the
  candidate `config.json`, tensor shards, and tokenizer auxiliary files).
- `$CANDIDATE_ID` — stable operator-chosen name for the selected candidate
  GGUF output (no path or host identity).
- `$MODEL` — the candidate GGUF under evaluation (operator sets it).
- `$MODEL_REF` — a well-supported reference-architecture MoE GGUF under
  `$MIMO_LAB/source-models/` (operator sets it).

Numbered 1–49. Gates are explicit stop conditions; a gate that cannot be
passed honestly blocks the pipeline — it is never waved through.

Arrival readiness state (each asset/config/command with its evidence class,
proof, missing prerequisite, next command, and the first-Halo sequence) lives
in `manifests/hardware/readiness.json`; evidence classes are `validated-on-Mac`,
`static-checked-for-Linux`, `untested-on-Halo`. The Mac prep box has 48 GiB
RAM: tiny probes and layer-streamed/bounded scratch only — no full candidate
or full-model materialization here; memory-heavy runs coordinate with
MiMoObservationRun. The 128 GB target is the Halo, and every Halo row stays
`untested-on-Halo` until SSH exists.

## A. Mac workstation preparation

1. `export MIMO_LAB=/path/to/lab` — required; every later step fails closed
   without it.
2. `python3 "$REPO/scripts/workspace.py" init --min-model-free-bytes 128849018880` (120 GiB floor) → creates
   the layout (mkdir only), reports free bytes, and **blocks model staging**
   when free bytes are short while still creating metadata dirs (exit 1 =
   staging blocked — an expected state on a nearly-full volume, not an error).
3. `python3 "$REPO/scripts/workspace.py" status` — read-only; confirm `model_staging`
   and `filesystem_free_bytes` before any staging decision (it never creates
   an absent root).
4. `python3 "$REPO/scripts/workspace.py" smoke` — hash round-trip probe: a uniquely
   named file under `caches/` is created exclusively, written, hashed, read
   back, verified and removed again; pre-existing files are never touched.
5. Inspect the external disk read-only:
   `diskutil list && diskutil info "$MIMO_LAB" && df -h "$MIMO_LAB" && mount | grep -i "$(basename "$MIMO_LAB")"`.
   `df -h` is human-readable only — never record it as a byte count. Exact
   free bytes come from `python3 "$REPO/scripts/workspace.py" status`
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
    blocks everything downstream. A fresh checkout from the manifest
    `fetch_method` carries NO local patches — apply and confirm each of the
    four patches listed for `strix-llama.cpp`, then compare each patch-file
    SHA-256 with its manifest entry (patch-file vs applied-diff hashes differ
    by the explanatory header; compare file hashes). The tokenizer NFC patch
    requires ICU4C `uc`; install/configure that dependency before building:

    ```sh
    for p in "$REPO"/patches/strix-*.patch; do
      git -C "$STRIX_SRC" apply --reverse --check "$p" 2>/dev/null ||
        git -C "$STRIX_SRC" apply "$p"
      git -C "$STRIX_SRC" apply --reverse --check "$p" || exit 1
      shasum -a 256 "$p"   # Linux: sha256sum "$p" — must equal manifests/upstreams.json
    done
    ```

## C. Mac correctness build (deferred to Main; exact commands)

13. On Mac, install the pinned ICU development package with `brew install icu4c@78`, then configure the pinned source with the explicit Homebrew prefix:
    ```sh
    cd "$STRIX_SRC" && cmake -B build-metal -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_METAL=ON -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DICU_ROOT=/opt/homebrew/opt/icu4c@78 -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/icu4c@78
    ```
    ICU4C component `uc` is REQUIRED by the NFC patch: CMake fails
    configuration when ICU is missing. Do not disable it or silently fall
    back to incomplete normalization tables.
14. `cmake --build build-metal --config Release -j 16`
15. `ctest --test-dir build-metal --output-on-failure`
16. Equivalently, via the driver:
    `ICU_ROOT=/opt/homebrew/opt/icu4c@78 CMAKE_PREFIX_PATH=/opt/homebrew/opt/icu4c@78 python3 "$REPO/scripts/build_runtime.py" --source "$STRIX_SRC" --backend metal --run --yes`
    (plan-only by default: `ICU_ROOT=/opt/homebrew/opt/icu4c@78 CMAKE_PREFIX_PATH=/opt/homebrew/opt/icu4c@78 python3 "$REPO/scripts/build_runtime.py" --source "$STRIX_SRC" --backend metal`;
    the plan prints each step as an argv list plus env requirements, defaults
    to the per-backend `build-metal` directory, and installs no tools).
17. Mac-side purpose: CPU/Metal correctness sanity of codecs and tooling.
    gfx1151 performance truth lives on the Halo box only; HIP is not
    buildable on macOS.

## D. Halo Linux box — inventory capture (read-only)

18. Write the command list once (file is operator-created, not shipped).
    One `printf` line so copy-paste works from any indentation — no heredoc
    `EOF` to mis-terminate:

    ```sh
    printf '%s\n' 'uname -a' 'lscpu' 'free -b' 'lsblk' 'lspci -nn -k' 'dmidecode -t bios            # optional, sudo-marked' 'dmidecode -t memory          # optional, sudo-marked' 'smartctl -a /dev/nvme0n1     # optional, sudo-marked' > "$REPO/configs/hw-commands.txt"
    ```

19. `python3 "$REPO/scripts/hardware_capture.py" --commands "$REPO/configs/hw-commands.txt"`
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
    `python3 "$REPO/scripts/workspace.py" init` against the Halo-side `$MIMO_LAB`.

## E. Backend builds on the Halo box

    The Strix Halo build also requires the standard ICU development package
    (for Debian/Ubuntu: `sudo apt-get install libicu-dev pkg-config`; install
    the distribution's ICU development package on other Linux systems). Set
    `ICU_PREFIX` to the actual ICU installation prefix and pass it to every
    CMake configure below; for Debian/Ubuntu:

    ```sh
    ICU_PREFIX=$(pkg-config --variable=prefix icu-uc)
    test -n "$ICU_PREFIX" || { echo "ICU prefix unavailable"; exit 1; }
    export ICU_PREFIX
    ```

    ICU4C `uc` is required, not optional. The patched CMake configuration
    fails if ICU cannot be found; do not disable NFC or silently fall back.
25. Vulkan (default recommendation): `cd "$STRIX_SRC" &&
    cmake -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_VULKAN=ON -DGGML_METAL=OFF -DGGML_HIP=OFF -DICU_ROOT="$ICU_PREFIX" -DCMAKE_PREFIX_PATH="$ICU_PREFIX" &&
    cmake --build build-vulkan --config Release -j 16`.
26. `ctest --test-dir build-vulkan --output-on-failure`
    — or `ICU_ROOT="$ICU_PREFIX" CMAKE_PREFIX_PATH="$ICU_PREFIX" python3 "$REPO/scripts/build_runtime.py" --source "$STRIX_SRC" --backend vulkan --run --yes`.
27. HIP (control lane): `cd "$STRIX_SRC" && HIPCXX="$(hipconfig -l)/clang"
    HIP_PATH="$(hipconfig -R)" cmake -B build-hip -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_HIP=ON
    -DGPU_TARGETS=gfx1151 -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DICU_ROOT="$ICU_PREFIX" -DCMAKE_PREFIX_PATH="$ICU_PREFIX" && cmake --build build-hip --config Release -j 16`.
28. Every HIP-path run sets `HIP_LAUNCH_BLOCKING=1` (and
    `HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1` where
    the ROCmFP4 quickstart says so). This is a documented correctness
    control (README:76-82, CI config), not a performance claim.
29. CPU reference build: `cd "$STRIX_SRC" && cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DICU_ROOT="$ICU_PREFIX" -DCMAKE_PREFIX_PATH="$ICU_PREFIX" &&
    cmake --build build-cpu --config Release -j 16` (AVX-512 Zen 5 baseline).

## F. Reference-model sanity ladder (bare-metal, before any virtualization)

30. Stage the synthetic graph smoke fixture — NOT a known-good quality
    model, compatibility proof only (identity anchored in
    `manifests/hardware/readiness.json`), and never overwrite an existing
    target without hash identity:

    ```sh
    F="$REPO/scratch/loadability/tiny_b_dense.gguf"   # sha256 04399ecf52904ce316e06ea4b73d3820793ee07c0c9eff2d43efd735386d9647
    T="$MIMO_LAB/source-models/tiny.gguf"
    if [ -e "$T" ]; then cmp -s "$F" "$T" || { echo "tiny.gguf exists with different identity — verify before replacing"; exit 1; }; else cp -n "$F" "$T"; fi
    shasum -a 256 "$T"   # Linux: sha256sum "$T" — must equal the hash above
    ```

    CPU sanity run (flag set proven on the pinned build — this pin REJECTS
    `--no-conversation`; use `--single-turn`):
    `"$STRIX_SRC/build-cpu/bin/llama-cli" --model "$MIMO_LAB/source-models/tiny.gguf" --offline --ctx-size 512 --n-gpu-layers 0 --threads 1 --seed 42 --temp 0 --n-predict 8 --single-turn --simple-io --prompt test`
    (exit 0, 8 generated tokens, ftype F16 — graph compatibility proof only;
    no model-quality or Halo speed claim).
31. Same tiny GGUF on Vulkan — identical command with
    `"$STRIX_SRC/build-vulkan/bin/llama-cli"` and `--n-gpu-layers 999`.
32. Same tiny GGUF on HIP with the control env from step 28 — identical
    command with `"$STRIX_SRC/build-hip/bin/llama-cli"`, `--n-gpu-layers 999`,
    and `HIP_LAUNCH_BLOCKING=1 HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1`
    prefixed.
33. Reference-architecture performance point (`$MODEL_REF`) on Vulkan, then
    HIP-with-control; record both, never mix lanes:
    `"$STRIX_SRC/build-vulkan/bin/llama-bench" -m "$MODEL_REF" -p 8192`,
    then
    `HIP_LAUNCH_BLOCKING=1 HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1 "$STRIX_SRC/build-hip/bin/llama-bench" -m "$MODEL_REF" -p 8192`.
34. MiMo selected original-derived native candidate conversion + candidate-gated
    resident smoke. The native converter patch is applied by step 12 and
    supports packed native MXFP4 expert payloads paired with their E8M0 scale
    siblings. Convert only the selected candidate checkpoint:

    ```sh
    mkdir -p "$MIMO_LAB/quantized-models"
    PYTHONPATH="$STRIX_SRC/gguf-py" \
      "$REPO/.venv/bin/python" "$STRIX_SRC/convert_hf_to_gguf.py" \
      "$CANDIDATE_DIR" \
      --outfile "$MIMO_LAB/quantized-models/$CANDIDATE_ID.gguf" \
      --outtype f16
    ```
    After successful conversion, use that output for all resident commands:

    ```sh
    export MODEL="$MIMO_LAB/quantized-models/$CANDIDATE_ID.gguf"
    ```

    `$CANDIDATE_DIR` must be a candidate derived from the pinned original
    Xiaomi checkpoint; record its selection/provenance separately. Do **not**
    make a full-original `mimo2-trunk.gguf` conversion a preparation step or
    resident-load it. The original source remains the layer-streamed ORIGINAL
    quality reference, and original-quality numbers come from that exact
    streamed scorer. If the selected checkpoint contains
    `second_gen_affine` expert tensors, this GGUF conversion is explicitly
    unsupported: the converter must refuse them rather than mislabel or
    requantize them, and the exact layer-streamed scorer is required.

    The supported native path repacks packed MXFP4 expert values losslessly
    into GGML `block_mxfp4` tensors and labels the output
    `MOSTLY_MXFP4_MOE`; this is format/converter evidence, not model-quality
    evidence. On the Mac, the observed proof is limited to synthetic
    `tiny_a` (6 experts) and non-power-of-two/pruned `tiny_b` (5 experts,
    top-k 2): the converted files are
    `scratch/loadability/tiny_a-native.gguf`
    (84,913,760 B; sha256
    `8237e9508ef07b17e02ac360bdabe5a63943cca77a979e11f69abccfb81a79ff`)
    and `scratch/loadability/tiny_b-native.gguf`
    (84,833,888 B; sha256
    `099e8be9f1305063a772e5489f5c00a54aca1ba6fc44208c546edcec6ad4d808`).
    These regenerated files include NFC metadata. Earlier hashes
    `0a7ff932...` and `cc005a09...` identify pre-NFC files only. Their
    CPU-generated eight-token smoke proves native format/graph only, not NFC
    behavior.

    The converter's existing tokenizer warning was observed and not suppressed
    or changed. Before the NFC patch, raw parity matched 12/13 broader cases;
    composed `U+00E9` and decomposed `U+0065 U+0301` both mapped to `[963]`
    in the HF source, while the pinned tokenizer mapped the decomposed form
    to `[68, 53839]`. The source-correct
    `tokenizer.ggml.normalizer.nfc` metadata and ICU-backed path pass bounded
    Mac regression coverage:

    ```sh
    env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -W error::ResourceWarning -m unittest tests.test_tokenizer_nfc -v
    ```

    Main observed the pre-rebuild red case (five decomposed spellings differed)
    and the earlier post-rebuild **4/4 passed** in 20.6 s. The final suite on
    corrected patch `03e4e745f1644eb9c1923b6c4d40ff90664ada36772a864603707f243ca92448`
    passes **6/6**, including valid NFC, malformed-byte red-to-green behavior,
    the legacy control, and refusal of non-BPE GGUF writing. For malformed
    `b"\xc3("`, rebuilt `llama-tokenize` uses HF-compatible U+FFFD
    replacement semantics; the no-NFC control is unchanged.

    The installed-consumer smoke also passed configure/build: a temporary
    static `libllama` and downstream CMake `find_package(llama)` consumer
    calling `llama_tokenize` compiled and linked successfully (no consumer
    runtime invocation); `pkg-config --static --libs llama`
    includes `-licuuc -licudata`. Current tiny_a/tiny_b native GGUF hashes
    remain unchanged, composed and decomposed `é` both produce token ID
    `[963]`, and actual CPU `llama-cli` smoke runs with a valid test prompt on
    both artifacts exited 0.
    These are bounded synthetic Mac proofs only, not real-candidate quality
    or Halo evidence. Real-candidate conversion/quality, measured resident
    memory, and Halo build/parity/runtime proof remain pending. Do not apply
    a blind tokenizer/regex workaround.

    Keep three quantities distinct in every report:

    - **Native storage/recipe band:** the accepted text-weight window
      `budget.window_bytes = [94704028877, 98569499443]` (90 GiB ±2%) from
      exact candidate accounting. It is not the unpruned original source
      size, the GGUF file size, or a runtime-memory measurement.
    - **Converted payload:** `--outtype f16` leaves native MXFP4 expert
      blocks lossless, but expands non-expert FP8 E4M3 qkv/dense payloads to
      F16 at approximately 2x their stored bytes; BF16 non-experts become
      F16 at approximately 1x, while one-dimensional norms/router/bias
      classes can become F32 at approximately 2x. GGUF metadata/alignment
      also contributes to the artifact. Measure the actual output; do not
      infer it from nominal bpw.
    - **Actual resident memory:** bytes reported by the loader/runtime on the
      target machine, including the runtime's allocations. This is the final
      admission and quality-report quantity, not the converted file size.

    An out-of-band diagnostic (including a tiny fixture or an artifact outside
    the accepted band) is never a matched-budget winner. Resident serving and
    quality comparisons consume the selected GGUF at `$MODEL` only after this
    conservative artifact-size preflight (admission ceiling =
    `98569499443` in `configs/experiments/compression-sweep.json`, the 90 GiB
    ±2% accepted upper band):

    ```sh
    CAND_FILE_BYTES=$(stat -c%s "$MODEL")                # artifact file size: conservative preflight proxy ONLY — NOT resident weight memory
    MEM_AVAIL=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
    ADMISSION_CEILING=98569499443                        # budget.window_bytes[1]: 90 GiB ±2% accepted upper band
    FIRST_BOOT_RESERVE_ASSUMPTION=$((16*1024*1024*1024))  # named assumption: conservative provisional first-boot reserve, not a measured hardware requirement
    test "$CAND_FILE_BYTES" -le "$ADMISSION_CEILING" || { echo "candidate artifact $CAND_FILE_BYTES B exceeds accepted band upper bound $ADMISSION_CEILING B"; exit 1; }
    test $((CAND_FILE_BYTES + FIRST_BOOT_RESERVE_ASSUMPTION)) -le "$MEM_AVAIL" || { echo "MemAvailable $MEM_AVAIL B too low for $CAND_FILE_BYTES B + first-boot reserve"; exit 1; }
    "$STRIX_SRC/build-vulkan/bin/llama-cli" --model "$MODEL" --offline --ctx-size 512 --n-gpu-layers 999 --threads 1 --seed 42 --temp 0 --n-predict 16 --single-turn --simple-io --prompt sanity
    ```

    File size remains only a conservative artifact-size preflight proxy — it
    does **not** equal resident weight memory. Final admission and quality
    reports must use actual loader/runtime resident-memory measurements, and
    the 16 GiB reserve above is a named conservative first-boot assumption,
    not a measured hardware requirement. Native runtime proof for the real
    selected candidate and for Halo is still pending; the tiny CPU proof above
    does not promote a candidate or claim Halo compatibility.

    **No selected candidate has been built yet — this resident leg stays
    BLOCKED; no fake substitution** (the tiny fixtures prove graph/format
    compatibility only, never quality). Perplexity leg on the candidate:
    `"$STRIX_SRC/build-vulkan/bin/llama-perplexity" -m "$MODEL" --file "$MIMO_LAB/datasets/perplexity-slice.txt" -ngl 999`
    — the slice must be cut from the real corpus governed by
    `configs/dataset.json` / `docs/datasets.md` (no placeholder data; the
    slice leg stays blocked until that corpus is materialized). Record actual
    expert count/layer count/top-k from the candidate's GGUF metadata and
    compare with the hypothesis ledger in `docs/runtime-audit.md` §5.
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
    `python3 "$REPO/scripts/benchmark_matrix.py" --mode bench --tool "$STRIX_SRC/build-vulkan/bin/llama-bench" --model "$MODEL" --output "$MIMO_LAB/metrics/pp-vulkan.json"`.
    These are in-process batched prompt-processing numbers (PP8K/32K/64K/128K)
    — never labelled as concurrency.
39. Concurrency matrix (Vulkan lane, real parallel requests):
    `python3 "$REPO/scripts/benchmark_matrix.py" --mode serve --server "$STRIX_SRC/build-vulkan/bin/llama-server" --model "$MODEL" --output "$MIMO_LAB/metrics/c-vulkan.json"`.
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
    `python3 "$REPO/scripts/burst_replay.py" --schedule timings.json`
43. Bursty replay — live:
    `python3 "$REPO/scripts/burst_replay.py" --schedule timings.json --server-url http://127.0.0.1:18080/completion`.
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
