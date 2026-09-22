# Public release intent (inactive P2)

The complete intent, implementation notes, and release checklist live in [#54 — Plan public release of Strix llama.cpp fork and MiMo compression artifacts](https://github.com/dymoo/mimo-halo-lab/issues/54) (parent: #1). This document duplicates only the stable intent pointer and the durable release requirements.

**Status: inactive P2.** This intent stays inactive until a genuinely worthwhile validated candidate exists; recording it authorizes nothing. No active P2 implementation or publication is underway or implied.

## Explicit non-actions

- No immediate publication of models, runtimes, or artifacts.
- No HF weight uploads.
- No new fork or runtime repository.
- No external commitments (upstream maintainers, HF releases, third parties).

## Requirements

1. **Public runtime fork.** Publish a Strix-Halo-focused llama.cpp-derived fork under dymoo with upstream attribution/licenses, documented upstream base SHA, isolated generic gfx1151 improvements, and documented Vulkan/ROCm builds, expert-count handling (160/176/nonstandard), MoE/SWA work, benchmark scripts, and correctness/regression suite — free of private infrastructure paths.
2. **Portable dynamic GGUF.** If derivative redistribution permits, publish portable llama.cpp-compatible coding-specialized MiMo GGUFs (e.g. UD-IQ4_XS / Q4_K_M / Q4_K_S, optional Q5) with broad CPU/CUDA/Metal/Vulkan compatibility and reproducible quality comparisons. Names must match the actually winning method/count.
3. **Separate Strix ROCm FP4 GGUF.** A distinct Strix-specific family (FAST/BALANCED/QUALITY only where measured useful), explicitly targeting AMD Strix Halo/gfx1151, never implying universally optimal quants.
4. **Structural maps and checkpoints.** Distribute source-precision pruned checkpoints if practical/licensed; otherwise publish deterministic HOPE160/HOPE176 maps, hashes, methodology, and reproduction scripts. Never mislabel a natively quantized source as BF16/unquantized.
5. **Generic tooling and upstreaming.** Publish generic MiMo REAP support, HOPE observer/selector, calibration, dynamic quant planner, ROCmFP4 export, and non-private evaluation tooling; prefer upstream PRs where genuinely generic. Upstream submission is a later deliberate action, not authorized now.
6. **Complete, truthful model cards.** Every model card states upstream revision, license, method/count, per-layer map/hash, calibration data description, actual private-data usage without content, recovery/QAT and quant methodology, tensor precision breakdown, fork commit, verified launch commands, measurements, limitations, quality-retention results with uncertainty, and checksums. Never invent "lossless", "same as full model", or "best quant".
7. **Public benchmark package.** Publish llama-bench/server commands, concurrency/context/KV/MTP settings, toolchain versions, power/hardware description, and prompt methodology. Private-derived evaluations expose aggregates, composition, methodology, and only explicitly permitted anonymized samples — never private code, transcripts, or NVMe state blobs.
8. **License, privacy, history-scan, and release gates.** Before any public weight/runtime release: upstream licensing and derivative-redistribution review; ROCmFP4 dependency/licensing review; private-data rights review; entire Git history and release-artifact secret/privacy scan (not just the latest tree); complete artifact provenance; verified checksums; green correctness/regression suite; complete quality and hardware benchmark reports; launch instructions tested from a clean machine.
9. **Complete provenance from now on.** Every generated model artifact records source model SHA/revision, source code commits, dataset-manifest hash, REAP/HOPE configuration, expert maps, recovery checkpoints/adapters, quant plan, quantizer commit, final GGUF checksum, runtime commit, and benchmark environment. Missing fields block promotion, not current development.
10. **Inactive until a worthwhile candidate.** This remains inactive P2 until validated quality/performance warrants explicit activation; current P0 work is never delayed for release packaging.

## Private-data attribution

State private-data use exactly as it happened: "private traces used only for calibration" only if factually true. If any recovery or training used private traces, disclose the actual use and complete rights/privacy review before release. Privacy applies to source, Git history, model metadata, benchmark packages, and NVMe conversation/prefix snapshots alike.
