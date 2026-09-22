# Baseline: tacodevs REAP25 mixed-3bit GPTQ MTP

`baseline_id = tacodevs-reap25-mixed3bit-gptq-mtp`

A published 25%-pruned MLX checkpoint used as a **comparison-only** baseline
(never a production source) for the mimo-halo-lab expert-selection work.

- Source: <https://huggingface.co/tacodevs/MiMo-V2.6-Flash-RL-MLX-REAP25-mixed3bit-GPTQ-MTP>
- Pinned revision: `95d80053eb22112f2acc4244330eb67b4f534c94` (confirmed by the HF
  revision API before any file fetch; immutable)
- Base model: `XiaomiMiMo/MiMo-V2.6-Flash-RL`. Independently confirmed: the official
  release stores its experts natively as MXFP4. The tacodevs card additionally claims
  MXFP4 QAT training and that no BF16 release exists — those are card claims, not
  officially confirmed facts.
- Official parent source: <https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL>,
  independently available at pinned revision
  `5711b268169967567844e1e560e8a3966da959b1` (recorded here for reference; **not
  fetched or archived by this baseline**, so no official-parent revision is claimed
  as pinned by our archive)
- Pipeline source: <https://github.com/irvollo/mimo-mlx-compress>

## What is archived

Only small public text metadata. **No `.safetensors`/`.pt`/`.bin` files are ever
fetched**; weight byte counts come from the API tree listing and the safetensors
index. The 11.4 MB `tokenizer.json` is excluded (size budget); its size is recorded
in `evidence/tree.json`. Every fetch must match the tree-declared size exactly and
the git blob sha1 must match the tree oid; mismatches abort.

Archive layout (`manifests/baselines/tacodevs-reap25/`):

```
raw/                      original bytes, repo-relative paths preserved
evidence/revision.json    HF revision API response (sha pin)
evidence/tree.json        recursive tree listing (declared sizes, oids)
evidence/files.json       per-file sha256 / blob sha1 / URL / fetched_at
evidence/checksums.sha256 raw-file checksums
manifest.json             summary + retrieved_at
normalized.json           schema_version=1 contract output (see below)
```

Archived files (18 files, 276,932 bytes total, fetched 2026-09-22):

| File | Bytes | sha256 (first 16 hex) |
|---|---|---|
| `README.md` | 5131 | `1473e497e158f5ea…` |
| `config.json` | 83217 | `e5b18ed1a3a34926…` |
| `compression_alloc.json` | 35735 | `8a54bc1e843d5f1e…` |
| `compression_eval.json` | 220 | `18275a29e2df9d17…` |
| `chat_template.jinja` | 3868 | `853650bee57bf950…` |
| `generation_config.json` | 195 | `eecf00d970192127…` |
| `tokenizer_config.json` | 885 | `8524fc932eb8e29e…` |
| `model.safetensors.index.json` | 104491 | `ac83ed75d7102082…` |
| `audio_tokenizer/chat_template.jinja` | 5588 | `cf1a0a0e5cbc6a6a…` |
| `audio_tokenizer/config.json` | 1215 | `e0702adae37947e0…` |
| `audio_tokenizer/generation_config.json` | 149 | `5e14358a9bd50424…` |
| `audio_tokenizer/tokenizer_config.json` | 6059 | `ac41378c3257a15e…` |
| `dflash/config.json` | 1242 | `3a65305bd8d9a8be…` |
| `dflash/dflash.py` | 14255 | `da5ab1738b954800…` |
| `dflash/model.safetensors.index.json` | 3870 | `89ec75e9a3f65afc…` |
| `mtp/config.json` | 2358 | `09c00efaf3b92d47…` |
| `omnimodal/config.json` | 8068 | `61bea4a0f7a0dd89…` |
| `omnimodal/manifest.json` | 386 | `f0ce7e551f234371…` |

Full sha256 values: `evidence/checksums.sha256`. Full URLs and fetch timestamps:
`evidence/files.json`.

Integrity vs authenticity: the recorded `sha256`/git blob sha1 values are
unsigned integrity evidence — `import` and `verify` recompute them to detect
modification of the archived bytes against the recorded fetch. They are not
signatures and do not prove publisher authenticity by themselves; the
provenance claim is the pinned revision fetched over HTTPS (revision-API sha
confirmation + tree-oid binding), and every `files.json` record URL is bound
to that pin.

`verify` (and `import`, before it writes anything) reconcile the whole
archive: the record set must be unique, allowlisted, and confined under
`raw/` (no `..`, absolute paths, or symlinks); it must match
`manifest.json` `file_count`/`total_archived_bytes` and the on-disk raw
files exactly — missing, extra, and duplicate entries all fail — and every
raw file must hash and size-match its record. `import` performs all of this
on the exact bytes it is about to parse, so a refusal never overwrites an
existing `normalized.json`.

## Observed metadata structure (real, not assumed)

- `compression_alloc.json`: `prune_k: 64`; `pruned` maps each MoE layer
  (`"1"`…`"47"`) to its 64 pruned **original** expert IDs (sorted, range 0–255);
  `experts` maps the same layers to `{gate_proj, up_proj, down_proj}` precision
  specs; `expert_bytes: 93465870336.0`; `cost: 2.4015…`.
- `config.json`: `model_type mimo_v2_flash`, 48 layers, `moe_layer_freq[0] = 0`
  (dense) then 47 MoE layers, `n_routed_experts: 192` (post-pruning),
  `num_experts_per_tok: 8`, `n_shared_experts: null` (no shared experts),
  `hidden_size: 4096`, `moe_intermediate_size: 2048`.
- The checkpoint stores experts as **packed** tensors
  (`model.layers.N.mlp.switch_mlp.{gate,up,down}_proj`) with no per-expert tensor
  names, so packed 192 indices are never original IDs. Retained original IDs are
  derived as the ascending complement of each layer's published pruned list; the
  derivation is recorded on every layer.

## Normalized output (schema_version=1)

Generated deterministically from the archived bytes:

- **Architecture**: 256 original experts/layer, 192 retained/layer, top_k 8,
  47 MoE layers, 9024 total retained expert instances. Original count is
  cross-checked three ways (max pruned ID + 1 = 256; n_routed 192 + prune_k 64 = 256;
  card statement "192 of 256", asserted present in the archived README). The
  `field_status` notes are derived from the parsed config (entry counts
  included, never literal) and a non-dense lead layer fails.
- **Precision allocation** (141 expert projections, derived from the raw specs —
  matches the card exactly): 93× 3-bit affine g128, 22× native MXFP4 (4-bit g32),
  18× 2-bit affine g128, 8× 3-bit affine g64. The 8 3-bit-affine-g64 projections are
  small-`group_size` outliers worth noting when reading overhead; per-projection
  `bits/group_size/mode` plus `source_key` live on each layer.
- **Expert bits/weight**: published 3.29 (parsed from the archived README
  claim `experts average 3.29 bits/weight`); computed 3.2926 from
  `expert_bytes` over 47 × 192 × 3 × 4096 × 2048 params (includes
  scales/biases overhead). The computed entry's agreement label is computed
  from the two values — on this archive they agree, recorded separately.
- **Sizes** (integer bytes; DecimalGB ≠ GiB):
  - text weights (tree-summed 22 shards, includes safetensors headers):
    **100,431,428,728 B = 100.431429 decimal GB = 93.534057 GiB**
  - safetensors index `total_size`: **100,431,282,304 B = 100.431 decimal GB =
    93.534 GiB** (this is what the card's "100.4 GB" matches)
  - auxiliary weights (mtp/dflash/omnimodal/audio_tokenizer, tree-summed):
    **7,343,989,824 B ≈ 7.34 decimal GB** (card: "7.3 GB"; not loaded for text
    inference)
  - `runtime_memory_measured: false` — published size is disk size; runtime
    memory, KV cache and headroom are measured separately in the experiment
    design, never inferred from these numbers.
- **Published evaluation** (source-cited, distribution metrics only):
  - Held-out (31×2048, from `compression_eval.json` + card table): PPL
    9.271 → 9.529 (eval JSON 9.528511 rounds to 9.529), KL(base‖quant) 0.8315,
    top-1 agreement 79.5%. `cross_check.card_ppl`/`card_top1_agree_percent`
    are parsed from the archived card table (never hardcoded) and `agrees` is
    computed against `compression_eval.json`.
  - On-policy (40 API responses from the real MiMo-V2.6-Flash, assistant tokens
    only, card table): ALL 90.4%, code 90.8%, agent 88.5%, reasoning 94.2%,
    general 88.5% top-1 agreement; NLL deltas +0.045…+0.102.
  - `task_success: null` — these are **not** measured task success. Never cite
    them as benchmark scores. `independently_reproduced: false`.
- **Provenance**: role `comparison_only`, `production_source_allowed: false`,
  official parent pinned above, `metadata_sha256` = sha256 of `manifest.json`
  (`9458cdef9a0068a8d0179b37b34e01ffb2f83a4266214bfe156b401d765c99c0`).

## Usage

Consumer-facing command (stdlib only, Python ≥ 3.11):

```bash
# Archive small public metadata from the pinned revision (network op, idempotent)
PYTHONPATH=src python -m mimo_halo.baselines.tacodevs_reap25 archive \
  --out manifests/baselines/tacodevs-reap25

# Regenerate normalized.json from the archived bytes (offline, deterministic;
# re-verifies every raw file against records first, refuses without overwriting)
PYTHONPATH=src python -m mimo_halo.baselines.tacodevs_reap25 import \
  --manifest manifests/baselines/tacodevs-reap25 \
  --output manifests/baselines/tacodevs-reap25/normalized.json

# Re-verify raw files against recorded checksums (tamper detection)
PYTHONPATH=src python -m mimo_halo.baselines.tacodevs_reap25 verify \
  --manifest manifests/baselines/tacodevs-reap25
```

Installable as a package (`pip install .`): src layout, no runtime dependencies.
`import` fails closed on any archive-integrity violation before touching its
output (hash/size mismatch, record set vs `manifest.json`/raw-directory
mismatch, missing/extra/duplicate records, forged or symlinked paths, records
not URL-bound to the pinned revision), then on malformed original IDs
(duplicates, out-of-range,
unsorted), contradicting expert counts/dimensions, unknown projection specs, and
any index that names individual experts (packed-index ambiguity — the source
provides no packed→original mapping, so such a checkpoint cannot be normalized
safely).

## Caveats

- Published metrics describe token-level distribution agreement on the
  calibration domain, not long-horizon coding task success.
- The precision allocation is the publisher's sensitivity evidence, useful for
  prioritization — it is not our measured sensitivity and is not copied into
  candidate pipelines.
- Pruning metadata records which experts were removed, not why; saliency
  methodology lives in the upstream pipeline repo.
- Auxiliary (MTP/DFlash/vision/audio) bytes are recorded separately and are
  excluded from text-weight budget math; KV cache and runtime headroom are
  never folded into these byte counts.