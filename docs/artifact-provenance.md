# Artifact provenance sidecars

Every large artifact in mimo-halo-lab lives in its own directory (the
**artifact root**) and carries seven sidecars plus its payload files:

| Sidecar | Contents |
| --- | --- |
| `manifest.json` | Stage, lineage, structured-input references, per-file digests, canonical manifest digest |
| `checksums.txt` | `sha256sum`-compatible lines explaining every payload file and sidecar (including `manifest.json` itself; excludes itself) |
| `commands.txt` | The exact caller-supplied reproduction commands, one per line |
| `source_commits.json` | Explicit `name`/`url`/`revision` source records (immutable full revisions), declared-signed flags, conversion and runtime commits (full 40-hex) |
| `dataset_manifest.json` | Dataset provenance: either a caller-supplied manifest or `name` + SHA-256 per dataset |
| `metrics.json` | Caller-supplied metrics (required for `evaluation`) |
| `environment.json` | **Sanitized** environment map supplied by the caller |

Tool: `mimo_halo.artifacts` (`PYTHONPATH=src python -m mimo_halo.artifacts`).
The public contract for `manifest.json` is `schemas/artifact.schema.json`.

## Explicit inputs only

`create` invents nothing. Kind, parent artifact digests, source revisions,
dataset hashes, calibration config, expert map, quant assignment, seed, and
conversion/runtime commits must all be passed explicitly, and stage-required
data is rejected when absent:

| Stage | Required inputs |
| --- | --- |
| `dataset` | sources, dataset provenance, commands, environment |
| `calibration` | + `--calibration-config`, `--seed` |
| `pruned` | + `--expert-map`, `--seed` |
| `recovered` | + `--parent pruned=...`, `--seed` |
| `quantized` | + parents `pruned`, `recovered`, `calibration-post-recovery`, `--quant-assignment` |
| `evaluation` | + `--parent candidate=...`, `--metrics` |

Unquantized stages require no quant fields. A `quantized` stage demands the
full source/recovery lineage **and** a post-recovery calibration parent
digest (role `calibration-post-recovery`, kind `calibration`) — recalibration
after merged recovery is part of the lineage, not an afterthought.

## Immutable revisions

A recorded revision must identify exactly one immutable version of the
source — never a name that can move:

- **Source revisions** (`--source`, `--official-source`) must be a full
  **40-hex git/HF commit SHA** or a **64-hex content digest**. Branch and
  tag names (`main`, `master`, `HEAD`, ...), abbreviated SHAs, and any
  other moving or ambiguous ref are rejected — there is no blocklist; the
  revision must parse as a full-length immutable identifier or it fails.
- **Conversion and runtime commits** (`--conversion-commit`,
  `--runtime-commit`) must each be a full **40-hex commit SHA**.
- The same policy applies on **`create` and `verify`**: `verify`
  re-checks every revision in the stored `source_commits.json` and the
  `production.official_source` recorded in the manifest, so an artifact
  whose lineage was rewritten to a moving ref fails verification.

## Fail-closed creation

`create` is preflighted so a refused run never leaves accepted state
behind:

- **All explicit inputs are validated before any write.** Stage
  requirements, sources and revisions, commands, dataset provenance,
  environment, structured inputs, and every payload name (regular file,
  no reserved-name collisions, no duplicates) are checked first; a
  failure writes no files at all, so a corrected retry starts clean.
- **Unexplained existing root content is refused.** If the root already
  contains files this run did not declare, `create` fails instead of
  folding them into the manifest — a stale file from an older run blocks
  the retry rather than being silently absorbed.
- **A declared payload already at its destination is adopted in place.**
  When the payload path *is* the file inside the artifact root, it is
  digested and referenced without being copied, so an adjacent manifest
  run does not need to copy a 100 GB model it already has; anything else
  that already exists at a destination is refused, never clobbered.
- **Sidecars are written atomically and `manifest.json` last.** Every
  sidecar lands via temp-file + rename, `checksums.txt` is computed from
  the exact manifest bytes, and `manifest.json` is the final write — no
  reader ever observes a partial sidecar, and an artifact only exists
  once all of its bytes are in place. Existing files are never cleaned
  up or deleted by the tool.

## Trust and enforcement boundaries

- **No private filesystem scanning, no environment collection.** The tool
  reads only the paths and files explicitly passed to it. `environment.json`
  must be a sanitized JSON object; keys that look like secrets (`*token*`,
  `*secret*`, `*password*`, `*api_key*`, `*private_key*`, `*authorization*`,
  `*credential*`, ...) fail creation **and** verification. Raw environment
  dumps are never accepted.
- **Path confinement.** All paths inside an artifact are relative POSIX
  paths; absolute paths, `..`, backslashes, symlinks, and non-regular files
  are rejected on both `create` and `verify`.
- **Production lineage is declared, not faked.** `--production` requires an
  explicit `--official-source URL@FULL_SHA` naming the official Xiaomi
  source at an immutable full revision (same revision policy as
  `--source`, enforced again by `verify`). The manifest records the
  declaration and the policy verbatim; no signature verification is
  performed and none is claimed. Promotion gates that need cryptographic
  assurance must consume separately declared signed sources — the sidecar
  deliberately does not pretend to provide this.
- **Provenance failure blocks promotion.** Any `verify` failure is fatal;
  there are no warnings, fallbacks, or "best effort" modes.

## Determinism and digests

- Payload and sidecar digests are SHA-256, streamed in 1 MiB chunks (no
  whole-file allocation).
- `manifest_sha256` is the SHA-256 of the manifest's **canonical JSON**
  (sorted keys, compact separators, ASCII-safe UTF-8) with the
  `manifest_sha256` field excluded — the self-reference cannot cover itself.
  It provides integrity over the canonical form only; it is **not** an
  authenticity or signature claim.
- `created_at` is opt-in (`--created-at`); when omitted it is absent, not
  zeroed, so identical inputs produce byte-identical sidecars in any root.
- `checksums.txt` explains every file listed in `manifest.files` plus
  `manifest.json`, sorted, in `sha256sum -c` format.

## What verify catches

Modified payload bytes, modified or malformed sidecars, a modified or
hand-crafted manifest (canonical digest mismatch), malformed parent digests
or role/kind mismatches, source revisions that are not immutable full SHAs
(moving refs or abbreviations, in `source_commits.json` and the production
official source), files on disk that the manifest does not explain,
stage precondition violations, secret-bearing environment sidecars, and any
symlink or path escape — each with a specific error and exit code 1.

## Minimal fixture (smoke)

```bash
REPO=/path/to/mimo-halo-lab
FIX=$(mktemp -d)
printf 'weights-placeholder\n' > "$FIX/model.bin"
printf '{"layers":[{"layer":0,"experts":{"0":"ROCmFP4"}}]}' > "$FIX/quant_assignment.json"
printf '{"platform":"linux","gpu_count":1}' > "$FIX/env.json"

# Revisions must be full immutable SHAs (40-hex git/HF commit SHA or
# 64-hex content digest) — never a branch name like @main.
COMMIT=0123456789abcdef0123456789abcdef01234567

PYTHONPATH="$REPO/src" python -m mimo_halo.artifacts create \
  --root "$FIX/candidate" \
  --kind quantized \
  --source model=https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL@$COMMIT \
  --signed-source model \
  --parent pruned=0000000000000000000000000000000000000000000000000000000000000001 \
  --parent recovered=0000000000000000000000000000000000000000000000000000000000000002 \
  --parent calibration-post-recovery=calibration:0000000000000000000000000000000000000000000000000000000000000003 \
  --quant-assignment "$FIX/quant_assignment.json" \
  --dataset corpus=1111111111111111111111111111111111111111111111111111111111111111 \
  --command "llama-quantize --out candidate.gguf" \
  --environment "$FIX/env.json" \
  --production \
  --official-source https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL@$COMMIT \
  --payload "$FIX/model.bin"

PYTHONPATH="$REPO/src" python -m mimo_halo.artifacts verify --root "$FIX/candidate"
```

Both commands print a JSON report; `verify` exits 0 with
`{"status": "verified", ...}`. Tamper checks:

```bash
echo x >> "$FIX/candidate/model.bin" && \
  PYTHONPATH="$REPO/src" python -m mimo_halo.artifacts verify --root "$FIX/candidate"  # exit 1, digest mismatch
printf '{"unexpected":true}' > "$FIX/candidate/extra.json" && \
  PYTHONPATH="$REPO/src" python -m mimo_halo.artifacts verify --root "$FIX/candidate"  # exit 1, unexplained file
rm "$FIX/candidate/extra.json"
python3 - "$FIX/candidate/source_commits.json" <<'EOF'
import json, sys
path = sys.argv[1]
with open(path) as fh:
    obj = json.load(fh)
obj["sources"][0]["revision"] = "main"
with open(path, "w") as fh:
    json.dump(obj, fh, indent=2, sort_keys=True)
EOF
PYTHONPATH="$REPO/src" python -m mimo_halo.artifacts verify --root "$FIX/candidate"  # exit 1, non-immutable source revision
```
