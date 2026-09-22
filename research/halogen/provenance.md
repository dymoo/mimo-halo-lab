# Halogen — provenance and pin (Phase 1 acquisition)

Acquisition date: **2026-09-22**. Machine-readable companion: [`pin.json`](pin.json).

## Product identification

**Pinned product: `halogen-flash-server` (trade name "halogen") by Peonist, LLC** —
a closed-source Qwen3.8-Flash-Next inference runtime built exclusively for AMD
Strix Halo (gfx1151), published only as a container image.

Identity evidence (quoted):

- Repository description: *"The fastest way to run Qwen3.8-Flash-Next on Strix
  Halo (gfx1151)"* — <https://github.com/peonist-ai/halogen-flash-server>
- README at tag v0.13.2, opening line: *"halogen™ is the fastest way to run
  Qwen3.8-Flash-Next on AMD Strix Halo, and it does not get there by spending
  fewer bits."* and *"Every kernel is written for this one GPU and this one
  model family. No general-purpose runtime, no portability layer, no fallback
  path."*
- The pinned artifact's own OCI label
  `org.opencontainers.image.description` = *"Qwen3.8-Flash-Next inference
  server for AMD Strix Halo (gfx1151)"*.
- LICENSE.md §3: *"the engine ships as a compiled binary containing gfx1151
  device code, and anyone with a disassembler can read those kernels"*.
- Third-party corroboration: DeepWiki — *"halogen-flash-server is a
  closed-source, hardware-specific inference engine optimized exclusively for
  the Qwen3.8-Flash-Next model family running on AMD Strix Halo (gfx1151)
  silicon … Shipped entirely as a container image"*
  (<https://deepwiki.com/peonist-ai/halogen-flash-server>); r/LocalLLaMA —
  *"there's also a closed-source solution called Halogen"*
  (<https://www.reddit.com/r/LocalLLaMA/comments/1weobt6/qwen38_flash_next_now_at_12k_ts_prefill_on_strix/>).

### Ambiguity report (same-name products)

Several unrelated products share the name "Halogen":

- **Halogen Software** — Canadian talent-management SaaS (later part of Saba
  Software); no inference runtime, no GPU code
  (<https://en.wikipedia.org/wiki/Halogen_Software>).
- **RANE Halogen** — control software for RANE DSP audio devices: *"The Halogen
  software manages global tasks such as discovering, connecting to, and
  applying configurations to RANE DSP devices"*
  (<https://www.ranecommercial.com/hal_software>).

**Verdict: no blocking ambiguity.** Only one product matches the specified
identity criteria (Qwen-Flash-Next inference runtime, gfx1151 / 128 GB Strix
Halo, published performance claims): `peonist-ai/halogen-flash-server`. It is
the product pinned here.

## Version and release

- **Version: 0.13.2.** Evidence: OCI label `org.opencontainers.image.version`
  = `0.13.2` inside the pinned artifact; annotated git tag `v0.13.2` (tagger
  `wenis`, 2026-09-22T11:38:18Z) → commit
  `4f707875cbefee9efb4ce0870a10d62afbe6c511` (2026-09-22T07:49:29Z); the
  README quickstart at that tag runs
  `ghcr.io/peonist-ai/halogen-flash-server:0.13.2`; `THIRD-PARTY-NOTICES.md`
  at that tag states `Image: ghcr.io/peonist-ai/halogen-flash-server:0.13.2`
  and `ROCm: 7.14.0`.
- **Release date:** image created 2026-09-22T05:24:34.478638025Z (image config
  `created`); git tag created 2026-09-22T11:38:18Z.
- **Discrepancy recorded (unrecoverable):** image label
  `org.opencontainers.image.revision` = `a39127e31db7` does not resolve in the
  public repository (GitHub API returns HTTP 422 *"No commit found for SHA:
  a39127e31db7"*). Recorded as-is; the public v0.13.2 tag target is the commit
  above.

## Canonical distribution channel

- Project site / docs / scripts: <https://github.com/peonist-ai/halogen-flash-server>
  (tag page <https://github.com/peonist-ai/halogen-flash-server/releases/tag/v0.13.2>).
- **Binary channel: the OCI container image
  `ghcr.io/peonist-ai/halogen-flash-server:0.13.2`**
  (registry manifest endpoint
  `https://ghcr.io/v2/peonist-ai/halogen-flash-server/manifests/0.13.2`).
  LICENSE.md defines the Software as *"the container image published by us
  containing the halogen inference engine binary, its serving front-end, and
  supporting scripts"*. The GitHub Releases API returns an empty list (no
  release assets), so the image is the only published binary artifact.

## Acquisition — exact commands

Run on 2026-09-22 (Docker CLI/Engine 29.4.0 on macOS; macOS `bsdtar`,
`shasum`, `file`, `jq` for hashing and unpacking):

```bash
docker pull --platform linux/amd64 ghcr.io/peonist-ai/halogen-flash-server:0.13.2
docker save -o halogen-flash-server-0.13.2-oci.tar ghcr.io/peonist-ai/halogen-flash-server:0.13.2
shasum -a 256 halogen-flash-server-0.13.2-oci.tar
tar -xf halogen-flash-server-0.13.2-oci.tar -C unpack
```

Storage: the archive lives at
`$MIMO_LAB/reference/halogen/halogen-flash-server-0.13.2-oci.tar`
(extracted tree beside it under `unpack/`, staged root filesystem under
`rootfs/`). **The binary is never committed to Git** — only hashes and
provenance live in this repository.

## Pinned hashes

### Archive

| artifact | sha256 | size | file type |
|---|---|---|---|
| `halogen-flash-server-0.13.2-oci.tar` | `0586f593e6ac34e93bc2274252df3c51fbfbbef1908f8f035bb8d764e6bb0192` | 3,567,376,896 B | POSIX tar archive (OCI layout: `index.json`, `manifest.json`, `blobs/sha256/*`) |

Packaging: OCI image `application/vnd.oci.image.manifest.v1+json`, platform
`linux/amd64`, base `python:3.12-slim-trixie`, built with buildah 1.43.2.

### Image identity (registry side)

| field | value |
|---|---|
| registry manifest digest | `sha256:fd071dfdebe618256d0653515b3c59c27302d5c1a8b74b9c0133fbe7740afd9e` (Docker-Content-Digest header; matches `docker pull` output) |
| config digest | `sha256:c49b9181f60a81380c5db5ecf36d9fc87b3cd367cb7d1801795ba47b36d258dd` (byte-identical registry and local) |
| local OCI manifest digest (after `docker save`) | `sha256:9f992494d52f07ce3f0ec45e08e803d0954cd06e46244552af8b2300338e752f` (differs from registry digest only because `docker save` stores layers uncompressed) |
| layers | 17; uncompressed total 3,550,835,351 B |

### Inner binaries (container paths; unpacked from the archive)

| path | sha256 | size | file type |
|---|---|---|---|
| `/usr/local/bin/flash_serve` | `c610eb79bf00e8a97b742b5fb8494ed0a7946e1baf2ae4ef2f2c7a138f30e5be` | 13,510,488 B | ELF 64-bit LSB pie executable, x86-64, dynamically linked, stripped — **the inference engine** (contains gfx1151 device code per LICENSE.md) |
| `/usr/local/bin/halogen-tools` | `1ad31d1fd9f3cec1a60c3c11f613099125bdbf64c28182d93aac8d3dc81995dc` | 13,416,136 B | ELF 64-bit LSB pie executable, x86-64, dynamically linked, stripped — checkpoint tools (`verify`/`inspect`/`ppl`/`niah`) |
| `/usr/local/bin/entrypoint.sh` | `460bd61348c1d25e6ec4f6c817f3516742fd906e71859f4aed15f3908548f019` | 77,594 B | Bourne-Again shell script, UTF-8 text |
| `/usr/local/bin/halogen-healthcheck` | `582537ecb6149e02accdea739aac36f443728535996846cec221ee11a8fa96e5` | 2,302 B | Bourne-Again shell script, ASCII text |
| `/halogen/tools/serve_api.py` | `6f68be846cf5b62ec7e87bde9c4560d06e507d584978879224c79538b86fca6d` | 264,626 B | Python script, UTF-8 text — OpenAI-compatible front end |
| `/halogen/tools/tool_parse.py` | `e2359e39e7dbfab90cc36a3dd7839d520eb86d5be45b18196ebd3d4e6d7cec12` | 24,169 B | Python script, UTF-8 text |
| `/halogen/tools/bench-serving.py` | `7edd38e6ce04854298cf7584ecac13d47afeb96c93e4c85fe113c14094944f83` | 18,431 B | Python script, UTF-8 text |
| `/halogen/tools/halogen_tools.py` | `4155711c1ef9128f2f28b842e6a0e0662f6553bc571595606343db8ff02ada30` | 12,012 B | Python script, ASCII text |
| `/halogen/tools/halogen-bench.py` | `0587690c50cfa6769b54d67c7553ba8aa3ec861e677082590d322b0b2f0f846f` | 10,165 B | Python script, UTF-8 text |
| `/halogen/tools/eval-prompts.json` | `23df9feb5ec2b343f675bb08dc6c0c2286e17dd6d73acf4e7d363d0c0c49922b` | 1,384 B | JSON data — benchmark prompt set |
| `/opt/halogen/flash-tune.plan` | `a053ca07a4de9642934dcbff91addf5dd457edb06169dc173ae354abe56754d1` | 30,644 B | binary data — baked matmul tuning plan |

Inventory scope: this is the complete set of halogen's own payload files in
the image (every halogen-shipped file under `/usr/local/bin`, all of
`/halogen/tools/*`, all of `/opt/halogen/*`; verified by scanning the staged
root filesystem for `*halogen*`/`flash*` names — no other matches).
Third-party contents (Debian base, Python 3.12, the ROCm 7.14.0 user-mode
stack) are governed by `THIRD-PARTY-NOTICES.md` and are out of scope for this
pin.

## Integrity verification performed

1. Registry manifest digest taken from the `Docker-Content-Digest` header and
   cross-checked against `docker pull` output:
   `sha256:fd071dfd…afd9e`.
2. All **36** blobs in the saved archive re-hashed: `sha256(content) == blob
   filename` — **0 mismatches**.
3. Config blob digest is byte-identical between registry manifest and local
   archive (`c49b9181…258dd`).
4. Registry↔local spot checks: registry gzip layer
   `9643fad16e85…` decompressed → `a6dc765193a5…` = local layer-1 blob;
   registry gzip layer `66ca5f2ba03a…` decompressed → `c321a426ca69…` =
   local layer-4 blob.
5. Payload layer mapping (which image layer writes each binary; layers applied
   in manifest order, last writer wins):
   `flash_serve` ← `c035cbc7…` (single writer; layer size
   13,513,728 B ≈ file + tar padding), `halogen-tools` ← `f4ef0f45…` (single
   writer), `entrypoint.sh` ← last writer `5b37a02a…` (of 2 writers),
   `serve_api.py` ← `0806e2b0…` (single writer), `flash-tune.plan` ←
   `f6bc69e8…` (single writer).
6. `pin.json` re-parsed with `jq` after writing; hash fields re-scanned for
   64-hex well-formedness; repo files scanned for absolute local paths (none).

## License

- **Name:** *halogen — End User License Agreement*, **Version 0.1**, effective
  25 August 2026 (proprietary; GitHub reports "License: Other"; OCI label
  identifier `LicenseRef-Peonist-EULA`). Publisher: **Peonist, LLC**
  (Copyright © 2026 Peonist, LLC; State of Florida law).
- **Where stated:** `LICENSE.md` at the repository root (full text, title
  block and version/date); README §License — *"The engine is distributed under
  the terms in LICENSE.md."*; inside the artifact itself via OCI label
  `org.opencontainers.image.licenses = LicenseRef-Peonist-EULA`;
  GitHub repository metadata `License: Other`.
- **Scope:** covers only Peonist's own code (engine binary, serving front end,
  benchmark tooling, container packaging). Model weights are **not** in the
  image and are licensed separately by their original authors. Third-party
  components inside the image (AMD ROCm runtime/libraries, Python
  interpreter and packages) are listed in `THIRD-PARTY-NOTICES.md` (repo root
  and inside the image), each under its own license, which governs on
  conflict.
- **Terms relevant to this research:** §3 restricts redistributing a modified
  image, removing notices, or using the marks for endorsement; on reverse
  engineering it states *"We do not prohibit it … We ask that you not
  redistribute derived source or kernels as your own work. Interoperability
  and security research are welcome."* §4 expressly permits benchmarking and
  publication (stating version + prompt set is requested, not a license
  condition). §5: no telemetry.

## Clean-room statement

This research phase is **clean-room by design**: we acquire the binary,
record identity/version/license provenance, and compute hashes — then perform
*observation only* (disassembly excerpts, ABI-level facts, machine-readable
catalogue entries, hypotheses) for an **independent** implementation. We do
**not** reproduce or redistribute Halogen source code, decompiled/reconstructed
source, or proprietary kernels; short disassembly excerpts appear only as
evidence tied to catalogue entries; the archive itself stays outside Git at
`$MIMO_LAB/reference/halogen/`. This matches LICENSE.md §3 (reverse
engineering not prohibited; interoperability/security research welcome;
derived source or kernels must not be redistributed as our own work).

## Toolchain used (recorded exactly)

- `docker` CLI/Engine **29.4.0** (daemon on the acquisition host, macOS
  arm64 host, image pulled for `--platform linux/amd64`; no GPU involved,
  static acquisition only)
- `ghcr.io` registry HTTP API (`token` + `manifests` + `blobs` endpoints)
  for digest/header verification
- macOS `/usr/bin/shasum` (LibreSSL SHA-256), `/usr/bin/tar` (bsdtar),
  `/usr/bin/file`, `/usr/bin/jq`, GitHub REST API (`commits`, `git/ref/tags`,
  `git/tags`, `releases`)
