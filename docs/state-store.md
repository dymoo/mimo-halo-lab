# State store (src/mimo_halo/state_tier)

Blocking, opaque snapshot store for the NVMe state tier. Durable put, verified
get, metadata inspect. Standard library only (Python >= 3.11), real filesystem
I/O only — no mocked IO, no runtime or model claims. This module stores and
verifies bytes; it never asserts that any server actually loaded an identity.

## CLI

    PYTHONPATH=src python3 -m mimo_halo.state_tier.store put \
        --root DIR --session OPAQUE_ID --generation N --state-file INPUT \
        --identity IDENTITY.json --token-count N --token-sha256 SHA256 \
        --output RECEIPT.json

    PYTHONPATH=src python3 -m mimo_halo.state_tier.store get \
        --root DIR --snapshot-id ID --expected-identity IDENTITY.json \
        --output RESTORED.bin

    PYTHONPATH=src python3 -m mimo_halo.state_tier.store inspect \
        --root DIR --snapshot-id ID

`put` first streams the input **read-only** to prehash every payload byte;
no payload temp file exists yet, so a generation collision is refused
before any payload write happens. If the content-addressed blob for that
hash is already present and verified, it is reused as-is: no payload temp
file is created and zero payload bytes are written (also when a new record
deduplicates onto another record's identical blob). Otherwise `put` writes
a private temporary file with exclusive creation (mode 0600) while
re-hashing the bytes it actually writes, verifies that copy against the
prehash (refusing if `--state-file` changed between the two passes), fsyncs
it, atomically renames it into place, fsyncs the parent directory, and only
then commits the index row (SQLite stdlib, `synchronous = FULL`). The
receipt — including `io_metrics.payload_bytes_written`, the honest count of
payload bytes this operation wrote — is written atomically to `--output`
and also printed to stdout.

`get` validates the expected identity against the stored identity first,
streams the blob to a private temporary file beside the destination while
re-hashing and re-counting, and verifies checksum and size **before** a
single byte is published at the destination path (atomic rename). Any
refusal leaves an existing destination untouched. On success it also bumps
`last_accessed` in the index (metadata only).

`inspect` prints the canonical record as JSON on stdout.

Exit status: `0` success, `1` named refusal or I/O failure (message on
stderr as `store: ...`), `2` CLI usage error (argparse).

## Store layout

    ROOT/
      blobs/<ns-shard>/<state_sha256>.bin   immutable content-addressed payloads
      active-index/index.sqlite             SQLite stdlib index (snapshots table)
      active-index/store.lock               advisory lock file

- `ROOT`, `blobs/`, `active-index/` and namespace shard directories are created
  with mode 0700; blob, lock and index files with mode 0600.
- `<ns-shard>` is the first 16 hex chars of `sha256(namespace)`: the raw
  security principal never appears in a path, and caller-supplied session ids
  or namespaces are never used as path components, so no value can escape the
  store root.
- Every command refuses a `--root` that lies inside any Git working tree.
  The root must be explicit (`--root` is required).

## Receipt / snapshot record

Canonical fields (schema: `schemas/state-snapshot.schema.json`,
schema_version 1): `schema_version`, `snapshot_id`, `session_id`,
`generation`, `token_count`, `token_sha256`, `identity`, `state_sha256`,
`serialized_bytes`, `created_at`, `last_accessed`, `blob_path` (store-relative),
`kind = "target_state"`.

- `snapshot_id` is derived from (namespace, session id, generation): at most
  one record exists per triple.
- No raw tokens, prompts or API keys are ever written to the index or
  receipts — only counts and digests.
- `created_at` / `last_accessed` are Unix seconds.
- `io_metrics.payload_bytes_written` (put receipts only) counts the payload
  blob bytes the emitting `put` operation actually wrote: full payload size
  when a new or healing blob was written, `0` when an existing verified
  blob was reused or deduplicated. It deliberately **excludes**
  SQLite/index writes, filesystem metadata, receipt bytes, and NAND/SSD
  write amplification — it is not a device-level write claim. It is
  per-operation, so it is not persisted in the index and `inspect` output
  omits it.

## Compatibility identity (schema_version 1)

Required fields, exact key set, no extras:

| field | format |
|---|---|
| `schema_version` | `1` |
| `model_sha256` | lowercase 64-hex |
| `quant_revision` | nonempty string (exact source identifier) |
| `tokenizer_sha256` | lowercase 64-hex |
| `chat_template_sha256` | lowercase 64-hex |
| `runtime_commit` | lowercase 40-hex |
| `runtime_patch_sha256` | lowercase 64-hex (empty-patch-set hash when none) |
| `cache_format_version` | nonempty string |
| `kv_config_sha256` | lowercase 64-hex (canonical semantic cache config) |
| `namespace` | nonempty opaque security principal ID |

These are **caller-supplied strong fingerprints**. The store validates
structure and equality only. It does **not** and **cannot** attest that an
arbitrary remote server actually loaded them; a runtime adapter must bind
them to a controlled server/model launch. Namespace isolation on `get` is
enforced by full identity equality, never inferred from filesystem layout.

## Integrity and atomicity ordering

1. Stream input read-only, hashing all bytes (prehash); no payload temp
   file exists yet.
2. Validate any existing record for (namespace, session, generation)
   against the prehash, identity and token metadata; refuse a collision
   here with zero payload writes.
3. Verify the existing content-addressed blob against the prehash; if it
   matches, reuse it with no payload temp file and zero payload bytes
   written (skip to step 5).
4. Otherwise stream input → private temp file (exclusive `mkstemp`
   creation) while re-hashing the copied bytes, verify the copy equals the
   prehash (a changed `--state-file` fails here, nothing published), fsync
   it, atomically rename into `blobs/...` and fsync the directory — the
   immutable blob is durable first.
5. Commit the index row (SQLite transaction, `synchronous = FULL`) — the
   index is published only after the blob it references is durable.
6. Atomically write the receipt, carrying
   `io_metrics.payload_bytes_written` measured at the write path.

A partial write therefore never looks valid: an index row can only reference
an already-durable blob, and `get` re-verifies checksum, size and identity
before exposing any restored output.

## Generation semantics

- Re-putting the same (namespace, session, generation) with identical
  content, identity and token metadata **reuses** the existing record and
  blob: the existing file is verified by hashing it read-only, no payload
  temp file is created, zero payload bytes are written
  (`io_metrics.payload_bytes_written` is `0`), and only `last_accessed`
  metadata changes.
- The same generation with different content, identity or token metadata is
  a **collision** and is refused with exit status 1 — detected right after
  the read-only prehash, before any payload temp file exists.
- Identical content under a new generation or another session deduplicates
  onto the same immutable blob file (content-addressed); rows are
  independent, and the deduplicating put writes zero payload bytes.

## Concurrency / locking guarantees

- Every command (`put`, `get`, `inspect`) holds an **exclusive POSIX advisory
  lock** (`fcntl.flock`) on `active-index/store.lock` for its entire
  duration. Concurrent invocations on the same root serialize completely:
  there is no window in which two writers can race the generation check, and
  no reader can observe a torn index or a not-yet-durable blob.
- The lock is released automatically by the kernel if the process dies; no
  stale-lock recovery is needed.
- This is the documented "serialize on a safe lock" option from the contract,
  chosen over fine-grained latching. Blocking CLI; no async machinery.

## Path and namespace safety

- Blob paths are generated from digests, validated on read
  (`blobs/` prefix, no `..`, realpath confined under `blobs/`), and opened
  with `O_NOFOLLOW` — symlinks are refused, absolute paths and `..` escapes
  are refused.
- Session ids and namespaces are opaque data stored only in the index; they
  never select a path component directly.

## Refusals (exit 1)

- generation collision (different content/identity/token metadata)
- `--state-file` changed between the prehash and the copy pass (nothing is
  published and no receipt is written)
- identity or namespace mismatch on `get`
- corrupt blob (checksum or size mismatch) on `get`
- symlinked or escaping blob path, unknown snapshot id
- malformed identity JSON (schema/key/format errors)
- store root inside a Git working tree, or missing store root

In every refusal path on `get`, an existing destination file is preserved
byte-for-byte; in every refusal path on `put`, the index and existing blobs
are untouched and no receipt is written — and the generation-collision and
changed-source refusals happen before any payload temp file is created, so
they perform zero payload writes.

## Limitations (deliberate, first blocking round)

- Blocking, synchronous I/O only. No async save/prefetch pipeline, no
  scheduler, no generation-fenced runtime integration yet.
- POSIX only (`fcntl.flock`); not ported to Windows.
- No retention, eviction, TTL or prefix-cache policy in this module: warm/cold
  demotion is future index metadata only, and blob bytes are never rewritten
  for retention class or access time. Retention decisions live in the
  independent `retention` module; its soft retention defaults are not
  deadlines, and prefix TTL is a separate concern.
- No delete/GC command yet. Hard kills can leave orphan `.staging-*` /
  `.restore-*` temp files; they are never referenced by the index and are
  safe to remove manually. Blobs are immutable and content-addressed.
- Crash reconstruction of logical sessions is a later milestone; today's
  guarantee is only that partial writes never look valid.
- Identity validation is structural/equality, never attestation.

## Verification

Consumer seam: `tests/test_state_store.py` — the first-round real-CLI
put→get round-trip tracer plus the critical refusal regressions (generation
collision, unchanged-state no-rewrite, corrupt blob and incompatible
identity preserving the destination, path/namespace confinement) and the
honest write-counter regressions (`io_metrics.payload_bytes_written` equals
the real payload size on a first put, is `0` on an unchanged repeat and on
content dedup, and follows the write path rather than index-row presence).
Fixtures are synthetic bytes and identities created in independent temporary
directories outside the Git tree. Main runs the full suite.
