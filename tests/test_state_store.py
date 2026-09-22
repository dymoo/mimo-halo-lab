"""First TDD tracer for the state-tier store consumer seam.

Real CLI put -> get binary round-trip over a known synthetic byte fixture and
a known identity, consuming the returned snapshot ID. No mocked internal IO;
invokes `python3 -m mimo_halo.state_tier.store` via subprocess with
PYTHONPATH=src exactly as the consumer contract specifies. Checksum and count
fields in the receipt are verified against an independently computed fixture,
not against store internals.

This test is expected to fail until the store implementation lands.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# Independent known fixture: deterministic binary bytes, small, not
# secret-shaped.
STATE_BYTES = bytes(range(256)) + b"mimo-halo state-tier tracer\x00\x01\xfe\xff"

# Known identity, schema_version 1, all fields strong synthetic fingerprints.
IDENTITY = {
    "schema_version": 1,
    "model_sha256": "a" * 64,
    "quant_revision": "q4_0",
    "tokenizer_sha256": "b" * 64,
    "chat_template_sha256": "c" * 64,
    "runtime_commit": "d" * 40,
    "runtime_patch_sha256": hashlib.sha256(b"").hexdigest(),
    "cache_format_version": "kv-v1",
    "kv_config_sha256": "e" * 64,
    "namespace": "ns-tracer-public",
}

TOKEN_COUNT = 7
TOKEN_SHA256 = hashlib.sha256(
    b"".join(i.to_bytes(4, "little") for i in range(TOKEN_COUNT))
).hexdigest()

SESSION_ID = "opaque-session-tracer-0001"
GENERATION = 3


def run_store(args, cwd=REPO_ROOT):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-m", "mimo_halo.state_tier.store", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
        timeout=60,
    )


class StateStoreRoundTripTest(unittest.TestCase):
    def test_put_get_round_trip_with_known_fixture(self):
        expected_state_sha256 = hashlib.sha256(STATE_BYTES).hexdigest()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "store-root"
            root.mkdir()
            state_file = Path(tmp) / "state.bin"
            state_file.write_bytes(STATE_BYTES)
            identity_file = Path(tmp) / "identity.json"
            identity_file.write_text(json.dumps(IDENTITY))
            receipt_file = Path(tmp) / "receipt.json"
            restored_file = Path(tmp) / "restored.bin"

            put = run_store(
                [
                    "put",
                    "--root", str(root),
                    "--session", SESSION_ID,
                    "--generation", str(GENERATION),
                    "--state-file", str(state_file),
                    "--identity", str(identity_file),
                    "--token-count", str(TOKEN_COUNT),
                    "--token-sha256", TOKEN_SHA256,
                    "--output", str(receipt_file),
                ]
            )
            self.assertEqual(
                put.returncode, 0,
                f"put failed:\nstdout={put.stdout}\nstderr={put.stderr}",
            )
            self.assertTrue(receipt_file.is_file(), "put wrote no receipt")

            receipt = json.loads(receipt_file.read_text())
            snapshot_id = receipt["snapshot_id"]
            self.assertTrue(snapshot_id, "receipt has empty snapshot_id")

            # Receipt consistency against the independent fixture.
            self.assertEqual(receipt["schema_version"], 1)
            self.assertEqual(receipt["session_id"], SESSION_ID)
            self.assertEqual(receipt["generation"], GENERATION)
            self.assertEqual(receipt["token_count"], TOKEN_COUNT)
            self.assertEqual(receipt["token_sha256"], TOKEN_SHA256)
            self.assertEqual(receipt["identity"], IDENTITY)
            self.assertEqual(receipt["state_sha256"], expected_state_sha256)
            self.assertEqual(receipt["serialized_bytes"], len(STATE_BYTES))
            self.assertEqual(receipt["kind"], "target_state")

            get = run_store(
                [
                    "get",
                    "--root", str(root),
                    "--snapshot-id", str(snapshot_id),
                    "--expected-identity", str(identity_file),
                    "--output", str(restored_file),
                ]
            )
            self.assertEqual(
                get.returncode, 0,
                f"get failed:\nstdout={get.stdout}\nstderr={get.stderr}",
            )
            restored = restored_file.read_bytes()
            self.assertEqual(restored, STATE_BYTES)
            self.assertEqual(
                hashlib.sha256(restored).hexdigest(), expected_state_sha256
            )
            self.assertEqual(len(restored), len(STATE_BYTES))


STATE_BYTES_B = b"different payload for collision\x00\x01\xfe\xff"
SENTINEL = b"pre-existing destination bytes"


class StateStoreConsumerRegressionTest(unittest.TestCase):
    """Critical consumer regressions beyond the first-round tracer.

    All assertions go through the real CLI into independent temporary
    directories outside the Git working tree: generation collision, unchanged
    state avoiding a blob rewrite, corrupt blob / incompatible identity
    preserving an existing destination, and path/namespace confinement.
    """

    def setUp(self):
        tmp_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_ctx.cleanup)
        self.tmp = Path(tmp_ctx.name)
        self.root = self.tmp / "store-root"
        self.root.mkdir()
        self.state_a = self.tmp / "state-a.bin"
        self.state_a.write_bytes(STATE_BYTES)
        self.state_b = self.tmp / "state-b.bin"
        self.state_b.write_bytes(STATE_BYTES_B)
        self.identity_file = self.tmp / "identity.json"
        self.identity_file.write_text(json.dumps(IDENTITY))
        self._seq = 0

    def put(self, state_file=None, generation=GENERATION, session=SESSION_ID,
            identity_file=None):
        self._seq += 1
        receipt_path = self.tmp / f"receipt-{self._seq}.json"
        result = run_store(
            [
                "put",
                "--root", str(self.root),
                "--session", session,
                "--generation", str(generation),
                "--state-file", str(state_file or self.state_a),
                "--identity", str(identity_file or self.identity_file),
                "--token-count", str(TOKEN_COUNT),
                "--token-sha256", TOKEN_SHA256,
                "--output", str(receipt_path),
            ]
        )
        receipt = None
        if result.returncode == 0 and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
        return result, receipt

    def get(self, snapshot_id, identity_file=None, dest=None):
        if dest is None:
            self._seq += 1
            dest = self.tmp / f"restored-{self._seq}.bin"
        result = run_store(
            [
                "get",
                "--root", str(self.root),
                "--snapshot-id", snapshot_id,
                "--expected-identity",
                str(identity_file or self.identity_file),
                "--output", str(dest),
            ]
        )
        return result, dest

    def test_generation_collision_different_content_fails_original_restorable(self):
        first, receipt = self.put()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIsNotNone(receipt)

        second, second_receipt = self.put(self.state_b)
        self.assertNotEqual(second.returncode, 0)
        self.assertIsNone(second_receipt, "colliding put must write no receipt")

        result, dest = self.get(receipt["snapshot_id"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(dest.read_bytes(), STATE_BYTES)

    def test_unchanged_state_reuses_blob_without_rewrite(self):
        first, receipt = self.put()
        self.assertEqual(first.returncode, 0, first.stderr)
        blob = self.root / receipt["blob_path"]
        before = blob.stat()

        second, again = self.put()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(again["snapshot_id"], receipt["snapshot_id"])

        after = blob.stat()
        self.assertEqual(before.st_ino, after.st_ino, "blob was rewritten")
        self.assertEqual(before.st_mtime_ns, after.st_mtime_ns, "blob mtime changed")
        self.assertEqual(before.st_size, after.st_size)

    def test_put_receipt_io_metrics_counts_payload_writes_and_zero_on_repeat(self):
        first, receipt = self.put()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIsNotNone(receipt)

        io_metrics = receipt.get("io_metrics")
        self.assertIsInstance(io_metrics, dict, "put receipt lacks io_metrics")
        self.assertEqual(
            set(io_metrics), {"payload_bytes_written"},
            "io_metrics must expose exactly payload_bytes_written",
        )
        written = io_metrics["payload_bytes_written"]
        self.assertIsInstance(written, int, "payload_bytes_written must be int")
        self.assertGreaterEqual(written, 0)
        self.assertEqual(
            written, len(STATE_BYTES),
            "first put must count the real payload bytes written at the "
            "write path, not a constant zero",
        )

        second, again = self.put()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(again["snapshot_id"], receipt["snapshot_id"])
        self.assertEqual(
            again["io_metrics"]["payload_bytes_written"], 0,
            "unchanged repeat must not rewrite payload bytes to the store",
        )
        # The retained record contract is unchanged by the new counter.
        self.assertEqual(again["generation"], GENERATION)
        self.assertEqual(again["serialized_bytes"], len(STATE_BYTES))
        self.assertEqual(again["serialized_bytes"], receipt["serialized_bytes"])
        self.assertEqual(again["state_sha256"], receipt["state_sha256"])
        self.assertEqual(again["blob_path"], receipt["blob_path"])

        # Restored bytes after the unchanged repeat are still the fixture.
        result, dest = self.get(again["snapshot_id"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(dest.read_bytes(), STATE_BYTES)

    def test_put_payload_bytes_written_follows_write_path_not_row_presence(self):
        base, base_receipt = self.put()
        self.assertEqual(base.returncode, 0, base.stderr)
        self.assertEqual(
            base_receipt["io_metrics"]["payload_bytes_written"],
            len(STATE_BYTES),
        )

        # A new record whose content already exists writes no payload bytes:
        # the counter follows the real write path, not index-row presence.
        dedup, dedup_receipt = self.put(generation=GENERATION + 1)
        self.assertEqual(dedup.returncode, 0, dedup.stderr)
        self.assertNotEqual(
            dedup_receipt["snapshot_id"], base_receipt["snapshot_id"]
        )
        self.assertEqual(
            dedup_receipt["io_metrics"]["payload_bytes_written"], 0,
            "dedup onto an existing valid blob must not write payload bytes",
        )

        # Fresh content in a fresh record writes and counts the full payload.
        fresh, fresh_receipt = self.put(
            state_file=self.state_b, generation=GENERATION + 2
        )
        self.assertEqual(fresh.returncode, 0, fresh.stderr)
        self.assertEqual(
            fresh_receipt["io_metrics"]["payload_bytes_written"],
            len(STATE_BYTES_B),
        )

    def test_corrupt_blob_refuses_get_and_preserves_destination(self):
        _, receipt = self.put()
        blob = self.root / receipt["blob_path"]
        data = bytearray(blob.read_bytes())
        data[-1] ^= 0xFF  # same size, different bytes
        blob.write_bytes(bytes(data))

        dest = self.tmp / "restored.bin"
        dest.write_bytes(SENTINEL)
        result, dest = self.get(receipt["snapshot_id"], dest=dest)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(dest.read_bytes(), SENTINEL)

    def test_incompatible_identity_refuses_get_and_preserves_destination(self):
        _, receipt = self.put()
        incompatible = dict(IDENTITY)
        incompatible["model_sha256"] = "f" * 64
        other_identity = self.tmp / "identity-other.json"
        other_identity.write_text(json.dumps(incompatible))

        dest = self.tmp / "restored.bin"
        dest.write_bytes(SENTINEL)
        result, dest = self.get(
            receipt["snapshot_id"], identity_file=other_identity, dest=dest
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(dest.read_bytes(), SENTINEL)

    def test_traversal_namespace_confined_and_symlink_blob_refused(self):
        weird = dict(IDENTITY)
        weird["namespace"] = "../../escape/../ns"
        weird_identity = self.tmp / "identity-weird.json"
        weird_identity.write_text(json.dumps(weird))

        result, receipt = self.put(
            state_file=self.state_a,
            generation=7,
            session="opaque-session-tracer-0002",
            identity_file=weird_identity,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        parts = Path(receipt["blob_path"]).parts
        self.assertEqual(parts[0], "blobs")
        self.assertNotIn("..", parts)
        blob = self.root / receipt["blob_path"]
        self.assertTrue(
            blob.resolve().is_relative_to(self.root.resolve()),
            "blob escaped the store root",
        )

        # A symlinked blob must be refused even when its target bytes match.
        outside = self.tmp / "outside.bin"
        outside.write_bytes(STATE_BYTES)
        blob.unlink()
        blob.symlink_to(outside)
        result, dest = self.get(receipt["snapshot_id"], identity_file=weird_identity)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(dest.exists(), "refusal must not create a destination")

    def test_foreign_namespace_identity_refused(self):
        _, receipt = self.put()
        foreign = dict(IDENTITY)
        foreign["namespace"] = "ns-other-principal"
        foreign_identity = self.tmp / "identity-foreign.json"
        foreign_identity.write_text(json.dumps(foreign))

        dest = self.tmp / "restored.bin"
        dest.write_bytes(SENTINEL)
        result, dest = self.get(
            receipt["snapshot_id"], identity_file=foreign_identity, dest=dest
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(dest.read_bytes(), SENTINEL)


if __name__ == "__main__":
    unittest.main()
