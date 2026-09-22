"""Tests for mimo_halo.artifacts.

Focus: tamper detection, path traversal/symlink rejection, stage
preconditions, canonical digest determinism, immutable revisions, and
fail-closed creation. Not field copies.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from mimo_halo import artifacts  # noqa: E402


def _h(n):
    return f"{n:064x}"


def _commit(n):
    """A full 40-hex commit-shaped revision for fixtures."""
    return f"{n:040x}"


class ArtifactTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = os.path.join(self._tmp.name, "scratch")
        os.makedirs(self.scratch)

    def write(self, name, text):
        path = os.path.join(self.scratch, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def base_kwargs(self):
        env = self.write("env.json", json.dumps({"platform": "linux", "gpu_count": 1}))
        qa = self.write(
            "quant_assignment.json",
            json.dumps({"layers": [{"layer": 0, "experts": {"0": "ROCmFP4"}}]}),
        )
        return {
            "sources": [
                artifacts.parse_source_spec(f"model=https://example.com/mimo@{_commit(0xabc1234)}")
            ],
            "commands": ["quantize --in parent --out candidate.bin"],
            "environment_path": env,
            "dataset_hashes": [f"corpus={_h(1)}"],
            "parents": [
                artifacts.parse_parent_spec(f"pruned=pruned:{_h(1)}"),
                artifacts.parse_parent_spec(f"recovered=recovered:{_h(2)}"),
                artifacts.parse_parent_spec(f"calibration-post-recovery=calibration:{_h(3)}"),
            ],
            "quant_assignment_path": qa,
            "payload_paths": [],
            "seed": None,
        }

    def create(self, root, **overrides):
        kwargs = self.base_kwargs()
        kwargs.update(overrides)
        return artifacts.create_artifact(root, "quantized", **kwargs)


class CreationAndVerificationTest(ArtifactTestCase):
    def test_roundtrip_create_and_verify(self):
        root = os.path.join(self._tmp.name, "art")
        manifest = self.create(root)
        report = artifacts.verify_artifact(root)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["manifest_sha256"], manifest["manifest_sha256"])
        sidecars = {entry["path"] for entry in manifest["files"]}
        self.assertIn("checksums.txt", artifacts.RESERVED_NAMES)
        self.assertNotIn("checksums.txt", sidecars)
        self.assertNotIn("manifest.json", sidecars)
        self.assertTrue(set(artifacts.ALL_SIDECARS) & sidecars)

    def test_checksums_file_explains_payloads_and_sidecars(self):
        root = os.path.join(self._tmp.name, "art")
        payload = self.write("weights.bin", "payload-bytes")
        self.create(root, payload_paths=[payload])
        with open(os.path.join(root, "checksums.txt"), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        explained = {line.split("  ", 1)[1] for line in lines}
        self.assertIn("weights.bin", explained)
        self.assertIn("manifest.json", explained)
        self.assertIn("environment.json", explained)
        self.assertNotIn("checksums.txt", explained)
        self.assertEqual(len(lines), len(explained))

    def test_failed_preflight_leaves_no_files(self):
        payload = self.write("weights.bin", "payload-bytes")
        secret_env = self.write("secret_env.json", json.dumps({"OPENAI_API_KEY": "sk-x"}))
        failures = (
            ("duplicate payload", {"payload_paths": [payload, payload]}),
            ("dataset hash", {"dataset_hashes": ["corpus=deadbeef"]}),
            ("looks like a secret", {"environment_path": secret_env}),
            ("at least one --command", {"commands": []}),
        )
        for index, (pattern, overrides) in enumerate(failures):
            root = os.path.join(self._tmp.name, f"preflight-{index}")
            with self.assertRaisesRegex(artifacts.ArtifactError, pattern):
                self.create(root, **overrides)
            self.assertTrue(
                not os.path.exists(root) or not any(os.scandir(root)),
                f"failed preflight left files behind in {root}",
            )

    def test_unexplained_stale_content_blocks_create_and_retry(self):
        root = os.path.join(self._tmp.name, "stale")
        os.makedirs(root)
        stale = os.path.join(root, "stale.bin")
        with open(stale, "w", encoding="utf-8") as handle:
            handle.write("leftover")
        with self.assertRaisesRegex(artifacts.ArtifactError, "unexplained"):
            self.create(root)
        # A retry that declares a same-named file from elsewhere must be
        # refused rather than clobber the stale content.
        replacement = self.write("stale.bin", "fresh-declared-bytes")
        with self.assertRaisesRegex(artifacts.ArtifactError, "unexplained"):
            self.create(root, payload_paths=[replacement])
        self.assertFalse(os.path.exists(os.path.join(root, "manifest.json")))
        with open(stale, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "leftover")

    def test_payload_already_at_destination_adopted_without_copy(self):
        root = os.path.join(self._tmp.name, "art")
        os.makedirs(root)
        payload = os.path.join(root, "big.bin")
        with open(payload, "w", encoding="utf-8") as handle:
            handle.write("declared-payload-in-place")
        before = os.stat(payload)
        manifest = self.create(root, payload_paths=[payload])
        after = os.stat(payload)
        self.assertEqual(
            (before.st_ino, before.st_size, before.st_mtime_ns),
            (after.st_ino, after.st_size, after.st_mtime_ns),
        )
        self.assertIn("big.bin", {entry["path"] for entry in manifest["files"]})
        self.assertEqual(artifacts.verify_artifact(root)["status"], "verified")


class CanonicalDigestTest(ArtifactTestCase):
    def test_identical_inputs_identical_digest_across_roots(self):
        one = os.path.join(self._tmp.name, "one")
        two = os.path.join(self._tmp.name, "two")
        payload = self.write("weights.bin", "same-bytes")
        m1 = self.create(one, payload_paths=[payload])
        m2 = self.create(two, payload_paths=[payload])
        self.assertEqual(m1["manifest_sha256"], m2["manifest_sha256"])

    def test_digest_excludes_self_reference(self):
        root = os.path.join(self._tmp.name, "art")
        manifest = self.create(root)
        self.assertEqual(
            artifacts.canonical_manifest_digest(manifest),
            manifest["manifest_sha256"],
        )
        # Changing only the recorded digest must break verification.
        path = os.path.join(root, "manifest.json")
        with open(path, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        on_disk["manifest_sha256"] = _h(99)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(on_disk, handle)
        with self.assertRaisesRegex(artifacts.ArtifactError, "canonical manifest digest"):
            artifacts.verify_artifact(root)


class TamperDetectionTest(ArtifactTestCase):
    def setUp(self):
        super().setUp()
        self.root = os.path.join(self._tmp.name, "art")
        self.payload = self.write("weights.bin", "payload-bytes")
        self.create(self.root, payload_paths=[self.payload])

    def assert_verify_fails(self, pattern):
        with self.assertRaisesRegex(artifacts.ArtifactError, pattern):
            artifacts.verify_artifact(self.root)

    def test_modified_payload_detected(self):
        path = os.path.join(self.root, "weights.bin")
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content[:-1] + "X")
        self.assert_verify_fails("digest mismatch for weights.bin")

    def test_modified_sidecar_detected(self):
        metrics = os.path.join(self.root, "metrics.json")
        with open(metrics, encoding="utf-8") as handle:
            content = handle.read()
        with open(metrics, "w", encoding="utf-8") as handle:
            handle.write(content.replace("metrics", "metricz", 1))
        self.assert_verify_fails("digest mismatch for metrics.json")

    def test_modified_checksums_detected(self):
        path = os.path.join(self.root, "checksums.txt")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.replace("weights.bin", "weights.bin "))
        self.assert_verify_fails("checksums.txt")

    def test_modified_manifest_detected(self):
        path = os.path.join(self.root, "manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["seed"] = 1234
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        self.assert_verify_fails("canonical manifest digest mismatch")

    def test_unexplained_file_detected(self):
        with open(os.path.join(self.root, "smuggled.bin"), "w") as handle:
            handle.write("x")
        self.assert_verify_fails("unexplained files")

    def test_missing_manifest_detected(self):
        os.remove(os.path.join(self.root, "manifest.json"))
        self.assert_verify_fails("missing manifest.json")

    def test_handcrafted_malformed_parent_digest_detected(self):
        path = os.path.join(self.root, "manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["parents"][0]["sha256"] = "z" * 64
        manifest["manifest_sha256"] = artifacts.canonical_manifest_digest(manifest)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        self.assert_verify_fails("malformed parent digest")

    def test_handcrafted_role_kind_mismatch_detected(self):
        path = os.path.join(self.root, "manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for parent in manifest["parents"]:
            if parent["role"] == "calibration-post-recovery":
                parent["kind"] = "pruned"
        manifest["manifest_sha256"] = artifacts.canonical_manifest_digest(manifest)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        self.assert_verify_fails("requires kind 'calibration'")

    def test_moving_source_revision_detected(self):
        path = os.path.join(self.root, "source_commits.json")
        with open(path, encoding="utf-8") as handle:
            commits = json.load(handle)
        commits["sources"][0]["revision"] = "main"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(commits, handle)
        self.assert_verify_fails("moving refs")

    def test_moving_official_source_revision_detected(self):
        path = os.path.join(self.root, "manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["production"]["official_source"] = {
            "url": "https://example.com/x",
            "revision": "main",
        }
        manifest["manifest_sha256"] = artifacts.canonical_manifest_digest(manifest)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        self.assert_verify_fails("official source revision")


class TraversalAndSymlinkTest(ArtifactTestCase):
    def test_relative_path_validation(self):
        for bad in ("../x", "/abs/x", "a\\b", "a/./b", "a//b", "", ".", "a/../b"):
            with self.assertRaises(artifacts.ArtifactError):
                artifacts._safe_relative_path(bad)
        self.assertEqual(artifacts._safe_relative_path("a/b.bin"), "a/b.bin")

    def test_symlink_in_root_rejected_on_scan(self):
        root = os.path.join(self._tmp.name, "linkroot")
        os.makedirs(root)
        target = self.write("real.txt", "data")
        os.symlink(target, os.path.join(root, "link.txt"))
        with self.assertRaisesRegex(artifacts.ArtifactError, "symlink"):
            artifacts.scan_artifact_files(root)

    def test_symlinked_directory_rejected_on_scan(self):
        root = os.path.join(self._tmp.name, "linkdir")
        os.makedirs(os.path.join(root, "real"))
        os.symlink(self.scratch, os.path.join(root, "real", "hop"))
        with self.assertRaisesRegex(artifacts.ArtifactError, "symlink"):
            artifacts.scan_artifact_files(root)

    def test_create_rejects_payload_named_as_sidecar(self):
        root = os.path.join(self._tmp.name, "art")
        evil = self.write("manifest.json", "{}")
        with self.assertRaisesRegex(artifacts.ArtifactError, "collides"):
            self.create(root, payload_paths=[evil])


class StagePreconditionTest(ArtifactTestCase):
    def create_stage(self, kind, **overrides):
        kwargs = self.base_kwargs()
        kwargs.update(overrides)
        return artifacts.create_artifact(os.path.join(self._tmp.name, kind), kind, **kwargs)

    def test_quantized_requires_full_lineage(self):
        parents = [
            artifacts.parse_parent_spec(f"pruned=pruned:{_h(1)}"),
            artifacts.parse_parent_spec(f"recovered=recovered:{_h(2)}"),
        ]
        with self.assertRaisesRegex(artifacts.ArtifactError, "calibration-post-recovery"):
            self.create_stage("quantized", parents=parents)

    def test_quantized_requires_quant_assignment(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "quant_assignment"):
            self.create_stage("quantized", quant_assignment_path=None)

    def test_unquantized_stage_needs_no_quant_fields(self):
        manifest = self.create_stage("recovered", seed=1, quant_assignment_path=None)
        self.assertEqual(manifest["inputs"], {})

    def test_calibration_requires_seed_and_config(self):
        cfg = self.write("calib.json", json.dumps({"tokens": 2048}))
        with self.assertRaisesRegex(artifacts.ArtifactError, "seed"):
            self.create_stage("calibration", calibration_config_path=cfg)
        with self.assertRaisesRegex(artifacts.ArtifactError, "calibration_config"):
            self.create_stage("calibration", seed=7)

    def test_calibration_with_seed_and_config_verifies(self):
        cfg = self.write("calib.json", json.dumps({"tokens": 2048}))
        root = os.path.join(self._tmp.name, "calib-art")
        kwargs = self.base_kwargs()
        kwargs.pop("parents")
        kwargs.pop("quant_assignment_path")
        kwargs["seed"] = 7
        kwargs["calibration_config_path"] = cfg
        artifacts.create_artifact(root, "calibration", **kwargs)
        self.assertEqual(artifacts.verify_artifact(root)["kind"], "calibration")

    def test_evaluation_requires_metrics_and_candidate_parent(self):
        metrics = self.write("metrics.json", json.dumps({"solved": 0.5}))
        with self.assertRaisesRegex(artifacts.ArtifactError, "candidate"):
            self.create_stage("evaluation", metrics_path=metrics, parents=[])
        with self.assertRaisesRegex(artifacts.ArtifactError, "metrics.json"):
            self.create_stage(
                "evaluation",
                parents=[artifacts.parse_parent_spec(f"candidate=quantized:{_h(4)}")],
            )
        root = os.path.join(self._tmp.name, "eval-art")
        kwargs = self.base_kwargs()
        kwargs.pop("quant_assignment_path")
        kwargs["parents"] = [artifacts.parse_parent_spec(f"candidate=quantized:{_h(4)}")]
        kwargs["metrics_path"] = metrics
        artifacts.create_artifact(root, "evaluation", **kwargs)
        self.assertEqual(artifacts.verify_artifact(root)["kind"], "evaluation")

    def test_pruned_requires_expert_map(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "expert_map"):
            self.create_stage("pruned", seed=1, quant_assignment_path=None)

    def test_pruned_with_expert_map_and_seed_verifies(self):
        emap = self.write("expert_map.json", json.dumps({"layers": [{"layer": 0, "retained": [0, 1]}]}))
        root = os.path.join(self._tmp.name, "pruned-art")
        kwargs = self.base_kwargs()
        kwargs.pop("parents")
        kwargs.pop("quant_assignment_path")
        kwargs["seed"] = 1
        kwargs["expert_map_path"] = emap
        artifacts.create_artifact(root, "pruned", **kwargs)
        self.assertEqual(artifacts.verify_artifact(root)["kind"], "pruned")


class InputValidationTest(ArtifactTestCase):
    def test_moving_revision_rejected(self):
        for bad in ("HEAD", "main", "master", "latest", "develop", "trunk", "nightly", "Main", "MASTER"):
            with self.assertRaisesRegex(artifacts.ArtifactError, "moving"):
                artifacts.parse_source_spec(f"m=https://example.com/x@{bad}")

    def test_abbreviated_revision_rejected(self):
        for bad in ("abc1234f", "0123456789abcdef", "0" * 39, "0" * 41, "v1.2.3"):
            with self.assertRaisesRegex(artifacts.ArtifactError, "immutable full revision"):
                artifacts.parse_source_spec(f"m=https://example.com/x@{bad}")

    def test_full_sha_revisions_accepted(self):
        self.assertEqual(
            artifacts.parse_source_spec(f"m=https://example.com/x@{_commit(7)}")["revision"],
            _commit(7),
        )
        self.assertEqual(
            artifacts.parse_source_spec(f"m=https://example.com/x@{_h(9)}")["revision"],
            _h(9),
        )

    def test_conversion_and_runtime_commits_must_be_full_sha(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "40-hex"):
            self.create(
                os.path.join(self._tmp.name, "short-commit"), conversion_commit="abc1234"
            )
        root = os.path.join(self._tmp.name, "commits")
        self.create(root, conversion_commit=_commit(11), runtime_commit=_commit(12))
        self.assertEqual(artifacts.verify_artifact(root)["status"], "verified")

    def test_non_https_source_rejected(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "https"):
            artifacts.parse_source_spec(f"m=ftp://example.com/x@{_commit(1)}")

    def test_parent_digest_must_be_lowercase_hex(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "64 lowercase hex"):
            artifacts.parse_parent_spec(f"pruned={'A' * 64}")

    def test_dataset_hash_malformed_rejected(self):
        root = os.path.join(self._tmp.name, "art")
        with self.assertRaisesRegex(artifacts.ArtifactError, "dataset hash"):
            self.create(root, dataset_hashes=["corpus=deadbeef"])

    def test_duplicate_dataset_name_rejected(self):
        root = os.path.join(self._tmp.name, "art")
        with self.assertRaisesRegex(artifacts.ArtifactError, "duplicate dataset name"):
            self.create(root, dataset_hashes=[f"corpus={_h(1)}", f"corpus={_h(2)}"])

    def test_dataset_provenance_required(self):
        root = os.path.join(self._tmp.name, "art")
        with self.assertRaisesRegex(artifacts.ArtifactError, "dataset provenance"):
            self.create(root, dataset_hashes=[])

    def test_environment_secret_rejected(self):
        root = os.path.join(self._tmp.name, "art")
        env = self.write("secret_env.json", json.dumps({"nested": {"OPENAI_API_KEY": "sk-x"}}))
        with self.assertRaisesRegex(artifacts.ArtifactError, "looks like a secret"):
            self.create(root, environment_path=env)

    def test_environment_must_be_object(self):
        root = os.path.join(self._tmp.name, "art")
        env = self.write("list_env.json", '["raw", "dump"]')
        with self.assertRaisesRegex(artifacts.ArtifactError, "JSON object"):
            self.create(root, environment_path=env)

    def test_production_requires_official_source(self):
        root = os.path.join(self._tmp.name, "art")
        with self.assertRaisesRegex(artifacts.ArtifactError, "official-source"):
            self.create(root, production=True)
        manifest = self.create(
            root + "2",
            production=True,
            official_source=f"https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL@{_commit(0xabc1234)}",
        )
        self.assertTrue(manifest["production"]["requested"])
        self.assertIn("declaration-based", manifest["production"]["policy"])

    def test_production_rejected_for_dataset_stage(self):
        kwargs = self.base_kwargs()
        kwargs.pop("parents")
        kwargs.pop("quant_assignment_path")
        kwargs["production"] = True
        kwargs["official_source"] = f"https://example.com/x@{_commit(5)}"
        with self.assertRaisesRegex(artifacts.ArtifactError, "not valid for stage"):
            artifacts.create_artifact(
                os.path.join(self._tmp.name, "ds"), "dataset", **kwargs
            )


class CliTest(ArtifactTestCase):
    def test_cli_create_and_verify_exit_codes(self):
        root = os.path.join(self._tmp.name, "cli-art")
        payload = self.write("weights.bin", "payload")
        self.write("env.json", json.dumps({"platform": "linux"}))
        self.write("quant_assignment.json", json.dumps({"layers": []}))
        env = os.path.join(self.scratch, "env.json")
        qa = os.path.join(self.scratch, "quant_assignment.json")
        rc = artifacts.main(
            [
                "create",
                "--root", root,
                "--kind", "quantized",
                "--source", f"model=https://example.com/mimo@{_commit(0xabc1234)}",
                "--parent", f"pruned=pruned:{_h(1)}",
                "--parent", f"recovered=recovered:{_h(2)}",
                "--parent", f"calibration-post-recovery=calibration:{_h(3)}",
                "--quant-assignment", qa,
                "--dataset", f"corpus={_h(1)}",
                "--command", "quantize --out candidate.bin",
                "--environment", env,
                "--payload", payload,
            ]
        )
        self.assertEqual(rc, 0)
        rc = artifacts.main(["verify", "--root", root])
        self.assertEqual(rc, 0)

        with open(os.path.join(root, "weights.bin"), "a") as handle:
            handle.write("tampered")
        rc = artifacts.main(["verify", "--root", root])
        self.assertEqual(rc, 1)

    def test_cli_rejects_absent_stage_data(self):
        root = os.path.join(self._tmp.name, "cli-bad")
        self.write("env.json", json.dumps({"platform": "linux"}))
        env = os.path.join(self.scratch, "env.json")
        rc = artifacts.main(
            [
                "create",
                "--root", root,
                "--kind", "quantized",
                "--source", f"model=https://example.com/mimo@{_commit(0xabc1234)}",
                "--parent", f"pruned=pruned:{_h(1)}",
                "--command", "quantize",
                "--environment", env,
            ]
        )
        self.assertEqual(rc, 1)


class SchemaContractTest(unittest.TestCase):
    def test_schema_has_no_duplicate_keys(self):
        def reject_duplicates(pairs):
            seen = set()
            for key, _value in pairs:
                if key in seen:
                    raise AssertionError(f"duplicate key in schema: {key!r}")
                seen.add(key)
            return dict(pairs)

        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "artifact.schema.json"
        with open(schema_path, encoding="utf-8") as handle:
            json.load(handle, object_pairs_hook=reject_duplicates)


if __name__ == "__main__":
    unittest.main()
