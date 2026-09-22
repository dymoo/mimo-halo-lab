"""Tests for the tacodevs REAP25 baseline importer.

Coverage per assignment: malformed original-ID handling, packed-index
ambiguity, provenance/tamper detection, and unit correctness of the
normalization core. All fixtures are synthetic public-metadata-shaped
dicts; no network access and no model weights are used.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mimo_halo.baselines import tacodevs_reap25 as m


# ---------------------------------------------------------------------------
# fixture builders (synthetic, deterministic, small but structurally faithful)
# ---------------------------------------------------------------------------


def _pruned_ids_for_layer(layer: int) -> list[int]:
    if layer == 47:
        return list(range(192, 256))  # includes original ID 255
    return sorted({(layer + 4 * k) % 256 for k in range(64)})


def _spec_cycle() -> list[dict]:
    """141 projection specs: 93x 3-bit affine g128, 22x mxfp4, 18x 2-bit g128, 8x 3-bit g64."""
    specs = [{"bits": 3, "group_size": 128, "mode": "affine"}] * 93
    specs += [{"bits": 4, "group_size": 32, "mode": "mxfp4"}] * 22
    specs += [{"bits": 2, "group_size": 128, "mode": "affine"}] * 18
    specs += [{"bits": 3, "group_size": 64, "mode": "affine"}] * 8
    return specs


def _fixture():
    moe_layers = list(range(1, 48))
    specs = _spec_cycle()
    experts = {}
    cursor = 0
    for layer in moe_layers:
        entry = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            entry[name] = specs[cursor]
            cursor += 1
        experts[str(layer)] = entry
    config = {
        "model_type": "mimo_v2_flash",
        "num_hidden_layers": 48,
        "moe_layer_freq": [0] + [1] * 47,
        "num_experts_per_tok": 8,
        "n_routed_experts": 192,
        "n_shared_experts": None,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
    }
    alloc = {
        "prune_k": 64,
        "pruned": {str(layer): _pruned_ids_for_layer(layer) for layer in moe_layers},
        "experts": experts,
        "attn_bits": 8,
        "expert_bytes": 93465870336.0,
        "cost": 2.4014980130047903,
    }
    compression_eval = {
        "ppl": 9.529,
        "top1_agree": 0.795,
        "kl_base_to_quant": 0.8315,
        "tokens": 63457,
        "gptq": True,
    }
    index = {
        "metadata": {"total_size": 100431282304},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.mlp.switch_mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.mlp.switch_mlp.up_proj.scales": "model-00001-of-00002.safetensors",
            "model.layers.1.mlp.gate.weight": "model-00001-of-00002.safetensors",
        },
    }
    readme = (
        "# title\n\nText model on disk: **100.4 GB** (192 experts/layer, experts average "
        "3.29 bits/weight). Auxiliary weights: 7.3 GB.\n\n"
        "192 of 256 experts remain in every MoE layer.\n\n"
        "## Quality\n\n"
        "| Metric (held-out mix, 31x2048) | Original (MXFP4/FP8) | This model |\n"
        "|---|---|---|\n"
        "| Perplexity | 9.271 | 9.529 |\n"
        "| KL(original || this) | 0 | 0.8315 |\n"
        "| Top-1 agreement | 100% | 79.5% |\n\n"
        "**On-policy** (40 responses sampled via API):\n\n"
        "| group | tokens | original NLL | this model NLL | Delta | KL | top-1 agree |\n"
        "|---|---|---|---|---|---|---|\n"
        "| ALL | 24136 | 0.530 | 0.609 | +0.079 | 0.105 | 90.4% |\n"
        "| code | 10627 | 0.511 | 0.589 | +0.078 | 0.106 | 90.8% |\n"
        "| agent | 4786 | 0.695 | 0.779 | +0.084 | 0.126 | 88.5% |\n"
        "| reasoning | 3815 | 0.313 | 0.358 | +0.045 | 0.052 | 94.2% |\n"
        "| general | 4908 | 0.580 | 0.682 | +0.102 | 0.125 | 88.5% |\n"
    )
    tree = [
        {"type": "file", "path": "model-00001-of-00002.safetensors", "size": 100_000_000},
        {"type": "file", "path": "model-00002-of-00002.safetensors", "size": 500},
        {"type": "file", "path": "mtp/model_mtp.safetensors", "size": 700},
        {"type": "file", "path": "dflash/mask_embedding.pt", "size": 10},
        {"type": "file", "path": "audio_tokenizer/model.safetensors", "size": 33},
        {"type": "file", "path": "dflash/dflash.py", "size": 42},
        {"type": "file", "path": "tokenizer.json", "size": 11_000_000},
        {"type": "directory", "path": "mtp", "size": 0},
    ]
    source_files = [
        {"path": "config.json", "sha256": "a" * 64, "size_bytes": 10, "url": "https://example.test/config.json"}
    ]
    return dict(
        config=config,
        alloc=alloc,
        compression_eval=compression_eval,
        index=index,
        readme_text=readme,
        source_files=source_files,
        tree=tree,
        retrieved_at="2026-09-22T00:00:00+00:00",
    )


def _normalize_fixture(**overrides):
    kwargs = _fixture()
    kwargs.update(overrides)
    return m.normalize(**kwargs)


def _write_manifest(tmp_path, **fixture_overrides):
    """Materialize a full manifest directory from the fixture; returns (dir,)."""
    fixture = _fixture()
    fixture.update(fixture_overrides)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    readme_bytes = fixture["readme_text"].encode("utf-8")
    raw_payloads = {
        "README.md": readme_bytes,
        "config.json": json.dumps(fixture["config"]).encode("utf-8"),
        "compression_alloc.json": json.dumps(fixture["alloc"]).encode("utf-8"),
        "compression_eval.json": json.dumps(fixture["compression_eval"]).encode("utf-8"),
        "model.safetensors.index.json": json.dumps(fixture["index"]).encode("utf-8"),
    }
    files = []
    for name, payload in raw_payloads.items():
        (raw / name).write_bytes(payload)
        files.append(
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "url": f"https://huggingface.co/{m.REPO_ID}/resolve/{m.REVISION}/{name}",
                "fetched_at": "2026-09-22T00:00:00+00:00",
            }
        )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "files.json").write_text(json.dumps(files), encoding="utf-8")
    (evidence / "tree.json").write_text(json.dumps(fixture["tree"]), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "baseline_id": m.BASELINE_ID,
        "retrieved_at": fixture["retrieved_at"],
        "file_count": len(files),
        "total_archived_bytes": sum(f["size_bytes"] for f in files),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _read_records(manifest_dir):
    return json.loads((manifest_dir / "evidence" / "files.json").read_text(encoding="utf-8"))


def _write_records(manifest_dir, records):
    (manifest_dir / "evidence" / "files.json").write_text(json.dumps(records), encoding="utf-8")


# ---------------------------------------------------------------------------
# unit correctness
# ---------------------------------------------------------------------------


class TestNormalizeCorrectness(unittest.TestCase):
    def test_normalize_derives_architecture(self) -> None:
        doc = _normalize_fixture()
        arch = doc["architecture"]
        self.assertEqual(arch["original_experts_per_layer"], 256)
        self.assertEqual(arch["retained_experts_per_layer"], 192)
        self.assertEqual(arch["top_k"], 8)
        self.assertEqual(arch["moe_layer_count"], 47)
        self.assertEqual(arch["total_retained_expert_instances"], 47 * 192)
        # field note derives the entry count from config, never a literal 48
        config = _fixture()["config"]
        config["num_hidden_layers"] = 49
        config["moe_layer_freq"] = [0] + [1] * 47 + [0]
        derived = _normalize_fixture(config=config)["architecture"]["field_status"]["moe_layer_count"]
        self.assertIn("config.moe_layer_freq (49 entries, first dense)", derived)

    def test_layer_expert_ids_are_original_ids_from_complement(self) -> None:
        doc = _normalize_fixture()
        layer1 = doc["layers"][0]
        self.assertEqual(layer1["layer"], 1)
        pruned = _pruned_ids_for_layer(1)
        expected_retained = sorted(set(range(256)) - set(pruned))
        self.assertEqual(layer1["pruned_expert_ids"], pruned)
        self.assertEqual(pruned, sorted(pruned))
        self.assertEqual(layer1["retained_expert_ids"], expected_retained)
        self.assertEqual(len(layer1["retained_expert_ids"]), 192)
        self.assertEqual(layer1["original_expert_count"], 256)
        # packed 192 indices (0..191) must never leak in as original IDs
        self.assertNotEqual(layer1["retained_expert_ids"], list(range(192)))
        self.assertIn("packed", layer1["retained_ids_derivation"])

    def test_projection_specs_and_counts(self) -> None:
        doc = _normalize_fixture()
        counts = doc["precision_summary"]["projection_counts"]
        self.assertEqual(counts, {"affine_b2_g128": 18, "affine_b3_g128": 93, "affine_b3_g64": 8, "mxfp4_b4_g32": 22})
        self.assertEqual(sum(counts.values()), 141)
        spec = doc["layers"][0]["projections"]["gate"]
        self.assertEqual(spec["bits"], 3)
        self.assertEqual(spec["group_size"], 128)
        self.assertEqual(spec["mode"], "affine")
        self.assertEqual(spec["source_key"], "experts.1.gate_proj")

    def test_computed_expert_bpw_matches_published(self) -> None:
        doc = _normalize_fixture()
        params = 47 * 192 * 3 * 4096 * 2048
        expected = round(93465870336.0 * 8.0 / params, 4)
        self.assertEqual(doc["precision_summary"]["computed_expert_bpw"]["value"], expected)
        self.assertTrue(abs(expected - 3.29) < 0.01)  # agrees with the published card value
        self.assertTrue(doc["precision_summary"]["computed_expert_bpw"]["source"].endswith("agrees with published 3.29"))
        self.assertEqual(
            doc["precision_summary"]["expert_bpw"],
            {
                "value": 3.29,
                "kind": "published",
                "includes_overhead": None,
                "source": "archived README.md: 'experts average 3.29 bits/weight'",
            },
        )

    def test_size_units_decimal_gb_vs_gib_differ(self) -> None:
        doc = _normalize_fixture()
        size = doc["size"]
        self.assertEqual(size["text_weight_bytes"], 100_000_500)  # tree sum, never a fetched weight
        self.assertEqual(size["text_weight_gb"], round(100_000_500 / 1e9, 6))
        self.assertEqual(size["text_weight_gib"], round(100_000_500 / 2**30, 6))
        self.assertNotEqual(size["text_weight_gb"], size["text_weight_gib"])  # DecimalGB is not GiB
        self.assertEqual(size["index_total_size_bytes"], 100431282304)
        self.assertEqual(size["auxiliary_weight_bytes"], 743)  # mtp 700 + pt 10 + audio 33; dflash.py is not a weight
        self.assertIs(size["runtime_memory_measured"], False)
        self.assertEqual(size["published_text_gb"], 100.4)

    def test_published_evaluation_tables_and_nulls(self) -> None:
        doc = _normalize_fixture()
        evaluation = doc["published_evaluation"]
        self.assertIsNone(evaluation["task_success"])
        self.assertIs(evaluation["independently_reproduced"], False)
        self.assertIsNone(evaluation["on_policy"]["task_success"])
        self.assertEqual(len(evaluation["on_policy"]["table"]), 5)
        self.assertEqual(evaluation["on_policy"]["table"][0]["top-1 agree"], "90.4%")
        self.assertEqual(len(evaluation["distribution"]["heldout_table"]), 3)
        self.assertEqual(evaluation["distribution"]["compression_eval"]["ppl"], 9.529)
        cross_check = evaluation["distribution"]["cross_check"]
        self.assertEqual(cross_check["card_ppl"], 9.529)  # parsed from the archived card table
        self.assertIs(cross_check["agrees"], True)
        self.assertEqual(cross_check["card_top1_agree_percent"], 79.5)

    def test_parse_published_tables_rejects_missing(self) -> None:
        with self.assertRaisesRegex(m.BaselineError, "Quality"):
            m.parse_published_tables("# no tables here\n")


# ---------------------------------------------------------------------------
# malformed original IDs / dimensions
# ---------------------------------------------------------------------------


class TestMalformedOriginalIds(unittest.TestCase):
    def test_duplicate_pruned_id_fails(self) -> None:
        alloc = _fixture()["alloc"]
        alloc["pruned"]["5"] = sorted(set(alloc["pruned"]["5"][:-1]))  # one short
        alloc["pruned"]["5"][-1] = alloc["pruned"]["5"][0]  # duplicate
        with self.assertRaisesRegex(m.BaselineError, "duplicate pruned original ID"):
            _normalize_fixture(alloc=alloc)

    def test_out_of_range_pruned_id_fails(self) -> None:
        alloc = _fixture()["alloc"]
        bad = sorted({*alloc["pruned"]["3"][:-1], 300})
        alloc["pruned"]["3"] = bad
        with self.assertRaisesRegex(m.BaselineError, r"out of range \[0, 255\].*packed"):
            _normalize_fixture(alloc=alloc)

    def test_wrong_pruned_count_fails(self) -> None:
        alloc = _fixture()["alloc"]
        alloc["pruned"]["9"] = alloc["pruned"]["9"][:63]
        with self.assertRaisesRegex(m.BaselineError, "prune_k"):
            _normalize_fixture(alloc=alloc)

    def test_contradicting_original_expert_count_fails(self) -> None:
        config = _fixture()["config"]
        config["n_routed_experts"] = 193  # 193 + 64 != 256 derived from max pruned ID
        with self.assertRaisesRegex(m.BaselineError, "contradictory original expert count"):
            _normalize_fixture(config=config)

    def test_missing_layer_metadata_fails(self) -> None:
        alloc = _fixture()["alloc"]
        del alloc["pruned"]["12"]
        with self.assertRaisesRegex(m.BaselineError, "missing pruned list"):
            _normalize_fixture(alloc=alloc)

    def test_non_dense_first_layer_fails(self) -> None:
        config = _fixture()["config"]
        config["moe_layer_freq"] = [1] + [1] * 46 + [0]  # lead layer must be dense
        with self.assertRaisesRegex(m.BaselineError, "dense lead layer"):
            _normalize_fixture(config=config)

    def test_unknown_projection_spec_fails(self) -> None:
        alloc = _fixture()["alloc"]
        alloc["experts"]["3"]["down_proj"] = {"bits": 9, "group_size": 128, "mode": "affine"}
        with self.assertRaisesRegex(m.BaselineError, "bits"):
            _normalize_fixture(alloc=alloc)

    def test_extra_projection_field_fails(self) -> None:
        alloc = _fixture()["alloc"]
        alloc["experts"]["4"]["up_proj"] = {**alloc["experts"]["4"]["up_proj"], "measured_sensitivity": 0.5}
        with self.assertRaisesRegex(m.BaselineError, "exactly bits/group_size/mode"):
            _normalize_fixture(alloc=alloc)


# ---------------------------------------------------------------------------
# packed-index ambiguity
# ---------------------------------------------------------------------------


class TestPackedIndexAmbiguity(unittest.TestCase):
    def test_per_expert_named_index_is_ambiguous_and_fails(self) -> None:
        index = _fixture()["index"]
        index["weight_map"]["model.layers.1.mlp.experts.17.up_proj.weight"] = "model-00001-of-00002.safetensors"
        with self.assertRaisesRegex(m.BaselineError, "packed-index ambiguity"):
            _normalize_fixture(index=index)

    def test_packed_index_passes_and_is_recorded(self) -> None:
        doc = _normalize_fixture()
        self.assertTrue(any("packed" in note for note in doc["provenance"]["notes"]))


# ---------------------------------------------------------------------------
# provenance / tamper detection
# ---------------------------------------------------------------------------


class TestProvenanceAndTamper(unittest.TestCase):
    def test_verify_detects_tampered_raw_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            result = m.verify(manifest_dir)
            self.assertIs(result["ok"], True)
            readme = manifest_dir / "raw" / "README.md"
            readme.write_bytes(readme.read_bytes() + b"tampered\n")
            with self.assertRaisesRegex(m.BaselineError, "verify failed"):
                m.verify(manifest_dir)

    def test_verify_detects_tampered_recorded_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            files_path = manifest_dir / "evidence" / "files.json"
            records = json.loads(files_path.read_text(encoding="utf-8"))
            records[0]["sha256"] = "0" * 64
            files_path.write_text(json.dumps(records), encoding="utf-8")
            with self.assertRaisesRegex(m.BaselineError, "sha256"):
                m.verify(manifest_dir)

    def test_verify_detects_missing_raw_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            (manifest_dir / "raw" / "README.md").unlink()
            with self.assertRaisesRegex(m.BaselineError, "missing raw file"):
                m.verify(manifest_dir)

    def test_import_writes_normalized_with_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            doc = m.import_manifest(manifest_dir, manifest_dir / "normalized.json")
            written = json.loads((manifest_dir / "normalized.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["schema_version"], 1)
            self.assertEqual(doc["baseline_id"], m.BASELINE_ID)
            self.assertEqual(
                doc["provenance"]["metadata_sha256"],
                hashlib.sha256((manifest_dir / "manifest.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(doc["provenance"]["role"], "comparison_only")
            self.assertIs(doc["provenance"]["production_source_allowed"], False)
            self.assertEqual(written["architecture"]["moe_layer_count"], 47)

    def test_import_fails_without_required_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            (manifest_dir / "raw" / "README.md").unlink()
            with self.assertRaisesRegex(m.BaselineError, "missing required archived files"):
                m.import_manifest(manifest_dir)

    def test_import_refuses_tampered_raw_and_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            output = manifest_dir / "normalized.json"
            m.import_manifest(manifest_dir, output)
            original = output.read_bytes()
            readme = manifest_dir / "raw" / "README.md"
            readme.write_bytes(readme.read_bytes() + b"\ntampered\n")
            with self.assertRaisesRegex(m.BaselineError, "verify failed"):
                m.import_manifest(manifest_dir, output)
            self.assertEqual(output.read_bytes(), original)  # refusal never overwrites the output

    def test_removed_record_and_raw_file_fails_verify_and_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            records = [r for r in _read_records(manifest_dir) if r["path"] != "model.safetensors.index.json"]
            _write_records(manifest_dir, records)
            (manifest_dir / "raw" / "model.safetensors.index.json").unlink()
            with self.assertRaisesRegex(m.BaselineError, "file_count"):
                m.verify(manifest_dir)
            with self.assertRaisesRegex(m.BaselineError, "file_count"):
                m.import_manifest(manifest_dir, manifest_dir / "normalized.json")
            self.assertFalse((manifest_dir / "normalized.json").exists())

    def test_verify_detects_unlisted_extra_raw_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            (manifest_dir / "raw" / "notes.md").write_bytes(b"not in the record set")
            with self.assertRaisesRegex(m.BaselineError, "not listed in evidence/files.json"):
                m.verify(manifest_dir)

    def test_verify_rejects_duplicate_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            records = _read_records(manifest_dir)
            records.append(dict(records[0]))
            _write_records(manifest_dir, records)
            with self.assertRaisesRegex(m.BaselineError, "duplicate record"):
                m.verify(manifest_dir)

    def test_verify_rejects_forged_record_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            records = _read_records(manifest_dir)
            records[0]["path"] = "../README.md"
            _write_records(manifest_dir, records)
            with self.assertRaisesRegex(m.BaselineError, "confined under raw"):
                m.verify(manifest_dir)
            records[0]["path"] = "/etc/passwd"
            _write_records(manifest_dir, records)
            with self.assertRaisesRegex(m.BaselineError, "confined under raw"):
                m.verify(manifest_dir)

    def test_verify_rejects_symlinked_raw_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            outside = manifest_dir / "outside.json"
            outside.write_text('{"outside": true}', encoding="utf-8")
            target = manifest_dir / "raw" / "config.json"
            target.unlink()
            target.symlink_to(outside)
            with self.assertRaisesRegex(m.BaselineError, "symlink"):
                m.verify(manifest_dir)

    def test_verify_rejects_records_unbound_from_pinned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_dir = _write_manifest(tmp_path)
            records = _read_records(manifest_dir)
            records[0]["url"] = "https://example.test/README.md"
            _write_records(manifest_dir, records)
            with self.assertRaisesRegex(m.BaselineError, "pinned revision"):
                m.verify(manifest_dir)
            records[0]["url"] = f"https://huggingface.co/{m.REPO_ID}/resolve/{m.REVISION}/README.md"
            records[0]["revision"] = "0" * 40  # explicit revision present but wrong
            _write_records(manifest_dir, records)
            with self.assertRaisesRegex(m.BaselineError, "pinned revision"):
                m.verify(manifest_dir)

    def test_card_values_parsed_from_archived_readme_not_hardcoded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _fixture()
            drifted = fixture["readme_text"].replace(
                "| Perplexity | 9.271 | 9.529 |", "| Perplexity | 9.271 | 9.999 |"
            ).replace("| Top-1 agreement | 100% | 79.5% |", "| Top-1 agreement | 100% | 70% |")
            manifest_dir = _write_manifest(tmp_path, readme_text=drifted)
            doc = m.import_manifest(manifest_dir)
            cross_check = doc["published_evaluation"]["distribution"]["cross_check"]
            self.assertEqual(cross_check["card_ppl"], 9.999)  # old literals would say 9.529 / agrees=True
            self.assertIs(cross_check["agrees"], False)  # eval JSON (9.529) no longer agrees with the drifted card
            self.assertEqual(cross_check["card_top1_agree_percent"], 70.0)  # old literal said 79.5

    def test_missing_published_bpw_claim_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _fixture()
            drifted = fixture["readme_text"].replace("experts average 3.29 bits/weight", "very compressed")
            manifest_dir = _write_manifest(tmp_path, readme_text=drifted)
            with self.assertRaisesRegex(m.BaselineError, "bits/weight"):
                m.import_manifest(manifest_dir)

    def test_bpw_disagreement_label_is_computed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _fixture()
            fixture["alloc"]["expert_bytes"] = 80_000_000_000.0  # computed bpw far from published 3.29
            manifest_dir = _write_manifest(tmp_path, alloc=fixture["alloc"])
            doc = m.import_manifest(manifest_dir)
            source = doc["precision_summary"]["computed_expert_bpw"]["source"]
            self.assertIn("does NOT agree with published 3.29", source)  # old code hardcoded "agrees"


# ---------------------------------------------------------------------------
# archive allowlist (no network)
# ---------------------------------------------------------------------------


class TestArchiveAllowlist(unittest.TestCase):
    def test_allowlist_rejects_weights_and_unknown_paths(self) -> None:
        rejected = [
            ("model-00001-of-00022.safetensors", 100, "non-allowlisted"),
            ("tokenizer.json", 11_423_819, "non-allowlisted"),
            ("dflash/mask_embedding.pt", 100, "non-allowlisted"),
            ("README.md", m.GLOBAL_FILE_LIMIT + 1, "exceeds global limit"),
        ]
        for path, size, pattern in rejected:
            with self.subTest(path=path):
                with self.assertRaisesRegex(m.BaselineError, pattern):
                    m._validate_allowlist(path, size)
        m._validate_allowlist("README.md", 5131)  # allowlisted, small, allowlisted extension

    def test_tree_weight_summaries_from_declared_sizes(self) -> None:
        tree = _fixture()["tree"]
        text = m.tree_weight_bytes(tree, prefix="model-", suffix=".safetensors")
        self.assertEqual(text, 100_000_500)
        aux = m.tree_weight_bytes(tree, prefix=None, suffix=".safetensors", dirs=m.AUX_WEIGHT_DIRS)
        self.assertEqual(aux, 733)
        self.assertIsNone(m.tree_weight_bytes(None, prefix=None, suffix=".safetensors"))


if __name__ == "__main__":
    unittest.main()
