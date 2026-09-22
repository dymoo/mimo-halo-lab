"""Acceptance tests for the Halogen Phase-1 catalogue generator.

Run:
  cd research/halogen/tools/tests && python3 -m unittest -v test_catalogue

Covers (acceptance list):
  * generator + classifier green on synthetic fixtures (>=2 kernels,
    one wave32 + one wave64), rows matching ground truth;
  * honest-null behaviour (missing metadata key, missing descriptor,
    absent toolchain) - null + reason, never a guess;
  * schema parses and the emitted catalogue validates;
  * classifier categories on synthetic instruction streams.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)

from halogen_tools import catalogue, classify, disasm, validate  # noqa: E402
import make_fixtures  # noqa: E402

FIXTURE_DIR = os.path.join(HERE, "fixtures")
SCHEMA_PATH = os.path.normpath(
    os.path.join(TOOLS, "..", "schemas", "kernel-catalogue.schema.json")
)
GT_PATH = os.path.join(FIXTURE_DIR, "ground_truth.json")


def fixture_paths():
    with open(GT_PATH, "r", encoding="utf-8") as fh:
        gt = json.load(fh)
    return [
        os.path.join(FIXTURE_DIR, name)
        for name, spec in gt["fixtures"].items()
        if "input_error_contains" not in spec
    ], gt


def insn(offset, mnemonic, operands="v0, v1"):
    return disasm.Insn(offset=offset, vma=offset, mnemonic=mnemonic,
                       operands=operands, raw_bytes="00000000")


class TestSchema(unittest.TestCase):
    def test_schema_parses(self):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
            schema = validate.load_schema(json.load(fh))
        self.assertEqual(schema["type"], "object")
        self.assertIn("kernel", schema["$defs"])

    def test_emitted_catalogue_validates(self):
        paths, _gt = fixture_paths()
        cat, _tc, errors = catalogue.build(paths, schema_path=SCHEMA_PATH)
        self.assertEqual(errors["schema"], [], errors["schema"])
        with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
        self.assertEqual(validate.validate(cat, schema), [])


class TestGroundTruth(unittest.TestCase):
    """Fixture rows must equal the fixture design values exactly."""

    def test_make_fixtures_verify(self):
        # make_fixtures.verify() performs field-by-field comparison against
        # ground_truth.json and raises on any mismatch
        make_fixtures.verify(GT_PATH)

    def test_wave32_and_wave64_present(self):
        paths, _gt = fixture_paths()
        cat, _tc, _e = catalogue.build(paths)
        waves = {r["wave_size"] for r in cat["kernels"]}
        self.assertIn(32, waves)
        self.assertIn(64, waves)
        self.assertGreaterEqual(len(cat["kernels"]), 2)

    def test_named_categories_classified(self):
        paths, gt = fixture_paths()
        cat, _tc, _e = catalogue.build(paths)
        rows = {(os.path.basename(r["input"]), r["name"]): r for r in cat["kernels"]}
        a = rows[("fx_qgemm_w32.o", "fx_qgemm_w32")]
        self.assertEqual(a["likely_operation"], "fused_dequant_gemm")
        self.assertGreaterEqual(a["confidence"], 0.5)
        self.assertTrue(
            any(e["source"] == "disasm" and e["mnemonic"] for e in a["evidence"]),
            "classification must cite disassembly evidence pointers",
        )
        b = rows[("fx_reduce_w64.o", "fx_reduce_w64")]
        self.assertEqual(b["likely_operation"], "reduction")
        self.assertGreaterEqual(b["confidence"], 0.6)


class TestHonestNulls(unittest.TestCase):
    def setUp(self):
        self.paths, self.gt = fixture_paths()
        self.cat, _tc, self.errors = catalogue.build(
            self.paths, schema_path=SCHEMA_PATH
        )
        self.rows = {
            (os.path.basename(r["input"]), r["name"]): r
            for r in self.cat["kernels"]
        }

    def test_missing_metadata_keys_are_null_with_reason(self):
        row = self.rows[("fx_missing_fields.o", "fx_missing_fields")]
        for field_name, needle in (
            ("vgpr", "absent or null in kernel metadata"),
            ("wave_size", "absent or null in kernel metadata"),
            ("kd_preload", "descriptor not found"),
        ):
            self.assertIsNone(row[field_name], field_name)
            self.assertIn(needle, row["null_reasons"][field_name])
        # fields that WERE present still resolve
        self.assertEqual(row["sgpr"], 16)
        self.assertEqual(row["lds"], 4096)

    def test_no_metadata_note_uses_descriptor_and_reports_gaps(self):
        row = self.rows[("fx_nometa.o", "fx_nometa")]
        self.assertEqual(row["metadata"]["schema"], None)
        self.assertEqual(row["field_sources"]["vgpr"], "kd")
        self.assertEqual(row["vgpr"], 16)
        self.assertEqual(row["lds"], 4096)
        self.assertIsNone(row["sgpr"])
        self.assertIn("granule encodes 0", row["null_reasons"]["sgpr"])
        self.assertIsNone(row["max_flat_workgroup_size"])
        self.assertIn(
            "not derivable", row["null_reasons"]["max_flat_workgroup_size"]
        )

    def test_every_null_contract_field_has_a_reason(self):
        for row in self.cat["kernels"]:
            for field_name in catalogue.NULLABLE_CONTRACT_FIELDS:
                if row.get(field_name) is None:
                    self.assertIn(
                        field_name, row["null_reasons"],
                        f"{row['name']}: {field_name} null without reason",
                    )
                    self.assertTrue(row["null_reasons"][field_name].strip())

    def test_no_disasm_tool_degrades_honestly(self):
        old = os.environ.get("HALOGEN_LLVM_OBJDUMP")
        os.environ["HALOGEN_LLVM_OBJDUMP"] = "/nonexistent/llvm-objdump"
        try:
            cat, tc, errors = catalogue.build(
                [os.path.join(FIXTURE_DIR, "fx_qgemm_w32.o")]
            )
        finally:
            if old is None:
                os.environ.pop("HALOGEN_LLVM_OBJDUMP", None)
            else:
                os.environ["HALOGEN_LLVM_OBJDUMP"] = old
        row = cat["kernels"][0]
        self.assertIsNone(row["likely_operation"])
        self.assertIsNone(row["confidence"])
        self.assertIn("llvm-objdump", row["null_reasons"]["likely_operation"])
        self.assertFalse(row["disassembly"]["available"])
        # metadata fields still resolved - only classification degraded
        self.assertEqual(row["wave_size"], 32)
        self.assertFalse(errors["schema"])

    def test_flag_no_disasm_degrades_honestly(self):
        cat, _tc, _e = catalogue.build(
            [os.path.join(FIXTURE_DIR, "fx_qgemm_w32.o")], no_disasm=True
        )
        row = cat["kernels"][0]
        self.assertIsNone(row["likely_operation"])
        self.assertIn("disabled", row["null_reasons"]["likely_operation"])

    def test_garbage_input_reports_error_not_crash(self):
        cat, _tc, errors = catalogue.build(
            [os.path.join(FIXTURE_DIR, "fx_garbage.bin")]
        )
        self.assertEqual(cat["kernels"], [])
        joined = " ".join(cat["inputs"][0]["errors"]) + " " + " ".join(errors["inputs"])
        self.assertIn("no AMDGPU code object found", joined)


class TestBundlesAndExtraction(unittest.TestCase):
    def test_clang_offload_bundle_extracted(self):
        path = os.path.join(FIXTURE_DIR, "fx_fat_bundle.bin")
        cat, _tc, _e = catalogue.build([path])
        self.assertEqual(len(cat["inputs"][0]["bundles"]), 1)
        bundle = cat["inputs"][0]["bundles"][0]
        self.assertEqual(bundle["format"], "clang-offload-bundle")
        ids = [e["id"] for e in bundle["entries"]]
        self.assertEqual(ids[1], "hipv4-amdgcn-amd-amdhsa--gfx1151")
        names = [r["name"] for r in cat["kernels"]]
        self.assertEqual(names, ["fx_qgemm_w32"])

    def test_bundle_inside_host_elf_extracted(self):
        path = os.path.join(FIXTURE_DIR, "fx_hostelf_bundle.o")
        cat, _tc, _e = catalogue.build([path])
        self.assertEqual(cat["inputs"][0]["kind"], "host-elf")
        self.assertEqual(cat["inputs"][0]["bundles"][0]["format"],
                         "clang-offload-bundle")
        self.assertEqual([r["name"] for r in cat["kernels"]], ["fx_qgemm_w32"])

    def test_classic_v3_schema_parsed(self):
        path = os.path.join(FIXTURE_DIR, "fx_v3_classic.o")
        cat, _tc, _e = catalogue.build([path])
        row = cat["kernels"][0]
        self.assertEqual(row["metadata"]["schema"], "classic")
        self.assertEqual(row["vgpr"], 32)
        self.assertEqual(row["sgpr"], 24)
        self.assertEqual(row["lds"], 2048)
        self.assertEqual(row["wave_size"], 64)
        self.assertIsNone(row["kd_preload"])
        self.assertIn("descriptor not found",
                      row["null_reasons"]["kd_preload"])


class TestPathPrivacy(unittest.TestCase):
    def test_absolute_paths_never_reach_output(self):
        # build() labels inputs through _input_label: absolute => basename
        cat, _tc, _e = catalogue.build(
            [os.path.join(FIXTURE_DIR, "fx_qgemm_w32.o")]
        )
        text = json.dumps(cat)
        self.assertNotIn("/Users/", text)
        self.assertNotIn("/Volumes/", text)
        for entry in cat["inputs"]:
            self.assertFalse(os.path.isabs(entry["path"]), entry["path"])
        for row in cat["kernels"]:
            self.assertFalse(os.path.isabs(row["input"]), row["input"])


class TestDisasmCompleteness(unittest.TestCase):
    """Parsed instruction bytes must tile the kernel's code range exactly:
    any silent parse drop would leave a gap (regression guard for report of
    lost v_dot2/dual-issue lines on real bundles)."""

    def _cover(self, fixture, symbol, isa_size):
        path = os.path.join(FIXTURE_DIR, fixture)
        with open(path, "rb") as fh:
            data = fh.read()
        tool, err = disasm.discover()
        if not tool:
            self.skipTest(f"llvm-objdump unavailable: {err}")
        res = disasm.disassemble(tool, data, "gfx1151", fixture)
        self.assertIsNone(res.error, res.error)
        # only the kernel's own code range; section fill after the symbol
        # belongs to nobody
        insns = [
            i for i in res.per_symbol[symbol]
            if i.raw_bytes and i.offset < isa_size
        ]
        self.assertTrue(insns, f"no parsed instructions for {symbol}")
        pos = 0
        for ins in insns:
            nbytes = len(ins.raw_bytes.replace(" ", "")) // 2
            self.assertEqual(
                ins.offset, pos,
                f"gap/overlap before {ins.mnemonic}@{ins.offset:#x} "
                f"(expected {pos:#x})",
            )
            pos = ins.offset + nbytes
        return pos

    def test_fixture_a_coverage(self):
        end = self._cover("fx_qgemm_w32.o", "fx_qgemm_w32", 128)
        self.assertEqual(end, 128)

    def test_fixture_b_coverage(self):
        end = self._cover("fx_reduce_w64.o", "fx_reduce_w64", 76)
        self.assertEqual(end, 76)


class TestClassifierCategories(unittest.TestCase):
    """Each contract category must be reachable from a synthetic stream."""

    def _classify(self, mnemonics):
        return classify.classify([insn(i * 4, m) for i, m in enumerate(mnemonics)])

    def test_quantized_gemm(self):
        c = self._classify([
            "v_dot2_f32_bf16", "buffer_load_u8", "buffer_load_u8",
        ])
        self.assertEqual(c.likely_operation, "quantized_gemm")
        self.assertGreater(c.confidence, 0.5)
        self.assertTrue(c.evidence)

    def test_fused_dequant_gemm(self):
        c = self._classify([
            "v_dot2_f32_bf16", "buffer_load_u8", "v_lshlrev_b32_e32",
            "v_pk_mul_f16", "v_pk_add_f16", "v_pk_lshrrev_b16",
            "v_cvt_f32_bf16",
        ])
        self.assertEqual(c.likely_operation, "fused_dequant_gemm")
        self.assertGreaterEqual(c.confidence, 0.5)

    def test_attention_beats_plain_gemm_with_softmax(self):
        c = self._classify([
            "wmma_f32_16x16x16_f16", "v_exp_f32_e32", "v_rcp_f32_e32",
            "ds_load_b32",
        ])
        self.assertEqual(c.likely_operation, "attention")
        self.assertGreaterEqual(c.confidence, 0.5)

    def test_reduction(self):
        c = self._classify([
            "v_add_f32_e32", "v_add_f32_e32", "v_add_f32_e32",
            "v_add_nc_u32_e32", "ds_load_b32", "v_cmp_gt_f32_e32",
        ])
        self.assertEqual(c.likely_operation, "reduction")
        self.assertGreaterEqual(c.confidence, 0.5)

    def test_activation(self):
        c = self._classify(["v_exp_f32_e32", "v_rcp_f32_e32", "v_add_f32_e32"])
        self.assertEqual(c.likely_operation, "activation")
        self.assertGreaterEqual(c.confidence, 0.5)

    def test_moe_routing(self):
        c = self._classify([
            "s_load_b128", "s_load_b128", "v_cmp_eq_u32_e32",
            "buffer_store_b32",
        ])
        self.assertEqual(c.likely_operation, "moe_routing")
        self.assertTrue(c.evidence)

    def test_expert_gather_scatter(self):
        c = self._classify([
            "v_lshlrev_b32_e32", "buffer_load_b32",
            "v_lshlrev_b32_e32", "buffer_load_b32",
            "v_lshlrev_b32_e32", "buffer_load_b32",
            "v_lshlrev_b32_e32", "buffer_load_b32",
            "buffer_store_b32",
        ])
        self.assertEqual(c.likely_operation, "expert_gather_scatter")
        self.assertGreaterEqual(c.confidence, 0.5)

    def test_kv_operation(self):
        c = self._classify([
            "buffer_load_b16", "buffer_load_b16", "buffer_load_b16",
            "buffer_load_b16", "buffer_store_b16", "buffer_store_b16",
            "buffer_store_b16", "s_add_i32", "s_cbranch_scc0",
        ])
        self.assertEqual(c.likely_operation, "kv_operation")
        self.assertGreaterEqual(c.confidence, 0.5)

    def test_tensor_repacking(self):
        c = self._classify([
            "v_cvt_f32_ubyte0_e32", "v_cvt_f32_ubyte1_e32",
            "v_cvt_f32_ubyte2_e32", "v_cvt_f32_ubyte3_e32",
            "v_cvt_f32_f16_e32", "v_cvt_f32_f16_e32",
            "ds_store_b32", "ds_load_b32", "v_lshlrev_b32_e32",
            "buffer_store_b32",
        ])
        self.assertEqual(c.likely_operation, "tensor_repacking")
        self.assertGreaterEqual(c.confidence, 0.5)

    def test_speculative_verification(self):
        c = self._classify([
            "buffer_load_b32", "buffer_load_b32", "v_cmp_eq_u32_e32",
            "v_cmp_ne_u32_e32", "v_sub_u32_e32", "s_cbranch_scc1",
        ])
        self.assertEqual(c.likely_operation, "speculative_verification")
        self.assertGreaterEqual(c.confidence, 0.5)

    def test_v_wmma_counts_as_matrix(self):
        c = self._classify([
            "v_wmma_f32_16x16x16_f16", "buffer_load_u8", "buffer_load_u8",
        ])
        self.assertEqual(c.likely_operation, "quantized_gemm")

    def test_pure_dequant_becomes_repacking(self):
        # k_deq-style stream: extract (bfe/and/shift) + cvt + typed store,
        # no matrix op -> tensor_repacking (rubric maps this to dequant-repack)
        c = self._classify([
            "buffer_load_b32", "v_bfe_u32", "v_and_b32", "v_lshrrev_b32_e32",
            "v_cvt_f32_ubyte0_e32", "v_cvt_f16_f32_e32", "global_store_d16",
        ])
        self.assertEqual(c.likely_operation, "tensor_repacking")
        self.assertGreaterEqual(c.confidence, 0.5)

    def test_unclassified_on_no_motif(self):
        c = self._classify(["s_endpgm"])
        self.assertEqual(c.likely_operation, "unclassified")
        self.assertEqual(c.confidence, 0.0)

    def test_none_disassembly_degrades(self):
        c = classify.classify(None)
        self.assertIsNone(c.likely_operation)
        self.assertIsNone(c.confidence)
        self.assertIn("disassembly unavailable", c.error)

    def test_confidence_never_claims_certainty(self):
        c = self._classify(["v_dot2_f32_bf16", "buffer_load_u8",
                            "buffer_load_u8"])
        self.assertLessEqual(c.confidence, 0.95)


class TestToolchainRecord(unittest.TestCase):
    def test_toolchain_json_written_and_reparsed(self):
        # generate_catalogue writes + reads back; emulate here on catalogue.build
        paths, _gt = fixture_paths()
        cat, tc, _e = catalogue.build(paths, schema_path=SCHEMA_PATH)
        text = json.dumps(tc)
        back = json.loads(text)
        self.assertEqual(back["llvm_objdump"], tc["llvm_objdump"])
        self.assertIn("llvm_objdump", cat["toolchain"])
        d = tc["llvm_objdump"]
        if d["path"]:
            self.assertTrue(d["version"], "discovery succeeded: version recorded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
