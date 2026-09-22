"""Tests for mimo_halo.models.inventory.

Focus areas (per contract): range refusal, truncated header, packed logical
counts, scale roles, duplicates, and component boundaries.  All tests are
offline: HTTP behaviour is exercised through a scripted fake response object,
header/packing logic through synthetic safetensors structures.
"""

from __future__ import annotations

import json
import struct
import unittest
import urllib.error
from unittest.mock import patch

from mimo_halo.models.inventory import (
    HeaderError,
    HttpsRangeClient,
    InventoryError,
    PackingError,
    RangeNotRespectedError,
    SafetensorsHeader,
    build_inventory,
    classify_component,
    memory_report,
    parse_safetensors_header,
    tensor_role,
)

REPO = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
REV = "5711b268169967567844e1e560e8a3966da959b1"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status, headers, body):
        self.status = status
        self.headers = headers
        self._body = body

    def read(self, amount=-1):
        if amount < 0:
            data, self._body = self._body, b""
        else:
            data, self._body = self._body[:amount], self._body[amount:]
        return data

    def close(self):
        pass


def make_client():
    return HttpsRangeClient(timeout=1.0)


def run_client_with_responses(client, responses, stream_past_window=False):
    """Monkeypatch _open to yield scripted responses in order.

    ``stream_past_window`` scripts a server that keeps sending beyond the
    requested window; the default scripts one that stops at it.
    """

    def fake_open(url, rng):
        status, headers, body = responses.pop(0)
        if rng is not None:
            start, length = rng[0], rng[1] - rng[0] + 1
            window = (
                body[start:]
                if stream_past_window
                else body[start : start + length]
            )
            return status, rng[0], length, FakeResponse(status, headers, window)
        return status, 0, -1, FakeResponse(status, headers, body)

    return patch.object(client, "_open", side_effect=fake_open)


def header_bytes_for(tensors: dict, metadata: dict | None = None) -> bytes:
    doc = dict(tensors)
    if metadata is not None:
        doc["__metadata__"] = metadata
    raw = json.dumps(doc).encode("utf-8")
    # pad to 8-byte alignment like real safetensors writers
    pad = (8 - len(raw) % 8) % 8
    raw += b" " * pad
    return struct.pack("<Q", len(raw)) + raw


def synth_header(tensors: dict, name: str = "model.safetensors") -> SafetensorsHeader:
    blob = header_bytes_for(tensors)
    n = struct.unpack("<Q", blob[:8])[0]
    file_size = len(blob) + sum(
        t["data_offsets"][1] for t in tensors.values()
    )
    return parse_safetensors_header(name, f"https://x/{name}", blob[:8], blob[8:], file_size)


def mxp4_pair(out_features=4, in_features=32, start=0):
    """One mxfp4-packed weight + its per-32 scale sibling, like the real shards."""
    packed_cols = in_features // 2
    scale_cols = in_features // 32
    return {
        "layer.weight": {
            "dtype": "U8",
            "shape": [out_features, packed_cols],
            "data_offsets": [start, start + out_features * packed_cols],
        },
        "layer.weight_scale": {
            "dtype": "U8",
            "shape": [out_features, scale_cols],
            "data_offsets": [
                start + out_features * packed_cols,
                start + out_features * packed_cols + out_features * scale_cols,
            ],
        },
    }


BASE_CONFIG = {
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "moe_layer_freq": [0] + [1] * 47,
    "quantization_config": {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "store_dtype": "mxfp4",
        "mxfp4_block_size": 32,
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    },
}


# ---------------------------------------------------------------------------
# Range reader: refusal / truncation / caps
# ---------------------------------------------------------------------------


class RangeRefusalTests(unittest.TestCase):
    def test_200_full_response_aborts(self):
        client = make_client()
        body = b"x" * 4096
        responses = [
            (200, {"Content-Length": str(len(body))}, body),
        ]
        with run_client_with_responses(client, responses):
            with self.assertRaises(RangeNotRespectedError) as ctx:
                client.read_range("https://huggingface.co/f.safetensors", 0, 8)
        self.assertIn("ignored Range", str(ctx.exception))

    def test_non_206_status_refused(self):
        client = make_client()
        responses = [(403, {}, b"")]
        with run_client_with_responses(client, responses):
            with self.assertRaises(RangeNotRespectedError):
                client.read_range("https://huggingface.co/f.safetensors", 0, 8)

    def test_content_range_mismatch_refused(self):
        client = make_client()
        responses = [
            (206, {"Content-Range": "bytes 0-7/100"}, b"12345678"),
        ]
        with run_client_with_responses(client, responses):
            with self.assertRaises(RangeNotRespectedError):
                # requested 10-17 but server claims it served 0-7
                client.read_range("https://huggingface.co/f.safetensors", 10, 8)

    def test_206_without_content_range_refused(self):
        client = make_client()
        responses = [(206, {}, b"12345678")]
        with run_client_with_responses(client, responses):
            with self.assertRaises(RangeNotRespectedError):
                client.read_range("https://huggingface.co/f.safetensors", 0, 8)

    def test_http_redirect_target_refused(self):
        client = make_client()
        err = urllib.error.HTTPError(
            "https://huggingface.co/f",
            302,
            "Found",
            {"Location": "http://evil.example/f"},
            None,
        )
        with patch.object(client.opener, "open", side_effect=err):
            with self.assertRaises(InventoryError) as ctx:
                client.read_range("https://huggingface.co/f", 0, 8)
        self.assertIn("allowlisted", str(ctx.exception))

    def test_non_https_url_refused(self):
        client = make_client()
        with self.assertRaises(InventoryError):
            client.read_range("http://huggingface.co/f", 0, 8)

    def test_short_response_detected(self):
        client = make_client()
        responses = [
            (206, {"Content-Range": "bytes 0-7/100", "Content-Length": "8"}, b"12345"),
        ]
        with run_client_with_responses(client, responses):
            with self.assertRaises(InventoryError) as ctx:
                client.read_range("https://huggingface.co/f.safetensors", 0, 8)
        self.assertIn("truncated", str(ctx.exception))

    def test_extra_bytes_aborts_full_read(self):
        client = make_client()
        # server keeps streaming beyond the requested window
        body = b"12345678" + b"Z" * 1000
        responses = [
            (206, {"Content-Range": "bytes 0-7/2000"}, body),
        ]
        with run_client_with_responses(
            client, responses, stream_past_window=True
        ):
            with self.assertRaises(InventoryError) as ctx:
                client.read_range("https://huggingface.co/f.safetensors", 0, 8)
        self.assertIn("more than requested", str(ctx.exception))


class HeaderTruncationTests(unittest.TestCase):
    def test_n_smaller_than_8_rejected(self):
        blob = struct.pack("<Q", 4) + b"{}"
        with self.assertRaises(HeaderError) as ctx:
            parse_safetensors_header("f", "https://x/f", blob[:8], blob[8:], len(blob))
        self.assertIn("truncated", str(ctx.exception))

    def test_header_extends_past_file_rejected(self):
        blob = header_bytes_for(mxp4_pair())
        n = struct.unpack("<Q", blob[:8])[0]
        with self.assertRaises(HeaderError) as ctx:
            parse_safetensors_header("f", "https://x/f", blob[:8], blob[8:], 8 + n - 1)
        self.assertIn("truncated header", str(ctx.exception))

    def test_declared_n_exceeds_cap_rejected(self):
        blob = struct.pack("<Q", 1 << 30) + b"{}"
        with self.assertRaises(HeaderError):
            parse_safetensors_header("f", "https://x/f", blob[:8], blob[8:], 1 << 40)

    def test_fetch_refuses_second_range_when_n_exceeds_file(self):
        from mimo_halo.models.inventory import fetch_safetensors_header

        client = make_client()
        # prefix says the header needs 1 MiB but file is only 64 bytes
        prefix = struct.pack("<Q", 1 << 20)
        responses = [
            (206, {"Content-Range": "bytes 0-7/64"}, prefix + b"payload"),
        ]
        with run_client_with_responses(client, responses):
            with self.assertRaises(HeaderError):
                fetch_safetensors_header(client, "f", "https://huggingface.co/f")


# ---------------------------------------------------------------------------
# Header structure: duplicates / offsets / dtypes
# ---------------------------------------------------------------------------


class HeaderStructureTests(unittest.TestCase):
    def test_duplicate_json_keys_rejected(self):
        raw = b'{"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}, "a": '
        raw += b'{"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}'
        blob = struct.pack("<Q", len(raw)) + raw
        with self.assertRaises(HeaderError) as ctx:
            parse_safetensors_header(
                "f", "https://x/f", blob[:8], blob[8:], len(blob)
            )
        self.assertIn("duplicate JSON key", str(ctx.exception))

    def test_overlapping_offsets_rejected(self):
        tensors = {
            "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
        }
        with self.assertRaises(HeaderError) as ctx:
            synth_header(tensors)
        self.assertIn("overlap", str(ctx.exception))

    def test_span_mismatch_rejected(self):
        tensors = {"a": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 4]}}
        with self.assertRaises(HeaderError) as ctx:
            synth_header(tensors)  # 4 elements x 2 bytes = 8, span is 4
        self.assertIn("span", str(ctx.exception))

    def test_unknown_dtype_rejected(self):
        tensors = {"a": {"dtype": "FP4_MADEUP", "shape": [4], "data_offsets": [0, 4]}}
        with self.assertRaises(HeaderError):
            synth_header(tensors)

    def test_valid_header_roundtrip(self):
        header = synth_header(mxp4_pair())
        self.assertEqual(header.tensors["layer.weight"]["dtype"], "U8")
        self.assertIsNone(header.metadata)


# ---------------------------------------------------------------------------
# Logical counting: packed weights, scales, unknown packing
# ---------------------------------------------------------------------------


class PackedLogicalCountTests(unittest.TestCase):
    def test_packed_mxfp4_doubles_last_dim(self):
        header = synth_header(mxp4_pair(out_features=4, in_features=32))
        from mimo_halo.models.inventory import logical_view

        config = dict(BASE_CONFIG)
        index_meta = {"save_format": "mxfp4"}
        logical, params = logical_view(
            "layer.weight", "U8", [4, 16], "parameter", header, config, index_meta
        )
        self.assertEqual(logical, [4, 32])
        self.assertEqual(params, 128)  # 2 nibbles per stored byte

    def test_scale_role_has_zero_logical_parameters(self):
        header = synth_header(mxp4_pair())
        from mimo_halo.models.inventory import logical_view

        config = dict(BASE_CONFIG)
        logical, params = logical_view(
            "layer.weight_scale", "U8", [4, 1], "quantization_auxiliary",
            header, config, {"save_format": "mxfp4"},
        )
        self.assertEqual(params, 0)
        self.assertEqual(logical, [4, 1])

    def test_u8_without_scale_sibling_fails(self):
        header = synth_header(
            {"w": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
        )
        from mimo_halo.models.inventory import logical_view

        with self.assertRaises(PackingError):
            logical_view(
                "w", "U8", [4], "parameter", header, dict(BASE_CONFIG),
                {"save_format": "mxfp4"},
            )

    def test_u8_without_mxfp4_config_fails(self):
        header = synth_header(mxp4_pair())
        from mimo_halo.models.inventory import logical_view

        config = dict(BASE_CONFIG)
        config["quantization_config"] = {"quant_method": "fp8", "store_dtype": "fp8"}
        with self.assertRaises(PackingError):
            logical_view(
                "layer.weight", "U8", [4, 16], "parameter", header, config,
                {"save_format": "mxfp4"},
            )

    def test_fp8_is_one_byte_per_parameter(self):
        header = synth_header(
            {"w": {"dtype": "F8_E4M3", "shape": [128, 128], "data_offsets": [0, 16384]}}
        )
        from mimo_halo.models.inventory import logical_view

        logical, params = logical_view(
            "w", "F8_E4M3", [128, 128], "parameter", header, dict(BASE_CONFIG), {}
        )
        self.assertEqual(params, 16384)
        self.assertEqual(logical, [128, 128])

    def test_bf16_is_exact_dense(self):
        header = synth_header(
            {"w": {"dtype": "BF16", "shape": [10], "data_offsets": [0, 20]}}
        )
        from mimo_halo.models.inventory import logical_view

        _, params = logical_view(
            "w", "BF16", [10], "parameter", header, dict(BASE_CONFIG), {}
        )
        self.assertEqual(params, 10)

    def test_role_classification(self):
        self.assertEqual(tensor_role("x.weight"), "parameter")
        self.assertEqual(tensor_role("x.weight_scale"), "quantization_auxiliary")
        self.assertEqual(tensor_role("x.weight_scale_inv"), "quantization_auxiliary")
        self.assertEqual(tensor_role("x.bias"), "parameter")
        self.assertEqual(tensor_role("x.attn.sinks"), "parameter")


# ---------------------------------------------------------------------------
# Component boundaries and inventory totals
# ---------------------------------------------------------------------------


class ComponentTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(classify_component("model.layers.3.mlp.gate.weight", "f"), "text")
        self.assertEqual(classify_component("model.mtp.layers.0.enorm.weight", "f"), "mtp")
        self.assertEqual(classify_component("anything", "model_mtp.safetensors"), "mtp")
        self.assertEqual(classify_component("anything", "dflash/dflash_draft_model.safetensors"), "dflash")
        self.assertEqual(classify_component("visual.blocks.2.attn.proj.weight", "f"), "vision")
        self.assertEqual(classify_component("audio_encoder.norm.weight", "f"), "audio")
        self.assertEqual(classify_component("speech_embeddings.0.weight", "f"), "audio")
        self.assertEqual(classify_component("model.embed_tokens.weight", "f"), "text")
        self.assertEqual(classify_component("lm_head.weight", "f"), "text")

    def test_layer_and_expert_parsed(self):
        name = "model.layers.7.mlp.experts.42.gate_proj.weight"
        pair = mxp4_pair()
        header = synth_header(
            {name: pair["layer.weight"], name + "_scale": pair["layer.weight_scale"]}
        )
        from mimo_halo.models.inventory import build_tensor_record

        record = build_tensor_record(
            name,
            header.tensors[name],
            header,
            "model_pp0_ep0_shard0.safetensors",
            dict(BASE_CONFIG),
            {"save_format": "mxfp4"},
        )
        self.assertEqual(record["layer"], 7)
        self.assertEqual(record["expert"], 42)
        self.assertEqual(record["component"], "text")
        self.assertEqual(record["logical_shape"], [4, 32])

    def test_inventory_totals_separate_components(self):
        config = dict(BASE_CONFIG)
        index = {
            "metadata": {"save_format": "mxfp4", "tp_size": 4, "total_size": 138},
            "weight_map": {
                "model.layers.1.mlp.experts.0.gate_proj.weight": "a.safetensors",
                "model.layers.1.mlp.experts.0.gate_proj.weight_scale": "a.safetensors",
                "model.embed_tokens.weight": "a.safetensors",
                "model.mtp.layers.0.enorm.weight": "m.safetensors",
                "visual.patch_embed.proj.weight": "a.safetensors",
            },
        }
        dflash_index = {
            "metadata": {"total_size": 8},
            "weight_map": {
                # bare filename as the dflash-dir index declares it; the
                # dflash/-prefixed header path is what classify_component sees
                "layers.0.self_attn.q_proj.weight": "dflash_draft_model.safetensors"
            },
        }
        tensors = {
            "model.layers.1.mlp.experts.0.gate_proj.weight": {
                "dtype": "U8", "shape": [2, 16], "data_offsets": [0, 32],
            },
            "model.layers.1.mlp.experts.0.gate_proj.weight_scale": {
                "dtype": "U8", "shape": [2, 1], "data_offsets": [32, 34],
            },
            "model.embed_tokens.weight": {
                "dtype": "BF16", "shape": [4, 4], "data_offsets": [34, 66],
            },
            "model.mtp.layers.0.enorm.weight": {
                "dtype": "BF16", "shape": [4], "data_offsets": [0, 8],
            },
            "visual.patch_embed.proj.weight": {
                "dtype": "BF16", "shape": [8, 4], "data_offsets": [66, 130],
            },
            "layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16", "shape": [4], "data_offsets": [0, 8],
            },
        }
        a_names = [
            k for k, f in index["weight_map"].items() if f == "a.safetensors"
        ]
        headers = {
            "a.safetensors": synth_header(
                {k: tensors[k] for k in a_names}, "a.safetensors"
            ),
            "m.safetensors": synth_header(
                {"model.mtp.layers.0.enorm.weight": tensors["model.mtp.layers.0.enorm.weight"]},
                "m.safetensors",
            ),
            "dflash/dflash_draft_model.safetensors": synth_header(
                {"layers.0.self_attn.q_proj.weight": tensors["layers.0.self_attn.q_proj.weight"]},
                "dflash/dflash_draft_model.safetensors",
            ),
        }
        inventory = build_inventory(
            config, index, dict(BASE_CONFIG), dflash_index, headers, []
        )
        totals = inventory["totals"]["by_component"]
        self.assertEqual(set(totals), {"text", "mtp", "vision", "dflash"})
        # packed expert: 32 stored bytes -> 64 logical; scale: 2 stored -> 0 logical
        self.assertEqual(totals["text"]["logical_parameters"], 64 + 16)
        self.assertEqual(totals["text"]["quantization_auxiliary_bytes"], 2)
        self.assertEqual(totals["mtp"]["logical_parameters"], 4)
        self.assertEqual(totals["dflash"]["logical_parameters"], 4)
        grand = inventory["totals"]["grand"]
        self.assertEqual(grand["tensors"], 6)
        self.assertEqual(inventory["architecture"]["original_experts_per_layer"], 256)
        self.assertEqual(inventory["architecture"]["moe_layers"], list(range(1, 48)))


# ---------------------------------------------------------------------------
# Index <-> header mapping / duplicates across shards
# ---------------------------------------------------------------------------


class MappingTests(unittest.TestCase):
    def _build(self, weight_map, headers, index_meta=None):
        config = dict(BASE_CONFIG)
        index = {
            "metadata": index_meta or {"save_format": "mxfp4", "total_size": 0},
            "weight_map": weight_map,
        }
        dflash_index = {"metadata": {}, "weight_map": {}}
        return build_inventory(config, index, dict(BASE_CONFIG), dflash_index, headers, [])

    def test_header_file_absent_from_index_files_rejected(self):
        tensors = mxp4_pair()
        header = synth_header(tensors, "a.safetensors")
        # a.safetensors claims tensor names the index maps to b.safetensors
        with self.assertRaises(HeaderError) as ctx:
            self._build(
                {"layer.weight": "b.safetensors", "layer.weight_scale": "b.safetensors"},
                {"a.safetensors": header},
            )
        self.assertIn("without headers", str(ctx.exception))

    def test_index_header_mismatch_rejected(self):
        tensors = mxp4_pair()
        header = synth_header(tensors, "a.safetensors")
        # index claims layer.weight lives in b.safetensors
        with self.assertRaises(HeaderError) as ctx:
            self._build(
                {"layer.weight": "b.safetensors", "layer.weight_scale": "a.safetensors"},
                {"a.safetensors": header},
            )
        self.assertIn("mismatch", str(ctx.exception))

    def test_missing_header_file_rejected(self):
        tensors = mxp4_pair()
        header = synth_header(tensors, "a.safetensors")
        with self.assertRaises(HeaderError) as ctx:
            self._build(
                {"layer.weight": "a.safetensors", "layer.weight_scale": "a.safetensors",
                 "other.weight": "c.safetensors"},
                {"a.safetensors": header},
            )
        self.assertIn("without headers", str(ctx.exception))

    def test_declared_total_mismatch_rejected(self):
        tensors = mxp4_pair()  # 36 stored bytes
        header = synth_header(tensors, "a.safetensors")
        with self.assertRaises(HeaderError) as ctx:
            self._build(
                {"layer.weight": "a.safetensors", "layer.weight_scale": "a.safetensors"},
                {"a.safetensors": header},
                index_meta={"save_format": "mxfp4", "total_size": 999},
            )
        self.assertIn("total_size", str(ctx.exception))


# ---------------------------------------------------------------------------
# Memory report: exact prune-candidate arithmetic
# ---------------------------------------------------------------------------


class MemoryReportTests(unittest.TestCase):
    def test_candidates_scale_exactly(self):
        # build a tiny inventory: 2 experts, 1 moe layer
        config = dict(BASE_CONFIG)
        config["n_routed_experts"] = 2
        config["moe_layer_freq"] = [0, 1]
        weight_map = {}
        tensors = {}
        offset = 0
        for e in range(2):
            pair = mxp4_pair(out_features=2, in_features=32, start=offset)
            for name, entry in pair.items():
                full = f"model.layers.1.mlp.experts.{e}.{name}"
                weight_map[full] = "a.safetensors"
                tensors[full] = entry
            offset += sum(
                entry["data_offsets"][1] - entry["data_offsets"][0]
                for entry in pair.values()
            )
        # fixed non-expert tensor
        weight_map["model.embed_tokens.weight"] = "a.safetensors"
        tensors["model.embed_tokens.weight"] = {
            "dtype": "BF16", "shape": [4], "data_offsets": [offset, offset + 8],
        }
        offset += 8
        header = synth_header(tensors, "a.safetensors")
        index = {
            "metadata": {"save_format": "mxfp4", "total_size": offset},
            "weight_map": weight_map,
        }
        inventory = build_inventory(
            config, index, dict(BASE_CONFIG),
            {"metadata": {}, "weight_map": {}}, {"a.safetensors": header}, [],
        )
        report = memory_report(inventory)
        # Default candidates: the original expert count only; 144/160/176/192
        # out of 2 experts must be absent, not fabricated.
        self.assertEqual(
            {c["retained_experts"] for c in report["prune_candidates"]}, {2}
        )

        report = memory_report(inventory, candidate_counts=[2, 1])
        native = report["native"]
        # Per retained expert index (== per layer-instance here: one MoE
        # layer): weight 2x16=32 bytes + scale 2x1=2 bytes = 34 stored, and
        # exactly 4.25 bpw -> 34 * 8 / 4.25 = 64 logical (scales count 0).
        self.assertEqual(
            native["per_retained_expert_across_moe_layers_stored_bytes"], 34
        )
        self.assertEqual(
            native["per_retained_expert_across_moe_layers_logical_parameters"],
            64,
        )
        layout = native["per_layer_expert_layout"]
        self.assertEqual(layout["stored_bytes"], 34)
        self.assertEqual(layout["logical_parameters"], 64)
        self.assertEqual(layout["tensors_per_expert_instance"], 2)
        self.assertEqual(
            report["precision_costs"]["mxfp4_native_u8_packed"][
                "bits_per_logical_parameter_exact"
            ],
            4.25,
        )
        by_retained = {
            c["retained_experts"]: c for c in report["prune_candidates"]
        }
        # non-expert text here is embed [4] BF16: 8 stored bytes, 4 logical
        self.assertEqual(set(by_retained), {2, 1})
        self.assertEqual(by_retained[2]["native_text_payload_bytes"], 8 + 34 * 2)
        self.assertEqual(by_retained[1]["native_text_payload_bytes"], 8 + 34)
        self.assertEqual(by_retained[2]["logical_parameters"], 4 + 64 * 2)
        self.assertEqual(by_retained[1]["logical_parameters"], 4 + 64)

    def test_expert_nonuniformity_fails(self):
        config = dict(BASE_CONFIG)
        config["n_routed_experts"] = 2
        config["moe_layer_freq"] = [0, 1]
        weight_map = {}
        tensors = {}
        offset = 0
        for e, out_features in enumerate((2, 4)):  # expert 1 differs, but the
            # total stays divisible by the expert count so the uniformity
            # check (not the divisibility check) is what fires
            pair = mxp4_pair(out_features=out_features, in_features=32, start=offset)
            for name, entry in pair.items():
                full = f"model.layers.1.mlp.experts.{e}.{name}"
                weight_map[full] = "a.safetensors"
                tensors[full] = entry
            offset += sum(
                entry["data_offsets"][1] - entry["data_offsets"][0]
                for entry in pair.values()
            )
        header = synth_header(tensors, "a.safetensors")
        index = {
            "metadata": {"save_format": "mxfp4", "total_size": offset},
            "weight_map": weight_map,
        }
        inventory = build_inventory(
            config, index, dict(BASE_CONFIG),
            {"metadata": {}, "weight_map": {}}, {"a.safetensors": header}, [],
        )
        with self.assertRaises(InventoryError) as ctx:
            memory_report(inventory)
        self.assertIn("differs from expert 0", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
