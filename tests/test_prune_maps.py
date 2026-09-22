"""Tests for src/mimo_halo/pruning — dry-run prune-map planning.

Synthetic inventories only: small fixtures plus one full 47-layer /
256-expert synthetic dry-run at 160 and 176. Every failure case asserts the
CLI fails closed with exit code 1 and a named reason. No weights, no
network, stdlib only.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimo_halo.pruning import (  # noqa: E402
    PruneMapError,
    generate_selection,
    load_inventory,
    parse_external_selection,
)
from mimo_halo.pruning.maps import main  # noqa: E402

HIDDEN = 32
INTER = 32
BLOCK = 32
MXFP4_DTYPE = "U8_mxfp4_packed"
ROUTER_DTYPE = "BF16"


def _mxfp4_expert_tensors(layer: int, expert: int, dtype: str = MXFP4_DTYPE) -> list[dict]:
    """One expert: 3 packed weights (2 nibbles/byte) + 3 per-32-block scales."""
    tensors = []
    for proj, logical in (
        ("gate_proj", [INTER, HIDDEN]),
        ("up_proj", [INTER, HIDDEN]),
        ("down_proj", [HIDDEN, INTER]),
    ):
        stored = logical.copy()
        stored[-1] //= 2  # two nibbles per stored byte
        tensors.append(
            {
                "name": f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
                "shape": stored,
                "dtype": dtype,
                "stored_bytes": stored[0] * stored[1],
                "logical_shape": logical,
                "logical_parameters": logical[0] * logical[1],
                "role": "parameter",
                "layer": layer,
                "expert": expert,
                "component": "text/main",
                "quantization": {"method": "mxfp4", "block_size": BLOCK},
            }
        )
        scale = logical.copy()
        scale[-1] //= BLOCK
        tensors.append(
            {
                "name": f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight_scale",
                "shape": scale,
                "dtype": "F32",
                "stored_bytes": scale[0] * scale[1] * 4,
                "logical_shape": scale,
                "logical_parameters": 0,
                "role": "quantization_auxiliary",
                "layer": layer,
                "expert": expert,
                "component": "text/main",
            }
        )
    return tensors


def make_inventory(
    moe_layer_ids: list[int],
    experts: int = 256,
    *,
    component: str | None = "text/main",
    router_shape: list[int] | None = None,
    router_bias_present: bool = True,
    drop_projection: tuple[int, int, str, str] | None = None,
    expert_dtype_override: dict[int, str] | None = None,
    include_vision: bool = True,
) -> dict:
    """Synthetic inventory matching the schema_version=1 reader contract."""
    dense_layers = sorted(set(range(max(moe_layer_ids) + 1)) - set(moe_layer_ids))
    tensors: list[dict] = []
    comp = {"component": component} if component is not None else {}

    for layer in dense_layers:  # dense FFN layers (e.g. layer 0)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tensors.append(
                {
                    "name": f"model.layers.{layer}.mlp.{proj}.weight",
                    "shape": [INTER, HIDDEN],
                    "dtype": "FP8_e4m3",
                    "stored_bytes": INTER * HIDDEN,
                    "logical_shape": [INTER, HIDDEN],
                    "logical_parameters": INTER * HIDDEN,
                    "role": "parameter",
                    "layer": layer,
                    **comp,
                }
            )
            tensors.append(
                {
                    "name": f"model.layers.{layer}.mlp.{proj}.weight_scale_inv",
                    "shape": [1, 1],
                    "dtype": "F32",
                    "stored_bytes": 4,
                    "logical_shape": [1, 1],
                    "logical_parameters": 0,
                    "role": "quantization_auxiliary",
                    "layer": layer,
                    **comp,
                }
            )

    dtype_override = expert_dtype_override or {}
    for layer in moe_layer_ids:
        tensors.append(
            {
                "name": f"model.layers.{layer}.mlp.gate.weight",
                "shape": router_shape or [experts, HIDDEN],
                "dtype": ROUTER_DTYPE,
                "stored_bytes": (router_shape or [experts, HIDDEN])[0] * HIDDEN * 2,
                "logical_shape": router_shape or [experts, HIDDEN],
                "logical_parameters": (router_shape or [experts, HIDDEN])[0] * HIDDEN,
                "role": "parameter",
                "layer": layer,
                **comp,
            }
        )
        if router_bias_present:
            tensors.append(
                {
                    "name": f"model.layers.{layer}.mlp.gate.e_score_correction_bias",
                    "shape": [experts],
                    "dtype": "F32",
                    "stored_bytes": experts * 4,
                    "logical_shape": [experts],
                    "logical_parameters": experts,
                    "role": "parameter",
                    "layer": layer,
                    **comp,
                }
            )
        for expert in range(experts):
            entries = _mxfp4_expert_tensors(layer, expert)
            if drop_projection and (layer, expert) == drop_projection[:2]:
                entries = [
                    t for t in entries if not t["name"].endswith("." + drop_projection[2] + "." + drop_projection[3])
                ]
            dtype = dtype_override.get(layer, MXFP4_DTYPE)
            for entry in entries:
                if entry["role"] == "parameter":
                    entry["dtype"] = dtype
            tensors.extend(entries)

    if include_vision:
        tensors.append(
            {
                "name": "visual.patch_embed.proj.weight",
                "shape": [8, 8],
                "dtype": "BF16",
                "stored_bytes": 128,
                "logical_shape": [8, 8],
                "logical_parameters": 64,
                "role": "parameter",
                "component": "vision",
            }
        )

    return {
        "schema_version": 1,
        "source": {
            "repo_id": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
            "revision": "5711b268169967567844e1e560e8a3966da959b1",
            "files": ["model_pp0_ep0_shard0.safetensors"],
        },
        "architecture": {
            "original_experts_per_layer": experts,
            "top_k": 8,
            "moe_layers": list(moe_layer_ids),
        },
        "tensors": tensors,
    }


class PruneMapsTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_json(self, name: str, payload: object) -> str:
        path = self.dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()


class ShapeOnlyDryRunTest(PruneMapsTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.inv_path = self.write_json("inv.json", make_inventory([1, 2], experts=8))
        self.map_path = str(self.dir / "map.json")

    def load_map(self) -> dict:
        return json.loads(Path(self.map_path).read_text(encoding="utf-8"))

    def test_end_to_end_shape_only_dry_run(self) -> None:
        code, _, err = self.run_cli(
            ["--inventory", self.inv_path, "--retained-count", "4", "--seed", "7",
             "--output", self.map_path]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("mode=shape_only", err)
        m = self.load_map()
        self.assertEqual(m["schema_version"], 1)
        self.assertEqual(m["kind"], "prune_map")
        self.assertFalse(m["applied"])
        self.assertFalse(m["quality_map"])
        self.assertIn("NOT a REAP/HOPE quality map", " ".join(m["warnings"]))
        self.assertIn("NOT a REAP/HOPE quality map", m["provenance"]["disclaimer"])
        self.assertEqual(m["source"]["revision"], "5711b268169967567844e1e560e8a3966da959b1")
        self.assertEqual(m["architecture"]["original_experts_per_layer"], 8)
        self.assertEqual(m["retained_count"], 4)

        # Exact per-layer math from fixture constants.
        expert_weight_params = 3 * INTER * HIDDEN          # 3 packed projections
        expert_scale_bytes = 3 * (INTER * (HIDDEN // BLOCK)) * 4
        expert_weight_bytes = 3 * (INTER * (HIDDEN // 2))
        for entry in m["layers"]:
            self.assertEqual(len(entry["old_to_new"]), 8)
            self.assertEqual(
                [k for k, v in entry["old_to_new"].items() if v is not None],
                [str(i) for i in entry["retained"]],
            )
            self.assertEqual(sorted(v for v in entry["old_to_new"].values() if v is not None), [0, 1, 2, 3])
            self.assertEqual(entry["pruned"], sorted(entry["pruned"]))
            self.assertEqual(set(entry["retained"]) | set(entry["pruned"]), set(range(8)))
            self.assertEqual(entry["tensor_counts"]["keep"], 4 * 6)
            self.assertEqual(entry["tensor_counts"]["drop"], 4 * 6)
            self.assertEqual(entry["tensor_counts"]["total"], 48)
            self.assertEqual(entry["logical_parameters"]["keep"], 4 * expert_weight_params)
            self.assertEqual(entry["logical_parameters"]["drop"], 4 * expert_weight_params)
            self.assertEqual(entry["stored_bytes"]["keep"], 4 * (expert_weight_bytes + expert_scale_bytes))
            router = entry["router"]
            self.assertEqual(router["weight"]["rows_kept"], entry["retained"])
            self.assertEqual(router["weight"]["new_logical_shape"], [4, HIDDEN])
            self.assertEqual(router["weight"]["new_logical_parameters"], 4 * HIDDEN)
            self.assertEqual(router["e_score_correction_bias"]["entries_kept"], entry["retained"])
            self.assertEqual(router["e_score_correction_bias"]["new_logical_parameters"], 4)
            # Dense quantization-null BF16/F32 routers: exact row-scaled bytes.
            self.assertEqual(router["weight"]["new_stored_bytes"], 4 * HIDDEN * 2)
            self.assertEqual(router["e_score_correction_bias"]["new_stored_bytes"], 4 * 4)
            self.assertIn("top-8", router["plan"])
            # scale tensors travel with their kept weights
            for name in entry["expert_tensors"]["keep"]:
                self.assertRegex(name, r"(weight|weight_scale)$")

        totals = m["totals"]
        self.assertEqual(totals["expert_tensors_kept"], 2 * 4 * 6)
        self.assertEqual(totals["expert_tensors_dropped"], 2 * 4 * 6)
        self.assertEqual(totals["logical_parameters_dropped"], 2 * 4 * expert_weight_params)
        self.assertEqual(
            totals["stored_bytes_dropped"], 2 * 4 * (expert_weight_bytes + expert_scale_bytes)
        )
        # untouched = dense layer 0 (6 tensors) + vision (1)
        self.assertEqual(totals["unchanged_tensors"], 7)
        self.assertEqual(totals["unchanged_logical_parameters"], 3 * INTER * HIDDEN + 64)

        # Candidate block follows the authoritative baseline comparison schema.
        c = m["candidate"]
        self.assertEqual(c["schema_version"], 1)
        self.assertEqual(len(c["layers"]), 2)
        for entry, layer_entry in zip(c["layers"], m["layers"]):
            self.assertEqual(
                set(entry),
                {"layer", "original_expert_count", "retained_expert_ids", "pruned_expert_ids"},
            )
            self.assertEqual(entry["original_expert_count"], 8)
            self.assertEqual(entry["retained_expert_ids"], layer_entry["retained"])
            self.assertEqual(entry["pruned_expert_ids"], layer_entry["pruned"])

    def test_full_sum_closure_including_router(self) -> None:
        """Regression: router tensors appear in totals and close the inventory sum."""
        inv = load_inventory(self.inv_path)
        code, _, err = self.run_cli(
            ["--inventory", self.inv_path, "--retained-count", "4", "--seed", "7",
             "--output", self.map_path]
        )
        self.assertEqual(code, 0, err)
        m = self.load_map()
        totals = m["totals"]

        # Known router bytes per layer: BF16 gate weight [8, HIDDEN] + F32 bias [8].
        router_weight_bytes = 8 * HIDDEN * 2
        router_bias_bytes = 8 * 4
        per_layer_router_bytes = router_weight_bytes + router_bias_bytes
        # Post-remap at 4 retained rows: 4 * HIDDEN * 2 + 4 * 4 (exact row scale).
        per_layer_router_post_bytes = 4 * HIDDEN * 2 + 4 * 4
        self.assertEqual(totals["router_tensors"], 4)
        self.assertEqual(totals["router_stored_bytes"], 2 * per_layer_router_bytes)
        self.assertEqual(totals["router_logical_parameters"], 2 * (8 * HIDDEN + 8))
        for entry in m["layers"]:
            accounting = entry["router"]["accounting"]
            self.assertEqual(accounting["tensor_count"], 2)
            self.assertEqual(accounting["original_stored_bytes"], per_layer_router_bytes)
            self.assertEqual(accounting["original_logical_parameters"], 8 * HIDDEN + 8)
            self.assertEqual(accounting["post_remap_logical_parameters"], 4 * HIDDEN + 4)
            # Dense quantization-null BF16/F32 routers scale exactly:
            # old_bytes // original_rows * retained_rows — never a guessed figure.
            self.assertEqual(
                accounting["post_remap_stored_bytes"], per_layer_router_post_bytes
            )
            self.assertTrue(accounting["post_remap_stored_bytes_known"])
        self.assertEqual(
            totals["router_stored_bytes_post_remap"],
            2 * per_layer_router_post_bytes,
        )
        self.assertTrue(totals["router_stored_bytes_post_remap_known"])

        # Full sum: kept + dropped + router + unchanged == every inventory tensor.
        self.assertEqual(
            totals["stored_bytes_kept"] + totals["stored_bytes_dropped"]
            + totals["router_stored_bytes"] + totals["unchanged_stored_bytes"],
            sum(t["stored_bytes"] for t in inv["tensors"]),
        )
        self.assertEqual(
            totals["logical_parameters_kept"] + totals["logical_parameters_dropped"]
            + totals["router_logical_parameters"] + totals["unchanged_logical_parameters"],
            sum(t["logical_parameters"] or 0 for t in inv["tensors"]),
        )
        self.assertEqual(
            totals["expert_tensors_kept"] + totals["expert_tensors_dropped"]
            + totals["router_tensors"] + totals["unchanged_tensors"],
            len(inv["tensors"]),
        )
        closure = totals["accounting_closure"]
        self.assertTrue(closure["closes"])
        self.assertEqual(closure["accounted_tensors"], closure["inventory_tensors"])
        self.assertEqual(closure["accounted_stored_bytes"], closure["inventory_stored_bytes"])
        self.assertEqual(
            closure["accounted_logical_parameters"], closure["inventory_logical_parameters"]
        )

        # Candidate parameter count includes the REMAPPED router, not kept+unchanged alone.
        expert_weight_params = 3 * INTER * HIDDEN
        candidate_params = (
            totals["logical_parameters_kept"]
            + totals["unchanged_logical_parameters"]
            + totals["logical_parameters_router_remapped"]
        )
        self.assertEqual(
            candidate_params,
            2 * 4 * expert_weight_params + (3 * INTER * HIDDEN + 64) + 2 * (4 * HIDDEN + 4),
        )
        self.assertNotEqual(
            candidate_params,
            totals["logical_parameters_kept"] + totals["unchanged_logical_parameters"],
        )
        # Remapped router params differ from the originals used in closure.
        self.assertEqual(totals["logical_parameters_router_remapped"], 2 * (4 * HIDDEN + 4))
        self.assertLess(
            totals["logical_parameters_router_remapped"], totals["router_logical_parameters"]
        )

    def test_candidate_out_writes_compare_compatible_file(self) -> None:
        cand_path = str(self.dir / "candidate.json")
        code, _, err = self.run_cli(
            ["--inventory", self.inv_path, "--retained-count", "4", "--seed", "3",
             "--output", self.map_path, "--candidate-out", cand_path]
        )
        self.assertEqual(code, 0, err)
        cand = json.loads(Path(cand_path).read_text(encoding="utf-8"))
        self.assertEqual(cand["schema_version"], 1)
        self.assertEqual(
            set(cand["layers"][0]),
            {"layer", "original_expert_count", "retained_expert_ids", "pruned_expert_ids"},
        )

    def test_deterministic_same_seed_different_seed(self) -> None:
        inv = load_inventory(self.inv_path)
        a = generate_selection(inv["architecture"], 4, 7)
        b = generate_selection(inv["architecture"], 4, 7)
        c = generate_selection(inv["architecture"], 4, 8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_component_name_prefix_fallback(self) -> None:
        inv = make_inventory([1], experts=4, component=None)
        self.assertEqual(load_inventory(self.write_json("inv2.json", inv))["tensors"][0].get("component"), None)
        code, _, err = self.run_cli(
            ["--inventory", self.write_json("inv3.json", make_inventory([1], experts=4, component=None)),
             "--retained-count", "2", "--seed", "1", "--output", self.map_path]
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.load_map()["layers"]), 1)

    def test_post_prune_totals_by_component(self) -> None:
        """Text bucket = prunable stack only; other components untouched."""
        inv = load_inventory(self.inv_path)
        code, _, err = self.run_cli(
            ["--inventory", self.inv_path, "--retained-count", "4", "--seed", "7",
             "--output", self.map_path]
        )
        self.assertEqual(code, 0, err)
        totals = self.load_map()["totals"]
        by = totals["post_prune_by_component"]
        self.assertEqual(list(by), ["text/main", "vision"])

        expert_weight_bytes = 3 * (INTER * (HIDDEN // 2))
        expert_scale_bytes = 3 * (INTER * (HIDDEN // BLOCK)) * 4
        expert_weight_params = 3 * INTER * HIDDEN
        kept_bytes = 2 * 4 * (expert_weight_bytes + expert_scale_bytes)
        kept_params = 2 * 4 * expert_weight_params
        dense_text_bytes = 3 * INTER * HIDDEN + 3 * 4   # layer-0 FFN + scale_inv
        dense_text_params = 3 * INTER * HIDDEN
        router_post_bytes = 2 * (4 * HIDDEN * 2 + 4 * 4)
        router_post_params = 2 * (4 * HIDDEN + 4)

        text = by["text/main"]
        self.assertTrue(text["stored_bytes_known"])
        self.assertEqual(
            text["stored_bytes"], kept_bytes + dense_text_bytes + router_post_bytes
        )
        self.assertEqual(
            text["logical_parameters"],
            kept_params + dense_text_params + router_post_params,
        )
        self.assertEqual(text["tensor_count"], 2 * 4 * 6 + 2 * 2 + 6)

        vision = by["vision"]
        self.assertTrue(vision["stored_bytes_known"])
        self.assertEqual(vision["stored_bytes"], 128)
        self.assertEqual(vision["logical_parameters"], 64)
        self.assertEqual(vision["tensor_count"], 1)

        # Exact identity behind the memory-report comparison: text post-prune
        # = inventory text total - dropped expert payload - router row delta.
        inv_text_bytes = sum(
            t["stored_bytes"]
            for t in inv["tensors"]
            if t.get("component") == "text/main"
        )
        inv_text_params = sum(
            t["logical_parameters"] or 0
            for t in inv["tensors"]
            if t.get("component") == "text/main"
        )
        router_delta_bytes = 2 * (8 - 4) * (HIDDEN * 2 + 4)
        router_delta_params = 2 * (8 - 4) * (HIDDEN + 1)
        self.assertEqual(
            text["stored_bytes"],
            inv_text_bytes - totals["stored_bytes_dropped"] - router_delta_bytes,
        )
        self.assertEqual(
            text["logical_parameters"],
            inv_text_params
            - totals["logical_parameters_dropped"]
            - router_delta_params,
        )

    def test_packed_router_post_bytes_stay_unknown(self) -> None:
        """Packed router weights keep an explicit null; the cost is never guessed."""
        inv = make_inventory([1, 2], experts=8)
        for t in inv["tensors"]:
            if t["name"].endswith("mlp.gate.weight"):
                t["quantization"] = {"method": "mxfp4", "block_size": BLOCK}
        path = self.write_json("inv_packed_router.json", inv)
        code, _, err = self.run_cli(
            ["--inventory", path, "--retained-count", "4", "--seed", "7",
             "--output", self.map_path]
        )
        self.assertEqual(code, 0, err)
        m = self.load_map()
        for entry in m["layers"]:
            accounting = entry["router"]["accounting"]
            self.assertIsNone(accounting["post_remap_stored_bytes"])
            self.assertFalse(accounting["post_remap_stored_bytes_known"])
            self.assertIn("unknown", accounting["post_remap_stored_bytes_note"])
            self.assertIsNone(entry["router"]["weight"]["new_stored_bytes"])
            # the dense F32 bias is still exactly row-scaled
            self.assertEqual(
                entry["router"]["e_score_correction_bias"]["new_stored_bytes"], 4 * 4
            )
        totals = m["totals"]
        self.assertIsNone(totals["router_stored_bytes_post_remap"])
        self.assertFalse(totals["router_stored_bytes_post_remap_known"])
        text = totals["post_prune_by_component"]["text/main"]
        self.assertIsNone(text["stored_bytes"])
        self.assertFalse(text["stored_bytes_known"])
        # The logical side stays exact regardless of byte-cost unknowns.
        self.assertEqual(
            text["logical_parameters"],
            2 * 4 * 3 * INTER * HIDDEN
            + 3 * INTER * HIDDEN
            + 2 * (4 * HIDDEN + 4),
        )
        # Original accounting closure still closes on original router bytes.
        closure = totals["accounting_closure"]
        self.assertTrue(closure["closes"])
        self.assertEqual(
            closure["accounted_stored_bytes"], closure["inventory_stored_bytes"]
        )

    def test_unrecognized_dense_router_dtype_bytes_not_guessed(self) -> None:
        """A router dtype outside BF16/F32 keeps null bytes: no dtype guessing."""
        inv = make_inventory([1, 2], experts=8)
        for t in inv["tensors"]:
            if t["name"].endswith("mlp.gate.weight"):
                t["dtype"] = "FP8_e4m3"
        path = self.write_json("inv_odd_router.json", inv)
        code, _, err = self.run_cli(
            ["--inventory", path, "--retained-count", "4", "--seed", "7",
             "--output", self.map_path]
        )
        self.assertEqual(code, 0, err)
        m = self.load_map()
        for entry in m["layers"]:
            accounting = entry["router"]["accounting"]
            self.assertIsNone(accounting["post_remap_stored_bytes"])
            self.assertFalse(accounting["post_remap_stored_bytes_known"])
        totals = m["totals"]
        self.assertIsNone(totals["router_stored_bytes_post_remap"])
        self.assertFalse(totals["router_stored_bytes_post_remap_known"])
        text = totals["post_prune_by_component"]["text/main"]
        self.assertIsNone(text["stored_bytes"])
        self.assertFalse(text["stored_bytes_known"])
        # Logical parameters remain exact — they never depend on a byte guess.
        self.assertEqual(
            text["logical_parameters"],
            2 * 4 * 3 * INTER * HIDDEN
            + 3 * INTER * HIDDEN
            + 2 * (4 * HIDDEN + 4),
        )


class ExternalSelectionTest(PruneMapsTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.inv_path = self.write_json("inv.json", make_inventory([1, 2], experts=8))
        self.map_path = str(self.dir / "map.json")

    def test_external_selection_preserves_original_ids_and_order(self) -> None:
        order = [5, 0, 7, 2]
        cand = {
            "schema_version": 1,
            "layers": [
                {"layer": 1, "retained_expert_ids": order},
                {"layer": 2, "retained_expert_ids": list(reversed(order))},
            ],
        }
        cand_path = self.write_json("sel.json", cand)
        code, _, err = self.run_cli(
            ["--inventory", self.inv_path, "--selection", cand_path, "--output", self.map_path]
        )
        self.assertEqual(code, 0, err)
        m = json.loads(Path(self.map_path).read_text(encoding="utf-8"))
        self.assertEqual(m["mode"], "external_selection")
        self.assertEqual(m["layers"][0]["retained"], order)
        self.assertEqual(m["layers"][0]["old_to_new"]["5"], 0)
        self.assertEqual(m["layers"][0]["old_to_new"]["0"], 1)
        self.assertEqual(m["layers"][1]["retained"], list(reversed(order)))
        import hashlib
        expected = hashlib.sha256(Path(cand_path).read_bytes()).hexdigest()
        self.assertEqual(m["provenance"]["sha256"], expected)

    def test_external_candidate_schema_keys(self) -> None:
        """Baseline-contract field names (retained_expert_ids) are accepted."""
        cand = {
            "schema_version": 1,
            "layers": [{"layer": 1, "original_expert_count": 8,
                        "retained_expert_ids": [3, 1, 4, 6],
                        "pruned_expert_ids": [0, 2, 5, 7]},
                       {"layer": 2, "original_expert_count": 8,
                        "retained_expert_ids": [3, 1, 4, 6],
                        "pruned_expert_ids": [0, 2, 5, 7]}],
        }
        code, _, err = self.run_cli(
            ["--inventory", self.inv_path, "--selection",
             self.write_json("sel.json", cand), "--output", self.map_path]
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(Path(self.map_path).read_text())["retained_count"], 4)


class MalformedSelectionTest(PruneMapsTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.inv_path = self.write_json("inv.json", make_inventory([1, 2], experts=8))
        self.map_path = str(self.dir / "map.json")

    def expect_failure(self, argv: list[str], needle: str) -> None:
        code, _, err = self.run_cli(argv)
        self.assertEqual(code, 1, err)
        self.assertIn(needle, err)

    def sel(self, payload: object) -> str:
        return self.write_json("bad_sel.json", payload)

    def test_duplicate_ids(self) -> None:
        self.expect_failure(
            ["--inventory", self.inv_path, "--selection",
             self.sel({"schema_version": 1, "layers": [
                 {"layer": 1, "retained_expert_ids": [1, 1, 2, 3]},
                 {"layer": 2, "retained_expert_ids": [0, 1, 2, 3]}]}),
             "--output", self.map_path],
            "duplicate",
        )

    def test_out_of_range_id(self) -> None:
        self.expect_failure(
            ["--inventory", self.inv_path, "--selection",
             self.sel({"schema_version": 1, "layers": [
                 {"layer": 1, "retained_expert_ids": [0, 1, 2, 8]},
                 {"layer": 2, "retained_expert_ids": [0, 1, 2, 3]}]}),
             "--output", self.map_path],
            "out of range",
        )

    def test_missing_layer(self) -> None:
        self.expect_failure(
            ["--inventory", self.inv_path, "--selection",
             self.sel({"schema_version": 1, "layers": [
                 {"layer": 1, "retained_expert_ids": [0, 1, 2, 3]}]}),
             "--output", self.map_path],
            "missing layers",
        )

    def test_non_moe_layer_rejected(self) -> None:
        self.expect_failure(
            ["--inventory", self.inv_path, "--selection",
             self.sel({"schema_version": 1, "layers": [
                 {"layer": 0, "retained_expert_ids": [0, 1, 2, 3]},
                 {"layer": 1, "retained_expert_ids": [0, 1, 2, 3]},
                 {"layer": 2, "retained_expert_ids": [0, 1, 2, 3]}]}),
             "--output", self.map_path],
            "not a MoE layer",
        )

    def test_unequal_count_across_layers(self) -> None:
        self.expect_failure(
            ["--inventory", self.inv_path, "--selection",
             self.sel({"schema_version": 1, "layers": [
                 {"layer": 1, "retained_expert_ids": [0, 1, 2, 3]},
                 {"layer": 2, "retained_expert_ids": [0, 1, 2, 3, 4]}]}),
             "--output", self.map_path],
            "unequal count",
        )

    def test_top_k_change_rejected(self) -> None:
        self.expect_failure(
            ["--inventory", self.inv_path, "--selection",
             self.sel({"schema_version": 1, "top_k": 4, "layers": [
                 {"layer": 1, "retained_expert_ids": [0, 1, 2, 3]},
                 {"layer": 2, "retained_expert_ids": [0, 1, 2, 3]}]}),
             "--output", self.map_path],
            "top_k change",
        )

    def test_bad_expert_type(self) -> None:
        self.expect_failure(
            ["--inventory", self.inv_path, "--selection",
             self.sel({"schema_version": 1, "layers": [
                 {"layer": 1, "retained_expert_ids": [0, 1, 2, "3"]},
                 {"layer": 2, "retained_expert_ids": [0, 1, 2, 3]}]}),
             "--output", self.map_path],
            "bad expert",
        )

    def test_duplicate_layer_entries(self) -> None:
        self.expect_failure(
            ["--inventory", self.inv_path, "--selection",
             self.sel({"schema_version": 1, "layers": [
                 {"layer": 1, "retained_expert_ids": [0, 1, 2, 3]},
                 {"layer": 1, "retained_expert_ids": [4, 5, 6, 7]},
                 {"layer": 2, "retained_expert_ids": [0, 1, 2, 3]}]}),
             "--output", self.map_path],
            "duplicate entries",
        )

    def test_generated_count_out_of_range(self) -> None:
        self.expect_failure(
            ["--inventory", self.inv_path, "--retained-count", "0", "--seed", "1",
             "--output", self.map_path],
            "out of range",
        )
        self.expect_failure(
            ["--inventory", self.inv_path, "--retained-count", "9", "--seed", "1",
             "--output", self.map_path],
            "out of range",
        )

    def test_retained_count_with_selection_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.run_cli(
                ["--inventory", self.inv_path, "--retained-count", "4",
                 "--selection", self.sel({"schema_version": 1, "layers": [
                     {"layer": 1, "retained_expert_ids": [0, 1, 2, 3]},
                     {"layer": 2, "retained_expert_ids": [0, 1, 2, 3]}]}),
                 "--output", self.map_path]
            )
        self.assertEqual(ctx.exception.code, 2)


class MalformedInventoryTest(PruneMapsTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.map_path = str(self.dir / "map.json")

    def expect_failure(self, payload: object, needle: str) -> None:
        path = self.write_json("bad_inv.json", payload)
        code, _, err = self.run_cli(
            ["--inventory", path, "--retained-count", "2", "--seed", "1",
             "--output", self.map_path]
        )
        self.assertEqual(code, 1, err)
        self.assertIn(needle, err)

    def base(self, **kwargs: object) -> dict:
        return make_inventory([1, 2], experts=8, **kwargs)

    def test_schema_version_mismatch(self) -> None:
        inv = self.base()
        inv["schema_version"] = 2
        self.expect_failure(inv, "unsupported inventory schema_version")

    def test_missing_projection(self) -> None:
        inv = self.base()
        inv["tensors"] = [
            t for t in inv["tensors"]
            if not t["name"].startswith("model.layers.1.mlp.experts.3.gate_proj.weight_scale")
        ]
        self.expect_failure(inv, "missing gate_proj.weight_scale")

    def test_missing_expert(self) -> None:
        inv = self.base()
        inv["tensors"] = [
            t for t in inv["tensors"] if ".experts.5." not in t["name"]
        ]
        self.expect_failure(inv, "missing experts")

    def test_missing_router_bias(self) -> None:
        inv = self.base(router_bias_present=False)
        self.expect_failure(inv, "missing router tensor *.mlp.gate.e_score_correction_bias")

    def test_router_dimension_mismatch(self) -> None:
        inv = self.base(router_shape=[7, 32])
        self.expect_failure(inv, "leading dimension 7 != original_experts_per_layer 8")

    def test_dense_router_bytes_not_row_uniform_fails(self) -> None:
        """A dense BF16 router whose bytes violate row-uniformity fails closed."""
        inv = self.base()
        for t in inv["tensors"]:
            if t["name"] == "model.layers.1.mlp.gate.weight":
                t["stored_bytes"] = 2 * HIDDEN * 8 - 1  # 511: not divisible by 8 rows
        self.expect_failure(inv, "row-uniform")

    def test_dtype_change_across_layers(self) -> None:
        inv = self.base(expert_dtype_override={2: "BF16"})
        self.expect_failure(inv, "dtype change")

    def test_unknown_logical_packing_fails(self) -> None:
        inv = self.base()
        for t in inv["tensors"]:
            if t["name"].endswith(".experts.0.gate_proj.weight"):
                del t["logical_parameters"]
        self.expect_failure(inv, "logical_parameters")

    def test_bad_expert_id_in_inventory(self) -> None:
        inv = self.base()
        for t in inv["tensors"]:
            if t["name"].startswith("model.layers.1.mlp.experts.9."):
                pass
        inv["tensors"].append(
            {
                "name": "model.layers.1.mlp.experts.8.gate_proj.weight",
                "shape": [INTER, HIDDEN // 2],
                "dtype": MXFP4_DTYPE,
                "stored_bytes": 512,
                "logical_shape": [INTER, HIDDEN],
                "logical_parameters": INTER * HIDDEN,
                "role": "parameter",
                "layer": 1,
                "component": "text/main",
            }
        )
        self.expect_failure(inv, "bad expert id 8")

    def test_expert_tensor_in_dense_layer(self) -> None:
        inv = self.base()
        inv["architecture"]["moe_layers"] = [1]
        self.expect_failure(inv, "non-MoE layer 2")


class GenericCountsTest(PruneMapsTestBase):
    """Counts 144/160/176/192/256 on a 256-expert synthetic architecture."""

    def setUp(self) -> None:
        super().setUp()
        self.arch = {
            "original_experts_per_layer": 256,
            "top_k": 8,
            "moe_layers": list(range(1, 4)),
        }

    def test_generic_counts_generate_valid_selections(self) -> None:
        for count in (144, 160, 176, 192, 256):
            for layer, ids in generate_selection(self.arch, count, seed=42).items():
                self.assertEqual(len(ids), count, (count, layer))
                self.assertEqual(len(set(ids)), count)
                self.assertTrue(all(0 <= i < 256 for i in ids))
                self.assertEqual(set(ids) <= set(range(256)), True)
        full = generate_selection(self.arch, 256, seed=42)
        for ids in full.values():
            self.assertEqual(sorted(ids), list(range(256)))

    def test_identity_256_dry_run_has_no_pruned(self) -> None:
        inv = make_inventory([1, 2, 3], experts=256)
        path = self.write_json("inv256.json", inv)
        map_path = str(self.dir / "map256.json")
        code, _, err = self.run_cli(
            ["--inventory", path, "--retained-count", "256", "--seed", "0",
             "--output", map_path]
        )
        self.assertEqual(code, 0, err)
        m = json.loads(Path(map_path).read_text(encoding="utf-8"))
        self.assertTrue(all(e["pruned"] == [] for e in m["layers"]))
        self.assertEqual(m["totals"]["stored_bytes_dropped"], 0)
        self.assertEqual(m["totals"]["expert_tensors_dropped"], 0)


class Full47LayerDryRunTest(PruneMapsTestBase):
    """47 MoE layers x 256 experts, official layout, at 160 and 176."""

    EXPERT_WEIGHT_PARAMS = 3 * INTER * HIDDEN
    EXPERT_WEIGHT_BYTES = 3 * (INTER * (HIDDEN // 2))
    EXPERT_SCALE_BYTES = 3 * (INTER * (HIDDEN // BLOCK)) * 4

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        cls.inv_path = str(cls.dir / "inv47.json")
        Path(cls.inv_path).write_text(
            json.dumps(make_inventory(list(range(1, 48)), experts=256)), encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def run_count(self, count: int) -> dict:
        map_path = str(self.dir / f"map47_{count}.json")
        code, _, err = self.run_cli(
            ["--inventory", self.inv_path, "--retained-count", str(count),
             "--seed", "11", "--output", map_path]
        )
        self.assertEqual(code, 0, err)
        return json.loads(Path(map_path).read_text(encoding="utf-8"))

    def test_47_layer_160(self) -> None:
        m = self.run_count(160)
        self.assertEqual(len(m["layers"]), 47)
        self.assertEqual([e["layer"] for e in m["layers"]], list(range(1, 48)))
        for entry in m["layers"]:
            self.assertEqual(len(entry["retained"]), 160)
            self.assertEqual(len(entry["pruned"]), 96)
            self.assertEqual(len(entry["old_to_new"]), 256)
            self.assertEqual(
                sorted(v for v in entry["old_to_new"].values() if v is not None),
                list(range(160)),
            )
            self.assertEqual(entry["tensor_counts"]["keep"], 160 * 6)
            self.assertEqual(entry["tensor_counts"]["drop"], 96 * 6)
            self.assertEqual(
                entry["logical_parameters"]["keep"], 160 * self.EXPERT_WEIGHT_PARAMS
            )
        totals = m["totals"]
        self.assertEqual(totals["expert_tensors_kept"], 47 * 160 * 6)
        self.assertEqual(totals["expert_tensors_dropped"], 47 * 96 * 6)
        self.assertEqual(
            totals["logical_parameters_dropped"], 47 * 96 * self.EXPERT_WEIGHT_PARAMS
        )
        self.assertEqual(
            totals["stored_bytes_dropped"], 47 * 96 * (self.EXPERT_WEIGHT_BYTES + self.EXPERT_SCALE_BYTES)
        )
        self.assertEqual(
            totals["logical_parameters_router_remapped"],
            47 * (160 * HIDDEN + 160),
        )
        self.assertEqual(totals["router_tensors"], 47 * 2)
        self.assertEqual(
            totals["router_stored_bytes"], 47 * (256 * HIDDEN * 2 + 256 * 4)
        )
        self.assertEqual(
            totals["router_logical_parameters"], 47 * (256 * HIDDEN + 256)
        )
        closure = totals["accounting_closure"]
        self.assertTrue(closure["closes"])
        self.assertEqual(closure["accounted_tensors"], closure["inventory_tensors"])
        self.assertEqual(closure["accounted_stored_bytes"], closure["inventory_stored_bytes"])
        self.assertEqual(
            closure["accounted_logical_parameters"], closure["inventory_logical_parameters"]
        )

        # Exact dense router row-scaling at 160: 47 x (160*HIDDEN*2 + 160*4).
        router_post = 47 * (160 * HIDDEN * 2 + 160 * 4)
        self.assertEqual(totals["router_stored_bytes_post_remap"], router_post)
        self.assertTrue(totals["router_stored_bytes_post_remap_known"])

        # By-component: text post-prune = kept experts + sliced routers +
        # untouched text; equivalently inventory text - dropped - router delta.
        by = totals["post_prune_by_component"]
        self.assertEqual(list(by), ["text/main", "vision"])
        text = by["text/main"]
        self.assertTrue(text["stored_bytes_known"])
        router_delta_bytes = 47 * (256 - 160) * (HIDDEN * 2 + 4)
        router_delta_params = 47 * (256 - 160) * (HIDDEN + 1)
        inv_text_bytes = closure["inventory_stored_bytes"] - 128  # only vision is non-text
        inv_text_params = closure["inventory_logical_parameters"] - 64
        self.assertEqual(
            text["stored_bytes"],
            totals["stored_bytes_kept"]
            + (totals["unchanged_stored_bytes"] - 128)
            + router_post,
        )
        self.assertEqual(
            text["stored_bytes"],
            inv_text_bytes - totals["stored_bytes_dropped"] - router_delta_bytes,
        )
        self.assertEqual(
            text["logical_parameters"],
            inv_text_params
            - totals["logical_parameters_dropped"]
            - router_delta_params,
        )
        self.assertEqual(text["tensor_count"], 47 * 160 * 6 + 94 + 6)
        self.assertEqual(by["vision"]["stored_bytes"], 128)
        self.assertEqual(by["vision"]["logical_parameters"], 64)

    def test_47_layer_176(self) -> None:
        m = self.run_count(176)
        self.assertEqual(m["retained_count"], 176)
        totals = m["totals"]
        self.assertEqual(totals["expert_tensors_kept"], 47 * 176 * 6)
        self.assertEqual(totals["expert_tensors_dropped"], 47 * 80 * 6)
        router_post = 47 * (176 * HIDDEN * 2 + 176 * 4)
        self.assertEqual(totals["router_stored_bytes_post_remap"], router_post)
        self.assertTrue(totals["router_stored_bytes_post_remap_known"])

        closure = totals["accounting_closure"]
        self.assertTrue(closure["closes"])
        self.assertEqual(closure["accounted_tensors"], closure["inventory_tensors"])
        self.assertEqual(closure["accounted_stored_bytes"], closure["inventory_stored_bytes"])
        self.assertEqual(
            closure["accounted_logical_parameters"], closure["inventory_logical_parameters"]
        )

        by = totals["post_prune_by_component"]
        self.assertEqual(list(by), ["text/main", "vision"])
        text = by["text/main"]
        router_delta_bytes = 47 * (256 - 176) * (HIDDEN * 2 + 4)
        router_delta_params = 47 * (256 - 176) * (HIDDEN + 1)
        self.assertEqual(
            text["stored_bytes"],
            totals["stored_bytes_kept"]
            + (totals["unchanged_stored_bytes"] - 128)
            + router_post,
        )
        self.assertEqual(
            text["stored_bytes"],
            closure["inventory_stored_bytes"]
            - 128
            - totals["stored_bytes_dropped"]
            - router_delta_bytes,
        )
        self.assertEqual(
            text["logical_parameters"],
            closure["inventory_logical_parameters"]
            - 64
            - totals["logical_parameters_dropped"]
            - router_delta_params,
        )
        self.assertEqual(text["tensor_count"], 47 * 176 * 6 + 94 + 6)

    def test_maps_differ_between_counts(self) -> None:
        self.assertNotEqual(self.run_count(160)["layers"][0]["retained"],
                            self.run_count(176)["layers"][0]["retained"])


class ParseSelectionUnitTest(PruneMapsTestBase):
    def test_parse_rejects_bad_schema(self) -> None:
        arch = {"original_experts_per_layer": 8, "top_k": 8, "moe_layers": [1]}
        with self.assertRaises(PruneMapError):
            parse_external_selection({"schema_version": 2, "layers": []}, arch)
        with self.assertRaises(PruneMapError):
            parse_external_selection({"schema_version": 1, "layers": []}, arch)


if __name__ == "__main__":
    unittest.main()
