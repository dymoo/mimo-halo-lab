"""Tests for the MiMo-V2.6-Flash one-pass REAP/HOPE observer wiring.

Covers, per assignment:

- a synthetic MiMo-shaped fixture (class names and formulas twin-ed from the
  pinned ``modeling_mimo_v2.py`` at revision 5711b268169967567844e1e560e8a3966da959b1)
  passing ``attach_model`` -> one-pass ``observe`` -> ``select`` -> merge
  roundtrip through the REAL pinned REAP observer and accumulator;
- fail-closed guards: wrong expert count, missing MXFP4 scale sibling, top-k
  mismatch, dtype divergence, wrong router/packing shapes, missing correction
  bias, active group routing, non-unit routed scaling, non-sigmoid scoring,
  divergent config, and unregistered architectures;
- REAP and HOPE outputs produced from ONE observation (same ids, gates,
  norms; exact frequency agreement; REAP ``reap`` metric == HopeStats
  ``first_order()`` within float tolerance; forward-call accounting);
- determinism of both metric families across repeated and rebuilt runs.

Fixtures are synthetic and tiny in hidden dims; the verified counts (256
experts, top-8) are enforced because the adapter hard-checks them against the
inventory metadata.  No model weights, no network access.
"""

import json
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from mimo_halo.pruning import hope  # noqa: E402
from mimo_halo.pruning import reap_adapter  # noqa: E402

HIDDEN = 64
MOE_INTER = 32
N_MOE_LAYERS = 2  # fixture layers 1..2 are MoE; layer 0 is dense (skipped)
TOKENS_PER_BATCH = 32  # batch=2 x seq=16


# ---------------------------------------------------------------------------
# synthetic MiMo-shaped fixture (pinned class names, pinned formulas)
# ---------------------------------------------------------------------------


class _FixtureConfig:
    def __init__(self, **overrides):
        values = dict(
            hidden_size=HIDDEN,
            intermediate_size=HIDDEN,
            moe_intermediate_size=MOE_INTER,
            n_routed_experts=reap_adapter.MIMO_EXPERTS_PER_LAYER,
            num_experts_per_tok=reap_adapter.MIMO_TOP_K,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=None,  # pinned config: null -> gate uses 1.0
            hidden_act="silu",
        )
        values.update(overrides)
        self.__dict__.update(values)


class _PackedProj(nn.Module):
    """Projection with verified native MXFP4 packing evidence.

    ``weight`` (U8 nibble-packed, ``(out, in // 2)``) plus ``weight_scale``
    (U8, ``(out, in // 32)``) — the packed-weight/scale sibling pair recorded
    for every expert tensor in inventory.json (``mxfp4_packed`` /
    ``mxfp4_scale``, block size 32).  ``forward`` simulates the dequantizing
    runtime the real observation will use: experts must be runnable, the
    stored tensors must stay packed.
    """

    def __init__(self, out_features: int, in_features: int, offset: int):
        super().__init__()
        assert in_features % 32 == 0
        packed = (torch.arange(out_features * (in_features // 2)) + offset) % 251
        scales = (torch.arange(out_features * (in_features // 32)) + offset) % 250
        self.weight = nn.Parameter(
            packed.to(torch.uint8).reshape(out_features, in_features // 2),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            scales.to(torch.uint8).reshape(out_features, in_features // 32),
            requires_grad=False,
        )

    def forward(self, x):
        # Fixture unpack: nibble-expand (out, in // 2) -> logical (out, in),
        # centre the unsigned payload and scale to a stable range (the real
        # runtime dequantizes MXFP4; raw 0..255 weights would overflow float32
        # across stacked layers, zero-centred-at-full-scale collapses them).
        logical = self.weight.repeat_interleave(2, dim=1).to(torch.float32)
        return F.linear(x, (logical * (1.0 / 250.0) - 0.5) * 0.4)


class MiMoV2MLP(nn.Module):
    """Fixture twin of pinned modeling_mimo_v2.py ``MiMoV2MLP``."""

    def __init__(self, config, intermediate_size=None, offset=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = _PackedProj(
            self.intermediate_size, self.hidden_size, offset + 1
        )
        self.up_proj = _PackedProj(self.intermediate_size, self.hidden_size, offset + 2)
        self.down_proj = _PackedProj(
            self.hidden_size, self.intermediate_size, offset + 3
        )

    def forward(self, hidden_states):
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class MiMoV2MoEGate(nn.Module):
    """Fixture twin of pinned modeling_mimo_v2.py ``MiMoV2MoEGate``.

    The forward formula is reproduced from the pinned source: sigmoid scores,
    noaux_tc selection ``topk(sigmoid(logits) + bias)`` under the checkpoint's
    degenerate ``n_group = topk_group = 1`` group mask, unbiased sigmoid gather
    renormalized over the selected set, scaled by ``routed_scaling_factor``.
    """

    def __init__(self, config, seed: int = 0):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = (
            config.routed_scaling_factor
            if config.routed_scaling_factor is not None
            else 1.0
        )
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(
            torch.randn(self.n_routed_experts, self.gating_dim, generator=generator).to(
                torch.bfloat16
            ),
            requires_grad=False,
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.randn(self.n_routed_experts, generator=generator),
                requires_grad=False,
            )

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f"Unsupported scoring function: {self.scoring_func}")
        if self.topk_method != "noaux_tc":
            raise NotImplementedError(f"Unsupported topk method: {self.topk_method}")
        scores_for_choice = scores.view(bsz * seq_len, -1) + (
            self.e_score_correction_bias.unsqueeze(0)
        )
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                bsz * seq_len,
                self.n_group,
                self.n_routed_experts // self.n_group,
            )
            .reshape(bsz * seq_len, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class MiMoV2MoE(nn.Module):
    """Fixture twin of pinned modeling_mimo_v2.py ``MiMoV2MoE``.

    ``forward`` returns a BARE tensor, exactly like the pinned source — the
    output contract the adapter's logits-exposure seam must bridge.
    """

    def __init__(self, config, seed: int = 0):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                MiMoV2MLP(
                    config,
                    intermediate_size=config.moe_intermediate_size,
                    offset=4 * index + seed,
                )
                for index in range(config.n_routed_experts)
            ]
        )
        self.gate = MiMoV2MoEGate(config, seed=seed)

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(
            topk_indices, num_classes=len(self.experts)
        ).permute(2, 0, 1)
        for expert_idx, expert in enumerate(self.experts):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_output = expert(hidden_states[token_indices])
                final.index_add_(
                    0, token_indices, expert_output * expert_weights.unsqueeze(-1)
                )
        return final.type(hidden_states.dtype).view(*orig_shape)  # bare tensor


class _DecoderLayer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


class _InnerModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class MiMoV2ForCausalLM(nn.Module):
    """Named exactly as source-metadata/config.json ``architectures[0]`` so the
    adapter's MODEL_ATTRS / observer-config rows resolve."""

    def __init__(self, config, layers):
        super().__init__()
        self.config = config
        self.model = _InnerModel(layers)

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def build_fixture(*, moe_layers=N_MOE_LAYERS, experts=256, top_k=8, seed=0):
    config = _FixtureConfig(
        n_routed_experts=experts, num_experts_per_tok=top_k
    )
    layers = [_DecoderLayer(MiMoV2MLP(config, offset=10_000))]  # dense layer 0
    for index in range(moe_layers):
        layers.append(_DecoderLayer(MiMoV2MoE(config, seed=seed + 17 * index)))
    model = MiMoV2ForCausalLM(config, layers)
    model.eval()
    return model


def _batches(count: int = 2, seed: int = 1234):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(2, TOKENS_PER_BATCH // 2, HIDDEN, generator=generator)
        for _ in range(count)
    ]


def _observe(model, inputs, labels=None):
    """attach -> install -> make_observer -> forward -> export (one pass)."""
    listener = reap_adapter.OnePassListener()
    notes = listener.attach_model(model)
    listener.install()
    observer = reap_adapter.make_observer(model)
    listener.bind_observer(observer)
    try:
        with torch.no_grad():
            for index, hidden in enumerate(inputs):
                label = labels[index] if labels else None
                with listener.capability(label):
                    model(hidden)
        stats = listener.export()
        reap_state = reap_adapter.extract_reap_state(observer)
        forward_passes = listener.forward_passes
    finally:
        listener.uninstall()
        observer.close_hooks()
    return notes, stats, reap_state, forward_passes


class RegistrationTests(unittest.TestCase):
    def test_registration_lands_in_both_pinned_registries(self):
        reap_adapter.ensure_mimo_registration()
        reap_adapter.ensure_mimo_registration()  # idempotent
        from mimo_halo.pruning import reap_adapter as ra

        ra.ensure_reap()
        import importlib

        model_util = importlib.import_module("reap.model_util")
        observer_mod = importlib.import_module("reap.observer")
        attrs = model_util.MODEL_ATTRS[reap_adapter.MIMO_MODEL_CLASS]
        self.assertEqual(attrs, ra._mimo_expected_attrs())
        self.assertFalse(attrs["fused"])
        config_cls = observer_mod.OBSERVER_CONFIG_REGISTRY[
            reap_adapter.MIMO_MODEL_CLASS
        ]
        probe = config_cls()
        expected = ra._mimo_expected_hook_config()
        self.assertEqual(
            {key: getattr(probe, key, None) for key in expected}, expected
        )
        # make_observer flips these two on regardless of row defaults:
        self.assertEqual(
            {key: getattr(probe, key) for key in ("renormalize_router_weights",
                                                  "record_pruning_metrics_only")},
            {"renormalize_router_weights": False, "record_pruning_metrics_only": False},
        )

    def test_unregistered_architecture_still_refuses(self):
        class NeverRegisteredModel:
            pass

        listener = reap_adapter.OnePassListener()
        with self.assertRaises(hope.HopeDependencyError) as ctx:
            listener.attach_model(NeverRegisteredModel())
        message = str(ctx.exception)
        self.assertIn("MODEL_ATTRS", message)
        self.assertIn(reap_adapter.MIMO_MODEL_CLASS, message)


class AttachGuardTests(unittest.TestCase):
    """Every fail-closed guard refuses with an explicit error."""

    @staticmethod
    def _attach(listener, model):
        return listener.attach_model(model)

    def _assert_refuses(self, model, needle):
        with self.assertRaises(hope.HopeValidationError) as ctx:
            self._attach(reap_adapter.OnePassListener(), model)
        self.assertIn(needle, str(ctx.exception))

    @staticmethod
    def _first_moe(model):
        return model.model.layers[1].mlp

    def test_correct_fixture_attaches_with_verified_notes(self):
        model = build_fixture()
        notes = reap_adapter.OnePassListener().attach_model(model)
        # Dense layer 0 is skipped; MoE layers 1..N carry verified specs.
        self.assertEqual(sorted(notes), list(range(1, N_MOE_LAYERS + 1)))
        for note in notes.values():
            self.assertIn("verified sigmoid + noaux_tc", note)
            self.assertIn("single shared observation", note)

    def test_wrong_expert_count_refuses(self):
        self._assert_refuses(build_fixture(experts=255), "wrong expert count")

    def test_truncated_expert_list_refuses_despite_config(self):
        model = build_fixture()
        experts = model.model.layers[1].mlp.experts
        model.model.layers[1].mlp.experts = nn.ModuleList(list(experts)[:-1])
        self._assert_refuses(model, "expert count mismatch")

    def test_top_k_mismatch_refuses(self):
        self._assert_refuses(build_fixture(top_k=4), "top-k mismatch")

    def test_missing_scale_sibling_refuses(self):
        model = build_fixture()
        del self._first_moe(model).experts[5].gate_proj.weight_scale
        self._assert_refuses(model, "scale sibling")

    def test_expert_weight_dtype_divergence_refuses(self):
        model = build_fixture()
        proj = self._first_moe(model).experts[3].up_proj
        proj.weight = nn.Parameter(torch.zeros(33, 33))  # float32, unpacked
        self._assert_refuses(model, "diverges")

    def test_router_weight_dtype_divergence_refuses(self):
        model = build_fixture()
        gate = self._first_moe(model).gate
        gate.weight = nn.Parameter(torch.zeros(256, HIDDEN))  # float32, not BF16
        self._assert_refuses(model, "BF16")

    def test_router_bias_dtype_divergence_refuses(self):
        model = build_fixture()
        gate = self._first_moe(model).gate
        gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(256, dtype=torch.bfloat16)
        )
        self._assert_refuses(model, "F32")

    def test_wrong_router_shape_refuses(self):
        model = build_fixture()
        gate = self._first_moe(model).gate
        gate.weight = nn.Parameter(torch.zeros(255, HIDDEN, dtype=torch.bfloat16))
        self._assert_refuses(model, "router weight shape")

    def test_wrong_packed_shape_refuses(self):
        model = build_fixture()
        proj = self._first_moe(model).experts[7].down_proj
        proj.weight = nn.Parameter(
            torch.zeros(33, HIDDEN // 2, dtype=torch.uint8), requires_grad=False
        )
        self._assert_refuses(model, "!= packed")

    def test_missing_correction_bias_refuses(self):
        model = build_fixture()
        del self._first_moe(model).gate.e_score_correction_bias
        self._assert_refuses(model, "e_score_correction_bias")

    def test_missing_router_module_refuses(self):
        model = build_fixture()
        del self._first_moe(model).gate
        self._assert_refuses(model, "no router module")

    def test_active_group_routing_refuses(self):
        model = build_fixture()
        self._first_moe(model).gate.n_group = 2
        self._assert_refuses(model, "group routing")

    def test_non_unit_routed_scaling_refuses(self):
        model = build_fixture()
        self._first_moe(model).gate.routed_scaling_factor = 2.5
        self._assert_refuses(model, "routed_scaling_factor")

    def test_non_sigmoid_scoring_refuses(self):
        model = build_fixture()
        self._first_moe(model).gate.scoring_func = "softmax"
        self._assert_refuses(model, "sigmoid")

    def test_non_renormalized_gates_refuse(self):
        model = build_fixture()
        self._first_moe(model).gate.norm_topk_prob = False
        self._assert_refuses(model, "renormalized")

    def test_config_expert_count_divergence_refuses(self):
        model = build_fixture()
        self._first_moe(model).config.n_routed_experts = 200
        self._assert_refuses(model, "config.n_routed_experts")


class OnePassObservationTests(unittest.TestCase):
    """One observation feeds BOTH metric families; attach/observe/select/merge."""

    @classmethod
    def setUpClass(cls):
        cls.model = build_fixture()
        cls.inputs = _batches()
        cls.labels = ["code", "reasoning"]
        (
            cls.notes,
            cls.stats,
            cls.reap_state,
            cls.forward_passes,
        ) = _observe(cls.model, cls.inputs, cls.labels)

    def test_observer_state_covers_every_moe_layer_only(self):
        self.assertEqual(sorted(self.stats), list(range(1, N_MOE_LAYERS + 1)))
        self.assertEqual(sorted(self.reap_state), list(range(1, N_MOE_LAYERS + 1)))
        # Dense layer 0 was never attached, never hooked.
        self.assertNotIn(0, self.stats)
        self.assertNotIn(0, self.reap_state)

    def test_single_accumulation_per_layer_per_forward(self):
        # One wrapped update_pruning_state call per MoE layer per forward:
        # any double instrumentation (REAP and HOPE counted separately) would
        # exceed this — the two families come from the same single call.
        self.assertEqual(
            self.forward_passes, len(self.inputs) * N_MOE_LAYERS
        )

    def test_rows_counts_and_capability_partitions(self):
        expect = len(self.inputs) * TOKENS_PER_BATCH
        for layer, stats in self.stats.items():
            stats.validate()
            self.assertEqual(stats.n_experts, reap_adapter.MIMO_EXPERTS_PER_LAYER)
            self.assertEqual(stats.top_k, reap_adapter.MIMO_TOP_K)
            self.assertEqual(stats.total_rows, expect)
            self.assertEqual(
                sorted(stats.capabilities), sorted(set(self.labels))
            )
            self.assertEqual(
                sum(cap.total_rows for cap in stats.capabilities.values()),
                expect,
            )
            for cap in stats.capabilities.values():
                cap.validate()

    def test_reap_and_hope_agree_from_the_same_observation(self):
        for layer, stats in self.stats.items():
            values = self.reap_state[layer]
            # Exact: routed-id counts on both families are integers counted
            # from the injected ids of this one observation.
            self.assertTrue(
                torch.equal(
                    values["expert_frequency"].to(torch.int64),
                    torch.tensor(stats.act_count, dtype=torch.int64),
                )
            )
            self.assertTrue(
                torch.equal(
                    values["pairwise_expert_frequency"].to(torch.int64),
                    torch.tensor(
                        [
                            [
                                stats.act_count[i] + stats.act_count[j]
                                for j in range(stats.n_experts)
                            ]
                            for i in range(stats.n_experts)
                        ],
                        dtype=torch.int64,
                    ),
                )
            )
            # REAP's `reap` metric == HopeStats.first_order(): both are the
            # mean over active tokens of g * ||activation||, computed by the
            # pinned accumulator (float32) vs the HOPE decodes (float64).
            hope_first = torch.tensor(stats.first_order(), dtype=torch.float32)
            self.assertTrue(
                torch.allclose(
                    values["reap"].to(torch.float32),
                    hope_first,
                    rtol=1e-3,
                    atol=1e-5,
                ),
                f"layer {layer}: REAP {values['reap']} != HOPE {hope_first}",
            )
            # Routing frequency agreement (same ids, same denominators).
            hope_freq = torch.tensor(stats.routing_frequency(), dtype=torch.float64)
            reap_freq = (
                values["expert_frequency"].to(torch.float64) / float(stats.total_rows)
            )
            self.assertTrue(torch.allclose(hope_freq, reap_freq, atol=1e-12))

    def test_select_and_merge_roundtrip(self):
        for layer, stats in self.stats.items():
            f_matrix = stats.conditional_f()
            budget = stats.n_experts // 4
            pruned = hope.diag_topk_pruned(f_matrix, budget)
            self.assertEqual(len(pruned), budget)
            self.assertEqual(pruned, sorted(set(pruned)))
            self.assertTrue(set(pruned) <= set(range(stats.n_experts)))
            # Selection is a pure function of the observed stats: a JSON
            # roundtrip must reproduce it exactly.
            restored = hope.HopeStats.from_json(stats.to_json())
            self.assertEqual(
                hope.diag_topk_pruned(restored.conditional_f(), budget), pruned
            )

        # Split-then-merge vs direct observation of the same rows.
        _, stats_a, _, _ = _observe(build_fixture(), self.inputs[:1], self.labels[:1])
        _, stats_b, _, _ = _observe(build_fixture(), self.inputs[1:], self.labels[1:])
        _, stats_direct, _, _ = _observe(
            build_fixture(), self.inputs, self.labels
        )
        for layer in stats_direct:
            merged = stats_a[layer].merge(stats_b[layer])
            self.assertEqual(merged.total_rows, stats_direct[layer].total_rows)
            self.assertEqual(
                sorted(merged.capabilities), sorted(stats_direct[layer].capabilities)
            )
            self.assertEqual(
                sum(c.total_rows for c in merged.capabilities.values()),
                merged.total_rows,
            )
            merged.validate()
            # Byte-stable JSON roundtrip on the merged document...
            payload = merged.to_json()
            self.assertEqual(hope.HopeStats.from_json(payload).to_json(), payload)
            # ...and float64 accumulation matches the direct observation
            # (addition order changes only rounding noise).
            for got, want in zip(merged.first_order(), stats_direct[layer].first_order()):
                self.assertTrue(
                    math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12),
                    f"{got} != {want}",
                )
            merged_f = merged.conditional_f()
            direct_f = stats_direct[layer].conditional_f()
            for got_row, want_row in zip(merged_f, direct_f):
                for got, want in zip(got_row, want_row):
                    self.assertTrue(math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12))

    def test_model_forward_output_contract_preserved(self):
        # make_observer installed augment/restore hooks around the observer;
        # the model's callers must still receive the bare tensor.
        with torch.no_grad():
            out = self.model(self.inputs[0])
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), tuple(self.inputs[0].shape))
        self.assertTrue(torch.isfinite(out).all())


class DeterminismTests(unittest.TestCase):
    def test_repeated_observation_is_byte_identical(self):
        inputs = _batches(count=1, seed=77)
        _, stats_a, reap_a, _ = _observe(build_fixture(), inputs)
        _, stats_b, reap_b, _ = _observe(build_fixture(), inputs)
        for layer in stats_a:
            self.assertEqual(stats_a[layer].to_json(), stats_b[layer].to_json())
        self.assertEqual(
            json.dumps(reap_adapter._jsonable_reap_state(reap_a), sort_keys=True),
            json.dumps(reap_adapter._jsonable_reap_state(reap_b), sort_keys=True),
        )
        # First-order scores are deterministic and finite for every expert.
        for layer, stats in stats_a.items():
            for value in stats.first_order():
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)

    def test_select_deterministic_across_observations(self):
        inputs = _batches(count=1, seed=99)
        _, stats_a, _, _ = _observe(build_fixture(), inputs)
        _, stats_b, _, _ = _observe(build_fixture(), inputs)
        budget = reap_adapter.MIMO_EXPERTS_PER_LAYER // 4
        for layer in stats_a:
            self.assertEqual(
                hope.diag_topk_pruned(stats_a[layer].conditional_f(), budget),
                hope.diag_topk_pruned(stats_b[layer].conditional_f(), budget),
            )


if __name__ == "__main__":
    unittest.main()
