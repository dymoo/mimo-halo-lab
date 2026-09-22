"""REAP integration for HOPE: one observation pass, two metric families.

Anchored on the pinned upstream clone ``CerebrasResearch/reap`` @
``1970473c51ca3caeb98c10392f15b3a08a672974`` (``upstreams/reap``, override
with ``$REAP_SRC``).  This module never edits the clone; the only upstream
change is the tracked import-deferral patch ``patches/reap-hope.patch``.

One pass, both metrics
----------------------
REAP's ``MoETransformerObserver`` hook calls
``reap.pruning_metrics.update_pruning_state(...)`` and discards the returned
``PreparedPruningBatch`` (activations, selected_experts, router_logits — the
exact expensive per-expert forward re-computation the observer already
performed).  :class:`OnePassListener` wraps that module-level name in
``reap.observer`` and accumulates HOPE statistics from the same batch object,
so REAP scores and HOPE F matrices come from ONE routed pass with no second
model execution and no upstream edit.

Actual routed ids and gates (shared observation seam)
-----------------------------------------------------
The upstream hook re-selects experts with ``topk`` on RAW router logits
(``observer.py``: ``_, selected_experts = torch.topk(router_logits, top_k)``).
That equals the model's actual routing only for bias-free softmax routers
(softmax is strictly increasing, so raw-logit top-k and softmax top-k select the
same experts — verified for ``MixtralSparseMoeBlock`` and
``Qwen3MoeSparseMoeBlock`` in the installed transformers source).  For every
other router the seam :func:`resolve_actual_routing` runs INSIDE the wrapped
``update_pruning_state`` *before* the pinned accumulator touches layer state,
injects the resolved actual ids as ``selected_experts``, and decodes HOPE
statistics from the same returned ``PreparedPruningBatch`` — both metric
families always consume one observation (same experts, same gates, same
expert outputs), and a post-call cross-check fails closed if the batch ever
carries anything but the injected ids.

For normalized sigmoid routers with ``e_score_correction_bias`` the seam
implements the source-grounded formula — selection
``topk(sigmoid(logits) + bias)``, gates gathered from UNBIASED
``sigmoid(logits)`` and renormalized over the selected set — and feeds the
accumulator ``log(sigmoid(logits))`` so its internal softmax + renormalization
reproduces exactly those gates.  MiMo-V2.6-Flash is the first correction-bias
architecture that passes full observation verification: this adapter
registers it at RUNTIME (the pinned clone has no MiMo row and is never
edited) with ``MODEL_ATTRS``/observer-config entries derived from the
verified manifests (``manifests/models/mimo-v2.6-flash-rl/{inventory,memory-
report}.json``) and the pinned modeling source, verifies every layer's
router, correction bias, expert count, top-k and MXFP4 packing evidence at
``attach_model`` (wrong shapes, missing router/bias, dtype divergence and
missing scale siblings all refuse), and exposes the router logits that
``MiMoV2MoE.forward`` does not return through an augment/restore
forward-hook pair around the pinned observer (same-input, same-op ``F.linear``
recompute of the gate's own projection — one extra router matmul, no second
model execution).  Verified degenerate group routing only:
``n_group == topk_group == 1`` makes the source group mask all-ones, so the
seam's plain sigmoid+top-k formula is exact; active groups,
``routed_scaling_factor != 1.0``, non-renormalized gates, and every other
correction-bias router still fail closed at ``attach_model`` with an explicit
unsupported error — never an approximation.  The paper (arXiv:2609.18916v1
§3.1) assumes softmax top-K routing; MiMo's sigmoid+correction-bias routing
violates that assumption, so even this verified integration is a documented
modeling caveat, not paper ground truth.

Slicing uses the REAL upstream ``reap.prune.prune`` (expert ModuleList
reindex, router row slicing, config patch, ``save_pretrained``).  REAP
selection runs natively via the upstream ``topk(saliency, largest=False)`` on
the ``reap`` metric; HOPE selection is injected as an exact 0/1 saliency so
the upstream top-k provably returns the pre-solved exhaustive prune set.

Mac / heavy-dependency note: ``reap.prune`` imports ``reap.eval`` which, at
the pinned revision, imports lm_eval/evalplus/vllm/uvloop at module scope;
``patches/reap-hope.patch`` moves those imports into the actual eval call
paths (the pattern the file already uses for lcb/helm/evalscope).  No mock
modules, no second pruning framework.

CLI::

    PYTHONPATH=src python -m mimo_halo.pruning.reap_adapter \
        tiny-mixtral-smoke --config configs/hope.json --out scratch/hope-tiny-smoke

Offline/telemetry: the smoke sets HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE,
HF_HUB_DISABLE_TELEMETRY, HF_DATASETS_OFFLINE and WANDB_* from the config's
``offline`` block before importing transformers, builds the tiny Mixtral from
a local config with random init, and only reads/writes local paths — no model
download, no telemetry.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Iterator, NamedTuple, Sequence

from .hope import (
    EXACT_MAX_EXPERTS,
    EXHAUSTIVE_METHOD,
    HopeDependencyError,
    HopeError,
    HopeValidationError,
    HopeStats,
    SCHEMA_VERSION,
    PAPER_REF,
    diag_topk_pruned,
    exhaustive_select,
    load_config,
    oracle_objective,
)

REAP_PIN = "1970473c51ca3caeb98c10392f15b3a08a672974"
REAP_UPSTREAM = "https://github.com/CerebrasResearch/reap"

_REAP_MODULE: Any = None
_PIN_STATE: dict[str, Any] = {}


def _repo_root() -> str:
    # src/mimo_halo/pruning/reap_adapter.py -> repo root is three levels up.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _reap_src_dir() -> str:
    override = os.environ.get("REAP_SRC")
    if override:
        return os.path.abspath(override)
    return os.path.join(_repo_root(), "upstreams", "reap", "src")


def ensure_reap() -> Any:
    """Make the pinned REAP clone importable; fail closed on pin mismatch."""
    global _REAP_MODULE
    if _REAP_MODULE is not None:
        return _REAP_MODULE
    src = _reap_src_dir()
    clone_root = os.path.dirname(src)
    if not os.path.isfile(os.path.join(src, "reap", "__init__.py")):
        raise HopeDependencyError(
            f"pinned REAP clone not found at {clone_root}. Clone and pin it with: "
            f"git clone {REAP_UPSTREAM} {clone_root} && "
            f"git -C {clone_root} checkout {REAP_PIN} "
            "(or set REAP_SRC to an existing checkout)"
        )
    if "pin_verified" not in _PIN_STATE:
        git_dir = os.path.join(clone_root, ".git")
        if os.path.exists(git_dir):
            try:
                head = subprocess.run(
                    ["git", "-C", clone_root, "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError) as exc:
                raise HopeDependencyError(
                    f"cannot verify REAP pin in {clone_root}: {exc}"
                ) from exc
            if head != REAP_PIN:
                raise HopeDependencyError(
                    f"REAP clone at {clone_root} is {head}, required pin {REAP_PIN}; "
                    f"run: git -C {clone_root} checkout {REAP_PIN}"
                )
            _PIN_STATE["pin_verified"] = head
        else:
            # No .git (copied tree): cannot prove the pin; record it explicitly
            # so reports state the weaker provenance instead of implying proof.
            _PIN_STATE["pin_verified"] = None
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        _REAP_MODULE = importlib.import_module("reap")
    except ImportError as exc:
        raise HopeDependencyError(
            f"pinned REAP at {clone_root} is present but failed to import: {exc}"
        ) from exc
    return _REAP_MODULE


def reap_pin_state() -> dict[str, Any]:
    ensure_reap()
    return {
        "required": REAP_PIN,
        "clone_src": _reap_src_dir(),
        "verified_head": _PIN_STATE.get("pin_verified"),
        "verification": (
            "git rev-parse matched required pin"
            if _PIN_STATE.get("pin_verified") == REAP_PIN
            else "unverifiable (no .git in clone) — pin claimed by provenance only"
        ),
    }


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise HopeDependencyError(
            "torch is required for the REAP integration path "
            "(stdlib HOPE core does not need it); install: python -m pip install torch"
        ) from exc
    return torch


def _layer_attr_path() -> str:
    return "model.layers"


def _model_attrs(model) -> dict[str, Any]:
    ensure_reap()
    if model.__class__.__name__ == MIMO_MODEL_CLASS:
        ensure_mimo_registration()
    model_util = importlib.import_module("reap.model_util")
    attrs = model_util.MODEL_ATTRS.get(model.__class__.__name__)
    if attrs is None:
        registered = sorted(model_util.MODEL_ATTRS)
        raise HopeDependencyError(
            f"architecture {model.__class__.__name__!r} has no MODEL_ATTRS entry in "
            f"pinned REAP (registered: {registered}). Adding one requires verified "
            "metadata for that checkpoint — unsupported, not silently approximated."
        )
    if attrs.get("fused"):
        raise HopeDependencyError(
            f"architecture {model.__class__.__name__!r} uses fused experts; the HOPE "
            "adapter only supports the per-expert (non-fused) observer path"
        )
    return attrs


def iter_moe_blocks(model, attrs: dict[str, Any]):
    """Yield (layer_index, moe_block) using the upstream attribute map."""
    layers_path = _layer_attr_path().split(".")
    try:
        container = model
        for part in layers_path:
            container = getattr(container, part)
    except AttributeError as exc:
        raise HopeDependencyError(
            f"model does not expose {_layer_attr_path()!r}: {exc}"
        ) from exc
    for layer_index in range(len(container)):
        block = getattr(container[layer_index], attrs["moe_block"])
        yield layer_index, block


def _find_router_bias(block, attrs: dict[str, Any]):
    router = getattr(block, attrs.get("router", "gate"), None)
    holders = [router, block, getattr(block, "moe_statics", None)]
    for holder in holders:
        if holder is None:
            continue
        bias = getattr(holder, "e_score_correction_bias", None)
        if bias is not None:
            return bias
    return None


_VERIFIED_SOFTMAX_BLOCKS = (
    "MixtralSparseMoeBlock",
    "Qwen3MoeSparseMoeBlock",
)


def _verified_softmax_spec(block) -> RoutingSpec:
    """Routing contract verified against the installed transformers source.

    Verified (transformers 4.57.6 on this workstation):

    - ``MixtralSparseMoeBlock`` — ``softmax -> topk -> /= selected sum``
      unconditionally (``models/mixtral/modeling_mixtral.py``,
      ``MixtralSparseMoeBlock.forward``).  Raw-logit top-k therefore equals
      the model's selection (softmax strictly increasing) and the pinned
      accumulator's softmax + renormalization reproduces the model's gates.
    - ``Qwen3MoeSparseMoeBlock`` — same softmax top-k with renormalization
      gated on ``norm_topk_prob`` (``models/qwen3_moe/modeling_qwen3_moe.py``);
      only the renormalized configuration is accepted because the pinned
      accumulator always divides by the selected-set sum.

    Every other block class fails closed: its logits-exposing output contract
    and router formula have not been verified here (DeepSeek-V2 groups and
    ``routed_scaling_factor``, Ernie, gpt-oss sigmoid clamping, fused Llama4,
    Glm4MoeMoE returning a bare tensor).
    """
    name = block.__class__.__name__
    if name == "MixtralSparseMoeBlock":
        return RoutingSpec(
            mode=SOFTMAX_TOPK_ROUTES,
            note="softmax top-k with unconditional selected-set renormalization",
        )
    if name == "Qwen3MoeSparseMoeBlock":
        if not bool(getattr(block, "norm_topk_prob", False)):
            raise HopeValidationError(
                "Qwen3MoeSparseMoeBlock.norm_topk_prob is False: the model's gates "
                "are unnormalized softmax values, which the pinned accumulator "
                "(renormalize_router_weights=True) cannot reproduce — unsupported, "
                "not approximated"
            )
        return RoutingSpec(
            mode=SOFTMAX_TOPK_ROUTES,
            note="softmax top-k, norm_topk_prob=True (renormalized)",
        )
    raise HopeValidationError(
        f"block class {name!r} has no source-verified actual-routing contract "
        f"(verified: {', '.join(_VERIFIED_SOFTMAX_BLOCKS)}); refusing to record "
        "'actual' model routing from an unverified formula"
    )


# ---------------------------------------------------------------------------
# MiMo-V2.6-Flash registration (verified metadata — adapter-owned, runtime)
# ---------------------------------------------------------------------------
#
# The pinned REAP clone has no MiMo row; this module never edits the clone,
# so the rows are registered at runtime from verified metadata.  Every value
# below is traced to the authoritative manifests of revision
# 5711b268169967567844e1e560e8a3966da959b1:
#
#   MIMO_MOE_LAYERS (47, layers 1..47; layer 0 dense)
#       -> inventory.json  architecture.moe_layers (47 entries, 1..47)
#   MIMO_EXPERTS_PER_LAYER (256)
#       -> inventory.json  architecture.original_experts_per_layer
#   MIMO_TOP_K (8)
#       -> inventory.json  architecture.top_k
#   hidden 4096 / router weight [256, 4096] BF16
#       -> inventory.json  tensors "model.layers.1.mlp.gate.weight"
#                          (shape [256, 4096], dtype BF16); hidden cross-checked
#                          against tensors "model.norm.weight" [4096] and
#                          "model.embed_tokens.weight" [152576, 4096]
#   router correction bias [256] F32
#       -> inventory.json  tensors "model.layers.1.mlp.gate.e_score_correction_bias";
#                          memory-report.json native.router_note ("dense BF16 gate
#                          weight + F32 correction bias per MoE layer")
#   moe intermediate 2048; experts U8 MXFP4 packed + per-32 scale siblings
#       -> inventory.json  tensors "model.layers.1.mlp.experts.0.{gate,up}_proj.weight"
#                          shape [2048, 2048] U8 / logical [2048, 4096] /
#                          quantization.kind "mxfp4_packed" block_size 32;
#                          "...down_proj.weight" [4096, 1024] U8 / logical
#                          [4096, 2048]; every weight carries a "...weight_scale"
#                          U8 sibling ([2048, 128] / [4096, 64],
#                          quantization.kind "mxfp4_scale" block_size 32);
#                          6 tensors per expert instance ->
#                          native.per_layer_expert_layout.tensors_per_expert_instance
#   router/gate formula (sigmoid + noaux_tc, degenerate groups)
#       -> source-metadata/config.json scoring_func "sigmoid",
#          topk_method "noaux_tc", n_group 1, topk_group 1,
#          norm_topk_prob true, routed_scaling_factor null (-> 1.0),
#          n_routed_experts 256, num_experts_per_tok 8
#       -> pinned modeling_mimo_v2.py MiMoV2MoEGate.forward (selection
#          topk(sigmoid(logits) + bias), gates = unbiased sigmoid gather
#          renormalized over the selected set, x routed_scaling_factor)

MIMO_MODEL_CLASS = "MiMoV2ForCausalLM"  # source-metadata/config.json architectures[0]
MIMO_BLOCK_CLASS = "MiMoV2MoE"  # pinned modeling_mimo_v2.py block class
MIMO_DENSE_BLOCK_CLASS = "MiMoV2MLP"  # layer-0 dense block (never observed)
MIMO_GATE_CLASS = "MiMoV2MoEGate"  # pinned modeling_mimo_v2.py router class
MIMO_MOE_LAYERS = 47  # inventory.json architecture.moe_layers (count)
MIMO_EXPERTS_PER_LAYER = 256  # inventory.json architecture.original_experts_per_layer
MIMO_TOP_K = 8  # inventory.json architecture.top_k
MIMO_EXPOSE_LOGITS = "recompute_gate_linear"


def _mimo_expected_attrs() -> dict[str, Any]:
    """The exact MODEL_ATTRS row this adapter registers for MiMo."""
    return {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
        # Adapter-owned keys (upstream consumers ignore them):
        "dense_block": MIMO_DENSE_BLOCK_CLASS,
        "expose_router_logits": MIMO_EXPOSE_LOGITS,
    }


def _mimo_expected_hook_config() -> dict[str, Any]:
    """The exact observer hook-config contract this adapter registers."""
    return {
        "module_class_name_to_hook_regex": MIMO_BLOCK_CLASS,
        "num_experts_attr_name": "config.n_routed_experts",
        "top_k_attr_name": "gate.top_k",
        "fused_experts": False,
        "expose_router_logits": True,
    }


def _mimo_hook_config_cls(observer_mod):
    @dataclass
    class MiMoV2ObserverHookConfig(observer_mod.MoETransformerObserverConfig):
        """Hook config for ``MiMoV2MoE`` blocks (hooked by exact class name).

        ``expose_router_logits`` is adapter-owned: MiMo's block forward returns
        a bare tensor, so :func:`make_observer` wraps the pinned observer with
        augment/restore forward hooks that expose the gate's router logits to
        the observer and hand the original tensor back to the model.
        """

        module_class_name_to_hook_regex: str | None = MIMO_BLOCK_CLASS
        num_experts_attr_name: str = "config.n_routed_experts"
        top_k_attr_name: str = "gate.top_k"
        fused_experts: bool = False
        expose_router_logits: bool = True

    return MiMoV2ObserverHookConfig


def ensure_mimo_registration() -> None:
    """Register the adapter-owned MiMo rows in the pinned REAP registries.

    Idempotent.  Fails closed if the pinned clone ever ships a *different*
    MiMo row: diverging metadata must never be silently overwritten or
    observed under.
    """
    ensure_reap()
    model_util = importlib.import_module("reap.model_util")
    observer_mod = importlib.import_module("reap.observer")
    expected_attrs = _mimo_expected_attrs()
    existing_attrs = model_util.MODEL_ATTRS.get(MIMO_MODEL_CLASS)
    if existing_attrs is None:
        model_util.MODEL_ATTRS[MIMO_MODEL_CLASS] = dict(expected_attrs)
    elif existing_attrs != expected_attrs:
        raise HopeDependencyError(
            f"pinned REAP ships a MODEL_ATTRS entry for {MIMO_MODEL_CLASS} that "
            "diverges from this adapter's verified metadata — refusing to observe "
            "under unverified attributes"
        )
    expected_cfg = _mimo_expected_hook_config()
    config_cls = observer_mod.OBSERVER_CONFIG_REGISTRY.get(MIMO_MODEL_CLASS)
    if config_cls is None:
        observer_mod.OBSERVER_CONFIG_REGISTRY[MIMO_MODEL_CLASS] = (
            _mimo_hook_config_cls(observer_mod)
        )
        config_cls = observer_mod.OBSERVER_CONFIG_REGISTRY[MIMO_MODEL_CLASS]
    probe = config_cls()
    actual_cfg = {key: getattr(probe, key, None) for key in expected_cfg}
    if actual_cfg != expected_cfg:
        raise HopeDependencyError(
            f"pinned REAP ships an observer hook config for {MIMO_MODEL_CLASS} that "
            f"diverges from this adapter's verified contract ({actual_cfg!r} != "
            f"{expected_cfg!r}) — refusing to observe"
        )


def _mimo_packed_projection_guard(
    layer: int,
    where: str,
    proj: Any,
    *,
    out_features: int,
    in_features: int,
    torch: Any,
) -> None:
    """Verify one expert projection carries native MXFP4 packing evidence.

    Checked against inventory.json tensor records: U8 nibble-packed ``weight``
    of shape ``(out, in // 2)`` plus its ``weight_scale`` sibling of shape
    ``(out, in // 32)`` (``quantization.kind`` ``mxfp4_packed`` / ``mxfp4_scale``,
    ``block_size`` 32).  Any missing sibling, dtype divergence, or shape
    mismatch refuses.
    """
    weight = getattr(proj, "weight", None)
    if weight is None:
        raise HopeValidationError(
            f"layer {layer}: {where} has no weight tensor — packing evidence "
            "absent, refusing"
        )
    scale = getattr(proj, "weight_scale", None)
    if scale is None:
        raise HopeValidationError(
            f"layer {layer}: {where} is missing its MXFP4 per-32 scale sibling "
            "'weight_scale' — packing evidence absent, refusing"
        )
    if weight.dtype != torch.uint8:
        raise HopeValidationError(
            f"layer {layer}: {where}.weight dtype {weight.dtype} diverges from the "
            "verified native MXFP4 U8 packed layout (inventory.json "
            "quantization.kind=mxfp4_packed) — refusing to observe a checkpoint "
            "that is not the original packed weights"
        )
    if tuple(weight.shape) != (out_features, in_features // 2):
        raise HopeValidationError(
            f"layer {layer}: {where}.weight shape {tuple(weight.shape)} != packed "
            f"({out_features}, {in_features // 2}) for logical "
            f"({out_features}, {in_features}) — wrong shapes, refusing"
        )
    if scale.dtype != torch.uint8:
        raise HopeValidationError(
            f"layer {layer}: {where}.weight_scale dtype {scale.dtype} diverges "
            "from the verified U8 MXFP4 scale layout — refusing"
        )
    if tuple(scale.shape) != (out_features, in_features // 32):
        raise HopeValidationError(
            f"layer {layer}: {where}.weight_scale shape {tuple(scale.shape)} != "
            f"({out_features}, {in_features // 32}) for block size 32 — wrong "
            "scale-sibling shape, refusing"
        )


def _verified_mimo_spec(block, attrs: dict[str, Any], layer: int) -> RoutingSpec:
    """Verified ``MiMoV2MoE`` routing contract (pinned modeling source).

    Source: revision 5711b268169967567844e1e560e8a3966da959b1
    ``modeling_mimo_v2.py`` ``MiMoV2MoEGate.forward`` plus
    ``source-metadata/config.json``: scoring ``sigmoid`` + ``noaux_tc``
    selection ``topk(sigmoid(logits) + e_score_correction_bias)`` with group
    mask over ``n_group``/``topk_group`` (checkpoint values 1/1 — the mask is
    all-ones, i.e. plain global top-k), gates gathered from UNBIASED
    ``sigmoid(logits)``, renormalized over the selected set
    (``norm_topk_prob``) and scaled by ``routed_scaling_factor`` (checkpoint
    value null -> 1.0).  Anything beyond that verified degenerate
    configuration refuses.
    """
    torch = _require_torch()
    if block.__class__.__name__ != MIMO_BLOCK_CLASS:
        raise HopeValidationError(
            f"layer {layer}: expected {MIMO_BLOCK_CLASS}, got "
            f"{block.__class__.__name__!r} — unverified block, refusing"
        )
    gate = getattr(block, attrs.get("router", "gate"), None)
    if gate is None:
        raise HopeValidationError(
            f"layer {layer}: MiMo MoE block has no router module "
            f"({attrs.get('router', 'gate')!r}) — refusing"
        )
    if gate.__class__.__name__ != MIMO_GATE_CLASS:
        raise HopeValidationError(
            f"layer {layer}: router class {gate.__class__.__name__!r} != verified "
            f"{MIMO_GATE_CLASS!r} — unverified routing formula, refusing"
        )
    config = getattr(block, "config", None)
    if config is None:
        raise HopeValidationError(
            f"layer {layer}: MoE block lacks .config (the pinned observer reads "
            "config.n_routed_experts from it) — refusing"
        )
    if getattr(config, "n_routed_experts", None) != MIMO_EXPERTS_PER_LAYER:
        raise HopeValidationError(
            f"layer {layer}: config.n_routed_experts="
            f"{getattr(config, 'n_routed_experts', None)!r} != verified "
            f"{MIMO_EXPERTS_PER_LAYER} — wrong expert count, refusing"
        )
    if getattr(gate, "scoring_func", None) != "sigmoid":
        raise HopeValidationError(
            f"layer {layer}: scoring_func={getattr(gate, 'scoring_func', None)!r} "
            "— only the verified sigmoid router is supported, refusing"
        )
    if getattr(gate, "topk_method", None) != "noaux_tc":
        raise HopeValidationError(
            f"layer {layer}: topk_method={getattr(gate, 'topk_method', None)!r} "
            "— only the verified noaux_tc selection is supported, refusing"
        )
    if getattr(gate, "top_k", None) != MIMO_TOP_K:
        raise HopeValidationError(
            f"layer {layer}: top-k mismatch — gate.top_k="
            f"{getattr(gate, 'top_k', None)!r}, verified {MIMO_TOP_K} "
            "(inventory.json architecture.top_k); refusing"
        )
    if getattr(gate, "n_group", None) != 1 or getattr(gate, "topk_group", None) != 1:
        raise HopeValidationError(
            f"layer {layer}: group routing n_group="
            f"{getattr(gate, 'n_group', None)!r}, topk_group="
            f"{getattr(gate, 'topk_group', None)!r} — only the verified "
            "degenerate 1/1 configuration (all-ones group mask) is supported; "
            "active group selection is not approximated, refusing"
        )
    if getattr(gate, "routed_scaling_factor", None) != 1.0:
        raise HopeValidationError(
            f"layer {layer}: routed_scaling_factor="
            f"{getattr(gate, 'routed_scaling_factor', None)!r} != 1.0 — the pinned "
            "accumulator cannot reproduce scaled output weights, refusing"
        )
    if not getattr(gate, "norm_topk_prob", False):
        raise HopeValidationError(
            f"layer {layer}: norm_topk_prob is not True — gates must be "
            "renormalized over the selected set for REAP and HOPE to observe the "
            "model's weights, refusing"
        )
    experts = getattr(block, attrs["experts"], None)
    if experts is None:
        raise HopeValidationError(
            f"layer {layer}: MoE block has no {attrs['experts']!r} container — "
            "refusing"
        )
    n_experts = len(experts)
    if n_experts != MIMO_EXPERTS_PER_LAYER:
        raise HopeValidationError(
            f"layer {layer}: expert count mismatch — {n_experts} != verified "
            f"{MIMO_EXPERTS_PER_LAYER} "
            "(inventory.json architecture.original_experts_per_layer); refusing"
        )
    weight = getattr(gate, "weight", None)
    if weight is None:
        raise HopeValidationError(f"layer {layer}: router has no weight — refusing")
    bias = getattr(gate, "e_score_correction_bias", None)
    if bias is None:
        raise HopeValidationError(
            f"layer {layer}: router lacks e_score_correction_bias — required by "
            "the verified noaux_tc contract (memory-report.json native.router_note) "
            "and by the seam's selection formula; refusing"
        )
    gating_dim = getattr(gate, "gating_dim", None)
    if type(gating_dim) is not int or gating_dim < 1:
        raise HopeValidationError(
            f"layer {layer}: gate.gating_dim={gating_dim!r} is not a positive int "
            "— hidden size unavailable, refusing"
        )
    if getattr(config, "hidden_size", gating_dim) != gating_dim:
        raise HopeValidationError(
            f"layer {layer}: config.hidden_size="
            f"{getattr(config, 'hidden_size', None)!r} != gate.gating_dim="
            f"{gating_dim} — router/hidden shape divergence, refusing"
        )
    if tuple(weight.shape) != (MIMO_EXPERTS_PER_LAYER, gating_dim):
        raise HopeValidationError(
            f"layer {layer}: router weight shape {tuple(weight.shape)} != "
            f"({MIMO_EXPERTS_PER_LAYER}, {gating_dim}) — wrong router shape, "
            "refusing"
        )
    if weight.dtype != torch.bfloat16:
        raise HopeValidationError(
            f"layer {layer}: router weight dtype {weight.dtype} diverges from the "
            "verified dense BF16 gate weight (memory-report.json native.router_note) "
            "— refusing"
        )
    if tuple(bias.shape) != (MIMO_EXPERTS_PER_LAYER,):
        raise HopeValidationError(
            f"layer {layer}: correction bias shape {tuple(bias.shape)} != "
            f"({MIMO_EXPERTS_PER_LAYER},) — wrong bias shape, refusing"
        )
    if bias.dtype != torch.float32:
        raise HopeValidationError(
            f"layer {layer}: correction bias dtype {bias.dtype} diverges from the "
            "verified F32 correction bias (memory-report.json native.router_note) "
            "— refusing"
        )
    # Expert packing evidence: 6 tensors per expert instance (3 packed weights
    # + 3 per-32 scale siblings).  Dimensions are cross-checked between the
    # router (gating_dim = hidden) and every expert projection.
    first_gate_proj = getattr(experts[0], attrs["gate_proj"], None)
    first_weight = getattr(first_gate_proj, "weight", None)
    if first_weight is None:
        raise HopeValidationError(
            f"layer {layer}: experts[0] has no {attrs['gate_proj']}.weight — "
            "refusing"
        )
    hidden = gating_dim
    inter = int(first_weight.shape[0])
    if inter < 1 or hidden % 32 or inter % 32:
        raise HopeValidationError(
            f"layer {layer}: hidden={hidden}, moe_intermediate={inter} must be "
            "positive multiples of the verified MXFP4 block size 32 "
            "(inventory.json precision_costs.mxfp4_native_u8_packed) — refusing"
        )
    projections = (
        (attrs["gate_proj"], inter, hidden),
        (attrs["up_proj"], inter, hidden),
        (attrs["down_proj"], hidden, inter),
    )
    for index, expert in enumerate(experts):
        for proj_name, out_features, in_features in projections:
            proj = getattr(expert, proj_name, None)
            if proj is None:
                raise HopeValidationError(
                    f"layer {layer}: experts.{index} has no {proj_name!r} "
                    "projection — refusing"
                )
            _mimo_packed_projection_guard(
                layer,
                f"experts.{index}.{proj_name}",
                proj,
                out_features=out_features,
                in_features=in_features,
                torch=torch,
            )
    return RoutingSpec(
        mode=SIGMOID_BIAS_TOPK_ROUTES,
        bias=tuple(float(v) for v in bias.detach().cpu().tolist()),
        note=(
            f"{MIMO_GATE_CLASS}: sigmoid + noaux_tc correction-bias routing with "
            "degenerate group mask (n_group=1/topk_group=1, "
            "routed_scaling_factor=1.0, norm_topk_prob=True — pinned "
            "modeling_mimo_v2.py / source-metadata/config.json); selection "
            "topk(sigmoid(logits)+bias), gates = unbiased sigmoid renormalized "
            f"over the selected set; hidden={hidden}, moe_intermediate={inter}, "
            f"experts={n_experts}, top_k={MIMO_TOP_K} verified against "
            "inventory.json tensor records (native MXFP4 U8 packed weights + "
            "per-32 scale siblings)"
        ),
    )


# ---------------------------------------------------------------------------
# Actual-routing seam (ONE observation for REAP and HOPE)
# ---------------------------------------------------------------------------

SOFTMAX_TOPK_ROUTES = "softmax_topk_renorm"
SIGMOID_BIAS_TOPK_ROUTES = "sigmoid_topk_bias_renorm"


class RoutingSpec(NamedTuple):
    """Source-verified router semantics for one MoE layer.

    ``mode`` is one of the module constants; the remaining fields narrow the
    contract.  :func:`resolve_actual_routing` refuses every combination it
    cannot reproduce exactly for BOTH metric families — group routing, routed
    scaling factors, and non-renormalized gates have no approximation branch.
    """

    mode: str
    renormalize: bool = True
    group_routing: bool = False
    routed_scaling_factor: float = 1.0
    bias: tuple[float, ...] | None = None
    note: str = ""


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _log_sigmoid(value: float) -> float:
    # log(sigmoid(v)) = -softplus(-v), evaluated without overflow.
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _select_topk(scores: Sequence[float], k: int) -> list[int]:
    """Descending top-k with ties broken to the lower expert id; ascending ids."""
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return sorted(order[:k])


def resolve_actual_routing(
    logits: Sequence[Sequence[float]],
    upstream_ids: Sequence[Sequence[int]],
    spec: RoutingSpec,
) -> tuple[list[list[int]], list[list[float]], list[list[float]] | None]:
    """Resolve the ACTUAL routed ids and gates — the ONE observation both the
    pinned REAP accumulators and HOPE statistics consume.

    Args:
        logits: raw router logits ``(tokens, E)`` exactly as the model's
            router produced them.
        upstream_ids: the pinned observer's raw-logit top-k proposal
            ``(tokens, k)``.  Trusted ONLY for :data:`SOFTMAX_TOPK_ROUTES`
            (softmax is strictly increasing, so raw-logit top-k selects the
            model's experts); ignored for selection under
            :data:`SIGMOID_BIAS_TOPK_ROUTES`.
        spec: the layer's source-verified routing contract.

    Returns:
        ``(ids, gates, upstream_router_logits)``:

        - ``ids`` — the model's actual selected experts per token;
        - ``gates`` — the model's actual output weights: unbiased score gather
          renormalized over the selected set (for the sigmoid+bias mode the
          correction bias affects ONLY selection, never the gathered weights);
        - values to feed the pinned REAP accumulator's ``router_logits`` so its
          internal ``softmax(...)`` plus configured selected-set renormalization
          reproduces exactly ``gates`` (``None`` = the raw logits are already
          such values, identity transform).

    Raises :class:`HopeValidationError` — for unsupported contracts and for
    non-finite/invalid inputs alike — BEFORE any caller mutates metrics.
    There is no silent approximation branch.
    """
    if spec.mode not in (SOFTMAX_TOPK_ROUTES, SIGMOID_BIAS_TOPK_ROUTES):
        raise HopeValidationError(
            f"unsupported routing mode {spec.mode!r} — actual routing cannot be "
            "resolved, refusing before metrics mutation"
        )
    if spec.group_routing:
        raise HopeValidationError(
            "group routing (n_group/topk_group masking) is unsupported — the seam "
            "will not approximate unverified group selection"
        )
    if spec.routed_scaling_factor != 1.0:
        raise HopeValidationError(
            f"routed_scaling_factor={spec.routed_scaling_factor} is unsupported — "
            "the pinned accumulator cannot reproduce scaled output weights, refusing "
            "to record diverging gates"
        )
    if not spec.renormalize:
        raise HopeValidationError(
            "non-renormalized gates are unsupported — REAP and HOPE must observe the "
            "same selected-set-renormalized weights"
        )
    if spec.mode == SIGMOID_BIAS_TOPK_ROUTES:
        if spec.bias is None:
            raise HopeValidationError(
                "sigmoid+bias routing requires the e_score_correction_bias vector; "
                "selection is topk(sigmoid(logits)+bias) and gathered weights come "
                "from UNBIASED sigmoid(logits)"
            )
    elif spec.bias is not None:
        raise HopeValidationError(
            "softmax routing must not carry a correction bias — resolve the actual "
            "routing contract instead of guessing"
        )

    if len(upstream_ids) != len(logits):
        raise HopeValidationError(
            f"router logits rows ({len(logits)}) and upstream top-k rows "
            f"({len(upstream_ids)}) diverge"
        )
    if not logits:
        z_rows: list[list[float]] | None = (
            [] if spec.mode == SIGMOID_BIAS_TOPK_ROUTES else None
        )
        return [], [], z_rows

    num_experts = len(logits[0])
    if num_experts == 0:
        raise HopeValidationError("router logits have zero experts")
    bias = spec.bias
    if bias is not None and len(bias) != num_experts:
        raise HopeValidationError(
            f"correction bias width {len(bias)} != router width {num_experts}"
        )

    top_k = len(upstream_ids[0])
    if top_k < 1:
        raise HopeValidationError("upstream top-k width must be >= 1")

    ids_rows: list[list[int]] = []
    gates_rows: list[list[float]] = []
    z_rows = [] if spec.mode == SIGMOID_BIAS_TOPK_ROUTES else None
    for token, (logit_row, id_row) in enumerate(zip(logits, upstream_ids)):
        if len(logit_row) != num_experts:
            raise HopeValidationError(
                f"token {token}: router logits width {len(logit_row)} != {num_experts}"
            )
        if len(id_row) != top_k:
            raise HopeValidationError(
                f"token {token}: upstream top-k width {len(id_row)} != {top_k}"
            )
        for value in logit_row:
            if type(value) not in (int, float) or (
                type(value) is float and not math.isfinite(value)
            ):
                raise HopeValidationError(
                    f"token {token}: router logits are not finite numbers — actual "
                    "routing unavailable, refusing before metrics mutation"
                )
        seen: set[int] = set()
        for expert in id_row:
            if type(expert) is not int or not 0 <= expert < num_experts:
                raise HopeValidationError(
                    f"token {token}: expert id {expert!r} outside [0, {num_experts})"
                )
            if expert in seen:
                raise HopeValidationError(
                    f"token {token}: duplicate expert id {expert}"
                )
            seen.add(expert)

        if spec.mode == SOFTMAX_TOPK_ROUTES:
            row_ids = list(id_row)
            # exp-shifted softmax is proportional to softmax; the shift cancels
            # under selected-set renormalization — same value as the model's
            # softmax -> topk -> renormalize gates.
            maximum = max(logit_row)
            weights = [math.exp(value - maximum) for value in logit_row]
        else:
            assert bias is not None  # guarded above
            scores = [_sigmoid(value) for value in logit_row]
            choice = [
                score + correction
                for score, correction in zip(scores, bias)
            ]
            row_ids = _select_topk(choice, top_k)
            weights = scores

        gathered = [weights[expert] for expert in row_ids]
        total = sum(gathered)
        if not total > 0.0 or not math.isfinite(total):
            raise HopeValidationError(
                f"token {token}: degenerate gathered gate mass {total!r} — actual "
                "routing unavailable, refusing before metrics mutation"
            )
        ids_rows.append(row_ids)
        gates_rows.append([value / total for value in gathered])
        if z_rows is not None:
            z_rows.append([_log_sigmoid(value) for value in logit_row])
    return ids_rows, gates_rows, z_rows


class _Observation(NamedTuple):
    """Resolved actual routing plus the tensors injected into the seam call.

    ``ids``/``gates`` are decode copies (padding rows already dropped); the
    ``selected``/``logits_feed`` tensors stay full length for upstream's own
    mask application.
    """

    ids: list[list[int]]
    gates: list[list[float]]
    selected: Any
    logits_feed: Any
    num_experts: int
    top_k: int


class OnePassListener:
    """Accumulate HOPE :class:`HopeStats` from REAP's own observer pass.

    Install wraps ``reap.observer.update_pruning_state``.  Each call first
    resolves the model's ACTUAL routed ids/gates at the shared seam
    (:func:`resolve_actual_routing` — fail closed BEFORE the pinned
    accumulator touches layer state), injects those ids into the single
    upstream call so REAP metrics accumulate over the same experts, then
    decodes the returned ``PreparedPruningBatch`` into HOPE statistics with
    the same ids/gates and a post-call cross-check.  One observation, two
    metric families, no post-hoc relabeling.
    """

    def __init__(self) -> None:
        self._original: Any = None
        self._installed = False
        self._observer: Any = None
        self._stats_by_state: dict[int, HopeStats] = {}
        self._layer_by_state: dict[int, int] = {}
        self._spec_by_layer: dict[int, RoutingSpec] = {}
        self._capability: str | None = None
        self.forward_passes = 0

    # -- wiring --------------------------------------------------------------

    def attach_model(self, model) -> dict[int, str]:
        """Resolve a source-verified actual-routing spec per MoE layer.

        Fails closed with an explicit unsupported error for every layer whose
        router formula has not been verified against source (correction-bias
        routers, active group routing, scaled or non-renormalized gates) — no
        mirroring, no approximation.  MiMo dispatches to the fully verified
        :func:`_verified_mimo_spec` contract (sigmoid + noaux_tc with the
        checkpoint's degenerate 1/1 group mask), and the architecture's dense
        block (``MODEL_ATTRS['dense_block']``) is skipped — it is not an MoE
        layer and the pinned observer never hooks it.  Committed only after
        every layer resolves.
        """
        attrs = _model_attrs(model)
        mimo = attrs.get("expose_router_logits") == MIMO_EXPOSE_LOGITS
        notes: dict[int, str] = {}
        specs: dict[int, RoutingSpec] = {}
        for layer, block in iter_moe_blocks(model, attrs):
            block_class = block.__class__.__name__
            if mimo and block_class == MIMO_DENSE_BLOCK_CLASS:
                # Verified dense layer (MiMo layer 0): not an MoE block; the
                # pinned observer hooks MiMoV2MoE class names only.
                continue
            if mimo and block_class == MIMO_BLOCK_CLASS:
                spec = _verified_mimo_spec(block, attrs, layer)
                specs[layer] = spec
                notes[layer] = (
                    f"{block_class}: verified sigmoid + noaux_tc "
                    "correction-bias routing with degenerate group mask "
                    "(n_group=1/topk_group=1, routed_scaling_factor=1.0, "
                    "norm_topk_prob=True); selection topk(sigmoid(logits)+bias), "
                    "gates = unbiased sigmoid renormalized over the selected set; "
                    "router logits exposed by the recompute/restore forward-hook "
                    "seam (block forward returns a bare tensor); REAP accumulators "
                    "receive these same ids/gates (single shared observation)"
                )
                continue
            bias = _find_router_bias(block, attrs)
            if bias is not None:
                raise HopeValidationError(
                    f"layer {layer}: {block_class} carries e_score_correction_bias — "
                    "bias-router observation is unsupported at this pin. The seam "
                    "implements the verified selection topk(sigmoid(logits)+bias) "
                    "with weights gathered from UNBIASED sigmoid(logits) "
                    "(unit-tested), but this architecture passes "
                    "output-contract/group/scaling verification only for the "
                    "adapter-registered MiMo contract (and a zero-initialized "
                    "buffer still leaves the block's formula unverified); "
                    "recording raw-logit top-k or biased gates as 'actual routing' "
                    "is rejected by design."
                )
            spec = _verified_softmax_spec(block)
            specs[layer] = spec
            notes[layer] = (
                f"{block_class}: {spec.note}; ids = raw-logit top-k (equal to the "
                "model's softmax top-k by strict monotonicity), gates = softmax "
                "renormalized over the selected set; REAP accumulators receive these "
                "same ids/gates (single shared observation)"
            )
        self._spec_by_layer = specs
        self._model_attrs_used = attrs  # type: ignore[attr-defined]
        return notes

    def bind_observer(self, observer) -> None:
        if not hasattr(observer, "state"):
            raise HopeValidationError("bind_observer expects a REAP observer instance")
        hook_config = getattr(observer, "hook_config", None)
        if not bool(getattr(hook_config, "renormalize_router_weights", False)):
            raise HopeValidationError(
                "gate agreement requires hook_config.renormalize_router_weights=True "
                "on the pinned accumulator (its selected-set renormalization "
                "reproduces the model's gates); the observer was built without it — "
                "refusing to observe rather than record diverging gates"
            )
        self._observer = observer

    def install(self) -> None:
        if self._installed:
            return
        ensure_reap()
        observer_mod = importlib.import_module("reap.observer")
        self._original = observer_mod.update_pruning_state
        original = self._original

        def _wrapper(layer_state, *args, **kwargs):
            # Resolve ACTUAL routing FIRST: if it is unavailable the seam raises
            # here, before `original` mutates any REAP metric (fail closed).
            layer = self._resolve_layer(layer_state)
            observation = self._observe(layer, kwargs)
            kwargs["selected_experts"] = observation.selected
            if observation.logits_feed is not None:
                kwargs["router_logits"] = observation.logits_feed
            batch = original(layer_state, *args, **kwargs)
            self._accumulate(layer_state, layer, batch, observation)
            self.forward_passes += 1
            return batch

        observer_mod.update_pruning_state = _wrapper
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        observer_mod = importlib.import_module("reap.observer")
        observer_mod.update_pruning_state = self._original
        self._installed = False

    @contextlib.contextmanager
    def capability(self, name: str | None) -> Iterator[None]:
        """Label subsequent routed rows with a capability partition."""
        previous = self._capability
        self._capability = name
        try:
            yield
        finally:
            self._capability = previous

    # -- accumulation --------------------------------------------------------

    def _resolve_layer(self, layer_state: object) -> int:
        key = id(layer_state)
        cached = self._layer_by_state.get(key)
        if cached is not None:
            return cached
        if self._observer is None:
            raise HopeValidationError(
                "OnePassListener.bind_observer(...) must be called before the "
                "observed forward pass so routed batches can be attributed to layers"
            )
        for layer, state in self._observer.state.items():
            if state is layer_state:
                self._layer_by_state[key] = layer
                return layer
        raise HopeValidationError(
            "observer state identity not found — the listener must be bound to the "
            "live observer that owns this state (stats lost, refusing to guess)"
        )

    def _observe(self, layer: int, kwargs: dict[str, Any]) -> _Observation:
        """Resolve actual routing from the seam inputs — BEFORE any mutation."""
        torch = _require_torch()
        spec = self._spec_by_layer.get(layer)
        if spec is None:
            raise HopeValidationError(
                "attach_model(model) must run before the observed forward pass so "
                "routing semantics are verified per layer (refusing to observe "
                "without a verified actual-routing contract)"
            )
        num_experts = int(kwargs["num_experts"])
        logits = kwargs["router_logits"]
        selected = kwargs["selected_experts"]
        logit_shape = tuple(getattr(logits, "shape", ()))
        if len(logit_shape) != 2 or logit_shape[1] != num_experts:
            raise HopeValidationError(
                f"layer {layer}: observer saw router logits shaped {logit_shape}, "
                f"expected (tokens, {num_experts}) — the block's output contract "
                "does not expose actual router logits; failing closed before any "
                "metric mutation"
            )
        selected_shape = tuple(getattr(selected, "shape", ()))
        if len(selected_shape) != 2 or selected_shape[0] != logit_shape[0]:
            raise HopeValidationError(
                f"layer {layer}: observer top-k shape {selected_shape} diverges "
                f"from logits {logit_shape} — actual routing unavailable, refusing "
                "before metrics mutation"
            )
        top_k = int(selected_shape[1])
        rows = logits.detach().to(torch.float32).cpu().tolist()
        upstream_ids = selected.detach().cpu().tolist()
        ids, gates, z_rows = resolve_actual_routing(rows, upstream_ids, spec)

        # Feeds to the accumulator stay FULL length: `_prepare_pruning_batch`
        # applies valid_token_mask itself, exactly once.
        selected_feed = torch.tensor(
            ids, dtype=torch.long, device=selected.device
        ).reshape(len(ids), top_k)
        logits_feed = None
        if z_rows is not None:
            # Feed the accumulator log(sigmoid(logits)): its internal
            # softmax(...) + selected-set renormalization then reproduces the
            # model's unbiased sigmoid gates exactly (non-selected entries
            # cancel under the renormalizing division).
            logits_feed = torch.tensor(
                z_rows, dtype=torch.float32, device=logits.device
            ).reshape(len(z_rows), num_experts)

        # Decode copies drop padding rows the same way upstream does, so HOPE
        # rows line up with the returned (already-masked) batch.
        decode_ids, decode_gates = ids, gates
        mask = kwargs.get("valid_token_mask")
        if mask is not None:
            keep = mask.reshape(-1).bool().cpu().tolist()
            if len(keep) != len(rows):
                raise HopeValidationError(
                    f"layer {layer}: attention mask flattens to {len(keep)} rows, "
                    f"logits have {len(rows)} — actual routing unavailable, "
                    "refusing before metrics mutation"
                )
            decode_ids = [row for row, kept in zip(ids, keep) if kept]
            decode_gates = [row for row, kept in zip(gates, keep) if kept]
        return _Observation(
            ids=decode_ids,
            gates=decode_gates,
            selected=selected_feed,
            logits_feed=logits_feed,
            num_experts=num_experts,
            top_k=top_k,
        )

    def _accumulate(
        self,
        layer_state: object,
        layer: int,
        batch,
        observation: _Observation,
    ) -> None:
        torch = _require_torch()
        selected = batch.selected_experts
        actual_ids = selected.detach().cpu().tolist()
        # Cross-check: the REAP batch must carry exactly the injected actual
        # ids — otherwise the two metric families would diverge (fail closed;
        # can only trigger on pinned-upstream contract drift).
        if actual_ids != observation.ids:
            raise HopeValidationError(
                f"layer {layer}: pinned REAP batch carries expert ids that differ "
                "from the resolved actual routing — refusing to accumulate "
                "mismatched REAP/HOPE statistics"
            )
        if not actual_ids:
            return
        stats_key = id(layer_state)
        stats = self._stats_by_state.get(stats_key)
        if stats is None:
            stats = HopeStats(observation.num_experts, observation.top_k)
            self._stats_by_state[stats_key] = stats
        elif stats.n_experts != observation.num_experts:
            raise HopeValidationError(
                f"layer {layer}: num_experts changed mid-run "
                f"{stats.n_experts} -> {observation.num_experts}"
            )
        # activations: (num_experts, tokens, hidden) — norm per (expert, token),
        # gathered at the ACTUAL routed ids so norms align with model routing.
        norms_all = batch.activations.detach().to(torch.float32).norm(dim=-1)
        row_norms = norms_all.transpose(0, 1).gather(1, selected)
        stats.add_batch(
            actual_ids,
            observation.gates,
            row_norms.tolist(),
            capability=self._capability,
        )

    # -- export --------------------------------------------------------------

    def export(self) -> dict[int, HopeStats]:
        """Resolve state identities to layers; fail closed on lost states."""
        if self._observer is None:
            raise HopeValidationError("bind_observer() before export")
        live = {id(state): layer for layer, state in self._observer.state.items()}
        out: dict[int, HopeStats] = {}
        for key, stats in self._stats_by_state.items():
            layer = live.get(key)
            if layer is None:
                if stats.total_rows:
                    raise HopeValidationError(
                        "accumulated stats belong to an observer state that no longer "
                        "exists — statistics lost; refusing to export partial data"
                    )
                continue
            if layer in out:
                raise HopeValidationError(f"duplicate state binding for layer {layer}")
            stats.validate()
            out[layer] = stats
        if not out:
            raise HopeValidationError(
                "no HOPE statistics accumulated — was the listener installed before "
                "the observed forward passes?"
            )
        return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Observer / selection / slicing
# ---------------------------------------------------------------------------


def _mimo_expose_hook():
    """Augment the block output with the gate's own router logits.

    ``MiMoV2MoE.forward`` returns a bare tensor while the pinned observer
    unpacks ``*_, router_logits = output``.  This hook recomputes the router's
    linear projection from the SAME input tensor the gate received during this
    forward (``F.linear`` over float32 casts — identical inputs and op to
    ``MiMoV2MoEGate.forward``) and returns ``(output, logits)`` for the
    observer hook that runs next; the paired restore hook (registered after
    the observer) strips the tuple again so the model's own callers still
    receive the tensor.  One extra router matmul on an already-computed block
    input, never a second model execution — the same recompute pattern the
    pinned observer's fused path uses (``module.router(flat_input)``).
    """
    torch = _require_torch()

    def hook(module, args, output):
        if isinstance(output, tuple):
            return None  # already exposed (seam installed twice without restore)
        gate = module.gate
        if not args:
            raise HopeValidationError(
                "MoE block forward received no input tensor — router logits "
                "unavailable, refusing before metrics mutation"
            )
        hidden = args[0]
        shape = tuple(getattr(hidden, "shape", ()))
        if len(shape) != 3:
            raise HopeValidationError(
                f"expected (batch, seq, hidden) block input, got {shape} — router "
                "logits unavailable, refusing before metrics mutation"
            )
        logits = torch.nn.functional.linear(
            hidden.reshape(-1, shape[-1]).to(torch.float32),
            gate.weight.to(torch.float32),
            None,
        )
        return (output, logits)

    return hook


def _mimo_restore_hook(module, args, output):
    """Hand the original bare-tensor output back to the model's callers."""
    if isinstance(output, tuple) and len(output) == 2:
        return output[0]
    return None


def _install_mimo_logits_seam(model, *, restore: bool) -> None:
    """Install (or reinstall) the augment/restore forward hooks on MiMo blocks.

    ``restore=False`` runs BEFORE the observer is constructed (so the
    observer's own hook sees ``(hidden, logits)``); ``restore=True`` runs
    after (so it runs after the observer's hook and before the model's
    callers).  Reinstalling removes this adapter's previous handles first;
    stale observer hooks from an unclosed observer then fail closed at the
    seam's shape check instead of silently observing.
    """
    seam = getattr(model, "_mimo_logits_seam", None)
    if seam is None:
        seam = {"expose": [], "restore": []}
        model._mimo_logits_seam = seam
    key = "restore" if restore else "expose"
    for handle in seam[key]:
        handle.remove()
    seam[key] = []
    attrs = _model_attrs(model)
    for _layer, block in iter_moe_blocks(model, attrs):
        if block.__class__.__name__ != MIMO_BLOCK_CLASS:
            continue
        if restore:
            seam[key].append(block.register_forward_hook(_mimo_restore_hook))
        else:
            seam[key].append(block.register_forward_hook(_mimo_expose_hook()))


def make_observer(model):
    """Create the pinned REAP MoETransformerObserver for this architecture.

    ``renormalize_router_weights`` is fixed to True: the shared seam feeds the
    accumulator router values whose selected-set renormalization reproduces
    the model's own gates exactly, and ``bind_observer`` refuses an observer
    built without it.  There is no configuration that lets REAP and HOPE
    observe different gates.

    For architectures whose hook config sets ``expose_router_logits`` (the
    adapter-registered MiMo row), the pinned observer is wrapped with the
    augment/restore forward-hook pair so it observes the block's true router
    logits while the model keeps its bare-tensor output contract.
    """
    ensure_reap()
    name = model.__class__.__name__
    if name == MIMO_MODEL_CLASS:
        ensure_mimo_registration()
    observer_mod = importlib.import_module("reap.observer")
    registry = observer_mod.OBSERVER_CONFIG_REGISTRY
    config_cls = registry.get(name)
    if config_cls is None:
        raise HopeDependencyError(
            f"no REAP observer hook config for {name!r} "
            f"(registered: {sorted(registry)})"
        )
    hook_config = config_cls()
    hook_config.renormalize_router_weights = True
    hook_config.record_pruning_metrics_only = True
    if bool(getattr(hook_config, "expose_router_logits", False)):
        _install_mimo_logits_seam(model, restore=False)
        observer = observer_mod.MoETransformerObserver(
            model, hook_config=hook_config
        )
        _install_mimo_logits_seam(model, restore=True)
        return observer
    return observer_mod.MoETransformerObserver(model, hook_config=hook_config)


def extract_reap_state(observer) -> dict[int, dict[str, Any]]:
    """REAP-side metrics from the same pass (OnlineStatsTracker -> mean)."""
    return {layer: dict(values) for layer, values in observer.report_state().items()}


def encode_exact_prune_set(n_experts: int, pruned: Sequence[int]) -> Any:
    """0.0 for pruned experts, 1.0 for retained — upstream
    ``topk(saliency, |P|, largest=False)`` then returns exactly ``pruned``:
    there are exactly |P| zeros and the next value is 1.0, so tie-breaking
    cannot alter the set."""
    torch = _require_torch()
    pruned_set = set(pruned)
    if len(pruned_set) != len(list(pruned)):
        raise HopeValidationError("duplicate ids in prune set")
    for expert in pruned:
        if type(expert) is not int or not 0 <= expert < n_experts:
            raise HopeValidationError(f"prune id {expert!r} outside [0, {n_experts})")
    if not pruned_set:
        raise HopeValidationError("empty prune set cannot drive top-k selection")
    values = [0.0 if i in pruned_set else 1.0 for i in range(n_experts)]
    return torch.tensor(values, dtype=torch.float32)


def _saliency_finite(layer: int, key: str, tensor: Any) -> None:
    torch = _require_torch()
    if not bool(torch.isfinite(tensor).all()):
        raise HopeValidationError(
            f"layer {layer}: non-finite '{key}' saliency (expert never active or "
            "corrupted observer state) — refusing to select"
        )


def slice_with_reap(
    model,
    observer_data: dict[int, dict[str, Any]],
    *,
    prune_method: str,
    n_experts_to_prune: int,
    out_dir: str,
) -> str:
    """Run the REAL upstream slicing path (reap.prune.prune) and save."""
    ensure_reap()
    prune_mod = importlib.import_module("reap.prune")
    import pathlib
    import types

    for layer, values in observer_data.items():
        saliency = values.get(prune_method)
        if saliency is None:
            raise HopeValidationError(
                f"layer {layer}: observer data lacks saliency key {prune_method!r}"
            )
        _saliency_finite(layer, prune_method, saliency)
    prune_args = types.SimpleNamespace(
        prune_method=prune_method,
        perserve_super_experts=False,
        perserve_outliers=False,
    )
    result = prune_mod.prune(
        observer_data,
        model,
        prune_args,
        n_experts_to_prune,
        pathlib.Path(out_dir),
    )
    return str(result)


def reload_and_verify(path: str, *, expect_experts: int, seq_len: int) -> dict[str, Any]:
    """Load a sliced checkpoint from local disk and prove it forwards."""
    torch = _require_torch()
    transformers = importlib.import_module("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained(path)
    config_experts = getattr(model.config, "num_local_experts", None)
    gate = model.model.layers[0].block_sparse_moe.gate
    gate_out = int(gate.out_features)
    generator = torch.Generator().manual_seed(1234)
    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (1, seq_len), generator=generator)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    shape = [int(s) for s in logits.shape]
    forward_ok = shape == [1, seq_len, vocab]
    total_bytes = 0
    file_names = []
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            full = os.path.join(root, name)
            total_bytes += os.path.getsize(full)
            file_names.append(os.path.relpath(full, path))
    checks = {
        "config_num_local_experts_matches": config_experts == expect_experts,
        "gate_out_features_matches": gate_out == expect_experts,
        "forward_logits_shape": forward_ok,
        "logits_finite": bool(torch.isfinite(logits).all()),
    }
    del model
    return {
        "path": path,
        "config_num_local_experts": config_experts,
        "gate_out_features": gate_out,
        "expected_experts": expect_experts,
        "logits_shape": shape,
        "total_bytes": total_bytes,
        "files": sorted(file_names),
        "checks": checks,
        "ok": all(checks.values()),
    }


def _hope_and_reap_sets(
    stats_by_layer: dict[int, HopeStats],
    reap_state: dict[int, dict[str, Any]],
    *,
    budget: int,
    max_exact_experts: int,
) -> dict[int, dict[str, Any]]:
    """Solve HOPE exactly per layer; report the native REAP top-k set."""
    torch = _require_torch()
    out: dict[int, dict[str, Any]] = {}
    for layer, stats in sorted(stats_by_layer.items()):
        if stats.n_experts > max_exact_experts:
            raise HopeValidationError(
                f"layer {layer}: E={stats.n_experts} exceeds exact range "
                f"{max_exact_experts}; the smoke requires the exhaustive oracle"
            )
        f_matrix = stats.conditional_f()
        cert = exhaustive_select(f_matrix, budget, max_experts=max_exact_experts)
        dense = oracle_objective(f_matrix, cert["pruned"])
        if dense != cert["objective"]:
            raise HopeValidationError(
                f"layer {layer}: objective {cert['objective']} != dense oracle {dense}"
            )
        reap_scores = reap_state[layer]["reap"]
        _saliency_finite(layer, "reap", reap_scores)
        k = min(budget, int(reap_scores.numel()))
        reap_pruned = sorted(
            int(i)
            for i in torch.topk(reap_scores, k, largest=False).indices.tolist()
        )
        out[layer] = {
            "hope": {
                "pruned": cert["pruned"],
                "objective": cert["objective"],
                "exact": cert["exact"],
                "ties": cert["ties"],
                "method": cert["method"],
                "subsets_evaluated": cert["subsets_evaluated"],
            },
            "reap": {
                "pruned": reap_pruned,
                "objective_under_hope_f": oracle_objective(f_matrix, reap_pruned),
                "method": "upstream reap saliency topk(largest=False)",
            },
            "first_order_from_stats": [
                round(v, 6) for v in stats.first_order()
            ],
            "routing_frequency": [round(v, 6) for v in stats.routing_frequency()],
            "mean_gates": [round(v, 6) for v in stats.mean_gates()],
        }
    return out


def _hope_objective_not_worse(sets: dict[int, dict[str, Any]]) -> bool:
    return all(
        layer_data["hope"]["objective"]
        <= layer_data["reap"]["objective_under_hope_f"] + 1e-9
        for layer_data in sets.values()
    )


def _jsonable_reap_state(reap_state: dict[int, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer, values in sorted(reap_state.items()):
        row: dict[str, Any] = {}
        for key, value in sorted(values.items()):
            if hasattr(value, "tolist"):
                converted = value.tolist()
                if isinstance(converted, float) and not math.isfinite(converted):
                    converted = str(converted)
                row[key] = converted
            elif isinstance(value, (int, float, str, bool)) or value is None:
                row[key] = value
            else:
                row[key] = str(value)
        out[str(layer)] = row
    return out


# ---------------------------------------------------------------------------
# Tiny integration smoke
# ---------------------------------------------------------------------------


def _apply_offline_env(offline: dict[str, Any]) -> dict[str, str]:
    applied = {}
    for key, value in offline.items():
        text = str(value)
        os.environ[key] = text
        applied[key] = text
    return applied


def _build_tiny_mixtral(smoke_cfg: dict[str, Any], seed: int):
    torch = _require_torch()
    transformers = importlib.import_module("transformers")
    torch.manual_seed(seed)
    config = transformers.MixtralConfig(
        vocab_size=int(smoke_cfg["vocab_size"]),
        hidden_size=int(smoke_cfg["hidden_size"]),
        intermediate_size=int(smoke_cfg["intermediate_size"]),
        num_hidden_layers=int(smoke_cfg["num_hidden_layers"]),
        num_attention_heads=int(smoke_cfg["num_attention_heads"]),
        num_key_value_heads=int(smoke_cfg["num_key_value_heads"]),
        max_position_embeddings=int(smoke_cfg["max_position_embeddings"]),
        num_local_experts=int(smoke_cfg["num_local_experts"]),
        num_experts_per_tok=int(smoke_cfg["num_experts_per_tok"]),
    )
    model = transformers.MixtralForCausalLM(config)
    model.eval()
    return model


def run_tiny_mixtral_smoke(config: dict[str, Any], out_dir: str) -> dict[str, Any]:
    """Full integration: local tiny Mixtral -> pinned REAP observer (one pass,
    both metrics) -> exact HOPE selection + native REAP selection -> real REAP
    slicing -> save -> reload -> forward.  INTEGRATION evidence only; this is
    NOT an agentic-quality or model-quality claim."""
    smoke_cfg = config.get("smoke")
    if not isinstance(smoke_cfg, dict):
        raise HopeValidationError("config is missing the 'smoke' section")
    # Offline/telemetry flags FIRST: transformers/huggingface_hub read these at
    # import time, so they must be set before either module loads.
    offline_applied = _apply_offline_env(config.get("offline", {}))
    torch = _require_torch()
    transformers = importlib.import_module("transformers")
    fixtures_cfg = config.get("fixtures", {})
    seed = int(smoke_cfg.get("seed", config.get("seed", 0)))
    budget = int(smoke_cfg["experts_to_prune"])
    n_experts = int(smoke_cfg["num_local_experts"])
    batches = int(smoke_cfg["batches"])
    batch_size = int(smoke_cfg["batch_size"])
    seq_len = int(smoke_cfg["seq_len"])
    max_exact = int(config["exact"]["max_experts"])
    capabilities = list(fixtures_cfg.get("capabilities") or [])

    ensure_reap()
    pin = reap_pin_state()
    model = _build_tiny_mixtral(smoke_cfg, seed)
    params_total = sum(int(p.numel()) for p in model.parameters())

    listener = OnePassListener()
    routing_notes = listener.attach_model(model)
    listener.install()
    observer = make_observer(model)
    listener.bind_observer(observer)

    generator = torch.Generator().manual_seed(seed)
    vocab = int(smoke_cfg["vocab_size"])
    expect_rows = 0
    capability_schedule = []
    try:
        for index in range(batches):
            ids = torch.randint(0, vocab, (batch_size, seq_len), generator=generator)
            label = capabilities[index % len(capabilities)] if capabilities else None
            capability_schedule.append(label)
            with listener.capability(label):
                with torch.no_grad():
                    model(input_ids=ids)
            expect_rows += batch_size * seq_len
        stats_by_layer = listener.export()
        reap_state = extract_reap_state(observer)
    finally:
        listener.uninstall()
        observer.close_hooks()

    for layer, stats in stats_by_layer.items():
        if stats.total_rows != expect_rows:
            raise HopeValidationError(
                f"layer {layer}: rows {stats.total_rows} != expected {expect_rows}"
            )
        stats.validate()

    sets = _hope_and_reap_sets(
        stats_by_layer,
        reap_state,
        budget=budget,
        max_exact_experts=max_exact,
    )
    hope_not_worse = _hope_objective_not_worse(sets)

    import copy
    import pathlib
    import shutil

    out_dir = os.path.abspath(out_dir)
    stats_dir = os.path.join(out_dir, "stats")
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    stat_files = {}
    for layer, stats in stats_by_layer.items():
        path = os.path.join(stats_dir, f"hope_layer{layer}.json")
        stats.save(path)
        stat_files[str(layer)] = path

    base_state = {
        layer: dict(values) for layer, values in reap_state.items()
    }

    def _observer_data() -> dict[int, dict[str, Any]]:
        return {layer: dict(values) for layer, values in base_state.items()}

    slices: dict[str, Any] = {}
    slice_models = {}
    for label in ("reap", "hope"):
        slice_models[label] = copy.deepcopy(model)
    del model

    reap_dir = os.path.join(models_dir, "reap_pruned")
    if os.path.exists(reap_dir):
        shutil.rmtree(reap_dir)
    slice_with_reap(
        slice_models["reap"],
        _observer_data(),
        prune_method="reap",
        n_experts_to_prune=budget,
        out_dir=reap_dir,
    )
    slices["reap"] = {"saved_dir": reap_dir}

    hope_dir = os.path.join(models_dir, "hope_pruned")
    if os.path.exists(hope_dir):
        shutil.rmtree(hope_dir)
    hope_data = _observer_data()
    for layer, stats in stats_by_layer.items():
        hope_data[layer]["hope_exact"] = encode_exact_prune_set(
            n_experts, sets[layer]["hope"]["pruned"]
        )
    slice_with_reap(
        slice_models["hope"],
        hope_data,
        prune_method="hope_exact",
        n_experts_to_prune=budget,
        out_dir=hope_dir,
    )
    slices["hope"] = {
        "saved_dir": hope_dir,
        "pruned_sets": {str(l): sets[l]["hope"]["pruned"] for l in sets},
    }

    expect_retained = n_experts - budget
    reload_reports = {}
    for label, entry in slices.items():
        verify = reload_and_verify(
            entry["saved_dir"], expect_experts=expect_retained, seq_len=seq_len
        )
        verify["size_within_tiny_bounds"] = (
            100_000 <= verify["total_bytes"] <= 512_000_000
        )
        verify["checks"]["size_within_tiny_bounds"] = verify["size_within_tiny_bounds"]
        verify["ok"] = all(verify["checks"].values())
        reload_reports[label] = verify
        del slice_models[label]

    checks = {
        "rows_match_forwards": all(
            s.total_rows == expect_rows for s in stats_by_layer.values()
        ),
        "count_safe_validation": True,
        "one_pass_both_metrics": len(stats_by_layer) == len(reap_state)
        == int(smoke_cfg["num_hidden_layers"]),
        "hope_objective_not_worse_than_reap_set": hope_not_worse,
        "exact_certificates": all(
            data["hope"]["exact"] for data in sets.values()
        ),
        "capability_partitions_populated": (
            not capabilities
            or any(s.capabilities for s in stats_by_layer.values())
        ),
        "reap_reload_ok": reload_reports["reap"]["ok"],
        "hope_reload_ok": reload_reports["hope"]["ok"],
        "tiny_local_size_only": all(
            r["size_within_tiny_bounds"] for r in reload_reports.values()
        ),
        "no_remote_download": True,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "hope-reap-tiny-integration-smoke",
        "paper": PAPER_REF,
        "evidence_class": (
            "INTEGRATION only: real Transformers checkpoint, real pinned REAP "
            "observer/slicing, real exhaustive selector. NOT agentic-quality, "
            "NOT model-quality, NOT a benchmark."
        ),
        "reap_pin": pin,
        "versions": {
            "torch": getattr(torch, "__version__", "?"),
            "transformers": getattr(transformers, "__version__", "?"),
            "python": sys.version.split()[0],
        },
        "offline_env": offline_applied,
        "telemetry": config.get("telemetry", "disabled"),
        "model": {
            "class": "MixtralForCausalLM",
            "construction": "local MixtralConfig + random init (no weights download)",
            "params_total": params_total,
            "layers": int(smoke_cfg["num_hidden_layers"]),
            "experts": n_experts,
            "hidden": int(smoke_cfg["hidden_size"]),
            "top_k": int(smoke_cfg["num_experts_per_tok"]),
        },
        "one_pass": {
            "forward_passes": batches,
            "rows_per_layer": expect_rows,
            "listener_hook": "reap.observer.update_pruning_state (actual-routing seam injects resolved ids; upstream metric code untouched; returned PreparedPruningBatch reused)",
            "capability_schedule": capability_schedule,
        },
        "routing": routing_notes,
        "stats_files": stat_files,
        "capability_rows": {
            str(layer): {
                name: sub.total_rows
                for name, sub in sorted(stats.capabilities.items())
            }
            for layer, stats in stats_by_layer.items()
        },
        "selections": {str(layer): data for layer, data in sets.items()},
        "reap_observer_state": _jsonable_reap_state(reap_state),
        "slices": slices,
        "reload": reload_reports,
        "checks": checks,
        "pass": all(checks.values()),
        "caveats": [
            "paper §3.1 assumes softmax top-K routing; sigmoid routers with a "
            "correction bias differ — MiMo's verified observation records this "
            "caveat per layer (see routing notes) and every other bias router "
            "still fails closed at attach_model — see docs/hope-objective.md §4",
            "MiMo-V2.6-Flash is now registered by this adapter from verified "
            "inventory/modeling metadata (MODEL_ATTRS + observer hook config + "
            "sigmoid/noaux_tc seam spec + router-logits recompute seam); every "
            "other correction-bias router still fails closed at attach_model "
            "(explicit error, never an approximation) and the real full-model "
            "observation run remains tracked in issue #21",
            "toy/tiny fixtures must never be reported as agent-quality evidence",
        ],
    }
    report_path = os.path.join(out_dir, "report.json")
    tmp = f"{report_path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=1, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, report_path)
    report["report_path"] = report_path
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.pruning.reap_adapter",
        description=(
            "Pinned-REAP one-pass HOPE integration (tiny Mixtral smoke). "
            "Local files only; offline and telemetry flags come from the config."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser(
        "tiny-mixtral-smoke",
        help="local tiny Mixtral: observer -> one-pass selection -> REAP slicing -> reload",
    )
    smoke.add_argument("--config", required=True, help="hope-config JSON")
    smoke.add_argument("--out", required=True, help="output directory (local)")
    smoke.set_defaults(func=_cmd_tiny_smoke)
    return parser


def _cmd_tiny_smoke(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = run_tiny_mixtral_smoke(config, args.out)
    print(json.dumps(report, sort_keys=True, indent=1, allow_nan=False))
    return 0 if report["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HopeError as exc:
        print(json.dumps({"kind": "hope-adapter-error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
