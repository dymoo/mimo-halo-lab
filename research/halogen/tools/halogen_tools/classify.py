"""Heuristic ISA motif classifier over AMDGPU disassembly.

Categories (Phase-1 contract):
  quantized_gemm, fused_dequant_gemm, moe_routing, expert_gather_scatter,
  reduction, activation, attention, kv_operation, tensor_repacking,
  speculative_verification, unclassified

Design rules:
  * evidence-first: every fired rule contributes bounded evidence pointers
    (kernel-relative byte offset + mnemonic + rule id);
  * confidence = top_score / total_fired_score, capped at 0.95 (these are
    heuristics; never claim certainty);
  * name-independent: classification uses ONLY instruction streams (HIP
    symbol names are stripped/obfuscated in the wild and would bias us);
  * deterministic tie-breaks via CATEGORY_PRIORITY.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CATEGORIES = (
    "quantized_gemm",
    "fused_dequant_gemm",
    "moe_routing",
    "expert_gather_scatter",
    "reduction",
    "activation",
    "attention",
    "kv_operation",
    "tensor_repacking",
    "speculative_verification",
    "unclassified",
)

# more specific category wins ties
CATEGORY_PRIORITY = {c: i for i, c in enumerate((
    "fused_dequant_gemm",
    "quantized_gemm",
    "attention",
    "moe_routing",
    "expert_gather_scatter",
    "speculative_verification",
    "kv_operation",
    "reduction",
    "tensor_repacking",
    "activation",
    "unclassified",
))}

_MATRIX_RE = re.compile(r"^(wmma_|v_wmma_|v_mfma_|v_dot|v_mac_)")
_PK_RE = re.compile(r"^v_pk_")
_CVT_RE = re.compile(r"^v_cvt")
_FMT_LOAD_RE = re.compile(
    r"^(buffer|flat|global)_load_(ubyte|sbyte|u8|s8|u16|s16|i16|"
    r"short|ushort|half|f16|b16|bf16)"
)
_D16_RE = re.compile(r"_d16|d16")
_DS_RE = re.compile(r"^ds_(read|write|rmw|load|store)")
_CMP_RE = re.compile(r"^v_cmp")
_ADD_RE = re.compile(r"^v_add_(nc_)?(f32|u32|i32|f16|i16|u16|i8|u8)")
_SUB_RE = re.compile(r"^v_sub")
_EXP_RE = re.compile(r"^v_exp")
_LOG_RE = re.compile(r"^v_log")
_RCP_RE = re.compile(r"^v_rcp|v_div_(fix|f32)|v_rsq")
_LSH_RE = re.compile(r"^v_(lshl|ashr|lshr|and|or|or3|bfe|bfi|alignbyte|perm)")
_BRANCH_RE = re.compile(r"^s_cbranch|^s_branch")
_LOOP_RE = re.compile(r"^s_(add|addc|sub)_")
_SLOAD_RE = re.compile(r"^s_load")
_LOAD_RE = re.compile(r"^(buffer|flat|global)_load|^(buffer|flat|global)_atomic")
_STORE_RE = re.compile(r"^(buffer|flat|global)_store")
_UMAD_RE = re.compile(r"^v_(mad|mul)_(u32|u16|i32|i16)")
_EXTRACT_RE = re.compile(r"^v_(bfe|bfi|and|or|or3|lshl|ashr|lshr|alignbyte|perm)")


@dataclass
class EvItem:
    offset: int | None
    mnemonic: str | None
    detail: str


@dataclass
class Classification:
    likely_operation: str | None
    confidence: float | None
    evidence: list = field(default_factory=list)     # [EvItem]
    fired_rules: list = field(default_factory=list)  # [(rule_id, category, score)]
    error: str | None = None


def _sample(insns, pred, rule_id, limit=4):
    out = []
    for ins in insns:
        if pred(ins):
            out.append(EvItem(ins.offset, ins.mnemonic, f"rule:{rule_id}"))
            if len(out) >= limit:
                break
    return out


def classify(insns) -> Classification:
    """Classify a kernel from its instruction list (possibly empty)."""
    if insns is None:
        return Classification(
            None, None,
            error="disassembly unavailable; ISA motif classification impossible",
        )

    total = len(insns)
    mnems = [i.mnemonic for i in insns]
    counts = {}
    for m in mnems:
        counts[m] = counts.get(m, 0) + 1

    def count_re(rx) -> int:
        return sum(1 for m in mnems if rx.match(m))

    matrix = count_re(_MATRIX_RE)
    pk = count_re(_PK_RE)
    cvt = count_re(_CVT_RE)
    fmt_loads = count_re(_FMT_LOAD_RE)
    d16 = sum(1 for m in mnems if _D16_RE.search(m))
    ds = count_re(_DS_RE)
    ds_read = sum(1 for m in mnems if m.startswith(("ds_read", "ds_load")))
    ds_write = sum(1 for m in mnems if m.startswith(("ds_write", "ds_store")))
    cmp = count_re(_CMP_RE)
    adds = count_re(_ADD_RE)
    subs = count_re(_SUB_RE)
    exps = count_re(_EXP_RE)
    logs = count_re(_LOG_RE)
    rcps = count_re(_RCP_RE)
    sloads = count_re(_SLOAD_RE)
    loads = count_re(_LOAD_RE)
    stores = count_re(_STORE_RE)
    branches = count_re(_BRANCH_RE)
    loop_ctl = count_re(_LOOP_RE)
    umad = count_re(_UMAD_RE)
    # compute density: conversion ops are a REPATTERNING signal, not math;
    # they get their own tensor_repacking rules and must not gate them out
    math = matrix + pk + exps + logs + rcps + adds + subs + umad

    # indexed gather/scatter: load/store preceded (3-insn window) by index math
    indexed_loads = _indexed_window(insns, _LOAD_RE)
    indexed_stores = _indexed_window(insns, _STORE_RE)
    masked_stores = _cmp_guarded(insns, _STORE_RE)
    math_density = math / max(total, 1)

    fired: list[tuple[str, str, float]] = []
    evidence: list[EvItem] = []

    def fire(rule_id: str, category: str, score: float, ev_preds=()):
        fired.append((rule_id, category, score))
        for pred in ev_preds:
            if isinstance(pred, list):  # pre-collected EvItems
                evidence.extend(pred)
            else:
                evidence.extend(_sample(insns, pred, rule_id))

    # ---- rule set -----------------------------------------------------
    if matrix:
        fire("matrix_ops", "quantized_gemm", 0.40, [lambda i: _MATRIX_RE.match(i.mnemonic)])
    if matrix and (pk or umad >= 2):
        fire("matrix_with_pack", "quantized_gemm", 0.10,
             [lambda i: _PK_RE.match(i.mnemonic)])
    if matrix and fmt_loads >= 2:
        fire("quant_load_matrix", "quantized_gemm", 0.25,
             [lambda i: _FMT_LOAD_RE.match(i.mnemonic)])
    if matrix and (pk >= 3 or (pk and cvt >= 3)) and fmt_loads >= 1:
        fire("dequant_pack_chain", "fused_dequant_gemm", 0.60,
             [lambda i: _PK_RE.match(i.mnemonic) or _CVT_RE.match(i.mnemonic)])
    if matrix and (pk or cvt) and _bitunpack(insns):
        fire("bit_unpack_chain", "fused_dequant_gemm", 0.35,
             [lambda i: _LSH_RE.match(i.mnemonic)])
    if matrix and exps and (rcps or adds >= 2) and ds_read:
        fire("matmul_softmax", "attention", 1.10,
             [lambda i: _EXP_RE.match(i.mnemonic), lambda i: _RCP_RE.match(i.mnemonic)])
    elif matrix and exps >= 2 and rcps:
        fire("softmax_core", "attention", 0.40,
             [lambda i: _EXP_RE.match(i.mnemonic)])
    if indexed_loads >= 4:
        fire("indexed_load_fanout", "expert_gather_scatter", 0.50,
             [_indexed_evidence(insns, _LOAD_RE, "indexed_load_fanout")])
    if indexed_stores >= 4:
        fire("indexed_store_fanout", "expert_gather_scatter", 0.45,
             [_indexed_evidence(insns, _STORE_RE, "indexed_store_fanout")])
    if (indexed_loads >= 2 or indexed_stores >= 2) and umad + count_re(
        re.compile(r"^v_(lshl|ashr|lshr)")
    ) >= 3:
        fire("gather_index_math", "expert_gather_scatter", 0.20)
    if sloads >= 2 and (cmp >= 2 or masked_stores >= 1):
        fire("kernarg_config_dispatch", "moe_routing", 0.35,
             [lambda i: _SLOAD_RE.match(i.mnemonic)])
    if masked_stores >= 1 and (cmp >= 2 or indexed_stores >= 1):
        fire("predicate_masked_stores", "moe_routing", 0.35,
             [lambda i: _CMP_RE.match(i.mnemonic)])
    if sloads >= 2 and indexed_loads >= 2 and cmp == 0:
        fire("expert_table_indexing", "moe_routing", 0.25,
             [lambda i: _SLOAD_RE.match(i.mnemonic)])
    if adds >= 4 and (ds_read >= 1 or cmp >= 1 or count_re(re.compile(
        r"^v_read(first)?lane"
    )) >= 1):
        fire("add_reduction_chain", "reduction", 0.60,
             [lambda i: _ADD_RE.match(i.mnemonic)])
    elif adds >= 6 and math_density > 0.15:
        fire("add_chain", "reduction", 0.35,
             [lambda i: _ADD_RE.match(i.mnemonic)])
    if exps >= 2 or (exps and rcps) or (exps and logs):
        fire("exp_log_rcp", "activation", 0.55,
             [lambda i: _EXP_RE.match(i.mnemonic) or _RCP_RE.match(i.mnemonic)
              or _LOG_RE.match(i.mnemonic)])
    if exps and subs == 0 and adds >= 1:
        fire("elementwise_act", "activation", 0.20,
             [lambda i: _EXP_RE.match(i.mnemonic)])
    if fmt_loads + d16 >= 4 and stores >= 3 and math_density < 0.10:
        fire("typed_load_store_shuffle", "kv_operation", 0.45,
             [lambda i: _FMT_LOAD_RE.match(i.mnemonic)])
    if loop_ctl >= 1 and branches >= 1 and math_density < 0.10 and loads >= 2:
        fire("ring_loop_control", "kv_operation", 0.30,
             [lambda i: _BRANCH_RE.match(i.mnemonic)])
    if cvt >= 6 and math_density < 0.25 and ds >= 2:
        fire("cvt_repack_via_lds", "tensor_repacking", 0.55,
             [lambda i: _CVT_RE.match(i.mnemonic)])
    elif cvt >= 6 and math_density < 0.25:
        fire("cvt_dense", "tensor_repacking", 0.40,
             [lambda i: _CVT_RE.match(i.mnemonic)])
    # pure dequant: nibble/byte extract chains + converts + typed stores and
    # NO matrix op (with a matrix op the same stream is fused_dequant_gemm)
    extract_ops = sum(1 for m in mnems if _EXTRACT_RE.match(m))
    if not matrix and extract_ops >= 3 and cvt >= 2 and stores >= 1 \
            and (fmt_loads >= 1 or d16 >= 1):
        fire("dequant_extract_repack", "tensor_repacking", 0.65,
             [lambda i: _EXTRACT_RE.match(i.mnemonic),
              lambda i: _CVT_RE.match(i.mnemonic),
              lambda i: _STORE_RE.match(i.mnemonic)])
    if pk + umad >= 2 and ds >= 2 and not matrix:
        fire("pack_shuffle_lds", "tensor_repacking", 0.30,
             [lambda i: _PK_RE.match(i.mnemonic)])
    if cmp >= 2 and subs >= 1 and branches >= 1 and not matrix:
        fire("compare_sub_branch", "speculative_verification", 0.60,
             [lambda i: _CMP_RE.match(i.mnemonic), lambda i: _SUB_RE.match(i.mnemonic)])
    if cmp >= 2 and branches >= 1 and loads >= 2 and math_density < 0.12:
        fire("token_id_predication", "speculative_verification", 0.35,
             [lambda i: _CMP_RE.match(i.mnemonic)])

    # ---- score aggregation --------------------------------------------
    scores: dict[str, float] = {}
    for rule_id, cat, score in fired:
        scores[cat] = scores.get(cat, 0.0) + score
    if not scores:
        if total == 0:
            return Classification(
                "unclassified", 0.0,
                fired_rules=[],
                error="empty disassembly (no instructions recovered for symbol)",
            )
        return Classification("unclassified", 0.0, fired_rules=[])

    total_score = sum(scores.values())
    best = max(
        scores,
        key=lambda c: (scores[c], -CATEGORY_PRIORITY[c]),
    )
    best_score = scores[best]
    confidence = min(0.95, round(best_score / total_score, 3))

    # evidence: rule samples for the winning category + top overall samples
    win_evidence = [
        item for item in evidence
        if any(r[1] == best and f"rule:{r[0]}" == item.detail for r in fired)
    ]
    if not win_evidence:
        win_evidence = evidence[:3]
    fired_out = [
        {"rule": r, "category": c, "score": round(s, 3)}
        for (r, c, s) in sorted(fired, key=lambda x: -x[2])
    ]
    return Classification(best, confidence, evidence=win_evidence[:8],
                          fired_rules=fired_out)


def _indexed_window(insns, rx) -> int:
    """Count rx-matching instructions with index math in the 3-insns before."""
    n = 0
    for idx, ins in enumerate(insns):
        if not rx.match(ins.mnemonic):
            continue
        window = insns[max(0, idx - 3) : idx]
        if any(_UMAD_RE.match(w.mnemonic) or _LSH_RE.match(w.mnemonic)
               or re.match(r"^v_mul_(lo|u32|i32)", w.mnemonic)
               for w in window):
            n += 1
    return n


def _indexed_evidence(insns, rx, rule_id: str, limit: int = 4):
    """Evidence samples: rx ops backed by index math in the preceding window."""
    out = []
    for idx, ins in enumerate(insns):
        if not rx.match(ins.mnemonic):
            continue
        window = insns[max(0, idx - 3) : idx]
        if any(_UMAD_RE.match(w.mnemonic) or _LSH_RE.match(w.mnemonic)
               or re.match(r"^v_mul_(lo|u32|i32)", w.mnemonic)
               for w in window):
            out.append(EvItem(ins.offset, ins.mnemonic, f"rule:{rule_id}"))
            if len(out) >= limit:
                break
    return out


def _cmp_guarded(insns, rx) -> int:
    """Stores within 3 insns after a v_cmp (predicate-masked write)."""
    n = 0
    last_cmp = -10
    for idx, ins in enumerate(insns):
        if _CMP_RE.match(ins.mnemonic):
            last_cmp = idx
        elif rx.match(ins.mnemonic) and idx - last_cmp in (1, 2, 3):
            n += 1
    return n


def _bitunpack(insns) -> bool:
    """AND+shift+OR style unpack chains (int packing -> float pieces)."""
    seq = [i.mnemonic for i in insns]
    window_has = False
    for i in range(len(seq) - 2):
        trio = seq[i : i + 3]
        if (_LSH_RE.match(trio[0]) or _LSH_RE.match(trio[1])) and any(
            _CVT_RE.match(t) or _PK_RE.match(t) for t in trio
        ):
            window_has = True
            break
    return window_has
