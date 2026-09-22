"""Selection comparison for expert-pruning baselines and candidates.

Compares a baseline normalized selection map (schema_version=1, produced by
``mimo_halo.baselines.tacodevs_reap25``) against a candidate selection map in
the same layer schema, plus optional capability-importance and pair evidence
files supplied explicitly on the command line.

Per layer the report states, over ORIGINAL expert IDs (packed/retained
indices are never compared):

- retained Jaccard: |retained_baseline INTERSECT retained_candidate| /
  |retained_baseline UNION retained_candidate| (1.0 means identical retained
  sets; it is a set agreement statistic, NOT a quality score);
- removed-both, baseline-only-removed and candidate-only-removed expert
  counts (asymmetric by construction), each ALSO emitted as a sorted list
  of ORIGINAL expert IDs (``removed_both_expert_ids``,
  ``baseline_only_removed_expert_ids``, ``candidate_only_removed_expert_ids``)
  so the report names exactly which experts disagree per layer, not only
  how many;
- expert counts for each side.

Global aggregates give layer patterns: Jaccard distribution histogram,
exact-agreement layers, layers where each side removed more, and identical
selection layers.

Optional inputs are reported as ``status: "unavailable"`` when absent or
without usable rows. Missing capability or pair evidence is NEVER inferred
from selection overlap and never reported as zero. Repeated layer entries in
the optional files fail closed (they would silently drop the earlier entry's
stats), as do duplicate pair/circuit identities within the same capability —
identity is (layer, kind, expert set, capability), so the same pair observed
under two different capabilities is two legitimate distinct observations.
Pair/circuit totals count supplied distinct capability observations, not
unique proven functional circuits.

The optional capability file supplies per-layer, per-capability expert
importance over original IDs; the report sums importance mass removed and
retained per side. The optional pair file supplies explicitly measured
expert pairs (and optional multi-expert circuit sets); the report labels the
key category "pair retention evidence" (candidate retained both endpoints
while baseline removed at least one) and never calls it proven functional
circuit survival.

The baseline's published precision allocation (projection format and group
histograms, sensitive projection references, projections allocated higher
bits) is echoed as a precision reference. It is PUBLISHED ALLOCATION
EVIDENCE, not measured sensitivity and not task-success evidence.

This module performs no model inference and fabricates no data: everything
in the report is derived from the supplied files or explicitly marked
unavailable.

CLI::

    PYTHONPATH=src python -m mimo_halo.baselines.compare \\
        --baseline normalized.json --candidate candidate.json \\
        --output report.json [--capabilities capabilities.json] [--pairs pairs.json]

Exit codes: 0 success, 2 validation/usage failure (fail closed).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

JACCARD_DEFINITION = (
    "retained_jaccard = |retained_baseline INTERSECT retained_candidate| / "
    "|retained_baseline UNION retained_candidate|, computed over original "
    "expert IDs per layer; 1.0 means identical retained sets. It is a set "
    "agreement statistic, not a quality or task-success measure."
)
PAIR_EVIDENCE_LABEL = (
    "pair retention evidence: candidate retained both endpoints while the "
    "baseline removed at least one endpoint. This is explicitly NOT proven "
    "functional circuit survival; it only restates supplied pair evidence "
    "against the two selections."
)
OBSERVATION_NOTE = (
    "Pair/circuit identity is (layer, kind, expert set, capability): the same "
    "expert set observed under two capabilities counts as two supplied "
    "distinct capability observations, while a duplicate within the same "
    "capability fails closed because it would double count. Totals are "
    "supplied distinct capability observations, not unique proven functional "
    "circuits."
)
PRECISION_REFERENCE_LABEL = (
    "Published precision allocation from the baseline normalization: "
    "allocation evidence for sensitivity prioritization only. It is NOT "
    "measured sensitivity, NOT produced by this lab, and NOT task-success "
    "evidence."
)
UNAVAILABLE_INFERENCE_NOTE = (
    "Missing capability/pair observations are reported as unavailable and "
    "are never inferred from selection overlap or reported as zero."
)
NO_INFERENCE_NOTE = (
    "This report contains selection-overlap statistics only; no inference "
    "functionality was executed and no quality outcome is claimed."
)

_STATUS_AVAILABLE = "available"
_STATUS_UNAVAILABLE = "unavailable"

# Jaccard histogram buckets, upper edge inclusive (bucket "1.0" is exact
# agreement, kept separate from "0.9-1.0").
_JACCARD_BUCKETS: tuple[tuple[str, float], ...] = (
    ("1.0", 1.0),
    ("0.9-1.0", 0.9),
    ("0.75-0.9", 0.75),
    ("0.5-0.75", 0.5),
    ("0.25-0.5", 0.25),
    ("0.0-0.25", 0.0),
)


class ComparisonError(Exception):
    """Fail-closed validation or IO error surfaced as exit code 2."""


# ---------------------------------------------------------------------------
# JSON loading and primitive validation
# ---------------------------------------------------------------------------


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ComparisonError(f"{label} file {path}: cannot read: {exc}") from exc

    def _reject_constant(name: str) -> Any:
        raise ValueError(f"non-finite JSON constant {name!r}")

    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ComparisonError(f"{label} file {path}: malformed JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ComparisonError(f"{label} file {path}: top level must be a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ComparisonError(
            f"{label} file {path}: schema_version must be {SCHEMA_VERSION}, "
            f"got {data.get('schema_version')!r}"
        )
    return data


def _require_object(value: Any, ctx: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ComparisonError(f"{ctx}: expected object, got {type(value).__name__}")
    return value


def _require_int(value: Any, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComparisonError(f"{ctx}: expected integer, got {value!r}")
    return value


def _require_finite_number(value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{ctx}: expected number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ComparisonError(f"{ctx}: non-finite number {value!r}")
    return number


def _require_id_list(value: Any, ctx: str, expert_count: int) -> list[int]:
    if not isinstance(value, list):
        raise ComparisonError(f"{ctx}: expected list of expert IDs, got {type(value).__name__}")
    ids: list[int] = []
    for item in value:
        expert_id = _require_int(item, f"{ctx} entry")
        if expert_id < 0 or expert_id >= expert_count:
            raise ComparisonError(
                f"{ctx}: expert ID {expert_id} out of range [0, {expert_count})"
            )
        ids.append(expert_id)
    duplicates = sorted({expert_id for expert_id in ids if ids.count(expert_id) > 1})
    if duplicates:
        raise ComparisonError(f"{ctx}: duplicate expert IDs {duplicates}")
    return ids


# ---------------------------------------------------------------------------
# Layer map parsing and universe validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerMap:
    layer: int
    original_expert_count: int
    retained: frozenset[int]
    pruned: frozenset[int]


def _parse_layer_map(entry: Any, label: str) -> LayerMap:
    entry = _require_object(entry, f"{label} layer entry")
    layer = _require_int(entry.get("layer"), f"{label} layer entry: layer")
    ctx = f"{label} layer {layer}"
    expert_count = _require_int(
        entry.get("original_expert_count"), f"{ctx}: original_expert_count"
    )
    if expert_count <= 0:
        raise ComparisonError(f"{ctx}: original_expert_count must be positive")
    retained_list = _require_id_list(
        entry.get("retained_expert_ids"), f"{ctx}: retained_expert_ids", expert_count
    )
    pruned_list = _require_id_list(
        entry.get("pruned_expert_ids"), f"{ctx}: pruned_expert_ids", expert_count
    )
    retained = frozenset(retained_list)
    pruned = frozenset(pruned_list)
    overlap = sorted(retained & pruned)
    if overlap:
        raise ComparisonError(f"{ctx}: expert IDs {overlap} appear in both retained and pruned")
    if len(retained) + len(pruned) != expert_count:
        raise ComparisonError(
            f"{ctx}: retained ({len(retained)}) + pruned ({len(pruned)}) != "
            f"original_expert_count ({expert_count}); layer universe incomplete"
        )
    return LayerMap(layer, expert_count, retained, pruned)


def _parse_layers(data: dict[str, Any], label: str) -> dict[int, LayerMap]:
    entries = data.get("layers")
    if not isinstance(entries, list) or not entries:
        raise ComparisonError(f"{label}: layers must be a non-empty list")
    layers: dict[int, LayerMap] = {}
    for entry in entries:
        layer_map = _parse_layer_map(entry, label)
        if layer_map.layer in layers:
            raise ComparisonError(f"{label}: duplicate layer index {layer_map.layer}")
        layers[layer_map.layer] = layer_map
    return layers


def _validate_universes(
    baseline: dict[int, LayerMap], candidate: dict[int, LayerMap]
) -> None:
    baseline_layers = set(baseline)
    candidate_layers = set(candidate)
    if baseline_layers != candidate_layers:
        only_baseline = sorted(baseline_layers - candidate_layers)
        only_candidate = sorted(candidate_layers - baseline_layers)
        raise ComparisonError(
            "layer universes differ between baseline and candidate: "
            f"baseline-only layers {only_baseline}, candidate-only layers {only_candidate}"
        )
    for layer in sorted(baseline_layers):
        b_count = baseline[layer].original_expert_count
        c_count = candidate[layer].original_expert_count
        if b_count != c_count:
            raise ComparisonError(
                f"layer {layer}: original_expert_count mismatch "
                f"(baseline {b_count}, candidate {c_count})"
            )


# ---------------------------------------------------------------------------
# Selection comparison
# ---------------------------------------------------------------------------


def _compare_layer(baseline_map: LayerMap, candidate_map: LayerMap) -> dict[str, Any]:
    retained_intersection = len(baseline_map.retained & candidate_map.retained)
    retained_union = len(baseline_map.retained | candidate_map.retained)
    removed_both = baseline_map.pruned & candidate_map.pruned
    baseline_only_removed = baseline_map.pruned - candidate_map.pruned
    candidate_only_removed = candidate_map.pruned - baseline_map.pruned
    return {
        "layer": baseline_map.layer,
        "original_expert_count": baseline_map.original_expert_count,
        "retained_baseline_count": len(baseline_map.retained),
        "retained_candidate_count": len(candidate_map.retained),
        "retained_intersection": retained_intersection,
        "retained_union": retained_union,
        "retained_jaccard": retained_intersection / retained_union,
        "removed_baseline": len(baseline_map.pruned),
        "removed_candidate": len(candidate_map.pruned),
        "removed_both": len(removed_both),
        "baseline_only_removed": len(baseline_only_removed),
        "candidate_only_removed": len(candidate_only_removed),
        # Sorted ORIGINAL expert IDs behind the three counts above, so the
        # report states which experts disagree, not only how many. Same
        # original-ID namespace as retained/pruned_expert_ids; packed or
        # retained-set indices never appear here.
        "removed_both_expert_ids": sorted(removed_both),
        "baseline_only_removed_expert_ids": sorted(baseline_only_removed),
        "candidate_only_removed_expert_ids": sorted(candidate_only_removed),
    }


def _bucket_jaccard(jaccard: float) -> str:
    for name, lower in _JACCARD_BUCKETS:
        if jaccard > lower or (name == "1.0" and jaccard == 1.0):
            return name
    return _JACCARD_BUCKETS[-1][0]


def _layer_patterns(per_layer: list[dict[str, Any]]) -> dict[str, Any]:
    jaccards = [entry["retained_jaccard"] for entry in per_layer]
    histogram = Counter(_bucket_jaccard(j) for j in jaccards)
    exact_match = sorted(
        entry["layer"] for entry in per_layer if entry["retained_jaccard"] == 1.0
    )
    baseline_removed_more = sorted(
        entry["layer"]
        for entry in per_layer
        if entry["removed_baseline"] > entry["removed_candidate"]
    )
    candidate_removed_more = sorted(
        entry["layer"]
        for entry in per_layer
        if entry["removed_candidate"] > entry["removed_baseline"]
    )
    identical_pruned = sorted(
        entry["layer"]
        for entry in per_layer
        if entry["baseline_only_removed"] == 0 and entry["candidate_only_removed"] == 0
    )
    return {
        "jaccard_histogram": {name: histogram.get(name, 0) for name, _ in _JACCARD_BUCKETS},
        "retained_jaccard_mean": sum(jaccards) / len(jaccards),
        "retained_jaccard_min": min(jaccards),
        "retained_jaccard_max": max(jaccards),
        "layers_with_identical_retained_set": exact_match,
        "layers_with_identical_pruned_sets": identical_pruned,
        "asymmetry": {
            "layers_baseline_removed_more": baseline_removed_more,
            "layers_candidate_removed_more": candidate_removed_more,
            "baseline_only_removed_total": sum(
                entry["baseline_only_removed"] for entry in per_layer
            ),
            "candidate_only_removed_total": sum(
                entry["candidate_only_removed"] for entry in per_layer
            ),
            "removed_both_total": sum(entry["removed_both"] for entry in per_layer),
        },
    }


# ---------------------------------------------------------------------------
# Optional capability importance comparison
# ---------------------------------------------------------------------------


def _parse_capabilities(
    data: dict[str, Any], universe: dict[int, LayerMap], label: str
) -> dict[int, dict[str, dict[int, float]]]:
    """Parse capability importance into {layer: {capability: {expert_id: importance}}}."""
    entries = data.get("layers")
    if not isinstance(entries, list) or not entries:
        raise ComparisonError(f"{label}: layers must be a non-empty list")
    parsed: dict[int, dict[str, dict[int, float]]] = {}
    for entry in entries:
        entry = _require_object(entry, f"{label} layer entry")
        layer = _require_int(entry.get("layer"), f"{label} layer entry: layer")
        if layer not in universe:
            raise ComparisonError(
                f"{label}: layer {layer} is not in the baseline/candidate layer universe"
            )
        if layer in parsed:
            raise ComparisonError(
                f"{label}: duplicate layer index {layer}; repeated layer entries "
                "would silently drop the earlier entry's stats"
            )
        capabilities = entry.get("capabilities", {})
        capabilities = _require_object(capabilities, f"{label} layer {layer}: capabilities")
        expert_count = universe[layer].original_expert_count
        layer_caps: dict[str, dict[int, float]] = {}
        for name, stats in capabilities.items():
            stats = _require_object(stats, f"{label} layer {layer} capability {name!r}")
            importance = stats.get("expert_importance", {})
            importance = _require_object(
                importance, f"{label} layer {layer} capability {name!r}: expert_importance"
            )
            if not importance:
                raise ComparisonError(
                    f"{label} layer {layer} capability {name!r}: empty expert_importance; "
                    "omit the capability or supply observed importance"
                )
            mapping: dict[int, float] = {}
            for key, value in importance.items():
                if not isinstance(key, str) or not key.lstrip("-").isdigit():
                    raise ComparisonError(
                        f"{label} layer {layer} capability {name!r}: expert ID key "
                        f"{key!r} is not an integer string"
                    )
                expert_id = int(key)
                if expert_id < 0 or expert_id >= expert_count:
                    raise ComparisonError(
                        f"{label} layer {layer} capability {name!r}: expert ID "
                        f"{expert_id} out of range [0, {expert_count})"
                    )
                mapping[expert_id] = _require_finite_number(
                    value,
                    f"{label} layer {layer} capability {name!r} expert {expert_id}",
                )
            layer_caps[name] = mapping
        parsed[layer] = layer_caps
    return parsed


def _capability_comparison(
    capabilities: dict[int, dict[str, dict[int, float]]],
    baseline: dict[int, LayerMap],
    candidate: dict[int, LayerMap],
) -> dict[str, Any]:
    names = sorted(
        {name for caps in capabilities.values() for name in caps}
    )
    per_capability: list[dict[str, Any]] = []
    for name in names:
        per_layer: list[dict[str, Any]] = []
        layers_available: list[int] = []
        layers_missing: list[int] = []
        totals = {
            "baseline_removed_importance_total": 0.0,
            "candidate_removed_importance_total": 0.0,
            "retained_importance_baseline_total": 0.0,
            "retained_importance_candidate_total": 0.0,
            "importance_observed_experts_total": 0,
        }
        for layer in sorted(baseline):
            mapping = capabilities.get(layer, {}).get(name)
            if mapping is None:
                layers_missing.append(layer)
                per_layer.append(
                    {
                        "layer": layer,
                        "status": _STATUS_UNAVAILABLE,
                        "reason": "no capability importance stats supplied for this layer",
                    }
                )
                continue
            baseline_map = baseline[layer]
            candidate_map = candidate[layer]
            all_experts = range(baseline_map.original_expert_count)
            observed = sum(1 for expert in all_experts if expert in mapping)
            baseline_removed_mass = sum(
                mapping[e] for e in baseline_map.pruned if e in mapping
            )
            candidate_removed_mass = sum(
                mapping[e] for e in candidate_map.pruned if e in mapping
            )
            baseline_retained_mass = sum(
                mapping[e] for e in baseline_map.retained if e in mapping
            )
            candidate_retained_mass = sum(
                mapping[e] for e in candidate_map.retained if e in mapping
            )
            layers_available.append(layer)
            totals["baseline_removed_importance_total"] += baseline_removed_mass
            totals["candidate_removed_importance_total"] += candidate_removed_mass
            totals["retained_importance_baseline_total"] += baseline_retained_mass
            totals["retained_importance_candidate_total"] += candidate_retained_mass
            totals["importance_observed_experts_total"] += observed
            per_layer.append(
                {
                    "layer": layer,
                    "status": _STATUS_AVAILABLE,
                    "importance_experts_observed": observed,
                    "importance_experts_expected": baseline_map.original_expert_count,
                    "baseline_removed_importance": baseline_removed_mass,
                    "candidate_removed_importance": candidate_removed_mass,
                    "delta_removed_importance_candidate_minus_baseline": (
                        candidate_removed_mass - baseline_removed_mass
                    ),
                    "retained_importance_baseline": baseline_retained_mass,
                    "retained_importance_candidate": candidate_retained_mass,
                }
            )
        per_capability.append(
            {
                "capability": name,
                "status": _STATUS_AVAILABLE if layers_available else _STATUS_UNAVAILABLE,
                "layers_available": layers_available,
                "layers_missing": layers_missing,
                "baseline_removed_importance_total": totals[
                    "baseline_removed_importance_total"
                ],
                "candidate_removed_importance_total": totals[
                    "candidate_removed_importance_total"
                ],
                "delta_removed_importance_total": (
                    totals["candidate_removed_importance_total"]
                    - totals["baseline_removed_importance_total"]
                ),
                "retained_importance_baseline_total": totals[
                    "retained_importance_baseline_total"
                ],
                "retained_importance_candidate_total": totals[
                    "retained_importance_candidate_total"
                ],
                "importance_observed_experts_total": totals[
                    "importance_observed_experts_total"
                ],
                "per_layer": per_layer,
            }
        )
    section: dict[str, Any] = {
        "status": _STATUS_AVAILABLE if names else _STATUS_UNAVAILABLE,
        "note": UNAVAILABLE_INFERENCE_NOTE,
    }
    if not names:
        section["reason"] = "capability file supplied but contains no capability entries"
        return section
    section["capabilities"] = per_capability
    return section


# ---------------------------------------------------------------------------
# Optional pair / circuit evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertSet:
    experts: tuple[int, ...]
    importance: float | None
    capability: str | None
    objective_source: str | None


def _parse_pair_file(
    data: dict[str, Any], universe: dict[int, LayerMap], label: str
) -> dict[int, dict[str, list[ExpertSet]]]:
    entries = data.get("layers")
    if not isinstance(entries, list) or not entries:
        raise ComparisonError(f"{label}: layers must be a non-empty list")
    parsed: dict[int, dict[str, list[ExpertSet]]] = {}
    for entry in entries:
        entry = _require_object(entry, f"{label} layer entry")
        layer = _require_int(entry.get("layer"), f"{label} layer entry: layer")
        if layer in parsed:
            raise ComparisonError(
                f"{label}: duplicate layer index {layer}; repeated layer entries "
                "would silently drop the earlier entry's pairs/circuits"
            )
        if layer not in universe:
            raise ComparisonError(
                f"{label}: layer {layer} is not in the baseline/candidate layer universe"
            )
        expert_count = universe[layer].original_expert_count

        def _parse_set(raw: Any, kind: str, require_importance: bool) -> ExpertSet:
            raw = _require_object(raw, f"{label} layer {layer} {kind} entry")
            experts_raw = raw.get("experts")
            if not isinstance(experts_raw, list):
                raise ComparisonError(
                    f"{label} layer {layer} {kind}: experts must be a list of expert IDs"
                )
            experts = tuple(
                _require_int(item, f"{label} layer {layer} {kind}: expert")
                for item in experts_raw
            )
            if len(experts) != len(set(experts)):
                raise ComparisonError(
                    f"{label} layer {layer} {kind}: experts must be distinct IDs"
                )
            for expert in experts:
                if expert < 0 or expert >= expert_count:
                    raise ComparisonError(
                        f"{label} layer {layer} {kind}: expert ID {expert} out of range "
                        f"[0, {expert_count})"
                    )
            if len(experts) < 2:
                raise ComparisonError(
                    f"{label} layer {layer} {kind}: needs at least 2 expert IDs"
                )
            importance: float | None = None
            if "importance" in raw:
                importance = _require_finite_number(
                    raw["importance"], f"{label} layer {layer} {kind}: importance"
                )
            elif require_importance:
                raise ComparisonError(
                    f"{label} layer {layer} {kind}: missing importance; pair evidence "
                    "must carry an observed importance number"
                )
            capability = raw.get("capability")
            if capability is not None and not isinstance(capability, str):
                raise ComparisonError(
                    f"{label} layer {layer} {kind}: capability must be a string"
                )
            objective_source = raw.get("objective_source")
            if objective_source is not None and not isinstance(objective_source, str):
                raise ComparisonError(
                    f"{label} layer {layer} {kind}: objective_source must be a string"
                )
            return ExpertSet(experts, importance, capability, objective_source)

        sets: dict[str, list[ExpertSet]] = {"pairs": [], "circuits": []}
        seen_identities: dict[str, set[tuple[tuple[int, ...], str | None]]] = {
            "pairs": set(),
            "circuits": set(),
        }
        for kind, key, require_importance in (
            ("pair", "pairs", True),
            ("circuit", "circuits", False),
        ):
            for raw in entry.get(key, []) or []:
                expert_set = _parse_set(raw, kind, require_importance=require_importance)
                # Identity: (layer, kind, expert set, capability). The same
                # pair observed under two capabilities is two distinct
                # supplied observations; a duplicate within the same
                # capability would double count evidence and fails closed.
                identity = (tuple(sorted(expert_set.experts)), expert_set.capability)
                if identity in seen_identities[key]:
                    raise ComparisonError(
                        f"{label} layer {layer}: duplicate {kind} identity "
                        f"{list(identity[0])} for capability {identity[1]!r} would "
                        "double count evidence; identity is (layer, kind, expert "
                        "set, capability)"
                    )
                seen_identities[key].add(identity)
                sets[key].append(expert_set)
        if not sets["pairs"] and not sets["circuits"]:
            raise ComparisonError(
                f"{label} layer {layer}: no pairs or circuits; omit the layer or supply "
                "explicit evidence"
            )
        parsed[layer] = sets
    return parsed


def _classify_set(
    expert_set: ExpertSet,
    baseline_map: LayerMap,
    candidate_map: LayerMap,
) -> dict[str, Any]:
    baseline_removed = sum(1 for e in expert_set.experts if e in baseline_map.pruned)
    candidate_removed = sum(1 for e in expert_set.experts if e in candidate_map.pruned)
    return {
        "experts": list(expert_set.experts),
        "importance": expert_set.importance,
        "capability": expert_set.capability,
        "objective_source": expert_set.objective_source,
        "baseline_removed_endpoints": baseline_removed,
        "candidate_removed_endpoints": candidate_removed,
        "baseline_removed_at_least_one": baseline_removed >= 1,
        "baseline_removed_all": baseline_removed == len(expert_set.experts),
        "candidate_removed_at_least_one": candidate_removed >= 1,
        "candidate_removed_all": candidate_removed == len(expert_set.experts),
        "candidate_retained_both_baseline_removed": (
            candidate_removed == 0 and baseline_removed >= 1
        ),
        "baseline_retained_both_candidate_removed": (
            baseline_removed == 0 and candidate_removed >= 1
        ),
    }


def _summarize_sets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "total": len(rows),
        "baseline_removed_at_least_one": sum(r["baseline_removed_at_least_one"] for r in rows),
        "baseline_removed_all": sum(r["baseline_removed_all"] for r in rows),
        "candidate_removed_at_least_one": sum(
            r["candidate_removed_at_least_one"] for r in rows
        ),
        "candidate_removed_all": sum(r["candidate_removed_all"] for r in rows),
        "candidate_retained_both_baseline_removed": sum(
            r["candidate_retained_both_baseline_removed"] for r in rows
        ),
        "baseline_retained_both_candidate_removed": sum(
            r["baseline_retained_both_candidate_removed"] for r in rows
        ),
    }
    importance_sums = {
        "candidate_retained_both_baseline_removed_importance": sum(
            r["importance"]
            for r in rows
            if r["candidate_retained_both_baseline_removed"] and r["importance"] is not None
        ),
        "baseline_retained_both_candidate_removed_importance": sum(
            r["importance"]
            for r in rows
            if r["baseline_retained_both_candidate_removed"] and r["importance"] is not None
        ),
    }
    by_capability: Counter[str] = Counter(
        r["capability"]
        for r in rows
        if r["candidate_retained_both_baseline_removed"] and r["capability"]
    )
    return {
        "counts": counts,
        "importance_sums": importance_sums,
        "candidate_retained_both_baseline_removed_by_capability": dict(
            sorted(by_capability.items())
        ),
    }


def _pair_evidence_section(
    parsed_pairs: dict[int, dict[str, list[ExpertSet]]],
    baseline: dict[int, LayerMap],
    candidate: dict[int, LayerMap],
) -> dict[str, Any]:
    per_layer: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    all_circuit_rows: list[dict[str, Any]] = []
    for layer in sorted(parsed_pairs):
        sets = parsed_pairs[layer]
        baseline_map = baseline[layer]
        candidate_map = candidate[layer]
        pair_rows = [
            _classify_set(expert_set, baseline_map, candidate_map)
            for expert_set in sets["pairs"]
        ]
        circuit_rows = [
            _classify_set(expert_set, baseline_map, candidate_map)
            for expert_set in sets["circuits"]
        ]
        all_pair_rows.extend(pair_rows)
        all_circuit_rows.extend(circuit_rows)
        per_layer.append(
            {
                "layer": layer,
                "pairs": _summarize_sets(pair_rows) if pair_rows else None,
                "circuits": _summarize_sets(circuit_rows) if circuit_rows else None,
            }
        )
    pairs_summary = _summarize_sets(all_pair_rows)
    circuits_summary = _summarize_sets(all_circuit_rows)
    return {
        "status": _STATUS_AVAILABLE,
        "evidence_label": PAIR_EVIDENCE_LABEL,
        "totals_note": OBSERVATION_NOTE,
        "pairs_total": pairs_summary["counts"]["total"],
        "circuits_total": circuits_summary["counts"]["total"],
        "pairs": pairs_summary,
        "circuits": circuits_summary,
        "per_layer": per_layer,
        "circuit_note": (
            "Multi-expert circuits are supported only with the explicitly supplied "
            "expert sets; interpretation is the file author's, not this tool's."
        ),
    }


# ---------------------------------------------------------------------------
# Baseline published precision reference
# ---------------------------------------------------------------------------


def _projection_format(projection: dict[str, Any]) -> str | None:
    mode = projection.get("mode")
    bits = projection.get("bits")
    group_size = projection.get("group_size")
    if mode is None and bits is None:
        return None
    return f"mode={mode} bits={bits} group_size={group_size}"


def _precision_reference(baseline_data: dict[str, Any]) -> dict[str, Any]:
    layers = baseline_data.get("layers")
    format_counts: Counter[str] = Counter()
    per_projection_type: dict[str, Counter[str]] = {}
    higher_bits: list[dict[str, Any]] = []
    low_bits_seen: list[int] = []
    for entry in layers if isinstance(layers, list) else []:
        if not isinstance(entry, dict):
            continue
        projections = entry.get("projections")
        if not isinstance(projections, dict):
            continue
        layer = entry.get("layer")
        for name, projection in sorted(projections.items()):
            if not isinstance(projection, dict):
                continue
            fmt = _projection_format(projection)
            if fmt is None:
                continue
            format_counts[fmt] += 1
            per_projection_type.setdefault(str(projection.get("component", name)), Counter())[
                fmt
            ] += 1
            bits = projection.get("bits")
            if isinstance(bits, (int, float)) and not isinstance(bits, bool):
                low_bits_seen.append(int(bits))
    section: dict[str, Any] = {
        "evidence_label": PRECISION_REFERENCE_LABEL,
        "status": _STATUS_AVAILABLE if format_counts else _STATUS_UNAVAILABLE,
    }
    if not format_counts:
        section["reason"] = (
            "baseline normalized file carries no projection format data under "
            "layers[].projections"
        )
        return section
    bit_counts = Counter(low_bits_seen)
    if bit_counts:
        modal_bits = min(
            bits
            for bits, count in bit_counts.items()
            if count == max(bit_counts.values())
        )
    else:
        modal_bits = None
    for entry in layers if isinstance(layers, list) else []:
        if not isinstance(entry, dict):
            continue
        projections = entry.get("projections")
        if not isinstance(projections, dict):
            continue
        for name, projection in sorted(projections.items()):
            if not isinstance(projection, dict):
                continue
            bits = projection.get("bits")
            if (
                modal_bits is not None
                and isinstance(bits, (int, float))
                and not isinstance(bits, bool)
                and bits > modal_bits
            ):
                higher_bits.append(
                    {
                        "layer": entry.get("layer"),
                        "projection": name,
                        "format": _projection_format(projection),
                    }
                )
    section.update(
        {
            "projection_format_histogram": dict(sorted(format_counts.items())),
            "projection_format_by_tensor_type": {
                name: dict(sorted(counter.items()))
                for name, counter in sorted(per_projection_type.items())
            },
            "modal_bits": modal_bits,
            "projections_allocated_higher_bits_than_modal": sorted(
                higher_bits, key=lambda item: (str(item["layer"]), item["projection"])
            ),
            "higher_bits_interpretation": (
                "Tensors allocated more bits than the modal expert projection are the "
                "published plan's implicit sensitivity ordering (allocation evidence); "
                "this is NOT measured sensitivity and NOT a task-success ranking."
            ),
        }
    )
    sensitive = baseline_data.get("precision_summary", {}).get(
        "sensitive_projection_references"
    )
    if isinstance(sensitive, list):
        section["published_sensitive_projection_references"] = sensitive
    return section


# ---------------------------------------------------------------------------
# Report assembly and CLI
# ---------------------------------------------------------------------------


def build_report(
    baseline_data: dict[str, Any],
    candidate_data: dict[str, Any],
    capabilities_data: dict[str, Any] | None,
    pairs_data: dict[str, Any] | None,
    input_names: dict[str, str],
) -> dict[str, Any]:
    baseline = _parse_layers(baseline_data, "baseline")
    candidate = _parse_layers(candidate_data, "candidate")
    _validate_universes(baseline, candidate)

    per_layer = [
        _compare_layer(baseline[layer], candidate[layer]) for layer in sorted(baseline)
    ]
    total_retained_baseline = sum(entry["retained_baseline_count"] for entry in per_layer)
    total_retained_candidate = sum(entry["retained_candidate_count"] for entry in per_layer)

    if capabilities_data is not None:
        capabilities = _capability_comparison(
            _parse_capabilities(capabilities_data, baseline, "capabilities"), baseline, candidate
        )
    else:
        capabilities = {
            "status": _STATUS_UNAVAILABLE,
            "reason": "no capability file supplied (--capabilities); capability importance "
            "is not inferred from selection overlap",
            "note": UNAVAILABLE_INFERENCE_NOTE,
        }
    if pairs_data is not None:
        pairs = _pair_evidence_section(
            _parse_pair_file(pairs_data, baseline, "pairs"), baseline, candidate
        )
    else:
        pairs = {
            "status": _STATUS_UNAVAILABLE,
            "reason": "no pair evidence file supplied (--pairs); pair retention evidence "
            "is not inferred from selection overlap",
            "note": UNAVAILABLE_INFERENCE_NOTE,
        }

    baseline_id = baseline_data.get("baseline_id")
    candidate_id = candidate_data.get("candidate_id") or candidate_data.get("baseline_id")

    return {
        "schema_version": SCHEMA_VERSION,
        "comparison": {
            "baseline_id": baseline_id,
            "candidate_id": candidate_id,
            "layers_compared": len(per_layer),
            "total_retained_expert_instances_baseline": total_retained_baseline,
            "total_retained_expert_instances_candidate": total_retained_candidate,
            "id_domain": (
                "original expert IDs (packed/retained indices are never "
                "compared); retained_* and removed_*_expert_ids all use this "
                "same original-ID namespace"
            ),
        },
        "definitions": {
            "retained_jaccard": JACCARD_DEFINITION,
            "pair_retention_evidence": PAIR_EVIDENCE_LABEL,
            "precision_reference": PRECISION_REFERENCE_LABEL,
            "missing_evidence": UNAVAILABLE_INFERENCE_NOTE,
            "pair_circuit_totals": OBSERVATION_NOTE,
            "no_inference": NO_INFERENCE_NOTE,
        },
        "inputs": input_names,
        "agreement": _layer_patterns(per_layer),
        "per_layer": per_layer,
        "capabilities": capabilities,
        "pairs": pairs,
        "baseline_precision_reference": _precision_reference(baseline_data),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.baselines.compare",
        description=(
            "Compare baseline and candidate expert-selection maps (schema_version=1) "
            "with optional capability-importance and pair evidence files."
        ),
    )
    parser.add_argument(
        "--baseline", required=True, help="baseline normalized selection JSON (schema_version=1)"
    )
    parser.add_argument(
        "--candidate", required=True, help="candidate selection JSON (schema_version=1)"
    )
    parser.add_argument("--output", required=True, help="path to write the comparison report JSON")
    parser.add_argument(
        "--capabilities",
        help="optional capability importance JSON: layers[].capabilities[name]."
        "expert_importance{'<originalID>': number}; repeated layer entries fail closed",
    )
    parser.add_argument(
        "--pairs",
        help="optional pair/circuit evidence JSON: layers[].pairs[{experts, importance, "
        "capability?, objective_source?}] and optional circuits[{experts, ...}]; repeated "
        "layer entries and duplicate same-capability expert-set identities fail closed",
    )
    args = parser.parse_args(argv)

    try:
        baseline_path = Path(args.baseline)
        candidate_path = Path(args.candidate)
        baseline_data = _load_json(baseline_path, "baseline")
        candidate_data = _load_json(candidate_path, "candidate")
        capabilities_data = (
            _load_json(Path(args.capabilities), "capabilities") if args.capabilities else None
        )
        pairs_data = _load_json(Path(args.pairs), "pairs") if args.pairs else None
        report = build_report(
            baseline_data,
            candidate_data,
            capabilities_data,
            pairs_data,
            {
                "baseline": baseline_path.name,
                "candidate": candidate_path.name,
                "capabilities": Path(args.capabilities).name if args.capabilities else None,
                "pairs": Path(args.pairs).name if args.pairs else None,
            },
        )
        output_path = Path(args.output)
        if output_path.parent and not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except ComparisonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    agreement = report["agreement"]
    print(
        f"wrote {args.output}: {report['comparison']['layers_compared']} layers, "
        f"mean retained Jaccard {agreement['retained_jaccard_mean']:.4f}, "
        f"capabilities {report['capabilities']['status']}, pairs {report['pairs']['status']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())