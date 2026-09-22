"""Weighted calibration distribution over normalized episode records.

Folds per-episode weights plus the contract capability anchors into a
per-partition / per-capability sampling distribution that targets the mix in
``configs/dataset.json`` (partition weights) and
``identity._CAPABILITY_WEIGHTS`` (capability anchors, whose contract comment
says they are "applied at aggregation time" — this module is that
aggregation site).

Contracts consumed, never reimplemented:

- ``events.episode_weight`` -> each record's ``weight`` field already carries
  the extraction-side semantics (bounded decision episodes outrank giant tool
  spam: spam-ratio base discount, decision/verification bonuses, floor 0.05).
- ``configs/dataset.json`` -> ``partition.weights`` (partition target mix) and
  ``episode.downweight.spam_weight`` (the configured contribution of one
  downweighted event: 0.25 of a full event).
- ``identity._CAPABILITY_WEIGHTS`` -> capability target mix (sums to 1.0;
  the ``events.episode_weight`` docstring anchors "15% exploration, 15%
  planning, 20% implementation, 15% debugging, 15% compiler/tests, 10%
  shell/git/tools, 10% late recovery" are consumed as that one dict).
- ``schemas/episode.schema.json`` (schema_version 1.0.0) -> the closed 22-tag
  capability taxonomy; a label outside it fails closed.
- docs/datasets.md "golden never participates in calibration/quant search" ->
  golden-partition records are excluded from the fold and reported.

Downweighted episodes, per the existing weight semantics plus the dataset
config's downweight factor::

    mass(e) = weight(e) * (1 - (1 - spam_weight) * downweighted_events / event_count)

``weight`` is consumed exactly as extraction produced it (it already applies
the spam-ratio base discount and the decision/verification bonuses); on top
of that, each of the episode's ``downweighted_events`` out of its
``event_seq`` events counts as ``spam_weight`` instead of 1.0 — the dataset
config's definition of a downweighted event's contribution, applied here at
aggregation time.  An episode with no downweighted events contributes its
weight unchanged.

Multi-label episodes split their mass evenly across ``capabilities`` so total
mass is conserved.  For every (partition, capability) cell the artifact
reports the target share vs realized share and the post-stratification
sampling weight ``target / realized``: ``0.0`` when the target is 0,
``null`` plus ``unreachable: true`` when a positive target has no realized
mass — a sampling gap is reported, never a fabricated number.

CLI::

    PYTHONPATH=src python3 -m mimo_halo.traces.calibration \
        --episodes episodes.json [--config configs/dataset.json] \
        [--output dist.json]

Input records: schema episode rows (``capabilities``, ``weight``,
``event_seq``, ``downweighted_events``) plus a ``partition`` field naming one
of the six contract partitions — the caller joins the partition from the
partition assignments.  The input file may be a JSON array, an
``{"episodes": [...]}`` document, or JSONL.  Exit codes: 0 success,
2 validation failure (fail closed), mirroring the traces CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .cli import CONFIG_PATH, SCHEMA_PATH, load_config
from .common import TraceError, atomic_write_json, stable_hash
from .identity import _CAPABILITY_WEIGHTS
from .partition import PARTITIONS

SCHEMA_VERSION = "1.0.0"
KIND = "mimo-halo-calibration-distribution"


def _load_taxonomy() -> list[str]:
    """Closed 22-tag capability enum from the episode schema (consumer contract)."""
    if not SCHEMA_PATH.is_file():
        raise TraceError(f"missing episode schema: {SCHEMA_PATH}")
    try:
        doc = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraceError(f"episode schema unreadable: {exc}") from exc
    try:
        props = doc["properties"]
        version = props["schema_version"]["const"]
        enum = props["episodes"]["items"]["properties"]["capabilities"]["items"]["enum"]
    except (KeyError, TypeError) as exc:
        raise TraceError(
            "episode schema: capabilities enum not found at "
            "properties.episodes.items.properties.capabilities.items.enum"
        ) from exc
    if version != SCHEMA_VERSION:
        raise TraceError(f"episode schema_version {version!r} != {SCHEMA_VERSION!r}")
    if not isinstance(enum, list) or not enum or not all(
        isinstance(tag, str) and tag for tag in enum
    ):
        raise TraceError("episode schema: capabilities enum is not a list of tags")
    return list(enum)


def _partition_weights(config: Any) -> dict[str, float]:
    try:
        weights = config["partition"]["weights"]
    except (KeyError, TypeError) as exc:
        raise TraceError("config: missing partition.weights") from exc
    if (
        not isinstance(weights, dict)
        or sorted(weights) != sorted(PARTITIONS)
        or any(
            isinstance(weights[p], bool)
            or not isinstance(weights[p], (int, float))
            or not math.isfinite(weights[p])
            for p in PARTITIONS
        )
        or abs(sum(weights[p] for p in PARTITIONS) - 1.0) > 1e-9
    ):
        raise TraceError(
            "partition weights must cover exactly the six partitions and sum to 1"
        )
    return {p: float(weights[p]) for p in PARTITIONS}


def _spam_weight(config: Any) -> float:
    try:
        spam = config["episode"]["downweight"]["spam_weight"]
    except (KeyError, TypeError) as exc:
        raise TraceError("config: missing episode.downweight.spam_weight") from exc
    if (
        isinstance(spam, bool)
        or not isinstance(spam, (int, float))
        or not math.isfinite(spam)
        or not 0.0 <= spam <= 1.0
    ):
        raise TraceError(
            f"episode.downweight.spam_weight must be a number in [0, 1], got {spam!r}"
        )
    return float(spam)


def load_episode_records(path: Path) -> list[dict]:
    """Read episode records from a JSON array, ``{"episodes": [...]}``, or JSONL."""
    path = Path(path)
    if not path.is_file():
        raise TraceError(f"missing episodes file: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise TraceError(f"{path}: empty episodes file")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = None
    if isinstance(doc, list):
        records = doc
    elif isinstance(doc, dict) and isinstance(doc.get("episodes"), list):
        records = doc["episodes"]
    elif doc is not None:
        raise TraceError(
            f"{path}: expected a JSON array of episode records or an "
            '{"episodes": [...]} document'
        )
    else:  # JSONL fallback: one record per line
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            records.append(record)
    if not records:
        raise TraceError(f"{path}: no episode records")
    if not all(isinstance(record, dict) for record in records):
        bad = next(i for i, r in enumerate(records) if not isinstance(r, dict))
        raise TraceError(f"{path}: record {bad} is not a JSON object")
    return records


def _coerce_record(
    record: dict, index: int, taxonomy: list[str]
) -> tuple[str, tuple[str, ...], float, int, int | None]:
    """Validate one episode row against the closed contracts; fail closed."""
    caps_raw = record.get("capabilities")
    if not isinstance(caps_raw, list) or not caps_raw:
        raise TraceError(f"record {index}: capabilities must be a non-empty list")
    caps: list[str] = []
    for tag in caps_raw:
        if not isinstance(tag, str) or tag not in taxonomy:
            raise TraceError(
                f"record {index}: unknown capability {tag!r} (closed "
                f"{len(taxonomy)}-tag taxonomy; labels outside "
                "schemas/episode.schema.json fail closed)"
            )
        caps.append(tag)
    if len(set(caps)) != len(caps):
        raise TraceError(f"record {index}: duplicate capability label in {caps}")
    primary = record.get("primary_capability")
    if primary is not None and (not isinstance(primary, str) or primary not in taxonomy):
        raise TraceError(
            f"record {index}: unknown primary_capability {primary!r} (closed "
            f"{len(taxonomy)}-tag taxonomy)"
        )
    weight = record.get("weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0.0
    ):
        raise TraceError(
            f"record {index}: weight must be a finite number >= 0, got {weight!r}"
        )
    partition = record.get("partition")
    if not isinstance(partition, str) or partition not in PARTITIONS:
        raise TraceError(
            f"record {index}: partition must be one of {', '.join(PARTITIONS)}, "
            f"got {partition!r}"
        )
    downweighted = record.get("downweighted_events", 0)
    if (
        isinstance(downweighted, bool)
        or not isinstance(downweighted, int)
        or downweighted < 0
    ):
        raise TraceError(
            f"record {index}: downweighted_events must be an int >= 0, "
            f"got {downweighted!r}"
        )
    seq = record.get("event_seq")
    event_count = len(seq) if isinstance(seq, list) else None
    if downweighted:
        if not isinstance(seq, list) or not seq:
            raise TraceError(
                f"record {index}: downweighted_events={downweighted} requires "
                "a non-empty event_seq"
            )
        if downweighted > len(seq):
            raise TraceError(
                f"record {index}: downweighted_events {downweighted} exceeds "
                f"event count {len(seq)}"
            )
    return partition, tuple(caps), float(weight), downweighted, event_count


def build_distribution(records: list[dict], config: Any) -> dict:
    """Fold episode weights + capability anchors into the calibration distribution.

    Deterministic given inputs: cells accumulate in record order, rows emit in
    contract order (PARTITIONS, then schema taxonomy order).
    """
    taxonomy = _load_taxonomy()
    missing_anchors = [tag for tag in _CAPABILITY_WEIGHTS if tag not in taxonomy]
    if missing_anchors:
        raise TraceError(
            f"capability anchors missing from episode taxonomy: {missing_anchors}"
        )
    anchors = {tag: float(_CAPABILITY_WEIGHTS.get(tag, 0.0)) for tag in taxonomy}
    partition_weights = _partition_weights(config)
    spam_weight = _spam_weight(config)
    if not records:
        raise TraceError("input contains no episode records")

    # Golden never participates in calibration: the five eligible partition
    # targets renormalize over everything except golden's reservation.
    eligible = 1.0 - partition_weights["golden"]
    if eligible <= 0.0:
        raise TraceError(
            "partition weights leave no calibration-eligible mass "
            "(golden reservation covers everything)"
        )

    cells: dict[tuple[str, str], dict[str, float]] = {}
    included = excluded = 0
    included_mass = excluded_mass = 0.0
    for index, record in enumerate(records):
        partition, caps, weight, downweighted, event_count = _coerce_record(
            record, index, taxonomy
        )
        if downweighted:
            factor = 1.0 - (1.0 - spam_weight) * (downweighted / event_count)
        else:
            factor = 1.0
        mass = weight * factor
        if not math.isfinite(mass):
            raise TraceError(f"record {index}: non-finite mass")
        if partition == "golden":
            excluded += 1
            excluded_mass += mass
            continue
        included += 1
        included_mass += mass
        cell_mass = mass / len(caps)  # multi-label split; mass conserved
        for tag in caps:
            cell = cells.get((partition, tag))
            if cell is None:
                cells[(partition, tag)] = {"episodes": 1, "mass": cell_mass}
            else:
                cell["episodes"] += 1
                cell["mass"] += cell_mass
    if included == 0:
        raise TraceError(
            "no calibration-eligible episodes after golden exclusion "
            "(docs/datasets.md: golden never participates in calibration)"
        )
    if included_mass <= 0.0:
        raise TraceError(
            "included episode mass is zero; cannot build a sampling distribution"
        )

    partition_targets = {
        p: (0.0 if p == "golden" else partition_weights[p] / eligible)
        for p in PARTITIONS
    }

    groups: list[dict[str, Any]] = []
    unreachable = 0
    for partition in PARTITIONS:
        if partition == "golden":
            continue  # excluded from the fold; target stays 0 in `targets`
        for tag in taxonomy:
            cell = cells.get((partition, tag))
            target = partition_targets[partition] * anchors[tag]
            if cell is None and target <= 0.0:
                continue
            episodes = int(cell["episodes"]) if cell else 0
            mass = float(cell["mass"]) if cell else 0.0
            realized = mass / included_mass
            if realized > 0.0:
                sampling_weight: float | None = target / realized
                is_unreachable = False
            elif target > 0.0:
                sampling_weight = None
                is_unreachable = True
                unreachable += 1
            else:
                sampling_weight = 0.0
                is_unreachable = False
            row: dict[str, Any] = {
                "partition": partition,
                "capability": tag,
                "episodes": episodes,
                "realized_mass": mass,
                "realized_share": realized,
                "target_share": target,
                "sampling_weight": sampling_weight,
            }
            if is_unreachable:
                row["unreachable"] = True
            groups.append(row)

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "inputs": {
            "digest": stable_hash(
                {
                    "records": records,
                    "partition_weights": partition_weights,
                    "spam_weight": spam_weight,
                    "capability_targets": anchors,
                }
            ),
            "records": len(records),
            "config": {
                "partition_weights": partition_weights,
                "spam_weight": spam_weight,
            },
        },
        "targets": {"partitions": partition_targets, "capabilities": anchors},
        "groups": groups,
        "totals": {
            "records_included": included,
            "records_excluded_golden": excluded,
            "included_mass": included_mass,
            "excluded_golden_mass": excluded_mass,
            "unreachable_targets": unreachable,
        },
        "definitions": {
            "episode_mass": (
                "mass(e) = weight(e) * (1 - (1 - spam_weight) * "
                "downweighted_events / event_count): record weight carries the "
                "existing episode_weight semantics consumed as-is "
                "(events.episode_weight: bounded decision episodes outrank "
                "giant tool spam; spam-ratio base discount + decision bonus + "
                "verification bonus, floor 0.05); spam_weight is the dataset "
                "config's episode.downweight.spam_weight, the configured "
                "contribution of a downweighted event (0.25 of a full event), "
                "applied at aggregation time; no downweighted events -> mass "
                "equals weight"
            ),
            "capability_split": (
                "each episode's mass is split evenly across its multi-label "
                "capabilities (total mass conserved)"
            ),
            "target_share": (
                "target(p, c) = partition_weights[p] / (1 - "
                "partition_weights['golden']) * capability_anchor[c]; "
                "golden's target is 0 by contract (docs/datasets.md: golden "
                "never participates in calibration/quant search) and the five "
                "eligible partition targets renormalize; capability anchors "
                "are identity._CAPABILITY_WEIGHTS, the events.episode_weight "
                "docstring anchors (15% exploration, 15% planning, 20% "
                "implementation, 15% debugging, 15% compiler/tests, 10% "
                "shell/git/tools, 10% late recovery) applied downstream per "
                "capability"
            ),
            "sampling_weight": (
                "sampling_weight = target_share / realized_share "
                "(post-stratification ratio: sampling with probability "
                "proportional to weight * sampling_weight draws the target "
                "mix); 0.0 when target_share is 0; null with unreachable=true "
                "when a positive target has no realized mass (a gap is "
                "reported, never a fabricated number)"
            ),
            "methodology": (
                "docs/evaluation-methodology.md section 4, level-3 row: "
                "| 3. Distribution | Routing and calibration distribution | "
                "Routing top8 overlap, gate correlation, expert frequency, "
                "Jaccard/convergence across token budgets, heldout "
                "perplexity. | — levels 1-3 are distribution controls "
                "('levels 1-3 are distribution controls, levels 4-7 decide'; "
                "section 5: 'token agreement never becomes task success'), "
                "never a quality or task-success claim"
            ),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m mimo_halo.traces.calibration",
        description=(
            "fold per-episode weights + capability anchors into a "
            "per-partition / per-capability calibration sampling distribution"
        ),
    )
    parser.add_argument(
        "--episodes",
        required=True,
        help=(
            "episode records (JSON array, {\"episodes\": [...]}, or JSONL); "
            "each row carries partition + capabilities + weight"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="dataset config with partition weights + spam_weight",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="write the distribution JSON here (also printed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        config = load_config(Path(args.config))
        records = load_episode_records(Path(args.episodes))
        doc = build_distribution(records, config)
        if args.output:
            atomic_write_json(Path(args.output), doc, indent=1)
        print(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False))
        return 0
    except TraceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
