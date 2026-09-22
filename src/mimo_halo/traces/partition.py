"""Deterministic grouped split into the six contract partitions.

- Split is BY TASK GROUP: all sessions/episodes of a task land in one partition
  (no group leakage across partitions).
- Assignment is a stable hash of (salt, group_id); re-running produces the same
  result and never reassigns groups already present in a frozen assignment file.
- Golden is a HASH RESERVATION: the golden slice (10% by DEFAULT_WEIGHTS) is
  reserved independent of oracle eligibility. A reserved group always keeps the
  golden assignment and is reported not_ready when ineligible -- it is never
  redistributed into the training partitions. Only reserved+eligible groups
  freeze into the executable golden bank; a training group never moves into
  golden when an oracle appears later.
- Frozen assignments are immutable. The single exception is the audited
  pre-calibration migration (guard_pre_calibration_migration +
  regenerated_reservations): it requires explicit authorization, refuses while
  frozen golden tasks or golden-consumption evidence exist, and records file
  backups, digests and an audit reason.
"""

from __future__ import annotations

import hashlib

from .common import TraceError

PARTITIONS = ("pruning", "quant", "recovery", "validation", "golden", "torture")

DEFAULT_WEIGHTS = {
    "pruning": 0.35, "quant": 0.20, "recovery": 0.20,
    "validation": 0.10, "golden": 0.10, "torture": 0.05,
}


def _unit_hash(salt: str, group_id: str) -> float:
    digest = hashlib.sha256(f"{salt}|{group_id}".encode()).hexdigest()
    return int(digest[:16], 16) / float(16 ** 16)


def assign_partition(salt: str, group_id: str, weights: dict[str, float] | None = None) -> str:
    weights = weights or DEFAULT_WEIGHTS
    if sorted(weights) != sorted(PARTITIONS) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise TraceError("partition weights must cover exactly the six partitions and sum to 1")
    u = _unit_hash(salt, group_id)
    acc = 0.0
    for name in PARTITIONS:
        acc += weights[name]
        if u < acc:
            return name
    return PARTITIONS[-1]


def partition_tasks(
    groups: list[dict],
    salt: str,
    frozen_assignments: dict[str, str] | None = None,
    golden_eligible_ids: set[str] | None = None,
) -> dict:
    """Assign each task group deterministically.

    frozen_assignments maps group_id -> partition from a previous freeze; those
    assignments are immutable. golden_eligible_ids only marks readiness of
    reserved groups: a reserved group missing from it keeps the golden
    assignment and is reported in reserved_not_ready -- never relabeled, never
    redistributed into training. Eligibility can never pull a non-reserved
    group into golden.
    """
    frozen = dict(frozen_assignments or {})
    golden_ok = golden_eligible_ids or set()
    assignments: dict[str, str] = {}
    reserved_not_ready: list[dict] = []

    for group in sorted(groups, key=lambda g: g["group_id"]):
        gid = group["group_id"]
        part = frozen[gid] if gid in frozen else assign_partition(salt, gid)
        assignments[gid] = part
        if part == "golden" and gid not in golden_ok:
            reserved_not_ready.append({
                "group_id": gid,
                "reason": "reserved_golden_missing_eligibility",
            })

    counts: dict[str, int] = {p: 0 for p in PARTITIONS}
    for part in assignments.values():
        counts[part] += 1
    reserved_ready = sorted(
        gid for gid, part in assignments.items()
        if part == "golden" and gid in golden_ok
    )
    return {
        "salt": salt,
        "assignments": dict(sorted(assignments.items())),
        "counts": counts,
        "frozen_respected": sum(1 for gid in assignments if gid in frozen),
        "reserved_golden_count": counts["golden"],
        "reserved_ready": reserved_ready,
        "reserved_not_ready": reserved_not_ready,
    }


def regenerated_reservations(salt: str, assignments: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Regenerate a frozen assignment map from the reservation hash.

    Audited pre-calibration migration only: re-derives every assignment from
    the stable (salt, group_id) hash and returns (regenerated, changed_ids).
    Raises TraceError when any changed group would NOT regenerate as reserved
    golden -- the only sanctioned correction is bringing hash-reserved golden
    groups back from a shadow training assignment; any other divergence means
    the file was not produced by that known defect and migration must stop.
    """
    regenerated = {gid: assign_partition(salt, gid) for gid in assignments}
    changed = sorted(gid for gid in assignments if assignments[gid] != regenerated[gid])
    unexpected = [gid for gid in changed if regenerated[gid] != "golden"]
    if unexpected:
        raise TraceError(
            f"refusing reservation regeneration: {len(unexpected)} group(s) diverge "
            "outside the reserved golden slice"
        )
    return {gid: regenerated[gid] for gid in sorted(regenerated)}, changed


def guard_pre_calibration_migration(
    freeze_manifest: dict | None,
    consumption_evidence: list[str],
) -> None:
    """Refuse the frozen-assignment migration unless it is provably pre-calibration.

    The migration is allowed only while no golden task has been frozen for
    execution (empty/absent freeze manifest) and no artifact shows the golden
    holdout was consumed. Both conditions raise TraceError fail-closed.
    """
    frozen_tasks = (freeze_manifest or {}).get("frozen_tasks") or []
    if frozen_tasks:
        raise TraceError(
            f"refusing migration: {len(frozen_tasks)} frozen golden task(s) already exist"
        )
    if consumption_evidence:
        raise TraceError(
            "refusing migration: golden consumption evidence: "
            + ", ".join(sorted(consumption_evidence))
        )
