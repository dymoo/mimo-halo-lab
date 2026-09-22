"""Golden held-out bank eligibility with eligibility-independent reservation.

A task group is golden-eligible only when ALL required fields are available
privately:
- task_id: the deterministic group id
- repo_identifier: remote origin URL recorded by the harness
- starting_revision: revision hash captured at session start
- task_prompt: the first user prompt
- test_oracle: an explicitly registered oracle (private oracle index); chat
  transcripts alone never make a task executable.

Bank READINESS additionally requires the reservation: only task groups in the
stable hash-reserved golden slice (see partition.assign_partition) may freeze
into the executable bank. Eligible groups outside the slice are reported
(never imported), and reserved groups that lack fields stay reserved/not_ready.
The evaluation reports the EXACT missing fields per candidate and a bank-level
readiness status. It never invents benchmark truth: if evidence is
insufficient, status is not_ready with exact counts.
"""

from __future__ import annotations

REQUIRED_FIELDS = ("task_id", "repo_identifier", "starting_revision", "task_prompt", "test_oracle")


def evaluate_golden_eligibility(
    groups: list[dict],
    oracle_index: dict[str, str] | None = None,
    min_tasks: int = 50,
    max_tasks: int = 200,
    reserved_ids: set[str] | None = None,
) -> dict:
    """Evaluate candidates and return bank status with exact missing prerequisites.

    oracle_index maps a task id (group_id) to a privately registered oracle
    descriptor (test command / expected evidence). Absent oracles keep the bank
    not_ready; they are never fabricated from conversation content.

    reserved_ids is the hash-reserved golden slice: bank readiness counts only
    reserved AND eligible groups, and eligible non-reserved groups are reported
    via non_reserved_eligible_count but never imported. When omitted, every
    candidate is treated as reserved (pure eligibility view).
    """
    oracle_index = oracle_index or {}
    candidates: list[dict] = []
    for group in sorted(groups, key=lambda g: g["group_id"]):
        task_id = group["group_id"]
        available = {
            "task_id": task_id,
            "repo_identifier": group.get("repo_identifier"),
            "starting_revision": group.get("start_revision"),
            "task_prompt": group.get("task_prompt"),
            "test_oracle": oracle_index.get(task_id),
        }
        missing = [f for f in REQUIRED_FIELDS if not available.get(f)]
        # An explicit issue ref strengthens identity but cannot substitute an oracle.
        candidates.append({
            "task_id": task_id,
            "repo_identifier": available["repo_identifier"],
            "starting_revision": available["starting_revision"],
            "issue_ref": group.get("issue_ref"),
            "sources": group.get("sources", []),
            "session_count": group.get("session_count", 0),
            "eligible": not missing,
            "missing_fields": missing,
            "chat_record_only": not missing or "test_oracle" in missing,
        })
    return summarize_golden_candidates(candidates, min_tasks, max_tasks, reserved_ids)


def summarize_golden_candidates(
    candidates: list[dict],
    min_tasks: int = 50,
    max_tasks: int = 200,
    reserved_ids: set[str] | None = None,
) -> dict:
    """Counts, bank status and blocker over prebuilt candidate rows.

    Shared by live evaluation and the audited reservation migration, which
    reuses the stored candidate rows instead of re-parsing transcripts; the
    summarization logic is single-sourced here.
    """
    for candidate in candidates:
        candidate["reserved"] = reserved_ids is None or candidate["task_id"] in reserved_ids

    eligible = [c for c in candidates if c["eligible"]]
    reserved = [c for c in candidates if c["reserved"]]
    bank = [c for c in candidates if c["reserved"] and c["eligible"]]
    non_reserved_eligible = len(eligible) - len(bank)

    missing_field_counts: dict[str, int] = {}
    for c in candidates:
        for f in c["missing_fields"]:
            missing_field_counts[f] = missing_field_counts.get(f, 0) + 1

    bank_ready = min_tasks <= len(bank) <= max_tasks
    return {
        "required_fields": list(REQUIRED_FIELDS),
        "candidates": candidates,
        "reserved_count": len(reserved),
        "eligible_count": len(eligible),
        "ineligible_count": len(candidates) - len(eligible),
        "reserved_eligible_count": len(bank),
        "non_reserved_eligible_count": non_reserved_eligible,
        "missing_field_counts": dict(sorted(missing_field_counts.items())),
        "bank_status": "ready" if bank_ready else "not_ready",
        "bank_bounds": {"min_tasks": min_tasks, "max_tasks": max_tasks},
        "blocker": None if bank_ready else _blocker_text(
            len(bank), min_tasks, missing_field_counts, non_reserved_eligible
        ),
    }


def _blocker_text(eligible: int, min_tasks: int, missing: dict[str, int],
                  non_reserved_eligible: int = 0) -> str:
    parts = [f"reserved+eligible tasks {eligible} < minimum {min_tasks}"]
    if missing:
        detail = ", ".join(f"{field} missing for {count}" for field, count in sorted(missing.items()))
        parts.append(f"missing prerequisites: {detail}")
    if non_reserved_eligible:
        parts.append(
            f"{non_reserved_eligible} eligible task(s) lie outside the reserved golden "
            "slice and are never imported into the bank"
        )
    parts.append("no oracles may be fabricated from chat records; register private "
                 "oracles and start revisions before freezing")
    return "; ".join(parts)
