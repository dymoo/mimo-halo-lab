"""Pure retention decision CLI for the NVMe state tier.

Consumes an explicit observation (fixed ``now``, pressure flags, sessions,
prefix entries) plus a validated config and emits a decision document:
``hot_evictions``, ``retention_actions``, ``prefix_evictions``, a
``rationale`` and the ``effective_config``.  Every returned value is a
*decision*; the module performs no runtime, scheduler or disk mutation and
never reads the wall clock.  Contracts: local://nvme-implementation-contract.md
(observation/decision schema_version 1) and local://nvme-contract.md.

CLI::

    PYTHONPATH=src python3 -m mimo_halo.state_tier.retention \
        --config configs/state-retention.json \
        --input OBSERVATIONS.json --output DECISIONS.json

Policy -- one deterministic pass, no plugin registry and no speculative
abstraction:

* No pressure/demand: nothing leaves HOT no matter how long it has been
  idle; warm/cold targets are soft and never act as hard TTL evictions.
* Slot/UMA pressure: only HOT+WAITING sessions are victims, ranked lowest
  priority first then longest idle; RUNNING/SUSPENDING/RESTORING are never
  victims; the hot idle grace holds unless a strictly higher-priority
  incoming request overrides it (soft priority override).
* The warm preferred window only demotes WARM to COLD (class metadata;
  blob bytes untouched).  Durable snapshots leave NVMe solely through
  quota overflow or an explicit disk free-bytes target, both LRU.
* Prefix entries never expire by age: only incompatibility, the prefix
  quota, or an explicit disk free-bytes target selects them.
* Where the observation cannot size a pressure decision, the rationale
  states the explicit shortfall or missing target instead of claiming
  invented freed memory.

Validation is fail-closed: finite integer byte caps, finite timestamps
consistent with the explicit clock, priorities 0..3, non-empty unique
opaque IDs, lifecycle/tier consistency, and boolean pressure flags are all
checked before any decision is computed; invalid input exits nonzero and
writes no output file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

SCHEMA_VERSION = 1

#: 0 interactive, 1 high-priority local, 2 normal, 3 background.
PRIORITIES = (0, 1, 2, 3)
LIFECYCLES = ("RUNNING", "WAITING", "SUSPENDING", "NVME_RESIDENT", "RESTORING")
TIERS = ("HOT", "WARM", "COLD")
#: Lifecycle states that hold or capture a hot slot: never eviction victims.
PROTECTED_LIFECYCLES = frozenset({"RUNNING", "SUSPENDING", "RESTORING"})
#: Lifecycle states whose target state is resident in UMA (hot tier).
HOT_LIFECYCLES = frozenset({"RUNNING", "WAITING", "SUSPENDING", "RESTORING"})

CONFIG_KEYS = (
    "schema_version",
    "hot_idle_grace_seconds",
    "warm_nvme",
    "cold_nvme",
    "prefix_cache",
    "quick_reactivation_window_seconds",
)
WINDOW_KEYS = ("preferred_retention_seconds", "max_size_bytes")
PREFIX_CONFIG_KEYS = ("preferred_retention_seconds", "eviction", "max_size_bytes")
PREFIX_EVICTION = "LRU"

OBSERVATION_KEYS = (
    "schema_version",
    "now",
    "pressure",
    "incoming_priority",
    "sessions",
    "prefix_entries",
)
PRESSURE_KEYS = ("needs_hot_slot", "uma_pressure", "disk_pressure", "required_free_bytes")
SESSION_KEYS = (
    "session_id",
    "lifecycle",
    "tier",
    "priority",
    "last_accessed",
    "state_bytes",
    "waiting_reason",
    "expected_wakeup",
)
PREFIX_KEYS = ("prefix_id", "last_accessed", "state_bytes", "compatible")


class RetentionError(Exception):
    """Invalid config, observation, or output environment (fail closed)."""


def _reject_constant(name: str) -> None:
    raise RetentionError(f"non-finite JSON numeric constant {name} is not allowed")


def _load_json(path: str, label: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_constant)
    except RetentionError:
        raise
    except json.JSONDecodeError as exc:
        raise RetentionError(f"{label} {path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise RetentionError(f"cannot read {label} {path}: {exc}") from exc


def _object(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise RetentionError(f"{path}: must be a JSON object")
    return value


def _exact_keys(value: dict, keys: tuple[str, ...], path: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise RetentionError(f"{path}: missing keys {missing}")
    unexpected = [key for key in value if key not in keys]
    if unexpected:
        raise RetentionError(f"{path}: unexpected keys {unexpected}")


def _present(value: dict, keys: tuple[str, ...], path: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise RetentionError(f"{path}: missing keys {missing}")


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise RetentionError(f"{path}: must be a boolean")
    return value


def _int(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RetentionError(f"{path}: must be an integer")
    if value < minimum:
        raise RetentionError(f"{path}: must be >= {minimum} (got {value})")
    return value


def _number(value: Any, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetentionError(f"{path}: must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise RetentionError(f"{path}: must be a finite number (got {value})")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise RetentionError(f"{path}: must be a non-empty string")
    return value


def _opt_string(value: Any, path: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise RetentionError(f"{path}: must be null or a string")
    return value


def _priority(value: Any, path: str) -> int:
    priority = _int(value, path)
    if priority not in PRIORITIES:
        raise RetentionError(f"{path}: must be one of {list(PRIORITIES)} (got {priority})")
    return priority


def _window(raw: Any, path: str) -> dict:
    obj = _object(raw, path)
    _exact_keys(obj, WINDOW_KEYS, path)
    return {
        "preferred_retention_seconds": _int(
            obj["preferred_retention_seconds"], f"{path}.preferred_retention_seconds"
        ),
        "max_size_bytes": _int(obj["max_size_bytes"], f"{path}.max_size_bytes"),
    }


def _validate_config(raw: Any) -> dict:
    cfg = _object(raw, "config")
    _exact_keys(cfg, CONFIG_KEYS, "config")
    version = _int(cfg["schema_version"], "config.schema_version")
    if version != SCHEMA_VERSION:
        raise RetentionError(f"config.schema_version: expected {SCHEMA_VERSION} (got {version})")
    prefix_raw = _object(cfg["prefix_cache"], "config.prefix_cache")
    _exact_keys(prefix_raw, PREFIX_CONFIG_KEYS, "config.prefix_cache")
    if prefix_raw["preferred_retention_seconds"] is not None:
        raise RetentionError(
            "config.prefix_cache.preferred_retention_seconds: must be null; "
            "prefix entries never expire by age"
        )
    if prefix_raw["eviction"] != PREFIX_EVICTION:
        raise RetentionError(
            f"config.prefix_cache.eviction: expected {PREFIX_EVICTION!r} "
            f"(got {prefix_raw['eviction']!r})"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "hot_idle_grace_seconds": _int(
            cfg["hot_idle_grace_seconds"], "config.hot_idle_grace_seconds"
        ),
        "warm_nvme": _window(cfg["warm_nvme"], "config.warm_nvme"),
        "cold_nvme": _window(cfg["cold_nvme"], "config.cold_nvme"),
        "prefix_cache": {
            "preferred_retention_seconds": None,
            "eviction": PREFIX_EVICTION,
            "max_size_bytes": _int(
                prefix_raw["max_size_bytes"], "config.prefix_cache.max_size_bytes"
            ),
        },
        "quick_reactivation_window_seconds": _int(
            cfg["quick_reactivation_window_seconds"],
            "config.quick_reactivation_window_seconds",
        ),
    }


def _validate_session(raw: Any, index: int, now: int | float) -> dict:
    path = f"sessions[{index}]"
    obj = _object(raw, path)
    _present(obj, SESSION_KEYS, path)
    session_id = _string(obj["session_id"], f"{path}.session_id")
    lifecycle = obj["lifecycle"]
    if lifecycle not in LIFECYCLES:
        raise RetentionError(f"{path}.lifecycle: must be one of {list(LIFECYCLES)}")
    tier = obj["tier"]
    if tier not in TIERS:
        raise RetentionError(f"{path}.tier: must be one of {list(TIERS)}")
    if tier == "HOT":
        if lifecycle not in HOT_LIFECYCLES:
            raise RetentionError(
                f"{path}: HOT tier requires an in-UMA lifecycle "
                f"{sorted(HOT_LIFECYCLES)} (got {lifecycle!r})"
            )
    elif lifecycle != "NVME_RESIDENT":
        raise RetentionError(
            f"{path}: {tier} tier requires lifecycle 'NVME_RESIDENT' (got {lifecycle!r})"
        )
    last_accessed = _number(obj["last_accessed"], f"{path}.last_accessed")
    if last_accessed > now:
        raise RetentionError(
            f"{path}.last_accessed: {last_accessed} is after observation now {now}"
        )
    return {
        "session_id": session_id,
        "lifecycle": lifecycle,
        "tier": tier,
        "priority": _priority(obj["priority"], f"{path}.priority"),
        "last_accessed": last_accessed,
        "state_bytes": _int(obj["state_bytes"], f"{path}.state_bytes"),
        "waiting_reason": _opt_string(obj["waiting_reason"], f"{path}.waiting_reason"),
        "expected_wakeup": (
            None
            if obj["expected_wakeup"] is None
            else _number(obj["expected_wakeup"], f"{path}.expected_wakeup")
        ),
    }


def _validate_prefix(raw: Any, index: int, now: int | float) -> dict:
    path = f"prefix_entries[{index}]"
    obj = _object(raw, path)
    _present(obj, PREFIX_KEYS, path)
    last_accessed = _number(obj["last_accessed"], f"{path}.last_accessed")
    if last_accessed > now:
        raise RetentionError(
            f"{path}.last_accessed: {last_accessed} is after observation now {now}"
        )
    return {
        "prefix_id": _string(obj["prefix_id"], f"{path}.prefix_id"),
        "last_accessed": last_accessed,
        "state_bytes": _int(obj["state_bytes"], f"{path}.state_bytes"),
        "compatible": _bool(obj["compatible"], f"{path}.compatible"),
    }


def _validate_observation(raw: Any) -> dict:
    obs = _object(raw, "observation")
    _exact_keys(obs, OBSERVATION_KEYS, "observation")
    version = _int(obs["schema_version"], "observation.schema_version")
    if version != SCHEMA_VERSION:
        raise RetentionError(
            f"observation.schema_version: expected {SCHEMA_VERSION} (got {version})"
        )
    now = _number(obs["now"], "observation.now")

    pressure_raw = _object(obs["pressure"], "observation.pressure")
    _exact_keys(pressure_raw, PRESSURE_KEYS, "observation.pressure")
    pressure = {
        "needs_hot_slot": _bool(
            pressure_raw["needs_hot_slot"], "observation.pressure.needs_hot_slot"
        ),
        "uma_pressure": _bool(pressure_raw["uma_pressure"], "observation.pressure.uma_pressure"),
        "disk_pressure": _bool(
            pressure_raw["disk_pressure"], "observation.pressure.disk_pressure"
        ),
        "required_free_bytes": _int(
            pressure_raw["required_free_bytes"],
            "observation.pressure.required_free_bytes",
        ),
    }

    incoming = obs["incoming_priority"]
    if incoming is not None:
        incoming = _priority(incoming, "observation.incoming_priority")

    if not isinstance(obs["sessions"], list):
        raise RetentionError("observation.sessions: must be a JSON array")
    sessions: list[dict] = []
    seen_ids: set[str] = set()
    for index, raw_session in enumerate(obs["sessions"]):
        session = _validate_session(raw_session, index, now)
        if session["session_id"] in seen_ids:
            raise RetentionError(f"duplicate session_id {session['session_id']!r}")
        seen_ids.add(session["session_id"])
        sessions.append(session)

    if not isinstance(obs["prefix_entries"], list):
        raise RetentionError("observation.prefix_entries: must be a JSON array")
    prefix_entries: list[dict] = []
    seen_prefixes: set[str] = set()
    for index, raw_prefix in enumerate(obs["prefix_entries"]):
        prefix = _validate_prefix(raw_prefix, index, now)
        if prefix["prefix_id"] in seen_prefixes:
            raise RetentionError(f"duplicate prefix_id {prefix['prefix_id']!r}")
        seen_prefixes.add(prefix["prefix_id"])
        prefix_entries.append(prefix)

    return {
        "schema_version": SCHEMA_VERSION,
        "now": now,
        "pressure": pressure,
        "incoming_priority": incoming,
        "sessions": sessions,
        "prefix_entries": prefix_entries,
    }


def _j(value: Any) -> str:
    """JSON scalar rendering for rationale text (true/false/null, no quoting gaps)."""
    return json.dumps(value, ensure_ascii=True)


def _lru_over_quota(pool: list[dict], cap: int) -> list[dict]:
    """Least-recently-accessed entries to evict until total bytes fit ``cap``."""
    total = sum(item["state_bytes"] for item in pool)
    chosen: list[dict] = []
    for item in sorted(pool, key=lambda entry: entry["last_accessed"]):
        if total <= cap:
            break
        chosen.append(item)
        total -= item["state_bytes"]
    return chosen


def _apply_policy(cfg: dict, obs: dict) -> dict:
    now = obs["now"]
    pressure = obs["pressure"]
    incoming = obs["incoming_priority"]
    sessions = obs["sessions"]
    prefixes = obs["prefix_entries"]
    grace = cfg["hot_idle_grace_seconds"]
    warm_window = cfg["warm_nvme"]["preferred_retention_seconds"]
    warm_cap = cfg["warm_nvme"]["max_size_bytes"]
    cold_cap = cfg["cold_nvme"]["max_size_bytes"]
    prefix_cap = cfg["prefix_cache"]["max_size_bytes"]
    required = pressure["required_free_bytes"]

    parts = [
        f"explicit clock: observation now={_j(now)}",
        (
            "pressure: needs_hot_slot={needs_hot_slot} uma_pressure={uma_pressure} "
            "disk_pressure={disk_pressure} required_free_bytes={required_free_bytes}"
        ).format(**pressure),
        f"incoming_priority={_j(incoming)}",
    ]
    if required and not (pressure["uma_pressure"] or pressure["disk_pressure"]):
        parts.append(
            f"required_free_bytes={_j(required)} carries no uma_pressure/disk_pressure "
            "flag: no byte target applied"
        )

    # --- hot pool: slot admission and UMA byte demand -------------------
    def eligible(session: dict) -> bool:
        # Only HOT WAITING can yield; RUNNING/SUSPENDING/RESTORING never do.
        if session["lifecycle"] != "WAITING":
            return False
        if now - session["last_accessed"] >= grace:
            return True
        # Soft priority override: a strictly higher-priority incoming
        # request may evict before grace; otherwise short waits hold.
        if incoming is not None and incoming < session["priority"]:
            return True
        # Real UMA byte demand overrides grace even with no incoming
        # admission: uma_pressure with positive required_free_bytes is a
        # sized target, not a boolean wish.
        return pressure["uma_pressure"] and required > 0

    ranked = sorted(
        (session for session in sessions if session["tier"] == "HOT" and eligible(session)),
        key=lambda session: (-session["priority"], session["last_accessed"]),
    )
    hot_evictions: list[str] = []
    hot_freed = 0

    if pressure["needs_hot_slot"]:
        if ranked:
            victim = ranked.pop(0)
            idle = now - victim["last_accessed"]
            if idle >= grace:
                basis = "idle past grace"
            elif incoming is not None and incoming < victim["priority"]:
                basis = "incoming priority outranks victim"
            else:
                basis = "uma byte demand overrides grace"
            hot_evictions.append(victim["session_id"])
            hot_freed += victim["state_bytes"]
            parts.append(
                f"slot admission demand: evicted {victim['session_id']} "
                f"(priority {_j(victim['priority'])}, idle {_j(idle)}s, {basis})"
            )
        else:
            parts.append(
                "slot admission demand unserved: no eligible HOT WAITING victim "
                "(RUNNING/SUSPENDING/RESTORING are never victims, and shorter waits "
                "hold grace without a strictly higher-priority incoming request or "
                "positive UMA byte demand); "
                "the boolean demand cannot size any further capacity"
            )

    uma_target = required if pressure["uma_pressure"] else 0
    if pressure["uma_pressure"]:
        if uma_target == 0:
            parts.append(
                "uma_pressure with required_free_bytes=0 gives no byte target: "
                "no hot bytes sized or claimed freed"
            )
        else:
            while ranked and hot_freed < uma_target:
                victim = ranked.pop(0)
                hot_evictions.append(victim["session_id"])
                hot_freed += victim["state_bytes"]
            if hot_freed >= uma_target:
                parts.append(
                    f"uma free-bytes target {_j(uma_target)}: evicted "
                    f"{len(hot_evictions)} hot waiter(s) freeing {_j(hot_freed)} "
                    "observed state_bytes; target met"
                )
            else:
                shortfall = uma_target - hot_freed
                parts.append(
                    f"uma free-bytes target {_j(uma_target)}: evicted "
                    f"{len(hot_evictions)} hot waiter(s) freeing {_j(hot_freed)} "
                    f"observed state_bytes; shortfall {_j(shortfall)} bytes unmet "
                    "and not claimed as freed"
                )

    # --- NVMe state pool: soft class demotion, quotas -------------------
    # Warm and cold are classes over the SAME immutable blobs: demotion is
    # metadata only -- it frees warm budget, never disk bytes -- and the
    # demoted bytes then consume the cold quota. Only the cold quota or an
    # explicit disk target selects an actual snapshot deletion.
    demote_reason: dict[str, str] = {}
    for session in sessions:
        if session["tier"] == "WARM" and now - session["last_accessed"] > warm_window:
            demote_reason[session["session_id"]] = (
                f"idle beyond warm preferred window {warm_window}s: "
                "class metadata only, blob bytes untouched"
            )
    if demote_reason:
        parts.append(
            f"warm preferred window {warm_window}s exceeded: demote_to_cold for "
            f"{list(demote_reason)}; class metadata only, blob bytes untouched"
        )

    effective_warm = [
        session
        for session in sessions
        if session["tier"] == "WARM" and session["session_id"] not in demote_reason
    ]
    effective_cold = [
        session
        for session in sessions
        if session["tier"] == "COLD" or session["session_id"] in demote_reason
    ]

    evict_reason: dict[str, str] = {}
    state_freed = 0  # disk bytes actually deleted; demotions free none.

    warm_chosen = _lru_over_quota(effective_warm, warm_cap)
    if warm_chosen:
        warm_ids = {session["session_id"] for session in warm_chosen}
        for session in warm_chosen:
            demote_reason[session["session_id"]] = (
                f"warm tier over max_size_bytes {_j(warm_cap)}: LRU demotion to cold, "
                "frees warm budget not disk bytes"
            )
        effective_warm = [
            session for session in effective_warm if session["session_id"] not in warm_ids
        ]
        effective_cold.extend(warm_chosen)
        ids = [session["session_id"] for session in warm_chosen]
        parts.append(
            f"warm quota {warm_cap} bytes exceeded: LRU demoted {ids} to cold "
            "(oldest access first); class metadata only, frees warm budget not "
            "disk bytes; demoted bytes now consume cold quota"
        )

    cold_chosen = _lru_over_quota(effective_cold, cold_cap)
    if cold_chosen:
        state_freed += sum(session["state_bytes"] for session in cold_chosen)
        for session in cold_chosen:
            evict_reason[session["session_id"]] = (
                f"cold tier over max_size_bytes {_j(cold_cap)}: LRU eviction"
            )
        ids = [session["session_id"] for session in cold_chosen]
        parts.append(
            f"cold quota {cold_cap} bytes exceeded: LRU evicted {ids} (oldest access first)"
        )

    # --- prefix pool: incompatibility, quota, then disk target ----------
    prefix_evicted: list[dict] = []
    prefix_freed = 0

    incompatible = [entry for entry in prefixes if not entry["compatible"]]
    live_prefixes = [entry for entry in prefixes if entry["compatible"]]
    if incompatible:
        prefix_evicted.extend(incompatible)
        prefix_freed += sum(entry["state_bytes"] for entry in incompatible)
        parts.append(
            "incompatible prefix entries evicted (can never match the required "
            f"identity again): {[entry['prefix_id'] for entry in incompatible]}"
        )

    prefix_quota = _lru_over_quota(live_prefixes, prefix_cap)
    if prefix_quota:
        prefix_evicted.extend(prefix_quota)
        prefix_freed += sum(entry["state_bytes"] for entry in prefix_quota)
        quota_ids = {entry["prefix_id"] for entry in prefix_quota}
        live_prefixes = [
            entry for entry in live_prefixes if entry["prefix_id"] not in quota_ids
        ]
        parts.append(
            f"prefix quota {prefix_cap} bytes exceeded: LRU evicted "
            f"{[entry['prefix_id'] for entry in prefix_quota]} (oldest access first)"
        )

    # --- disk free-bytes target: cache (prefix) first, then snapshots ---
    disk_target = required if pressure["disk_pressure"] else 0
    if disk_target > 0:
        for entry in sorted(live_prefixes, key=lambda item: item["last_accessed"]):
            if prefix_freed + state_freed >= disk_target:
                break
            prefix_evicted.append(entry)
            prefix_freed += entry["state_bytes"]

        snapshot_pool = [
            session
            for session in effective_warm + effective_cold
            if session["session_id"] not in evict_reason
        ]
        for session in sorted(snapshot_pool, key=lambda item: item["last_accessed"]):
            if prefix_freed + state_freed >= disk_target:
                break
            evict_reason[session["session_id"]] = (
                "disk_pressure free-bytes target: LRU eviction"
            )
            state_freed += session["state_bytes"]

        freed = prefix_freed + state_freed
        if freed >= disk_target:
            parts.append(
                f"disk free-bytes target {_j(disk_target)}: prefix evictions freed "
                f"{_j(prefix_freed)} observed bytes, snapshot evictions freed "
                f"{_j(state_freed)} observed bytes; target met"
            )
        else:
            shortfall = disk_target - freed
            parts.append(
                f"disk free-bytes target {_j(disk_target)}: prefix evictions freed "
                f"{_j(prefix_freed)} observed bytes, snapshot evictions freed "
                f"{_j(state_freed)} observed bytes; shortfall {_j(shortfall)} bytes "
                "unmet and not claimed as freed"
            )

    if prefixes:
        parts.append(
            f"{len(prefixes)} prefix entries: no age expiry applied; selections come "
            "only from incompatibility, prefix quota, or the disk free-bytes target"
        )

    # --- one action per session, input order preserved ------------------
    hot_set = set(hot_evictions)
    actions: list[dict] = []
    kept = 0
    for session in sessions:
        session_id = session["session_id"]
        if session_id in hot_set:
            action = "keep"
            reason = "yielded from hot by hot_evictions: durable state retained for restore"
        elif session_id in evict_reason:
            action = "evict_snapshot"
            reason = evict_reason[session_id]
        elif session_id in demote_reason:
            action = "demote_to_cold"
            reason = demote_reason[session_id]
        elif session["tier"] == "HOT":
            action = "keep"
            if session["lifecycle"] in PROTECTED_LIFECYCLES:
                reason = "protected lifecycle: never an eviction victim"
            else:
                reason = "kept hot: no pressure demand selected this session"
        else:
            action = "keep"
            reason = f"kept {session['tier']}: within quota and not selected by disk LRU"
        if action == "keep":
            kept += 1
        actions.append({"session_id": session_id, "action": action, "reason": reason})

    parts.append(f"kept {kept} of {len(sessions)} sessions without snapshot eviction")

    return {
        "schema_version": SCHEMA_VERSION,
        "hot_evictions": hot_evictions,
        "retention_actions": actions,
        "prefix_evictions": [entry["prefix_id"] for entry in prefix_evicted],
        "rationale": ". ".join(parts) + ".",
        "effective_config": cfg,
    }


def decide(config: Any, observation: Any) -> dict:
    """Validate config + observation and return the schema_version 1 decision."""
    cfg = _validate_config(config)
    obs = _validate_observation(observation)
    return _apply_policy(cfg, obs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.state_tier.retention",
        description=(
            "Pure retention decision CLI: config + observation in, decision JSON out. "
            "No runtime or disk side effects; soft targets only, never hard TTL eviction."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="retention config JSON (e.g. configs/state-retention.json)",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="observation JSON (schema_version 1, explicit now clock)",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="decision JSON to write (schema_version 1)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = _load_json(args.config, "config")
        observation = _load_json(args.input, "observation")
        decision = decide(config, observation)
        text = json.dumps(decision, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            raise RetentionError(f"cannot write output {args.output}: {exc}") from exc
    except RetentionError as exc:
        print(f"retention: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
