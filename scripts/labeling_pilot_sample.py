"""One-off sampler for the trace-labeling pilot (docs/experiments/labeling-pilot.md).

Single pass over the raw harness transcripts (read-only at origin):

1. re-derives the deterministic rule labels with the checked-in labeler
   (src/mimo_halo/traces/labeling.py) on freshly parsed SessionTraces,
2. resolves each session's task group / partition the way the partition
   command does (identity.build_task_groups formulas) and joins the frozen
   task-assignments.json,
3. draws a deterministic stratified sample of episodes (non-golden partitions
   only -- the golden partition never enters observation corpora),
4. writes PRIVATE pilot inputs under <output-root>/scratch/labeling-pilot/:
   frame-summary.json, sample-index.json (rule labels + hashed ids) and
   sample-states.jsonl (blinded redacted episode excerpts for typed judging).

Nothing here writes into the Git repository and no raw path, prompt or payload
is emitted: sample ids are hashes and event excerpts are credential-redacted
bounded text kept only in the private output root.

Usage:
    PYTHONPATH=src python3 scripts/labeling_pilot_sample.py \
        --config configs/dataset.json \
        --output-root /Volumes/4tbRackNvme/mimo-halo \
        --target 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mimo_halo.traces.common import atomic_write_json, ensure_dir, sha256_text  # noqa: E402
from mimo_halo.traces.discovery import expand_root, list_transcripts  # noqa: E402
from mimo_halo.traces.identity import find_issue_refs  # noqa: E402
from mimo_halo.traces.labeling import (  # noqa: E402
    CAPABILITIES,
    DIFFICULTY_LEVELS,
    label_episodes,
)
from mimo_halo.traces.session import normalize_session, segment_episodes  # noqa: E402

# Contamination rule: the golden partition never enters observation corpora.
NON_GOLDEN = ("pruning", "quant", "recovery", "validation", "torture")

# Per-tag Noul subset: the five boundary-critical tags from docs/datasets.md
# ("Tagging rules and boundary cases") plus the most frequent remaining tags,
# filled to TAG_SUBSET_SIZE by frame frequency (ties alphabetical).
BOUNDARY_TAGS = ("debugging", "compiler_interpretation", "test_interpretation",
                 "verification", "recovery")
TAG_SUBSET_SIZE = 8
DEFAULT_TARGET = 200
DEFAULT_SEED = "labeling-pilot-v1"

# Verdict criteria fixed BEFORE the judgment batch runs (recorded in
# frame-summary.json, which is written before any judgment is collected).
PRE_REGISTERED_CRITERIA = {
    "unit": "10 hand-inspected disagreements, verdicts in {judge_better, rule_better, ambiguous}",
    "supported": "judge_better >= 6 and rule_better <= 2",
    "falsified": "rule_better >= 6 and judge_better <= 2",
    "inconclusive": "anything else, including fewer than 10 inspectable disagreements",
    "construct_check_not_gating": (
        "reported separately: difficulty exact agreement and within-one-level "
        "rate between judge and rule labels"),
}

_STATE_PROMPT_CHARS = 600
_STATE_TEXT_BUDGET = 9000
_STATE_TEXT_CAP = 700
_STATE_TEXT_FLOOR = 150
_STATE_MAX_EVENTS = 80


def map_sessions_to_groups(traces) -> tuple[dict[str, list[str]], dict]:
    """session source_path_sha256 -> sorted group ids, replicating the exact
    membership rules of identity.build_task_groups (including retry-deduped
    sessions, which the group's session_refs list does not retain)."""
    explicit: dict[str, str | None] = {}
    prompt_info: dict[str, tuple[str | None, str]] = {}
    prompt_sources: dict[tuple[str | None, str], set[str]] = {}
    stats = {"no_cwd": 0, "no_prompt_no_ref": 0, "cross_harness_prompt": 0}

    for t in traces:
        ref = t.source_path_sha256
        if not t.cwd_sha256:
            stats["no_cwd"] += 1
            continue
        refs = find_issue_refs(t.first_user_prompt)
        if refs:
            explicit[ref] = t.cwd_sha256
            continue
        if not t.first_user_prompt:
            stats["no_prompt_no_ref"] += 1
            continue
        digest = sha256_text(t.first_user_prompt)[:16]  # identity.prompt_hash_key
        prompt_info[ref] = (t.cwd_sha256, digest)
        prompt_sources.setdefault((t.cwd_sha256, digest), set()).add(t.source)

    refs_by_ref = {t.source_path_sha256: find_issue_refs(t.first_user_prompt) for t in traces}
    mapping: dict[str, list[str]] = {}
    for ref, cwd in explicit.items():
        mapping[ref] = sorted({
            sha256_text(f"explicit|{cwd}|{r}")[:24] for r in refs_by_ref[ref]
        })
    for t in traces:
        ref = t.source_path_sha256
        if ref in mapping or ref not in prompt_info:
            continue
        key = prompt_info[ref]
        if len(prompt_sources[key]) > 1:
            stats["cross_harness_prompt"] += 1
            continue
        mapping[ref] = [sha256_text(f"prompt|{t.source}|{key[0]}|{key[1]}")[:24]]
    return mapping, stats


def build_state(trace, ep, events) -> dict:
    """Blinded redacted episode excerpt: raw facts only (types, tool names,
    error flags, credential-redacted bounded text). Rule labels and the
    labeler's input hints (verification_hint/spam_hint/weights) are withheld."""
    shown = list(events)
    gap = 0
    if len(shown) > _STATE_MAX_EVENTS:
        gap = len(shown) - _STATE_MAX_EVENTS
        shown = shown[:60] + shown[-20:]
    text_events = [e for e in shown if e.text]
    cap = _STATE_TEXT_BUDGET // max(1, len(text_events))
    cap = max(_STATE_TEXT_FLOOR, min(_STATE_TEXT_CAP, cap))

    out_events = []
    for e in shown:
        entry: dict = {"seq": e.seq, "type": e.type}
        if e.tool_name:
            entry["tool"] = e.tool_name
        if e.result_is_error:
            entry["error"] = True
        if e.text:
            t = e.text
            if len(t) > cap:
                t = t[:cap] + " [truncated]"
            entry["text"] = t
        out_events.append(entry)
        if gap and len(out_events) == 60:
            out_events.append({"gap_events": gap})

    prompt = trace.first_user_prompt or ""
    if len(prompt) > _STATE_PROMPT_CHARS:
        prompt = prompt[:_STATE_PROMPT_CHARS] + " [truncated]"
    return {
        "harness": trace.source,
        "turn": ep.turn_index,
        "event_count": len(events),
        "first_prompt": prompt,
        "events": out_events,
    }


def allocate_quotas(cells: dict, target: int) -> dict:
    """Largest-remainder proportional allocation with a 2-per-cell floor."""
    keys = sorted(cells)
    base = {c: min(2, len(cells[c])) for c in keys}
    quotas = dict(base)
    pool = max(0, min(target, sum(len(v) for v in cells.values())) - sum(base.values()))
    weights = {c: len(cells[c]) - base[c] for c in keys}
    total_w = sum(weights.values())
    if pool and total_w:
        exact = {c: pool * weights[c] / total_w for c in keys}
        for c in keys:
            add = min(weights[c], int(exact[c]))
            quotas[c] += add
            exact[c] -= add
        left = min(target, sum(len(v) for v in cells.values())) - sum(quotas.values())
        order = sorted(keys, key=lambda c: (-(exact[c] / weights[c] if weights[c] else 0.0), c))
        for c in order:
            if left <= 0:
                break
            if quotas[c] < len(cells[c]):
                quotas[c] += 1
                left -= 1
    return quotas


def pick_episodes(episodes: list, quota: int, seed: str) -> list:
    """Deterministic capability round-robin inside one stratum cell."""
    buckets: dict[str, list] = {}
    for ep in episodes:
        buckets.setdefault(ep.primary_capability or "_unlabeled", []).append(ep)
    for k in buckets:
        buckets[k].sort(key=lambda ep: sha256_text(f"{seed}|{ep.episode_id}"))
    names = sorted(buckets)
    picked: list = []
    idx = {k: 0 for k in names}
    pos = 0
    while len(picked) < quota:
        progressed = False
        for i in range(len(names)):
            k = names[(pos + i) % len(names)]
            if idx[k] < len(buckets[k]):
                picked.append(buckets[k][idx[k]])
                idx[k] += 1
                progressed = True
                if len(picked) >= quota:
                    break
        if not progressed:
            break
        pos = (pos + 1) % max(1, len(names))
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.json")
    ap.add_argument("--output-root", required=True, help="private output root outside the repo")
    ap.add_argument("--assignments", default=None,
                    help="task-assignments.json (default <output-root>/traces/partition/task-assignments.json)")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="per-source cap for smoke runs only; the pilot run must be full")
    args = ap.parse_args()

    t0 = time.time()
    config = json.loads(Path(args.config).read_text())
    out_dir = ensure_dir(Path(args.output_root) / "scratch" / "labeling-pilot")
    assign_path = Path(args.assignments) if args.assignments else (
        Path(args.output_root) / "traces" / "partition" / "task-assignments.json")
    assignments = json.loads(assign_path.read_text())["assignments"]

    traces = []
    parse_failures = 0
    for source, cfg in config["traces"]["sources"].items():
        root = expand_root(cfg["root"])
        transcripts, _records = list_transcripts(source, root, cfg.get("transcript_glob", "**/*.jsonl"))
        paths = transcripts[: args.max_sessions] if args.max_sessions else transcripts
        for p in paths:
            try:
                traces.append(normalize_session(p, source))
            except Exception:
                parse_failures += 1
        print(f"parsed {source}: {len(paths)} transcripts", file=sys.stderr)
    t_parse = time.time() - t0

    mapping, group_stats = map_sessions_to_groups(traces)

    excluded = {"golden_episodes": 0, "sessions_multi_partition": 0,
                "sessions_unassigned": 0, "sessions_without_episodes": 0}
    frame: dict[tuple[str, str], list] = {}
    cap_freq: dict[str, int] = {}
    ctx: dict[str, tuple] = {}  # episode_id -> (trace, episode, partition)
    episodes_total = 0
    for t in traces:
        gids = mapping.get(t.source_path_sha256, [])
        parts = {assignments[g] for g in gids if g in assignments}
        if not parts:
            excluded["sessions_unassigned"] += 1
            continue
        if len(parts) > 1:
            excluded["sessions_multi_partition"] += 1
            continue
        part = next(iter(parts))
        eps = label_episodes(t, segment_episodes(t))
        episodes_total += len(eps)
        if part == "golden":
            excluded["golden_episodes"] += len(eps)
            continue
        if not eps:
            excluded["sessions_without_episodes"] += 1
            continue
        for ep in eps:
            frame.setdefault((part, ep.difficulty), []).append(ep)
            ctx[ep.episode_id] = (t, ep, part)
            for c in ep.capabilities:
                cap_freq[c] = cap_freq.get(c, 0) + 1

    quotas = allocate_quotas(frame, args.target)
    sampled = [ep for cell in sorted(frame) for ep in pick_episodes(frame[cell], quotas[cell], args.seed)]

    subset = list(BOUNDARY_TAGS)
    for tag in sorted(cap_freq, key=lambda k: (-cap_freq[k], k)):
        if len(subset) >= TAG_SUBSET_SIZE:
            break
        if tag not in subset:
            subset.append(tag)

    sample_records = []
    lines = []
    for ep in sampled:
        t, e, part = ctx[ep.episode_id]
        by_seq = {ev.seq: ev for ev in t.events}
        events = [by_seq[s] for s in e.event_seq if s in by_seq]
        state = build_state(t, e, events)
        lines.append(json.dumps({"episode_id": e.episode_id, "state": state},
                                sort_keys=True, ensure_ascii=False))
        sample_records.append({
            "episode_id": e.episode_id,
            "source": t.source,
            "stratum": {"partition": part, "difficulty": e.difficulty,
                        "primary_capability": e.primary_capability},
            "rule": {
                "difficulty": e.difficulty,
                "primary_capability": e.primary_capability,
                "capabilities": e.capabilities,
                "languages": e.languages,
            },
            "structure": {
                "span": len(e.event_seq),
                "tool_calls": e.tool_calls,
                "has_decision": e.has_decision,
                "verification_events": e.verification_events,
                "recovered": e.recovered,
                "downweighted_events": e.downweighted_events,
            },
        })

    frame_summary = {
        "schema_version": "1.0.0",
        "kind": "labeling-pilot-sample",
        "params": {"seed": args.seed, "target": args.target, "config": "configs/dataset.json",
                   "max_sessions": args.max_sessions},
        "rule_labeler": {
            "module": "src/mimo_halo/traces/labeling.py",
            "method": "recomputed in memory from raw transcripts with the checked-in labeler",
            "note": "the hash-light normalized corpus predates the labeler and carries no labels",
        },
        "corpus": {
            "sessions_parsed": len(traces),
            "sessions_failed": parse_failures,
            "episodes_seen": episodes_total,
            "group_stats": group_stats,
        },
        "excluded": excluded,
        "frame": {
            "episodes": sum(len(v) for v in frame.values()),
            "cells": {f"{p}|{d}": len(frame[(p, d)]) for (p, d) in sorted(frame)},
            "capability_frame_freq": {c: cap_freq[c] for c in CAPABILITIES if c in cap_freq},
        },
        "allocation": {f"{p}|{d}": quotas[(p, d)] for (p, d) in sorted(quotas)},
        "sample_size": len(sample_records),
        "tag_subset": subset,
        "tag_subset_rule": "five boundary tags + top frame-frequency fill to 8 (ties alphabetical)",
        "pre_registered_criteria": PRE_REGISTERED_CRITERIA,
        "wall_seconds": {"parse": round(t_parse, 1), "total": round(time.time() - t0, 1)},
    }

    states_path = out_dir / "sample-states.jsonl"
    states_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_write_json(out_dir / "sample-index.json", sample_records, indent=None)
    atomic_write_json(out_dir / "frame-summary.json", frame_summary)

    # read-back verification (write hazard): re-parse every output
    parsed_lines = [json.loads(x) for x in states_path.read_text(encoding="utf-8").splitlines()]
    assert [x["episode_id"] for x in parsed_lines] == [r["episode_id"] for r in sample_records]
    json.loads((out_dir / "sample-index.json").read_text())
    json.loads((out_dir / "frame-summary.json").read_text())

    print(json.dumps({
        "sessions": len(traces), "episodes_seen": episodes_total,
        "frame": frame_summary["frame"]["episodes"], "sample": len(sample_records),
        "cells": frame_summary["frame"]["cells"],
        "tag_subset": subset,
        "out": str(out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
