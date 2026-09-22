#!/usr/bin/env python3
"""Bursty workload replay (stdlib only, Python >= 3.11).

Replays a bursty agent-style timing schedule against a llama-server. The
schedule comes from an explicit JSON file — either a public sanitized timing
input or local trace-derived timings. The tool never invents performance:
every reported number is measured from this replay, and the schedule itself
is echoed into the report.

Input JSON (schema_version 1):
{
  "events": [
    {"t_s": 0.0, "kind": "shell",        "prompt_tokens": 512,  "gen_tokens": 64},
    {"t_s": 1.7, "kind": "compile",      "prompt_tokens": 1024, "gen_tokens": 32},
    {"t_s": 9.2, "kind": "test",         "prompt_tokens": 4096, "gen_tokens": 128},
    {"t_s": 12.0, "kind": "files",       "prompt_tokens": 256,  "gen_tokens": 512}
  ],
  "scale": 1.0,                 // optional time scale for the schedule
  "notes": "source of the timings"
}

Events are sorted by t_s. Each event sends one /completion request at its
scheduled offset (relative to a common replay start), waits for completion,
and records measured latency plus server-reported token counts. The aggregate
"successful requests per replay wall-hour" is reported as an observed replay
throughput of THIS schedule only.

--plan mode emits the schedule with no model calls and no measurements.
Timeouts, HTTP failures and parse failures are recorded per event; missing or
malformed input fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

KINDS_HINT = "free-form (e.g. shell, compile, test, files, prefill_wait); recorded verbatim"


def _fail(message: str) -> None:
    print(f"burst_replay: {message}", file=sys.stderr)
    raise SystemExit(2)


def load_schedule(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema_version") != 1:
        _fail(f"{path}: expected schema_version 1")
    events = data.get("events")
    if not isinstance(events, list) or not events:
        _fail(f"{path}: events must be a non-empty list")
    scale = data.get("scale", 1.0)
    parsed = []
    for i, ev in enumerate(events):
        try:
            t = float(ev["t_s"]) * float(scale)
            ptok = int(ev["prompt_tokens"])
            gtok = int(ev["gen_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            _fail(f"{path}: event {i} malformed: {exc}")
        if t < 0 or ptok <= 0 or gtok <= 0:
            _fail(f"{path}: event {i} out of range (t_s>=0, positive token counts required)")
        parsed.append({"index": i, "t_s": t, "kind": str(ev.get("kind", "unknown")), "prompt_tokens": ptok, "gen_tokens": gtok})
    parsed.sort(key=lambda e: e["t_s"])
    return parsed


def filler_prompt(tokens: int) -> str:
    unit = "def compute(a, b):\n    return a + b * 2\n"
    return unit * max(1, tokens // 8)


def one_event(url: str, event: dict, start: float, timeout: float, results: dict) -> None:
    delay = event["t_s"] - (time.monotonic() - start)
    if delay > 0:
        time.sleep(delay)
    send_at = time.monotonic()
    body = json.dumps({"prompt": filler_prompt(event["prompt_tokens"]), "n_predict": event["gen_tokens"]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        predicted = payload.get("timings", {}).get("predicted_n")
        if predicted is None:
            results[event["index"]] = {"status": "parse_error", "error": "timings.predicted_n missing"}
            return
        results[event["index"]] = {
            "status": "ok", "predicted_tokens": predicted,
            "start_lag_s": round(send_at - start - event["t_s"], 3),
            "latency_s": round(time.monotonic() - send_at, 3),
        }
    except Exception as exc:  # noqa: BLE001 - explicit recorded failure
        status = "timeout" if isinstance(exc, TimeoutError) else "failed"
        results[event["index"]] = {"status": status, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", required=True, help="timing JSON (schema_version 1)")
    ap.add_argument("--server-url", default=None, help="llama-server /completion URL, e.g. http://127.0.0.1:18080/completion")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout seconds")
    ap.add_argument("--output", default=None, help="write the JSON report here (default: stdout)")
    args = ap.parse_args()

    events = load_schedule(args.schedule)
    if args.server_url is None:
        plan = {"schema_version": 1, "action": "plan", "event_count": len(events),
                "total_scheduled_s": events[-1]["t_s"], "events": events,
                "note": "plan mode: no model calls, no measurements, no invented performance"}
        text = json.dumps(plan, sort_keys=True, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        print(text)
        return

    url = args.server_url
    results: dict[int, dict] = {}
    start = time.monotonic()
    threads = [threading.Thread(target=one_event, args=(url, ev, start, args.timeout, results)) for ev in events]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - start

    ok = [r for r in results.values() if r["status"] == "ok"]
    report = {
        "schema_version": 1,
        "mode": "replay",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schedule": os.path.basename(args.schedule),
        "event_count": len(events),
        "replay_wall_s": round(wall, 3),
        "successful_requests": len(ok),
        "failed_requests": sum(1 for r in results.values() if r["status"] == "failed"),
        "timeout_requests": sum(1 for r in results.values() if r["status"] == "timeout"),
        "parse_error_requests": sum(1 for r in results.values() if r["status"] == "parse_error"),
        "observed_requests_per_replay_hour": round(len(ok) / (wall / 3600.0), 3) if wall > 0 else None,
        "throughput_note": "observed throughput of this replay schedule only; not a model quality or speed claim",
        "events": [dict(events[i], **results.get(i, {"status": "missing"})) for i in range(len(events))],
    }
    text = json.dumps(report, sort_keys=True, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    raise SystemExit(0 if not (report["failed_requests"] or report["timeout_requests"]
                               or report["parse_error_requests"]) else 1)


if __name__ == "__main__":
    main()
