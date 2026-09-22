#!/usr/bin/env python3
"""Strix Halo benchmark matrix runner (stdlib only, Python >= 3.11).

Two modes, kept explicitly distinct:

bench   Run llama-bench per prompt-processing length (PP8K/32K/64K/128K from
        the config). llama-bench measures a single in-process context; its
        -b/-ub batch-size knobs are recorded as batch-size knobs and are
        NEVER reported as concurrency. Output parsed from `-o json`.

serve   Concurrency levels C1/C2/C4/C8 are measured as that many actually
        concurrent HTTP requests in flight against one llama-server instance
        (thread pool, one socket per request). Aggregate tok/s is derived from
        the server-reported predicted token counts over the wall time of the
        concurrent wave. No llama.cpp batch-size flag is ever labelled
        concurrency.

Every point has an explicit timeout and records an explicit status
(ok / timeout / failed / parse_error); a failed point never silently
disappears. Results are written as JSON with schema_version 1. The model is
recorded by basename only, so public reports carry no local paths.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

STATUSES = ("ok", "timeout", "failed", "parse_error")


def _fail(message: str) -> None:
    print(f"benchmark_matrix: {message}", file=sys.stderr)
    raise SystemExit(2)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if cfg.get("schema_version") != 1:
        _fail(f"{path}: expected schema_version 1")
    return cfg


def run_llama_bench(tool: str, model: str, pp_tokens: int, cfg: dict, extra: list[str], timeout: float) -> dict:
    argv = [tool, "-m", model, "-p", str(pp_tokens), "-n", "0", "-o", "json", *extra]
    public_argv = list(argv)
    public_argv[public_argv.index("-m") + 1] = os.path.basename(model)  # no local paths in reports
    start = time.monotonic()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "pp_tokens": pp_tokens, "duration_s": round(time.monotonic() - start, 3),
                "command": public_argv}
    if proc.returncode != 0:
        return {"status": "failed", "pp_tokens": pp_tokens, "duration_s": round(time.monotonic() - start, 3),
                "command": public_argv, "stderr_tail": proc.stderr[-2000:]}
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "parse_error", "pp_tokens": pp_tokens, "command": public_argv, "error": str(exc),
                "stdout_head": proc.stdout[:400]}
    points = []
    for row in rows if isinstance(rows, list) else []:
        test = row.get("test", "")
        if test.startswith("pp"):
            points.append({"test": test, "avg_ts": row.get("avg_ts"), "stddev_ts": row.get("stddev_ts"),
                           "n_prompt": row.get("n_prompt"), "avg_ns": row.get("avg_ns")})
    if not points:
        return {"status": "parse_error", "pp_tokens": pp_tokens, "command": public_argv,
                "error": "no pp test rows in llama-bench json output"}
    return {"status": "ok", "pp_tokens": pp_tokens, "method": "in-process batched prompt processing (batch-size dependent; NOT concurrency)",
            "duration_s": round(time.monotonic() - start, 3), "command": public_argv, "points": points}


def filler_prompt(prompt_tokens: int) -> str:
    """Deterministic code-shaped filler (~4 bytes/token is approximate; the
    server-reported prompt_n is recorded for honesty)."""
    unit = "def compute(a, b):\n    return a + b * 2\n"
    return unit * max(1, prompt_tokens // 8)


def one_request(url: str, prompt: str, gen_tokens: int, timeout: float, results: dict, idx: int) -> None:
    body = json.dumps({"prompt": prompt, "n_predict": gen_tokens}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        predicted = payload.get("timings", {}).get("predicted_n")
        prompt_n = payload.get("timings", {}).get("prompt_n")
        if predicted is None:
            results[idx] = {"status": "parse_error", "error": "timings.predicted_n missing",
                            "latency_s": round(time.monotonic() - start, 3)}
            return
        results[idx] = {"status": "ok", "predicted_tokens": predicted, "prompt_tokens_recorded": prompt_n,
                        "latency_s": round(time.monotonic() - start, 3)}
    except Exception as exc:  # noqa: BLE001 - explicit recorded failure
        status = "timeout" if isinstance(exc, TimeoutError) else "failed"
        results[idx] = {"status": status, "error": f"{type(exc).__name__}: {exc}",
                        "latency_s": round(time.monotonic() - start, 3)}


def wait_for_server(port: int, boot_timeout: float) -> bool:
    deadline = time.monotonic() + boot_timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(2.0)
    return False


def run_serve(server: str, model: str, cfg: dict, port: int, extra: list[str], requests_per_level: int,
              timeout: float, boot_timeout: float) -> list[dict]:
    env = dict(os.environ)
    for key, value in cfg.get("runtime_controls", {}).get("hip", {}).items():
        env[key] = value
    argv = [server, "-m", model, "--port", str(port), *extra]
    public_argv = list(argv)
    public_argv[public_argv.index("-m") + 1] = os.path.basename(model)  # no local paths in reports
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    entries = []
    try:
        if not wait_for_server(port, boot_timeout):
            entries.append({"mode": "serve", "status": "failed",
                            "error": f"llama-server did not become healthy within {boot_timeout}s"})
            return entries
        url = f"http://127.0.0.1:{port}/completion"
        conc_cfg = cfg["concurrency"]
        prompt = filler_prompt(conc_cfg["request"]["prompt_tokens_default"])
        for level in conc_cfg["levels"]:
            results: dict[int, dict] = {}
            wave_start = time.monotonic()
            # each thread keeps `requests_per_level` requests in flight back
            # to back; all threads run concurrently
            def worker(idx: int) -> None:
                for rep in range(requests_per_level):
                    one_request(url, prompt, conc_cfg["request"]["gen_tokens_default"], timeout, results, idx * requests_per_level + rep)
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(level)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.monotonic() - wave_start
            attempts = results.values()
            total_tokens = sum(r["predicted_tokens"] for r in attempts if r.get("status") == "ok")
            entries.append({
                "mode": "serve",
                "concurrency": level,
                "method": "actual parallel HTTP requests against one llama-server",
                "requests_per_thread": requests_per_level,
                "wall_s": round(wall, 3),
                "aggregate_tokens_per_s": round(total_tokens / wall, 3) if wall > 0 else None,
                "per_request": [results[i] for i in sorted(results)],
                "statuses": {"ok": sum(1 for r in attempts if r["status"] == "ok"),
                             "failed": sum(1 for r in attempts if r["status"] == "failed"),
                             "timeout": sum(1 for r in attempts if r["status"] == "timeout"),
                             "parse_error": sum(1 for r in attempts if r["status"] == "parse_error")},
            })
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        if proc.returncode not in (0, -15, None):
            tail = (out or "")[-2000:].replace(model, os.path.basename(model))
            entries.append({"mode": "serve", "status": "failed", "error": f"llama-server exited {proc.returncode}",
                            "server_output_tail": tail})
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join("configs", "hardware-benchmarks.json"))
    ap.add_argument("--mode", choices=("bench", "serve"), required=True)
    ap.add_argument("--tool", default=None, help="llama-bench binary (bench mode)")
    ap.add_argument("--server", default=None, help="llama-server binary (serve mode)")
    ap.add_argument("--model", required=True, help="GGUF model path (recorded by basename only)")
    ap.add_argument("--output", default=None, help="write the JSON report here (default: stdout)")
    ap.add_argument("--port", type=int, default=18080, help="llama-server port (serve mode)")
    ap.add_argument("--bench-extra", nargs="*", default=None, help="extra llama-bench args, e.g. -ngl 999 -fa on")
    ap.add_argument("--server-extra", nargs="*", default=None, help="extra llama-server args, e.g. -ngl 999 -fa on -c 131072")
    ap.add_argument("--pp", type=int, nargs="*", default=None, help="subset of PP lengths (default: all from config)")
    ap.add_argument("--concurrency", type=int, nargs="*", default=None, help="subset of C levels (default: all from config)")
    ap.add_argument("--requests-per-level", type=int, default=None, help="requests per thread per level (default from config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode == "bench" and not args.tool:
        _fail("bench mode requires --tool (llama-bench)")
    if args.mode == "serve" and not args.server:
        _fail("serve mode requires --server (llama-server)")

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": args.mode,
        "model": os.path.basename(args.model),
        "model_recorded_as_basename": True,
        "config": os.path.basename(args.config),
        "runtime_controls": cfg.get("runtime_controls", {}),
        "results": [],
    }
    failures = 0
    if args.mode == "bench":
        extra = args.bench_extra if args.bench_extra is not None else cfg["llama_bench"]["extra_args_default"]
        timeout = cfg["timeouts_seconds"]["bench_per_point"]
        for pp in (args.pp if args.pp is not None else cfg["prompt_processing"]["tokens"]):
            entry = run_llama_bench(args.tool, args.model, pp, cfg, extra, timeout)
            report["results"].append(entry)
            if entry["status"] != "ok":
                failures += 1
    else:
        extra = args.server_extra if args.server_extra is not None else ["-ngl", "999", "-fa", "on"]
        entries = run_serve(args.server, args.model, cfg, args.port, extra,
                            args.requests_per_level if args.requests_per_level is not None
                            else cfg["concurrency"]["requests_per_level_default"],
                            cfg["timeouts_seconds"]["server_request"], cfg["timeouts_seconds"]["server_boot"])
        report["results"] = entries
        failures = sum(1 for e in entries if e.get("status") == "failed")

    text = json.dumps(report, sort_keys=True, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
