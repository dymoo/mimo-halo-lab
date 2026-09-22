#!/usr/bin/env python3
"""Portable hardware capture (stdlib only, Python >= 3.11).

Runs an operator-supplied list of shell commands with per-command timeouts,
saves raw output locally, and emits a sanitized public summary.

Safety rules:
- Commands come only from the --commands file; nothing is invented or
  auto-discovered. Lines starting with '#' are comments.
- sudo is never injected automatically. A line may be marked as sudo-dependent
  with a leading 'sudo ' (or 'sudo:' prefix form); the tool runs it as-is.
  If it fails (no tty, no password), that single command is recorded as
  failed and the run continues.
- Raw output (full stdout/stderr, exit codes, durations) stays on the local
  machine. The public summary replaces machine identifiers with deterministic
  placeholder tokens so structure is preserved but nothing identifies the
  host: MACs, IPv4/IPv6, UUIDs, serial numbers, hostnames and user home paths.
- Each command runs with a timeout (default 30 s); a timeout is an explicit,
  recorded outcome, never silent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

REPLACERS = [
    ("MAC", re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("IPV6", re.compile(r"\b(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}\b")),
    ("UUID", re.compile(r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b")),
    ("SERIAL", re.compile(r"(?i)\b(serial(?:[\s_-]*(?:number|no\.?)?)|serial=|sn=)\s*[:=]?\s*([0-9A-Za-z_/+\-]{4,})")),
    ("HOSTNAME", None),  # handled after regex replacers, needs the resolved hostname
    ("USERHOME", re.compile(r"(?i)(?<![\w/.-])(/home/[\w.]+(?:/[\w.]+)*|/Users/[\w.]+(?:/[\w.]+)*)")),
]


def _fail(message: str) -> None:
    print(f"hardware_capture: {message}", file=sys.stderr)
    raise SystemExit(2)


def sanitize(text: str, hostname: str) -> tuple[str, dict[str, int]]:
    """Replace identifiers with deterministic __KIND_n__ tokens."""
    counters: dict[str, int] = {}
    cache: dict[str, str] = {}

    def replace(kind: str, value: str) -> str:
        if value not in cache:
            counters[kind] = counters.get(kind, 0) + 1
            cache[value] = f"__{kind}_{counters[kind]}__"
        return cache[value]

    out = text
    for kind, pattern in REPLACERS:
        if pattern is None:
            continue
        if kind == "SERIAL":
            out = pattern.sub(lambda m: f"{m.group(1)} {replace('SERIAL', m.group(2))}", out)
        else:
            out = pattern.sub(lambda m, k=kind: replace(k, m.group(0)), out)
    if hostname:
        out = out.replace(hostname, "__HOSTNAME_1__")
    return out, counters


def load_commands(path: str) -> list[dict]:
    commands = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            sudo = line.startswith("sudo ") or line.startswith("sudo:")
            if line.startswith("sudo:"):
                line = line[len("sudo:"):].strip()
            try:
                argv = shlex.split(line)
            except ValueError as exc:
                _fail(f"commands file line {lineno}: cannot parse ({exc})")
            if not argv:
                _fail(f"commands file line {lineno}: empty command")
            commands.append({"argv": argv, "line": line, "sudo": sudo, "lineno": lineno})
    if not commands:
        _fail(f"no commands found in {path}")
    return commands


def run_capture(command: str, timeout: float) -> dict:
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(command, shell=True, executable="/bin/sh",
                              capture_output=True, text=True, timeout=timeout)
        result = {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        result = {
            "exit_code": None,
            "stdout": exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr": exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
            "timed_out": True,
        }
    result["duration_s"] = round(time.monotonic() - start, 3)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commands", required=True, help="text file of commands, one per line")
    ap.add_argument("--raw-dir", default=None,
                    help="directory for raw capture output (default: $MIMO_LAB/traces/private/hardware-evidence; "
                         "raw output contains identifiers and must stay off shared storage)")
    ap.add_argument("--public", default=None, help="path for the sanitized public summary JSON (default: stdout only)")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-command timeout seconds (default 30)")
    ap.add_argument("--summary-bytes", type=int, default=2000, help="max stdout/stderr bytes kept in the public summary")
    args = ap.parse_args()

    raw_dir = args.raw_dir or os.path.join(os.environ.get("MIMO_LAB", ""), "traces", "private", "hardware-evidence")
    if not raw_dir or raw_dir == os.path.join("", "traces", "private", "hardware-evidence"):
        _fail("no raw output destination: pass --raw-dir or set MIMO_LAB")
    os.makedirs(raw_dir, exist_ok=True)

    commands = load_commands(args.commands)
    hostname = os.uname().nodename if hasattr(os, "uname") else ""
    entries = []
    failures = 0

    for cmd in commands:
        res = run_capture(cmd["line"], args.timeout)
        # public-facing fields are sanitized end to end: the command line, the
        # slug and the captured output all go through the same replacer
        san_line, _ = sanitize(cmd["line"], hostname)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", san_line)[:80] or f"line{cmd['lineno']}"
        raw_path = os.path.join(raw_dir, f"{slug}.txt")
        if os.path.exists(raw_path):
            _fail(f"refusing to overwrite existing raw capture file {slug}.txt")
        with open(raw_path, "w", encoding="utf-8") as fh:
            fh.write(f"# command: {cmd['line']}\n# exit_code: {res['exit_code']}\n"
                     f"# timed_out: {res['timed_out']}\n# duration_s: {res['duration_s']}\n"
                     f"# --- stdout ---\n{res['stdout']}\n# --- stderr ---\n{res['stderr']}\n")
        status = "ok" if res["exit_code"] == 0 else ("timeout" if res["timed_out"] else "failed")
        if status != "ok":
            failures += 1
        san_out, _ = sanitize(res["stdout"], hostname)
        san_err, _ = sanitize(res["stderr"], hostname)
        entries.append({
            "command": san_line,
            "sudo_marked": cmd["sudo"],
            "exit_code": res["exit_code"],
            "timed_out": res["timed_out"],
            "duration_s": res["duration_s"],
            "status": status,
            "raw_file": os.path.basename(raw_path),
            "stdout_sanitized": san_out[: args.summary_bytes],
            "stderr_sanitized": san_err[: args.summary_bytes],
        })

    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commands_file": os.path.basename(args.commands),
        "timeout_s": args.timeout,
        "counts": {"total": len(entries), "ok": sum(1 for e in entries if e["status"] == "ok"),
                   "failed": sum(1 for e in entries if e["status"] == "failed"),
                   "timeout": sum(1 for e in entries if e["status"] == "timeout")},
        "results": entries,
        "note": "stdout/stderr in this summary are sanitized: MAC/IP/UUID/serial/hostname/home-path "
                "values replaced by __KIND_n__ tokens; full raw output remains local only",
    }
    text = json.dumps(summary, sort_keys=True, indent=2)
    if args.public:
        with open(args.public, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
