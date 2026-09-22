#!/usr/bin/env python3
"""Pinned runtime build driver (stdlib only, Python >= 3.11).

Plans (default) or runs (--run --yes) the pinned strix-llama.cpp CMake builds
per backend, plus the matching unit-test commands. Default mode only prints
the plan — each step as an explicit argv list plus env requirements, never a
shell command string. --run executes every step argv with shell=False and
fails closed if the checkout does not match the pin recorded in
manifests/upstreams.json; --skip-pin-check is plan-only and is refused
together with --run. Each backend defaults to its own build directory
(build-cpu, build-metal, build-vulkan, build-hip), --jobs must be > 0, and no
tools are ever installed implicitly.

Backends select exactly one accelerator (explicit -D flags; the two others
are always passed as OFF, and CPU disables all three):
- cpu      GGML_METAL/HIP/VULKAN all OFF (Zen 5 AVX-512 baseline sanity)
- metal    GGML_METAL=ON — macOS correctness build, deferred to Main; --run
           requires the explicit --yes confirmation
- vulkan   GGML_VULKAN=ON (RADV on the Strix Halo iGPU is the default
           recommendation of the strix tree)
- hip      GGML_HIP=ON with GPU_TARGETS=gfx1151; HIPCXX/HIP_PATH requirements
           are recorded in the plan without substitution and resolved by
           separate `hipconfig` subprocesses only at --run time.
           HIP_LAUNCH_BLOCKING=1 is a documented runtime correctness
           control, not a build flag

This tool never builds the ROCmFP4 patched fork; the patch flow is owned by
upstream rocmfp4 scripts (apply-rocmfp4.sh, build-strix-rocmfp4-mtp.sh) and is
documented in docs/hardware-bringup.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "manifests", "upstreams.json")
PIN_NAME = "strix-llama.cpp"


def _fail(message: str) -> None:
    print(f"build_runtime: {message}", file=sys.stderr)
    raise SystemExit(2)


def pinned_revision() -> str:
    with open(MANIFEST, encoding="utf-8") as fh:
        manifest = json.load(fh)
    for entry in manifest.get("upstreams", []):
        if entry.get("name") == PIN_NAME:
            return entry["revision"]
    _fail(f"{PIN_NAME} not found in {MANIFEST}")


def verify_pin(source: str) -> str:
    try:
        head = subprocess.run(["git", "-C", source, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        _fail(f"cannot resolve HEAD of {source}: {exc}")
    expected = pinned_revision()
    if head != expected:
        _fail(f"pin mismatch in {source}: HEAD {head} != pinned {expected}")
    return head


# Explicit backend selection: the chosen accelerator ON, the others OFF
# (CPU disables all three). Keys map to the strix-llama.cpp CMake options.
BACKEND_GPU_FLAGS: dict[str, dict[str, str]] = {
    "cpu": {"GGML_METAL": "OFF", "GGML_HIP": "OFF", "GGML_VULKAN": "OFF"},
    "metal": {"GGML_METAL": "ON", "GGML_HIP": "OFF", "GGML_VULKAN": "OFF"},
    "vulkan": {"GGML_METAL": "OFF", "GGML_HIP": "OFF", "GGML_VULKAN": "ON"},
    "hip": {"GGML_METAL": "OFF", "GGML_HIP": "ON", "GGML_VULKAN": "OFF"},
}
# Env requirements recorded as data (no shell substitution in the plan);
# resolved by separate subprocesses only when --run executes a HIP build.
HIP_ENV_REQUIREMENTS: dict[str, dict[str, str]] = {
    "HIPCXX": {"tool": "hipconfig", "flag": "-l", "suffix": "/clang"},
    "HIP_PATH": {"tool": "hipconfig", "flag": "-R", "suffix": ""},
}


def backend_plan(backend: str, build_dir: str, jobs: int) -> list[dict]:
    """Steps as {argv, env}; never shell command strings."""
    configure = ["cmake", "-B", build_dir,
                 "-DCMAKE_BUILD_TYPE=Release",
                 "-DLLAMA_BUILD_TESTS=ON"]
    configure += [f"-D{flag}={value}" for flag, value in BACKEND_GPU_FLAGS[backend].items()]
    env: dict[str, dict[str, str]] = {}
    if backend == "hip":
        configure.append("-DGPU_TARGETS=gfx1151")
        env = dict(HIP_ENV_REQUIREMENTS)
    build = ["cmake", "--build", build_dir, "--config", "Release", "-j", str(jobs)]
    test = ["ctest", "--test-dir", build_dir, "--output-on-failure"]  # unit tests
    return [{"argv": configure, "env": dict(env)},
            {"argv": build, "env": dict(env)},
            {"argv": test, "env": {}}]


def resolve_env(requirements: dict[str, dict[str, str]]) -> dict[str, str] | None:
    """Resolve recorded env requirements via separate subprocesses (HIP --run only)."""
    if not requirements:
        return None
    env = dict(os.environ)
    for key, spec in requirements.items():
        tool, flag = spec["tool"], spec["flag"]
        try:
            out = subprocess.run([tool, flag], capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            _fail(f"cannot resolve {key}: `{tool} {flag}` failed: {exc}")
        env[key] = out + spec.get("suffix", "")
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.path.join("upstreams", PIN_NAME))
    ap.add_argument("--backend", choices=("cpu", "metal", "vulkan", "hip"), required=True)
    ap.add_argument("--build-dir", default=None,
                    help="build directory (default: build-<backend> inside the source tree)")
    ap.add_argument("--jobs", type=int, default=16, help="parallel build jobs (default 16; must be > 0)")
    ap.add_argument("--run", action="store_true", help="execute instead of only planning")
    ap.add_argument("--yes", action="store_true", help="explicit confirmation required for --run")
    ap.add_argument("--skip-pin-check", action="store_true", help="plan-only convenience; refused together with --run")
    args = ap.parse_args()

    if args.run and not args.yes:
        _fail("--run requires --yes (builds are never started by accident)")
    if args.skip_pin_check and args.run:
        _fail("--skip-pin-check cannot be combined with --run")
    if args.jobs <= 0:
        _fail("--jobs must be a positive integer")
    build_dir = args.build_dir or f"build-{args.backend}"

    if args.run:
        verify_pin(args.source)
        for step in backend_plan(args.backend, build_dir, args.jobs):
            print(f"+ {shlex.join(step['argv'])}", file=sys.stderr)
            try:
                subprocess.run(step["argv"], shell=False, check=True, cwd=args.source,
                               env=resolve_env(step["env"]))
            except FileNotFoundError:
                _fail(f"command not found: {step['argv'][0]}")
            except subprocess.CalledProcessError as exc:
                _fail(f"command exited {exc.returncode}: {shlex.join(step['argv'])}")
        print(json.dumps({"schema_version": 2, "action": "build", "backend": args.backend,
                          "source": os.path.basename(args.source), "pin": pinned_revision(),
                          "status": "done", "at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                         sort_keys=True))
        return

    if not args.skip_pin_check:
        verify_pin(args.source)
    else:
        print("build_runtime: pin check skipped (plan-only)", file=sys.stderr)
    plan = {
        "schema_version": 2,
        "action": "plan",
        "backend": args.backend,
        "source": args.source,
        "build_dir": build_dir,
        "jobs": args.jobs,
        "pin": pinned_revision(),
        "commands": backend_plan(args.backend, build_dir, args.jobs),
        "runtime_env": ({"HIP_LAUNCH_BLOCKING": "1", "HSA_OVERRIDE_GFX_VERSION": "11.5.1",
                         "GGML_HIP_ENABLE_UNIFIED_MEMORY": "1"} if args.backend == "hip" else {}),
        "runtime_env_note": "runtime correctness control from strix-llama.cpp README/CI; a control, not a performance claim",
        "mac_note": "Mac (Metal) build is deferred to Main; the exact commands above are the handoff",
    }
    print(json.dumps(plan, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
