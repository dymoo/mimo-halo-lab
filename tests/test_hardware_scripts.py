"""Behavioral tests for the hardware/runtime scripts.

Covers observable contracts only:
- scripts/workspace.py: explicit MIMO_LAB requirement, mkdir-only layout,
  model-staging block on low free bytes (metadata unaffected), exclusive
  unique hash round-trip probe that never clobbers pre-existing files or
  symlinks, symlinked workspace paths rejected, status read-only (never
  creates an absent root), no path leakage into recorded output.
- scripts/hardware_capture.py: sanitized public summary (identifiers
  replaced, structure kept), timeouts recorded explicitly, sudo never
  auto-injected.
- scripts/benchmark_matrix.py: bench/serve modes label their methods
  honestly; failed points exit nonzero; model recorded by basename only.
- scripts/build_runtime.py: pin mismatch fails closed; plans are argv+env
  steps (never shell strings) with explicit per-backend flags, unique
  per-backend build dirs, --jobs > 0, --run requires --yes and refuses
  --skip-pin-check, and a shell-metacharacter path never reaches a shell.
- scripts/burst_replay.py: malformed schedules fail closed; plan mode emits
  no measurements.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def run_script(script: str, *args: str, env_extra: dict[str, str] | None = None):
    env = dict(os.environ)
    env.pop("MIMO_LAB", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["python3", str(SCRIPTS / script), *args],
        capture_output=True, text=True, env=env, timeout=120,
    )


class WorkspaceTests(unittest.TestCase):
    def test_requires_explicit_mimo_lab(self):
        proc = run_script("workspace.py", "status")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("MIMO_LAB", proc.stderr)

    def test_init_creates_layout_mkdir_only_and_smoke_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_script("workspace.py", "init", "--min-model-free-bytes", "1000",
                              env_extra={"MIMO_LAB": tmp})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            for d in ("source-models", "pruned-models", "traces/discovered", "traces/private",
                      "caches", "manifests", "metrics"):
                self.assertTrue((Path(tmp) / d).is_dir(), d)
            record = json.loads((Path(tmp) / "manifests" / "workspace.json").read_text())
            self.assertEqual(record["schema_version"], 1)
            self.assertEqual(record["hash_smoke"]["status"], "verified")
            self.assertFalse(record["root_recorded"])
            # no absolute paths anywhere in the recorded file or public summary
            self.assertNotIn(str(Path(tmp)), (Path(tmp) / "manifests" / "workspace.json").read_text())
            self.assertNotIn(str(Path(tmp)), proc.stdout)
            # the probe is unique, removed after verification, and leaves no litter
            self.assertTrue(record["hash_smoke"]["probe"].startswith("caches/.workspace-hash-probe-"))
            self.assertEqual(list((Path(tmp) / "caches").glob(".workspace-hash-probe*")), [])

    def test_low_free_space_blocks_model_staging_not_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            # absurdly high floor forces the block path deterministically
            proc = run_script("workspace.py", "init", "--min-model-free-bytes", str(2**70),
                              env_extra={"MIMO_LAB": tmp})
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertIn("MODEL STAGING BLOCKED", proc.stderr)
            self.assertFalse((Path(tmp) / "source-models").exists())
            self.assertTrue((Path(tmp) / "manifests").exists())  # metadata unaffected
            self.assertTrue((Path(tmp) / "traces" / "private").exists())
            record = json.loads((Path(tmp) / "manifests" / "workspace.json").read_text())
            self.assertEqual(record["model_staging"], "blocked")

    def test_smoke_roundtrip_verified_and_own_probe_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_script("workspace.py", "init", "--min-model-free-bytes", "1000",
                              env_extra={"MIMO_LAB": tmp})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            proc = run_script("workspace.py", "smoke", env_extra={"MIMO_LAB": tmp})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out = json.loads(proc.stdout)
            self.assertEqual(out["hash_smoke"]["status"], "verified")
            self.assertTrue(out["hash_smoke"]["probe"].startswith("caches/.workspace-hash-probe-"))
            # cleanup removes only the probe this run created
            self.assertEqual(list((Path(tmp) / "caches").glob(".workspace-hash-probe*")), [])

    def test_smoke_never_clobbers_existing_probe_file_or_symlink(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            proc = run_script("workspace.py", "init", "--min-model-free-bytes", "1000",
                              env_extra={"MIMO_LAB": tmp})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            fixed = Path(tmp) / "caches" / ".workspace-hash-probe"
            victim = Path(outside) / "victim.txt"

            # a pre-existing file at the old fixed probe name is never touched
            fixed.write_text("keep-me")
            proc = run_script("workspace.py", "smoke", env_extra={"MIMO_LAB": tmp})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(fixed.read_text(), "keep-me")

            # a symlink at the probe path is never followed or overwritten
            fixed.unlink()
            victim.write_text("outside-keep-me")
            fixed.symlink_to(victim)
            proc = run_script("workspace.py", "smoke", env_extra={"MIMO_LAB": tmp})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(victim.read_text(), "outside-keep-me")
            self.assertTrue(fixed.is_symlink())

    def test_init_rejects_symlinked_workspace_path(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            (Path(tmp) / "caches").symlink_to(outside, target_is_directory=True)
            proc = run_script("workspace.py", "init", "--min-model-free-bytes", "1000",
                              env_extra={"MIMO_LAB": tmp})
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn("symlink", proc.stderr)
            # nothing was written through the symlink into the outside directory
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_status_missing_root_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent = Path(tmp) / "absent-lab"
            proc = run_script("workspace.py", "status", env_extra={"MIMO_LAB": str(absent)})
            self.assertEqual(proc.returncode, 2)
            self.assertFalse(absent.exists())

    def test_status_is_read_only_on_existing_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = sorted(p.name for p in root.iterdir())
            proc = run_script("workspace.py", "status", env_extra={"MIMO_LAB": tmp})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            json.loads(proc.stdout)
            self.assertEqual(sorted(p.name for p in root.iterdir()), before)  # nothing created


class HardwareCaptureTests(unittest.TestCase):
    def test_sanitize_replaces_identifiers_keeps_structure(self):
        sys.path.insert(0, str(SCRIPTS))
        try:
            import hardware_capture as hc
        finally:
            sys.path.pop(0)
        text = ("MAC aa:bb:cc:dd:ee:ff ip 10.0.0.7 uuid 123e4567-e89b-12d3-a456-426614174000 "
                "Serial Number: ZZZ123456 home /Users/someone host hidden")
        out, counters = hc_sanitize(text, hostname="hidden")
        self.assertNotIn("aa:bb:cc:dd:ee:ff", out)
        self.assertNotIn("10.0.0.7", out)
        self.assertNotIn("123e4567", out)
        self.assertNotIn("ZZZ123456", out)
        self.assertNotIn("hidden", out)
        self.assertNotIn("/Users/someone", out)
        self.assertIn("__MAC_1__", out)
        self.assertIn("__USERHOME_1__", out)
        self.assertIn("__IPV4_1__", out)
        self.assertIn("__UUID_1__", out)
        self.assertIn("__SERIAL_1__", out)
        self.assertIn("__HOSTNAME_1__", out)
        self.assertEqual(counters["MAC"], 1)

    def test_run_records_timeout_explicitly(self):
        sys.path.insert(0, str(SCRIPTS))
        try:
            import hardware_capture as hc
        finally:
            sys.path.pop(0)
        res = hc.run_capture("python3 -c \"import time; time.sleep(5)\"", timeout=0.5)
        self.assertTrue(res["timed_out"])
        self.assertLess(res["duration_s"], 4)

    def test_no_auto_sudo(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmds = Path(tmp) / "cmds.txt"
            cmds.write_text("echo plain-ok\nsudo: echo sudo-marked\n")
            raw = Path(tmp) / "raw"
            proc = run_script("hardware_capture.py", "--commands", str(cmds), "--raw-dir", str(raw),
                              "--public", str(Path(tmp) / "public.json"))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            summary = json.loads((Path(tmp) / "public.json").read_text())
            by_cmd = {e["command"]: e for e in summary["results"]}
            self.assertEqual(by_cmd["echo plain-ok"]["status"], "ok")
            sudo_entry = by_cmd["echo sudo-marked"]
            self.assertTrue(sudo_entry["sudo_marked"])
            self.assertEqual(sudo_entry["status"], "ok")  # ran as written, unmodified
            self.assertTrue((raw / "echo_sudo-marked.txt").exists())


class BenchmarkMatrixTests(unittest.TestCase):
    def make_config(self, tmp: Path) -> Path:
        cfg = {
            "schema_version": 1,
            "prompt_processing": {"method": "llama-bench", "tokens": [8192],
                                  "labels": ["PP8K"], "interpretation": "batched pp"},
            "concurrency": {"method": "actual parallel HTTP requests", "levels": [1, 2],
                            "requests_per_level_default": 2,
                            "request": {"endpoint": "/completion", "gen_tokens_default": 2,
                                        "prompt_tokens_default": 8}},
            "runtime_controls": {"hip": {}},
            "llama_bench": {"output": "json", "extra_args_default": []},
            "timeouts_seconds": {"bench_per_point": 30, "server_boot": 10, "server_request": 30,
                                 "capture_command": 10},
            "soak": {"durations_hours": [1]},
        }
        path = tmp / "bench-config.json"
        path.write_text(json.dumps(cfg))
        return path

    def test_bench_mode_labels_not_concurrency_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = self.make_config(tmp)
            out = tmp / "report.json"
            proc = run_script("benchmark_matrix.py", "--mode", "bench", "--tool", "false",  # noqa: no such tool
                              "--model", "/some/private/path/model.gguf", "--config", str(cfg),
                              "--pp", "8192", "--output", str(out))
            self.assertNotEqual(proc.returncode, 0)
            report = json.loads(out.read_text())
            self.assertEqual(report["model"], "model.gguf")  # basename only
            self.assertNotIn("/some/private/path", out.read_text())
            self.assertEqual(report["results"][0]["status"], "failed")

    def test_serve_mode_uses_real_parallel_requests_and_reports_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # minimal fake llama-server: HTTP server answering /health and /completion
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    n = body["n_predict"]
                    payload = {"content": "x" * n, "timings": {"predicted_n": n, "prompt_n": 8}}
                    data = json.dumps(payload).encode()
                    time.sleep(0.05)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            port = server.server_address[1]
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                # happy path first: two genuinely concurrent requests against
                # the local fake server via a plain HTTP helper
                url = f"http://127.0.0.1:{port}/completion"
                results = {}
                workers = [threading.Thread(target=_req, args=(url, 3, results, i)) for i in range(2)]
                for w in workers:
                    w.start()
                for w in workers:
                    w.join()
                payloads = [results[i] for i in sorted(results)]
                self.assertEqual([p["timings"]["predicted_n"] for p in payloads], [3, 3])

                # a fake "llama-server" that exits immediately: wait_for_server
                # must fail explicitly rather than hang
                cfg = self.make_config(tmp)
                out = tmp / "serve.json"
                fake = tmp / "fake-server.sh"
                fake.write_text("#!/bin/sh\nexit 7\n")
                os.chmod(fake, 0o755)
                proc = run_script("benchmark_matrix.py", "--mode", "serve", "--server", str(fake),
                                  "--model", "m.gguf", "--config", str(cfg), "--port", str(port + 1),
                                  "--output", str(out))
                self.assertNotEqual(proc.returncode, 0)
                serve_report = json.loads(out.read_text())
                self.assertEqual(serve_report["model"], "m.gguf")
                self.assertEqual(serve_report["results"][0]["status"], "failed")
                self.assertIn("did not become healthy", serve_report["results"][0]["error"])
            finally:
                server.shutdown()

    def test_config_requires_schema_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"schema_version": 99}))
            proc = run_script("benchmark_matrix.py", "--mode", "bench", "--tool", "x",
                              "--model", "m.gguf", "--config", str(bad))
            self.assertEqual(proc.returncode, 2)


class BuildRuntimeTests(unittest.TestCase):
    def test_plan_refuses_pin_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "strix"
            (src / ".git").mkdir(parents=True)
            subprocess.run(["git", "init", "-q", str(src)], check=True)
            subprocess.run(["git", "-C", str(src), "remote", "add", "origin", "x"], check=True)
            # unborn HEAD: rev-parse fails -> explicit failure
            proc = run_script("build_runtime.py", "--source", str(src), "--backend", "vulkan")
            self.assertEqual(proc.returncode, 2)

    def plan_steps(self, backend: str, build_dir: str = "build-x", jobs: int = 8):
        sys.path.insert(0, str(SCRIPTS))
        try:
            import build_runtime as br
        finally:
            sys.path.pop(0)
        return br.backend_plan(backend, build_dir, jobs)

    def test_plan_prints_argv_steps_without_executing(self):
        proc = run_script("build_runtime.py", "--backend", "vulkan", "--skip-pin-check")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        plan = json.loads(proc.stdout)
        self.assertEqual(plan["action"], "plan")
        self.assertIsInstance(plan["commands"], list)
        for step in plan["commands"]:
            self.assertIsInstance(step["argv"], list)
            self.assertTrue(all(isinstance(a, str) for a in step["argv"]))
            self.assertIsInstance(step["env"], dict)
        self.assertEqual(plan["commands"][0]["argv"][0], "cmake")
        self.assertEqual(plan["commands"][-1]["argv"][0], "ctest")

    def test_steps_are_argv_not_shell_strings(self):
        steps = self.plan_steps("vulkan")
        self.assertIn("-DGGML_VULKAN=ON", steps[0]["argv"])
        self.assertIn("-DLLAMA_BUILD_TESTS=ON", steps[0]["argv"])
        self.assertIn("-DGGML_METAL=OFF", steps[0]["argv"])
        self.assertIn("-DGGML_HIP=OFF", steps[0]["argv"])
        # the build dir travels as one literal argv element, never interpolated
        self.assertEqual(steps[0]["argv"][1:3], ["-B", "build-x"])
        self.assertEqual(steps[1]["argv"][1:3], ["--build", "build-x"])

    def test_cpu_and_metal_flags_differ_explicitly(self):
        cpu_cfg = self.plan_steps("cpu")[0]["argv"]
        metal_cfg = self.plan_steps("metal")[0]["argv"]
        self.assertNotEqual(cpu_cfg, metal_cfg)
        for flag in ("-DGGML_METAL=OFF", "-DGGML_HIP=OFF", "-DGGML_VULKAN=OFF"):
            self.assertIn(flag, cpu_cfg)
        self.assertIn("-DGGML_METAL=ON", metal_cfg)
        self.assertIn("-DGGML_HIP=OFF", metal_cfg)
        self.assertIn("-DGGML_VULKAN=OFF", metal_cfg)

    def test_backend_build_dirs_are_unique_by_default(self):
        for backend, expected in (("cpu", "build-cpu"), ("metal", "build-metal"),
                                  ("vulkan", "build-vulkan"), ("hip", "build-hip")):
            proc = run_script("build_runtime.py", "--backend", backend, "--skip-pin-check")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            plan = json.loads(proc.stdout)
            self.assertEqual(plan["build_dir"], expected)
            self.assertIn(expected, plan["commands"][0]["argv"])

    def test_hip_plan_pins_gfx1151_and_records_env_without_substitution(self):
        proc = run_script("build_runtime.py", "--backend", "hip", "--skip-pin-check")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        plan = json.loads(proc.stdout)
        cfg = plan["commands"][0]
        self.assertIn("-DGGML_HIP=ON", cfg["argv"])
        self.assertIn("-DGPU_TARGETS=gfx1151", cfg["argv"])
        self.assertIn("-DGGML_METAL=OFF", cfg["argv"])
        self.assertIn("-DGGML_VULKAN=OFF", cfg["argv"])
        # env requirements are data, resolved only at --run (no $(...) in the plan)
        self.assertEqual(cfg["env"]["HIPCXX"], {"tool": "hipconfig", "flag": "-l", "suffix": "/clang"})
        self.assertEqual(cfg["env"]["HIP_PATH"], {"tool": "hipconfig", "flag": "-R", "suffix": ""})
        self.assertNotIn("$(", json.dumps(cfg["env"]))
        self.assertEqual(plan["runtime_env"]["HIP_LAUNCH_BLOCKING"], "1")

    def test_jobs_must_be_positive(self):
        proc = run_script("build_runtime.py", "--backend", "cpu", "--skip-pin-check", "--jobs", "0")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--jobs", proc.stderr)

    def test_run_requires_yes(self):
        proc = run_script("build_runtime.py", "--backend", "cpu", "--run")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--yes", proc.stderr)

    def test_skip_pin_check_refused_with_run(self):
        proc = run_script("build_runtime.py", "--backend", "cpu", "--run", "--yes", "--skip-pin-check")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--skip-pin-check", proc.stderr)

    def test_run_executes_argv_without_a_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = json.loads((REPO_ROOT / "manifests" / "upstreams.json").read_text())
            revision = next(e["revision"] for e in manifest["upstreams"]
                            if e["name"] == "strix-llama.cpp")
            src = tmp / "src"
            src.mkdir()
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            log = tmp / "argv.log"
            (bin_dir / "git").write_text(f"#!/bin/sh\necho {revision}\n")
            recorder = ('#!/bin/sh\n{\n  echo "---"\n'
                        '  for a in "$@"; do printf \'%s\\n\' "$a"; done\n'
                        '} >> "$ARGV_LOG"\nexit 0\n')
            for tool in ("cmake", "ctest"):
                (bin_dir / tool).write_text(recorder)
            for script in bin_dir.iterdir():
                os.chmod(script, 0o755)

            # shell metacharacters in the build dir: executed via a shell, the
            # `;` would run `touch` as a separate command
            marker = tmp / "pwned"
            evil_dir = f"build;touch {marker}"
            proc = run_script(
                "build_runtime.py", "--run", "--yes", "--backend", "cpu",
                "--source", str(src), "--build-dir", evil_dir,
                env_extra={"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                           "ARGV_LOG": str(log)},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(marker.exists(), "path must never be handed to a shell")
            invocations = [block.splitlines()
                           for block in log.read_text().split("---\n")[1:]]
            configure, build, test = invocations
            # $@ excludes argv[0]; each token is one literal element
            self.assertEqual(configure[:2], ["-B", evil_dir])
            self.assertEqual(build[:3], ["--build", evil_dir, "--config"])
            self.assertEqual(test[:2], ["--test-dir", evil_dir])


class BurstReplayTests(unittest.TestCase):
    def test_malformed_schedule_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"schema_version": 1, "events": [{"t_s": -1}]}))
            proc = run_script("burst_replay.py", "--schedule", str(bad))
            self.assertEqual(proc.returncode, 2)

    def test_plan_mode_emits_no_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = Path(tmp) / "s.json"
            sched.write_text(json.dumps({"schema_version": 1, "scale": 1.0, "events": [
                {"t_s": 0.0, "kind": "shell", "prompt_tokens": 64, "gen_tokens": 8},
                {"t_s": 2.5, "kind": "test", "prompt_tokens": 128, "gen_tokens": 8}]}))
            proc = run_script("burst_replay.py", "--schedule", str(sched))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            plan = json.loads(proc.stdout)
            self.assertEqual(plan["action"], "plan")
            self.assertEqual(plan["event_count"], 2)
            self.assertEqual(plan["events"][1]["t_s"], 2.5)
            self.assertNotIn("latency_s", json.dumps(plan))

    def test_replay_measures_real_requests(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = {"content": "y", "timings": {"predicted_n": body["n_predict"], "prompt_n": 5}}
                data = json.dumps(payload).encode()
                time.sleep(0.02)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                sched = Path(tmp) / "s.json"
                sched.write_text(json.dumps({"schema_version": 1, "events": [
                    {"t_s": 0.0, "kind": "shell", "prompt_tokens": 16, "gen_tokens": 4},
                    {"t_s": 0.3, "kind": "compile", "prompt_tokens": 16, "gen_tokens": 4}]}))
                proc = run_script("burst_replay.py", "--schedule", str(sched),
                                  "--server-url", f"http://127.0.0.1:{port}/completion",
                                  "--timeout", "30")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                report = json.loads(proc.stdout)
                self.assertEqual(report["successful_requests"], 2)
                self.assertEqual([e["kind"] for e in report["events"]], ["shell", "compile"])
                self.assertGreater(report["replay_wall_s"], 0.2)
        finally:
            server.shutdown()


def _req(url: str, n: int, results: dict, idx: int) -> None:
    import json as _json
    body = _json.dumps({"prompt": "x", "n_predict": n}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        results[idx] = _json.loads(resp.read().decode())


def hc_sanitize(text: str, hostname: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("hardware_capture", SCRIPTS / "hardware_capture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sanitize(text, hostname)


if __name__ == "__main__":
    unittest.main()
