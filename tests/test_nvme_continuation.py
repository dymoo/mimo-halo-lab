"""Real-HTTP regression for NVMe/SWA whole-context continuation.

Consumer of scripts/state_restore_harness.py: runs the instrumented
hot-vs-save/erase/restore comparison against an EXTERNALLY managed
llama-server and validates the JSON report, not just an HTTP status.

Skips ONLY when the explicit external endpoint env is absent — the server
is never spawned here (Main provides the real endpoint for the RED run).

    NVME_CONTINUATION_BASE_URL=http://127.0.0.1:8080 \
        python3 -m unittest discover -s tests -p 'test_nvme_continuation.py' -v

Optional env:
    NVME_CONTINUATION_VARIANT      small (default) | slow   (100K+4K)
    NVME_CONTINUATION_TIMEOUT_S    harness subprocess timeout (default 7200)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "scripts" / "state_restore_harness.py"
BASE_URL_ENV = "NVME_CONTINUATION_BASE_URL"
VARIANT_ENV = "NVME_CONTINUATION_VARIANT"
TIMEOUT_ENV = "NVME_CONTINUATION_TIMEOUT_S"


@unittest.skipUnless(
    os.environ.get(BASE_URL_ENV),
    f"explicit external server env {BASE_URL_ENV} not set; "
    "Main provides the endpoint (harness never spawns a server)",
)
class NvmeContinuationTest(unittest.TestCase):
    def _run_harness(self) -> tuple[subprocess.CompletedProcess, dict]:
        variant = os.environ.get(VARIANT_ENV, "small")
        timeout = int(os.environ.get(TIMEOUT_ENV, "7200"))
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            proc = subprocess.run(
                [sys.executable, str(HARNESS), "run",
                 "--base-url", os.environ[BASE_URL_ENV],
                 "--variant", variant,
                 "--report", str(report_path)],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(REPO_ROOT),
            )
            self.assertTrue(
                report_path.exists(),
                f"harness wrote no report (rc={proc.returncode})\n"
                f"stdout:\n{proc.stdout[-4000:]}\n"
                f"stderr:\n{proc.stderr[-4000:]}",
            )
            report = json.loads(report_path.read_text())
            return proc, report

    def test_hot_vs_restored_continuation_all_assertions_pass(self) -> None:
        proc, report = self._run_harness()

        # Transport/fixture failures carry `error` instead of a summary.
        self.assertNotIn(
            "error", report,
            f"harness transport failure: {report.get('error')}\n"
            f"stderr:\n{proc.stderr[-4000:]}",
        )

        failed = [c for c in report["assertions"] if not c["ok"]]
        self.assertEqual(
            failed, [],
            f"assertion failures (rc={proc.returncode}): "
            f"{json.dumps(failed, indent=2)}\n"
            f"summary={report['summary']}\nstderr:\n{proc.stderr[-4000:]}",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        self.assertTrue(report["summary"]["ok"])

        # Instrumented observations, not status-only checks. The cold probe
        # is config-gated, so require it only when the config enables it.
        cfg = json.loads(
            (REPO_ROOT / "configs" / "state-restore-fixture.json")
            .read_text())
        required = ["warm", "hot_continuation", "restore",
                    "restored_continuation"]
        if cfg["variants"].get(os.environ.get(VARIANT_ENV, "small"),
                               {}).get("cold_probe"):
            required.append("cold_probe_after_erase")
        obs = report["observations"]
        for key in required:
            self.assertIn(key, obs, f"missing observation {key}")
        hot = obs["hot_continuation"]
        restored = obs["restored_continuation"]
        for label, rec in (("hot", hot), ("restored", restored)):
            self.assertIn("prompt_n", rec["timings"], label)
            self.assertIn("cache_n", rec["timings"], label)
            self.assertIn("llamacpp:prompt_tokens_total",
                          rec["metrics_delta"], label)
            self.assertIn("llamacpp:prompt_tokens_cached_total",
                          rec["metrics_delta"], label)

        # Physical prefill suffix bound (named constants from the config).
        bound = report["bounds"]["restored_prompt_n_max"]
        self.assertLessEqual(restored["timings"]["prompt_n"], bound,
                             f"restored prompt_n exceeds suffix bound {bound}")
        self.assertGreaterEqual(restored["timings"]["prompt_n"],
                                report["bounds"]["prompt_n_min"])

        # Hot/restored parity under identical request bytes.
        self.assertEqual(hot["request"], restored["request"])
        comparison = report["comparison"]
        self.assertTrue(comparison["prompt_n"]["equal"])
        self.assertTrue(comparison["generated_tokens"]["equal"])
        self.assertTrue(comparison["identical_request"])

        # Cumulative physical counter identity.
        metrics = report["metrics"]
        self.assertEqual(metrics["global_prompt_tokens_delta"],
                         metrics["sum_per_request_prompt_n"])
        self.assertEqual(metrics["global_prompt_tokens_cached_delta"],
                         metrics["sum_per_request_cache_n"])


if __name__ == "__main__":
    unittest.main()
