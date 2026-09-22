"""Red tracer for the NVMe retention policy CLI (mimo_halo.state_tier.retention).

Consumer-level test of the approved observation/decision seam from
local://nvme-implementation-contract.md:

    PYTHONPATH=src python3 -m mimo_halo.state_tier.retention \
        --config CFG --input OBSERVATIONS.json --output DECISIONS.json

Scenario (fixed observation clock, no wall-time reads): hot-slot admission
pressure with an interactive incoming request while one session is actively
RUNNING and a background WAITING session has been idle only 20 seconds --
far inside the 180-second hot idle grace. Per contract, pressure/admission
lets an eligible WAITING state yield before grace, chosen lowest-priority /
longest-idle first, while RUNNING slots are never eviction victims. The
observable decision must therefore be hot_evictions == [background-waiter].

The config carries the complete contract defaults in decimal bytes. Config,
input and output live in a temporary directory outside the Git working tree.
Everything is asserted from the CLI's DECISIONS.json output file via the real
`python3 -m` command target -- no private helper wiring, no mocks. Written
first for Main's red run; implementation follows only after Main confirms red.
Main confirmed red (ModuleNotFoundError); the implementation and the
regression cases appended below then landed through the same CLI seam:
no-pressure 20-minute idleness, short-wait non-churn, active-decode
protection, soft priority override of the grace window, UMA byte-demand
override of grace without an incoming priority, prefix non-expiry,
warm-quota LRU demotion to cold, explicit pressure-sizing rationales
and invalid-input refusal.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fixed clock: decision idleness must derive from observation["now"],
# never from the wall clock.
NOW = 1_700_000_000

ACTIVE_SESSION = "active-slot"
WAITING_SESSION = "background-waiter"

# Complete defaults from the contract, decimal bytes.
CONFIG = {
    "schema_version": 1,
    "hot_idle_grace_seconds": 180,
    "warm_nvme": {
        "preferred_retention_seconds": 3600,
        "max_size_bytes": 512000000000,
    },
    "cold_nvme": {
        "preferred_retention_seconds": 86400,
        "max_size_bytes": 1000000000000,
    },
    "prefix_cache": {
        "preferred_retention_seconds": None,
        "eviction": "LRU",
        "max_size_bytes": 500000000000,
    },
    "quick_reactivation_window_seconds": 30,
}


class RetentionCliPressureTest(unittest.TestCase):
    def test_pressure_admission_evicts_idle_waiting_preserves_running(self):
        observations = {
            "schema_version": 1,
            "now": NOW,
            "pressure": {
                "needs_hot_slot": True,
                "uma_pressure": False,
                "disk_pressure": False,
                "required_free_bytes": 0,
            },
            "incoming_priority": 0,  # interactive admission
            "sessions": [
                {
                    "session_id": ACTIVE_SESSION,
                    "lifecycle": "RUNNING",
                    "tier": "HOT",
                    "priority": 0,
                    "last_accessed": NOW,
                    "state_bytes": 8000000000,
                    "waiting_reason": None,
                    "expected_wakeup": None,
                },
                {
                    "session_id": WAITING_SESSION,
                    "lifecycle": "WAITING",
                    "tier": "HOT",
                    "priority": 3,
                    "last_accessed": NOW - 20,  # idle 20s, grace is 180s
                    "state_bytes": 3000000000,
                    "waiting_reason": "tool_wait",
                    "expected_wakeup": None,
                },
            ],
            "prefix_entries": [],
        }

        # System temp dir: config/input/output stay outside the Git tree.
        with tempfile.TemporaryDirectory(prefix="mimo-halo-retention-") as tmp:
            tmp_dir = Path(tmp)
            config_path = tmp_dir / "state-retention.json"
            input_path = tmp_dir / "observations.json"
            output_path = tmp_dir / "decisions.json"
            config_path.write_text(json.dumps(CONFIG, indent=2) + "\n")
            input_path.write_text(json.dumps(observations, indent=2) + "\n")

            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT / "src")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mimo_halo.state_tier.retention",
                    "--config",
                    str(config_path),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=(
                    "retention CLI exited nonzero\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                ),
            )
            self.assertTrue(output_path.is_file(), "CLI wrote no DECISIONS.json")

            decision = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(decision.get("schema_version"), 1)
            # Exact expected decision: the 20-second-idle background WAITING
            # session yields despite the 180s grace because admission pressure
            # demands a hot slot, and the active RUNNING session must be
            # preserved (absent from hot_evictions).
            self.assertEqual(decision.get("hot_evictions"), [WAITING_SESSION])


# ---------------------------------------------------------------------------
# Regression coverage for the implemented policy. Every case drives the same
# public CLI seam as the tracer above: config/input/output files in a temp
# directory outside the Git tree, `python3 -m mimo_halo.state_tier.retention`
# via subprocess with PYTHONPATH=src, all assertions on DECISIONS.json.

DEFAULT_CONFIG = REPO_ROOT / "configs" / "state-retention.json"


def _run_retention(config, observations):
    """Run the literal CLI; return (proc, decision-or-None).

    ``config`` is either an existing config path (the shipped default) or a
    config object written into the temporary directory. On exit 0 the
    decision file must exist and is parsed; on failure the decision file
    must not exist (fail closed).
    """
    with tempfile.TemporaryDirectory(prefix="mimo-halo-retention-") as tmp:
        tmp_dir = Path(tmp)
        if isinstance(config, Path):
            config_path = config
        else:
            config_path = tmp_dir / "state-retention.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        input_path = tmp_dir / "observations.json"
        output_path = tmp_dir / "decisions.json"
        input_path.write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mimo_halo.state_tier.retention",
                "--config",
                str(config_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0:
            if not output_path.is_file():
                raise AssertionError(
                    "retention CLI exited 0 but wrote no DECISIONS.json\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            return proc, json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists():
            raise AssertionError(
                "retention CLI failed but still wrote DECISIONS.json\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return proc, None


def _session(
    session_id,
    *,
    lifecycle,
    tier,
    priority,
    idle,
    state_bytes=1_000_000_000,
    waiting_reason=None,
    expected_wakeup=None,
):
    return {
        "session_id": session_id,
        "lifecycle": lifecycle,
        "tier": tier,
        "priority": priority,
        "last_accessed": NOW - idle,
        "state_bytes": state_bytes,
        "waiting_reason": waiting_reason,
        "expected_wakeup": expected_wakeup,
    }


def _observation(
    sessions,
    *,
    pressure=None,
    incoming_priority=None,
    prefix_entries=None,
):
    return {
        "schema_version": 1,
        "now": NOW,
        "pressure": pressure
        if pressure is not None
        else {
            "needs_hot_slot": False,
            "uma_pressure": False,
            "disk_pressure": False,
            "required_free_bytes": 0,
        },
        "incoming_priority": incoming_priority,
        "sessions": sessions,
        "prefix_entries": prefix_entries if prefix_entries is not None else [],
    }


def _slot_pressure():
    return {
        "needs_hot_slot": True,
        "uma_pressure": False,
        "disk_pressure": False,
        "required_free_bytes": 0,
    }


def _actions(decision):
    return {entry["session_id"]: entry for entry in decision["retention_actions"]}


class RetentionPolicyRegressionTest(unittest.TestCase):
    """Consumer regressions: no-pressure keep, grace/priority/UMA demand, tier quota, fail-closed."""

    def test_no_pressure_twenty_minute_idle_stays_hot(self):
        sessions = [
            _session("idle-twenty-min", lifecycle="WAITING", tier="HOT", priority=3, idle=1200),
            _session("warm-recent", lifecycle="NVME_RESIDENT", tier="WARM", priority=2, idle=1200),
        ]
        _, decision = _run_retention(DEFAULT_CONFIG, _observation(sessions))
        self.assertEqual(decision.get("schema_version"), 1)
        self.assertEqual(decision["hot_evictions"], [])
        actions = _actions(decision)
        self.assertEqual(actions["idle-twenty-min"]["action"], "keep")
        self.assertEqual(actions["warm-recent"]["action"], "keep")
        # The shipped config's exact defaults are echoed as effective_config.
        self.assertEqual(
            decision["effective_config"],
            json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8")),
        )

    def test_short_wait_does_not_churn_without_pressure(self):
        sessions = [
            _session("compiler-10s", lifecycle="WAITING", tier="HOT", priority=1, idle=10),
            _session("compiler-30s", lifecycle="WAITING", tier="HOT", priority=2, idle=30),
        ]
        _, decision = _run_retention(DEFAULT_CONFIG, _observation(sessions))
        self.assertEqual(decision["hot_evictions"], [])
        # Exactly one action per session, input order preserved.
        self.assertEqual(
            [entry["session_id"] for entry in decision["retention_actions"]],
            ["compiler-10s", "compiler-30s"],
        )
        for entry in decision["retention_actions"]:
            self.assertEqual(entry["action"], "keep")

    def test_active_running_slot_is_never_a_victim(self):
        # Idle far past the 180s grace: RUNNING is still never a victim.
        sessions = [_session("decoder", lifecycle="RUNNING", tier="HOT", priority=3, idle=600)]
        obs = _observation(sessions, pressure=_slot_pressure(), incoming_priority=0)
        _, decision = _run_retention(DEFAULT_CONFIG, obs)
        self.assertEqual(decision["hot_evictions"], [])
        self.assertIn("slot admission demand unserved", decision["rationale"])
        action = _actions(decision)["decoder"]
        self.assertEqual(action["action"], "keep")
        self.assertIn("protected lifecycle", action["reason"])

    def test_soft_priority_override_background_admission_respects_grace(self):
        sessions = [
            _session("short-wait", lifecycle="WAITING", tier="HOT", priority=3, idle=20),
            _session("long-wait", lifecycle="WAITING", tier="HOT", priority=3, idle=300),
        ]
        # Background admission (3) does not outrank an equal-priority waiter,
        # so grace still protects the 20s short wait; only the grace-elapsed
        # waiter (oldest eligible, same rank) yields. Interactive admission
        # overriding grace for a 20s waiter is covered by the tracer above.
        obs = _observation(sessions, pressure=_slot_pressure(), incoming_priority=3)
        _, decision = _run_retention(DEFAULT_CONFIG, obs)
        self.assertEqual(decision["hot_evictions"], ["long-wait"])
        self.assertEqual(_actions(decision)["short-wait"]["action"], "keep")

    def test_uma_byte_demand_overrides_grace_without_incoming_priority(self):
        # Real UMA pressure with a positive byte target and NO admission
        # pending (incoming_priority null): positive byte demand overrides
        # the 180s grace, so the recent low-priority waiter yields before
        # grace; the RUNNING decoder is never a victim under any pressure.
        sessions = [
            _session(
                "decoder",
                lifecycle="RUNNING",
                tier="HOT",
                priority=0,
                idle=4,
                state_bytes=8_000_000_000,
            ),
            _session(
                "recent-bg-wait",
                lifecycle="WAITING",
                tier="HOT",
                priority=3,
                idle=20,  # idle 20s, grace is 180s
                state_bytes=3_000_000_000,
            ),
        ]
        uma = {
            "needs_hot_slot": False,
            "uma_pressure": True,
            "disk_pressure": False,
            "required_free_bytes": 3_000_000_000,
        }
        _, decision = _run_retention(DEFAULT_CONFIG, _observation(sessions, pressure=uma))
        self.assertEqual(decision["hot_evictions"], ["recent-bg-wait"])
        actions = _actions(decision)
        # Yielding from HOT is not snapshot deletion: state stays restorable.
        self.assertEqual(actions["recent-bg-wait"]["action"], "keep")
        self.assertIn("yielded from hot", actions["recent-bg-wait"]["reason"])
        self.assertEqual(actions["decoder"]["action"], "keep")
        self.assertIn("protected lifecycle", actions["decoder"]["reason"])
        self.assertIn("target met", decision["rationale"])

    def test_prefix_entries_never_expire_by_age(self):
        sessions = [
            _session("warm-old", lifecycle="NVME_RESIDENT", tier="WARM", priority=1, idle=7200)
        ]
        prefixes = [
            {
                "prefix_id": "p-harness",
                "last_accessed": NOW - 30 * 86400,
                "state_bytes": 10_000_000,
                "compatible": True,
            },
            {
                "prefix_id": "p-repo",
                "last_accessed": NOW - 86400,
                "state_bytes": 40_000_000,
                "compatible": True,
            },
        ]
        _, decision = _run_retention(
            DEFAULT_CONFIG, _observation(sessions, prefix_entries=prefixes)
        )
        # Conversation warm window still acts (class demotion only) while
        # month-old prefixes are untouched: no age expiry, ever.
        self.assertEqual(decision["prefix_evictions"], [])
        self.assertEqual(_actions(decision)["warm-old"]["action"], "demote_to_cold")

    def test_warm_quota_overflow_demotes_lru_to_cold_while_cold_has_room(self):
        # Warm/cold are classes over the SAME blobs: warm quota overflow
        # first demotes least-recently-accessed warm sessions to cold while
        # the cold quota still has room -- no evict_snapshot and no
        # disk-bytes-freed claim for a metadata-only demotion.
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["warm_nvme"]["max_size_bytes"] = 1000
        sessions = [
            _session(
                "w-oldest",
                lifecycle="NVME_RESIDENT",
                tier="WARM",
                priority=2,
                idle=900,
                state_bytes=600,
            ),
            _session(
                "w-middle",
                lifecycle="NVME_RESIDENT",
                tier="WARM",
                priority=2,
                idle=500,
                state_bytes=600,
            ),
            _session(
                "w-newest",
                lifecycle="NVME_RESIDENT",
                tier="WARM",
                priority=2,
                idle=100,
                state_bytes=600,
            ),
            _session(
                "c-resident",
                lifecycle="NVME_RESIDENT",
                tier="COLD",
                priority=2,
                idle=5000,
                state_bytes=100,
            ),
        ]
        _, decision = _run_retention(config, _observation(sessions))
        self.assertEqual(decision["hot_evictions"], [])
        # 1800 > 1000: demote LRU warm first (w-oldest, then w-middle until
        # the warm pool fits); the cold quota (1300 << 1 TB) absorbs the
        # demoted bytes, so no snapshot on disk is deleted.
        self.assertEqual(
            [entry["session_id"] for entry in decision["retention_actions"]],
            ["w-oldest", "w-middle", "w-newest", "c-resident"],
        )
        actions = _actions(decision)
        self.assertEqual(actions["w-oldest"]["action"], "demote_to_cold")
        self.assertIn("max_size_bytes", actions["w-oldest"]["reason"])
        self.assertEqual(actions["w-middle"]["action"], "demote_to_cold")
        self.assertEqual(actions["w-newest"]["action"], "keep")
        self.assertEqual(actions["c-resident"]["action"], "keep")
        self.assertNotIn(
            "evict_snapshot",
            [entry["action"] for entry in decision["retention_actions"]],
        )
        # Honest accounting: warm demotion frees WARM budget, never disk
        # bytes, and the demoted bytes now count against the cold quota.
        self.assertIn("frees warm budget not disk bytes", decision["rationale"])
        self.assertIn("demoted bytes now consume cold quota", decision["rationale"])

    def test_pressure_sizing_is_explicit_not_faked(self):
        # (a) UMA pressure without a byte target: say so, free nothing.
        uma_unsized = {
            "needs_hot_slot": False,
            "uma_pressure": True,
            "disk_pressure": False,
            "required_free_bytes": 0,
        }
        sessions = [_session("hot-wait", lifecycle="WAITING", tier="HOT", priority=3, idle=400)]
        _, decision = _run_retention(DEFAULT_CONFIG, _observation(sessions, pressure=uma_unsized))
        self.assertEqual(decision["hot_evictions"], [])
        self.assertIn(
            "required_free_bytes=0 gives no byte target", decision["rationale"]
        )

        # (b) Disk target beyond everything evictable: real observed bytes
        # only, explicit shortfall, hot state untouched.
        disk = {
            "needs_hot_slot": False,
            "uma_pressure": False,
            "disk_pressure": True,
            "required_free_bytes": 10**15,
        }
        sessions = [
            _session("decoder", lifecycle="RUNNING", tier="HOT", priority=0, idle=30),
            _session(
                "cold-snap",
                lifecycle="NVME_RESIDENT",
                tier="COLD",
                priority=2,
                idle=5000,
                state_bytes=5_000_000_000,
            ),
        ]
        prefixes = [
            {
                "prefix_id": "p-live",
                "last_accessed": NOW - 900,
                "state_bytes": 1_000_000_000,
                "compatible": True,
            },
            {
                "prefix_id": "p-stale",
                "last_accessed": NOW - 2000,
                "state_bytes": 1_000_000_000,
                "compatible": False,
            },
        ]
        _, decision = _run_retention(
            DEFAULT_CONFIG,
            _observation(sessions, pressure=disk, prefix_entries=prefixes),
        )
        self.assertEqual(decision["hot_evictions"], [])
        expected_shortfall = 10**15 - 7_000_000_000
        self.assertIn(
            f"shortfall {expected_shortfall} bytes unmet and not claimed as freed",
            decision["rationale"],
        )
        # Incompatible first, then compatible LRU against the disk target.
        self.assertEqual(decision["prefix_evictions"], ["p-stale", "p-live"])
        actions = _actions(decision)
        self.assertEqual(actions["cold-snap"]["action"], "evict_snapshot")
        self.assertEqual(actions["decoder"]["action"], "keep")

    def test_invalid_inputs_fail_closed(self):
        base_config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        good = _observation(
            [_session("s1", lifecycle="WAITING", tier="HOT", priority=2, idle=60)]
        )

        missing_grace = {k: v for k, v in base_config.items() if k != "hot_idle_grace_seconds"}
        typed_prefix = {
            **base_config,
            "prefix_cache": {**base_config["prefix_cache"], "preferred_retention_seconds": 3600},
        }
        typo_key = {**base_config, "hot_idle_grace": 5}
        fractional_cap = {
            **base_config,
            "warm_nvme": {**base_config["warm_nvme"], "max_size_bytes": 1.5},
        }
        priority_out_of_range = _observation(
            [_session("s1", lifecycle="WAITING", tier="HOT", priority=4, idle=60)]
        )
        unknown_lifecycle = _observation(
            [_session("s1", lifecycle="SLEEPING", tier="HOT", priority=2, idle=60)]
        )
        tier_mismatch = _observation(
            [_session("s1", lifecycle="NVME_RESIDENT", tier="HOT", priority=2, idle=60)]
        )
        future_access = _observation(
            [_session("s1", lifecycle="WAITING", tier="HOT", priority=2, idle=-5)]
        )
        missing_now = _observation(
            [_session("s1", lifecycle="WAITING", tier="HOT", priority=2, idle=60)]
        )
        del missing_now["now"]
        nan_now = _observation(
            [_session("s1", lifecycle="WAITING", tier="HOT", priority=2, idle=60)]
        )
        nan_now["now"] = float("nan")
        duplicate_id = _observation(
            [
                _session("s1", lifecycle="WAITING", tier="HOT", priority=2, idle=60),
                _session("s1", lifecycle="WAITING", tier="WARM", priority=3, idle=90),
            ]
        )
        duplicate_id["sessions"][1]["lifecycle"] = "NVME_RESIDENT"
        bad_flag = _observation(
            [_session("s1", lifecycle="WAITING", tier="HOT", priority=2, idle=60)],
            pressure={
                "needs_hot_slot": "yes",
                "uma_pressure": False,
                "disk_pressure": False,
                "required_free_bytes": 0,
            },
        )
        negative_bytes = _observation(
            [_session("s1", lifecycle="WAITING", tier="HOT", priority=2, idle=60, state_bytes=-1)]
        )

        cases = [
            ("fractional byte cap", fractional_cap, good, "must be an integer"),
            ("missing config key", missing_grace, good, "missing keys"),
            ("prefix age window not null", typed_prefix, good, "must be null"),
            ("unknown config key", typo_key, good, "unexpected keys"),
            ("priority out of range", base_config, priority_out_of_range, ".priority: must be one of"),
            ("unknown lifecycle", base_config, unknown_lifecycle, ".lifecycle: must be one of"),
            ("tier/lifecycle mismatch", base_config, tier_mismatch, "HOT tier requires"),
            ("future last_accessed", base_config, future_access, "is after observation now"),
            ("missing explicit clock", base_config, missing_now, "observation: missing keys"),
            ("non-finite now", base_config, nan_now, "non-finite"),
            ("duplicate session id", base_config, duplicate_id, "duplicate session_id"),
            ("non-boolean pressure flag", base_config, bad_flag, "must be a boolean"),
            ("negative state_bytes", base_config, negative_bytes, "must be >= 0"),
        ]
        for label, config, observations, fragment in cases:
            with self.subTest(label):
                proc, decision = _run_retention(config, observations)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIsNone(decision)  # fail closed: no output file
                self.assertIn(fragment, proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main()
