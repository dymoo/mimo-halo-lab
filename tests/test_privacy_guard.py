"""Behavioral regression tests for scripts/privacy_guard.py and
scripts/install_hooks.py.

Each guard case drives the real guard executable against a throwaway Git
repository so the staged-blob contract (not an internal API) is what is
exercises: forced-gitignore bypass, staged-vs-working divergence, tricky
filenames, secrets, full-blob scanning with the size cap, structural
JSONL-trace recognition, env-file variants, symlink/case hardening, diff
status coverage (rename/typechange/gitlink), and safe source / public
metadata acceptance. Installer cases drive the real installer for legacy
hook preservation and worktree-root path resolution.

Secret literals are assembled at runtime so this test source can itself pass
through the guard.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / "scripts" / "privacy_guard.py"

AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_" + "T" * 36

SAFE_PY = "def add(a, b):\n    return a + b\n"
PUBLIC_METADATA_JSON = """{
  "schema_version": 1,
  "baseline_id": "tacodevs-reap25-mixed3bit-gptq-mtp",
  "published_evaluation": {"task_success": null, "ppl": 9.529}
}
"""


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


class GuardTestCase(unittest.TestCase):
    def make_repo(self, initial_commit: bool = False) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "repo"
        repo.mkdir()
        git(repo, "init", "-q")
        git(repo, "config", "user.email", "guard-test@example.invalid")
        git(repo, "config", "user.name", "Guard Test")
        if initial_commit:
            (repo / "README.md").write_text("safe\n")
            git(repo, "add", "README.md")
            git(repo, "commit", "-q", "-m", "init")
        return repo

    def stage(self, repo: Path, relpath: str, content: str | bytes) -> None:
        target = repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode() if isinstance(content, str) else content)
        git(repo, "add", "--", relpath)

    def run_guard(
        self,
        repo: Path,
        env: dict[str, str] | None = None,
        paths: list[str] | None = None,
    ):
        cmd = [sys.executable, str(GUARD)]
        cmd += ["--paths", *paths] if paths is not None else ["--staged"]
        return subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            env=env,
        )

    def assert_blocked(self, proc, *fragments: str):
        self.assertEqual(proc.returncode, 1, f"stdout={proc.stdout}\nstderr={proc.stderr}")
        combined = proc.stdout + proc.stderr
        for fragment in fragments:
            self.assertIn(fragment, combined)

    def assert_clean(self, proc):
        self.assertEqual(
            proc.returncode, 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )


class AcceptanceTests(GuardTestCase):
    def test_safe_source_and_public_metadata_accepted_on_unborn_head(self):
        """Unborn HEAD (first commit) with only safe content passes."""
        repo = self.make_repo()
        self.stage(repo, "src/mimo_halo/demo.py", SAFE_PY)
        self.stage(repo, "manifests/baselines/tacodevs-reap25/normalized.json", PUBLIC_METADATA_JSON)
        self.stage(repo, "docs/agents/note.md", "# note\n\nplain prose only\n")
        self.assert_clean(self.run_guard(repo))

    def test_env_example_allowed_env_blocked(self):
        repo = self.make_repo()
        self.stage(repo, ".env.example", "DATABASE_URL=postgres://localhost/db\n")
        self.assert_clean(self.run_guard(repo))

        repo2 = self.make_repo()
        self.stage(repo2, ".env", "DATABASE_URL=postgres://localhost/db\n")
        self.assert_blocked(self.run_guard(repo2), "env-file")


class SecretTests(GuardTestCase):
    def test_secrets_blocked_and_never_printed(self):
        repo = self.make_repo(initial_commit=True)
        self.stage(repo, "src/creds.py", f"AWS_KEY = \"{AWS_KEY}\"\n")
        self.stage(repo, "src/tok.txt", f"token: {GITHUB_TOKEN}\n")
        proc = self.run_guard(repo)
        self.assert_blocked(proc, "src/creds.py", "src/tok.txt", "aws-access-key", "github-token")
        combined = proc.stdout + proc.stderr
        self.assertNotIn(AWS_KEY, combined, "guard must never echo the secret value")
        self.assertNotIn(GITHUB_TOKEN, combined, "guard must never echo the secret value")

    def test_generic_credential_assignment_blocked(self):
        repo = self.make_repo()
        value = "s" * 24
        self.stage(repo, "src/settings.cfg", f"api_key = \"{value}\"\n")
        self.assert_blocked(self.run_guard(repo), "generic-credential-assignment")


class PayloadTests(GuardTestCase):
    def test_protected_extensions_and_dirs_blocked_without_reading_blob(self):
        repo = self.make_repo()
        self.stage(repo, "model.gguf", b"GGUF-fake-bytes")
        self.stage(repo, "traces/raw/session-0001.jsonl", "raw trace payload\n")
        self.stage(repo, "manifests/ok.json", PUBLIC_METADATA_JSON)
        proc = self.run_guard(repo)
        self.assert_blocked(proc, "model.gguf", "traces/raw/session-0001.jsonl")
        self.assertIn("protected-extension", proc.stdout)
        self.assertIn("protected-dir:traces/raw", proc.stdout)
        self.assertNotIn("manifests/ok.json", proc.stdout)

    def test_binary_and_encoded_payload_content_blocked(self):
        repo = self.make_repo()
        self.stage(repo, "data/blob.dat", b"prefix\x00\x01\x02" + b"\xff" * 64)
        self.stage(repo, "data/encoded.txt", "A" * 4200 + "\n")
        self.assert_blocked(
            self.run_guard(repo), "data/blob.dat", "data/encoded.txt", "binary-payload", "encoded-payload"
        )


class IgnoreBypassTests(GuardTestCase):
    def test_forced_gitignore_bypass_still_blocked(self):
        """git add -f defeats .gitignore but must not defeat the guard."""
        repo = self.make_repo(initial_commit=True)
        (repo / ".gitignore").write_text("*.local.md\n")
        git(repo, "add", ".gitignore")
        git(repo, "commit", "-q", "-m", "ignore rule")
        secret_note = f"note\n{AWS_KEY}\n"
        target = repo / "leaked.local.md"
        target.write_text(secret_note)
        # Without -f the path is ignored; with -f it is staged. The guard is
        # independent of gitignore either way.
        subprocess.run(["git", "-C", str(repo), "add", "leaked.local.md"], capture_output=True)
        git(repo, "add", "-f", "leaked.local.md")
        proc = self.run_guard(repo)
        self.assert_blocked(proc, "leaked.local.md", "aws-access-key")
        self.assertNotIn(AWS_KEY, proc.stdout + proc.stderr)


class StagedVsWorkingTests(GuardTestCase):
    def test_guard_scans_staged_blob_not_working_tree(self):
        repo = self.make_repo(initial_commit=True)
        self.stage(repo, "src/app.py", SAFE_PY)

        # Working tree diverges into a secret after staging: staged blob is
        # clean, so the commit must pass.
        (repo / "src" / "app.py").write_text(f"KEY = \"{AWS_KEY}\"\n")
        self.assert_clean(self.run_guard(repo))

        # Staging the secret version must block, and cleaning the working
        # tree afterwards must not unblock (staged blob wins).
        git(repo, "add", "src/app.py")
        (repo / "src" / "app.py").write_text(SAFE_PY)
        self.assert_blocked(self.run_guard(repo), "src/app.py", "aws-access-key")


class TrickyPathTests(GuardTestCase):
    def test_tricky_filenames_safe_content_pass(self):
        repo = self.make_repo(initial_commit=True)
        for name in (
            "src/file with space.py",
            "src/quote's.py",
            "src/uni-ünïcødé-文件.py",
            "src/line\nbreak.py",
            "src/back\\slash.py",
        ):
            self.stage(repo, name, SAFE_PY)
        self.assert_clean(self.run_guard(repo))

    def test_tricky_filename_secret_blocked_reported_safely(self):
        repo = self.make_repo(initial_commit=True)
        self.stage(repo, "tricky dir/key file.txt", f"token {GITHUB_TOKEN}\n")
        proc = self.run_guard(repo)
        self.assert_blocked(proc, "tricky dir/key file.txt", "github-token")
        self.assertNotIn(GITHUB_TOKEN, proc.stdout + proc.stderr)


class FailClosedTests(GuardTestCase):
    def test_guard_fails_closed_when_git_is_unavailable(self):
        repo = self.make_repo()
        self.stage(repo, "src/app.py", SAFE_PY)
        env = dict(os.environ)
        env["PATH"] = "/nonexistent-dir-for-guard-test"
        proc = self.run_guard(repo, env=env)
        self.assertEqual(
            proc.returncode, 2, f"must fail closed, got: {proc.stdout} {proc.stderr}"
        )

    def test_paths_mode_fails_closed_on_missing_file(self):
        proc = subprocess.run(
            [sys.executable, str(GUARD), "--paths", "does/not/exist.txt"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)


class FullScanTests(GuardTestCase):
    def test_late_secret_past_old_8mib_window_blocked(self):
        """Regression: content rules must see the WHOLE blob, not a prefix."""
        repo = self.make_repo(initial_commit=True)
        line = "# padding line to push the credential past any scan window\n"
        payload = line * 150000 + "late credential: " + GITHUB_TOKEN + "\n"
        self.assertGreater(len(payload), 8 * 1024 * 1024)
        self.stage(repo, "docs/session.log", payload)
        proc = self.run_guard(repo)
        self.assert_blocked(proc, "docs/session.log", "github-token")
        self.assertNotIn(GITHUB_TOKEN, proc.stdout + proc.stderr)

    def test_late_nul_bytes_past_old_8kib_sniff_blocked(self):
        """Regression: binary detection must cover every byte, not 8 KiB."""
        repo = self.make_repo(initial_commit=True)
        content = b"safe ascii padding line\n" * 500 + b"\x00\x01binary tail\n"
        self.assertGreater(content.index(b"\x00"), 8192)
        self.stage(repo, "logs/output.txt", content)
        proc = self.run_guard(repo)
        self.assert_blocked(proc, "logs/output.txt", "binary-payload")

    def test_oversize_blob_fails_closed_before_read_in_both_modes(self):
        """Accepted blobs are capped at 64 MiB; over-cap fails closed (exit 2)
        before any read -- if the content were read instead, this all-'A'
        file would report exit 1 as an encoded-payload violation."""
        repo = self.make_repo(initial_commit=True)
        self.stage(repo, "docs/oversize.txt", b"A" * (64 * 1024 * 1024 + 1))
        staged = self.run_guard(repo)
        self.assertEqual(staged.returncode, 2, staged.stdout + staged.stderr)
        manual = self.run_guard(repo, paths=["docs/oversize.txt"])
        self.assertEqual(manual.returncode, 2, manual.stdout + manual.stderr)


class CredentialIdentifierTests(GuardTestCase):
    @staticmethod
    def assignment(name: str, value: str) -> str:
        return f'{name} = "{value}"\n'

    def test_prefixed_identifiers_and_punctuated_values_blocked(self):
        repo = self.make_repo()
        payload = (
            self.assignment("db_password", "hunter2" + ".local.production.secret")
            + self.assignment(
                "AWS_SECRET_ACCESS_KEY", "wJalrXUt" + "nFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
            )
            + self.assignment("client_secret", "s3cr3t-" + "client-secret-value-0123456789")
            + self.assignment("access_token", "tok-" + "value.with.punctuation:and@more-123")
        )
        self.stage(repo, "src/settings.cfg", payload)
        proc = self.run_guard(repo)
        self.assert_blocked(proc, "src/settings.cfg", "generic-credential-assignment")
        self.assertNotIn("hunter2.local.production", proc.stdout + proc.stderr)

    def test_tokenizer_metadata_identifiers_not_flagged(self):
        repo = self.make_repo()
        meta = (
            '{"tokenizer": "xlm-roberta-tokenizer-v2-frozen-readonly-copy"}\n'
            '{"tokenizer_class": "LlamaTokenizerFast-slow-pretokenizer-impl",'
            ' "max_tokens": 4096, "input_token_count": 512,'
            ' "token_encoding": "cl100k_base"}\n'
        )
        self.stage(repo, "configs/tokenizer-metadata.jsonl", meta)
        self.assert_clean(self.run_guard(repo))

    def test_placeholder_and_interpolation_values_not_flagged(self):
        """Value-shape exemptions only: interpolation and clearly fake
        placeholders pass; the same rule still fires on real values."""
        repo = self.make_repo()
        cfg = (
            "API_TOKEN=${SERVICE_TOKEN}\n"
            'SERVICE_TOKEN="${SERVICE_TOKEN_URL}"\n'
            'DB_PASSWORD="<your-database-password-here>"\n'
            "HUGGING_FACE_HUB_TOKEN=hf_" + "x" * 30 + "\n"
        )
        self.stage(repo, "configs/placeholders.cfg", cfg)
        self.assert_clean(self.run_guard(repo))


class EnvVariantTests(GuardTestCase):
    def test_env_family_paths_blocked(self):
        repo = self.make_repo()
        self.stage(repo, ".envrc", "export SOME_SETTING=value\n")
        self.stage(repo, "config/production.env", "SETTING=another\n")
        self.stage(repo, "config/.env.local", "SETTING=third\n")
        proc = self.run_guard(repo)
        self.assert_blocked(
            proc, ".envrc", "config/production.env", "config/.env.local"
        )
        self.assertEqual(proc.stdout.count("rule=env-file"), 3)

    def test_env_example_allowed_but_still_credential_scanned(self):
        repo = self.make_repo()
        self.stage(
            repo,
            ".env.example",
            "DATABASE_URL=postgres://localhost/db\n"
            "HUGGING_FACE_HUB_TOKEN=hf_" + "x" * 30 + "\n",
        )
        self.assert_clean(self.run_guard(repo))

        repo2 = self.make_repo()
        self.stage(
            repo2, ".env.example", "HUGGING_FACE_HUB_TOKEN=hf_" + "A1b2C3d4" * 4 + "\n"
        )
        proc = self.run_guard(repo2)
        self.assert_blocked(proc, ".env.example", "huggingface-token")


class StripeKeyTests(GuardTestCase):
    def test_publishable_key_allowed_secret_key_blocked(self):
        repo = self.make_repo()
        self.stage(
            repo,
            "web/checkout.js",
            'const key = Stripe("' + "pk_test_" + "51AbcDefGhiJklMnoPqrStuVw" + '");\n',
        )
        self.assert_clean(self.run_guard(repo))

        repo2 = self.make_repo()
        self.stage(
            repo2,
            "web/server.js",
            'const secretKey = "' + "sk_live_" + "51AbcDefGhiJklMnoPqrStuVw" + '";\n',
        )
        proc = self.run_guard(repo2)
        self.assert_blocked(proc, "web/server.js", "stripe-secret-key")


class TraceJsonlTests(GuardTestCase):
    def test_conversation_jsonl_blocked_at_repo_root(self):
        repo = self.make_repo(initial_commit=True)
        chat = "".join(
            '{"role": "user", "content": "private question %d"}\n' % i for i in range(5)
        ) + "".join(
            '{"role": "assistant", "content": "private answer %d"}\n' % i for i in range(5)
        )
        self.stage(repo, "eval-notes.jsonl", chat)
        proc = self.run_guard(repo)
        self.assert_blocked(proc, "eval-notes.jsonl", "trace-jsonl")

    def test_session_event_shape_jsonl_blocked(self):
        repo = self.make_repo()
        events = (
            '{"message": {"role": "assistant", "content": "private reply"}}\n'
            '{"ts": 1757000000, "event": "turn_end"}\n'
        )
        self.stage(repo, "docs/session-events.jsonl", events)
        self.assert_blocked(
            self.run_guard(repo), "docs/session-events.jsonl", "trace-jsonl"
        )

    def test_non_chat_jsonl_and_tensor_inventory_accepted(self):
        repo = self.make_repo()
        inventory = (
            '{"shape": [4096, 4096], "dtype": "bf16", "name": "experts.0.up"}\n'
            '{"shape": [11008, 4096], "dtype": "int4", "name": "experts.0.down"}\n'
        )
        metrics = '{"event": "latency_ms", "value": 12}\n' * 5
        self.stage(repo, "manifests/tensor-inventory.jsonl", inventory)
        self.stage(repo, "docs/session-metrics.jsonl", metrics)
        self.assert_clean(self.run_guard(repo))

    def test_pretty_chat_example_and_minified_metadata_accepted(self):
        """No blanket raw-JSON rule and no long-line rule: pretty-printed
        docs examples, huge minified public metadata, and the public
        baseline archive shape all commit cleanly."""
        repo = self.make_repo()
        pretty = (
            "{\n"
            '  "messages": [\n'
            '    {"role": "user", "content": "hello"}\n'
            "  ]\n"
            "}\n"
        )
        big_meta = json.dumps(
            {
                "weight_map": {
                    f"layers.{i}.weight": "model-00001-of-00002.safetensors"
                    for i in range(40000)
                }
            },
            separators=(",", ":"),
        )
        self.assertGreater(len(big_meta), 100_000)
        self.stage(repo, "docs/chat-schema-example.json", pretty)
        self.stage(repo, "manifests/weight-inventory.json", big_meta + "\n")
        self.assert_clean(self.run_guard(repo))


class SymlinkTests(GuardTestCase):
    def stage_symlink(self, repo: Path, linkpath: str, target: str) -> None:
        link = repo / linkpath
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, link)
        git(repo, "add", "--", linkpath)

    def test_protected_absolute_and_escaping_targets_blocked(self):
        repo = self.make_repo(initial_commit=True)
        self.stage_symlink(repo, "docs/link.jsonl", "../traces/raw/session-0001.jsonl")
        self.stage_symlink(repo, "docs/rootish.jsonl", "traces/raw/session-0001.jsonl")
        self.stage_symlink(repo, "docs/cased.jsonl", "../TRACES/raw/session-0001.jsonl")
        self.stage_symlink(
            repo, "docs/abs", "/Users/example/mimo-halo-lab/traces/raw/s.jsonl"
        )
        self.stage_symlink(repo, "docs/esc", "../../etc/passwd")
        self.stage_symlink(repo, "docs/weights.gguf", "../model.gguf")
        proc = self.run_guard(repo)
        self.assert_blocked(
            proc,
            "docs/link.jsonl",
            "docs/rootish.jsonl",
            "docs/cased.jsonl",
            "docs/abs",
            "docs/esc",
            "docs/weights.gguf",
        )
        self.assertIn("rule=symlink-protected-target", proc.stdout)
        self.assertIn("rule=symlink-absolute-target", proc.stdout)
        self.assertIn("rule=symlink-escaping-target", proc.stdout)
        combined = proc.stdout + proc.stderr
        self.assertNotIn("/Users/example", combined, "link target must never be echoed")

    def test_safe_relative_targets_accepted(self):
        repo = self.make_repo(initial_commit=True)
        self.stage_symlink(repo, "docs/demo.py", "../src/demo.py")
        self.stage_symlink(repo, "README-shortcut.md", "docs/demo.py")
        self.assert_clean(self.run_guard(repo))

    def test_paths_mode_symlink_policy_and_never_followed(self):
        repo = self.make_repo()
        raw_dir = repo / "traces" / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "session-0001.jsonl").write_text(
            '{"role": "user", "content": "private chatter here"}\n' * 3
        )
        (repo / "notes.txt").write_text("value: " + AWS_KEY + "\n")
        (repo / "docs").mkdir()
        os.symlink("../traces/raw/session-0001.jsonl", repo / "docs" / "l.jsonl")
        os.symlink("../notes.txt", repo / "docs" / "n.txt")

        direct = self.run_guard(repo, paths=["traces/raw/session-0001.jsonl"])
        self.assert_blocked(
            direct, "traces/raw/session-0001.jsonl", "protected-dir:traces/raw"
        )

        via_link = self.run_guard(repo, paths=["docs/l.jsonl"])
        self.assert_blocked(via_link, "docs/l.jsonl", "symlink-protected-target")

        note_direct = self.run_guard(repo, paths=["notes.txt"])
        self.assert_blocked(note_direct, "notes.txt", "aws-access-key")
        # The link's target STRING is what would commit; its content is
        # never followed, so the link itself is clean while the real path
        # above is blocked.
        note_via_link = self.run_guard(repo, paths=["docs/n.txt"])
        self.assert_clean(note_via_link)


class CaseHardeningTests(GuardTestCase):
    def test_case_variants_of_protected_paths_blocked(self):
        repo = self.make_repo(initial_commit=True)
        self.stage(repo, "TRACES/RAW/session.jsonl", '{"a": 1}\n')
        self.stage(repo, "Models/weights-notes.txt", "safe note\n")
        self.stage(repo, "config/PRODUCTION.ENV", "SETTING=x\n")
        proc = self.run_guard(repo)
        self.assert_blocked(
            proc,
            "TRACES/RAW/session.jsonl",
            "Models/weights-notes.txt",
            "config/PRODUCTION.ENV",
        )
        self.assertIn("rule=protected-dir:traces/raw", proc.stdout)
        self.assertIn("rule=protected-dir:models", proc.stdout)
        self.assertIn("rule=env-file", proc.stdout)


class DiffStatusTests(GuardTestCase):
    def test_rename_destination_content_scanned(self):
        repo = self.make_repo(initial_commit=True)
        (repo / "src").mkdir()
        (repo / "src" / "app.py").write_text(SAFE_PY * 40)
        git(repo, "add", "src/app.py")
        git(repo, "commit", "-q", "-m", "app")
        git(repo, "mv", "src/app.py", "src/renamed.py")
        with open(repo / "src" / "renamed.py", "a") as handle:
            handle.write('KEY = "' + AWS_KEY + '"\n')
        git(repo, "add", "src/renamed.py")
        proc = self.run_guard(repo)
        self.assert_blocked(proc, "src/renamed.py", "aws-access-key")

    def test_typechange_to_symlink_policed(self):
        repo = self.make_repo(initial_commit=True)
        note = repo / "docs" / "note.md"
        note.parent.mkdir(parents=True)
        note.write_text("safe note\n")
        git(repo, "add", "docs/note.md")
        git(repo, "commit", "-q", "-m", "note")
        note.unlink()
        os.symlink("../traces/raw/session.jsonl", note)
        git(repo, "add", "docs/note.md")
        proc = self.run_guard(repo)
        self.assert_blocked(proc, "docs/note.md", "symlink-protected-target")

    def test_gitlink_pointer_accepted(self):
        repo = self.make_repo()
        git(
            repo,
            "update-index",
            "--add",
            "--cacheinfo",
            "160000,1111111111111111111111111111111111111111,vendor/submodule",
        )
        self.assert_clean(self.run_guard(repo))


class InstallerTests(GuardTestCase):
    INSTALLER = REPO_ROOT / "scripts" / "install_hooks.py"
    WRAPPER = REPO_ROOT / ".githooks" / "pre-commit"

    def make_target_repo(self) -> Path:
        repo = self.make_repo()
        (repo / ".githooks").mkdir()
        shutil.copy(self.WRAPPER, repo / ".githooks" / "pre-commit")
        return repo

    def run_installer(self, repo: Path, subcwd: Path | None = None):
        return subprocess.run(
            [sys.executable, str(self.INSTALLER)],
            cwd=str(subcwd if subcwd is not None else repo),
            capture_output=True,
            text=True,
        )

    def hooks_path(self, repo: Path) -> str | None:
        proc = subprocess.run(
            ["git", "-C", str(repo), "config", "--get", "core.hooksPath"],
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    def make_legacy_hook(self, repo: Path, name: str) -> None:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--git-path", "hooks"],
            capture_output=True,
            text=True,
            check=True,
        )
        hooks_dir = Path(proc.stdout.strip())
        if not hooks_dir.is_absolute():
            hooks_dir = repo / hooks_dir
        hook = hooks_dir / name
        hook.write_text("#!/bin/sh\nexit 0\n")
        hook.chmod(0o755)

    def test_sample_hooks_do_not_block_install_and_it_is_idempotent(self):
        """git init ships only *.sample hooks; they must not be conflicts."""
        repo = self.make_target_repo()
        first = self.run_installer(repo)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(self.hooks_path(repo), ".githooks")
        second = self.run_installer(repo)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(self.hooks_path(repo), ".githooks")

    def test_active_legacy_hook_any_name_refuses_before_changing_config(self):
        """A non-pre-commit legacy hook (pre-push) that switching would
        silently disable must refuse, with core.hooksPath left untouched."""
        repo = self.make_target_repo()
        self.make_legacy_hook(repo, "pre-push")
        proc = self.run_installer(repo)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("pre-push", proc.stderr)
        self.assertIsNone(self.hooks_path(repo), "core.hooksPath must stay unset")

    def test_subdir_invocation_still_installs(self):
        repo = self.make_target_repo()
        sub = repo / "tests"
        sub.mkdir()
        proc = self.run_installer(repo, subcwd=sub)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.hooks_path(repo), ".githooks")

    def test_outside_relative_config_refused_from_any_cwd(self):
        """core.hooksPath resolves against the WORKTREE ROOT at hook time
        (githooks(5)); a value pointing outside must conflict from the root
        AND from a subdirectory, never report success."""
        repo = self.make_target_repo()
        git(repo, "config", "core.hooksPath", "../.githooks")
        from_root = self.run_installer(repo)
        self.assertEqual(from_root.returncode, 1, from_root.stdout + from_root.stderr)
        sub = repo / "docs"
        sub.mkdir()
        from_sub = self.run_installer(repo, subcwd=sub)
        self.assertEqual(from_sub.returncode, 1, from_sub.stdout + from_sub.stderr)
        self.assertEqual(
            self.hooks_path(repo), "../.githooks", "user config must not be rewritten"
        )

    def test_exact_config_accepted(self):
        repo = self.make_target_repo()
        git(repo, "config", "core.hooksPath", ".githooks")
        proc = self.run_installer(repo)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.hooks_path(repo), ".githooks")


if __name__ == "__main__":
    unittest.main()
