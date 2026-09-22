#!/usr/bin/env python3
"""Install the versioned pre-commit privacy hook for mimo-halo-lab.

Standard library only. Never overwrites, moves or deletes an existing hook or
an existing ``core.hooksPath`` configuration: on conflict it reports the
problem and exits non-zero so a human can decide. Idempotent once installed.

Conflict policy:
  - If ``core.hooksPath`` is already set, it must resolve to this worktree's
    ``.githooks`` directory. Relative values are resolved against the
    WORKTREE ROOT -- exactly how githooks(5) resolves them at hook time --
    never against this process' current working directory.
  - If ``core.hooksPath`` is unset, switching it would hide the entire
    legacy hooks directory, so ANY active legacy hook there (every file
    except ``*.sample``) is a conflict: nothing is changed and the hook
    names are listed for a human to merge into ``.githooks``.
  - The stored value is re-read and verified after being written.

The hook itself is the versioned wrapper at ``.githooks/pre-commit`` which
delegates to ``scripts/privacy_guard.py``; this installer only wires
``core.hooksPath`` to ``.githooks`` when nothing conflicts.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys

HOOK_REL_PATH = ".githooks/pre-commit"
HOOKS_PATH_CONFIG = ".githooks"


def git(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True)
    except OSError as exc:
        print(f"install-hooks: ERROR: git could not be executed: {exc}", file=sys.stderr)
        sys.exit(2)


def expected_hooks_dir(toplevel: str) -> str:
    return os.path.normpath(os.path.join(toplevel, HOOKS_PATH_CONFIG))


def resolve_hooks_path(configured: str, toplevel: str) -> str:
    """Resolve core.hooksPath the way git resolves it at hook time:
    relative to the worktree root, never to the caller's cwd."""
    if os.path.isabs(configured):
        return os.path.normpath(configured)
    return os.path.normpath(os.path.join(toplevel, configured))


def legacy_active_hooks(toplevel: str) -> tuple[list[str], str | None]:
    """Active (non-``*.sample``) hook files in the CURRENT legacy hooks
    directory that a core.hooksPath switch would silently stop running.

    Returns (names, None) on success or (None, message) on failure.
    """
    proc = git("rev-parse", "--git-path", "hooks")
    if proc.returncode != 0:
        return None, f"could not resolve the legacy hooks directory: {proc.stderr.strip()}"
    hooks_dir = proc.stdout.strip()
    if not hooks_dir:
        return None, "git returned an empty legacy hooks directory"
    if not os.path.isabs(hooks_dir):
        # git prints this path relative to ITS cwd, which is ours here.
        hooks_dir = os.path.abspath(hooks_dir)
    if not os.path.isdir(hooks_dir):
        return [], None
    active: list[str] = []
    try:
        with os.scandir(hooks_dir) as entries:
            for entry in entries:
                if entry.name.endswith(".sample"):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    continue
                active.append(entry.name)
    except OSError as exc:
        return None, f"could not enumerate {hooks_dir!r}: {exc}"
    return sorted(active), None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="store_true", help="print installer version and exit")
    args = parser.parse_args(argv)
    if args.version:
        print("install-hooks 1.1.0")
        return 0

    toplevel_proc = git("rev-parse", "--show-toplevel")
    if toplevel_proc.returncode != 0:
        print("install-hooks: ERROR: not inside a Git repository (fail closed).", file=sys.stderr)
        return 2
    toplevel = toplevel_proc.stdout.strip()

    wrapper = os.path.join(toplevel, HOOK_REL_PATH)
    if not os.path.isfile(wrapper):
        print(
            f"install-hooks: ERROR: {HOOK_REL_PATH} is missing; the versioned "
            "wrapper must exist before installation (fail closed).",
            file=sys.stderr,
        )
        return 2

    expected = expected_hooks_dir(toplevel)
    configured = git("config", "--get", "core.hooksPath")
    current = configured.stdout.strip() if configured.returncode == 0 else ""
    if current:
        resolved = resolve_hooks_path(current, toplevel)
        if resolved != expected:
            print(
                f"install-hooks: CONFLICT: core.hooksPath is already '{current}', "
                f"which resolves against the worktree root to '{resolved}', not "
                f"'{expected}'. Not touching it; resolve manually if you want "
                "the privacy gate active.",
                file=sys.stderr,
            )
            return 1
        action = "already configured"
    else:
        active, err = legacy_active_hooks(toplevel)
        if err is not None:
            print(f"install-hooks: ERROR: {err} (fail closed).", file=sys.stderr)
            return 2
        if active:
            print(
                "install-hooks: CONFLICT: switching core.hooksPath would "
                f"silently disable existing hook(s): {', '.join(active)}. "
                "Not changing anything; merge them into .githooks first if "
                "they should keep running.",
                file=sys.stderr,
            )
            return 1
        config_proc = git("config", "core.hooksPath", HOOKS_PATH_CONFIG)
        if config_proc.returncode != 0:
            print(
                f"install-hooks: ERROR: could not set core.hooksPath: "
                f"{config_proc.stderr.strip()}",
                file=sys.stderr,
            )
            return 2
        verify = git("config", "--get", "core.hooksPath")
        stored = verify.stdout.strip() if verify.returncode == 0 else ""
        if resolve_hooks_path(stored, toplevel) != expected:
            print(
                f"install-hooks: ERROR: core.hooksPath verified as '{stored}', "
                f"expected '{HOOKS_PATH_CONFIG}' (fail closed).",
                file=sys.stderr,
            )
            return 2
        action = "installed"

    mode = os.stat(wrapper).st_mode
    if not mode & stat.S_IXUSR:
        os.chmod(wrapper, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        action += " (made wrapper executable)"

    print(f"install-hooks: {action}: core.hooksPath -> '{HOOKS_PATH_CONFIG}'.")
    print("install-hooks: pre-commit now runs scripts/privacy_guard.py --staged (fail closed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
