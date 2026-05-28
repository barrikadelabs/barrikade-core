#!/usr/bin/env python3
"""
Developer utility to install git hooks in the local repository.
Installs the Conventional Commits message linter as a native Git `commit-msg` hook.
"""

import stat
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_DIR = REPO_ROOT / ".git"
HOOKS_DIR = GIT_DIR / "hooks"
COMMIT_MSG_HOOK = HOOKS_DIR / "commit-msg"

HOOK_CONTENT = """#!/usr/bin/env bash
# Git hook to validate commit messages against Conventional Commits.
# This runs on all local commits (e.g., from terminal git, VS Code, Cursor, Copilot).

# Locate repo root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
[ -n "$REPO_ROOT" ] || REPO_ROOT="."

# Run our self-contained Python commit linter
"$REPO_ROOT/.claude/hooks/commitlint-before-commit.py" "$1"
"""


def install_hooks():
    if not GIT_DIR.exists():
        print(f"Error: .git directory not found at {GIT_DIR}", file=sys.stderr)
        print("Are you running this from the repository root?", file=sys.stderr)
        return 1

    # Ensure hooks directory exists
    HOOKS_DIR.mkdir(parents=True, exist_ok=True)

    # Write hook content
    try:
        COMMIT_MSG_HOOK.write_text(HOOK_CONTENT, encoding="utf-8")
        print(f"✓ Created Git commit-msg hook at {COMMIT_MSG_HOOK}")

        # Make hook executable (chmod +x equivalent in Python)
        st = COMMIT_MSG_HOOK.stat()
        COMMIT_MSG_HOOK.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print("✓ Set hook as executable.")
        print("\n🎉 Commit message linting hook installed successfully!")
        print("All local commits will now be validated against Conventional Commits.")
        print("Example: 'feat(sdk): implement baseline detector'")
        return 0

    except Exception as e:
        print(f"Error installing hook: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(install_hooks())
