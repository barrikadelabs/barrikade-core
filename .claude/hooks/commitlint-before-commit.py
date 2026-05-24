#!/usr/bin/env python3
"""
Self-contained commit message linter for Barrikade.
Supports both:
  1. Claude Code PreToolUse hook (reads JSON from stdin).
  2. Git commit-msg hook (reads commit message from file argument).

Rules enforced (aligned with @commitlint/config-conventional):
  - Format: type(scope): description
  - Scope: REQUIRED (scope-empty: never)
  - Max header length: 69 characters
  - Allowed types: feat, fix, chore, refactor, test, docs, perf, build, ci, style, revert
  - Description: lowercase, imperative mood preferred, no trailing period
"""

import sys
import json
import re
import textwrap
from pathlib import Path

ALLOWED_TYPES = {
    "feat",
    "fix",
    "chore",
    "refactor",
    "test",
    "docs",
    "perf",
    "build",
    "ci",
    "style",
    "revert",
}


def check_commit_message(msg: str) -> list[str]:
    """Lints a commit message and returns a list of error strings (empty if valid)."""
    errors = []
    lines = msg.splitlines()
    if not lines:
        return ["Commit message is empty"]

    header = lines[0].strip()

    # 1. Header Length Check
    if len(header) > 69:
        errors.append(f"Header is too long ({len(header)} chars; max is 69)")

    # 2. Pattern Matching
    # Regex: matches type(scope): description
    match = re.match(r"^([a-zA-Z0-9_\-]+)(?:\(([^)]+)\))?:\s*(.+)$", header)
    if not match:
        errors.append(
            "Header must follow the 'type(scope): description' format. "
            "Note that the scope is REQUIRED in this repository."
        )
        return errors

    commit_type, scope, description = match.groups()

    # 3. Type Check
    if commit_type not in ALLOWED_TYPES:
        errors.append(
            f"Type '{commit_type}' is not allowed. Must be one of: {', '.join(sorted(ALLOWED_TYPES))}"
        )

    # 4. Scope Check (Required)
    if not scope:
        errors.append("Scope is required (e.g. 'feat(sdk): add baseline detector')")

    # 5. Description Checks
    if description:
        # Check first character lowercase
        first_char = description[0]
        if first_char.isupper() and not description[:3].isupper():
            # Exception for obvious acronyms like "GCS", "API", "LLM"
            errors.append("Description must start with a lowercase letter")

        # Check no trailing period
        if description.endswith("."):
            errors.append("Description must not end with a period")

    # 6. Body Separation Check
    if len(lines) > 1:
        # Check if the second line is blank
        if lines[1].strip() != "":
            errors.append("Header and body must be separated by a blank line")

    return errors


def emit_deny(reason: str) -> None:
    """Emits Claude PreToolUse deny response to stdin."""
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def extract_message(command: str) -> str | None:
    """Extracts commit message from a git commit bash command."""
    # Pattern: -m "$(cat <<'DELIM' ... DELIM)"
    m = re.search(
        r"-m\s+\"?\$\(\s*cat\s+<<(-?)\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n[\t ]*\2\s*$",
        command,
        re.DOTALL | re.MULTILINE,
    )
    if m:
        body = m.group(3)
        return textwrap.dedent(body) if m.group(1) == "-" else body

    # Pattern: -m "..."
    m = re.search(r'-m\s+"((?:[^"\\]|\\.)*)"', command)
    if m:
        return re.sub(r'\\([$"\\`])', r"\1", m.group(1))

    # Pattern: -m '...'
    m = re.search(r"-m\s+'([^']*)'", command)
    if m:
        return m.group(1)

    return None


def run_claude_hook(data_str: str) -> int:
    """Processes stdin JSON from Claude Code PreToolUse hook."""
    try:
        data = json.loads(data_str)
    except Exception:
        return 0

    command = data.get("tool_input", {}).get("command", "") or ""

    # Match `git commit` allowing flags between
    if not re.search(r"(^|[\s&;|])git(\s+\S+)*?\s+commit(\s|$)", command):
        return 0

    # Avoid linting if --no-edit is used
    args_prefix = re.split(r"\s-m\b|\s--message\b", command, maxsplit=1)[0]
    if "--no-edit" in args_prefix:
        return 0

    msg = extract_message(command)
    if not msg:
        # Message not found in -m flags (e.g. interactive git commit), defer to Git hook
        return 0

    errors = check_commit_message(msg)
    if errors:
        error_msg = (
            "Commit message fails Conventional Commits validation!\n\n"
            + "\n".join(f"  ✖  {err}" for err in errors)
            + "\n\nSee .github/rules/conventional-commits.md for the rules."
        )
        emit_deny(error_msg)
    return 0


def run_git_hook(commit_msg_filepath: str) -> int:
    """Lints a commit message from a git commit-msg hook."""
    path = Path(commit_msg_filepath)
    if not path.exists():
        print(f"Error: Commit message file {commit_msg_filepath} does not exist.", file=sys.stderr)
        return 1

    with open(path, "r", encoding="utf-8") as f:
        msg = f.read()

    # Ignore merge commits, squash commits, etc.
    if msg.startswith("Merge branch ") or msg.startswith("Revert "):
        return 0

    errors = check_commit_message(msg)
    if errors:
        print("\n\033[31m✖ Commit message fails Conventional Commits validation!\033[0m", file=sys.stderr)
        for err in errors:
            print(f"  \033[31m✖\033[0m {err}", file=sys.stderr)
        print("\nRefer to .github/rules/conventional-commits.md for correct format.", file=sys.stderr)
        print("Example: \033[32mfeat(sdk): implement new detector interface\033[0m\n", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    # If args are provided, we act as a native Git commit-msg hook
    if len(sys.argv) > 1:
        return run_git_hook(sys.argv[1])

    # Otherwise, read stdin as a Claude PreToolUse hook
    stdin_data = sys.stdin.read().strip()
    if stdin_data:
        return run_claude_hook(stdin_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
