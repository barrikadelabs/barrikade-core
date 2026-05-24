#!/usr/bin/env bash
# PostToolUse hook: run ruff check --fix and ruff format on an edited .py file in barrikade.
# Reads Claude Code's hook JSON from stdin and no-ops for non-Python paths.
set -u

# Extract file path from Claude's hook JSON
f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty' 2>/dev/null)
[ -n "$f" ] && [ "${f##*.}" = "py" ] || exit 0

# Check if file exists (it could have been deleted)
[ -f "$f" ] || exit 0

# Detect repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
[ -n "$REPO_ROOT" ] || REPO_ROOT="."

# Check for ruff in virtual environments or path
if [ -x "$REPO_ROOT/venv/bin/ruff" ]; then
  RUFF="$REPO_ROOT/venv/bin/ruff"
elif [ -x "$REPO_ROOT/.venv/bin/ruff" ]; then
  RUFF="$REPO_ROOT/.venv/bin/ruff"
elif command -v ruff >/dev/null 2>&1; then
  RUFF="ruff"
else
  # Ruff not installed/found, skip silently
  exit 0
fi

# Run check & format
"$RUFF" check --fix "$f" 2>/dev/null && "$RUFF" format "$f" 2>/dev/null
exit 0
