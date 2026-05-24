# Barrikade Developer Guidelines (Claude Code)

This document contains repo-specific commands and rules for Claude Code.

## Centralized Project Rules
For detailed rules on code styles, git workflows, and commits, refer to:
1. **Conventional Commits**: See `.github/rules/conventional-commits.md`
2. **Git Workflow**: See `.github/rules/git-workflow.md`
3. **Python Code Style**: See `.github/rules/python-code-style.md`

---

## Pre-commit & Post-edit Automated Hooks

This repository has automated Claude hooks registered in `.claude/settings.json`:
- **PreCommit Hook**: Automatically runs `.claude/hooks/commitlint-before-commit.py` to validate Conventional Commit messages before any `git commit` tool is called.
- **PostEdit Hook**: Automatically runs `.claude/hooks/ruff-fix.sh` to run `ruff check --fix` and `ruff format` on any `.py` file immediately after a write or edit tool is called.

---

## Build, Test & Lint Commands

### 1. Environment Setup
- Activate virtual environment: `source venv/bin/activate` or `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### 2. Linting & Formatting
- Manual lint check: `ruff check`
- Manual formatting check: `ruff format --check --diff`
- Manual auto-fix: `ruff check --fix && ruff format`

### 3. Running Tests
- Run all tests: `pytest`
- Run quiet/short output: `pytest -q`
- Run specific test file: `pytest tests/test_sdk.py`
- Run specific test pattern: `pytest -k "detector"`
- Run tests with coverage: `pytest --cov=core --cov=barrikade --cov-report=term-missing`
