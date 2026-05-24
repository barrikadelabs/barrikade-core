# Barrikade Developer & Contributor Guide

Welcome to the Barrikade Developer Guide. This document provides an in-depth look at our local development workflows, code style configurations, testing guidelines, automated hooks, and codebase architecture.

---

## 1. Local Environment & Setup

We recommend developing on **Python 3.11** to ensure full compatibility with our production Docker containers, though we support any version from **Python 3.10+**.

### Virtual Environment Setup
1. **Initialize and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install all packages:**
   ```bash
   pip install -r requirements.txt
   ```
   This command installs runtime dependencies along with development utilities (`pytest`, `pytest-cov`, and `ruff`).

### Initial Model Bundles Setup
Barrikade relies on runtime models and datasets distributed via a public Google Cloud Storage bucket. On first import of `barrikade`, the SDK will automatically fetch and unpack the bundle into your `~/.barrikade/bundle/` directory.

To trigger the download manually or skip auto-download:
- **Trigger manual download:**
  ```bash
  python scripts/bundling/gcs_download.py --bucket barrikade-bundles
  python scripts/download_qwen3guard.py
  ```
- **Skip automatic check** (by setting the bypass environment variable):
  ```bash
  export BARRIKADA_SKIP_IMPORT_BUNDLE_CHECK=1
  ```
For more information, see [MODEL_HOSTING.md](file:///Users/ishaan/Documents/Data%20Science/FYP/code/docs/MODEL_HOSTING.md).

---

## 2. Universal Code Style (Ruff)

This project strictly enforces uniform formatting and style guidelines. All developers and **AI agents** (Claude, Cursor, Copilot, Antigravity) MUST conform to these rules.

> [!NOTE]
> Detailed style guidelines are stored in `.github/rules/python-code-style.md`.

### Our Configuration Standards
- **Style Rules**: Strictly PEP-8 aligned.
- **Line Length**: Set to exactly **100 characters** in `pyproject.toml`.
- **Top-level imports only** (`PLC0415`): All imports must live at the top of the file. Function-local imports are forbidden except where necessary to untangle circular dependencies (requires an inline explanatory comment).
- **No private name imports** (`PLC2701`): Do not import symbols starting with `_` from other modules. Promote them to public instead.
- **Modern type hinting**: Always use Python 3.10+ typing syntax (`list[str]`, `str | None`) rather than obsolete typing wrappers (`typing.List`, `typing.Optional`).

### Running Style Commands
Use Ruff to clean and lint your code before committing changes:
```bash
# Verify code quality and styling errors
ruff check

# View formatting differences
ruff format --check --diff

# Automatically fix lint issues and format files
ruff check --fix && ruff format
```

---

## 3. Git Operations & Commit Conventions

To maintain a clean and reviewable history, all commits must follow our Git policies.

> [!NOTE]
> Branching and commit rules are detailed in `.github/rules/git-workflow.md` and `.github/rules/conventional-commits.md`.

### Branching Policy
Branch names must reflect the change type and kebab-case description:
- `feature/<short-desc>` (e.g. `feature/add-baseline-detector`)
- `fix/<issue-number>-<short-desc>` (e.g. `fix/142-handle-empty-input`)
- `chore/<short-desc>` (e.g. `chore/setup-hooks`)

### Commit message guidelines
We use **Conventional Commits** (v1.0.0) with a **REQUIRED scope**. Headers must not exceed **69 characters**:
```
type(scope): description
```
- **Allowed Types**: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `perf`, `build`, `ci`, `style`, `revert`
- **Scope**: Must specify the subject area (e.g. `feat(sdk)`, `fix(core)`, `docs(DEVELOPMENT)`).
- **Sign-off**: Every commit must be signed off to certify its origin using `git commit -s`.

---

## 4. Automated Developer Hooks

We have set up automation to validate commits and format code automatically.

### Native Git hooks (For human developers and general editors)
We provide a setup script to install a native git `commit-msg` hook:
```bash
python3 scripts/setup_git_hooks.py
```
This links our lightweight, regex-based Python commit validator script (`.claude/hooks/commitlint-before-commit.py`) into your local `.git/hooks/commit-msg` configuration. It will automatically intercept any local commit command (from terminal git, VS Code, Cursor, Copilot) and block it if the message violates Conventional Commits.

### Claude Code Hooks (For AI agents)
Claude Code loads project hooks defined in `.claude/settings.json`:
- **PreCommit Hook**: Automatically executes `.claude/hooks/commitlint-before-commit.py` to validate conventional commit formatting before allowing a git commit bash command.
- **PostEdit Hook**: Automatically runs `.claude/hooks/ruff-fix.sh` to run `ruff check --fix` and `ruff format` on any `.py` file immediately after Claude writes or modifies code.

---

## 5. Running Tests

Testing is handled via **pytest**. Ensure your virtual environment is active and all test suites pass before submitting pull requests.

```bash
# Run all tests quietly
pytest -q

# Run specific test file
pytest tests/test_sdk.py

# Run specific test matching name pattern
pytest -k "signature"

# Run tests with terminal coverage report
pytest --cov=core --cov=barrikade --cov-report=term-missing
```

Our test suite uses markers to differentiate test categories (configured in `pyproject.toml`):
- `@pytest.mark.telemetry`: Observability and telemetry logging tests.
- `@pytest.mark.slow`: Long integration tests or full dataset-driven evaluations.

---

## 6. Codebase Architecture

Here is a quick overview of how the Barrikade project directory is organized:

```
├── barrikade/            # Python package entrypoints and main interface
│   ├── __init__.py
│   └── __main__.py       # CLI launcher
├── core/                 # Core scanning and detection engines
│   ├── layer_a/          # Layer A: Rule-based, heuristics, confusion checks
│   ├── layer_b/          # Layer B: Vector embeddings, dual-encoder scanning
│   ├── layer_e/          # Layer E: Deep Guard models (Qwen3Guard etc.)
│   └── telemetry/        # Observability, analytics, and centralized telemetry
├── models/               # Model definitions, dataset signatures, storage
├── tests/                # Test suite directory matching package structure
├── docs/                 # Detailed documentation guides
│   ├── DEVELOPMENT.md    # This guide
│   ├── MODEL_HOSTING.md  # Models and weights downloading workflow
│   └── DOCKER.md         # Building containers and compose setups
├── scripts/              # Utility scripts, downloaders, and bundlers
├── pyproject.toml        # Ruff, pytest, setuptools configurations
└── requirements.txt      # Python dependencies manifest
```
