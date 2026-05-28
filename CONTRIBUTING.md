# Contributing to Barrikade

Thank you for contributing to Barrikade! We are committed to building a highly secure, production-grade runtime prompt-injection detection SDK for AI agents. 

To maintain the highest level of code quality and engineering velocity, we enforce strict standards for linting, testing, and git operations. Please read and follow this guide before submitting changes.

---

## 1. Quick Onboarding

### System Prerequisites
- **Python 3.10+** (Python 3.11 is recommended to match the production container).
- **Docker** and **Docker Compose** (for running model servers).
- **Ruff** (for python linting and formatting).

### Local Environment Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/barrikade-ai/barrikade.git
   cd barrikade
   ```

2. **Initialize and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This installs development dependencies like `pytest`, `pytest-cov`, and `ruff`.*

4. **Install Git commit validation hooks:**
   We strictly enforce Conventional Commits. Install our automated local commit validator hook using:
   ```bash
   python3 scripts/setup_git_hooks.py
   ```
   *This ensures any commit made in your environment is linted locally before being created.*

5. **Verify your setup by running the baseline quickstart:**
   ```bash
   python examples/quickstart.py
   ```

---

## 2. Universal Git Workflow

All contributors (including human developers and AI assistants like Claude, Cursor, or Copilot) MUST follow these git rules. For full details, see the centralized rules inside `.github/rules/git-workflow.md`.

### Branching Model
- Name your working branches with a type prefix and kebab-case description:
  - `feature/add-layer-c-detector`
  - `fix/142-handle-empty-input`
  - `chore/update-configs`
  - `docs/clarify-download-gcs`

### Atomic Commits
- Keep commits highly focused, logical, and self-contained.
- Sign off every commit per the Developer Certificate of Origin: Use `git commit -s` (appends a `Signed-off-by:` trailer).

### Conventional Commits
- Commit messages must follow Conventional Commits format with a **REQUIRED scope**:
  ```
  type(scope): description
  ```
- Allowed types: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `perf`, `build`, `ci`, `style`, `revert`
- Examples:
  - `feat(sdk): add baseline prompt injection detector`
  - `fix(core): handle empty input string in layer a scanner`
  - `docs(DEVELOPMENT): document local ruff formatting setup`
- See the centralized rules inside `.github/rules/conventional-commits.md` for full details.

---

## 3. Code Style & Quality Checks

We enforce automatic checks and styling to keep the codebase highly uniform. For full details, see the centralized rules inside `.github/rules/python-code-style.md`.

### Code Style (Ruff)
- We use **Ruff** for linting and code formatting.
- **Project standards**: Line length of **100 characters** (aligned with PEP 8 structure otherwise).
- Enforces strict rules like **top-level imports only** (`PLC0415`), **no private imports** (`PLC2701`), and **modern Python 3.10+ type syntax** (`list[str]` instead of `List[str]`).
- Run checks manually:
  ```bash
  # Check style & lint rules
  ruff check
  
  # Check formatting differences
  ruff format --check --diff
  
  # Auto-fix lint violations & format code
  ruff check --fix && ruff format
  ```

### Running Tests
Always run our test suites before opening a Pull Request to ensure nothing was broken:
```bash
# Run all tests quietly
pytest -q

# Run specific test file
pytest tests/test_sdk.py

# Run tests with coverage
pytest --cov=core --cov=barrikade --cov-report=term-missing
```

---

## 4. Pull Requests

1. **Keep PRs small and focused**: Split unrelated modifications into separate PRs.
2. **Link Issues in PR Body**: Reference any issues (e.g. `Closes #123`) in the PR description, never in the PR title.
3. **Squash-Merging**: All PRs are squash-merged. Ensure the PR title conforms to Conventional Commits so that the final merge commit is clean.
4. **Documentation**: When making feature changes, update the relevant documentation inside `docs/` in the same PR.
