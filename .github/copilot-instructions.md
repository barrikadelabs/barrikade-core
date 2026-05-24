# Project Guidelines for GitHub Copilot

This project enforces strict code style, git workflows, and commit conventions. You MUST adhere to these rules in all chat suggestions, inline completions, and code generation.

## Reference Centralized Rules
Please read and follow the full details inside:
1. **Conventional Commits**: See `.github/rules/conventional-commits.md`
2. **Git Workflow**: See `.github/rules/git-workflow.md`
3. **Python Code Style**: See `.github/rules/python-code-style.md`

## Summary of Core Rules

### Python Style & Formatting
- **Formatter**: PEP-8 aligned via Ruff.
- **Line Length**: Exactly **100 characters**.
- **Imports**: Strict top-level imports only (no function-local imports).
- **Private Imports**: Do NOT import private names (symbols with leading underscores) from other modules.
- **Type Hints**: Use modern Python 3.10+ syntax (`list[X]`, `dict[K, V]`, `X | Y`) instead of the old `typing` module wrappers.

### Git Commits
- **Format**: `type(scope): description` (e.g. `feat(sdk): add dual encoder detector`).
- **Scope**: **REQUIRED** (never empty).
- **Header Limit**: **69 characters** maximum.
- **Header Rules**: Lowercase description, imperative mood, no trailing period.
- **Sign-off**: Always sign off commits (`git commit -s`).
