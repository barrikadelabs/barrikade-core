# Conventional Commits for AI Agents & Developers

This repository strictly enforces the **Conventional Commits** specification (v1.0.0) for all changes. 

Whether you are an **AI coding assistant** (such as Claude Code, GitHub Copilot, Cursor, or a Google Antigravity agent) or a **human developer**, you MUST ensure that every commit message you generate or construct conforms to these rules.

> [!IMPORTANT]
> Commit message validation is strictly enforced by:
> 1. A Claude PreToolUse hook (`.claude/hooks/commitlint-before-commit.py`) that checks commands before execution.
> 2. A native Git `commit-msg` hook located in `.git/hooks/commit-msg` which intercepts commits from any other environment (e.g., Cursor terminal, VS Code git UI, command-line git).

## Header Format

Every commit header must follow this exact pattern:
```
type(scope): description
```

### Max Length
- The commit header MUST NOT exceed **69 characters** in total.

### Types
Use one of the following lowercase types:
- **`feat`**: A new user-facing capability or public SDK function.
- **`fix`**: A bug fix (restoring expected behavior).
- **`chore`**: Internal developer tooling changes, dependencies updates, environment setups, or config files (like `.cursorrules`, `.gitignore`).
- **`refactor`**: Restructuring code without changing behavior or adding features.
- **`test`**: Adding, modifying, or fixing test cases.
- **`docs`**: Any documentation files (markdown files in `docs/`, `README.md`, `CONTRIBUTING.md`).
- **`perf`**: A code change that specifically improves performance.
- **`build`**: Changes that affect the build system or packaging.
- **`ci`**: CI workflow configuration updates (e.g., GitHub Actions in `.github/workflows/`).
- **`style`**: Markup, white-space, formatting, semi-colons, etc., with no code changes.
- **`revert`**: Reverting a previous commit.

### Scope (Required)
The scope is **REQUIRED** in this repository. Use a scope representing the primary subject of the change:
- **For documentation (`docs`)**: Use the doc filename without extension (e.g., `docs(DEVELOPMENT)`, `docs(MODEL_HOSTING)`, `docs(README)`).
- **For CI workflows (`ci`)**: Use the workflow filename without extension (e.g., `ci(ci)`, `ci(publish)`).
- **For SDK code**: Use the main module or layer name (e.g., `feat(sdk)`, `fix(core)`, `refactor(layer_b)`, `feat(models)`).
- **For tests (`test`)**: Use the specific test component (e.g., `test(telemetry)`, `test(layer_b)`).
- **For internal chores (`chore`)**: Use `chore(harness)` for agent setups/hooks, or `chore(deps)` for dependency bumps.

### Description
- **Lowercase**: Start the description with a lowercase letter (except for capitalized acronyms like `GCS`, `API`, `LLM`).
- **Imperative Mood**: Write in the present imperative ("add baseline detector" rather than "adds baseline detector" or "added baseline detector").
- **No Trailing Period**: Do NOT end the header with a period.

## Body and Footers
- If a commit contains a body, separate it from the header with a **blank line**.
- To reference issues, use `Refs #<issue>` in the body.
- Do NOT use GitHub magic keywords (like `Closes #123`, `Fixes #123`) in the commit message. Those belong in the Pull Request body so that issues close on PR merge, not on every individual push.

## Example Good Commits
- `feat(sdk): add baseline prompt injection detector`
- `fix(core): handle empty input string in layer a scanner`
- `docs(DEVELOPMENT): document local ruff formatting setup`
- `test(telemetry): verify logs are dispatched to backup endpoint`
- `ci(publish): add upload retry to twine package publish workflow`
