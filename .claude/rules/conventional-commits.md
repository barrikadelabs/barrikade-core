Commit messages MUST follow Conventional Commits (https://www.conventionalcommits.org/en/v1.0.0/).

*Enforced by Claude PreToolUse hook (`.claude/hooks/commitlint-before-commit.py`) and native git commit-msg hook.*

Format: `type(scope): description` — max 69 characters in the header.

- Types: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `perf`, `build`, `ci`, `style`, `revert`
- If a change ships to users, it's `feat` or `fix`. Internal dev configs, scripts, bumps use `chore`. Other types (`refactor`, `test`, `docs`, etc.) take precedence when fitting.
- Scope: always include a scope representing the primary subject of the change:
  - For `docs`: the doc file name without extension (e.g. `docs(DEVELOPMENT)`, `docs(README)`)
  - For `ci`: the workflow file name without extension (e.g. `ci(publish)`)
  - For code: the module or layer name (e.g. `feat(sdk)`, `fix(core)`, `refactor(layer_b)`)
  - For `chore`: use `chore(harness)` for AI setups/hooks, or `chore(deps)` for dependency bumps.
- Description: lowercase, imperative mood, no trailing period.
