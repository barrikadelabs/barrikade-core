---
paths:
  - "**/*.py"
---

## Python code style

Formatting is `ruff format` — PEP 8-aligned, line length 100 (project standard; PEP 8 recommends 79). `ruff check` enforces our selected rule subset (`E4`, `E7`, `E9`, `F`, `I`, `PLC0415`, `PLC2701`) — see `[tool.ruff.lint]` in `pyproject.toml`. Run manual formatting or rely on the automatic PostToolUse hook before completing tasks.

- **Top-level imports only.** Every `import` belongs at the module top, above the first definition. No function-local imports for lazy-loading or startup-cost reasons. Enforced by ruff `PLC0415`.
  - **Exception: breaking an import cycle.** When you must inline an import to avoid a circular dependency, leave a one-line comment explaining why the cycle can't be untangled.

- **Don't import other modules' private names.** Symbols prefixed with `_` are module-private. If another module needs one, promote it to public (rename without the leading underscore) rather than cross-importing `_foo`. Enforced by ruff `PLC2701`.

- **Modern type syntax for Python 3.10+.** Prefer `list[str]`, `dict[str, int]` (PEP 585) and `X | None` (PEP 604) over `typing.List` / `typing.Optional`.
