<!--
  When this package is published to PyPI, this README becomes its long
  description, and the line below is the MCP registry ownership marker. Replace
  YOUR_GH_USERNAME so it matches the `name` in server.json exactly, then keep it
  in sync. It is a comment so it doesn't render on PyPI.
  mcp-name: io.github.YOUR_GH_USERNAME/barrikade-mcp
-->

# barrikade-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
**Barrikade** prompt-injection detection to coding agents (Claude Code, Claude
Desktop, Cursor, …). Point your agent at it and it gains a tool to screen
untrusted text — tool output, retrieved documents, web pages, user input —
**before** acting on it.

It wraps the in-process Barrikade pipeline (`barrikade.PIPipeline`) and serves it
over stdio. Detection runs locally; no data leaves your machine.

## The tool

| Tool | Arguments | Returns |
|------|-----------|---------|
| `detect_prompt_injection` | `text: str` (1–50000 chars), `include_diagnostics: bool = false` | `{ verdict, decision_layer, confidence, processing_time_ms, diagnostics? }` |

- `verdict` — `"allow"` (no injection), `"block"` (injection/jailbreak detected),
  or `"flag"` (inconclusive; rare).
- `decision_layer` — which tier resolved it: `"A"` normalisation, `"B"` signature
  engine, `"C"`/`"D"` ML classifiers, `"E"` LLM judge.
- `confidence` — 0.0–1.0 for the deciding layer (Layer E is binary, `1.0`).
- `diagnostics` — present only with `include_diagnostics=true`; a curated
  per-layer breakdown that **omits the verbatim input and internal model paths**.

## Install

> **Status:** the detection core (`barrikade`) is already published on PyPI, so
> there is no core-dependency blocker. The only thing left for a clean
> `pip install barrikade-mcp` is publishing *this* package to PyPI (the last task
> of issue #29). Until then, install from a checkout or the built wheel.

**Once `barrikade-mcp` is published** (the `barrikade` core resolves from PyPI automatically):

```bash
pip install barrikade-mcp
# or isolated — recommended for a server a client launches repeatedly, so the
# heavy ML deps never touch your project env:
uv tool install barrikade-mcp      # persistent; exposes the `barrikade-mcp` command
uvx barrikade-mcp                  # ephemeral
```

**Today (from a repo checkout or the built wheel):**

```bash
pip install -e . && pip install -e barrikade-mcp        # editable, from the repo root
# or build the wheel and install it (pulls `barrikade` from PyPI):
python -m build barrikade-mcp && pip install barrikade-mcp/dist/*.whl
```

> `torch` defaults to the large CUDA wheel on PyPI. The pipeline runs on CPU, so
> install into a CPU-only environment to save ~700 MB, e.g.
> `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu pip install barrikade-mcp`.

## Model artifacts (one-time)

The detection layers need model artifacts (~GBs) that are **not** bundled in the
package. Download them once before first use:

```bash
barrikade download-artifacts        # uses the public GCS bundle; no auth needed
```

By default this server **does not** download at startup (it sets
`BARRIKADA_SKIP_IMPORT_BUNDLE_CHECK=1` so the MCP handshake stays fast). If
artifacts are missing, the first `detect_prompt_injection` call returns a clear
error telling you to run the command above.

## Configure your agent

All clients use the same stdio launcher: a `command` plus an optional `env`
block. Any of these launch forms work — pick one:

- **`barrikade-mcp`** — the console script. Simplest, but the command must be on
  the client's `PATH` (GUI apps often don't inherit your shell `PATH`).
- **`python -m barrikade_mcp`** — robust when the script isn't on `PATH`; point
  `command` at the venv's `python`.
- **`uvx barrikade-mcp`** — isolated, once published.

> **Windows:** launching the bare `barrikade-mcp` command can fail with
> `WinError 2 (The system cannot find the file specified)` — the MCP client's
> process launcher doesn't append `.exe`. Use the **full path** to
> `...\Scripts\barrikade-mcp.exe`, or the `python -m barrikade_mcp` form
> (`"command": "python", "args": ["-m", "barrikade_mcp"]`). App Control /
> Smart App Control policies can also block pip's generated `.exe` shims
> outright ("An Application Control policy has blocked this file") — the
> `python -m` form sidesteps both issues, since it runs through the signed
> `python.exe`.

### Claude Code

```bash
claude mcp add barrikade -- barrikade-mcp
```

Options go **before** the name; `--` separates the server name from its launch
command. Pick a scope with `--scope` (`local` default · `project` writes a
committed `.mcp.json` · `user` applies to all your projects). Manage with
`claude mcp list` / `claude mcp get barrikade` / `claude mcp remove barrikade`, or
`/mcp` in a session.

### Claude Desktop

Edit the config via **Settings → Developer → Edit Config**, then restart the app.
Config file: macOS `~/Library/Application Support/Claude/claude_desktop_config.json`,
Windows `%APPDATA%\Claude\claude_desktop_config.json` (Linux is not officially
supported).

```json
{
  "mcpServers": {
    "barrikade": {
      "command": "barrikade-mcp"
    }
  }
}
```

Use **absolute paths** in JSON (relative paths fail), and forward slashes or
escaped `\\` on Windows. Logs: macOS `~/Library/Logs/Claude/mcp*.log`, Windows
`%APPDATA%\Claude\logs\mcp*.log`.

### Cursor

Project-level `<project-root>/.cursor/mcp.json` (takes precedence) or global
`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "barrikade": {
      "command": "barrikade-mcp"
    }
  }
}
```

## Environment variables

Set these in the client's `env` block. Forwarded to the detection core.

| Variable | Default | Purpose |
|----------|---------|---------|
| `BARRIKADA_SKIP_IMPORT_BUNDLE_CHECK` | `1` (set by this server) | Skip the import-time artifact check so startup is fast; artifacts load lazily on first detect. |
| `BARRIKADA_MCP_LOG_LEVEL` | `WARNING` | Server log level on stderr (`INFO` for verbose startup/per-layer logs). |
| `BARRIKADA_LAYER_*_*` | — | Point individual layers at pre-staged artifact paths (see core `settings.py`). |

## How it works

Each call runs the tiered pipeline with cost-aware early exits:

```
text → A (normalisation) → B (signature engine) → C (XGBoost) → D (ModernBERT) → E (LLM judge)
```

A layer returns as soon as it reaches a confident `allow`/`block`; otherwise it
cascades to the next. Detection runs in a worker thread so model load/inference
never blocks the server's event loop — your client's own tool timeout still
applies if you set one.

## Development

```bash
pip install -e . && pip install -e barrikade-mcp
pytest barrikade-mcp -q          # unit tests mock the pipeline (no artifacts needed)
python -m barrikade_mcp          # run the server on stdio (Ctrl-C to stop)
```

## Publishing to the MCP registry (maintainers, future)

The [official registry](https://modelcontextprotocol.io/registry/quickstart)
(in preview as of 2026) stores **metadata only** — the package must be on PyPI
first. The `barrikade` core is already on PyPI, so this just needs `barrikade-mcp`
itself published:

1. Publish `barrikade-mcp` to PyPI (`python -m build` → `twine upload`). Ensure
   the `mcp-name:` marker at the top of this README matches the `server.json`
   `name`.
2. Install the `mcp-publisher` CLI (`brew install mcp-publisher`, or a prebuilt
   binary from the registry releases page).
3. `mcp-publisher init` to scaffold `server.json` (a draft is committed here),
   then fill `name` (`io.github.<your-username>/barrikade-mcp`), `version`, and
   the `packages[]` entry (`registryType: "pypi"`, `identifier: "barrikade-mcp"`,
   `runtimeHint: "uvx"`, `transport: { "type": "stdio" }`).
4. `mcp-publisher login github` (device-code flow), then `mcp-publisher publish`.
5. Verify: `curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.<your-username>/barrikade-mcp"`.
