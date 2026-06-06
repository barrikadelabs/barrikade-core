"""Barrikada MCP server.

Exposes Barrikada prompt-injection detection over the Model Context Protocol.
This package namespace is kept import-light; the server (which pulls in the
heavy detection core) lives in ``barrikada_mcp.server``.

Importing this package sets a few environment defaults *before* anything imports
the detection core, so that (a) the core's import-time artifact bundle check does
not block the MCP startup handshake, and (b) third-party ML libraries don't emit
progress bars / chatter that could corrupt the stdio JSON-RPC stream on stdout.
``setdefault`` is used throughout so an operator can still override any of these
via the MCP client's ``env`` block.
"""

import os


__version__ = "0.1.0"

# Applied before ``barrikade`` (and its torch/transformers/HF stack) is imported
# by ``barrikada_mcp.server``. The package __init__ always runs first, so this is
# the correct place to set them (server.py keeps imports at the top per lint).
_ENV_DEFAULTS = {
    # Defer the core's artifact bundle check to lazy pipeline construction so a
    # missing or multi-GB bundle cannot crash or hang the MCP handshake at import.
    "BARRIKADA_SKIP_IMPORT_BUNDLE_CHECK": "1",
    # Keep progress bars / advisory chatter off stdout (stdio JSON-RPC lives there).
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TQDM_DISABLE": "1",
}
for _key, _value in _ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)
