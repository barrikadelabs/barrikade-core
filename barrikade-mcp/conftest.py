"""Test bootstrap for the barrikade-mcp package.

Makes both the MCP package (this dir) and the detection core (repo root, for
``barrikade`` / ``models``) importable without an editable install, and pins the
CI-safety env defaults before anything imports the core.
"""

import os
import sys


os.environ.setdefault("BARRIKADA_SKIP_IMPORT_BUNDLE_CHECK", "1")
os.environ.setdefault("BARRIKADA_AUTO_DOWNLOAD_ARTIFACTS", "0")

_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.dirname(_HERE)
for _path in (_REPO_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)
