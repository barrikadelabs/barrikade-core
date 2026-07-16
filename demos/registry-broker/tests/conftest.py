import sys
from pathlib import Path


# The broker modules live one level up (demos/registry-broker is not a package —
# the dir name has a hyphen), so put that dir on sys.path for the tests.
sys.path.insert(0, str(Path(__file__).parent.parent))
