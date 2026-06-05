"""Pytest config: make ``rlpe`` importable when tests are run from the repo root."""
from __future__ import annotations

import sys
from pathlib import Path

# Add the project's ``src`` directory to sys.path so ``import rlpe`` works
# whether tests are invoked as ``python3 -m pytest`` or via a build tool.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
