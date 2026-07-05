"""Pytest config: make ``rlpe`` and ``tests`` sub-packages importable."""

from __future__ import annotations

import sys
from pathlib import Path

# Add the project's ``src`` directory to sys.path so ``import rlpe`` works
# whether tests are invoked as ``python3 -m pytest`` or via a build tool.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Add the ``tests/`` directory itself so ``import tests.fakes.fake_m3_backend``
# works in tests that need the FakeM3Backend stub. The ``tests`` package
# itself has an ``__init__.py`` so this is a regular import path.
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))
