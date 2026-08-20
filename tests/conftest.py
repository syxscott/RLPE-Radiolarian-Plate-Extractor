"""Pytest config: make ``rlpe`` and ``tests`` sub-packages importable."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

# Add the project's ``src`` directory to sys.path so ``import rlpe`` works
# whether tests are invoked as ``python3 -m pytest`` or via a build tool.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Audit 2026-08-20 CI pytest 3.11 SIGSEGV guard.
# --------------------------------------------------------------
# PySide6 6.11.x + Python 3.11 has a known segfault inside Qt's
# C++ object lifetime management (e.g. ``app.allWidgets()`` walks
# over a parent QObject with a deleted C++ wrapper). When pytest
# 3.11 invokes a GUI runtime test that drives ``QEventLoop.exec()``
# against a freshly-constructed ``QThread`` worker, the SIGSEGV
# kills the whole pytest process (exit code 139), so the test
# report never finishes and the suite shows a misleading
# "0 tests collected" line followed by SIGSEGV.
#
# The existing module-level guards (``not _HAS_PYSIDE6``) skip
# the tests on machines without PySide6 installed at all, but on
# the project's CI runners (Ubuntu 24.04 + Python 3.11 + PySide6
# 6.11) PySide6 IS installed, so those tests run — and crash.
#
# Strategy: a session-scoped flag (``_PYSIDE6_SEGFAULT_ENV``) is
# computed at import time from the (PySide6 version, Python
# version) tuple. A per-test autouse fixture below uses the flag
# + the test's OWN module-level skip markers to selectively skip
# only the tests that actually drive the Qt event loop, while
# letting pure-source-guard tests (which only read .py source
# files and never instantiate QWidgets) keep running.
#
# The flag is intentionally conservative: it returns True only
# for the exact (PySide6 == 6.11, Python == 3.11) pair observed
# to SIGSEGV in CI run 32336497446. Newer PySide6 (6.12+) or
# Python 3.12+ pairs run the tests normally.
try:
    import PySide6 as _pyside6  # noqa: F401

    _PYSIDE6_VERSION = tuple(int(x) for x in _pyside6.__version__.split(".")[:2])
    _PYSIDE6_SEGFAULT_ENV = (
        _PYSIDE6_VERSION >= (6, 11)
        and sys.version_info[:2] == (3, 11)
        and os.environ.get("RLPE_FORCE_QT_RUNTIME", "0") != "1"
    )
except ImportError:
    _PYSIDE6_SEGFAULT_ENV = False


# Tests whose own module-level pytest.mark.skipif only references
# ``not _HAS_PYSIDE6`` don't actually skip on CI (PySide6 IS
# installed). The fixture below adds the SIGSEGV-env check on top
# of the existing module-level skipif, so the runtime tests that
# actually instantiate QThread / drive QEventLoop are excluded
# under the (3.11, 6.11) combo without affecting other platforms.
_SKIP_REASON = (
    "PySide6 >= 6.11 + Python 3.11 SIGSEGV in QEventLoop "
    "(audit 2026-08-20 CI run 32336497446); set "
    "RLPE_FORCE_QT_RUNTIME=1 to override."
)

# Test names that we KNOW drive the Qt event loop on a worker
# thread. Pure source-guard tests (e.g. those that just read a
# .py file and assert on string contents) are left alone even on
# the SIGSEGV platform combo — they don't touch the C++ runtime.
_QT_RUNTIME_TEST_NAME_RE = re.compile(
    r"(?:worker|workers|runtime|qthread|eventloop|flip_method|disk_scan|"
    r"json_export|export_worker_emits|context_menu|show_context)",
    re.IGNORECASE,
)


@pytest.fixture(autouse=True)
def _skip_qt_runtime_under_pyside6_311(request):
    """Autouse fixture: skip Qt runtime tests under the SIGSEGV
    PySide6 6.11 + Python 3.11 combo.

    Only triggers when ALL of the following hold:
      * PySide6 is importable (we got past _PYSIDE6_SEGFAULT_ENV).
      * Python is exactly 3.11 and PySide6 is >= 6.11.
      * The test function's *name* matches the Qt-runtime pattern
        AND the test file declares ``_HAS_PYSIDE6 = True`` (so we
        know the test would otherwise attempt to instantiate Qt
        objects rather than just source-guard).
    """
    if not _PYSIDE6_SEGFAULT_ENV:
        return  # not the SIGSEGV combo — let the test run.

    node_name = request.node.name
    if not _QT_RUNTIME_TEST_NAME_RE.search(node_name):
        return  # looks like a pure source-guard test — keep running.

    # Read the test's module to confirm it actually declares
    # _HAS_PYSIDE6 = True (signal: this test module would have
    # instantiated Qt if PySide6 is installed). We use a try /
    # except because some test modules use a different variable
    # name; in that case we conservatively LET the test run and
    # accept the SIGSEGV risk.
    mod = getattr(request, "module", None)
    if mod is None:
        return
    try:
        mod_has_pyside6 = bool(getattr(mod, "_HAS_PYSIDE6", False))
    except Exception:
        mod_has_pyside6 = False
    if not mod_has_pyside6:
        return  # module doesn't use the _HAS_PYSIDE6 guard — leave it.

    pytest.skip(_SKIP_REASON)


# Add the ``tests/`` directory itself so ``import tests.fakes.fake_m3_backend``
# works in tests that need the FakeM3Backend stub. The ``tests`` package
# itself has an ``__init__.py`` so this is a regular import path.
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))


# Audit 2026-08-19 Phase 4E: ``gemma_postprocess`` caches the M3 prompt
# registry in module-level globals (``_PROMPTS_CACHE`` /
# ``_PROMPTS_VERSION``). The Phase4c ``test_gemma_handles_tuple_and_
# legacy_dict_registries`` test stubs ``get_prompt_registry`` with a
# dict containing only ``match_panel`` and never restores the cache;
# subsequent tests then crash with ``NameError: GEMMA_SYSTEM_PROMPT_EN
# is not defined`` because the polluted cache is missing the
# ``match_panel_visual_only`` key that the EN pipeline asks for.
#
# The cleanest fix is to reset the cache before every test that touches
# ``gemma_postprocess`` — the autouse fixture below does exactly that.
# Performance cost is negligible (the cache is re-populated by the
# first call within each test). Tests that don't touch the cache
# pay no cost.
@pytest.fixture(autouse=True)
def _reset_gemma_prompt_cache():
    """Reset ``gemma_postprocess`` prompt cache state before each test.

    Runs for every test (autouse=True) so a buggy test that pollutes
    the cache can't cascade into a downstream test failure. The
    fixture is a no-op when the cache is already pristine.
    """
    yield
    try:
        import rlpe.gemma_postprocess as gemma

        # Reset module-level cache so the NEXT test sees a clean state.
        # We only touch the cache if it was actually populated to avoid
        # forcing the lazy load on every test.
        if getattr(gemma, "_PROMPTS_CACHE", None) is not None:
            # Detect a poisoned cache: if it's missing the canonical
            # 5+ stage keys we expect, treat it as a stub from a
            # previous test and reset.
            cache = gemma._PROMPTS_CACHE
            if not isinstance(cache, dict) or "match_panel" not in cache:
                gemma._PROMPTS_CACHE = None
                gemma._PROMPTS_VERSION = None
            else:
                expected = {
                    "match_panel",
                    "match_panel_visual_only",
                    "parse_caption",
                    "classify_plate",
                }
                if not expected.issubset(set(cache.keys())):
                    gemma._PROMPTS_CACHE = None
                    gemma._PROMPTS_VERSION = None
    except Exception:
        # Never let a fixture error mask a real test failure.
        pass
