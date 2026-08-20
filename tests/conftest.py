"""Pytest config: make ``rlpe`` and ``tests`` sub-packages importable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
