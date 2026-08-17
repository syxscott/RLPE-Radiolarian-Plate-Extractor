"""Phase 61 Plan 4 (Bug 4.10): retry with a different backend on persistent 4xx.

Previously every retry used the IDENTICAL prompt against the same backend.
For 4xx errors (400 / 422) the root cause is almost always a request-shape
or prompt-size issue — retrying the same prompt on the same backend never
helps. The fix: after 1 retry on a 4xx, switch to a configured fallback
backend (extra['fallback_llm_backend']).

The dispatcher lives in a small pure-Python helper
``select_backend_after_4xx`` that takes the current backend name and a
configured fallback name and returns the one to use on the next attempt.
The retry-loop in ``_call_api`` consumes the same helper.
"""

from __future__ import annotations

import pytest

from rlpe.llm_backends import select_backend_after_4xx


def test_4xx_triggers_fallback_backend():
    """After 1 retry on 4xx, switch to the configured fallback."""
    out = select_backend_after_4xx(
        current_backend="MiniMax",
        configured_fallback="ollama",
        attempts_made=2,  # 1 initial + 1 retry
    )
    assert out == "ollama"


def test_4xx_no_fallback_returns_current():
    """No fallback configured → keep retrying current backend."""
    out = select_backend_after_4xx(
        current_backend="MiniMax",
        configured_fallback=None,
        attempts_made=2,
    )
    assert out == "MiniMax"


def test_4xx_first_attempt_no_switch():
    """On the first 4xx attempt we still retry the current backend once."""
    out = select_backend_after_4xx(
        current_backend="MiniMax",
        configured_fallback="ollama",
        attempts_made=1,
    )
    assert out == "MiniMax"


def test_4xx_already_on_fallback_returns_same():
    """When the current backend IS the fallback, keep it (avoid loop)."""
    out = select_backend_after_4xx(
        current_backend="ollama",
        configured_fallback="ollama",
        attempts_made=2,
    )
    assert out == "ollama"
