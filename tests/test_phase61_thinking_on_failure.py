"""Phase 61 Plan 4 (Bug 4.8): surface thinking tokens on JSON parse failure.

When MiniMax extended-thinking produced a JSON parse error, the
thinking text was preserved on ``result["thinking"]`` but never made
it into ``cost_summary`` aggregations nor any UI dashboard field. The
operator had no way to tell "this call was paid for but the model
choked on its own output" from "no thinking at all happened".

The fix stamps ``cost_summary["failed_with_thinking"] = True`` and
exposes a counter ``failed_with_thinking`` on the backend so the
``/system/llm-status`` route can surface the rate.
"""

from __future__ import annotations

import pytest

from rlpe.llm_backends import MiniMaxM3Backend


def test_cost_summary_marks_failed_with_thinking():
    """A parse-failure with non-empty thinking bumps the dedicated counter."""
    # Construct a backend in local_only mode to skip real API init.
    backend = MiniMaxM3Backend(
        api_key="",
        base_url="http://localhost:0",
        model="MiniMax-M3",
        data_outbound_policy="local_only",
    )
    # Simulate a JSON parse failure with thinking present.
    backend.record_failed_with_thinking(thinking_text="reasoning chain...")

    status = backend.llm_status()
    assert status["failed_with_thinking"] >= 1
    assert status["errors"] >= 1


def test_failed_with_thinking_increments_once_per_call():
    """Multiple calls each bump the counter independently."""
    backend = MiniMaxM3Backend(
        api_key="",
        base_url="http://localhost:0",
        model="MiniMax-M3",
        data_outbound_policy="local_only",
    )
    backend.record_failed_with_thinking("thinking 1")
    backend.record_failed_with_thinking("thinking 2")
    backend.record_failed_with_thinking("thinking 3")
    status = backend.llm_status()
    assert status["failed_with_thinking"] == 3
