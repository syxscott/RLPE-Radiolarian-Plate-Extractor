"""Unit tests for ``rlpe.llm_usage.collect_llm_usage``.

This module deliberately has no cv2 / paddleocr / torch dependency so
the sidecar aggregation can be exercised in any test environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.llm_usage import collect_llm_usage  # noqa: E402


class _Backend:
    def __init__(
        self,
        *,
        backend_name: str = "MiniMax",
        model: str = "MiniMax-M3",
        cost_summary_payload: dict | None = None,
        raise_on_cost_summary: bool = False,
        total_calls: int | None = None,
        total_errors: int | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
    ):
        self.backend_name = backend_name
        self.model = model
        self._payload = cost_summary_payload
        self._raise = raise_on_cost_summary
        if total_calls is not None:
            self.total_calls = total_calls
        if total_errors is not None:
            self.total_errors = total_errors
        if total_input_tokens is not None:
            self.total_input_tokens = total_input_tokens
        if total_output_tokens is not None:
            self.total_output_tokens = total_output_tokens

    def cost_summary(self):
        if self._raise:
            raise RuntimeError("synthetic backend failure")
        return self._payload


class _Runtime:
    def __init__(self, backend: _Backend | None):
        self.backend = backend


def test_returns_none_when_runtime_is_none():
    assert collect_llm_usage(None) is None


def test_returns_none_when_backend_is_none():
    assert collect_llm_usage(_Runtime(None)) is None


def test_uses_cost_summary_dict():
    runtime = _Runtime(
        _Backend(
            cost_summary_payload={
                "calls": 12,
                "errors": 1,
                "input_tokens": 8000,
                "output_tokens": 1500,
                "total_cost_cny": 0.42,
            }
        )
    )
    out = collect_llm_usage(runtime)
    assert out["backend"] == "MiniMax"
    assert out["model"] == "MiniMax-M3"
    assert out["calls"] == 12
    assert out["errors"] == 1
    assert out["input_tokens"] == 8000
    assert out["output_tokens"] == 1500
    assert out["total_cost_cny"] == 0.42


def test_falls_back_to_counters_when_no_cost_summary():
    backend = _Backend(
        total_calls=7,
        total_errors=2,
        total_input_tokens=4000,
        total_output_tokens=800,
    )
    runtime = _Runtime(backend)
    out = collect_llm_usage(runtime)
    assert out["backend"] == "MiniMax"
    assert out["total_calls"] == 7
    assert out["total_errors"] == 2
    assert out["total_input_tokens"] == 4000
    assert out["total_output_tokens"] == 800


def test_cost_summary_keys_take_precedence_over_counters():
    """If both cost_summary and counters exist, the summary wins."""
    backend = _Backend(
        cost_summary_payload={"calls": 99, "input_tokens": 1234, "total_cost_cny": 0.7},
        total_calls=1,  # should NOT clobber summary's calls=99
        total_input_tokens=1,  # ditto
    )
    runtime = _Runtime(backend)
    out = collect_llm_usage(runtime)
    assert out["calls"] == 99
    assert out["input_tokens"] == 1234
    assert out["total_cost_cny"] == 0.7


def test_returns_none_when_no_signal_present():
    """Backend with only name/model but no usage data must not emit."""
    runtime = _Runtime(_Backend())
    assert collect_llm_usage(runtime) is None


def test_cost_summary_exception_does_not_crash():
    """A raising cost_summary() must not abort the run; counters still usable."""
    backend = _Backend(
        raise_on_cost_summary=True,
        total_calls=3,
        total_input_tokens=100,
        total_output_tokens=50,
    )
    runtime = _Runtime(backend)
    out = collect_llm_usage(runtime)
    assert out["backend"] == "MiniMax"
    assert out["total_calls"] == 3
    assert out["total_input_tokens"] == 100
    assert out["total_output_tokens"] == 50


def test_total_cost_cny_alone_is_sufficient_signal():
    """A run that only saw cost_cny (no call counter) still emits a sidecar."""
    backend = _Backend(cost_summary_payload={"total_cost_cny": 0.001})
    runtime = _Runtime(backend)
    out = collect_llm_usage(runtime)
    assert out is not None
    assert out["total_cost_cny"] == 0.001


def test_returns_none_when_all_counters_are_zero():
    """A fresh backend that has not been called must NOT emit a sidecar.

    Reproduces the case where every counter attribute exists (because the
    backend constructor initialises them to 0) but no API call has run
    yet. Before the fix this returned ``{total_calls: 0, ...}`` and the
    pipeline wrote a misleading zero-sidecar into output/manifests/.
    """
    backend = _Backend(
        total_calls=0,
        total_errors=0,
        total_input_tokens=0,
        total_output_tokens=0,
    )
    # _Backend with no cost_summary_payload, no method overridden.
    runtime = _Runtime(backend)
    assert collect_llm_usage(runtime) is None
