"""Tests for tests/fakes/fake_m3_backend.py.

Lock down the fake's public contract so M3-stage tests + smoke runs
have a stable target. The fake must:

* implement ``infer_text`` + ``infer_panel`` with the production shape
* track calls in ``self.calls`` for assertion
* aggregate ``cost_summary()`` the same way MiniMaxM3Backend does
* never make a network call
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fakes.fake_m3_backend import FakeM3Backend  # noqa: E402


class TestInferText:
    def test_returns_expected_shape(self):
        backend = FakeM3Backend()
        out = backend.infer_text(system_prompt="x", user_prompt="y")
        for key in (
            "label",
            "species",
            "confidence",
            "reasoning",
            "request_id",
            "model_version",
            "usage",
            "cost_cny",
        ):
            assert key in out

    def test_records_call(self):
        backend = FakeM3Backend()
        backend.infer_text(system_prompt="sysA", user_prompt="userA")
        assert len(backend.calls) == 1
        call = backend.calls[0]
        assert call.method == "infer_text"
        assert call.system_prompt == "sysA"
        assert call.user_prompt == "userA"
        assert call.image is None


class TestInferPanel:
    def test_returns_expected_shape(self):
        backend = FakeM3Backend()
        out = backend.infer_panel(
            panel_image=b"fake-png-bytes",
            caption_text="cap",
            ocr_labels=["1", "2"],
            system_prompt="sys",
            user_prompt="user",
        )
        assert out["label"] is not None
        assert out["species"] is not None

    def test_records_call_with_image(self):
        backend = FakeM3Backend()
        backend.infer_panel(
            panel_image=b"\x89PNG\r\n",
            caption_text="cap",
            system_prompt="s",
            user_prompt="u",
        )
        assert backend.calls[0].method == "infer_panel"
        assert backend.calls[0].image == b"\x89PNG\r\n"


class TestCannedResponses:
    def test_match_predicate_selects_specific_response(self):
        backend = FakeM3Backend(
            canned_responses=[
                {
                    "match": lambda s: "map_geo" in s,
                    "label": "map-only",
                    "species": "X",
                    "confidence": 0.9,
                    "reasoning": "from map prompt",
                },
                {
                    "label": "default",
                    "species": "Y",
                    "confidence": 0.1,
                    "reasoning": "default fallback",
                },
            ]
        )
        a = backend.infer_text(system_prompt="this is a map_geo prompt")
        b = backend.infer_text(system_prompt="totally unrelated")
        assert a["label"] == "map-only"
        assert b["label"] == "default"

    def test_default_response_when_no_canned(self):
        backend = FakeM3Backend()
        out = backend.infer_text(system_prompt="x", user_prompt="y")
        assert out["request_id"] == "fake-req-0"


class TestCostSummary:
    def test_zero_calls_returns_zero(self):
        s = FakeM3Backend().cost_summary()
        assert s["calls"] == 0
        assert s["errors"] == 0
        assert s["total_cost_cny"] == 0.0

    def test_increments_calls_after_infer_text(self):
        backend = FakeM3Backend()
        backend.infer_text(system_prompt="a", user_prompt="b")
        backend.infer_text(system_prompt="c", user_prompt="d")
        s = backend.cost_summary()
        assert s["calls"] == 2
        assert s["input_tokens"] >= 200
        assert s["output_tokens"] >= 100

    def test_increments_calls_after_infer_panel(self):
        backend = FakeM3Backend()
        backend.infer_panel(panel_image=b"x", system_prompt="s", user_prompt="u")
        s = backend.cost_summary()
        assert s["calls"] == 1


class TestNoNetwork:
    """The fake must not import any networking library."""

    def test_no_requests_import(self):
        import sys as _sys

        had = "requests" in _sys.modules
        saved = _sys.modules.pop("requests", None)
        try:
            from importlib import reload

            if "tests.fakes.fake_m3_backend" in _sys.modules:
                reload(_sys.modules["tests.fakes.fake_m3_backend"])
            assert "requests" not in _sys.modules
        finally:
            if saved is not None:
                _sys.modules["requests"] = saved
            elif had:
                pass
