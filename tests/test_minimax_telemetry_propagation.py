"""Regression tests for MiniMax telemetry propagation.

These cover the case where ``apply_gemma_to_matches()`` was previously
gating cost / request_id / model_version behind the failure branch.
Successful high-confidence MiniMax calls must still propagate telemetry
so /system/llm-status reports non-zero usage and cost.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_match(panel_path: Path) -> MagicMock:
    m = MagicMock()
    m.metadata = {}
    m.panel_path = str(panel_path)
    m.panel_id = None
    m.species = None
    m.label_text = None
    m.confidence = 0.5
    return m


def _patch_image_open(monkeypatch) -> None:
    class _FakeImage:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def convert(self, _mode):
            return self

    fake_image_module = MagicMock()
    fake_image_module.open = lambda *_, **__: _FakeImage()
    monkeypatch.setattr("rlpe.gemma_postprocess.Image", fake_image_module)


def test_successful_call_stamps_minimax_telemetry(monkeypatch, tmp_path):
    from rlpe.gemma_postprocess import apply_gemma_to_matches

    runtime = MagicMock()

    out = {
        "fallback_used": False,
        "label": "3",
        "species": "Actinomma leptodermum",
        "confidence": 0.92,
        "reasoning": "ok",
        "request_id": "req-success-1",
        "model_version": "MiniMax-M3",
        "cost_cny": 0.0123,
        "usage": {"input_tokens": 1000, "output_tokens": 50},
    }

    monkeypatch.setattr(
        "rlpe.gemma_postprocess.gemma_match_panel",
        lambda **kwargs: out,
    )
    _patch_image_open(monkeypatch)

    panel = tmp_path / "panel.png"
    panel.write_bytes(b"\x89PNG\r\n\x1a\n")
    m = _make_match(panel)
    apply_gemma_to_matches(
        runtime=runtime,
        matches=[m],
        caption_text="Plate 1. figs 3. Actinomma leptodermum",
        ocr_labels=["3"],
        conf_threshold=0.70,
        prompt_lang="en",
    )

    assert m.metadata["MiniMax_request_id"] == "req-success-1"
    assert m.metadata["MiniMax_cost_cny"] == pytest.approx(0.0123)
    assert m.metadata["MiniMax_model_version"] == "MiniMax-M3"
    assert m.metadata["MiniMax_usage"] == {"input_tokens": 1000, "output_tokens": 50}
    assert m.metadata["gemma_used"] is True
    # No error / error_type keys must have been written on a successful call.
    assert "gemma_error" not in m.metadata
    assert "gemma_error_type" not in m.metadata


def test_failed_call_keeps_gemma_error_but_still_records_telemetry(monkeypatch, tmp_path):
    from rlpe.gemma_postprocess import apply_gemma_to_matches

    runtime = MagicMock()

    out = {
        "fallback_used": True,
        "label": None,
        "species": None,
        "confidence": 0.10,
        "reasoning": "fail",
        "request_id": "req-fail-1",
        "model_version": "MiniMax-M3",
        "cost_cny": 0.005,
        "error": "rate limited",
        "error_type": "RateLimit",
    }

    monkeypatch.setattr(
        "rlpe.gemma_postprocess.gemma_match_panel",
        lambda **kwargs: out,
    )
    _patch_image_open(monkeypatch)

    panel = tmp_path / "panel.png"
    panel.write_bytes(b"\x89PNG\r\n\x1a\n")
    m = _make_match(panel)
    apply_gemma_to_matches(
        runtime=runtime,
        matches=[m],
        caption_text="Plate 1.",
        ocr_labels=[],
        conf_threshold=0.70,
        prompt_lang="en",
    )

    assert m.metadata["MiniMax_request_id"] == "req-fail-1"
    assert m.metadata["MiniMax_cost_cny"] == pytest.approx(0.005)
    assert m.metadata["MiniMax_model_version"] == "MiniMax-M3"
    assert m.metadata["gemma_error"] == "rate limited"
    assert m.metadata["gemma_error_type"] == "RateLimit"
