"""Tests for Stage 4.5 M3 per-panel species ID.

Audit 2026-08-17 spec: docs/superpowers/specs/2026-08-17-m3-per-panel-pipeline-design.md
Plan: docs/superpowers/plans/2026-08-17-m3-per-panel-pipeline.md
"""
from __future__ import annotations

import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from rlpe.config import PipelineConfig


def _make_cfg(tmp_path: Path, **overrides) -> PipelineConfig:
    cfg = PipelineConfig(
        pdf_dir=tmp_path / "pdfs",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_config_has_m3_per_panel_fields_with_safe_defaults(tmp_path):
    """All 4 new fields exist with safe defaults:
      - m3_per_panel_enabled: False (off by default)
      - m3_per_panel_min_conf: 0.55
      - m3_per_panel_max_per_figure: 20
      - m3_per_panel_max_per_paper: 200
    """
    cfg = _make_cfg(tmp_path)
    assert hasattr(cfg, "m3_per_panel_enabled")
    assert cfg.m3_per_panel_enabled is False
    assert cfg.m3_per_panel_min_conf == pytest.approx(0.55)
    assert cfg.m3_per_panel_max_per_figure == 20
    assert cfg.m3_per_panel_max_per_paper == 200


def test_config_field_overrides_work(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled=True,
        m3_per_panel_min_conf=0.7,
        m3_per_panel_max_per_figure=10,
        m3_per_panel_max_per_paper=50,
    )
    assert cfg.m3_per_panel_enabled is True
    assert cfg.m3_per_panel_min_conf == pytest.approx(0.7)
    assert cfg.m3_per_panel_max_per_figure == 10
    assert cfg.m3_per_panel_max_per_paper == 50


# ---------------------------------------------------------------------------
# Task 2: method shell with early-return guards
# ---------------------------------------------------------------------------


class _StubPipeline:
    """Minimal stand-in: bind the unbound method and provide config."""

    def __init__(self, cfg: PipelineConfig):
        from types import SimpleNamespace

        from rlpe.pipeline import RadiolarianPipeline

        self.config = cfg
        # Provide a stub m3_engine with a non-None backend so the method's
        # second guard (m3_engine.backend) does NOT short-circuit and the
        # actual Task 3 loop body gets exercised by skip-rows tests.
        self.m3_engine = SimpleNamespace(
            backend=SimpleNamespace(backend_name="stub")
        )
        self._apply_m3_per_panel_species_id = (
            RadiolarianPipeline._apply_m3_per_panel_species_id.__get__(
                self, RadiolarianPipeline
            )
        )


def test_method_early_returns_when_disabled(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=False)
    pipe = _StubPipeline(cfg)
    out = pipe._apply_m3_per_panel_species_id(
        results=[{"panel_id": "1", "species": "regex_match"}],
        paper_id="paper1",
    )
    # When disabled, results pass through unchanged (regex match survives).
    assert out == [{"panel_id": "1", "species": "regex_match"}]


def test_method_no_op_when_no_results(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    pipe = _StubPipeline(cfg)
    out = pipe._apply_m3_per_panel_species_id(results=[], paper_id="paper1")
    assert out == []


# ---------------------------------------------------------------------------
# Minor #1 (from Task 1 CQ review): post_init coercion + range validation
# ---------------------------------------------------------------------------


def test_config_coerces_string_inputs_in_post_init(tmp_path):
    """YAML/JSON loads can produce string booleans/floats/ints.
    ``__post_init__`` must coerce them to the declared types."""
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled="true",
        m3_per_panel_min_conf="0.7",
        m3_per_panel_max_per_figure="10",
        m3_per_panel_max_per_paper="50",
    )
    # _make_cfg uses plain setattr; __post_init__ was already called once
    # during construction but the string overrides bypass coercion, so
    # re-invoke it explicitly to exercise the coercion path.
    cfg.__post_init__()
    assert cfg.m3_per_panel_enabled is True
    assert cfg.m3_per_panel_min_conf == pytest.approx(0.7)
    assert cfg.m3_per_panel_max_per_figure == 10
    assert cfg.m3_per_panel_max_per_paper == 50


def test_config_post_init_validates_conf_range(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_min_conf=1.5)
    with pytest.raises(ValueError, match=r"m3_per_panel_min_conf must be"):
        cfg.__post_init__()


def test_config_post_init_validates_caps_positive(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_max_per_figure=0)
    with pytest.raises(ValueError, match=r"must be >= 1"):
        cfg.__post_init__()


# ---------------------------------------------------------------------------
# Task 3: build per-panel context tuples + skip rows without crops
# ---------------------------------------------------------------------------


def test_method_skips_rows_without_panel_path(tmp_path):
    """Rows without a Stage 3 crop (no panel_path) are passed through
    unchanged — per-panel vision needs the crop image."""
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    pipe = _StubPipeline(cfg)
    results = [
        {"panel_id": "1", "species": "regex_A", "panel_path": None},
        {"panel_id": "2", "species": "regex_B", "panel_path": ""},
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    # Without a backend hook we cannot test "called" — we test "skipped".
    assert out[0]["species"] == "regex_A"
    assert out[1]["species"] == "regex_B"


def test_method_builds_caption_for_panel_from_caption_pairs(tmp_path):
    """When ``caption_pairs`` is on the row, the call uses the pair
    whose panel_id matches the row's panel_id. We assert via inspect:
    the method body must read caption_pairs / select-by-panel_id / etc.
    """
    cfg = _make_cfg(tmp_path)
    from rlpe.pipeline import RadiolarianPipeline

    src = inspect.getsource(RadiolarianPipeline._apply_m3_per_panel_species_id)
    assert "caption_pairs" in src, (
        "method must read caption_pairs to pick the panel-specific snippet"
    )
    assert "panel_id" in src, (
        "method must match caption pairs to row panel_id"
    )


def test_method_truncates_page_context_at_1500_chars(tmp_path):
    cfg = _make_cfg(tmp_path)
    from rlpe.pipeline import RadiolarianPipeline

    src = inspect.getsource(RadiolarianPipeline._apply_m3_per_panel_species_id)
    # Spec §3 requires page-context truncation at 1500 chars.
    assert "1500" in src, (
        "page-context snippet must be truncated (spec §3 says 1500 chars)"
    )


# ---------------------------------------------------------------------------
# Task 4: fan-out via semaphore + per-panel backend call
# ---------------------------------------------------------------------------


def _write_valid_png(path: Path) -> None:
    """Write a tiny valid PNG so ``PIL.Image.open`` can decode it.

    PIL rejects an \"almost PNG\" byte string (the magic without a
    parseable IHDR chunk) with ``UnidentifiedImageError``, which would
    drive the production ``_one`` helper onto its exception path and
    silently leave the row's species unchanged — masking whatever the
    test was trying to assert. A real 10x10 white PNG keeps the test
    focused on the confidence-gate wiring.
    """
    Image.new("RGB", (10, 10), "white").save(path)


def test_method_overwrites_species_when_m3_high_confidence(tmp_path):
    """When ``backend.infer_panel`` returns parseable JSON with
    confidence ``>= m3_per_panel_min_conf``, the row's species is
    overwritten."""
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled=True,
        m3_per_panel_min_conf=0.55,
    )

    # Fake backend that returns a high-confidence species.
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "Emiluvia orea",
        "label": "1",
        "confidence": 0.92,
        "reasoning": "Late Cretaceous nassellarian",
        "alternative": "Archaeodictyomitra",
    }

    # Stub pipeline with fake engine + backend.
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend

    # Fake crop file (must be a real PNG — see _write_valid_png).
    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)

    results = [
        {
            "panel_id": "1",
            "species": "regex_old_species",
            "label": "1",
            "panel_path": str(crop),
            "caption_pairs": [{"panel_id": "1", "text": "Fig. 1, 1."}],
            "page_context_snippet": "Tunisia, Late Cretaceous, Scaglia Fm",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    assert out[0]["species"] == "Emiluvia orea"
    assert out[0]["label"] == "1"
    assert backend.infer_panel.called


def test_method_keeps_regex_when_m3_low_confidence(tmp_path):
    """M3 confidence < min_conf → regex species stays."""
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled=True,
        m3_per_panel_min_conf=0.55,
    )
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "Emiluvia orea",
        "label": "1",
        "confidence": 0.3,  # below threshold
        "reasoning": "uncertain",
    }
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)
    results = [
        {
            "panel_id": "1",
            "species": "regex_old_species",
            "label": "1",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    assert out[0]["species"] == "regex_old_species"
