"""Tests for Stage 4.5 M3 per-panel species ID.

Audit 2026-08-17 spec: docs/superpowers/specs/2026-08-17-m3-per-panel-pipeline-design.md
Plan: docs/superpowers/plans/2026-08-17-m3-per-panel-pipeline.md
"""
from __future__ import annotations

from pathlib import Path

import pytest

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
