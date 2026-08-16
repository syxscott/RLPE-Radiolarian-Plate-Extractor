"""Tests for --use-neural-matcher gate behavior.

Without --matcher-checkpoint-path, --use-neural-matcher is a no-op
(heuristic fallback). This test guards against silent regressions
where the warning gets dropped.
"""
from __future__ import annotations

import logging
from pathlib import Path

from rlpe.config import PipelineConfig
from rlpe.pipeline import RadiolarianPipeline


def test_neural_matcher_warns_without_checkpoint(caplog):
    """--use-neural-matcher without checkpoint should WARN, not silently fallback."""
    cfg = PipelineConfig(
        pdf_dir=Path("/tmp/nonexistent"),
        work_dir=Path("/tmp/nonexistent_work"),
        extra={"use_neural_matcher": True, "matcher_checkpoint_path": None},
    )
    with caplog.at_level(logging.WARNING):
        RadiolarianPipeline(cfg)  # constructor only, no run()
    assert "falling back to heuristic" in caplog.text
    assert "--matcher-checkpoint-path" in caplog.text


def test_neural_matcher_no_warning_with_checkpoint(tmp_path, caplog):
    """When checkpoint is provided, no fallback warning should fire."""
    fake_ckpt = tmp_path / "fake_matcher.pt"
    fake_ckpt.write_text("not a real checkpoint")
    cfg = PipelineConfig(
        pdf_dir=Path("/tmp/nonexistent"),
        work_dir=Path("/tmp/nonexistent_work"),
        extra={"use_neural_matcher": True, "matcher_checkpoint_path": str(fake_ckpt)},
    )
    with caplog.at_level(logging.WARNING):
        RadiolarianPipeline(cfg)
    assert "falling back to heuristic" not in caplog.text
