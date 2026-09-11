"""Phase 55 audit: smoke tests for config_io module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestConfigIO:
    """Smoke tests for config_io module."""

    def test_save_and_load_yaml_config(self, tmp_path):
        """Test that YAML config files can be saved and loaded."""
        from rlpe.config import PipelineConfig
        from rlpe.config_io import load_config, save_config

        config_file = tmp_path / "test_config.yaml"
        original = PipelineConfig(
            pdf_dir=Path("/data/pdfs"),
            work_dir=Path("/data/work"),
            grobid_url="http://localhost:8070",
        )
        save_config(original, config_file)
        loaded = load_config(config_file)
        assert loaded is not None
        assert str(loaded.pdf_dir) == "/data/pdfs"
        assert str(loaded.work_dir) == "/data/work"
        assert loaded.grobid_url == "http://localhost:8070"

    def test_load_missing_file_raises(self):
        """Test that loading a missing config file raises ValueError."""
        from rlpe.config_io import load_config

        with pytest.raises(ValueError, match="could not read"):
            load_config(Path("/nonexistent/config.yaml"))


class TestTypedLLMFlagRoundTrip:
    """2026-09-11: the typed LLM stage flags set by the CLI
    (--use-llm-stage3 / --llm-per-panel / --llm-multi-plate-enrich)
    must survive save_config -> load_config so the
    batch_isolation="subprocess" worker children see the same stage
    gates as the parent. Previously save_config never wrote them and
    every worker child ran with the dataclass defaults."""

    def test_typed_llm_flags_round_trip(self, tmp_path):
        from rlpe.config import PipelineConfig
        from rlpe.config_io import load_config, save_config

        config = PipelineConfig(
            pdf_dir=tmp_path / "pdfs",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "out",
        )
        # Simulate the CLI's post-construction typed-attr assignment.
        config.llm_stage3_enabled = True
        config.llm_per_panel_enabled = True
        config.llm_per_panel_min_conf = 0.61
        config.llm_per_panel_max_per_figure = 7
        config.llm_per_panel_max_per_paper = 33
        config.llm_multi_plate_enrich_enabled = True
        config.llm_stage_6 = True

        cfg_path = tmp_path / "worker_config.json"
        save_config(config, cfg_path)
        loaded = load_config(cfg_path)

        assert loaded.llm_stage3_enabled is True
        assert loaded.llm_per_panel_enabled is True
        assert loaded.llm_per_panel_min_conf == 0.61
        assert loaded.llm_per_panel_max_per_figure == 7
        assert loaded.llm_per_panel_max_per_paper == 33
        assert loaded.llm_multi_plate_enrich_enabled is True
        assert loaded.llm_stage_6 is True

    def test_typed_llm_flags_default_false_when_absent(self, tmp_path):
        from rlpe.config import PipelineConfig
        from rlpe.config_io import load_config, save_config

        config = PipelineConfig(
            pdf_dir=tmp_path / "pdfs",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "out",
        )
        cfg_path = tmp_path / "worker_config.json"
        save_config(config, cfg_path)
        loaded = load_config(cfg_path)

        # Defaults stay False so a config written before this change
        # loads cleanly with the original behaviour.
        assert loaded.llm_stage3_enabled is False
        assert loaded.llm_per_panel_enabled is False
        assert loaded.llm_multi_plate_enrich_enabled is False
