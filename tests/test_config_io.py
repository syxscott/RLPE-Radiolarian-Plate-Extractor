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
