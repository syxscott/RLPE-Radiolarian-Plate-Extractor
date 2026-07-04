"""Tests for Round-6 CLI flag wiring.

The previous CLI surface didn't expose
``--use-geo-vision`` or ``--use-m3-stage3`` despite pipeline
supporting the corresponding ``extra`` config keys. Users had to
hand-edit config dictionaries to enable Round-5 routing. The
new flags round-trip the config so a typical CLI invocation
``--use-opendataloader --use-geo-vision`` works end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestCliFlags:
    """Source guard: the new CLI flags must exist and route into extra."""

    def test_use_geo_vision_flag_exists(self):
        text = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py").read_text(
            encoding="utf-8"
        )
        assert "--use-geo-vision" in text, (
            "CLI must expose --use-geo-vision so users can enable "
            "Round-5 multi-modal geology vision from the command line"
        )

    def test_geo_vision_figure_types_flag_exists(self):
        text = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py").read_text(
            encoding="utf-8"
        )
        assert "--geo-vision-figure-types" in text, (
            "CLI must expose --geo-vision-figure-types so users can "
            "restrict the geo-vision allowlist"
        )

    def test_use_m3_stage3_flag_exists(self):
        text = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py").read_text(
            encoding="utf-8"
        )
        assert "--use-m3-stage3" in text, (
            "CLI must expose --use-m3-stage3 so users can enable Stage 3 bbox/crop enrichment"
        )

    def test_use_geo_vision_flag_is_store_true(self):
        """``--use-geo-vision`` must be a boolean flag, not a value-taking arg."""
        from pathlib import Path as _Path

        text = (_Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py").read_text(
            encoding="utf-8"
        )
        # Find the add_argument block for --use-geo-vision (it may
        # span multiple lines if it has a multi-line `help=` arg).
        idx = text.find('"--use-geo-vision"')
        assert idx > 0, "Could not find --use-geo-vision string in cli.py"
        # Look at the surrounding ~600 chars (covers the action= line
        # which always appears within the add_argument call).
        snippet = text[max(0, idx - 200) : idx + 400]
        assert 'action="store_true"' in snippet, (
            f"--use-geo-vision must be a store_true boolean, got: {snippet!r}"
        )

    def test_use_geo_vision_routes_into_extra(self):
        """The new flag must land in the ``extra`` dict passed to
        PipelineConfig so the pipeline's geo_vision_enabled check
        actually triggers."""
        from pathlib import Path as _Path

        text = (_Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py").read_text(
            encoding="utf-8"
        )
        # Must appear as a key in the extra dict alongside other CLI flags
        assert '"use_geo_vision":' in text, (
            "CLI must route use_geo_vision into the PipelineConfig extra dict"
        )
        assert '"use_m3_stage3":' in text, (
            "CLI must route use_m3_stage3 into the PipelineConfig extra dict"
        )
