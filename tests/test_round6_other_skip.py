"""Tests for Round-6 'other' fig_type skip in classical CV path.

The previous pipeline ran classical panel segmentation on every
figure regardless of fig_type. For micro-CT, XCT, tomographic,
and cross-section captions, this produced thousands of spurious
rows with no species (audit: Xiao_2017 micro-CT paper → 1216
rows, all conf=0.01). The fix adds an 'other' branch that
skips classical segmentation entirely; the user can opt into
geo_vision for non-classical content.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestOtherFigTypeSkipSourceGuard:
    """Source guard: fig_type='other' must skip classical CV."""

    def test_other_branch_in_process_one_pdf_od(self):
        from pathlib import Path as _Path

        text = (_Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py").read_text(
            encoding="utf-8"
        )
        # audit 2026-07-31: the per-PDF body moved to
        # ``_process_one_pdf_od_inner`` (the public entry now only
        # applies the GROBID↔OD cycle guard).
        marker = "def _process_one_pdf_od_inner("
        i = text.find(marker)
        assert i > 0
        next_def = text.find("\n    def ", i + 1)
        assert next_def > 0
        body = text[i:next_def]
        # The other branch must come BEFORE the map / strat_column /
        # litholog_column / paleogeographic_map branches so it
        # gets first crack at non-specimen figures.
        assert 'if fig_type == "other":' in body, (
            "fig_type='other' skip branch missing (audit Round-6 P3): "
            "classical CV must skip Micro-CT/XCT/tomographic/cross-section"
        )
        # Must `continue` past the classical segmentation block.
        # Find the other branch and verify it has `continue` somewhere
        # in the next 1200 chars (we have a long docstring above the
        # `continue`).
        other_pos = body.find('if fig_type == "other":')
        assert other_pos > 0
        snippet = body[other_pos : other_pos + 1200]
        assert "continue" in snippet, (
            "fig_type='other' branch must `continue` past the classical segmentation block"
        )
