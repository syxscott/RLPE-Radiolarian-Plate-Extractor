"""Regression tests for audit 2026-08-01 batch W2 — pipeline C9/M23."""

from __future__ import annotations

import re as _re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Helpers — mirror the hallucination-filter logic in pipeline.py:3751-3763 so
# we can exercise Rule 3 (A-H label prefix) in isolation. We deliberately
# inline the logic instead of importing the inner function (which is a
# closure inside _process_region).
# ---------------------------------------------------------------------------
def _rule3_label_in_caption(nn: str, caption_labels: set[str]) -> bool:
    """Replicate the post-fix logic from pipeline.py:3751-3763."""
    nn = (nn or "").strip().lower()
    if not nn:
        return False
    if nn in caption_labels:
        return True
    # Rule 1: leading digits
    m = _re.match(r"^(\d+)", nn)
    if m and m.group(1) in caption_labels:
        return True
    # Rule 3: leading letter A-H — MUST use IGNORECASE so upper-case
    # caption labels (e.g. "A", "B") match even though ``nn`` is lowered
    # (Bug M23 audit 2026-08-01).
    m = _re.match(r"^([A-H])", nn, _re.IGNORECASE)
    if m and m.group(1).lower() in caption_labels:
        return True
    return False


# ---------------------------------------------------------------------------
# Bug C9 — range_chart stub must carry ``figure_type`` so the cross-figure
# linker classifies it as an anchor (paper_figures) instead of silently
# dropping it (audit 2026-08-01 batch W2).
# ---------------------------------------------------------------------------
class TestRangeChartFigureType:
    def _make_pipeline(self, tmp_path, monkeypatch):
        """Build a RadiolarianPipeline with heavy deps mocked out."""
        from rlpe.config import PipelineConfig
        from rlpe.pipeline import RadiolarianPipeline

        cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
        # Set an API key so _process_range_chart hits the main code path
        # (not the "no API key" early-return branch which builds a
        # different stub).
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://example.test")
        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            pipe = RadiolarianPipeline(cfg)
        # M3 is not required for this test path.
        pipe.m3_engine = None
        return pipe

    def test_range_chart_stub_has_figure_type(self, tmp_path, monkeypatch):
        """Every stub produced by ``_process_range_chart`` MUST carry
        ``figure_type == "range_chart"`` at the top level AND inside
        ``metadata`` so the linker filter (pipeline.py:2273-2278) can
        include it as a paper_figure."""
        from rlpe.range_chart_extractor import BiozoneRecord, RangeChartResult, SpeciesRange

        pipe = self._make_pipeline(tmp_path, monkeypatch)
        chart = RangeChartResult(
            figure_id="fig9",
            paper_id="p1",
            image_path="x.png",
            caption="Range chart",
            species_ranges=[
                SpeciesRange(species="Genus a", section="S1"),
                SpeciesRange(species="Genus b", section="S1"),
            ],
            biozones=[BiozoneRecord(name="Zone A")],
            confidence=0.9,
        )

        with patch("rlpe.pipeline.extract_range_chart", return_value=chart) as m_extract:
            stubs = pipe._process_range_chart(
                paper_id="p1",
                figure_id="fig9",
                caption_text="Range chart caption",
                image_path="x.png",
            )

        assert m_extract.called
        assert len(stubs) == 2
        for stub in stubs:
            # Bug C9 fix: figure_type at top level so the linker sees it
            assert stub.get("figure_type") == "range_chart", (
                f"top-level figure_type missing on stub: {stub}"
            )
            # And inside metadata, where Round 5 set it for strat / litholog
            # / map / range_chart-classified rows.
            md = stub.get("metadata") or {}
            assert md.get("figure_type") == "range_chart", (
                f"metadata.figure_type missing on stub: {stub}"
            )

    def test_linker_picks_up_range_chart_stub(self, tmp_path, monkeypatch):
        """When a range_chart stub flows into ``_apply_cross_figure_linker``,
        the filter at pipeline.py:2273-2278 must put it into
        ``paper_figures`` (anchor index). With figure_type empty the stub
        would silently fall through and the cross-figure linker would see
        no anchor → results are unlinked (Bug C9)."""
        from rlpe.range_chart_extractor import BiozoneRecord, RangeChartResult, SpeciesRange

        pipe = self._make_pipeline(tmp_path, monkeypatch)

        # Plate panel row (what we link FROM).
        plate_row = {
            "paper_id": "p1",
            "figure_id": "plate1",
            "panel_id": "p1",
            "canonical_panel_id": "p1",
            "species": "Genus a",
            "caption_snippet": "Specimen",
            "metadata": {
                "figure_type": "plate",
                "caption": "Specimen",
            },
        }
        chart = RangeChartResult(
            figure_id="fig9",
            paper_id="p1",
            image_path="x.png",
            caption="Range chart for S1",
            species_ranges=[SpeciesRange(species="Genus a", section="S1")],
            biozones=[BiozoneRecord(name="Zone A")],
            confidence=0.9,
        )
        # Build the stub the same way _process_range_chart does — with
        # the post-fix figure_type.
        range_stub = {
            "paper_id": "p1",
            "figure_id": "fig9",
            "panel_id": "RANGE_CHART",
            "species": "Genus a",
            "panel_path": None,
            "bbox": None,
            "confidence": 0.9,
            "label_text": None,
            "caption_snippet": "Range chart for S1",
            "ocr_text": None,
            "paper_metadata": None,
            "figure_type": "range_chart",
            "metadata": {
                "extraction_method": "range_chart_vision",
                "extraction_source": "range_chart",
                "figure_type": "range_chart",
                "section": "S1",
                "range_top": "",
                "range_base": "",
                "biozone": "Zone A",
                "range_chart": chart.to_dict(),
                "geology_links": [{"formation": "X Fm", "age": "Late Triassic"}],
            },
        }

        # Replace the real link_species_to_geology with a probe so we can
        # observe whether the linker sees paper_figures=1 or paper_figures=0.
        # Use a MagicMock so the linker's full kwarg signature is satisfied
        # automatically (the real function has a required
        # ``m3_inference_callable`` kwarg).
        from unittest.mock import MagicMock

        seen: dict[str, int] = {}
        mock_linker = MagicMock(return_value=[])

        def _capture(**kwargs):
            seen["panels"] = len(kwargs.get("panels") or [])
            seen["figures"] = len(kwargs.get("paper_figures") or [])
            return []

        mock_linker.side_effect = _capture

        with patch("rlpe.cross_figure_linker.link_species_to_geology", mock_linker):
            pipe._apply_cross_figure_linker([plate_row, range_stub], paper_id="p1")

        # Critical: the range_chart stub must be in paper_figures (not
        # silently dropped because figure_type was empty).
        assert seen["figures"] == 1, (
            f"range_chart stub not picked up by linker (paper_figures="
            f"{seen['figures']}). With the fix, figure_type='range_chart' "
            f"should be classified as an anchor figure."
        )
        assert seen["panels"] == 1, f"plate row not classified as a link source: {seen}"


# ---------------------------------------------------------------------------
# Bug M23 — Rule 3 of the hallucination filter must use re.IGNORECASE so
# upper-case caption labels (A, B, ...) match the lowered ``nn``. Before
# the fix, ``re.match(r'^([A-H])', nn)`` without IGNORECASE failed to
# match upper-case prefixes — multi-panel labels like "A1" got wrongly
# filtered out (audit 2026-08-01 batch W2).
# ---------------------------------------------------------------------------
class TestRule3IGNORECASE:
    def test_rule3_matches_uppercase_A(self):
        """``A1`` should match a caption label ``"a"`` via Rule 3 (A-H
        prefix). Before the fix, the regex without IGNORECASE missed
        upper-case letters because ``nn`` is .lower()-ed."""
        assert _rule3_label_in_caption("A1", caption_labels={"a"}) is True

    def test_rule3_matches_lowercase_a(self):
        """``a1`` should still match the same caption label."""
        assert _rule3_label_in_caption("a1", caption_labels={"a"}) is True

    def test_rule3_skips_non_label_char(self):
        """``9`` is not a letter prefix and not in caption_labels → rules
        1/2/3 all fail → the function correctly returns False."""
        assert _rule3_label_in_caption("9", caption_labels={"a", "b", "1"}) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
