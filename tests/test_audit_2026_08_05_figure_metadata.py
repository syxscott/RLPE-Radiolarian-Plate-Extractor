"""Regression tests for audit 2026-08-05 (Fill Gaps) — Fix 3+4.

Fix 3: ``FigureRecord.figure_type`` / ``figure_image_path`` / ``image_path``
       / ``panel_ids[]`` propagation.
  - audit 2026-08-05 verified on Beccaro 2006 that the FigureRecord
    emitted by the regular plate path (OpenDataLoader + classical CV
    matcher) had ``figure_type = None``, ``image_path = None``,
    ``bbox = None``, ``panel_ids = []``. Only the range_chart /
    geo_vision / schematic_vision branches stamped their own
    ``figure_type``.

Fix 4: ``PanelRecord.extraction_method`` plate-path population.
  - Same audit showed ``extraction_method = ""`` (default value) on
    Beccaro panels. The classical plate path never stamped a
    meaningful value; only the LLM-first path set ``"llm_first"``.

These tests exercise ``figure_records_from_matches`` directly with a
hand-built MatchResult list to assert the four FigureRecord fields
plus the PanelRecord ``extraction_method`` land at expected values
when the producer pipeline stamps the metadata keys the Fix 3+4
patches add.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestFigureRecordsFromMatchesStamps:
    """End-to-end: MatchResult → FigureRecord via figure_records_from_matches."""

    def _make_match(self, paper_id, figure_id, panel_id, **meta_overrides):
        from rlpe.types import MatchResult

        meta = {
            "figure_type": "plate",
            "image_path": f"/tmp/{figure_id}.png",
            "figure_image_path": f"/tmp/{figure_id}.png",
            "panel_ids": [panel_id],
            "extraction_method": "heuristic",
            "extraction_source": "opendataloader",
        }
        meta.update(meta_overrides)
        return MatchResult(
            paper_id=paper_id,
            figure_id=figure_id,
            panel_id=panel_id,
            species="Genus species",
            panel_path=f"/tmp/{panel_id}.png",
            bbox=[10, 20, 100, 200],
            confidence=0.7,
            label_text=panel_id,
            caption_snippet="Plate 1",
            ocr_text=None,
            metadata=meta,
            paper_metadata=None,
        )

    def test_figure_type_propagated_from_metadata(self):
        from rlpe.converters import figure_records_from_matches

        matches = [
            self._make_match("p1", "f1", "1", figure_type="plate"),
            self._make_match("p1", "f1", "2", figure_type="plate"),
        ]
        figs = figure_records_from_matches(matches)
        assert len(figs) == 1
        assert figs[0]["figure_type"] == "plate"

    def test_figure_image_path_propagated(self):
        from rlpe.converters import figure_records_from_matches

        matches = [
            self._make_match("p1", "f1", "1", image_path="/path/f1.png"),
        ]
        figs = figure_records_from_matches(matches)
        # meta.get("image_path") or meta.get("figure_image_path")
        assert figs[0]["image_path"] == "/path/f1.png"

    def test_panel_ids_propagated(self):
        from rlpe.converters import figure_records_from_matches

        # Each match carries its own ``panel_ids`` list — set by the
        # pipeline-side audit 2026-08-05 Fix 3 loop. The converter
        # uses the first match seen for any (paper_id, figure_id).
        matches = [
            self._make_match("p1", "f1", "1", panel_ids=["1", "2", "3"]),
        ]
        figs = figure_records_from_matches(matches)
        assert figs[0]["panel_ids"] == ["1", "2", "3"]

    def test_missing_metadata_defaults_to_none(self):
        from rlpe.converters import figure_records_from_matches

        # No figure_type / image_path / panel_ids in metadata.
        m = self._make_match("p1", "f1", "1")
        m.metadata.pop("figure_type", None)
        m.metadata.pop("image_path", None)
        m.metadata.pop("figure_image_path", None)
        m.metadata.pop("panel_ids", None)
        figs = figure_records_from_matches([m])
        assert figs[0]["figure_type"] is None
        assert figs[0]["image_path"] is None
        assert figs[0]["panel_ids"] == []

    def test_figure_records_dedup_by_paper_figure(self):
        from rlpe.converters import figure_records_from_matches

        # Two MatchResults with the same (paper_id, figure_id) should
        # collapse to one FigureRecord.
        matches = [
            self._make_match("p1", "f1", "1"),
            self._make_match("p1", "f1", "2"),
            self._make_match("p1", "f2", "1"),  # different figure
        ]
        figs = figure_records_from_matches(matches)
        assert len(figs) == 2
        assert {f["figure_id"] for f in figs} == {"f1", "f2"}


class TestExtractionMethodOnPanelRecord:
    """``PanelRecord.extraction_method`` reads match.metadata correctly."""

    def test_heuristic_string_round_trips(self):
        from rlpe.converters import panel_record_from_match
        from rlpe.types import MatchResult

        m = MatchResult(
            paper_id="p", figure_id="f", panel_id="1",
            species="Genus species", panel_path="/tmp/1.png",
            bbox=[10, 20, 100, 200], confidence=0.6,
            label_text="1", caption_snippet="Plate 1",
            ocr_text=None,
            metadata={"extraction_method": "heuristic"},
            paper_metadata=None,
        )
        rec = panel_record_from_match(m)
        assert rec.extraction_method == "heuristic"

    def test_llm_first_string_round_trips(self):
        from rlpe.converters import panel_record_from_match
        from rlpe.types import MatchResult

        m = MatchResult(
            paper_id="p", figure_id="f", panel_id="1",
            species="Genus species", panel_path="/tmp/1.png",
            bbox=[10, 20, 100, 200], confidence=0.6,
            label_text="1", caption_snippet="Plate 1",
            ocr_text=None,
            metadata={"extraction_method": "llm_first"},
            paper_metadata=None,
        )
        rec = panel_record_from_match(m)
        assert rec.extraction_method == "llm_first"

    def test_missing_metadata_defaults_to_empty_string(self):
        from rlpe.converters import panel_record_from_match
        from rlpe.types import MatchResult

        m = MatchResult(
            paper_id="p", figure_id="f", panel_id="1",
            species="Genus species", panel_path="/tmp/1.png",
            bbox=[10, 20, 100, 200], confidence=0.6,
            label_text="1", caption_snippet="Plate 1",
            ocr_text=None,
            metadata={},  # no extraction_method
            paper_metadata=None,
        )
        rec = panel_record_from_match(m)
        assert rec.extraction_method == ""  # schema default


class TestFigureTypeSourceGuard:
    """Source guard: pipeline.py must write figure_type in both plate
    and GROBID paths so a future regression that deletes those lines
    surfaces as a test failure.
    """

    def test_plate_path_stamps_figure_type(self):
        text = Path(_SRC, "rlpe", "pipeline.py").read_text(encoding="utf-8")
        # The plate-path loop (added in audit 2026-08-05 Fix 3) must
        # contain a line that stamps figure_type onto match.metadata.
        assert 'meta["figure_type"]' in text
        assert 'meta["image_path"]' in text
        assert 'meta["figure_image_path"]' in text
        assert 'meta["panel_ids"]' in text
        assert 'meta["extraction_method"]' in text

    def test_grobid_path_stamps_figure_type(self):
        text = Path(_SRC, "rlpe", "pipeline.py").read_text(encoding="utf-8")
        # The GROBID-region dedup loop (audit 2026-08-05 Fix 3) must
        # stamp figure_type via the new ``_grobid_fig_type`` variable.
        assert "_grobid_fig_type" in text
        assert "_grobid_panel_ids" in text
        assert "_grobid_image_path" in text
        assert '"grobid_heuristic"' in text