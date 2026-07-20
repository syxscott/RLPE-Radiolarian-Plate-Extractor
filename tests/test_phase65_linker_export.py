"""Phase 65 Plan A.5 — linker export tests (converters + xlsx + DwC-A).

Note on xlsx coverage: the worktree's venv lacks ``openpyxl`` (the
heavyweight dep that drives ``_row_for_panel`` end-to-end). We use the
same source-guard pattern as Phase 64 tests: ``_PANEL_HEADERS`` is
imported as long as openpyxl is present; otherwise we fall back to
source-level assertions on xlsx.py. End-to-end runtime validation
lives in the conda env that has openpyxl installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rlpe.exporters.archive import (
    _linker_dynamic_properties,
    _merged_dynamic_properties,
    _schematic_dynamic_properties,
)

try:
    import openpyxl  # noqa: F401

    _HAS_OPENPYXL = True
except Exception:
    _HAS_OPENPYXL = False

if _HAS_OPENPYXL:
    from rlpe.exporters.xlsx import _PANEL_HEADERS, _row_for_panel
else:
    _SRC_XLSX = (
        Path(__file__).resolve().parents[1] / "src" / "rlpe" / "exporters" / "xlsx.py"
    )
    _XLSX_SOURCE = _SRC_XLSX.read_text(encoding="utf-8") if _SRC_XLSX.exists() else ""
    _PANEL_HEADERS = []
    _row_for_panel = None  # type: ignore[assignment]


def _make_panel_metadata(
    link_source: str | None = None,
    link_confidence: float = 0.0,
    link_figure_id: str | None = None,
    schematic_data: dict | None = None,
):
    """Build a PanelMetadata with the given fields."""
    from rlpe.schema_models import PanelMetadata
    return PanelMetadata(
        link_source=link_source,
        link_confidence=link_confidence,
        link_figure_id=link_figure_id,
        figure_schematic_data=schematic_data,
    )


class TestConvertersPropagateLinkSource:
    def test_panel_metadata_from_match_propagates(self):
        from rlpe.converters import panel_metadata_from_match
        from rlpe.types import MatchResult

        m = MatchResult(
            paper_id="p1",
            figure_id="f1",
            panel_id="p1",
            species="X",
            panel_path=None,
            bbox=None,
            confidence=0.5,
            metadata={
                "link_source": "sample_match",
                "link_confidence": 0.95,
                "link_figure_id": "strat1",
            },
        )
        pm = panel_metadata_from_match(m)
        assert pm.link_source == "sample_match"
        assert abs(pm.link_confidence - 0.95) < 1e-6
        assert pm.link_figure_id == "strat1"

    def test_panel_metadata_default_none(self):
        from rlpe.converters import panel_metadata_from_match
        from rlpe.types import MatchResult

        m = MatchResult(
            paper_id="p1",
            figure_id="f1",
            panel_id="p1",
            species="X",
            panel_path=None,
            bbox=None,
            confidence=0.5,
            metadata={},
        )
        pm = panel_metadata_from_match(m)
        assert pm.link_source is None
        assert pm.link_confidence == 0.0
        assert pm.link_figure_id is None


@pytest.mark.skipif(not _HAS_OPENPYXL, reason="openpyxl not installed")
class TestXlsxExporter:
    def test_link_source_column_present(self):
        assert "Link Source" in _PANEL_HEADERS
        assert "Link Confidence" in _PANEL_HEADERS
        assert "Link Figure" in _PANEL_HEADERS

    def test_row_includes_link_data(self):
        panel = {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "p1",
            "species": "Genus species",
            "bbox": None,
            "needs_review": False,
            "review_reasons": [],
            "metadata": {
                "link_source": "sample_match",
                "link_confidence": 1.0,
                "link_figure_id": "strat1",
                "extraction_method": "test",
            },
        }
        row = _row_for_panel(panel)
        assert row[-3] == "sample_match"
        assert row[-2] == 1.0
        assert row[-1] == "strat1"

    def test_row_empty_link_data(self):
        panel = {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "p1",
            "species": "X",
            "bbox": None,
            "needs_review": False,
            "review_reasons": [],
            "metadata": {},
        }
        row = _row_for_panel(panel)
        assert row[-3] == ""
        assert row[-2] == ""
        assert row[-1] == ""


@pytest.mark.skipif(_HAS_OPENPYXL, reason="source-guard fallback only")
class TestXlsxExporterSourceGuard:
    """Source-guard tests for when openpyxl isn't installed.

    Ensures the linker columns exist in the source even when we can't
    import the module.
    """

    def test_link_source_in_source(self):
        assert '"Link Source"' in _XLSX_SOURCE
        assert '"Link Confidence"' in _XLSX_SOURCE
        assert '"Link Figure"' in _XLSX_SOURCE

    def test_row_for_panel_includes_link_in_source(self):
        assert "md.get(\"link_source\")" in _XLSX_SOURCE
        assert "md.get(\"link_confidence\")" in _XLSX_SOURCE
        assert "md.get(\"link_figure_id\")" in _XLSX_SOURCE


class TestArchiveExporter:
    def test_linker_only(self):
        md = _make_panel_metadata(
            link_source="sample_match",
            link_confidence=1.0,
            link_figure_id="strat1",
        )
        blob = _linker_dynamic_properties(md)
        assert blob
        data = json.loads(blob)
        assert data["source"] == "sample_match"
        assert data["confidence"] == 1.0
        assert data["figure_id"] == "strat1"

    def test_linker_unlinked_returns_empty(self):
        md = _make_panel_metadata(link_source=None)
        assert _linker_dynamic_properties(md) == ""

    def test_schematic_only_unchanged(self):
        sch = {"figure_type": "schematic", "text_elements": [], "confidence": 0.9}
        md = _make_panel_metadata(schematic_data=sch)
        sch_blob = _schematic_dynamic_properties(md.figure_schematic_data)
        assert sch_blob
        data = json.loads(sch_blob)
        assert data["figure_type"] == "schematic"
        assert "cross_figure_link" not in data

    def test_merged_both_present(self):
        sch = {"figure_type": "schematic", "text_elements": [], "confidence": 0.9}
        md = _make_panel_metadata(
            link_source="m3_inference",
            link_confidence=0.5,
            link_figure_id="strat2",
            schematic_data=sch,
        )
        merged = _merged_dynamic_properties(md)
        assert merged
        data = json.loads(merged)
        assert data["figure_type"] == "schematic"
        assert data["cross_figure_link"]["source"] == "m3_inference"
        assert data["cross_figure_link"]["confidence"] == 0.5
        assert data["cross_figure_link"]["figure_id"] == "strat2"

    def test_merged_neither_returns_empty(self):
        md = _make_panel_metadata()
        assert _merged_dynamic_properties(md) == ""

    def test_merged_linker_only(self):
        md = _make_panel_metadata(
            link_source="locality_match",
            link_confidence=0.7,
            link_figure_id="map1",
        )
        merged = _merged_dynamic_properties(md)
        assert merged
        data = json.loads(merged)
        assert "cross_figure_link" in data
        assert "figure_type" not in data


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
