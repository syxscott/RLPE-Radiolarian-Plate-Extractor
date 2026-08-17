"""Tests for Phase 63 Plan 6 — Bug 6.8: NaN/Inf sanitisation in CSV/xlsx.

Before: ``analysis.write_csv`` wrote ``float('nan')`` / ``float('inf')``
as the Python repr ``"nan"`` / ``"inf"`` into the CSV. Excel /
LibreOffice then showed ``#NAME?`` (Excel) or empty (LibreOffice);
GBIF/PBDB ingest rejected the row.

After: ``_sanitise_csv_cell`` in ``analysis.py`` replaces NaN/Inf with
the empty string. ``xlsx._row_for_panel`` does the same for the
workbook writer.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.exporters import AnalysisOptions, write_csv  # noqa: E402
from rlpe.exporters.analysis import _sanitise_csv_cell  # noqa: E402

# xlsx source-guard: read the file directly to extract the helper so
# the test works in envs without openpyxl installed. The runtime
# behaviour is pinned in the conda env that ships openpyxl.
XLSX_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "rlpe" / "exporters" / "xlsx.py"
).read_text(encoding="utf-8")
from rlpe.provenance import build_provenance  # noqa: E402
from rlpe.schema_models import (  # noqa: E402
    GeologyLinkRecord,
    PanelMetadata,
    PanelRecord,
    PaperMetadataRecord,
    ProvenanceRecord,
    RunOutput,
    ScaleBarRecord,
)


def test_sanitise_csv_cell_handles_nan():
    assert _sanitise_csv_cell(float("nan")) == ""


def test_sanitise_csv_cell_handles_inf():
    assert _sanitise_csv_cell(float("inf")) == ""
    assert _sanitise_csv_cell(float("-inf")) == ""


def test_sanitise_csv_cell_preserves_normal_floats():
    assert _sanitise_csv_cell(3.14) == 3.14
    assert _sanitise_csv_cell(-1.5) == -1.5
    assert _sanitise_csv_cell(0.0) == 0.0


def test_sanitise_csv_cell_preserves_int_none():
    assert _sanitise_csv_cell(None) == ""
    assert _sanitise_csv_cell(42) == 42


def test_sanitise_csv_cell_still_handles_formula_prefix():
    """Formula sanitisation is preserved alongside the NaN fix."""
    assert _sanitise_csv_cell("=cmd|'/c calc'!A1") == "'=cmd|'/c calc'!A1"
    assert _sanitise_csv_cell("-1+2") == "'-1+2"


def _panel_with_nan_coords() -> PanelRecord:
    """Panel whose first geology link has NaN coordinates.

    This simulates a Round 26 era scale-bar / coordinate parsing
    bug that emitted ``float('nan')`` rather than None for an
    illegible digit.

    PanelRecord is pydantic-validated with ``latitude`` /
    ``modern_latitude`` constrained to ``-90..90``. Constructing
    with NaN would fail validation, so we build the panel dict
    directly (the writers consume dicts in production via
    ``model_dump`` anyway) and use a sentinel that lets us
    simulate the buggy round-trip.
    """
    pm = PaperMetadataRecord(
        title="T", authors=["A"], year=2020, doi="10.1/x", source="opendataloader", confidence=0.8
    )
    geo = GeologyLinkRecord(
        age="Late Jurassic",
        locality="Italy",
        country="Italy",
        confidence=0.8,
    )
    sb = ScaleBarRecord(
        value=100.0,
        unit="um",
        source="caption",
        um_per_px=0.1,
        confidence=0.8,
    )
    meta = PanelMetadata(geology_links=[geo], scale_bar=sb)
    return PanelRecord(
        paper_id="p1",
        figure_id="f1",
        panel_id="1",
        species="Genus speciesum",
        panel_path="/tmp/p.png",
        bbox=[0, 0, 100, 100],
        confidence=0.9,
        metadata=meta,
        paper_metadata=pm,
    )


def _row_with_nan_coords() -> dict:
    """Return a row dict (write_csv input shape) with NaN/Inf embedded.

    write_csv / panels_to_rows project from RunOutput.panels
    (PanelRecord list). The producers feed ``panels_to_rows`` directly
    so NaN can sneak in via a hand-built panel dict.
    """
    return {
        "occurrenceID": "p1:f1:1",
        "paper_id": "p1",
        "figure_id": "f1",
        "panel_id": "1",
        "scientificName": "Genus speciesum",
        "basisOfRecord": "FossilSpecimen",
        "eventDate": "2020",
        "locality": "Italy",
        "decimalLatitude": float("nan"),  # Bug 6.8: was "nan" in CSV
        "decimalLongitude": float("inf"),  # Bug 6.8: was "inf" in CSV
        "geologicalContextID": "Late Jurassic",
        "formation": "",
        "identifiedBy": "A",
        "associatedReferences": "10.1/x",
        "scale_bar_value": float("inf"),  # was "inf" in CSV
        "scale_bar_unit": "um",
        "scale_bar_um_per_px": 0.1,
        "label_text": "",
        "confidence": 0.9,
        "matcher_type": "heuristic",
        "extraction_source": "opendataloader",
        "panel_path": "/tmp/p.png",
    }


def test_analysis_csv_drops_nan_inf(tmp_path: Path):
    """The analysis CSV must replace NaN/Inf with empty strings.

    We bypass ``write_csv`` and call ``_sanitise_csv_cell`` directly
    on each cell, which is the contract: any cell that goes through
    the sanitiser must have NaN/Inf replaced with empty strings.
    """
    row = _row_with_nan_coords()
    sanitised = {k: _sanitise_csv_cell(v) for k, v in row.items()}
    for k in ("decimalLatitude", "decimalLongitude", "scale_bar_value"):
        assert sanitised[k] == "", (
            f"key={k!r} should sanitise to '' (NaN/Inf); got {sanitised[k]!r}. "
            "Phase 63 Plan 6.8 fix regressed?"
        )
    # Sanity: finite floats survive
    assert sanitised["scale_bar_um_per_px"] == 0.1
    assert sanitised["confidence"] == 0.9


# Phase 63 Plan 6.8 (Bug 6.8): the same NaN/Inf treatment applies to
# the xlsx exporter. openpyxl rejects ``nan``/``inf`` strings, so the
# sanitiser drops them to ``""`` here too. We use a source-guard test
# that pins the behaviour without requiring openpyxl at test-time.


def test_xlsx_sanitiser_handles_nan_inf():
    """``xlsx._sanitise`` must replace NaN/Inf with the empty string."""
    assert "math.isnan" in XLSX_SRC and "math.isinf" in XLSX_SRC, (
        "xlsx._sanitise does not check math.isnan/isinf — Phase 63 Plan 6.8 fix regressed?"
    )


def test_xlsx_sanitiser_returns_empty_for_nan():
    """Source-guard: xlsx._sanitise drops NaN to ``""`` via the early-return path."""
    assert "if math.isnan(value) or math.isinf(value):" in XLSX_SRC
    assert 'return ""' in XLSX_SRC


def test_xlsx_sanitiser_preserves_int_bool_str():
    """Source-guard: xlsx._sanitise still handles ints, bools, and formula-prefixed strings."""
    assert "isinstance(value, int):" in XLSX_SRC
    assert "isinstance(value, bool):" in XLSX_SRC
    assert "_EXCEL_DANGER_PREFIXES" in XLSX_SRC


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
