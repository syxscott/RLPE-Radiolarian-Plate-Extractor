"""Phase 63 Plan 6 — Tasks 6.2/6.3 pinning regression tests.

Pins the Phase 58 Plan 1.2 (commit 169ee41) fix: DwC-A + analysis
exports must emit modern_latitude/modern_longitude in
decimalLatitude/decimalLongitude (the GBIF/PBDB-required fields).

Before the Phase 58 fix, every DwC-A export shipped null decimals
because archive.py read geo.latitude / geo.longitude, but Round 25+
converters populate geo.modern_latitude / modern_longitude.

These tests pin the precedence order and the legacy fallback so a
future refactor that accidentally re-reads the legacy fields will
fail loudly.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.exporters.analysis import AnalysisOptions, panels_to_rows  # noqa: E402
from rlpe.exporters.archive import write_dwca_zip  # noqa: E402
from rlpe.provenance import build_provenance  # noqa: E402
from rlpe.schema_models import (  # noqa: E402
    GeologyLinkRecord,
    PanelMetadata,
    PanelRecord,
    PaperMetadataRecord,
    ProvenanceRecord,
    RunOutput,
)


def _panel(
    latitude=None,
    longitude=None,
    modern_latitude=None,
    modern_longitude=None,
) -> PanelRecord:
    geo = GeologyLinkRecord(
        age="Late Jurassic",
        locality="Italy",
        country="Italy",
        latitude=latitude,
        longitude=longitude,
        modern_latitude=modern_latitude,
        modern_longitude=modern_longitude,
        confidence=0.8,
    )
    pm = PaperMetadataRecord(
        title="T",
        authors=["A"],
        year=2020,
        doi="10.1/test",
        source="opendataloader",
        confidence=0.8,
    )
    meta = PanelMetadata(geology_links=[geo])
    return PanelRecord(
        paper_id="p1",
        figure_id="f1",
        panel_id="1",
        species="Genus speciesum",
        panel_path="/tmp/p.png",
        bbox=[0, 0, 100, 100],
        confidence=0.7,
        metadata=meta,
        paper_metadata=pm,
    )


def _run(panel: PanelRecord) -> RunOutput:
    prov = ProvenanceRecord(**build_provenance().to_dict())
    return RunOutput(provenance=prov, panels=[panel])


def _read_occurrence_row(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as zf:
        occ = zf.read("occurrence.txt").decode()
    lines = occ.strip().split("\n")
    header = lines[0].split("\t")
    row = lines[1].split("\t")
    return dict(zip(header, row))


def test_dwca_pins_modern_coords(tmp_path: Path):
    """Phase 58 Plan 1.2: when modern_* is set, decimalLatitude/Longitude
    must come from modern_*, NOT from legacy latitude/longitude.

    Pins commit 169ee41.
    """
    panel = _panel(
        latitude=None,
        longitude=None,
        modern_latitude=36.5,
        modern_longitude=4.8,
    )
    out_zip = tmp_path / "out.zip"
    write_dwca_zip(_run(panel), out_zip)
    row = _read_occurrence_row(out_zip)
    assert row["decimalLatitude"] == "36.5"
    assert row["decimalLongitude"] == "4.8"


def test_dwca_pins_modern_coords_precedence(tmp_path: Path):
    """When BOTH modern_* and legacy are populated, modern_* wins."""
    panel = _panel(
        latitude=46.5,
        longitude=11.5,
        modern_latitude=36.5,
        modern_longitude=4.8,
    )
    out_zip = tmp_path / "out.zip"
    write_dwca_zip(_run(panel), out_zip)
    row = _read_occurrence_row(out_zip)
    assert row["decimalLatitude"] == "36.5", (
        f"modern_* must take precedence over legacy; got {row['decimalLatitude']!r}"
    )
    assert row["decimalLongitude"] == "4.8"


def test_analysis_pins_modern_coords():
    """Analysis CSV must also read modern_latitude/longitude first."""
    panel = _panel(
        latitude=None,
        longitude=None,
        modern_latitude=36.5,
        modern_longitude=4.8,
    )
    rows = panels_to_rows(_run(panel), AnalysisOptions(include_unmatched=True))
    assert len(rows) == 1
    r = rows[0]
    # The values are emitted as raw floats in the analysis dict (becomes
    # ``str`` only after passing through csv.DictWriter).
    assert r["decimalLatitude"] == 36.5
    assert r["decimalLongitude"] == 4.8


def test_legacy_coords_still_emit():
    """Backwards-compat: legacy latitude/longitude (no modern_*) still emit."""
    panel = _panel(latitude=46.5, longitude=11.5)
    rows = panels_to_rows(_run(panel), AnalysisOptions(include_unmatched=True))
    r = rows[0]
    assert r["decimalLatitude"] == 46.5
    assert r["decimalLongitude"] == 11.5


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
