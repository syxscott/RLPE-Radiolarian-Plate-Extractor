"""P0 regression: Phase 58 Plan 1.2 (Bug 1.2).

``GeologyLinkRecord`` carries both legacy ``latitude/longitude`` and
newer ``modern_latitude/modern_longitude`` fields. Converters (Round 25
onwards) populate ``modern_*``; the DwC-A exporter was reading the
legacy fields, exporting null coords. GBIF/PBDB reject nulls.

Fix: read ``modern_latitude/longitude`` first, fall back to legacy
``latitude/longitude``.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

from rlpe.exporters.analysis import AnalysisOptions, panels_to_rows
from rlpe.exporters.archive import write_dwca_zip
from rlpe.provenance import build_provenance
from rlpe.schema_models import (
    GeologyLinkRecord,
    PanelMetadata,
    PanelRecord,
    PaperMetadataRecord,
    ProvenanceRecord,
    RunOutput,
    ScaleBarRecord,
)


def _panel_with_modern_coords() -> PanelRecord:
    """Panel whose only geology link uses modern_latitude/longitude.

    legacy ``latitude/longitude`` left as None to simulate the Round 25+
    converter path that exclusively fills modern_* fields.
    """
    geo = GeologyLinkRecord(
        age="Late Jurassic",
        locality="Italy",
        country="Italy",
        # legacy fields left None
        latitude=None,
        longitude=None,
        # modern coords present
        modern_latitude=36.5,
        modern_longitude=4.8,
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
    sb = ScaleBarRecord(value=100.0, unit="um", source="caption", um_per_px=0.1, confidence=0.8)
    meta = PanelMetadata(
        geology_links=[geo],
        scale_bar=sb,
    )
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


def _make_run() -> RunOutput:
    prov = ProvenanceRecord(**build_provenance().to_dict())
    return RunOutput(provenance=prov, panels=[_panel_with_modern_coords()])


def test_dwca_modern_coords_emitted(tmp_path: Path) -> None:
    """decimalLatitude/decimalLongitude must be populated from modern_*."""
    target = tmp_path / "out.zip"
    n = write_dwca_zip(_make_run(), target)
    assert n == 1
    with zipfile.ZipFile(target) as zf:
        occ = zf.read("occurrence.txt").decode()
    lines = occ.strip().split("\n")
    header = lines[0].split("\t")
    row = lines[1].split("\t")
    d = dict(zip(header, row))
    # The fix: these must NOT be empty.
    assert d["decimalLatitude"] != "", (
        f"decimalLatitude was empty; got row {d!r}. "
        "archive.py is reading legacy geo.latitude (None) instead of "
        "geo.modern_latitude (36.5)."
    )
    assert d["decimalLongitude"] != "", (
        f"decimalLongitude was empty; got row {d!r}. "
        "archive.py is reading legacy geo.longitude (None) instead of "
        "geo.modern_longitude (4.8)."
    )
    assert d["decimalLatitude"] == "36.5"
    assert d["decimalLongitude"] == "4.8"


def test_analysis_csv_modern_coords_emitted() -> None:
    """Analysis CSV (analysis.py) must also read modern_latitude/longitude."""
    rows = panels_to_rows(_make_run(), AnalysisOptions(include_unmatched=True))
    assert len(rows) == 1
    r = rows[0]
    assert r["decimalLatitude"] != "", (
        f"decimalLatitude empty in analysis row {r!r}; "
        "analysis.py is reading legacy geo.latitude instead of modern_latitude."
    )
    assert r["decimalLongitude"] != "", (
        f"decimalLongitude empty in analysis row {r!r}; "
        "analysis.py is reading legacy geo.longitude instead of modern_longitude."
    )
    assert r["decimalLatitude"] == 36.5
    assert r["decimalLongitude"] == 4.8


def test_dwca_legacy_coords_still_work(tmp_path: Path) -> None:
    """Backwards-compat: legacy ``latitude/longitude`` (no modern_*) still emit."""
    geo = GeologyLinkRecord(
        age="Late Jurassic",
        locality="Italy",
        country="Italy",
        latitude=46.5,
        longitude=11.5,
        confidence=0.8,
    )
    pm = PaperMetadataRecord(
        title="T", authors=["A"], year=2020, source="opendataloader", confidence=0.8
    )
    meta = PanelMetadata(geology_links=[geo])
    panel = PanelRecord(
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
    prov = ProvenanceRecord(**build_provenance().to_dict())
    run = RunOutput(provenance=prov, panels=[panel])
    target = tmp_path / "out.zip"
    write_dwca_zip(run, target)
    with zipfile.ZipFile(target) as zf:
        occ = zf.read("occurrence.txt").decode()
    lines = occ.strip().split("\n")
    header = lines[0].split("\t")
    row = lines[1].split("\t")
    d = dict(zip(header, row))
    assert d["decimalLatitude"] == "46.5"
    assert d["decimalLongitude"] == "11.5"


def test_dwca_modern_coords_take_precedence(tmp_path: Path) -> None:
    """If BOTH modern_* and legacy are set, modern_* wins (Round 25+ convention)."""
    geo = GeologyLinkRecord(
        age="Late Jurassic",
        locality="Italy",
        country="Italy",
        latitude=46.5,  # legacy
        longitude=11.5,  # legacy
        modern_latitude=36.5,  # newer (preferred)
        modern_longitude=4.8,  # newer (preferred)
        confidence=0.8,
    )
    pm = PaperMetadataRecord(
        title="T", authors=["A"], year=2020, source="opendataloader", confidence=0.8
    )
    meta = PanelMetadata(geology_links=[geo])
    panel = PanelRecord(
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
    prov = ProvenanceRecord(**build_provenance().to_dict())
    run = RunOutput(provenance=prov, panels=[panel])
    target = tmp_path / "out.zip"
    write_dwca_zip(run, target)
    with zipfile.ZipFile(target) as zf:
        occ = zf.read("occurrence.txt").decode()
    lines = occ.strip().split("\n")
    header = lines[0].split("\t")
    row = lines[1].split("\t")
    d = dict(zip(header, row))
    assert d["decimalLatitude"] == "36.5"
    assert d["decimalLongitude"] == "4.8"
