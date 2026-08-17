"""Tests for Phase 63 Plan 6 — Bug 6.10: CSV exports written without
UTF-8 BOM.

Before: ``analysis.write_csv`` and the legacy ``export_csv`` both
wrote with ``encoding="utf-8"``. Excel on Windows defaults to ANSI
when reading CSVs (no BOM) and mangles Greek / CJK chars in
scientificName / locality fields.

After: both default to ``encoding="utf-8-sig"`` — the 3-byte UTF-8
BOM ``\\xEF\\xBB\\xBF`` lets Excel detect the encoding and render
the cell values verbatim. ``csv.DictReader`` and Pandas both
transparently skip the BOM.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.export import export_csv  # noqa: E402
from rlpe.exporters import AnalysisOptions, write_csv  # noqa: E402
from rlpe.exporters.analysis import AnalysisOptions as AO  # noqa: E402
from rlpe.provenance import build_provenance  # noqa: E402
from rlpe.schema_models import (  # noqa: E402
    PanelMetadata,
    PanelRecord,
    PaperMetadataRecord,
    ProvenanceRecord,
    RunOutput,
)


def test_analysis_csv_encoding_default_is_utf8_sig():
    """``AnalysisOptions.csv_encoding`` must default to ``utf-8-sig``."""
    # ``AnalysisOptions`` uses slots, so reading the class attribute
    # directly returns the descriptor. Construct an empty instance
    # to read the actual default.
    opts = AnalysisOptions()
    assert opts.csv_encoding == "utf-8-sig", (
        f"AnalysisOptions.csv_encoding default is {opts.csv_encoding!r}, expected 'utf-8-sig'. "
        "Phase 63 Plan 6.10 fix regressed?"
    )


def _dummy_run() -> RunOutput:
    prov = ProvenanceRecord(**build_provenance().to_dict())
    pm = PaperMetadataRecord(
        title="Test",
        authors=["Author"],
        year=2020,
        doi="10.1/t",
        source="opendataloader",
        confidence=0.8,
    )
    meta = PanelMetadata()
    panel = PanelRecord(
        paper_id="p1",
        figure_id="f1",
        panel_id="1",
        species="Genus species",
        panel_path="/tmp/p.png",
        bbox=[0, 0, 100, 100],
        confidence=0.9,
        metadata=meta,
        paper_metadata=pm,
    )
    return RunOutput(provenance=prov, panels=[panel])


def test_analysis_csv_has_bom(tmp_path: Path):
    """The written CSV starts with the 3-byte UTF-8 BOM."""
    target = tmp_path / "out.csv"
    write_csv(_dummy_run(), target)
    raw = target.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf", (
        f"analysis CSV missing UTF-8 BOM; first bytes={raw[:6]!r}. "
        "Phase 63 Plan 6.10 fix regressed?"
    )


def test_analysis_csv_readable_as_utf8_sig(tmp_path: Path):
    """Reading with ``utf-8-sig`` yields the BOM-stripped first line."""
    target = tmp_path / "out.csv"
    write_csv(_dummy_run(), target)
    with open(target, encoding="utf-8-sig") as f:
        first_line = f.readline()
    assert first_line.startswith("occurrenceID"), first_line[:80]


def test_export_csv_has_bom(tmp_path: Path):
    """The legacy ``export_csv`` (rlpe.export) writes with BOM too."""
    target = tmp_path / "legacy.csv"
    rows = [
        {"paper_id": "p1", "scientificName": "Genus species", "locality": "China"},
        {"paper_id": "p1", "scientificName": "Genus alterum", "locality": "Japan"},
    ]
    export_csv(rows, target)
    raw = target.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf", (
        f"export_csv missing UTF-8 BOM; first bytes={raw[:6]!r}. Phase 63 Plan 6.10 fix regressed?"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
