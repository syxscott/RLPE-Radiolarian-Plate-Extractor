"""P0 crash regression tests for Phase 58 Plan 1.1.

Bug 1.1: GUI "Export DwC-A" button crashed because ``write_dwca_zip``
expected a ``RunOutput`` object but received a plain ``dict``.

The fix must accept both shapes (dict OR RunOutput) so the GUI code
that builds a ``dict`` payload keeps working.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from rlpe.exporters.archive import write_dwca_zip


def _gui_run_dict() -> dict:
    """Mirror the exact dict shape that ResultsTab._build_run_output returns.

    Note: this dict has a *minimal* provenance (``job_id`` + ``source``)
    rather than a full ``ProvenanceRecord``. The signature relaxation
    must accommodate that mismatch by filling in stub fields instead of
    raising ``ValidationError``.
    """
    return {
        "schema_version": "1.0.0",
        "provenance": {"job_id": "gui-test-job", "source": "rlpe-gui"},
        "papers": [],
        "figures": [],
        "panels": [],
        "taxa": [],
        "samples": [],
        "geology_contexts": [],
        "localities": [],
        "paleo_coordinates": [],
        "warnings": [],
    }


def test_write_dwca_zip_accepts_dict(tmp_path: Path) -> None:
    """Bug 1.1: ``write_dwca_zip`` must accept a ``dict`` payload.

    The GUI's ``_build_run_output`` returns a plain dict — calling
    ``write_dwca_zip(dict, path)`` previously raised ``TypeError``
    (object has no attribute ``panels``). After the fix, it should
    transparently coerce via ``RunOutput.model_validate``.
    """
    target = tmp_path / "out.zip"
    n = write_dwca_zip(_gui_run_dict(), target)
    assert n == 0
    assert target.exists()
    with zipfile.ZipFile(target) as zf:
        assert "occurrence.txt" in zf.namelist()
        assert "meta.xml" in zf.namelist()
        assert "eml.xml" in zf.namelist()


def test_write_dwca_zip_dict_with_panels(tmp_path: Path) -> None:
    """Passing a dict with real panels should produce a row per panel.

    A panel without ``species`` is dropped by default (no DwC row),
    so we include ``species`` to make sure the panel survives the
    coercion path.
    """
    run_dict = _gui_run_dict()
    run_dict["panels"] = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "Genus speciesum",
            "panel_path": "/tmp/p.png",
            "bbox": [10, 20, 100, 200],
            "confidence": 0.7,
            "metadata": {"geology_links": []},
            "paper_metadata": {
                "title": "T",
                "authors": ["A"],
                "year": 2020,
                "source": "opendataloader",
                "confidence": 0.8,
            },
        }
    ]
    target = tmp_path / "out.zip"
    n = write_dwca_zip(run_dict, target)
    assert n == 1
    with zipfile.ZipFile(target) as zf:
        occ = zf.read("occurrence.txt").decode()
    lines = occ.strip().split("\n")
    assert len(lines) == 2  # header + 1 row
    header = lines[0].split("\t")
    row = lines[1].split("\t")
    d = dict(zip(header, row))
    assert d["scientificName"] == "Genus speciesum"
    assert d["occurrenceID"] == "p1:f1:1"
