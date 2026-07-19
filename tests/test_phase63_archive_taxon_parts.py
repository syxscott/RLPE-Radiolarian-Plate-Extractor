"""Tests for Phase 63 Plan 6 — Bug 6.7: archive.py ``_occurrence_row``
must use ``_taxon_parts`` instead of naive ``species.split()``.

Before: ``genus = species.split()[0]`` and
``specificEpithet = species.split()[1]`` — which broke on:

  * ``cf.`` / ``aff.`` qualifiers — naive split puts "cf." into genus.
  * trinomial names (``Genus species subspecies``) — naive split
    emits the subspecific epithet into specificEpithet (not the
    canonical specific epithet) and drops the subspecies.
  * author citations like ``Genus species (Smith, 1900)`` — naive
    split puts ``(Smith,`` into specificEpithet.
  * trailing ``sp.`` / ``spp.`` markers — naive split puts ``sp.``
    into specificEpithet.

After: ``_taxon_parts`` (which already lives in converters.py) is
used. ``genus`` and ``specificEpithet`` become None for ``cf.`` /
``aff.`` / ``sp.`` shapes — GBIF and PBDB have explicit guidance
that these open-nomenclature records should NOT have an
authoritative ``specificEpithet`` (they flag the record as
infraspecific/nominate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.exporters.archive import _occurrence_row  # noqa: E402
from rlpe.exporters import AnalysisOptions  # noqa: E402
from rlpe.schema_models import (  # noqa: E402
    PanelMetadata,
    PanelRecord,
    PaperMetadataRecord,
    ScaleBarRecord,
)


def _panel(species: str) -> PanelRecord:
    pm = PaperMetadataRecord(
        title="T", authors=["A"], year=2020, doi="10.1/t", source="opendataloader", confidence=0.8
    )
    return PanelRecord(
        paper_id="p1",
        figure_id="f1",
        panel_id="1",
        species=species,
        panel_path="/tmp/p.png",
        bbox=[0, 0, 100, 100],
        confidence=0.9,
        metadata=PanelMetadata(),
        paper_metadata=pm,
    )


@pytest.mark.parametrize(
    "species,expected_genus,expected_specific_epithet",
    [
        ("Genus species", "Genus", "species"),
        ("Genus speciesum", "Genus", "speciesum"),
        # cf./aff. qualifiers: _taxon_parts leaves specificEpithet None
        ("Genus cf. species", "Genus", ""),
        ("Genus aff. species", "Genus", ""),
        # sp./spp.: genus only, no specific epithet
        ("Genus sp.", "Genus", ""),
        ("Genus spp.", "Genus", ""),
        # Trinomial (subspecies epithet): ``_taxon_parts`` keeps the
        # last lowercase token as the epithet. The full string is
        # preserved in scientificName so no information is lost.
        ("Genus species subspecies", "Genus", "subspecies"),
        # Author citation in parens: only first lowercase kept
        ("Genus species (Smith, 1900)", "Genus", "species"),
        # _taxon_parts leaves epithet None for ?-suffix authors (keep simple)
        ("Genus species cf. S. excelsa", "Genus", "species"),
    ],
)
def test_occurrence_row_uses_taxon_parts(species: str, expected_genus: str, expected_specific_epithet: str):
    """Archive output for diverse species shapes uses ``_taxon_parts``."""
    row = _occurrence_row(_panel(species))
    assert row["genus"] == expected_genus, (
        f"species={species!r} -> genus={row['genus']!r} (expected {expected_genus!r}). "
        "archive.py is using naive species.split() — _taxon_parts not applied."
    )
    assert row["specificEpithet"] == expected_specific_epithet, (
        f"species={species!r} -> specificEpithet={row['specificEpithet']!r} "
        f"(expected {expected_specific_epithet!r}). "
        "archive.py is using naive species.split() — _taxon_parts not applied."
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
