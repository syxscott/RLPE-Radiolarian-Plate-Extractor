"""Regression: audit 2026-09-04 taxon-4 — range-chart linkage treats
cf./aff. as equivalent and promotes uncertain determinations.

``range_chart_extractor._norm`` stripped ``cf.`` and ``aff.`` before
species comparison, so ``_species_match('Parvicingula cf. jamesi',
'Parvicingula jamesi')`` returned True. ``build_geology_links_for_panels``
then emitted a ``GeologyLinkRecord`` carrying the chart row's biozone,
section, age_range and formations under the *chart* species name —
and a panel determined only as ``cf. jamesi`` was reported as an
occurrence of definite ``jamesi`` in the exported occurrence table,
PBDB submission, and GBIF Darwin Core record. cf. and aff. are
different ICZN assertions (cf. = tentative identification, aff. =
similar but distinct species); conflating either with the definite
species corrupts downstream analysis and overstates the dataset.

Fix contract: ``_norm`` preserves the qualifier; ``_species_match``
rejects comparisons where the two sides disagree on the qualifier
status (cf. vs definite, cf. vs aff., aff. vs definite); only an
exact qualifier match (cf. vs cf., aff. vs aff.) passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.range_chart_extractor import _norm, _species_match


class TestQualifierPreservedInNorm:
    def test_norm_preserves_cf_qualifier(self):
        # cf. is no longer silently deleted — the qualifier must
        # survive so downstream logic can distinguish the assertion.
        out = _norm("Parvicingula cf. jamesi")
        assert "cf." in out.split()
        # And it lower-cases + collapses whitespace for case-insensitive
        # comparison.
        assert out == "parvicingula cf. jamesi"

    def test_norm_preserves_aff_qualifier(self):
        assert _norm("Parvicingula aff. jamesi") == "parvicingula aff. jamesi"

    def test_norm_strips_trailing_sp_still(self):
        # Existing behaviour pinned: trailing " sp." stays stripped.
        assert _norm("Parvicingula sp.") == "parvicingula"
        assert _norm("Parvicingula sp") == "parvicingula"


class TestSpeciesMatchQualifierAware:
    def test_cf_vs_definite_does_not_match(self):
        # The exact historical bug: cf. jamesi (panel) vs jamesi (chart)
        # was a silent True and built a wrong biozone link.
        assert _species_match("Parvicingula jamesi", "Parvicingula cf. jamesi") is False
        assert _species_match("Parvicingula cf. jamesi", "Parvicingula jamesi") is False

    def test_cf_vs_aff_does_not_match(self):
        # Different ICZN assertions must not collapse.
        assert _species_match("Parvicingula cf. jamesi", "Parvicingula aff. jamesi") is False
        assert _species_match("Parvicingula aff. jamesi", "Parvicingula cf. jamesi") is False

    def test_aff_vs_definite_does_not_match(self):
        assert _species_match("Parvicingula jamesi", "Parvicingula aff. jamesi") is False

    def test_same_cf_cf_matches(self):
        assert _species_match(
            "Parvicingula cf. jamesi", "Parvicingula cf. jamesi"
        ) is True

    def test_same_aff_aff_matches(self):
        assert _species_match(
            "Parvicingula aff. jamesi", "Parvicingula aff. jamesi"
        ) is True

    def test_definite_definite_still_matches(self):
        # No qualifier on either side: behaviour unchanged.
        assert _species_match(
            "Parvicingula jamesi", "Parvicingula jamesi"
        ) is True

    def test_bare_genus_vs_definite_still_matches(self):
        # Existing behaviour pinned: a bare genus links to its definite
        # species (this is the bare-genus disambiguation path the
        # pipeline relies on).
        assert _species_match("Parvicingula", "Parvicingula jamesi") is True
        assert _species_match("Parvicingula jamesi", "Parvicingula") is True