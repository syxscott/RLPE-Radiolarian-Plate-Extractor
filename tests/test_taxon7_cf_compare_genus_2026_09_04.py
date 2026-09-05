"""Regression: audit 2026-09-04 taxon-7 —
:func:`rlpe.association.extract_taxa_from_caption`'s cf./aff. comparison
fallback appended the COMPARED SPECIES' epithet alone, dropping the
genus / author-initial token that disambiguates it.

Example real caption in Bandini 2011 pl08::

    "Archaeodictyomitra mitra cf. S. excelsa"

The first sweep consumed ``Archaeodictyomitra mitra`` as the canonical
binomial. The CF-COMPARE fallback then surfaced just ``excelsa`` (the
epithet) — but ``S.`` (author initial for the compared species) is
the critical disambiguator because multiple genera have an ``excelsa``
epithet. Without the ``S.`` token the downstream mapping cannot tell
which ``excelsa`` the determiner meant.

Fix contract: append ``"{genus_or_initial} {epithet}"`` so both
shapes — ``cf. S. excelsa`` and ``cf. Stichocapsa excelsa`` — are
surfaced as a unit. A bare ``excelsa`` alone (no preceding genus /
initial) is never appended.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.association import (  # noqa: E402
    TAXON_CF_COMPARE_PATTERN,
    extract_taxa_from_caption,
)


class TestCfCompareSurfacesFullComparedBinomial:
    def test_author_initial_plus_epithet_surfaced(self):
        taxa = extract_taxa_from_caption(
            "Archaeodictyomitra mitra cf. S. excelsa",
        )
        bearing = [t for t in taxa if "excelsa" in t]
        assert bearing, taxa
        # The "S." disambiguator must ride along with "excelsa".
        assert any("S." in t for t in bearing), taxa

    def test_full_genus_plus_epithet_comparison_surfaced(self):
        taxa = extract_taxa_from_caption(
            "Stichomitra cf. Stichocapsa excelsa",
        )
        assert any("Stichocapsa" in t and "excelsa" in t for t in taxa), taxa

    def test_aff_form_also_surfaces_genus(self):
        taxa = extract_taxa_from_caption(
            "Pseudoeucyrtis aff. S. hannai",
        )
        assert any("hannai" in t and "S." in t for t in taxa), taxa

    def test_bare_epithet_no_longer_pollutes(self):
        taxa = extract_taxa_from_caption(
            "Archaeodictyomitra mitra cf. S. excelsa",
        )
        bare = [t for t in taxa if t.strip().lower() == "excelsa"]
        assert not bare, taxa


class TestRegexShapeUnchanged:
    def test_pattern_still_captures_both_shapes(self):
        m = TAXON_CF_COMPARE_PATTERN.search("cf. S. excelsa")
        assert m is not None
        assert m.group(2) == "excelsa"
        m = TAXON_CF_COMPARE_PATTERN.search("cf. Stichocapsa excelsa")
        assert m is not None
        assert m.group(2) == "excelsa"
