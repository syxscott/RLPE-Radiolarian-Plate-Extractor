"""Audit 2026-09-03 (BLOCKER-#9): geo_coords British/French spelling.

The previous ``_PALEO_KEYWORDS_GEO`` table only matched American
spellings ("Palaeocene", "Palaeogene", "Palaeozoic"). Classical
radiolarian literature uses British ("Palaeocene") and French
("Paléocène") spellings extensively (De Wever 2001, O'Dogherty
1994, Hollis 1997). Missing these caused 56 Ma occurrences in
"Palaeocene this region lay at 35°S, 110°E" text to be
mis-classified as modern coordinates — corrupting downstream GBIF
submissions.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestBritishPalaeoceneSpellings:
    """British English — the spelling actually used in De Wever 2001,
    O'Dogherty 1994, Hollis 1997."""

    def test_palaeocene_detected(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "...during the Palaeocene this region lay at 35°S, 110°E"
        idx = text.find("35")
        assert _is_paleo_text(text, idx) is True, (
            "British 'Palaeocene' not detected as paleo context"
        )

    def test_palaeogene_detected(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "reconstructed position in the Palaeogene at 25°N, 60°E"
        idx = text.find("25")
        assert _is_paleo_text(text, idx) is True

    def test_palaeozoic_detected(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "during the Palaeozoic, the region was situated at 45°S"
        idx = text.find("45")
        assert _is_paleo_text(text, idx) is True

    def test_cainozoic_detected(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "in the Cainozoic this region lay at 20°N, 30°E"
        idx = text.find("20")
        assert _is_paleo_text(text, idx) is True

    def test_palaeontological_detected(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "palaeontological evidence at 10°S, 80°E"
        idx = text.find("10")
        assert _is_paleo_text(text, idx) is True


class TestFrenchAccentSpellings:
    """French — De Wever 2001 et al. native spelling."""

    def test_paleocene_accent_detected(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "pendant le Paléocène, à 35°S, 110°E"
        idx = text.find("35")
        assert _is_paleo_text(text, idx) is True, "French accented 'Paléocène' not detected"

    def test_mesozoique_detected(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "pendant le Mésozoïque, à 35°S, 110°E"
        idx = text.find("35")
        assert _is_paleo_text(text, idx) is True

    def test_cenozoique_detected(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "Cénozoïque, à 25°N, 60°E"
        idx = text.find("25")
        assert _is_paleo_text(text, idx) is True


class TestAmericanSpellingsStillWork:
    """The previous American spellings must still resolve to keep
    the no-regression invariant."""

    def test_paleocene_american_still_works(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "during the Paleocene this region lay at 35°S, 110°E"
        idx = text.find("35")
        assert _is_paleo_text(text, idx) is True

    def test_mesozoic_american_still_works(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "in the Mesozoic this region lay at 20°N, 30°E"
        idx = text.find("20")
        assert _is_paleo_text(text, idx) is True

    def test_cenozoic_american_still_works(self):
        from rlpe.geo_coords import _is_paleo_text

        text = "Cenozoic, 25°N, 60°E"
        idx = text.find("25")
        assert _is_paleo_text(text, idx) is True


class TestSourceGuard:
    """Source guard: the table must include the new variants. A
    future refactor that reverts the table drops this test."""

    def test_keywords_include_british_palaeo_variants(self):
        from rlpe.geo_coords import _PALEO_KEYWORDS_GEO

        table = " ".join(_PALEO_KEYWORDS_GEO)
        for kw in (
            "palaeocene",
            "palaeogene",
            "palaeozoic",
            "palaeontological",
            "cainozoic",
            "caenozoic",
            "paléocène",
            "paléogène",
            "paléozoïque",
            "mésozoïque",
            "cénozoïque",
        ):
            assert kw in _PALEO_KEYWORDS_GEO, (
                f"_PALEO_KEYWORDS_GEO missing '{kw}' (audit 2026-09-03 BLOCKER-#9 regression)"
            )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
