"""Regression tests for audit 2026-08-01 batch W3 — cross_figure M7 formation blocklist."""

from __future__ import annotations

from rlpe.cross_figure_linker import _extract_locality_phrases
from rlpe.sample_id_extractor import _LOCALITY_BLOCKLIST


def _phrase_values(phrases: list[str]) -> list[str]:
    """Helper: return only the lowercase phrase values for easy membership checks."""
    return [p.casefold() for p in phrases]


class TestCrossFigureLocalityBlocklist:
    """_extract_locality_phrases must skip formation-name false-positives."""

    def test_scaglia_not_locality(self) -> None:
        """Scaglia Fm should not contribute 'Scaglia' as a locality phrase."""
        phrases = _extract_locality_phrases("Scaglia Fm, late Cretaceous")
        assert "scaglia" not in _phrase_values(phrases), (
            f"'Scaglia' is a lithostratigraphic formation name, not a locality. Got: {phrases}"
        )

    def test_maiolica_not_locality(self) -> None:
        """Maiolica Fm should not contribute 'Maiolica' as a locality phrase."""
        phrases = _extract_locality_phrases("Maiolica Fm")
        assert "maiolica" not in _phrase_values(phrases), (
            f"'Maiolica' is a formation name, not a locality. Got: {phrases}"
        )

    def test_rosso_ammonitico_blocked(self) -> None:
        """Rosso Ammonitico is a multi-word formation name; both words must not appear."""
        phrases = _extract_locality_phrases("Rosso Ammonitico Fm, late Jurassic")
        vals = _phrase_values(phrases)
        assert "rosso ammonitico" not in vals, (
            f"'Rosso Ammonitico' is a formation, not a locality. Got: {phrases}"
        )
        # Sanity: also ensure 'rosso' alone is filtered (case-insensitive).
        assert "rosso" not in vals

    def test_real_locality_still_extracted(self) -> None:
        """A real locality name alongside a formation must still be extracted."""
        phrases = _extract_locality_phrases("Tunisia, Scaglia Fm")
        vals = _phrase_values(phrases)
        assert "tunisia" in vals, (
            f"Real locality 'Tunisia' must still be extracted alongside "
            f"the formation 'Scaglia'. Got: {phrases}"
        )
        assert "scaglia" not in vals

    def test_radiolarian_chert_blocked(self) -> None:
        """'Radiolarian Chert' is a lithology term, not a locality."""
        phrases = _extract_locality_phrases("Radiolarian Chert")
        assert "radiolarian chert" not in _phrase_values(phrases), (
            f"'Radiolarian Chert' is a lithology, not a locality. Got: {phrases}"
        )

    def test_blocklist_terms_canonical(self) -> None:
        """All required formation-name terms live in the canonical blocklist."""
        for term in (
            "scaglia",
            "rosso ammonitico",
            "maiolica",
            "biancone",
            "fonzaso",
            "sicani",
            "radiolarian chert",
        ):
            assert term in _LOCALITY_BLOCKLIST, (
                f"Expected {term!r} in _LOCALITY_BLOCKLIST (audit 2026-08-01 M7)."
            )
