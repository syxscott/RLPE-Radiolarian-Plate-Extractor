"""Regression tests for audit 2026-08-01 batch W3 — converters M1/M2."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.converters import _extract_authorship, _taxon_parts  # noqa: E402


class TestSubgenusPostfix:
    """Bug M1 — postfix subgenus ``Podocyrtis amphora (Podocyrtites)``."""

    def test_postfix_subgenus_extracted(self) -> None:
        result = _taxon_parts("Podocyrtis amphora (Podocyrtites)")
        assert result["generic_name"] == "Podocyrtites"
        assert result["genus"] == "Podocyrtis"
        assert result["specific_epithet"] == "amphora"

    def test_prefix_subgenus_still_works(self) -> None:
        result = _taxon_parts("Podocyrtis (Podocyrtites) amphora")
        assert result["generic_name"] == "Podocyrtites"
        assert result["genus"] == "Podocyrtis"
        assert result["specific_epithet"] == "amphora"

    def test_no_subgenus(self) -> None:
        result = _taxon_parts("Triassocampe sp.")
        assert result["generic_name"] is None

    def test_extract_authorship_postfix(self) -> None:
        # _extract_authorship must also recognise the postfix shape so
        # the call site in ``_unique_taxon_records`` populates
        # ``generic_name`` on the TaxonRecord.
        _, subgenus, _ = _extract_authorship("Podocyrtis amphora (Podocyrtites)")
        assert subgenus == "Podocyrtites"


class TestAuthorityVsQualifier:
    """Bug M2 — ICZN authority citation must NOT become a qualifier."""

    def test_year_in_parens_is_authority(self) -> None:
        result = _taxon_parts("Podocyrtis (Podocyrtites) amphora (Haeckel, 1887)")
        assert result["authority"] == "(Haeckel, 1887)"
        assert result["qualifier"] is None
        # Subgenus (prefix shape) is still extracted.
        assert result["generic_name"] == "Podocyrtites"
        assert result["genus"] == "Podocyrtis"
        assert result["specific_epithet"] == "amphora"

    def test_qualifier_without_year(self) -> None:
        # Bare "?" on the genus, no parenthetical authority, no
        # subgenus.  The "?" is carried by the genus; the trailing
        # ``sp.`` lands in the qualifier field.  Critical: no
        # authority must be invented from the open-nomenclature
        # marker.
        result = _taxon_parts("Cenosphaera? sp.")
        assert result["genus"] == "Cenosphaera"
        assert result["qualifier"] is not None
        # The qualifier is a string capturing the open-nomenclature
        # marker (either ``?`` or ``sp.``/``sp`` — the existing
        # _taxon_parts absorbs ``sp.`` as a qualifier token).  Either
        # way, the parenthesised-authority fix must not have leaked
        # in here.
        assert isinstance(result["qualifier"], str)
        assert result["authority"] is None
        assert result["generic_name"] is None

    def test_authority_with_surname_only(self) -> None:
        # Single-token prefix with a parenthesised surname — the
        # parens are an authority citation, not a postfix subgenus
        # (no epithet to anchor the subgenus shape).
        result = _taxon_parts("X (Smith)")
        assert result["authority"] == "(Smith)"
        assert result["generic_name"] is None

    def test_extract_authorship_year_in_parens(self) -> None:
        # _extract_authorship must extract the authority string (no
        # parens) into the third tuple element for the call site.
        _, _, authorship = _extract_authorship("Podocyrtis (Podocyrtites) amphora (Haeckel, 1887)")
        assert authorship == "Haeckel, 1887"
