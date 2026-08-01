"""Regression tests for audit 2026-08-01 batch W3 — taxon M6 author surnames."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.taxon import (  # noqa: E402
    _KNOWN_AUTHOR_SURNAMES,
    _is_valid_species,
)


class TestAuthorSurnames:
    """Audit 2026-08-01 batch W3 (Bug M6): _KNOWN_AUTHOR_SURNAMES was
    missing common Asian and Russian radiolarian worker surnames. The
    LLM-first hybrid path was reading "Wu (in Wu, 1986)" as the species
    "Wu 1986" and producing bogus species records."""

    def test_Wu_recognized(self) -> None:
        """\"Wu 1986\" attribution is rejected as a species, because
        the genus token ``Wu`` is a known author surname."""
        # Sanity: Wu is in the blocklist (case-insensitive comparison).
        assert "wu" in _KNOWN_AUTHOR_SURNAMES
        # The LLM-validity check must reject "Wu 1986" as a species
        # because the genus token is an author surname.
        assert _is_valid_species("Wu 1986") is False
        # Mixed-case also rejected.
        assert _is_valid_species("wu 1986") is False
        assert _is_valid_species("WU 1986") is False

    def test_Bragin_recognized(self) -> None:
        """\"Bragin 2025\" attribution is rejected as a species."""
        assert "bragin" in _KNOWN_AUTHOR_SURNAMES
        assert _is_valid_species("Bragin 2025") is False

    def test_Sashida_recognized(self) -> None:
        """\"Sashida\" attribution (e.g. \"Sashida 1983\") is rejected."""
        assert "sashida" in _KNOWN_AUTHOR_SURNAMES
        assert _is_valid_species("Sashida 1983") is False

    def test_full_list_present(self) -> None:
        """All the audit 2026-08-01 batch W3 (Bug M6) Chinese, Japanese
        and Russian surnames must be present in the blocklist. Zhang
        was already in the ``Core`` set so it is excluded here."""
        new_surnames = {
            # Chinese
            "wu",
            "li",
            "wang",
            "chen",
            "liu",
            "yang",
            "huang",
            # Japanese
            "sashida",
            "kamata",
            "nagai",
            "mizutani",
            "sakai",
            "toyota",
            "kurihara",
            # Russian
            "bragin",
            "korchagin",
            "tikhomirova",
            "kazintsova",
            "afanasieva",
        }
        missing = new_surnames - _KNOWN_AUTHOR_SURNAMES
        assert not missing, (
            f"Audit 2026-08-01 Bug M6: missing surnames in "
            f"_KNOWN_AUTHOR_SURNAMES: {sorted(missing)}"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
