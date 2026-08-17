"""Tests for Phase 60 Plan 3 — Bug 3.2: known paleontologist author
surnames must not be matched as a genus token.

A common radiolarian-caption shape is::

    "figs 1-2. Entactinia compacta Riedel & Sanfilippo"

The species extraction code in :mod:`rlpe.taxon` treats
``Riedel & Sanfilippo`` as a binomial ``Riedel Sanfilippo`` and emits
the author surname as a fake species. The fix adds
``_KNOWN_AUTHOR_SURNAMES`` so any capitalised token whose ``lower()``
is in the set is rejected as a genus.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.taxon import (  # noqa: E402
    _KNOWN_AUTHOR_SURNAMES,
    TaxonRecognizer,
)


def _reco() -> TaxonRecognizer:
    return TaxonRecognizer()


def test_known_author_surnames_blocklist_is_nonempty():
    """Sanity check: the blocklist must contain at least the canonical
    radiolarian workers. If it ever drops to zero the regex will start
    emitting author surnames as binomials again."""
    assert len(_KNOWN_AUTHOR_SURNAMES) >= 25, (
        f"_KNOWN_AUTHOR_SURNAMES shrunk to {len(_KNOWN_AUTHOR_SURNAMES)}; "
        "add surnames back to maintain the fix"
    )


def test_author_surname_not_matched_as_genus():
    """\"Riedel & Sanfilippo\" as an author citation must NOT be
    matched as a binomial by the fallback species recognizer."""
    reco = _reco()
    text = "figs 1-2. Entactinia compacta Riedel & Sanfilippo"
    entities = reco._fallback_predict(text)
    taxon_texts = [e.text for e in entities]
    # The real species must still be present.
    assert any("Entactinia compacta" in t for t in taxon_texts), taxon_texts
    # The author surname must not be treated as a genus.
    assert not any(t == "Riedel" or t.startswith("Riedel ") for t in taxon_texts), (
        f"Author surname 'Riedel' leaked into taxon list: {taxon_texts}"
    )
    assert not any("Sanfilippo" in t for t in taxon_texts), (
        f"Author surname 'Sanfilippo' leaked into taxon list: {taxon_texts}"
    )


def test_author_surname_with_amperstand_blocked():
    """\"Bütschli & Haeckel\" — both authors blocked."""
    reco = _reco()
    text = "Some discussion by Bütschli & Haeckel about Entactinia"
    entities = reco._fallback_predict(text)
    taxon_texts = [e.text for e in entities]
    # Neither surname should appear as a genus.
    for surname in ("Bütschli", "Haeckel"):
        assert not any(surname in t for t in taxon_texts), (
            f"{surname!r} leaked into taxon list: {taxon_texts}"
        )


def test_author_surname_only_no_species_emits_nothing():
    """If the entire text is just an author surname + a common noun,
    no fake binomial should be produced."""
    reco = _reco()
    text = "Following Riedel and Kozur, the assemblage changes."
    entities = reco._fallback_predict(text)
    # No Riedel taxon, no Kozur taxon.
    for surname in ("Riedel", "Kozur"):
        assert not any(t == surname for t in (e.text for e in entities)), (
            f"{surname!r} leaked: {[e.text for e in entities]}"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
