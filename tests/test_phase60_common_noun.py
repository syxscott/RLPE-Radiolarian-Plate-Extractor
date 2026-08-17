"""Tests for Phase 60 Plan 3 — Bug 3.6: common nouns matched as
binomial species.

Real caption text frequently contains title-case bigram pairs that
look like a binomial but are actually a topic phrase:

  * ``Crystal Structure``
  * ``Late Jurassic``
  * ``Rosso Ammonitico Formation``
  * ``Crystal Distribution``
  * ``Sample Recovery``

The previous ``_NON_TAXON_SECOND_WORDS`` list blocked only ~20
high-frequency stopwords; the new fix extends it with a curated list
of common geological / microscopy / stratigraphy nouns that appear
in title case in real captions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.taxon import _NON_TAXON_SECOND_WORDS, TaxonRecognizer  # noqa: E402


def _reco() -> TaxonRecognizer:
    return TaxonRecognizer()


def test_extract_taxa_rejects_common_nouns():
    """Common-noun bigrams must NOT be emitted as binomials."""
    reco = _reco()
    cases = [
        "Crystal Structure of the matrix",
        "Late Jurassic ammonite assemblage",
        "Rosso Ammonitico Formation locality",
        "Crystal Distribution across the slide",
        "Sample Recovery from the borehole",
    ]
    for text in cases:
        entities = reco._fallback_predict(text)
        texts = [e.text for e in entities]
        # None of the title-case bigrams should appear.
        for bigram in [
            "Crystal Structure",
            "Late Jurassic",
            "Rosso Ammonitico",
            "Crystal Distribution",
            "Sample Recovery",
        ]:
            assert bigram not in texts, (
                f"{bigram!r} leaked into taxon list for text {text!r}: {texts}"
            )


def test_common_noun_blocklist_is_extended():
    """Sanity check: the new blocklist contains at least the canonical
    nouns from Bug 3.6's description."""
    needed = {
        "structure",
        "distribution",
        "occurrence",
        "assemblage",
        "fauna",
        "flora",
        "biostratigraphy",
        "recovery",
        "extinction",
        "diversity",
        "abundance",
        "range",
        "radiolarian",
        "sponge",
        "section",
        "outcrop",
        "sample",
        "specimen",
        "species",
        "genus",
        "group",
        "member",
        "zone",
        "age",
        "epoch",
        "era",
        "period",
        "stage",
        "system",
    }
    missing = needed - _NON_TAXON_SECOND_WORDS
    assert not missing, f"_NON_TAXON_SECOND_WORDS missing: {sorted(missing)}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
