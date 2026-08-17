"""Tests for Phase 60 Plan 3 — Bug 3.5: Greek-letter open nomenclature.

Real radiolarian captions use Greek letters as informal epithets:

  * ``Entactinia α``     (alpha)
  * ``Stichocapsa β``    (beta)
  * ``Cenodiscus sp. A``  (capital Latin A as informal variant)

The previous regex required the second token to start with a
lowercase Latin letter ``[a-z][a-zA-Z-]{2,}`` and the isolated-genus
qualifier branch only accepted ``sp. / spp. / n. sp. / sp. nov. / nom.
nov. / comb. nov.`` — Greek letters were silently dropped, and the
single uppercase ``A`` form was rejected because the second token
must be lowercase or one of the qualifiers.

The fix extends the isolated-genus branch to accept a Greek letter
OR a single capital letter as a valid second token.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.taxon import TaxonRecognizer  # noqa: E402


def _reco() -> TaxonRecognizer:
    return TaxonRecognizer()


def test_extract_taxa_handles_greek_letters():
    """Greek letters α / β / γ as second token must be accepted."""
    reco = _reco()
    text = "figs 1-2. Entactinia α"
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    # The folded surface form (``alpha`` or ``Entactinia α``) is
    # acceptable. We check that the genus name was matched with the
    # Greek-letter second token attached.
    assert any("Entactinia" in t and ("α" in t or "alpha" in t) for t in texts), texts


def test_extract_taxa_handles_greek_letter_beta():
    """β variant."""
    reco = _reco()
    text = "figs 1-2. Stichocapsa β"
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    assert any("Stichocapsa" in t and ("β" in t or "beta" in t) for t in texts), texts


def test_extract_taxa_handles_sp_capital_letter():
    """``Genus sp. A`` — capital letter as the informal-variant
    descriptor (common in De Wever bandini-style plates)."""
    reco = _reco()
    text = "figs 1-2. Entactinia sp. A"
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    assert any("Entactinia sp. A" in t for t in texts), texts


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
