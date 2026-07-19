"""Tests for Phase 60 Plan 3 — Bug 3.3: ``Genus sp.``, ``Genus n. sp.``,
``Genus sp. nov.``, ``Genus nom. nov.``, ``Genus comb. nov.``.

The previous regex in :mod:`rlpe.taxon` only matched the canonical
``Genus + lowercase`` shape; isolated-genus forms like ``Genus sp.``
or ``Genus n. sp.`` were silently dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.taxon import TaxonRecognizer  # noqa: E402


def _reco() -> TaxonRecognizer:
    return TaxonRecognizer()


def test_extract_genus_sp():
    """``Genus sp.`` (open nomenclature) must be extracted."""
    reco = _reco()
    text = "figs 1-2. Entactinia sp."
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    assert any("Entactinia sp." in t for t in texts), texts


def test_extract_genus_spp():
    """``Genus spp.`` (plural open nomenclature) must be extracted."""
    reco = _reco()
    text = "figs 1-2. Entactinia spp."
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    assert any("Entactinia spp." in t for t in texts), texts


def test_extract_genus_n_sp():
    """``Genus n. sp.`` (new species, open nomenclature) must be
    extracted."""
    reco = _reco()
    text = "figs 1-2. Entactinia n. sp."
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    assert any("Entactinia n. sp." in t or "Entactinia n.sp." in t for t in texts), texts


def test_extract_genus_sp_nov():
    """``Genus sp. nov.`` must be extracted."""
    reco = _reco()
    text = "figs 1-2. Entactinia sp. nov."
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    assert any("Entactinia sp. nov." in t for t in texts), texts


def test_extract_full_binomial_still_works():
    """Regression guard: the canonical Genus + epithet shape must
    still match."""
    reco = _reco()
    text = "figs 1-2. Entactinia compacta"
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    assert "Entactinia compacta" in texts, texts


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])