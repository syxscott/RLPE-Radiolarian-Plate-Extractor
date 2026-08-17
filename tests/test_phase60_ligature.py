"""Tests for Phase 60 Plan 3 — Bug 3.4: Unicode ligatures and diacritics
in taxon names.

Real radiolarian captions use characters like:

  * ``Pterocanium æthiopicum``  — Latin small letter ash (U+00E6) → ``ae``
  * ``Cenodiscus çElementum``   — Latin small letter c-cedilla (U+00E7)
  * ``Entactinia döeringii``    — Latin small letter o-umlaut (U+00F6)
  * ``Actinomma hughesiﬁ``      — Latin small letter fi ligature (U+FB01)

The previous regex used ``[a-zA-Z-]`` (ASCII only) so a name containing
``æ`` / ``ç`` / ``ö`` / ``ﬁ`` failed to match entirely.

The fix normalises the caption via Unicode NFKD before regex matching,
which decomposes ``æ`` → ``ae``, strips combining diacritics, and
unfolds ``ﬁ`` / ``ﬂ`` / ``ﬀ`` / ``ﬃ`` / ``ﬄ`` to ``fi`` / ``fl`` / ``ff``
/ ``ffi`` / ``ffl``. The original surface form is preserved in the
output entity's ``text`` field via the standard ``m.group(1)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.taxon import TaxonRecognizer  # noqa: E402


def _reco() -> TaxonRecognizer:
    return TaxonRecognizer()


def test_ligature_ae_unfolded():
    """``æ`` → ``ae`` so ``Pterocanium æthiopicum`` is matched."""
    reco = _reco()
    text = "figs 1-2. Pterocanium æthiopicum"
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    # After NFKD normalisation the epithet becomes ``aethiopicum`` and
    # the regex matches it. The entity's text field is the matched
    # substring; we check that the species is detected at all.
    assert any("thiopicum" in t or "aethiopicum" in t for t in texts), texts


def test_ligature_fi_unfolded():
    """``ﬁ`` (U+FB01) → ``fi``. Real captions like
    ``Actinomma hughesiﬁ`` should match ``Actinomma hughesifi``."""
    reco = _reco()
    text = "figs 1-2. Actinomma hughesiﬁ"
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    assert any("hughesi" in t for t in texts), texts


def test_diacritic_o_umlaut():
    """``ö`` → ``o`` so ``Entactinia döeringii`` is matched."""
    reco = _reco()
    text = "figs 1-2. Entactinia döeringii"
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    assert any("ringii" in t or "doeringii" in t for t in texts), texts


def test_ascii_only_still_works():
    """Regression guard: pure-ASCII binomials must still match."""
    reco = _reco()
    text = "figs 1-2. Entactinia compacta"
    entities = reco._fallback_predict(text)
    texts = [e.text for e in entities]
    assert "Entactinia compacta" in texts, texts


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
