"""Phase 61 Plan 4 (Bug 4.4): OCRToken carries a token_type discriminator.

Previously OCRToken had text/conf/bbox/metadata — downstream
``match_panels`` had no way to tell a panel-label token ("1a", "Fig. 3")
from a species token ("Genus species"), so the matcher treated them
identically.

The fix adds ``OCRToken.token_type: Literal["label", "species", "other"]``
defaulting to "other" (backward-compatible), and:
  - ``recognize_panel_label()`` stamps ``token_type="label"`` on every
    returned token.
  - ``extract_species_tokens()`` (new helper) recognises taxa-shaped
    strings and stamps ``token_type="species"`` on those.
"""
from __future__ import annotations

import pytest

from rlpe.ocr import OCRToken, OCRBackend, extract_species_tokens


def test_ocr_token_has_type_field():
    """OCRToken exposes a ``token_type`` attribute (default 'other')."""
    t = OCRToken(text="hello", confidence=0.9, bbox=(0, 0, 10, 10))
    assert t.token_type == "other"
    # explicit constructor override is allowed
    t2 = OCRToken(text="x", confidence=0.5, bbox=(0, 0, 1, 1), token_type="label")
    assert t2.token_type == "label"


def test_recognize_panel_label_stamps_label_type():
    """recognize_panel_label() must stamp token_type='label' on output."""
    # We don't want to spin up PaddleOCR / EasyOCR — patch the
    # internal _ocr_array to return one synthetic token per corner band.
    backend = OCRBackend(backend="easyocr")
    # Stub the internal helper so we don't need a real OCR engine.
    def _fake_ocr_array(img):
        return [OCRToken(text="3", confidence=0.9, bbox=(0, 0, 4, 4))]
    backend._ocr_array = _fake_ocr_array  # type: ignore[assignment]
    # Recognise on a small synthetic image, panel bbox anywhere.
    import numpy as np
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    tokens = backend.recognize_panel_label(img, bbox=(0, 0, 100, 100), label_corner="tl")
    assert tokens, "recognize_panel_label returned no tokens"
    for t in tokens:
        assert t.token_type == "label", f"got {t.token_type}"


def test_extract_species_tokens_returns_only_species():
    """extract_species_tokens() stamps species tokens and rejects short
    text that cannot be a binomial."""
    tokens = [
        OCRToken(text="3", confidence=0.9, bbox=(0, 0, 4, 4)),  # label, short
        OCRToken(text="Genus species", confidence=0.9, bbox=(0, 0, 30, 10)),
        OCRToken(text="Genus cf. species", confidence=0.9, bbox=(0, 0, 30, 10)),
        OCRToken(text="Genus sp.", confidence=0.9, bbox=(0, 0, 30, 10)),
        OCRToken(text="random caption fragment", confidence=0.9, bbox=(0, 0, 30, 10)),
    ]
    out = extract_species_tokens(tokens)
    species_texts = {t.text for t in out if t.token_type == "species"}
    other_texts = {t.text for t in out if t.token_type == "other"}
    # The taxon-shaped inputs should be stamped 'species'.
    assert "Genus species" in species_texts
    assert "Genus sp." in species_texts
    # Non-taxon text passes through unchanged with token_type='other'.
    assert "3" in other_texts
    assert "random caption fragment" in other_texts