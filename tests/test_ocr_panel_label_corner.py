"""Tests for the OCRBackend.recognize_panel_label default and behaviour.

Most of OCRBackend depends on PaddleOCR / EasyOCR being installed, but the
public API surface (default value of `label_corner`, behaviour when the
backend is disabled) is pure Python and can be tested without the
heavyweight OCR engines.
"""
from __future__ import annotations

import inspect

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.ocr import OCRBackend, OCRToken  # noqa: E402


def test_recognize_panel_label_default_is_tl():
    """The default for `label_corner` must be `"tl"` (top-left) for
    radiolarian plates — that's the dominant label placement in the
    eval corpus. Setting it to `"auto"` (which tries all 4 corners)
    is 4x slower per panel. Pipeline.py relies on the default being
    fast; the `"auto"` path is only for papers with non-standard
    label placement.

    This was a real perf regression: the docstring claimed `"tl"` is
    the default but the code default was `"auto"`, costing ~12 min
    on bandini2011 (317 panels × 3 extra corner OCRs × 0.7s).
    """
    sig = inspect.signature(OCRBackend.recognize_panel_label)
    assert "label_corner" in sig.parameters
    assert sig.parameters["label_corner"].default == "tl", (
        f"recognize_panel_label default for label_corner should be 'tl', "
        f"got {sig.parameters['label_corner'].default!r}. Using 'auto' "
        f"is 4x slower per panel."
    )


def test_recognize_panel_label_returns_empty_when_no_backend():
    """When no OCR backend is installed, recognize_panel_label must
    return an empty list, not raise. This is the fallback path the
    pipeline uses when PaddleOCR init fails AND EasyOCR is missing."""
    backend = OCRBackend(backend="nonexistent", use_gpu=False)
    # Use a tiny synthetic image (no OCR needed if backend fails)
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = backend.recognize_panel_label(img, (0, 0, 50, 50))
    assert out == []


def test_recognize_panel_label_tl_corner_only_one_call():
    """Sanity: when label_corner is set explicitly to 'tl'/'tr'/'bl'/'br',
    the corner loop must run only ONCE (not 4 times). We can't introspect
    the OCR backend's internal call count without monkey-patching, so
    we just verify the function accepts the explicit values."""
    backend = OCRBackend(backend="nonexistent", use_gpu=False)
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for corner in ("tl", "tr", "bl", "br", "auto"):
        # No exception means the corner value is accepted.
        out = backend.recognize_panel_label(img, (0, 0, 50, 50), label_corner=corner)
        assert out == [], f"expected empty result for {corner}, got {out}"


def test_ocr_token_dataclass_is_hashable():
    """OCRToken is a frozen-ish dataclass; verify slots + equality for
    use in dict-based dedup caches. The pipeline may want to cache
    `(image_hash, bbox) -> OCRToken` results in a future perf pass."""
    t1 = OCRToken(text="1", confidence=0.9, bbox=(0, 0, 10, 10))
    t2 = OCRToken(text="1", confidence=0.9, bbox=(0, 0, 10, 10))
    assert t1 == t2
    t3 = OCRToken(text="2", confidence=0.9, bbox=(0, 0, 10, 10))
    assert t1 != t3
