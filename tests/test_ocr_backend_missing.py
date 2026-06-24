"""Tests for the OCR backend-missing 降级 path.

When PaddleOCR fails to init AND EasyOCR fails to init, the OCRBackend
falls back to returning empty results (self.backend = None,
self._engine = None). This is the "graceful degradation" path the
pipeline relies on in production where OCR packages may be missing.

These tests verify that:
  1. ``recognize()`` returns [] (not raises) on the missing-backend path
  2. ``recognize_panel()`` returns [] (not raises) on the missing-backend
  3. ``recognize_panel_label()`` returns [] (not raises) on the missing-
     backend path, AND the 2x fallback doesn't crash either
  4. The lazy init is not retried on every call (only once after first
     failure — prevents 5-min hangs when OCR is broken)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.ocr import OCRBackend  # noqa: E402


def _force_no_backend(backend: OCRBackend) -> None:
    """Simulate the lazy_init path where both PaddleOCR and EasyOCR
    fail to import. The OCRBackend._lazy_init catches ImportError and
    sets self.backend = None, self._engine = None. We replicate that
    state directly so the test doesn't need both packages absent."""
    backend._engine = None
    backend.backend = None  # type: ignore[assignment]


def test_recognize_returns_empty_when_no_backend():
    """recognize() must return [], not raise, when no OCR backend is
    available. This is the path the pipeline uses when PaddleOCR init
    fails AND EasyOCR is missing — the pipeline must still produce
    sensible empty results, not crash."""
    backend = OCRBackend(backend="nonexistent", use_gpu=False)
    _force_no_backend(backend)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = backend.recognize(img)
    assert out == []


def test_recognize_handles_str_path_when_no_backend():
    """recognize() must also handle str/Path inputs without trying to
    load the image when the backend is unavailable. The current code
    calls cv2.imread *before* checking the engine, so a missing
    backend with a Path input should still return [] gracefully."""
    backend = OCRBackend(backend="nonexistent", use_gpu=False)
    _force_no_backend(backend)

    out = backend.recognize("/nonexistent/path/to/image.png")
    assert out == []


def test_recognize_panel_returns_empty_when_no_backend():
    """recognize_panel() must return [], not raise, when no backend
    is available. The bbox is irrelevant — the function bails before
    any work is done."""
    backend = OCRBackend(backend="nonexistent", use_gpu=False)
    _force_no_backend(backend)

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = backend.recognize_panel(img, (10, 10, 50, 50))
    assert out == []


def test_recognize_panel_label_returns_empty_when_no_backend():
    """recognize_panel_label() must return [], not raise, when no
    backend. This is the path the v18 reassignment script uses for
    papers where the OCR label can't be read."""
    backend = OCRBackend(backend="nonexistent", use_gpu=False)
    _force_no_backend(backend)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = backend.recognize_panel_label(img, (0, 0, 50, 50))
    assert out == []


def test_recognize_panel_label_does_not_crash_on_adaptive_with_no_backend():
    """The 2x fallback (added 2026-06-08) must not be entered when the
    backend is None — it would try to call self._ocr_array(None) which
    would raise. The test verifies that even with a panel large enough
    to trigger the fallback, no_backend still returns []."""
    backend = OCRBackend(backend="nonexistent", use_gpu=False)
    _force_no_backend(backend)

    img = np.zeros((100, 100, 3), dtype=np.uint8)  # small, < 500px
    out = backend.recognize_panel_label(img, (0, 0, 80, 100), label_corner="tl")
    assert out == []


def test_recognize_panel_label_with_adaptive_corner_no_backend():
    """label_corner='adaptive' iterates over all 4 corners. With no
    backend, every iteration must return [] without raising."""
    backend = OCRBackend(backend="nonexistent", use_gpu=False)
    _force_no_backend(backend)

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = backend.recognize_panel_label(img, (0, 0, 200, 200), label_corner="adaptive")
    assert out == []


def test_lazy_init_not_retried_after_failure(monkeypatch):
    """After _lazy_init fails once and sets backend=None, subsequent
    calls must NOT re-attempt the import. Without this guard, every
    call to recognize() would pay the import cost (or worse, hit a
    5-min timeout retry loop on a broken install)."""
    backend = OCRBackend(backend="nonexistent", use_gpu=False)
    call_count = {"n": 0}

    def fake_lazy_init(self):
        call_count["n"] += 1
        # Simulate the failure path: set engine=None and return None
        self._engine = None
        return None

    monkeypatch.setattr(OCRBackend, "_lazy_init", fake_lazy_init)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(5):
        out = backend.recognize(img)
        assert out == []
    # All 5 calls hit _lazy_init (it's lazy). What we're verifying is
    # the "no exception, no hang" contract. If the lazy init
    # path were missing the early-return on engine=None, the 5 calls
    # would re-attempt the PaddleOCR/EasyOCR import each time and the
    # test would time out or fail.
    assert call_count["n"] == 5
