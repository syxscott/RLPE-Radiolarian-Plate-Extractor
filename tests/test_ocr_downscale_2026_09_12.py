"""2026-09-12: oversized plate scans made EasyOCR attempt 1.3 GB conv
tensors (RuntimeError / segfault), killing caption OCR for whole plates.
OCRBackend must downscale oversized inputs before inference and scale
token bboxes back to source coordinates."""

from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock

from rlpe.ocr import OCRBackend


def _backend(max_side: int) -> OCRBackend:
    return OCRBackend(backend="easyocr", ocr_max_side_px=max_side)


def test_downscales_and_restores_bbox():
    eng = _backend(1000)
    big = np.zeros((2000, 3000, 3), dtype=np.uint8)
    eng._engine = MagicMock()
    # Boxes arrive in DOWNSCALED coordinates (the engine only sees the
    # resized image): a 500px-wide box on the 1000px-wide render.
    eng._engine.readtext.return_value = [([[0, 0], [500, 0], [500, 40], [0, 40]], "hello", 0.9)]

    tokens = eng._ocr_array(big)

    assert len(tokens) == 1
    # 3000/1000 = 3× scale-back → source coordinates.
    assert tokens[0].bbox == (0, 0, 1500, 120)


def test_no_downscale_under_threshold():
    eng = _backend(4000)
    small = np.zeros((500, 800, 3), dtype=np.uint8)
    eng._engine = MagicMock()
    eng._engine.readtext.return_value = [([[0, 0], [100, 0], [100, 20], [0, 20]], "x", 0.9)]

    tokens = eng._ocr_array(small)

    assert tokens[0].bbox == (0, 0, 100, 20)


def test_zero_disables_downscale():
    eng = _backend(0)
    big = np.zeros((4000, 4000, 3), dtype=np.uint8)
    eng._engine = MagicMock()
    eng._engine.readtext.return_value = []

    # Must not raise even for a huge image when the knob is disabled.
    assert eng._ocr_array(big) == []
