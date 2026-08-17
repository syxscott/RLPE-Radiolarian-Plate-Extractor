"""Scale bar length detection — HoughLinesP ndim-guard regression.

Round 25 live integration surfaced a crash on OpenCV >=5 where
``cv2.HoughLinesP`` returns a 2-D array ``(N, 4)`` instead of the
historical 3-D ``(1, N, 4)``. The pipeline called
``detect_scale_bar_length_px(region_img)`` inside the per-region
match path; when the OCR backend fell back to EasyOCR (also missing
in this env) and ``HoughLinesP`` returned a 2-D array, the
``lines[:, 0, :]`` index raised ``IndexError: too many indices for
array`` and the entire paper produced zero rows.

The fix in ``scale_bar.py`` accepts both shapes. These tests verify
the function returns a non-zero length for the synthetic happy path
and ``None`` for the empty / malformed path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _draw_horizontal_line(img: np.ndarray, length: int = 100) -> np.ndarray:
    """Draw a thick horizontal black line on a white image."""
    out = img.copy()
    h, w = out.shape[:2]
    cy = h // 2
    x1 = max(0, (w - length) // 2)
    x2 = min(w - 1, x1 + length)
    cv2 = __import__("cv2")
    cv2.line(out, (x1, cy), (x2, cy), (0, 0, 0), 3)
    return out


def test_detect_scale_bar_length_px_returns_length_for_clear_line():
    import cv2

    from rlpe.scale_bar import detect_scale_bar_length_px

    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    img = _draw_horizontal_line(img, length=120)
    px_len = detect_scale_bar_length_px(img)
    assert px_len is not None and px_len > 50, f"Expected a long horizontal line; got {px_len!r}"


def test_detect_scale_bar_length_px_handles_grayscale_input():
    from rlpe.scale_bar import detect_scale_bar_length_px

    img_gray = np.ones((100, 200), dtype=np.uint8) * 255
    img_gray = _draw_horizontal_line(img_gray, length=80)
    px_len = detect_scale_bar_length_px(img_gray)
    assert px_len is not None and px_len > 30, f"Grayscale input not handled; got {px_len!r}"


def test_detect_scale_bar_length_px_returns_none_on_blank():
    from rlpe.scale_bar import detect_scale_bar_length_px

    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    px_len = detect_scale_bar_length_px(img)
    # No edges -> HoughLinesP returns None -> our function returns None.
    assert px_len is None, f"blank image should yield None; got {px_len!r}"


def test_detect_scale_bar_length_px_handles_none_input():
    from rlpe.scale_bar import detect_scale_bar_length_px

    assert detect_scale_bar_length_px(None) is None
