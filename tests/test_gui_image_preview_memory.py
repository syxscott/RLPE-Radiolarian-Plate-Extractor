"""Phase F-1 (audit 2026-08-20) B-4/M-20 — image_preview memory guards.

Covers:
* ``_load_pixmap`` downsamples oversized SEM scans to MAX_PREVIEW_LONG_EDGE
* normal-size images pass through untouched
* ``clear()`` releases the QPixmap / current path
* ``clear()`` removes bbox + label overlay items from the scene
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Boilerplate: headless Qt + src on sys.path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

pytest.importorskip("PySide6")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from rlpe.gui.image_preview import (  # noqa: E402
    MAX_IMAGE_PIXELS,
    MAX_PREVIEW_LONG_EDGE,
    ImagePreviewWidget,
)


def _write_png(tmp_path: Path, w: int, h: int, name: str = "img.png") -> Path:
    """Write a solid-colour RGB PNG of the requested size."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 2] = 200  # BGR blue-ish so the channel swap is exercised
    path = tmp_path / name
    assert cv2.imwrite(str(path), arr)
    return path


def test_constants_are_sane() -> None:
    assert MAX_IMAGE_PIXELS == 50_000_000
    assert MAX_PREVIEW_LONG_EDGE == 4096


def test_load_pixmap_downsamples_oversized(tmp_path: Path) -> None:
    """8000x8000 (64 MP) exceeds MAX_IMAGE_PIXELS → must be downsampled."""
    assert 8000 * 8000 > MAX_IMAGE_PIXELS
    path = _write_png(tmp_path, 8000, 8000, "big.png")
    w = ImagePreviewWidget()
    pix = w._load_pixmap(path)
    assert pix is not None
    assert not pix.isNull()
    assert max(pix.width(), pix.height()) <= MAX_PREVIEW_LONG_EDGE
    # Aspect ratio preserved (square in, square out)
    assert pix.width() == pix.height()


def test_load_pixmap_normal_size_unchanged(tmp_path: Path) -> None:
    path = _write_png(tmp_path, 800, 600, "small.png")
    w = ImagePreviewWidget()
    pix = w._load_pixmap(path)
    assert pix is not None
    assert (pix.width(), pix.height()) == (800, 600)


def test_clear_releases_pixmap(tmp_path: Path) -> None:
    path = _write_png(tmp_path, 800, 600)
    w = ImagePreviewWidget()
    w.set_image(path)
    assert w._pixmap is not None
    assert w._current_path is not None
    w.clear()
    assert w._pixmap is None
    assert w._current_path is None


def test_clear_resets_overlay_state(tmp_path: Path) -> None:
    path = _write_png(tmp_path, 800, 600)
    w = ImagePreviewWidget()
    w.set_image(path)
    w.set_bboxes(
        [
            {"bbox": (10, 10, 100, 100), "species": "Foo bar", "confidence": 0.9},
            {"bbox": (200, 200, 50, 50), "species": "Baz qux"},
        ]
    )
    assert w._bbox_items and w._text_items
    w.clear()
    assert w._bbox_items == []
    assert w._text_items == []
    assert w._bboxes == []
    assert w._scene.items() == []


def test_clear_resets_drag_state(tmp_path: Path) -> None:
    """clear() mid-drag must not leave the view stuck in pan mode."""
    path = _write_png(tmp_path, 800, 600)
    w = ImagePreviewWidget()
    w.set_image(path)
    w._view._drag_mode = True
    w.clear()
    assert w._view._drag_mode is False
    assert w._view._drag_start is None


def test_clear_then_overlay_is_noop(tmp_path: Path) -> None:
    """After clear(), _overlay_bboxes must not resurrect anything."""
    path = _write_png(tmp_path, 800, 600)
    w = ImagePreviewWidget()
    w.set_image(path)
    w.set_bboxes([{"bbox": (10, 10, 100, 100)}])
    w.clear()
    w._overlay_bboxes()
    assert w._scene.items() == []
