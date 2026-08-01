"""Regression tests for audit 2026-08-01 batch W1 — C1 image_preview.py:274 self._pixmap never assigned."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Boilerplate: headless Qt + src on sys.path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from rlpe.gui.image_preview import ImagePreviewWidget  # noqa: E402


def _make_test_png(tmp_path: Path, w: int = 100, h: int = 100) -> Path:
    """Create a small PNG on disk for ImagePreviewWidget.set_image()."""
    try:
        from PIL import Image

        img = Image.new("RGB", (w, h), color=(255, 0, 0))
        path = tmp_path / "test.png"
        img.save(path)
        return path
    except ImportError:
        # Fallback: write a minimal 1x1 PNG without PIL. We don't
        # depend on cv2 here either; cv2 is imported by the widget
        # itself and we only care that it returns *some* QPixmap so
        # we can check ``_pixmap is not None``. The widget's
        # _load_pixmap handles cv2/PIL fallback; if neither is
        # installed the test will be skipped via a separate path.
        import struct
        import zlib

        def _png_chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        width = max(w, 1)
        height = max(h, 1)
        # IHDR
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        # IDAT — single-row red pixel repeated
        raw = b""
        for _ in range(height):
            raw += b"\x00" + b"\xff\x00\x00" * width
        idat = zlib.compress(raw, 9)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", idat)
            + _png_chunk(b"IEND", b"")
        )
        path = tmp_path / "test.png"
        path.write_bytes(png)
        return path


class TestImagePreviewPixmap:
    """Regression tests for bug C1 — self._pixmap never assigned.

    Before the fix, ``_overlay_bboxes`` read ``self._pixmap`` at line
    274 to guard against off-image bboxes, but ``set_image()`` never
    stored the loaded pixmap on the instance, so the guard silently
    fell through to the ``else 0`` branch and the bounds-skip at
    line 281 never fired.
    """

    def test_set_image_assigns_self_pixmap(self, tmp_path: Path) -> None:
        """After set_image() the widget must keep a reference to the pixmap."""
        widget = ImagePreviewWidget()
        png = _make_test_png(tmp_path, w=100, h=100)
        widget.set_image(png)
        try:
            assert widget._pixmap is not None, (
                "audit 2026-08-01 C1 regression: self._pixmap must be assigned "
                "in set_image() so _overlay_bboxes can read its dimensions"
            )
        finally:
            widget.deleteLater()

    def test_set_bboxes_uses_correct_pixmap_dimensions(self, tmp_path: Path) -> None:
        """After set_image() + set_bboxes(), _pixmap dimensions must match.

        The bug previously caused the guard at line 274 to read 0/0,
        silently disabling the off-image bbox skip. Asserting the
        dimensions on ``widget._pixmap`` is enough to lock in the fix.
        """
        widget = ImagePreviewWidget()
        png = _make_test_png(tmp_path, w=100, h=100)
        widget.set_image(png)
        # Bbox list intentionally includes an off-image rect that, before
        # the fix, would have been silently drawn because img_w=img_h=0.
        widget.set_bboxes(
            [
                {"bbox": (10, 10, 20, 20), "species": "A"},
                {"bbox": (500, 500, 20, 20), "species": "off-image"},
            ]
        )
        try:
            assert widget._pixmap is not None
            assert widget._pixmap.width() == 100
            assert widget._pixmap.height() == 100
            # After the fix, the off-image bbox must have been skipped
            # so only one QGraphicsRectItem is present in the scene.
            scene_rects = [
                item
                for item in widget._scene.items()
                if item.__class__.__name__ == "QGraphicsRectItem"
            ]
            assert len(scene_rects) == 1, (
                "audit 2026-08-01 C1 regression: the off-image bbox should "
                "have been skipped via the (x >= img_w) guard but was drawn"
            )
        finally:
            widget.deleteLater()
