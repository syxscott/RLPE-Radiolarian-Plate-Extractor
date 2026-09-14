"""2026-09-14 — unicode-safe cv2 imread/imwrite regression tests.

Root cause of the "Volcano Ark" 12-second failures: ``cv2.imread`` /
``cv2.imwrite`` on Windows open files with the ANSI locale codec, so a
non-ASCII path component (the comparison harness named run directories
after the profile, e.g. ``…\\火山方舟_run1\\od_output\\…\\imageFile1.png``)
silently returned None — every plate image became "unreadable", the
figure loop skipped all 20 figures, and the paper collapsed to a
0-row stub. These tests pin the unicode-safe wrappers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.preprocess import imread_unicode, imwrite_unicode  # noqa: E402


@pytest.fixture()
def unicode_image(tmp_path: Path) -> tuple[Path, np.ndarray]:
    img = (np.random.rand(32, 48, 3) * 255).astype("uint8")
    d = tmp_path / "火山方舟测试目录"
    d.mkdir()
    return d / "图版01.png", img


class TestUnicodeSafeIO:
    def test_cv2_imread_fails_on_unicode_path(self, unicode_image):
        """Documents the platform bug the wrappers exist for (Windows)."""
        p, img = unicode_image
        ok, buf = cv2.imencode(".png", img)
        assert ok
        buf.tofile(str(p))
        if sys.platform == "win32":
            assert cv2.imread(str(p)) is None, (
                "cv2.imread unexpectedly handled a unicode path — the "
                "platform bug this module guards against is fixed upstream"
            )

    def test_imwrite_then_imread_roundtrip(self, unicode_image):
        p, img = unicode_image
        assert imwrite_unicode(p, img) is True
        back = imread_unicode(p)
        assert back is not None
        assert back.shape == img.shape
        assert (back == img).all()

    def test_imread_missing_file_returns_none(self, tmp_path):
        assert imread_unicode(tmp_path / "不存在.png") is None

    def test_imwrite_bad_dir_returns_false(self, tmp_path):
        img = np.zeros((4, 4, 3), dtype="uint8")
        assert imwrite_unicode(tmp_path / "no_such_dir" / "x.png", img) is False

    def test_imwrite_params_forwarded(self, unicode_image):
        p, img = unicode_image
        assert imwrite_unicode(p, img, [cv2.IMWRITE_PNG_COMPRESSION, 0]) is True
        assert imread_unicode(p) is not None

    def test_imread_flags_forwarded(self, unicode_image):
        p, img = unicode_image
        assert imwrite_unicode(p, img)
        unchanged = imread_unicode(p, cv2.IMREAD_UNCHANGED)
        assert unchanged is not None and unchanged.shape == img.shape
