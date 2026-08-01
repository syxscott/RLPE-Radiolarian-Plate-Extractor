"""Regression tests for audit 2026-08-01 batch W2 — ocr M25/C12."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestOCRLangAliases:
    """Bug M25: PaddleOCR-native long spellings must be accepted by OCRBackend."""

    def test_japan_alias_accepted(self, caplog):
        from rlpe.ocr import OCRBackend

        with caplog.at_level(logging.WARNING, logger="rlpe.ocr"):
            be = OCRBackend(backend="paddleocr", lang="japan")
        # No "unknown OCR lang" warning should fire for an aliased form.
        assert not any("unknown OCR lang" in rec.message for rec in caplog.records), (
            f"unexpected warning: {[r.message for r in caplog.records]}"
        )
        # "japan" must be normalised to internal "ja".
        assert "ja" in be.lang, f"expected 'ja' in lang, got {be.lang!r}"
        assert "japan" not in be.lang

    def test_ch_alias_accepted(self, caplog):
        from rlpe.ocr import OCRBackend

        with caplog.at_level(logging.WARNING, logger="rlpe.ocr"):
            be = OCRBackend(backend="paddleocr", lang="ch")
        assert not any("unknown OCR lang" in rec.message for rec in caplog.records)
        assert "ch_sim" in be.lang, f"expected 'ch_sim' in lang, got {be.lang!r}"

    def test_chinese_cht_alias(self, caplog):
        from rlpe.ocr import OCRBackend

        with caplog.at_level(logging.WARNING, logger="rlpe.ocr"):
            be = OCRBackend(backend="paddleocr", lang="chinese_cht")
        assert not any("unknown OCR lang" in rec.message for rec in caplog.records)
        assert "ch_tra" in be.lang, f"expected 'ch_tra' in lang, got {be.lang!r}"

    def test_ja_unchanged(self, caplog):
        """Internal short codes must still work unchanged."""
        from rlpe.ocr import OCRBackend

        with caplog.at_level(logging.WARNING, logger="rlpe.ocr"):
            be = OCRBackend(backend="paddleocr", lang="ja")
        assert not any("unknown OCR lang" in rec.message for rec in caplog.records)
        assert "ja" in be.lang

    def test_unknown_lang_rejected(self, caplog):
        """A truly unknown lang must still be dropped with a warning."""
        from rlpe.ocr import OCRBackend

        with caplog.at_level(logging.WARNING, logger="rlpe.ocr"):
            be = OCRBackend(backend="paddleocr", lang="klingon")
        assert any(
            "unknown OCR lang" in rec.message and "klingon" in rec.message for rec in caplog.records
        ), "expected warning for klingon"
        # Falls back to ["en"] when nothing valid was supplied.
        assert be.lang == ["en"], f"expected fallback to ['en'], got {be.lang!r}"


class TestPaddlePolysInt:
    """Bug C12: _normalize_paddle_result must handle int entries in dt_polys."""

    def test_int_polys_handled(self):
        from rlpe.ocr import OCRBackend

        # PaddleOCR 3.x edge case: dt_polys contains a bare int
        # sentinel instead of a polygon. box[0] would raise TypeError
        # on the int path.
        result = {
            "rec_texts": ["x"],
            "rec_scores": [0.9],
            "dt_polys": [42],
        }
        out = OCRBackend._normalize_paddle_result(result)
        # Must not raise; gracefully skip the bad entry.
        assert isinstance(out, list)
        assert out == [], f"expected graceful empty list, got {out!r}"

    def test_normal_polys_unchanged(self):
        from rlpe.ocr import OCRBackend

        # Sanity: normal 4-point polygon entry must still round-trip.
        result = {
            "rec_texts": ["x"],
            "rec_scores": [0.9],
            "dt_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
        }
        out = OCRBackend._normalize_paddle_result(result)
        assert len(out) == 1
        box, text, conf = out[0]
        assert text == "x"
        assert conf == pytest.approx(0.9)
        assert box == [[0, 0], [1, 0], [1, 1], [0, 1]]
