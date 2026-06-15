"""Tests for extended scale-bar regex (bare, range, en-dash, Greek mu)."""
from __future__ import annotations

from rlpe.scale_bar import (
    ScaleInfo,
    extract_scale_from_caption,
    extract_scale_from_ocr_text,
    merge_scale_info,
    normalize_unit,
    to_um,
)


class TestNormalizeUnit:
    def test_um_variants(self):
        assert normalize_unit("μm") == "um"
        assert normalize_unit("µm") == "um"
        assert normalize_unit("um") == "um"

    def test_micron_words(self):
        assert normalize_unit("micron") == "um"
        assert normalize_unit("microns") == "um"
        assert normalize_unit("Micron") == "um"

    def test_mm(self):
        assert normalize_unit("mm") == "mm"

    def test_unknown(self):
        assert normalize_unit("parsec") == "parsec"

    def test_empty(self):
        assert normalize_unit("") == ""


class TestToUm:
    def test_um(self):
        assert to_um(100, "um") == 100

    def test_mm_to_um(self):
        assert to_um(1, "mm") == 1000

    def test_cm_to_um(self):
        assert to_um(1, "cm") == 10000

    def test_nm_to_um(self):
        assert to_um(1000, "nm") == 1

    def test_unknown_unit(self):
        assert to_um(1, "parsec") is None


class TestExtractScaleFromCaption:
    def test_bare_value(self):
        info = extract_scale_from_caption("100 μm")
        assert info.value == 100.0
        assert info.unit == "um"
        assert info.confidence > 0

    def test_bare_value_no_space(self):
        info = extract_scale_from_caption("100μm")
        assert info.value == 100.0

    def test_with_scale_bar_label(self):
        info = extract_scale_from_caption("Scale bar = 50 μm")
        assert info.value == 50.0

    def test_range_with_dash(self):
        info = extract_scale_from_caption("scale bar 5-10 μm")
        # Range form: midpoint is stored in value
        assert info.value == 7.5

    def test_range_with_en_dash(self):
        info = extract_scale_from_caption("scale bar 5–10 μm")
        assert info.value == 7.5

    def test_greek_mu(self):
        info = extract_scale_from_caption("100 µm")
        assert info.unit == "um"

    def test_microns_word(self):
        info = extract_scale_from_caption("100 microns")
        assert info.unit == "um"

    def test_no_match(self):
        info = extract_scale_from_caption("not a scale bar at all")
        assert info.value is None
        assert info.confidence == 0.0

    def test_empty_string(self):
        info = extract_scale_from_caption("")
        assert info.value is None


class TestExtractScaleFromOcr:
    def test_basic(self):
        info = extract_scale_from_ocr_text("50 um")
        assert info.value == 50.0
        assert info.source == "ocr"

    def test_no_match(self):
        info = extract_scale_from_ocr_text("nothing here")
        assert info.value is None


class TestMergeScaleInfo:
    def test_prefers_higher_confidence(self):
        caption = ScaleInfo(value=100.0, unit="um", confidence=0.9, source="caption")
        ocr = ScaleInfo(value=50.0, unit="um", confidence=0.5, source="ocr")
        merged = merge_scale_info(caption, ocr)
        assert merged.value == 100.0
        assert merged.source == "caption"

    def test_falls_back_to_other(self):
        caption = ScaleInfo(confidence=0.0)
        ocr = ScaleInfo(value=75.0, unit="um", confidence=0.6, source="ocr")
        merged = merge_scale_info(caption, ocr)
        assert merged.value == 75.0

    def test_computes_um_per_px(self):
        caption = ScaleInfo(value=100.0, unit="um", confidence=0.8, source="caption")
        merged = merge_scale_info(caption, ScaleInfo(), pixel_length=200.0)
        assert merged.um_per_px == 0.5
