"""Tests for panel-label validators and OCR-noise rejection.

These tests pin down the contract of ``is_valid_panel_label`` and
``_normalize_panel_label``: which strings are panel-label-shaped vs.
OCR noise. PaddleOCR / EasyOCR commonly misread small numerals as
random letters (``3`` → ``ean``, ``2a`` → ``P1``); a permissive
validator that lets those through pollutes the figure's label space
and collides with real labels via positional fallback.

The shape rules (locked here):

  ACCEPT  single uppercase A–H        (figure-level decorative markers)
  ACCEPT  digit with optional trailing [a-z]    (1, 2a, 12b)
  REJECT  anything else (ean / foo / L / P1 / ,1 / 251.90 / long strings)

Length cap remains 16 chars as a backstop against caption fragments.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from rlpe.association import _normalize_panel_label, is_valid_panel_label  # noqa: E402


class TestIsValidPanelLabelAccepts:
    """Strings that ARE valid panel labels must pass."""

    @pytest.mark.parametrize("label", ["1", "2", "12", "99"])
    def test_pure_digits(self, label):
        assert is_valid_panel_label(label) is True

    @pytest.mark.parametrize("label", ["1a", "2b", "12c", "99z"])
    def test_digit_with_trailing_lowercase(self, label):
        assert is_valid_panel_label(label) is True

    @pytest.mark.parametrize("label", ["A", "B", "C", "D", "E", "F", "G", "H"])
    def test_single_uppercase_marker(self, label):
        assert is_valid_panel_label(label) is True


class TestIsValidPanelLabelRejects:
    """Strings that are OCR noise / caption fragments must be rejected."""

    @pytest.mark.parametrize(
        "label",
        [
            "ean",  # PaddleOCR misread of '3'
            "L",  # OCR misread of '1' / '4' / '7'
            "P1",  # OCR misread of '21'
            "foo",  # arbitrary 3-char alpha
            ",1",  # comma leftover from punctuation
            "251.90",  # Ma-range fragment
            "Figure",  # caption keyword
            "plate",  # caption keyword
            "Plate1",  # compound word
            "I",  # uppercase outside the A-H marker range
            "J",
            "Z",
            "ab",  # lowercase alpha
            "AA",  # multi-letter
        ],
    )
    def test_ocr_noise_rejected(self, label):
        assert is_valid_panel_label(label) is False, (
            f"{label!r} must be rejected as OCR noise / caption fragment"
        )

    @pytest.mark.parametrize("label", ["", " ", "  "])
    def test_empty_or_whitespace_only(self, label):
        # Empty / whitespace stripped to empty must be rejected.
        assert is_valid_panel_label(label.strip()) is False

    def test_none_rejected(self):
        assert is_valid_panel_label(None) is False

    def test_non_string_rejected(self):
        assert is_valid_panel_label(123) is False
        assert is_valid_panel_label(["1"]) is False

    def test_overly_long_rejected(self):
        # 17-char label passes the regex check but fails length cap.
        assert is_valid_panel_label("12345678901234567") is False


class TestNormalizePanelLabelRoundtrip:
    """normalize then validate: well-formed input survives, noise doesn't."""

    def test_normalized_digit_is_valid(self):
        n = _normalize_panel_label("04")
        assert n == "4"
        assert is_valid_panel_label(n) is True

    def test_normalized_alpha_passes_validation_only_if_in_a_h(self):
        # "A" is a valid marker, normalize keeps it as-is.
        n = _normalize_panel_label("A")
        assert n == "A"
        assert is_valid_panel_label(n) is True

    def test_normalized_garbage_alpha_stays_invalid(self):
        # "ean" stays "ean" through normalize but must NOT be valid.
        n = _normalize_panel_label("ean")
        assert n == "ean"
        assert is_valid_panel_label(n) is False
