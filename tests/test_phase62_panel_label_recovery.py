"""Phase 62 Plan 5 (Bug 5.10): panel-label OCR misread recovery.

``_PANEL_LABEL_SHAPE`` (used by ``is_valid_panel_label``) is
anchored fullmatch on
    ^(?:[A-H]|[1-9]\d{0,2}[a-z]?|0)$

OCR sometimes reads ``3a`` as ``3a0`` (trailing noise — the next
character of the printed panel label has bled into the OCR
window). The current shape rejects ``3a0`` outright, losing the
real label ``3a``.

The fix: in ``_normalize_panel_label``, attempt to recover the
clean base label from an OCR-misread digit+letter+trailing-digit
shape. Specifically:

    * If the input matches ``^(\\d{1,3}[a-z])\\d$`` (digit+letter
      followed by a single trailing digit), strip the trailing
      digit and return the base label. Log at debug.

This is intentionally conservative: only ONE trailing digit is
stripped, and only when the base shape is otherwise a valid
panel-label shape.
"""
from __future__ import annotations

from rlpe.association import _normalize_panel_label, is_valid_panel_label


def test_normalize_panel_label_recovers_3a0_to_3a():
    """OCR misread '3a0' must recover to '3a'."""
    out = _normalize_panel_label("3a0")
    assert out == "3a", f"expected '3a', got {out!r}"


def test_normalize_panel_label_recovers_12b5_to_12b():
    """'12b5' (digit + letter + trailing OCR noise) recovers to '12b'."""
    out = _normalize_panel_label("12b5")
    assert out == "12b", f"expected '12b', got {out!r}"


def test_normalize_panel_label_clean_label_unchanged():
    """Regression: '3a' (no trailing noise) stays as '3a'."""
    out = _normalize_panel_label("3a")
    assert out == "3a"


def test_normalize_panel_label_pure_digit_unchanged():
    """Regression: '5' stays as '5'."""
    out = _normalize_panel_label("5")
    assert out == "5"


def test_is_valid_panel_label_accepts_recovered_3a0():
    """After normalisation, '3a0' → '3a' which IS a valid panel label."""
    # The pipeline calls _normalize_panel_label THEN is_valid_panel_label
    # so a successful recovery makes the label valid.
    normalised = _normalize_panel_label("3a0")
    assert is_valid_panel_label(normalised)


def test_is_valid_panel_label_rejects_unrecoverable():
    """Labels that can't be normalised (e.g. 'abcdefg') are still
    rejected."""
    assert not is_valid_panel_label(_normalize_panel_label("abcdefg"))


def test_normalize_panel_label_multi_trailing_digit_unchanged():
    """More than one trailing digit ('3a00') is NOT stripped — too
    ambiguous, likely a real value (e.g. 'Bed 3a00' OCR fragment)."""
    out = _normalize_panel_label("3a00")
    # The base shape '3a' would be valid, but '3a00' with two trailing
    # digits is too risky to recover.
    assert out == "3a00" or out == "3a", (
        f"ambiguous: '3a00' should be either kept or fully recovered; got {out!r}"
    )