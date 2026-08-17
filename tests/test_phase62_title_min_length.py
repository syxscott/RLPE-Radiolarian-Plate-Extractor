"""Phase 62 Plan 5 (Bug 5.13): apply _MIN_TITLE_LEN to short
alphanumeric titles.

``_MIN_TITLE_LEN = 8`` is defined at the top of
``paper_metadata_cleanup.py`` but was never applied to the title
itself. A title like ``"A GIS map"`` (8 chars) is fine — but
``"1234567"`` (7 digits, no letters, looks like a page number) or
``"a1b2c3"`` (6 alphanumerics, no real word) should also be
flagged as a parse artifact.

The existing ``_TITLE_GARBAGE_PATTERNS`` already catches page
ranges, filenames, and pure digits. The remaining gap is titles
that are 8+ characters but consist entirely of short
alphanumeric gibberish (no real words). The fix: as a final
filter, check the title has at least one run of 3+ consecutive
alphabetic characters — otherwise flag it as garbage.

This catches the cases like:
  - ``"1234567a"`` (digits + 1 letter, no real word)
  - ``"a1b2c3"`` (alternating, no word)
  - ``"x y z w"`` (single letters separated by spaces)

while passing real titles:
  - ``"Late Triassic radiolarians"`` (consecutive letters)
  - ``"GIS-based analysis"`` (GIS is 3 consecutive letters)
  - ``"A 2-D map of Italy"`` (multiple consecutive letters)
"""

from __future__ import annotations

from rlpe.paper_metadata_cleanup import (
    _MIN_TITLE_LEN,
    cleanup_title,
    looks_like_garbage_title,
)


def test_short_alphanumeric_title_rejected():
    """'1234567' (7 digits, no letters) must be rejected — too short."""
    title, reason = cleanup_title("1234567")
    assert reason == "title_extraction_failed"
    assert title is None


def test_alphanumeric_gibberish_rejected():
    """'a1b2c3d' (alternating single letters + digits) must be
    rejected as a parse artifact — no real word."""
    title, reason = cleanup_title("a1b2c3d")
    assert reason == "title_extraction_failed", (
        f"gibberish title not rejected; got title={title!r} reason={reason!r}"
    )


def test_short_title_real_word_kept():
    """'A GIS map' (real short title with real words) must be kept."""
    title, reason = cleanup_title("A GIS map")
    assert title == "A GIS map"
    assert reason is None


def test_long_title_kept():
    """Regression: 'Late Triassic radiolarians from Sicily' is kept."""
    t = "Late Triassic radiolarians from Sicily"
    title, reason = cleanup_title(t)
    assert title == t
    assert reason is None


def test_short_real_word_at_threshold():
    """'GIS data' (8 chars, contains 'GIS' which has 3 consecutive
    letters) is right at the threshold and must be kept."""
    title, reason = cleanup_title("GIS data")
    assert title == "GIS data"
    assert reason is None


def test_short_real_word_below_threshold():
    """'ab1' (3 chars, alternating, no 3-letter run) is below
    _MIN_TITLE_LEN and has no real word — must be rejected."""
    title, reason = cleanup_title("ab1")
    assert reason == "title_extraction_failed"


def test_min_title_len_constant_is_8():
    """Lock down the _MIN_TITLE_LEN constant value."""
    assert _MIN_TITLE_LEN == 8


def test_looks_like_garbage_title_helper():
    """The helper must classify short alphanumerics as garbage."""
    assert looks_like_garbage_title("1234567")
    assert looks_like_garbage_title("a1b2c3d")
    # Real titles are not garbage.
    assert not looks_like_garbage_title("Late Triassic radiolarians")
    assert not looks_like_garbage_title("GIS data")
