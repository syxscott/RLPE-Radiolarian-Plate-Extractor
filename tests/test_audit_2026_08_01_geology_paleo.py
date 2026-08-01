"""Regression tests for audit 2026-08-01 batch W4 — geology_extraction M4/M5.

Covers two related fixes in the paleo-coordinate classification path:

  * Bug M4 — ``_classify_coordinate_age`` defaulted to ``"modern"``
    when neither paleo nor modern keyword appeared in the 120-char
    prefix window. The 120-char window was too short when the age
    lives in the plate header (often 200+ chars before the
    coordinate). Wrong default → paleo-reconstruction silently
    seeds off modern coordinates → meaningless paleo maps.

    Fix:
      1. Widen the context window from 120 to 400 chars.
      2. Line-level scan (split on ``\\n``) so header lines like
         "Tertiary" or "Pleistocene" are searchable regardless of
         position within the window.
      3. Return ``None`` (was ``"modern"``) when neither keyword
         appears — caller decides what to do with the ambiguous
         record instead of silently stamping it as modern.

  * Bug M5 — ``_PALEO_KEYWORDS`` (and the mirror copy
    ``_PALEO_KEYWORDS_GEO`` in geo_coords.py) used bare substring
    ``kw in ctx`` without word boundaries. Captures like
    "paleogeneously" or "subpaleogene" falsely matched "paleogene";

    "concurrently" falsely matched "currently". The phrase
    "sandstone-dominated Neogene succession" (where "Neogene" is
    200 chars before the coordinate) DID match — but only by luck
    because the bare substring wasn't tripped by a longer word in
    that capture.

    Fix: pre-compile a word-boundary regex
    ``\\b(?:" + "|".join(map(re.escape, keywords)) + r")\\b`` and
    use ``.search()`` instead of ``any(kw in ctx ...)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.geo_coords import _is_paleo_text  # noqa: E402
from rlpe.geology_extraction import (  # noqa: E402
    _MODERN_KEYWORDS,
    _PALEO_KEYWORDS,
    _classify_coordinate_age,
)

# A dummy coordinate inserted at the end of every test caption so
# we can call ``_classify_coordinate_age(text, coord_start,
# coord_end)`` deterministically. The actual number is irrelevant —
# the function only looks at the prefix BEFORE the match.
_COORD = "35.7N, 14.3E"
_COORD_START = 500  # set per-test (see helper below)
_COORD_END = _COORD_START + len(_COORD)


def _ctx(prefix: str) -> str:
    """Return a string of length exactly ``_COORD_START`` followed by
    the dummy coordinate. The function looks at ``text[match_start -
    400 : match_start]`` so the prefix must be at least 400 chars
    long (or the test is really about a shorter window — see
    individual tests).
    """
    assert len(prefix) <= _COORD_START
    return prefix + " " * (_COORD_START - len(prefix)) + _COORD


class TestPaleoCoordinateClassification:
    """Bug M4 — window widened, line scan added, None on ambiguous."""

    def test_modern_default_when_no_keyword(self):
        """Old behaviour: no keyword in 120-char window → ``"modern"``
        (BAD default; would seed paleo-reconstruction with modern
        coords).

        New behaviour: no keyword in 400-char window → ``None`` so
        the caller leaves both modern_ and paleo_ latitude unset.
        """
        # 400 chars of innocuous text — no paleo / modern keyword.
        # Use deliberately chosen filler ('g', 'h', 'j', 'k') that
        # avoids accidental matches: e.g. "during the 1980s" would
        # trigger the "during the " paleo keyword even with a
        # window fix. Pure letter runs avoid that footgun.
        prefix = "g" * 400
        text = _ctx(prefix)
        out = _classify_coordinate_age(text, _COORD_START, _COORD_END)
        assert out is None, (
            f"Expected None when no keyword in 400-char window, "
            f"got {out!r}. Old behaviour returned 'modern' which "
            f"would silently seed paleo-reconstruction with modern "
            f"coordinates."
        )

    def test_paleo_keyword_at_300_chars_detected(self):
        """Caption has 'Eocene' 300 chars before the coordinate →
        ``is_paleo`` True. With the OLD 120-char window this would
        have been missed (Eocene was outside the window) and the
        default 'modern' would have been returned.

        Window math: the coord starts at position 500 in the text;
        the 400-char window covers positions 100-500. Placing
        'Eocene' at position 200 puts it 300 chars before the
        coord — well inside the 400-char window.
        """
        # 194 chars of filler, then ' Eocene ' (8 chars: spaces
        # matter! The word-boundary regex requires Eocene to be
        # bordered by non-word chars; glued to 'g' it would not
        # match), then 99 filler, total 300 chars. 'Eocene' starts
        # at position 195 — well inside the 400-char window.
        prefix = "g" * 194 + " Eocene " + "g" * 98
        assert len(prefix) == 300
        text = _ctx(prefix)
        out = _classify_coordinate_age(text, _COORD_START, _COORD_END)
        assert out == "paleo", (
            f"Expected 'paleo' (Eocene 300 chars before coord), "
            f"got {out!r}. The 120-char window was too short for "
            f"plate-header age labels."
        )

    def test_modern_keyword_within_window(self):
        """Standard case: 'today' immediately before the coordinate →
        ``is_paleo`` == 'modern' (still works)."""
        prefix = (
            "Today the locality is at the following position, "
            "with fresh outcrops visible along the road " * 5
        )
        prefix = prefix[:400]
        text = _ctx(prefix)
        out = _classify_coordinate_age(text, _COORD_START, _COORD_END)
        assert out == "modern", f"Expected 'modern' (today keyword in window), got {out!r}"


class TestPaleoKeywordsWordBoundary:
    """Bug M5 — word-boundary regex in _PALEO_KEYWORDS / _is_paleo_text."""

    # The window used by geo_coords._is_paleo_text is 120 chars.
    # geo_coords parses the FIRST coordinate in the text, so we use
    # a small text where the coord is at the end and the keyword
    # (or fake keyword) is inside the 120-char prefix.
    def _build(self, prefix_text: str):
        coord = "35.7N, 14.3E"
        return prefix_text + " " + coord

    def test_paleogeneously_not_matched(self):
        """'paleogeneously terrigenous' must NOT match. Old bare
        substring found 'paleogene' inside 'paleogeneously' → false
        paleo=True. Word boundary rejects it.
        """
        text = self._build("paleogeneously terrigenous succession")
        is_paleo = _is_paleo_text(text, text.index("35.7N"))
        assert is_paleo is False, (
            "paleogeneously must NOT match 'paleogene' — the bare "
            "substring used to falsely match, now rejected by \\b"
        )

    def test_paleogene_alone_matched(self):
        """'Paleogene' alone → matched."""
        text = self._build("Deposited during the Paleogene")
        is_paleo = _is_paleo_text(text, text.index("35.7N"))
        assert is_paleo is True, "Standalone 'Paleogene' must still match the paleo keyword"

    def test_neogene_in_long_paragraph_matched(self):
        """'sandstone-dominated Neogene succession' 200 chars before
        the coord → still matched. The keyword remains searchable
        even at the edge of the 120-char window for geo_coords, and
        inside the 400-char window for _classify_coordinate_age.
        """
        # geo_coords uses 120 chars; pad to keep 'Neogene' inside.
        prefix = (
            "sandstone-dominated Neogene succession with thin "
            "limestone interbeds and conglomerate lenses"
        )
        # 120 chars total — 'Neogene' sits at position 20-26.
        prefix = prefix[:120]
        assert "Neogene" in prefix
        text = self._build(prefix)
        is_paleo = _is_paleo_text(text, text.index("35.7N"))
        assert is_paleo is True, "Neogene in a longer paragraph must still match the paleo keyword"

    def test_substring_word_boundary(self):
        """'subpaleogene' must NOT match. The 'p' of 'paleogene' is
        glued to 'sub' (no word boundary), so ``\\b`` rejects it.
        """
        text = self._build("subpaleogene lithofacies")
        is_paleo = _is_paleo_text(text, text.index("35.7N"))
        assert is_paleo is False, (
            "subpaleogene must NOT match — 'paleogene' is glued to 'sub' (no word boundary)"
        )


# ---------------------------------------------------------------------------
# Source-guards: confirm the bug fix is actually in the source so a
# future regression that reverts to bare substring / 120-char
# window / "modern" default is caught immediately.
# ---------------------------------------------------------------------------


class TestSourceGuard:
    """Source-guard checks: the fix must be visible in the source."""

    def test_widened_window_400(self):
        src = (_SRC / "rlpe" / "geology_extraction.py").read_text()
        assert "match_start - 400" in src, (
            "geology_extraction._classify_coordinate_age must use "
            "a 400-char window (audit 2026-08-01 Bug M4)"
        )

    def test_returns_none_on_ambiguous(self):
        src = (_SRC / "rlpe" / "geology_extraction.py").read_text()
        assert "return None" in src, (
            "geology_extraction._classify_coordinate_age must "
            "return None on ambiguous (no keyword) — the old "
            "behaviour returned 'modern'"
        )

    def test_word_boundary_regex_present(self):
        src = (_SRC / "rlpe" / "geology_extraction.py").read_text()
        assert "_PALEO_KEYWORDS_RE" in src, (
            "geology_extraction.py must use a pre-compiled "
            "_PALEO_KEYWORDS_RE word-boundary regex (Bug M5)"
        )
        src2 = (_SRC / "rlpe" / "geo_coords.py").read_text()
        assert "_PALEO_KEYWORDS_GEO_RE" in src2, (
            "geo_coords.py must use the matching word-boundary regex (Bug M5)"
        )

    def test_no_bare_substring_kw_in_ctx(self):
        """After the fix no ``kw in ctx`` / ``kw in line`` /
        ``any(kw in ...)`` pattern should remain in the two target
        functions.
        """
        import re

        for path, fname in (
            (_SRC / "rlpe" / "geology_extraction.py", "geology_extraction"),
            (_SRC / "rlpe" / "geo_coords.py", "geo_coords"),
        ):
            src = path.read_text()
            # find references to old patterns (line-by-line scan)
            for i, line in enumerate(src.splitlines(), 1):
                # Allow comments, but not real code.
                if "any(kw in" in line or "if kw in" in line:
                    # Skip if it's a comment (after a #).
                    code = line.split("#", 1)[0]
                    if "any(kw in" in code or "if kw in" in code:
                        pytest.fail(
                            f"{fname}:{i}: bare-substring pattern "
                            f"remains ({line.strip()!r}). Use the "
                            f"pre-compiled _PALEO_KEYWORDS_RE / "
                            f"_PALEO_KEYWORDS_GEO_RE instead."
                        )

    def test_keyword_lists_unchanged(self):
        """The keyword list itself must not be modified — only the
        matching algorithm. (Catches accidental renames.)"""
        for needle in ("paleogene", "neogene", "eocene", "pleistocene"):
            assert needle in _PALEO_KEYWORDS, f"_PALEO_KEYWORDS missing {needle!r}"
        for needle in ("today", "present-day"):
            assert needle in _MODERN_KEYWORDS, f"_MODERN_KEYWORDS missing {needle!r}"
