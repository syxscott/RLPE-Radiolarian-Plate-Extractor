"""Tests for Phase 60 Plan 3 — Bug 3.12: lexicon match uses naive
substring search, so the lexicon entry ``can`` (a hypothetical genus)
would match inside ``canned``, ``scan``, ``cancels``, ``canvas``, etc.

The previous ``_lexicon_predict`` used ``lower.find(name.lower())``
which is a substring match with no word-boundary check.

Phase 60 Plan 3 fix: use ``re.search(rf\"\\b{re.escape(word)}\\b\",
text)`` so the lexicon entry must start/end at a word boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.taxon import TaxonRecognizer  # noqa: E402


def _reco_with_lexicon(lexicon: list[str]) -> TaxonRecognizer:
    """Build a TaxonRecognizer with an in-memory lexicon (no disk I/O)."""
    reco = TaxonRecognizer()
    reco._lexicon = set(lexicon)
    return reco


def test_lexicon_substring_does_not_match():
    """``can`` must NOT match inside ``canned``."""
    reco = _reco_with_lexicon(["can"])
    out = reco._lexicon_predict("the canned samples were collected")
    assert out == [], f"substring match leaked: {[(e.text, e.start) for e in out]}"


def test_lexicon_substring_does_not_match_scan():
    """``can`` must NOT match inside ``scan``."""
    reco = _reco_with_lexicon(["can"])
    out = reco._lexicon_predict("we scan the slides for radiolarians")
    assert out == [], f"substring match leaked: {[(e.text, e.start) for e in out]}"


def test_lexicon_substring_does_not_match_canvas():
    """``can`` must NOT match inside ``canvas``."""
    reco = _reco_with_lexicon(["can"])
    out = reco._lexicon_predict("a canvas of diverse species")
    assert out == [], f"substring match leaked: {[(e.text, e.start) for e in out]}"


def test_lexicon_whole_word_still_matches():
    """Regression guard: ``can`` as a standalone word still matches."""
    reco = _reco_with_lexicon(["can"])
    out = reco._lexicon_predict("we can identify the species")
    assert any(e.text == "can" for e in out), (
        f"standalone 'can' missed: {[(e.text, e.start) for e in out]}"
    )


def test_lexicon_multibyte_entry():
    """Lexicon entries with hyphens are still matched (hyphen is
    a non-word char so ``\\b`` fires on either side)."""
    reco = _reco_with_lexicon(["Actinomma-holmesi"])
    out = reco._lexicon_predict("figs 1-2. Actinomma-holmesi")
    assert any(e.text == "Actinomma-holmesi" for e in out), (
        f"hyphenated entry missed: {[(e.text, e.start) for e in out]}"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
