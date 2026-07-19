"""Tests for Phase 60 Plan 3 — Bug 3.9: ``find_ages_in_text`` lacks
``\\b`` word boundaries and matches age names inside longer words.

``Cambrianian`` (a hypothetical mineral / journal name) must NOT match
``Cambrian``. ``Permianian`` must NOT match ``Permian``. The previous
implementation used a substring match with the comment "we add a
small negative lookahead" but no such lookahead was actually
present in the regex.

Phase 60 Plan 3 fix: add a negative lookahead for word / Chinese
characters after the matched name so the match must end at a
non-alphanumeric boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.stratigraphy import find_ages_in_text  # noqa: E402


def test_find_ages_word_boundary():
    """Substring-within-word must NOT match."""
    # Cambrian must NOT match inside Cambrianian.
    out = find_ages_in_text("the Cambrianian formation of uncertain age")
    assert all(c.period != "Cambrian" for c in out), (
        f"Cambrian matched inside 'Cambrianian': "
        f"{[(c.period, c.age) for c in out]}"
    )
    # Permian must NOT match inside Permianian.
    out = find_ages_in_text("Permianian strata")
    assert all(c.period != "Permian" for c in out), (
        f"Permian matched inside 'Permianian': "
        f"{[(c.period, c.age) for c in out]}"
    )
    # Jurassic must NOT match inside Jurassicpark (hypothetical).
    out = find_ages_in_text("Jurassicpark fossils")
    assert all(c.period != "Jurassic" for c in out), (
        f"Jurassic matched inside 'Jurassicpark': "
        f"{[(c.period, c.age) for c in out]}"
    )


def test_find_ages_word_boundary_real_words_still_match():
    """Regression guard: real-world mentions with adjacent word
    characters must still match."""
    out = find_ages_in_text("Late Cambrian and Early Ordovician")
    periods = {c.period for c in out}
    assert "Cambrian" in periods, periods
    assert "Ordovician" in periods, periods


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])