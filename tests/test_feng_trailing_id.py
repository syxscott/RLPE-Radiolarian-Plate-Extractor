from __future__ import annotations

r"""Regression tests for feng2007-style trailing-identifier parsing.

feng2007 ("Upper Permian radiolarians from the Dongpan section") uses
a "sp. N" trailing-identifier convention to distinguish multiple
"sp." specimens in the same genus (e.g. ``Entactinia sp. 1``,
``Triaenosphaera sp. 4``). The identifier is a single digit (or
letter) after ``sp.`` and a whitespace, e.g. ``figs 7-8. Entactinia
sp. 1`` and ``fig. 12. Triaenosphaera sp. 5``.

The original ``_CAPTION_CLAUSE_RE`` captured only the modifier
(`` sp.``) and dropped the identifier, so the parser emitted
``Entactinia sp.`` — the gold form is ``Entactinia sp. 1``. The fix
adds a fourth capturing group for the trailing identifier and folds
it into the species string in ``_regex_parse_caption``.

The Genus-uncertainty form (``Trilonche? sp. 1``) is also covered.
Previously the ``?`` was tied to a dedicated epithet branch that
fired before the modifier group, so the identifier never reached the
trailing-ID group. The fix moves the ``?`` into the genus token
(``(?:\?)?``) so the modifier + trailing-ID path is reached normally.
"""

from rlpe.m3_engine import _regex_parse_caption


def test_entactinia_sp_1():
    """The headline case: 'figs 7-8. Entactinia sp. 1' must capture
    'Entactinia sp. 1' (with the trailing ' 1'), not bare 'Entactinia sp.'."""
    pairs = _regex_parse_caption("figs 7-8. Entactinia sp. 1")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.labels == ["7", "8"], p.labels
    assert p.species == "Entactinia sp. 1", p.species
    assert p.modifier == "", (
        f"modifier should be cleared once trailing_id is folded in, got {p.modifier!r}"
    )


def test_triaenosphaera_sp_4():
    """Multi-digit panel label + trailing ID: 'fig. 10. Triaenosphaera
    sp. 4' must capture 'Triaenosphaera sp. 4'."""
    pairs = _regex_parse_caption("fig. 10. Triaenosphaera sp. 4")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.labels == ["10"], p.labels
    assert p.species == "Triaenosphaera sp. 4", p.species


def test_praewilliriedellum_sp_letter_id():
    """The trailing identifier can be a letter, not just a digit:
    'Praewilliriedellum sp. A' must capture 'Praewilliriedellum sp. A'."""
    pairs = _regex_parse_caption("Figs 1-3. Praewilliriedellum sp. A")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.labels == ["1", "2", "3"], p.labels
    assert p.species == "Praewilliriedellum sp. A", p.species


def test_trilonche_question_sp_1():
    """Genus-uncertainty form: 'Trilonche? sp. 1' must capture
    'Trilonche? sp. 1' (with the '?'), not 'Trilonche' or 'Trilonche? sp.'.
    The '?' has to be folded into the genus token so the modifier +
    trailing-ID path is reached."""
    pairs = _regex_parse_caption("figs 7-8. Trilonche? sp. 1")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.labels == ["7", "8"], p.labels
    assert p.species == "Trilonche? sp. 1", p.species


def test_trilonche_question_sp_2():
    """Sequential specimen: 'Trilonche? sp. 2' must capture
    'Trilonche? sp. 2'."""
    pairs = _regex_parse_caption("figs 13-16. Trilonche? sp. 2")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.labels == ["13", "14", "15", "16"], p.labels
    assert p.species == "Trilonche? sp. 2", p.species


def test_no_trailing_id_keeps_modifier():
    """Without a trailing identifier the modifier must remain in
    its own field (e.g. 'Praewilliriedellum sp.' → species='Praewilliriedellum',
    modifier=' sp.'), so the existing gold-format match still works."""
    pairs = _regex_parse_caption("Fig. 6. Praewilliriedellum sp.")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.labels == ["6"], p.labels
    assert p.species == "Praewilliriedellum", p.species
    assert p.modifier == "sp.", p.modifier


def test_trinomial_still_works():
    """Trinomial names ('Lamptonium fabaeforme fabaeforme') must not
    be confused with the trailing-identifier pattern. The third
    epithet branch uses the same word-boundary lookahead, but
    'fabaeforme' is not a modifier keyword, so it still matches."""
    pairs = _regex_parse_caption("Fig. 6. Lamptonium fabaeforme fabaeforme")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.species == "Lamptonium fabaeforme fabaeforme", p.species
    assert p.modifier == "", p.modifier
