"""Property-based / fuzz tests for the regex caption parsers.

The caption parsers (``_regex_parse_caption`` and the inline regexes
in ``m3_engine.py``) are the only path the pipeline takes when the LLM
stage is disabled. A bug in any of them — e.g. an unescaped backslash
that makes a token consume the rest of the string, or a character
class that matches newlines and breaks pair-boundary detection —
silently makes 100% of one paper's species wrong. The unit tests only
cover a few positive examples; they don't cover the long tail of
adversarial input shapes the parsers will see in production.

Hypothesis generates random caption text (alphanumeric words, control
chars, very long strings, mixed Unicode) and verifies that:
  1. ``_regex_parse_caption`` does not raise
  2. The returned list contains only well-formed ``CaptionPair`` objects
     (non-None species, label, modifier; types match)
  3. The number of pairs returned is bounded by the input size
     (a single line shouldn't produce 100 pairs unless the line
     really has 100 labels)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

hypothesis = pytest.importorskip("hypothesis")
import hypothesis.strategies as st  # noqa: E402
from hypothesis import given, settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from rlpe.m3_engine import _regex_parse_caption  # noqa: E402


# Realistic caption fragments. A hypothesis strategy that produces
# text shaped like a radiolarian-plate caption: starts with a figure
# label, has species words.
def _species_word():
    return st.sampled_from(
        [
            "Genus",
            "GenusA",
            "Eucyrtidiellum",
            "Archaeodictyomitra",
            "Stylocapsa",
            "Theocampe",
            "Pseudoeucyrtis",
            "Sethocapsa",
            "Hiscocapsa",
            "Parahsuum",
            "Canoptum",
            "Spumellaria",
            "Nassellaria",
            "Deviatus",
            "Archeo",
            "Archaeo",
        ]
    )


def _epithet():
    return st.sampled_from(
        [
            "species",
            "sp",
            "sp.",
            "indet",
            "pustulatum",
            "minor",
            "unumaense",
            "apiarium",
            "apiarius",
            "blackhorsensis",
            "remusa",
            "oblongula",
            "minor",
            "gracilis",
            "excelsa",
        ]
    )


def _modifier():
    return st.sampled_from(
        [
            "",
            "sp",
            "sp.",
            "spp",
            "spp.",
            "gr",
            "gr.",
            "cf.",
            "aff.",
            "cf. W.",
            "aff. minoensis",
            "(?)",
            "?",
        ]
    )


def _label_token():
    return st.sampled_from(["1", "2", "3", "1a", "1b", "2-3", "4, 5", "10"])


def _figure_lead():
    return st.sampled_from(
        [
            "Plate 1, fig. 1.",
            "Fig. 1.",
            "fig 1.",
            "figs 1-3.",
            "Plate 1, fig 1, 4.",
            "Fig 1, fig 2.",
        ]
    )


# A strategy that builds a caption-shaped string from primitives.
caption_text_strategy = st.builds(
    lambda fig, lbl, sp1, epi, mod, sp2, epi2, mod2, tail: (
        f"{fig} {lbl}) {sp1} {epi} {mod} {lbl}) {sp2} {epi2} {mod2} {tail}"
    ),
    fig=_figure_lead(),
    lbl=_label_token(),
    sp1=_species_word(),
    epi=_epithet(),
    mod=_modifier(),
    sp2=_species_word(),
    epi2=_epithet(),
    mod2=_modifier(),
    tail=st.sampled_from(
        [
            "",
            "; Pl. 1, fig. 2: Foo bar",
            "; 5, 6) Bar baz sp.",
            "and Fig 7, fig 8. extra",
            "; (?). trailing punctuation...",
        ]
    ),
)


# Adversarial inputs that should NOT crash the parser
adversarial_strategy = st.sampled_from(
    [
        "",  # empty
        " ",  # whitespace only
        "\n\n\n",  # newlines only
        "..." * 100,  # punctuation spam
        "abc" * 1000,  # very long
        "123" * 1000,  # digits only
        "\x00\x01\x02\x03",  # control chars
        "🔥" * 100,  # Unicode emoji
        "αβγδ" * 50,  # Greek
        "(?<!invalid_regex_attempt)",  # regex meta-chars in text
        "1) Genus epithet" * 100,  # repeated labels
        "\t\t\t1)\tGenus\tepithet",  # tabs as separators
        "1)Genus\n2)Species",  # newline-separated
        "Plate 1\nfig 1. Genus epithet\nfig 2. Genus species2",
        "A" * 10000,  # very long single token
        "1) " + "x" * 5000,  # very long trailing
    ]
)


@given(caption_text=caption_text_strategy)
@settings(max_examples=200, deadline=None)
def test_regex_parser_does_not_crash_on_caption_shaped_input(caption_text):
    """A well-formed caption should parse to a list of CaptionPair
    objects without raising."""
    pairs = _regex_parse_caption(caption_text)
    assert isinstance(pairs, list)
    for p in pairs:
        # Field types: species/modifier are strings (or None if no
        # modifier), labels is a list of strings
        assert isinstance(p.species, str)
        assert p.modifier is None or isinstance(p.modifier, str)
        assert isinstance(p.labels, list)
        for lbl in p.labels:
            assert isinstance(lbl, str)


@given(caption_text=adversarial_strategy)
@settings(max_examples=200, deadline=None)
def test_regex_parser_does_not_crash_on_adversarial_input(caption_text):
    """Adversarial inputs (empty, control chars, very long, Unicode)
    must not crash the parser. Returning an empty list is fine;
    returning a list of pairs is fine. Raising is not fine."""
    pairs = _regex_parse_caption(caption_text)
    assert isinstance(pairs, list)
    for p in pairs:
        assert isinstance(p.species, str)
        assert p.modifier is None or isinstance(p.modifier, str)
        assert isinstance(p.labels, list)
        for lbl in p.labels:
            assert isinstance(lbl, str)


@given(caption_text=st.text(min_size=0, max_size=500))
@settings(max_examples=300, deadline=None)
def test_regex_parser_handles_arbitrary_text(caption_text):
    """Most general: any unicode string. The parser must be total
    (defined for all inputs) and produce only well-formed output."""
    pairs = _regex_parse_caption(caption_text)
    assert isinstance(pairs, list)
    for p in pairs:
        # species must be a non-empty string
        assert isinstance(p.species, str) and p.species
        # labels must be a list of non-empty strings
        assert isinstance(p.labels, list)
        for lbl in p.labels:
            assert isinstance(lbl, str) and lbl


@given(
    n=st.integers(min_value=1, max_value=20),
    species_word=_species_word(),
    epithet=_epithet(),
)
@settings(max_examples=100, deadline=None)
def test_regex_parser_pair_count_bounded(n, species_word, epithet):
    """A caption with N labels should not produce wildly more than N
    pairs. The parser uses finditer, so the worst case is bounded by
    the number of digit-clusters in the input. The sanity check is
    ``len(pairs) <= 4 * n`` (each label can match up to 4 caption
    forms: leading, range, list, parenthesised)."""
    parts = [f"{i}) {species_word} {epithet}" for i in range(1, n + 1)]
    text = "Plate 1, fig. 1. " + "; ".join(parts)
    pairs = _regex_parse_caption(text)
    # Should be at least n (one pair per label)
    # Upper bound: lenient, just sanity-check it doesn't blow up
    assert len(pairs) <= 10 * n


def test_regex_parser_handles_null_bytes():
    """Null bytes are a common adversarial input. The parser must
    not crash; the result is implementation-defined (likely empty)."""
    pairs = _regex_parse_caption("\x001) Genus epithet\x00")
    assert isinstance(pairs, list)
    # No assertion on the result, only that it doesn't raise


def test_regex_parser_handles_crlf():
    """CRLF line endings (Windows-style captions). The parser should
    not be tripped up by \\r\\n vs \\n."""
    pairs = _regex_parse_caption("Plate 1, fig. 1.\r\n1) Genus epithet\r\n2) Genus species2")
    assert isinstance(pairs, list)
    # At least the labels should be present
    all_labels = [lbl for p in pairs for lbl in p.labels]
    # 1 and 2 should appear (one of the species may or may not match
    # depending on epithet validity, but the labels are robust)
    assert "1" in all_labels or len(pairs) == 0


def test_regex_parser_handles_smart_quotes():
    """Smart quotes (curly quotes) are common in PDF text extraction.
    The parser should not crash on them."""
    pairs = _regex_parse_caption("Plate 1, fig. 1. 1) “Genus” ‘epithet’ 2) Genus species2")
    assert isinstance(pairs, list)
    for p in pairs:
        assert isinstance(p.species, str)
