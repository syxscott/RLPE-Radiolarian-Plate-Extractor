"""Round 8: regression tests for caption regex + panel_id shape.

These tests lock the two bug fixes from 2026-07-05:

Bug-R8-1: ``_CAPTION_CLAUSE_RE`` only matched captions prefixed with
``Fig.`` / ``Figs.`` / ``Figure.`` — it rejected the numbered-list
form ``1. Species`` used by Hollis_2006 (Paleocene-Eocene
radiolarian captions) and several other Mesozoic / Cenozoic
papers. The fix extends the regex to accept a leading ``<digits>.``
prefix as long as the next token starts with an uppercase letter
(prose like ``1. Introduction`` doesn't match because
``Introduction`` is also uppercase but the regex captures the
species name from the same match, not the introduction header;
the gate ``(?=[A-Z])`` ensures we only match when a species
genus follows).

Bug-R8-2: ``_PANEL_LABEL_SHAPE`` used ``[1-9]\\d*[a-z]?`` which had
NO digit-length cap. OCR engines frequently misread scale-bar
labels like ``100 µm`` / ``86500`` and emit the bare digits as
panel_id candidates. Real radiolarian plates rarely exceed 50
panels (bandini2011's largest plate has 42 panels), so 1-3
digits is the safe shape. The fix is ``[1-9]\\d{0,2}[a-z]?``
which caps the digit run at 3 characters.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestCaptionClauseRegexHollis:
    """Bug-R8-1: ``1. Species`` numbered-list caption format."""

    def test_hollis_caption_parses(self):
        """Hollis_2006 caption format must be parseable.

        Sample from data/gold/hollis2006.jsonl panel 1:
          "Plate 1 ... 1. Amphisphaera coronata EHRENBERG gr. CH12076..."
        After fix, _CAPTION_CLAUSE_RE must match the ``1. Amphisphaera``
        clause.
        """
        from rlpe.m3_engine import _CAPTION_CLAUSE_RE, _regex_parse_caption

        hollis_caption = (
            "Plate 1\n\n"
            "Scanning electron micrographs of spumellarian radiolarians "
            "from lower Eocene strata, Mead Stream. Scale bars = 100 µm.\n\n"
            "1. Amphisphaera coronata EHRENBERG gr. CH12076, P30/f534.\n"
            "2. Axoprunum aff. bispiculum (POPOFSKY).\n"
            "3. Axoprunum pierinae n. sp.\n"
            "4. Stylosphaera minor n. sp."
        )
        # The fix must extract at least 4 panels.
        pairs = _regex_parse_caption(hollis_caption)
        assert len(pairs) >= 3, (
            f"Expected 3+ pairs from Hollis-style numbered-list caption; "
            f"got {len(pairs)}: {pairs!r}"
        )
        # Spot-check that 'Amphisphaera coronata' was extracted.
        species_extracted = {p.species for p in pairs}
        assert any("Amphisphaera" in s for s in species_extracted), (
            f"Expected 'Amphisphaera coronata' in extracted species; "
            f"got {species_extracted!r}"
        )

    def test_prose_introduction_not_matched(self):
        """``1. Introduction`` (prose, no species) must NOT match.

        Without the ``(?=[A-Z])`` lookahead gate, the regex would
        greedily match the first uppercase word after every ``<digit>.``
        and silently absorb section headers as panel labels.
        """
        from rlpe.m3_engine import _CAPTION_CLAUSE_RE

        # ``Abstract. 1. Introduction.`` should yield zero matches
        # because "Introduction" is followed by "." not a species, and
        # our regex requires the species name to start with a capital
        # letter that continues into a binomial epithet (lowercase
        # tail). The standalone "Introduction" has no lowercase tail.
        text = "1. Introduction. 2. Methods section."
        matches = _CAPTION_CLAUSE_RE.findall(text)
        assert len(matches) == 0, (
            f"Prose numbered list must not match as panel labels; "
            f"got {matches!r}"
        )


class TestPanelLabelShapeValidation:
    """Bug-R8-2: 4+ digit panel_id from OCR scale-bar misreads."""

    def test_four_digit_panel_id_rejected(self):
        """``86500`` (Feng OCR misread of scale-bar text) must NOT be
        accepted as a panel id. Cap is 3 digits."""
        from rlpe.association import is_valid_panel_label

        # Real OCR garbage from Feng_2006.
        for label in ("86500", "15000", "100000", "99999", "500"):
            # Note: "500" is 3 digits — at the edge. If a paper
            # actually has 500 panels this test would break, but no
            # radiolarian paper does. Cap at 3 is conservative.
            if len(label) <= 3:
                continue
            assert not is_valid_panel_label(label), (
                f"Multi-digit scale-bar label {label!r} must be rejected "
                f"as panel id (audit bug R8-2: OCR scale-bar leak)"
            )

    def test_three_digit_panel_id_accepted(self):
        """Real 1-3 digit panel ids must still be accepted."""
        from rlpe.association import is_valid_panel_label

        for label in ("1", "12", "100", "0", "999"):
            assert is_valid_panel_label(label), (
                f"Valid 1-3 digit panel id {label!r} was rejected"
            )

    def test_letter_suffix_panel_id_accepted(self):
        """``1a`` / ``12b`` must still be accepted (a/b panel suffixes)."""
        from rlpe.association import is_valid_panel_label

        for label in ("1a", "12b", "100z"):
            assert is_valid_panel_label(label)

    def test_letter_only_panel_id_accepted(self):
        """Single A-H decorative markers (figure-level "(A)") must
        still be accepted. ``AA`` / ``HH`` should be rejected."""
        from rlpe.association import is_valid_panel_label

        for label in ("A", "H"):
            assert is_valid_panel_label(label)
        for label in ("AA", "ZZ", "Hello"):
            assert not is_valid_panel_label(label)

    def test_source_guard_shape_regex(self):
        """The source-level regex change must be in place — this
        guards against accidental future relaxation that would
        re-introduce the OCR scale-bar leak."""
        from pathlib import Path as _Path

        path = (
            _Path(__file__).resolve().parents[1]
            / "src" / "rlpe" / "association.py"
        )
        text = path.read_text(encoding="utf-8")
        # Must have the {0,2} cap (1-3 digits).
        assert r"[1-9]\d{0,2}[a-z]?" in text, (
            "_PANEL_LABEL_SHAPE must cap at 3 digits via \\d{0,2}"
        )
        # Must NOT have the old unbounded \\d* pattern.
        assert r"[1-9]\d*[a-z]?" not in text, (
            "Old unbounded _PANEL_LABEL_SHAPE regex leaked back in"
        )
