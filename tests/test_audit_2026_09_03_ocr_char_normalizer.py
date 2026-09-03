"""Audit 2026-09-03 (BLOCKER-#6): OCR character-level normaliser tests.

Verifies the new ``_normalize_ocr_chars`` pre-pass in
``rlpe.ocr_corrections`` handles the most common radiolarian OCR
character confusions:

  * ``1`` between two letters → ``l``     (Sponguru1 → Spongurul)
  * Capital ``I`` between lowercase letters → ``l`` (TheocorIs → Theocorls)
  * Long-vowel marks (ā, ē, ī, ō, ū) → base vowel (Archaeodictyomitrā → Archaeodictyomitra)

The look-around guards MUST prevent false positives on:
  * "iuncus" (leading i is at a word edge, not word-medial)
  * "ITA" (capital I at a word start or after another capital)
  * "Fig 1" (trailing 1 is at a word edge)
  * "1991" (year tokens; only digits, no letter context)

The C5 substring lock (2-entry ``CORRECTIONS`` table) is preserved
— see ``tests/test_audit_2026_08_01_ocr_corrections_lock.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _norm(s: str) -> str:
    """Apply the pre-pass directly so tests pin its behavior even
    when the public apply_corrections() entry point's paper-specific
    whitelist semantics shift."""
    from rlpe.ocr_corrections import _normalize_ocr_chars
    return _normalize_ocr_chars(s)


class TestDigitOneToLowerL:
    """``1`` between two letters → ``l``."""

    def test_sponguru1_word_medial(self):
        # BLOCKER-#6 canonical example from the audit.
        assert _norm("Sponguru1") == "Spongurul"

    def test_dactylioceras1_digit_l(self):
        # Dactylioceras is a key Toarcian ammonite genus used as
        # the biostratigraphic anchor in early-Jurassic radiolarian
        # papers; OCR frequently reads the final "s" as "1".
        assert _norm("Dactylioceras1") == "Dactyliocerasl"

    def test_trailing_one_at_word_edge_unchanged(self):
        # "Fig 1" — trailing 1 at a word edge must NOT be touched.
        assert _norm("Fig 1") == "Fig 1"

    def test_leading_one_at_word_edge_unchanged(self):
        # "1st" — leading digit at a word start.
        assert _norm("1st") == "1st"

    def test_year_token_unchanged(self):
        # "1991" — pure digit run, no letter context.
        assert _norm("1991") == "1991"

    def test_only_digits_unchanged(self):
        assert _norm("12345") == "12345"


class TestCapitalIToLowerL:
    """Capital ``I`` between two lowercase letters → ``l``."""

    def test_theocoris(self):
        # Theocorys is a key Cretaceous nassellarian — OCR commonly
        # misreads the medial "I" as a lowercase l (or vice versa).
        assert _norm("TheocorIs") == "Theocorls"

    def test_internal_i_between_lowercase(self):
        assert _norm("aIbus") == "albus"

    def test_word_initial_capital_unchanged(self):
        # Leading capital is legitimate Latin: "Internus" not "lnternus".
        assert _norm("Internus") == "Internus"

    def test_acronym_unchanged(self):
        # All-caps acronyms keep their I.
        assert _norm("ITA") == "ITA"

    def test_i_after_capital_unchanged(self):
        # "IBT" — second I is between two capitals; the rule requires
        # the lookbehind to be lowercase.
        assert _norm("IBT") == "IBT"


class TestLongVowelStripping:
    """ā/ē/ī/ō/ū → a/e/i/o/u."""

    def test_archaeodictyomitra_macrone(self):
        assert _norm("Archaeodictyomitrā") == "Archaeodictyomitra"

    def test_mesozoicum_o_macrone(self):
        # ̄o → o
        assert _norm("Mesōzoicum") == "Mesozoicum"

    def test_pachyncus_y_macrone(self):
        # ȳ → y
        assert _norm("Pachȳncus") == "Pachyncus"

    def test_u_macrone_to_u(self):
        # ū → u. Use a simple test word.
        assert _norm("ū") == "u"
        assert _norm("Hū") == "Hu"

    def test_etoile_e_macrone(self):
        assert _norm("Ētoile") == "Etoile"

    def test_nfc_decomposed_to_precomposed(self):
        # "a" + combining-macron → "ā" (NFC precomposed form), then
        # the long-vowel table maps to base vowel.
        import unicodedata
        a_tilde = "a" + "̄"  # decomposed form
        assert unicodedata.normalize("NFC", a_tilde) == "ā"
        assert _norm(a_tilde) == "a"


class TestComposeUnchanged:
    """The pre-pass MUST be a no-op on clean strings."""

    def test_no_op_clean_string(self):
        assert _norm("Theocorys") == "Theocorys"

    def test_no_op_canonical_species(self):
        # Hollis 2006 / De Wever 2001 reference species.
        for s in (
            "Archaeodictyomitra vulgaris",
            "Parvicingula jamesi",
            "Pseudodictyomitra pseudomacrocephala",
            "Cryptocapsa texta",
            "Haliomma gr. b",
        ):
            assert _norm(s) == s, f"Clean string modified: {s!r}"

    def test_no_op_punctuation(self):
        # Periods, parentheses, brackets — character-level rules
        # should never fire on punctuation.
        assert _norm("sp. (cf. X)") == "sp. (cf. X)"


class TestPrepassEndToEndViaApplyCorrections:
    """Verify the public ``apply_corrections`` entry point actually
    invokes the pre-pass. Indirectly tests the call site at the
    top of apply_corrections()."""

    def test_apply_corrections_digit_one(self):
        from rlpe.ocr_corrections import apply_corrections
        # Without the pre-pass, "Sponguru1" would round-trip unchanged.
        # The pre-pass normalises "1" to "l" before the C5 substring
        # table even sees it.
        out = apply_corrections("Sponguru1", paper_id=None)
        assert "Spongurul" in out, (
            f"apply_corrections did not normalise '1' to 'l': {out!r}"
        )

    def test_apply_corrections_long_vowel(self):
        from rlpe.ocr_corrections import apply_corrections
        out = apply_corrections("Archaeodictyomitrā", paper_id=None)
        assert out == "Archaeodictyomitra", (
            f"apply_corrections did not strip macron: {out!r}"
        )

    def test_apply_corrections_preserves_c5_rules(self):
        """The two C5 substring rules must still fire on the
        post-normalised string."""
        from rlpe.ocr_corrections import apply_corrections
        out = apply_corrections("Archaeodictyomitracf X", paper_id=None)
        assert "Archaeodictyomitra cf." in out

    def test_apply_corrections_passes_through_clean(self):
        from rlpe.ocr_corrections import apply_corrections
        assert apply_corrections("Theocorys", paper_id=None) == "Theocorys"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
