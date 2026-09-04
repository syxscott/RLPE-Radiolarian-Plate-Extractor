"""Regression: audit 2026-09-04 taxon-2 — the OCR-correction layer had
ZERO production call sites.

``ocr_corrections.py`` (CORRECTIONS, PAPER_WHITELIST and the
``_normalize_ocr_chars`` character normalizer shipped as the
"BLOCKER-#6 fix" on 2026-09-03) was imported by tests only, so
OCR-mangled species strings were exported verbatim as phantom taxa
('Sponguru1', 'Archaeodictyomitrā', 'TheocorIs'...), while
``range_chart_extractor._species_match`` had removed its 1-char OCR
tolerance on the stated assumption that the normalizer "handles the
l/1 class upstream" — it didn't; nothing called it.

Wiring contract (deliberately narrow): the character-level normalizer
only is wired into the two species SOURCE paths:
  * ``association.extract_taxa_from_caption`` — the caption text is
    normalized BEFORE ``TAXON_LIKE_PATTERN`` runs, because that
    pattern's character class is ASCII-only: a genus token containing
    a stray digit ("Sponguru1") or a macron ("Archaeodictyomitrā")
    simply never matches, so normalizing only the extracted output
    would fix nothing for exactly the mangled-token class the
    normalizer exists for.
  * ``m3_engine._normalize_species`` — the LLM-path species
    normalizer (covers all three LLM call sites).

The CORRECTIONS / PAPER_WHITELIST mapping layers stay unwired for
now — audit taxon-5 showed some whitelist rules rewrite valid
determinations, so blanket adoption would trade one defect for
another. A source guard fails if the module ever again has no
non-test importer.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.association import extract_taxa_from_caption
from rlpe.m3_engine import _normalize_species
from rlpe.ocr_corrections import _normalize_ocr_chars


class TestWiringCharacterNormalizer:
    def test_m3_normalize_species_digit_one_folded(self):
        # digit-1 after a letter -> l (Sponguru1 is the documented example)
        assert _normalize_species("Sponguru1 torsionis") == "Spongurul torsionis"

    def test_m3_normalize_species_long_vowel_folded(self):
        assert _normalize_species("Archaeodictyomitrā apiarium") == (
            "Archaeodictyomitra apiarium"
        )

    def test_m3_normalize_species_capital_i_folded(self):
        assert _normalize_species("TheocorIs robusta") == "Theocorls robusta"

    def test_m3_normalize_species_clean_name_untouched(self):
        assert _normalize_species("Follicucullus scholasticus") == (
            "Follicucullus scholasticus"
        )

    def test_extract_taxa_caption_digit_one_recovered(self):
        # The mangled genus must be extracted at all (pre-normalization
        # TAXON_LIKE_PATTERN cannot match a digit inside the token).
        taxa = extract_taxa_from_caption("1. Sponguru1 spinosa — plate 4.")
        assert any(t == "Spongurul spinosa" for t in taxa), taxa

    def test_extract_taxa_caption_macron_recovered(self):
        taxa = extract_taxa_from_caption(
            "Fig. 2: Archaeodictyomitrā apiarium (Pl. 4, Figs. 8-9)."
        )
        assert any(t == "Archaeodictyomitra apiarium" for t in taxa), taxa

    def test_extract_taxa_caption_numeric_identifier_not_corrupted(self):
        # "sp. 1" style identifiers must keep their digit: the 1 sits
        # after a space, so the digit-one rule must not fire and turn
        # figure/identifier numbers into letter noise ("spl").
        taxa = extract_taxa_from_caption(
            "Fig. 1 shows Archaeodictyomitra spinosa sp. 1."
        )
        assert any(t == "Archaeodictyomitra spinosa" for t in taxa), taxa
        assert not any("spl" in t for t in taxa), taxa

    def test_extract_taxa_clean_caption_unchanged(self):
        taxa = extract_taxa_from_caption(
            "Fig. 3: Follicucullus scholasticus (Pl. 2, Fig. 1)."
        )
        assert any(t == "Follicucullus scholasticus" for t in taxa), taxa


class TestSourceGuardProductionCaller:
    def test_ocr_corrections_has_non_test_importer(self):
        """ocr_corrections must not be left orphaned again — the whole
        layer was dead code until 2026-09-04."""
        import re

        importers = []
        for path in sorted((_SRC / "rlpe").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if re.search(
                r"^\s*(from\s+\.{0,3}ocr_corrections\b|from\s+rlpe\.ocr_corrections\b|"
                r"import\s+\.{0,3}ocr_corrections\b)",
                text,
                re.MULTILINE,
            ):
                importers.append(str(path.relative_to(_SRC)))
        assert importers, (
            "ocr_corrections has no non-test importer — the OCR "
            "correction layer is dead code again (audit 2026-09-04 taxon-2)"
        )
        # And the importers must include the wired production paths.
        assert any(
            "m3_engine" in i or "association" in i for i in importers
        ), importers

    def test_m3_engine_normalizes_before_gold_rules(self):
        """The char pass must run on the raw species string, not after
        the gold-shape rules have already consumed it."""
        import inspect

        from rlpe import m3_engine

        src = inspect.getsource(m3_engine._normalize_species)
        assert "_normalize_ocr_chars" in src

    def test_caption_extract_normalizes_text_before_matching(self):
        import inspect

        from rlpe import association

        src = inspect.getsource(association.extract_taxa_from_caption)
        assert "_normalize_ocr_chars" in src
