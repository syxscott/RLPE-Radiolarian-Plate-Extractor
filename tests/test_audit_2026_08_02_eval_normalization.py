"""Regression tests for audit 2026-08-02 — eval panel_id + species normalization.

The hollis2006 (61.9% F1) and feng2007 (83.9% F1) gaps in the 9-paper
eval were dominated by surface-form mismatches between predicted and
gold species names — Roman vs Arabic numerals, "cf." vs "cf", trailing
punctuation, whitespace runs, etc. The two eval-side normalization
layers below close that gap WITHOUT touching any ``data/gold/*.jsonl``
file:

* **Layer A (panel_id)** — ``gold.match_panel`` now lowercases + splits
  comma-separated gold entries so a gold row labelled ``"1, 2, 3"``
  matches three independent pred rows while compound forms like
  ``"1-3"`` or ``"1a"`` are preserved unchanged.

* **Layer B (species)** — ``gold.normalize_species`` lowercases, converts
  Roman → Arabic (II → 2, III → 3, IV → 4), strips ``cf.``/``aff.`` to
  ``cf``/``aff``, drops parenthesised content, and collapses whitespace.
  ``metrics._norm_species`` runs on top of this for taxonomy rules
  (trinomial fold, sp./spp. strip, ``Archaeo`` → ``Archeo``).

These tests cover both layers through the public ``normalize_species``
and ``match_panel`` entry points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.evaluation.gold import GoldPanel, match_panel, normalize_species  # noqa: E402


# ---------------------------------------------------------------------------
# Layer B — species normalisation
# ---------------------------------------------------------------------------


class TestSpeciesNormalization:
    def test_roman_to_arabic(self):
        """``Hastigerina II`` (gold) must match ``Hastigerina 2`` (pred).

        Real-world hollis2006 panel labels mix Roman numerals in the
        figure caption with Arabic in the OCR'd species list. The eval
        should treat them as the same name.
        """
        assert normalize_species("Hastigerina II") == normalize_species("Hastigerina 2")

    def test_cf_normalization(self):
        """``cf.`` (gold) and ``cf`` (pred) are the same open-nomenclature
        qualifier — only the trailing period differs.
        """
        assert normalize_species("Archaeodictyomitra cf. tumandae") == normalize_species(
            "Archaeodictyomitra cf tumandae"
        )

    def test_aff_normalization(self):
        """``Williriedellum aff. W. sp. S`` vs ``Williriedellum aff W. sp. S``.

        Same open-nomenclature rule as cf., but for ``affinis``. The
        suffix ``W. sp. S`` is preserved — only ``aff.`` is normalised.
        """
        assert normalize_species("Williriedellum aff. W. sp. S") == normalize_species(
            "Williriedellum aff W. sp. S"
        )

    def test_whitespace_collapsing(self):
        """Run of whitespace (typical OCR artefact) collapses to single
        space. ``Entactinia  modesta`` (two spaces) is the same as
        ``Entactinia modesta`` (one space)."""
        assert normalize_species("Entactinia  modesta") == normalize_species("Entactinia modesta")

    def test_case_insensitive(self):
        """All-caps ``ARCHAEODICTYOMITRA`` (typical of older figure OCR
        that lifted the genus name straight from the captioned header)
        matches the canonical Title-case spelling."""
        assert normalize_species("ARCHAEODICTYOMITRA") == normalize_species("Archaeodictyomitra")


# ---------------------------------------------------------------------------
# Layer A — panel_id normalization via match_panel
# ---------------------------------------------------------------------------


class TestPanelIdNormalization:
    def test_comma_split(self):
        """A gold row labelled ``"1, 2, 3"`` represents THREE panels.

        After Layer A split, each predicted row matches the gold entry
        independently. This is the dominant feng2007 pattern where the
        caption lists "fig. 1, 2, 3" against a single plate caption.
        """
        gold = GoldPanel(paper_id="p", figure_id="f", panel_id="1, 2, 3", species=None)
        for tok in ("1", "2", "3"):
            assert match_panel(gold, "p", tok), f"pred {tok!r} should match gold '1, 2, 3'"
        # Pred 4 must NOT match — it's outside the gold range.
        assert not match_panel(gold, "p", "4")

    def test_compound_preserved(self):
        """Compound forms like ``1a`` MUST be preserved verbatim because
        they are a single panel (sub-label of "1") and must NOT be
        split into ``1`` + ``a`` which would falsely match two
        different gold entries.
        """
        gold = GoldPanel(paper_id="p", figure_id="f", panel_id="1a", species=None)
        # Exact match still works.
        assert match_panel(gold, "p", "1a")
        # Single "1" no longer matches compound "1a" (different panel).
        # NOTE: prefix-extension would normally match here — pred "1"
        # against gold "1a" gives suffix "a" which is alphabetic, so
        # extension-is-alpha does match. That's the EXISTING behavior
        # carried forward from the test_numeric_label_no_collapse test
        # in tests/test_gold.py and is intentional: a pred labelled
        # "1" is a candidate sub-panel of gold "1a" (the macro panel).
        # The KEY assertion is that "1a" is NOT split into "1" and "a"
        # by the comma logic.
        gold_macros = GoldPanel(paper_id="p", figure_id="f", panel_id="1", species=None)
        assert match_panel(gold_macros, "p", "1a"), "prefix-extension still works: 1a↔1"
        # And no spurious split on a comma-bearing gold entry that has
        # no comma — should behave exactly as before.
        gold_simple = GoldPanel(paper_id="p", figure_id="f", panel_id="1", species=None)
        assert match_panel(gold_simple, "p", "1")

    def test_lowercase(self):
        """Letter labels are case-insensitive at the panel_id level.

        Gold ``"A"`` (label printed in figure in upper case, e.g. for
        SEM plates that use letter gridding) must match pred ``"a"``.
        """
        gold = GoldPanel(paper_id="p", figure_id="f", panel_id="A", species=None)
        assert match_panel(gold, "p", "a")
        # And compound alphabetic suffix still works with the case fold
        # (alphabetic suffix check ignores case via ``str.isalpha()``
        # being case-aware and both sides having lowercased beforehand).
        gold_a_ext = GoldPanel(paper_id="p", figure_id="f", panel_id="A1", species=None)
        # Pred "a" must NOT match gold "A1" — the suffix "1" is numeric.
        assert not match_panel(gold_a_ext, "p", "a")
        # Exact match (lowered) works.
        assert match_panel(gold, "p", "A")


# ---------------------------------------------------------------------------
# Source-guard — guard against the two-layer design being quietly reverted
# ---------------------------------------------------------------------------


class TestSourceGuard:
    def test_normalize_species_exists_in_gold(self):
        """Source guard: ``normalize_species`` must remain exported from
        ``gold`` for the regression to stay alive."""
        import rlpe.evaluation.gold as gold_mod

        assert hasattr(gold_mod, "normalize_species"), (
            "normalize_species was removed from rlpe.evaluation.gold — "
            "Layer B normalization has been silently dropped from eval"
        )
        assert callable(gold_mod.normalize_species)

    def test_match_panel_uses_comma_split(self):
        """Source guard: ``match_panel`` must continue to split comma-
        separated gold panel_ids. Guard against a future "simplification"
        that drops the split loop and breaks Layer A."""
        import rlpe.evaluation.gold as gold_mod

        gold = GoldPanel(paper_id="p", figure_id="f", panel_id="1, 2, 3", species=None)
        # If the comma split was reverted, this would return False.
        assert gold_mod.match_panel(gold, "p", "2")
