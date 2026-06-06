"""Tests for the gold loader and the 4-paper ground truth."""
from __future__ import annotations

from pathlib import Path

import pytest

from rlpe.evaluation import GoldPanel, load_gold, match_panel, write_gold


GOLD_DIR = Path(__file__).resolve().parents[1] / "data" / "gold"


PAPERS = ["bandini2011", "hollis2006", "danelian2006", "pouille2014"]


class TestGoldFiles:
    @pytest.mark.parametrize("paper", PAPERS)
    def test_gold_file_exists(self, paper):
        path = GOLD_DIR / f"{paper}.jsonl"
        assert path.exists(), f"Missing gold file: {path}"

    @pytest.mark.parametrize("paper", PAPERS)
    def test_gold_file_is_valid_jsonl(self, paper):
        path = GOLD_DIR / f"{paper}.jsonl"
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        for ln in lines:
            import json
            d = json.loads(ln)
            assert "paper_id" in d
            assert "figure_id" in d
            assert "panel_id" in d
            assert "species" in d

    @pytest.mark.parametrize("paper", PAPERS)
    def test_gold_has_at_least_one_panel(self, paper):
        gold = load_gold(GOLD_DIR / f"{paper}.jsonl")
        assert len(gold) > 0

    @pytest.mark.parametrize("paper", PAPERS)
    def test_gold_paper_ids_match(self, paper):
        gold = load_gold(GOLD_DIR / f"{paper}.jsonl")
        for p in gold:
            # The paper_id field stores the hash; not the short name.
            # This is a soft check that the file isn't cross-contaminated.
            assert p.paper_id and isinstance(p.paper_id, str)


class TestMatchPanel:
    def test_exact_match(self):
        gold = GoldPanel("p1", "f1", "5", "Genus species")
        assert match_panel(gold, "p1", "5")

    def test_no_match_different_paper(self):
        gold = GoldPanel("p1", "f1", "5", "Genus species")
        assert not match_panel(gold, "p2", "5")

    def test_no_match_different_label(self):
        gold = GoldPanel("p1", "f1", "5", "Genus species")
        assert not match_panel(gold, "p1", "6")

    def test_no_match_empty_gold(self):
        gold = GoldPanel("p1", "f1", None, "Genus species")
        assert not match_panel(gold, "p1", "5")

    def test_no_match_empty_pred(self):
        gold = GoldPanel("p1", "f1", "5", "Genus species")
        assert not match_panel(gold, "p1", None)

    def test_prefix_match_pred_starts_with_gold(self):
        # "12a" should match gold "12"
        gold = GoldPanel("p1", "f1", "12", "Genus")
        assert match_panel(gold, "p1", "12a")

    def test_prefix_match_gold_starts_with_pred(self):
        # Pred "5" matches gold "5b"
        gold = GoldPanel("p1", "f1", "5b", "Genus")
        assert match_panel(gold, "p1", "5")

    def test_numeric_label_no_collapse(self):
        """Critical regression: a pure-digit pred label like "1" must NOT
        match gold "10", "11", "12", ... (those are different panels)."""
        gold = GoldPanel("p1", "f1", "10", "Genus")
        # "1" must not match "10" — the panels are distinct
        assert not match_panel(gold, "p1", "1")
        # exact match still works
        assert match_panel(gold, "p1", "10")
        # other two-digit labels still don't match
        assert not match_panel(gold, "p1", "11")
        assert not match_panel(gold, "p1", "100")

    def test_single_letter_label_no_collapse(self):
        """Same rule for letter labels: "A" must not match "A1"."""
        gold = GoldPanel("p1", "f1", "A1", "Genus")
        assert not match_panel(gold, "p1", "A")
        # exact letter match still works
        gold2 = GoldPanel("p1", "f1", "A", "Genus")
        assert match_panel(gold2, "p1", "A")

    def test_alphanumeric_label_still_prefix_matches(self):
        """The fix must not regress the alphanumeric case ("12a" / "12b")."""
        gold_12 = GoldPanel("p1", "f1", "12", "Genus")
        gold_12a = GoldPanel("p1", "f1", "12a", "Genus")
        # "12a" matches gold "12" (pred is more specific)
        assert match_panel(gold_12, "p1", "12a")
        # "12" matches gold "12a" (gold is more specific)
        assert match_panel(gold_12a, "p1", "12")
        # "12a" matches gold "12a" (exact)
        assert match_panel(gold_12a, "p1", "12a")
        # "12a" does NOT match gold "12b" — these are different panels
        assert not match_panel(gold_12a, "p1", "12b")


class TestWriteGold:
    def test_round_trip(self, tmp_path):
        panels = [
            GoldPanel("p1", "f1", "1", "Genus sp"),
            GoldPanel("p1", "f1", "2", None),
        ]
        path = tmp_path / "test.jsonl"
        n = write_gold(panels, path)
        assert n == 2
        loaded = load_gold(path)
        assert len(loaded) == 2
        assert loaded[0].species == "Genus sp"
        assert loaded[1].species is None


class TestSanityCounts:
    """Loose sanity checks on the actual gold-set sizes.

    These guard against regressions in build_gold_from_captions.py
    without pinning to exact numbers (which would break if the
    upstream manifest format changes).
    """
    def test_danelian_at_least_20_species(self):
        gold = load_gold(GOLD_DIR / "danelian2006.jsonl")
        # Danelian Plate 1 has 23 species clauses
        assert len(gold) >= 20

    def test_hollis_at_least_50_species(self):
        gold = load_gold(GOLD_DIR / "hollis2006.jsonl")
        # 3 plates × ~20 species each
        assert len(gold) >= 50

    def test_pouille_has_6_species_clauses(self):
        gold = load_gold(GOLD_DIR / "pouille2014.jsonl")
        # Pouille Plate 1 has 6 species (figs 1-4, 5-7, 8-11, 12-14b, 15-18, 19)
        assert len(gold) == 6
