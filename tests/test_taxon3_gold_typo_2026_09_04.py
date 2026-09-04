"""Regression: audit 2026-09-04 taxon-3 — ``Amphiscraspedum`` misspelling
in the gold standard.

Gold panel 14 / 20 / 21 of ``data/gold/hollis2006.jsonl`` carried
"Amphiscraspedum prolixum" (a non-accepted genus name) on the same
plate as lines 15-19 "Amphicraspedum gracilis/murrayanum" (the
accepted Haeckel 1882/1887 genus). Mikrotax.org and the DSDP Leg 10
report both confirm ``Amphicraspedum`` Sanfilippo & Riedel 1973 as
the valid species; ``Amphiscraspedum`` is not an accepted name in any
of the consulted radiolarian taxonomic databases.

Consequence (verified during audit): ``_species_compatible`` returned
false even when the prediction was the correct accepted spelling,
so gold panel 14 could never match its own prediction — the tool was
measuring string-distance from corrupted ground truth. Production
output also leaked the misspelling into 3 rows of the exported
species table, so a PBDB/GBIF submission would have shipped a genus
that does not exist.

Fix: normalise the misspelling to the accepted spelling at the gold
file. A source guard asserts the misspelling never re-appears in
``data/gold/*.jsonl`` so a future gold rebuild (e.g. when the
parser-derived caveat is finally retired) cannot regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.evaluation.gold import normalize_species


_GOLD_DIR = Path(__file__).resolve().parent.parent / "data" / "gold"


class TestGoldMisspellingFixed:
    def test_hollis2006_panel_14_accepted_spelling(self):
        from rlpe.evaluation.gold import load_gold  # type: ignore[attr-defined]

        # The fix is on disk; loading and checking is sufficient and
        # survives any future re-ordering.
        rows = load_gold(_GOLD_DIR / "hollis2006.jsonl")
        misspell_rows = [
            r for r in rows if (r.species or "").startswith("Amphiscraspedum")
        ]
        accepted_rows = [
            r for r in rows if (r.species or "").startswith("Amphicraspedum")
        ]
        assert not misspell_rows, (
            "gold still contains the non-accepted genus 'Amphiscraspedum' "
            f"on rows {[(r.panel_id, r.species) for r in misspell_rows]}"
        )
        assert accepted_rows, "expected at least one accepted-spelling row to remain"

    def test_accepted_spelling_in_gold_matches_eval_normaliser(self):
        # The eval-side normaliser must recognise the accepted
        # spelling without altering it (lowercase is applied by
        # ``_species_compatible`` downstream, not here).
        assert normalize_species("Amphicraspedum prolixum") == "Amphicraspedum prolixum"


class TestSourceGuardNoMisspellingInGold:
    def test_no_accepted_misspelling_anywhere_in_gold(self):
        """Gold re-builds (from caption parser) cannot re-introduce
        the misspelling. A bare substring scan catches regressions
        across all papers, not just hollis2006."""
        offenders: list[str] = []
        for path in sorted(_GOLD_DIR.glob("*.jsonl")):
            for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "Amphiscraspedum" in line:
                    offenders.append(f"{path.name}:{ln}")
        assert not offenders, (
            f"gold directory still contains the non-accepted genus "
            f"'Amphiscraspedum': {offenders}"
        )