"""Regression: audit 2026-09-04 eval-4 — the anchored-caption
selector in :mod:`scripts.gold_eval_anchored` ranked candidate
paragraphs by counting how many gold species appeared in each
candidate paragraph. The one with the most gold-species matches
won. This is circular evaluation: the eval is steered toward
the paragraph that already mentions gold answers, so it ALWAYS
looks correct.

Real failure mode: a paper's actual caption was at the top of the
page ("Plate 5. Spumellarians from the Lower Cretaceous") but a
later paragraph ("Comparison with Bandini 2011 plate 5: Genus
alpha, Genus beta, Genus gamma…") mentioned more gold species.
The script selected the comparison paragraph instead of the real
caption, inflating F1.

Fix contract: the selector must NOT look at gold species when
ranking. Ranking uses structural cues only:
    * Anchor match (``Plate 5.``, ``Fig. 5.``) — preferred
    * First qualifying paragraph on the page — tiebreak
    * Length filter (50-4000 chars) — sanity

This test pins the fix via a source-guard AST check (the
``scripts/gold_eval_anchored.py`` module eagerly initialises an
Anthropic client at import time, so a behavioural test would need
PDF fixtures we don't want to ship).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"


class TestSourceGuardAgainstGoldOverlapRanking:
    def test_no_gold_species_score_loop(self):
        # Source guard: no ``for ... in gold_species`` loop combined
        # with a score variable tied to gold species must remain in
        # the file. The ranking must use structural cues only.
        src = (_SCRIPTS / "gold_eval_anchored.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                src_segment = ast.unparse(node)
                # The bug shape was: ``for sp in gold_species: if sp in p: score += 1``
                if "gold_species" in src_segment and "score" in src_segment:
                    raise AssertionError(
                        "audit 2026-09-04 eval-4 regression: ranking by gold-species "
                        "overlap re-introduced in:\n" + src_segment[:300]
                    )

    def test_no_gold_overlap_log_line(self):
        # The previous ``print(f"...gold_species_overlap={best_score}...")``
        # log line is the only consumer of the dead ``best_score``
        # variable — its removal confirms the variable is gone too.
        src = (_SCRIPTS / "gold_eval_anchored.py").read_text(encoding="utf-8")
        assert "gold_species_overlap" not in src, (
            "audit 2026-09-04 eval-4 regression: log line that emits "
            "gold-overlap score is still in gold_eval_anchored.py — "
            "the variable it references only existed to rank by gold."
        )

    def test_best_score_removed(self):
        # ``best_score`` was the variable that accumulated the gold-
        # species overlap count. Removing it (no other use) confirms
        # the gold-rank code is fully gone.
        src = (_SCRIPTS / "gold_eval_anchored.py").read_text(encoding="utf-8")
        assert "best_score" not in src, (
            "audit 2026-09-04 eval-4 regression: best_score variable "
            "re-introduced — only existed for gold-species ranking"
        )

    def test_anchor_match_first_paragraph_wins(self):
        # The fix replaced ``if score > best_score: best_para = p``
        # with ``best_para = p; break`` — the FIRST anchor-matched
        # paragraph wins (no scoring loop). Confirm the structural
        # shape.
        src = (_SCRIPTS / "gold_eval_anchored.py").read_text(encoding="utf-8")
        # The phrase ``best_para = p`` followed by ``break`` (within
        # 2 lines) is the structural contract.
        import re as _re

        m = _re.search(r"best_para = p\s*\n\s*break", src)
        assert m, (
            "audit 2026-09-04 eval-4: anchor-matched paragraph ranking "
            "no longer uses 'first wins' pattern (best_para = p; break)"
        )
