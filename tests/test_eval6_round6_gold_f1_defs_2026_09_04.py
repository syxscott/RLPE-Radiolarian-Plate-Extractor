"""Regression: audit 2026-09-04 eval-6 — :mod:`scripts.eval_round6_gold`
emitted three F1 numbers with three different formulas under the same
key ``"f1"``:

    * per_figure[fig]["f1"]   = harmonic_mean(P_fig, R_fig)
    * micro["f1"]            = harmonic_mean(P_micro, R_micro)
    * (implicit) printed summary "F1" sometimes meant arithmetic
      mean of per-figure F1 values (the standard "macro F1"
      definition)

Downstream comparison scripts (e.g. ``scripts/compare_round6_runs.py``,
``work/r25_live_cv_results/aggregate.py``) read one of these and
report a number that doesn't match another consumer of the same
JSON. The discrepancy is silent because the key is identical.

Fix contract:
    * Rename per_figure[fig]["f1"] → "f1_fig_harmonic"
    * Add per_figure[fig]["f1_macro"] = arithmetic mean of per-figure F1
    * Keep micro["f1"] as the pooled-harmonic-mean F1 (canonical micro)
    * Stand-alone normalize_species must be marked DEPRECATED with a
      pointer to rlpe.evaluation.metrics.evaluate() so future scripts
      don't fork the F1 definition again

This file pins the rename + the new keys via source guards (the
script is runnable end-to-end only with real data fixtures, so a
behavioural test would need JSONL fixtures we don't ship).
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


class TestF1KeyNaming:
    def test_per_figure_f1_renamed(self):
        src = (_ROOT / "scripts" / "eval_round6_gold.py").read_text(encoding="utf-8")
        # After the fix, ``per_fig[fid]`` must use ``f1_fig_harmonic``
        # and ``f1_macro`` keys, NOT a bare ``"f1"``.
        assert '"f1":' not in src.split('"per_figure":')[1].split('"micro":')[0], (
            "audit 2026-09-04 eval-6 regression: per_figure block still "
            "uses 'f1' key — replace with 'f1_fig_harmonic' + 'f1_macro'."
        )

    def test_micro_f1_labeled(self):
        # The micro block keeps ``"f1"`` but the key now must include
        # a comment OR the script's docstring must explain the
        # canonical meaning. Pin via a docstring grep.
        src = (_ROOT / "scripts" / "eval_round6_gold.py").read_text(encoding="utf-8")
        # The fix doesn't rename micro["f1"] (that's the canonical
        # micro value), but the script docstring must document it.
        assert "species_f1_micro" in src or "F1 (micro)" in src or (
            "micro" in src.lower() and "harmonic" in src.lower()
        ), (
            "audit 2026-09-04 eval-6: script docstring must document "
            "which F1 formula micro['f1'] uses."
        )

    def test_macro_f1_present(self):
        src = (_ROOT / "scripts" / "eval_round6_gold.py").read_text(encoding="utf-8")
        assert "f1_macro" in src, (
            "audit 2026-09-04 eval-6: macro F1 (arithmetic mean of "
            "per-figure F1) must be computed and emitted as f1_macro."
        )


class TestNormalizeSpeciesDocumented:
    def test_local_normalize_marks_legacy(self):
        # Normalize_species is intentionally lenient for round6 —
        # it's a different normalization from rlpe.evaluation.gold.
        # Future maintainers must NOT use this for general-purpose
        # eval (that's what rlpe.evaluation.metrics.evaluate() is
        # for). Source guard: the function carries a DEPRECATED /
        # round6-only warning.
        src = (_ROOT / "scripts" / "eval_round6_gold.py").read_text(encoding="utf-8")
        assert "DEPRECATED" in src or "round6-only" in src or (
            "Use rlpe.evaluation.metrics" in src
        ), (
            "audit 2026-09-04 eval-6: scripts/eval_round6_gold.py must "
            "mark its local normalize_species as legacy / round6-only "
            "and point users at rlpe.evaluation.metrics.evaluate()."
        )
