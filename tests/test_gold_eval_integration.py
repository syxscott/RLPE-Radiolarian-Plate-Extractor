"""Task 5: 5-fold CV + bootstrap CI integration tests.

Tests for load_split, compute_aggregate_with_ci, run_5fold_cv functions
in scripts/gold_eval_anchored.py.

NOTE: ``scripts/gold_eval_anchored.py`` is a runnable *script* (not a
library). At import time it kicks off a 9-paper eval loop that calls
the real M3 API and ``time.sleep(60)`` between papers. We do NOT want
any of that for the unit test, so before importing the module we
stub out two collaborators:

* ``rlpe.llm_backends.MiniMaxM3Backend`` — replaced with a no-op class
  that returns a deterministic ``{"error": "stubbed", ...}`` payload
  from ``infer_panel``, so the import-time loop short-circuits via
  the ``r.get('error') or r.get('fallback_used')`` guard in
  ``gold_eval_anchored.py``.
* ``time.sleep`` — autouse fixture monkeypatches only within each
  test function, so the per-paper 60s wait does not block the test
  runner without leaking into other tests.

We never call any external API and never invoke the actual backend;
the test runs offline.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Safe env-var sentinels — required because the existing module body
# unconditionally reads ANTHROPIC_API_KEY at top-level.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key-no-api-calls")
os.environ.setdefault("ANTHROPIC_BASE_URL", "https://test.invalid")
os.environ.setdefault("ANTHROPIC_MODEL", "test-dummy-model")


class _StubBackend:
    """Lightweight stand-in that captures constructor kwargs."""

    def __init__(self, *a, **kw):
        pass

    def infer_panel(self, **_kw):
        return {"error": "stubbed", "fallback_used": True, "panels": []}


def _import_gold_eval_anchored():
    """Import ``gold_eval_anchored`` with ``MiniMaxM3Backend`` and
    ``time.sleep`` stubbed, then RESTORE both attributes.

    Audit 2026-09-04 (CI regression): this file used to replace
    ``rlpe.llm_backends.MiniMaxM3Backend`` with ``_StubBackend`` and
    no-op ``time.sleep`` at module level and never restore them.
    Every test file collected *after* this one then saw the stub
    instead of the real backend class (→ AttributeError in
    test_phase61 / test_run_research_eval_wiring / llm_backends
    suites) and a no-op sleep (→ M9 backoff-jitter failures). The
    stubs are only needed while ``gold_eval_anchored``'s module-level
    backend construction + 60s paper loop execute, so keep them
    scoped to the import.
    """
    import time as _time  # noqa: E402

    import rlpe.llm_backends  # noqa: E402

    real_backend = rlpe.llm_backends.MiniMaxM3Backend
    real_sleep = _time.sleep
    # Monkey-patch the symbol BEFORE gold_eval_anchored imports it. The
    # file uses ``from rlpe.llm_backends import MiniMaxM3Backend`` at
    # module load, so the binding is captured from rlpe.llm_backends at
    # the time of import — replacing the attribute on the source module
    # is sufficient.
    rlpe.llm_backends.MiniMaxM3Backend = _StubBackend  # type: ignore[attr-defined]
    _time.sleep = lambda *_a, **_kw: None
    try:
        import gold_eval_anchored

        return gold_eval_anchored
    finally:
        rlpe.llm_backends.MiniMaxM3Backend = real_backend  # type: ignore[attr-defined]
        _time.sleep = real_sleep


_gold_eval = _import_gold_eval_anchored()

import pytest  # noqa: E402

load_split = _gold_eval.load_split
compute_aggregate_with_ci = _gold_eval.compute_aggregate_with_ci
run_5fold_cv = _gold_eval.run_5fold_cv


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Per-test guard: undo any test-side sleeps. (time.sleep was
    already stubbed at module load to skip the import-time 60s
    waits.) This fixture only matters if a test reassigns time.sleep.
    """
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)


def test_load_split_v1():
    """Load the v1 split (6 train + 3 test)."""
    split = load_split("data/splits/research_v1.json")
    assert "train" in split
    assert "test" in split
    assert len(split["train"]) == 6
    assert len(split["test"]) == 3


def test_compute_aggregate_with_ci_multi_paper():
    """Multi-paper fixture: A has 1 FN, B has 1 FP. Bootstrap must give
    a non-degenerate CI (lower < upper) and the point estimate must lie
    inside the CI.
    """
    preds = [
        # Paper A: 2 perfect, 1 missing (1 FN)
        {"paper_id": "A", "figure_id": "f1", "panel_id": "1", "species": "A sp", "confidence": 0.9},
        {"paper_id": "A", "figure_id": "f1", "panel_id": "2", "species": "B sp", "confidence": 0.9},
        # Paper B: 1 wrong (1 FP), 1 missing (1 FN)
        {
            "paper_id": "B",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "WRONG sp",
            "confidence": 0.5,
        },
    ]
    gold = [
        {"paper_id": "A", "figure_id": "f1", "panel_id": "1", "species": "A sp"},
        {"paper_id": "A", "figure_id": "f1", "panel_id": "2", "species": "B sp"},
        {"paper_id": "A", "figure_id": "f1", "panel_id": "3", "species": "C sp"},
        {"paper_id": "B", "figure_id": "f1", "panel_id": "1", "species": "CORRECT sp"},
    ]
    f1, ci = compute_aggregate_with_ci(preds, gold, n_bootstrap=100)
    # Micro F1 with TP=2, FP=1, FN=2 → P=2/3, R=2/4, F1=0.5
    assert 0.4 < f1 < 0.6, f"expected ~0.5, got {f1}"
    assert ci[0] < ci[1], f"CI should have width > 0, got {ci}"
    assert ci[0] <= f1 <= ci[1], f"point {f1} should be in CI {ci}"


def test_run_5fold_cv_holds_out_correctly():
    """5-fold CV must hold out the fold (not train on it)."""
    # 5 papers with 5 folds → each fold is exactly one paper (LOO-like).
    # 'BAD' has a wrong species; the others are perfect. With proper
    # held-out CV, the BAD fold must have F1=0 and the others F1=1.
    papers = ["p1", "p2", "p3", "p4", "BAD"]
    preds_by_paper = {
        p: [
            {
                "paper_id": p,
                "figure_id": "f1",
                "panel_id": "1",
                "species": "X sp",
                "confidence": 1.0,
            }
        ]
        for p in papers
        if p != "BAD"
    }
    preds_by_paper["BAD"] = [
        {
            "paper_id": "BAD",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "WRONG sp",
            "confidence": 1.0,
        }
    ]
    gold_by_paper = {
        p: [{"paper_id": p, "figure_id": "f1", "panel_id": "1", "species": "X sp"}] for p in papers
    }
    out = run_5fold_cv(preds_by_paper, gold_by_paper, papers, n_folds=5)
    # Should produce exactly 5 folds (BLOCKER-2 fix: numpy.array_split)
    assert len(out["folds"]) == 5, f"expected 5 folds, got {len(out['folds'])}"
    # The fold containing 'BAD' should have F1=0; the other folds F1=1.0
    # (BLOCKER-1 fix: held-out, not train complement)
    for fm in out["folds"]:
        papers_in_fold = [str(x) for x in fm["papers"]]
        if "BAD" in papers_in_fold:
            assert fm["f1"] < 0.1, f"fold with BAD should have low F1, got {fm['f1']}"
        else:
            assert fm["f1"] > 0.9, f"fold without BAD should have high F1, got {fm['f1']}"
