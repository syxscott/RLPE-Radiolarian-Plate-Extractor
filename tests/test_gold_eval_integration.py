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
* ``time.sleep`` — replaced with a no-op so the per-paper 60s wait
  does not block the test runner.

We never call any external API and never invoke the actual backend;
the test runs offline.
"""
import os
import sys


# Safe env-var sentinels — required because the existing module body
# unconditionally reads ANTHROPIC_API_KEY at top-level.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key-no-api-calls")
os.environ.setdefault("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
os.environ.setdefault("ANTHROPIC_MODEL", "MiniMax-M3[1M]")


class _StubBackend:
    """Lightweight stand-in that captures constructor kwargs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._kwargs = kwargs

    def infer_panel(self, **_kw):
        return {"error": "stubbed", "fallback_used": True, "panels": []}


# Monkeypatch the symbol BEFORE gold_eval_anchored imports it. The
# file uses ``from rlpe.llm_backends import MiniMaxM3Backend`` at
# module load, so the binding is captured from rlpe.llm_backends at
# the time of import — replacing the attribute on the source module
# is sufficient.
import rlpe.llm_backends  # noqa: E402

rlpe.llm_backends.MiniMaxM3Backend = _StubBackend  # type: ignore[attr-defined]


# Skip the 60s sleep between papers in the import-time loop.
import time as _time  # noqa: E402

_real_sleep = _time.sleep
_time.sleep = lambda *_a, **_kw: None


sys.path.insert(0, 'scripts')
from gold_eval_anchored import (  # noqa: E402
    load_split,
    compute_aggregate_with_ci,
    run_5fold_cv,
)


def test_load_split_v1():
    """Load the v1 split (6 train + 3 test)."""
    split = load_split('data/splits/research_v1.json')
    assert 'train' in split
    assert 'test' in split
    assert len(split['train']) == 6
    assert len(split['test']) == 3


def test_compute_aggregate_with_ci():
    """Bootstrap CI is a tuple (low, high) of length 2."""
    preds = [
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '1', 'species': 'A', 'confidence': 0.9},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '2', 'species': 'B', 'confidence': 0.9},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '3', 'species': 'A', 'confidence': 0.8},
    ]
    gold = [
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '1', 'species': 'A'},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '2', 'species': 'B'},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '3', 'species': 'A'},
    ]
    f1, ci = compute_aggregate_with_ci(preds, gold, n_bootstrap=100)
    assert isinstance(f1, float)
    assert len(ci) == 2
    assert ci[0] <= f1 <= ci[1]
