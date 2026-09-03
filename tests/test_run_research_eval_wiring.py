"""Smoke test that run_research_eval.py correctly wires text_extract + occurrence."""

import os
import sys
from pathlib import Path

REPO = Path("/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Stub MiniMaxM3Backend so import of run_research_eval doesn't try a real call
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("ANTHROPIC_BASE_URL", "https://test.invalid")
os.environ.setdefault("ANTHROPIC_MODEL", "dummy")

import rlpe.llm_backends


class _StubBackend:
    def __init__(self, *a, **kw):
        # Audit 2026-09-03 (CI regression): MiniMaxM3Backend.__post_init__
        # now reads ``self.data_outbound_policy`` and checks the env var
        # for ``api_full`` opt-in. The stub must expose the field
        # (with a default that doesn't trigger the opt-in requirement)
        # so downstream construction doesn't AttributeError.
        self.data_outbound_policy = kw.get("data_outbound_policy", "api_redacted")
        # The user-reported BLOCKER fix also assumes ``__post_init__``
        # exists; some tests do ``mock.patch.object(MiniMaxM3Backend,
        # "__post_init__", lambda self: None)`` which requires the
        # attribute to exist. Add a no-op stub for that case.
        self.__post_init__ = lambda: None

    def infer_panel(self, *a, **kw):
        return {"error": "stubbed"}


rlpe.llm_backends.MiniMaxM3Backend = _StubBackend

# gold_eval_anchored.py is a script-style module: its top-level
# ``for i, slug in enumerate(slugs)`` loop calls ``time.sleep(60)``
# between papers. Stub it out BEFORE importing run_research_eval so
# the transitive import doesn't take 9 minutes.
import time as _time

_time.sleep = lambda *_a, **_kw: None

from run_research_eval import _enrich_preds_with_text_and_group


def test_enrich_attaches_occurrence_group_id():
    preds = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A sp",
            "confidence": 0.9,
        },
        {
            "paper_id": "p1",
            "figure_id": "f2",
            "panel_id": "1",
            "species": "A sp",
            "confidence": 0.8,
        },
    ]
    out = _enrich_preds_with_text_and_group(preds)
    assert all("occurrence_group_id" in r for r in out)
    assert out[0]["occurrence_group_id"] == out[1]["occurrence_group_id"]


def test_enrich_does_not_modify_input():
    preds = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A sp",
            "confidence": 0.9,
        },
    ]
    snapshot = list(preds)
    _enrich_preds_with_text_and_group(preds)
    assert preds == snapshot
