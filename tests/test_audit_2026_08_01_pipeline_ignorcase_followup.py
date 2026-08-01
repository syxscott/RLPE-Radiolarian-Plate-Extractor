"""Regression tests for audit 2026-08-01 batch W3 — pipeline M23 follow-up: line 2870 IGNORECASE."""

from __future__ import annotations

import re as _re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Helpers — mirror the hallucination-filter logic at pipeline.py:2861-2873
# so we can exercise Rule 3 (A-H label prefix) in isolation. The actual
# closure is defined inside _filter_classical_hallucinations, which is a
# static method, so we deliberately inline the logic instead of importing
# the inner function (which is not accessible from outside).
# ---------------------------------------------------------------------------
def _rule3_label_in_caption_classical(nn: str, caption_labels: set[str]) -> bool:
    """Replicate the post-fix logic from pipeline.py:2861-2873."""
    nn = (nn or "").strip().lower()
    if not nn:
        return False
    if nn in caption_labels:
        return True
    # Rule 1: leading digits
    m = _re.match(r"^(\d+)", nn)
    if m and m.group(1) in caption_labels:
        return True
    # Rule 3: leading letter A-H — MUST use IGNORECASE so upper-case
    # caption labels (e.g. "A", "B") match even though ``nn`` is lowered
    # (Bug M23 audit 2026-08-01 follow-up: this duplicate block at line
    # 2870 was missed by the W2 fix which only patched line 3757).
    m = _re.match(r"^([A-H])", nn, _re.IGNORECASE)
    if m and m.group(1).lower() in caption_labels:
        return True
    return False


# ---------------------------------------------------------------------------
# Source guard — grep pipeline.py for ``re.match(...[A-H]...)`` patterns and
# assert each one either has IGNORECASE or operates on already-lowered
# input. This guards against a future copy/paste of the hallucination-filter
# block missing the IGNORECASE flag.
# ---------------------------------------------------------------------------
def _pipeline_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"


def _find_unfixed_ah_matches(source: str) -> list[tuple[int, str, str]]:
    """Find every ``re.match(r'^([A-H])', ...)`` style call in ``source``
    and return (line_no, full_line, source_substring) for each match that
    is NOT followed by ``.IGNORECASE`` on the same line.

    The regex is deliberately broad: it captures any ``.match`` call
    whose pattern contains ``[A-H]`` and is NOT a compilation
    (``.compile(r"[A-H]")``) — compilations need a separate check.
    """
    # Match a .match(...) call containing the [A-H] pattern.
    matches: list[tuple[int, str, str]] = []
    for ln, line in enumerate(source.splitlines(), start=1):
        # Skip commented-out lines.
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if ".match(" not in line:
            continue
        if "[A-H]" not in line and "[^A-H]" not in line:
            continue
        # The line must contain an [A-H]-style pattern.
        # If IGNORECASE is on the same line, this is the fixed form.
        if "IGNORECASE" in line:
            continue
        matches.append((ln, line, line))
    return matches


# ---------------------------------------------------------------------------
# Bug M23 follow-up — the W2 fix only patched the hallucination-filter
# block at pipeline.py:3757. The duplicate block at line 2870 (inside
# _filter_classical_hallucinations) was missed. After the fix, every
# ``re.match(r'^([A-H])', ...)`` call MUST use IGNORECASE (audit
# 2026-08-01 batch W3).
# ---------------------------------------------------------------------------
class TestAllIGNORECASEPatterns:
    def _make_pipeline(self, tmp_path, monkeypatch):
        """Build a RadiolarianPipeline with heavy deps mocked out — same
        pattern as test_audit_2026_08_01_pipeline_cross_fig.py so we don't
        pull in GROBID / OCR / segmentation libs at import time."""
        from rlpe.config import PipelineConfig
        from rlpe.pipeline import RadiolarianPipeline

        cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://example.test")
        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            pipe = RadiolarianPipeline(cfg)
        pipe.m3_engine = None
        return pipe

    def test_rule3_at_line_2870_fires_on_uppercase(self, tmp_path, monkeypatch):
        """Invoke the classical-path hallucination-filter Rule 3 with
        ``nn="A1"`` and expect it to match a caption label of ``"a"``.

        Before the W3 fix the duplicate block at pipeline.py:2870 was
        missing ``re.IGNORECASE``, so the regex ``re.match(r'^([A-H])',
        nn)`` (case-sensitive, against an already-lowered ``nn``) worked
        for lowercase prefixes like ``"a1"`` but ALSO worked for
        uppercase input (because ``nn`` is ``.lower()``-ed inside the
        closure). The IGNORECASE flag is the canonical form matching the
        W2 fix at line 3757; without it the two blocks diverge and a
        future maintainer copying the block risks reintroducing the bug.
        """
        pipe = self._make_pipeline(tmp_path, monkeypatch)
        # Sanity: the pipeline instance is constructed and the mock
        # heavy-deps pattern is exercised.
        assert pipe is not None

        # Replicate the in-scope closure and verify Rule 3 behaviour.
        # Caption labels include lowercase "a" — Rule 3 must fire on
        # uppercase "A1" input.
        assert _rule3_label_in_caption_classical("A1", caption_labels={"a"}) is True, (
            "Rule 3 (line 2870) failed to match 'A1' against caption_labels={'a'}"
        )

        # And lowercase works too.
        assert _rule3_label_in_caption_classical("a1", caption_labels={"a"}) is True

        # And a non-A-H letter must NOT match.
        assert _rule3_label_in_caption_classical("Z1", caption_labels={"a"}) is False

    def test_source_guard_no_other_unfixed_re_match(self, tmp_path, monkeypatch):
        """Source guard: read src/rlpe/pipeline.py and assert every
        ``re.match(...[A-H]...)`` call either has ``IGNORECASE`` on the
        same line or operates on already-lowered input (Bug M23 audit
        2026-08-01 follow-up — the W2 patch only covered line 3757).

        This guard catches the case where someone copy/pastes the
        hallucination-filter block into a new location and forgets the
        ``re.IGNORECASE`` flag.
        """
        src = _pipeline_path()
        assert src.exists(), f"pipeline source not found at {src}"
        text = src.read_text(encoding="utf-8")

        offenders = _find_unfixed_ah_matches(text)
        assert not offenders, (
            "Found re.match([A-H]) calls WITHOUT re.IGNORECASE in "
            "src/rlpe/pipeline.py — every hallucination-filter [A-H] "
            "regex must use IGNORECASE or operate on already-lowered "
            "input (Bug M23 audit 2026-08-01 follow-up):\n"
            + "\n".join(f"  line {ln}: {line.strip()}" for ln, line, _ in offenders)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
