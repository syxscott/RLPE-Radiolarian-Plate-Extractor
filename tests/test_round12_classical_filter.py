"""Round 12 source-guard test: classical-path hallucination filter.

Locks in the fix for Bug 7 — classical path emitting rows whose
panel_id is NOT mentioned in the caption. Live smoke on Pouille 2014
pl02 found 11 rows (gold has 0 in pl02) because the segmenter
over-segments and OCR mis-reads some labels, producing phantom
panel_ids (10a, 10b, 11b, 11c) that the caption-derived pair set
rejects.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_classical_hallucination_filter_present():
    """The static source check: the new ``_filter_classical_hallucinations``
    method must exist in ``RadiolarianPipeline`` and be called from the
    classical path branch (right after ``match_panels``).
    """
    src = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
    src_text = src.read_text(encoding="utf-8")
    assert "def _filter_classical_hallucinations(" in src_text, (
        "Missing _filter_classical_hallucinations method on RadiolarianPipeline"
    )
    # It must be called from the classical path branch (after
    # match_panels, before M3 stage 4).
    assert "self._filter_classical_hallucinations(" in src_text, (
        "_filter_classical_hallucinations must be called from the classical path branch"
    )
    # Filter count: at least one call site.
    call_count = src_text.count("self._filter_classical_hallucinations(")
    assert call_count >= 1, (
        f"_filter_classical_hallucinations must have ≥ 1 call site, got {call_count}"
    )


def test_classical_filter_drops_phantom_panels():
    """Runtime test: with caption-derived pair_lookup = {1, 2, 3, 4, 5, 6, 7, 8, 9},
    a MatchResult with panel_id=10a should be dropped while panel_id=5b
    (5-prefix) is kept.
    """
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class FakeMatch:
        paper_id: str = "p1"
        figure_id: str = "f1"
        panel_id: str | None = None
        species: str | None = None
        confidence: float = 0.0
        bbox: list = field(default_factory=list)
        label_text: str | None = None
        panel_path: str | None = None
        metadata: dict[str, Any] = field(default_factory=dict)

    from rlpe.pipeline import RadiolarianPipeline

    caption_labels = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    # audit 2026-07-31: the caption used to list 10a-b and 11a-c as
    # EXPLICIT clauses — with the letter-suffix label parsing fixed
    # (batch 3), those are now recognised as real caption labels and
    # the filter correctly KEEPS them. Phantom filtering only applies
    # to labels the caption does NOT mention, so the caption here
    # lists only 1-9.
    caption_text = "figs 1-9. Sample."

    @dataclass
    class FakeCaption:
        caption: str = caption_text

    rows = [
        FakeMatch(panel_id="1", species="A"),
        FakeMatch(panel_id="5b", species="B"),  # 5-prefix → keep
        FakeMatch(panel_id="6b", species="B"),  # 6-prefix → keep
        FakeMatch(panel_id="10a", species="X"),  # 10 not in {1..9} → drop
        FakeMatch(panel_id="10b", species="X"),  # 10 not in {1..9} → drop
        FakeMatch(panel_id="11b", species="X"),  # 11 not in {1..9} → drop
        FakeMatch(panel_id="11c", species="X"),  # 11 not in {1..9} → drop
        FakeMatch(panel_id=None, species=None),  # empty → drop
    ]
    kept = RadiolarianPipeline._filter_classical_hallucinations(
        rows,
        FakeCaption(),
        "p1",
        "f1",
    )
    kept_pids = {r.panel_id for r in kept}
    assert "10a" not in kept_pids, "Filter missed 10a"
    assert "10b" not in kept_pids, "Filter missed 10b"
    assert "11b" not in kept_pids, "Filter missed 11b"
    assert "11c" not in kept_pids, "Filter missed 11c"
    assert "1" in kept_pids, "Filter wrongly dropped 1"
    assert "5b" in kept_pids, "Filter wrongly dropped 5b"
    assert "6b" in kept_pids, "Filter wrongly dropped 6b"


def test_classical_filter_no_caption_keeps_everything():
    """When there's no caption to filter against, the filter is a
    no-op (we have no ground truth)."""
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class FakeMatch:
        paper_id: str = "p1"
        figure_id: str = "f1"
        panel_id: str | None = None
        species: str | None = None
        bbox: list = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class FakeCaption:
        caption: str = ""

    from rlpe.pipeline import RadiolarianPipeline

    rows = [FakeMatch(panel_id="1"), FakeMatch(panel_id="999")]
    kept = RadiolarianPipeline._filter_classical_hallucinations(
        rows,
        FakeCaption(),
        "p1",
        "f1",
    )
    assert len(kept) == 2, f"No caption should keep all rows, got {len(kept)}: {kept}"
