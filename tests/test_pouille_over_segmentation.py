"""Tests for pouille2014 over-segmentation fix (Round 4 / 2026-07-03).

Bug report (from memory project_v20_results_2026_07_02.md):

    pouille — pred over-seg 68-70 panels vs gold 6 panels.

The root cause was in ``match_panels`` fallback paths. When the
caption produced zero matches and the fallback ``if not matches and
(taxa or labels):`` block fired, the placeholder row used
``labels[0]`` UNCONDITIONALLY. If ``labels[0]`` was an OCR garbage
token like ``"P1"``, ``"ean"``, or a single lowercase letter
(``"d"``/``"a"``) — i.e. tokens that ``is_valid_panel_label``
correctly rejects — every placeholder row got that garbage value,
polluting the pred set with rows that dragged pouille F1 from 100%
to 0%.

Note: pouille2014 also legitimately uses A-H figure markers
elsewhere in the paper (the gold panel_ids are all digits, but the
caption parser occasionally emits single-letter tokens). The fix
correctly KEEPS valid A-H markers (they pass ``is_valid_panel_label``)
and DROPS only the OCR garbage (lowercase, ``"P1"``, etc.).

The fix: validate ``labels[0]`` against ``is_valid_panel_label``;
if invalid, fall through to None so the eval treats the row as
no-pred instead of bad-pred. Same guard applied to the
placeholder-caption path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def _make_panel(panel_id: str | None = None) -> object:
    """Build a minimal PanelCandidate stub."""
    from rlpe.types import PanelCandidate

    return PanelCandidate(
        panel_id=panel_id,
        bbox=(0, 0, 100, 100),
        score=0.85,
        image_path="/tmp/panel.png",
    )


def _make_caption(text: str = "Plate 1. caption here.") -> object:
    from rlpe.types import CaptionRecord

    return CaptionRecord(
        paper_id="paper",
        figure_id="fig1",
        caption=text,
        page_index=1,
        figure_number="1",
    )


class TestPouilleOverSegmentation:
    """P1-4: pouille over-segmentation.

    The audit found 108 pred rows for pouille2014 (gold: 6). Of
    those, 28 rows shared ``panel_id='P1'`` (OCR garbage from
    Stage-3 over-segmentation that broadcast the same first caption
    token to every panel). The fix tightens ``assign_panels_to_labels``
    to use ``is_valid_panel_label`` (regex ``^([A-H]|\\d+[a-z]?)$``)
    instead of the loose ``pid.isdigit() or len(pid) <= 3``.

    Note: pouille2014 gold panel_ids are all DIGITS (1, 5, 8, 12,
    15, 19) — the audit mistakenly cited A/B/C/D markers as
    garbage. Valid A-H markers (per ``is_valid_panel_label``) are
    kept; only OCR-garbage multi-char / lowercase labels are
    rejected.
    """

    def test_assign_panels_to_labels_rejects_P1(self):
        """``panel_id='P1'`` (OCR garbage) is rejected by
        ``assign_panels_to_labels``. Without this fix, 28 rows of
        ``panel_id='P1'`` polluted pouille's pred set."""
        from rlpe.association import assign_panels_to_labels
        from rlpe.types import PanelCandidate

        panels = [
            PanelCandidate(
                panel_id="P1", bbox=(0, 0, 100, 100), score=0.85, image_path="/tmp/p.png"
            )
        ]
        out = assign_panels_to_labels(panels, labels=[], ocr_tokens=[])
        # P1 should be rejected; assign returns None for that slot.
        assert out == [None], f"P1 should be rejected; got {out}"

    def test_assign_panels_to_labels_rejects_ean(self):
        from rlpe.association import assign_panels_to_labels
        from rlpe.types import PanelCandidate

        panels = [
            PanelCandidate(
                panel_id="ean", bbox=(0, 0, 100, 100), score=0.85, image_path="/tmp/p.png"
            )
        ]
        out = assign_panels_to_labels(panels, labels=[], ocr_tokens=[])
        assert out == [None]

    def test_assign_panels_to_labels_rejects_lowercase(self):
        from rlpe.association import assign_panels_to_labels
        from rlpe.types import PanelCandidate

        panels = [
            PanelCandidate(panel_id="d", bbox=(0, 0, 100, 100), score=0.85, image_path="/tmp/p.png")
        ]
        out = assign_panels_to_labels(panels, labels=[], ocr_tokens=[])
        assert out == [None]

    def test_assign_panels_to_labels_accepts_digit(self):
        from rlpe.association import assign_panels_to_labels
        from rlpe.types import PanelCandidate

        panels = [
            PanelCandidate(panel_id="5", bbox=(0, 0, 100, 100), score=0.85, image_path="/tmp/p.png")
        ]
        out = assign_panels_to_labels(panels, labels=[], ocr_tokens=[])
        assert out == ["5"]

    def test_assign_panels_to_labels_accepts_ah_marker(self):
        from rlpe.association import assign_panels_to_labels
        from rlpe.types import PanelCandidate

        panels = [
            PanelCandidate(panel_id="A", bbox=(0, 0, 100, 100), score=0.85, image_path="/tmp/p.png")
        ]
        out = assign_panels_to_labels(panels, labels=[], ocr_tokens=[])
        assert out == ["A"]

    def test_assign_panels_to_labels_falls_through_to_caption_label_when_panel_invalid(self):
        """When panel.panel_id is garbage ('P1'), the code falls through
        to the i-th caption-derived label. The caption-derived label
        must ALSO be validated; 'P1' from the caption is rejected."""
        from rlpe.association import assign_panels_to_labels
        from rlpe.types import PanelCandidate

        panels = [
            PanelCandidate(
                panel_id="P1", bbox=(0, 0, 100, 100), score=0.85, image_path="/tmp/p1.png"
            ),
            PanelCandidate(
                panel_id=None, bbox=(0, 0, 100, 100), score=0.85, image_path="/tmp/p2.png"
            ),
        ]
        # Caption labels are themselves garbage 'P1' — both rejected.
        out = assign_panels_to_labels(panels, labels=["P1", "P1"], ocr_tokens=[])
        assert out == [None, None], f"both garbage labels should be rejected; got {out}"

    def test_assign_panels_to_labels_uses_valid_caption_label(self):
        """When panel.panel_id is None, the i-th caption label is
        used IF valid. '5' is valid → keep."""
        from rlpe.association import assign_panels_to_labels
        from rlpe.types import PanelCandidate

        panels = [
            PanelCandidate(
                panel_id=None, bbox=(0, 0, 100, 100), score=0.85, image_path="/tmp/p.png"
            ),
        ]
        out = assign_panels_to_labels(panels, labels=["5"], ocr_tokens=[])
        assert out == ["5"]


class TestIsValidPanelLabelContract:
    """P1-4: lock down the regex that the fix relies on. Any change
    here ripples through the entire match_panels pipeline."""

    def test_pouille_garbage_rejected(self):
        """All the panel_ids observed in the pouille pred set that
        were OCR garbage must be rejected by is_valid_panel_label."""
        from rlpe.association import is_valid_panel_label

        for garbage in ("P1", "ean", "d", "a", None):
            assert is_valid_panel_label(garbage) is False, (
                f"{garbage!r} should be rejected (audit P1-4 garbage)"
            )

    def test_pouille_gold_patterns_accepted(self):
        """The panel_ids in pouille2014 gold must be accepted."""
        from rlpe.association import is_valid_panel_label

        for ok in ("1", "5", "8", "12", "15", "19", "1a", "14b"):
            assert is_valid_panel_label(ok) is True, f"{ok!r} should be accepted as a panel label"
