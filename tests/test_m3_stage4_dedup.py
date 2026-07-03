"""Tests for _apply_m3_stage4 panel dedup (Round 6 fix).

A real-world test on the Beccaro_2006 PDF (35 panels across one
plate) showed panel_id="1" repeated 4 times with different
confidence values. Root cause: the classical CV detector produced
N physical panel detections per logical panel (over-segmentation),
all with the same panel_id="1" and similar bboxes. The previous
``_apply_m3_stage4`` called M3 once per row, so the same physical
panel was sent to the API N times — wasting cost and producing
duplicate rows in the output.

The fix: dedup by (panel_id, bbox-tuple) before calling M3.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

HAS_CV2 = True
try:
    import cv2  # noqa: F401
except Exception:
    HAS_CV2 = False

requires_cv2 = pytest.mark.skipif(not HAS_CV2, reason="pipeline import requires cv2")


class TestApplyM3Stage4Dedup:
    """Static source guard — _apply_m3_stage4 must dedup by (panel_id, bbox)."""

    def test_source_has_dedup_loop(self):
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
        text = path.read_text(encoding="utf-8")
        # Locate _apply_m3_stage4 function body.
        marker = "def _apply_m3_stage4("
        i = text.find(marker)
        assert i > 0
        # Use the next top-level ``def `` as the end-marker.
        next_def = text.find("\n    def ", i + 1)
        assert next_def > 0
        body = text[i:next_def]
        # The fix: a ``seen_panel_keys`` set + ``deduped_matches`` list
        # must be built BEFORE the ``for m in matches`` loop so the
        # iteration consumes the deduped list.
        assert "seen_panel_keys" in body, (
            "_apply_m3_stage4 must dedup panels by (panel_id, bbox) "
            "to avoid wasted API calls on over-segmented figures"
        )
        assert "deduped_matches" in body
        # The for loop must iterate over deduped_matches, NOT matches.
        assert "for m in deduped_matches" in body, (
            "The for loop must iterate over deduped_matches, not the "
            "original matches list — otherwise the dedup is no-op"
        )
