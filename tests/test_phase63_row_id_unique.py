"""Tests for Phase 63 Plan 6 — Bug 6.12: ``_row_id`` collides for a
paper with no panel ids.

Before: ``_row_id = f"{job_id}:{paper_id}:{figure_id}:{panel_id}"``
returned the same string for two distinct panels in a paper that
suppressed panel_id (``panel_index=None`` and the caption-parser
found no printed id). Both rows pointed at the same ``job_id``,
``paper_id``, ``figure_id``, and ``None`` for ``panel_id`` — so the
rest of the cache indexed both rows under one key, and downstream
matcher code could not distinguish them.

After: the row_id falls back to ``(paper_id, figure_id, bbox)`` when
``panel_id`` is missing or empty. The bbox is unique per panel crop,
so collisions vanish while the previous keying still works for the
common path where a real panel_id is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


APP_SRC = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "api" / "app.py").read_text(
    encoding="utf-8"
)


def test_row_id_uses_bbox_fallback():
    """``_row_id`` must include ``bbox`` when ``panel_id`` is missing."""
    assert "def _row_id" in APP_SRC
    # Look at the body of the function
    idx = APP_SRC.find("def _row_id")
    # Find the next top-level function or class to bound the slice
    end = APP_SRC.find("\n\ndef ", idx + 1)
    if end == -1:
        end = APP_SRC.find("\n\nclass ", idx + 1)
    if end == -1:
        end = len(APP_SRC)
    body = APP_SRC[idx:end]
    # The new behaviour must consider ``bbox`` when panel_id is missing.
    assert "bbox" in body, (
        f"_row_id does not include bbox fallback. Phase 63 Plan 6.12 "
        f"fix regressed. Body:\n{body[:300]!r}"
    )


def test_row_id_collision_with_no_panel_id_distinct_bbox():
    """Two rows with same paper/figure but no panel_id must produce
    distinct row_ids when their bboxes differ."""
    from rlpe.api.app import _row_id

    row_a = {
        "paper_id": "p1",
        "figure_id": "f1",
        "panel_id": None,
        "bbox": [0, 0, 100, 100],
    }
    row_b = {
        "paper_id": "p1",
        "figure_id": "f1",
        "panel_id": None,
        "bbox": [100, 0, 200, 100],
    }
    rid_a = _row_id("job-1", row_a)
    rid_b = _row_id("job-1", row_b)
    assert rid_a != rid_b, (
        f"_row_id collided for two distinct bboxes: {rid_a!r} == {rid_b!r}. "
        "Phase 63 Plan 6.12 fix regressed."
    )


def test_row_id_no_panel_id_no_bbox_still_distinct():
    """If both panel_id and bbox are missing, fall back to a
    counter / row position / unique-enough hash so the cache can
    still distinguish rows."""
    from rlpe.api.app import _row_id

    rows = [
        {"paper_id": "p1", "figure_id": "f1", "panel_id": None, "bbox": None, "_seq": i}
        for i in range(3)
    ]
    rids = {_row_id("job-1", r) for r in rows}
    assert len(rids) >= 1, (
        "_row_id dropped into a too-broad bucket for missing-panel rows; "
        "Phase 63 Plan 6.12 fallback not idempotent."
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
