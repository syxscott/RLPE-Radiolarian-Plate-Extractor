"""Phase 63 Plan 6 — Task 6.4 pinning regression test.

Pins the Phase 58 Plan 1.3 (commit 3f20766) fix: xlsx.py ``_row_for_panel``
must read ``panel_id_source`` from ``metadata["panel_id_source"]`` first,
then top-level ``panel_id_source``, and fall back to ``""``. Before the fix,
``xlsx.py`` referenced a non-existent legacy key ``v18_panel_id_source``,
so every exported xlsx had an empty "Panel来源" column.

Source-guard pinning is in ``tests/test_p0_xlsx_panel_id_source.py``;
this file pins the runtime behaviour when xlsx is invoked with a
properly-stamped panel dict.

If openpyxl is not installed in the test env we skip this test
gracefully; the source-guard test still runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

openpyxl = pytest.importorskip("openpyxl")  # noqa: F401

from rlpe.exporters.xlsx import _row_for_panel  # noqa: E402


def _panel_dict(panel_id_source: str | None = "caption") -> dict:
    return {
        "paper_id": "p1",
        "figure_id": "f1",
        "panel_id": "1",
        "panel_path": "/tmp/p.png",
        "bbox": [0, 0, 100, 100],
        "confidence": 0.9,
        "species": "Genus species",
        # PanelRecord schema field — top-level fallback
        "panel_id_source": panel_id_source,
        "metadata": {
            "geology_links": [],
            # The pipeline actually writes the source here, not at top-level
            "panel_id_source": "image_ocr",
        },
    }


def test_xlsx_metadata_panel_id_source_wins():
    """When metadata has panel_id_source, it wins (the pipeline-write path)."""
    row = _row_for_panel(_panel_dict(panel_id_source="caption"))
    # The row schema: paper_id, figure_id, panel_id, panel_id_source, ...
    assert row[3] == "image_ocr", (
        f"xlsx _row_for_panel did not read metadata.panel_id_source; got row[3]={row[3]!r}. "
        "Phase 58 Plan 1.3 fix regressed?"
    )


def test_xlsx_top_level_panel_id_source_fallback():
    """When metadata is empty, top-level panel_id_source becomes the value."""
    panel = _panel_dict(panel_id_source="caption")
    panel["metadata"] = {"geology_links": []}  # no panel_id_source
    row = _row_for_panel(panel)
    assert row[3] == "caption"


def test_xlsx_panel_id_source_empty_default():
    """When neither location has panel_id_source, the cell is empty."""
    panel = _panel_dict(panel_id_source=None)
    panel["metadata"] = {"geology_links": []}
    row = _row_for_panel(panel)
    assert row[3] == ""


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
