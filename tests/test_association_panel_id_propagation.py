"""Tests for propagating image-OCR panel-id metadata through match_panels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.association import match_panels  # noqa: E402
from rlpe.types import CaptionRecord, PanelCandidate  # noqa: E402


def _caption(text: str = "Plate 1. figs 1. Actinomma leptodermum") -> CaptionRecord:
    return CaptionRecord(
        paper_id="p1",
        figure_id="fig1",
        caption=text,
        page_index=1,
        figure_number="1",
    )


def _panel(pid: str | None = "1", metadata: dict | None = None) -> PanelCandidate:
    return PanelCandidate(
        panel_id=pid,
        bbox=(0, 0, 100, 100),
        score=0.8,
        image_path="panel.png",
        metadata=metadata or {},
    )


def test_match_panels_propagates_printed_panel_id_from_panel_metadata():
    panel = _panel(
        "3",
        metadata={
            "printed_panel_id": "3",
            "panel_id_source": "image_ocr",
            "label_region_ocr": "3 2a 1",
            "label_region_picked": "3",
        },
    )

    matches = match_panels(
        paper_id="p1",
        figure_id="fig1",
        caption=_caption("Plate 1. figs 3. Actinomma leptodermum"),
        panels=[panel],
        ocr_tokens=[],
        taxon_entities=[],
    )

    assert len(matches) == 1
    assert matches[0].metadata["printed_panel_id"] == "3"
    assert matches[0].metadata["panel_id_source"] == "image_ocr"
    assert matches[0].metadata["label_region_ocr"] == "3 2a 1"
    assert matches[0].metadata["label_region_picked"] == "3"


def test_match_panels_propagates_caption_panel_id():
    panel = _panel(
        "2a",
        metadata={
            "printed_panel_id": "2a",
            "caption_panel_id": "1",
            "panel_id_source": "image_ocr",
        },
    )

    matches = match_panels(
        paper_id="p1",
        figure_id="fig1",
        caption=_caption(),
        panels=[panel],
        ocr_tokens=[],
        taxon_entities=[],
    )

    assert len(matches) == 1
    assert matches[0].metadata["caption_panel_id"] == "1"
    assert matches[0].metadata["printed_panel_id"] == "2a"
    assert matches[0].metadata["panel_id_source"] == "image_ocr"


def test_match_panels_keeps_association_metadata_when_propagating_panel_keys():
    panel = _panel(
        "5",
        metadata={
            "printed_panel_id": "5",
            "panel_id_source": "image_ocr",
            "panel_ocr_text": "5",
            "panel_ocr_token_count": 1,
        },
    )

    matches = match_panels(
        paper_id="p1",
        figure_id="fig1",
        caption=_caption("Plate 2. figs 5. Actinomma leptodermum"),
        panels=[panel],
        ocr_tokens=[],
        taxon_entities=[],
    )

    assert len(matches) == 1
    meta = matches[0].metadata
    assert meta["printed_panel_id"] == "5"
    assert meta["panel_ocr_text"] == "5"
    assert meta["panel_ocr_token_count"] == 1
    assert meta["figure_number"] == "1"
    assert meta["panel_score"] == 0.8
    assert meta["matcher_type"] == "heuristic"
