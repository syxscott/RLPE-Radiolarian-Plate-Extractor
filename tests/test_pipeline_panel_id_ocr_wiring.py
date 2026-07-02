"""Tests for printed_panel_id wiring into exported PanelRecord fields."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.association import match_panels  # noqa: E402
from rlpe.converters import panel_record_from_match  # noqa: E402
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
        score=0.9,
        image_path="panel.png",
        metadata=metadata or {},
    )


def test_valid_image_ocr_label_reaches_panel_record_fields():
    panel = _panel(
        "3",
        metadata={
            "printed_panel_id": "3",
            "caption_panel_id": "1",
            "panel_id_source": "image_ocr",
            "label_region_ocr": "3",
            "label_region_picked": "3",
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

    record = panel_record_from_match(matches[0])
    assert record.panel_id == "3"
    assert record.caption_panel_id == "1"
    assert record.printed_panel_id == "3"
    assert record.canonical_panel_id == "3"
    assert record.panel_id_source == "image_ocr"
    assert "missing_printed_panel_id" not in record.review_reasons


def test_caption_panel_id_is_preserved_when_image_ocr_differs():
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

    record = panel_record_from_match(matches[0])
    assert record.caption_panel_id == "1"
    assert record.printed_panel_id == "2a"
    assert record.panel_id_source == "image_ocr"
    assert "missing_printed_panel_id" not in record.review_reasons


def test_missing_image_ocr_label_keeps_legacy_source_and_review_reason():
    panel = _panel("1", metadata={})

    matches = match_panels(
        paper_id="p1",
        figure_id="fig1",
        caption=_caption(),
        panels=[panel],
        ocr_tokens=[],
        taxon_entities=[],
    )

    record = panel_record_from_match(matches[0])
    assert record.printed_panel_id is None
    assert record.panel_id_source == "legacy"
    assert "missing_printed_panel_id" in record.review_reasons
