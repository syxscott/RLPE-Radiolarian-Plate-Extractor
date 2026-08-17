"""Phase 66 Plan C.2 — cross_figure_visual_links schema field tests.

The schema field carries the VISION-coordinate links that the
Phase C linker emits, parallel to ``link_source`` /
``link_confidence`` / ``link_figure_id`` (which the Phase A text-only
linker populated).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlpe.schema_models import PanelMetadata, PanelRecord, StrictModel

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "rlpe-v1.0.0.json"


class TestCrossFigureVisualLinksField:
    def test_field_exists_on_panel_metadata(self):
        meta = PanelMetadata()
        assert hasattr(meta, "cross_figure_visual_links")
        assert meta.cross_figure_visual_links == []

    def test_field_default_is_empty_list(self):
        meta = PanelMetadata.model_validate({})
        assert meta.cross_figure_visual_links == []

    def test_field_accepts_list_of_dicts(self):
        meta = PanelMetadata(
            cross_figure_visual_links=[
                {
                    "target_figure_id": "fig_strat_2",
                    "target_layer": 3,
                    "target_age": "Late Cretaceous",
                    "target_formation": "Scaglia Rossa",
                    "confidence": 0.92,
                    "source": "m3_visual",
                }
            ]
        )
        assert len(meta.cross_figure_visual_links) == 1
        entry = meta.cross_figure_visual_links[0]
        assert entry["target_figure_id"] == "fig_strat_2"
        assert entry["target_layer"] == 3
        assert entry["confidence"] == 0.92
        assert entry["source"] == "m3_visual"

    def test_field_can_hold_multiple_entries(self):
        meta = PanelMetadata(
            cross_figure_visual_links=[
                {
                    "target_figure_id": "fig_strat_2",
                    "target_layer": 1,
                    "target_age": "Late Cretaceous",
                    "target_formation": "Scaglia",
                    "confidence": 0.88,
                    "source": "m3_visual",
                },
                {
                    "target_figure_id": "fig_strat_2",
                    "target_layer": 2,
                    "target_age": "Late Cretaceous",
                    "target_formation": "Scaglia",
                    "confidence": 0.81,
                    "source": "m3_visual",
                },
            ]
        )
        assert len(meta.cross_figure_visual_links) == 2

    def test_field_optional_keys_use_none(self):
        """target_layer / target_age / target_formation are all optional
        in the visual-link contract — the M3 prompt contract marks
        them all as nullable. The schema must round-trip None values."""
        meta = PanelMetadata(
            cross_figure_visual_links=[
                {
                    "target_figure_id": "fig_strat_2",
                    "target_layer": None,
                    "target_age": None,
                    "target_formation": None,
                    "confidence": 0.7,
                    "source": "m3_visual",
                }
            ]
        )
        entry = meta.cross_figure_visual_links[0]
        assert entry["target_layer"] is None
        assert entry["target_age"] is None
        assert entry["target_formation"] is None


class TestPanelRecordRoundTrip:
    def test_panel_record_round_trip(self):
        """A PanelRecord carrying the new field must round-trip through
        JSON serialization without losing data."""
        rec = PanelRecord(
            paper_id="test_paper",
            figure_id="fig1",
            panel_id="fig1_1",
            species="Genus species",
            confidence=0.9,
            panel_path="path/to/panel.png",
            metadata=PanelMetadata(
                cross_figure_visual_links=[
                    {
                        "target_figure_id": "fig_strat_2",
                        "target_layer": 5,
                        "target_age": "Late Triassic",
                        "target_formation": "Scaglia",
                        "confidence": 0.95,
                        "source": "m3_visual",
                    }
                ]
            ),
        )
        dumped = rec.model_dump_json()
        parsed = PanelRecord.model_validate_json(dumped)
        assert parsed.metadata is not None
        assert len(parsed.metadata.cross_figure_visual_links) == 1
        link = parsed.metadata.cross_figure_visual_links[0]
        assert link["target_figure_id"] == "fig_strat_2"
        assert link["target_layer"] == 5
        assert link["confidence"] == 0.95


class TestSchemaJsonRegenerated:
    def test_schema_json_contains_new_field(self):
        if not SCHEMA_PATH.exists():
            pytest.skip("schema dump not generated yet")
        schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
        assert "cross_figure_visual_links" in schema_text


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
