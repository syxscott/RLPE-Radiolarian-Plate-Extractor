"""Phase 66 Plan C.5 — export visual links tests.

Visual-coordinate cross-reference links must reach the JSONL / xlsx /
DwC-A export chain so operators can audit Phase C's precision
refinements alongside Phase A's text-only links.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rlpe.converters import panel_metadata_from_match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_match(metadata: dict[str, Any]):
    """Build a tiny stand-in for ``MatchResult`` with just the fields
    ``panel_metadata_from_match`` reads."""
    from types import SimpleNamespace

    return SimpleNamespace(metadata=metadata)


# ---------------------------------------------------------------------------
# Converters tests
# ---------------------------------------------------------------------------


class TestConvertersForwardVisualLinks:
    def test_cross_figure_visual_links_round_trip(self):
        """panel_metadata_from_match must forward cross_figure_visual_links
        onto the exported PanelMetadata."""
        match = _make_match({
            "cross_figure_visual_links": [
                {
                    "target_figure_id": "strat1",
                    "target_layer": 3,
                    "target_age": "Late Triassic",
                    "target_formation": "Scaglia Fm",
                    "confidence": 0.9,
                    "source": "m3_visual",
                }
            ]
        })
        pm = panel_metadata_from_match(match)
        assert len(pm.cross_figure_visual_links) == 1
        entry = pm.cross_figure_visual_links[0]
        assert entry["target_figure_id"] == "strat1"
        assert entry["confidence"] == 0.9

    def test_empty_visual_links_default(self):
        match = _make_match({})
        pm = panel_metadata_from_match(match)
        assert pm.cross_figure_visual_links == []

    def test_multiple_visual_links_round_trip(self):
        match = _make_match({
            "cross_figure_visual_links": [
                {"target_figure_id": "strat1", "target_layer": 1,
                 "target_age": "Late Cretaceous", "target_formation": "Scaglia",
                 "confidence": 0.88, "source": "m3_visual"},
                {"target_figure_id": "strat1", "target_layer": 2,
                 "target_age": "Late Cretaceous", "target_formation": "Scaglia",
                 "confidence": 0.81, "source": "m3_visual"},
            ]
        })
        pm = panel_metadata_from_match(match)
        assert len(pm.cross_figure_visual_links) == 2


# ---------------------------------------------------------------------------
# Archive (DwC-A) dynamicProperties tests
# ---------------------------------------------------------------------------


class TestArchiveDynamicPropertiesForVisualLinks:
    def test_visual_links_appear_in_dynamic_properties(self):
        """The DwC-A exporter should include cross_figure_visual_links
        in the dynamicProperties JSON when present."""
        from rlpe.exporters.archive import _merged_dynamic_properties

        meta = {
            "figure_schematic_data": None,
            "link_source": "locality_match",
            "link_confidence": 0.7,
            "link_figure_id": "strat1",
            "cross_figure_visual_links": [
                {
                    "target_figure_id": "strat1",
                    "target_layer": 3,
                    "target_age": "Late Triassic",
                    "target_formation": "Scaglia Fm",
                    "confidence": 0.9,
                    "source": "m3_visual",
                }
            ],
        }
        # Build a fake metadata object with getattr support.
        from types import SimpleNamespace
        meta_obj = SimpleNamespace(**meta)

        blob = _merged_dynamic_properties(meta_obj)
        assert blob
        parsed = json.loads(blob)
        assert "cross_figure_visual_links" in parsed
        links = parsed["cross_figure_visual_links"]
        assert len(links) == 1
        assert links[0]["target_figure_id"] == "strat1"
        assert links[0]["confidence"] == 0.9
        # The Phase A linker block must still be present.
        assert parsed.get("cross_figure_link", {}).get("source") == "locality_match"

    def test_no_visual_links_returns_normal_payload(self):
        from rlpe.exporters.archive import _merged_dynamic_properties
        from types import SimpleNamespace

        meta_obj = SimpleNamespace(
            figure_schematic_data=None,
            link_source="sample_match",
            link_confidence=1.0,
            link_figure_id="strat1",
            cross_figure_visual_links=[],
        )
        blob = _merged_dynamic_properties(meta_obj)
        assert blob
        parsed = json.loads(blob)
        # No visual_links key added when list is empty.
        assert "cross_figure_visual_links" not in parsed
        # Linker block still there.
        assert parsed["cross_figure_link"]["source"] == "sample_match"

    def test_visual_links_alone_with_no_linker(self):
        """Defensive: visual links present even if Phase A linker
        didn't fire (e.g. linker disabled). Still serialise them."""
        from rlpe.exporters.archive import _merged_dynamic_properties
        from types import SimpleNamespace

        meta_obj = SimpleNamespace(
            figure_schematic_data=None,
            link_source=None,
            link_confidence=0.0,
            link_figure_id=None,
            cross_figure_visual_links=[
                {"target_figure_id": "strat1", "target_layer": 5,
                 "target_age": "Late Triassic", "target_formation": "Scaglia",
                 "confidence": 0.92, "source": "m3_visual"}
            ],
        )
        blob = _merged_dynamic_properties(meta_obj)
        assert blob
        parsed = json.loads(blob)
        assert "cross_figure_visual_links" in parsed
        assert len(parsed["cross_figure_visual_links"]) == 1
        # No Phase A linker block present.
        assert "cross_figure_link" not in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-q"])