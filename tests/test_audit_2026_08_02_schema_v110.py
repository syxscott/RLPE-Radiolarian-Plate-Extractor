"""Regression tests for audit 2026-08-02 — schema v1.1.0 new fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _minimal_panel_record_kwargs(**overrides):
    """Build the minimum kwargs needed to construct a valid PanelRecord.

    Centralised so each test only varies the field under test. The
    ``paper_id`` / ``figure_id`` / ``panel_id`` / ``species`` /
    ``panel_path`` / ``confidence`` sextet is the documented minimum
    per the schema_models docstring.
    """
    base = {
        "paper_id": "test_paper",
        "figure_id": "fig1",
        "panel_id": "1",
        "species": "Genus species",
        "panel_path": "/tmp/panel.png",
        "confidence": 0.85,
    }
    base.update(overrides)
    return base


class TestSchemaV110:
    def test_schema_version_is_1_1_0(self):
        from rlpe.schema_models import SCHEMA_VERSION

        assert SCHEMA_VERSION == "1.1.0"

    def test_panel_record_accepts_new_fields(self):
        from rlpe.schema_models import PanelRecord

        rec = PanelRecord(
            **_minimal_panel_record_kwargs(
                confidence_interval_low=0.7,
                confidence_interval_high=0.95,
                image_verified=True,
                review_priority=2,
            )
        )
        assert rec.confidence_interval_low == 0.7
        assert rec.confidence_interval_high == 0.95
        assert rec.image_verified is True
        assert rec.review_priority == 2

        dumped = rec.model_dump()
        assert dumped["confidence_interval_low"] == 0.7
        assert dumped["confidence_interval_high"] == 0.95
        assert dumped["image_verified"] is True
        assert dumped["review_priority"] == 2

        # Round-trip via model_validate (the path the
        # ``validate_run_output`` helper uses for JSONL re-load).
        reloaded = PanelRecord.model_validate(dumped)
        assert reloaded.confidence_interval_low == 0.7
        assert reloaded.confidence_interval_high == 0.95
        assert reloaded.image_verified is True
        assert reloaded.review_priority == 2

    def test_panel_record_defaults_for_new_fields(self):
        from rlpe.schema_models import PanelRecord

        rec = PanelRecord(**_minimal_panel_record_kwargs())
        assert rec.confidence_interval_low is None
        assert rec.confidence_interval_high is None
        assert rec.image_verified is False
        assert rec.review_priority == 0

    def test_panel_record_validates_priority_range(self):
        from rlpe.schema_models import PanelRecord

        # 3 is out of [0, 2]
        with pytest.raises(ValidationError):
            PanelRecord(**_minimal_panel_record_kwargs(review_priority=3))
        # -1 is out of [0, 2]
        with pytest.raises(ValidationError):
            PanelRecord(**_minimal_panel_record_kwargs(review_priority=-1))
        # Boundary 0 and 2 must pass
        PanelRecord(**_minimal_panel_record_kwargs(review_priority=0))
        PanelRecord(**_minimal_panel_record_kwargs(review_priority=2))

    def test_panel_record_validates_confidence_interval_range(self):
        from rlpe.schema_models import PanelRecord

        # Above 1.0 must be rejected for both bounds
        with pytest.raises(ValidationError):
            PanelRecord(**_minimal_panel_record_kwargs(confidence_interval_low=1.5))
        with pytest.raises(ValidationError):
            PanelRecord(**_minimal_panel_record_kwargs(confidence_interval_high=2.0))
        # Below 0.0 must be rejected for both bounds
        with pytest.raises(ValidationError):
            PanelRecord(**_minimal_panel_record_kwargs(confidence_interval_low=-0.1))
        with pytest.raises(ValidationError):
            PanelRecord(**_minimal_panel_record_kwargs(confidence_interval_high=-1.0))
        # Boundary 0.0 and 1.0 must pass
        PanelRecord(**_minimal_panel_record_kwargs(confidence_interval_low=0.0))
        PanelRecord(**_minimal_panel_record_kwargs(confidence_interval_high=1.0))
        # None must pass (the "not computed" case)
        PanelRecord(**_minimal_panel_record_kwargs(confidence_interval_low=None))
        PanelRecord(**_minimal_panel_record_kwargs(confidence_interval_high=None))

    def test_v110_schema_file_exists(self):
        schema_path = Path(__file__).resolve().parent.parent / "schemas" / "rlpe-v1.1.0.json"
        assert schema_path.is_file(), f"Expected schema at {schema_path}"
        # Parses as valid JSON
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        # $id reflects the v1.1.0 path
        assert data["$id"].endswith("/rlpe-v1.1.0.json")

    def test_v110_schema_has_new_fields(self):
        schema_path = Path(__file__).resolve().parent.parent / "schemas" / "rlpe-v1.1.0.json"
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        # Pydantic v2 emits model defs under ``$defs``
        defs = data.get("$defs", {})
        assert "PanelRecord" in defs, "PanelRecord def missing from emitted schema"
        props = defs["PanelRecord"]["properties"]
        for field_name in (
            "confidence_interval_low",
            "confidence_interval_high",
            "image_verified",
            "review_priority",
        ):
            assert field_name in props, (
                f"Expected {field_name!r} in PanelRecord properties; got: {sorted(props.keys())}"
            )
        # review_priority carries the ge=0, le=2 bound in the JSON schema
        rp = props["review_priority"]
        assert rp.get("minimum") == 0
        assert rp.get("maximum") == 2
        # confidence_interval_* are Optional[float], so Pydantic emits
        # them as ``anyOf: [number, null]`` with bounds on the
        # number branch. Verify the bounds live on the number branch.
        for ci_field in ("confidence_interval_low", "confidence_interval_high"):
            ci = props[ci_field]
            number_branch = next(
                (b for b in ci.get("anyOf", []) if b.get("type") == "number"),
                None,
            )
            assert number_branch is not None, (
                f"Expected a number branch in {ci_field} anyOf; got: {ci}"
            )
            assert number_branch.get("minimum") == 0.0
            assert number_branch.get("maximum") == 1.0

    def test_converters_panel_record_from_match_forwards_new_fields(self):
        """converters.panel_record_from_match must populate the v1.1.0
        fields from match.metadata so downstream JSONL exports carry
        them without a second pass over the records."""
        from rlpe.converters import panel_record_from_match
        from rlpe.types import MatchResult

        match = MatchResult(
            paper_id="p",
            figure_id="f",
            panel_id="1",
            species="Genus species",
            panel_path="/tmp/panel.png",
            bbox=None,
            confidence=0.9,
            label_text=None,
            caption_snippet=None,
            ocr_text=None,
            metadata={
                "confidence_interval_low": 0.75,
                "confidence_interval_high": 0.99,
                "image_verified": True,
                "review_priority": 1,
            },
            paper_metadata=None,
        )
        rec = panel_record_from_match(match)
        assert rec.confidence_interval_low == 0.75
        assert rec.confidence_interval_high == 0.99
        assert rec.image_verified is True
        assert rec.review_priority == 1

    def test_converters_panel_record_from_match_uses_defaults_when_missing(self):
        """A MatchResult with empty metadata must still produce a valid
        PanelRecord with the v1.1.0 defaults (None / False / 0)."""
        from rlpe.converters import panel_record_from_match
        from rlpe.types import MatchResult

        match = MatchResult(
            paper_id="p",
            figure_id="f",
            panel_id="1",
            species="Genus species",
            panel_path="/tmp/panel.png",
            bbox=None,
            confidence=0.9,
            label_text=None,
            caption_snippet=None,
            ocr_text=None,
            metadata={},
            paper_metadata=None,
        )
        rec = panel_record_from_match(match)
        assert rec.confidence_interval_low is None
        assert rec.confidence_interval_high is None
        assert rec.image_verified is False
        assert rec.review_priority == 0
