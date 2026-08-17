"""Phase 64 Plan B Task 2: PanelMetadata.figure_schematic_data field.

The Phase 64 Plan B M3 pipeline writes the
``extract_schematic()`` output JSON into a new per-panel field
``PanelMetadata.figure_schematic_data``. The field is the storage
location for the conceptual-figure extraction result on
schematic / diagram / reconstruction / phylogenetic figures.

The shape stored matches the M3 prompt contract:

  {
    "figure_type": "schematic" | "diagram" | "reconstruction" | "phylogenetic",
    "text_elements": [
        {"text": str, "type": str, "confidence": float},
        ...
    ],
    "relationships": [
        {"from": str, "to": str, "label": str},
        ...
    ],
    "extracted_facts": {
        "ages_mentioned": [str, ...],
        "geographic_names": [str, ...],
        "taxa_mentioned": [str, ...],
    },
    "confidence": float,
  }

We store it as a free-form ``dict[str, Any]`` (not a nested
Pydantic model) so the downstream export path can iterate the
same shape without re-marshalling.

This test file locks:
  1. The field exists on PanelMetadata and defaults to None.
  2. The field accepts the M3 prompt contract shape.
  3. The published JSON schema (`schemas/rlpe-v1.0.0.json`) lists
     the new field so downstream consumers see it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rlpe.schema_models import PanelMetadata

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_PATH = _REPO_ROOT / "schemas" / "rlpe-v1.0.0.json"


class TestPanelMetadataSchematicField:
    """Phase 64 Plan B Task 2: PanelMetadata.figure_schematic_data."""

    def test_field_exists_with_none_default(self) -> None:
        """Default PanelMetadata has figure_schematic_data == None
        so existing JSONL records (without the new field) remain
        valid."""
        pm = PanelMetadata()
        assert pm.figure_schematic_data is None

    def test_field_accepts_m3_prompt_contract_shape(self) -> None:
        """A representative M3 prompt-contract JSON is round-tripped
        through PanelMetadata without data loss."""
        schematic = {
            "figure_type": "schematic",
            "text_elements": [
                {"text": "Late Triassic", "type": "age", "confidence": 0.98},
                {"text": "Tethys Ocean", "type": "geographic", "confidence": 0.95},
            ],
            "relationships": [
                {"from": "box1", "to": "box2", "label": "evolved into"},
            ],
            "extracted_facts": {
                "ages_mentioned": ["Late Triassic", "Carnian"],
                "geographic_names": ["Tethys", "Panthalassa"],
                "taxa_mentioned": ["Genus species"],
            },
            "confidence": 0.92,
        }
        pm = PanelMetadata(figure_schematic_data=schematic)
        assert pm.figure_schematic_data == schematic

    def test_field_appears_in_published_json_schema(self) -> None:
        """The published JSON Schema file must declare the new
        field so downstream consumers (DwC-A exporters, dashboard
        validators, etc.) can introspect it."""
        assert _SCHEMA_PATH.exists(), (
            f"Schema file missing: {_SCHEMA_PATH}. Run "
            "`python -m rlpe.schema_dump --out schemas/rlpe-v1.0.0.json`."
        )
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        # Find the PanelMetadata properties block.
        panel_metadata = None
        for defn in schema.get("$defs", {}).values():
            if defn.get("title") == "PanelMetadata":
                panel_metadata = defn
                break
        assert panel_metadata is not None, "PanelMetadata $defs entry missing"
        props = panel_metadata.get("properties", {})
        assert "figure_schematic_data" in props, (
            f"figure_schematic_data missing from published schema; "
            f"properties are: {list(props.keys())}"
        )
        # The field is free-form dict — schema should reflect that.
        fsd = props["figure_schematic_data"]
        # Either nullable (anyOf with null) or with a default of null.
        assert "anyOf" in fsd or fsd.get("type") == "object" or "default" in fsd, (
            f"Unexpected schema for figure_schematic_data: {fsd}"
        )

    def test_field_roundtrips_through_json(self) -> None:
        """The field round-trips through model_dump / model_validate
        so JSONL exports remain faithful."""
        schematic = {
            "figure_type": "phylogenetic",
            "text_elements": [
                {"text": "Nassellaria", "type": "taxon", "confidence": 0.99},
            ],
            "relationships": [
                {"from": "node_a", "to": "node_b", "label": "sister to"},
            ],
            "extracted_facts": {
                "ages_mentioned": [],
                "geographic_names": [],
                "taxa_mentioned": ["Nassellaria"],
            },
            "confidence": 0.88,
        }
        pm = PanelMetadata(figure_schematic_data=schematic)
        dumped = pm.model_dump()
        assert dumped["figure_schematic_data"] == schematic
        # Re-validate.
        pm2 = PanelMetadata.model_validate(dumped)
        assert pm2.figure_schematic_data == schematic

    def test_field_defaults_omitted_from_minimal_dump(self) -> None:
        """A PanelMetadata that never set figure_schematic_data still
        dumps cleanly. ``exclude_none`` keeps backward-compat JSONL
        records (no new field) clean."""
        pm = PanelMetadata()
        dumped = pm.model_dump(exclude_none=True)
        assert "figure_schematic_data" not in dumped
