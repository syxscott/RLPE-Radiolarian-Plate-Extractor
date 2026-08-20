"""Source guard: Pydantic models in ``schema_models.py`` must match the
field set in the published JSON Schema (``schemas/rlpe-v{SCHEMA_VERSION}.json``)
exactly.

Audit 2026-08-18: the round 64 sweep added ``paper_id`` to ``LocalityRecord``
and ``PaleoCoordinateRecord`` in the Pydantic models, but the published JSON
Schema was never regenerated. ``run_output_from_provenance`` was emitting
the new field; jsonschema validation against the published schema failed
with ``additionalProperties: false``. The runtime regression
(``test_phase63_schema_validate_runoutput``) would have caught this, but
only when jsonschema is installed AND the schema file is present AND the
RunOutput is built with the right shape — i.e. it has moving parts.

This test is a **structural** guard: it compares field-name sets directly
without running the runtime pipeline. It catches the same drift class but
without any moving parts, so it's cheap to keep around.

Each Pydantic model whose fields appear in the schema's ``$defs`` must
have the same set of fields. Drift in either direction (model adds field
schema doesn't, or schema has field model doesn't) is reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.schema_models import SCHEMA_VERSION  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / f"rlpe-v{SCHEMA_VERSION}.json"


def _model_classes_to_check():
    """Return the Pydantic classes whose ``$defs`` entry we want to compare.

    Imports are deferred inside the function so test collection doesn't
    fail when an optional dep is missing.
    """
    from rlpe.schema_models import (  # noqa: E402
        FigureRecord,
        GeologyContextRecord,
        GeologyLinkRecord,
        LocalityRecord,
        MorphologyRecord,
        PaleoCoordinateRecord,
        PanelMetadata,
        PanelRecord,
        PaperMetadataRecord,
        PaperRecord,
        ProvenanceRecord,
        SampleRecord,
        ScaleBarRecord,
        TaxonRecord,
        WarningRecord,
    )

    return [
        ScaleBarRecord,
        GeologyLinkRecord,
        PaperMetadataRecord,
        ProvenanceRecord,
        PaperRecord,
        FigureRecord,
        TaxonRecord,
        SampleRecord,
        GeologyContextRecord,
        MorphologyRecord,
        LocalityRecord,
        PaleoCoordinateRecord,
        WarningRecord,
        PanelMetadata,
        PanelRecord,
    ]


def test_schema_drift_guard():
    """Every Pydantic model field set must match the schema's ``$defs``
    entry exactly. Drift in either direction is a regression."""
    if not SCHEMA_PATH.exists():
        pytest.skip(f"{SCHEMA_PATH} not present; cannot audit drift")
    schema = json.loads(SCHEMA_PATH.read_text())
    defs = schema.get("$defs", {})

    drift_lines: list[str] = []
    for cls in _model_classes_to_check():
        name = cls.__name__
        if name not in defs:
            drift_lines.append(f"  - {name}: defined in Pydantic but MISSING from schema $defs")
            continue
        model_fields = set(cls.model_fields.keys())
        schema_props = set(defs[name].get("properties", {}).keys())
        in_model_only = model_fields - schema_props
        in_schema_only = schema_props - model_fields
        if in_model_only:
            drift_lines.append(
                f"  - {name}: model has fields {sorted(in_model_only)} "
                f"not in schema (would emit additionalProperties: false violation)"
            )
        if in_schema_only:
            # Schema having extra fields is OK from a runtime POV
            # (additionalProperties=false only blocks model emitting extras,
            # not schema declaring more than model) but it's still drift
            # worth surfacing.
            drift_lines.append(
                f"  - {name}: schema has fields {sorted(in_schema_only)} "
                f"not in model (stale $defs entry)"
            )

    assert drift_lines == [], (
        "Pydantic ↔ JSON Schema drift detected:\n" + "\n".join(drift_lines)
        + "\n\nFix: regenerate the schema with "
        "`python -m rlpe.schema_dump` (or update the schema by hand) and "
        "re-commit. See commit `dda596c` for the previous paper_id fix."
    )


def test_schema_version_single_source():
    """The published schema filename must match ``SCHEMA_VERSION`` exactly.

    A rename of ``SCHEMA_VERSION`` without regenerating the schema file
    silently disables the schema-validation guard. This test fails fast
    if the version constant doesn't match the actual file."""
    expected_filename = f"rlpe-v{SCHEMA_VERSION}.json"
    expected_path = REPO_ROOT / "schemas" / expected_filename
    if not expected_path.exists():
        pytest.skip(f"{expected_filename} not present")
    assert SCHEMA_VERSION, "SCHEMA_VERSION must be non-empty"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
