"""Tests that the published JSON Schema is in sync with the code.

These guard against drift: if anyone changes ``rlpe.schema_models``
without re-running ``python -m rlpe.schema_dump``, the schema file
will not match what Pydantic would emit today, and this test fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlpe.provenance.stamp import build_provenance
from rlpe.schema_models import (
    SCHEMA_VERSION,
    ProvenanceRecord,
    RunOutput,
    emit_json_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / f"rlpe-v{SCHEMA_VERSION}.json"

# Optional: jsonschema lets us validate the published file end-to-end.
# If it's not installed we still run the sync test, which is the more
# important guard.
try:
    import jsonschema  # type: ignore[import-untyped]
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


def test_schema_file_exists():
    assert SCHEMA_PATH.exists(), (
        f"Schema not found at {SCHEMA_PATH}. "
        f"Run: PYTHONPATH=src python -m rlpe.schema_dump"
    )


def test_schema_file_is_in_sync():
    """The published schema must match the live Pydantic emission."""
    if not SCHEMA_PATH.exists():
        pytest.skip(f"{SCHEMA_PATH} not yet emitted; run schema_dump first")
    # Re-emit to a temp file and compare
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tmp = Path(f.name)
    try:
        emit_json_schema(tmp)
        a = json.loads(SCHEMA_PATH.read_text())
        b = json.loads(tmp.read_text())
        assert a == b, "Published schema is out of sync. Run schema_dump."
    finally:
        if tmp.exists():
            tmp.unlink()


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_run_output_validates_against_published_schema():
    """End-to-end: a fresh RunOutput validates against the published file."""
    if not SCHEMA_PATH.exists():
        pytest.skip(f"{SCHEMA_PATH} not yet emitted")
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    prov = ProvenanceRecord(**build_provenance().to_dict())
    out = RunOutput(provenance=prov, panels=[])
    payload = out.model_dump()
    errors = list(validator.iter_errors(payload))
    assert errors == [], f"RunOutput fails published schema: {errors}"
