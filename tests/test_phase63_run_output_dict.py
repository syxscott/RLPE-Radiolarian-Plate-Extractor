"""Tests for Phase 63 Plan 6 — Bug 6.1: ``run_output_from_provenance``
must accept either a :class:`ProvenanceRecord` or a plain ``dict``.

Before: signature hard-coded ``ProvenanceRecord``; the GUI / web
``Job.rows`` code path that builds a RunOutput dict from a partial
provenance dict crashed with::

    TypeError: argument 'provenance': missing 1 required keyword argument

After: callers can pass either type. A ``dict`` is coerced via
``ProvenanceRecord.model_validate`` (with stub backfill on missing
fields) so the GUI's truncated provenance (``job_id`` + ``source``
only) still produces a RunOutput dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.converters import run_output_from_provenance  # noqa: E402
from rlpe.schema_models import ProvenanceRecord  # noqa: E402
from rlpe.types import MatchResult  # noqa: E402


def _match(species: str = "Genus species") -> MatchResult:
    return MatchResult(
        paper_id="p1",
        figure_id="fig1",
        panel_id="1",
        species=species,
        panel_path="/tmp/p1.png",
        bbox=[10, 20, 100, 200],
        confidence=0.9,
        label_text=None,
        caption_snippet=None,
        ocr_text=None,
        metadata={},
    )


def _valid_prov_dict() -> dict:
    """All required fields populated; mirrors provenance.build_provenance().to_dict()."""
    return {
        "pipeline_version": "1.2.0",
        "schema_version": "1.0.0",
        "git_commit": "abc1234",
        "git_dirty": False,
        "config_snapshot": {},
        "input_sha256": {},
        "timestamp_utc": "2026-07-20T00:00:00Z",
        "host": "linux/test/localhost",
        "python_version": "3.12.0",
    }


def test_run_output_from_provenance_accepts_provenance_record():
    """The legacy signature (typed ``ProvenanceRecord``) keeps working."""
    prov = ProvenanceRecord(**_valid_prov_dict())
    out = run_output_from_provenance(prov, [_match()])
    assert out["provenance"]["pipeline_version"] == "1.2.0"
    assert len(out["panels"]) == 1


def test_run_output_from_provenance_accepts_dict_full():
    """A complete-prov provenance dict is normalised to a ProvenanceRecord."""
    out = run_output_from_provenance(_valid_prov_dict(), [_match()])
    # The dict roundtrips; the resulting RunOutput dict carries the same fields.
    assert out["provenance"]["pipeline_version"] == "1.2.0"
    assert out["provenance"]["git_commit"] == "abc1234"
    assert len(out["panels"]) == 1
    assert out["panels"][0]["species"] == "Genus species"


def test_run_output_from_provenance_accepts_dict_partial():
    """A partial-prov dict (just GUI fields) is backfilled to a complete stamp."""
    gui_payload = {
        "job_id": "job-abc",
        "source": "gui",
    }
    out = run_output_from_provenance(gui_payload, [_match()])
    # All required ProvenanceRecord keys must end up set:
    for key in (
        "pipeline_version",
        "schema_version",
        "git_commit",
        "git_dirty",
        "timestamp_utc",
        "host",
        "python_version",
        "config_snapshot",
        "input_sha256",
    ):
        assert key in out["provenance"], f"missing provenance key: {key}"
    assert len(out["panels"]) == 1


def test_run_output_from_provenance_accepts_none_matches():
    """``matches=None`` is treated as an empty list."""
    prov = ProvenanceRecord(**_valid_prov_dict())
    out = run_output_from_provenance(prov, None)
    assert out["panels"] == []
    # Provenance is still attached.
    assert out["provenance"]["pipeline_version"] == "1.2.0"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
