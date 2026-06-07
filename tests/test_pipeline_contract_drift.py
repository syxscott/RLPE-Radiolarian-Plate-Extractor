"""Contract-drift guard: validate real pipeline output against the published schema.

Why: The 46ef988 commit fixed a case where adding a runtime metadata field
(reassigned_from_figure / reassigned_reason in commit 5f50d67) wasn't
mirrored in PanelMetadata, so 15/65 smoke-test records were rejected by
PanelRecord.model_validate_json. Synthetic unit tests in
``test_schema_models.py`` only exercise the schema in isolation — they
miss drift introduced by the actual pipeline. This test runs against real
pipeline output to close that gap.

Strategy: if a recent smoke test run is present at
``work/papers_smoke/output/manifests/matches.jsonl``, validate every
record. The smoke test fixture is git-ignored, so this test is a
no-op on a fresh checkout. When run locally after a real pipeline
run, it surfaces schema drift as a red build.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlpe.schema_models import PanelRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_OUTPUT = REPO_ROOT / "work" / "papers_smoke" / "output" / "manifests" / "matches.jsonl"


def _smoke_records() -> list[str]:
    if not SMOKE_OUTPUT.exists():
        return []
    with SMOKE_OUTPUT.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class TestPipelineContract:
    def test_every_record_validates_against_panelrecord(self):
        records = _smoke_records()
        if not records:
            pytest.skip(
                f"No smoke test output at {SMOKE_OUTPUT}. "
                "Run `python -m rlpe.cli --pdf-dir work/papers_smoke "
                "--work-dir work/papers_smoke --use-opendataloader` first."
            )

        failures: list[tuple[int, str]] = []
        for i, line in enumerate(records):
            try:
                PanelRecord.model_validate_json(line)
            except Exception as e:
                failures.append((i, str(e).splitlines()[0][:240]))

        assert not failures, (
            f"{len(failures)}/{len(records)} smoke-test records failed "
            f"PanelRecord validation. First failures:\n"
            + "\n".join(f"  [{i}] {msg}" for i, msg in failures[:5])
        )

    def test_metadata_field_set_is_subset_of_panelmetadata(self):
        """If runtime adds a new metadata field, this test fails until
        the field is also added to PanelMetadata. Acts as a fast-fail
        guard for the same class of bug 46ef988 fixed."""
        records = _smoke_records()
        if not records:
            pytest.skip("No smoke test output to inspect")

        # Discover runtime metadata keys from real pipeline output
        runtime_keys: set[str] = set()
        for line in records:
            d = json.loads(line)
            runtime_keys.update((d.get("metadata") or {}).keys())

        # Use the model class (not instance) — instance access is deprecated
        from rlpe.schema_models import PanelMetadata
        allowed = set(PanelMetadata.model_fields.keys())

        unknown = runtime_keys - allowed
        assert not unknown, (
            f"Runtime metadata fields {sorted(unknown)} are not in "
            f"PanelMetadata. Add them to schema_models.PanelMetadata and "
            f"re-run `python -m rlpe.schema_dump`."
        )
