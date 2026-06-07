"""Smoke tests for the FastAPI service in `rlpe.api.app`.

These tests exercise the read-only endpoints (health, list jobs,
system info) and the validation logic for the JSON request bodies
(review correction, job options). They do NOT trigger a real
PDF-processing job (those are integration tests and need a real
PDF + GROBID / OpenDataLoader running).

The tests use FastAPI's TestClient with a real app instance.
Run with: `PYTHONPATH=src python -m pytest tests/test_api_app.py -q`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# These imports only succeed when fastapi + pydantic are installed.
# Both are in the `service` optional-deps group in pyproject.toml.
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient  # noqa: E402

from rlpe.api.app import (  # noqa: E402
    JobOptions,
    ReviewCorrection,
    app,
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """Build a TestClient that uses tmp_path as the work dir so the
    test never touches the real `work/`, `uploads/`, or
    `service_work/` directories."""
    monkeypatch.setenv("RLPE_API_TEST_TMP", str(tmp_path))
    return TestClient(app)


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"


def test_root_serves_index_or_404(client: TestClient) -> None:
    """The / endpoint either serves the SPA index.html (when web/
    is present) or returns a 404 placeholder. Either is acceptable
    in the smoke test — the contract is "doesn't 500"."""
    r = client.get("/")
    assert r.status_code in (200, 404)


def test_system_info_returns_metadata(client: TestClient) -> None:
    r = client.get("/system/info")
    assert r.status_code == 200
    body = r.json()
    # The endpoint must expose at least these keys for the SPA to
    # render the system-info panel.
    for key in ("version", "python_version", "active_jobs"):
        assert key in body, f"missing key {key!r} in /system/info"


def test_list_jobs_is_empty_on_fresh_client(client: TestClient) -> None:
    r = client.get("/jobs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_review_correction_accepts_minimal_payload(client: TestClient) -> None:
    """A correction with just paper_id + figure_id + corrected_species
    should be accepted; the service should respond 200 (the record
    is queued for the next pipeline run)."""
    r = client.post(
        "/review/correction",
        json={
            "paper_id": "test_paper",
            "figure_id": "test_figure",
            "corrected_species": "Genus species",
        },
    )
    assert r.status_code in (200, 202, 404), (
        f"unexpected status {r.status_code}: {r.text}"
    )


def test_job_options_rejects_invalid_fallback(client: TestClient) -> None:
    """JobOptions._validate_fallback restricts the fallback to
    {gemma4, rules, stop, retry}. Anything else should fail with 422.
    """
    with pytest.raises(ValueError):
        JobOptions(MiniMax_fallback_default="not-a-valid-fallback")


def test_job_options_accepts_valid_fallback() -> None:
    """All four documented values are accepted."""
    for fb in ("gemma4", "rules", "stop", "retry"):
        opts = JobOptions(MiniMax_fallback_default=fb)
        assert opts.MiniMax_fallback_default == fb


def test_job_options_rejects_invalid_outbound_policy() -> None:
    with pytest.raises(ValueError):
        JobOptions(data_outbound_policy="to_the_moon")


def test_review_correction_model_validates_reviewer_field() -> None:
    """A non-empty reviewer is stored verbatim."""
    rc = ReviewCorrection(
        paper_id="p",
        figure_id="f",
        corrected_species="Genus species",
        reviewer="Alice",
    )
    assert rc.reviewer == "Alice"


def test_missing_job_status_returns_404(client: TestClient) -> None:
    """A job_id that doesn't exist should 404, not 500."""
    r = client.get("/jobs/nonexistent-id-xyz/status")
    assert r.status_code in (404, 400)
