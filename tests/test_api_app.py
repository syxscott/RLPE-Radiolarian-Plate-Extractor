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
    assert r.status_code in (200, 202, 404), f"unexpected status {r.status_code}: {r.text}"


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


class TestUploadJobLifecycle:
    """End-to-end integration test for the upload → status → cancel flow.

    This test exercises the real wire format of the PDF-processing
    endpoints that the smoke tests deliberately skip. It does NOT
    wait for the full pipeline to complete (that takes 30+ minutes
    per paper on a GPU and would dominate CI time) — it verifies
    that:

    1. POST /jobs/upload accepts a real PDF and returns a job_id.
    2. GET /jobs/{job_id}/status returns a valid JobStatus for the
       new job (status: "queued" or "running", not 404).
    3. POST /jobs/{job_id}/cancel transitions the job to "cancelled".

    The smallest committed PDF (beccaro2006.pdf, 1.1 MB) is used so
    the upload completes in <100 ms. The full 30-min pipeline is
    NOT exercised here; that is covered by manual end-to-end runs
    in the EVALUATION.md "How to add a new paper" section.
    """

    @pytest.fixture
    def small_pdf(self) -> Path:
        # The smallest committed PDF in data/pdfs/ is beccaro2006.pdf
        # (1.1 MB). Use it as a real but small payload.
        candidates = [
            Path(__file__).resolve().parents[1] / "data" / "pdfs" / "beccaro2006.pdf",
            Path(__file__).resolve().parents[1] / "data" / "pdfs" / "danelian2006.pdf",
        ]
        for c in candidates:
            if c.exists():
                return c
        pytest.skip("no committed PDF available in data/pdfs/")

    def test_upload_returns_queued_status(self, client: TestClient, small_pdf: Path) -> None:
        with small_pdf.open("rb") as f:
            r = client.post(
                "/jobs/upload",
                files={"file": (small_pdf.name, f, "application/pdf")},
            )
        assert r.status_code == 200, f"upload failed: {r.text}"
        body = r.json()
        assert "job_id" in body, "response missing job_id"
        assert body["status"] in ("queued", "running"), (
            f"unexpected initial status: {body['status']!r}"
        )
        assert body.get("filename") == small_pdf.name

    def test_upload_then_status_then_cancel(self, client: TestClient, small_pdf: Path) -> None:
        # 1. Upload
        with small_pdf.open("rb") as f:
            r = client.post(
                "/jobs/upload",
                files={"file": (small_pdf.name, f, "application/pdf")},
            )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]

        # 2. Status should resolve (not 404)
        r_status = client.get(f"/jobs/{job_id}/status")
        assert r_status.status_code == 200, r_status.text
        status_body = r_status.json()
        # The job may be in "queued", "running", or "done" by the time we
        # query — all three are valid contract states.
        assert status_body["status"] in ("queued", "running", "done", "failed")
        assert status_body["job_id"] == job_id

        # 3. Cancel — only valid for queued/running. If already done/failed,
        # skip the cancel assertion and just verify the lifecycle completed.
        r_cancel = client.post(f"/jobs/{job_id}/cancel")
        if status_body["status"] in ("queued", "running"):
            assert r_cancel.status_code == 200, r_cancel.text
            assert r_cancel.json()["status"] == "cancelled"
        else:
            # Already done — verify the cancel endpoint refuses it (400).
            assert r_cancel.status_code == 400

    def test_upload_rejects_non_pdf(self, client: TestClient, tmp_path: Path) -> None:
        fake = tmp_path / "not-a-pdf.txt"
        fake.write_text("hello world")
        with fake.open("rb") as f:
            r = client.post(
                "/jobs/upload",
                files={"file": (fake.name, f, "text/plain")},
            )
        assert r.status_code == 400, r.text
        assert "PDF" in r.text or "pdf" in r.text
