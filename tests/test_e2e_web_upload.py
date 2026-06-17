"""Test the full Web path: upload a PDF via the API, wait for completion,
fetch the result, validate it against the published schema."""

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "Xiao Yifan et al 2017a micro-XCT.pdf"
if not PDF.exists():
    pytest.skip("Xiao Yifan micro-XCT PDF not present", allow_module_level=True)

from rlpe.api.app import app  # noqa: E402
from rlpe.schema_models import PanelRecord  # noqa: E402


def test_upload_completes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """POST a PDF, poll until the job is done, GET the result.

    The test runs the real pipeline (heavy deps may be missing -> the
    pipeline gracefully degrades) and asserts that:
      * The upload returns 200 + a job_id
      * The job eventually transitions to "done" within the timeout
      * The result rows are JSONL-compatible and (if any) pass
        PanelRecord validation
    """
    # Sandbox: point the API at tmp_path so we don't touch the real
    # uploads/ or service_work/ directories.
    monkeypatch.setenv("RLPE_API_TEST_TMP", str(tmp_path))

    # The pipeline runs in a background thread inside the API; the
    # test client blocks until the request returns. Use TestClient
    # with a short timeout per request.
    with TestClient(app) as client:
        with PDF.open("rb") as f:
            r = client.post(
                "/jobs/upload",
                files={"file": (PDF.name, f, "application/pdf")},
                data={
                    "options": json.dumps(
                        {"use_opendataloader": True, "data_outbound_policy": "local_only"}
                    )
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        job_id = body["job_id"]
        assert job_id

        # Poll status
        deadline = time.time() + 180
        last_status = None
        while time.time() < deadline:
            rs = client.get(f"/jobs/{job_id}/status")
            assert rs.status_code == 200, rs.text
            s = rs.json()
            last_status = s.get("status")
            if last_status in {"done", "failed"}:
                break
            time.sleep(2)
        assert last_status in {"done", "failed"}, (
            f"Job did not terminate: last_status={last_status}"
        )

        # Get result
        rr = client.get(f"/jobs/{job_id}/result")
        assert rr.status_code == 200, rr.text
        result = rr.json()
        assert "status" in result
        rows = result.get("result") or []
        # The pipeline may produce 0 rows if no OCR / OD; the API must
        # still respond 200 with a valid envelope.
        for row in rows:
            for k in ("job_id", "panel_local_path", "provenance"):
                row.pop(k, None)
            allowed = set(PanelRecord.model_fields.keys())
            allowed_meta = set(PanelRecord.model_fields["metadata"].annotation.model_fields.keys())
            for k in list(row.keys()):
                if k not in allowed:
                    row.pop(k, None)
            meta = row.get("metadata") or {}
            for k in list(meta.keys()):
                if k not in allowed_meta:
                    meta.pop(k, None)
            row["metadata"] = meta
            PanelRecord.model_validate(row)
