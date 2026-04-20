from __future__ import annotations

import json
import shutil
import sys
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"FastAPI dependencies not available: {exc}")

from ..config import PipelineConfig
from ..pipeline import RadiolarianPipeline
from ..utils import ensure_dir


APP_ROOT = Path.cwd()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ensure_dir(APP_ROOT / "static")
UPLOAD_DIR = ensure_dir(APP_ROOT / "uploads")
WORK_DIR = ensure_dir(APP_ROOT / "service_work")
RESULT_CACHE: dict[str, dict[str, Any]] = {}

WEB_DIR: Path | None = None
for _candidate in (APP_ROOT / "web", PROJECT_ROOT / "web"):
    if _candidate.exists() and (_candidate / "index.html").exists():
        WEB_DIR = _candidate
        break


class JobStatus(BaseModel):
    job_id: str
    status: str
    detail: str | None = None
    created_at: str | None = None
    filename: str | None = None
    progress: int = 0


class ReviewCorrection(BaseModel):
    paper_id: str
    figure_id: str
    panel_path: str | None = None
    corrected_species: str | None = None
    corrected_label: str | None = None
    reviewer: str | None = None
    notes: str | None = None


class ResultRecord(BaseModel):
    job_id: str | None = None
    paper_id: str
    figure_id: str
    panel_id: str | None = None
    species: str | None = None
    confidence: float
    panel_path: str | None = None
    label_text: str | None = None


app = FastAPI(
    title="RLPE API - Radiolarian Plate Extractor",
    version="0.2.0",
    description="Web API for radiolarian figure extraction from PDF literature"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR is not None:
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


@app.get("/")
def root():
    if WEB_DIR is not None:
        index_path = WEB_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
    return {
        "status": "ok",
        "service": "rlpe-api",
        "docs": "/docs",
        "web": "/web"
    }


@app.get("/css/{file_path:path}")
def web_css(file_path: str):
    if WEB_DIR is None:
        raise HTTPException(status_code=404, detail="Web assets not found")
    target = WEB_DIR / "css" / file_path
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target)


@app.get("/js/{file_path:path}")
def web_js(file_path: str):
    if WEB_DIR is None:
        raise HTTPException(status_code=404, detail="Web assets not found")
    target = WEB_DIR / "js" / file_path
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target)


@app.get("/favicon.ico")
def favicon():
    if WEB_DIR is None:
        return Response(status_code=204)
    target = WEB_DIR / "favicon.ico"
    if not target.exists() or not target.is_file():
        return Response(status_code=204)
    return FileResponse(target)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/jobs/upload", response_model=JobStatus)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    now = datetime.now().isoformat()
    RESULT_CACHE[job_id] = {
        "status": "queued",
        "result": None,
        "error": None,
        "detail": None,
        "created_at": now,
        "filename": file.filename,
        "progress": 0
    }
    background_tasks.add_task(_run_job, job_id, save_path)
    return JobStatus(
        job_id=job_id,
        status="queued",
        created_at=now,
        filename=file.filename
    )


@app.get("/jobs/{job_id}/status", response_model=JobStatus)
def job_status(job_id: str):
    job = RESULT_CACHE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        detail=job.get("error") or job.get("detail"),
        created_at=job.get("created_at"),
        filename=job.get("filename"),
        progress=job.get("progress", 0)
    )


@app.get("/jobs")
def list_jobs() -> list[JobStatus]:
    """List all jobs with their current status."""
    return [
        JobStatus(
            job_id=job_id,
            status=job["status"],
            detail=job.get("error") or job.get("detail"),
            created_at=job.get("created_at"),
            filename=job.get("filename"),
            progress=job.get("progress", 0)
        )
        for job_id, job in RESULT_CACHE.items()
    ]


@app.get("/jobs/{job_id}/files/{file_path:path}")
def job_file(job_id: str, file_path: str):
    job_root = (WORK_DIR / job_id).resolve()
    if not job_root.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    target = (job_root / file_path).resolve()
    try:
        target.relative_to(job_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = RESULT_CACHE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in {"done", "failed"}:
        raise HTTPException(status_code=202, detail="Job not finished")
    return job


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a pending or running job."""
    job = RESULT_CACHE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in {"queued", "running"}:
        raise HTTPException(status_code=400, detail="Cannot cancel a finished job")
    job["status"] = "cancelled"
    return {"status": "cancelled", "job_id": job_id}


@app.post("/review/correction")
def submit_correction(payload: ReviewCorrection):
    corrections_dir = ensure_dir(WORK_DIR / "corrections")
    target = corrections_dir / "corrections.jsonl"
    row = {
        **payload.model_dump(),
        "timestamp": datetime.now().isoformat()
    }
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"status": "ok", "saved_to": str(target)}


@app.get("/results")
def get_results() -> list[ResultRecord]:
    """Get all accumulated results from completed jobs."""
    results = []
    for job_id, job in RESULT_CACHE.items():
        if job["status"] == "done" and job.get("result"):
            for row in job["result"]:
                results.append(ResultRecord(
                    job_id=job_id,
                    **row
                ))
    return results


@app.get("/system/info")
def system_info() -> dict[str, Any]:
    """Get system and configuration information."""
    return {
        "version": "0.2.0",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "grobid_url": "http://localhost:8070",
        "active_jobs": sum(1 for j in RESULT_CACHE.values() if j["status"] in {"queued", "running"}),
        "total_jobs": len(RESULT_CACHE),
        "completed_jobs": sum(1 for j in RESULT_CACHE.values() if j["status"] == "done"),
        "failed_jobs": sum(1 for j in RESULT_CACHE.values() if j["status"] == "failed"),
    }


def _run_job(job_id: str, pdf_path: Path) -> None:
    RESULT_CACHE[job_id]["status"] = "running"
    RESULT_CACHE[job_id]["progress"] = 10
    try:
        ensure_dir(APP_ROOT / "static")
        pdf_dir = ensure_dir(WORK_DIR / job_id / "pdfs")
        moved_path = pdf_dir / pdf_path.name
        shutil.move(str(pdf_path), moved_path)
        RESULT_CACHE[job_id]["progress"] = 20

        cfg = PipelineConfig(
            pdf_dir=pdf_dir,
            work_dir=WORK_DIR / job_id,
            output_dir=None,
            save_intermediate=True,
            extra={
                "use_gemma4": False,
            },
        )
        RESULT_CACHE[job_id]["progress"] = 30
        rows = RadiolarianPipeline(cfg).run()
        normalized_rows: list[dict[str, Any]] = []
        job_root = (WORK_DIR / job_id).resolve()
        for row in rows:
            normalized = asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row)
            panel_path = normalized.get("panel_path")
            if panel_path:
                panel_abs = Path(panel_path).resolve()
                try:
                    rel = panel_abs.relative_to(job_root)
                    normalized["panel_local_path"] = str(panel_abs)
                    normalized["panel_path"] = f"/jobs/{job_id}/files/{rel.as_posix()}"
                except ValueError:
                    # Keep original path when file is outside this job workspace.
                    pass
            normalized_rows.append(normalized)
        RESULT_CACHE[job_id]["progress"] = 90
        RESULT_CACHE[job_id]["status"] = "done"
        RESULT_CACHE[job_id]["result"] = normalized_rows
        if normalized_rows:
            RESULT_CACHE[job_id]["detail"] = f"Generated {len(normalized_rows)} result rows"
        else:
            RESULT_CACHE[job_id]["detail"] = "Pipeline finished but no panels/matches were produced"
        RESULT_CACHE[job_id]["progress"] = 100
    except Exception as exc:
        RESULT_CACHE[job_id]["status"] = "failed"
        tb = traceback.format_exc(limit=8)
        err = str(exc)
        if "object has no attribute 'route'" in err and "Starlette" in err:
            err = (
                f"{err}. Possible PyMuPDF/fitz package conflict. "
                "Install `pymupdf` and uninstall non-PyMuPDF `fitz`."
            )
        RESULT_CACHE[job_id]["error"] = err
        RESULT_CACHE[job_id]["error_traceback"] = tb
        RESULT_CACHE[job_id]["detail"] = "Pipeline execution failed"
        RESULT_CACHE[job_id]["progress"] = 0
