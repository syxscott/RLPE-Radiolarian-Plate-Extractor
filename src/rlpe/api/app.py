from __future__ import annotations

import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"FastAPI dependencies not available: {exc}")

from ..config import PipelineConfig
from ..pipeline import RadiolarianPipeline
from ..utils import ensure_dir


APP_ROOT = Path.cwd()
UPLOAD_DIR = ensure_dir(APP_ROOT / "uploads")
WORK_DIR = ensure_dir(APP_ROOT / "service_work")
RESULT_CACHE: dict[str, dict[str, Any]] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str
    detail: str | None = None


class ReviewCorrection(BaseModel):
    job_id: str
    paper_id: str
    figure_id: str
    panel_path: str | None = None
    corrected_species: str | None = None
    corrected_label: str | None = None
    reviewer: str | None = None


app = FastAPI(title="RLPE API", version="0.2.0")


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

    RESULT_CACHE[job_id] = {"status": "queued", "result": None, "error": None}
    background_tasks.add_task(_run_job, job_id, save_path)
    return JobStatus(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}/status", response_model=JobStatus)
def job_status(job_id: str):
    job = RESULT_CACHE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(job_id=job_id, status=job["status"], detail=job.get("error"))


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = RESULT_CACHE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in {"done", "failed"}:
        raise HTTPException(status_code=202, detail="Job not finished")
    return job


@app.post("/review/correction")
def submit_correction(payload: ReviewCorrection):
    corrections_dir = ensure_dir(WORK_DIR / "corrections")
    target = corrections_dir / f"{payload.job_id}.jsonl"
    row = payload.model_dump()
    with target.open("a", encoding="utf-8") as f:
        import json

        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"status": "ok", "saved_to": str(target)}


def _run_job(job_id: str, pdf_path: Path) -> None:
    RESULT_CACHE[job_id]["status"] = "running"
    try:
        pdf_dir = ensure_dir(WORK_DIR / job_id / "pdfs")
        moved_path = pdf_dir / pdf_path.name
        shutil.move(str(pdf_path), moved_path)

        cfg = PipelineConfig(
            pdf_dir=pdf_dir,
            work_dir=WORK_DIR / job_id,
            output_dir=None,
            save_intermediate=True,
            extra={
                "use_gemma4": False,
            },
        )
        rows = RadiolarianPipeline(cfg).run()
        RESULT_CACHE[job_id]["status"] = "done"
        RESULT_CACHE[job_id]["result"] = rows
    except Exception as exc:
        RESULT_CACHE[job_id]["status"] = "failed"
        RESULT_CACHE[job_id]["error"] = str(exc)
