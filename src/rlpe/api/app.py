from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("rlpe.api")

try:
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"FastAPI dependencies not available: {exc}")

from ..config import PipelineConfig
from ..pipeline import RadiolarianPipeline
from ..utils import ensure_dir

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Use project root so dirs are predictable regardless of launch directory.
APP_ROOT = PROJECT_ROOT
ensure_dir(APP_ROOT / "static")
UPLOAD_DIR = ensure_dir(APP_ROOT / "uploads")
WORK_DIR = ensure_dir(APP_ROOT / "service_work")
MAX_UPLOAD_SIZE_MB = 256
RESULT_CACHE: dict[str, dict[str, Any]] = {}
# All compound writes to ``RESULT_CACHE`` and ``FALLBACK_PENDING`` go
# through this lock. The GIL makes individual dict operations atomic,
# but the cancel / progress / heartbeat / fallback-popup paths all
# do read-check-write sequences on the same per-job dict; without
# the lock, a status flip from cancel can race a progress callback
# and the job can end up in a state the API can't reason about
# (e.g. status=cancelled but progress=80%). Single lock keeps the
# invariants simple — the job-state machine is small enough that
# contention isn't a real cost.
RESULT_LOCK = threading.Lock()

# Cached GROBID URL — read from the GROBID_URL env var once at
# module load. The previous version re-read the env var on every
# /system/info request, which is wasteful and also has a subtle
# correctness issue: if a user updates the env var mid-session, the
# pipeline keeps using the old URL. Caching at startup makes the
# behaviour predictable. The fallback matches PipelineConfig's
# default ("http://localhost:8070").
GROBID_URL: str = os.environ.get("GROBID_URL") or "http://localhost:8070"

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
    stage: str | None = None  # human-readable "what is the pipeline doing right now"
    elapsed_sec: int | None = None  # how long the job has been running


class ReviewCorrection(BaseModel):
    # ``extra="forbid"`` makes the API surface honest: if the
    # frontend sends a typo (``panelpath`` instead of ``panel_path``),
    # the user gets a 400 instead of a silently-dropped field that
    # would have produced a no-op review entry. Pydantic v2's default
    # is ``extra="ignore"`` which silently dropped reviewer notes.
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    figure_id: str
    panel_path: str | None = None
    corrected_species: str | None = None
    corrected_label: str | None = None
    reviewer: str | None = None
    notes: str | None = None


class ResultRecord(BaseModel):
    # ``extra="forbid"`` surfaces new pipeline fields (added to the
    # internal row dict but not yet declared here) as a 500/422
    # response rather than silently dropping them from the API
    # response. The result is an honest contract: anything the UI
    # shows must be in the schema. If a new field needs to be
    # exposed, it should be added to this model intentionally,
    # not slipped in via the ``**row`` spread at the call site.
    model_config = ConfigDict(extra="forbid")
    job_id: str | None = None
    paper_id: str
    figure_id: str
    panel_id: str | None = None
    species: str | None = None
    confidence: float
    panel_path: str | None = None
    panel_local_path: str | None = None
    label_text: str | None = None
    bbox: list[int] | None = None
    caption_snippet: str | None = None
    ocr_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    paper_metadata: dict[str, Any] | None = None


class JobOptions(BaseModel):
    """Per-job overrides for the pipeline configuration.

    Validated server-side; invalid values are rejected with HTTP 400.
    """
    use_gemma4: bool = False
    llm_backend: str | None = None  # "transformers" | "ollama" | "llamacpp" | "MiniMax"
    gemma_conf_threshold: float = 0.70
    # Local LLM backends (llamacpp / ollama). Optional host overrides
    # for when the LLM server runs on a different machine.
    llama_host: str | None = None
    llama_model: str | None = None
    llama_timeout_sec: int | None = None
    ollama_host: str | None = None
    ollama_model: str | None = None
    gemma_timeout_sec: int | None = None
    MiniMax_api_key: str | None = None
    MiniMax_endpoint: str | None = None
    MiniMax_model: str | None = None
    MiniMax_enable_thinking: bool = True
    MiniMax_thinking_budget_tokens: int = 1024
    MiniMax_max_output_tokens: int | None = None
    MiniMax_max_concurrent: int | None = None
    MiniMax_timeout_sec: int | None = None
    MiniMax_max_retries: int | None = None
    MiniMax_fallback_default: str = "rules"  # gemma4 | rules | stop | retry
    data_outbound_policy: str = "api_full"  # api_full | api_redacted | local_only
    # ---- PDF figure extractor (GROBID vs OpenDataLoader) ----
    # Default to OpenDataLoader: it runs in-process and doesn't need a
    # separate GROBID server. Override to False to use GROBID explicitly.
    use_opendataloader: bool = True
    # ---- M3 5-stage engine overrides ----
    m3_enhanced_mode: bool | None = None
    m3_stage_1: bool | None = None
    m3_stage_2: bool | None = None
    m3_stage_3: bool | None = None
    m3_stage_4: bool | None = None
    m3_stage_5: bool | None = None
    m3_match_samples: int | None = None
    # ---- Paleobiology Database (opt-in) ----
    use_paleodb: bool = False
    paleodb_max_occurrences: int = 25
    paleodb_endpoint: str | None = None
    paleodb_cache_dir: str | None = None
    paleodb_offline: bool = False

    @field_validator("llm_backend")
    @classmethod
    def _validate_backend(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"transformers", "ollama", "llamacpp", "MiniMax"}
        if v not in allowed:
            raise ValueError(f"llm_backend must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("MiniMax_fallback_default")
    @classmethod
    def _validate_fallback(cls, v: str) -> str:
        allowed = {"gemma4", "rules", "stop", "retry"}
        if v not in allowed:
            raise ValueError(f"MiniMax_fallback_default must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("data_outbound_policy")
    @classmethod
    def _validate_policy(cls, v: str) -> str:
        allowed = {"api_full", "api_redacted", "local_only"}
        if v not in allowed:
            raise ValueError(f"data_outbound_policy must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("gemma_conf_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        # 0.0 means "always trust LLM"; 1.0 means "never trust LLM"
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"gemma_conf_threshold must be in [0.0, 1.0], got {v!r}")
        return v

    @field_validator("MiniMax_thinking_budget_tokens")
    @classmethod
    def _validate_thinking_budget(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"MiniMax_thinking_budget_tokens must be >= 0, got {v!r}")
        if v > 32_000:
            raise ValueError(f"MiniMax_thinking_budget_tokens must be <= 32000, got {v!r}")
        return v

    @field_validator("MiniMax_max_output_tokens", "MiniMax_max_concurrent",
                     "MiniMax_timeout_sec", "MiniMax_max_retries")
    @classmethod
    def _validate_positive_int(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v <= 0:
            raise ValueError(f"must be a positive integer, got {v!r}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _log_unknown_fields(cls, values: Any) -> Any:
        # Pydantic v2 defaults to ``extra="ignore"`` which silently
        # drops unknown keys — a frontend typo (e.g. ``minimax_api_key``
        # instead of ``MiniMax_api_key``) would run the pipeline without
        # the API key with no visible failure. Surface the dropped
        # keys at warning level so the typo is at least visible in
        # the server logs. (We don't use ``extra="forbid"`` because
        # the public API surface is allowed to grow over time, and
        # breaking every consumer on each new field is worse than
        # the silent-drop risk once the warning is in place.)
        if isinstance(values, dict):
            known = set(cls.model_fields.keys())
            unknown = sorted(set(values.keys()) - known)
            if unknown:
                logger.warning(
                    "JobOptions dropped unknown fields: %s (check the "
                    "frontend / caller for typos)", unknown,
                )
        return values


class FallbackDecisionRequest(BaseModel):
    """User response when MiniMax M3 API errors and the pipeline is paused."""
    job_id: str
    action: str  # "gemma4" | "rules" | "stop" | "retry"


# In-memory fallback-request registry: job_id -> (error_info, threading.Event, decision)
# WARNING: this dict is PER-PROCESS. If you run uvicorn with --workers N>1, the
# frontend may poll a different worker than the one running the job. To support
# multi-worker mode, set RLPE_REDIS_URL and use the redis-backed registry below.
FALLBACK_PENDING: dict[str, dict[str, Any]] = {}


app = FastAPI(
    title="RLPE API - Radiolarian Plate Extractor",
    version="0.2.0",
    description="Web API for radiolarian figure extraction from PDF literature"
)

# Enable CORS for frontend. Browsers reject
# ``Access-Control-Allow-Origin: *`` combined with
# ``Access-Control-Allow-Credentials: true`` (per the CORS spec), so
# credentials must be ``False`` unless we list a specific origin set.
# For a research tool that is normally served from the same origin
# via FastAPI's StaticFiles, wildcard origin + no credentials is the
# correct pairing — the previous combination silently dropped the
# ``Allow-Credentials`` header in browsers, breaking any cookie-based
# auth and confusing the operator about why logins weren't sticking.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR is not None:
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


# ------------------------------------------------------------------
# Startup: load previously completed jobs from disk into RESULT_CACHE
# ------------------------------------------------------------------
def _load_existing_jobs_from_disk() -> int:
    """Scan WORK_DIR and PROJECT_ROOT/work for completed jobs and populate
    RESULT_CACHE so the Web UI's results view shows them after a server restart.

    Returns the number of jobs loaded.
    """
    from datetime import datetime as _dt
    loaded = 0
    # Candidate roots: service_work/<job_id>/output/manifests/matches.jsonl
    # and the dev work/ directory at project root.
    roots: list[tuple[Path, str]] = []
    if WORK_DIR.exists():
        for child in sorted(WORK_DIR.iterdir()):
            if child.is_dir():
                roots.append((child, child.name))
    # Also scan project root work/ for ad-hoc CLI runs.
    cli_work = APP_ROOT / "work"
    if cli_work.exists() and cli_work.resolve() != WORK_DIR.resolve():
        # Synthesize a stable job_id from a hash of the path so it can be referenced.
        import hashlib
        jid = "cli_" + hashlib.md5(str(cli_work.resolve()).encode()).hexdigest()[:12]
        roots.append((cli_work, jid))

    for root, jid in roots:
        matches_path = root / "output" / "manifests" / "matches.jsonl"
        if not matches_path.exists():
            continue
        try:
            rows: list[dict[str, Any]] = []
            with matches_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception:
            continue
        if not rows:
            continue
        # Locate the original PDF (best-effort) and remember the row count.
        pdf_name: str | None = None
        pdfs_dir = root / "pdfs"
        if pdfs_dir.exists():
            pdfs = list(pdfs_dir.glob("*.pdf"))
            if pdfs:
                pdf_name = pdfs[0].name
        # Try to get a creation timestamp from filesystem.
        try:
            ts = _dt.fromtimestamp(matches_path.stat().st_mtime).isoformat()
        except Exception:
            ts = _dt.now().isoformat()
        RESULT_CACHE[jid] = {
            "status": "done",
            "result": rows,
            "error": None,
            "detail": f"loaded from disk ({len(rows)} rows)",
            "created_at": ts,
            "filename": pdf_name,
            "progress": 100,
            "_root": str(root.resolve()),
        }
        # Rewrite absolute panel_path to a URL the browser can fetch
        # (frontend's resolveAssetUrl treats /-prefixed paths as API URLs).
        for row in rows:
            p = row.get("panel_path")
            if p and not str(p).startswith("/jobs/"):
                try:
                    rel = _resolve_under(p, root).relative_to(root.resolve())
                    row["panel_path"] = f"/jobs/{jid}/files/{rel.as_posix()}"
                except ValueError:
                    # path is outside the job root; leave it as-is
                    pass
        loaded += 1
    return loaded


@app.on_event("startup")
def _on_startup() -> None:
    n = _load_existing_jobs_from_disk()
    if n:
        import logging as _log
        _log.getLogger("rlpe.api").info("Loaded %d existing job(s) from disk", n)


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
    target = (WEB_DIR / "css" / file_path).resolve()
    # Reject path traversal: file_path may contain ".." segments
    # that escape the css/ directory. The previous version didn't
    # validate this, so a request for /css/../../app.py would return
    # the project's main source file. Mirror the check in
    # /jobs/{id}/files/{file_path} below.
    css_root = (WEB_DIR / "css").resolve()
    try:
        target.relative_to(css_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid asset path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target)


@app.get("/js/{file_path:path}")
def web_js(file_path: str):
    if WEB_DIR is None:
        raise HTTPException(status_code=404, detail="Web assets not found")
    target = (WEB_DIR / "js" / file_path).resolve()
    js_root = (WEB_DIR / "js").resolve()
    try:
        target.relative_to(js_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid asset path")
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
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: str | None = Form(None),
):
    original_filename = file.filename or ""
    safe_filename = Path(original_filename).name
    if not safe_filename or safe_filename != original_filename or "/" in original_filename or "\\" in original_filename:
        raise HTTPException(status_code=400, detail="Invalid upload filename.")
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    # Optional: parse JSON options from form field (use_gemma4, llm_backend, MiniMax_*).
    # Even when no options field is supplied, instantiate JobOptions so
    # direct API/Swagger uploads get the same defaults as the web UI
    # (notably use_opendataloader=True).
    raw_options: Any = {}
    if options:
        try:
            raw_options = json.loads(options)
        except Exception:
            raise HTTPException(status_code=400, detail="`options` must be valid JSON.")
    try:
        validated = JobOptions(**(raw_options if isinstance(raw_options, dict) else {}))
        job_options = validated.model_dump(exclude_none=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid options: {exc}")
    # Read content to check size before writing.
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_SIZE_MB} MB limit.")
    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}_{safe_filename}"
    with save_path.open("wb") as f:
        f.write(content)

    now = datetime.now().isoformat()
    with RESULT_LOCK:
        RESULT_CACHE[job_id] = {
            "status": "queued",
            "result": None,
            "error": None,
            "detail": None,
            "created_at": now,
            "filename": safe_filename,
            "progress": 0
        }
    background_tasks.add_task(_run_job, job_id, save_path, job_options)
    return JobStatus(
        job_id=job_id,
        status="queued",
        created_at=now,
        filename=safe_filename
    )


@app.get("/jobs/{job_id}/status", response_model=JobStatus)
def job_status(job_id: str):
    with RESULT_LOCK:
        job = dict(RESULT_CACHE.get(job_id) or {})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        detail=job.get("error") or job.get("detail"),
        created_at=job.get("created_at"),
        filename=job.get("filename"),
        progress=job.get("progress", 0),
        stage=job.get("stage"),
        elapsed_sec=job.get("elapsed_sec"),
    )


@app.get("/jobs")
def list_jobs() -> list[JobStatus]:
    """List all jobs with their current status."""
    with RESULT_LOCK:
        items = [(job_id, dict(job)) for job_id, job in RESULT_CACHE.items()]
    return [
        JobStatus(
            job_id=job_id,
            status=job["status"],
            detail=job.get("error") or job.get("detail"),
            created_at=job.get("created_at"),
            filename=job.get("filename"),
            progress=job.get("progress", 0),
            stage=job.get("stage"),
            elapsed_sec=job.get("elapsed_sec"),
        )
        for job_id, job in items
    ]


@app.get("/jobs/{job_id}/files/{file_path:path}")
def job_file(job_id: str, file_path: str):
    # Resolve the actual job root: standard jobs live in WORK_DIR, but
    # CLI/imported jobs (e.g. loaded from work/) may live elsewhere.
    job = RESULT_CACHE.get(job_id)
    if job and job.get("_root"):
        job_root = Path(job["_root"]).resolve()
    else:
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
    """Cancel a pending or running job.

    Returns ``was_running`` (True if the job was already past the "queued"
    state when cancelled) and ``cancelled_at`` (ISO timestamp) so the UI
    can distinguish "user cancel" from "server auto-cancel" and show a
    sensible toast.
    """
    # All the read-check-write sequences below touch shared cache
    # state, and the heartbeat thread / progress callback / fallback
    # popup worker can be writing to the same per-job dict at the
    # same moment we are. Hold ``RESULT_LOCK`` for the whole cancel
    # sequence so the status flip + FALLBACK_PENDING release are
    # atomic with respect to the worker thread.
    with RESULT_LOCK:
        job = RESULT_CACHE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] not in {"queued", "running", "awaiting_user_decision"}:
            raise HTTPException(status_code=400, detail="Cannot cancel a finished job")
        was_running = job["status"] == "running"
        job["status"] = "cancelled"
        job["cancelled_at"] = datetime.now().isoformat()
        # If the pipeline is currently blocked in _web_fallback_popup
        # waiting on a user decision, release the event so the popup
        # returns the default action immediately. Without this, the
        # popup's 5-minute timeout keeps the worker thread alive even
        # though the user has already decided to abort.
        pending = FALLBACK_PENDING.get(job_id)
        if pending is not None:
            ev = pending.get("event")
            if ev is not None:
                ev.set()
            FALLBACK_PENDING.pop(job_id, None)
    return {
        "status": "cancelled",
        "job_id": job_id,
        "was_running": was_running,
        "cancelled_at": job["cancelled_at"],
    }


# ------------------------------------------------------------------
# Delete: remove one or more jobs from RESULT_CACHE and (optionally)
# delete their on-disk files. Hardened against path traversal.
# ------------------------------------------------------------------
def _is_relative_to(p: Path, base: Path) -> bool:
    """Compat helper: True if p is the same as or under base."""
    try:
        p.relative_to(base)
        return True
    except ValueError:
        return False


def _resolve_job_root(job_id: str) -> Path | None:
    """Return the on-disk root directory for a job, or None if not found."""
    job = RESULT_CACHE.get(job_id)
    if not job:
        return None
    # Prefer the recorded root (used by jobs loaded from disk);
    # fall back to WORK_DIR/job_id for normal uploaded jobs.
    if job.get("_root"):
        return Path(job["_root"]).resolve()
    return (WORK_DIR / job_id).resolve()


def _purge_job(job_id: str, delete_files: bool) -> dict[str, Any]:
    """Delete a single job. Returns a per-job status dict."""
    job = RESULT_CACHE.get(job_id)
    if not job:
        return {"job_id": job_id, "status": "not_found"}
    # Refuse to delete a running OR queued job. For "running": the
    # background thread is still alive and will keep writing to
    # RESULT_CACHE[job_id] (heartbeat, progress, status transitions,
    # final result). Popping the entry from under it surfaces as
    # KeyError in the worker thread — and the except handler in
    # _run_job also reads from RESULT_CACHE[job_id], so the failure
    # propagates as an unhandled error. For "queued": the background
    # task is scheduled but hasn't started yet; when it does, its
    # first line of work is `RESULT_CACHE[job_id]["status"] =
    # "running"`, which raises KeyError on the deleted entry, then the
    # except handler tries the same write and raises again. To safely
    # delete, the user must cancel the job first (which sets
    # status="cancelled"); the worker thread sees this in its progress
    # callback, raises _JobCancelledError, and exits cleanly.
    if job.get("status") in {"running", "queued"}:
        return {
            "job_id": job_id,
            "status": "refused",
            "error": (
                f"Job is currently {job.get('status')}; cancel it first via "
                f"/jobs/{job_id}/cancel, then delete."
            ),
        }
    files_removed = False
    bytes_freed = 0
    if delete_files:
        root = _resolve_job_root(job_id)
        if root is not None and root.exists():
            # Only allow deletion under known safe roots.
            safe_roots = [WORK_DIR.resolve(), (APP_ROOT / "work").resolve()]
            if not any(_is_relative_to(root, sr) for sr in safe_roots):
                return {
                    "job_id": job_id,
                    "status": "refused",
                    "error": f"root {root} not under safe dirs",
                }
            try:
                # Compute size before deletion for reporting.
                bytes_freed = sum(
                    f.stat().st_size for f in root.rglob("*") if f.is_file()
                )
                shutil.rmtree(root)
                files_removed = True
            except Exception as exc:
                return {
                    "job_id": job_id,
                    "status": "file_error",
                    "error": str(exc),
                }
    # Always remove from in-memory caches.
    RESULT_CACHE.pop(job_id, None)
    FALLBACK_PENDING.pop(job_id, None)
    return {
        "job_id": job_id,
        "status": "deleted",
        "files_removed": files_removed,
        "bytes_freed": bytes_freed,
    }


class BatchDeleteRequest(BaseModel):
    job_ids: list[str]
    delete_files: bool = True


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, delete_files: bool = True):
    """Delete a single job. Use ``?delete_files=false`` to keep on-disk data."""
    return _purge_job(job_id, delete_files=delete_files)


@app.post("/jobs/batch-delete")
def batch_delete_jobs(req: BatchDeleteRequest) -> dict[str, Any]:
    """Delete multiple jobs in one call. Returns a per-job status list."""
    if not req.job_ids:
        raise HTTPException(status_code=400, detail="job_ids must not be empty")
    if len(req.job_ids) > 200:
        raise HTTPException(status_code=400, detail="too many job_ids (max 200)")
    results = [_purge_job(jid, delete_files=req.delete_files) for jid in req.job_ids]
    return {
        "delete_files": req.delete_files,
        "requested": len(req.job_ids),
        "deleted": sum(1 for r in results if r["status"] == "deleted"),
        "files_removed_count": sum(1 for r in results if r.get("files_removed")),
        "bytes_freed": sum(r.get("bytes_freed", 0) for r in results),
        "results": results,
    }


@app.get("/jobs/{job_id}/MiniMax-fallback")
def get_MiniMax_fallback(job_id: str) -> dict[str, Any]:
    """Frontend polls this endpoint to detect when MiniMax API needs a user decision."""
    pending = FALLBACK_PENDING.get(job_id)
    if not pending:
        return {"status": "none", "job_id": job_id}
    return {
        "status": "awaiting_decision",
        "job_id": job_id,
        "error_info": pending.get("error_info", {}),
        "options": ["gemma4", "rules", "stop", "retry"],
    }


@app.post("/jobs/{job_id}/MiniMax-fallback")
def post_MiniMax_fallback(job_id: str, req: FallbackDecisionRequest) -> dict[str, Any]:
    """Frontend posts the user's choice; releases the pipeline."""
    if req.action not in {"gemma4", "rules", "stop", "retry"}:
        raise HTTPException(status_code=400, detail=f"Invalid action: {req.action}")
    # Hold ``RESULT_LOCK`` across the read of FALLBACK_PENDING and
    # the subsequent decision write + status flip so we can't race
    # cancel_job (which would pop the entry under the same lock).
    with RESULT_LOCK:
        pending = FALLBACK_PENDING.get(job_id)
        if not pending:
            raise HTTPException(status_code=404, detail="No pending fallback for this job.")
        pending["decision"] = req.action
        ev = pending.get("event")
        if ev is not None:
            ev.set()
        # Restore status to running
        if job_id in RESULT_CACHE:
            RESULT_CACHE[job_id]["status"] = "running"
    return {"status": "ok", "job_id": job_id, "action": req.action}


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
    with RESULT_LOCK:
        items = [(job_id, dict(job)) for job_id, job in RESULT_CACHE.items()]
    for job_id, job in items:
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
    with RESULT_LOCK:
        jobs = [dict(j) for j in RESULT_CACHE.values()]
    return {
        "version": "0.2.0",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "grobid_url": GROBID_URL,
        "active_jobs": sum(1 for j in jobs if j["status"] in {"queued", "running", "awaiting_user_decision"}),
        "total_jobs": len(jobs),
        "completed_jobs": sum(1 for j in jobs if j["status"] == "done"),
        "failed_jobs": sum(1 for j in jobs if j["status"] == "failed"),
    }


def _safe_value(v: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON serialization.

    Falls back to ``str(v)`` for unknown types (``Path``,
    ``datetime``, ``numpy.ndarray``, dataclass instances) so the
    API response is always JSON-serialisable. The previous
    pass-through of unknown types caused a 500 with no indication of
    which field tripped ``json.dumps`` downstream — by stringifying
    the last-resort case we at least produce a usable response body
    the operator can inspect.
    """
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    try:
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.ndarray):
            # Convert to a Python list (recursively safe) so the
            # element-wise types still go through _safe_value.
            return [_safe_value(x) for x in v.tolist()]
    except Exception:
        pass
    # Last-resort: stringify. ``datetime`` and ``Path`` round-trip
    # through ISO / str() fine; arbitrary objects get a repr that
    # the operator can read in the response body.
    try:
        return str(v)
    except Exception:
        return repr(v)


class _JobCancelledError(Exception):
    """Raised from the pipeline progress callback when the user cancels.

    The progress callback polls ``RESULT_CACHE[job_id]["status"]``; if it
    sees ``"cancelled"`` (set by ``/jobs/{id}/cancel``), it raises this
    exception. The outer try/except in ``_run_job`` catches it and writes
    a "cancelled" terminal state without re-marking the job as failed.
    """


def _resolve_under(path_like: str | Path, base: Path) -> Path:
    """Resolve ``path_like`` against ``base`` if it's relative.

    The previous code used ``Path(p).resolve()`` directly, which depends on
    the process CWD — if the server was started from a different directory
    than the one that produced ``matches.jsonl``, relative panel_paths in
    the jsonl silently fail to resolve (their `relative_to(root)` raises
    ValueError, panel_path stays relative, and the browser tries to fetch
    `/work/...` which has no route → 404 thumbnail).
    """
    p = Path(path_like)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert numpy types in a result row."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, dict):
            out[k] = _sanitize_row(v)
        elif isinstance(v, list):
            out[k] = [_sanitize_row(item) if isinstance(item, dict) else _safe_value(item) for item in v]
        else:
            out[k] = _safe_value(v)
    return out


def _run_job(job_id: str, pdf_path: Path, options: dict[str, Any] | None = None) -> None:
    options = options or {}
    import time as _time
    t_start = _time.time()
    stop_hb = threading.Event()
    def _heartbeat_loop() -> None:
        # Background thread that refreshes elapsed_sec every second so the
        # UI shows "alive" progress even when the pipeline is mid-figure and
        # hasn't called its own progress_cb tick.
        while not stop_hb.is_set():
            try:
                # Hold ``RESULT_LOCK`` so the heartbeat's elapsed write
                # can't race the cancel endpoint's status flip. The
                # write is a single dict key update, so contention
                # stays near-zero.
                with RESULT_LOCK:
                    if job_id in RESULT_CACHE:
                        RESULT_CACHE[job_id]["elapsed_sec"] = int(_time.time() - t_start)
            except Exception:
                pass
            stop_hb.wait(1.0)
    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name=f"rlpe-hb-{job_id[:8]}")
    hb_thread.start()
    # Pre-flight cancel check: if the user hit /jobs/{id}/cancel between
    # the upload's "queued" registration and the moment our worker thread
    # actually started running, the cancel endpoint set status="cancelled".
    # The unconditional "running" flip below would clobber that signal,
    # and the in-pipeline progress callback (which polls status) would
    # never see "cancelled" again. The result: the job would run to
    # completion, then silently transition cancelled → done. Bail out
    # before doing any work.
    if RESULT_CACHE.get(job_id, {}).get("status") == "cancelled":
        stop_hb.set()
        return
    # Transition out of "queued" so the UI stops showing the waiting spinner
    # once the worker thread has actually started running.
    RESULT_CACHE[job_id]["status"] = "running"
    try:
        ensure_dir(APP_ROOT / "static")
        pdf_dir = ensure_dir(WORK_DIR / job_id / "pdfs")
        moved_path = pdf_dir / pdf_path.name
        shutil.move(str(pdf_path), moved_path)
        RESULT_CACHE[job_id]["progress"] = 20
        RESULT_CACHE[job_id]["stage"] = "构建配置…"

        # Build extra config, allowing the web client to inject M3 options.
        extra: dict[str, Any] = {"use_gemma4": False}
        # NOTE: keep this list in sync with the CLI flags in cli.py
        for key in (
            "llm_backend", "MiniMax_api_key", "MiniMax_endpoint", "MiniMax_model",
            "MiniMax_enable_thinking", "MiniMax_thinking_budget_tokens",
            "MiniMax_max_output_tokens", "MiniMax_max_concurrent",
            "MiniMax_timeout_sec", "MiniMax_max_retries",
            "MiniMax_fallback_default", "data_outbound_policy",
            "MiniMax_interactive",
            # Local LLM backends (llamacpp / ollama). The web UI exposes
            # these in the LLM config panel; if the user fills in a custom
            # host, it must reach the pipeline (previously silently dropped).
            "llama_host", "llama_model", "llama_timeout_sec",
            "ollama_host", "ollama_model", "gemma_timeout_sec",
            # PDF figure extractor
            "use_opendataloader",
            # M3 5-stage engine toggles
            "m3_enhanced_mode", "m3_stage_1", "m3_stage_2",
            "m3_stage_3", "m3_stage_4", "m3_stage_5",
            "m3_match_samples", "m3_diagnostic_dir",
            # Paleobiology Database (opt-in)
            "use_paleodb", "paleodb_max_occurrences", "paleodb_endpoint",
            "paleodb_cache_dir", "paleodb_offline",
        ):
            if key in options and options[key] is not None:
                extra[key] = options[key]
        if options.get("use_gemma4"):
            extra["use_gemma4"] = True
        # For web mode, we never block on stdin; default to non-interactive.
        extra.setdefault("MiniMax_interactive", False)

        # Auto-detect GPU; user can override via options.use_gpu
        try:
            import torch
            default_use_gpu = bool(torch.cuda.is_available())
        except ImportError:
            default_use_gpu = False
        use_gpu_flag = bool(options.get("use_gpu", default_use_gpu))

        cfg = PipelineConfig(
            pdf_dir=pdf_dir,
            work_dir=WORK_DIR / job_id,
            output_dir=None,
            save_intermediate=True,
            use_gpu=use_gpu_flag,
            extra=extra,
        )

        # If using MiniMax, register a web-popup fallback handler.
        if str(extra.get("llm_backend", "")).lower() in {"minimax", "minimax-m3", "minimax_api"}:
            from ..llm_backends import FallbackHandler
            handler = FallbackHandler(default_action=extra.get("MiniMax_fallback_default", "rules"))

            def _web_fallback_popup(error_info: dict[str, Any]) -> str:
                import threading
                event = threading.Event()
                # Register the pending decision under the same lock
                # that cancel_job and post_MiniMax_fallback hold, so
                # the user can't race a cancel against the popup
                # registering.
                with RESULT_LOCK:
                    FALLBACK_PENDING[job_id] = {
                        "error_info": error_info,
                        "event": event,
                        "decision": None,
                    }
                    RESULT_CACHE[job_id]["status"] = "awaiting_user_decision"
                    RESULT_CACHE[job_id]["detail"] = (
                        f"MiniMax API error: {error_info.get('error_type', '?')} - "
                        f"{error_info.get('error', '?')[:200]}"
                    )
                # Poll for cancellation during the wait. A single
                # 300-second blocking wait would prevent the user from
                # cancelling the job while the popup is open — the cancel
                # endpoint does set the event, but we still want to
                # short-circuit if the user backs out of the popup (e.g.
                # closes the browser tab). A 0.5s tick keeps the wait
                # responsive without burning CPU.
                waited_ms = 0
                TIMEOUT_MS = 300_000
                TICK_MS = 500
                while waited_ms < TIMEOUT_MS:
                    if event.wait(timeout=TICK_MS / 1000):
                        break  # user posted a decision (or cancelled)
                    waited_ms += TICK_MS
                    # Cancellation check: if /cancel ran, the event was
                    # set AND FALLBACK_PENDING[job_id] was popped. Treat
                    # the absence of the entry as "cancelled" and
                    # return the default action so the pipeline can
                    # continue past this point (or fail cleanly — either
                    # way, the worker is no longer blocked).
                    if job_id not in FALLBACK_PENDING:
                        return handler.default_action
                decision = FALLBACK_PENDING.pop(job_id, {}).get("decision") or handler.default_action
                return decision

            handler.on_error = _web_fallback_popup
            cfg.extra["_MiniMax_external_handler"] = handler
            RESULT_CACHE[job_id]["MiniMax_fallback_handler"] = handler
        RESULT_CACHE[job_id]["progress"] = 30
        RESULT_CACHE[job_id]["stage"] = "开始处理 PDF…"

        # Build the pipeline with a progress callback that maps the pipeline's
        # (current, total) onto the 30-90% band of the job progress bar. This
        # is the missing piece that was making the UI look "stuck at 30%".
        def _on_progress(current: int, total: int, message: str) -> None:
            if total <= 0:
                pct = 50  # unknown total → sit in the middle
            else:
                pct = 30 + int(60 * current / total)
            RESULT_CACHE[job_id]["progress"] = max(30, min(89, pct))
            RESULT_CACHE[job_id]["stage"] = message
            # Cancellation poll: the user may have POSTed /jobs/{id}/cancel
            # while the pipeline is mid-figure. Without this check, the task
            # would run to completion and overwrite the "cancelled" status
            # with "done", making the cancel look like a no-op. Raising
            # here propagates out of pipeline.run() and is caught below.
            if RESULT_CACHE.get(job_id, {}).get("status") == "cancelled":
                raise _JobCancelledError(f"Job {job_id} cancelled by user")

        rows = RadiolarianPipeline(cfg, progress_callback=_on_progress).run()
        normalized_rows: list[dict[str, Any]] = []
        job_root = (WORK_DIR / job_id).resolve()
        for row in rows:
            normalized = _sanitize_row(asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row))
            panel_path = normalized.get("panel_path")
            if panel_path:
                # Resolve relative panel_paths against job_root, NOT against
                # the process CWD. Previously `Path(p).resolve()` would silently
                # fail to map CLI-generated `work/...` paths back to /jobs/{id}/files/...
                # when the server was started from a non-project directory.
                panel_abs = _resolve_under(panel_path, job_root)
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
    except _JobCancelledError:
        # Cancellation raised from the progress callback. Keep the
        # "cancelled" status that /jobs/{id}/cancel set; don't overwrite it
        # with "failed" (which would have been the previous behaviour for
        # any uncaught exception in the pipeline).
        RESULT_CACHE[job_id]["status"] = "cancelled"
        RESULT_CACHE[job_id]["detail"] = "Cancelled by user"
        RESULT_CACHE[job_id]["progress"] = RESULT_CACHE[job_id].get("progress", 0)
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
        RESULT_CACHE[job_id]["error_trace"] = tb
        RESULT_CACHE[job_id]["detail"] = "Pipeline execution failed"
        RESULT_CACHE[job_id]["progress"] = 0
    finally:
        # Stop the heartbeat thread so it doesn't keep a reference to the
        # job entry in RESULT_CACHE forever.
        try:
            stop_hb.set()
        except Exception:
            pass
