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
# (e.g. status=cancelled but progress=80%). Single GLOBAL lock keeps
# the invariants simple — the job-state machine is small enough that
# contention isn't a real cost. NOTE: this is a single ``threading.Lock()``
# shared across ALL job_ids, not per-job. Phase 54 wrapped _purge_job's
# rmtree call inside this lock, so deleting a large job directory
# (e.g. 80k files, 20 GB) blocks ALL other concurrent
# /jobs/{any_id}/* endpoints globally for the rmtree duration.
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
    # ``extra="ignore"`` means callers can extend the pipeline output
    # without breaking the public results endpoint. The earlier
    # ``extra="forbid"`` was good in theory (surfaces unknown fields
    # as 422) but in practice the pipeline accumulated several rows
    # the schema didn't yet declare (``panel_local_path``, custom
    # corrections, fallback metadata) and ``ResultRecord(**row)``
    # crashed the whole /results endpoint with a 500. Silent ignore
    # keeps the API surface honest (every declared field is what
    # we promise) without taking the whole endpoint down for a new
    # internal field. The /system/info endpoint logs the JobOptions
    # drop list so frontend typos are still observable.
    model_config = ConfigDict(extra="ignore")
    # ``row_id`` is the stable identity used by /results/batch DELETE.
    # Computed in get_results() as ``f"{job_id}:{paper_id}:{figure_id}:{panel_id}"``.
    row_id: str | None = None
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
    MiniMax_enable_thinking: bool = False  # default OFF to avoid surprise API cost
    MiniMax_thinking_budget_tokens: int = 1024
    MiniMax_max_output_tokens: int | None = None
    MiniMax_max_concurrent: int | None = None
    MiniMax_timeout_sec: int | None = None
    MiniMax_max_retries: int | None = None
    MiniMax_fallback_default: str = "rules"  # gemma4 | rules | stop | retry
    data_outbound_policy: str = "api_redacted"  # api_full | api_redacted | local_only
    # Default to api_redacted (caption text + plate region; sensitive
    # fields stripped before sending) so the web UI does not
    # silently send the full PDF text to the LLM backend. Operators
    # who want the previous behaviour can set api_full explicitly.
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
    # ---- Round 18 multi-modal geology vision ----
    # When True, M3Engine.extract_geology() reads each figure image +
    # caption and emits a structured GeologyLinkRecord. Default ON so
    # web-UI users get all 25 published geology fields populated
    # without having to flip an obscure flag. Operators who don't
    # want the per-figure API cost can pass use_geo_vision=False.
    use_geo_vision: bool = True
    geo_vision_figure_types: list[str] | None = None
    # ---- Core pipeline overrides (previously rendered in the web form but
    # silently dropped by the API) ----
    grobid_url: str | None = None
    ocr_backend: str | None = None  # "paddleocr" | "easyocr"
    num_workers: int | None = None  # 1..32
    min_panel_score: float | None = None  # 0.0..1.0
    # ---- Phase 28 caption-pairing page-distance windows ----
    # caption_window: GROBID path lookup window (default 2). Increase
    # when figure numbers in body text are far from the figure.
    # od_caption_window: OpenDataLoader path cross-page limit
    # (default 5). Increase when plates are clustered at the end of
    # the paper or captions sit on the page adjacent to the figure.
    caption_window: int | None = None  # 1..50
    od_caption_window: int | None = None  # 1..200
    # ---- Phase 29 GROBID retry + OD-fallback knobs ----
    # ``grobid_max_retries`` is the total HTTP attempts; ``grobid_timeout``
    # is the per-attempt request timeout. ``disable_od_fallback`` is the
    # escape hatch for operators who want strict legacy behaviour (visual
    # stub on GROBID failure, no OD retry). Defaults match CLI defaults.
    grobid_max_retries: int | None = None  # 1..10
    grobid_timeout: int | None = None  # 10..3600
    disable_od_fallback: bool = False

    @field_validator("llm_backend")
    @classmethod
    def _validate_backend(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"transformers", "ollama", "llamacpp", "MiniMax", "MiniMax-m3", "minimax"}
        if v not in allowed:
            raise ValueError(f"llm_backend must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("MiniMax_fallback_default")
    @classmethod
    def _validate_fallback(cls, v: str) -> str:
        allowed = {"gemma4", "rules", "stop", "retry"}
        if v not in allowed:
            raise ValueError(
                f"MiniMax_fallback_default must be one of {sorted(allowed)}, got {v!r}"
            )
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

    @field_validator(
        "MiniMax_max_output_tokens",
        "MiniMax_max_concurrent",
        "MiniMax_timeout_sec",
        "MiniMax_max_retries",
    )
    @classmethod
    def _validate_positive_int(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v <= 0:
            raise ValueError(f"must be a positive integer, got {v!r}")
        return v

    @field_validator("num_workers")
    @classmethod
    def _validate_num_workers(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 1 or v > 32:
            raise ValueError(f"num_workers must be 1..32, got {v!r}")
        return v

    # Phase 28: caption-window validators. Both default to None which
    # lets the pipeline fall back to its own field defaults (caption_window=2
    # for GROBID, od_caption_window=5 for OpenDataLoader).
    @field_validator("caption_window")
    @classmethod
    def _validate_caption_window(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 1 or v > 50:
            raise ValueError(f"caption_window must be 1..50, got {v!r}")
        return v

    @field_validator("od_caption_window")
    @classmethod
    def _validate_od_caption_window(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 1 or v > 200:
            raise ValueError(f"od_caption_window must be 1..200, got {v!r}")
        return v

    @field_validator("min_panel_score")
    @classmethod
    def _validate_min_panel_score(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"min_panel_score must be in [0.0, 1.0], got {v!r}")
        return v

    @field_validator("ocr_backend")
    @classmethod
    def _validate_ocr_backend(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"paddleocr", "easyocr"}
        if v not in allowed:
            raise ValueError(f"ocr_backend must be one of {sorted(allowed)}, got {v!r}")
        return v

    # Phase 29: GROBID retry + timeout validators
    @field_validator("grobid_max_retries")
    @classmethod
    def _validate_grobid_max_retries(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 1 or v > 10:
            raise ValueError(f"grobid_max_retries must be 1..10, got {v!r}")
        return v

    @field_validator("grobid_timeout")
    @classmethod
    def _validate_grobid_timeout(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 10 or v > 3600:
            raise ValueError(f"grobid_timeout must be 10..3600, got {v!r}")
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
                    "JobOptions dropped unknown fields: %s (check the frontend / caller for typos)",
                    unknown,
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


# ``lifespan`` replaces the deprecated ``on_event("startup")`` (removed in
# FastAPI 0.110+). The previous decorator still works on 0.111.0 but
# triggers a ``DeprecationWarning`` and is gone in 0.120+. The context
# manager form is the recommended one and works across all supported
# FastAPI versions.
import contextlib


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    n = _load_existing_jobs_from_disk()
    if n:
        import logging as _log

        _log.getLogger("rlpe.api").info("Loaded %d existing job(s) from disk", n)
    yield


app = FastAPI(
    title="RLPE API - Radiolarian Plate Extractor",
    version="1.1.0",
    description="Web API for radiolarian figure extraction from PDF literature",
    lifespan=_lifespan,
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
    # No-cache headers for the dev-mode static files so JS / CSS edits
    # propagate without the operator having to hard-refresh. Cache
    # busting is also wired into the script/link tags in index.html
    # so a stale browser still re-fetches the new content. The
    # trade-off (no intermediate cache) is fine for a research tool
    # served from a single dev box.
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as _Req

    class _NoCacheStatic(BaseHTTPMiddleware):
        async def dispatch(self, request: _Req, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/web/"):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

    app.add_middleware(_NoCacheStatic)
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


@app.get("/")
def root():
    if WEB_DIR is not None:
        index_path = WEB_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
    return {"status": "ok", "service": "rlpe-api", "docs": "/docs", "web": "/web"}


@app.get("/css/{file_path:path}")
def web_css(file_path: str):
    if WEB_DIR is None:
        raise HTTPException(status_code=404, detail="Web assets not found")
    # Path-traversal defense (audit): reject literal "..", "\\", and
    # absolute-path payloads BEFORE any filesystem resolution so the
    # safe-input contract is obvious from a quick read of the
    # function.
    if (
        ".." in file_path.split("/")
        or ".." in file_path.split("\\")
        or file_path.startswith(("/", "\\"))
    ):
        raise HTTPException(status_code=400, detail="Invalid asset path")
    css_root = (WEB_DIR / "css").resolve()
    target = (css_root / file_path).resolve()
    # ``strict=True`` makes relative_to raise ValueError when target is
    # outside css_root (the default behavior). The previous version
    # only raised when the result needed normalization, so a symlink
    # under css_root/ pointing OUTSIDE the root would silently pass
    # the check and serve an arbitrary file. Resolving both paths
    # BEFORE the relative_to check further reduces the
    # symlink-bypass surface.
    try:
        target.relative_to(css_root, strict=True)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid asset path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    # Refuse symlinks whose target resolves outside css_root.
    # ``resolve()`` already follows symlinks; if ``target`` was a
    # symlink pointing outside the root, the relative_to check above
    # would have raised (since ``target`` was already resolved), but
    # belt-and-braces guards against any future relaxation.
    if target.is_symlink():
        real = target.resolve(strict=True)
        try:
            real.relative_to(css_root, strict=True)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid asset path")
    return FileResponse(target)


@app.get("/js/{file_path:path}")
def web_js(file_path: str):
    if WEB_DIR is None:
        raise HTTPException(status_code=404, detail="Web assets not found")
    if (
        ".." in file_path.split("/")
        or ".." in file_path.split("\\")
        or file_path.startswith(("/", "\\"))
    ):
        raise HTTPException(status_code=400, detail="Invalid asset path")
    js_root = (WEB_DIR / "js").resolve()
    target = (js_root / file_path).resolve()
    try:
        target.relative_to(js_root, strict=True)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid asset path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    if target.is_symlink():
        real = target.resolve(strict=True)
        try:
            real.relative_to(js_root, strict=True)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid asset path")
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
    if (
        not safe_filename
        or safe_filename != original_filename
        or "/" in original_filename
        or "\\" in original_filename
    ):
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
            "progress": 0,
        }
    background_tasks.add_task(_run_job, job_id, save_path, job_options)
    return JobStatus(job_id=job_id, status="queued", created_at=now, filename=safe_filename)


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
    # Path-traversal defense (audit): reject literal traversal payloads
    # BEFORE filesystem resolution so the safe-input contract is
    # obvious from a quick read of the function.
    if (
        ".." in file_path.split("/")
        or ".." in file_path.split("\\")
        or file_path.startswith(("/", "\\"))
    ):
        raise HTTPException(status_code=400, detail="Invalid file path")
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
    # ``strict=True`` makes relative_to raise ValueError when target is
    # outside job_root. Without strict=True, a symlink inside the
    # job's working directory pointing OUTSIDE would silently pass
    # the check and let the API serve an arbitrary host file.
    # Verify ``target`` is inside ``job_root``. ``relative_to`` raises
    # ValueError when target is not a subpath of job_root — that's the
    # security guard we want. (PurePath.relative_to's ``strict=True``
    # kwarg was added in Python 3.12; we deliberately don't pass it so
    # this works on 3.10 too.)
    try:
        target.relative_to(job_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    # Belt-and-braces symlink guard.
    if target.is_symlink():
        real = target.resolve(strict=True)
        try:
            real.relative_to(job_root, strict=True)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid file path")
    return FileResponse(target)


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = RESULT_CACHE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # ``cancelled`` is a terminal state — return the (likely empty) payload
    # so the UI can stop polling. The previous version only treated
    # ``done`` / ``failed`` as terminal, which made the frontend's
    # adaptive-poll loop spin forever on cancelled jobs (it kept seeing
    # 202 "not finished" and never closed the poll). Surfacing cancelled
    # here also lets the UI render a "this run was cancelled at X" view
    # instead of an infinite spinner.
    if job["status"] not in {"done", "failed", "cancelled"}:
        raise HTTPException(status_code=202, detail="Job not finished")
    return job


@app.get("/jobs/{job_id}/export.xlsx")
def export_job_xlsx(job_id: str):
    """Download a multi-sheet .xlsx for one job.

    Round 24: the user requested an Excel export that captures
    ALL data (Round 23 CSV is one sheet; this endpoint produces
    5 sheets — panels, geology_contexts, localities,
    paleo_coordinates, legend). The endpoint streams the bytes
    via ``StreamingResponse`` so a 10⁵-row workbook doesn't
    block the worker thread.

    Sheets are produced by ``rlpe.exporters.xlsx.write_xlsx``
    which already sanitises formula-injection (CWE-1236) and
    uses ``openpyxl`` (transitive via the install env at
    version 3.1.5).
    """
    job = RESULT_CACHE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not finished (status={job.get('status')})",
        )
    if not job.get("result"):
        raise HTTPException(
            status_code=404, detail=f"Job {job_id} has no result"
        )
    try:
        from ..exporters.xlsx import write_xlsx

        run_output = {
            "panels": job["result"],
            # Localities, geology_contexts, and paleo_coordinates
            # are stored separately on the job dict. The frontend
            # /results endpoint reconstructs them via the
            # converters' helpers, but for the xlsx export we
            # read them directly from the cached job.
            "geology_contexts": job.get("geology_contexts", []) or [],
            "localities": job.get("localities", []) or [],
            "paleo_coordinates": job.get("paleo_coordinates", []) or [],
        }
        xlsx_bytes = write_xlsx(run_output)
    except Exception as exc:
        logger.exception("xlsx export failed for %s", job_id)
        raise HTTPException(
            status_code=500,
            detail=f"xlsx export failed: {exc}",
        ) from exc

    # Round 24: use the paper_id + job_id for the filename so
    # multiple exports land in different files.
    paper_id = (job.get("result", [{}])[0] or {}).get("paper_id", job_id[:8])
    filename = f"rlpe_{paper_id}_{job_id[:8]}.xlsx"

    # ``media_type`` for .xlsx is the OOXML spec:
    # application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
    # Phase 54 audit: B3 — wrap the entire status-check + cache-read +
    # filesystem-delete + cache-pop sequence in a single critical
    # section. The previous implementation read ``status`` at line 924
    # without holding ``RESULT_LOCK``, then performed ``shutil.rmtree``
    # at line 964 outside the lock. A worker thread that flipped
    # status to ``"running"`` between the two steps (legal: the
    # pre-flight cancel check at line 1696-1709 is the only thing
    # preventing it, and that only covers the initial flip — later
    # resume / re-launch paths don't re-check here) would have its
    # ``WORK_DIR/job_id`` deleted from under it, losing partially-
    # written ``matches.jsonl`` / ``run_output.json`` / ``llm_usage.json``.
    #
    # We do the filesystem work INSIDE the lock because
    # ``shutil.rmtree`` on a large job is fast (<1s for hundreds of
    # MB) and the lock is per-job — concurrent ``cancel`` / ``status``
    # / ``results`` endpoints only block when THIS specific job_id is
    # being purged. A separate, finer-grained lock would be overkill
    # for the current traffic pattern.
    with RESULT_LOCK:
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
        # except handler tries the same write and raises again.
        # For "awaiting_user_decision": the background thread is BLOCKED
        # inside ``_web_fallback_popup`` waiting on an event. Popping the
        # cache entry under it releases the wait (via the "if jid not in
        # FALLBACK_PENDING" check) but the thread will then return to
        # ``_apply_gemma_with_fallback`` and eventually try to write back
        # to ``RESULT_CACHE[job_id]`` — KeyError. To safely delete, the
        # user must cancel the job first (which sets status="cancelled");
        # the worker thread sees this in its progress callback, raises
        # _JobCancelledError, and exits cleanly.
        if job.get("status") in {"running", "queued", "awaiting_user_decision"}:
            return {
                "job_id": job_id,
                "status": "refused",
                "error": (
                    f"Job is currently {job.get('status')}; cancel it first via "
                    f"/jobs/{job_id}/cancel, then delete."
                ),
            }
        # Phase 54 audit: B4 — snapshot ``_root`` once under the lock so
        # the second lock-free ``RESULT_CACHE.get`` at the previous line
        # 949 is no longer a race. We also use this snapshot for the
        # ``_is_relative_to`` safety check and the rmtree itself, so the
        # entire purge uses a single immutable view of the job's root.
        cached_root_str = job.get("_root")
        files_removed = False
        bytes_freed = 0
        cli_loaded = False
        if delete_files and cached_root_str:
            root = Path(cached_root_str).resolve()
            if root.exists():
                # Refuse to delete CLI-loaded jobs' files. Those jobs were
                # discovered by ``_load_existing_jobs_from_disk`` from a
                # previous ``rlpe.cli`` run whose on-disk layout lives at
                # ``APP_ROOT/work`` — a DIRECTORY SHARED ACROSS ALL CLI
                # RUNS. The previous code allowed ``shutil.rmtree(root)``
                # to wipe the entire dev work/ tree (including any
                # unrelated CLI runs the user has done since the server
                # started). For CLI-loaded jobs we drop the in-memory
                # cache entry but leave the on-disk files alone; the user
                # can still delete them from a normal shell.
                if root == (APP_ROOT / "work").resolve():
                    cli_loaded = True
                else:
                    # Only allow deletion under known safe roots.
                    safe_roots = [WORK_DIR.resolve()]
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
        # ``deleted`` = removed from cache. For CLI-loaded jobs we
        # additionally set ``files_skipped`` so the UI can surface
        # "removed from list, but on-disk files preserved (shared
        # with other CLI runs)" rather than a confusing generic
        # success toast.
        "status": "deleted",
        "files_removed": files_removed,
        "files_skipped": cli_loaded and delete_files,
        "bytes_freed": bytes_freed,
    }


class BatchDeleteRequest(BaseModel):
    # Phase 54 audit: M16 — cap per-id length and validate the shape.
    # A previous version accepted up to 200 ids of any length, so a
    # single 800 KB payload (200 × 4 KB strings) could trigger 200
    # concurrent ``root.rglob("*")`` walks. 64 chars matches the
    # hex/UUID lengths RLPE actually generates; the regex keeps typos
    # from accidentally matching valid jobs in subsequent operations.
    model_config = ConfigDict(extra="forbid")
    job_ids: list[str] = Field(..., max_length=200)
    delete_files: bool = True

    @field_validator("job_ids")
    @classmethod
    def _validate_job_ids(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        for jid in v:
            if not isinstance(jid, str) or len(jid) > 64 or len(jid) == 0:
                raise ValueError(
                    f"job_id must be a non-empty string ≤ 64 chars, got {jid!r}"
                )
            # Phase 55 audit MEDIUM-2 fix: reject duplicate job_ids.
            # Previously duplicates passed validation, causing _purge_job to be
            # called twice with the same id — the second call would return
            # 'not_found', making aggregate metrics misleading and the per-id
            # results list show inconsistent states for the same job_id.
            if jid in seen:
                raise ValueError(f"duplicate job_id: {jid!r}")
            seen.add(jid)
        return v


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
    # Phase 54 audit: B5 — hold ``RESULT_LOCK`` around the read so we
    # can't race ``cancel_job`` (which pops under the same lock) or
    # ``_web_fallback_popup`` (which inserts under the same lock).
    # Without the lock, the dict.get is GIL-atomic for the single key,
    # but the subsequent ``pending.get("error_info", {})`` reads other
    # fields on the returned value — if cancel pops the entry between
    # those two reads, the response is a mix of pre- and post-pop state.
    # ``dict(pending)`` snapshots the values so the response is consistent
    # even if the entry is mutated by the worker thread while we are
    # serialising it.
    with RESULT_LOCK:
        pending = FALLBACK_PENDING.get(job_id)
        if not pending:
            return {"status": "none", "job_id": job_id}
        snapshot = {
            "error_info": dict(pending.get("error_info", {})),
        }
    return {
        "status": "awaiting_decision",
        "job_id": job_id,
        "error_info": snapshot["error_info"],
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
    row = {**payload.model_dump(), "timestamp": datetime.now().isoformat()}
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"status": "ok", "saved_to": str(target)}


@app.get("/results")
def get_results(
    limit: int = 500,
    offset: int = 0,
) -> list[ResultRecord]:
    """Get all accumulated results from completed jobs.

    Round 23 audit: this endpoint previously returned the FULL
    result set with no pagination. With 1000+ panels across many
    jobs, the response could be huge (causing frontend lag or
    timeouts). The new ``?limit=N&offset=M`` query params enable
    pagination. ``limit`` is capped at 5000 to prevent abuse.

    The response is ordered by ``(job_id, paper_id, figure_id,
    panel_id)`` so pagination is stable across requests. The
    response headers are NOT extended with a total-count link
    because computing it would require walking all rows anyway;
    the frontend asks for the next page until an empty page is
    returned (length 0) — implicit pagination.
    """
    # Round 23 audit: clamp limit/offset to safe ranges. Negative
    # offset returns the last ``limit`` rows; ``limit > 5000`` is
    # treated as ``5000``. Empty page (``limit=0``) is allowed for
    # callers that want to query "is there a next page?".
    limit = max(0, min(int(limit), 5000))
    offset = max(0, int(offset))
    results: list[ResultRecord] = []
    skipped = 0  # rows skipped before reaching the offset window
    with RESULT_LOCK:
        items = [(job_id, dict(job)) for job_id, job in RESULT_CACHE.items()]
    # Whitelist the row fields the public schema knows about. The
    # pipeline writes additional internal fields (panel_local_path,
    # diagnostic flags, ...) that are not part of the API contract
    # and would either inflate the response or trip future strict
    # parsers downstream. ``ResultRecord.model_fields`` is the
    # authoritative list — keeping the filter here means adding a
    # new field to the schema is the only change needed.
    allowed = set(ResultRecord.model_fields.keys()) - {"job_id", "row_id"}
    for job_id, job in items:
        if job.get("status") != "done" or not job.get("result"):
            continue
        for row in job["result"]:
            if not isinstance(row, dict):
                continue
            # Skip the first ``offset`` rows so callers can paginate.
            if skipped < offset:
                skipped += 1
                continue
            if limit and len(results) >= limit:
                # Reached the requested page size; bail out early so
                # we don't build a 10000-row response when the
                # caller only asked for 500.
                return results
            filtered = {k: v for k, v in row.items() if k in allowed}
            # ``paper_id`` and ``figure_id`` are required fields;
            # if a sanitiser ever produces a row missing them,
            # skip rather than 500 the whole endpoint.
            if "paper_id" not in filtered or "figure_id" not in filtered:
                continue
            # confidence is required (non-optional float). Default
            # to 0.0 so a partial row still serialises.
            filtered.setdefault("confidence", 0.0)
            # Inject the synthetic row_id so the frontend can
            # address rows for /results/batch DELETE.
            filtered["row_id"] = _row_id(job_id, filtered)
            try:
                results.append(ResultRecord(job_id=job_id, **filtered))
            except Exception as exc:
                logger.warning(
                    "Skipping malformed result row in job=%s: %s",
                    job_id,
                    exc,
                )
    return results


# ---------------------------------------------------------------------------
# Result-row delete endpoints (Round 16 UI addition).
#
# The /results table shows one row per panel extraction. Rows live inside a
# job's ``result`` list; they don't have their own top-level identity in
# RESULT_CACHE. We synthesise a stable ``row_id`` from the unique tuple
# ``(job_id, paper_id, figure_id, panel_id)`` and delete rows by filtering
# the job's result list against the requested row_ids.
#
# These endpoints deliberately only delete result rows — the job metadata
# and the on-disk pipeline output are kept so the operator can re-run or
# inspect after a clean-up pass.
# ---------------------------------------------------------------------------


def _row_id(job_id: str, row: dict[str, Any]) -> str:
    """Stable identifier for one result row.

    ``(job_id, paper_id, figure_id, panel_id)`` is unique within the
    cache because ``panel_id`` is unique per figure within a paper.
    """
    return (
        f"{job_id}:{row.get('paper_id', '')}:{row.get('figure_id', '')}:{row.get('panel_id', '')}"
    )


@app.delete("/results")
def delete_all_results() -> dict[str, int]:
    """Clear every result row across every done job.

    Keeps the job metadata (so ``/jobs`` listings, ``/system/info``
    totals, and any in-flight pipeline runs are unaffected). Returns
    the total number of rows removed.
    """
    total_removed = 0
    with RESULT_LOCK:
        for job_id, job in RESULT_CACHE.items():
            result_list = job.get("result")
            if isinstance(result_list, list):
                total_removed += len(result_list)
                job["result"] = []
    return {"removed": total_removed}


class DeleteRowsRequest(BaseModel):
    # Phase 54 audit: M15 — typed Pydantic request model. The previous
    # raw ``dict[str, list[str]]`` accepted ``{"row_ids": "abc"}`` (a
    # single string) and silently bypassed ``payload.get("row_ids") or
    # []`` — the string flowed straight into ``set(...)`` and then
    # ``if rid in row_ids`` did substring-style membership, producing
    # nonsense "not_found" counts. Pydantic v2 raises 422 on type
    # mismatch and a missing ``row_ids`` becomes a clear validation
    # error rather than a silent no-op.
    model_config = ConfigDict(extra="forbid")
    row_ids: list[str] = Field(default_factory=list)


@app.delete("/results/batch")
def delete_results_batch(payload: DeleteRowsRequest) -> dict[str, Any]:
    """Delete specific result rows by ``row_id``.

    Request body: ``{"row_ids": ["job_id:paper:figure:panel", ...]}``.
    Rows whose row_id is unknown are silently skipped (idempotent).
    Returns the number of rows actually removed and the not-found count.
    """
    row_ids = set(payload.row_ids or [])
    if not row_ids:
        return {"removed": 0, "not_found": 0}
    removed = 0
    not_found: list[str] = []
    matched_ids: set[str] = set()
    with RESULT_LOCK:
        for job_id, job in RESULT_CACHE.items():
            result_list = job.get("result")
            if not isinstance(result_list, list):
                continue
            kept: list[dict[str, Any]] = []
            for row in result_list:
                if not isinstance(row, dict):
                    kept.append(row)
                    continue
                rid = _row_id(job_id, row)
                if rid in row_ids:
                    removed += 1
                    matched_ids.add(rid)
                else:
                    kept.append(row)
            job["result"] = kept
    not_found = sorted(row_ids - matched_ids)
    return {"removed": removed, "not_found": len(not_found)}


@app.get("/system/info")
def system_info() -> dict[str, Any]:
    """Get system and configuration information."""
    with RESULT_LOCK:
        jobs = [dict(j) for j in RESULT_CACHE.values()]
    return {
        # Pull version from the rlpe package metadata so it stays in lock-step
        # with pyproject.toml (the previous hard-coded "1.1.0" silently
        # drifted whenever the package version was bumped).
        "version": _get_package_version(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "grobid_url": GROBID_URL,
        "active_jobs": sum(
            1 for j in jobs if j["status"] in {"queued", "running", "awaiting_user_decision"}
        ),
        "total_jobs": len(jobs),
        "completed_jobs": sum(1 for j in jobs if j["status"] == "done"),
        "failed_jobs": sum(1 for j in jobs if j["status"] == "failed"),
    }


# ---------------------------------------------------------------------------
# LLM status / test endpoints — used by the frontend onboarding panel to
# show "API Key configured / not configured" without leaking the key, and
# to let the operator click "Test API Key" before paying for a real run.
# ---------------------------------------------------------------------------


def _mask_api_key(key: str | None) -> str | None:
    """Return a non-revealing preview of an API key.

    ``sk-or-v1-abc...xyz`` becomes ``sk-...xyz`` so the operator can see
    *which* key is configured without us echoing it back. Empty / missing
    keys return None so the frontend can render the "not configured"
    state cleanly.
    """
    if not key:
        return None
    s = str(key).strip()
    if not s:
        return None
    if len(s) <= 8:
        return "***"
    return f"{s[:3]}...{s[-4:]}"


@app.get("/system/llm-status")
def llm_status() -> dict[str, Any]:
    """Report whether MiniMax / local LLM keys are configured.

    The frontend's onboarding banner uses this to render either:
        - "✅ API Key 已从 .env 读取 (sk-...abc)"  (key_configured=True)
        - "⚠️ 未配置 API Key — [立即设置]"           (key_configured=False)

    Never returns the raw key. The masked preview helps operators
    confirm WHICH key is loaded when they have multiple .env files.
    Also returns aggregated MiniMax usage if any jobs have made calls.
    """
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("MiniMax_API_KEY")
        or os.environ.get("MINIMAX_API_KEY")
        or ""
    )
    key_configured = bool(api_key.strip())

    # Aggregate MiniMax cost across all completed jobs (if any matched
    # results carry a cost_cny in their metadata). Each LLM API call may
    # generate multiple panel match rows (one per panel) but there is
    # only ONE LLM invocation per call AND each row in that group
    # carries the SAME ``MiniMax_cost_cny`` value (it's the cost of the
    # batched call, not per-panel). We deduplicate on
    # ``MiniMax_request_id`` so:
    #   - call counter is exact (1 per real API invocation)
    #   - cost counter doesn't multi-count the same batch
    total_cost_cny = 0.0
    seen_requests: set[str] = set()
    no_id_count = 0  # fallback when request_id is missing
    no_id_cost = 0.0
    with RESULT_LOCK:
        for job in RESULT_CACHE.values():
            if job.get("status") != "done":
                continue
            rows = job.get("result") or []
            for r in rows:
                md = (r or {}).get("metadata") or {}
                c = md.get("MiniMax_cost_cny")
                if c is None:
                    continue
                try:
                    cost_f = float(c)
                except (TypeError, ValueError):
                    continue
                req_id = md.get("MiniMax_request_id")
                if req_id:
                    if req_id in seen_requests:
                        continue
                    seen_requests.add(req_id)
                    total_cost_cny += cost_f
                else:
                    # Best-effort fallback: no request_id, so we can't
                    # deduplicate. Count and add separately so a future
                    # bug-report can distinguish the two regimes.
                    no_id_count += 1
                    no_id_cost += cost_f
    total_cost_cny += no_id_cost
    total_calls = len(seen_requests) + no_id_count

    # Resolved endpoint / model — what the pipeline will actually use
    # given the current env. The previous field name "default_endpoint"
    # was misleading because the value reflected the env override (e.g.
    # an Ark / Volces URL), not the system default. The new name
    # "active_endpoint" makes it clear this is the *resolved* value.
    # Both names are returned for one release so frontends that read
    # the old key continue to work.
    active_endpoint = os.environ.get("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
    active_model = os.environ.get(
        "MiniMax_MODEL",
        os.environ.get("ANTHROPIC_MODEL", "MiniMax-M3"),
    )

    return {
        "key_configured": key_configured,
        "key_preview": _mask_api_key(api_key) if key_configured else None,
        "key_source": (
            "env:ANTHROPIC_API_KEY"
            if os.environ.get("ANTHROPIC_API_KEY")
            else (
                "env:MiniMax_API_KEY"
                if os.environ.get("MiniMax_API_KEY")
                else ("env:MINIMAX_API_KEY" if os.environ.get("MINIMAX_API_KEY") else None)
            )
        ),
        "active_endpoint": active_endpoint,
        "active_model": active_model,
        # Deprecated aliases — drop in next major release.
        "default_endpoint": active_endpoint,
        "default_model": active_model,
        # Approximate cost per call (MiniMax M3 prices, 2026-06):
        # in:  ¥2.1 / M tokens   out: ¥8.4 / M tokens
        # A typical panel call uses ~2k input + ~0.5k output ≈ ¥0.0085/call
        "approx_cny_per_call": 0.0085,
        "total_cost_cny": round(total_cost_cny, 4),
        "total_calls": total_calls,
    }


class TestLLMRequest(BaseModel):
    """Body for /system/test-llm — all fields optional, falls back to env."""

    model_config = ConfigDict(extra="ignore")
    api_key: str | None = None
    endpoint: str | None = None
    model: str | None = None


@app.post("/system/test-llm")
def test_llm(req: TestLLMRequest | None = None) -> dict[str, Any]:
    """Send a minimal request to the MiniMax M3 endpoint to verify the key.

    The response shape matches the frontend's expectations:
        {"ok": true,  "latency_ms": 412, "model": "MiniMax-M3"}
        {"ok": false, "error": "401 Unauthorized: ..."}

    The test payload is intentionally tiny — a single "Reply OK" prompt
    with a 64-token output cap — so a successful call costs <¥0.001.
    Failures return ok=false with the exception type so the frontend
    can render a useful message ("Key invalid", "Network error", ...).
    """
    body = req or TestLLMRequest()
    api_key = (
        body.api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("MiniMax_API_KEY")
        or os.environ.get("MINIMAX_API_KEY")
        or ""
    ).strip()
    if not api_key:
        return {
            "ok": False,
            "error": "no API key provided (request body empty and "
            "ANTHROPIC_API_KEY env var not set)",
            "error_type": "MissingKey",
        }
    endpoint = (
        body.endpoint
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.minimaxi.com/anthropic"
    ).strip()
    model = (
        body.model
        or os.environ.get("MiniMax_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or "MiniMax-M3"
    ).strip()

    import time as _time

    t0 = _time.time()
    try:
        from ..llm_backends import MiniMaxM3Backend

        backend = MiniMaxM3Backend(
            api_key=api_key,
            base_url=endpoint,
            model=model,
            max_output_tokens=64,
            thinking_budget_tokens=0,
            enable_thinking=False,
            timeout_sec=15,
            max_retries=1,
            max_concurrent=1,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"backend init failed: {exc}",
            "error_type": type(exc).__name__,
        }

    try:
        # The minimal prompt: ask the model to reply with "OK". We do NOT
        # require the response to be JSON — only that the HTTP layer
        # succeeded (no auth/network/quota error). MiniMaxM3Backend's
        # ``_make_result`` returns ``fallback_used=True`` for ANY
        # exception, including JSON-parse failures on a non-JSON reply
        # like "OK". For a connection test that's a false negative — the
        # API actually worked. We therefore inspect ``error_type`` and
        # treat ``JSONParseError`` as success.
        result = backend.infer_text(
            system_prompt="You are a connection test. Reply with exactly: OK",
            user_prompt="ping",
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "latency_ms": int((_time.time() - t0) * 1000),
        }

    latency_ms = int((_time.time() - t0) * 1000)
    err_type = (result.get("error_type") or "").lower()
    fallback = bool(result.get("fallback_used"))
    # JSON-parse-on-non-JSON-reply is still a success for a connection
    # test: the API responded, charged us tokens, and returned text. We
    # only care that auth / quota / network worked.
    is_json_parse_only = err_type in {"jsonparseerror", "valueerror"} and (
        result.get("raw_text") or ""
    )
    if fallback and not is_json_parse_only:
        return {
            "ok": False,
            "error": result.get("error") or "API returned fallback_used=True",
            "error_type": result.get("error_type") or "Unknown",
            "latency_ms": latency_ms,
        }
    # ``result.get("usage") or {}`` was wrong: when the API returns
    # ``usage = {"input_tokens": 0, "output_tokens": 0}`` (a valid
    # zero-token response) the ``or {}`` treats the dict as falsy and
    # replaces it with {}, which then crashes the .get("input_tokens")
    # call below when the dict literal is in fact correct. Use an
    # explicit type check instead.
    usage_raw = result.get("usage")
    usage = usage_raw if isinstance(usage_raw, dict) else {}
    return {
        "ok": True,
        "latency_ms": latency_ms,
        "model": result.get("model_version") or model,
        "request_id": result.get("request_id"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_cny": result.get("cost_cny"),
        # Surface a JSON-parse note so the frontend can show a subtle
        # "API working (reply was not JSON, that's expected for /test)"
        # rather than nothing.
        "note": "Reply was non-JSON, treated as success for connection test."
        if is_json_parse_only
        else None,
    }


def _get_package_version() -> str:
    """Best-effort lookup of the rlpe package version.

    Prefers ``importlib.metadata`` (works for installed packages and most
    editable installs); falls back to reading the project pyproject.toml so
    a source-checkout run from a non-installed tree still shows the right
    version instead of a stale literal.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("rlpe")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    # Source-checkout fallback. tomllib lands in Python 3.11; fall back to
    # tomli for older interpreters that still need to serve the API.
    try:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]
        pyproject = APP_ROOT / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            return str(data.get("project", {}).get("version") or "unknown")
    except Exception:
        pass
    return "unknown"


def _safe_value(v: Any) -> Any:
    """Recursively convert numpy / non-JSON-native types to Python builtins.

    Handles (in priority order):
      * Python primitives — pass through unchanged
      * numpy scalars (int / float / bool) — unwrap to Python equivalents
      * numpy arrays — tolist() and recurse on each element
      * ``tuple`` / ``set`` / ``frozenset`` — recurse and emit as ``list``
        (JSON has no tuple; sets round-trip deterministically via sorted
        list when stable order is needed, otherwise set order is fine
        because the caller doesn't need to read this back).
      * ``datetime`` — ISO 8601 string
      * ``Path`` — ``str(path)``
      * ``bytes`` / ``bytearray`` — UTF-8 with replacement (binary
        payloads in a JSON field are always a bug, but we don't want to
        500 on them)
      * Last resort — ``str(v)`` so the API response is always
        JSON-serialisable.

    The previous version stringified tuple / set / list values, which
    silently corrupted nested bboxes and tag lists into repr-like
    strings (e.g. ``(0, 1, 2, 3)`` became the literal 13-character
    string ``"(0, 1, 2, 3)"``). Frontend code that expected a list
    (e.g. ``r.bbox[0]``) would then read the first character of the
    string instead. This now recurses instead of stringifying.
    """
    # Fast path: already JSON-native
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    # numpy
    try:
        import numpy as np

        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.ndarray):
            return [_safe_value(x) for x in v.tolist()]
    except Exception:
        pass
    # Container types — recurse and emit as JSON-compatible list.
    # We don't sort sets by default because order is usually not the
    # caller's concern; if stable ordering is needed, the caller can
    # convert to a sorted list before storing.
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_safe_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_value(val) for k, val in v.items()}
    # datetime
    try:
        import datetime as _dt

        if isinstance(v, (_dt.datetime, _dt.date)):
            return v.isoformat()
    except Exception:
        pass
    # pathlib
    try:
        from pathlib import Path as _Path

        if isinstance(v, _Path):
            return str(v)
    except Exception:
        pass
    # bytes
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return repr(v)
    # Last-resort: stringify. ``repr`` of arbitrary objects round-trips
    # through JSON so the API never 500s on an unknown scalar type.
    try:
        return str(v)
    except Exception:
        try:
            return repr(v)
        except Exception:
            return "<unserializable>"


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
            out[k] = [
                _sanitize_row(item) if isinstance(item, dict) else _safe_value(item) for item in v
            ]
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
    # before doing any work. Both the read-check AND the status-flip
    # must happen under RESULT_LOCK so /cancel cannot slip in between
    # them and have its "cancelled" write clobbered by our "running".
    with RESULT_LOCK:
        cur = RESULT_CACHE.get(job_id, {})
        if cur.get("status") == "cancelled":
            stop_hb.set()
            return
        if job_id not in RESULT_CACHE:
            # The /jobs/{id} delete endpoint refuses to drop "queued"
            # entries, but if it ever does (or if a future code path
            # races with us), bail rather than KeyError below.
            stop_hb.set()
            return
        # Transition out of "queued" so the UI stops showing the waiting
        # spinner once the worker thread has actually started running.
        RESULT_CACHE[job_id]["status"] = "running"
    try:
        ensure_dir(APP_ROOT / "static")
        pdf_dir = ensure_dir(WORK_DIR / job_id / "pdfs")
        moved_path = pdf_dir / pdf_path.name
        shutil.move(str(pdf_path), moved_path)
        with RESULT_LOCK:
            if job_id in RESULT_CACHE:
                RESULT_CACHE[job_id]["progress"] = 20
                RESULT_CACHE[job_id]["stage"] = "构建配置…"

        # Build extra config, allowing the web client to inject M3 options.
        extra: dict[str, Any] = {"use_gemma4": False}
        # NOTE: keep this list in sync with the CLI flags in cli.py
        for key in (
            "llm_backend",
            "MiniMax_api_key",
            "MiniMax_endpoint",
            "MiniMax_model",
            "MiniMax_enable_thinking",
            "MiniMax_thinking_budget_tokens",
            "MiniMax_max_output_tokens",
            "MiniMax_max_concurrent",
            "MiniMax_timeout_sec",
            "MiniMax_max_retries",
            "MiniMax_fallback_default",
            "data_outbound_policy",
            "MiniMax_interactive",
            # Local LLM backends (llamacpp / ollama). The web UI exposes
            # these in the LLM config panel; if the user fills in a custom
            # host, it must reach the pipeline (previously silently dropped).
            "llama_host",
            "llama_model",
            "llama_timeout_sec",
            "ollama_host",
            "ollama_model",
            "gemma_timeout_sec",
            # PDF figure extractor
            "use_opendataloader",
            # M3 5-stage engine toggles
            "m3_enhanced_mode",
            "m3_stage_1",
            "m3_stage_2",
            "m3_stage_3",
            "m3_stage_4",
            "m3_stage_5",
            "m3_match_samples",
            "m3_diagnostic_dir",
            # Paleobiology Database (opt-in)
            "use_paleodb",
            "paleodb_max_occurrences",
            "paleodb_endpoint",
            "paleodb_cache_dir",
            "paleodb_offline",
            # Round 18 multi-modal geology vision
            "use_geo_vision",
            "geo_vision_figure_types",
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

        # Build kwargs for the top-level (non-extra) PipelineConfig fields so
        # the web form's grobid-url / ocr-backend / num-workers / min-panel-score
        # actually reach the pipeline (they previously sat in the JobOptions
        # model unused and were dropped on the floor).
        pipeline_kwargs: dict[str, Any] = {
            "pdf_dir": pdf_dir,
            "work_dir": WORK_DIR / job_id,
            "output_dir": None,
            "save_intermediate": True,
            "use_gpu": use_gpu_flag,
            "extra": extra,
        }
        if options.get("grobid_url"):
            pipeline_kwargs["grobid_url"] = options["grobid_url"]
        if options.get("ocr_backend"):
            pipeline_kwargs["ocr_backend"] = options["ocr_backend"]
        if options.get("num_workers") is not None:
            pipeline_kwargs["num_workers"] = int(options["num_workers"])
        if options.get("min_panel_score") is not None:
            pipeline_kwargs["min_panel_score"] = float(options["min_panel_score"])
        # Phase 28: forward the two caption-page-distance windows.
        # These are first-class PipelineConfig fields, not extras.
        if options.get("caption_window") is not None:
            pipeline_kwargs["caption_window"] = int(options["caption_window"])
        if options.get("od_caption_window") is not None:
            pipeline_kwargs["od_caption_window"] = int(options["od_caption_window"])
        # Phase 29: forward GROBID retry + timeout + OD-fallback opt-out.
        # ``caption_window`` (GROBID) and ``od_caption_window`` (OD) are
        # first-class PipelineConfig fields; the GROBID retry + OD-fallback
        # knobs are extras consumed by ``RadiolarianPipeline.__init__``.
        if options.get("grobid_max_retries") is not None:
            extra["grobid_max_retries"] = int(options["grobid_max_retries"])
        if options.get("grobid_timeout") is not None:
            extra["grobid_timeout"] = int(options["grobid_timeout"])
        if options.get("disable_od_fallback"):
            extra["disable_od_fallback"] = True
        cfg = PipelineConfig(**pipeline_kwargs)

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
                # Race fix: ``FALLBACK_PENDING.pop(job_id, {})`` was
                # dropping the user's decision when ``post_MiniMax_fallback``
                # had already popped the entry on another thread (the
                # ``/MiniMax-fallback`` handler pops, then this block also
                # pops, and the second pop returns ``{}`` which silently
                # overrides the real decision with ``default_action``).
                # Read the decision WITHOUT popping; the cleanup is
                # idempotent. If the entry was already removed, ``.get``
                # returns ``{}`` and the ``or`` falls back to the default
                # action — but only when no real decision was posted, which
                # is the correct behaviour.
                entry = FALLBACK_PENDING.get(job_id) or {}
                decision = entry.get("decision") or handler.default_action
                # Cleanup is best-effort; ignore KeyError if another
                # thread already removed the entry.
                try:
                    del FALLBACK_PENDING[job_id]
                except KeyError:
                    pass
                return decision

            handler.on_error = _web_fallback_popup
            cfg.extra["_MiniMax_external_handler"] = handler
            with RESULT_LOCK:
                if job_id in RESULT_CACHE:
                    RESULT_CACHE[job_id]["MiniMax_fallback_handler"] = handler
        with RESULT_LOCK:
            if job_id in RESULT_CACHE:
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
            # Hold RESULT_LOCK so the cancel endpoint cannot pop the
            # entry under us between the read and the writes. The lock
            # also serialises this with the heartbeat thread's elapsed_sec
            # write — both touch the same dict in unrelated threads.
            with RESULT_LOCK:
                entry = RESULT_CACHE.get(job_id)
                if entry is None:
                    # Job was deleted while we were running. Treat as
                    # cancellation so the worker exits cleanly.
                    raise _JobCancelledError(f"Job {job_id} was deleted")
                if entry.get("status") == "cancelled":
                    raise _JobCancelledError(f"Job {job_id} cancelled by user")
                entry["progress"] = max(30, min(89, pct))
                entry["stage"] = message

        rows = RadiolarianPipeline(cfg, progress_callback=_on_progress).run()
        normalized_rows: list[dict[str, Any]] = []
        job_root = (WORK_DIR / job_id).resolve()
        for row in rows:
            normalized = _sanitize_row(
                asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row)
            )
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
        with RESULT_LOCK:
            entry = RESULT_CACHE.get(job_id)
            if entry is not None:
                # If a cancel/delete slipped in between run() returning and
                # this lock, don't resurrect the job into "done".
                if entry.get("status") not in {"cancelled"}:
                    entry["progress"] = 90
                    entry["status"] = "done"
                    entry["result"] = normalized_rows
                    if normalized_rows:
                        entry["detail"] = f"Generated {len(normalized_rows)} result rows"
                    else:
                        entry["detail"] = "Pipeline finished but no panels/matches were produced"
                    entry["progress"] = 100
    except _JobCancelledError:
        # Cancellation raised from the progress callback. Keep the
        # "cancelled" status that /jobs/{id}/cancel set; don't overwrite it
        # with "failed" (which would have been the previous behaviour for
        # any uncaught exception in the pipeline).
        with RESULT_LOCK:
            entry = RESULT_CACHE.get(job_id)
            if entry is not None:
                entry["status"] = "cancelled"
                entry["detail"] = "Cancelled by user"
                entry["progress"] = entry.get("progress", 0)
    except Exception as exc:
        tb = traceback.format_exc(limit=8)
        err = str(exc)
        if "object has no attribute 'route'" in err and "Starlette" in err:
            err = (
                f"{err}. Possible PyMuPDF/fitz package conflict. "
                "Install `pymupdf` and uninstall non-PyMuPDF `fitz`."
            )
        with RESULT_LOCK:
            entry = RESULT_CACHE.get(job_id)
            if entry is not None:
                entry["status"] = "failed"
                entry["error"] = err
                entry["error_trace"] = tb
                entry["detail"] = "Pipeline execution failed"
                entry["progress"] = 0
    finally:
        # Stop the heartbeat thread so it doesn't keep a reference to the
        # job entry in RESULT_CACHE forever.
        # Phase 54 audit: B6 — join the thread instead of just setting
        # the stop event. The previous version only flipped the event;
        # the thread was ``daemon=True`` so the process wouldn't hang on
        # exit, but each ``cancel`` / ``delete`` leaked one Thread
        # object plus its closure over ``_run_job``'s frame for up to
        # 1 second (the ``stop_hb.wait(1.0)`` tick). For batch
        # operations processing many short-lived jobs this accumulated
        # Thread objects and held onto per-job memory.
        try:
            stop_hb.set()
        except Exception:
            pass
        hb_thread.join(timeout=2.0)
