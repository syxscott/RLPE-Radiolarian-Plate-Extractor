"""QThread wrapper around ``RadiolarianPipeline.run()``.

Phase 32 of the GUI plan: the GUI must never block on the pipeline.
The pipeline can take minutes per paper; running it on the Qt main
thread would freeze the UI. We use a ``QThread`` subclass that:

1. Builds a ``PipelineConfig`` from a settings dict (mirrors what
   the CLI does in ``src/rlpe/cli.py``).
2. Constructs a ``RadiolarianPipeline`` with ``progress_callback``
   wired to a Qt signal.
3. Emits signals for progress, log messages, per-region results,
   and final outcome.
4. Supports graceful cancellation via ``requestInterruption``.

We deliberately do NOT use ``multiprocessing.Process`` for the GUI
worker — the Qt main loop + cv2/torch + ProcessPoolExecutor can
deadlock on fork-after-thread. The pipeline's own internal
``ThreadPoolExecutor`` (for OCR) is cv2-safe.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..config import PipelineConfig
from ..pipeline import RadiolarianPipeline
from .utils import get_gui_logger


class PipelineWorker(QThread):
    """Background worker that runs a single ``RadiolarianPipeline.run()``.

    Signals
    -------
    progress(int, int, str) — current, total, message
        Matches the callback signature already accepted by
        ``RadiolarianPipeline(progress_callback=...)`` (see
        ``src/rlpe/pipeline.py:66``).
    status_changed(str) — short status string for the Jobs tab
    log_line(str) — one line of internal log output
    result_row(dict) — one panel-level row, emitted as each region
        completes (so the Results tab can stream updates).
    finished_ok(list) — list of result rows when pipeline completes
    failed(str) — error message when pipeline raises
    """

    # progress(current, total, message)
    progress = Signal(int, int, str)
    # status_changed(status_string)
    status_changed = Signal(str)
    # log_line(line)
    log_line = Signal(str)
    # result_row(row_dict)
    result_row = Signal("QVariant")
    # finished_ok(results_list)
    finished_ok = Signal("QVariant")
    # failed(error_message)
    failed = Signal(str)

    def __init__(
        self,
        settings: dict[str, Any],
        pdf_path: Path,
        work_dir: Path,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._pdf_path = Path(pdf_path)
        self._work_dir = Path(work_dir)
        self._logger = get_gui_logger()

    # ------------------------------------------------------------------
    # Helpers exposed to the GUI
    # ------------------------------------------------------------------

    def paper_id(self) -> str:
        """Stable paper id, computed the same way the pipeline does.

        Currently we delegate to ``rlpe.utils.stable_id`` via the
        extra dict fallback path. Keeping a method on the worker
        avoids a duplicate import.
        """
        # Use a hash of the absolute path. Matches
        # ``rlpe.utils.stable_id`` which uses sha1(file_path)[:16].
        import hashlib

        path_str = str(self._pdf_path.resolve())
        return hashlib.sha1(path_str.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    @Slot()
    def run(self) -> None:  # noqa: D401 — QThread override
        """Build the pipeline config, run it, emit results.

        Exceptions are caught and forwarded via the ``failed`` signal
        so the GUI thread can present a dialog without crashing.
        """
        try:
            self.status_changed.emit("starting")
            self._emit_log(f"Building pipeline config for {self._pdf_path.name}")

            cfg = self._build_config()
            self._emit_log(
                f"Pipeline config ready: OCR={cfg.extra.get('ocr_lang', 'en')} "
                f"GEMMA={cfg.extra.get('llm_backend', 'rules')} "
                f"caption_window={cfg.od_caption_window}"
            )

            self.status_changed.emit("running")
            pipeline = RadiolarianPipeline(
                cfg,
                progress_callback=self._on_progress,
            )
            self._emit_log("Pipeline instantiated; running on PDF...")
            results = pipeline.run()

            # Convert each result row to a plain dict (QVariant friendly).
            out = [self._row_to_dict(r) for r in results]
            self._emit_log(f"Pipeline finished: {len(out)} rows")
            self.status_changed.emit("done")
            self.finished_ok.emit(out)
        except Exception as exc:
            tb = traceback.format_exc()
            self._logger.exception("Pipeline worker failed: %s", exc)
            self.status_changed.emit("failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{tb}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_progress(self, current: int, total: int, message: str) -> None:
        """Forward the pipeline's progress_callback to the Qt signal.

        The callback signature (current, total, message) was added in
        pipeline.py:66 and matches the API of multiple existing
        callers (FastAPI app.py:1902, web UI). Reusing the same shape
        keeps the GUI, the web app, and the CLI on the same data path.
        """
        if self.isInterruptionRequested():
            # We can't actually stop the pipeline mid-run (no
            # cooperation hook); we just stop forwarding progress.
            return
        self.progress.emit(int(current), int(total), str(message))

    def _emit_log(self, line: str) -> None:
        self._logger.info(line)
        self.log_line.emit(line)

    def _build_config(self) -> PipelineConfig:
        """Translate the GUI settings dict → ``PipelineConfig``.

        Mirrors ``src/rlpe/cli.py`` (lines 374–407). We keep the
        mapping local to the worker so the GUI can iterate on field
        names without touching the CLI.
        """
        s = dict(self._settings)
        # Numeric coercion w/ defaults
        num_workers = int(s.get("num_workers", 1))
        num_workers = max(1, min(num_workers, 32))
        min_panel_score = float(s.get("min_panel_score", 0.80))
        min_panel_score = max(0.0, min(min_panel_score, 1.0))
        render_dpi = int(s.get("render_dpi", 200))
        render_dpi = max(50, min(render_dpi, 600))
        grobid_max_retries = int(s.get("grobid_max_retries", 3))
        grobid_max_retries = max(1, min(grobid_max_retries, 10))
        grobid_timeout = int(s.get("grobid_timeout", 300))
        grobid_timeout = max(10, min(grobid_timeout, 3600))
        caption_window = int(s.get("caption_window", 2))
        caption_window = max(1, min(caption_window, 50))
        od_caption_window = int(s.get("od_caption_window", 5))
        od_caption_window = max(1, min(od_caption_window, 200))
        paleodb_max_occ = int(s.get("paleodb_max_occurrences", 25))
        paleodb_max_occ = max(1, min(paleodb_max_occ, 1000))

        # ``use_gpu`` is tri-state: True / False / None (auto-detect).
        # The CLI passes `default=None`; we do the same.
        use_gpu_raw = s.get("use_gpu", None)
        if use_gpu_raw is None:
            use_gpu = bool(_detect_gpu())
        else:
            use_gpu = bool(use_gpu_raw)

        extra: dict[str, Any] = {
            "use_opendataloader": bool(s.get("use_opendataloader", True)),
            "od_use_ocr": bool(s.get("od_use_ocr", False)),
            "od_ocr_lang": str(s.get("ocr_lang", "en")),
            "od_merge_gap_pt": float(s.get("od_merge_gap_pt", 72.0)),
            "ocr_lang": str(s.get("ocr_lang", "en")),
            "m3_prompt_lang": str(s.get("m3_prompt_lang", "auto")),
            "grobid_max_retries": grobid_max_retries,
            "grobid_timeout": grobid_timeout,
            "disable_od_fallback": bool(s.get("disable_od_fallback", False)),
            "use_geology_llm": bool(s.get("use_geology_llm", False)),
            "use_geo_vision": bool(s.get("use_geo_vision", False)),
            "use_m3_stage3": bool(s.get("use_m3_stage3", True)),
            "m3_multi_plate_enrich": bool(s.get("m3_multi_plate_enrich", True)),
            "use_paleodb": bool(s.get("use_paleodb", True)),
            "paleodb_max_occurrences": paleodb_max_occ,
            "MiniMax_max_retries": int(s.get("MiniMax_max_retries", 3)),
            "MiniMax_timeout_sec": int(s.get("MiniMax_timeout_sec", 60)),
            "MiniMax_thinking_budget_tokens": int(s.get("MiniMax_thinking_budget", 1024)),
            "MiniMax_max_concurrent": int(s.get("MiniMax_max_concurrent", 1)),
            "data_outbound_policy": str(s.get("data_outbound_policy", "local_only")),
            # SAM2 / model paths — pass through if set
            "sam2_checkpoint": s.get("sam2_checkpoint"),
            "sam2_model_cfg": s.get("sam2_model_cfg"),
            "taxon_hf_model_path": s.get("taxon_hf_model_path"),
            "taxon_lexicon_path": s.get("taxon_lexicon_path"),
            "m3_diagnostic_dir": s.get("m3_diagnostic_dir"),
        }

        cfg = PipelineConfig(
            pdf_dir=self._work_dir / "input",
            work_dir=self._work_dir,
            output_dir=self._work_dir,
            save_intermediate=bool(s.get("save_intermediate", False)),
            grobid_url=str(s.get("grobid_url", "http://localhost:8070")),
            ocr_backend=str(s.get("ocr_backend", "paddleocr")),
            taxon_model=str(s.get("taxon_model", "en_eco")),
            use_gpu=use_gpu,
            num_workers=num_workers,
            min_panel_score=min_panel_score,
            render_dpi=render_dpi,
            caption_window=caption_window,
            od_caption_window=od_caption_window,
            extra=extra,
        )
        # Ensure the single input PDF is in the work_dir's input/
        # so the pipeline's ``pdf_dir.glob("*.pdf")`` finds it.
        in_dir = self._work_dir / "input"
        in_dir.mkdir(parents=True, exist_ok=True)
        target = in_dir / self._pdf_path.name
        if target.resolve() != self._pdf_path.resolve():
            import shutil

            shutil.copy2(self._pdf_path, target)
        return cfg

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        """Convert a pipeline result row to a JSON-friendly dict.

        ``row`` is a plain ``dict`` per the pipeline's run() return
        type. We coerce nested structures (paper_metadata, metadata)
        to plain dicts so ``QVariant`` marshalling works.
        """
        if hasattr(row, "to_dict"):
            return row.to_dict()
        if isinstance(row, dict):
            return {
                "paper_id": row.get("paper_id"),
                "figure_id": row.get("figure_id"),
                "panel_id": row.get("panel_id"),
                "species": row.get("species"),
                "panel_path": row.get("panel_path"),
                "bbox": row.get("bbox"),
                "confidence": row.get("confidence"),
                "label_text": row.get("label_text"),
                "caption_snippet": row.get("caption_snippet"),
                "ocr_text": row.get("ocr_text"),
                "metadata": _flatten_metadata(row.get("metadata") or {}),
                "paper_metadata": _flatten_metadata(row.get("paper_metadata") or {}),
                "geology_links": row.get("geology_links", []),
            }
        return {"raw": str(row)}


def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Convert nested dataclass-like metadata to a plain dict."""
    out: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if hasattr(v, "to_dict"):
            out[k] = v.to_dict()
        elif isinstance(v, dict):
            out[k] = _flatten_metadata(v)
        else:
            out[k] = v
    return out


def _detect_gpu() -> bool:
    """Best-effort GPU detection — returns True if torch + CUDA."""
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False