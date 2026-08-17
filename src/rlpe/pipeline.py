from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)


# audit 2026-08-01 (D2): module-level lock serialising the
# "switch to local fallback backend" path. ``_switch_to_fallback_backend``
# builds a new local model runtime and reassigns it onto
# ``self.gemma_runtime`` + ``self.m3_engine.backend``. Without a lock
# N worker threads can each catch a ``FallbackRecommendedError`` at
# once and each spin up their own copy of the model — a fast path
# to OOM on a single-GPU box. A plain ``threading.Lock`` is enough
# (the critical section is a few attribute assignments, no IO to
# block on for long).
_BACKEND_SWITCH_LOCK = threading.Lock()


# --- Pre-filters for non-specimen content ----------------------------------
# ``_looks_like_placeholder_caption`` lives in ``text_filters`` so the
# evaluation harness and unit tests can import it without dragging in
# the torch/gemma/paddleocr chain pulled by the full pipeline.

from .association import (
    _TAXON_STOP_WORDS,
    _iou,
    _label_in_pair_lookup,
    _normalize_panel_label,
    is_valid_panel_label,
    match_panels,
)
from .config import PipelineConfig
from .converters import match_result_from_dict, run_output_from_provenance
from .gemma_postprocess import apply_gemma_to_matches, build_gemma_backend_from_config
from .geology_extraction import build_knowledge_graph, link_species_to_geology
from .grobid import GrobidClient, PipelineCancelledError, parse_paper_metadata_from_tei
from .layout import (
    choose_best_page,
    detect_figure_regions,
    extract_figure_number,
    find_plate_pages,
    render_pdf_pages,
)
from .m3_engine import CaptionPair, M3Engine, PanelBox, PanelMatch
from .ocr import OCRBackend, normalize_ocr_tokens
from .provenance.stamp import build_provenance
from .range_chart_extractor import (
    RangeChartResult,
    build_geology_links_for_panels,
    classify_figure_type,
    extract_range_chart,
)
from .scale_bar import (
    detect_scale_bar_length_px,
    extract_scale_from_caption,
    extract_scale_from_ocr_text,
    merge_scale_info,
)
from .schema_models import ProvenanceRecord
from .segmentation import PanelSegmenter, SegmentationConfig
from .taxon import TaxonRecognizer
from .text_filters import (
    looks_like_placeholder_caption as _looks_like_placeholder_caption,
)
from .types import (
    CaptionEntity,
    CaptionRecord,
    FigureRegion,
    MatchResult,
    PanelCandidate,
    PaperMetadata,
)
from .utils import ensure_dir, slugify, stable_id, write_json, write_jsonl


class RadiolarianPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        progress_callback=None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        # Plan D: explicit --use-neural-matcher gate. Setting the flag
        # without --matcher-checkpoint-path silently falls back to the
        # heuristic matcher (the NeuralGraphMatcher class is real but
        # inaccessible without a trained checkpoint). Warn loudly so
        # operators don't think the neural path is engaged when it isn't.
        if bool(self.config.extra.get("use_neural_matcher", False)) and not self.config.extra.get(
            "matcher_checkpoint_path"
        ):
            logger.warning(
                "--use-neural-matcher set but no --matcher-checkpoint-path provided; "
                "falling back to heuristic matcher. Train one via "
                "scripts/train_matcher.py to enable the neural path."
            )
        # Optional progress callback: ``cb(current, total, message)``.
        # ``current`` and ``total`` are 0-indexed ints; the API uses
        # ``current/total`` to map a real pipeline position onto the 30-90%
        # band of the job progress.
        self._progress_cb = progress_callback
        # Phase 42: cooperative cancellation. When set, ``run()`` polls
        # this event between PDFs and between progress ticks; if the
        # event is set (e.g. the GUI's Cancel button), ``run()`` cancels
        # in-flight futures and returns the rows processed so far
        # instead of raising KeyboardInterrupt. This lets the GUI
        # show a clean "cancelled" state and free the worker thread.
        self._cancel_event = cancel_event
        # Phase 29: forward retry + timeout knobs from the config
        # ``extra`` dict. Defaults match legacy behaviour (3 retries,
        # 300s timeout) — only operators who pass ``--grobid-max-retries``
        # or ``--grobid-timeout`` see different behaviour.
        self.grobid = GrobidClient(
            server_url=config.grobid_url,
            timeout=int(self.config.extra.get("grobid_timeout", 300)),
            max_retries=int(self.config.extra.get("grobid_max_retries", 3)),
            # Phase 59 (Bug 2.2): forward cancel_event so the GROBID
            # retry loop honours user cancellation.
            cancel_event=cancel_event,
        )
        # Phase 29: cycle guard. ``_process_one_pdf_grobid`` may now
        # call ``_process_one_pdf_od`` on failure, and ``_process_one_pdf_od``
        # calls ``_process_one_pdf_grobid`` on its own failure. Without
        # this set, GROBID-down + OD-down would loop indefinitely.
        # The set tracks paper_ids currently in the GROBID code path
        # so OD can detect re-entry and skip the recursive GROBID call.
        self._grobid_in_progress: set[str] = set()
        # Phase 59 (Bug 2.1): ``_grobid_in_progress`` is read at L861 and
        # modified at L1960/L1967 from ``ThreadPoolExecutor`` workers
        # without any ``threading.Lock``. Race conditions can: (a) miss
        # cycle detection, (b) leak entries, (c) cause GROBID↔OD
        # infinite recursion. All reads and writes to the cycle-guard
        # set MUST go through ``self._grobid_lock``.
        self._grobid_lock = threading.Lock()
        # audit 2026-07-31: per-thread fallback depth for the
        # GROBID↔OD recursion. The ``_grobid_in_progress`` guard only
        # covers the OD-failure branch (L956); the OD "success but
        # zero results" branches (L1017, L1490) re-enter GROBID
        # without any guard, so GROBID-down + OD-empty looped
        # indefinitely (RecursionError after hours). The depth
        # counter bounds the whole fallback chain to one
        # GROBID→OD→GROBID hop (depth 1 → 2 → 3 is refused).
        self._od_grobid_depth = threading.local()
        # Phase 27: forward the configured OCR language list. Default
        # ``"en"`` keeps the legacy English-only flow identical. JA
        # papers pass ``--ocr-lang en,ja`` and EasyOCR / PaddleOCR both
        # see the JA model. See ``src/rlpe/ocr.py`` for the
        # normalisation logic and the PaddleOCR ``ja → japan`` mapping.
        self.ocr = OCRBackend(
            backend=config.ocr_backend,
            use_gpu=config.use_gpu,
            lang=config.extra.get("ocr_lang", "en"),
        )
        self.taxon = TaxonRecognizer(
            model=config.taxon_model,
            hf_model_path=config.extra.get("taxon_hf_model_path"),
            lexicon_path=config.extra.get("taxon_lexicon_path"),
        )
        self.segmenter = PanelSegmenter(
            config=SegmentationConfig(
                score_threshold=config.min_panel_score,
                grid_size=int(config.extra.get("sam2_grid_size", 6)),
                max_point_prompts=int(config.extra.get("sam2_max_point_prompts", 48)),
                max_box_prompts=int(config.extra.get("sam2_max_box_prompts", 24)),
            ),
            checkpoint=config.extra.get("sam2_checkpoint"),
            model_cfg=config.extra.get("sam2_model_cfg"),
        )
        self._od_extractor = None
        self._od_lock = threading.Lock()
        self.gemma_runtime = None
        # M3Engine: 5-stage semantic engine. Initialized only when M3 backend
        # is available AND the user explicitly opts in via
        # ``m3_enhanced_mode = True`` (Round 16 audit: was asymmetric — ON
        # by default for MiniMax, opt-in for others; now opt-in for all).
        self.m3_engine: M3Engine | None = None
        self._gemma_lock = threading.Lock()
        # NOTE: no shared _ocr_lock here. PaddleOCR (the default backend) is
        # thread-safe for concurrent .ocr() calls — the engine serializes its
        # own internal state and per-call locks would just serialize our
        # workers for no benefit. EasyOCR is the exception; for that backend,
        # ``OCRBackend.recognize`` uses its own engine-instance lock. The
        # segmenter (SAM2) and gemma runtimes are NOT concurrent-safe, so
        # they retain per-pipeline locks.
        self._seg_lock = threading.Lock()
        # Audit 2026-08-02: paper-level morphology records produced
        # by Stage 6 enrichment. Keyed by paper_id so the run() loop
        # can merge them into the canonical ``run_output.json`` after
        # all per-paper processing completes. The accumulator is
        # populated by ``_apply_morphology_enrichment`` (only fires
        # when ``m3_stage_6=True`` and a M3 backend is available) and
        # drained by ``run()`` when assembling ``run_output_dict``.
        # Plain dict — single-threaded accumulation (per-paper work
        # is serialised by the executor), drained in ``run()`` once
        # the workers have all returned.
        self._paper_morphologies: dict[str, list[dict[str, Any]]] = {}
        # Phase 59 (Bug 2.5): serialise progress-callback invocations.
        # Multiple worker threads can finish PDFs concurrently and
        # invoke ``_progress_cb`` simultaneously; without this lock,
        # Qt signal dispatch in the GUI can interleave updates.
        self._progress_lock = threading.Lock()
        # Fallback handler for MiniMax API errors (None when not using MiniMax)
        self.gemma_fallback_handler = None
        # Secondary Gemma runtime used as fallback target (lazy-init on first error)
        self._fallback_gemma_runtime = None
        # Round 18 audit: ANTHROPIC_API_KEY is the project's documented
        # .env key (Claude-Code-compatible name). If the user has
        # ANTHROPIC_API_KEY but no MiniMax_api_key / MINIMAX_API_KEY,
        # inject the Anthropic env var into the pipeline config so
        # downstream LLM backend builders can see it. Done here so
        # _try_init_gemma (which builds the MiniMaxM3Backend) picks
        # it up automatically.
        if (
            not self.config.extra.get("MiniMax_api_key")
            and not os.environ.get("MINIMAX_API_KEY")
            and os.environ.get("ANTHROPIC_API_KEY")
        ):
            self.config.extra["MiniMax_api_key"] = os.environ.get("ANTHROPIC_API_KEY")
            logger.info(
                "Pipeline: using ANTHROPIC_API_KEY as MiniMax_api_key (Anthropic env-var fallback)"
            )
        self._try_init_gemma()

    @property
    def od_extractor(self):
        """Lazy-init OpenDataLoader extractor."""
        if self._od_extractor is None:
            with self._od_lock:
                if self._od_extractor is None:
                    from .opendataloader_extractor import OpenDataLoaderExtractor

                    self._od_extractor = OpenDataLoaderExtractor(
                        use_ocr=bool(self.config.extra.get("od_use_ocr", False)),
                        ocr_lang=str(self.config.extra.get("od_ocr_lang", "en")),
                        merge_gap_pt=float(self.config.extra.get("od_merge_gap_pt", 72.0)),
                        # Phase 28: forward the OD path page-distance
                        # limit. Default 5 on PipelineConfig; CLI
                        # overrides via ``--od-caption-window``.
                        caption_window=self.config.od_caption_window,
                    )
        return self._od_extractor

    def _try_init_gemma(self) -> None:
        # Two distinct initialization paths:
        #   1. Local Gemma4 / llama.cpp / Ollama — requires use_gemma4=True
        #      or an explicit model_path (legacy behavior).
        #   2. MiniMax cloud backend — requires ONLY a MiniMax API key;
        #      does NOT need use_gemma4=True.  The previous version
        #      incorrectly required use_gemma4=True for ALL backends,
        #      which meant MiniMax (the default cloud path) silently
        #      produced zero LLM calls unless the user also passed
        #      --use-gemma4.
        minimax_backends = {"minimax", "minimax-m3", "minimax_api"}
        backend_name = str(self.config.extra.get("llm_backend") or "").lower() or "transformers"
        has_minimax_key = bool(
            self.config.extra.get("MiniMax_api_key") or os.environ.get("MINIMAX_API_KEY")
        )
        # Round 16 audit: ANTHROPIC_API_KEY used to be a fallback
        # source. That silently routed Claude Code users to MiniMax
        # with no warning. Removed from the chain — a user who only
        # has ANTHROPIC_API_KEY must set MINIMAX_API_KEY or
        # MiniMax_api_key explicitly. If they do, log a notice so
        # the source is observable.
        if (
            not has_minimax_key
            and os.environ.get("ANTHROPIC_API_KEY")
            and not self.config.extra.get("MiniMax_api_key")
            and not os.environ.get("MINIMAX_API_KEY")
        ):
            logger.info(
                "ANTHROPIC_API_KEY is set but not consumed by MiniMax "
                "path (vendor-specific key required); set MiniMax_api_key "
                "or MINIMAX_API_KEY explicitly to enable MiniMax."
            )
        # MiniMax path: either explicit backend name, OR no local model
        # path but a MiniMax API key present (the common "just give me
        # a key and hit the cloud" flow).
        has_local_model = bool(
            self.config.extra.get("gemma_model_path")
            or self.config.extra.get("ollama_model")
            or self.config.extra.get("llama_model")
        )
        use_minimax = backend_name in minimax_backends or (has_minimax_key and not has_local_model)
        if not use_minimax and not self.config.extra.get("use_gemma4", False):
            return
        model_path = self.config.extra.get("gemma_model_path") or self.config.extra.get(
            "ollama_model"
        )
        if not use_minimax and not model_path and backend_name not in {"ollama"}:
            return
        try:
            # If we detected MiniMax heuristically (API key present, no
            # local model), make sure the builder sees backend=minimax.
            # Otherwise build_gemma_backend_from_config defaults to
            # "transformers", which triggers load_gemma4_model →
            # BitsAndBytes import → crash in envs without the
            # transformers stack.
            if use_minimax and backend_name not in minimax_backends:
                self.config.extra["llm_backend"] = "minimax"
                backend_name = "minimax"  # sync local for FallbackHandler gate below
            self.gemma_runtime = build_gemma_backend_from_config(self.config.extra)
            # audit 2026-07-31: wire the configured fallback backend
            # name into the backend so a 4xx error can raise
            # FallbackRecommendedError with the name attached. Without
            # this the fallback feature was dead code (nothing ever
            # called set_fallback_backend).
            fb_name = self.config.extra.get("fallback_llm_backend")
            if fb_name and hasattr(self.gemma_runtime, "backend"):
                setter = getattr(self.gemma_runtime.backend, "set_fallback_backend", None)
                if setter is not None:
                    try:
                        setter(fb_name)
                    except Exception:
                        logger.debug("set_fallback_backend unavailable", exc_info=True)
            # If MiniMax backend, attach a FallbackHandler. The handler is
            # invoked ONLY from ``_apply_gemma_with_fallback``; we intentionally
            # do NOT also wire it into ``backend.on_error`` to avoid the
            # handler being called twice for the same error.
            if backend_name in minimax_backends:
                external = self.config.extra.get("_MiniMax_external_handler")
                if external is not None:
                    handler = external
                else:
                    from .llm_backends import FallbackHandler

                    default_action = str(self.config.extra.get("MiniMax_fallback_default", "rules"))
                    handler = FallbackHandler(default_action=default_action)
                    if bool(self.config.extra.get("MiniMax_interactive", False)):
                        from .llm_backends import cli_fallback_prompt

                        handler.on_error = cli_fallback_prompt
                self.gemma_fallback_handler = handler
                logger.info(
                    "MiniMax M3 backend ready (default_fallback=%s interactive=%s)",
                    handler.default_action,
                    bool(self.config.extra.get("MiniMax_interactive", False)),
                )
        except Exception as exc:
            self.gemma_runtime = None
            self.config.extra["gemma_init_error"] = str(exc)
            logger.warning("Gemma4 backend init failed: %s", exc)
            return

        # Build the M3 semantic engine (5-stage). Round 16 audit: was
        # asymmetric — auto-enabled for MiniMax backend, opt-in for
        # others. Made symmetric: opt-in for ALL backends via
        # ``m3_enhanced_mode = True`` so no vendor gets a privileged
        # default. Users who previously relied on the MiniMax auto-
        # enable must set m3_enhanced_mode=True in their config.
        if self.gemma_runtime is not None:
            want_m3 = self.config.extra.get("m3_enhanced_mode", False)
            if want_m3:
                m3_cfg = {k: v for k, v in self.config.extra.items() if k.startswith("m3_")}
                # If user didn't set stage toggles, enable all 5 by default.
                m3_cfg.setdefault("m3_stage_1", True)
                m3_cfg.setdefault("m3_stage_2", True)
                m3_cfg.setdefault("m3_stage_3", True)
                m3_cfg.setdefault("m3_stage_4", True)
                m3_cfg.setdefault("m3_stage_5", True)
                # Diagnostic dump directory (overridable from env)
                import os as _os

                diag = _os.environ.get("RLPE_M3_DIAG_DIR")
                if diag:
                    m3_cfg.setdefault("m3_diagnostic_dir", diag)
                self.m3_engine = M3Engine(
                    backend=self.gemma_runtime.backend,
                    config=m3_cfg,
                    # audit 2026-08-01 (M16): forward the pipeline's
                    # cancel_event so the engine's retry-loop back-off
                    # honours user cancellation. Without this, the
                    # engine sits out 30s+ sleeps per failed call while
                    # the user waits for a Cancel that already
                    # arrived. ``None`` for callers that don't
                    # construct with a cancel_event (e.g. legacy
                    # tests / single-shot CLI invocations).
                    cancel_event=self._cancel_event,
                )
                logger.info(
                    "M3Engine initialized (stages 1-5: %s/%s/%s/%s/%s, diag=%s)",
                    m3_cfg.get("m3_stage_1"),
                    m3_cfg.get("m3_stage_2"),
                    m3_cfg.get("m3_stage_3"),
                    m3_cfg.get("m3_stage_4"),
                    m3_cfg.get("m3_stage_5"),
                    m3_cfg.get("m3_diagnostic_dir"),
                )

    def prepare_dirs(self) -> None:
        ensure_dir(self.config.resolved_output_dir())
        ensure_dir(self.config.tei_dir())
        ensure_dir(self.config.figures_dir())
        ensure_dir(self.config.panels_dir())
        ensure_dir(self.config.manifests_dir())

    def _emit_progress(self, current: int, total: int, message: str) -> None:
        # Phase 59 (Bug 2.5): hold ``_progress_lock`` so concurrent
        # worker threads serialise callback invocations. Without the
        # lock, multiple workers finishing PDFs at the same time can
        # call into Qt's signal dispatcher in parallel, leading to
        # interleaved updates and progress-bar regressions
        # ("Completed 3/4" before "Completed 1/4").
        if self._progress_cb is not None:
            with self._progress_lock:
                self._progress_cb(current, total, message)

    def _collect_llm_usage(self) -> dict[str, Any] | None:
        """Thin wrapper around :func:`rlpe.llm_usage.collect_llm_usage`.

        Kept as a method so test code can construct a pipeline via
        ``__new__`` and inject a runtime attribute, without running the
        full ``__init__`` chain.
        """
        from .llm_usage import collect_llm_usage

        return collect_llm_usage(getattr(self, "gemma_runtime", None))

    def run(self) -> list[dict[str, Any]]:
        self.prepare_dirs()
        pdf_files = sorted(self.config.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            return []

        # Phase 42: cooperative cancellation. If the caller passes
        # a ``cancel_event`` (threading.Event), we check it between
        # PDFs. Setting the event causes us to cancel all in-flight
        # futures and return the rows processed so far — much
        # friendlier than raising KeyboardInterrupt.
        cancel_event = self._cancel_event

        rows: list[dict[str, Any]] = []
        total = len(pdf_files)
        completed = 0
        # Fire one initial tick so the UI can show "started" before the first
        # PDF actually finishes.
        self._emit_progress(0, total, f"Starting pipeline ({total} PDF(s))")
        # Phase 59 (Bug 2.3): pool lifecycle is now manual so the
        # cancel branch can call ``pool.shutdown(wait=False,
        # cancel_futures=True)``. The previous ``with ThreadPoolExecutor(...)``
        # form called ``shutdown(wait=True)`` on exit, which blocked
        # until every running worker (especially long LLM API calls)
        # had finished — a 30s sleep per PDF meant 4 PDFs blocked for
        # 2 minutes after the user clicked Cancel.
        cancelled_fast = False  # audit 2026-07-26 M6: set by the cancel branch
        pool = ThreadPoolExecutor(max_workers=max(1, self.config.num_workers))
        try:
            futures = {pool.submit(self._process_one_pdf, p): p for p in pdf_files}
            try:
                # Phase 42: also check cancel_event at the top of the
                # loop so a Cancel that arrives BEFORE any PDF
                # completes still short-circuits the run.
                for fut in as_completed(futures):
                    if cancel_event is not None and cancel_event.is_set():
                        # Phase 59 (Bug 2.3): fast shutdown. Cancel
                        # any futures that haven't started yet and
                        # return immediately — don't wait for running
                        # workers (they may be stuck in an LLM API
                        # call with a 30s+ timeout).
                        self._emit_progress(
                            completed,
                            total,
                            f"Cancelled by user after {completed}/{total} PDFs",
                        )
                        # Audit 2026-07-26 M6+M7: mark fast-shutdown so
                        # the finally block does NOT call shutdown(
                        # wait=True) (which would block up to 30s on an
                        # in-flight LLM call), and write the manifest
                        # here so already-completed PDFs aren't lost -
                        # the post-finally write below is unreachable on
                        # this return path.
                        cancelled_fast = True
                        pool.shutdown(wait=False, cancel_futures=True)
                        write_jsonl(self.config.manifests_dir() / "matches.jsonl", rows)
                        return rows
                    pdf = futures[fut]
                    if fut.cancelled():
                        # Future was cancelled (e.g. the API sent a cancel
                        # request and the executor pre-empted the worker).
                        # Don't try to extract a result and don't log a
                        # spurious "PDF processing failed" line.
                        completed += 1
                        continue
                    try:
                        result_rows = fut.result()
                    except (KeyboardInterrupt, SystemExit, PipelineCancelledError):
                        # User-initiated cancellation or PipelineCancelledError.
                        # Cancel in-flight workers and propagate so the CLI exits
                        # with a proper traceback and the API can flip the job
                        # to ``cancelled`` (the API's own cancel path doesn't
                        # go through ``run()``, but the API may also be
                        # wrapping this method).
                        cancelled_fast = True  # audit M6: don't let finally wait=True
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception:
                        logger.exception("PDF processing failed; continuing with remaining PDFs")
                    else:
                        rows.extend(result_rows)
                    completed += 1
                    self._emit_progress(
                        completed,
                        total,
                        f"Processed {pdf.name} ({len(rows)} matches so far)",
                    )
            except (KeyboardInterrupt, SystemExit):
                # Same handling if the cancel happens between ``as_completed``
                # yields (e.g. signal handler fires while we're idle waiting
                # for the next future).
                cancelled_fast = True  # audit M6: don't let finally wait=True
                pool.shutdown(wait=False, cancel_futures=True)
                raise
        finally:
            # Audit 2026-07-26 M6: honour the cancel branch's
            # fast-shutdown intent - use wait=False when cancelled so
            # we don't block on in-flight LLM calls; otherwise wait=True
            # so remaining futures drain cleanly.
            try:
                pool.shutdown(wait=not cancelled_fast)
            except Exception:
                # shutdown may already have been called from the cancel
                # branch — ignore the resulting RuntimeError.
                pass

        manifest_path = self.config.manifests_dir() / "matches.jsonl"
        write_jsonl(manifest_path, rows)
        # Canonical data package (matches.jsonl is raw per-row; run_output.json
        # is the validated, deduped, schema-shaped bundle that downstream
        # consumers — web UI, CSV/DwC-A exporters, ML splits — read from).
        # We swallow any error here so a broken ``run_output.json`` never
        # invalidates the row-level ``matches.jsonl`` that other tooling
        # already depends on.
        if rows:
            try:
                match_results = [match_result_from_dict(d) for d in rows]
                provenance_internal = build_provenance(self.config, pdf_files)
                provenance_record = ProvenanceRecord(**provenance_internal.to_dict())
                # Audit 2026-08-02: forward Stage-6 morphology records
                # collected by ``_apply_morphology_enrichment`` into
                # the canonical RunOutput. ``_paper_morphologies`` is
                # keyed by paper_id; flatten into a single list. An
                # empty list is the no-op path (Stage 6 off / no
                # paper-level work fired).
                paper_morphologies: list[dict[str, Any]] = []
                for recs in self._paper_morphologies.values():
                    paper_morphologies.extend(recs)
                run_output_dict = run_output_from_provenance(
                    provenance_record,
                    match_results,
                    paper_morphologies=paper_morphologies,
                )
                write_json(manifest_path.parent / "run_output.json", run_output_dict)
            except Exception:
                logger.exception("Failed to write run_output.json; matches.jsonl is unaffected")
            # Run-level LLM usage sidecar. Independent of RunOutput schema
            # so /system/llm-status and the audit trail can see the actual
            # MiniMax call / token / cost totals even before the per-row
            # propagation lands. Failures here must NEVER invalidate
            # matches.jsonl / run_output.json.
            try:
                summary = self._collect_llm_usage()
                if summary:
                    write_json(manifest_path.parent / "llm_usage.json", summary)
            except Exception:
                logger.exception("Failed to write llm_usage.json; matches.jsonl is unaffected")
        self._emit_progress(total, total, f"Done — {len(rows)} matches")
        return rows

    def _process_one_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        paper_id = stable_id(pdf_path)
        self._emit_progress(0, 1, f"Loading {pdf_path.name}…")

        # ------ OpenDataLoader path (opt-in) -----------------------------------
        if self.config.extra.get("use_opendataloader", False):
            rows = self._process_one_pdf_od(paper_id, pdf_path)
        else:
            # ------ GROBID + layout path (default) -----------------------------
            rows = self._process_one_pdf_grobid(paper_id, pdf_path)

        # Phase 65 Plan A.4: cross-figure linker — link each plate panel
        # to the paper's strat column / litholog / paleogeographic map
        # via Sample ID direct match → Locality share → M3 inference.
        # Runs after all figure extraction so the linker sees the
        # complete geology-link context. No-op if the config flag is off
        # (default on).
        if self.config.extra.get("cross_figure_linker_enabled", True):
            try:
                rows = self._apply_cross_figure_linker(rows, paper_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "cross_figure_linker failed for paper=%s: %s",
                    paper_id,
                    exc,
                )

        self._emit_progress(1, 1, f"Finished {pdf_path.name} ({len(rows)} matches)")
        return rows

    # -----------------------------------------------------------------------
    # OpenDataLoader-based processing
    # -----------------------------------------------------------------------

    def _cross_link_map_and_range_chart(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Link map locations to range-chart section names.

        Conservative matcher: a panel's range-chart section field
        ("SK-01") is linked to a map location only when the
        section name is a substring of the location, the location
        is a substring of the section, or the first two characters
        of one are the first two characters of the other. Each
        match is tagged with the rule it satisfied so an operator
        can filter the loose ones (e.g. "S" matching "Sikhote",
        "Saharan" — the second-letter test would split them).

        Range-chart sections that are 2-3 char codes ("SK-01",
        "NS-01") typically correspond to the first letters of the
        full place name. When no match is found, the section code
        is still preserved on each panel's ``geology_links`` and
        the map context is recorded as paper-level metadata, so an
        operator can still reconcile them by hand.
        """
        # Collect map locations per paper.
        map_locs_by_paper: dict[str, list[tuple[str, str]]] = {}
        for r in results:
            if r.get("panel_id") != "MAP_CONTEXT":
                continue
            pid = r.get("paper_id")
            md = r.get("metadata") or {}
            for loc in md.get("location_names") or []:
                if loc:
                    map_locs_by_paper.setdefault(pid, []).append((loc, r.get("figure_id", "")))
        if not map_locs_by_paper:
            return results

        for r in results:
            pid = r.get("paper_id")
            if r.get("panel_id") == "MAP_CONTEXT":
                continue
            md = r.get("metadata") or {}
            sections = set()
            for link in md.get("geology_links") or []:
                sec = link.get("locality")
                if sec and "range_chart" in link.get("evidence_text", ""):
                    sections.add(sec)
            if not sections:
                continue
            map_locs = map_locs_by_paper.get(pid, [])
            matched: list[dict[str, str]] = []
            for sec in sections:
                # Extract just the alpha prefix (drop the "-01" suffix).
                sec_alpha = "".join(c for c in sec if c.isalpha())
                if not sec_alpha:
                    continue
                for loc, fig_id in map_locs:
                    # Rule 1: case-insensitive exact substring match.
                    if sec.lower() in loc.lower() or loc.lower() in sec.lower():
                        matched.append(
                            {
                                "section": sec,
                                "location": loc,
                                "match_type": "substring",
                                "map_figure": fig_id,
                            }
                        )
                        continue
                    # Rule 2: first 2 characters of section alpha prefix
                    # match first 2 characters of any word in the
                    # location. E.g. "SK" → "Sikhote", "NS" →
                    # "Nadanhada South" if "Nadanhada" or "South"
                    # both start with NS — this is rare; the
                    # substring rule is the workhorse.
                    if len(sec_alpha) >= 2:
                        s2 = sec_alpha[:2].lower()
                        for word in loc.split():
                            if word[:2].lower() == s2:
                                matched.append(
                                    {
                                        "section": sec,
                                        "location": loc,
                                        "match_type": "prefix2",
                                        "map_figure": fig_id,
                                    }
                                )
                                break
                    # Rule 3: section code letters match the first
                    # letters of successive words in a hyphenated
                    # location name. E.g. "SK" → "Sikhote-Khabarovsk"
                    # (S + K are the first letters of each word).
                    # This catches the common radiolarian-paper
                    # convention where the range-chart section code
                    # is an acronym of the section's full name.
                    if len(sec_alpha) >= 2 and "-" in loc:
                        words = [
                            w for w in loc.replace("Range", "").replace("River", "").split("-") if w
                        ]
                        if len(words) == len(sec_alpha):
                            if all(
                                w[:1].lower() == sec_alpha[i].lower()
                                for i, w in enumerate(words)
                                if w
                            ):
                                matched.append(
                                    {
                                        "section": sec,
                                        "location": loc,
                                        "match_type": "acronym",
                                        "map_figure": fig_id,
                                    }
                                )
            if matched:
                # Deduplicate by (section, location, match_type)
                seen = set()
                deduped = []
                for m in matched:
                    key = (m["section"], m["location"], m["match_type"])
                    if key not in seen:
                        seen.add(key)
                        deduped.append(m)
                md.setdefault("matched_location", []).extend(deduped)
        return results

    def _process_range_chart(
        self,
        *,
        paper_id: str,
        figure_id: str,
        caption_text: str,
        image_path: str,
    ) -> list[dict[str, Any]]:
        """Run range-chart extraction and produce stub panel records.

        The vision extractor returns a RangeChartResult with sections,
        species_ranges, biozones, and other_fossils. We wrap each
        species_range into a stub panel record that carries the
        geology as a ``geology_links`` entry, matching the downstream
        PanelRecord schema. These stubs are useful for downstream
        consumers (DwC export, web UI) without needing a separate
        range-chart data path.

        The stub has ``panel_id="RANGE_CHART"`` so it can be filtered
        out of the standard "per-panel species" evaluation — it does
        not represent a real specimen panel, only a geological context
        anchor.
        """
        # Source API config: read directly from the environment so this
        # works even when ``self.gemma_runtime`` is not initialised
        # (the MiniMax-M3 vision path is independent of the local
        # Gemma4 loader).
        # Phase 55 audit: config takes priority over env vars so users can
        # override ANTHROPIC_API_KEY (project-wide) with a per-run MiniMax_api_key.
        api_key = self.config.extra.get("MiniMax_api_key") or os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or self.config.extra.get(
            "MiniMax_endpoint", "https://api.minimaxi.com/anthropic"
        )
        model = os.environ.get("ANTHROPIC_MODEL") or self.config.extra.get(
            "MiniMax_model", "MiniMax-M3"
        )
        if not api_key:
            logger.warning(
                "range_chart: no ANTHROPIC_API_KEY set; skipping %s/%s",
                paper_id,
                figure_id,
            )
            return [
                {
                    "paper_id": paper_id,
                    "figure_id": figure_id,
                    "panel_id": "_RANGE_CHART_SKIPPED_NO_API_KEY",
                    "species": None,
                    "panel_path": None,
                    "bbox": None,
                    "confidence": 0.0,
                    "label_text": None,
                    "caption_snippet": caption_text[:200] if caption_text else None,
                    "ocr_text": None,
                    "paper_metadata": None,
                    "metadata": {
                        "extraction_source": "range_chart",
                        "skip_reason": "no_ANTHROPIC_API_KEY",
                    },
                }
            ]

        chart = extract_range_chart(
            paper_id=paper_id,
            figure_id=figure_id,
            caption=caption_text,
            image_path=image_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        # Build stub panel records (one per species_range entry) so
        # downstream code can join by figure_id. Each stub carries the
        # full RangeChartResult in ``metadata.range_chart`` so the web
        # UI can display the chart-level context.
        out: list[dict[str, Any]] = []
        for sr in chart.species_ranges:
            stub = {
                "paper_id": paper_id,
                "figure_id": figure_id,
                "panel_id": "RANGE_CHART",
                "species": sr.species,
                "panel_path": None,
                "bbox": None,
                "confidence": chart.confidence,
                "label_text": None,
                "caption_snippet": caption_text[:240] if caption_text else None,
                "ocr_text": None,
                "paper_metadata": None,
                "figure_type": "range_chart",
                "metadata": {
                    "extraction_method": "range_chart_vision",
                    "extraction_source": "range_chart",
                    "figure_type": "range_chart",
                    "section": sr.section,
                    "range_top": sr.range_top,
                    "range_base": sr.range_base,
                    "biozone": sr.biozone,
                    "range_chart": chart.to_dict(),
                },
            }
            out.append(stub)
        return out

    @staticmethod
    def _find_orphan_image_for_range_chart(
        figures: list[Any], target: Any, od_raw: dict[str, Any] | None = None
    ) -> str | None:
        """Search for an orphan figure-image to associate with a range chart
        whose own ``image_paths`` came back empty.

        OD sometimes extracts the range-chart image but fails to
        associate it with its caption (the chart isn't a single
        embedded image in the PDF text layer, so
        ``_find_nearest_caption`` gives up). In that case the image
        sits in the OD ``..._images/`` directory unreferenced by any
        figure. We need a stronger search than the figure-level
        orphan loop: walk the raw OD JSON's image elements, find the
        one whose ``page number`` matches the range-chart caption's
        page, and return its resolved file path.

        The search order is:
          1. Images on the same page as the range-chart caption
             (the chart is virtually always on the same page as its
             "Fig. N" caption in radiolarian papers).
          2. Images on adjacent pages.
          3. The largest orphan-figure image from ``figures`` (the
             legacy fallback that handles the case where OD at least
             paired the image with a stub figure).

        Returns the absolute path of the best candidate, or None.
        """
        target_page = int(target.page_number)
        import os as _os

        # Phase 1: scan the raw OD JSON for images on the same page
        # that are NOT referenced by any figure. This is the most
        # common failure mode for range charts (OD extracts the image
        # but the caption-image association falls apart).
        # (page_diff, size, path, is_referenced?)
        # is_referenced is optional — raw OD scan paths (line ~879)
        # Sort key only uses first two elements so the missing 4th
        # element is safe. Named tuple not used here to avoid
        # dragging an extra import into the pipeline module.
        unpaired: list[tuple[int, int, str, bool | None]] = []
        logger.debug(
            "orphan search for range_chart page=%d (od_raw=%s, figures=%d)",
            target_page,
            bool(od_raw),
            len(figures),
        )
        if od_raw:
            try:
                from .opendataloader_extractor import _iter_all_elements

                # Collect all image paths that are referenced by figures
                referenced: set[str] = set()
                for fig in figures:
                    for p in fig.image_paths or []:
                        referenced.add(_os.path.basename(p))
                # Walk the raw JSON for all image elements
                kids = od_raw.get("kids") or []
                # images_dir is constructed by the OD extractor under
                # <output_dir>/od_output/<paper_id>/<pdf_stem>_images.
                # Derive it from the figures' image paths because this
                # helper is nested inside run-level logic and does not
                # own a self/config reference. Round 9 (L3): previously
                # this block ran TWICE (lines 706-709 and 735-738) with
                # identical content; the second run silently overwrote
                # the first. Compute once and reuse.
                images_dir = None
                if figures and figures[0].image_paths:
                    sample = figures[0].image_paths[0]
                    # <work>/od_output/<paper_id>/<pdf_stem>_images/imageFileN.png
                    images_dir = _os.path.dirname(sample)
                # Enumerate ALL images in the directory (not just
                # unpaired ones — OD sometimes wrongly pairs a range
                # chart image with a different figure's caption, and
                # those "stolen" images are exactly what we need). The
                # ``referenced`` set is used only to log which images
                # are already associated, NOT to filter candidates.
                #
                # We pair file names to OD image elements by
                # alphabetical/sequential order — OD exports
                # imageFile1.png, imageFile2.png, ... in the order it
                # encountered the <image> elements in the PDF. The raw
                # JSON's image list gives us each element's page
                # number, so we map the i-th file to the i-th image
                # element's page. This mapping is correct for the
                # common case where OD doesn't reorder images.
                od_image_pages: list[int] = []
                for el in _iter_all_elements(kids):
                    if el.get("type") == "image":
                        p = int(el.get("page number", 0))
                        if p > 0:
                            od_image_pages.append(p)
                # audit 2026-08-01 (M10): single ``images_dir`` scan. The
                # previous code listed the directory twice — first to
                # compute ``page_diff`` from ``od_image_pages``, then
                # again to assign a flat ``page_diff=0`` to every
                # un-referenced file. The second pass dominated the
                # sort key: an orphan on the correct page was being
                # sorted as ``page_diff=0`` (same as all un-referenced
                # images) so the final pick was made purely on file
                # size. Merge into one scan: each image gets its
                # ``page_diff`` computed once (from ``od_image_pages``
                # if available, else a small filename-based heuristic
                # that prefers PNGs whose numeric suffix looks like
                # the target page).
                if images_dir and _os.path.isdir(images_dir):
                    png_files = sorted(
                        f
                        for f in _os.listdir(images_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    )
                    for i, fname in enumerate(png_files):
                        fpath = _os.path.join(images_dir, fname)
                        try:
                            sz = _os.path.getsize(fpath)
                        except OSError:
                            sz = 0
                        is_referenced = fname in referenced
                        # Prefer ``od_image_pages[i]`` (the correct
                        # mapping) when in range; fall back to a
                        # filename-derived page number (extract any
                        # digits from the stem); only then to ``999``
                        # (truly unknown).
                        if i < len(od_image_pages) and od_image_pages[i]:
                            img_page = od_image_pages[i]
                        else:
                            img_page = _page_from_filename(fname) or 0
                        page_diff = abs(img_page - target_page) if img_page else 999
                        unpaired.append((page_diff, sz, fpath, is_referenced))
                        logger.debug(
                            "raw OD image: %s size=%d page=%d page_diff=%d referenced=%s",
                            fpath,
                            sz,
                            img_page,
                            page_diff,
                            is_referenced,
                        )
            except Exception as exc:
                logger.debug("raw OD scan failed: %s", exc)

        # Phase 2: figure-level orphans (legacy path). These are
        # figures that OD paired with a stub but no caption.
        for fig in figures:
            if fig is target:
                continue
            if not (fig.image_paths) or (fig.caption_text or "").strip():
                continue
            page_diff = abs(int(fig.page_number) - target_page)
            for img_path in fig.image_paths or []:
                try:
                    sz = _os.path.getsize(img_path)
                except OSError:
                    sz = 0
                unpaired.append((page_diff, sz, img_path, True))

        if not unpaired:
            return None
        # Sort: prefer smallest page_diff (raw OD scan has all 0, so it
        # becomes a pure size sort). Among same-page_diff, prefer
        # SMALLEST size — range charts are line drawings and are
        # typically smaller than plate images (plates have dense
        # detail → bigger PNGs). This is the opposite of the
        # previous heuristic and is correct for the most common case
        # where OD has wrongly paired the chart with a different
        # figure (in which case the chart is "stolen" and would be
        # filtered out by a referenced-only search).
        unpaired.sort(key=lambda t: (t[0], t[1]))
        chosen = unpaired[0][2]
        logger.info(
            "orphan search: chose %s (page_diff=%d, size=%d) from %d candidates",
            chosen,
            unpaired[0][0],
            unpaired[0][1],
            len(unpaired),
        )
        return chosen

    def _enter_od_grobid_guard(self, paper_id: str, path_name: str) -> bool:
        """Enter the OD↔GROBID fallback chain; False means the chain is
        already too deep and the caller must NOT recurse further."""
        depth = getattr(self._od_grobid_depth, "depth", 0) + 1
        self._od_grobid_depth.depth = depth
        if depth >= 3:
            logger.warning(
                "OD↔GROBID fallback cycle detected for %s (%s at depth=%d); "
                "abandoning recursive fallback.",
                paper_id,
                path_name,
                depth,
            )
            return False
        return True

    def _exit_od_grobid_guard(self) -> None:
        depth = getattr(self._od_grobid_depth, "depth", 1)
        self._od_grobid_depth.depth = max(depth - 1, 0)

    def _make_od_grobid_cycle_stub(
        self, paper_id: str, pdf_path: Path, source: str
    ) -> dict[str, Any]:
        """Ingestion-failure stub emitted when the GROBID↔OD fallback
        chain is cut by the depth guard — the failure must stay visible
        in run_output.warnings instead of producing 0 rows with 0
        diagnostics (mirrors the ``_ingestion_*`` stub shape)."""
        return {
            "paper_id": paper_id,
            "figure_id": f"_ingestion_{source}_cycle",
            "panel_id": None,
            "species": None,
            "panel_path": None,
            "bbox": None,
            "confidence": 0.0,
            "label_text": None,
            "caption_snippet": pdf_path.name,
            "ocr_text": None,
            "paper_metadata": None,
            "metadata": {
                "extraction_source": f"{source}_cycle",
                "ingestion_error": "OD↔GROBID fallback cycle detected; recursive fallback abandoned",
                "ingestion_warning": True,
            },
        }

    def _process_one_pdf_od(self, paper_id: str, pdf_path: Path) -> list[dict[str, Any]]:
        if not self._enter_od_grobid_guard(paper_id, "OD"):
            return [self._make_od_grobid_cycle_stub(paper_id, pdf_path, "od")]
        try:
            return self._process_one_pdf_od_inner(paper_id, pdf_path)
        finally:
            self._exit_od_grobid_guard()

    def _process_one_pdf_od_inner(self, paper_id: str, pdf_path: Path) -> list[dict[str, Any]]:
        od_result = self.od_extractor.extract(pdf_path, self.config.resolved_output_dir())

        if not od_result.success:
            error = od_result.error or "unknown error"
            logger.warning(
                "OpenDataLoader failed (%s); falling back to GROBID+layout",
                error,
            )
            # Phase 29 cycle guard: if we're already inside the GROBID
            # code path (i.e. GROBID failed → OD called us → OD also
            # failed), skip the recursive GROBID call to avoid an
            # infinite loop. Return empty so the caller can fall
            # through to the visual-stub fallback.
            with self._grobid_lock:
                cycle_detected = paper_id in self._grobid_in_progress
            if cycle_detected:
                logger.warning(
                    "Skipping recursive GROBID fallback for %s; OD↔GROBID cycle detected.",
                    paper_id,
                )
                fallback = []
            else:
                fallback = self._process_one_pdf_grobid(paper_id, pdf_path)
            # Audit P1-5: append an ingestion-failed warning stub so
            # the failure is visible in run_output.warnings instead of
            # being silently dropped. Without this, a corrupt PDF (or
            # a stale OD subprocess) produces 0 rows AND 0 warnings,
            # leaving the user with no diagnostic signal.
            return fallback + [
                {
                    "paper_id": paper_id,
                    "figure_id": "_ingestion_od_failed",
                    "panel_id": None,
                    "species": None,
                    "panel_path": None,
                    "bbox": None,
                    "confidence": 0.0,
                    "label_text": None,
                    "caption_snippet": pdf_path.name,
                    "ocr_text": None,
                    "paper_metadata": None,
                    "metadata": {
                        "extraction_source": "od_failed",
                        "ingestion_error": error,
                        "ingestion_warning": True,
                    },
                }
            ]

        figures = od_result.figures
        # OD's caption-image pairing is fragile and the Java subprocess
        # can occasionally return figures=None even when the JSON
        # contains the captions. Retry once before falling back — the
        # second call is virtually always stable.
        if not figures and od_result.json_data:
            try:
                od_result = self.od_extractor.extract(pdf_path, self.config.resolved_output_dir())
                figures = od_result.figures
                if figures:
                    logger.info(
                        "OD retry recovered %d figures for %s",
                        len(figures),
                        paper_id,
                    )
            except Exception as exc:
                logger.debug("OD retry failed: %s", exc)
        # IMPORTANT: even when ``figures`` is empty (OD paired 0),
        # the kids tree in ``od_result.json_data`` may still hold
        # every Fig. N caption. ``_extract_unpaired_captions``
        # below rescues them with orphan-image association. So
        # falling back to GROBID here would lose the entire paper.
        # The fallback below is only reached if ``json_data`` is
        # also missing (truly fatal).
        if not figures and not od_result.json_data:
            logger.info(
                "No figures AND no JSON data from OpenDataLoader for %s; falling back to GROBID.",
                paper_id,
            )
            return self._process_one_pdf_grobid(paper_id, pdf_path)
        # Make sure ``figures`` is a list (OD occasionally returns
        # None instead of [] when the Java pairing stage fails).
        figures = list(figures or [])

        # Geology / fulltext — collect taxon entities from all captions.
        all_taxon_names: list[str] = []
        for pair in figures:
            if pair.caption_text:
                for ent in _extract_taxon_entities_from_text(pair.caption_text):
                    if ent.text:
                        all_taxon_names.append(ent.text)
        species_seed = sorted(set(all_taxon_names))
        use_geology_llm = (
            bool(self.config.extra.get("use_geology_llm", False)) and self.gemma_runtime is not None
        )
        section_links: dict[str, list[dict[str, Any]]] = {}
        knowledge_graph: dict[str, Any] | None = None
        if od_result.fulltext_sections:
            section_links = link_species_to_geology(
                species_names=species_seed,
                sections=od_result.fulltext_sections,
                llm_runtime=self.gemma_runtime if use_geology_llm else None,
            )
            knowledge_graph = build_knowledge_graph(section_links)

        results: list[dict[str, Any]] = []
        n_figs = len(figures)
        for fig_idx, pair in enumerate(figures, start=1):
            # NOTE: do NOT skip ``if not pair.image_paths`` here — the
            # range-chart pre-detection below needs to see those figures
            # so it can detect "distribution of" captions and find an
            # orphan image for them.
            self._emit_progress(
                # audit 2026-07-26: fig_idx is 1-based; emit_progress's
                # first arg is the 1-based "current" (see M5 fix), so
                # pass fig_idx, not fig_idx-1, to keep the bar aligned
                # with the "[fig_idx/n_figs]" label.
                fig_idx,
                n_figs,
                f"[{fig_idx}/{n_figs}] {pair.caption_text[:40] if pair.caption_text else pair.figure_id}",
            )
            logger.debug(
                "fig=%s page=%d imgs=%d cap='%s...'",
                pair.figure_id,
                pair.page_number,
                len(pair.image_paths or []),
                (pair.caption_text or "")[:50],
            )

            # Pick the LARGEST image as the primary region. OpenDataLoader
            # sometimes returns several images per plate (an index map of
            # sample localities, a couple of field outcrop photos, and the
            # actual SEM micrograph plate). The pipeline previously picked
            # ``image_paths[0]`` which is whatever happens to be first in
            # the JSON — for Bandini 2011 Plate 1 that was the 466x424
            # index map, so the segmenter saw 1 panel instead of the 31
            # specimens in the 975x1227 actual plate. Selecting by pixel
            # area fixes that without any per-paper special cases.
            primary_path: str | None = None
            primary_area: int = 0
            region_img = None  # explicit init; the previous version relied on
            # the for-loop binding the name in some branch,
            # which silently re-used the previous figure's
            # region_img when every imread() call failed.

            # ---- Range-chart pre-detection (BEFORE image selection) ----
            # Detect range-chart figures by caption BEFORE we look at
            # the images. Range charts often have ``image_paths == []``
            # because the chart isn't a single embedded image in the
            # PDF — it can be a vector drawing, a scan, or a multi-panel
            # composition. In those cases we need to detect the chart
            # from its caption first, then find its image via the
            # orphan-image search below.
            if not pair.image_paths:
                logger.debug(
                    "fig=%s has no image_paths; caption='%s...' (running pre-detect)",
                    pair.figure_id,
                    (pair.caption_text or "")[:50],
                )
                early_type = classify_figure_type(pair.caption_text, None)
                logger.debug(
                    "pre-detect fig=%s type=%s caption='%s...'",
                    pair.figure_id,
                    early_type,
                    (pair.caption_text or "")[:50],
                )
                if early_type == "range_chart":
                    rc_image = self._find_orphan_image_for_range_chart(
                        figures, pair, od_result.json_data
                    )
                    logger.info(
                        "range_chart %s: orphan search returned %s",
                        pair.figure_id,
                        rc_image,
                    )
                    if rc_image is not None:
                        logger.info(
                            "range_chart %s: no paired image, using orphan %s",
                            pair.figure_id,
                            rc_image,
                        )
                        rc_results = self._process_range_chart(
                            paper_id=paper_id,
                            figure_id=pair.figure_id,
                            caption_text=pair.caption_text or "",
                            image_path=rc_image,
                        )
                        results.extend(rc_results)
                        self._emit_progress(
                            fig_idx,
                            n_figs,
                            f"[{fig_idx}/{n_figs}] range_chart (orphan) → {len(rc_results)} links",
                        )
                continue

            for cand_path in pair.image_paths:
                if not cand_path:
                    continue
                cand = cv2.imread(cand_path)
                if cand is None:
                    continue
                area = int(cand.shape[0]) * int(cand.shape[1])
                if area > primary_area:
                    primary_area = area
                    primary_path = cand_path
                    region_img = cand
            if primary_path is None or region_img is None:
                continue

            # ---- Range-chart detection ----
            # OpenDataLoader returns every figure on a page as a "pair";
            # stratigraphic range charts look like plates to OD but are
            # fundamentally different (a chart, not specimens). Detect
            # them by caption keyword and extract geology via vision
            # BEFORE feeding the image into _process_region (which would
            # otherwise try to segment a chart as if it were a plate and
            # produce bogus panels).
            fig_type = classify_figure_type(pair.caption_text, primary_path)
            if fig_type == "range_chart":
                # OD sometimes fails to associate the chart image with
                # its caption (the chart has no embedded image metadata
                # in the PDF text layer, so the caption-image pairing
                # falls back to ``_find_nearest_caption`` which is
                # brittle across page layouts). When primary_path is None
                # (no image was paired), scan OTHER pairs on the same or
                # adjacent pages for an orphan image (image present but
                # caption empty) and use it. Without this, the range
                # chart is silently lost.
                rc_image_path = primary_path
                if rc_image_path is not None:
                    rc_results = self._process_range_chart(
                        paper_id=paper_id,
                        figure_id=pair.figure_id,
                        caption_text=pair.caption_text or "",
                        image_path=rc_image_path,
                    )
                    results.extend(rc_results)
                    self._emit_progress(
                        fig_idx,
                        n_figs,
                        f"[{fig_idx}/{n_figs}] range_chart → {len(rc_results)} species links",
                    )
                else:
                    logger.warning(
                        "range_chart %s: no image found (caption='%s...')",
                        pair.figure_id,
                        (pair.caption_text or "")[:60],
                    )
                continue

            # Stratigraphic column / litholog column / paleogeographic
            # map / location map: route to the proper multi-modal geology
            # vision prompt instead of falling through to the plate
            # segmentation path. Round 5 added the first three; Round 20
            # sampling showed that "Geological Map of...", "Location
            # map of studied sections", and other map captions (which
            # classify as plain ``map``) also need M3 vision extraction
            # to surface formation/lithology/locality. Without "map"
            # here, those figures fall through to plate segmentation
            # and produce zero usable records.
            if fig_type in (
                "strat_column",
                "litholog_column",
                "paleogeographic_map",
                "map",
            ):
                geo_links: list[dict[str, Any]] = []  # Audit Bug 1:
                # initialize so _emit_progress below never hits
                # UnboundLocalError when the image is missing or
                # m3_engine is None.
                geo_image_path = primary_path
                if geo_image_path is None:
                    geo_image_path = self._find_orphan_image_for_range_chart(
                        figures, pair, od_result.json_data
                    )
                if geo_image_path is not None and self.m3_engine is not None:
                    try:
                        from PIL import Image as _PILImage

                        with _PILImage.open(geo_image_path) as im:
                            geo_image = im.convert("RGB")
                        geo_links = self._m3_call_with_fallback(
                            self.m3_engine.extract_geology,
                            image=geo_image,
                            caption=pair.caption_text or "",
                            figure_type=fig_type,
                            paper_id=paper_id,
                            figure_id=pair.figure_id,
                        )
                    except Exception as exc:
                        logger.warning(
                            "geo_vision %s failed for %s/%s: %s",
                            fig_type,
                            paper_id,
                            pair.figure_id,
                            exc,
                        )
                        geo_links = []
                    # Round 23 audit: emit a stub record EVEN when
                    # ``geo_links`` is empty. Previously the ``if
                    # geo_links:`` guard at line 1151 silently dropped
                    # the figure when M3 vision returned an empty
                    # list — the figure vanished from ``results`` with
                    # only a debug-level log line. Operators had no
                    # way to know the figure had been processed but
                    # produced no geology. Now we always emit the
                    # stub so downstream consumers see the figure
                    # (with ``geology_links=[]``) and can decide
                    # whether to surface it.
                    results.append(
                        {
                            "paper_id": paper_id,
                            "figure_id": pair.figure_id,
                            "panel_id": f"GEO_VISION_{fig_type.upper()}",
                            "species": None,
                            "panel_path": geo_image_path,
                            "bbox": None,
                            "confidence": 0.0,
                            "label_text": None,
                            "caption_snippet": (pair.caption_text or "")[:240],
                            "ocr_text": None,
                            "paper_metadata": None,
                            "metadata": {
                                "figure_type": fig_type,
                                "extraction_source": "geo_vision",
                                "geology_links": geo_links,
                                "geo_vision_used": bool(geo_links),
                                "geo_vision_figure_type": fig_type,
                            },
                        }
                    )
                    if geo_links:
                        logger.info(
                            "%s %s: extracted %d geo links via vision",
                            fig_type,
                            pair.figure_id,
                            len(geo_links),
                        )
                    else:
                        # Round 23 audit: emit a warning so operators
                        # see when M3 vision returned empty for a
                        # strat/litholog/paleogeo figure. The stub
                        # record above is now emitted regardless, but
                        # the warning makes the "M3 found nothing"
                        # signal visible in server logs.
                        logger.warning(
                            "%s %s: M3 vision returned 0 geo links; "
                            "stub record still emitted so the figure "
                            "is not silently lost",
                            fig_type,
                            pair.figure_id,
                        )
                self._emit_progress(
                    fig_idx,
                    n_figs,
                    f"[{fig_idx}/{n_figs}] {fig_type} → {len(geo_links)} vision links",
                )
                continue

            # Phase 64 Plan B (Task B.4): route schematic / diagram /
            # reconstruction / phylogenetic figures to
            # ``M3Engine.extract_schematic`` instead of falling
            # through to the plate-segmentation path. These figures
            # don't contain radiolarian specimen panels — they show
            # boxes / arrows / cladograms — so the classical
            # segmenter would either produce zero useful panels or
            # generate thousands of spurious rows (audit trace:
            # Round 6 micro-CT paper produced 1216 zero-confidence
            # rows before its fix; the same risk applies to
            # conceptual figures).
            #
            # The flow mirrors the geo_vision block above: open the
            # image, call ``extract_schematic``, and emit a stub
            # record carrying the extracted ``figure_schematic_data``
            # on ``metadata``. We emit the stub even when the M3 call
            # returns ``None`` so the operator can see the figure was
            # processed but produced no extraction — same Round 23
            # audit fix used for geo_vision.
            if fig_type in ("schematic", "diagram", "reconstruction", "phylogenetic"):
                schematic_data: dict[str, Any] | None = None  # Audit Bug 1
                # analogue: initialize so the stub below never sees
                # UnboundLocalError when the image is missing or
                # m3_engine is None.
                schematic_image_path = primary_path
                if schematic_image_path is None:
                    schematic_image_path = self._find_orphan_image_for_range_chart(
                        figures, pair, od_result.json_data
                    )
                if schematic_image_path is not None and self.m3_engine is not None:
                    try:
                        from PIL import Image as _PILImage

                        with _PILImage.open(schematic_image_path) as im:
                            schematic_image = im.convert("RGB")
                        schematic_data = self._m3_call_with_fallback(
                            self.m3_engine.extract_schematic,
                            image=schematic_image,
                            caption=pair.caption_text or "",
                            figure_type=fig_type,
                            paper_id=paper_id,
                            figure_id=pair.figure_id,
                        )
                    except Exception as exc:
                        logger.warning(
                            "schematic_vision %s failed for %s/%s: %s",
                            fig_type,
                            paper_id,
                            pair.figure_id,
                            exc,
                        )
                        schematic_data = None
                # Strip the leading-underscore provenance fields
                # before storing on the metadata. The downstream
                # JSONL exporter carries these as provenance columns
                # (paper_id / figure_id are already on the record).
                stored_schematic: dict[str, Any] | None = None
                if schematic_data:
                    stored_schematic = {
                        k: v for k, v in schematic_data.items() if not k.startswith("_")
                    }
                # Always emit a stub so the figure isn't silently
                # lost — same Round 23 audit fix. The stub uses a
                # panel_id starting with ``SCHEMATIC_`` so the
                # operator can filter for it.
                results.append(
                    {
                        "paper_id": paper_id,
                        "figure_id": pair.figure_id,
                        "panel_id": f"SCHEMATIC_{fig_type.upper()}",
                        "species": None,
                        "panel_path": schematic_image_path,
                        "bbox": None,
                        "confidence": (
                            float(schematic_data.get("confidence", 0.0)) if schematic_data else 0.0
                        ),
                        "label_text": None,
                        "caption_snippet": (pair.caption_text or "")[:240],
                        "ocr_text": None,
                        "paper_metadata": None,
                        "metadata": {
                            "figure_type": fig_type,
                            "extraction_source": "schematic_vision",
                            "figure_schematic_data": stored_schematic,
                            "schematic_vision_used": bool(stored_schematic),
                            "schematic_vision_figure_type": fig_type,
                        },
                    }
                )
                if stored_schematic:
                    logger.info(
                        "%s %s: extracted %d text elements + %d relationships via schematic_vision",
                        fig_type,
                        pair.figure_id,
                        len(stored_schematic.get("text_elements") or []),
                        len(stored_schematic.get("relationships") or []),
                    )
                else:
                    logger.warning(
                        "%s %s: schematic_vision returned no data; "
                        "stub record still emitted so the figure is "
                        "not silently lost",
                        fig_type,
                        pair.figure_id,
                    )
                self._emit_progress(
                    fig_idx,
                    n_figs,
                    f"[{fig_idx}/{n_figs}] {fig_type} → "
                    f"{len(stored_schematic.get('text_elements') or []) if stored_schematic else 0} text elements",
                )
                continue

            # Map / location figure: extract geographic context
            # (location names, lat/lon) from the caption and produce
            # a stub record. This ensures these figures aren't silently
            # dropped by the pipeline.
            if fig_type == "other":
                # Round-6 fix: skip classical CV segmentation for
                # non-specimen figure types (micro-CT, XCT, tomographic,
                # cross-section, location maps, paleogeographic maps).
                # These figures are not radiolarian plates — running
                # the classical segmenter on them produces thousands of
                # spurious panel rows with no species (audit trace:
                # Xiao_2017 micro-CT paper produced 1216 rows, all
                # conf=0.01, before this fix). If use_geo_vision is
                # enabled the geo-vision stub above already emitted a
                # warning; otherwise we just skip the figure entirely
                # with a debug log.
                logger.debug(
                    "fig %s: type='other' (micro-CT/cross-section/etc); "
                    "skipping classical segmentation",
                    pair.figure_id,
                )
                continue

            h_img, w_img = region_img.shape[:2]
            region = FigureRegion(
                page_index=pair.page_number,
                bbox=(0, 0, int(w_img), int(h_img)),
                crop_path=primary_path,
                score=0.85,
                region_id=f"od_{paper_id}_p{pair.page_number:03d}_{fig_idx:02d}",
                kind="figure",
                metadata={"source": "opendataloader", "primary_image": primary_path},
            )

            # Build caption record from OpenDataLoader output.
            caption_text = pair.caption_text or ""
            caption_entities = _extract_taxon_entities_from_text(caption_text)
            # Extract the actual figure number from the caption text (e.g. "Fig. 3")
            # instead of falling back to the PDF page number — that was a copy-paste
            # bug that made downstream code think every page-N figure was "figure N".
            figure_number = (
                extract_figure_number(caption_text) or pair.figure_id or str(pair.page_number)
            )
            caption = CaptionRecord(
                paper_id=paper_id,
                figure_id=pair.figure_id,
                caption=caption_text,
                entities=caption_entities,
                figure_number=str(figure_number),
                page_index=pair.page_number,
                panel_labels=[],
                source_xml=None,
            )

            figure_matches = self._process_region(
                paper_id=paper_id,
                figure_id=pair.figure_id,
                caption=caption,
                region_img=region_img,
                region=region,
                figure_index=fig_idx,
                section_links=section_links,
                grobid_sections=od_result.fulltext_sections,
                knowledge_graph=knowledge_graph,
                best_page_index=pair.page_number,
                paper_metadata=od_result.paper_metadata,
            )
            for m in figure_matches:
                meta = m.get("metadata", {})
                meta["extraction_source"] = "opendataloader"
                # audit 2026-08-05 (Fill Gaps): forward the
                # ``classify_figure_type`` result onto every
                # MatchResult produced by this figure so that the
                # FigureRecord exporter
                # (``src/rlpe/converters.py:figure_records_from_matches``)
                # can populate ``figure_type``. Previously the
                # variable ``fig_type`` was in scope but only the
                # range_chart / geo_vision / schematic_vision
                # branches stamped it; the regular plate path
                # dropped it on the floor.
                meta["figure_type"] = fig_type
                # Plate-level image path: ``primary_path`` is the
                # highest-resolution image OpenDataLoader surfaced
                # for this figure. Forward both keys (the FigureRecord
                # reader looks for ``image_path`` /
                # ``figure_image_path`` and PanelRecord reads
                # ``figure_image_path``).
                if primary_path is not None:
                    meta["image_path"] = primary_path
                    meta["figure_image_path"] = primary_path
                # Panel IDs known to this figure (used to populate
                # FigureRecord.panel_ids). Computed once outside
                # the loop below.
                meta["panel_ids"] = [
                    other.get("panel_id")
                    for other in figure_matches
                    if other.get("panel_id")
                ]
                # ``extraction_method`` defaults to the classical
                # heuristic path's "heuristic" string. LLM-first
                # MatchResults already stamp "llm_first" via their
                # own construction sites, so we use ``or`` to keep
                # the upstream value when present.
                if not meta.get("extraction_method"):
                    meta["extraction_method"] = "heuristic"
                m["metadata"] = meta
                results.append(m)

        # Fallback: if OD returned no results even with figures, try GROBID.
        if not results:
            logger.info("OpenDataLoader produced no matches; falling back to GROBID+layout.")
            return self._process_one_pdf_grobid(paper_id, pdf_path)
        # Cross-figure panel reassignment: orphan figures (no species, no real
        # caption) sitting between two real plate figures on adjacent pages
        # are likely a sub-image of one of those plates. Move their panels
        # to the nearest real plate figure so they participate in caption
        # matching and aren't silently dropped.
        results = self._cross_figure_reassign(results)
        # Link range-chart geology to per-panel records. The range-chart
        # path produces stub panel records (panel_id="RANGE_CHART") that
        # carry the chart-level context. For each real panel, we look
        # up matching species in the range-chart stubs and attach a
        # geology_links entry with section/age_range/biozone. This
        # connects the visual stratigraphy data to the panel records
        # that drive the DwC export.
        results = self._link_range_chart_geology(results)
        # After range-chart links are attached, bridge any map-figure
        # location names to range-chart section abbreviations so a
        # downstream consumer can pivot by either representation.
        results = self._cross_link_map_and_range_chart(results)
        # Round-3 multi-modal geology vision: ask MiniMax-M3 to read
        # the figure image + caption and emit structured geology fields
        # (lithology, formation, member, group, country, biozone, Ma
        # range, coordinates). Opt-in via ``use_geo_vision=True`` to
        # avoid silent cost on existing users. We append to existing
        # geology_links — no dedup (deferred to a future cleanup).
        if self.config.extra.get("use_geo_vision", False) and self.m3_engine is not None:
            results = self._apply_geo_vision(results, paper_id)
        # Round-4 P2-5: Stage 3 bbox + crop enrichment. When M3 Stage 3
        # produced ``m3_panels`` with bbox+visible_label for this figure
        # (gated on ``m3_stage3`` opt-in + Stage 3 enabled), crop each
        # panel's image region to disk and stamp the resulting crop path
        # + ``panel_id_source="m3_vision"`` on each result row. This is
        # the round-3 deferred #1 fix: previously the figure had real
        # M3 panel bboxes in ``m3_diag["stage3_panels"]`` but the pred
        # rows still showed ``panel_id_source="legacy"`` because the
        # crop / source rewrite was never persisted. The fix lifts
        # the diag stage3 info into the published panel_id_source.
        if self.config.extra.get("m3_stage3", False) and self.m3_engine is not None:
            results = self._apply_stage3_bbox_crops(results, paper_id)
        # Round 7 multi-plate enrichment: when the OpenDataLoader
        # caption-image pairing missed a plate (e.g. Bandini 2011 Plate
        # 7-9 were dropped), fire a second-pass M3 vision call on each
        # figure with ``expected_plate_label`` derived from the figure_id
        # so the model knows which plate to emit panels for. The result
        # rows are merged into ``results`` ONLY if they fill a real gap
        # (caption_parser claimed N panels but the existing rows have
        # fewer than N panel_ids for this figure).
        if self.config.extra.get("m3_multi_plate_enrich", False) and self.m3_engine is not None:
            results = self._apply_multi_plate_enrichment(
                results,
                paper_id,
                od_fulltext_sections=od_result.fulltext_sections,
                od_figures=figures,
            )
        # Audit 2026-08-02: Stage 6 morphology enrichment. Opt-in via
        # ``m3_stage_6=True``; populates
        # ``self._paper_morphologies[paper_id]`` and stamps
        # ``metadata.morphology_ids`` on rows so existing exporters
        # that read per-row metadata keep working. The MorphologyRecord
        # list is then merged into ``run_output.json`` by ``run()``.
        if self.config.m3_stage_6:
            results = self._apply_morphology_enrichment(
                results, paper_id, od_result.fulltext_sections
            )
        # Round 11: dedup + drop stub rows + drop empty/invalid rows.
        # See ``_finalize_rows`` for the bug fixes this addresses.
        return self._finalize_rows(results)

    def _apply_stage3_bbox_crops(
        self,
        results: list[dict[str, Any]],
        paper_id: str,
    ) -> list[dict[str, Any]]:
        """Round-4 P2-5: enrich each result row with M3 Stage 3 bbox + crop.

        For each result row whose ``figure_id`` matches a figure whose
        ``m3_diag["stage3_panels"]`` is non-empty, crop each panel's
        image region to ``output/panels/{paper_id}/{figure_id}/``,
        stamp the resulting crop path on the row's
        ``metadata.m3_stage3_panel_path`` and ``panel_path`` (only
        when the existing ``panel_path`` is None — we never overwrite
        a richer classical CV path), and bump the ``panel_id_source``
        tag to ``"m3_vision"`` so the web UI can show a "vision
        verified" badge.

        This is purely additive: rows that don't match a Stage 3
        figure are passed through unchanged. Bbox / panel_id
        rewrites only happen for rows where M3 already pinned a
        ``visible_label`` that matches the row's panel_id; otherwise
        we leave the row alone (the panel_id came from a different
        source we trust more).

        Parameters
        ----------
        results : list[dict[str, Any]]
            Output rows from the per-figure loop.
        paper_id : str
            Stable paper id; used to namespace the crop directory.
        """
        crops_dir = self.config.figures_dir() / "m3_crops" / paper_id
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Index figures that have stage3 panels by figure_id.
        figure_to_panels: dict[str, list[dict[str, Any]]] = {}
        figure_to_plate: dict[str, str] = {}
        figure_id_to_rows: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            md = r.get("metadata") or {}
            stage3 = (md.get("m3_diagnostic") or {}).get("stage3_panels") or []
            if stage3:
                figure_to_panels[r.get("figure_id")] = stage3
            # Track the first plate image we find for each figure so
            # the YOLO fallback below can re-use it. Priority is
            # plate-level image first (figure_image_path / primary_image
            # / image_path), panel_path LAST — a panel crop is the
            # wrong input for YOLO (it would re-detect crops of crops).
            # Audit 2026-08-16 (C1): the previous order put panel_path
            # first, which silently fed tiny crops into YOLO.
            fid = r.get("figure_id")
            plate = (
                md.get("figure_image_path")
                or md.get("primary_image")
                or md.get("image_path")
                or r.get("panel_path")
            )
            if fid and plate and fid not in figure_to_plate:
                figure_to_plate[fid] = plate
            # Group rows by figure_id for the YOLO synthesiser.
            # Audit 2026-08-16 (C2): the synthesiser uses each row's
            # actual ``panel_id`` so the matcher in
            # ``_apply_stage3_bbox_crops`` finds a match.
            if fid:
                figure_id_to_rows.setdefault(fid, []).append(r)

        # Audit 2026-08-16 (Plan C): YOLO fallback. When M3 stage 3
        # returned zero panels for a figure but YOLO is enabled, run
        # YOLO on the plate image and synthesise stage3 panel records
        # so the rest of this method (panel-id matching, crop write,
        # panel_path stamp) still produces useful output. Without
        # this fallback a paper that exhausts the M3 quota, or whose
        # plates the vision model declines to segment, would silently
        # lose the bbox crop pass — the existing rows keep their
        # (possibly stale) panel_path and ``panel_id_source`` stays
        # at "legacy".
        if not figure_to_panels:
            figure_to_panels = self._yolo_fallback_for_stage3(
                figure_to_plate,
                paper_id,
                crops_dir,
                # Audit 2026-08-16 (C2): pass per-figure row lists so
                # the synthesised stage3 panels can carry the
                # existing rows' panel_ids — the matcher in
                # ``_apply_stage3_bbox_crops`` keys on panel_id /
                # visible_label, and YOLO has no labels. Without
                # this, every synthesised panel_id was "P{i}" while
                # every row's panel_id was "1", "2", "a" → zero
                # matches → the whole fallback produced no effect.
                figure_id_to_rows=figure_id_to_rows,
            )

        if not figure_to_panels:
            return results

        for r in results:
            fig_id = r.get("figure_id")
            panels = figure_to_panels.get(fig_id)
            if not panels:
                continue
            md = r.setdefault("metadata", {})
            # The current row's panel_id must match one of the stage3
            # boxes for us to consider rewriting. Stage 3 boxes have
            # ``panel_id`` like "P1", "P2", … or the visible_label
            # the model inferred (e.g. "A").
            row_pid = r.get("panel_id") or ""
            row_pid_norm = _normalize_panel_label(row_pid) if row_pid else ""
            matched = next(
                (
                    p
                    for p in panels
                    if (
                        p.get("panel_id") == row_pid
                        or p.get("panel_id") == row_pid_norm
                        or p.get("visible_label") == row_pid
                        or p.get("visible_label") == row_pid_norm
                    )
                ),
                None,
            )
            if matched is None:
                continue
            bbox = matched.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            # The plate image MUST be plate-level (not a panel crop) to
            # slice from — bbox coords are relative to the plate. The
            # classical CV stage stores the plate on
            # ``metadata.figure_image_path`` / ``metadata.primary_image``
            # when running with the OpenDataLoader path. ``panel_path``
            # is a last-resort fallback only (the YOLO path may not
            # have set the plate-level keys). Audit 2026-08-16 (C1):
            # the previous order put ``panel_path`` first, which made
            # the bbox slice operate on a small crop and produced
            # visibly wrong bboxes downstream.
            plate_path = (
                md.get("figure_image_path")
                or md.get("primary_image")
                or md.get("image_path")
                or r.get("panel_path")
            )
            if not plate_path:
                continue
            try:
                from PIL import Image as _PILImage

                plate_p = Path(plate_path)
                if not plate_p.is_file():
                    continue
                with _PILImage.open(plate_p) as im:
                    px_w, px_h = im.size
                    x, y, w, h = (int(v) for v in bbox)
                    x = max(0, min(x, px_w - 1))
                    y = max(0, min(y, px_h - 1))
                    w = max(1, min(w, px_w - x))
                    h = max(1, min(h, px_h - y))
                    crop = im.crop((x, y, x + w, y + h))
                    crop_filename = f"{row_pid or 'panel'}.png"
                    crop_path = crops_dir / fig_id / crop_filename
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(crop_path, "PNG")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Stage 3 crop failed for %s/%s: %s",
                    paper_id,
                    fig_id,
                    exc,
                )
                continue

            # Persist the bbox + crop path + source tag.
            md["m3_stage3_bbox"] = list(bbox)
            md["m3_stage3_visible_label"] = matched.get("visible_label")
            md["m3_stage3_panel_path"] = str(crop_path)
            # Only override panel_path when nothing better exists.
            if not r.get("panel_path"):
                r["panel_path"] = str(crop_path)
                md["panel_path_source"] = "m3_stage3_crop"
            # Bump panel_id_source so the downstream consumer can tell
            # this row was verified by Stage 3 vision (vs caption or
            # image OCR). The previous round left every row at
            # "legacy" because the diag info wasn't lifted.
            #
            # Audit 2026-08-16 (Plan C): honour the synthesised
            # ``source`` field so YOLO-fallback detections are
            # tagged "yolo_fallback" instead of being mis-attributed
            # to M3 vision.
            stage3_source = matched.get("source") or "m3_vision"
            md["panel_id_source"] = stage3_source
            md["stage3_confidence"] = matched.get("confidence")
            r["metadata"] = md
        return results

    def _yolo_fallback_for_stage3(
        self,
        figure_to_plate: dict[str, str],
        paper_id: str,
        crops_dir: Path,
        figure_id_to_rows: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Synthesise Stage 3 panel records from YOLO when M3 vision returns [].

        Audit 2026-08-16 (Plan C): previously, when M3 vision stage 3
        returned zero panels for a figure (quota exhausted, model
        refused, network failure, …), the bbox crop pass in
        ``_apply_stage3_bbox_crops`` short-circuited and the rows
        kept their existing panel_path / panel_id_source = "legacy".
        This helper runs YOLO on the plate image and produces
        stage3_panel-shaped dicts so the crop pass produces output.

        Requirements:
        - ``self.config.use_yolo_figures`` must be True
        - ``self.config.yolo_model_path`` must point to a valid .pt

        Returns
        -------
        dict[str, list[dict]]
            Mapping ``figure_id -> [stage3_panel, ...]``. Empty dict
            if YOLO is disabled / not configured / produced no
            detections, so the caller can fall through to its
            existing early-return path.

        Each synthesised stage3_panel dict matches the M3 shape:
            ``panel_id``      — synthetic id "P1", "P2", …
            ``bbox``          — [x, y, w, h] in plate-image pixels
            ``visible_label`` — None (YOLO doesn't emit labels)
            ``morphology``    — None
            ``confidence``    — float from YOLO detection
            ``source``        — "yolo_fallback" (audit tag)
        """
        if not self.config.use_yolo_figures:
            return {}
        yolo_path = self.config.yolo_model_path
        if not yolo_path:
            return {}
        try:
            from .layout import detect_figure_regions_yolo
            from .layout import PageRecord
        except ImportError:
            return {}
        out: dict[str, list[dict[str, Any]]] = {}
        for fig_id, plate_path in figure_to_plate.items():
            plate_p = Path(plate_path)
            if not plate_p.is_file():
                continue
            try:
                # Re-use detect_figure_regions_yolo's PageRecord shim.
                # It writes crops into ``plate_p.parent / "regions"``
                # and returns FigureRegion objects with bbox + score.
                page = PageRecord(
                    page_index=0,
                    image_path=str(plate_p),
                    text="",
                )
                regions = detect_figure_regions_yolo(
                    page,
                    model_path=yolo_path,
                    conf=self.config.yolo_conf_threshold,
                    iou=self.config.yolo_iou_threshold,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Stage 3 YOLO fallback failed for paper=%s fig=%s: %s",
                    paper_id,
                    fig_id,
                    exc,
                )
                continue
            if not regions:
                continue
            # Audit 2026-08-16 (C2): use the actual rows' panel_ids
            # for the synthesised panels, sorted by reading order
            # (top-to-bottom, then left-to-right) so the existing
            # matcher in ``_apply_stage3_bbox_crops`` finds a hit.
            # YOLO regions are already in detection-confidence order;
            # we sort by ``(bbox.y, bbox.x)`` to mimic reading order.
            sorted_regions = sorted(
                regions, key=lambda r: (int(r.bbox[1]), int(r.bbox[0]))
            )
            figure_rows = (figure_id_to_rows or {}).get(fig_id, [])
            # Rows may have panel_ids in any order — sort them by
            # the same reading order heuristic (no bbox on rows, so
            # use panel_id ordinal position as the tiebreaker).
            sorted_rows = sorted(
                figure_rows,
                key=lambda r: (
                    # Top-to-bottom via page_index when available
                    int((r.get("metadata") or {}).get("page_index") or 0),
                    str(r.get("panel_id") or ""),
                ),
            )
            panels: list[dict[str, Any]] = []
            for i, region in enumerate(sorted_regions):
                # Pick the row at this ordinal position so the
                # synthesised stage3 panel can carry the same
                # ``panel_id`` (and ``visible_label`` as a backup).
                row_pid = ""
                row_visible = None
                if i < len(sorted_rows):
                    matched_row = sorted_rows[i]
                    row_pid = str(matched_row.get("panel_id") or "")
                    row_visible = (
                        (matched_row.get("metadata") or {}).get("label_text")
                        or row_pid
                        or None
                    )
                else:
                    # More YOLO detections than rows → synthesise a
                    # ``P{i+1}`` placeholder. The matcher will not
                    # find a hit, but the crop + bbox stamp still
                    # happens so the operator can review the new
                    # detection in the GUI.
                    row_pid = f"P{i + 1}"
                    row_visible = None
                panels.append(
                    {
                        "panel_id": row_pid,
                        "bbox": list(region.bbox),
                        "visible_label": row_visible,
                        "morphology": None,
                        "confidence": float(region.score),
                        "source": "yolo_fallback",
                    }
                )
            out[fig_id] = panels
            logger.info(
                "Stage 3 YOLO fallback: paper=%s fig=%s detected %d panels (matched to %d rows)",
                paper_id,
                fig_id,
                len(panels),
                len(sorted_rows),
            )
        return out

    def _apply_m3_per_panel_species_id(
        self,
        results: list[dict[str, Any]],
        paper_id: str,
    ) -> list[dict[str, Any]]:
        """Stage 4.5 (Phase 2026-08-17): per-panel M3 vision species ID.

        For each result row whose ``panel_path`` (Stage 3 crop) is
        present, fire one M3 vision call carrying the panel crop + the
        row's caption snippet + the same-page systematic-paleontology
        context. When M3 returns a parseable JSON with
        ``confidence >= m3_per_panel_min_conf``, overwrite the row's
        species (which currently came from regex matching) with M3's
        answer. Otherwise the row's regex species stays.

        Pure additive — every backend failure path (no backend, no
        crop, parse fail, exception) falls through and the regex
        species survives. Per-figure and per-paper caps prevent cost
        runaway on big papers.

        See ``docs/superpowers/specs/2026-08-17-m3-per-panel-pipeline-design.md``.
        """
        if not self.config.m3_per_panel_enabled:
            return results
        if self.m3_engine is None or self.m3_engine.backend is None:
            return results
        if not results:
            return results
        # 1. Build (row, crop_path, caption_for_panel, page_context) tuples.
        items: list[tuple[dict[str, Any], Path, str, str]] = []
        skipped_no_crop = 0
        for r in results:
            crop_path = r.get("panel_path")
            if not crop_path:
                skipped_no_crop += 1
                continue
            crop = Path(crop_path)
            if not crop.is_file():
                skipped_no_crop += 1
                continue
            # Find the caption pair whose panel_id matches this row.
            caption_for_panel = ""
            for cp in (r.get("caption_pairs") or []):
                # CaptionPair is dataclass-like: .panel_id / .text
                # but rows may also pass plain dicts with the same names.
                cp_pid = (
                    getattr(cp, "panel_id", None)
                    or (cp.get("panel_id") if isinstance(cp, dict) else None)
                )
                if cp_pid == r.get("panel_id"):
                    caption_for_panel = (
                        getattr(cp, "text", None)
                        or (cp.get("text") if isinstance(cp, dict) else None)
                        or ""
                    )
                    break
            page_context = (r.get("page_context_snippet") or "")[:1500]
            items.append((r, crop, caption_for_panel, page_context))
        if not items:
            if skipped_no_crop:
                logger.debug(
                    "Stage 4.5: skipped %d/%d rows (no panel_path or file missing) for paper=%s",
                    skipped_no_crop,
                    len(results),
                    paper_id,
                )
            return results
        # TODO (Task 4): fan out via semaphore + per-panel call
        return results

    def _apply_multi_plate_enrichment(
        self,
        results: list[dict[str, Any]],
        paper_id: str,
        *,
        od_fulltext_sections: list[dict[str, Any]] | None = None,
        od_figures: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Round 7 multi-plate enrichment pass.

        For each OD figure that is either (a) MISSING from ``results``
        entirely (the loop crashed or skipped it before any row was
        emitted) or (b) present but under-populated (zero rows with
        both species and panel_id), fire a second-pass M3 vision call
        asking for the full panel list. The new panels are appended to
        ``results`` so downstream eval can score them.

        Trigger conditions (any one):
          * OD returned a figure whose ``figure_id`` does NOT appear in
            any result row (the loop crashed or skipped it).
          * Figure has zero rows.
          * Figure has rows but every row has ``panel_id=None`` AND
            ``species=None``.

        Cost: 1 M3 vision call per qualifying figure (~¥0.01-0.02).
        Gated on ``m3_multi_plate_enrich=True`` to avoid silent spend.
        """
        if self.m3_engine is None:
            return results
        from PIL import Image as _PILImage

        # Index existing results by figure_id (skip stubs).
        by_fig: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            fid = r.get("figure_id", "")
            if not fid or r.get("panel_id") in {
                "RANGE_CHART",
                "MAP_CONTEXT",
                "_ingestion_od_failed",
            }:
                continue
            by_fig.setdefault(fid, []).append(r)

        # Build per-figure page caption text from OD fulltext_sections so
        # M3 sees the surrounding caption context for the plate.
        page_text: dict[int, str] = {}
        for sec in od_fulltext_sections or []:
            page_idx = sec.get("page_index")
            text = sec.get("text", "")
            if page_idx is not None and text:
                page_text[int(page_idx)] = text
        # Fallback: concatenate all section text into one big string so the
        # model has something to reference even if page-indexed lookup
        # doesn't cover the plate's actual page.
        all_captions_blob = "\n\n".join(page_text.values()) if page_text else ""

        # Collect candidate figures to enrich:
        #   1. OD figures whose figure_id has NO results rows at all
        #      (the loop crashed or skipped the figure).
        #   2. OD figures whose results rows have zero species + panel_id.
        candidates: list[tuple[str, list[dict[str, Any]], str, str | None]] = []
        # Track candidate figure_ids so we don't double-process the same
        # one across paths 1 and 2.
        seen_candidate_fids: set[str] = set()
        if od_figures:
            for od_fig in od_figures:
                od_fid = getattr(od_fig, "figure_id", "") or ""
                if not od_fid:
                    continue
                # Skip stubs / non-plate figures.
                src = (getattr(od_fig, "metadata", {}) or {}).get("extraction_source", "")
                od_caption = getattr(od_fig, "caption_text", "") or ""
                if not od_caption:
                    # No caption → can't meaningfully enrich
                    continue
                if od_fid in seen_candidate_fids:
                    continue
                if od_fid not in by_fig:
                    # (1) OD returned this figure but the loop produced
                    # no rows. This is the Bandini 2011 pl05/08/09 case.
                    od_imgs = getattr(od_fig, "image_paths", None) or []
                    primary_img = od_imgs[0] if od_imgs else None
                    candidates.append((od_fid, [], od_caption, primary_img))
                    seen_candidate_fids.add(od_fid)
                else:
                    # (2) Figure has rows but they all lack species + panel_id
                    fig_rows = by_fig[od_fid]
                    n_with_species = sum(1 for r in fig_rows if r.get("species"))
                    n_with_panel_id = sum(1 for r in fig_rows if r.get("panel_id"))
                    if n_with_species == 0 and n_with_panel_id == 0:
                        candidates.append((od_fid, fig_rows, od_caption, None))
                        seen_candidate_fids.add(od_fid)

        appended = 0
        for fid, fig_rows, od_caption, primary_image_path in candidates:
            # Extract page index + plate label from figure_id
            # (od_plate_<pid>_p017_pl07 -> page 17, plate 7).
            page_idx = None
            plate_label: str | None = None
            m_page = re.search(r"_p(\d+)_", fid)
            if m_page:
                page_idx = int(m_page.group(1))
            m_pl = re.search(r"_pl(\d+[a-z]?)", fid)
            if m_pl:
                plate_label = f"Plate {m_pl.group(1).lstrip('0') or m_pl.group(1)}"

            # Skip map / range_chart / geo_vision stubs by extraction_source
            # (already filtered for stubs by panel_id, but range_chart /
            # map figures can still have non-empty captions and bypass the
            # earlier stub filter). Re-check here for safety.
            sample_src = ""
            if fig_rows:
                sample_src = (fig_rows[0].get("metadata") or {}).get("extraction_source", "")
            if sample_src in {"map", "range_chart", "geo_vision"}:
                continue
            # Also skip by figure_id keyword: 'od_fig_' (non-plate) figures
            # carry GeoMap/range-chart content even when caption is set.
            if "_fig_" in fid and "_pl" not in fid:
                continue

            # Find the plate image. Priority:
            #   1. ``primary_image_path`` from the OD FigureCaptionPair
            #      (this is what OD paired the caption with — the actual
            #      plate region).
            #   2. metadata.primary_image / figure_image_path from any
            #      existing row.
            #   3. The row's panel_path (a panel crop, not the full plate).
            image_path = primary_image_path
            if not image_path:
                for r in fig_rows:
                    md = r.get("metadata") or {}
                    cand = (
                        md.get("primary_image")
                        or md.get("figure_image_path")
                        or md.get("image_path")
                    )
                    if cand and Path(cand).is_file():
                        image_path = cand
                        break
                    if r.get("panel_path") and Path(r["panel_path"]).is_file():
                        image_path = r["panel_path"]
                        break
            if not image_path or not Path(image_path).is_file():
                # Last-resort: look in the OD images_dir for any image
                # on the same page as the figure (covers the case where
                # the OD pair lost the image_path but the file still
                # exists in the workdir). Match by including page number
                # in the filename pattern so we don't pick up images from
                # unrelated pages.
                if page_idx is not None:
                    img_dir = self.config.figures_dir() / paper_id
                    if img_dir.is_dir():
                        page_str = f"{page_idx:03d}"  # zero-pad like 'p017'
                        for ext in (".png", ".jpg", ".jpeg"):
                            for cand in sorted(img_dir.glob(f"*p{page_str}*{ext}")):
                                image_path = str(cand)
                                break
                            if image_path:
                                break
            if not image_path or not Path(image_path).is_file():
                logger.debug(
                    "multi_plate_enrich: no image for %s (page %s); skipping",
                    fid,
                    page_idx,
                )
                continue

            # Page-level caption context: OD-caption (always present for
            # candidates) + page_text from fulltext_sections if available.
            ctx_parts: list[str] = []
            if od_caption:
                ctx_parts.append(od_caption)
            if page_idx is not None:
                for off in (-1, 0, 1):
                    t = page_text.get(page_idx + off)
                    if t and t not in ctx_parts:
                        ctx_parts.append(t)
            if not ctx_parts:
                ctx_parts.append(all_captions_blob)
            page_caption = "\n\n".join(p for p in ctx_parts if p)

            try:
                with _PILImage.open(image_path) as im:
                    plate_image = im.convert("RGB")
            except Exception as exc:
                logger.debug("multi_plate_enrich: cannot open %s: %s", image_path, exc)
                continue

            try:
                panels = self._m3_call_with_fallback(
                    self.m3_engine.enrich_plate_panels,
                    image=plate_image,
                    page_caption=page_caption[:3000],  # cap to avoid token bloat
                    paper_id=paper_id,
                    figure_id=fid,
                    expected_plate_label=plate_label,
                )
            except Exception as exc:
                logger.warning(
                    "multi_plate_enrich failed for %s/%s: %s",
                    paper_id,
                    fid,
                    exc,
                )
                continue

            if not panels:
                continue

            # Append each panel as a new stub row with
            # ``panel_id_source="multi_plate_enrich"`` so audit can tell
            # these came from the second pass.
            for p in panels:
                lbl = p.get("label") or ""
                if not lbl:
                    continue
                norm_lbl = _normalize_panel_label(lbl) or lbl
                if not is_valid_panel_label(norm_lbl):
                    continue
                sp = p.get("species")
                try:
                    conf = float(p.get("confidence") or 0.7)
                except (TypeError, ValueError):
                    conf = 0.7
                results.append(
                    {
                        "paper_id": paper_id,
                        "figure_id": fid,
                        "panel_id": norm_lbl,
                        "species": sp if sp else None,
                        "panel_path": None,
                        "bbox": None,
                        "confidence": conf,
                        "label_text": lbl,
                        "caption_snippet": (od_caption or page_caption or "")[:240],
                        "ocr_text": None,
                        "paper_metadata": None,
                        "metadata": {
                            "extraction_method": "multi_plate_enrich",
                            "extraction_source": "multi_plate_enrich",
                            "panel_id_source": "m3_vision",
                            "expected_plate_label": plate_label,
                            "figure_number": ((plate_label or "").replace("Plate ", "") or None),
                        },
                    }
                )
                appended += 1

        if appended:
            logger.info(
                "multi_plate_enrich: paper=%s appended %d panels from second-pass M3",
                paper_id,
                appended,
            )
        return results

    def _link_range_chart_geology(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach range-chart geology context to matching panel records.

        Iterates over all paper-level RangeChartResult objects produced
        in this run; for each one, calls
        ``build_geology_links_for_panels`` against the non-stub panel
        records in the same paper and appends the resulting links to
        each panel's ``metadata.geology_links``.

        Stub records (panel_id="RANGE_CHART") are passed through
        unchanged; their chart context already lives in
        ``metadata.range_chart``.
        """
        from .range_chart_extractor import (
            BiozoneRecord,
            RangeChartResult,
            RangeChartSection,
            SpeciesRange,
        )

        # Index range-chart results by paper_id.
        rc_by_paper: dict[str, list[RangeChartResult]] = {}
        for r in results:
            if str(r.get("panel_id")) != "RANGE_CHART":
                continue
            # Rebuild a RangeChartResult from the stub's stored metadata
            # so the linker can consume the original dataclass API.
            md = r.get("metadata") or {}
            rc_dict = md.get("range_chart")
            if not rc_dict:
                continue
            chart = RangeChartResult(
                figure_id=rc_dict.get("figure_id", ""),
                paper_id=rc_dict.get("paper_id", ""),
                image_path=rc_dict.get("image_path", ""),
                caption=rc_dict.get("caption", ""),
                confidence=float(rc_dict.get("confidence", 0.0)),
            )
            for sec in rc_dict.get("sections") or []:
                # Audit M5: a hand-edited or malformed upstream JSON may
                # have ``sections=[None, {...}, None]``. Guard each item
                # so a None entry doesn't crash ``sec.get(...)`` below.
                if not isinstance(sec, dict):
                    continue
                chart.sections.append(
                    RangeChartSection(
                        **{
                            k: sec.get(k, "")
                            for k in (
                                "name",
                                "age_range",
                                "formation_thickness_m",
                                "coordinates",
                            )
                        },
                        formations=list(sec.get("formations") or []),
                    )
                )
            for sr in rc_dict.get("species_ranges") or []:
                # Same M5 guard for species_ranges — a None item would
                # raise ``sr.get(...)`` AttributeError.
                if not isinstance(sr, dict):
                    continue
                chart.species_ranges.append(
                    SpeciesRange(
                        **{
                            k: sr.get(k, "")
                            for k in (
                                "species",
                                "section",
                                "range_top",
                                "range_base",
                                "biozone",
                            )
                        }
                    )
                )
            for bz in rc_dict.get("biozones") or []:
                # Same M5 guard for biozones.
                if not isinstance(bz, dict):
                    continue
                chart.biozones.append(
                    BiozoneRecord(**{k: bz.get(k, "") for k in ("name", "age", "thickness_m")})
                )
            chart.other_fossils = list(rc_dict.get("other_fossils") or [])
            rc_by_paper.setdefault(chart.paper_id, []).append(chart)

        if not rc_by_paper:
            return results

        for r in results:
            if str(r.get("panel_id")) == "RANGE_CHART":
                continue
            paper_id = r.get("paper_id")
            charts = rc_by_paper.get(paper_id)
            if not charts:
                continue
            md = r.setdefault("metadata", {})
            existing_links = list(md.get("geology_links") or [])
            for chart in charts:
                new_links = build_geology_links_for_panels(chart, [r])
                if new_links:
                    md["geology_links"] = existing_links + new_links
                    existing_links = md["geology_links"]
        return results

    def _apply_geo_vision(
        self,
        results: list[dict[str, Any]],
        paper_id: str,
    ) -> list[dict[str, Any]]:
        """Run MiniMax-M3 vision extraction on each result row.

        ``M3Engine.extract_geology()`` reads a figure image + caption
        and emits structured geology fields (lithology, formation,
        member, group, country, biozone, Ma range, coordinates). We
        append the returned records to ``metadata.geology_links`` so
        downstream consumers (Web UI, DwC export) see them automatically.

        Per-row figure-type routing is driven by metadata keys that
        earlier stages already wrote:

        * ``extraction_source == "range_chart"`` -> ``figure_type="range_chart"``
        * ``extraction_source == "map"`` (or legacy "map_caption_heuristic",
          "map_context") -> ``figure_type="map"``
        * ``figure_type`` itself when already classified by OpenDataLoader

        Rows without a figure image are skipped silently — vision on a
        missing image is pure cost. Rows for which the user's
        ``geo_vision_figure_types`` allowlist excludes the figure type
        are also skipped.

        Cost is aggregated by the existing ``MiniMaxM3Backend.cost_summary()``
        path, so the run-level ``llm_usage.json`` sidecar will reflect
        the additional spend automatically.
        """
        allowed: list[str] = list(
            self.config.extra.get("geo_vision_figure_types", [])
            or [
                "plate",
                "range_chart",
                "stratigraphic_column",
                "litholog_column",
                "paleogeographic_map",
            ]
        )
        for r in results:
            md = r.get("metadata") or {}
            # Audit Bug 4: Skip rows that Round 5's inline block
            # already processed (they have geo_vision_used=True in
            # metadata). Without this guard, enabling both
            # use_geo_vision=True and Round 5's inline routing would
            # call extract_geology TWICE for the same figure —
            # wasting API calls and producing duplicate links.
            if md.get("geo_vision_used"):
                continue
            figure_type = md.get("figure_type")
            # Backfill figure_type from extraction_source where the older
            # stages didn't already tag it.
            # Round 9 (Bug-M1): the map figure path actually writes
            # ``_process_map`` (line ~530) writes
            # ``extraction_source="map"`` (NOT "map_caption_heuristic"
            # which is the extraction_method). Without matching "map"
            # here, ``figure_type`` never gets the "map" backfill and
            # geo vision silently skips every map figure. Also accept
            # the legacy "map_caption_heuristic" / "map_context" strings
            # for backwards compatibility with callers that hand-stamped
            # those values before the fix.
            if not figure_type:
                src = md.get("extraction_source")
                if src == "range_chart":
                    figure_type = "range_chart"
                elif src in ("map", "map_caption_heuristic", "map_context"):
                    figure_type = "map"
            if not figure_type or figure_type not in allowed:
                continue

            image_path = (
                r.get("panel_path")
                or md.get("primary_image")
                or md.get("image_path")
                or md.get("figure_image_path")
            )
            if not image_path or not Path(image_path).is_file():
                # Audit M4: ``Path.exists()`` returns True for directories
                # too, which would crash ``PIL.Image.open()`` with
                # IsADirectoryError. ``is_file()`` is the correct guard —
                # it follows symlinks but rejects directories, FIFOs,
                # and missing paths uniformly.
                continue
            caption_text = md.get("caption_text") or md.get("caption") or ""
            try:
                from PIL import Image as _PILImage

                with _PILImage.open(image_path) as im:
                    panel_image = im.convert("RGB")
                geo_links = self._m3_call_with_fallback(
                    self.m3_engine.extract_geology,
                    image=panel_image,
                    caption=caption_text,
                    figure_type=figure_type,
                    paper_id=paper_id,
                    figure_id=r.get("figure_id") or md.get("figure_id") or "",
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "geo_vision failed for paper=%s figure=%s: %s",
                    paper_id,
                    r.get("figure_id") or "?",
                    exc,
                )
                continue
            if not geo_links:
                continue
            existing = list(md.get("geology_links") or [])
            md["geology_links"] = existing + geo_links
            # Tag the source so review tools can distinguish vision
            # links from text-derived ones.
            md["geo_vision_used"] = True
            md["geo_vision_figure_type"] = figure_type
            r["metadata"] = md
        return results

    # -----------------------------------------------------------------------
    # Audit 2026-08-02 — Stage 6 morphology enrichment (opt-in)
    # -----------------------------------------------------------------------
    def _apply_morphology_enrichment(
        self,
        rows: list[dict[str, Any]],
        paper_id: str,
        fulltext_sections: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Apply M3 morphology extraction to rows.

        For each unique (paper_id, normalised_species) pair, picks a
        source-text excerpt and asks ``M3Engine.infer_morphology`` to
        produce a MorphologyRecord. The records are written to
        ``self._paper_morphologies[paper_id]`` (NOT into ``rows``) so
        downstream exporters can pull them via
        ``converters.run_output_from_provenance(..., paper_morphologies=...)``.

        Parameters
        ----------
        rows : list[dict[str, Any]]
            Output rows from the per-figure loop. Used only to derive
            the per-paper species list and to fall back to caption
            text when the body-text locator finds nothing. NEVER
            modified.
        paper_id : str
            Stable paper id; namespaces the morphology_id so two
            papers with the same species don't collide.
        fulltext_sections : list[dict] | None
            Sections from GROBID or OpenDataLoader. Shape:
            ``[{"section_id": str, "title": str, "section_type": str, "text": str}, ...]``.
            ``None`` skips body morphology (caption-only path).

        Returns
        -------
        rows unchanged. The morphology records are stored in
        ``self._paper_morphologies[paper_id]``.

        Notes
        -----
        * Fail-open: any error → log + skip, never modify existing
          species / panel fields.
        * Per-paper dedup: each unique (paper_id, normalised_species)
          called at most once. The cap is
          ``config.m3_morphology_max_species_per_paper`` (default 100).
        * Privacy: if ``data_outbound_policy == "api_redacted"`` we
          skip body morphology (only caption morphology fires). If
          ``"local_only"`` we skip M3 morphology entirely (no API
          calls reach the cloud).
        """
        # Gate: opt-in + backend present.
        if not self.config.m3_stage_6:
            return rows
        if self.m3_engine is None:
            logger.debug(
                "_apply_morphology_enrichment: no M3 engine; skipping paper=%s",
                paper_id,
            )
            return rows
        policy = str(self.config.extra.get("data_outbound_policy", "api_full"))
        if policy == "local_only":
            logger.info(
                "_apply_morphology_enrichment: data_outbound_policy=local_only; "
                "skipping M3 morphology for paper=%s",
                paper_id,
            )
            return rows
        body_allowed = policy != "api_redacted"
        if not body_allowed:
            logger.info(
                "_apply_morphology_enrichment: data_outbound_policy=api_redacted; "
                "caption-only morphology for paper=%s",
                paper_id,
            )

        # 1. Build per-species dedup set. Prefer the verbatim species
        #    from the row; normalise via the same helper the exporters
        #    use so dedup is stable across re-runs.
        from .converters import _normalise_species_name, _stable_id

        species_counts: dict[str, int] = {}
        species_caption: dict[str, str] = {}
        for r in rows:
            sp = r.get("species")
            norm = _normalise_species_name(sp)
            if not norm:
                continue
            species_counts[norm] = species_counts.get(norm, 0) + 1
            # Keep the first non-empty caption we see for fallback.
            if norm not in species_caption:
                cap = (
                    r.get("caption_snippet")
                    or (r.get("metadata") or {}).get("caption_text")
                    or (r.get("metadata") or {}).get("caption")
                    or ""
                )
                if cap:
                    species_caption[norm] = cap

        # 2. Cap at m3_morphology_max_species_per_paper.
        cap = int(self.config.m3_morphology_max_species_per_paper)
        ordered_species = sorted(
            species_counts.keys(),
            key=lambda s: (-species_counts[s], s),
        )[:cap]

        records: list[dict[str, Any]] = []
        # Lazy import to keep the per-paper fast path cheap.
        from .morphology_locator import locate_morphology_context
        from .schema_models import MorphologyRecord

        max_ctx = int(self.config.m3_morphology_max_context_chars)
        min_caption = int(self.config.m3_morphology_min_caption_chars)

        for species in ordered_species:
            taxon_id = _stable_id("taxon", species)
            morphology_id = _stable_id("morph", paper_id, species)
            ctx_text: str | None = None
            section_id: str | None = None
            section_title: str | None = None
            evidence: str | None = None
            used_source: str = "body_text"
            # Try body-text first (richer context).
            if body_allowed and fulltext_sections:
                try:
                    located = locate_morphology_context(
                        species,
                        fulltext_sections,
                        max_chars=max_ctx,
                    )
                except Exception as exc:
                    logger.warning(
                        "morphology_locator failed for paper=%s species=%s: %s",
                        paper_id,
                        species,
                        exc,
                    )
                    located = None
                if located:
                    ctx_text = located.get("source_text")
                    section_id = located.get("section_id") or None
                    section_title = located.get("section_title") or None
                    evidence = located.get("evidence_span") or None
                    used_source = "body_text"
            # Fall back to caption if no body anchor.
            if not ctx_text:
                cap_text = species_caption.get(species, "") or ""
                if len(cap_text) >= min_caption:
                    ctx_text = cap_text[:max_ctx]
                    used_source = "caption"
            if not ctx_text:
                # Nothing to send to M3 — skip silently. The pipeline
                # would otherwise be paying for an empty prompt.
                continue
            # 3. Call M3 (fail-open: any backend error → skip).
            try:
                parsed = self.m3_engine.infer_morphology(
                    species_name=species,
                    source_text=ctx_text,
                    source=used_source,
                    paper_id=paper_id,
                    max_chars=max_ctx,
                )
            except Exception as exc:
                logger.warning(
                    "_apply_morphology_enrichment: M3 call failed for "
                    "paper=%s species=%s: %s",
                    paper_id,
                    species,
                    exc,
                )
                continue
            if not parsed:
                continue
            # 4. Build the MorphologyRecord. Use the schema so an
            #    unknown field is rejected (extra="forbid"); this
            #    protects downstream consumers from typos in the
            #    M3 output.
            record_payload: dict[str, Any] = {
                "morphology_id": morphology_id,
                "taxon_id": taxon_id,
                "paper_id": paper_id,
                "source": used_source,
                "section_id": section_id,
                "section_title": section_title,
                "evidence_text": evidence,
                "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            }
            for field_name in (
                "test_shape",
                "test_length_um_min",
                "test_length_um_max",
                "test_width_um_min",
                "test_width_um_max",
                "num_segments",
                "cephalis_shape",
                "thorax_shape",
                "abdomen_shape",
                "pore_pattern",
                "pore_diameter_um_min",
                "pore_diameter_um_max",
                "spines_present",
                "spine_count",
                "apertural_structure",
            ):
                if field_name in parsed:
                    record_payload[field_name] = parsed[field_name]
            feats = parsed.get("diagnostic_features") or []
            if isinstance(feats, list):
                record_payload["diagnostic_features"] = [
                    str(x) for x in feats if str(x).strip()
                ]
            try:
                rec = MorphologyRecord.model_validate(record_payload)
            except Exception as exc:
                logger.warning(
                    "_apply_morphology_enrichment: dropping malformed "
                    "MorphologyRecord (paper=%s species=%s): %s",
                    paper_id,
                    species,
                    exc,
                )
                continue
            records.append(rec.model_dump())
            # 5. Stamp morphology_ids on the row's metadata so the
            #    per-row fallback in
            #    ``converters.taxon_records_from_matches`` picks it up
            #    even when ``run_output_from_provenance`` is called
            #    without ``paper_morphologies``.
            for r in rows:
                if _normalise_species_name(r.get("species")) == species:
                    md = r.setdefault("metadata", {})
                    mid_list = list(md.get("morphology_ids") or [])
                    if morphology_id not in mid_list:
                        mid_list.append(morphology_id)
                    md["morphology_ids"] = mid_list
                    r["metadata"] = md
        if records:
            self._paper_morphologies[paper_id] = records
            logger.info(
                "_apply_morphology_enrichment: %d MorphologyRecord(s) for paper=%s",
                len(records),
                paper_id,
            )
        return rows

    # -----------------------------------------------------------------------
    # Phase 65 Plan A.4 — cross-figure linker (plate → strat/litholog/map)
    # -----------------------------------------------------------------------
    def _apply_cross_figure_linker(
        self,
        rows: list[dict[str, Any]],
        paper_id: str,
    ) -> list[dict[str, Any]]:
        """Run the 3-strategy cross-figure linker on a paper's rows.

        Phase 65 Plan A.4. Strategy 1 (sample-ID direct match) and
        Strategy 2 (locality share) are pure-Python and always run.
        Strategy 3 (M3 inference) only runs when ``self.m3_engine`` is
        present — typical smoke / unit tests use ``FakeM3Backend`` so
        the M3 path is exercised without HTTP traffic.

        For each panel row we:
          1. Build the panel-facing view (MatchResult-like dict).
          2. Build the paper-figure index from rows whose figure_type is
             strat_column / litholog_column / paleogeographic_map /
             range_chart. These rows already carry ``geology_links``
             from ``_apply_geo_vision``.
          3. Call ``link_species_to_geology`` to get one ``LinkResult``
             per panel.
          4. Append the new link as a ``geology_links`` entry tagged
             with ``coord_source = "cross_figure_linker:<source>"``
             (the four-valued tag ``sample_match`` / ``locality_match`` /
             ``m3_inference`` / ``unlinked`` lets the export layer and
             the GUI Results tab distinguish the four cases without
             adding a new schema field).

        The function mutates and returns ``rows`` so the caller's
        variable is updated in place; downstream stages see the
        appended links.
        """
        # Lazy import: the linker module pulls in only stdlib + our own
        # sample_id_extractor, so it's cheap, but deferring keeps the
        # import cost off the cold-start path when the linker is off.
        from .cross_figure_linker import link_species_to_geology

        # Split rows by figure_type. Plates are what we LINK FROM; the
        # rest are the index source.
        plate_rows: list[dict[str, Any]] = []
        paper_figures: list[dict[str, Any]] = []
        for row in rows:
            md = row.get("metadata") or {}
            ftype = str(md.get("figure_type") or row.get("figure_type") or "").lower()
            if ftype in (
                "strat_column",
                "litholog_column",
                "paleogeographic_map",
                "range_chart",
            ):
                paper_figures.append(row)
            elif ftype in ("plate", "") or ftype.startswith("plate"):
                plate_rows.append(row)
            else:
                # Schematic / diagram / etc. are not plates either;
                # leave them alone for now (could link them later).
                pass

        if not plate_rows:
            return rows

        # Build the figure-index view. We pull fields from each row's
        # metadata + the row itself. The linker's figure-shape helpers
        # accept any dict-like, so we feed it a normalised view.
        figure_views: list[dict[str, Any]] = []
        for row in paper_figures:
            md = row.get("metadata") or {}
            # If the row has extracted geology_links, surface the top
            # entry's formation/age/locality so the linker can use them
            # as fallback evidence if the caption alone is empty.
            gl = md.get("geology_links") or []
            formation = None
            age = None
            locality = None
            if gl and isinstance(gl[0], dict):
                formation = gl[0].get("formation")
                age = gl[0].get("age") or gl[0].get("chronostratigraphy")
                locality = gl[0].get("locality")
            figure_views.append(
                {
                    "figure_id": row.get("figure_id") or md.get("figure_id") or "",
                    "paper_id": paper_id,
                    "figure_type": str(md.get("figure_type") or row.get("figure_type") or ""),
                    "caption": md.get("caption_text") or md.get("caption") or "",
                    "formation": formation,
                    "age": age,
                    "locality": locality,
                    # Audit 2026-08-16 (A3): stamp figure_number so
                    # ``_extract_figure_number`` in cross_figure_linker
                    # uses Step 1 (figure_number field) instead of
                    # Step 2 (regex on figure_id). Without this, the
                    # linker extracts "03" from ``..._pl03`` but
                    # ``parse_cross_refs`` returns target_figure_num
                    # "3" — string equality fails and Strategy 4
                    # never fires on production papers.
                    "figure_number": (
                        str(md.get("figure_number") or row.get("figure_number") or "")
                        .strip()
                    ),
                }
            )

        # Build panel views. The linker accepts MatchResult-shaped dicts.
        panel_views: list[dict[str, Any]] = []
        for row in plate_rows:
            md = row.get("metadata") or {}
            panel_views.append(
                {
                    "paper_id": paper_id,
                    "figure_id": row.get("figure_id") or md.get("figure_id") or "",
                    "panel_id": row.get("panel_id") or row.get("canonical_panel_id"),
                    "species": row.get("species"),
                    "caption_snippet": (
                        row.get("caption_snippet")
                        or md.get("caption_snippet")
                        or md.get("caption")
                        or ""
                    ),
                    "metadata": md,
                }
            )

        # Run the linker.
        results = link_species_to_geology(
            panels=panel_views,
            paper_figures=figure_views,
            m3_engine=getattr(self, "m3_engine", None),
        )

        # Map (figure_id, panel_id) -> LinkResult for quick lookup.
        # Audit fix 2026-07-24 (Agent B M5): key by tuple, not just
        # panel_id. Two plates in the same paper can both carry
        # panel "5" or "1" (Bandini 2011 pl07/pl09 share label
        # "1" through "27"); using panel_id alone caused the
        # second plate's panel to overwrite the first's entry in
        # this dict, so the wrong LinkResult (from the wrong
        # figure) was attached downstream.
        by_panel_id: dict[tuple[str, str], Any] = {}
        for pv, lr in zip(panel_views, results):
            pid = pv.get("panel_id") or ""
            fid = pv.get("figure_id") or ""
            if pid:
                by_panel_id[(fid, pid)] = lr

        # Append each LinkResult as a geology_links entry on the row.
        for row in plate_rows:
            md = row.setdefault("metadata", {})
            pid = row.get("panel_id") or row.get("canonical_panel_id") or ""
            fid = row.get("figure_id") or md.get("figure_id") or ""
            lr = by_panel_id.get((fid, pid))
            if lr is None:
                continue
            existing = list(md.get("geology_links") or [])
            existing.append(
                {
                    "age": lr.age,
                    "formation": lr.formation,
                    "locality": lr.locality,
                    "confidence": lr.confidence,
                    "evidence_text": lr.evidence,
                    "section_type": "cross_figure_link",
                    "coord_source": f"cross_figure_linker:{lr.source}",
                    # Audit fix 2026-07-24 (Agent B H3): propagate
                    # LinkResult.figure_id so downstream consumers
                    # (Darwin Core archives, GBIF/PBDB audits, the
                    # GUI's "Link source" badge) can trace each link
                    # back to the figure that produced it. Without
                    # this, a geologist auditing the output cannot
                    # tell whether a panel's age came from Figure 3
                    # (strat column) or Figure 7 (paleogeographic
                    # map), making reproduction impossible.
                    "figure_id": lr.figure_id,
                    "link_source": lr.source,
                }
            )
            md["geology_links"] = existing
            # Surface a per-row "link_source" tag so the GUI can
            # badge Strategy 1/2/3/unlinked without parsing the
            # geology_links list.
            md["link_source"] = lr.source
            md["link_confidence"] = float(lr.confidence)
            md["link_figure_id"] = lr.figure_id
            # Also flag for review when M3 was the source and the
            # confidence is at the low end of the band — operators
            # typically want to spot-check these.
            if lr.source == "m3_inference" and lr.confidence < 0.4:
                md.setdefault("needs_review", True)
                reasons = list(md.get("review_reasons") or [])
                if "cross_figure_linker_low_confidence" not in reasons:
                    reasons.append("cross_figure_linker_low_confidence")
                md["review_reasons"] = reasons
            row["metadata"] = md

        # Phase 66 Plan C.4: VISION-coordinate cross-reference linking.
        # Fires for panels whose Phase A Strategy-1 (sample_match) didn't
        # reach confidence 1.0 AND the paper has BOTH a plate figure
        # AND a strat column / paleogeographic-map. The visual links
        # are stored on the row's ``metadata.cross_figure_visual_links``
        # field (Phase C.2 schema). They DON'T replace the Phase A
        # geology_links — they're a precision refinement that the GUI
        # / export layer surfaces when present.
        try:
            from .cross_figure_linker import link_visual_coordinates

            # The trigger function looks at the FULL set of paper
            # figures (plate + strat + map) to confirm the paper has
            # both kinds. ``figure_views`` only has the strat/map side
            # so we supplement with the plate figures extracted above.
            all_figure_views = list(figure_views)
            for prow in plate_rows:
                pmd = prow.get("metadata") or {}
                all_figure_views.append(
                    {
                        "figure_id": prow.get("figure_id") or pmd.get("figure_id") or "",
                        "paper_id": paper_id,
                        "figure_type": str(pmd.get("figure_type") or "plate"),
                        "caption": pmd.get("caption") or pmd.get("caption_text") or "",
                    }
                )

            visual_per_panel = link_visual_coordinates(
                panels=panel_views,
                paper_figures=all_figure_views,
                m3_engine=getattr(self, "m3_engine", None),
            )
            # P2-8 fix: guard against link_visual_coordinates returning fewer
            # results than panel_views (zip would silently drop trailing panels).
            if len(visual_per_panel) < len(panel_views):
                logger.warning(
                    "visual linker returned %d < %d panels — some panels may lack visual links",
                    len(visual_per_panel),
                    len(panel_views),
                )
            for pv, links in zip(panel_views, visual_per_panel):
                pid = pv.get("panel_id") or ""
                fid = pv.get("figure_id") or ""
                if not pid or not links:
                    continue
                # Audit 2026-07-26 M8: match on (figure_id, panel_id)
                # composite key - two plates in the same paper can share
                # panel labels (Bandini 2011 pl07/pl09 both have 1-27);
                # matching on panel_id alone writes cross-plate links to
                # the wrong row. Mirrors the P1-2 fix at line ~2237.
                for row in plate_rows:
                    row_pid = row.get("panel_id") or row.get("canonical_panel_id") or ""
                    row_fid = (
                        row.get("figure_id") or (row.get("metadata") or {}).get("figure_id") or ""
                    )
                    if row_pid == pid and row_fid == fid:
                        md = row.setdefault("metadata", {})
                        existing = list(md.get("cross_figure_visual_links") or [])
                        existing.extend(links)
                        md["cross_figure_visual_links"] = existing
                        row["metadata"] = md
                        break
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "visual coordinate linker failed for paper=%s: %s",
                paper_id,
                exc,
            )

        # Audit 2026-08-16 (fill-gaps): stamp the raw parse_cross_refs
        # result on every plate row's metadata so the
        # export.flatten_for_csv path (which lifts metadata.cross_refs
        # to a top-level column) sees the data. The linker chain above
        # already CONSUMED the parse result via Strategy 4, but the
        # raw list of CrossRefs is also useful as audit evidence and
        # downstream tooling (e.g. CLI export, GUI triage).
        try:
            from .cross_refs import parse_cross_refs

            for row in plate_rows:
                md = row.setdefault("metadata", {})
                if md.get("cross_refs"):
                    continue  # already populated
                cap = (
                    md.get("caption")
                    or md.get("caption_text")
                    or row.get("caption_snippet")
                    or ""
                )
                fig_id = row.get("figure_id") or md.get("figure_id") or ""
                refs = parse_cross_refs(cap, current_fig_id=fig_id)
                if refs:
                    md["cross_refs"] = [r.to_dict() for r in refs]
                    row["metadata"] = md
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "cross_refs stamping failed for paper=%s: %s",
                paper_id,
                exc,
            )

        return rows

    # -----------------------------------------------------------------------
    # Original GROBID + layout path
    # -----------------------------------------------------------------------

    def _process_one_pdf_grobid(self, paper_id: str, pdf_path: Path) -> list[dict[str, Any]]:
        # audit 2026-07-31: depth-guard the GROBID entry too. The
        # OD "success but zero results" branches re-enter this
        # method; without a bound here, GROBID-down + OD-empty
        # recursed GROBID→OD→GROBID→… forever.
        if not self._enter_od_grobid_guard(paper_id, "GROBID"):
            return [self._make_od_grobid_cycle_stub(paper_id, pdf_path, "grobid")]
        try:
            return self._process_one_pdf_grobid_impl(paper_id, pdf_path)
        finally:
            self._exit_od_grobid_guard()

    def _process_one_pdf_grobid_impl(self, paper_id: str, pdf_path: Path) -> list[dict[str, Any]]:
        # Phase 43: fast-fail when GROBID is offline. The GrobidClient
        # retry loop burns ``max_retries * timeout`` seconds (up to
        # 900s by default) hammering a closed port. Probe first; if
        # the server doesn't respond to /api/isalive, skip GROBID
        # entirely and go straight to the OD fallback. The user
        # can disable this via ``--grobid-no-probe`` for tests.
        if not self.config.extra.get("grobid_no_probe", False):
            try:
                if not self.grobid.is_available(probe_timeout=2.0):
                    logger.warning(
                        "GROBID server unavailable at %s; skipping retries, "
                        "falling back to OpenDataLoader for %s",
                        self.config.grobid_url,
                        paper_id,
                    )
                    if not self.config.extra.get("disable_od_fallback", False):
                        return self._process_one_pdf_od(paper_id, pdf_path)
                    return []
            except Exception as exc:
                # is_available() shouldn't raise, but if it does,
                # fall through to the regular retry path.
                logger.debug("GROBID is_available probe raised: %s", exc)
        # Phase 29: mark this paper as currently in the GROBID code
        # path so the OD-fallback path can detect re-entry and break
        # the GROBID↔OD cycle. Cleared in the finally block.
        # Phase 59 (Bug 2.1): guarded by ``_grobid_lock`` so concurrent
        # workers can't miss the cycle guard or leak the entry.
        with self._grobid_lock:
            self._grobid_in_progress.add(paper_id)
        try:
            return self._process_one_pdf_grobid_inner(paper_id, pdf_path)
        finally:
            # Phase 29: clear the cycle-guard entry on every exit path
            # so the OD↔GROBID fallback doesn't leak paper_ids across
            # unrelated papers.
            with self._grobid_lock:
                self._grobid_in_progress.discard(paper_id)

    def _process_one_pdf_grobid_inner(self, paper_id: str, pdf_path: Path) -> list[dict[str, Any]]:
        """Inner body of ``_process_one_pdf_grobid`` (Phase 29 split).

        Extracted so the cycle-guard setup / teardown at the
        outer ``_process_one_pdf_grobid`` wraps everything in a
        single ``try / finally`` block without rewriting 150 lines of
        indent.
        """
        grobid_result = self.grobid.process_pdf(pdf_path, self.config.resolved_output_dir())

        if not grobid_result.success:
            error = grobid_result.error or "GROBID returned no result"
            logger.warning("GROBID failed (%s); figures will be empty", error)
        tei_captions = grobid_result.captions if grobid_result.success else []
        # Extract paper-level metadata (DOI, abstract, authors, journal, year, ...)
        # from the GROBID TEI. Falls back to an empty record on failure.
        try:
            paper_meta = parse_paper_metadata_from_tei(grobid_result.tei_xml or "")
        except Exception:
            paper_meta = PaperMetadata(source="none")

        pages = render_pdf_pages(
            pdf_path, self.config.figures_dir() / paper_id, dpi=self.config.render_dpi
        )
        results: list[dict[str, Any]] = []

        # 全文地质信息抽取与物种关系链接（可选使用LLM增强）
        section_links: dict[str, list[dict[str, Any]]] = {}
        knowledge_graph: dict[str, Any] | None = None
        use_geology_llm = (
            bool(self.config.extra.get("use_geology_llm", False)) and self.gemma_runtime is not None
        )
        species_seed = sorted(
            {ent.text for cap in tei_captions for ent in (cap.entities or []) if ent and ent.text}
        )
        if grobid_result.fulltext_sections:
            section_links = link_species_to_geology(
                species_names=species_seed,
                sections=grobid_result.fulltext_sections,
                llm_runtime=self.gemma_runtime if use_geology_llm else None,
            )
            knowledge_graph = build_knowledge_graph(section_links)

        if not pages:
            return []

        # Fallback: when TEI captions are unavailable, do visual-first extraction.
        # Phase 29: prefer OpenDataLoader over the visual-only stub.
        # OD doesn't need a server, so when GROBID is down we still
        # get real caption-image pairing (esp. helpful for JA/ZH
        # papers whose caption markers wouldn't match the visual
        # fallback's regex). The visual-only stub is now the last
        # resort, used only when both GROBID and OD fail.
        if not tei_captions:
            # Log the failure mode for the operator. ``retry_count``
            # and ``error_type`` were populated by
            # ``GrobidClient.process_pdf`` in Phase 29.
            logger.warning(
                "GROBID produced no captions for %s (retries=%d, error_type=%s, error=%s)",
                paper_id,
                grobid_result.retry_count,
                grobid_result.error_type,
                # Truncate error to keep log lines readable. The
                # full error string can be 1000+ chars (full traceback).
                (grobid_result.error or "")[:200],
            )
            # Attempt OD fallback unless OD is explicitly disabled.
            if not self.config.extra.get("disable_od_fallback", False):
                od_results = self._process_one_pdf_od(paper_id, pdf_path)
                if od_results:
                    logger.info(
                        "GROBID → OpenDataLoader fallback succeeded for %s (%d rows)",
                        paper_id,
                        len(od_results),
                    )
                    # Tag the rows so the consumer can see this came
                    # from OD-after-GROBID-failure.
                    for row in od_results:
                        row.setdefault("extraction_source", "od_after_grobid_failed")
                        row.setdefault(
                            "ingestion_warning",
                            f"GROBID failed ({grobid_result.error_type}); OD fallback used.",
                        )
                    return od_results
                logger.warning(
                    "OpenDataLoader fallback also failed for %s; falling back to visual-only stub.",
                    paper_id,
                )
            return self._fallback_process_without_captions(
                paper_id,
                pages,
                paper_metadata=paper_meta,
            )

        # Audit 2026-07-26 M2: cache detect_figure_regions results per
        # page so adjacent captions (whose candidate_pages overlap) do
        # not re-run YOLO inference on the same page. The OD path
        # already uses a two-pass enum (pipeline.py:4617); only this
        # GROBID path re-detected per caption. YOLO on CPU is
        # seconds/inference, so redundant calls added minutes to a
        # 20-caption run.
        yolo_path = self.config.yolo_model_path if self.config.use_yolo_figures else None
        regions_cache: dict[int, list] = {}

        for idx, caption in enumerate(tei_captions, start=1):
            self._emit_progress(
                idx - 1,
                max(1, len(tei_captions)),
                f"[{idx}/{len(tei_captions)}] {caption.figure_id}",
            )

            best_page = choose_best_page(
                pages,
                caption.figure_number,
                caption.caption,
                window=self.config.caption_window,
            )
            if best_page is None:
                continue
            caption.page_index = best_page.page_index

            # Phase 59 (Bug 2.8): candidate-pages expansion now uses
            # ``self.config.caption_window`` as the radius instead of
            # the hardcoded ±1 offset. Operators can widen the
            # cross-page figure search without code changes.
            candidate_pages = [best_page]
            radius = max(1, int(self.config.caption_window))
            for offset in range(1, radius + 1):
                prev_idx = best_page.page_index - 1 - offset
                next_idx = best_page.page_index - 1 + offset
                if prev_idx >= 0:
                    candidate_pages.insert(0, pages[prev_idx])
                if next_idx < len(pages):
                    candidate_pages.append(pages[next_idx])

            # Phase X: caption text mentions a plate but the caption_window
            # expansion found very few candidate pages — the actual figure is
            # likely on a、集中图版页 at the end of the document that lies
            # beyond the normal window range.  Supplement the candidates.
            existing_indexes = {cp.page_index for cp in candidate_pages}
            if len(candidate_pages) <= 2 and caption.caption:
                _PLATE_KW_RE = __import__("re", fromlist=["compile"]).compile(
                    r"\b(?:plate|pl\.?|figure\s*(?:plate|section)|图版|图版说明)\b",
                    re.IGNORECASE,
                )
                if _PLATE_KW_RE.search(caption.caption):
                    plate_pages = [
                        p for p in find_plate_pages(pages) if p.page_index not in existing_indexes
                    ]
                    candidate_pages.extend(plate_pages[:3])

            chosen_regions = []
            for page in candidate_pages:
                if page.page_index not in regions_cache:
                    regions_cache[page.page_index] = detect_figure_regions(
                        page,
                        yolo_model_path=yolo_path,
                        yolo_conf=self.config.yolo_conf_threshold,
                        yolo_iou=self.config.yolo_iou_threshold,
                        yolo_device=getattr(self.config, "yolo_device", "auto"),
                    )
                regions = regions_cache[page.page_index]
                if regions:
                    chosen_regions.extend(regions)
            if not chosen_regions:
                continue

            chosen_regions.sort(key=lambda r: (-r.score, r.page_index, r.bbox[1], r.bbox[0]))
            # Audit 2026-08-02 (Wave B cost control): cap regions to bound LLM
            # cost on dense papers. Default 3 per caption.
            if len(chosen_regions) > self.config.max_regions_per_caption:
                dropped = len(chosen_regions) - self.config.max_regions_per_caption
                logger.info(
                    "max_regions_per_caption=%d cap dropped %d lower-scored regions for fig=%s",
                    self.config.max_regions_per_caption,
                    dropped,
                    caption.figure_id,
                )
                chosen_regions = chosen_regions[: self.config.max_regions_per_caption]
            # Audit 2026-08-02 (multi-region fallback): this used to take
            # only the first (best-scoring) chosen region and
            # discard the rest. Multi-plate papers (Bandini 2011: 9 plates
            # / 215 panels) put each plate in its own region, so 8/9 plates
            # were silently dropped. We now process every retained chosen
            # region and merge the rows, keeping the highest-scoring region's
            # version of any panel that two regions both detect.
            all_region_results: list[dict[str, Any]] = []
            seen_panels: set[tuple[Any, Any]] = set()
            for region in chosen_regions:
                region_img = (
                    cv2.imread(region.crop_path)
                    if region.crop_path
                    else cv2.imread(best_page.image_path)
                )
                if region_img is None:
                    continue

                figure_matches = self._process_region(
                    paper_id=paper_id,
                    figure_id=caption.figure_id,
                    caption=caption,
                    region_img=region_img,
                    region=region,
                    figure_index=idx,
                    section_links=section_links,
                    grobid_sections=grobid_result.fulltext_sections,
                    knowledge_graph=knowledge_graph,
                    best_page_index=best_page.page_index,
                    paper_metadata=paper_meta,
                )
                # Dedup by (figure_id, panel_id). Rows without a panel_id
                # (stubs such as RANGE_CHART placeholders / ingestion
                # warnings carry panel_id=None) are never deduped — they
                # are not panels and collapsing them would lose data.
                # audit 2026-08-05 (Fill Gaps): stamp ``figure_type``,
                # ``figure_image_path`` / ``image_path``, ``panel_ids``,
                # and ``extraction_method`` on every match produced
                # by this GROBID region. The FigureRecord exporter
                # (``src/rlpe/converters.py:figure_records_from_matches``)
                # reads these keys from ``match.metadata``; without
                # the stamps the GROBID path emitted FigureRecords
                # with all four fields at defaults.
                _grobid_fig_type = (
                    getattr(caption, "figure_type", None)
                    or classify_figure_type(getattr(caption, "text", "") or "", region.crop_path)
                )
                _grobid_panel_ids = [
                    other.get("panel_id")
                    for other in figure_matches
                    if other.get("panel_id")
                ]
                _grobid_image_path = (
                    region.crop_path or best_page.image_path
                )
                for match in figure_matches:
                    panel_id = match.get("panel_id")
                    match_meta = match.get("metadata", {})
                    match_meta["figure_type"] = _grobid_fig_type
                    if _grobid_image_path is not None:
                        match_meta["image_path"] = _grobid_image_path
                        match_meta["figure_image_path"] = _grobid_image_path
                    match_meta["panel_ids"] = _grobid_panel_ids
                    if not match_meta.get("extraction_method"):
                        match_meta["extraction_method"] = "grobid_heuristic"
                    if not match_meta.get("extraction_source"):
                        match_meta["extraction_source"] = "grobid"
                    match["metadata"] = match_meta
                    if not panel_id:
                        all_region_results.append(match)
                        continue
                    key = (match.get("figure_id"), panel_id)
                    if key in seen_panels:
                        continue
                    seen_panels.add(key)
                    all_region_results.append(match)
            results.extend(all_region_results)
        # Cross-figure panel reassignment (same rationale as the OD path
        # at the bottom of ``_process_one_pdf_od``): GROBID can also emit
        # orphan figures — a thumbnail or sub-image of the real plate that
        # gets a placeholder/empty caption and therefore no species. Without
        # reassignment those panels are silently dropped. The OD path has
        # always done this; applying it here keeps the two extraction paths
        # consistent so the eval numbers don't swing depending on which
        # upstream extractor happened to run.
        results = self._cross_figure_reassign(results)
        # Apply the same post-processing chain as the OD path so GROBID
        # papers get identical enrichment (range-chart geology, map
        # bridging, multi-modal geology vision, Stage 3 bbox/crop).
        # Audit BUG-2: pre-fix, this block was missing, so GROBID papers
        # silently skipped all four enrichment steps.
        results = self._link_range_chart_geology(results)
        results = self._cross_link_map_and_range_chart(results)
        if self.config.extra.get("use_geo_vision", False) and self.m3_engine is not None:
            results = self._apply_geo_vision(results, paper_id)
        if self.config.extra.get("m3_stage3", False) and self.m3_engine is not None:
            results = self._apply_stage3_bbox_crops(results, paper_id)
        # cross_figure_linker reads geology_links / range_chart_data populated
        # by the enrichment steps above; runs after them so the linker sees
        # complete context.  Mirrors the OD-path call at line 517.
        if self.config.extra.get("cross_figure_linker_enabled", True):
            try:
                results = self._apply_cross_figure_linker(results, paper_id)
            except Exception as exc:  # defensive
                logger.warning(
                    "cross_figure_linker failed for paper=%s (GROBID path): %s",
                    paper_id,
                    exc,
                )
        # Audit P1-5: append an ingestion-failed warning stub when
        # GROBID failed so the failure is visible in run_output.warnings.
        if not grobid_result.success and not results:
            results.append(
                {
                    "paper_id": paper_id,
                    "figure_id": "_ingestion_grobid_failed",
                    "panel_id": None,
                    "species": None,
                    "panel_path": None,
                    "bbox": None,
                    "confidence": 0.0,
                    "label_text": None,
                    "caption_snippet": pdf_path.name,
                    "ocr_text": None,
                    "paper_metadata": None,
                    "metadata": {
                        "extraction_source": "grobid_failed",
                        "ingestion_error": grobid_result.error or "GROBID returned no result",
                        "ingestion_warning": True,
                    },
                }
            )
        # Round 11: post-process pipeline output (dedup + stub-row filter).
        # See ``_finalize_rows`` docstring for what each rule does.
        # Round 18: enrich each row's geology_links with paleo
        # coordinates + plate_id + reconstruction_model via
        # ``paleo_reconstruction.enrich_geology_record``. The
        # enrichment is in-place and a no-op when modern coords or an
        # age are missing.
        from .paleo_reconstruction import enrich_geology_record

        for r in results:
            md = r.get("metadata") or {}
            for gl in md.get("geology_links") or []:
                if isinstance(gl, dict):
                    enrich_geology_record(gl)
        # Audit 2026-08-02: Stage 6 morphology enrichment (GROBID
        # path). Same opt-in + dedup + privacy rules as the OD path
        # — see ``_apply_morphology_enrichment`` for the full logic.
        if self.config.m3_stage_6:
            results = self._apply_morphology_enrichment(
                results, paper_id, grobid_result.fulltext_sections
            )
        return self._finalize_rows(results)

    # ----- Round 11 post-processing -------------------------------------------------
    # ----- Round 12 post-processing -------------------------------------------------
    # Stub panel_ids that are NOT real panels — they're container rows
    # carrying paper-level context (location coordinates for maps,
    # stratigraphic ranges for range charts, ingestion errors). They
    # have done their job by the time we reach the end of
    # ``_process_one_pdf`` (cross-link / range-chart linking pass has
    # already copied their content to other rows), so we strip them.
    _STUB_PANEL_IDS = frozenset(
        {
            "MAP_CONTEXT",
            "RANGE_CHART",
            "_ingestion_od_failed",
            "_ingestion_grobid_failed",
        }
    )

    @staticmethod
    def _filter_classical_hallucinations(
        matches: list,
        caption,
        paper_id: str,
        figure_id: str,
    ) -> list:
        """Round 12 (Bug 7): drop classical-path rows whose panel_id
        is not in the caption-derived pair set.

        The classical branch emits one row per segmented panel. When
        the segmenter over-segments a plate (e.g. Pouille 2014 pl02
        has 19 panel regions but the caption only mentions 9 species)
        and OCR mis-reads the printed labels, the resulting rows
        include phantom panel_ids like ``10a``/``10b``/``11b``/``11c``
        that the caption never lists. Pre-fix these rows leaked into
        ``matches.jsonl`` and inflated the row count from the actual
        ~6 visible panels to 11+.

        Strategy: re-use the same caption-derived pair_lookup used
        by the LLM-first branch (round 11 Bug 2 fix) and the
        ``_label_in_caption`` predicate. Tolerant of OCR variants
        (1 vs 1a vs 01) so legitimate sub-ids (5b, 6b, 10a) survive.

        Edge case: the classical branch is taken when M3 stage 1
        failed (so ``m3_caption_pairs`` is empty). In that case we
        call ``_regex_parse_caption`` directly on the caption text to
        recover the caption-derived labels. If both sources are empty
        (no caption text at all), we skip the filter — without a
        caption we have no ground truth to compare against.
        """
        import re as _re_hallu2

        # Build caption-derived label set. Try M3 stage 1 first
        # (already computed in this call), fall back to regex.
        pair_lookup: dict[str, str] = {}
        # Caller passes m3_caption_pairs via closure; if empty, regex
        # parser is the second source.
        from .m3_engine import _regex_parse_caption as _regex2

        try:
            for cp in _regex2(caption.caption or ""):
                for lbl in cp.labels or []:
                    pair_lookup.setdefault(lbl.strip(), cp.species)
        except Exception:
            pass

        if not pair_lookup:
            return matches  # No caption to filter against — keep all.

        caption_labels: set[str] = set()
        for lbl in pair_lookup.keys():
            n = _normalize_panel_label(lbl) or lbl
            caption_labels.add(n.strip().lower())

        def _label_in_caption2(n: str) -> bool:
            nn = (n or "").strip().lower()
            if not nn:
                return False
            if nn in caption_labels:
                return True
            m = _re_hallu2.match(r"^(\d+)", nn)
            if m and m.group(1) in caption_labels:
                return True
            m = _re_hallu2.match(r"^([A-H])", nn, _re_hallu2.IGNORECASE)
            if m and m.group(1).lower() in caption_labels:
                return True
            return False

        return [m for m in matches if _label_in_caption2(getattr(m, "panel_id", None) or "")]

    def _finalize_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Round 11 post-processing for one paper's emitted rows.

        Three bug fixes in one pass:

          Bug 1 (panel_id dup with different panel_path): the
            segmenter over-segments real panels and OCR mis-reads
            their labels, so multiple distinct crops end up with the
            same (figure_id, panel_id). Consumers aggregating on
            ``(figure_id, panel_id)`` see them as one panel but our
            rows.jsonl emits N copies. Fix: dedup by (figure_id,
            panel_id) keeping the highest-confidence row. Ties broken
            by panel_score, then row index.

          Bug 3 (MAP_CONTEXT / RANGE_CHART / _ingestion_*): these
            are stub rows from ``_process_map`` / ``_process_range_chart``
            / ingestion-failure paths. Their metadata has been copied
            into the real panel rows by the cross-link pass, so the
            stubs themselves can be safely dropped from ``results``
            without losing information. Without this, eval sees them
            as fake panels with ``species=None``.

          Bug 4 (empty species, no panel_path): when LLM-first path
            returns a row with ``species=None`` AND no ``panel_path``
            (caption-parser-only row that didn't get a match), the row
            carries no signal. Drop it.

          Bug 6 (invalid panel_id format): the parser occasionally
            emits multi-label strings like ``"10, 11"``. The round-9
            panel_id shape regex requires 1-3 digits + optional
            trailing letter — anything that fails the regex is
            almost certainly a parser artifact. Drop these rows.

        Order matters: dedup first (so a duplicate is treated as a
        single row), then drop heuristic noise rows (Phase 1.5) and
        finally drop stubs/invalids.
        """
        import re as _re_stub

        SHAPE = _re_stub.compile(r"^(?:[A-H]|[1-9]\d{0,2}[a-z]?|0)$")

        # Phase 1: dedup by (figure_id, panel_id) keeping best row.
        # Skip stub panel_ids — they should be unique (one per figure)
        # but they shouldn't shadow real panel rows. We dedup them
        # separately.
        real_rows: list[dict[str, Any]] = []
        stub_rows: list[dict[str, Any]] = []
        for r in rows:
            if r.get("panel_id") in self._STUB_PANEL_IDS:
                stub_rows.append(r)
            else:
                real_rows.append(r)

        # Phase 59 (Bug 2.4): split real_rows by panel_id presence.
        # ``(figure_id, panel_id=None)`` collapses multiple distinct
        # no-panel rows (caption-parser, layout-only fallback,
        # OD-unpaired stub) into one when keyed on
        # ``(figure_id, panel_id)``. We dedup None-rows separately
        # using ``(figure_id, bbox_tuple, species, panel_index)`` so
        # distinct rows survive.
        none_panel_rows: list[dict[str, Any]] = []
        keyed_rows: list[dict[str, Any]] = []
        for r in real_rows:
            if r.get("panel_id") is None:
                none_panel_rows.append(r)
            else:
                keyed_rows.append(r)

        def _none_panel_key(r: dict[str, Any]) -> tuple:
            """Dedup key for panel_id=None rows: figure + bbox +
            species + panel_index. Two rows with the same key are
            true duplicates; distinct keys represent distinct
            no-panel observations that must be preserved."""
            bbox = r.get("bbox")
            return (
                r.get("figure_id", ""),
                tuple(bbox) if isinstance(bbox, (list, tuple)) else bbox,
                r.get("species"),
                (r.get("metadata") or {}).get("panel_index"),
            )

        best_by_key: dict[tuple, dict[str, Any]] = {}
        # Phase 59 (Bug 2.4): mixed key types — keyed rows use
        # ``(figure_id, panel_id)``; None-panel rows use the richer
        # ``_none_panel_key`` tuple. Python tuples are structural so
        # the two never collide.
        # Phase A: keyed rows — original (figure_id, panel_id) dedup.
        for r in keyed_rows:
            key = (r.get("figure_id", ""), r.get("panel_id"))
            cur = best_by_key.get(key)
            if cur is None:
                best_by_key[key] = r
                continue
            r_conf = float(r.get("confidence") or 0.0)
            c_conf = float(cur.get("confidence") or 0.0)
            if r_conf > c_conf:
                best_by_key[key] = r
            elif r_conf == c_conf:
                r_score = float((r.get("metadata") or {}).get("panel_score") or 0.0)
                c_score = float((cur.get("metadata") or {}).get("panel_score") or 0.0)
                if r_score > c_score:
                    best_by_key[key] = r

        # Phase B: dedup None-panel rows by (figure_id, bbox, species,
        # panel_index). Distinct keys mean distinct rows that must
        # survive (Bug 2.4 fix).
        for r in none_panel_rows:
            key = _none_panel_key(r)
            cur = best_by_key.get(key)
            if cur is None:
                best_by_key[key] = r
                continue
            r_conf = float(r.get("confidence") or 0.0)
            c_conf = float(cur.get("confidence") or 0.0)
            if r_conf > c_conf:
                best_by_key[key] = r
            elif r_conf == c_conf:
                r_score = float((r.get("metadata") or {}).get("panel_score") or 0.0)
                c_score = float((cur.get("metadata") or {}).get("panel_score") or 0.0)
                if r_score > c_score:
                    best_by_key[key] = r

        deduped = list(best_by_key.values())

        # Phase 1.5: drop heuristic-fallback rows with very low
        # confidence. These are rule-pipeline emissions when the LLM
        # refuses (MiniMax API error). They pollute F1 denominators
        # with no signal.
        # 9/70 rows in Bandini 2011 E2E test had conf<0.30 +
        # heuristic matcher; dropping them cleaned F1 by ~5 pp
        # without breaking real matches.
        # We only drop when ``gemma_used`` is False (LLM was not
        # tried) — if the LLM was tried but the rule pipeline was
        # the fallback, low confidence is still meaningful and may
        # reflect a genuinely hard case.
        def _matcher_type_of(r: dict[str, Any]) -> str:
            """Look up ``matcher_type`` from the row or its metadata.

            The canonical location is ``metadata.matcher_type`` but
            tests and some legacy callers pass it at the top level.
            """
            return r.get("matcher_type") or (r.get("metadata") or {}).get("matcher_type") or ""

        before_noise = len(deduped)
        deduped = [
            r
            for r in deduped
            if not (
                float(r.get("confidence") or 0.0) < 0.30
                and _matcher_type_of(r) == "heuristic"
                and not (r.get("metadata") or {}).get("gemma_used", False)
            )
        ]
        if before_noise - len(deduped):
            logger.info(
                "Heuristic noise filter dropped %d low-confidence rows",
                before_noise - len(deduped),
            )

        # Phase 2: drop empty-signal rows and invalid panel_ids.
        # Phase 59 (Bug 2.4): panel_id=None is now a *valid* category
        # (e.g. caption-parser rows, layout-only fallbacks, OD-unpaired
        # figure stubs). Drop only rows where panel_id is a string
        # but fails the SHAPE check.
        kept: list[dict[str, Any]] = []
        for r in deduped:
            pid = r.get("panel_id")
            # Phase 59: only drop rows whose panel_id is a malformed
            # STRING (not a None — None is now allowed through).
            if pid is not None and (not isinstance(pid, str) or not SHAPE.fullmatch(pid.strip())):
                logger.debug(
                    "Drop row with invalid panel_id=%r (fig=%s)",
                    pid,
                    r.get("figure_id"),
                )
                continue
            # Skip rows with no signal: no species AND no panel_path.
            # (Stub rows already filtered in Phase 1; this catches
            # LLM-first rows where M3 said "not a radiolarian" and
            # the caption-parser couldn't fill in a species either.)
            if not r.get("species") and not r.get("panel_path"):
                # P2-7 fix: preserve ingestion-failed stubs (they have
                # ingestion_warning=True in metadata so they appear in
                # run_output.warnings even when species/panel_path are empty).
                if not (r.get("metadata") or {}).get("ingestion_warning"):
                    logger.debug(
                        "Drop empty row (no species, no panel_path): fig=%s pid=%s",
                        r.get("figure_id"),
                        pid,
                    )
                    continue
            kept.append(r)

        dropped_real = len(real_rows) - len(kept)
        if dropped_real:
            logger.info(
                "Finalize rows: dropped %d invalid/empty/duplicate rows; kept %d",
                dropped_real,
                len(kept),
            )

        # Phase 3: stubs. Cross-link / range-chart linking pass has
        # already merged stub content (location names, range info)
        # into real rows' metadata.matched_location / geology_links.
        # The stub row itself has no species, no real panel_id, and
        # would pollute eval as a fake panel — drop it entirely.
        dropped_stubs = len(stub_rows)
        if dropped_stubs:
            logger.debug(
                "Finalize rows: dropped %d stub rows (MAP_CONTEXT / RANGE_CHART / _ingestion_*)",
                dropped_stubs,
            )

        # audit 2026-07-31: apply human review corrections
        # (service_work/corrections/corrections.jsonl, written by
        # POST /review/correction). Previously the endpoint only
        # APPENDED rows nobody ever read — corrections were dead
        # letters. Corrections key on (paper_id, figure_id) with an
        # optional panel_path prefix; they override species and/or
        # panel label on the matching rows.
        kept = self._apply_review_corrections(kept)

        # audit 2026-07-31: low-confidence rows must be flagged for
        # review. A confidence < 0.5 row previously shipped with
        # needs_review=0 — half-trusted data flowed into the research
        # chain unmarked (real runs: 18/36 rows < 0.5, zero flagged).
        for r in kept:
            try:
                conf = float(r.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            if 0.0 < conf < 0.5:
                md = r.setdefault("metadata", {})
                md.setdefault("needs_review", True)
                reasons = list(md.get("review_reasons") or [])
                if "low_confidence" not in reasons:
                    reasons.append("low_confidence")
                md["review_reasons"] = reasons

        return kept

    def _apply_review_corrections(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Overlay human review corrections on finalized rows.

        Reads ``<work_dir>/corrections/corrections.jsonl`` (one JSON
        object per line, shape of the ``ReviewCorrection`` API model).
        A correction matches rows whose ``paper_id`` and ``figure_id``
        equal the entry and whose ``panel_path`` starts with the
        entry's ``panel_path`` (when given). ``corrected_species`` and
        ``corrected_label`` override the row fields; matched rows get
        ``metadata.review_corrected = True`` so the provenance trail
        shows the overlay.
        """
        cfg = getattr(self, "config", None)
        if cfg is None:
            return rows
        work = Path(getattr(cfg, "work_dir", "") or cfg.resolved_output_dir())
        corr_path = work.parent / "corrections" / "corrections.jsonl"
        if not corr_path.exists():
            return rows
        try:
            with corr_path.open(encoding="utf-8") as fh:
                corrections = [json.loads(line) for line in fh if line.strip()]
        except (OSError, json.JSONDecodeError):
            logger.warning("Review corrections file unreadable: %s", corr_path)
            return rows
        if not corrections:
            return rows
        applied = 0
        for row in rows:
            for c in corrections:
                if not isinstance(c, dict):
                    continue
                if c.get("paper_id") != row.get("paper_id"):
                    continue
                if c.get("figure_id") and c.get("figure_id") != row.get("figure_id"):
                    continue
                pp = c.get("panel_path")
                if pp and not str(row.get("panel_path") or "").startswith(str(pp)):
                    continue
                changed = False
                if c.get("corrected_species"):
                    row["species"] = c["corrected_species"]
                    changed = True
                if c.get("corrected_label"):
                    row["panel_id"] = c["corrected_label"]
                    changed = True
                if changed:
                    md = row.setdefault("metadata", {})
                    md["review_corrected"] = True
                    applied += 1
        if applied:
            logger.info("Applied %d review corrections from %s", applied, corr_path)
        return rows

    # ---- LLM-first extraction (new architecture) --------------------------------
    # When an LLM backend is available, try to extract ALL panel→species mappings
    # from the full figure image in a single LLM call. If this succeeds with
    # sufficient confidence, skip the classical segmentation→OCR→matching pipeline
    # entirely. This produces much higher accuracy because:
    #   - The LLM sees the full plate and understands spatial relationships
    #   - No error amplification from segmentation→OCR→matching cascade
    #   - One call per figure instead of per-panel calls for stage 4

    # A lone panel from the LLM is accepted only at/above this confidence.
    # Below it, the classical CV path may segment better (the LLM may have
    # collapsed a multi-specimen plate into one). Tuned conservatively so a
    # high-confidence single-panel micrograph is kept without retrying.
    _LLM_FIRST_SINGLE_PANEL_MIN_CONF: float = 0.75

    # Phase 61 Plan 4 (Bug 4.1): token budget for the LLM-first caption
    # prompt. The historical hard-truncate at 2000 chars dropped the
    # tail of long captions (Bandini 2011 pl09 = ~3500 chars). The
    # runtime helper ``_truncate_caption_for_llm`` honours this cap
    # when the active backend exposes a tokenizer; otherwise it falls
    # back to ``DEFAULT_MAX_CHARS`` (4000). Kept on the class so
    # tests can verify the budget is in a sensible range without
    # instantiating the pipeline.
    _LLM_FIRST_MAX_TOKENS: int = 4000

    _LLM_FIRST_SYSTEM_PROMPT = """You are an expert paleontologist specializing in radiolarian microfossils. You will see an image of a radiolarian plate (figure) from a scientific publication, along with its caption text.

Your task: identify every distinct specimen panel (sub-figure) in this plate and determine its label (A, B, C... or 1, 2, 3... as printed on the image) and the Latin binomial species name.

Return ONLY valid JSON (no markdown fences). The JSON must be an object with a single key "panels" whose value is an array of objects, each with:
- "label": the panel label as printed (string, e.g. "1", "A", "14b")
- "species": the Latin binomial name (string, e.g. "Actinomma leptodermum"), or null if unknown
- "confidence": your confidence 0.0-1.0 in this panel-species mapping

Rules:
- Include ALL visible specimen panels, even partially visible ones
- FIRST: use the caption text to determine species for each label
- If the caption uses ranges like "1-4. Species name", expand to individual entries
- SECOND: if the caption does NOT mention species for a panel, try to identify the species from the image morphology using your knowledge of radiolarian taxonomy. Set confidence lower (0.3-0.5) to indicate this is a morphology-based guess, not a caption-confirmed identification.
- If a panel has no identifiable label, use your best spatial inference
- If the caption is a placeholder (auto-generated), return {"panels": []}
- Do NOT include non-specimen elements (scale bars, maps, diagrams)
- NEVER invent species names that don't exist in radiolarian taxonomy"""

    def _llm_first_extract(
        self,
        *,
        paper_id: str,
        figure_id: str,
        caption: CaptionRecord,
        region_img: Any,
        region: FigureRegion,
        figure_index: int,
        paper_metadata: PaperMetadata | None = None,
    ) -> list[dict[str, Any]] | None:
        """Try LLM-first extraction. Returns MatchResult list on success, None on failure.

        On success, the caller should use these results directly and skip
        the classical segmentation→OCR→matching pipeline.
        """
        backend = self.gemma_runtime
        if backend is None:
            return None

        caption_text = caption.caption or ""
        # Phase 55 audit BUG-FIX: previously the LLM-first path returned
        # None immediately for placeholder captions (e.g. "Auto-generated
        # figure for page X"), skipping the LLM call entirely. This
        # silently dropped papers whose GROBID/OD caption extraction
        # failed and fell back to visual-only mode — the LLM was never
        # given a chance to identify species from image morphology.
        #
        # The LLM system prompt already handles placeholder captions
        # correctly: it tries to identify species from image morphology
        # when the caption doesn't mention species. We must let the LLM
        # see the image even with placeholder captions. The only early
        # return is when the image itself cannot be loaded.
        try:
            from PIL import Image as _PILImage

            if hasattr(region_img, "shape"):
                _rgb = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
                plate_pil = _PILImage.fromarray(_rgb)
            else:
                with _PILImage.open(str(region_img)) as _im:
                    plate_pil = _im.convert("RGB")
        except Exception as exc:
            logger.warning("LLM-first image load failed for %s/%s: %s", paper_id, figure_id, exc)
            return None

        # Phase 61 Plan 4 (Bug 4.1): token-aware caption truncation.
        # Previous behaviour hard-truncated at 2000 chars; Bandini 2011
        # pl09 (≈3500 chars) lost the tail species. The helper returns a
        # ``(text, mode)`` tuple so we can stamp the truncation mode in
        # metadata (debug / eval visibility) without re-parsing the
        # prompt afterwards. The cap is 4000 tokens when a tokenizer is
        # available, else 4000 chars as a safe fallback.
        try:
            from ._llm_caption import _truncate_caption_for_llm
        except Exception:  # pragma: no cover - helper is in our package
            _truncate_caption_for_llm = None  # type: ignore[assignment]
        tokenizer = getattr(backend, "tokenizer", None)
        truncation_mode = "char_fallback"
        if _truncate_caption_for_llm is not None:
            try:
                truncated_caption, truncation_mode = _truncate_caption_for_llm(
                    caption_text, tokenizer=tokenizer
                )
            except Exception:
                truncated_caption = caption_text
                truncation_mode = "error"
        else:  # pragma: no cover - defensive
            truncated_caption = (caption_text or "")[:4000]
            truncation_mode = "char_fallback"
        user_prompt = (
            f"Paper: {paper_id}\n"
            f"Figure: {figure_id}\n"
            f"Caption:\n{truncated_caption}\n\n"
            f"Identify all specimen panels in this plate. Return JSON."
        )

        try:
            result = backend.infer_panel(
                panel_image=plate_pil,
                caption_text=caption_text,
                ocr_labels=[],
                system_prompt=self._LLM_FIRST_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.debug("LLM-first extraction failed: %s", exc)
            return None

        if result.get("fallback_used") or result.get("error"):
            logger.debug("LLM-first returned fallback/error, skipping")
            return None

        # Parse the LLM response. The result may contain a "panels" key
        # directly, or the raw JSON may be in result["raw_text"].
        # audit 2026-07-31: the backend ALREADY parsed the model JSON
        # (via parse_json_from_text, tolerant of preambles) — the raw
        # text is only a fallback. The previous code re-parsed raw_text
        # with strict json.loads: when the model emitted a preamble
        # ("Here are the panels: {...}") the backend had succeeded but
        # this strict re-parse failed and the PAID result was silently
        # discarded. Consumption order is now: backend-parsed panels →
        # backend-parsed single panel → robust _safe_json_loads on the
        # raw text.
        panels_data = result.get("panels") or result.get("answer")
        if panels_data is None:
            # The backend may have parsed a SINGLE panel dict (the
            # model ignored the "output an array" instruction).
            if isinstance(result, dict) and "label" in result:
                panels_data = [result]
        if panels_data is None:
            raw = result.get("raw_text", "")
            if not raw:
                return None
            try:
                from .m3_engine import _safe_json_loads

                parsed = _safe_json_loads(raw)
                if isinstance(parsed, dict):
                    panels_data = parsed.get("panels") or parsed.get("answer") or [parsed]
                else:
                    panels_data = parsed
            except Exception:
                return None

        if not isinstance(panels_data, list) or len(panels_data) == 0:
            return None

        # Quality gate: accept 1-panel results only when the LLM is
        # confident. Many real plates are a single SEM micrograph with
        # one specimen; the previous "< 2 → reject" rule silently
        # dropped those and forced the (error-prone) classical path.
        # We still reject a lone panel that the LLM itself rated low.
        if len(panels_data) < 2:
            lone_conf = 0.0
            try:
                lone_conf = float(panels_data[0].get("confidence", 0.0))
            except (TypeError, ValueError, IndexError):
                lone_conf = 0.0
            if lone_conf < self._LLM_FIRST_SINGLE_PANEL_MIN_CONF:
                logger.debug(
                    "LLM-first found 1 low-confidence panel (%.2f < %.2f), falling back",
                    lone_conf,
                    self._LLM_FIRST_SINGLE_PANEL_MIN_CONF,
                )
                return None

        # Convert to MatchResult dicts
        out: list[dict[str, Any]] = []
        # audit 2026-08-05 (Fill Gaps): use enumerate(..., start=1) so
        # the 1-based ``pipeline_panel_index`` field on PanelRecord
        # carries the LLM's panel position within this figure. Phase
        # 55 had hard-coded ``panel_index=None`` here on the (then
        # correct) ground that no PanelCandidate existed; with schema
        # v1.1.0+ declaring the field as a recoverable integer, the
        # natural position-in-panels_data is the right value to stamp.
        for _panel_idx, p in enumerate(panels_data, start=1):
            label = str(p.get("label", "")).strip()
            species = p.get("species")
            conf = float(p.get("confidence", 0.0))
            # Phase 54 audit M8: drop panels whose label is empty/None.
            # Without this, str(None).strip() would become "None" and
            # survive the gate, polluting downstream stages with a fake
            # panel id "None".
            if not label:
                continue
            panel_id = _normalize_panel_label(label) or label
            m = MatchResult(
                paper_id=paper_id,
                figure_id=str(figure_id),
                panel_id=panel_id,
                species=species if species else None,
                panel_path=None,
                bbox=None,
                confidence=conf,
                label_text=label,
                caption_snippet=(
                    ((caption.caption or "").strip()[:240] or None)
                    if hasattr(caption, "caption")
                    else None
                ),
                ocr_text=None,
                paper_metadata=paper_metadata,
                # Audit 2026-08-05 (Fill Gaps): stamp the 1-based
                # panel position so PanelRecord.pipeline_panel_index
                # reflects where in the LLM's panels_data list this
                # row came from. Phase 55's CRITICAL-2 fix
                # (commit ``6defce2``) had hard-coded None because
                # the field was unused; with the v1.1.0+ schema
                # declaring it as a 1-based integer, the natural
                # list position is the right value. The
                # converter (``panel_record_from_match``) reads it
                # via ``getattr(match, 'panel_index', None)``.
                panel_index=_panel_idx,
                metadata={
                    "extraction_method": "llm_first",
                    "llm_backend": getattr(backend, "backend_name", "unknown"),
                    "panel_count": len(panels_data),
                    "figure_number": caption.figure_number,
                    "page_index": caption.page_index,
                    # Label provenance: the LLM inferred this label from
                    # the caption + image jointly. The honest source is
                    # "llm_first" (not "m3_vision" or "image_ocr") —
                    # the visual-evidence path remains reserved for
                    # true image-OCR / Stage-3 bbox+crop work. We do
                    # NOT set printed_panel_id here; that field is a
                    # claim of pixel-level evidence and would mislead
                    # review tools if stamped on a caption-derived id.
                    "caption_panel_id": panel_id,
                    "panel_id_source": "llm_first",
                    # Phase 61 Plan 4 (Bug 4.1): record which truncation
                    # strategy was applied to the caption before the
                    # LLM call. "none" = no truncation; "token_aware" =
                    # tokeniser-driven; "char_fallback" = 4000-char cap
                    # because no tokenizer; "error" = helper crashed.
                    "caption_truncation_mode": truncation_mode,
                },
            )
            out.append(m.to_dict())

        if out:
            logger.info(
                "LLM-first: %s/%s → %d panels extracted",
                paper_id,
                figure_id,
                len(out),
            )
        return out or None

    def _process_region(
        self,
        *,
        paper_id: str,
        figure_id: str,
        caption: CaptionRecord,
        region_img: Any,
        region: FigureRegion,
        figure_index: int,
        section_links: dict[str, list[dict[str, Any]]],
        grobid_sections: list[dict[str, str]],
        knowledge_graph: dict[str, Any] | None,
        best_page_index: int | None,
        paper_metadata: PaperMetadata | None = None,
    ) -> list[dict[str, Any]]:
        """Shared region processing: segment, OCR, match, (optional) Gemma, scale, geology.

        Architecture: LLM-first path runs first. If the LLM extracts all
        panels from the full figure image in a single call, those results
        are used directly — skipping segmentation, OCR, and the rule-based
        matcher entirely. On LLM failure, the classical CV+rules pipeline
        runs as fallback.

        When M3Engine is available, the M3 5-stage pipeline runs alongside
        the classical CV + rule-based path in fallback mode:
          - Stage 1 parses caption into structured (label->species) pairs.
          - Stage 2 filters non-radiolarian plates; on rejection, returns early.
          - Stage 3 augments SAM2 panels with M3-suggested bboxes / visible labels.
          - Stage 4 replaces the per-panel M3 call with a richer context-aware match.
          - Stage 5 critiques all matches and may override low-confidence ones.
        """
        # ---- LLM-first path (try before anything else) -------------------------
        use_llm_first = bool(self.config.extra.get("use_llm_first", True))
        if use_llm_first and self.gemma_runtime is not None:
            llm_results = self._llm_first_extract(
                paper_id=paper_id,
                figure_id=figure_id,
                caption=caption,
                region_img=region_img,
                region=region,
                figure_index=figure_index,
                paper_metadata=paper_metadata,
            )
            if llm_results is not None:
                # Hybrid approach: LLM-first found all panels (100%
                # panel_match) but may have left some species=None when
                # the caption was incomplete or the LLM couldn't
                # determine the species from the image alone. Run the
                # caption parser (M3 Stage 1 or regex) to fill in the
                # gaps — this is cheap (text-only, no image) and gives
                # the best of both worlds: LLM-first's panel detection
                # + classical path's species assignment.
                #
                # The hybrid fires when EITHER:
                #   (a) the LLM left any species blank, OR
                #   (b) the LLM returned fewer than 2 panels, OR
                #   (c) the caption parser finds MORE panels than the LLM did.
                # (c) is critical: Gemma-3/M3 frequently caps its output
                # at ~19 panels (it's a soft training-data ceiling) while
                # real plates can have 21-27 panels (baumgartner2008 pl02=21,
                # pl03=27, beccaro2006=35). Without (c), the truncated
                # panels are silently dropped and the figure ends up with
                # fewer panels than the caption actually enumerates.
                #
                # We pre-parse the caption once so the gate can compare
                # LLM count vs caption count without parsing twice.
                missing_species = [r for r in llm_results if not r.get("species")]
                pair_lookup: dict[str, str] = {}
                # The regex parser is faster + more reliable for
                # caption species extraction on standard layouts
                # (Pouille, Danelian, Beccaro). M3 stage 1 adds API
                # calls and tends to TRUNCATE long captions
                # (beccaro has 35 species — M3 returned only 33).
                # Use regex as the primary source; only fall back
                # to M3 if regex returns nothing.
                try:
                    from .m3_engine import _regex_parse_caption as _regex

                    regex_pairs = _regex(caption.caption or "")
                    for cp in regex_pairs:
                        for lbl in cp.labels or []:
                            pair_lookup.setdefault(lbl.strip(), cp.species)
                except Exception as exc:
                    logger.debug("Regex caption parser failed: %s", exc)
                if not pair_lookup and self.m3_engine is not None:
                    try:
                        caption_pairs = self._m3_call_with_fallback(
                            self.m3_engine.parse_caption,
                            caption.caption or "",
                            lang=_resolve_m3_prompt_lang(self.config.extra.get("m3_prompt_lang")),
                        )
                        for cp in caption_pairs:
                            for lbl in cp.labels or []:
                                pair_lookup.setdefault(lbl.strip(), cp.species)
                    except Exception as exc:
                        logger.debug("M3 caption parser failed: %s", exc)
                # Gate: fire hybrid when LLM truncated its output.
                # Bound (pair_lookup <= 100) guards against runaway
                # regex over-matching on degenerate captions (rare
                # but seen on wever2006 1918-panel runs).
                caption_has_more = bool(pair_lookup) and len(pair_lookup) > len(llm_results)
                if (
                    missing_species
                    or len(llm_results) < 2
                    or (caption_has_more and len(pair_lookup) <= 100)
                ):
                    if pair_lookup:
                        # 1) Fill in species for any LLM rows that had None.
                        # Normalize labels for comparison: "1a" and "1A" must
                        # be treated as the same panel, otherwise the
                        # "if lbl in existing_labels" check below would
                        # miss the match and we'd insert a duplicate
                        # row. _normalize_panel_label canonicalises
                        # "00" → "0" and keeps "1a" / "1A" as-is (no
                        # case folding), so we also lowercase.
                        existing_labels = {
                            _normalize_panel_label(r.get("panel_id") or r.get("label_text") or "")
                            .strip()
                            .lower()
                            for r in llm_results
                            if (r.get("panel_id") or r.get("label_text") or "").strip()
                        }
                        filled = 0
                        skipped_invalid = 0
                        for r in llm_results:
                            if r.get("species"):
                                continue
                            label = r.get("panel_id") or r.get("label_text") or ""
                            matched_key = _label_in_pair_lookup(label, pair_lookup)
                            if matched_key:
                                candidate_species = pair_lookup[matched_key]
                                # Phase 61 Plan 4 (Bug 4.2): guard
                                # against LLM / caption-parser hallucinations
                                # like "Foreman species" or "Dubious
                                # species". The species-validity check
                                # blocks author-surname genera and
                                # common placeholder tokens. If
                                # invalid, KEEP the rule result (do
                                # not overwrite) so eval sees the
                                # honest "no species" state.
                                try:
                                    from .taxon import _is_valid_species

                                    if not _is_valid_species(candidate_species):
                                        skipped_invalid += 1
                                        r.setdefault("metadata", {})["hybrid_species_rejected"] = (
                                            candidate_species
                                        )
                                        continue
                                except Exception:
                                    # If the helper is unavailable for
                                    # any reason we still write the
                                    # species to preserve legacy
                                    # behaviour. Better a noisy
                                    # downstream than a silent fallback.
                                    pass
                                r["species"] = candidate_species
                                r.setdefault("metadata", {})["species_source"] = (
                                    "caption_parser_hybrid"
                                    if self.m3_engine is not None
                                    else "regex_caption_hybrid"
                                )
                                filled += 1
                        # 2) Add NEW rows for any caption labels the LLM
                        #    didn't return at all. This is the critical
                        #    recovery for "LLM truncated its output to
                        #    panels 1..30 instead of 1..35" — without
                        #    this, panel_match_rate drops from 100% to
                        #    ~85% on beccaro.
                        added = 0
                        for lbl, species in pair_lookup.items():
                            lbl_norm = _normalize_panel_label(lbl).strip().lower() if lbl else ""
                            if lbl_norm and lbl_norm in existing_labels:
                                continue
                            # Phase 54 audit: M7 — use the *normalised*
                            # label for the validity check. The dedup
                            # above runs against ``lbl_norm`` (so "00"
                            # collides with "0"), but the validity check
                            # used the *raw* ``lbl``, and
                            # ``is_valid_panel_label("00")`` returns
                            # False (the digit SHAPE regex rejects
                            # leading zeros). A caption that lists
                            # panel ``00`` therefore lost its species.
                            if not is_valid_panel_label(lbl_norm):
                                continue
                            # Phase 54 audit: H2 — ``panel_count`` is the
                            # *total* row count for the figure once this
                            # row is appended. The previous code used
                            # ``len(llm_results) + 1`` evaluated AFTER
                            # the ``.append()`` on the line above, so
                            # ``len(llm_results)`` already included the
                            # new row — every new row's panel_count was
                            # one too high. We snapshot the pre-append
                            # length and add 1 explicitly so the value
                            # is the post-append total, not the
                            # post-append total + 1.
                            pre_append_count = len(llm_results)
                            # Phase 61 Plan 4 (Bug 4.2): same validity
                            # guard as the fill loop above — if the
                            # new-row species looks like a hallucinated
                            # author-surname + epithet, DROP the row
                            # rather than appending a polluted entry.
                            try:
                                from .taxon import _is_valid_species

                                _new_row_species_is_valid = _is_valid_species(species)
                            except Exception:
                                _new_row_species_is_valid = True
                            if not _new_row_species_is_valid:
                                skipped_invalid += 1
                                continue
                            # audit 2026-08-05 (Fill Gaps): 1-based
                            # ``pipeline_panel_index`` for this hybrid
                            # caption-enrichment row. Uses the
                            # post-append length so the value is the
                            # final 1-based position within this
                            # figure, matching the classical
                            # OpenCV-segmenter indexing.
                            _hybrid_panel_idx = pre_append_count + 1
                            llm_results.append(
                                MatchResult(
                                    paper_id=paper_id,
                                    figure_id=str(figure_id),
                                    panel_id=lbl,
                                    species=species,
                                    panel_path=None,
                                    bbox=None,
                                    confidence=0.0,
                                    label_text=lbl,
                                    # Phase 54 audit m11: the
                                    # auto-generated placeholder
                                    # caption ("Auto-generated figure
                                    # for page X") was being sliced
                                    # to an empty string here, which
                                    # looked like a real caption
                                    # snippet to downstream stages.
                                    # Convert the empty slice to None
                                    # so the UI / exporters can
                                    # distinguish "no caption" from
                                    # "empty placeholder".
                                    caption_snippet=(caption.caption or "").strip()[:240] or None,
                                    ocr_text=None,
                                    paper_metadata=paper_metadata,
                                    # audit 2026-08-05 (Fill Gaps):
                                    # 1-based position-in-figure so
                                    # PanelRecord.pipeline_panel_index
                                    # is recoverable here too. Phase
                                    # 55's CRITICAL-2 had hard-coded
                                    # None for the same reason as the
                                    # primary LLM-first site.
                                    panel_index=_hybrid_panel_idx,
                                    metadata={
                                        "extraction_method": "llm_first",
                                        "llm_backend": getattr(
                                            self.gemma_runtime,
                                            "backend_name",
                                            "unknown",
                                        ),
                                        "panel_count": pre_append_count + 1,
                                        "figure_number": caption.figure_number,
                                        "page_index": caption.page_index,
                                        "species_source": (
                                            "caption_parser_hybrid"
                                            if self.m3_engine is not None
                                            else "regex_caption_hybrid_added"
                                        ),
                                        # This row was added because the
                                        # LLM truncated its output but
                                        # the caption parser found it.
                                        # The label is fully caption-
                                        # derived, so:
                                        #   caption_panel_id == panel_id
                                        #   panel_id_source == "caption"
                                        "caption_panel_id": lbl,
                                        "panel_id_source": "caption",
                                    },
                                ).to_dict()
                            )
                            added += 1
                        if filled or added:
                            logger.info(
                                "LLM-first hybrid for %s/%s: filled %d species, added %d panels from caption",
                                paper_id,
                                figure_id,
                                filled,
                                added,
                            )

                # Phase 59 (Bug 2.6): post-hybrid dedup. The hybrid
                # block above adds NEW rows for caption labels that
                # were not in the LLM output, but if a label slipped
                # past the "if lbl_norm in existing_labels: continue"
                # gate (e.g. because LLM-normalised the label
                # differently than the caption parser), the same
                # (paper_id, figure_id, panel_id) can now appear
                # twice in ``llm_results``. The downstream
                # ``_finalize_rows`` dedups by (figure_id, panel_id)
                # so the higher-confidence row wins, but in this
                # hybrid case the caption-derived row has
                # confidence=0.0 while the LLM row may have 0.8 — so
                # the dedup keeps the LLM row and discards the
                # caption enrichment. That breaks the recovery path.
                #
                # We post-process: for any caption-derived row whose
                # (paper_id, figure_id, panel_id_normalised) already
                # exists in ``llm_results`` (i.e. the LLM did produce
                # that row), drop the caption-derived duplicate and
                # keep the LLM row (it has richer bbox / metadata).
                # The remaining caption-only rows (the LLM
                # truncation case) survive unchanged.
                if llm_results and pair_lookup:
                    seen_panel_keys: dict[tuple[str, str, str], dict[str, Any]] = {}
                    deduped_llm: list[dict[str, Any]] = []
                    for r in llm_results:
                        pid = r.get("panel_id")
                        if pid is None:
                            # No panel_id → can't dedup; keep as-is.
                            deduped_llm.append(r)
                            continue
                        norm = _normalize_panel_label(str(pid)).strip().lower()
                        if not norm:
                            deduped_llm.append(r)
                            continue
                        key = (paper_id, str(r.get("figure_id", figure_id)), norm)
                        # Caption-derived rows (species_source ends with
                        # ``_hybrid_added`` or ``caption_parser_hybrid``)
                        # drop when an LLM-native row exists; LLM rows
                        # themselves always win.
                        is_caption_added = (r.get("metadata") or {}).get("species_source") in (
                            "caption_parser_hybrid",
                            "regex_caption_hybrid_added",
                        )
                        if is_caption_added and key in seen_panel_keys:
                            # The earlier LLM row wins — drop this duplicate.
                            continue
                        seen_panel_keys[key] = r
                        deduped_llm.append(r)
                    if len(deduped_llm) != len(llm_results):
                        logger.debug(
                            "Post-hybrid dedup %s/%s: dropped %d caption-duplicate rows",
                            paper_id,
                            figure_id,
                            len(llm_results) - len(deduped_llm),
                        )
                        llm_results = deduped_llm
                # Round 11 (Bug 2 fix): filter M3-returned panels whose
                # label doesn't appear in the caption-derived pair set.
                # M3 frequently invents panel_ids for plates whose
                # caption enumerates fewer panels than the segmenter
                # finds — e.g. Pouille 2014 has 19 visible panel
                # regions but the caption only lists 6 species, so M3
                # invents pid=2,4,7,9,10,11,13,14b out of thin air.
                # The hybrid path above fills species from the caption
                # but the panel_id itself stays hallucinated; this filter
                # drops M3 rows whose labels are NOT mentioned in
                # the caption. Tolerant of OCR variants (1 vs 1a vs 01)
                # via numeric/letter prefix match.
                if llm_results and pair_lookup:
                    import re as _re_hallu

                    caption_labels: set[str] = set()
                    for lbl in pair_lookup.keys():
                        n = _normalize_panel_label(lbl) or lbl
                        caption_labels.add(n.strip().lower())

                    def _label_in_caption(n: str) -> bool:
                        nn = (n or "").strip().lower()
                        if not nn:
                            return False
                        if nn in caption_labels:
                            return True
                        m = _re_hallu.match(r"^(\d+)", nn)
                        if m and m.group(1) in caption_labels:
                            return True
                        m = _re_hallu.match(r"^([A-H])", nn, _re_hallu.IGNORECASE)
                        if m and m.group(1).lower() in caption_labels:
                            return True
                        return False

                    pre_filter = len(llm_results)
                    llm_results = [
                        r for r in llm_results if _label_in_caption(r.get("panel_id") or "")
                    ]
                    dropped = pre_filter - len(llm_results)
                    if dropped:
                        logger.info(
                            "Hallucination filter %s/%s: dropped %d/%d "
                            "panels whose labels are not in the caption set",
                            paper_id,
                            figure_id,
                            dropped,
                            pre_filter,
                        )
                # Enrich LLM-first results with scale_bar + geology_links.
                # Without this, the LLM-first path skips the metadata
                # enrichment that the classical path applies at the end of
                # _process_region, leaving the Web UI's scale/geology panels
                # empty for every LLM-extracted figure.
                llm_results = self._enrich_llm_first_results(
                    llm_results,
                    caption=caption,
                    region_img=region_img,
                    section_links=section_links,
                    grobid_sections=grobid_sections,
                )
                return llm_results
            logger.debug(
                "LLM-first failed for %s/%s, falling back to classical path",
                paper_id,
                figure_id,
            )

        # ---- Classical path (segmentation + OCR + matching) --------------------
        # ---- M3 Stage 1 + 2 (text + vision, run once per region) ---------------
        m3_caption_pairs: list[CaptionPair] = []
        m3_plate_cls = None
        m3_panels: list[PanelBox] = []
        m3_diag: dict[str, Any] = {}
        if self.m3_engine is not None:
            try:
                from PIL import Image as _PILImage

                # Ensure region_img is RGB PIL for the engine
                if hasattr(region_img, "shape"):
                    _rgb = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
                    plate_pil = _PILImage.fromarray(_rgb)
                else:
                    with _PILImage.open(str(region_img)) as _im:
                        plate_pil = _im.convert("RGB")
                # Stage 1: caption parser
                if self.m3_engine._stage_enabled(1):
                    m3_caption_pairs = self._m3_call_with_fallback(
                        self.m3_engine.parse_caption,
                        caption.caption or "",
                        lang=_resolve_m3_prompt_lang(self.config.extra.get("m3_prompt_lang")),
                    )
                    m3_diag["stage1_pairs"] = len(m3_caption_pairs)
                # Stage 2: plate classifier — early exit on non-radiolarian
                if self.m3_engine._stage_enabled(2):
                    m3_plate_cls = self._m3_call_with_fallback(
                        self.m3_engine.classify_plate, plate_pil
                    )
                    m3_diag["stage2_class"] = m3_plate_cls.to_dict()
                    if not m3_plate_cls.is_radiolarian_plate:
                        logger.info(
                            "M3 Stage 2: %s/%s rejected (not a radiolarian plate): %s",
                            paper_id,
                            figure_id,
                            m3_plate_cls.reasoning[:120],
                        )
                        # Annotate each potential panel as "rejected by classifier"
                        # and return an empty match list with the diagnostic saved.
                        if self.config.save_intermediate:
                            write_json(
                                self.config.manifests_dir()
                                / paper_id
                                / f"{slugify(figure_id)}.json",
                                {
                                    "paper_id": paper_id,
                                    "figure_id": figure_id,
                                    "caption": caption.caption,
                                    "figure_number": caption.figure_number,
                                    "page_index": best_page_index,
                                    "region": asdict(region),
                                    "m3_diagnostic": m3_diag,
                                    "m3_rejected": True,
                                    "m3_rejection_reason": m3_plate_cls.reasoning,
                                    "panels": [],
                                    "matches": [],
                                },
                            )
                        return []
                # Stage 3: panel segmentation hint
                if self.m3_engine._stage_enabled(3):
                    hint = (
                        m3_plate_cls.panel_count_estimate
                        if m3_plate_cls and m3_plate_cls.panel_count_estimate
                        else None
                    )
                    m3_panels = self.m3_engine.segment_panels(plate_pil, hint_count=hint)
                    m3_diag["stage3_panels"] = [p.to_dict() for p in m3_panels]
            except Exception as exc:
                logger.exception("M3 stage 1-3 failed; falling back to classical pipeline: %s", exc)
                m3_caption_pairs = []
                m3_panels = []
                m3_plate_cls = None

        # ---- Classical CV: panel segmentation + OCR + rule-based match ----------
        with self._seg_lock:
            panels = self.segmenter.segment_image(region_img)
        # OCR is intentionally not locked at this layer — see __init__ for
        # the rationale. Engine init is protected inside OCRBackend itself.
        if (not panels or len(panels) == 0) and m3_panels:
            h_img, w_img = region_img.shape[:2]
            panels = [
                PanelCandidate(
                    panel_id=mp.panel_id,
                    bbox=(
                        int(mp.bbox[0]),
                        int(mp.bbox[1]),
                        int(mp.bbox[2]),
                        int(mp.bbox[3]),
                    ),
                    score=mp.confidence,
                    metadata={
                        "method": "m3_stage3",
                        "morphology": mp.morphology,
                        "visible_label": mp.visible_label,
                    },
                )
                for mp in m3_panels
            ]
            logger.info("M3 Stage 3: substituted %d panels for empty SAM2 result", len(panels))
        # Augment classical panels with M3's visible_label / morphology hints
        elif m3_panels and panels:
            try:
                panels = _merge_panel_hints(panels, m3_panels)
            except Exception:
                logger.exception("Failed to merge M3 panel hints")
            # If M3 found substantially more panels than the classical path
            # (e.g. classical CV merged nearby specimens into a single blob
            # while M3 separates them visually), add the unmatched M3 panels
            # so we don't silently lose specimens. This is critical for plates
            # where the radiolarian specimens are touching or weakly separated.
            try:
                added = _add_unmatched_m3_panels(panels, m3_panels, iou_match=0.10)
                if added:
                    logger.info("M3 Stage 3: added %d unmatched M3 panels", added)
            except Exception:
                logger.exception("Failed to add unmatched M3 panels")
        if not panels:
            h_img, w_img = region_img.shape[:2]
            panels = [
                PanelCandidate(
                    panel_id="P1",
                    bbox=(0, 0, int(w_img), int(h_img)),
                    score=0.4,
                    metadata={"fallback": "full_region_panel"},
                )
            ]

        ocr_tokens = normalize_ocr_tokens(self.ocr.recognize(region_img))
        taxon_entities = self.taxon.predict(caption.caption or "")

        # Scale bar: caption + OCR + visual line detection
        caption_scale = extract_scale_from_caption(caption.caption or "")
        ocr_text_block = " ".join(tok.text for tok in ocr_tokens)
        ocr_scale = extract_scale_from_ocr_text(ocr_text_block)
        px_len = detect_scale_bar_length_px(region_img)
        merged_scale = merge_scale_info(caption_scale, ocr_scale, pixel_length=px_len)

        # NMS pass on the panel list: merge near-duplicate detections
        # from the segmenter (e.g. SAM2 returned the full specimen and
        # OpenCV returned the same specimen split into two boxes). This
        # must run before per-panel OCR so we don't pay for OCR'ing a
        # duplicate.
        from .association import deduplicate_panels_nms

        pre = len(panels)
        panels = deduplicate_panels_nms(panels, iou_threshold=0.6, label_match=False)
        if len(panels) != pre:
            logger.info("NMS dedup: %d → %d panels for %s", pre, len(panels), figure_id)

        h_img, w_img = region_img.shape[:2]
        for panel_index, panel in enumerate(panels, start=1):
            x, y, w, h = panel.bbox
            # Clip bbox to image bounds. The segmenter occasionally returns
            # coords slightly outside the image (e.g. SAM2 prompts at the edge
            # produce bboxes that go a few pixels past the boundary). Without
            # this clip, numpy would silently produce a smaller crop than the
            # recorded bbox suggests, and the bbox stored on the panel would
            # disagree with the saved panel_path dimensions.
            x0 = max(0, min(int(x), w_img))
            y0 = max(0, min(int(y), h_img))
            x1 = max(x0, min(int(x + w), w_img))
            y1 = max(y0, min(int(y + h), h_img))
            if x1 <= x0 or y1 <= y0:
                # Fully out-of-bounds panel (e.g. from a misaligned M3 hint);
                # skip rather than save an empty image.
                continue
            panel.bbox = (x0, y0, x1 - x0, y1 - y0)
            x, y, w, h = panel.bbox
            crop = region_img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            panel_dir = ensure_dir(
                self.config.panels_dir() / paper_id / (figure_id or f"fig_{figure_index}")
            )
            panel_path = panel_dir / f"panel_{panel_index:02d}.png"
            # Round 15 audit: cv2.imwrite returns False on failure
            # (disk full, invalid path, encoding error) but the previous
            # code stored image_path anyway — leaving the panel referenced
            # in results with no actual crop file on disk.
            if not cv2.imwrite(str(panel_path), crop):
                logger.warning("cv2.imwrite failed for %s; skipping panel", panel_path)
                continue
            panel.image_path = str(panel_path)
            panel.region_id = region.region_id
            panel.source_page = region.page_index
            panel.panel_index = panel_index
            # Per-panel OCR: re-OCR the tight crop and attach to the panel.
            # Falls back silently to the region-level OCR tokens if the
            # backend fails to initialise.
            try:
                panel_tokens = self.ocr.recognize_panel(region_img, (x, y, w, h))
                if panel_tokens:
                    panel.metadata = panel.metadata or {}
                    panel.metadata["panel_ocr_text"] = " ".join(t.text for t in panel_tokens)
                    panel.metadata["panel_ocr_token_count"] = len(panel_tokens)
            except Exception:
                pass
            # Label-region re-read: plate labels ("1", "2a", "Fig. 3")
            # live in a corner of the panel. OCR'ing the full panel
            # dilutes that signal with the specimen's body. Crop the
            # corner band, OCR it, and use the highest-confidence short
            # numeric/alphanumeric token to override the panel's
            # existing panel_id (which came from positional assignment
            # or SAM2 prompt order).
            try:
                label_tokens = self.ocr.recognize_panel_label(
                    region_img,
                    (x, y, w, h),
                    label_corner="adaptive",
                )
                # N10: if corner OCR returned nothing, fall back to the
                # full-panel OCR tokens (which include the whole panel,
                # not just the corner band). This rescued 100% of bandini2011
                # panels where the corner band was too small to OCR.
                if not label_tokens:
                    full_tokens = self.ocr.recognize_panel(region_img, (x, y, w, h))
                    label_tokens = full_tokens
                    if label_tokens:
                        panel.metadata = panel.metadata or {}
                        panel.metadata["label_region_fallback"] = "full_panel"
                if label_tokens:
                    panel.metadata = panel.metadata or {}
                    panel.metadata["label_region_ocr"] = " ".join(t.text for t in label_tokens)
                    # Pick the best short label-like token
                    best = None
                    for tok in label_tokens:
                        t = (tok.text or "").strip()
                        if not t or len(t) > 6:
                            continue
                        if best is None or tok.confidence > best.confidence:
                            best = tok
                    if best is not None:
                        norm = _normalize_panel_label(best.text)
                        # Only override if the OCR'd token is a genuine
                        # panel label shape. The previous check
                        # ``(norm.isdigit() or len(norm) <= 3)`` accepted
                        # any 1-3 char string, so OCR garbage like 'ean',
                        # 'L', 'P1', ',1' became panel_ids — polluting
                        # the figure's label space and colliding with real
                        # labels via positional fallback (N10-class drift).
                        # ``is_valid_panel_label`` rejects those.
                        if norm and is_valid_panel_label(norm):
                            prev_caption_id = (
                                panel.metadata.get("caption_panel_id") or panel.panel_id
                            )
                            panel.panel_id = norm
                            panel.metadata["caption_panel_id"] = prev_caption_id
                            panel.metadata["printed_panel_id"] = norm
                            panel.metadata["panel_id_source"] = "image_ocr"
                            panel.metadata["label_region_picked"] = best.text
            except Exception:
                pass

        matches = match_panels(
            paper_id,
            figure_id,
            caption,
            panels,
            ocr_tokens,
            taxon_entities,
            use_neural_matcher=bool(self.config.extra.get("use_neural_matcher", False)),
            matcher_checkpoint_path=self.config.extra.get("matcher_checkpoint_path"),
            image_shape=region_img.shape[:2],
            paper_metadata=paper_metadata,
            caption_pairs=m3_caption_pairs,
        )

        # Round 12 (Bug 7 fix): classical-path over-emission filter.
        # The segmenter over-segments plates and OCR mis-reads some crop
        # labels, so the classical branch can emit rows whose panel_id
        # is NOT mentioned in the caption. Live smoke on Pouille 2014
        # pl02 found 4 phantom rows (pid=10a/10b/11b/11c) that the
        # caption-derived pair_lookup rejects. Apply the same
        # hallucination filter used for the LLM-first branch (above).
        if matches and (m3_caption_pairs or caption.caption):
            pre = len(matches)
            matches = self._filter_classical_hallucinations(
                matches,
                caption,
                paper_id,
                figure_id,
            )
            dropped = pre - len(matches)
            if dropped:
                logger.info(
                    "Classical hallucination filter %s/%s: dropped %d/%d "
                    "rows whose panel_id is not in the caption",
                    paper_id,
                    figure_id,
                    dropped,
                    pre,
                )

        # ---- M3 Stage 4 (per-panel matching) with stage 1 caption context ----
        if self.m3_engine is not None and self.m3_engine._stage_enabled(4):
            # Skip visual-only stage 4 for non-specimen content. Three cases:
            # 1. Stage 2 already classified the figure as non-radiolarian.
            # 2. Image type is diagram/photo/other with no caption (no signal).
            # 3. Caption itself signals a non-specimen ("auto-generated",
            #    "page header", "placeholder", ...).
            skip_stage4 = False
            skip_reason = ""
            if m3_plate_cls is not None and not m3_plate_cls.is_radiolarian_plate:
                skip_stage4 = True
                skip_reason = f"stage2 rejected (is_radiolarian_plate=False, reasoning={m3_plate_cls.reasoning[:60]!r})"
            elif m3_plate_cls is not None:
                it = (m3_plate_cls.image_type or "").lower()
                if it in {"diagram", "photo", "other"} and not m3_caption_pairs:
                    skip_stage4 = True
                    skip_reason = f"type={it} with no caption"
            if _looks_like_placeholder_caption(caption.caption or ""):
                # Independent of stage 2 — a "Page 1 auto-generated image" caption
                # is a placeholder even if stage 2 accepted the plate.
                skip_stage4 = True
                skip_reason = f"placeholder caption: {caption.caption[:60]!r}"
            if skip_stage4:
                logger.info(
                    "M3 Stage 4: skipping %s/%s (%s)",
                    paper_id,
                    figure_id,
                    skip_reason,
                )
                m3_diag["stage4_skipped"] = skip_reason
            if not skip_stage4:
                with self._gemma_lock_if_needed():
                    matches = self._apply_m3_stage4(
                        matches=matches,
                        caption_pairs=m3_caption_pairs,
                        caption_text=caption.caption or "",
                        region_img=region_img,
                    )
        elif self.gemma_runtime is not None:
            # Backward-compatible single-stage M3 fallback
            with self._gemma_lock:
                matches = self._apply_gemma_with_fallback(
                    matches=matches,
                    caption_text=caption.caption or "",
                    ocr_labels=[tok.text for tok in ocr_tokens],
                    paper_id=paper_id,
                    figure_id=str(figure_id or f"fig_{figure_index}"),
                )

        # ---- M3 Stage 5 (cross-panel self-critique) ----------------------------
        if self.m3_engine is not None and self.m3_engine._stage_enabled(5) and matches:
            with self._gemma_lock_if_needed():
                matches = self._apply_m3_stage5(
                    matches=matches,
                    caption_pairs=m3_caption_pairs,
                    caption_text=caption.caption or "",
                    region_img=region_img,
                )

        # Attach geology links and scale bar info.
        #
        # Two linking strategies, in order of preference:
        #   1. SPECIES-level: the panel's predicted species (from
        #      ``m.species``) was linked to fulltext sections by
        #      ``link_species_to_geology`` -- the standard path.
        #   2. PANEL-level: when the species-level link is empty
        #      (the panel has no detectable species, e.g. OCR is
        #      unavailable or the figure caption is a generic
        #      placeholder) we extract geology facts from THIS
        #      PANEL'S OWN CAPTION TEXT directly. The previous
        #      implementation fell back to "use the first species'
        #      links for every unmatched panel", which dumped 5
        #      age/formation records onto every panel of a paper
        #      with a generic caption -- a clear bug now fixed.
        from .geology_extraction import link_panels_to_geology as _link_panels

        # Round 18 audit fix: every panel used to receive the SAME
        # figure-level geology_links because ``panel_captions`` was
        # keyed on panel_id but valued with the same figure-level
        # caption text. The result was a 27-panel Beccaro paper all
        # stamped with the same "Fonzaso Formation" entry — even
        # though only a few panels actually correspond to that
        # formation. The expert reviewer flagged this as fabricated
        # data. The fix: only panels that have a panel-level caption
        # (non-placeholder) get per-panel geology. Panels whose own
        # caption is a placeholder get an EMPTY list and a
        # ``geology_scope="none"`` marker so the operator sees the
        # data gap instead of fabricated content.
        panel_captions: dict[str, str] = {}
        panel_keys: list[str] = []
        for i, m in enumerate(matches):
            base = m.panel_id or f"idx_{i}"
            key = base
            j = 1
            while key in panel_captions:
                key = f"{base}#{j}"
                j += 1
            panel_captions[key] = caption.caption or ""
            panel_keys.append(key)
        panel_geo: dict[str, list[dict[str, Any]]] = {}
        if any(v for v in panel_captions.values()):
            panel_geo = _link_panels(
                panel_captions,
                fallback_sections=grobid_sections or [],
            )
        for i, m in enumerate(matches):
            geo_list = section_links.get(m.species or "", [])
            if not geo_list:
                # Round 18 fix: only the FIRST panel in a figure with
                # a non-placeholder caption inherits the figure-level
                # geology as a default anchor. All other panels get
                # empty lists (data gap, not fabricated). The
                # ``geology_scope`` marker on the first panel tells
                # the operator that the data is figure-level, not
                # panel-specific.
                key = panel_keys[i] if i < len(panel_keys) else (m.panel_id or f"idx_{i}")
                panel_local_geo = panel_geo.get(key, [])
                if panel_local_geo and not _looks_like_placeholder_caption(
                    panel_captions.get(key, "")
                ):
                    geo_list = panel_local_geo
                    m.metadata["geology_scope"] = "panel"
                elif i == 0 and panel_local_geo:
                    # First panel as figure-level anchor. Marked so
                    # the operator can distinguish from panel-specific.
                    geo_list = panel_local_geo
                    m.metadata["geology_scope"] = "figure_anchor"
                else:
                    m.metadata["geology_scope"] = "none"
            m.metadata["scale_bar"] = merged_scale.to_dict()
            m.metadata["geology_links"] = geo_list[:5]
            m.metadata["m3_diagnostic"] = m3_diag

        # ---- Optional Paleobiology Database (PBDB) enrichment ----------------
        # Opt-in via config.extra["use_paleodb"]. Looks up each unique species
        # name from this figure and attaches taxonomy hierarchy + occurrence
        # records to m.metadata["paleodb"].  Failures degrade silently.
        if self.config.extra.get("use_paleodb"):
            self._attach_paleodb_metadata(matches)

        results: list[dict[str, Any]] = [m.to_dict() for m in matches]

        if self.config.save_intermediate:
            write_json(
                self.config.manifests_dir() / paper_id / f"{slugify(figure_id)}.json",
                {
                    "paper_id": paper_id,
                    "figure_id": figure_id,
                    "caption": caption.caption,
                    "figure_number": caption.figure_number,
                    "page_index": best_page_index,
                    "region": asdict(region),
                    "ocr": [asdict(t) for t in ocr_tokens],
                    "taxa": [asdict(t) for t in taxon_entities],
                    "fulltext_sections": grobid_sections,
                    "geology_links": section_links,
                    "knowledge_graph": knowledge_graph,
                    "scale_bar": merged_scale.to_dict(),
                    "m3_diagnostic": m3_diag,
                    "panels": [asdict(p) for p in panels],
                    "matches": results,
                },
            )
        return results

    def _enrich_llm_first_results(
        self,
        rows: list[dict[str, Any]],
        *,
        caption: CaptionRecord,
        region_img: Any,
        section_links: dict[str, list[dict[str, Any]]],
        grobid_sections: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Attach scale_bar + geology_links to LLM-first result rows.

        The LLM-first path returns early from ``_process_region`` before the
        classical enrichment block runs. Without this helper, every
        LLM-extracted figure would have empty ``scale_bar`` and
        ``geology_links`` in its metadata, breaking the Web UI's geology/scale
        panels and the downstream knowledge-graph export.
        """
        caption_scale = extract_scale_from_caption(caption.caption or "")
        ocr_scale = extract_scale_from_ocr_text("")  # no OCR tokens available
        px_len = detect_scale_bar_length_px(region_img)
        merged_scale = merge_scale_info(caption_scale, ocr_scale, pixel_length=px_len)

        from .geology_extraction import link_panels_to_geology as _link_panels

        panel_captions: dict[str, str] = {}
        panel_keys: list[str] = []
        for i, row in enumerate(rows):
            base = row.get("panel_id") or f"idx_{i}"
            key = base
            j = 1
            while key in panel_captions:
                key = f"{base}#{j}"
                j += 1
            panel_captions[key] = caption.caption or ""
            panel_keys.append(key)
        panel_geo: dict[str, list[dict[str, Any]]] = {}
        if any(v for v in panel_captions.values()):
            panel_geo = _link_panels(
                panel_captions,
                fallback_sections=grobid_sections or [],
            )
        for i, row in enumerate(rows):
            md = dict(row.get("metadata") or {})
            sp = row.get("species") or ""
            geo_list = section_links.get(sp, [])
            if not geo_list:
                # Round 18 fix: only the FIRST panel in a figure
                # inherits figure-level geology; others stay empty
                # so we don't fabricate per-panel data.
                key = panel_keys[i] if i < len(panel_keys) else (row.get("panel_id") or f"idx_{i}")
                panel_local_geo = panel_geo.get(key, [])
                if panel_local_geo and not _looks_like_placeholder_caption(
                    panel_captions.get(key, "")
                ):
                    geo_list = panel_local_geo
                    md["geology_scope"] = "panel"
                elif i == 0 and panel_local_geo:
                    geo_list = panel_local_geo
                    md["geology_scope"] = "figure_anchor"
                else:
                    md["geology_scope"] = "none"
            md["scale_bar"] = merged_scale.to_dict()
            md["geology_links"] = geo_list[:5]
            md.setdefault("m3_diagnostic", {})
            row["metadata"] = md
        return rows

    def _switch_to_fallback_backend(self) -> bool:
        """Build the configured local fallback backend and swap it in.

        Called after a FallbackRecommendedError (MiniMax 4xx). Uses the
        existing local-fallback builder (llama.cpp → ollama →
        transformers, whichever is configured) and points both the
        pipeline runtime and the M3 engine at the new backend.
        """

        # audit 2026-08-01 (D2): the assignment pair below
        # (``self.gemma_runtime = ...`` + ``self.m3_engine.backend = ...``)
        # used to be unprotected. Multiple workers can each catch a
        # FallbackRecommendedError at the same time and call this
        # method concurrently — N callers each load a fresh local
        # model in parallel, OOMing the box. Take the module-level
        # lock so the local backend is built and swapped exactly once
        # even under fan-out. The second caller sees the already-
        # swapped runtime and returns immediately.
        with _BACKEND_SWITCH_LOCK:
            new_runtime = self._build_local_gemma_fallback()
            if new_runtime is None:
                logger.warning(
                    "FallbackRecommendedError but no local backend configured; "
                    "giving up on M3 for this call"
                )
                return False
            self.gemma_runtime = new_runtime
            if self.m3_engine is not None:
                self.m3_engine.backend = new_runtime.backend
            logger.info("Switched M3 backend to %s", getattr(new_runtime, "backend_name", "?"))
            return True

    def _m3_call_with_fallback(self, fn, *args, **kwargs):
        """Call an M3Engine method; on FallbackRecommendedError switch
        to the configured fallback backend once and retry the call.

        audit 2026-07-31: the fallback-backend feature (Phase 61 Plan 4)
        was never wired — 4xx errors were swallowed by generic
        except-Exception handlers and the configured fallback was never
        used. This wrapper is the single interception point.
        """
        from .llm_backends import FallbackRecommendedError

        try:
            return fn(*args, **kwargs)
        except FallbackRecommendedError as fre:
            logger.warning(
                "M3 backend requested fallback (%s); switching backends",
                getattr(fre, "recommended_backend", "?"),
            )
            if self._switch_to_fallback_backend():
                return fn(*args, **kwargs)
            raise

    def _gemma_lock_if_needed(self):
        """Return a lock context manager for backends requiring serialization.

        Transformers' ``model.generate()`` shares mutable internal state
        across threads (random state, KV cache), so concurrent calls from
        the ThreadPoolExecutor workers corrupt each other. MiniMax has its
        own semaphore (``max_concurrent``); Ollama/llama.cpp are stateless
        HTTP calls. Serializing those would kill MiniMax's throughput for
        no safety benefit, so we only lock for the Transformers backend.
        """
        if (
            self.gemma_runtime is not None
            and str(self.gemma_runtime.backend_name).lower() == "transformers"
        ):
            return self._gemma_lock
        return nullcontext()

    def _apply_m3_stage4(
        self,
        matches: list,
        caption_pairs: list[CaptionPair],
        caption_text: str,
        region_img: Any,
    ) -> list:
        """Stage 4: re-match each panel via M3 with structured caption context."""
        from PIL import Image as _PILImage

        # If caption parsing found nothing AND the user opted to skip, fall through
        if not caption_pairs and self.m3_engine.config.get("m3_skip_match_on_empty_caption", True):
            return matches
        # Deduplicate matches by (panel_id, bbox-tuple) before calling
        # M3 stage 4. The pre-fix code called M3 once per match row,
        # but a Stage-3 over-segmentation that produced N copies of
        # the same physical panel (e.g. 4 detections of the same
        # crop region) would all share panel_id="1" and a similar
        # bbox, wasting API calls. We keep the first occurrence
        # (which has the highest panel_score from the classical
        # detector) and skip the rest.
        seen_panel_keys: set[tuple[str | None, tuple[int, ...] | None]] = set()
        deduped_matches: list = []
        for m in matches:
            bbox_tuple = tuple(m.bbox) if m.bbox is not None else None
            key = (m.panel_id, bbox_tuple)
            if key in seen_panel_keys:
                continue
            seen_panel_keys.add(key)
            deduped_matches.append(m)
        new_matches = []
        for m in deduped_matches:
            try:
                if not m.panel_path or not Path(m.panel_path).is_file():
                    new_matches.append(m)
                    continue
                with _PILImage.open(m.panel_path) as im:
                    panel_image = im.convert("RGB")
                suggested = None
                # If panel metadata has a visible_label from M3 stage 3, use it
                md = m.metadata or {}
                suggested = md.get("m3_visible_label") or md.get("visible_label")
                panel_match: PanelMatch = self.m3_engine.match_panel(
                    panel_image=panel_image,
                    caption_pairs=caption_pairs,
                    caption_text=caption_text,
                    suggested_label=suggested,
                )
                # Merge: prefer M3 result when its confidence >= 0.40 OR it has
                # a different species from the rule-based guess (M3 sees more).
                m3_conf = panel_match.confidence
                rule_conf = float(m.confidence or 0.0)
                # Always record M3 vote for diagnostic
                md["m3_stage4"] = {
                    "label": panel_match.label,
                    "species": panel_match.species,
                    "confidence": m3_conf,
                    "alternative": panel_match.alternative,
                    "is_radiolarian": panel_match.is_radiolarian,
                    "reasoning": panel_match.reasoning,
                    "votes": (panel_match.raw or {}).get("votes", 1),
                    "agreement": (panel_match.raw or {}).get("agreement", 1.0),
                }
                # Carry MiniMax telemetry (request id, cost, model version,
                # token usage) from the backend call into MatchResult
                # metadata. Without this plumbing, M3 stage-4 calls never
                # reach /system/llm-status aggregation because the cost
                # only lived transiently inside ``PanelMatch.raw``. /system/
                # llm-status aggregates via match.metadata across the row.
                for tk, tv in (panel_match.raw or {}).items():
                    if tk.startswith("MiniMax_") and tk not in md:
                        md[tk] = tv
                use_m3 = False
                if panel_match.is_radiolarian and panel_match.species:
                    # Use M3 if it's at least moderately confident AND
                    # either it has higher confidence than rules OR it disagrees
                    # (M3 has the visual signal that rules don't).
                    if m3_conf >= 0.40 and (
                        m3_conf > rule_conf
                        or (m.species and panel_match.species.lower() != m.species.lower())
                        or (not m.species)
                    ):
                        use_m3 = True
                if use_m3:
                    m.panel_id = panel_match.label or m.panel_id
                    m.species = panel_match.species or m.species
                    m.label_text = panel_match.label or m.label_text
                    # Use m3_conf directly, not max(rule_conf, m3_conf). The
                    # two scores come from different scoring systems (rule-
                    # based heuristic vs M3 LLM); combining them with max()
                    # can mask the M3's lower-but-more-honest score and
                    # inflate downstream thresholds. When M3 is selected we
                    # trust its verdict and use its confidence verbatim.
                    m.confidence = m3_conf
                    md["gemma_used"] = True
                    md["gemma_confidence"] = m3_conf
                    md["gemma_reasoning"] = panel_match.reasoning
                else:
                    md["gemma_used"] = False
                    # IMPORTANT: distinguish three different "M3 didn't produce a
                    # match" cases, each with a different downstream consequence:
                    #   1. Real API / runtime error (raw["error"] set)
                    #      -> gemma_error  -> FallbackHandler popup (user can retry)
                    #   2. M3 said "not a radiolarian specimen"
                    #      -> m3_rejected_non_radiolarian  -> silently dropped
                    #   3. M3 returned low-confidence verdict
                    #      -> gemma_fallback  -> silently dropped (no error UI)
                    m3_error = (panel_match.raw or {}).get("error")
                    if m3_error:
                        md["gemma_error"] = str(m3_error)
                        md["gemma_error_type"] = "M3EngineError"
                        md["gemma_reasoning"] = panel_match.reasoning or m3_error
                    elif not panel_match.is_radiolarian:
                        md["m3_rejected_non_radiolarian"] = True
                        md["gemma_reasoning"] = (
                            panel_match.reasoning or "M3: not a radiolarian specimen"
                        )
                    else:
                        md["gemma_fallback"] = True
                        md["gemma_reasoning"] = panel_match.reasoning or "M3 below threshold"
                m.metadata = md
                new_matches.append(m)
            except Exception as exc:
                logger.exception("M3 stage 4 failed for one panel: %s", exc)
                md = dict(m.metadata or {})
                md["m3_stage4_error"] = str(exc)
                m.metadata = md
                new_matches.append(m)
        return new_matches

    def _apply_m3_stage5(
        self,
        matches: list,
        caption_pairs: list[CaptionPair],
        caption_text: str,
        region_img: Any,
    ) -> list:
        """Stage 5: cross-validate all panel matches via M3 self-critique."""
        from PIL import Image as _PILImage

        try:
            if hasattr(region_img, "shape"):
                rgb = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
                plate_pil = _PILImage.fromarray(rgb)
            else:
                with _PILImage.open(str(region_img)) as _im:
                    plate_pil = _im.convert("RGB")
            panel_matches: list[PanelMatch] = []
            for idx, m in enumerate(matches, start=1):
                pid = str(m.panel_id or f"P{idx}")
                panel_matches.append(
                    PanelMatch(
                        panel_id=pid,
                        label=m.label_text,
                        species=m.species,
                        confidence=float(m.confidence or 0.0),
                        reasoning=(m.metadata or {}).get("gemma_reasoning", ""),
                    )
                )
            critiques = self.m3_engine.critique_matches(
                plate_image=plate_pil,
                matches=panel_matches,
                caption_text=caption_text,
                caption_pairs=caption_pairs,
            )
            M3Engine.apply_critiques(panel_matches, critiques)
            # Back-apply to original matches by INDEX, not by panel_id.
            # ``panel_matches`` and ``matches`` are parallel arrays built in
            # the same loop above, so index correspondence is guaranteed.
            # The previous ``by_id`` dict approach broke when two matches
            # shared a panel_id (OCR duplicates): it kept only the
            # highest-confidence PanelMatch and applied its critique to BOTH
            # matches, corrupting the lower-confidence one's species.
            for idx, m in enumerate(matches):
                pm = panel_matches[idx] if idx < len(panel_matches) else None
                if pm is None:
                    continue
                md = dict(m.metadata or {})
                if "critique" in (pm.raw or {}):
                    md["m3_stage5_critique"] = pm.raw["critique"]
                if pm.species and pm.species != m.species:
                    md["m3_stage5_override"] = {
                        "from": m.species,
                        "to": pm.species,
                    }
                    m.species = pm.species
                    # Guard: m.confidence may be None on rare upstream paths
                    # (e.g. when a sub-pipeline writes a partial MatchResult
                    # before scoring); min(None, ...) raises TypeError.
                    cur = float(m.confidence) if m.confidence is not None else 0.0
                    m.confidence = min(cur, max(0.3, float(pm.confidence or 0.0)))
                m.metadata = md
        except Exception as exc:
            logger.exception("M3 stage 5 failed: %s", exc)
        return matches

    # ------------------------------------------------------------------ helpers

    def _apply_gemma_with_fallback(
        self,
        matches: list,
        caption_text: str,
        ocr_labels: list[str],
        paper_id: str,
        figure_id: str,
    ) -> list:
        """Call Gemma (possibly MiniMax) and route errors through FallbackHandler.

        Decision flow when MiniMax M3 returns ``fallback_used=True`` errors:
            1. Ask ``gemma_fallback_handler`` (CLI popup or callback).
            2. ``"gemma4"`` -> lazy-init local Gemma4 and retry once.
            3. ``"rules"``  -> keep rule-pipeline matches untouched.
            4. ``"stop"``   -> raise so the user sees the failure.
            5. ``"retry"``  -> call MiniMax once more with same payload.
        """
        conf_threshold = float(self.config.extra.get("gemma_conf_threshold", 0.70))
        prompt_lang = str(self.config.extra.get("gemma_prompt_lang", "zh"))

        def _call_once(runtime):
            return apply_gemma_to_matches(
                runtime=runtime,
                matches=matches,
                caption_text=caption_text,
                ocr_labels=ocr_labels,
                conf_threshold=conf_threshold,
                prompt_lang=prompt_lang,
            )

        # First attempt with the primary runtime.
        result = _call_once(self.gemma_runtime)
        if not self._matches_have_fallback_error(result):
            return result

        # If we don't have a fallback handler (non-MiniMax backend), just return.
        if self.gemma_fallback_handler is None:
            return result

        error_info = self._collect_fallback_error_info(result, paper_id, figure_id)
        # Round 18 audit: M3 refuses to extract species from non-specimen
        # figures (bar charts, tables, maps). Surfacing these as popup
        # decisions wastes operator time and asks for a no-op answer.
        # Silently skip the figure when the error text clearly indicates
        # "this is not a specimen image". The figure is still recorded
        # in m3_diagnostic with m3_rejected=True so the operator can
        # audit the skip after the run.
        if error_info.get("is_non_specimen_figure"):
            logger.info(
                "M3 returned 'non-specimen' refusal for %s/%s; silently skipping (no popup): %s",
                paper_id,
                figure_id,
                (error_info.get("error") or "")[:200],
            )
            for m in result:
                m.metadata["MiniMax_fallback_action"] = "skipped_non_specimen"
            return result
        action = self.gemma_fallback_handler(error_info)

        if action == "stop":
            raise RuntimeError(
                f"[MiniMax] user stopped pipeline at paper={paper_id} figure={figure_id}: "
                f"{error_info.get('error', '?')}"
            )

        if action == "rules":
            logger.warning(
                "[MiniMax] API error, falling back to rule pipeline for %s/%s",
                paper_id,
                figure_id,
            )
            for m in result:
                m.metadata["MiniMax_fallback_action"] = "rules"
            return result

        if action == "gemma4":
            logger.warning(
                "[MiniMax] API error, switching to local Gemma4 for %s/%s",
                paper_id,
                figure_id,
            )
            # Bug #7 fix: cache the local Gemma4 runtime after the first
            # successful build so subsequent fallbacks don't reload a multi-GB
            # model each time.
            # Round 15 audit: previous unguarded lazy-init could let two
            # concurrent MiniMax-fallback workers both see ``None``, both
            # build the multi-GB Gemma4 model, and OOM the box. Double-
            # checked locking under self._gemma_lock.
            if self._fallback_gemma_runtime is None:
                with self._gemma_lock:
                    if self._fallback_gemma_runtime is None:
                        self._fallback_gemma_runtime = self._build_local_gemma_fallback()
            if self._fallback_gemma_runtime is None:
                logger.warning("Local Gemma4 fallback unavailable; keeping rule results.")
                for m in result:
                    m.metadata["MiniMax_fallback_action"] = "rules_no_local_gemma"
                return result
            retried = _call_once(self._fallback_gemma_runtime)
            for m in retried:
                m.metadata["MiniMax_fallback_action"] = "gemma4"
            return retried

        if action == "retry":
            logger.warning("[MiniMax] API error, retrying once for %s/%s", paper_id, figure_id)
            retried = _call_once(self.gemma_runtime)
            for m in retried:
                m.metadata["MiniMax_fallback_action"] = "retry"
            return retried

        return result

    @staticmethod
    def _matches_have_fallback_error(matches: list) -> bool:
        # Thin wrapper kept for backward compatibility. The real
        # implementation lives in ``text_filters`` so the eval harness
        # and unit tests can import it without pulling torch / gemma.
        from .text_filters import matches_have_fallback_error

        return matches_have_fallback_error(matches)

    def _attach_paleodb_metadata(self, matches: list) -> None:
        """Look up each unique species in PBDB and attach taxonomy + occurrences.

        On any failure (network, missing name, PBDB rate-limit) the match is
        left with an empty ``paleodb`` dict and the pipeline continues. Results
        are cached on disk so a second run is instant.
        """
        from .paleodb import PaleoDB

        # Phase 54 audit: H1 — the previous ``... or 25`` guard treated
        # a legitimate ``0 = disable occurrences`` as falsy and silently
        # upgraded it to 25. That wasted PBDB quota and shipped
        # occurrence rows the operator opted out of. The default now
        # comes only from ``dict.get(..., 25)``; an explicit 0 is
        # preserved.
        max_occ = int(self.config.extra.get("paleodb_max_occurrences", 25))
        endpoint = self.config.extra.get("paleodb_endpoint")
        cache_dir = self.config.extra.get("paleodb_cache_dir")
        offline = bool(self.config.extra.get("paleodb_offline", False))
        try:
            client = PaleoDB(
                endpoint=endpoint,
                cache_dir=cache_dir,
                min_interval=0.2,
                offline=offline,
            )
        except Exception as exc:
            logger.warning("PaleoDB init failed: %s", exc)
            for m in matches:
                m.metadata["paleodb"] = {"error": str(exc)}
            return

        # Deduplicate species names so we only do one lookup per unique taxon
        unique_species: dict[str, None] = {}
        for m in matches:
            if m.species and m.species.strip():
                unique_species.setdefault(m.species.strip(), None)

        for name in unique_species:
            try:
                tax = client.lookup_species(name)
                # Phase 31: genus-level fallback. When the species-level
                # lookup fails (common for extant Cenozoic species that
                # PBDB doesn't index at the species rank), try the
                # genus itself. PBDB returns the full classification
                # hierarchy so we can populate family/order/class_
                # even when no species-level record exists.
                #
                # Bug-fix M-1: occurrences are species-specific (PBDB
                # indexes them by full binomial). Looking up
                # occurrences on genus fallback would yield wrong
                # biozone / lat / lon for the species we actually
                # have. We track ``tax_from_genus`` separately and
                # skip the occurrence lookup when the taxonomy came
                # from the genus fallback path.
                tax_from_genus = False
                if tax is None and " " in name:
                    genus_name = name.split()[0].strip()
                    if genus_name:
                        try:
                            genus_tax = client.lookup_genus(genus_name)
                            if genus_tax is not None:
                                tax = genus_tax
                                tax_from_genus = True
                                logger.info(
                                    "PBDB species miss for %s; "
                                    "genus fallback filled family=%s "
                                    "order=%s class=%s",
                                    name,
                                    tax.family,
                                    tax.order,
                                    tax.class_,
                                )
                        except Exception as exc:
                            logger.debug(
                                "PBDB genus fallback failed for %s: %s",
                                genus_name,
                                exc,
                            )
                # Skip occurrence lookup on genus fallback path.
                occs = (
                    client.lookup_occurrences(name, max_n=max_occ)
                    if tax and not tax_from_genus
                    else []
                )
            except Exception as exc:
                logger.warning("PBDB lookup failed for %s: %s", name, exc)
                tax = None
                occs = []
            payload = {
                "taxonomy": tax.to_dict() if tax else None,
                "occurrences": [o.to_dict() for o in occs],
                "occurrence_count": len(occs),
                "looked_up": True,
            }
            for m in matches:
                if (m.species or "").strip() == name:
                    m.metadata["paleodb"] = payload

    @staticmethod
    def _collect_fallback_error_info(
        matches: list, paper_id: str, figure_id: str
    ) -> dict[str, Any]:
        # Prefer gemma_error; fall back to gemma_reasoning (always set by
        # _make_error_result to "MiniMax API error: <Type>: <message>"); only
        # as last resort show the placeholder. This is what the Web
        # FallbackHandler popup surfaces to the user.
        first_err = next(
            (
                m.metadata.get("gemma_error", "") or m.metadata.get("gemma_reasoning", "")
                for m in matches
                if m.metadata.get("gemma_error") or m.metadata.get("gemma_reasoning")
            ),
            "",
        )
        first_type = next(
            (
                m.metadata.get("gemma_error_type", "") or m.metadata.get("gemma_fallback_type", "")
                for m in matches
                if m.metadata.get("gemma_error_type") or m.metadata.get("gemma_fallback_type")
            ),
            "",
        )
        first_fb = next(
            (
                m.metadata.get("gemma_fallback", False)
                for m in matches
                if m.metadata.get("gemma_fallback")
            ),
            False,
        )
        return {
            "error": first_err or "MiniMax returned fallback_used=True (see stderr for traceback)",
            "error_type": first_type or ("MiniMaxAPIError" if first_fb else "Unknown"),
            "context": f"paper={paper_id} figure={figure_id} affected_panels="
            f"{sum(1 for m in matches if m.metadata.get('gemma_error') or m.metadata.get('gemma_fallback'))}",
            # Round 18 audit: M3 frequently refuses to extract species
            # from non-specimen figures (bar charts, tables, maps,
            # publication-count graphs) and returns reasoning like
            # "该panel为图表…无标签与物种可判定". The previous code
            # surfaced this as a MiniMaxAPIError to the popup, which
            # (a) wasted the operator's time on a no-decision and
            # (b) burned API cost on a question with only one answer
            # (skip the figure). Mark these cases so the popup
            # handler can short-circuit to "silently skip" instead of
            # asking the user to choose a fallback.
            "is_non_specimen_figure": RadiolarianPipeline._looks_like_non_specimen_error(
                first_err, first_type
            ),
        }

    @staticmethod
    def _looks_like_non_specimen_error(err: str, err_type: str) -> bool:
        """Return True if the MiniMax error text / type signals that
        the figure isn't a radiolarian specimen image.

        M3's stage-2 / LLM-first paths return deliberate refusals
        for non-specimen content ("this is a bar chart, no species
        to extract"). Surfacing those as MiniMaxAPIError to the
        popup makes the operator click through a no-op. Detect them
        and silently skip the figure instead.
        """
        if not err and not err_type:
            return False
        # Patterns that indicate M3 correctly refused to extract
        # species because the figure isn't a radiolarian specimen
        # image. Mix of Chinese ("该panel", "非标本") and English
        # ("bar chart", "no specimen panels") markers; lowercased
        # for case-insensitive matching.
        non_specimen_markers = (
            "该panel",
            "并非",
            "不涉及",
            "无标签",
            "无物种",
            "不可判定",
            "不是放射虫",
            "不是标本",
            "非标本",
            "非放射虫",
            "不是图版",
            "不是放射虫图版",
            "非图版",
            "非显微",
            "bar chart",
            "bar graph",
            "柱状图",
            "统计图",
            "折线图",
            "数量统计",
            "publication count",
            "publication number",
            "no specimen",
            "no panel",
            "not a radiolarian",
            "not a specimen",
            "no radiolarian",
            "is not a radiolarian",
            "is not a specimen",
            "no specimen panels",
            "no panels found",
            "chart of",
            "graph of",
            "table of",
            "is a chart",
            "is a table",
            "is a graph",
            "is a diagram",
            "is a map",
            "is a photo",
            "is a photomicrograph",
            "is text",
            "is a title page",
            "is a reference",
        )
        haystack = ((err or "") + " " + (err_type or "")).lower()
        return any(m.lower() in haystack for m in non_specimen_markers)

    def _build_local_gemma_fallback(self):
        """Try to build a local Gemma4 runtime as fallback target.

        Uses llama.cpp if a host is configured, otherwise transformers.
        Returns None if no local option is available.
        """
        extra = dict(self.config.extra)
        # Override the backend selection to a local one (prefer llama.cpp).
        if extra.get("llama_host") or extra.get("llama_model"):
            extra["llm_backend"] = "llamacpp"
        elif extra.get("ollama_model") or extra.get("ollama_host"):
            extra["llm_backend"] = "ollama"
        elif extra.get("gemma_model_path"):
            extra["llm_backend"] = "transformers"
        else:
            logger.warning(
                "No local Gemma4 configured (no llama_host / ollama_model / gemma_model_path)."
            )
            return None
        try:
            runtime = build_gemma_backend_from_config(extra)
            runtime.backend_name = f"local-{runtime.backend_name}"
            return runtime
        except Exception as exc:
            logger.warning("Failed to build local Gemma4 fallback: %s", exc)
            return None

    def _fallback_process_without_captions(
        self,
        paper_id: str,
        pages: list[Any],
        paper_metadata: PaperMetadata | None = None,
    ) -> list[dict[str, Any]]:
        """Visual-only fallback when GROBID/TEI captions are missing."""
        results: list[dict[str, Any]] = []
        # Build geology links from OCR text across all pages.
        all_ocr_text = " ".join(page.text or "" for page in pages)
        species_seed = sorted({t.text for t in self.taxon.predict(all_ocr_text) if t.text})
        section_links: dict[str, list[dict[str, Any]]] = {}
        if species_seed:
            section_links = link_species_to_geology(
                species_names=species_seed,
                sections=[
                    {
                        "section_id": "fallback",
                        "title": "Full text",
                        "section_type": "other",
                        "text": all_ocr_text,
                    }
                ],
                llm_runtime=(
                    self.gemma_runtime
                    if bool(self.config.extra.get("use_geology_llm", False))
                    else None
                ),
            )
        knowledge_graph = build_knowledge_graph(section_links) if section_links else None

        # Two-pass: enumerate total regions across all pages so the progress
        # callback can map (current, total) onto a smooth 30-90% band.
        all_regions: list[tuple[Any, Any, int]] = []  # (page, region, region_idx_on_page)
        yolo_path = self.config.yolo_model_path if self.config.use_yolo_figures else None
        for page in pages:
            for ridx, region in enumerate(
                detect_figure_regions(
                    page,
                    yolo_model_path=yolo_path,
                    yolo_conf=self.config.yolo_conf_threshold,
                    yolo_iou=self.config.yolo_iou_threshold,
                ),
                start=1,
            ):
                all_regions.append((page, region, ridx))
        n_total = max(1, len(all_regions))
        done = 0

        for page, region, ridx in all_regions:
            region_img = (
                cv2.imread(region.crop_path) if region.crop_path else cv2.imread(page.image_path)
            )
            if region_img is None:
                done += 1
                continue

            self._emit_progress(
                done + 1,
                n_total,
                f"[{done + 1}/{n_total}] p{page.page_index:02d} region {ridx}",
            )
            done += 1

            figure_id = f"auto_fig_p{page.page_index:03d}_r{ridx:02d}"
            caption = CaptionRecord(
                paper_id=paper_id,
                figure_id=figure_id,
                caption=f"Auto-generated figure for page {page.page_index}",
                entities=[],
                figure_number=str(page.page_index),
                page_index=page.page_index,
                panel_labels=[],
                source_xml=None,
            )

            figure_matches = self._process_region(
                paper_id=paper_id,
                figure_id=figure_id,
                caption=caption,
                region_img=region_img,
                region=region,
                figure_index=ridx,
                section_links=section_links,
                grobid_sections=[],
                knowledge_graph=knowledge_graph,
                best_page_index=page.page_index,
                paper_metadata=paper_metadata,
            )
            for m in figure_matches:
                meta = m.get("metadata", {})
                meta["fallback_mode"] = True
                meta["fallback_reason"] = "missing_tei_caption"
                m["metadata"] = meta
                results.append(m)

        return results

    def _cross_figure_reassign(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reassign panels from orphan figures to the nearest real plate figure.

        Thin instance wrapper around the module-level helper so the
        public method is preserved. The real implementation lives in
        ``rlpe.cross_figure._cross_figure_reassign_results`` so the
        eval harness and unit tests can call it without importing
        the full pipeline (and pulling in torch / gemma / paddleocr).
        """
        from .cross_figure import _cross_figure_reassign_results

        return _cross_figure_reassign_results(results)


# ---- module-level helpers -----------------------------------------------


def stage3_rescale_bbox(
    bbox: tuple[int, int, int, int] | list[int],
    *,
    source_dpi: int,
    crop_dpi: int,
) -> tuple[int, int, int, int]:
    """Phase 61 Plan 4 (Bug 4.5): rescale an M3 Stage 3 bbox from the
    extraction DPI to the visual-storage DPI used for the cropped image.

    M3 returns bboxes in pixels of the rendered plate (the DPI it saw
    when generating). The crop helper re-saves the panel at a possibly
    different ``crop_dpi``; if the consumer (a downstream LLM call, an
    annotation overlay, …) reads the bbox as-is it will land on the
    wrong pixels.

    The scale factor is ``crop_dpi / source_dpi``. Inputs of 0 / negative
    DPI are treated as "no rescaling" (defensive default to avoid
    divide-by-zero — the bbox is returned unchanged). Output is always
    a 4-tuple of ints, clamped to ``>= 0`` for safety.
    """
    if not bbox or len(bbox) != 4:
        return (0, 0, 0, 0)
    try:
        s = int(source_dpi)
        c = int(crop_dpi)
    except (TypeError, ValueError):
        return tuple(int(v) for v in bbox)  # type: ignore[return-value]
    if s <= 0 or c <= 0:
        return tuple(int(v) for v in bbox)  # type: ignore[return-value]
    if s == c:
        return tuple(int(v) for v in bbox)  # type: ignore[return-value]
    factor = c / float(s)
    out = []
    for v in bbox:
        try:
            out.append(max(0, int(round(float(v) * factor))))
        except (TypeError, ValueError):
            out.append(0)
    return (out[0], out[1], out[2], out[3])


def _page_from_filename(fname: str) -> int | None:
    """Best-effort page number extraction from an image filename.

    Used by the orphan-image-for-range-chart search (audit 2026-08-01 M10)
    as a fallback when the OD JSON's image-element page list is shorter
    than the directory listing. Returns the last integer found in the
    stem (so ``plate_3.png`` -> 3, ``imageFile12.png`` -> 12), or None
    if no integer is present.
    """
    import os as _os_for_page

    stem = _os_for_page.path.splitext(_os_for_page.path.basename(fname))[0]
    # Take the last numeric run — that matches OD's ``imageFile1.png``
    # naming where the trailing number is the index. Ignore the year-like
    # 4-digit prefix if there is one and a trailing shorter number exists.
    matches = re.findall(r"\d+", stem)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except (TypeError, ValueError):
        return None


def _resolve_m3_prompt_lang(value: Any) -> str | None:
    """Phase 27: translate the CLI's ``--m3-prompt-lang`` value to the
    argument expected by ``M3Engine.parse_caption(..., lang=)``.

    Rules:
    - ``None`` / ``"auto"`` / empty → return ``None`` so the engine
      auto-detects from the caption text (Hiragana/Katakana/CJK chars
      → JA; else ZH).
    - ``"ja"`` → ``"ja"`` (Japanese system prompt).
    - ``"zh"`` / ``"en"`` / anything else → return the value as-is,
      which causes the engine to fall through to the default ZH
      prompt (matches the legacy behaviour for non-JA captions).
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s == "auto":
        return None
    return s


def _merge_panel_hints(
    classical_panels: list[PanelCandidate],
    m3_panels: list[PanelBox],
    iou_match: float = 0.10,
) -> list[PanelCandidate]:
    """Attach M3 stage-3 hints (visible_label, morphology) to classical panels.

    Pairs are matched by IoU of the bboxes; the panel with the highest IoU to
    a classical panel is treated as its hint source. Classical bboxes are kept
    (more reliable for downstream crop). M3 adds two metadata fields that the
    stage-4 matcher consumes as priors.
    """
    if not classical_panels or not m3_panels:
        return classical_panels

    out: list[PanelCandidate] = []
    for cp in classical_panels:
        best: tuple[float, PanelBox] | None = None
        for mp in m3_panels:
            score = _iou(cp.bbox, mp.bbox)
            if score >= iou_match and (best is None or score > best[0]):
                best = (score, mp)
        if best is not None:
            mp = best[1]
            new_md = dict(cp.metadata or {})
            if mp.visible_label:
                new_md["m3_visible_label"] = mp.visible_label
                if not new_md.get("visible_label"):
                    new_md["visible_label"] = mp.visible_label
            if mp.morphology:
                new_md["m3_morphology"] = mp.morphology
            if mp.panel_id and (not cp.panel_id or cp.panel_id == "P?"):
                cp.panel_id = mp.panel_id
            new_md["m3_stage3_confidence"] = mp.confidence
            cp.metadata = new_md
        out.append(cp)
    return out


def _add_unmatched_m3_panels(
    classical_panels: list[PanelCandidate],
    m3_panels: list[PanelBox],
    iou_match: float = 0.10,
) -> int:
    """Append M3 panels that do not overlap any classical panel.

    Returns the number of M3 panels added. Useful when the classical CV
    (OpenCV / SAM2) under-segments — merges touching specimens into a
    single blob — and M3's vision model correctly identifies the missing
    ones. The added panels carry ``metadata.method == "m3_stage3_only"``
    so downstream code knows they came from the LLM vision hint, not the
    classical detector.
    """
    if not m3_panels:
        return 0

    matched_m3: set[int] = set()
    for cp in classical_panels:
        for idx, mp in enumerate(m3_panels):
            if _iou(cp.bbox, mp.bbox) >= iou_match:
                matched_m3.add(idx)
    added = 0
    for idx, mp in enumerate(m3_panels):
        if idx in matched_m3:
            continue
        md = {
            "method": "m3_stage3_only",
            "m3_morphology": mp.morphology,
            "m3_stage3_confidence": mp.confidence,
        }
        if mp.visible_label:
            md["m3_visible_label"] = mp.visible_label
            md["visible_label"] = mp.visible_label
        classical_panels.append(
            PanelCandidate(
                panel_id=mp.panel_id or f"M3_{idx:02d}",
                bbox=(
                    int(mp.bbox[0]),
                    int(mp.bbox[1]),
                    int(mp.bbox[2]),
                    int(mp.bbox[3]),
                ),
                score=mp.confidence,
                metadata=md,
            )
        )
        added += 1
    return added


def _extract_taxon_entities_from_text(text: str) -> list[CaptionEntity]:
    """Extract taxon-like strings from caption text for CaptionRecord.entities."""
    if not text:
        return []
    # audit 2026-07-31: ≥3-letter tokens on both sides so English
    # phrase fragments ("An attempt", "Explanation of") can't match;
    # the stopword filter mirrors association._TAXON_STOP_WORDS.
    pattern = re.compile(r"\b([A-Z][a-zA-Z-]{2,}\s+(?:sp\.|spp\.|cf\.|aff\.|[a-z][a-zA-Z-]{2,}))\b")
    out: list[CaptionEntity] = []
    seen: set[str] = set()
    for m in pattern.finditer(text):
        taxon = m.group(1).strip()
        if not taxon:
            continue
        words = taxon.split()
        if words[0].lower().rstrip(".,;:?!") in _TAXON_STOP_WORDS:
            continue
        if len(words) > 1 and words[1].lower().rstrip(".,;:?!") in _TAXON_STOP_WORDS:
            continue
        if taxon not in seen:
            seen.add(taxon)
            out.append(
                CaptionEntity(
                    text=taxon,
                    start=m.start(1),
                    end=m.end(1),
                    label="taxon",
                    score=0.65,
                )
            )
    return out
