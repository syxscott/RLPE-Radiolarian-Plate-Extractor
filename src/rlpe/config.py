from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# audit 2026-08-02 (Wave D): default to radiolarian-trained model
DEFAULT_YOLO_MODEL_PATH = "models/radiolarian_yolo_v1.pt"  # radiolarian-tuned

# Audit 2026-08-02 (Fix 1-B2): Ultralytics' stock COCO checkpoints. Picking
# one of these means the detector was never trained on radiolarian plates.
_COCO_YOLO_BASENAMES = {
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
}

# Audit 2026-08-02 (M5): valid ``yolo_device`` strings. Without this
# allow-list a typo (``yolo_device="garbage"``) silently sails through
# config parsing and only blows up inside ``YOLO(... device=...)``
# with a cryptic ultralytics AssertionError. Values per the design
# comment at line 182 + ultralytics conventions:
#   - "auto" / "" : let ultralytics auto-select (default)
#   - "cpu"       : force CPU
#   - "cuda"      : any CUDA-capable GPU
#   - "mps"       : Apple Silicon GPU
#   - "0".."7"    : specific GPU index
_VALID_YOLO_DEVICES = frozenset({"auto", "cpu", "cuda", "mps"} | {str(i) for i in range(8)})

# Recognised extra-config keys; any key outside this set triggers a warning.
_KNOWN_EXTRA_KEYS = {
    "sam2_checkpoint",
    "sam2_model_cfg",
    "sam2_grid_size",
    "sam2_max_point_prompts",
    "sam2_max_box_prompts",
    "use_neural_matcher",
    "matcher_checkpoint_path",
    "taxon_hf_model_path",
    "taxon_lexicon_path",
    "use_gemma4",
    "llm_backend",
    "gemma_model_path",
    "llama_model",
    "llama_host",
    "llama_timeout_sec",
    "ollama_model",
    "ollama_host",
    "gemma_timeout_sec",
    "gemma_conf_threshold",
    "gemma_prompt_lang",
    "gemma_use_4bit",
    "gemma_bfloat16",
    "gemma_device_map",
    "use_geology_llm",
    "gemma_init_error",
    # LLM cloud API (Anthropic-compatible, provider-agnostic)
    "llm_api_key",
    "llm_base_url",
    "llm_model",
    "llm_max_concurrent",
    "llm_timeout_sec",
    "llm_max_retries",
    "llm_enable_thinking",
    "llm_thinking_budget_tokens",
    "llm_max_output_tokens",
    "llm_fallback_default",
    "llm_interactive",
    "data_outbound_policy",
    "_llm_external_handler",  # injected by web/API layer
    # F19: batch subprocess isolation
    "batch_isolation",
    "batch_worker_timeout_sec",
    # Legacy vendor-branded aliases of the llm_* keys above (F17
    # rename). Kept listed so old configs load without typo warnings;
    # the backend builders read them as fallbacks.
    "MiniMax_api_key",
    "MiniMax_endpoint",
    "MiniMax_model",
    "MiniMax_max_concurrent",
    "MiniMax_timeout_sec",
    "MiniMax_max_retries",
    "MiniMax_enable_thinking",
    "MiniMax_thinking_budget_tokens",
    "MiniMax_max_output_tokens",
    "MiniMax_fallback_default",
    "MiniMax_interactive",
    "_MiniMax_external_handler",
    # OpenDataLoader integration
    "use_opendataloader",
    "od_use_ocr",
    "od_ocr_lang",
    "od_merge_gap_pt",
    # 2026-09-12: disable the orphan-page rescue's full-page OCR — the
    # native-crash hotspot in multi-DLL Windows processes (EasyOCR/torch
    # access violation). Default True preserves upstream behaviour.
    "od_orphan_rescue_ocr",
    # 2026-09-12: journal-style cross-page caption binding — bind body
    # paragraphs "Figs. X–Y. Species ..." (on the page after a full-bleed
    # plate) to that plate when ALL evidence gates hold (bare-Plate-ref
    # on the caption page, image page species-free, plate unclaimed).
    # Default True; multi-strategy caption resolution, strategy 4.
    "od_cross_page_captions",
    # 2026-09-12: Stage-2 verdict bridge — a figure the vision classifier
    # rejects as a plate but types as chart/diagram/table with a
    # species-bearing caption routes to the range-chart extractor
    # instead of being dropped. Default True. Read by pipeline.py.
    "stage2_range_chart_bridge",
    # 2026-09-12: collapse rows describing the same species with
    # identical geology data into one row — first panel image becomes
    # the table association, the rest move to
    # metadata.additional_panel_paths (files stay on disk). Default
    # True; opt-out for occurrence-level exports that need one row per
    # specimen panel.
    "dedup_species_panels",
    # 2026-09-12 (Arrow Lake): clamp torch intra-op threads to 1 at
    # pipeline init — the hybrid-core CPU segfaults in torch conv when
    # the default pool fans out (EasyOCR CRAFT victim). False restores
    # the unclamped default. Read by pipeline.py.
    "clamp_torch_threads",
    # 2026-09-12: fresh-worker retries after a NATIVE worker crash
    # (0xC0000005-class) — probabilistic on hybrid CPUs; each retry is
    # a new process/memory layout. Default 3; 0 disables.
    "batch_worker_crash_retries",
    # Phase 27: multilingual OCR + caption language selection
    "ocr_lang",
    "llm_prompt_lang",
    # 2026-09-12: oversized scans made EasyOCR attempt 1.3 GB conv
    # tensors (RuntimeError on torch 2.14, segfault on 2.8), killing
    # caption OCR for whole plates. Inputs with a side longer than this
    # are downscaled before inference (0 disables). Read by
    # pipeline.py at the OCRBackend construction site.
    "ocr_max_side_px",
    # Phase 29: GROBID retry + OD-fallback knobs. ``grobid_max_retries``
    # is the total HTTP attempts; ``grobid_timeout`` is the per-attempt
    # request timeout. ``disable_od_fallback`` is an escape hatch for
    # operators who want strict legacy behaviour (visual stub on
    # GROBID failure, no OD retry).
    "grobid_max_retries",
    "grobid_timeout",
    "max_regions_per_caption",
    "grobid_no_probe",  # Phase 43: skip is_available() probe
    "disable_od_fallback",
    # LLM 5-stage semantic engine
    "llm_enhanced_mode",
    "llm_stage_1",
    "llm_stage_2",
    "llm_stage_3",
    "llm_stage_4",
    "llm_stage_5",
    "llm_match_samples",
    "llm_diagnostic_dir",
    "llm_skip_match_on_empty_caption",
    "llm_retry_without_thinking",
    "llm_temperature",
    "llm_thinking_budget",
    # Round 6 + Round 7 multi-modal vision toggles
    "use_llm_stage3",
    "llm_multi_plate_enrich",
    # Sweep 6 (audit 2026-08-02 C4): ``llm_stage_6`` and the four
    # ``llm_per_panel_*`` knobs were listed here as legitimate extras,
    # but they're real ``PipelineConfig`` dataclass fields (see
    # ``llm_stage_6`` at line 211, ``llm_per_panel_*`` at lines
    # 218-220). Code that wrote them to ``cfg.extra`` was wrong
    # and would have tripped the unknown-keys warning. Removed.
    # Audit 2026-09-05 (tier3-D2): ``llm_stage_6`` is back — BOTH the
    # CLI and the Web extra builder write it into ``extra``, and
    # ``__post_init__`` now promotes it onto the typed field. Without
    # whitelisting it, every ``--llm-stage-6`` run tripped the
    # unknown-extra-key warning.
    "llm_stage_6",
    # LLM-first extraction (opt-in; default True when Gemma runtime is set)
    "use_llm_first",
    # Multi-modal geology vision (Commit 2 / Round 3)
    "use_geo_vision",
    "geo_vision_figure_types",
    # Paleobiology Database (opt-in)
    "use_paleodb",
    "paleodb_max_occurrences",
    "paleodb_endpoint",
    "paleodb_cache_dir",
    "paleodb_offline",
    # Audit 2026-09-07: read by pipeline.py:7277 but was missing from
    # this whitelist — JSON config writes triggered the unknown-key
    # warning (and hard failure under RLPE_STRICT_EXTRA=1).
    "paleodb_min_interval",
    # Audit 2026-09-07: read by llm_backends.py:2776/2778 (Gemma
    # temperature / top_p sampling params). Previously missing.
    "gemma_temperature",
    "gemma_top_p",
    # Phase 61 Plan 4 (Bug 4.3): reproducible-runs knob. When True, every
    # LLM backend sets temperature=0.0 / do_sample=False and seeds the
    # Python + numpy + torch RNGs so two consecutive runs on the same
    # paper produce identical species lists. See
    # ``llm_backends.resolve_deterministic_kwargs``.
    "deterministic",
    "deterministic_seed",
    # Phase 61 Plan 4 (Bug 4.10): optional name of a fallback LLM backend
    # for 4xx-then-retry.
    "fallback_llm_backend",
    # Audit 2026-09-05 (tier3-A6): both keys are read by pipeline.py
    # (``run`` resume check and the cross-figure-linker gate) but were
    # missing here, so setting them via a JSON config file tripped the
    # unknown-extra-key warning (and a hard failure under
    # RLPE_STRICT_EXTRA=1).
    # Audit 2026-09-07 (F7): YOLO panel detector for the OD path.
    "od_panel_detector",
    "resume",
    "cross_figure_linker_enabled",
    # Sweep 6 (audit 2026-08-02 C4): the five YOLO knobs (use_yolo_figures,
    # yolo_model_path, yolo_conf_threshold, yolo_iou_threshold,
    # yolo_device) used to be listed here as legitimate extras, but
    # they're real ``PipelineConfig`` dataclass fields (lines 178-185).
    # Code that wrote them to ``cfg.extra`` was wrong and would now trip
    # the unknown-keys warning. Removed so the validator catches any
    # straggler.
}


@dataclass(slots=True)
class PipelineConfig:
    """Pipeline configuration."""

    pdf_dir: Path
    work_dir: Path
    grobid_url: str = "http://localhost:8070"
    output_dir: Path | None = None
    use_gpu: bool = True
    ocr_backend: str = "paddleocr"
    taxon_model: str = "en_eco"
    min_panel_score: float = 0.80
    # GROBID path only: how far ahead/behind the page of a TEI caption
    # to look for a matching figure when figure numbers don't appear in
    # body text. Default 2 (legacy). Operators can widen via
    # ``--caption-window N``.
    caption_window: int = 2
    # Audit 2026-08-02 (Wave B cost control): cap per-caption regions
    # to prevent LLM cost explosion on dense papers.
    max_regions_per_caption: int = 3
    # Phase 28: OpenDataLoader path page-distance limit for caption↔image
    # pairing. Replaces four previously hard-coded limits in
    # ``opendataloader_extractor.py`` (the +2 plate forward window,
    # ±2 Fig. cross-page offsets, ±20 rescue hard cap, +3 body-ref
    # reconstruction window). Default 5 catches appendix-style layouts
    # (plates clustered at end of paper, caption on the previous/next
    # page from the figure) without enlarging enough to cause
    # cross-plate theft. Operators can widen via ``--od-caption-window N``.
    od_caption_window: int = 5
    # YOLO-based figure detection (alternative to OpenCV detect_figure_regions).
    # When ``use_yolo_figures=True``, ``yolo_model_path`` must point to a
    # trained YOLO ``.pt`` file. ``yolo_conf_threshold`` (default 0.25) discards
    # detections below this confidence; ``yolo_iou_threshold`` (default 0.45)
    # merges overlapping detections via Non-Maximum Suppression.
    use_yolo_figures: bool = False
    yolo_model_path: str = DEFAULT_YOLO_MODEL_PATH
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    # audit 2026-07-27 M-YO-1: YOLO device independent of ``use_gpu``.
    # "auto" = let ultralytics auto-select (cuda/cpu); "0" = GPU 0;
    # "" or "cpu" = force CPU.  Default "auto" preserves historical behaviour.
    yolo_device: str = "auto"
    num_workers: int = 4
    render_dpi: int = 200
    save_intermediate: bool = False  # Round 14 audit: default OFF — flipping this to True dumps per-region / per-page JSON to disk; on a 200-page PDF this produced ~117 GB of smoke-output during a Round 14 replay. Keep the default off; only set True for short debugging runs.
    # Audit 2026-08-02: LLM morphology extraction (Stage 6). Opt-in.
    # When True, the pipeline asks ``SemanticEngine.infer_morphology`` for
    # one MorphologyRecord per unique (paper, species) pair that has
    # an anchorable Description / Diagnosis section. Per-paper dedup
    # caps the API cost at
    # ``llm_morphology_max_species_per_paper`` species (default 100).
    # When False (default), no morphology records are produced and
    # no LLM call is made for morphology.
    llm_stage_6: bool = False
    llm_morphology_max_species_per_paper: int = 100
    llm_morphology_max_context_chars: int = 6000
    llm_morphology_min_caption_chars: int = 120
    # Phase 2026-08-17: Stage 4.5 per-panel LLM vision species ID.
    # Opt-in; when True, fans out one LLM vision call per Stage-3 panel
    # crop and overwrites the regex-matched species when LLM's confidence
    # meets the threshold. Pure additive — falls back to regex on any
    # backend error. See ``_apply_llm_per_panel_species_id``.
    llm_per_panel_enabled: bool = False
    llm_per_panel_min_conf: float = 0.55
    llm_per_panel_max_per_figure: int = 20
    llm_per_panel_max_per_paper: int = 200
    # Audit 2026-08-17: Stage 3 bbox + crop enrichment gate (was previously
    # read from ``config.extra["llm_stage3"]`` but the CLI set the key as
    # ``use_llm_stage3`` -- the gate never fired). Now a typed attribute
    # populated by the CLI as ``llm_stage3_enabled=args.use_llm_stage3``.
    # When True, ``_apply_stage3_bbox_crops`` crops each Stage 3 panel
    # bbox to disk and stamps ``panel_id_source="llm_vision"`` on the
    # matching result rows.
    llm_stage3_enabled: bool = False
    # Audit 2026-08-17: Round 7 multi-plate enrichment gate (was
    # previously read from ``config.extra["llm_multi_plate_enrich"]`` but
    # the CLI path never populated the key, so the second-pass was
    # silently disabled). Now a typed attribute populated by the CLI as
    # ``llm_multi_plate_enrich_enabled=args.llm_multi_plate_enrich``. When
    # True, ``_apply_multi_plate_enrichment`` fires a second-pass LLM
    # vision call on figures where the caption-image pairing missed a
    # plate (e.g. Bandini 2011 Plates 7-9).
    llm_multi_plate_enrich_enabled: bool = False
    # Round 14: default OFF. When True, _process_region dumps a
    # per-region ``auto_fig_pNNN_rNN.json`` to disk (34 MB each on
    # 200-page PDFs) and _process_one_pdf dumps the full
    # ``{slugify(figure_id)}.json``. A typical 5-paper OA smoke run
    # produces ~9000 such files = ~117 GB of intermediate state.
    # None of the eval scripts (eval_v19_normalized, simulate_v20_fix,
    # simulate_v21, evaluate_image_verified, eval_round6_gold) read
    # these files; the canonical output is the per-row
    # ``manifests/matches.jsonl`` + ``run_output.json``. Set
    # save_intermediate=True only when debugging the per-region
    # processing chain (see ``pipeline._process_region`` call sites
    # for the exact list of what gets written).
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Phase 38: type coercion for fields that come in as strings
        # from YAML / JSON configs. Without this, ``PipelineConfig(
        # num_workers="4")`` (string from yaml.safe_load) bypasses the
        # type hint and crashes ``ThreadPoolExecutor(max_workers="4")``
        # later with a confusing TypeError.
        try:
            self.num_workers = int(self.num_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"num_workers must be an integer, got {self.num_workers!r} ({exc})"
            ) from exc
        try:
            self.render_dpi = int(self.render_dpi)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"render_dpi must be an integer, got {self.render_dpi!r} ({exc})"
            ) from exc
        try:
            self.min_panel_score = float(self.min_panel_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"min_panel_score must be a float, got {self.min_panel_score!r} ({exc})"
            ) from exc
        try:
            self.caption_window = int(self.caption_window)
            self.od_caption_window = int(self.od_caption_window)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"caption_window / od_caption_window must be integers ({exc})"
            ) from exc
        try:
            self.yolo_conf_threshold = float(self.yolo_conf_threshold)
            self.yolo_iou_threshold = float(self.yolo_iou_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"yolo_conf_threshold / yolo_iou_threshold must be floats ({exc})"
            ) from exc

        # Phase 38: validation. Bad config values used to silently
        # produce zero-panel runs (min_panel_score out of range) or
        # cryptic PyMuPDF crashes (render_dpi ≤ 0).
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        if not (50 <= self.render_dpi <= 600):
            raise ValueError(f"render_dpi must be in [50, 600], got {self.render_dpi}")
        if not (0.0 <= self.min_panel_score <= 1.0):
            raise ValueError(f"min_panel_score must be in [0.0, 1.0], got {self.min_panel_score}")
        if self.caption_window < 1 or self.caption_window > 50:
            raise ValueError(f"caption_window must be in [1, 50], got {self.caption_window}")
        if self.max_regions_per_caption < 1 or self.max_regions_per_caption > 50:
            raise ValueError(
                f"max_regions_per_caption must be in [1, 50], got {self.max_regions_per_caption}"
            )
        # Align with gui/constants.RANGE_OD_CAPTION_WINDOW=(1,200), the
        # web JobOptions validator, and the CLI help text. Phase 28's
        # rescue ×4 = 20 default already implies 200 is the design
        # intent for appendix-style plate layouts; the previous cap of
        # 50 crashed the pipeline for any GUI/web-submitted value in
        # 51..200 (B1, audit 2026-07-26).
        if self.od_caption_window < 1 or self.od_caption_window > 200:
            raise ValueError(f"od_caption_window must be in [1, 200], got {self.od_caption_window}")
        if self.use_yolo_figures:
            if not self.yolo_model_path:
                raise ValueError("use_yolo_figures=True requires yolo_model_path to be set")
            model_path = Path(self.yolo_model_path)
            if not model_path.is_file():
                raise ValueError(f"yolo_model_path={model_path!r} does not exist or is not a file")
            # Audit 2026-08-02 (Fix 1-B2): warn when the user picks a known
            # COCO placeholder. Filename "yolo11*.pt" is the Ultralytics
            # generic COCO checkpoint — detecting radiolarian plates with it
            # produces random COCO-class bboxes (person, bicycle, etc.) that
            # downstream caption routing will ingest as if correct.
            if model_path.name in _COCO_YOLO_BASENAMES:
                logger.warning(
                    "yolo_model_path=%r looks like a generic COCO-pretrained "
                    "model. Detections will be unreliable on radiolarian "
                    "pages — train a domain-specific .pt or set "
                    "--use-yolo-figures=False. See docs/ for the training "
                    "recipe.",
                    model_path.name,
                )
        # audit 2026-07-26: align with gui.constants.RANGE_YOLO_CONF/IOU
        # (0.01 lower bound) and the settings_tab spinbox minimum 0.01.
        if not (0.01 <= self.yolo_conf_threshold <= 1.0):
            raise ValueError(
                f"yolo_conf_threshold must be in [0.01, 1.0], got {self.yolo_conf_threshold}"
            )
        if not (0.01 <= self.yolo_iou_threshold <= 1.0):
            raise ValueError(
                f"yolo_iou_threshold must be in [0.01, 1.0], got {self.yolo_iou_threshold}"
            )

        # Audit 2026-08-02 (M5): validate ``yolo_device`` early. Without
        # this, a typo (e.g. ``yolo_device="garbage"``) silently sails
        # through config parsing and only blows up inside the first
        # ``YOLO(... device=...)`` call with a cryptic
        # ``AssertionError`` from ultralytics. Valid values per the
        # design comment at line 182 + ultralytics conventions:
        #   - "auto" / "" : let ultralytics auto-select (default)
        #   - "cpu"       : force CPU
        #   - "cuda"      : any CUDA-capable GPU
        #   - "mps"       : Apple Silicon GPU
        #   - "0".."7"    : specific GPU index
        if self.yolo_device not in _VALID_YOLO_DEVICES:
            raise ValueError(
                f"yolo_device must be one of {sorted(_VALID_YOLO_DEVICES)!r} "
                f"(or empty string ''), got {self.yolo_device!r}"
            )

        # Audit 2026-08-02: LLM morphology Stage-6 validation. Coerce
        # incoming types (string from YAML/JSON) and clamp to safe
        # ranges so a bad operator value doesn't crash the run.
        # Audit 2026-09-05 (tier3-D2 follow-up): promote
        # ``extra["llm_stage_6"]`` onto the typed field. Both the CLI and
        # the Web extra builder write the key into ``extra``, but only
        # the CLI also set the typed attribute — a web job with
        # ``llm_stage_6: true`` never reached
        # ``_apply_morphology_enrichment`` (which reads the typed
        # field) and silently produced nothing. ``None`` (the CLI's
        # BooleanOptionalAction default) is skipped; an explicit CLI
        # flag still wins because it assigns the typed field after
        # construction.
        if self.extra.get("llm_stage_6") is not None:
            self.llm_stage_6 = bool(self.extra["llm_stage_6"])
        try:
            self.llm_stage_6 = bool(self.llm_stage_6)
            self.llm_morphology_max_species_per_paper = int(
                self.llm_morphology_max_species_per_paper
            )
            self.llm_morphology_max_context_chars = int(self.llm_morphology_max_context_chars)
            self.llm_morphology_min_caption_chars = int(self.llm_morphology_min_caption_chars)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "llm_morphology_* fields must be ints/bools, got "
                f"max={self.llm_morphology_max_species_per_paper!r}, "
                f"context={self.llm_morphology_max_context_chars!r}, "
                f"min_caption={self.llm_morphology_min_caption_chars!r} ({exc})"
            ) from exc
        if self.llm_morphology_max_species_per_paper < 1:
            raise ValueError(
                "llm_morphology_max_species_per_paper must be >= 1, "
                f"got {self.llm_morphology_max_species_per_paper}"
            )
        if self.llm_morphology_max_context_chars < 200:
            raise ValueError(
                "llm_morphology_max_context_chars must be >= 200, "
                f"got {self.llm_morphology_max_context_chars}"
            )
        if self.llm_morphology_min_caption_chars < 0:
            raise ValueError(
                "llm_morphology_min_caption_chars must be >= 0, "
                f"got {self.llm_morphology_min_caption_chars}"
            )

        # Phase 2026-08-17 Stage 4.5: coerce the 4 new fields from YAML
        # string inputs and validate their ranges. Matches the
        # ``llm_morphology_*`` precedent right above.
        self.llm_per_panel_enabled = bool(self.llm_per_panel_enabled)
        self.llm_per_panel_min_conf = float(self.llm_per_panel_min_conf)
        if not (0.0 <= self.llm_per_panel_min_conf <= 1.0):
            raise ValueError(
                f"llm_per_panel_min_conf must be in [0, 1], got {self.llm_per_panel_min_conf}"
            )
        self.llm_per_panel_max_per_figure = int(self.llm_per_panel_max_per_figure)
        self.llm_per_panel_max_per_paper = int(self.llm_per_panel_max_per_paper)
        if self.llm_per_panel_max_per_figure < 1 or self.llm_per_panel_max_per_paper < 1:
            raise ValueError(
                "llm_per_panel_max_per_figure / llm_per_panel_max_per_paper "
                f"must be >= 1 (got {self.llm_per_panel_max_per_figure}, "
                f"{self.llm_per_panel_max_per_paper})"
            )

        # Audit 2026-08-17: coerce the two Stage 3 / multi-plate enrichment
        # gate flags that were promoted from ``config.extra`` to typed
        # attributes. Default False for backward compat; CLI sets them to
        # ``args.use_llm_stage3`` / ``args.llm_multi_plate_enrich``.
        self.llm_stage3_enabled = bool(self.llm_stage3_enabled)
        self.llm_multi_plate_enrich_enabled = bool(self.llm_multi_plate_enrich_enabled)

        # Phase 38: warn (don't raise) for unknown extra-config keys.
        # A typo like ``minimax_api_key`` (lowercase) silently produces
        # a config that ignores the value.
        # Audit 2026-09-01 (Step 2): when ``RLPE_STRICT_EXTRA=1`` (or the
        # ``strict_extra`` env override is set on the JobOptions
        # wrapper), HARD-FAIL on unknown keys. The default behaviour
        # remains a warning to preserve backwards compatibility with
        # operator configs that pre-date some of the typed fields.
        unknown = set(self.extra.keys()) - _KNOWN_EXTRA_KEYS
        if unknown:
            # Phase 38: offer Levenshtein-style suggestions so users
            # can spot typos.
            from difflib import get_close_matches

            suggestions = []
            for u in sorted(unknown):
                matches = get_close_matches(u, _KNOWN_EXTRA_KEYS, n=1, cutoff=0.6)
                if matches:
                    suggestions.append(f"{u!r} → did you mean {matches[0]!r}?")
                else:
                    suggestions.append(repr(u))
            strict = __import__("os").environ.get("RLPE_STRICT_EXTRA", "0") == "1"
            if strict:
                raise ValueError(
                    f"Unknown extra config keys (strict mode): {sorted(unknown)}. "
                    f"Hints: {'; '.join(suggestions)}. "
                    f"Set RLPE_STRICT_EXTRA=0 to fall back to warning-only."
                )
            logger.warning(
                "Unknown extra config keys (typo?): %s%s",
                sorted(unknown),
                f"  hints: {'; '.join(suggestions)}" if suggestions else "",
            )

    def resolved_output_dir(self) -> Path:
        return self.output_dir or (self.work_dir / "output")

    def figures_dir(self) -> Path:
        return self.resolved_output_dir() / "figures"

    def tei_dir(self) -> Path:
        return self.resolved_output_dir() / "tei"

    def panels_dir(self) -> Path:
        return self.resolved_output_dir() / "panels"

    def manifests_dir(self) -> Path:
        return self.resolved_output_dir() / "manifests"
