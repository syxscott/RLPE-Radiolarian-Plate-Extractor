from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)


# --- Pre-filters for non-specimen content ----------------------------------

_PLACEHOLDER_CAPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"page\s+\d+\s*(auto[- ]?generated|placeholder|header|footer)", re.IGNORECASE),
    re.compile(r"(auto[- ]?generated|placeholder)\s+(image|figure|page)", re.IGNORECASE),
    # Chinese: 自动生成 (auto-generated), 占位 (placeholder), 页眉/页脚 (header/footer)
    re.compile(r"(自动生成|占位图|占位|页眉|页脚)"),
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*(running\s+head|header|footer)\s*$", re.IGNORECASE),
    # Copyright/license lines (allow attribution continuation)
    re.compile(r"^\s*(©|copyright|licen[sc]e|creative\s+commons)[\s.©]", re.IGNORECASE),
    re.compile(r"\b(scientific|elsevier|springer|wiley|tandfonline)\s*$", re.IGNORECASE),
)


def _looks_like_placeholder_caption(caption_text: str) -> bool:
    """Heuristic: return True when the caption itself signals non-specimen content.

    The OpenDataLoader extractor sometimes picks up page headers, running
    titles, or auto-generated watermarks as a "figure" with a short caption.
    Sending those to M3 wastes API calls and produces confusing "not a
    specimen" responses that get surfaced as fallback errors.
    """
    if not caption_text:
        return False
    text = caption_text.strip()
    if len(text) <= 3:
        return True
    return any(p.search(text) for p in _PLACEHOLDER_CAPTION_PATTERNS)

from .geology_extraction import build_knowledge_graph, link_species_to_geology
from .config import PipelineConfig
from .grobid import GrobidClient, parse_paper_metadata_from_tei
from .association import match_panels
from .gemma_postprocess import apply_gemma_to_matches, build_gemma_backend_from_config
from .layout import choose_best_page, detect_figure_regions, extract_figure_number, render_pdf_pages
from .m3_engine import CaptionPair, M3Engine, PanelBox, PanelMatch
from .ocr import OCRBackend, normalize_ocr_tokens
from .scale_bar import detect_scale_bar_length_px, extract_scale_from_caption, extract_scale_from_ocr_text, merge_scale_info
from .segmentation import PanelSegmenter, SegmentationConfig
from .taxon import TaxonRecognizer
from .types import CaptionEntity, CaptionRecord, FigureRegion, PanelCandidate, PaperMetadata
from .utils import ensure_dir, slugify, stable_id, write_json, write_jsonl


class RadiolarianPipeline:
    def __init__(self, config: PipelineConfig, progress_callback=None) -> None:
        self.config = config
        # Optional progress callback: ``cb(current, total, message)``.
        # ``current`` and ``total`` are 0-indexed ints; the API uses
        # ``current/total`` to map a real pipeline position onto the 30-90%
        # band of the job progress.
        self._progress_cb = progress_callback
        self.grobid = GrobidClient(server_url=config.grobid_url)
        self.ocr = OCRBackend(backend=config.ocr_backend, use_gpu=config.use_gpu)
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
        # is available AND the user opts in via ``m3_enhanced_mode`` (default ON
        # when M3 backend is selected).
        self.m3_engine: M3Engine | None = None
        self._gemma_lock = threading.Lock()
        self._ocr_lock = threading.Lock()
        self._seg_lock = threading.Lock()
        # Fallback handler for MiniMax API errors (None when not using MiniMax)
        self.gemma_fallback_handler = None
        # Secondary Gemma runtime used as fallback target (lazy-init on first error)
        self._fallback_gemma_runtime = None
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
                    )
        return self._od_extractor

    def _try_init_gemma(self) -> None:
        if not self.config.extra.get("use_gemma4", False):
            return
        model_path = self.config.extra.get("gemma_model_path") or self.config.extra.get("ollama_model")
        backend_name = str(self.config.extra.get("llm_backend", "transformers")).lower()
        if not model_path and backend_name not in {"ollama", "MiniMax", "minimax", "MiniMax-m3"}:
            return
        try:
            self.gemma_runtime = build_gemma_backend_from_config(self.config.extra)
            # If MiniMax backend, attach a FallbackHandler. The handler is
            # invoked ONLY from ``_apply_gemma_with_fallback``; we intentionally
            # do NOT also wire it into ``backend.on_error`` to avoid the
            # handler being called twice for the same error.
            if backend_name in {"MiniMax", "minimax", "MiniMax-m3"}:
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

        # Build the M3 semantic engine (5-stage). ON by default when M3 is the
        # active backend and ``m3_enhanced_mode`` is not explicitly disabled.
        # For other backends, opt-in via ``m3_enhanced_mode = True``.
        if self.gemma_runtime is not None:
            want_m3 = self.config.extra.get("m3_enhanced_mode")
            if want_m3 is None:
                want_m3 = backend_name in {"MiniMax", "minimax", "MiniMax-m3"}
            if want_m3:
                m3_cfg = {
                    k: v for k, v in self.config.extra.items()
                    if k.startswith("m3_")
                }
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
                )
                logger.info(
                    "M3Engine initialized (stages 1-5: %s/%s/%s/%s/%s, diag=%s)",
                    m3_cfg.get("m3_stage_1"), m3_cfg.get("m3_stage_2"),
                    m3_cfg.get("m3_stage_3"), m3_cfg.get("m3_stage_4"),
                    m3_cfg.get("m3_stage_5"), m3_cfg.get("m3_diagnostic_dir"),
                )

    def prepare_dirs(self) -> None:
        ensure_dir(self.config.resolved_output_dir())
        ensure_dir(self.config.tei_dir())
        ensure_dir(self.config.figures_dir())
        ensure_dir(self.config.panels_dir())
        ensure_dir(self.config.manifests_dir())

    def run(self) -> list[dict[str, Any]]:
        self.prepare_dirs()
        pdf_files = sorted(self.config.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            return []

        rows: list[dict[str, Any]] = []
        total = len(pdf_files)
        completed = 0
        # Fire one initial tick so the UI can show "started" before the first
        # PDF actually finishes.
        if self._progress_cb is not None:
            try:
                self._progress_cb(0, total, f"Starting pipeline ({total} PDF(s))")
            except Exception:
                logger.debug("progress_cb(0) failed", exc_info=True)
        with ThreadPoolExecutor(max_workers=max(1, self.config.num_workers)) as pool:
            futures = {pool.submit(self._process_one_pdf, p): p for p in pdf_files}
            for fut in as_completed(futures):
                pdf = futures[fut]
                try:
                    result_rows = fut.result()
                    rows.extend(result_rows)
                except Exception:
                    logger.exception("PDF processing failed; continuing with remaining PDFs")
                completed += 1
                if self._progress_cb is not None:
                    try:
                        self._progress_cb(
                            completed, total,
                            f"Processed {pdf.name} ({len(rows)} matches so far)",
                        )
                    except Exception:
                        logger.debug("progress_cb tick failed", exc_info=True)

        manifest_path = self.config.manifests_dir() / "matches.jsonl"
        write_jsonl(manifest_path, rows)
        if self._progress_cb is not None:
            try:
                self._progress_cb(total, total, f"Done — {len(rows)} matches")
            except Exception:
                logger.debug("progress_cb(done) failed", exc_info=True)
        return rows

    def _process_one_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        paper_id = stable_id(pdf_path)
        if self._progress_cb is not None:
            try:
                self._progress_cb(0, 1, f"Loading {pdf_path.name}…")
            except Exception:
                logger.debug("progress_cb(load) failed", exc_info=True)

        # ------ OpenDataLoader path (opt-in) -----------------------------------
        if self.config.extra.get("use_opendataloader", False):
            rows = self._process_one_pdf_od(paper_id, pdf_path)
        else:
            # ------ GROBID + layout path (default) -----------------------------
            rows = self._process_one_pdf_grobid(paper_id, pdf_path)

        if self._progress_cb is not None:
            try:
                self._progress_cb(1, 1, f"Finished {pdf_path.name} ({len(rows)} matches)")
            except Exception:
                logger.debug("progress_cb(finish) failed", exc_info=True)
        return rows

    # -----------------------------------------------------------------------
    # OpenDataLoader-based processing
    # -----------------------------------------------------------------------

    def _process_one_pdf_od(self, paper_id: str, pdf_path: Path) -> list[dict[str, Any]]:
        od_result = self.od_extractor.extract(pdf_path, self.config.resolved_output_dir())

        if not od_result.success:
            logger.warning(
                "OpenDataLoader failed (%s); falling back to GROBID+layout",
                od_result.error or "unknown error",
            )
            return self._process_one_pdf_grobid(paper_id, pdf_path)

        figures = od_result.figures
        if not figures:
            logger.info("No figures found by OpenDataLoader for %s; falling back.", paper_id)
            return self._process_one_pdf_grobid(paper_id, pdf_path)

        # Geology / fulltext — collect taxon entities from all captions.
        all_taxon_names: list[str] = []
        for pair in figures:
            if pair.caption_text:
                for ent in _extract_taxon_entities_from_text(pair.caption_text):
                    if ent.text:
                        all_taxon_names.append(ent.text)
        species_seed = sorted(set(all_taxon_names))
        use_geology_llm = bool(self.config.extra.get("use_geology_llm", False)) and self.gemma_runtime is not None
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
            if not pair.image_paths:
                continue
            if self._progress_cb is not None:
                try:
                    self._progress_cb(
                        fig_idx - 1, n_figs,
                        f"[{fig_idx}/{n_figs}] {pair.caption_text[:40] if pair.caption_text else pair.figure_id}",
                    )
                except Exception:
                    logger.debug("progress_cb(figure) failed", exc_info=True)

            # Load the first image as the region image.
            region_img = cv2.imread(pair.image_paths[0])
            if region_img is None:
                continue

            h_img, w_img = region_img.shape[:2]
            region = FigureRegion(
                page_index=pair.page_number,
                bbox=(0, 0, int(w_img), int(h_img)),
                crop_path=pair.image_paths[0],
                score=0.85,
                region_id=f"od_{paper_id}_p{pair.page_number:03d}_{fig_idx:02d}",
                kind="figure",
                metadata={"source": "opendataloader"},
            )

            # Build caption record from OpenDataLoader output.
            caption_text = pair.caption_text or ""
            caption_entities = _extract_taxon_entities_from_text(caption_text)
            # Extract the actual figure number from the caption text (e.g. "Fig. 3")
            # instead of falling back to the PDF page number — that was a copy-paste
            # bug that made downstream code think every page-N figure was "figure N".
            figure_number = extract_figure_number(caption_text) or pair.figure_id or str(pair.page_number)
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
                m["metadata"] = meta
                results.append(m)

        # Fallback: if OD returned no results even with figures, try GROBID.
        if not results:
            logger.info("OpenDataLoader produced no matches; falling back to GROBID+layout.")
            return self._process_one_pdf_grobid(paper_id, pdf_path)
        return results

    # -----------------------------------------------------------------------
    # Original GROBID + layout path
    # -----------------------------------------------------------------------

    def _process_one_pdf_grobid(self, paper_id: str, pdf_path: Path) -> list[dict[str, Any]]:
        grobid_result = self.grobid.process_pdf(pdf_path, self.config.resolved_output_dir())

        tei_captions = grobid_result.captions if grobid_result.success else []
        # Extract paper-level metadata (DOI, abstract, authors, journal, year, ...)
        # from the GROBID TEI. Falls back to an empty record on failure.
        try:
            paper_meta = parse_paper_metadata_from_tei(grobid_result.tei_xml or "")
        except Exception:
            paper_meta = PaperMetadata(source="none")

        pages = render_pdf_pages(pdf_path, self.config.figures_dir() / paper_id, dpi=self.config.render_dpi)
        results: list[dict[str, Any]] = []

        # 全文地质信息抽取与物种关系链接（可选使用LLM增强）
        section_links: dict[str, list[dict[str, Any]]] = {}
        knowledge_graph: dict[str, Any] | None = None
        use_geology_llm = bool(self.config.extra.get("use_geology_llm", False)) and self.gemma_runtime is not None
        species_seed = sorted({ent.text for cap in tei_captions for ent in cap.entities if ent.text})
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
        if not tei_captions:
            return self._fallback_process_without_captions(paper_id, pages)

        for idx, caption in enumerate(tei_captions, start=1):
            if self._progress_cb is not None:
                try:
                    self._progress_cb(
                        idx - 1, max(1, len(tei_captions)),
                        f"[{idx}/{len(tei_captions)}] {caption.figure_id}",
                    )
                except Exception:
                    logger.debug("progress_cb(grobid-caption) failed", exc_info=True)

            best_page = choose_best_page(pages, caption.figure_number, caption.caption, window=self.config.caption_window)
            if best_page is None:
                continue
            caption.page_index = best_page.page_index

            candidate_pages = [best_page]
            if best_page.page_index > 1:
                prev_page = pages[best_page.page_index - 2]
                candidate_pages.insert(0, prev_page)
            if best_page.page_index < len(pages):
                next_page = pages[best_page.page_index]
                candidate_pages.append(next_page)

            chosen_regions = []
            for page in candidate_pages:
                regions = detect_figure_regions(page)
                if regions:
                    chosen_regions.extend(regions)
            if not chosen_regions:
                continue

            chosen_regions.sort(key=lambda r: (-r.score, r.page_index, r.bbox[1], r.bbox[0]))
            region = chosen_regions[0]
            region_img = cv2.imread(region.crop_path) if region.crop_path else cv2.imread(best_page.image_path)
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
            results.extend(figure_matches)
        return results

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

        Used by both the normal caption-driven path and the visual-only fallback.

        When M3Engine is available, the M3 5-stage pipeline runs alongside the
        classical CV + rule-based path:
          - Stage 1 parses caption into structured (label->species) pairs.
          - Stage 2 filters non-radiolarian plates; on rejection, returns early.
          - Stage 3 augments SAM2 panels with M3-suggested bboxes / visible labels.
          - Stage 4 replaces the per-panel M3 call with a richer context-aware match.
          - Stage 5 critiques all matches and may override low-confidence ones.
        """
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
                    plate_pil = _PILImage.open(str(region_img)).convert("RGB")
                # Stage 1: caption parser
                if self.m3_engine._stage_enabled(1):
                    m3_caption_pairs = self.m3_engine.parse_caption(caption.caption or "")
                    m3_diag["stage1_pairs"] = len(m3_caption_pairs)
                # Stage 2: plate classifier — early exit on non-radiolarian
                if self.m3_engine._stage_enabled(2):
                    m3_plate_cls = self.m3_engine.classify_plate(plate_pil)
                    m3_diag["stage2_class"] = m3_plate_cls.to_dict()
                    if not m3_plate_cls.is_radiolarian_plate:
                        logger.info(
                            "M3 Stage 2: %s/%s rejected (not a radiolarian plate): %s",
                            paper_id, figure_id, m3_plate_cls.reasoning[:120],
                        )
                        # Annotate each potential panel as "rejected by classifier"
                        # and return an empty match list with the diagnostic saved.
                        if self.config.save_intermediate:
                            write_json(self.config.manifests_dir() / paper_id / f"{slugify(figure_id)}.json", {
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
                            })
                        return []
                # Stage 3: panel segmentation hint
                if self.m3_engine._stage_enabled(3):
                    hint = (m3_plate_cls.panel_count_estimate
                            if m3_plate_cls and m3_plate_cls.panel_count_estimate else None)
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
        # If M3 found panels and the classical path found none, use M3 boxes
        if (not panels or len(panels) == 0) and m3_panels:
            h_img, w_img = region_img.shape[:2]
            panels = [
                PanelCandidate(
                    panel_id=mp.panel_id,
                    bbox=(int(mp.bbox[0]), int(mp.bbox[1]), int(mp.bbox[2]), int(mp.bbox[3])),
                    score=mp.confidence,
                    metadata={"method": "m3_stage3", "morphology": mp.morphology, "visible_label": mp.visible_label},
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

        with self._ocr_lock:
            ocr_tokens = normalize_ocr_tokens(self.ocr.recognize(region_img))
        taxon_entities = self.taxon.predict(caption.caption or "")

        # Scale bar: caption + OCR + visual line detection
        caption_scale = extract_scale_from_caption(caption.caption or "")
        ocr_text_block = " ".join(tok.text for tok in ocr_tokens)
        ocr_scale = extract_scale_from_ocr_text(ocr_text_block)
        px_len = detect_scale_bar_length_px(region_img)
        merged_scale = merge_scale_info(caption_scale, ocr_scale, pixel_length=px_len)

        for panel_index, panel in enumerate(panels, start=1):
            x, y, w, h = panel.bbox
            crop = region_img[y : y + h, x : x + w]
            panel_dir = ensure_dir(self.config.panels_dir() / paper_id / (figure_id or f"fig_{figure_index}"))
            panel_path = panel_dir / f"panel_{panel_index:02d}.png"
            cv2.imwrite(str(panel_path), crop)
            panel.image_path = str(panel_path)
            panel.region_id = region.region_id
            panel.source_page = region.page_index
            panel.panel_index = panel_index
            # Per-panel OCR: re-OCR the tight crop and attach to the panel.
            # Falls back silently to the region-level OCR tokens if the
            # backend fails to initialise.
            try:
                with self._ocr_lock:
                    panel_tokens = self.ocr.recognize_panel(region_img, (x, y, w, h))
                if panel_tokens:
                    panel.metadata = panel.metadata or {}
                    panel.metadata["panel_ocr_text"] = " ".join(t.text for t in panel_tokens)
                    panel.metadata["panel_ocr_token_count"] = len(panel_tokens)
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
                    paper_id, figure_id, skip_reason,
                )
                m3_diag["stage4_skipped"] = skip_reason
            if not skip_stage4:
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
        if (self.m3_engine is not None and self.m3_engine._stage_enabled(5)
                and matches):
            matches = self._apply_m3_stage5(
                matches=matches,
                caption_pairs=m3_caption_pairs,
                caption_text=caption.caption or "",
                region_img=region_img,
            )

        # Attach geology links and scale bar info
        for m in matches:
            geo_list = section_links.get(m.species or "", [])
            if not geo_list and section_links:
                first_key = next(iter(section_links.keys())) if section_links else None
                geo_list = section_links.get(first_key, []) if first_key else []
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
            write_json(self.config.manifests_dir() / paper_id / f"{slugify(figure_id)}.json", {
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
            })
        return results

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
        new_matches = []
        for m in matches:
            try:
                if not m.panel_path or not Path(m.panel_path).exists():
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
                    m.confidence = max(m.confidence, m3_conf)
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
                        md["gemma_reasoning"] = panel_match.reasoning or "M3: not a radiolarian specimen"
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
                plate_pil = _PILImage.open(str(region_img)).convert("RGB")
            panel_matches: list[PanelMatch] = []
            for idx, m in enumerate(matches, start=1):
                pid = str(m.panel_id or f"P{idx}")
                panel_matches.append(PanelMatch(
                    panel_id=pid,
                    label=m.label_text,
                    species=m.species,
                    confidence=float(m.confidence or 0.0),
                    reasoning=(m.metadata or {}).get("gemma_reasoning", ""),
                ))
            critiques = self.m3_engine.critique_matches(
                plate_image=plate_pil,
                matches=panel_matches,
                caption_text=caption_text,
                caption_pairs=caption_pairs,
            )
            M3Engine.apply_critiques(panel_matches, critiques)
            # Back-apply to original matches
            by_id = {pm.panel_id: pm for pm in panel_matches}
            for idx, m in enumerate(matches, start=1):
                pid = str(m.panel_id or f"P{idx}")
                pm = by_id.get(pid)
                if not pm:
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
                    m.confidence = min(m.confidence, max(0.3, float(pm.confidence or 0.0)))
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
        action = self.gemma_fallback_handler(error_info)

        if action == "stop":
            raise RuntimeError(
                f"[MiniMax] user stopped pipeline at paper={paper_id} figure={figure_id}: "
                f"{error_info.get('error', '?')}"
            )

        if action == "rules":
            logger.warning("[MiniMax] API error, falling back to rule pipeline for %s/%s",
                           paper_id, figure_id)
            for m in result:
                m.metadata["MiniMax_fallback_action"] = "rules"
            return result

        if action == "gemma4":
            logger.warning("[MiniMax] API error, switching to local Gemma4 for %s/%s",
                           paper_id, figure_id)
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
        return any(
            # gemma_error = real API/runtime error
            # gemma_fallback = M3 returned a low-confidence verdict
            # We ignore m3_rejected_non_radiolarian — that's a normal
            # "this isn't a specimen" answer, not an error.
            (m.metadata.get("gemma_error") or m.metadata.get("gemma_fallback"))
            and not m.metadata.get("m3_rejected_non_radiolarian")
            for m in matches
        )

    def _attach_paleodb_metadata(self, matches: list) -> None:
        """Look up each unique species in PBDB and attach taxonomy + occurrences.

        On any failure (network, missing name, PBDB rate-limit) the match is
        left with an empty ``paleodb`` dict and the pipeline continues. Results
        are cached on disk so a second run is instant.
        """
        from .paleodb import PaleoDB

        max_occ = int(self.config.extra.get("paleodb_max_occurrences", 25) or 25)
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
                occs = client.lookup_occurrences(name, max_n=max_occ) if tax else []
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
    def _collect_fallback_error_info(matches: list, paper_id: str, figure_id: str) -> dict[str, Any]:
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
            (m.metadata.get("gemma_fallback", False) for m in matches if m.metadata.get("gemma_fallback")),
            False,
        )
        return {
            "error": first_err or "MiniMax returned fallback_used=True (see stderr for traceback)",
            "error_type": first_type or ("MiniMaxAPIError" if first_fb else "Unknown"),
            "context": f"paper={paper_id} figure={figure_id} affected_panels="
                       f"{sum(1 for m in matches if m.metadata.get('gemma_error') or m.metadata.get('gemma_fallback'))}",
        }

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
            logger.warning("No local Gemma4 configured (no llama_host / ollama_model / gemma_model_path).")
            return None
        try:
            runtime = build_gemma_backend_from_config(extra)
            runtime.backend_name = f"local-{runtime.backend_name}"
            return runtime
        except Exception as exc:
            logger.warning("Failed to build local Gemma4 fallback: %s", exc)
            return None

    def _fallback_process_without_captions(self, paper_id: str, pages: list[Any]) -> list[dict[str, Any]]:
        """Visual-only fallback when GROBID/TEI captions are missing."""
        results: list[dict[str, Any]] = []
        # Build geology links from OCR text across all pages.
        all_ocr_text = " ".join(page.text or "" for page in pages)
        species_seed = sorted({t.text for t in self.taxon.predict(all_ocr_text) if t.text})
        section_links: dict[str, list[dict[str, Any]]] = {}
        if species_seed:
            section_links = link_species_to_geology(
                species_names=species_seed,
                sections=[{"section_id": "fallback", "title": "Full text", "section_type": "other", "text": all_ocr_text}],
                llm_runtime=self.gemma_runtime if bool(self.config.extra.get("use_geology_llm", False)) else None,
            )
        knowledge_graph = build_knowledge_graph(section_links) if section_links else None

        # Two-pass: enumerate total regions across all pages so the progress
        # callback can map (current, total) onto a smooth 30-90% band.
        all_regions: list[tuple[Any, Any, int]] = []  # (page, region, region_idx_on_page)
        for page in pages:
            for ridx, region in enumerate(detect_figure_regions(page), start=1):
                all_regions.append((page, region, ridx))
        n_total = max(1, len(all_regions))
        done = 0

        for page, region, ridx in all_regions:
            region_img = cv2.imread(region.crop_path) if region.crop_path else cv2.imread(page.image_path)
            if region_img is None:
                done += 1
                continue

            if self._progress_cb is not None:
                try:
                    self._progress_cb(
                        done, n_total,
                        f"[{done + 1}/{n_total}] p{page.page_index:02d} region {ridx}",
                    )
                except Exception:
                    logger.debug("progress_cb(fallback-region) failed", exc_info=True)
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
            )
            for m in figure_matches:
                meta = m.get("metadata", {})
                meta["fallback_mode"] = True
                meta["fallback_reason"] = "missing_tei_caption"
                m["metadata"] = meta
                results.append(m)

        return results


# ---- module-level helpers -----------------------------------------------


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

    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / max(1, union)

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

    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / max(1, union)

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
        classical_panels.append(PanelCandidate(
            panel_id=mp.panel_id or f"M3_{idx:02d}",
            bbox=(int(mp.bbox[0]), int(mp.bbox[1]), int(mp.bbox[2]), int(mp.bbox[3])),
            score=mp.confidence,
            metadata=md,
        ))
        added += 1
    return added


def _extract_taxon_entities_from_text(text: str) -> list[CaptionEntity]:
    """Extract taxon-like strings from caption text for CaptionRecord.entities."""
    if not text:
        return []
    pattern = re.compile(r"\b([A-Z][a-zA-Z-]+\s+(?:sp\.|spp\.|cf\.|aff\.|[a-z][a-zA-Z-]+))\b")
    out: list[CaptionEntity] = []
    seen: set[str] = set()
    for m in pattern.finditer(text):
        taxon = m.group(1).strip()
        if taxon and taxon not in seen:
            seen.add(taxon)
            out.append(CaptionEntity(text=taxon, start=m.start(1), end=m.end(1), label="taxon", score=0.65))
    return out

