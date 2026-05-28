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

from .geology_extraction import build_knowledge_graph, link_species_to_geology
from .config import PipelineConfig
from .grobid import GrobidClient
from .association import match_panels
from .gemma_postprocess import apply_gemma_to_matches, build_gemma_backend_from_config
from .layout import choose_best_page, detect_figure_regions, render_pdf_pages
from .ocr import OCRBackend, normalize_ocr_tokens
from .scale_bar import detect_scale_bar_length_px, extract_scale_from_caption, extract_scale_from_ocr_text, merge_scale_info
from .segmentation import PanelSegmenter, SegmentationConfig
from .taxon import TaxonRecognizer
from .types import CaptionEntity, CaptionRecord, FigureRegion, PanelCandidate
from .utils import ensure_dir, slugify, stable_id, write_json, write_jsonl


class RadiolarianPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
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
        self._gemma_lock = threading.Lock()
        self._ocr_lock = threading.Lock()
        self._seg_lock = threading.Lock()
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
        if not model_path and str(self.config.extra.get("llm_backend", "transformers")).lower() != "ollama":
            return
        try:
            self.gemma_runtime = build_gemma_backend_from_config(self.config.extra)
        except Exception as exc:
            self.gemma_runtime = None
            self.config.extra["gemma_init_error"] = str(exc)
            logger.warning("Gemma4 backend init failed: %s", exc)

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
        with ThreadPoolExecutor(max_workers=max(1, self.config.num_workers)) as pool:
            futures = [pool.submit(self._process_one_pdf, pdf_path) for pdf_path in pdf_files]
            for fut in as_completed(futures):
                try:
                    result_rows = fut.result()
                    rows.extend(result_rows)
                except Exception:
                    logger.exception("PDF processing failed; continuing with remaining PDFs")

        manifest_path = self.config.manifests_dir() / "matches.jsonl"
        write_jsonl(manifest_path, rows)
        return rows

    def _process_one_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        paper_id = stable_id(pdf_path)

        # ------ OpenDataLoader path (opt-in) -----------------------------------
        if self.config.extra.get("use_opendataloader", False):
            return self._process_one_pdf_od(paper_id, pdf_path)

        # ------ GROBID + layout path (default) ---------------------------------
        return self._process_one_pdf_grobid(paper_id, pdf_path)

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
        for fig_idx, pair in enumerate(figures, start=1):
            if not pair.image_paths:
                continue

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
            caption = CaptionRecord(
                paper_id=paper_id,
                figure_id=pair.figure_id,
                caption=caption_text,
                entities=caption_entities,
                figure_number=str(pair.page_number),
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
    ) -> list[dict[str, Any]]:
        """Shared region processing: segment, OCR, match, (optional) Gemma, scale, geology.

        Used by both the normal caption-driven path and the visual-only fallback.
        """
        with self._seg_lock:
            panels = self.segmenter.segment_image(region_img)
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
        )

        if self.gemma_runtime is not None:
            with self._gemma_lock:
                matches = apply_gemma_to_matches(
                    runtime=self.gemma_runtime,
                    matches=matches,
                    caption_text=caption.caption or "",
                    ocr_labels=[tok.text for tok in ocr_tokens],
                    conf_threshold=float(self.config.extra.get("gemma_conf_threshold", 0.70)),
                    prompt_lang=str(self.config.extra.get("gemma_prompt_lang", "zh")),
                )

        # Attach geology links and scale bar info
        for m in matches:
            geo_list = section_links.get(m.species or "", [])
            if not geo_list and section_links:
                first_key = next(iter(section_links.keys())) if section_links else None
                geo_list = section_links.get(first_key, []) if first_key else []
            m.metadata["scale_bar"] = merged_scale.to_dict()
            m.metadata["geology_links"] = geo_list[:5]

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
                "panels": [asdict(p) for p in panels],
                "matches": results,
            })
        return results

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

        for page in pages:
            regions = detect_figure_regions(page)
            if not regions:
                continue

            for ridx, region in enumerate(regions, start=1):
                region_img = cv2.imread(region.crop_path) if region.crop_path else cv2.imread(page.image_path)
                if region_img is None:
                    continue

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

