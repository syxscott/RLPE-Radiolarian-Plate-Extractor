from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from .config import PipelineConfig
from .grobid import GrobidClient, parse_captions_from_tei
from .association import match_panels
from .gemma_postprocess import apply_gemma_to_matches, load_gemma4_model
from .layout import choose_best_page, detect_figure_regions, render_pdf_pages
from .ocr import OCRBackend, normalize_ocr_tokens
from .segmentation import PanelSegmenter, SegmentationConfig
from .taxon import TaxonRecognizer
from .types import MatchResult
from .utils import ensure_dir, slugify, stable_id, write_json, write_jsonl


class RadiolarianPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.grobid = GrobidClient(server_url=config.grobid_url)
        self.ocr = OCRBackend(backend=config.ocr_backend, use_gpu=config.use_gpu)
        self.taxon = TaxonRecognizer(model=config.taxon_model)
        self.segmenter = PanelSegmenter(
            config=SegmentationConfig(score_threshold=config.min_panel_score),
            checkpoint=config.extra.get("sam2_checkpoint"),
            model_cfg=config.extra.get("sam2_model_cfg"),
        )
        self.gemma_runtime = None
        self._try_init_gemma()

    def _try_init_gemma(self) -> None:
        if not self.config.extra.get("use_gemma4", False):
            return
        model_path = self.config.extra.get("gemma_model_path")
        if not model_path:
            return
        try:
            self.gemma_runtime = load_gemma4_model(
                model_path=model_path,
                use_4bit=bool(self.config.extra.get("gemma_use_4bit", True)),
                bfloat16=bool(self.config.extra.get("gemma_bfloat16", True)),
                device_map=str(self.config.extra.get("gemma_device_map", "auto")),
            )
        except Exception as exc:
            self.gemma_runtime = None
            self.config.extra["gemma_init_error"] = str(exc)

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
                result_rows = fut.result()
                rows.extend(result_rows)

        manifest_path = self.config.manifests_dir() / "matches.jsonl"
        write_jsonl(manifest_path, rows)
        return rows

    def _process_one_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        paper_id = stable_id(pdf_path)
        grobid_result = self.grobid.process_pdf(pdf_path, self.config.resolved_output_dir())

        tei_captions = grobid_result.captions
        if not grobid_result.success:
            return []

        pages = render_pdf_pages(pdf_path, self.config.figures_dir() / paper_id, dpi=self.config.render_dpi)
        results: list[dict[str, Any]] = []

        captions_by_figure = {cap.figure_id: cap for cap in tei_captions}
        if not pages:
            return []

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

            # Use the best region first, then others by score.
            chosen_regions.sort(key=lambda r: (-r.score, r.page_index, r.bbox[1], r.bbox[0]))
            region = chosen_regions[0]
            region_img = cv2.imread(region.crop_path) if region.crop_path else cv2.imread(best_page.image_path)
            if region_img is None:
                continue

            panels = self.segmenter.segment_image(region_img)
            if not panels:
                continue

            ocr_tokens = normalize_ocr_tokens(self.ocr.recognize(region_img))
            taxon_entities = self.taxon.predict(caption.caption or "")

            for panel_index, panel in enumerate(panels, start=1):
                x, y, w, h = panel.bbox
                crop = region_img[y : y + h, x : x + w]
                panel_dir = ensure_dir(self.config.panels_dir() / paper_id / (caption.figure_id or f"fig_{idx}"))
                panel_path = panel_dir / f"panel_{panel_index:02d}.png"
                cv2.imwrite(str(panel_path), crop)
                panel.image_path = str(panel_path)
                panel.region_id = region.region_id
                panel.source_page = region.page_index
                panel.panel_index = panel_index

            matches = match_panels(paper_id, caption.figure_id, caption, panels, ocr_tokens, taxon_entities)

            if self.gemma_runtime is not None:
                matches = apply_gemma_to_matches(
                    runtime=self.gemma_runtime,
                    matches=matches,
                    caption_text=caption.caption,
                    ocr_labels=[tok.text for tok in ocr_tokens],
                    conf_threshold=float(self.config.extra.get("gemma_conf_threshold", 0.70)),
                    prompt_lang=str(self.config.extra.get("gemma_prompt_lang", "zh")),
                )

            for m in matches:
                results.append(m.to_dict())

            if self.config.save_intermediate:
                write_json(self.config.manifests_dir() / paper_id / f"{slugify(caption.figure_id)}.json", {
                    "paper_id": paper_id,
                    "figure_id": caption.figure_id,
                    "caption": caption.caption,
                    "figure_number": caption.figure_number,
                    "page_index": best_page.page_index if best_page else None,
                    "region": asdict(region),
                    "ocr": [asdict(t) for t in ocr_tokens],
                    "taxa": [asdict(t) for t in taxon_entities],
                    "panels": [asdict(p) for p in panels],
                    "matches": [m.to_dict() for m in matches],
                })
        return results

