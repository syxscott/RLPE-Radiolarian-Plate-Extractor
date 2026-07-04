from __future__ import annotations

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


# --- Pre-filters for non-specimen content ----------------------------------
# ``_looks_like_placeholder_caption`` lives in ``text_filters`` so the
# evaluation harness and unit tests can import it without dragging in
# the torch/gemma/paddleocr chain pulled by the full pipeline.

from .association import (
    _label_in_pair_lookup,
    _normalize_panel_label,
    is_valid_panel_label,
    match_panels,
)
from .config import PipelineConfig
from .converters import match_result_from_dict, run_output_from_provenance
from .gemma_postprocess import apply_gemma_to_matches, build_gemma_backend_from_config
from .geology_extraction import build_knowledge_graph, link_species_to_geology
from .grobid import GrobidClient, parse_paper_metadata_from_tei
from .layout import choose_best_page, detect_figure_regions, extract_figure_number, render_pdf_pages
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
from .text_filters import looks_like_placeholder_caption as _looks_like_placeholder_caption
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
        # NOTE: no shared _ocr_lock here. PaddleOCR (the default backend) is
        # thread-safe for concurrent .ocr() calls — the engine serializes its
        # own internal state and per-call locks would just serialize our
        # workers for no benefit. EasyOCR is the exception; for that backend,
        # ``OCRBackend.recognize`` uses its own engine-instance lock. The
        # segmenter (SAM2) and gemma runtimes are NOT concurrent-safe, so
        # they retain per-pipeline locks.
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
            self.config.extra.get("MiniMax_api_key")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("MINIMAX_API_KEY")
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

        # Build the M3 semantic engine (5-stage). ON by default when M3 is the
        # active backend and ``m3_enhanced_mode`` is not explicitly disabled.
        # For other backends, opt-in via ``m3_enhanced_mode = True``.
        if self.gemma_runtime is not None:
            want_m3 = self.config.extra.get("m3_enhanced_mode")
            if want_m3 is None:
                want_m3 = backend_name in minimax_backends
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
        if self._progress_cb is not None:
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

        rows: list[dict[str, Any]] = []
        total = len(pdf_files)
        completed = 0
        # Fire one initial tick so the UI can show "started" before the first
        # PDF actually finishes.
        self._emit_progress(0, total, f"Starting pipeline ({total} PDF(s))")
        with ThreadPoolExecutor(max_workers=max(1, self.config.num_workers)) as pool:
            futures = {pool.submit(self._process_one_pdf, p): p for p in pdf_files}
            try:
                for fut in as_completed(futures):
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
                    except (KeyboardInterrupt, SystemExit):
                        # User-initiated cancellation. Cancel in-flight
                        # workers and propagate so the CLI exits with a
                        # proper traceback and the API can flip the job to
                        # ``cancelled`` (the API's own cancel path doesn't
                        # go through ``run()``, but the API may also be
                        # wrapping this method).
                        for f in futures:
                            f.cancel()
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
                for f in futures:
                    f.cancel()
                raise

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
                run_output_dict = run_output_from_provenance(provenance_record, match_results)
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

        self._emit_progress(1, 1, f"Finished {pdf_path.name} ({len(rows)} matches)")
        return rows

    # -----------------------------------------------------------------------
    # OpenDataLoader-based processing
    # -----------------------------------------------------------------------

    def _process_map(
        self,
        *,
        paper_id: str,
        figure_id: str,
        caption_text: str,
        image_path: str,
    ) -> list[dict[str, Any]]:
        """Process a map / paleogeographic-map figure and produce stub
        panel records carrying the geographic context.

        Maps don't have species or panel_id, so the output is a single
        stub record (panel_id="MAP_CONTEXT") whose metadata carries:
          - location names mentioned in the caption
          - lat/lon coordinates extracted from the caption text
          - the full caption as evidence
          - the image path for downstream display
        Downstream ``_link_range_chart_geology`` can link this stub's
        context to other panels via the shared paper_id + section
        name. For now we just record it as a paper-level context
        anchor so an operator can find it.

        Heuristic-only — map caption parsing is hard and the existing
        regex-based geology_extraction already covers most of the
        location name extraction. This method mostly ensures the
        map figure isn't silently dropped by the pipeline.
        """
        loc_names: list[str] = []
        coords: list[tuple[float, float, str]] = []
        # Lightweight location-name extraction: capitalized
        # multi-word tokens that aren't common English words. The
        # full geology_extraction module handles the more complex
        # patterns; this is a quick safety net for map-only figures
        # that don't reach the caption parser.
        import re as _re

        for m in _re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", caption_text or ""):
            tok = m.group(1)
            if tok in {"Fig", "Figure", "Scale", "Bar", "The", "This", "Map"}:
                continue
            loc_names.append(tok)
        # Try to extract coordinates.
        for m in _re.finditer(
            r"\b(\d{1,3}(?:\.\d+)?)\s*°?\s*([NSns])?[,\s]+(\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])?\b",
            caption_text or "",
        ):
            try:
                lat = float(m.group(1))
                lon = float(m.group(3))
                if m.group(2) and m.group(2).upper() == "S":
                    lat = -lat
                if m.group(4) and m.group(4).upper() == "W":
                    lon = -lon
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    coords.append((lat, lon, m.group(0)))
            except ValueError:
                continue
        # Cap to a reasonable number of location names to avoid
        # noise from generic capitalized words.
        loc_names = loc_names[:10]
        coords = coords[:5]
        return [
            {
                "paper_id": paper_id,
                "figure_id": figure_id,
                "panel_id": "MAP_CONTEXT",
                "species": None,
                "panel_path": image_path,
                "bbox": None,
                "confidence": 0.0,
                "label_text": None,
                "caption_snippet": (caption_text or "")[:240],
                "ocr_text": None,
                "paper_metadata": None,
                "metadata": {
                    "extraction_method": "map_caption_heuristic",
                    "extraction_source": "map",
                    "location_names": loc_names,
                    "coordinates": [
                        {"lat": lat, "lon": lon, "raw": raw} for lat, lon, raw in coords
                    ],
                    "evidence_text": (caption_text or "")[:300],
                },
            }
        ]

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
        api_key = os.environ.get("ANTHROPIC_API_KEY") or self.config.extra.get("MiniMax_api_key")
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or self.config.extra.get(
            "MiniMax_endpoint", "https://api.minimaxi.com/anthropic"
        )
        model = os.environ.get("ANTHROPIC_MODEL") or self.config.extra.get(
            "MiniMax_model", "MiniMax-M3"
        )
        if not api_key:
            logger.warning(
                "range_chart: no ANTHROPIC_API_KEY set; skipping %s/%s", paper_id, figure_id
            )
            return []

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
                "metadata": {
                    "extraction_method": "range_chart_vision",
                    "extraction_source": "range_chart",
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
        unpaired: list[tuple[int, int, str]] = []  # (page_diff, size, path)
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
                images_dir = None
                # images_dir is constructed by the OD extractor under
                # <output_dir>/od_output/<paper_id>/<pdf_stem>_images.
                # Derive it from the figures' image paths because this
                # helper is nested inside run-level logic and does not
                # own a self/config reference.
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
                # images_dir is constructed by the OD extractor under
                # <output_dir>/od_output/<paper_id>/<pdf_stem>_images/.
                # We don't know paper_id from the target pair here, so
                # derive it from the figures' image paths.
                if figures and figures[0].image_paths:
                    sample = figures[0].image_paths[0]
                    # <work>/od_output/<paper_id>/<pdf_stem>_images/imageFileN.png
                    images_dir = _os.path.dirname(sample)
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
                        # Map file index → OD image page. If we have
                        # fewer OD entries than files, fall back to 0
                        # (unknown page).
                        img_page = od_image_pages[i] if i < len(od_image_pages) else 0
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
                # Enumerate the images directory and match by reading
                # the page number from each file (use PyMuPDF to get
                # the page count — too expensive). Simpler: just
                # collect all PNGs in images_dir and let the caller
                # pick by page_diff.
                if images_dir and _os.path.isdir(images_dir):
                    for fname in sorted(_os.listdir(images_dir)):
                        if not fname.lower().endswith(".png"):
                            continue
                        fpath = _os.path.join(images_dir, fname)
                        if _os.path.basename(fpath) in referenced:
                            logger.debug("raw OD scan: skipping %s (referenced)", fname)
                            continue
                        try:
                            sz = _os.path.getsize(fpath)
                        except OSError:
                            sz = 0
                        # All raw OD unpaired images are assigned page_diff=0
                        # because we don't have per-file page info here
                        # (the OD JSON's image elements have page numbers,
                        # but the file naming is sequential, not page-
                        # indexed). The caller will then merge these with
                        # the figure-level orphans which DO have page_diff.
                        # For now, treat them all as same-page candidates.
                        unpaired.append((0, sz, fpath))
                        logger.info(
                            "raw OD unpaired image: %s (size=%d)",
                            fpath,
                            sz,
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

    def _process_one_pdf_od(self, paper_id: str, pdf_path: Path) -> list[dict[str, Any]]:
        od_result = self.od_extractor.extract(pdf_path, self.config.resolved_output_dir())

        if not od_result.success:
            error = od_result.error or "unknown error"
            logger.warning(
                "OpenDataLoader failed (%s); falling back to GROBID+layout",
                error,
            )
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
                fig_idx - 1,
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
                        "range_chart %s: orphan search returned %s", pair.figure_id, rc_image
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
                if rc_image_path is None:
                    rc_image_path = self._find_orphan_image_for_range_chart(
                        figures, pair, od_result.json_data
                    )
                    logger.info(
                        "range_chart %s: no image paired; using orphan image %s",
                        pair.figure_id,
                        rc_image_path,
                    )
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
            # map: route to the proper multi-modal geology vision
            # prompt instead of falling through to the plate
            # segmentation path. Round 5 — previously these were
            # misclassified as plate/range_chart and silently lost
            # their specialized geological content.
            if fig_type in ("strat_column", "litholog_column", "paleogeographic_map"):
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
                        geo_links = self.m3_engine.extract_geology(
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
                    if geo_links:
                        # Wrap geology links into a stub record so the
                        # downstream eval/export pipeline sees them.
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
                                    "geo_vision_used": True,
                                    "geo_vision_figure_type": fig_type,
                                },
                            }
                        )
                        logger.info(
                            "%s %s: extracted %d geo links via vision",
                            fig_type,
                            pair.figure_id,
                            len(geo_links),
                        )
                self._emit_progress(
                    fig_idx,
                    n_figs,
                    f"[{fig_idx}/{n_figs}] {fig_type} → "
                    f"{len(geo_links) if geo_links else 0} vision links",
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
            if fig_type == "map":
                # Use the largest image on the same page (or
                # primary_path if available).
                map_image = primary_path
                if map_image is None:
                    map_image = self._find_orphan_image_for_range_chart(
                        figures, pair, od_result.json_data
                    )
                if map_image is not None:
                    map_results = self._process_map(
                        paper_id=paper_id,
                        figure_id=pair.figure_id,
                        caption_text=pair.caption_text or "",
                        image_path=map_image,
                    )
                    results.extend(map_results)
                    logger.info(
                        "map %s: extracted %d location names, %d coords",
                        pair.figure_id,
                        len(map_results[0]["metadata"]["location_names"]) if map_results else 0,
                        len(map_results[0]["metadata"]["coordinates"]) if map_results else 0,
                    )
                self._emit_progress(
                    fig_idx,
                    n_figs,
                    f"[{fig_idx}/{n_figs}] map → {len(map_results) if map_results else 0} context",
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
                results, paper_id, od_fulltext_sections=od_result.fulltext_sections
            )
        return results

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
        for r in results:
            md = r.get("metadata") or {}
            stage3 = (md.get("m3_diagnostic") or {}).get("stage3_panels") or []
            if stage3:
                figure_to_panels[r.get("figure_id")] = stage3

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
            # The plate image is the row's panel_path or its
            # figure_image_path; we need the plate-level image (not a
            # panel crop) to slice from. The classical CV stage
            # stores the plate on ``metadata.figure_image_path`` /
            # ``metadata.primary_image`` when running with the
            # OpenDataLoader path.
            plate_path = (
                r.get("panel_path")
                or md.get("figure_image_path")
                or md.get("primary_image")
                or md.get("image_path")
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
            md["panel_id_source"] = "m3_vision"
            md["stage3_confidence"] = matched.get("confidence")
            r["metadata"] = md
        return results

    def _apply_multi_plate_enrichment(
        self,
        results: list[dict[str, Any]],
        paper_id: str,
        *,
        od_fulltext_sections: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Round 7 multi-plate enrichment pass.

        For each plate figure whose row count is markedly below what the
        caption_parser would imply, fire a second-pass M3 vision call
        with ``expected_plate_label`` (the figure_id encodes it as
        ``od_plate_<pid>_p<page>_pl<N>`` so ``pl07`` -> ``Plate 7``) and
        the page-level caption text. M3 returns the panel list for THAT
        plate; we append the new panels to ``results`` so the downstream
        eval can score them.

        Trigger conditions (any one):
          * Figure has zero rows (OD missed the entire plate)
          * Figure has rows but every row has ``panel_id=None`` or
            every row has ``species=None``
          * Caption parser for the figure's caption text produced more
            than 2× the figure's actual row count

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
            if not fid or r.get("panel_id") in {"RANGE_CHART", "MAP_CONTEXT", "_ingestion_od_failed"}:
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

        # Walk each figure that looks under-populated.
        appended = 0
        for fid, fig_rows in by_fig.items():
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

            # Skip maps, range charts, geo_vision stubs.
            sample_src = (fig_rows[0].get("metadata") or {}).get("extraction_source", "")
            if sample_src in {"map", "range_chart", "geo_vision"}:
                continue

            # Trigger condition: zero rows OR every row missing species.
            n_with_species = sum(1 for r in fig_rows if r.get("species"))
            n_with_panel_id = sum(1 for r in fig_rows if r.get("panel_id"))
            is_underpopulated = (
                len(fig_rows) == 0
                or (n_with_species == 0 and n_with_panel_id == 0)
            )
            if not is_underpopulated:
                continue

            # Find the plate image: prefer the largest image_path in any
            # row's metadata, else the row's panel_path.
            image_path = None
            for r in fig_rows:
                md = r.get("metadata") or {}
                cand = md.get("primary_image") or md.get("figure_image_path") or md.get("image_path")
                if cand and Path(cand).is_file():
                    image_path = cand
                    break
                if r.get("panel_path") and Path(r["panel_path"]).is_file():
                    image_path = r["panel_path"]
                    break
            if not image_path:
                continue

            # Page-level caption context: page text + adjacent pages.
            ctx_pages = []
            if page_idx is not None:
                for off in (-1, 0, 1):
                    t = page_text.get(page_idx + off)
                    if t:
                        ctx_pages.append(t)
            page_caption = "\n\n".join(ctx_pages) or all_captions_blob

            try:
                with _PILImage.open(image_path) as im:
                    plate_image = im.convert("RGB")
            except Exception as exc:
                logger.debug(
                    "multi_plate_enrich: cannot open %s: %s", image_path, exc
                )
                continue

            try:
                panels = self.m3_engine.enrich_plate_panels(
                    image=plate_image,
                    page_caption=page_caption[:3000],  # cap to avoid token bloat
                    paper_id=paper_id,
                    figure_id=fid,
                    expected_plate_label=plate_label,
                )
            except Exception as exc:
                logger.warning(
                    "multi_plate_enrich failed for %s/%s: %s",
                    paper_id, fid, exc,
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
                conf = float(p.get("confidence") or 0.7)
                results.append({
                    "paper_id": paper_id,
                    "figure_id": fid,
                    "panel_id": norm_lbl,
                    "species": sp if sp else None,
                    "panel_path": None,
                    "bbox": None,
                    "confidence": conf,
                    "label_text": lbl,
                    "caption_snippet": (page_caption or "")[:240],
                    "ocr_text": None,
                    "paper_metadata": None,
                    "metadata": {
                        "extraction_method": "multi_plate_enrich",
                        "extraction_source": "multi_plate_enrich",
                        "panel_id_source": "m3_vision",
                        "expected_plate_label": plate_label,
                        "figure_number": (
                            (plate_label or "").replace("Plate ", "") or None
                        ),
                    },
                })
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
                            for k in ("name", "age_range", "formation_thickness_m", "coordinates")
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
                            for k in ("species", "section", "range_top", "range_base", "biozone")
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
        * ``extraction_source == "map_context"`` -> ``figure_type="map"``
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
            or ["range_chart", "stratigraphic_column", "litholog_column", "paleogeographic_map"]
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
            if not figure_type:
                src = md.get("extraction_source")
                if src == "range_chart":
                    figure_type = "range_chart"
                elif src == "map_context":
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
                geo_links = self.m3_engine.extract_geology(
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
    # Original GROBID + layout path
    # -----------------------------------------------------------------------

    def _process_one_pdf_grobid(self, paper_id: str, pdf_path: Path) -> list[dict[str, Any]]:
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
        if not tei_captions:
            return self._fallback_process_without_captions(
                paper_id,
                pages,
                paper_metadata=paper_meta,
            )

        for idx, caption in enumerate(tei_captions, start=1):
            self._emit_progress(
                idx - 1,
                max(1, len(tei_captions)),
                f"[{idx}/{len(tei_captions)}] {caption.figure_id}",
            )

            best_page = choose_best_page(
                pages, caption.figure_number, caption.caption, window=self.config.caption_window
            )
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
            results.extend(figure_matches)
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
                        "ingestion_error": error,
                        "ingestion_warning": True,
                    },
                }
            )
        return results

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
        if _looks_like_placeholder_caption(caption_text):
            return None

        try:
            from PIL import Image as _PILImage

            if hasattr(region_img, "shape"):
                _rgb = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
                plate_pil = _PILImage.fromarray(_rgb)
            else:
                with _PILImage.open(str(region_img)) as _im:
                    plate_pil = _im.convert("RGB")
        except Exception:
            return None

        user_prompt = (
            f"Paper: {paper_id}\n"
            f"Figure: {figure_id}\n"
            f"Caption:\n{caption_text[:2000]}\n\n"
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
        panels_data = result.get("panels") or result.get("answer")
        if panels_data is None:
            raw = result.get("raw_text", "")
            if not raw:
                return None
            try:
                import json as _json
                import re as _re

                # Strip markdown code fences (```json ... ```) that M3
                # wraps around its JSON output. Without this, _json.loads
                # fails on the raw_text and we silently fall back to the
                # classical pipeline (which drops to ~83% F1 on beccaro).
                cleaned = raw.strip()
                cleaned = _re.sub(r"^```(?:json)?\s*", "", cleaned, flags=_re.MULTILINE)
                cleaned = _re.sub(r"\s*```\s*$", "", cleaned, flags=_re.MULTILINE)
                cleaned = cleaned.strip()
                parsed = _json.loads(cleaned)
                panels_data = parsed.get("panels") or parsed
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
        for p in panels_data:
            label = str(p.get("label", "")).strip()
            species = p.get("species")
            conf = float(p.get("confidence", 0.0))
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
                caption_snippet=caption.caption[:240] if hasattr(caption, "caption") else "",
                ocr_text=None,
                paper_metadata=paper_metadata,
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
                        caption_pairs = self.m3_engine.parse_caption(caption.caption or "")
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
                        for r in llm_results:
                            if r.get("species"):
                                continue
                            label = r.get("panel_id") or r.get("label_text") or ""
                            matched_key = _label_in_pair_lookup(label, pair_lookup)
                            if matched_key:
                                r["species"] = pair_lookup[matched_key]
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
                            if not is_valid_panel_label(lbl):
                                continue
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
                                    caption_snippet=(caption.caption or "")[:240],
                                    ocr_text=None,
                                    paper_metadata=paper_metadata,
                                    metadata={
                                        "extraction_method": "llm_first",
                                        "llm_backend": getattr(
                                            self.gemma_runtime, "backend_name", "unknown"
                                        ),
                                        "panel_count": len(llm_results) + 1,
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
                "LLM-first failed for %s/%s, falling back to classical path", paper_id, figure_id
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
                    m3_caption_pairs = self.m3_engine.parse_caption(caption.caption or "")
                    m3_diag["stage1_pairs"] = len(m3_caption_pairs)
                # Stage 2: plate classifier — early exit on non-radiolarian
                if self.m3_engine._stage_enabled(2):
                    m3_plate_cls = self.m3_engine.classify_plate(plate_pil)
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
                    bbox=(int(mp.bbox[0]), int(mp.bbox[1]), int(mp.bbox[2]), int(mp.bbox[3])),
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
            cv2.imwrite(str(panel_path), crop)
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

        # Build a unique panel_caption key per match. Two matches in the
        # same figure can share a panel_id (OCR misread duplicates) — a
        # plain ``{m.panel_id: caption}`` dict comprehension would
        # silently drop all but the last occurrence, and the lookup
        # below (also keyed on panel_id) would assign the same geo list
        # to every duplicate. Fall back to ``f"idx_{i}"`` when the
        # panel_id is missing OR has already been seen.
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
                # Panel-level fallback: scan this panel's own caption for
                # age/formation/locality mentions. Use ``panel_keys[i]``
                # (the same uniquified key we stored above) so a duplicate
                # panel_id doesn't accidentally pull the previous match's
                # geo list.
                key = panel_keys[i] if i < len(panel_keys) else (m.panel_id or f"idx_{i}")
                geo_list = panel_geo.get(key, [])
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
                key = panel_keys[i] if i < len(panel_keys) else (row.get("panel_id") or f"idx_{i}")
                geo_list = panel_geo.get(key, [])
            md["scale_bar"] = merged_scale.to_dict()
            md["geology_links"] = geo_list[:5]
            md.setdefault("m3_diagnostic", {})
            row["metadata"] = md
        return rows

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
        action = self.gemma_fallback_handler(error_info)

        if action == "stop":
            raise RuntimeError(
                f"[MiniMax] user stopped pipeline at paper={paper_id} figure={figure_id}: "
                f"{error_info.get('error', '?')}"
            )

        if action == "rules":
            logger.warning(
                "[MiniMax] API error, falling back to rule pipeline for %s/%s", paper_id, figure_id
            )
            for m in result:
                m.metadata["MiniMax_fallback_action"] = "rules"
            return result

        if action == "gemma4":
            logger.warning(
                "[MiniMax] API error, switching to local Gemma4 for %s/%s", paper_id, figure_id
            )
            # Bug #7 fix: cache the local Gemma4 runtime after the first
            # successful build so subsequent fallbacks don't reload a multi-GB
            # model each time.
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
                llm_runtime=self.gemma_runtime
                if bool(self.config.extra.get("use_geology_llm", False))
                else None,
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
            region_img = (
                cv2.imread(region.crop_path) if region.crop_path else cv2.imread(page.image_path)
            )
            if region_img is None:
                done += 1
                continue

            self._emit_progress(
                done,
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
        classical_panels.append(
            PanelCandidate(
                panel_id=mp.panel_id or f"M3_{idx:02d}",
                bbox=(int(mp.bbox[0]), int(mp.bbox[1]), int(mp.bbox[2]), int(mp.bbox[3])),
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
    pattern = re.compile(r"\b([A-Z][a-zA-Z-]+\s+(?:sp\.|spp\.|cf\.|aff\.|[a-z][a-zA-Z-]+))\b")
    out: list[CaptionEntity] = []
    seen: set[str] = set()
    for m in pattern.finditer(text):
        taxon = m.group(1).strip()
        if taxon and taxon not in seen:
            seen.add(taxon)
            out.append(
                CaptionEntity(text=taxon, start=m.start(1), end=m.end(1), label="taxon", score=0.65)
            )
    return out
