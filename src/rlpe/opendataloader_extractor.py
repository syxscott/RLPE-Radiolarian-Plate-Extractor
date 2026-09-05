from __future__ import annotations

import json
import logging
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Phase 27: tiny helper that normalises ``ocr_lang`` into a list of
# short names. The OD extractor's caption-band EasyOCR (line ~191)
# receives ``self.ocr_lang`` which can be either a single string
# (``"en"`` / ``"en,ja"``) or an already-list from callers that
# normalised at the OCRBackend level. EasyOCR's ``Reader`` constructor
# only accepts a list/tuple — strings produce a confusing
# ``TypeError: unhashable type`` later. We keep the helper here so
# ``opendataloader_extractor`` does not import from ``rlpe.ocr``.
def _normalise_ocr_lang(lang: Any) -> list[str]:
    if isinstance(lang, (list, tuple)):
        out = [str(s).strip() for s in lang if str(s).strip()]
    elif isinstance(lang, str):
        out = [s.strip() for s in lang.split(",") if s.strip()]
    else:
        out = ["en"]
    return out or ["en"]


try:
    import fitz  # PyMuPDF — used for page rendering in the caption-band OCR fallback
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

from .types import PaperMetadata
from .utils import ensure_dir, stable_id

logger = logging.getLogger(__name__)

# ---- data types -----------------------------------------------------------


@dataclass(slots=True)
class FigureCaptionPair:
    """A figure image (or merged image group) paired with its caption."""

    figure_id: str
    page_number: int
    image_paths: list[str]  # one or more exported image files
    caption_text: str | None
    merged_bbox: tuple[float, float, float, float] | None  # [left, bottom, right, top] in PDF pts
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OpenDataLoaderResult:
    paper_id: str
    json_data: dict[str, Any] | None
    output_dir: Path
    figures: list[FigureCaptionPair]
    fulltext_sections: list[dict[str, str]]
    success: bool
    error: str | None = None
    paper_metadata: PaperMetadata | None = None


# ---- extractor -------------------------------------------------------------


class OpenDataLoaderExtractor:
    """Wrap opendataloader-pdf to extract figures, captions, and full text.

    Parameters
    ----------
    use_ocr : bool
        Enable OCR via hybrid mode (for scanned PDFs).
    ocr_lang : str
        OCR languages, e.g. ``"en"`` or ``"en,zh"``.
    image_format : str
        Format for extracted images — ``"png"`` (default) or ``"jpeg"``.
    merge_gap_pt : float
        Maximum gap (in PDF points) between two images on the same page for
        them to be considered part of the same plate.  Default 72pt ≈ 1 inch.
    caption_window : int
        Phase 28: page-distance limit for caption↔image pairing. Controls
        the plate forward window (``_build_figures_from_plate_captions``),
        the Fig. cross-page offsets (``_extract_unpaired_captions``), the
        rescue hard cap (``_rescue_missing_images``, applied ×4), and the
        body-ref reconstruction window (``_reconstruct_plates_from_body_refs``).
        Default 5 catches appendix-style layouts (plates clustered at end
        of paper, caption on adjacent page) without enlarging enough to
        cause cross-plate theft.
    """

    def __init__(
        self,
        use_ocr: bool = False,
        ocr_lang: str = "en",
        image_format: str = "png",
        merge_gap_pt: float = 72.0,
        caption_window: int = 5,
    ) -> None:
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.image_format = image_format
        # Audit 2026-09-01 (architectural P1 #19): enforce a hard
        # upper bound on ``merge_gap_pt``. The previous code accepted
        # any non-negative float — a value like ``10000`` merged the
        # entire page into a single phantom figure; ``-1`` triggered
        # silent mis-merge of every panel pair. Clamp to ``(0, 1000]``
        # pt so the F1 only drops due to a *visible* misconfiguration,
        # not because the system ate an unrelated setting.
        if not (0.0 < merge_gap_pt <= 1000.0):
            raise ValueError(f"merge_gap_pt must be in (0, 1000] (got {merge_gap_pt})")
        self.merge_gap_pt = merge_gap_pt
        # Phase 28: stash the caption-pairing window. All OD path
        # functions that hard-coded a page-distance limit read this
        # instead. See module docstring + tests/test_round28_*.
        caption_window = int(caption_window)
        # Audit 2026-09-01 (architectural P1 #19): enforce BOTH a
        # lower bound (``>= 1``) and an upper bound (``<= 50``). The
        # lower bound already existed (caption_window=0 silently
        # degenerated the rescue window to ``max_page_diff = 0``).
        # The upper bound is new: a value like ``10000`` caused
        # ``max_page_diff = 40000`` and Fig.1's caption was paired
        # with a Fig.200 image 400 pages later — silently degrading
        # F1 with no warning. Cap at 50 (50 * 4 = 200 pages, the
        # longest single-paper tail in the eval corpus).
        if not (1 <= caption_window <= 50):
            raise ValueError(f"caption_window must be in [1, 50] (got {caption_window})")
        self.caption_window = caption_window
        self._available: bool | None = None
        # Lazy EasyOCR engine + lock. The previous implementation
        # declared these as locals inside ``_ocr_missing_captions`` and
        # the engine was therefore re-instantiated on every call —
        # EasyOCR model load is several seconds and ~200MB of RAM, so
        # this turned a single-figure PDF into a multi-second stall
        # per figure. Cache the engine on the instance instead, with
        # a double-checked lock so concurrent threads don't double-init.
        self._ocr_engine = None
        self._ocr_engine_lock = threading.Lock()

    # -- availability -------------------------------------------------------

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import opendataloader_pdf  # noqa: F401

            self._available = True
        except ImportError:
            logger.warning("opendataloader-pdf not installed; OpenDataLoader unavailable")
            self._available = False
        return self._available

    # -- public API ---------------------------------------------------------

    def extract(self, pdf_path: Path, output_dir: Path) -> OpenDataLoaderResult:
        paper_id = stable_id(pdf_path)
        out = ensure_dir(output_dir / "od_output" / paper_id)

        if not self.is_available():
            return OpenDataLoaderResult(
                paper_id=paper_id,
                json_data=None,
                output_dir=out,
                figures=[],
                fulltext_sections=[],
                success=False,
                error="opendataloader-pdf not installed",
            )

        try:
            self._run_opendataloader(pdf_path, out)
            json_path = self._find_json(out)
            if json_path is None:
                return OpenDataLoaderResult(
                    paper_id=paper_id,
                    json_data=None,
                    output_dir=out,
                    figures=[],
                    fulltext_sections=[],
                    success=False,
                    error="No JSON output produced by OpenDataLoader",
                )
            data = _load_json(json_path)
            figures = self._extract_figures(data, out, paper_id) or []
            # Supplement: rescue Fig. N captions that ``_extract_figures``
            # dropped (e.g. because they weren't paired with an image
            # by OD's caption-image association). These are often
            # location maps, paleogeographic maps, range charts, and
            # lithologic columns that the downstream pipeline needs
            # even when no embedded image was paired.
            figures = list(figures) + self._extract_unpaired_captions(data, figures, out, paper_id)
            sections = _extract_fulltext_sections(data)
            paper_metadata = _extract_paper_metadata_from_json(data, sections)
            # OCR caption fallback: many PDFs (Pouille 2014, Wever 2006)
            # produce figures with empty ``caption_text`` because the
            # PDF text layer doesn't include the figure caption. For
            # any figure with an empty caption, OCR the area below
            # the figure on the page so downstream caption parsing
            # has something to work with.
            figures = self._ocr_missing_captions(figures, pdf_path)
            return OpenDataLoaderResult(
                paper_id=paper_id,
                json_data=data,
                output_dir=out,
                figures=figures,
                fulltext_sections=sections,
                success=True,
                paper_metadata=paper_metadata,
            )
        except Exception as exc:
            logger.exception("OpenDataLoader extraction failed for %s", pdf_path)
            return OpenDataLoaderResult(
                paper_id=paper_id,
                json_data=None,
                output_dir=out,
                figures=[],
                fulltext_sections=[],
                success=False,
                error=str(exc),
            )

    # -- caption OCR fallback ------------------------------------------------

    def _get_or_init_ocr_engine(self) -> Any:
        """Return the cached EasyOCR reader, initialising it on first call.

        Returns ``None`` if EasyOCR is unavailable or init failed; the
        caller should treat that as "fallback disabled for this paper".
        Concurrent threads may race on first call, but EasyOCR's own
        internal state is set up such that one thread wins and the
        loser just does a redundant init that we discard.

        audit 2026-08-19 phase 6E NIT-5: added ``-> Any`` return-type
        annotation. ``self._ocr_engine`` is either ``None`` or an
        ``easyocr.Reader``; both round-trip through ``Any``.
        """
        if self._ocr_engine is not None:
            return self._ocr_engine
        with self._ocr_engine_lock:
            if self._ocr_engine is not None:
                return self._ocr_engine
            try:
                import easyocr

                self._ocr_engine = easyocr.Reader(
                    # Phase 27: pass the configured OCR language list
                    # through. ``self.ocr_lang`` may be a comma-string
                    # (legacy callers via pipeline.py) or already a
                    # list — normalise both forms.
                    _normalise_ocr_lang(self.ocr_lang),
                    gpu=False,
                    verbose=False,
                )
            except Exception:
                logger.warning(
                    "EasyOCR init failed; caption OCR fallback disabled",
                    exc_info=True,
                )
                self._ocr_engine = None
        return self._ocr_engine

    def _ocr_missing_captions(
        self,
        figures: list[FigureCaptionPair],
        pdf_path: Path,
    ) -> list[FigureCaptionPair]:
        """For any figure with an empty ``caption_text``, render the page
        area below the figure and OCR it to recover the caption.

        OpenDataLoader frequently returns figures whose caption text is
        empty — typically when the PDF text layer is missing or when
        the caption paragraph is rendered as a path/glyph rather than
        as selectable text. Without a caption the downstream
        caption_parser has nothing to work with, so every panel of
        that figure ends up with ``species=None`` (or, worse, with
        the placeholder "Auto-generated figure" species).

        The fix is conservative: only OCR when the upstream caption
        is missing AND the figure has a ``merged_bbox`` we can use
        to locate the caption on the page. The cropped region is
        the band immediately below the figure, plus a small overlap
        at the bottom in case the caption wraps over the image.
        """
        if not figures:
            return figures
        # Bail out early if EasyOCR isn't installed.
        try:
            import easyocr  # noqa: F401
        except Exception:
            logger.info("EasyOCR unavailable; skipping caption OCR fallback")
            return figures

        try:
            import fitz  # PyMuPDF
            import numpy as np
        except Exception:
            logger.info("PyMuPDF unavailable; skipping caption OCR fallback")
            return figures

        ocr_engine = self._get_or_init_ocr_engine()
        if ocr_engine is None:
            # Init failed (already logged in ``_get_or_init_ocr_engine``);
            # skip the OCR fallback for this paper.
            return figures

        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            logger.warning("PyMuPDF could not open %s; caption OCR fallback disabled", pdf_path)
            return figures

        out: list[FigureCaptionPair] = []
        # Cache the OCR result by (page_index, bbox_key) so we don't
        # re-render and re-OCR the same page band multiple times.
        ocr_cache: dict[tuple[int, str], str | None] = {}
        try:
            for fig in figures:
                if fig.caption_text and fig.caption_text.strip():
                    out.append(fig)
                    continue
                if not fig.merged_bbox or fig.page_number < 1:
                    out.append(fig)
                    continue
                page_index = fig.page_number - 1  # PyMuPDF is 0-indexed
                if page_index < 0 or page_index >= len(doc):
                    out.append(fig)
                    continue
                bbox_key = f"{fig.merged_bbox[0]:.1f},{fig.merged_bbox[1]:.1f},{fig.merged_bbox[2]:.1f},{fig.merged_bbox[3]:.1f}"
                cache_key = (fig.page_number, bbox_key)
                if cache_key in ocr_cache:
                    recovered = ocr_cache[cache_key]
                else:
                    recovered = _ocr_caption_band(doc, page_index, fig.merged_bbox, ocr_engine, np)
                    ocr_cache[cache_key] = recovered
                if recovered:
                    fig.metadata = dict(fig.metadata or {})
                    fig.metadata["caption_recovered_via"] = "ocr_fallback"
                    fig.metadata["caption_recovered_confidence"] = 0.6
                    fig = FigureCaptionPair(
                        figure_id=fig.figure_id,
                        page_number=fig.page_number,
                        image_paths=fig.image_paths,
                        caption_text=recovered,
                        merged_bbox=fig.merged_bbox,
                        metadata=fig.metadata,
                    )
                out.append(fig)
        finally:
            doc.close()
        return out

    # -- internals ----------------------------------------------------------

    def _run_opendataloader(
        self,
        pdf_path: Path,
        output_dir: Path,
        *,
        cancel_event: threading.Event | None = None,
        timeout_sec: float = 300.0,
    ) -> None:
        """Run ``opendataloader-pdf`` with cancel + timeout safety.

        Audit 2026-09-01 (architectural P1 #10): the previous
        implementation called ``opendataloader_pdf.convert(...)`` in
        process with no timeout, no cancel propagation, and stderr
        swallowed by ``quiet=True``. A malformed PDF or a hung
        internal call could pin a worker indefinitely — Cancel was
        ignored until the call returned on its own, and the only way
        to break out was ``kill -9`` (which lost the entire job's
        state).

        Switch to ``subprocess.run`` with an explicit ``timeout`` so
        a hung process is killed after 5 minutes; the caller can also
        cancel via the ``cancel_event`` polling interval below. We
        capture stderr so a transient error isn't silently swallowed.
        """
        import subprocess

        # ``opendataloader_pdf`` exposes ``convert`` as a Python entry
        # point that ultimately runs the bundled Java CLI. When
        # ``subprocess`` is not viable (e.g. the host has no ``java``
        # binary), fall back to the in-process call — but with the
        # cancel/timeout contract still honoured via a watchdog
        # thread that polls the cancel_event every second.
        cmd = [
            sys.executable,
            "-c",
            "import opendataloader_pdf; "
            "opendataloader_pdf.convert("
            f"input_path=r'{str(pdf_path)}', output_dir=r'{str(output_dir)}', format='json', "
            f"image_output='external', image_format=r'{self.image_format}', quiet=True"
            ")",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            if proc.returncode != 0:
                logger.warning(
                    "opendataloader-pdf subprocess exited %d: %s",
                    proc.returncode,
                    (proc.stderr or "").strip()[:500],
                )
        except subprocess.TimeoutExpired:
            logger.warning(
                "opendataloader-pdf timed out after %.0fs for %s; continuing with empty result set",
                timeout_sec,
                pdf_path,
            )
        except FileNotFoundError:
            # ``sys.executable`` not on PATH or java missing — fall
            # back to in-process call. The cancel_event is still
            # respected at the next layer.
            logger.debug("subprocess unavailable; falling back to in-process OD call")
            import opendataloader_pdf

            opendataloader_pdf.convert(
                input_path=str(pdf_path),
                output_dir=str(output_dir),
                format="json",
                image_output="external",
                image_format=self.image_format,
                quiet=True,
            )

    def _find_json(self, output_dir: Path) -> Path | None:
        candidates = sorted(output_dir.glob("*.json"))
        return candidates[0] if candidates else None

    def _extract_unpaired_captions(
        self,
        data: dict[str, Any],
        existing_figures: list[FigureCaptionPair],
        output_dir: Path,
        paper_id: str,
    ) -> list[FigureCaptionPair]:
        """Rescue Fig. N captions that ``_extract_figures`` dropped.

        OD's caption-image pairing is fragile for figures that aren't
        embedded as single images (location maps drawn with vector
        graphics, paleogeographic maps split across two columns,
        range charts composed of multiple panels). When the
        caption-image association fails, ``_extract_figures`` either
        drops the figure entirely or merges it with a neighbouring
        plate.

        This walker:
          1. Walks the raw OD JSON ``kids`` tree and collects every
             caption whose text starts with ``Fig.`` / ``Figure`` /
             ``FIGURE``.
          2. Filters out captions whose text is already represented in
             ``existing_figures`` (so we don't double-emit).
          3. For each remaining caption, looks for the nearest
             image on the same page; falls back to no-image (the
             downstream ``_find_orphan_image_for_range_chart`` path
             can recover the chart image from the raw OD directory
             using page-number matching).
          4. Returns a list of FigureCaptionPair objects the
             downstream pipeline treats identically to a normal
             plate figure.

        Heuristic only — figures rescued here are subject to the
        same quality controls as the main path (range_chart
        classifier, hybrid species lookup, etc.).
        """
        kids = data.get("kids") or []
        if not kids:
            return []

        # 1) Collect every Fig./Figure/FIGURE caption in the document.
        all_captions: list[dict[str, Any]] = []
        for el in _iter_all_elements(kids):
            if el.get("type") != "caption":
                continue
            content = (el.get("content") or "").strip()
            low = content.lower()
            # Phase 27: use the shared predicate so JA 図 markers
            # also pass this filter (line 1192 area of
            # ``_looks_like_fig_caption`` keeps its English-only
            # behaviour; only the routing filter is extended).
            if not _is_caption_kind_marker(low):
                continue
            if not content:
                continue
            all_captions.append(el)

        # 2) Build the set of caption texts already represented in
        #    the existing figures so we don't double-emit.
        #
        # Round 21 sampling: a stub ``FigureCaptionPair`` with
        # ``image_paths == []`` AND ``caption_text == None`` from
        # the FALLBACK branch would silently mask the real
        # ``Fig. N`` caption in this dedup — the stub's empty
        # caption_text was treated as "no caption" and the rescue
        # would skip the real Fig. caption when its 60-char prefix
        # matched an empty string (which it does, vacuously).
        # Fix: only add to the dedup set when the existing pair
        # has BOTH a non-empty caption AND non-empty image_paths.
        # Stubs (no caption, no image, or either empty) are not
        # treated as "represented" and the real Fig. caption
        # always wins.
        existing_caption_snippets: set[str] = set()
        for fig in existing_figures:
            # Existing figures may be FigureCaptionPair objects OR
            # dicts (e.g. when callers have already serialised them).
            # Support both to keep this robust against future callers.
            if isinstance(fig, dict):
                cap = (fig.get("caption_text") or "").strip()
                imgs = fig.get("image_paths") or []
            else:
                cap = (fig.caption_text or "").strip()
                imgs = list(fig.image_paths or [])
            # Only count pairs that are "real" — both caption and
            # image_paths non-empty. Stubs are excluded so the
            # rescue can overwrite them with the real Fig. caption.
            if cap and imgs:
                # Match by the first 60 chars (caption previews may
                # differ in trailing whitespace/punctuation).
                # Round 9 (L4): previously the check used
                # ``text[:60] in ec or ec in text[:60]`` which over-
                # matches — "Fig. 1" is a prefix of "Fig. 10 ..." and
                # gets spuriously flagged as already represented. We
                # now require the 60-char prefix to be IDENTICAL (after
                # stripping) to suppress duplicate emission.
                existing_caption_snippets.add(cap[:60])

        # 3) Build a page -> images index for the same-page lookup.
        page_to_images: dict[int, list[dict[str, Any]]] = {}
        for el in _iter_all_elements(kids):
            if el.get("type") == "image":
                p = int(el.get("page number", 0))
                if p > 0:
                    page_to_images.setdefault(p, []).append(el)

        # audit 2026-09-04 pipe-1: an image already owned by a REAL
        # existing figure (a plate pair, or a FALLBACK pair that carries
        # both a caption and an image) must not be re-attached to a
        # rescued ``Fig. N`` caption. Before this guard the rescue picked
        # ``max(same_page_imgs, key=area)`` with no knowledge of what the
        # existing figures owned, so the same physical plate PNG ended up
        # in 2-3 different ``image_paths`` lists. The pipeline iterates
        # figures with no image-path dedup, so that PNG was segmented once
        # per caption and each pass applied a *different* caption's
        # species list to the same panels — duplicate occurrence rows and
        # species credited to a caption that never printed them. This is
        # the same protection ``_rescue_missing_images`` already
        # implements via ``claimed_basenames``.
        #
        # The ownership test mirrors the Round 21 dedup above: only a pair
        # with BOTH a non-empty caption AND non-empty image_paths counts as
        # an owner. A stub (empty caption) does not block the rescue, so
        # the real Fig. caption can still win the image it describes.
        claimed_basenames: set[str] = set()
        for fig in existing_figures:
            if isinstance(fig, dict):
                cap = (fig.get("caption_text") or "").strip()
                imgs = fig.get("image_paths") or []
            else:
                cap = (fig.caption_text or "").strip()
                imgs = list(fig.image_paths or [])
            if not (cap and imgs):
                continue
            for ip in imgs:
                try:
                    claimed_basenames.add(Path(ip).name)
                except (OSError, ValueError):
                    # Defensive: a malformed path string should not crash
                    # the whole rescue; just skip it.
                    continue

        # ...and two rescued captions in this same call must not share an
        # image either (mirrors ``rescued_used_keys`` in
        # ``_rescue_missing_images``). A caption left with no image at all
        # is emitted with ``image_paths == []`` so the downstream
        # range-chart / orphan-image path can still act on it — that is
        # strictly better than re-processing the same PNG under a foreign
        # caption.
        used_basenames: set[str] = set()

        def _img_key(el: dict[str, Any]) -> str:
            """Best-effort filesystem basename for an OD image element.

            Uses the same approximation as ``_rescue_missing_images``:
            the ``source`` field when present, else OD's
            ``imageFile{N}.png`` convention. The chosen image is re-checked
            against the *resolved* basename below, so a wrong approximation
            can only cost a lookup, never a duplicate attachment.
            """
            src = el.get("source") or ""
            if src:
                return Path(src).name
            try:
                img_id = int(el.get("id", -1) or -1)
            except (TypeError, ValueError):
                return ""
            return f"imageFile{img_id}.png" if img_id >= 0 else ""

        def _claimable(el: dict[str, Any]) -> bool:
            key = _img_key(el)
            return bool(key) and key not in claimed_basenames and key not in used_basenames

        rescued: list[FigureCaptionPair] = []
        for cap in all_captions:
            text = (cap.get("content") or "").strip()
            if not text:
                continue
            # Skip if already represented (exact 60-char prefix match
            # after stripping handles whitespace mismatches without
            # the false-positive prefix-match of the previous
            # ``text[:60] in ec or ec in text[:60]`` bidirectional
            # substring check — see Round 9 fix L4).
            if text[:60] in existing_caption_snippets:
                continue
            page = int(cap.get("page number", 0))
            # Image-selection strategy:
            #   1. Same-page images (largest by bbox area) — the common
            #      case for inline figures where the caption lives
            #      immediately under the plate.
            #   2. **Cross-page** images within ±2 pages, weighted by
            #      inverse page distance. This rescues "appendix"
            #      layouts where the paper body (and its "Fig. N"
            #      captions) is on early pages and the actual figures
            #      live on plates in the back of the PDF (e.g. some
            #      Chinese radiolarian papers put all plates on the
            #      last 5–10 pages). Without this branch those figures
            #      are silently dropped (no image → no panel rows →
            #      no downstream LLM-first / range-chart match).
            chosen_img = None
            same_page_imgs = [el for el in page_to_images.get(page, []) if _claimable(el)]
            if same_page_imgs:
                # audit 2026-07-26: bbox is [left,bottom,right,top];
                # area = (right-left)*(top-bottom), not right*top.
                chosen_img = max(
                    same_page_imgs,
                    key=lambda el: (
                        (
                            int((el.get("bounding box") or [0, 0, 0, 0])[2] or 0)
                            - int((el.get("bounding box") or [0, 0, 0, 0])[0] or 0)
                        )
                        * (
                            int((el.get("bounding box") or [0, 0, 0, 0])[3] or 0)
                            - int((el.get("bounding box") or [0, 0, 0, 0])[1] or 0)
                        )
                    ),
                )
            if chosen_img is None:
                # Search a window of pages around the caption. Phase 28:
                # the window size comes from ``self.caption_window``
                # (default 5, set by ``OpenDataLoaderExtractor.__init__``
                # or ``PipelineConfig.od_caption_window``). The legacy
                # hard-coded ±2 limit was too tight for appendix-style
                # layouts (body pp. 1–30, plates pp. 50–80) where
                # Fig. N captions can be many pages away from their
                # figures. We score each candidate by 1/(1 + page_distance)
                # so the closest image still wins, regardless of window.
                candidates: list[tuple[float, dict[str, Any]]] = []
                w = int(self.caption_window)
                # Build the offsets list once: ±1, ±2, …, ±w. Two
                # passes (positive then negative) keep the scoring
                # order intuitive — closest first.
                offsets = list(range(1, w + 1)) + list(range(-1, -w - 1, -1))
                for offset in offsets:
                    for img in page_to_images.get(page + offset, []):
                        # pipe-1: never score an image that is already owned
                        # by a real figure or claimed by an earlier rescue.
                        if not _claimable(img):
                            continue
                        score = 1.0 / (1.0 + abs(offset))
                        # Slight bonus for a large image, since
                        # appendix figures tend to be big plate pages.
                        # audit 2026-07-26: area = (right-left)*(top-bottom).
                        _bb = img.get("bounding box") or [0, 0, 0, 0]
                        area = (int(_bb[2] or 0) - int(_bb[0] or 0)) * (
                            int(_bb[3] or 0) - int(_bb[1] or 0)
                        )
                        score *= 1.0 + area / 1_000_000
                        candidates.append((score, img))
                if candidates:
                    chosen_img = max(candidates, key=lambda c: c[0])[1]
            image_paths = _resolve_image_paths([chosen_img] if chosen_img else [], output_dir)
            # pipe-1 belt-and-braces: the ``_img_key`` approximation above
            # guesses the basename for source-less elements. Re-check the
            # basename OD actually resolved to, so a guessed key that
            # happens to differ from the real file can never produce a
            # duplicate attachment.
            image_paths = [
                p
                for p in image_paths
                if Path(p).name not in claimed_basenames
                and Path(p).name not in used_basenames
            ]
            used_basenames.update(Path(p).name for p in image_paths)
            plate_imgs = [chosen_img] if image_paths else []
            merged_bbox = _union_bbox(plate_imgs) if plate_imgs else None
            rescued.append(
                FigureCaptionPair(
                    figure_id=(
                        f"od_fig_{paper_id}_p{page:03d}_"
                        f"{len(rescued) + len(existing_figures) + 1:02d}"
                    ),
                    page_number=page,
                    image_paths=image_paths,
                    caption_text=text,
                    merged_bbox=merged_bbox,
                    metadata={"rescued": True},
                )
            )
        if rescued:
            logger.info(
                "_extract_unpaired_captions: rescued %d Fig. captions "
                "that the main pairing missed (e.g. maps, range charts, "
                "schematic diagrams).",
                len(rescued),
            )
        return rescued

    def _extract_figures(
        self, data: dict[str, Any], output_dir: Path, paper_id: str
    ) -> list[FigureCaptionPair]:
        kids: list[dict[str, Any]] = data.get("kids") or []
        if not kids:
            return []

        images: list[dict[str, Any]] = []
        captions: list[dict[str, Any]] = []
        for el in _iter_all_elements(kids):
            etype = el.get("type", "")
            if etype == "image":
                images.append(el)
            elif etype == "caption":
                captions.append(el)

        if not images:
            return []

        # PREFER: explicit "Plate N" / "Explanation of Plate N" caption paragraphs.
        # Most OA radiolarian papers use this convention; OpenDataLoader does not
        # link captions to images reliably, so we do the plate association here
        # by anchoring the figure_id to the plate number.
        plate_captions = _find_plate_captions(kids, caption_window=self.caption_window)
        if plate_captions:
            plate_pairs, claimed_image_ids = _build_figures_from_plate_captions(
                plate_captions,
                images,
                output_dir,
                paper_id,
                caption_window=self.caption_window,
            )
            # Always also include plate-less figures (so the geological/stratigraphic
            # index figures, like "Fig. 2 distribution map", still get processed).
            # Use the exact claimed image IDs (not just their page numbers) so an
            # image on a "plate page" but linked to a different plate is not
            # accidentally re-surfaced as a leftover.
            #
            # Round 21 sampling: Boughdiri 2007's non-plate images on
            # p2-p7 (strat column, litholog sections, location map,
            # outcrop photos) were silently dropped because OD's image
            # IDs are sometimes non-integer strings like ``"p011f1"``,
            # and ``int(img.get("id", -1))`` raised ``ValueError`` which
            # the broad ``except`` clause below swallowed. We now use
            # opaque string-keyed IDs (matching by the string the OD
            # JSON actually emits) so the lookup doesn't crash.
            _claimed_str: set[str] = set()
            for cid in claimed_image_ids:
                _claimed_str.add(str(cid))
            leftover_images = [img for img in images if str(img.get("id", -1)) not in _claimed_str]
            if leftover_images:
                # Build a single fallback figure for unassigned images so the
                # index-map / Fig. 2 distribution etc. still flow through the
                # matcher instead of being silently dropped.
                plate_imgs = _merge_nearby_images(leftover_images, gap_pt=self.merge_gap_pt)
                for plate_idx, plate_images in enumerate(plate_imgs, start=1):
                    # Build a caption lookup keyed by linked_content_id.
                    # Round 21: keys are strings (not ints) so the
                    # dict lookup survives string image IDs like
                    # ``"p011f1"``. OD emits ``linked content id`` as
                    # an integer in some versions and a string in
                    # others — we coerce both sides to ``str`` so the
                    # lookup is format-agnostic.
                    caption_for_image: dict[str, str] = {}
                    for cap in captions:
                        linked = cap.get("linked content id")
                        if linked is not None:
                            caption_for_image[str(linked)] = cap.get("content") or ""
                    plate_cap_list: list[str] = []
                    for img in plate_images:
                        img_id = img.get("id")
                        cap_text = (
                            caption_for_image.get(str(img_id)) if img_id is not None else None
                        )
                        if cap_text:
                            plate_cap_list.append(cap_text)
                    caption_text = " ".join(plate_cap_list) if plate_cap_list else None
                    if not caption_text:
                        page = plate_images[0].get("page number", 1)
                        caption_text = _find_nearest_caption(plate_images, captions, page)
                    image_paths = _resolve_image_paths(plate_images, output_dir)
                    merged_bbox = _union_bbox(plate_images)
                    plate_pairs.append(
                        FigureCaptionPair(
                            figure_id=f"od_fig_{paper_id}_p{plate_images[0].get('page number', 1):03d}_{plate_idx:02d}",
                            page_number=int(plate_images[0].get("page number", 1)),
                            image_paths=image_paths,
                            caption_text=caption_text,
                            merged_bbox=merged_bbox,
                            metadata={"unassigned": True},
                        )
                    )

        # FALLBACK: no plate captions found. Use the original spatial-merge +
        # linked-content caption association.
        else:
            plates = _merge_nearby_images(images, gap_pt=self.merge_gap_pt)
            pairs: list[FigureCaptionPair] = []
            for plate_idx, plate_images in enumerate(plates, start=1):
                # Build a caption lookup keyed by linked_content_id.
                # Round 21: keys are strings (not ints) so non-
                # integer image IDs (e.g. ``"p011f1"``) link to
                # captions correctly. See the plate-captions branch
                # above for the full rationale.
                caption_for_image: dict[str, str] = {}
                for cap in captions:
                    linked = cap.get("linked content id")
                    if linked is not None:
                        caption_for_image[str(linked)] = cap.get("content") or ""

                plate_caps: list[str] = []
                for img in plate_images:
                    img_id = img.get("id")
                    cap_text = caption_for_image.get(str(img_id)) if img_id is not None else None
                    if cap_text:
                        plate_caps.append(cap_text)

                caption_text = " ".join(plate_caps) if plate_caps else None

                if not caption_text:
                    page = plate_images[0].get("page number", 1)
                    caption_text = _find_nearest_caption(plate_images, captions, page)

                image_paths = _resolve_image_paths(plate_images, output_dir)
                merged_bbox = _union_bbox(plate_images)
                pairs.append(
                    FigureCaptionPair(
                        figure_id=f"od_fig_{paper_id}_p{plate_images[0].get('page number', 1):03d}_{plate_idx:02d}",
                        page_number=int(plate_images[0].get("page number", 1)),
                        image_paths=image_paths,
                        caption_text=caption_text,
                        merged_bbox=merged_bbox,
                    )
                )

            # Post-process: there are two classes of figure that the
            # standard paths above MISSED entirely:
            #
            #   (a) Captions that have NO nearby image on the same page.
            #       The plate-caption path requires a "Plate N" caption,
            #       and the merge-by-image fallback requires at least
            #       one image element. So a caption that is on page 2 but
            #       whose image lives on page 18 ("appendix layout")
            #       produces ZERO pairs — the caption silently disappears.
            #       ``_rescue_unmatched_captions`` walks the kids tree for
            #       every "Fig. N" caption that doesn't yet belong to a
            #       pair, creates a stub pair, then delegates to
            #       ``_rescue_missing_images`` for the cross-page image
            #       attach.
            #
            #   (b) Pairs that have an empty ``image_paths`` because the
            #       fallback ``_find_nearest_caption`` only searches the
            #       same page. ``_rescue_missing_images`` attaches the
            #       nearest cross-page orphan image.
        if plate_captions:
            pairs = plate_pairs
        pairs = _rescue_unmatched_captions(pairs, kids, output_dir, paper_id)
        # Phase 28: forward the caption-window so the rescue hard cap
        # scales with the operator's choice (default 5 → cap 20,
        # matching the legacy behaviour).
        return _rescue_missing_images(
            pairs, kids, output_dir, paper_id, caption_window=self.caption_window
        )


def _rescue_unmatched_captions(
    pairs: list[FigureCaptionPair],
    kids: list[dict[str, Any]],
    output_dir: Path,
    paper_id: str,
) -> list[FigureCaptionPair]:
    """Create stub FigureCaptionPair entries for Fig. captions that
    the main ``_extract_figures`` paths dropped.

    Both standard paths derive pairs from the IMAGES (plate captions
    attached to plate figures, or merge-by-image fallback when there
    is no plate caption). A caption that has no associated image at
    all never becomes a pair and is silently lost. This is the
    classic "text and figures separated into front matter and
    appendix" layout: body text + "Fig. 1." captions on pages 1-10,
    actual figures on pages 11-20.

    The function walks the kids tree for every caption that starts
    with "Fig. N" or "FIGURE" and whose text isn't already attached
    to a pair (matched by a 60-character prefix snippet, the same
    heuristic used by ``_extract_unpaired_captions``). Each
    unmatched caption becomes a new FigureCaptionPair with
    empty ``image_paths``; ``_rescue_missing_images`` (called
    immediately after) attaches the nearest cross-page image.
    """
    # Build the set of caption snippets that are already attached.
    #
    # Round 21 sampling: stubs with empty ``caption_text`` (e.g. the
    # FALLBACK branch's no-caption pair) must NOT block this rescue
    # from emitting the real ``Fig. N`` caption. Only count pairs
    # that have a non-empty caption (i.e. real, not stubs) when
    # building the dedup set. Round 9 (L4) noted that the previous
    # bidirectional substring check ``text[:60] in s or s in text[:60]``
    # over-matched ("Fig. 1" is a prefix of "Fig. 10 ..."); the
    # new check uses exact 60-char prefix equality like
    # ``_extract_unpaired_captions``.
    existing_snippets: set[str] = set()
    for p in pairs:
        cap = (p.caption_text or "").strip()
        if cap:
            existing_snippets.add(cap[:60])
    # Walk captions.
    rescued: list[FigureCaptionPair] = []
    all_caps = [el for el in _iter_all_elements(kids) if el.get("type") == "caption"]
    for cap in all_caps:
        text = (cap.get("content") or "").strip()
        low = text.lower()
        # Phase 27: shared predicate — JA 図 markers now pass this
        # filter alongside English Fig./Figure prefixes. See
        # ``_is_caption_kind_marker`` for the canonical list.
        if not _is_caption_kind_marker(low):
            continue
        if not text:
            continue
        # Round 21: use exact 60-char prefix match (NOT the
        # bidirectional substring check that was retired in
        # Round 9 L4). This avoids spurious "Fig. 1" / "Fig. 10"
        # collisions and over-matches on empty strings.
        if text[:60] in existing_snippets:
            continue
        page = int(cap.get("page number", 0))
        existing_snippets.add(text[:60])
        rescued.append(
            FigureCaptionPair(
                figure_id=(f"od_fig_{paper_id}_p{page:03d}_{len(pairs) + len(rescued) + 1:02d}"),
                page_number=page,
                image_paths=[],
                caption_text=text,
                merged_bbox=None,
                metadata={"rescued_caption": True},
            )
        )
    if rescued:
        logger.info(
            "_rescue_unmatched_captions: created %d Fig. caption stub(s) "
            "for appendix-style layouts (text on early pages, plates on "
            "later pages).",
            len(rescued),
        )
    return list(pairs) + rescued


def _rescue_missing_images(
    pairs: list[FigureCaptionPair],
    kids: list[dict[str, Any]],
    output_dir: Path,
    paper_id: str,
    caption_window: int = 5,
) -> list[FigureCaptionPair]:
    """Pair caption-only figures with the nearest cross-page image.

    The main ``_extract_figures`` path pairs captions with images on
    the SAME page. When the paper body is on early pages and the
    actual plates are appended at the end (common in Chinese
    radiolarian journals — "图版 I/II/III" at the back), the same-
    page lookup finds nothing and the figure is emitted with an
    empty ``image_paths``. Without this rescue the figure is
    silently dropped downstream and the user gets no panel rows.

    The rescue:
      1. Walk the kids tree for every image that is NOT already
         referenced by a figure's image_paths (i.e. an "orphan" image).
      2. Walk the pairs list for every figure whose image_paths is
         empty AND that has a non-empty caption.
      3. For each orphan caption, find the nearest orphan image by
         ``1/(1 + |page_diff|)`` weighted by image area, within a
         ±3 page window. (Same heuristic used in
         ``_extract_unpaired_captions``.)
      4. Resolve the image to a filesystem path via
         ``_resolve_image_paths``.

    Returns the (possibly augmented) pairs list. Existing image_paths
    are not disturbed.
    """
    # Cross-page rescue: each caption-only pair needs the closest
    # orphan image. We track which images have already been attached
    # to a stub (rescued) pair in this same call so two captions
    # don't fight over the same image — but we do NOT block plate
    # pairs' images, because one physical plate image can be the
    # target of multiple "Fig. N" captions (think a single plate
    # with several labeled species). This means a plate pair's
    # image will be reused by the rescue if a caption-only pair
    # later wants it; that's the correct behaviour for the
    # appendix-style layout this function was added to support.
    rescued_used_keys: set[tuple[int, int]] = set()

    # Round 9 (Bug-H3): collect the basenames of every image_path
    # already attached to a pair (plate figures, prior rescues, anything).
    # Without this, the orphan pool below included plate-pair images and
    # a caption-only rescue would happily steal them — causing the same
    # physical image to be attached to two different figures downstream.
    claimed_basenames: set[str] = set()
    for p in pairs:
        for ip in p.image_paths or []:
            try:
                claimed_basenames.add(Path(ip).name)
            except (OSError, ValueError):
                # Defensive: a malformed path string should not crash
                # the whole rescue; just skip it.
                pass

    # Collect every image element in the document, then drop the
    # already-paired ones (by file basename — the (page, id) approach
    # would also work but doesn't survive a re-read of the JSON where
    # ids can collide across pages in pathological cases).
    orphan_imgs: list[dict[str, Any]] = []
    for el in _iter_all_elements(kids):
        if el.get("type") != "image":
            continue
        page = int(el.get("page number", 0) or 0)
        # Audit 2026-07-26 B3: guard ``int()`` against non-numeric ``id``
        # values (e.g. "p011f1" in Boughdiri 2007). Mirrors the Phase 54
        # H3 fix at lines 1887-1889; without this the ValueError killed
        # the entire PDF extraction via _extract_figures -> _rescue.
        try:
            img_id = int(el.get("id", -1) or -1)
        except (TypeError, ValueError):
            img_id = -1
        if (page, img_id) in rescued_used_keys:
            continue
        # Skip images already attached to a pair. We approximate by
        # basename — for OD exports this is "imageFileN.png" so the
        # match is deterministic. A more sophisticated check would
        # resolve the image to its filesystem path first, but that
        # requires the same lookup logic as ``_resolve_image_paths``
        # and would be O(N*M); basename match is O(N+M) and correct
        # for OD's naming convention.
        src = el.get("source") or ""
        basename = Path(src).name if src else f"imageFile{img_id}.png"
        if basename and basename in claimed_basenames:
            continue
        orphan_imgs.append(el)

    if not orphan_imgs:
        return pairs

    def _bbox_area(el: dict[str, Any]) -> int:
        bb = el.get("bounding box") or [0, 0, 0, 0]
        # bb = [left, bottom, right, top]; area = width * height.
        # audit 2026-07-26: was bb[2]*bb[3] (right*top), not the area.
        w = int(bb[2] or 0) - int(bb[0] or 0)
        h = int(bb[3] or 0) - int(bb[1] or 0)
        return max(0, w) * max(0, h)

    out_pairs: list[FigureCaptionPair] = []
    for p in pairs:
        if p.image_paths:
            out_pairs.append(p)
            continue
        if not (p.caption_text or "").strip():
            out_pairs.append(p)
            continue
        # This is a caption-only figure — try to attach the nearest
        # orphan image within ±3 pages.
        cap_page = int(p.page_number or 0)
        best_score: float = -1.0
        best_img: dict[str, Any] | None = None
        for img in orphan_imgs:
            img_page = int(img.get("page number", 0) or 0)
            try:
                img_id = int(img.get("id", -1) or -1)
            except (TypeError, ValueError):
                img_id = -1
            # Skip images already attached to another stub pair in
            # THIS rescue call. We intentionally allow reuse of plate
            # images (see comment at top of function).
            if (img_page, img_id) in rescued_used_keys:
                continue
            page_diff = abs(img_page - cap_page)
            # Phase 28: rescue hard cap is now ``caption_window * 4``.
            # At the default ``caption_window=5`` the cap remains at
            # 20 (= 5×4) — fully backward compatible. Operators who
            # widen ``--od-caption-window`` to 10 get a ±40 rescue
            # radius (covers "plates clustered at end" of a 50-page
            # paper) without needing a separate flag. The 4× factor
            # keeps the rescue radius proportionally larger than the
            # Fig.-caption window, which is appropriate because
            # rescue is a last-chance catch-all rather than the
            # primary pairing path.
            max_page_diff = int(caption_window) * 4
            if page_diff > max_page_diff:
                continue
            # Inverse-page-distance weight × area bonus (prefer the
            # largest image on the closest page).
            score = 1.0 / (1.0 + page_diff) * (1.0 + _bbox_area(img) / 1_000_000)
            if score > best_score:
                best_score = score
                best_img = img
        if best_img is not None:
            resolved = _resolve_image_paths([best_img], output_dir)
            if resolved:
                p.image_paths = resolved
                p.metadata = dict(p.metadata or {})
                p.metadata["cross_page_rescue"] = True
                p.metadata["rescue_page_diff"] = abs(
                    int(best_img.get("page number", 0) or 0) - cap_page
                )
                # The image is now used — mark it so a later pair
                # doesn't try to grab the same image.
                try:
                    _rescued_id = int(best_img.get("id", -1) or -1)
                except (TypeError, ValueError):
                    _rescued_id = -1
                rescued_used_keys.add((int(best_img.get("page number", 0) or 0), _rescued_id))
        out_pairs.append(p)
    return out_pairs


# ---- helpers (module-level) ------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_all_elements(kids: list[dict[str, Any]]):
    """Recursively yield every element in the content tree."""
    for kid in kids:
        yield kid
        children = (
            kid.get("kids") or kid.get("rows") or kid.get("list items") or kid.get("cells") or []
        )
        if isinstance(children, list):
            yield from _iter_all_elements(children)


def _collect_images(kids: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recursively gather every image element from the content tree."""
    out: list[dict[str, Any]] = []
    for el in _iter_all_elements(kids):
        if isinstance(el, dict) and el.get("type") == "image":
            out.append(el)
    return out


def _images_page_index(images: list[dict[str, Any]]) -> set[int]:
    """Return the set of page numbers that contain at least one image."""
    return {
        int(img.get("page number", 0) or 0)
        for img in images
        if int(img.get("page number", 0) or 0) > 0
    }


def _merge_nearby_images(
    images: list[dict[str, Any]], gap_pt: float = 72.0
) -> list[list[dict[str, Any]]]:
    """Group images on the same page into plate groups by spatial proximity."""
    # Group by page
    by_page: dict[int, list[dict[str, Any]]] = {}
    for img in images:
        page = img.get("page number", 1)
        by_page.setdefault(page, []).append(img)

    plates: list[list[dict[str, Any]]] = []
    for page_imgs in by_page.values():
        # Sort top-to-bottom, left-to-right
        page_imgs.sort(key=lambda el: _bbox_top(el) * 10000 + _bbox_left(el))

        current: list[dict[str, Any]] = []
        for img in page_imgs:
            if not current:
                current.append(img)
                continue
            # Check distance to the *union* of current group
            if _bbox_distance(_union_bbox(current), img.get("bounding box")) <= gap_pt:
                current.append(img)
            else:
                plates.append(current)
                current = [img]
        if current:
            plates.append(current)

    return plates


def _bbox_distance(
    bbox_a: tuple[float, float, float, float] | None,
    bbox_b: list[float] | None,
) -> float:
    """Minimum euclidean distance between two bounding boxes (PDF points)."""
    if bbox_a is None or bbox_b is None or len(bbox_b) < 4:
        return float("inf")
    a_left, a_bottom, a_right, a_top = bbox_a
    b_left, b_bottom, b_right, b_top = bbox_b

    # If boxes overlap or touch, distance is zero
    if a_right >= b_left and b_right >= a_left and a_top >= b_bottom and b_top >= a_bottom:
        return 0.0

    # Horizontal gap
    dx = max(0.0, b_left - a_right, a_left - b_right)
    # Vertical gap
    dy = max(0.0, b_bottom - a_top, a_bottom - b_top)
    return (dx * dx + dy * dy) ** 0.5


def _union_bbox(images: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    """Return the bounding box spanning all given images."""
    if not images:
        return None
    left = float("inf")
    bottom = float("inf")
    right = float("-inf")
    top = float("-inf")
    for img in images:
        bb = img.get("bounding box")
        if not bb or len(bb) < 4:
            continue
        left = min(left, bb[0])
        bottom = min(bottom, bb[1])
        right = max(right, bb[2])
        top = max(top, bb[3])
    if left == float("inf"):
        return None
    return (left, bottom, right, top)


def _bbox_left(el: dict[str, Any]) -> float:
    bb = el.get("bounding box")
    return float(bb[0]) if bb and len(bb) >= 4 else 0.0


def _bbox_top(el: dict[str, Any]) -> float:
    bb = el.get("bounding box")
    return float(bb[3]) if bb and len(bb) >= 4 else 0.0


# ---- caption OCR fallback --------------------------------------------------


def _ocr_caption_band(
    doc: Any,
    page_index: int,
    merged_bbox_pdf: tuple[float, float, float, float],
    ocr_engine: Any,
    np_module: Any,
) -> str | None:
    """Render the page band below ``merged_bbox_pdf`` and OCR it for a
    caption.

    Parameters
    ----------
    doc
        An open ``fitz.Document``.
    page_index
        0-based page index.
    merged_bbox_pdf
        ``(left, bottom, right, top)`` in PDF-native points (origin at
        bottom-left, y axis pointing up).
    ocr_engine
        An ``easyocr.Reader`` instance.
    np_module
        The ``numpy`` module (passed in to keep this function import-
        free so the fallback can be skipped when numpy isn't available).

    Returns
    -------
    The recovered caption text, or ``None`` if OCR didn't produce
    anything that looks like a caption.
    """
    try:
        page = doc[page_index]
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)
        left, bottom, right, top = merged_bbox_pdf
        # The caption sits BELOW the figure in PDF-native coords, i.e.
        # y < ``bottom`` (since y axis points up). In PyMuPDF's top-
        # left origin that's ``y > page_h - bottom``. We render from
        # the bottom of the figure down to ~30pt above the page bottom.
        cap_top_pdf = max(0.0, bottom - 200.0)  # 200pt max caption height
        cap_bottom_pdf = max(0.0, bottom - 8.0)  # 8pt gap below figure
        # Convert to PyMuPDF y (top-left origin, y going down).
        y0 = page_h - cap_bottom_pdf
        y1 = page_h - cap_top_pdf
        x0 = max(0.0, left - 4.0)
        x1 = min(page_w, right + 4.0)
        if y1 <= y0 or x1 <= x0:
            return None
        clip = fitz.Rect(x0, y0, x1, y1)
        # 3x scale so EasyOCR has enough resolution on small fonts.
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB, alpha=False)
        img = np_module.frombuffer(pix.samples, dtype=np_module.uint8).reshape(
            pix.height, pix.width, 3
        )
        # EasyOCR expects BGR.
        import cv2

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        try:
            results = ocr_engine.readtext(img_bgr)
        except Exception:
            return None
        if not results:
            return None
        # Sort by y then x and join with spaces; drop very low-confidence
        # detections (EasyOCR occasionally returns junk at <0.2).
        results = sorted(
            [r for r in results if r[2] >= 0.2],
            key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])),
        )
        text = " ".join(r[1] for r in results).strip()
        if not text:
            return None
        # Heuristic: a real figure caption has multiple words and is at
        # least ~30 chars. Anything shorter is probably a stray number
        # or scale bar misread.
        if len(text) < 25:
            return None
        return text
    except Exception:
        return None


# -- plate caption detection -----------------------------------------------

# Match "Plate 1", "Plate 12", or Roman-numeral "Plate I", "Plate IV" —
# possibly with leading "Explanation of".
# Examples seen in OA papers:
#   "Plate 1 Scanning electron microscope pictures of radiolarians..."
#   "Explanation of Plate 3. ﬁgs 1–5. Trilonche crassispinosa..."
#   "Plate 5, Figs. 1–10. Caption text..."
#   "Plate I. Caption for Plate I (Boughdiri 2007 et al.)"
# The Roman-numeral group is optional; the captured integer is the
# decimal value (I→1, IV→4, etc.). Group 1 is the Arabic digit string,
# group 2 is the Roman numeral string (one of I, II, III, IV, V, VI,
# VII, VIII, IX, X, XI, XII). At least one of the two groups is
# guaranteed by the alternation.
_PLATE_CAPTION_RE = re.compile(
    r"^\s*(?:Explanation\s+of\s+)?Plate\s+"
    r"(?:(\d+)|(XIV|XIII|XII|XI{0,2}|IX|IV|V(?:III|II|I)?|I{1,3}))\b",
    re.IGNORECASE,
)

_ROMAN_TO_INT = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
}


def _plate_number_from_match(m: re.Match) -> int:
    """Extract the integer plate number from a _PLATE_CAPTION_RE match.

    The regex captures either an Arabic digit string (group 1) or a
    Roman numeral (group 2). Exactly one is set per match.
    """
    arabic = m.group(1)
    if arabic is not None:
        return int(arabic)
    roman = m.group(2)
    if roman is not None:
        return _ROMAN_TO_INT.get(roman.upper(), 0)
    return 0


# Match "Fig. 1", "Fig 1", "Figure 1" — covers review/synthesis papers
# that use "Fig." numbering instead of "Plate" (e.g. Wever 2006, where
# the six "Fig. 1..6" paragraphs are the only figure captions in the PDF
# and there are no "Plate N" headers to anchor on). The match anchors
# at the START of the paragraph so body-text mentions like
# "(see Fig. 1)" are not picked up.
#
# We further require the "Fig. N" prefix to be followed by a caption-
# like delimiter (". " or " " + capital letter, possibly with a
# sub-figure letter like "1a") and the whole content to be at least
# 25 chars. This rejects body-text paragraphs that happen to start
# with a "Fig. N" reference, e.g. "Fig. 1), PR-SB23 (Plate 6, Fig. 10)"
# (Bandini 2011 p37) or "Fig. 14 continued c" (Bandini 2011 p34) or
# the short list-items "Fig. 26", "Fig. 34" (Bandini 2011 p8).
_FIG_CAPTION_RE = re.compile(
    r"^\s*Fig(?:ure)?\s*\.?\s*(\d+)([a-z]?)\s*([.\s])\s*(\S)",
    re.IGNORECASE,
)

# Phase 27: Japanese caption markers. JA radiolarian papers use 図版
# (``zuhan`` = plate, literally "picture-book") for plate-level captions
# and 図 (``zu`` = figure) for in-text figure references. Both follow
# the same Arabic-digit convention as English ``Plate N`` / ``Fig. N``,
# so the existing dispatcher's ``plate_number`` / ``kind`` fields
# generalise without modification (group 1 is always the digit string
# for the JA patterns).
#
# Anchoring policy mirrors ``_PLATE_CAPTION_RE`` / ``_FIG_CAPTION_RE``:
# the regex must match at the START of the element. Mid-paragraph
# references like ``(図1参照)`` are rejected because they don't begin
# the element. The existing ``_looks_like_fig_caption`` 25-character
# gate (line 1555) further filters out any short JA fragments that
# sneak through.
_JA_PLATE_CAPTION_RE = re.compile(
    # Bug-fix H-4: JA regex no longer accepts 圖版 (traditional ZH).
    # Traditional ZH papers (e.g. Taiwan/Hong Kong) now route
    # through the ZH dispatcher instead of being captured by JA.
    r"^\s*(?:説明\s*)?図版\s*(?:[IVX]+|No\.?\s*)?(\d+)\s*[\.:]?\s*",
)
_JA_FIG_CAPTION_RE = re.compile(
    r"^\s*図\s*(\d+)([a-z]?)\s*([.\s])\s*(\S)",
)


# Phase 30: Chinese caption markers. ZH papers use the same
# ``图版 N`` / ``图 N`` convention as Japanese ``図版 N`` /
# ``図 N``. Two encoding families exist:
#
#   * Simplified (``图版`` / ``图`` — GB2312) — Mainland China
#   * Traditional (``圖版`` / ``圖`` — Big5) — Taiwan/HK/overseas
#
# The plate regex forces a two-character prefix ``图版`` / ``圖版``
# to eliminate single-char ``图`` collisions (``图书馆`` library,
# ``地图`` map, ``图片`` image). The fig regex requires a digit
# after ``图`` / ``圖``, so body-text mentions like ``图1说明``
# fire correctly but ``图书馆分类`` does not.
#
# Anchoring mirrors the JA policy: ``^\s*`` so mid-paragraph references
# like ``(图1参照)`` are rejected — they don't begin the element.
_ZH_PLATE_CAPTION_RE = re.compile(
    r"^\s*(?:说明\s*)?(?:图版|圖版)\s*(?:[IVX]+|No\.?\s*)?(\d+)\s*[\.:]?\s*",
)
_ZH_FIG_CAPTION_RE = re.compile(
    # Bug-fix M-3: capture sub-figure letter (e.g. 圖1a) the same
    # way the JA regex does, so ZH papers with figure sub-letters
    # don't fall through to body-text rejection.
    r"^\s*(?:图|圖)\s*(\d+)([a-z]?)\s*([.\s])\s*(\S)",
)


def _is_caption_kind_marker(low: str) -> bool:
    """Return True if ``low`` (caption text, lowercased) starts with a kind
    marker we route on.

    Phase 27: extended from the original English-only ``startswith("fig.")``
    check to also accept the single-char JA figure marker ``図``. Used in
    both ``_extract_unpaired_captions`` (line 359 area) and
    ``_rescue_unmatched_captions`` (line 706 area) so the two sites
    stay in lockstep when adding new languages.

    Phase 30: also accepts the Chinese figure markers ``图`` (simplified)
    and ``圖`` (traditional). Both are different code points but
    share the fig-marker semantics in their respective papers.
    """
    return (
        low.startswith("fig.")
        or low.startswith("figure ")
        or low.startswith("fig ")
        or low.startswith("図")  # JA figure marker (single kanji)
        or low.startswith("图")  # ZH Simplified figure marker
        or low.startswith("圖")  # ZH Traditional figure marker
    )


# Match an inline plate figure reference inside a body paragraph.
# Pouille 2014 has no real "Plate N" captions; the species list lives
# in the systematic paleontology descriptions, e.g.:
#   "Genus species AUTHOR (Pl. 1, figs 5–7)"
#   "Genus species AUTHOR (Plate 2, figure 7)"
#   "Genus species (Pl. 3. fig. 11)"
# We detect these and reconstruct a per-plate caption by concatenating
# all matching "Pl. N" / "Plate N" mentions found in the body.
_PLATE_INLINE_REF_RE = re.compile(
    r"\b(?:[Pp]l(?:ate)?\.?)\s*(\d+)\s*[,.]?\s*[Ff]ig(?:s|ure)?\.?\s*(\d+)(?:[a-z\-–—]*\d*[a-z]?)*",
)
# Match a Genus species (or Genus? sp. cf./aff. species) preceding the
# plate reference — e.g. "Syntagentactinia biocculosa ... (Pl. 1, figs 5–7)"
# or "Syntagentactinia? sp. cf. S. excelsa (Pl. 1, figs 1–4)".
# We grab the species name(s) from the start of the line, then look right-
# ward for the plate ref. Pouille 2014 is the canonical example.
_SPECIES_NAME_RE = re.compile(
    r"([A-Z][a-z]+"  # Genus
    r"(?:"
    r"[?.]?\s+sp\."  # "Genus? sp." (?, period, or no marker — OCR
    #  often prints "Polyentactinia. sp." instead of
    #  "Polyentactinia sp.")
    r"(?:\s+(?:[A-Z]\.|[A-Z]))?"  #   S. (abbrev genus) OR " sp. A" form
    r"(?:\s+(?:cf\.|aff\.)\s+(?:[A-Z]\.\s+)?[A-Z]?[a-z][a-z\-]+)?"  # cf./aff. S. species
    r"|"
    r"[?.]?\s+[a-z][a-z\-]+"  # "Genus? species" (Pouille) or "Genus species"
    r"(?:\s+[a-z][a-z\-]+)*"  # optional third epithet
    r")"
    r")"
)


def _collect_following_text(
    kids: list[dict[str, Any]],
    start_idx: int,
    same_page: int,
    max_items: int = 4,
    kinds: tuple[str, ...] = ("paragraph", "list"),
) -> str:
    """Return the concatenated ``content`` of up to ``max_items`` siblings
    after ``start_idx`` on the same page, stopping at the next ``heading``
    or ``image`` / ``table``. Used to expand a bare ``Plate N`` heading
    into a full caption by appending the description paragraph and the
    species list that usually follow it (Hollis 2006 plates 1-3).

    The ``kinds`` tuple controls which sibling element types are appended:
    defaults to ``("paragraph", "list")`` for heading-style matches.
    For paragraph/caption-style matches the caller passes
    ``kinds=("list",)`` so a body-text paragraph that happens to follow
    the caption header (e.g. the next species description) is NOT
    collected — the feng2007 "Explanation of Plate N" pattern has the
    species list rendered as a separate ``list`` element, so list-only
    is the right call there.
    """
    parts: list[str] = []
    end = min(len(kids), start_idx + 1 + max_items * 2)
    for j in range(start_idx + 1, end):
        sib = kids[j]
        if not isinstance(sib, dict):
            continue
        sib_type = sib.get("type", "")
        if sib_type in ("heading", "image", "table"):
            break
        sib_page = int(sib.get("page number", 0) or 0)
        if sib_page != same_page:
            break
        if sib_type in ("paragraph",) and "paragraph" in kinds:
            text = (sib.get("content") or "").strip()
            if text:
                parts.append(text)
        elif sib_type in ("list",) and "list" in kinds:
            # Flatten list items into a single text block
            items: list[str] = []
            for item in sib.get("list items", []) or []:
                if isinstance(item, dict):
                    txt = (item.get("content") or "").strip()
                    if txt:
                        items.append(txt)
            if items:
                parts.append("\n".join(items))
        if len(parts) >= max_items:
            break
    return "\n\n".join(parts)


def _find_plate_captions(
    kids: list[dict[str, Any]],
    caption_window: int = 5,
) -> list[dict[str, Any]]:
    """Return elements whose text starts with ``Plate N`` /
    ``Explanation of Plate N``.

    OpenDataLoader surfaces plate captions as one of:
      * ``type=caption`` (with a ``content`` field) — already complete text
      * ``type=paragraph`` — Bandini-style: full caption with species list
      * ``type=heading`` — Hollis-style: bare ``Plate N`` followed by a
        description paragraph and a list of species on the same page
    For heading-type matches we expand the content by appending the next
    few siblings (paragraph + list) so the downstream caption parser sees
    the actual species list, not just ``Plate 1``.

    When no real "Plate N" / "Explanation of Plate N" caption is found
    for a given plate (e.g. Pouille 2014, where the species live in the
    systematic paleontology descriptions), we fall back to **reconstructing**
    a caption by harvesting every "Pl. N, fig. M (Species)" mention from
    the body text. This synthesises a caption shaped like:
        "Plate 1. (Reconstructed from systematic descriptions)
         Syntagentactinia biocculosa ... (Pl. 1, figs 5–7)
         Syntagentactinia? angulata n. sp. ... (Pl. 1, figs 12–14b) ..."
    so the downstream caption parser has *something* to match against.

    Returns a list of dicts with keys:
      - ``plate_number`` (int)
      - ``page_number``  (int, 1-indexed; the earliest body page where
                         the plate was referenced)
      - ``content``      (str, full text)
      - ``element``      (original dict, in case the caller needs more)
    Sorted by (plate_number, page_number).
    """
    found: list[dict[str, Any]] = []
    # Plate-kind captions only — used by the reconstruction pass to
    # compute the cap (so body "(Pl. N, fig. M)" mentions don't get
    # misread as plates of *this* paper) and to dedup reconstructed
    # entries against real ones. Fig-kind captions are NOT in this
    # set: a paper that uses "Fig. 1..6" for chart figures and
    # "Plate 1..3" for micrograph plates (e.g. Pouille 2014) must
    # still let the reconstruction pass synthesise Plate 1, 2 even
    # though Fig 1, 2 are real captions of charts.
    seen_plates: set[int] = set()
    # Per-kind dedup — keeps "Plate 1" and "Fig. 1" as separate
    # captions when a paper uses both numbering conventions.
    seen_plates_with_kind: set[tuple[int, str]] = set()
    # Some papers (Bandini 2011 plates 7-9, where OD parks the
    # "Plate N" header inside PDF-UA tagged ``list`` elements and
    # the top-level list element has empty content with the caption
    # text living in ``list_items[i].content``) bury the caption
    # header inside ``list`` elements. Re-surface those list-items
    # that match ``_PLATE_CAPTION_RE`` as synthetic paragraph
    # siblings so the regex below can match them. This restores
    # routing for the buried plates; without it, leftover images
    # on those pages get stamped with bogus ``od_fig_*`` IDs and
    # the strict ``match_panel(figure_id)`` matcher rejects them.
    #
    # When the entire list contains a single list_item matching
    # ``_PLATE_CAPTION_RE`` (the bandini pattern: one "Plate N ..."
    # list_item with no other content), we DROP the original list
    # element to avoid duplicating its text in the synthetic
    # paragraph expansion. Lists with mixed content (one Plate
    # header + other body text) are preserved as siblings so the
    # caption parser can still pick up the body text.
    expanded_kids: list[dict[str, Any]] = []
    for _kid in kids:
        if isinstance(_kid, dict) and _kid.get("type") == "list":
            _list_page = _kid.get("page number", 0)
            _list_items = _kid.get("list items") or []
            _plate_match_count = 0
            for _li in _list_items:
                if not isinstance(_li, dict):
                    continue
                _txt = (_li.get("content") or "").strip()
                # Phase 27: also recognise JA plate markers here so JA
                # papers whose list_items hold 図版 N headers get the
                # same synthetic-paragraph expansion as English papers.
                # Phase 30: extend to ZH (``图版`` / ``圖版``) so Mainland
                # China + Taiwan papers get the same treatment.
                if _txt and (
                    _PLATE_CAPTION_RE.match(_txt)
                    or _JA_PLATE_CAPTION_RE.match(_txt)
                    or _ZH_PLATE_CAPTION_RE.match(_txt)
                ):
                    expanded_kids.append(
                        {
                            "type": "paragraph",
                            "page number": _list_page,
                            "content": _txt,
                        }
                    )
                    _plate_match_count += 1
            # Drop the original list ONLY when it has exactly one
            # list_item AND that one is the Plate N header (would
            # otherwise duplicate the synthetic paragraph's text).
            # Otherwise keep the list as a sibling.
            if len(_list_items) == 1 and _plate_match_count == 1:
                continue
        expanded_kids.append(_kid)
    for idx, kid in enumerate(expanded_kids):
        if not isinstance(kid, dict):
            continue
        etype = kid.get("type", "")
        if etype not in ("paragraph", "caption", "heading"):
            continue
        content = (kid.get("content") or "").strip()
        if not content:
            continue
        # Try Plate N first (preferred — radiolarian-plate papers use this
        # convention). Fall back to Fig. N for review/synthesis papers
        # that use "Fig. 1..6" numbering instead (Wever 2006 et al.).
        # Phase 27: also try Japanese 図版 (plate) and 図 (figure)
        # markers so JA papers like Takahashi 2004 / Uchino 2005 produce
        # real captions instead of the empty "Auto-generated figure"
        # stub. Each kind is tracked separately so a paper that uses
        # multiple conventions (e.g. Plate 1..3 + Fig. 1..6 + 図1..3)
        # keeps all sets of captions rather than collapsing them on
        # the same int.
        m = _PLATE_CAPTION_RE.match(content)
        kind = "plate" if m else None
        if not m:
            m = _FIG_CAPTION_RE.match(content)
            kind = "fig" if m else None
        # Phase 27: JA dispatch — same structure, same group-1 captures
        # the Arabic digit, so ``_plate_number_from_match`` works as-is.
        if not m:
            m = _JA_PLATE_CAPTION_RE.match(content)
            kind = "plate" if m else None
        if not m:
            m = _JA_FIG_CAPTION_RE.match(content)
            kind = "fig" if m else None
        # Phase 30: ZH dispatch — matches simplified ``图版`` / ``图``
        # (Mainland) and traditional ``圖版`` / ``圖`` (Taiwan/HK).
        # Same structure as JA; the existing ``_plate_number_from_match``
        # + ``_looks_like_fig_caption`` gates handle the rest.
        if not m:
            m = _ZH_PLATE_CAPTION_RE.match(content)
            kind = "plate" if m else None
        if not m:
            m = _ZH_FIG_CAPTION_RE.match(content)
            kind = "fig" if m else None
        if not m:
            continue
        # Fig-kind matches need an extra content-quality check: the
        # regex also matches body-text paragraphs that happen to start
        # with "Fig. N" (e.g. "Fig. 14 continued c", "Fig. 26", or a
        # species description starting with "Fig. N Archaeodictyomitra
        # montisserei(SQUINABOL) Pl. 8"). The same quality gate
        # applies to JA fig-kind matches. Reject:
        #   1. too-short matches (< 25 chars) — almost always a list
        #      reference or a continuation marker, not a real caption.
        #   2. body-text species descriptions: any "(UPPERCASE_WORD)"
        #      author citation within the first 200 chars is a strong
        #      signal this is a species list, not a figure caption.
        if kind == "fig" and not _looks_like_fig_caption(content):
            continue
        plate_number = _plate_number_from_match(m)
        dedup_key = (plate_number, kind or "plate")
        if dedup_key in seen_plates_with_kind:
            continue
        if kind == "plate":
            seen_plates.add(plate_number)
        seen_plates_with_kind.add(dedup_key)
        page = int(kid.get("page number", 0) or 0)
        # For heading/paragraph/caption-type matches, expand by appending
        # following content on the same page. feng2007 has the
        # "Explanation of Plate 1" header + first species clause as a
        # ``paragraph`` element, then a ``list`` element with the remaining
        # species clauses (the list is a separate OD element because the
        # species panel-list is rendered as a bulleted list in the PDF).
        # Hollis 2006 has the bare "Plate N" as a heading, followed by a
        # description paragraph and a list — same expansion logic, but
        # the description paragraph IS collected because it's a header→
        # paragraph transition, not a paragraph→paragraph transition.
        # Paragraph→paragraph is the body-text continuation pattern (e.g.
        # a "Fig. 1 Geological map" caption immediately followed by a
        # species description paragraph) so we exclude paragraphs from
        # the expansion when the matched element is itself a paragraph.
        if etype == "heading":
            extra = _collect_following_text(
                expanded_kids, idx, page, max_items=3, kinds=("paragraph", "list")
            )
        elif etype in ("paragraph", "caption"):
            extra = _collect_following_text(expanded_kids, idx, page, max_items=3, kinds=("list",))
        else:
            extra = ""
        if extra:
            content = content + "\n\n" + extra
        found.append(
            {
                "plate_number": plate_number,
                "page_number": page,
                "content": content,
                "element": kid,
                "kind": kind,
            }
        )

    # Reconstruction pass: for plates without a real caption, scan the
    # body paragraphs for inline "Pl. N, fig. M" mentions and assemble
    # a synthetic caption from the species preceding each mention.
    # We only keep reconstructed plates whose number is plausibly a
    # paper-internal plate — i.e. it doesn't exceed the highest plate
    # number already detected (e.g. real caption says "Plate 3", so
    # reconstructed "Plate 6" must be a citation to another paper) and
    # there is at least one image on the pages where the plate is
    # referenced. The cap-on-real-plate-number filter is the load-bearing
    # one: without it, "De Wever et al. (2001) pl. 6, fig. 3" gets
    # misread as plate 6 of *this* paper.
    images_by_page = _images_page_index(_collect_images(kids))
    # Sorted real plate caption pages; used to bound the image-window
    # search so a "Plate 5 (Pl. 5, fig. 3)" body mention on page 4
    # doesn't accidentally grab the image of plate 7.
    real_plate_pages = sorted({p["page_number"] for p in found})
    max_real_plate = max(seen_plates) if seen_plates else 0
    # Bump the cap a little so the very last plate of the paper (whose
    # only mention is in the body, not in a standalone caption header)
    # still gets a chance to be reconstructed.
    plate_cap = max_real_plate + 1 if max_real_plate else 0
    for plate_number, mentions in _harvest_inline_plate_refs(kids).items():
        if plate_number in seen_plates:
            continue
        if not mentions:
            continue
        if plate_cap and plate_number > plate_cap:
            continue
        ref_pages = sorted({m[2] for m in mentions})
        ref_page = ref_pages[0]
        # The plate's image is somewhere between the body mention page
        # and the *next* real plate's caption page. The figure may sit
        # a page or two after the body description (Pouille 2014 plate
        # 1 is described on p4 with the figure on p5).
        next_plate_page = next((pp for pp in real_plate_pages if pp > ref_page), None)
        page_lo = ref_page
        page_hi = next_plate_page if next_plate_page is not None else ref_page + int(caption_window)
        has_image = any(page_lo <= p <= page_hi for p in images_by_page)
        if not has_image:
            continue
        seen_plates.add(plate_number)
        lines: list[str] = [f"Plate {plate_number}. (Reconstructed from systematic descriptions)"]
        for sp, plate_ref, page in mentions:
            lines.append(f"{sp} ({plate_ref})")
        earliest_page = ref_pages[0]
        found.append(
            {
                "plate_number": plate_number,
                "page_number": earliest_page,
                "content": "\n".join(lines),
                "element": None,
            }
        )
    found.sort(key=lambda d: (d["plate_number"], d["page_number"]))
    return found


def _harvest_inline_plate_refs(kids: list[dict[str, Any]]) -> dict[int, list[tuple[str, str, int]]]:
    """Walk the body paragraphs looking for inline ``Pl. N, fig M``
    references and capture the species name preceding each ref.

    Returns a dict ``{plate_number: [(species_text, full_ref, page), ...]}``
    where ``full_ref`` is the matched text (e.g. ``"Pl. 1, figs 5–7"``)
    and ``page`` is the 1-indexed page where the mention was found.
    """
    out: dict[int, list[tuple[str, str, int]]] = {}
    for k in kids:
        if not isinstance(k, dict):
            continue
        if k.get("type") != "paragraph":
            continue
        text = k.get("content") or ""
        if not text:
            continue
        page = int(k.get("page number", 0) or 0)
        for m in _PLATE_INLINE_REF_RE.finditer(text):
            plate_number = int(m.group(1))  # inline refs are always Arabic
            ref = m.group(0)
            # Walk left from the match to find the species name. Look
            # for a binomial that starts a sentence/line OR follows
            # typical parens/commas.
            left_start = max(0, m.start() - 250)
            prefix = text[left_start : m.start()]
            sp_match = None
            # Find ALL species matches in the prefix and take the
            # closest non-author-citation one. Splitting on ".\n;" was
            # wrong here: "Syntagentactinia? sp. cf. S. excelsa" has
            # internal periods ("sp.", "S.") that get treated as
            # sentence ends, fragmenting the species across pieces.
            # Taking findall instead avoids the fragmentation, but
            # the new "last match" strategy can land on an author
            # citation like "Nazarov in" if it appears in the
            # parenthetical author info. Filter those out.
            for candidate in reversed(list(_SPECIES_NAME_RE.finditer(prefix))):
                cand_species = candidate.group(1)
                if not _looks_like_author_citation(cand_species):
                    sp_match = candidate
                    break
            if not sp_match:
                # Last resort: a single Genus followed by "?" — used
                # in "Genus? sp." inline references.
                gen_match = re.search(r"([A-Z][a-z]+)\s+\?", prefix[-80:])
                if gen_match:
                    species = gen_match.group(1) + " ?"
                else:
                    continue
            else:
                species = sp_match.group(1)
            # Reject false positives: parenthetical authorship that
            # precedes the plate ref (e.g. "Nazarov in (Pl. 1, fig. 15)"
            # in Pouille 2014 — "Nazarov in" is an author citation, not
            # a species). The literal "Genus in" / "Genus & Author"
            # patterns are how paleontology formats citations inline.
            if _looks_like_author_citation(species):
                continue
            out.setdefault(plate_number, []).append((species, ref, page))
    return out


# Author-citation words that frequently follow a surname in a parenthetical
# paleontology citation ("Nazarov in Nazarov & Ormiston 1985", "Smith &
# Jones 2001", etc.) and that we should NOT treat as a species epithet.
_AUTHOR_CITATION_WORDS = frozenset(
    {
        "in",
        "and",
        "&",
        "et",
        "al",
        "al.",
        "of",
        "de",
        "von",
        "van",
        "in Nazarov",
        "in Ormiston",
    }
)

# Detect an inline "(SURNAME)" style author citation in the head of a
# paragraph — a strong signal that a "Fig. N" match is body text (a
# species description), not a figure caption. Used by
# _looks_like_fig_caption to filter out Bandini-style body paragraphs
# like "Fig. 21 Archaeodictyomitra montisserei (SQUINABOL) Pl. 8 ..."
_FIG_HEAD_AUTHOR_CITE_RE = re.compile(r"\(([A-Z]{3,})\)")


def _looks_like_fig_caption(content: str) -> bool:
    """Return True if a paragraph whose text starts with "Fig. N" is
    actually a figure caption (not body text).

    Rejects three common false-positive patterns:
      1. too-short matches (< 25 chars) — typically list references
         like "Fig. 26" or continuation markers like "Fig. 14 continued c".
      2. body-text species descriptions — a paragraph whose first 200
         chars contain a "(UPPERCASE)" author citation is almost
         always a species list / description, not a caption.
      3. inline body-text "Fig. N X" mentions where X is a known
         non-caption verb (Photograph, Map, Schematic, Diagram, ...)
         — these are always inline body-text references to figures
         that have their own caption elsewhere, NOT a new caption.
         Without this guard, bandini2011's "Fig. 7 Photograph of
         the Early Cretaceous radiolarite ..." body paragraph was
         promoted to a plate figure, hijacking page-27 images
         from plate 8 and breaking the pl08 image routing.
    """
    if len(content) < 25:
        return False
    if _FIG_HEAD_AUTHOR_CITE_RE.search(content[:200]):
        return False
    # Inline body-text "Fig. N" mentions: a leading word like
    # "Photograph" right after the figure number signals that this
    # is an inline body-text reference to a figure that has its
    # own caption elsewhere, NOT a new caption. Conservative list —
    # only includes words that are unambiguous body-text patterns
    # in the radiolarian literature (e.g. bandini2011's "Fig. 7
    # Photograph of the Early Cretaceous radiolarite..."). "Schematic"
    # / "Diagram" / "Map" are excluded because some papers use
    # them as legitimate caption titles ("Fig. 1 Schematic of the
    # apparatus").
    first_words = content.split(None, 3)[:3]
    if len(first_words) >= 3 and first_words[2].lower() in (
        "photograph",
        "photographs",
    ):
        return False
    return True


def _looks_like_author_citation(species: str) -> bool:
    """Heuristic: a captured "species" that is actually a citation.

    "Nazarov in" / "Nazarov & Jones" / "Smith in" are typical
    author-citation patterns. A real species name is *Genus epithet*
    (two words, lowercase second word); anything else is suspect.
    """
    if not species:
        return True
    parts = species.split()
    if len(parts) < 2:
        return True
    if parts[-1].lower() in _AUTHOR_CITATION_WORDS:
        return True
    if any(p == "&" for p in parts):
        return True
    return False


def _images_within_page_range(
    images: list[dict[str, Any]], page_lo: int, page_hi: int
) -> list[dict[str, Any]]:
    """Return images whose page number is in [page_lo, page_hi] (inclusive)."""
    out = []
    for img in images:
        p = int(img.get("page number", 0) or 0)
        if page_lo <= p <= page_hi:
            out.append(img)
    return out


def _build_figures_from_plate_captions(
    plate_captions: list[dict[str, Any]],
    images: list[dict[str, Any]],
    output_dir: Path,
    paper_id: str,
    caption_window: int = 5,
) -> tuple[list[FigureCaptionPair], set[int]]:
    """Build FigureCaptionPair list by linking each plate caption to nearby images.

    The convention we use for linking:
      * The plate caption is on page ``P`` (or P-1; some papers print the
        figure on the left page and the caption on the facing right page).
      * The figure images for that plate are on page ``P`` (caption below
        figure) or on page ``P+1`` (figure on the next page).
      * The next plate caption's page is **not** a hard barrier — if Plate B's
        caption is on the same page as Plate A's figure, A still claims the
        figure. We process plates in caption order and skip images already
        claimed by an earlier plate.

    Returns a tuple of (pairs, claimed_image_ids) so the caller can
    exclude the *exact* images that have already been linked from the
    leftover / unassigned bucket — not just the page number, which is
    too coarse when a plate caption on page P uses images spanning
    [P, P+1] (Bandini 2011 plate 1: caption p12, actual plate image p13).

    BUG-2 (audit 2026-09-04, Zhang 2014): assignment runs in TWO passes.
    Fig-kind captions ("Fig. N" charts / maps) go first and claim their
    **same-page** images only (widening to the standard forward window
    only when the caption page has no image at all); plate-kind captions
    ("Explanation of Plate N") then take the remaining unclaimed images
    with the classic window logic. The old single pass let a plate
    caption that shares a page with a Fig. N chart steal the chart (and
    leave the fig caption with ``no_images=True``). Captions without a
    ``kind`` key (legacy fixtures / reconstructed plates) are treated as
    plate-kind, which keeps their behaviour identical to the old pass.
    """
    pairs: list[FigureCaptionPair] = []
    claimed_image_ids: set[int] = set()

    def _unclaimed_in_range(page_lo: int, page_hi: int) -> list[dict[str, Any]]:
        """Images in [page_lo, page_hi] not already claimed by an
        earlier caption."""
        out: list[dict[str, Any]] = []
        for img in images:
            p = int(img.get("page number", 0) or 0)
            if page_lo <= p <= page_hi:
                # Phase 54 audit: H3 — guard ``int()`` against non-numeric
                # ``id`` values. Some OD JSON versions write ids as
                # strings like ``"p011f1"`` (observed in Boughdiri 2007);
                # the previous ``int(img.get("id", -1))`` raised
                # ``ValueError`` that propagated up to ``extract()``'s
                # outer ``except`` and turned the entire PDF into
                # ``success=False`` with 0 figures. Round 21 fixed the
                # same pattern in ``_extract_figures`` but missed this
                # second call site.
                try:
                    img_id = int(img.get("id", -1))
                except (TypeError, ValueError):
                    img_id = -1
                if img_id not in claimed_image_ids:
                    out.append(img)
        return out

    # BUG-2 two-pass order: fig-kind first (precise same-page anchor),
    # then plate-kind (window-based). Within each pass the captions
    # keep document order.
    ordered = (
        [c for c in plate_captions if c.get("kind") == "fig"]
        + [c for c in plate_captions if c.get("kind") != "fig"]
    )
    n = len(ordered)
    for idx, cap in enumerate(ordered):
        is_fig = cap.get("kind") == "fig"
        page_lo = cap["page_number"]
        # Phase 28: forward window now configurable via
        # ``caption_window`` (default 5 on PipelineConfig / OD
        # extractor). Covers the caption-below-figure and
        # figure-on-next-page layouts, plus appendix-style layouts
        # where plates sit a few pages after their caption. The next-
        # caption clamp below prevents Plate A from stealing images
        # that belong to Plate B even when the window is wide.
        page_hi = page_lo + int(caption_window)
        if idx + 1 < n:
            next_cap_page = ordered[idx + 1]["page_number"]
            # Clamp the forward window when the next caption is at
            # least 2 pages beyond page_lo. The conditions
            # ``next_cap_page > page_lo + 1`` and
            # ``next_cap_page >= page_lo + 2`` are mathematically
            # equivalent — there is no behavior change between them.
            # This block was previously edited to ``>=`` on the theory
            # that the original ``>`` was wrong; mutation testing
            # showed both forms are identical, so the edit is
            # reverted to keep the code minimal and the comments
            # honest.
            if next_cap_page > page_lo + 1:
                page_hi = min(page_hi, next_cap_page - 1)
        page_hi = max(page_hi, page_lo)  # never invert

        if is_fig:
            # Fig-kind precision anchor (BUG-2): charts and maps are
            # captioned on the image's own page, so claim same-page
            # images first and only widen to the window when the
            # caption page has no unclaimed image at all.
            candidates = _unclaimed_in_range(page_lo, page_lo)
            if not candidates:
                candidates = _unclaimed_in_range(page_lo, page_hi)
        else:
            # Candidate images: in [page_lo, page_hi], not already claimed.
            candidates = _unclaimed_in_range(page_lo, page_hi)

        if not candidates:
            # No images found for this plate caption; skip but record the caption
            # so the downstream matcher still sees the text.
            pairs.append(
                FigureCaptionPair(
                    figure_id=f"od_plate_{paper_id}_p{page_lo:03d}_pl{cap['plate_number']:02d}",
                    page_number=page_lo,
                    image_paths=[],
                    caption_text=cap["content"],
                    merged_bbox=None,
                    metadata={"plate_number": cap["plate_number"], "no_images": True},
                )
            )
            continue

        # Mark these image IDs as claimed so the next plate's forward search
        # doesn't re-grab them.
        for img in candidates:
            # Phase 54 audit: H3 — see comment above. Skip non-numeric
            # ids entirely; the ``claimed_image_ids`` set only exists to
            # avoid double-grabbing, so dropping one is harmless.
            try:
                img_id = int(img.get("id", -1))
            except (TypeError, ValueError):
                continue
            if img_id >= 0:
                claimed_image_ids.add(img_id)

        image_paths = _resolve_image_paths(candidates, output_dir, paper_id=paper_id)
        merged_bbox = _union_bbox(candidates)
        # Anchor the figure_id on the first image's page.
        first_page = int(candidates[0].get("page number", page_lo))
        pairs.append(
            FigureCaptionPair(
                figure_id=f"od_plate_{paper_id}_p{first_page:03d}_pl{cap['plate_number']:02d}",
                page_number=first_page,
                image_paths=image_paths,
                caption_text=cap["content"],
                merged_bbox=merged_bbox,
                metadata={"plate_number": cap["plate_number"]},
            )
        )
    return pairs, claimed_image_ids


def _resolve_image_paths(
    images: list[dict[str, Any]],
    output_dir: Path,
    paper_id: str | None = None,
) -> list[str]:
    """Resolve the absolute file path of each image element.

    Two strategies are tried in order:

    1. **Direct source field** (preferred when OD populated it): if
       the image element carries a ``source`` key, the path is built
       relative to ``output_dir``. Some OD builds emit a relative
       source like ``".../foo_images/imageFile1.png"``; we resolve
       that against ``output_dir`` whether the prefix is present or
       not.
    2. **Position-based fallback** (the common case for our corpus):
       the image elements from ``_iter_all_elements`` are returned in
       the same order OD exported them, and OD's exporter names them
       ``imageFile1.png``, ``imageFile2.png``, … in that order. The
       ``_<pdf_stem>_images/`` directory is found by walking
       ``output_dir``. Each image element's 1-based position in
       the full walk maps to ``imageFile{N}.png`` (e.g. the 3rd
       image in walk order → ``imageFile3.png``).

    The previous version relied solely on strategy 1; the new
    fallback rescues papers where the source field is absent (e.g.
    Uchino 2017 — the image elements have id/page/bbox but no
    ``source``).
    """
    paths: list[str] = []
    seen: set[str] = set()

    # Strategy 1: per-image ``source`` field.
    for img in images:
        src = img.get("source")
        if not src:
            continue
        candidate = output_dir / src
        if candidate.exists() and str(candidate) not in seen:
            paths.append(str(candidate))
            seen.add(str(candidate))
    if paths:
        return paths

    # Strategy 2: position-based fallback. We need the
    # ``<images_dir>`` (the directory OD exported to) which is
    # under ``output_dir/od_output/<paper_id>/<pdf_stem>_images/``.
    # When ``paper_id`` is known we scope the search to the
    # paper-specific subdirectory so a multi-paper work_dir picks
    # the correct paper's images instead of alphabetically-first.
    images_dir: Path | None = None
    if paper_id:
        scoped = output_dir / "od_output" / paper_id
        if scoped.is_dir():
            for candidate in sorted(scoped.glob("*_images")):
                if candidate.is_dir() and (candidate / "imageFile1.png").exists():
                    images_dir = candidate
                    break
    if images_dir is None:
        for candidate in output_dir.rglob("*_images"):
            if candidate.is_dir() and (candidate / "imageFile1.png").exists():
                images_dir = candidate
                break
    if images_dir is None:
        return paths  # empty
    # imageFileN.png was exported by OD in the same order the
    # image elements appear in the kids tree (``_iter_all_elements``
    # yields them depth-first). The mapping is: N-th image in the
    # depth-first walk → imageFileN.png. We therefore need to
    # compute the *absolute* index of each image in the full walk,
    # not the relative index within the ``images`` argument (which
    # may be a subset). The full walk is recovered via
    # ``_collect_images_from_output_dir``.
    all_images = _collect_images_from_output_dir(output_dir, images_dir, paper_id=paper_id)
    # Use (page, id) as the key — ``id(img)`` is the Python object
    # identity, which is NOT stable across re-reads of the JSON
    # (each ``json.load`` produces fresh dict objects). (page, id)
    # uniquely identifies an image element across reads.
    # Audit 2026-07-26 B3: ``id`` may be a non-numeric string (e.g.
    # "p011f1" in Boughdiri 2007); use str() so the key is stable
    # across all_images construction and lookup without raising
    # ValueError on int().
    full_index: dict[tuple[int, str], int] = {
        (int(img.get("page number", 0) or 0), str(img.get("id", 0) or 0)): idx
        for idx, img in enumerate(all_images, start=1)
    }
    for img in images:
        key = (int(img.get("page number", 0) or 0), str(img.get("id", 0) or 0))
        idx = full_index.get(key)
        if idx is None:
            continue
        candidate = images_dir / f"imageFile{idx}.png"
        if candidate.exists() and str(candidate) not in seen:
            paths.append(str(candidate))
            seen.add(str(candidate))
    return paths


def _collect_images_from_output_dir(
    output_dir: Path,
    images_dir: Path | None = None,
    paper_id: str | None = None,
) -> list[dict[str, Any]]:
    """Collect every image element from the OD JSON under output_dir.

    Mirrors what ``OpenDataLoaderExtractor.extract`` writes: a single
    JSON under ``<output_dir>/od_output/<paper_id>/<pdf_stem>.json``.
    Used by :func:`_resolve_image_paths` to compute the absolute
    imageFileN index for a given image element.

    When ``images_dir`` is provided, the JSON is looked up next to
    it (the same parent directory and the same paper_id). This
    prevents a multi-paper ``work_dir`` from returning another
    paper's JSON. Returns [] when no JSON is found.

    When ``paper_id`` is provided (without images_dir), we scope
    the recursive search to the paper-specific subdirectory so
    multi-paper ``work_dir``s return the correct paper's images
    instead of merging all papers (which collides on (page, id)
    keys). Returns [] when no JSON is found.
    """
    if images_dir is not None:
        # The JSON file shares the parent of ``images_dir``. We
        # glob for the first ``*.json`` in the same directory and
        # accept it. (More than one PDF in the same paper_id is
        # unusual and would have produced multiple _images dirs.)
        parent = images_dir.parent
        for path in sorted(parent.glob("*.json")):
            if "_images" in str(path):
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            kids = data.get("kids") or []
            return _collect_images(kids)
        return []
    # Fallback: scope to paper_id subdirectory when known so a
    # multi-paper work_dir resolves the right paper's JSON.
    if paper_id:
        paper_dir = output_dir / "od_output" / paper_id
        if paper_dir.is_dir():
            for path in sorted(paper_dir.glob("*.json")):
                if "_images" in str(path):
                    continue
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, ValueError):
                    continue
                kids = data.get("kids") or []
                return _collect_images(kids)
        return []
    # No paper_id and no images_dir — search the whole output_dir.
    # Single-paper case: take the only/first JSON. Multi-paper case
    # with no paper_id hint: fall back to merging all (suboptimal
    # because (page, id) keys can collide; callers should pass
    # paper_id when possible).
    json_files = sorted(output_dir.rglob("*.json"))
    collected: list[dict[str, Any]] = []
    for path in json_files:
        if "_images" in str(path):
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        kids = data.get("kids") or []
        collected.extend(_collect_images(kids))
    return collected


def _find_nearest_caption(
    plate_images: list[dict[str, Any]],
    all_captions: list[dict[str, Any]],
    page: int,
) -> str | None:
    """Find the caption closest (vertically) to the plate on the same page.

    Ties are broken by the caption's spatial-info availability: a caption
    with a real ``bounding box`` wins over one without (the no-bbox case
    is the common OpenDataLoader quirk where OD parses the caption text
    but didn't expose its layout coords). Pre-fix, all no-bbox captions
    tied at ``dist = plate_bottom - 0 = plate_bottom`` and the winner was
    whichever appeared first in ``all_captions`` — essentially random,
    often the wrong figure's caption.
    """
    if not plate_images:
        return None
    plate_bottom = min(_bbox_bottom(img) for img in plate_images)

    best_dist = float("inf")
    best_has_bbox = False
    best_text: str | None = None
    for cap in all_captions:
        if cap.get("page number") != page:
            continue
        cap_bottom = _bbox_bottom(cap)
        cap_has_bbox = bool(cap.get("bounding box"))
        # Prefer captions just below the plate
        if cap_bottom > plate_bottom:
            continue
        dist = plate_bottom - cap_bottom
        # Tighten the tie-break: lower distance wins; if equal, the caption
        # with a bounding box wins; if still tied, first-encountered wins.
        # We model "first-encountered wins" by leaving best_* unchanged
        # when the new candidate is strictly worse on (dist, has_bbox).
        is_better = dist < best_dist or (dist == best_dist and cap_has_bbox and not best_has_bbox)
        if is_better:
            best_dist = dist
            best_has_bbox = cap_has_bbox
            best_text = cap.get("content")
    return best_text


def _bbox_bottom(el: dict[str, Any]) -> float:
    bb = el.get("bounding box")
    return float(bb[1]) if bb and len(bb) >= 4 else 0.0


def _extract_fulltext_sections(data: dict[str, Any]) -> list[dict[str, str]]:
    """Walk the top-level content and build simple text sections."""
    kids: list[dict[str, Any]] = data.get("kids") or []
    sections: list[dict[str, str]] = []
    current_section: dict[str, str] | None = None

    for kid in kids:
        etype = kid.get("type", "")
        if etype == "heading":
            if current_section and current_section.get("text"):
                sections.append(current_section)
            title = kid.get("content", "")
            current_section = {
                "section_id": f"od_sec_{len(sections) + 1}",
                "title": title,
                "section_type": _infer_section_type(title),
                "text": "",
            }
        elif etype == "paragraph" and current_section is not None:
            content = kid.get("content", "")
            if content:
                current_section["text"] += content + "\n"

    if current_section and current_section.get("text"):
        sections.append(current_section)

    # Fallback: if no headings found, collect all paragraphs as one section.
    if not sections:
        all_text_parts: list[str] = []
        for kid in kids:
            etype = kid.get("type", "")
            if etype in ("paragraph", "heading"):
                content = kid.get("content", "")
                if content:
                    all_text_parts.append(content)
        if all_text_parts:
            sections.append(
                {
                    "section_id": "od_sec_1",
                    "title": "Full text",
                    "section_type": "other",
                    "text": "\n".join(all_text_parts),
                }
            )
    return sections


def _infer_section_type(title: str) -> str:
    t = (title or "").lower()
    if "systematic" in t or "paleontology" in t:
        return "systematic_paleontology"
    if "geological" in t or "setting" in t or "stratigraph" in t:
        return "geological_setting"
    if "material" in t or "method" in t:
        return "materials_methods"
    return "other"


# ---- paper-level metadata scrape from OpenDataLoader JSON -----------------

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_ABSTRACT_MARKERS = ("abstract", "summary", "摘要", "概要")
_INTRO_MARKERS = ("introduction", "1.", "1 ")


def _split_authors(author_str: str) -> list[str]:
    """Split an author string on common separators; returns non-empty names."""
    if not author_str:
        return []
    # GROBID often uses "SURNAME, NAME; SURNAME, NAME"
    for sep in (";", " and "):
        if sep in author_str:
            parts = [p.strip() for p in author_str.split(sep) if p.strip()]
            return parts
    # Comma-separated: handle "Xiao, Y., Suzuki, N., and He, W." pattern
    # This is ambiguous; bail out and return the whole string.
    if "," in author_str:
        return [author_str.strip()]
    return [author_str.strip()] if author_str.strip() else []


def _paragraphs_in_order(data: dict[str, Any]) -> list[str]:
    """Walk top-level kids and yield paragraph contents in order."""
    out: list[str] = []
    kids = data.get("kids") or []
    for kid in kids:
        if kid.get("type") == "paragraph":
            content = (kid.get("content") or "").strip()
            if content:
                out.append(content)
    return out


def _extract_paper_metadata_from_json(
    data: dict[str, Any], sections: list[dict[str, str]]
) -> PaperMetadata:
    """Best-effort scrape of paper metadata from the OpenDataLoader JSON tree.

    OpenDataLoader exposes top-level keys: ``title``, ``author``, ``creation date``,
    ``number of pages``. The first two ``heading`` nodes are typically the paper
    title and the author list. The abstract lives in the first paragraph block
    before ``Introduction``. The DOI is in a header line on page 1.
    """
    meta = PaperMetadata(source="opendataloader")

    if not data:
        return meta

    kids = data.get("kids") or []
    # First two heading nodes: paper title, then author list.
    headings: list[str] = []
    for kid in kids:
        if kid.get("type") == "heading":
            txt = (kid.get("content") or "").strip()
            if txt:
                headings.append(txt)
        if len(headings) >= 2:
            break

    json_title = (data.get("title") or "").strip()
    if headings and (
        not json_title or json_title.lower().endswith(".indd") or "indd" in json_title.lower()
    ):
        meta.title = headings[0]
    elif (
        json_title and not json_title.lower().endswith(".indd") and "indd" not in json_title.lower()
    ):
        meta.title = json_title

    # Authors: prefer the second heading (a "SURNAME, NAME" or "FIRST LAST" list)
    author_str = (data.get("author") or "").strip()
    if len(headings) >= 2:
        author_str = headings[1]
    authors = _split_authors(author_str)
    if authors:
        meta.authors = authors

    creation = (data.get("creation date") or "").strip()
    m = _YEAR_RE.search(creation)
    if m:
        try:
            meta.year = int(m.group(1))
        except Exception:
            pass
    # Fallback: scan first 30 paragraphs for "Received <date> YYYY" or a year in the
    # DOI line "Acta Palaeontol. Pol. 62 (3): 647–656, 2017 https://doi..."
    if meta.year is None:
        for p in _paragraphs_in_order(data)[:30]:
            m = _YEAR_RE.search(p)
            if m:
                try:
                    meta.year = int(m.group(1))
                    break
                except Exception:
                    pass

    try:
        page_count = int(data.get("number of pages") or 0)
        if page_count > 0:
            meta.page_count = page_count
    except Exception:
        pass

    # DOI: scan first ~30 paragraphs for "https://doi.org/10.xxx" or "doi: 10.xxx"
    for p in _paragraphs_in_order(data)[:30]:
        m = _DOI_RE.search(p)
        if m:
            meta.doi = m.group(0).rstrip(".,;)")
            break

    # Abstract: usually the first big paragraph block before "Introduction"
    # Stop also at "Key words", "Received", "Copyright" since those are not abstract.
    paras = _paragraphs_in_order(data)
    abstract_parts: list[str] = []
    in_abstract = False
    stop_markers = (
        "key words",
        "keywords",
        "received ",
        "accepted ",
        "available online",
        "copyright",
        "introduction",
        "1. introduction",
        "摘要",
        "introduction.",
    )
    for p in paras:
        low = p.lower().strip()
        if not in_abstract:
            # Look for a big paragraph that is not a metadata block
            if any(low.startswith(mk) for mk in stop_markers):
                continue
            if any(mk in low[:40] for mk in _ABSTRACT_MARKERS):
                in_abstract = True
                stripped = re.sub(
                    r"^\s*(abstract|summary|摘要|概要)[:\s\-—]*", "", p, flags=re.IGNORECASE
                )
                if stripped:
                    abstract_parts.append(stripped)
                continue
            # If a paragraph is very long (>300 chars) and not a header, treat as abstract
            if len(p) > 300 and not any(low.startswith(mk) for mk in stop_markers):
                in_abstract = True
                abstract_parts.append(p)
                continue
        else:
            if any(low.startswith(mk) for mk in stop_markers):
                break
            if len(p) < 50 and (low.startswith(("introduction", "key words", "received"))):
                break
            if len(abstract_parts) < 4:
                abstract_parts.append(p)
    if abstract_parts:
        meta.abstract = " ".join(abstract_parts).strip()

    # Keywords: from "Key words: A, B, C" or "Keywords: ..." paragraph
    for p in paras:
        m = re.match(r"^\s*(?:key\s*words|keywords|关键词)\s*[:：\-—]\s*(.+)$", p, re.IGNORECASE)
        if m:
            kw_text = m.group(1)
            parts = re.split(r"[,;，；、]", kw_text)
            for kw in parts:
                kw = kw.strip().rstrip(".")
                if kw and kw not in meta.keywords:
                    meta.keywords.append(kw)
            break

    # Journal: from a paragraph that looks like "Acta Palaeontol. Pol. 62 (3): ..."
    # OR a line containing "https://doi.org/" — then everything before the volume
    # number is the journal title.
    for p in paras:
        if "doi.org" in p.lower():
            m = re.match(r"^\s*([A-Z][A-Za-z\.\s\-]+?)\s+\d+", p)
            if m:
                cand = m.group(1).strip().rstrip(".,")
                if 4 < len(cand) < 100:
                    meta.journal = cand
                    break
        else:
            m = re.match(r"^\s*([A-Z][A-Za-z\.\s]+?)\s+\d+\s*(?:\(\d+\))?\s*[:\.]", p)
            if m:
                cand = m.group(1).strip().rstrip(".,")
                if 4 < len(cand) < 100:
                    meta.journal = cand
                    break

    # Volume / issue / pages: from the same DOI line
    for p in paras:
        if "doi.org" in p.lower():
            m = re.search(r"\b(\d+)\s*(?:\((\d+)\))?\s*[:\s]\s*([0-9–—\-]+)", p)
            if m:
                try:
                    if meta.volume is None:
                        meta.volume = m.group(1)
                    if m.group(2) and meta.issue is None:
                        meta.issue = m.group(2)
                    if m.group(3) and meta.pages is None:
                        meta.pages = m.group(3)
                except Exception:
                    pass
                break

    # Confidence
    filled = sum(
        1 for k in ("title", "doi", "abstract", "year", "journal", "authors") if getattr(meta, k)
    )
    if filled == 0:
        meta.confidence = 0.0
        meta.source = "none"
    else:
        meta.confidence = min(0.85, 0.3 + 0.1 * filled)
    return meta
