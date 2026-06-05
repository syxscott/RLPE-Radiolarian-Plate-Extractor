from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import ensure_dir, stable_id
from .types import PaperMetadata

logger = logging.getLogger(__name__)

# ---- data types -----------------------------------------------------------

@dataclass(slots=True)
class FigureCaptionPair:
    """A figure image (or merged image group) paired with its caption."""
    figure_id: str
    page_number: int
    image_paths: list[str]          # one or more exported image files
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
    """

    def __init__(
        self,
        use_ocr: bool = False,
        ocr_lang: str = "en",
        image_format: str = "png",
        merge_gap_pt: float = 72.0,
    ) -> None:
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.image_format = image_format
        self.merge_gap_pt = merge_gap_pt
        self._available: bool | None = None

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
                paper_id=paper_id, json_data=None, output_dir=out,
                figures=[], fulltext_sections=[],
                success=False, error="opendataloader-pdf not installed",
            )

        try:
            self._run_opendataloader(pdf_path, out)
            json_path = self._find_json(out)
            if json_path is None:
                return OpenDataLoaderResult(
                    paper_id=paper_id, json_data=None, output_dir=out,
                    figures=[], fulltext_sections=[],
                    success=False, error="No JSON output produced by OpenDataLoader",
                )
            data = _load_json(json_path)
            figures = self._extract_figures(data, out, paper_id)
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
                paper_id=paper_id, json_data=data, output_dir=out,
                figures=figures, fulltext_sections=sections,
                success=True,
                paper_metadata=paper_metadata,
            )
        except Exception as exc:
            logger.exception("OpenDataLoader extraction failed for %s", pdf_path)
            return OpenDataLoaderResult(
                paper_id=paper_id, json_data=None, output_dir=out,
                figures=[], fulltext_sections=[],
                success=False, error=str(exc),
            )

    # -- caption OCR fallback ------------------------------------------------

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

        ocr_engine = None
        ocr_lock = threading.Lock()
        try:
            with ocr_lock:
                if ocr_engine is None:
                    import easyocr
                    ocr_engine = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            logger.warning("EasyOCR init failed; caption OCR fallback disabled")
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
                    recovered = _ocr_caption_band(
                        doc, page_index, fig.merged_bbox, ocr_engine, np
                    )
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

    def _run_opendataloader(self, pdf_path: Path, output_dir: Path) -> None:
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
        plate_captions = _find_plate_captions(kids)
        if plate_captions:
            plate_pairs, claimed_image_ids = _build_figures_from_plate_captions(
                plate_captions, images, output_dir, paper_id
            )
            # Always also include plate-less figures (so the geological/stratigraphic
            # index figures, like "Fig. 2 distribution map", still get processed).
            # Use the exact claimed image IDs (not just their page numbers) so an
            # image on a "plate page" but linked to a different plate is not
            # accidentally re-surfaced as a leftover.
            leftover_images = [
                img for img in images
                if int(img.get("id", -1)) not in claimed_image_ids
            ]
            if leftover_images:
                # Build a single fallback figure for unassigned images so the
                # index-map / Fig. 2 distribution etc. still flow through the
                # matcher instead of being silently dropped.
                plate_imgs = _merge_nearby_images(leftover_images, gap_pt=self.merge_gap_pt)
                for plate_idx, plate_images in enumerate(plate_imgs, start=1):
                    # Build a caption lookup keyed by linked_content_id
                    caption_for_image: dict[int, str] = {}
                    for cap in captions:
                        linked = cap.get("linked content id")
                        if linked is not None:
                            caption_for_image[int(linked)] = cap.get("content") or ""
                    plate_cap_list: list[str] = []
                    for img in plate_images:
                        img_id = img.get("id")
                        cap_text = caption_for_image.get(int(img_id)) if img_id is not None else None
                        if cap_text:
                            plate_cap_list.append(cap_text)
                    caption_text = " ".join(plate_cap_list) if plate_cap_list else None
                    if not caption_text:
                        page = plate_images[0].get("page number", 1)
                        caption_text = _find_nearest_caption(plate_images, captions, page)
                    image_paths = _resolve_image_paths(plate_images, output_dir)
                    merged_bbox = _union_bbox(plate_images)
                    plate_pairs.append(FigureCaptionPair(
                        figure_id=f"od_fig_{paper_id}_p{plate_images[0].get('page number', 1):03d}_{plate_idx:02d}",
                        page_number=int(plate_images[0].get("page number", 1)),
                        image_paths=image_paths,
                        caption_text=caption_text,
                        merged_bbox=merged_bbox,
                        metadata={"unassigned": True},
                    ))
            return plate_pairs

        # FALLBACK: no plate captions found. Use the original spatial-merge +
        # linked-content caption association.
        plates = _merge_nearby_images(images, gap_pt=self.merge_gap_pt)
        pairs: list[FigureCaptionPair] = []
        for plate_idx, plate_images in enumerate(plates, start=1):
            # Build a caption lookup keyed by linked_content_id
            caption_for_image: dict[int, str] = {}
            for cap in captions:
                linked = cap.get("linked content id")
                if linked is not None:
                    caption_for_image[int(linked)] = cap.get("content") or ""

            plate_caps: list[str] = []
            for img in plate_images:
                img_id = img.get("id")
                cap_text = caption_for_image.get(int(img_id)) if img_id is not None else None
                if cap_text:
                    plate_caps.append(cap_text)

            caption_text = " ".join(plate_caps) if plate_caps else None

            if not caption_text:
                page = plate_images[0].get("page number", 1)
                caption_text = _find_nearest_caption(plate_images, captions, page)

            image_paths = _resolve_image_paths(plate_images, output_dir)
            merged_bbox = _union_bbox(plate_images)
            pairs.append(FigureCaptionPair(
                figure_id=f"od_fig_{paper_id}_p{plate_images[0].get('page number', 1):03d}_{plate_idx:02d}",
                page_number=int(plate_images[0].get("page number", 1)),
                image_paths=image_paths,
                caption_text=caption_text,
                merged_bbox=merged_bbox,
            ))
        return pairs


# ---- helpers (module-level) ------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_all_elements(kids: list[dict[str, Any]]):
    """Recursively yield every element in the content tree."""
    for kid in kids:
        yield kid
        children = kid.get("kids") or kid.get("rows") or kid.get("list items") or kid.get("cells") or []
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
    return {int(img.get("page number", 0) or 0) for img in images if int(img.get("page number", 0) or 0) > 0}


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

import re as _re

# Match "Plate 1", "Plate 12" — possibly with leading "Explanation of".
# Examples seen in OA papers:
#   "Plate 1 Scanning electron microscope pictures of radiolarians..."
#   "Explanation of Plate 3. ﬁgs 1–5. Trilonche crassispinosa..."
#   "Plate 5, Figs. 1–10. Caption text..."
_PLATE_CAPTION_RE = _re.compile(
    r"^\s*(?:Explanation\s+of\s+)?Plate\s+(\d+)\b",
    _re.IGNORECASE,
)

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
_FIG_CAPTION_RE = _re.compile(
    r"^\s*Fig(?:ure)?\s*\.?\s*(\d+)([a-z]?)\s*([.\s])\s*(\S)",
    _re.IGNORECASE,
)

# Match an inline plate figure reference inside a body paragraph.
# Pouille 2014 has no real "Plate N" captions; the species list lives
# in the systematic paleontology descriptions, e.g.:
#   "Genus species AUTHOR (Pl. 1, figs 5–7)"
#   "Genus species AUTHOR (Plate 2, figure 7)"
#   "Genus species (Pl. 3. fig. 11)"
# We detect these and reconstruct a per-plate caption by concatenating
# all matching "Pl. N" / "Plate N" mentions found in the body.
_PLATE_INLINE_REF_RE = _re.compile(
    r"\b(?:[Pp]l(?:ate)?\.?)\s*(\d+)\s*[,.]?\s*[Ff]ig(?:s|ure)?\.?\s*\d+[a-z\-]*",
)
# Match a Genus species (or Genus? sp. cf./aff. species) preceding the
# plate reference — e.g. "Syntagentactinia biocculosa ... (Pl. 1, figs 5–7)"
# or "Syntagentactinia? sp. cf. S. excelsa (Pl. 1, figs 1–4)".
# We grab the species name(s) from the start of the line, then look right-
# ward for the plate ref. Pouille 2014 is the canonical example.
_SPECIES_NAME_RE = _re.compile(
    r"([A-Z][a-z]+"          # Genus
    r"(?:"
    r"\s+\?\s+sp\."          # Genus? sp.
    r"(?:\s+[A-Z]\.)?"        #   S.
    r"(?:\s+(?:cf\.|aff\.)\s+[A-Z]?[a-z][a-z\-]+)?"  #   cf./aff. S. species
    r"|"
    r"\s+[a-z][a-z\-]+"        # Genus species
    r"(?:\s+[a-z][a-z\-]+)*"  # optional third epithet
    r")"
    r")"
)


def _collect_following_text(kids: list[dict[str, Any]], start_idx: int,
                             same_page: int, max_items: int = 4) -> str:
    """Return the concatenated ``content`` of up to ``max_items`` siblings
    after ``start_idx`` on the same page, stopping at the next ``heading``
    or ``image`` / ``table``. Used to expand a bare ``Plate N`` heading
    into a full caption by appending the description paragraph and the
    species list that usually follow it (Hollis 2006 plates 1-3)."""
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
        if sib_type in ("paragraph",):
            text = (sib.get("content") or "").strip()
            if text:
                parts.append(text)
        elif sib_type in ("list",):
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


def _find_plate_captions(kids: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    for idx, kid in enumerate(kids):
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
        # Each kind is tracked separately so a paper that uses BOTH
        # conventions (e.g. Plate 1..3 + Fig. 1..6 charts) keeps both
        # sets of captions rather than collapsing them on the same int.
        m = _PLATE_CAPTION_RE.match(content)
        kind = "plate" if m else None
        if not m:
            m = _FIG_CAPTION_RE.match(content)
            kind = "fig" if m else None
        if not m:
            continue
        # Fig-kind matches need an extra content-quality check: the
        # regex also matches body-text paragraphs that happen to start
        # with "Fig. N" (e.g. "Fig. 14 continued c", "Fig. 26", or a
        # species description starting with "Fig. N Archaeodictyomitra
        # montisserei(SQUINABOL) Pl. 8"). Reject:
        #   1. too-short matches (< 25 chars) — almost always a list
        #      reference or a continuation marker, not a real caption.
        #   2. body-text species descriptions: any "(UPPERCASE_WORD)"
        #      author citation within the first 200 chars is a strong
        #      signal this is a species list, not a figure caption.
        if kind == "fig" and not _looks_like_fig_caption(content):
            continue
        plate_number = int(m.group(1))
        dedup_key = (plate_number, kind or "plate")
        if dedup_key in seen_plates_with_kind:
            continue
        if kind == "plate":
            seen_plates.add(plate_number)
        seen_plates_with_kind.add(dedup_key)
        page = int(kid.get("page number", 0) or 0)
        # For heading-type matches, expand by appending following paragraphs
        # / lists on the same page (Hollis 2006 has Plate 1 + description
        # + species list as three separate elements).
        if etype == "heading":
            extra = _collect_following_text(kids, idx, page, max_items=3)
            if extra:
                content = content + "\n\n" + extra
        found.append({
            "plate_number": plate_number,
            "page_number": page,
            "content": content,
            "element": kid,
            "kind": kind,
        })

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
        next_plate_page = next(
            (pp for pp in real_plate_pages if pp > ref_page), None
        )
        page_lo = ref_page
        page_hi = next_plate_page if next_plate_page is not None else ref_page + 3
        has_image = any(page_lo <= p <= page_hi for p in images_by_page)
        if not has_image:
            continue
        seen_plates.add(plate_number)
        lines: list[str] = [f"Plate {plate_number}. (Reconstructed from systematic descriptions)"]
        for sp, plate_ref, page in mentions:
            lines.append(f"{sp} ({plate_ref})")
        earliest_page = ref_pages[0]
        found.append({
            "plate_number": plate_number,
            "page_number": earliest_page,
            "content": "\n".join(lines),
            "element": None,
        })
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
        text = (k.get("content") or "")
        if not text:
            continue
        page = int(k.get("page number", 0) or 0)
        for m in _PLATE_INLINE_REF_RE.finditer(text):
            plate_number = int(m.group(1))
            ref = m.group(0)
            # Walk left from the match to find the species name. Look
            # for a binomial that starts a sentence/line OR follows
            # typical parens/commas.
            left_start = max(0, m.start() - 250)
            prefix = text[left_start:m.start()]
            sp_match = None
            # Try each line in the prefix (the species is usually on the
            # same line as the plate ref, e.g. "Genus species (Pl. N, ...)")
            for piece in _re.split(r"[\n;\.]+", prefix)[-3:]:
                sp_match = _SPECIES_NAME_RE.search(piece)
                if sp_match:
                    break
            if not sp_match:
                # Last resort: a single Genus
                gen_match = _re.search(r"([A-Z][a-z]+)\s+\?", prefix[-80:])
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
_AUTHOR_CITATION_WORDS = frozenset({
    "in", "and", "&", "et", "al", "al.", "of", "de", "von", "van",
    "in Nazarov", "in Ormiston",
})

# Detect an inline "(SURNAME)" style author citation in the head of a
# paragraph — a strong signal that a "Fig. N" match is body text (a
# species description), not a figure caption. Used by
# _looks_like_fig_caption to filter out Bandini-style body paragraphs
# like "Fig. 21 Archaeodictyomitra montisserei (SQUINABOL) Pl. 8 ..."
_FIG_HEAD_AUTHOR_CITE_RE = _re.compile(r"\(([A-Z]{3,})\)")


def _looks_like_fig_caption(content: str) -> bool:
    """Return True if a paragraph whose text starts with "Fig. N" is
    actually a figure caption (not body text).

    Rejects two common false-positive patterns:
      1. too-short matches (< 25 chars) — typically list references
         like "Fig. 26" or continuation markers like "Fig. 14 continued c".
      2. body-text species descriptions — a paragraph whose first 200
         chars contain a "(UPPERCASE)" author citation is almost
         always a species list / description, not a caption.
    """
    if len(content) < 25:
        return False
    if _FIG_HEAD_AUTHOR_CITE_RE.search(content[:200]):
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
    """
    pairs: list[FigureCaptionPair] = []
    n = len(plate_captions)
    claimed_image_ids: set[int] = set()
    for idx, cap in enumerate(plate_captions):
        page_lo = cap["page_number"]
        page_hi = page_lo + 2
        if idx + 1 < n:
            next_cap_page = plate_captions[idx + 1]["page_number"]
            # Only clamp if the next caption is at least 1 page beyond
            # ``page_lo``; otherwise both plates are tightly packed and we
            # need the forward window to be respected for both.
            if next_cap_page > page_lo + 1:
                page_hi = min(page_hi, next_cap_page - 1)
        page_hi = max(page_hi, page_lo)  # never invert

        # Candidate images: in [page_lo, page_hi], not already claimed.
        candidates: list[dict[str, Any]] = []
        for img in images:
            p = int(img.get("page number", 0) or 0)
            if page_lo <= p <= page_hi:
                img_id = int(img.get("id", -1))
                if img_id not in claimed_image_ids:
                    candidates.append(img)

        if not candidates:
            # No images found for this plate caption; skip but record the caption
            # so the downstream matcher still sees the text.
            pairs.append(FigureCaptionPair(
                figure_id=f"od_plate_{paper_id}_p{page_lo:03d}_pl{cap['plate_number']:02d}",
                page_number=page_lo,
                image_paths=[],
                caption_text=cap["content"],
                merged_bbox=None,
                metadata={"plate_number": cap["plate_number"], "no_images": True},
            ))
            continue

        # Mark these image IDs as claimed so the next plate's forward search
        # doesn't re-grab them.
        for img in candidates:
            img_id = int(img.get("id", -1))
            if img_id >= 0:
                claimed_image_ids.add(img_id)

        image_paths = _resolve_image_paths(candidates, output_dir)
        merged_bbox = _union_bbox(candidates)
        # Anchor the figure_id on the first image's page.
        first_page = int(candidates[0].get("page number", page_lo))
        pairs.append(FigureCaptionPair(
            figure_id=f"od_plate_{paper_id}_p{first_page:03d}_pl{cap['plate_number']:02d}",
            page_number=first_page,
            image_paths=image_paths,
            caption_text=cap["content"],
            merged_bbox=merged_bbox,
            metadata={"plate_number": cap["plate_number"]},
        ))
    return pairs, claimed_image_ids


def _resolve_image_paths(images: list[dict[str, Any]], output_dir: Path) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for img in images:
        src = img.get("source")
        if not src:
            continue
        candidate = output_dir / src
        if candidate.exists() and str(candidate) not in seen:
            paths.append(str(candidate))
            seen.add(str(candidate))
    return paths


def _find_nearest_caption(
    plate_images: list[dict[str, Any]],
    all_captions: list[dict[str, Any]],
    page: int,
) -> str | None:
    """Find the caption closest (vertically) to the plate on the same page."""
    if not plate_images:
        return None
    plate_bottom = min(_bbox_bottom(img) for img in plate_images)
    plate_top = max(_bbox_top(img) for img in plate_images)

    best: tuple[float, str | None] = (float("inf"), None)
    for cap in all_captions:
        if cap.get("page number") != page:
            continue
        cap_bottom = _bbox_bottom(cap)
        # Prefer captions just below the plate
        if cap_bottom <= plate_bottom:
            dist = plate_bottom - cap_bottom
            if dist < best[0]:
                best = (dist, cap.get("content"))
    return best[1]


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
                "section_id": f"od_sec_{len(sections)+1}",
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
            sections.append({
                "section_id": "od_sec_1",
                "title": "Full text",
                "section_type": "other",
                "text": "\n".join(all_text_parts),
            })
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

import re as _re

_DOI_RE = _re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_YEAR_RE = _re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
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
    if headings and (not json_title or json_title.lower().endswith(".indd") or "indd" in json_title.lower()):
        meta.title = headings[0]
    elif json_title and not json_title.lower().endswith(".indd") and "indd" not in json_title.lower():
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
    stop_markers = ("key words", "keywords", "received ", "accepted ", "available online",
                    "copyright", "introduction", "1. introduction", "摘要", "introduction.")
    for p in paras:
        low = p.lower().strip()
        if not in_abstract:
            # Look for a big paragraph that is not a metadata block
            if any(low.startswith(mk) for mk in stop_markers):
                continue
            if any(mk in low[:40] for mk in _ABSTRACT_MARKERS):
                in_abstract = True
                stripped = _re.sub(
                    r"^\s*(abstract|summary|摘要|概要)[:\s\-—]*", "", p, flags=_re.IGNORECASE
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
        m = _re.match(r"^\s*(?:key\s*words|keywords|关键词)\s*[:：\-—]\s*(.+)$", p, _re.IGNORECASE)
        if m:
            kw_text = m.group(1)
            parts = _re.split(r"[,;，；、]", kw_text)
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
            m = _re.match(r"^\s*([A-Z][A-Za-z\.\s\-]+?)\s+\d+", p)
            if m:
                cand = m.group(1).strip().rstrip(".,")
                if 4 < len(cand) < 100:
                    meta.journal = cand
                    break
        else:
            m = _re.match(r"^\s*([A-Z][A-Za-z\.\s]+?)\s+\d+\s*(?:\(\d+\))?\s*[:\.]", p)
            if m:
                cand = m.group(1).strip().rstrip(".,")
                if 4 < len(cand) < 100:
                    meta.journal = cand
                    break

    # Volume / issue / pages: from the same DOI line
    for p in paras:
        if "doi.org" in p.lower():
            m = _re.search(r"\b(\d+)\s*(?:\((\d+)\))?\s*[:\s]\s*([0-9–—\-]+)", p)
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
    filled = sum(1 for k in ("title", "doi", "abstract", "year", "journal", "authors") if getattr(meta, k))
    if filled == 0:
        meta.confidence = 0.0
        meta.source = "none"
    else:
        meta.confidence = min(0.85, 0.3 + 0.1 * filled)
    return meta
