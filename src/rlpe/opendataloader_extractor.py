from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import ensure_dir, stable_id

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
            return OpenDataLoaderResult(
                paper_id=paper_id, json_data=data, output_dir=out,
                figures=figures, fulltext_sections=sections,
                success=True,
            )
        except Exception as exc:
            logger.exception("OpenDataLoader extraction failed for %s", pdf_path)
            return OpenDataLoaderResult(
                paper_id=paper_id, json_data=None, output_dir=out,
                figures=[], fulltext_sections=[],
                success=False, error=str(exc),
            )

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

        # Merge nearby images on the same page into plates.
        plates = _merge_nearby_images(images, gap_pt=self.merge_gap_pt)

        # Attach captions to each plate.
        pairs: list[FigureCaptionPair] = []
        for plate_idx, plate_images in enumerate(plates, start=1):
            # Build a caption lookup keyed by linked_content_id
            caption_for_image: dict[int, str] = {}
            for cap in captions:
                linked = cap.get("linked content id")
                if linked is not None:
                    caption_for_image[int(linked)] = cap.get("content") or ""

            # Collect captions from any image in this plate.
            plate_captions: list[str] = []
            for img in plate_images:
                img_id = img.get("id")
                cap_text = caption_for_image.get(int(img_id)) if img_id is not None else None
                if cap_text:
                    plate_captions.append(cap_text)

            caption_text = " ".join(plate_captions) if plate_captions else None

            # If no linked caption found, fall back to nearest caption on same page.
            if not caption_text:
                page = plate_images[0].get("page number", 1)
                caption_text = _find_nearest_caption(plate_images, captions, page)

            image_paths = _resolve_image_paths(plate_images, output_dir)

            # Compute the union bbox across all images in the plate.
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
