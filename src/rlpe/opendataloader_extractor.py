from __future__ import annotations

import json
import logging
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
            plate_pairs = _build_figures_from_plate_captions(
                plate_captions, images, output_dir, paper_id
            )
            # Always also include plate-less figures (so the geological/stratigraphic
            # index figures, like "Fig. 2 distribution map", still get processed).
            plates_in_set = {
                p.metadata.get("plate_number") for p in plate_pairs if p.metadata.get("plate_number") is not None
            }
            plate_pages: set[int] = set()
            for p in plate_pairs:
                if p.image_paths:
                    plate_pages.add(p.page_number)
            leftover_images = [
                img for img in images
                if int(img.get("page number", 0) or 0) not in plate_pages
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


def _find_plate_captions(kids: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return elements (paragraph OR caption) whose text starts with
    ``Plate N`` / ``Explanation of Plate N``.

    OpenDataLoader sometimes surfaces plate captions as ``type=caption`` (with a
    ``content`` field) and other times as ``type=paragraph``; we accept both.

    Returns a list of dicts with keys:
      - ``plate_number`` (int)
      - ``page_number``  (int, 1-indexed)
      - ``content``      (str, full text)
      - ``element``      (original dict, in case the caller needs more)
    Sorted by (plate_number, page_number).
    """
    found: list[dict[str, Any]] = []
    for kid in kids:
        if not isinstance(kid, dict):
            continue
        etype = kid.get("type", "")
        if etype not in ("paragraph", "caption"):
            continue
        content = (kid.get("content") or "").strip()
        if not content:
            continue
        m = _PLATE_CAPTION_RE.match(content)
        if not m:
            continue
        plate_number = int(m.group(1))
        page = int(kid.get("page number", 0) or 0)
        found.append({
            "plate_number": plate_number,
            "page_number": page,
            "content": content,
            "element": kid,
        })
    found.sort(key=lambda d: (d["plate_number"], d["page_number"]))
    return found


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
) -> list[FigureCaptionPair]:
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
    return pairs


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
