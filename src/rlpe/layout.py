from __future__ import annotations

import re
from pathlib import Path

import cv2

from .types import FigureRegion, PageRecord
from .utils import ensure_dir, slugify

FIG_REF_PATTERN = re.compile(r"\b(?:fig(?:ure)?|plate)\s*\.?\s*(\d+[A-Za-z]?)\b", re.IGNORECASE)
CAPTION_LEAD_PATTERN = re.compile(
    r"^(?:fig(?:ure)?|plate)\s*\.?\s*(\d+[A-Za-z]?)\b[:\-\.]?\s*", re.IGNORECASE
)


def render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int = 200) -> list[PageRecord]:
    ensure_dir(out_dir)
    pages: list[PageRecord] = []
    fitz = _import_pymupdf()

    doc = fitz.open(str(pdf_path))
    try:
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            image_path = out_dir / f"page_{idx:03d}.png"
            pix.save(str(image_path))
            text = page.get_text("text") or ""
            pages.append(
                PageRecord(
                    page_index=idx,
                    image_path=str(image_path),
                    text=text,
                    width=pix.width,
                    height=pix.height,
                    metadata={"dpi": dpi},
                )
            )
        return pages
    finally:
        doc.close()


def _import_pymupdf():
    """Import PyMuPDF safely and avoid the unrelated `fitz` package collision."""
    try:
        import pymupdf as fitz  # PyMuPDF>=1.24 preferred import

        return fitz
    except Exception:
        pass

    try:
        import fitz  # legacy PyMuPDF import path

        # Guard against wrong `fitz` package (not PyMuPDF).
        if not hasattr(fitz, "open"):
            raise RuntimeError("Imported module 'fitz' is not PyMuPDF")
        return fitz
    except Exception as exc:
        raise RuntimeError(
            "PyMuPDF import failed. Please ensure `pymupdf` is installed and uninstall conflicting `fitz` package."
        ) from exc


def extract_figure_number(text: str | None) -> str | None:
    if not text:
        return None
    m = CAPTION_LEAD_PATTERN.search(text.strip())
    if m:
        return m.group(1)
    return None


def find_caption_pages(
    pages: list[PageRecord], figure_number: str | None, window: int = 2
) -> list[PageRecord]:
    if not pages:
        return []
    candidates: list[PageRecord] = []
    if figure_number:
        for page in pages:
            text = page.text or ""
            if re.search(
                rf"\b(?:fig(?:ure)?|plate)\s*\.?\s*{re.escape(figure_number)}\b",
                text,
                re.IGNORECASE,
            ):
                candidates.append(page)
    if candidates:
        return candidates
    return pages[: min(len(pages), window + 1)]


def detect_figure_regions(
    page: PageRecord,
    min_area: int = 8000,
    *,
    yolo_model_path: str | Path | None = None,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
) -> list[FigureRegion]:
    """Detect figure regions in a rendered PDF page.

    Uses YOLO when ``yolo_model_path`` is set, otherwise falls back to
    the OpenCV connected-component detector.
    """
    if yolo_model_path:
        return detect_figure_regions_yolo(
            page,
            model_path=yolo_model_path,
            conf=yolo_conf,
            iou=yolo_iou,
            min_area=min_area,
        )
    image = cv2.imread(page.image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return []
    # Bug #11 fix: PyMuPDF may save RGBA PNGs. cv2.threshold on a 4-channel
    # image has version-dependent behaviour and can produce an all-zero binary,
    # causing every page to fall back to the fullpage-region branch. Strip
    # alpha explicitly and convert to BGR before grayscale conversion.
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Figures are often non-white objects; invert so dark content becomes foreground.
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    merged = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    img_area = image.shape[0] * image.shape[1]
    regions: list[FigureRegion] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if area > img_area * 0.98:
            continue
        aspect = w / max(1, h)
        if aspect < 0.15 or aspect > 8.0:
            continue
        crop = image[y : y + h, x : x + w]
        crop_dir = ensure_dir(Path(page.image_path).parent / "regions")
        region_id = f"p{page.page_index:03d}_{slugify(f'region_{x}_{y}_{w}_{h}')}"
        crop_path = crop_dir / f"{region_id}.png"
        cv2.imwrite(str(crop_path), crop)
        regions.append(
            FigureRegion(
                page_index=page.page_index,
                bbox=(int(x), int(y), int(w), int(h)),
                crop_path=str(crop_path),
                score=min(0.99, area / img_area),
                region_id=region_id,
                kind="figure",
                metadata={"area": int(area), "aspect": float(aspect)},
            )
        )
    regions.sort(key=lambda r: (r.page_index, r.bbox[1], r.bbox[0]))
    # If no regions were found, use the full page as a fallback region.
    if not regions:
        h, w = image.shape[:2]
        crop_dir = ensure_dir(Path(page.image_path).parent / "regions")
        region_id = f"p{page.page_index:03d}_fullpage"
        crop_path = crop_dir / f"{region_id}.png"
        cv2.imwrite(str(crop_path), image)
        regions.append(
            FigureRegion(
                page_index=page.page_index,
                bbox=(0, 0, w, h),
                crop_path=str(crop_path),
                score=0.5,
                region_id=region_id,
                kind="page",
                metadata={"fallback": True},
            )
        )
    return regions


# ── YOLO-based figure detector ────────────────────────────────────────────────


def detect_figure_regions_yolo(
    page: PageRecord,
    model_path: str | Path,
    conf: float = 0.25,
    iou: float = 0.45,
    min_area: int = 5000,
) -> list[FigureRegion]:
    """Detect figure regions in a PDF page using a YOLO model.

    Parameters
    ----------
    page:
        PageRecord for the rendered page image.
    model_path:
        Path to the YOLO ``.pt`` model file.
    conf:
        Confidence threshold (0–1); detections below this are discarded.
    iou:
        IoU threshold for Non-Maximum Suppression (0–1); overlapping
        detections with IoU > iou are merged.
    min_area:
        Minimum pixel area of a detection (below this is filtered out).

    Returns
    -------
    list[FigureRegion]
        Detected regions, each with ``kind="figure"`` and a saved crop.
    """
    image_path = Path(page.image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    h_img, w_img = image.shape[:2]

    # Lazy-load the YOLO model (loaded once per session, cached on the
    # function object so repeated calls on the same path reuse it).
    cache_key = f"_yolo_model_{model_path}"
    if not hasattr(detect_figure_regions_yolo, cache_key):
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        # Warm up CUDA if available (first inference is slow).
        model(image_path, verbose=False, conf=conf, iou=iou)
        setattr(detect_figure_regions_yolo, cache_key, model)

    model: "ultralytics.YOLO" = getattr(detect_figure_regions_yolo, cache_key)
    results = model(image_path, verbose=False, conf=conf, iou=iou)

    regions: list[FigureRegion] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            # xyxy: (x1, y1, x2, y2) in pixel coordinates; the Boxes
            # object exposes it as shape (1, 4), so index [0] to get (4,).
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            w = x2 - x1
            ht = y2 - y1
            area = w * ht
            if area < min_area:
                continue
            if area > w_img * h_img * 0.98:
                continue
            crop = image[y1:y2, x1:x2]
            crop_dir = ensure_dir(image_path.parent / "regions")
            region_id = f"p{page.page_index:03d}_{slugify(f'yolo_{x1}_{y1}_{w}_{ht}')}"
            crop_path = crop_dir / f"{region_id}.png"
            cv2.imwrite(str(crop_path), crop)
            conf_score = float(box.conf.cpu().numpy()[0])
            regions.append(
                FigureRegion(
                    page_index=page.page_index,
                    bbox=(x1, y1, w, ht),
                    crop_path=str(crop_path),
                    score=conf_score,
                    region_id=region_id,
                    kind="figure",
                    metadata={
                        "area": int(area),
                        "detector": "yolo",
                        "model": str(model_path),
                        "conf": conf_score,
                    },
                )
            )

    regions.sort(key=lambda r: (r.page_index, r.bbox[1], r.bbox[0]))
    # Full-page fallback (same as OpenCV path): if YOLO finds nothing,
    # the page is not silently dropped — it contributes a full-page region.
    if not regions:
        h, w = image.shape[:2]
        crop_dir = ensure_dir(Path(page.image_path).parent / "regions")
        region_id = f"p{page.page_index:03d}_yolo_fullpage"
        crop_path = crop_dir / f"{region_id}.png"
        cv2.imwrite(str(crop_path), image)
        regions.append(
            FigureRegion(
                page_index=page.page_index,
                bbox=(0, 0, w, h),
                crop_path=str(crop_path),
                score=0.5,
                region_id=region_id,
                kind="page",
                metadata={"fallback": True, "detector": "yolo"},
            )
        )
    return regions


def page_text_density(page: PageRecord) -> float:
    text = (page.text or "").strip()
    if not text:
        return 0.0
    words = re.findall(r"\w+", text)
    return len(words) / max(1, page.width * page.height / 100000.0)


def choose_best_page(
    pages: list[PageRecord], figure_number: str | None, caption_text: str, window: int = 2
) -> PageRecord | None:
    if not pages:
        return None
    candidates = find_caption_pages(pages, figure_number, window=window)
    if candidates:
        # Phase 59 (Bug 2.7): rank candidates by plate-region score
        # (lowest text density = highest score) and return the best.
        # The previous code returned ``candidates[0]`` (always the
        # first match), which on figure-heavy plates where the same
        # "Fig. N" caption text repeats across adjacent pages picked
        # the page with the densest text — the worst plate page.
        return max(candidates, key=lambda p: -page_text_density(p))
    # Fallback to pages around where the figure number first appears in the text.
    if figure_number:
        for i, page in enumerate(pages):
            if re.search(rf"\b{re.escape(figure_number)}\b", page.text or ""):
                return page
    # Otherwise choose the page with lowest text density among pages near the caption text.
    if not pages:
        return None
    return min(pages, key=page_text_density)
