from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import cv2

from .types import FigureRegion, PageRecord
from .utils import ensure_dir, slugify

FIG_REF_PATTERN = re.compile(r"\b(?:fig(?:ure)?|plate)\s*\.?\s*(\d+[A-Za-z]?)\b", re.IGNORECASE)
CAPTION_LEAD_PATTERN = re.compile(
    r"^(?:fig(?:ure)?|plate)\s*\.?\s*(\d+[A-Za-z]?)\b[:\-\.]?\s*", re.IGNORECASE
)
# Phase X: pattern for detecting plate-related keywords in caption text or page text.
_PLATE_KEYWORD_RE = re.compile(
    r"\b(?:plate|pl\.?|figure\s*(?:plate|section)|图版|图版说明)\b",
    re.IGNORECASE,
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


def _import_pymupdf() -> Any:
    """Import PyMuPDF safely and avoid the unrelated `fitz` package collision."""
    try:
        import pymupdf as fitz  # PyMuPDF>=1.24 preferred import

        return fitz
    except Exception:
        pass

    try:
        # Re-import in fallback try block after pymupdf import failed
        # above. Mypy sees the second ``import fitz`` as a name
        # redefinition because the upper try introduced ``fitz`` as
        # a local alias for ``pymupdf``; ruff sees it as a duplicate
        # import (F811). The runtime behaviour is correct: the upper
        # try re-raised before reaching here so ``fitz`` is unbound
        # when this block runs. ``type: ignore[no-redef]`` on the
        # import line silences mypy; ``noqa: F811`` silences ruff.
        import fitz  # type: ignore[no-redef]  # noqa: F811

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
    yolo_device: str = "auto",
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
            device=yolo_device,
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
        # audit 2026-07-27 B1: check imwrite return AND catch cv2.error
        # (C-level I/O errors raise cv2.error, not Python exceptions).
        try:
            if not cv2.imwrite(str(crop_path), crop):
                raise RuntimeError(f"cv2.imwrite returned False for {crop_path}")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to write crop %s: %s; skipping this region", crop_path, exc
            )
            continue
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
        # audit 2026-07-27 B1: same imwrite guard for fullpage fallback.
        try:
            if not cv2.imwrite(str(crop_path), image):
                raise RuntimeError(f"cv2.imwrite returned False for {crop_path}")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to write fullpage crop %s: %s; proceeding without crop",
                crop_path,
                exc,
            )
            # Still emit a region with a None-ish crop_path so the page
            # is not silently dropped; downstream can decide how to handle it.
            regions.append(
                FigureRegion(
                    page_index=page.page_index,
                    bbox=(0, 0, w, h),
                    crop_path="",
                    score=0.0,
                    region_id=region_id,
                    kind="page",
                    metadata={"fallback": True, "crop_write_failed": True},
                )
            )
            return regions
        regions.append(
            FigureRegion(
                page_index=page.page_index,
                bbox=(0, 0, w, h),
                crop_path=str(crop_path),
                score=0.0,
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
    *,
    device: str = "auto",
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
    # Guard against invalid min_area values.
    if min_area <= 0:
        min_area = 5000
    image_path = Path(page.image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    # PyMuPDF may save RGBA PNGs; convert to BGR before any processing so
    # the YOLO inference and saved crops have correct colour ordering.
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    h_img, w_img = image.shape[:2]

    # Lazy-load the YOLO model (loaded once per session, cached on the
    # function object so repeated calls on the same path reuse it).
    # audit 2026-07-27 M2: use a lock so two concurrent threads don't
    # both see hasattr=False and both load the model simultaneously.
    # The lock is a class-level sentinel so it persists across calls.
    _lock_attr = "_yolo_load_lock"
    if not hasattr(detect_figure_regions_yolo, _lock_attr):
        setattr(detect_figure_regions_yolo, _lock_attr, threading.Lock())

    # audit 2026-07-27 B4: use .resolve() so that
    # models/yolo.pt, ./models/yolo.pt, and symlinked paths all
    # normalise to the same cache entry.
    cache_key = f"_yolo_model_{Path(model_path).resolve()}"
    with getattr(detect_figure_regions_yolo, _lock_attr):
        if not hasattr(detect_figure_regions_yolo, cache_key):
            # audit 2026-07-27 B2: guard YOLO() constructor for corrupt .pt.
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO figure detection requires the `ultralytics` "
                    "package. Install it with `pip install ultralytics` "
                    "or disable YOLO in Settings."
                ) from exc
            try:
                # Audit 2026-09-07: YOLO() does not accept a ``device``
                # constructor kwarg in ultralytics >=8.3 (inference device
                # is selected per-predict call). The old
                # ``YOLO(path, device=...)`` raised
                # "unexpected keyword argument 'device'" on every load.
                # The TypeError fallback stays for older ultralytics
                # whose stubs accepted the kwarg; mypy is silenced since
                # the installed stubs reject it but old versions need it.
                try:
                    _loaded_model = YOLO(str(model_path))  # type: ignore[call-arg]
                except TypeError:
                    _loaded_model = YOLO(str(model_path), device=device)  # type: ignore[call-arg]
            except Exception as exc:
                raise RuntimeError(
                    f"YOLO model failed to load from {model_path}: {exc}. "
                    "The model file may be corrupted or incomplete; "
                    "please re-download it."
                ) from exc
            # audit 2026-07-27 warmup fix: use a small dummy image instead
            # of the real page image so we don't double-infer on the first
            # actual page (and the warmup image is always fast).
            try:
                import numpy as np

                _dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                _loaded_model(_dummy, verbose=False)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "YOLO warmup failed (%s); continuing without warmup", exc
                )
            # Audit 2026-08-02 (Fix 1-B2 runtime check): if the loaded model
            # exposes class 0 as 'person', it's clearly a COCO model, not a
            # radiolarian-trained one. Warn loudly (don't raise — backward
            # compat for any non-COCO users).
            try:
                names = getattr(_loaded_model, "names", None) or {}
                if isinstance(names, dict) and str(names.get(0, "")).lower() in {
                    "person",
                    "bicycle",
                    "car",
                }:
                    import logging

                    logging.getLogger(__name__).warning(
                        "YOLO model at %r exposes COCO classes (class 0=%r) — "
                        "this is NOT a radiolarian-trained detector. Results "
                        "will be unreliable. Train a domain-specific .pt or "
                        "disable YOLO.",
                        str(model_path),
                        names.get(0),
                    )
            except Exception:
                pass  # don't fail the load just because class-name check failed
            setattr(detect_figure_regions_yolo, cache_key, _loaded_model)

    model: Any = getattr(detect_figure_regions_yolo, cache_key)

    # audit 2026-07-27 M1: inference exception should return the fullpage
    # fallback region, not []. Returning [] silently drops the page from
    # the output; the fullpage fallback preserves the page with a
    # kind="page" region so downstream caption routing can still match it.
    #
    # Audit 2026-09-01 (systemic #1 — YOLO half): ultralytics ``YOLO``
    # shares a CUDA context + model weights across instances; concurrent
    # inference is **not** thread-safe (no exception, but bounding boxes
    # can be attributed to the wrong page). Add a separate inference
    # lock so the load lock above can be released for the duration of the
    # (slow) inference. Lock contention is negligible vs. GPU time.
    _infer_lock_attr = "_yolo_infer_lock"
    if not hasattr(detect_figure_regions_yolo, _infer_lock_attr):
        setattr(detect_figure_regions_yolo, _infer_lock_attr, threading.Lock())
    try:
        with getattr(detect_figure_regions_yolo, _infer_lock_attr):
            results = model(image_path, verbose=False, conf=conf, iou=iou)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "YOLO inference failed on %s (%s); using fullpage fallback",
            image_path,
            exc,
        )
        h, w = image.shape[:2]
        crop_dir = ensure_dir(image_path.parent / "regions")
        region_id = f"p{page.page_index:03d}_yolo_fullpage"
        crop_path = crop_dir / f"{region_id}.png"
        # audit 2026-07-27 B1: same imwrite guard for fullpage.
        try:
            if not cv2.imwrite(str(crop_path), image):
                raise RuntimeError(f"cv2.imwrite returned False for {crop_path}")
        except Exception as write_exc:
            logging.getLogger(__name__).warning(
                "Failed to write YOLO fullpage crop %s: %s; proceeding without crop",
                crop_path,
                write_exc,
            )
            return [
                FigureRegion(
                    page_index=page.page_index,
                    bbox=(0, 0, w, h),
                    crop_path="",
                    score=0.0,
                    region_id=region_id,
                    kind="page",
                    metadata={"fallback": True, "detector": "yolo", "crop_write_failed": True},
                )
            ]
        return [
            FigureRegion(
                page_index=page.page_index,
                bbox=(0, 0, w, h),
                crop_path=str(crop_path),
                score=0.0,
                region_id=region_id,
                kind="page",
                metadata={"fallback": True, "detector": "yolo"},
            )
        ]

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
            # audit 2026-07-27 B1: same imwrite guard for per-detection crops.
            try:
                if not cv2.imwrite(str(crop_path), crop):
                    raise RuntimeError(f"cv2.imwrite returned False for {crop_path}")
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to write YOLO crop %s: %s; skipping region",
                    crop_path,
                    exc,
                )
                continue
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
    # Full-page fallback: if YOLO finds nothing, the page is not silently
    # dropped — it contributes a full-page region.
    if not regions:
        h, w = image.shape[:2]
        crop_dir = ensure_dir(image_path.parent / "regions")
        region_id = f"p{page.page_index:03d}_yolo_fullpage"
        crop_path = crop_dir / f"{region_id}.png"
        # audit 2026-07-27 B1: imwrite guard for YOLO fullpage fallback.
        try:
            if not cv2.imwrite(str(crop_path), image):
                raise RuntimeError(f"cv2.imwrite returned False for {crop_path}")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to write YOLO fullpage crop %s: %s; proceeding without crop",
                crop_path,
                exc,
            )
            regions.append(
                FigureRegion(
                    page_index=page.page_index,
                    bbox=(0, 0, w, h),
                    crop_path="",
                    score=0.0,
                    region_id=region_id,
                    kind="page",
                    metadata={"fallback": True, "detector": "yolo", "crop_write_failed": True},
                )
            )
            return regions
        regions.append(
            FigureRegion(
                page_index=page.page_index,
                bbox=(0, 0, w, h),
                crop_path=str(crop_path),
                score=0.0,
                region_id=region_id,
                kind="page",
                metadata={"fallback": True, "detector": "yolo"},
            )
        )
    return regions


def is_likely_plate_page(page: PageRecord) -> bool:
    """Return True if this page looks like a集中图版页 (末尾大图页).

    Plate pages typically contain keywords like "Plate", "pl.", and
    have relatively little body text compared to discussion pages.
    """
    text = page.text or ""
    return bool(_PLATE_KEYWORD_RE.search(text))


def find_plate_pages(pages: list[PageRecord]) -> list[PageRecord]:
    """Return pages in the second half of the document that look like集中图版页.

    Searches only the back half of the paper because radiolarian plates
    are conventionally placed at the end of an article.  Returns pages
    sorted by page_index (earliest plate pages first).
    """
    if not pages:
        return []
    mid = len(pages) // 2
    return sorted(
        [p for p in pages[mid:] if is_likely_plate_page(p)],
        key=lambda p: p.page_index,
    )


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
    # audit 2026-07-26: caption_text was previously ignored; use it to
    # pick a page whose text overlaps the caption before falling back to
    # the lowest-density page. (The `if not pages: return None` that was
    # here was dead - already guarded at the top of the function.)
    if caption_text:
        cap_tokens = [t for t in re.findall(r"\w+", caption_text) if len(t) > 3]
        for page in pages:
            pt = (page.text or "").lower()
            if any(t.lower() in pt for t in cap_tokens[:5]):
                return page
    # Phase X: caption mentions a plate keyword but every normal candidate
    # has already been exhausted — the figure is likely on a、集中图版页
    # at the end of the document.  Return the first plate page as a fallback.
    if caption_text and _PLATE_KEYWORD_RE.search(caption_text):
        plate_pages = find_plate_pages(pages)
        if plate_pages:
            return plate_pages[0]
    return min(pages, key=page_text_density)
