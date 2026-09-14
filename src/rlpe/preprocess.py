from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Unicode-safe :func:`cv2.imread`.

    2026-09-14 (Ark 12s-failure root cause): ``cv2.imread`` on Windows
    opens the file with the ANSI locale codec, so any non-ASCII path
    component (Chinese output directories, ``C:\\Users\\<中文名>\\``, …)
    silently returns ``None`` — every plate image became "unreadable"
    and whole papers collapsed to 0 rows. ``np.fromfile`` goes through
    Python's own file API, so round-tripping the bytes through
    ``cv2.imdecode`` reads any path OpenCV itself cannot.
    """
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray, params: list[int] | None = None) -> bool:
    """Unicode-safe :func:`cv2.imwrite` (same Windows ANSI-path issue as
    :func:`imread_unicode`; ``cv2.imwrite`` silently fails for non-ASCII
    paths). Returns True when the encoded bytes were written.

    ``params`` is cv2's imwrite parameter list, e.g.
    ``[cv2.IMWRITE_PNG_COMPRESSION, 3]``.
    """
    path = Path(path)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image, params or [])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


def load_image(image_path: str | Path) -> np.ndarray | None:
    return imread_unicode(image_path)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(image: np.ndarray) -> np.ndarray:
    gray = enhance_contrast(image)
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def binarize(image: np.ndarray) -> np.ndarray:
    gray = denoise(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def resize_keep_ratio(image: np.ndarray, max_side: int = 2000) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def preprocess_for_ocr(image_path: str | Path) -> np.ndarray | None:
    image = load_image(image_path)
    if image is None:
        return None
    image = resize_keep_ratio(image)
    return binarize(image)


def crop_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return image[y : y + h, x : x + w]
