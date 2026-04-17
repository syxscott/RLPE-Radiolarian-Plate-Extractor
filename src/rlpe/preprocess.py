from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray | None:
    return cv2.imread(str(image_path))


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
