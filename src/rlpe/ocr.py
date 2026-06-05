from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OCRToken:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    metadata: dict[str, Any] | None = None


class OCRBackend:
    def __init__(self, backend: str = "paddleocr", use_gpu: bool = True) -> None:
        self.backend = backend.lower()
        self.use_gpu = use_gpu
        self._engine = None
        self._lock = threading.Lock()

    def _lazy_init(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            if self.backend == "paddleocr":
                try:
                    from paddleocr import PaddleOCR

                    self._engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=self.use_gpu)
                    return self._engine
                except Exception:
                    logger.warning("PaddleOCR init failed; falling back to EasyOCR")
                    self.backend = "easyocr"
            if self.backend == "easyocr":
                try:
                    import easyocr

                    self._engine = easyocr.Reader(["en"], gpu=self.use_gpu)
                    return self._engine
                except Exception:
                    logger.warning("EasyOCR init failed; OCR disabled")
                    self.backend = None  # prevent repeated retries
                    self._engine = None
        return self._engine

    def recognize(self, image: np.ndarray | str | Path) -> list[OCRToken]:
        engine = self._lazy_init()
        if engine is None:
            return []

        if isinstance(image, (str, Path)):
            import cv2

            image = cv2.imread(str(image))
        if image is None:
            return []

        return self._ocr_array(image)

    def _ocr_array(self, image: np.ndarray) -> list[OCRToken]:
        engine = self._engine  # guaranteed non-None by caller
        tokens: list[OCRToken] = []
        try:
            if self.backend == "paddleocr":
                result = engine.ocr(image, cls=True)
                for line in result[0] if result and result[0] else []:
                    box, (text, conf) = line
                    x = min(p[0] for p in box)
                    y = min(p[1] for p in box)
                    w = max(p[0] for p in box) - x
                    h = max(p[1] for p in box) - y
                    tokens.append(OCRToken(text=text, confidence=float(conf), bbox=(int(x), int(y), int(w), int(h))))
            else:
                result = engine.readtext(image)
                for box, text, conf in result:
                    x = min(p[0] for p in box)
                    y = min(p[1] for p in box)
                    w = max(p[0] for p in box) - x
                    h = max(p[1] for p in box) - y
                    tokens.append(OCRToken(text=text, confidence=float(conf), bbox=(int(x), int(y), int(w), int(h))))
        except Exception:
            return []
        return tokens

    def recognize_panel(
        self,
        image: np.ndarray | str | Path,
        bbox: tuple[int, int, int, int],
        padding: int = 4,
    ) -> list[OCRToken]:
        """OCR a single panel sub-region.

        ``bbox`` is ``(x, y, w, h)`` in ``image`` pixel coordinates. The crop
        is padded by ``padding`` pixels on every side (default 4) so the OCR
        engine has a small margin to work with, and tokens are returned with
        their bbox translated back to ``image`` coordinates.
        """
        engine = self._lazy_init()
        if engine is None:
            return []
        if isinstance(image, (str, Path)):
            import cv2
            image = cv2.imread(str(image))
        if image is None:
            return []
        import cv2
        h_img, w_img = image.shape[:2]
        x, y, w, h = bbox
        x0 = max(0, int(x) - padding)
        y0 = max(0, int(y) - padding)
        x1 = min(w_img, int(x + w) + padding)
        y1 = min(h_img, int(y + h) + padding)
        if x1 <= x0 or y1 <= y0:
            return []
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        local_tokens = self._ocr_array(crop)
        # Translate token bboxes back to image coordinates
        out: list[OCRToken] = []
        for tok in local_tokens:
            tx, ty, tw, th = tok.bbox
            out.append(
                OCRToken(
                    text=tok.text,
                    confidence=tok.confidence,
                    bbox=(x0 + tx, y0 + ty, tw, th),
                )
            )
        return out


def normalize_ocr_tokens(tokens: list[OCRToken]) -> list[OCRToken]:
    out: list[OCRToken] = []
    for tok in tokens:
        text = tok.text.strip()
        if not text:
            continue
        out.append(OCRToken(text=text, confidence=tok.confidence, bbox=tok.bbox, metadata=tok.metadata))
    return out
