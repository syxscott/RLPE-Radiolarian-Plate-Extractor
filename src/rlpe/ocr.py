from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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

    def _lazy_init(self):
        if self._engine is not None:
            return self._engine
        if self.backend == "paddleocr":
            try:
                from paddleocr import PaddleOCR

                self._engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=self.use_gpu)
                return self._engine
            except Exception:
                self.backend = "easyocr"
        if self.backend == "easyocr":
            try:
                import easyocr

                self._engine = easyocr.Reader(["en"], gpu=self.use_gpu)
                return self._engine
            except Exception:
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


def normalize_ocr_tokens(tokens: list[OCRToken]) -> list[OCRToken]:
    out: list[OCRToken] = []
    for tok in tokens:
        text = tok.text.strip()
        if not text:
            continue
        out.append(OCRToken(text=text, confidence=tok.confidence, bbox=tok.bbox, metadata=tok.metadata))
    return out
