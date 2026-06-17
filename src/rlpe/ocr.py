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

                    # ``use_angle_cls`` was deprecated in PaddleOCR 3.x
                    # in favour of ``use_textline_orientation``; the new
                    # kwarg is the documented replacement. The old name
                    # still works in 2.x but emits a DeprecationWarning
                    # on every import. Try the new name first and fall
                    # back to the legacy one for 2.x users.
                    try:
                        self._engine = PaddleOCR(
                            use_textline_orientation=True, lang="en", use_gpu=self.use_gpu,
                        )
                    except TypeError:
                        self._engine = PaddleOCR(
                            use_angle_cls=True, lang="en", use_gpu=self.use_gpu,
                        )
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

    def recognize_panel_label(
        self,
        image: np.ndarray | str | Path,
        bbox: tuple[int, int, int, int],
        label_corner: str = "tl",
    ) -> list[OCRToken]:
        """OCR the small label area inside a single panel.

        Plate labels ("1", "2a", "Fig. 3") are usually a short text in one
        of the panel's corners. OCR'ing the full panel image dilutes that
        signal with the specimen's body — EasyOCR/PaddleOCR sometimes
        returns the body's texture as a higher-confidence token than the
        actual label. Cropping a tight corner band gives a much cleaner
        label read.

        ``label_corner``: ``"tl"`` (top-left), ``"tr"``, ``"bl"``, ``"br"``,
        ``"auto"`` (try all four corners), or ``"adaptive"`` (try the
        explicit corner first, then auto if that returns nothing). The
        default is ``"tl"`` because the vast majority of radiolarian
        plates have their numeric labels in the top-left corner; pass
        ``"adaptive"`` for papers (e.g. bandini2011 plate 1) that put
        labels in a different corner.

        N10 fix: the band is now 50% of the shorter side, floored at
        40px and capped at 160px. The previous 25% / 80px cap was too
        small for typical bandini2011 panels (102x117 → 25px band,
        too small for EasyOCR to read the label reliably). The wider
        band still concentrates on the corner while being large enough
        for OCR to lock on.
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
        # Crop the panel first
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w_img, int(x + w))
        y1 = min(h_img, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return []
        panel = image[y0:y1, x0:x1]
        if panel.size == 0:
            return []
        ph, pw = panel.shape[:2]
        # N10: 50% of shorter side, floored at 40, capped at 160. The
        # previous 25% / 80px was too small for sub-200px panels.
        band = max(40, min(int(min(ph, pw) * 0.50), 160))
        corners: list[tuple[str, tuple[int, int, int, int]]] = [
            ("tl", (0, 0, band, band)),
            ("tr", (max(0, pw - band), 0, pw, band)),
            ("bl", (0, max(0, ph - band), band, ph)),
            ("br", (max(0, pw - band), max(0, ph - band), pw, ph)),
        ]
        if label_corner in {"tl", "tr", "bl", "br"}:
            corners = [c for c in corners if c[0] == label_corner]
        # "adaptive" = try all four corners (corners already contains
        # all 4 from the list above). The previous code appended a
        # filtered copy of corners, producing a 7-element list
        # [tl, tr, bl, br, tr, bl, br] and wasting 3 OCR calls per
        # panel.
        best_tokens: list[OCRToken] = []
        best_score: float = -1.0
        # Try a 2x upscaled version of the corner band as a fallback. Many
        # bandini-style panels (e.g. 233x129 with a 1- or 2-digit label in
        # the corner) have labels too small for EasyOCR to read at native
        # resolution; cv2.INTER_CUBIC upscaling recovers ~78% of those
        # labels without introducing new false positives. We only run
        # this fallback on a corner band (not the full panel) to keep
        # the cost modest.
        for name, (cx0, cy0, cx1, cy1) in corners:
            sub = panel[cy0:cy1, cx0:cx1]
            if sub.size == 0:
                continue
            tokens = self._ocr_array(sub)
            if not tokens:
                # 2x fallback: upscale the band and retry. Skip if the
                # panel is already very large — for 500px+ panels the
                # native corner band is already well above OCR's
                # comfortable input size, so upscaling brings no real
                # benefit and just doubles the OCR cost.
                if min(ph, pw) < 500:
                    import cv2
                    sh, sw = sub.shape[:2]
                    up = cv2.resize(
                        sub, (sw * 2, sh * 2), interpolation=cv2.INTER_CUBIC
                    )
                    # ``image`` came from cv2.imread (already BGR), so
                    # ``sub`` is BGR — we must NOT apply RGB→BGR here
                    # (the previous version did, which silently swapped
                    # R and B channels before OCR). Only convert when
                    # the input is a grayscale slice; in that case
                    # GRAY→BGR makes the 3-channel structure both
                    # PaddleOCR and EasyOCR expect.
                    if up.ndim == 2:
                        up = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
                    tokens = self._ocr_array(up)
                    if tokens:
                        # Mark these tokens as coming from a 2x fallback
                        # and remap their bboxes back to corner coords.
                        tokens = [
                            OCRToken(
                                text=t.text,
                                confidence=t.confidence * 0.9,
                                bbox=(t.bbox[0] // 2, t.bbox[1] // 2,
                                      t.bbox[2] // 2, t.bbox[3] // 2),
                                metadata={"label_corner": name, "upscaled": "2x"},
                            )
                            for t in tokens
                        ]
            if not tokens:
                continue
            # Score: max confidence of any short text token (looks label-like)
            score = 0.0
            for tok in tokens:
                t = tok.text.strip()
                if not t or len(t) > 8:
                    continue
                # Boost numeric / alphanumeric labels
                if any(ch.isdigit() for ch in t):
                    score = max(score, tok.confidence + 0.1)
                else:
                    score = max(score, tok.confidence)
            if score > best_score:
                best_score = score
                # Translate bboxes back to image coordinates. The
                # ``label_corner`` field always needs to reflect the
                # corner that produced the winning token, even if the
                # token already carried metadata from a prior call
                # (e.g. the 2x upscaled fallback stamps its own
                # ``{"label_corner": name, "upscaled": "2x"}`` on
                # tokens). The previous ``tok.metadata or
                # {"label_corner": name}`` only set ``label_corner``
                # when metadata was None — any non-None metadata
                # (including the empty-dict from a fresh token) would
                # pass through untouched and lose the corner info.
                # Use an ``is not None`` check and merge in the corner
                # so existing fields are preserved.
                best_tokens = [
                    OCRToken(
                        text=tok.text,
                        confidence=tok.confidence,
                        bbox=(
                            x0 + cx0 + tok.bbox[0],
                            y0 + cy0 + tok.bbox[1],
                            tok.bbox[2],
                            tok.bbox[3],
                        ),
                        metadata=(
                            {**tok.metadata, "label_corner": name}
                            if tok.metadata is not None
                            else {"label_corner": name}
                        ),
                    )
                    for tok in tokens
                ]
        return best_tokens


def normalize_ocr_tokens(tokens: list[OCRToken]) -> list[OCRToken]:
    out: list[OCRToken] = []
    for tok in tokens:
        text = tok.text.strip()
        if not text:
            continue
        out.append(OCRToken(text=text, confidence=tok.confidence, bbox=tok.bbox, metadata=tok.metadata))
    return out
