from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


# Phase 61 Plan 4 (Bug 4.4): discriminated token type so downstream
# ``match_panels`` can tell panel-label tokens apart from species tokens
# (and from generic OCR noise). The default "other" preserves legacy
# behaviour — existing callers that ignore ``token_type`` keep working.
_OCRTokenType = Literal["label", "species", "other"]


@dataclass(slots=True)
class OCRToken:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    metadata: dict[str, Any] | None = None
    # Phase 61 Plan 4 (Bug 4.4): token type discriminator.
    #   "label"   - this token is a printed panel label ("1a", "Fig. 3").
    #               Stamped by ``recognize_panel_label()``.
    #   "species" - this token looks like a radiolarian taxon (binomial,
    #               "Genus cf. species", "Genus sp.", etc.). Stamped by
    #               ``extract_species_tokens()``.
    #   "other"   - generic OCR text; default for backward compatibility.
    token_type: _OCRTokenType = "other"


# Phase 27: map our internal short names to the engine-native spellings
# the PaddleOCR / EasyOCR packages expect. PaddleOCR uses "japan" and "ch"
# while EasyOCR uses "ja" and "ch_sim". We accept either form from the
# caller (CLI uses the short names; legacy callers pass the long names).
_PADDLE_LANG_MAP: dict[str, str] = {
    "ja": "japan",
    "ch_sim": "ch",
    "ch_tra": "chinese_cht",
    "zh": "ch",          # zh (Simplified Chinese) -> ch for PaddleOCR
    "de": "german",
    # Note: "en", "fr", "ko", "ru", "japan", "ch" all pass through
    # unchanged (PaddleOCR accepts those spellings natively).
}


class OCRBackend:
    # Whitelist of internal lang short names. Anything outside this set
    # is silently dropped at construction time (with a warning) so a
    # typo in --ocr-lang never crashes the pipeline.
    SUPPORTED_LANGS = {"en", "ja", "ch_sim", "ch_tra", "zh", "fr", "de", "ko", "ru"}

    def __init__(
        self,
        backend: str = "paddleocr",
        use_gpu: bool = True,
        lang: str | list[str] = "en",
    ) -> None:
        self.backend = backend.lower()
        self.use_gpu = use_gpu
        self.lang: list[str] = self._normalise_lang(lang)
        self._engine = None
        self._lock = threading.Lock()

    @staticmethod
    def _normalise_lang(lang: str | list[str]) -> list[str]:
        """Accept ``"en,ja"`` / ``["en","ja"]`` / ``"en"`` → ``["en","ja"]``.

        Unknown short names are dropped with a warning rather than raising,
        so a typo in --ocr-lang does not break the pipeline. An empty
        result falls back to ``["en"]`` to preserve legacy behaviour.
        """
        if isinstance(lang, str):
            langs = [s.strip() for s in lang.split(",") if s.strip()]
        else:
            langs = [str(s).strip() for s in lang]
        out: list[str] = []
        for l in langs:
            if l in OCRBackend.SUPPORTED_LANGS:
                out.append(l)
            else:
                logger.warning(
                    "OCRBackend: unknown OCR lang %r (supported: %s); ignoring",
                    l,
                    sorted(OCRBackend.SUPPORTED_LANGS),
                )
        return out or ["en"]

    def _paddle_lang(self) -> str:
        """First configured lang, mapped to PaddleOCR's spelling.

        PaddleOCR accepts only a single lang string; if the caller asked
        for multi-lang we use the first one and log a notice — EasyOCR is
        the right choice when multi-lang is required.
        """
        if len(self.lang) > 1:
            logger.info(
                "OCRBackend: PaddleOCR supports a single language; "
                "using first of %r",
                self.lang,
            )
        first = self.lang[0]
        mapped = _PADDLE_LANG_MAP.get(first, first)
        if mapped != first:
            logger.info("OCRBackend: PaddleOCR lang %r → %r", first, mapped)
        return mapped

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
                    # in favour of ``use_textline_orientation``. PaddleOCR
                    # 3.x also removed ``use_gpu`` in favour of ``device``
                    # (``"cpu"`` / ``"gpu"`` / ``"gpu:0"``). The legacy
                    # 2.x form takes ``use_gpu=bool``; passing it to 3.x
                    # raises ValueError. Pick the right kwarg name first,
                    # fall back to the legacy combo on TypeError.
                    device_kw = "cpu" if not self.use_gpu else "gpu"
                    paddle_lang = self._paddle_lang()
                    try:
                        self._engine = PaddleOCR(
                            use_textline_orientation=True,
                            lang=paddle_lang,
                            device=device_kw,
                        )
                    except TypeError:
                        # 2.x legacy form
                        self._engine = PaddleOCR(
                            use_angle_cls=True,
                            lang=paddle_lang,
                            use_gpu=self.use_gpu,
                        )
                    return self._engine
                except Exception as exc:
                    logger.warning(
                        "PaddleOCR init failed (%s: %s); falling back to EasyOCR",
                        type(exc).__name__,
                        exc,
                    )
                    self.backend = "easyocr"
            if self.backend == "easyocr":
                try:
                    import easyocr

                    # EasyOCR natively accepts a list of lang codes and
                    # handles multi-lang in one Reader (it downloads each
                    # model's weights on first use). Phase 27: pass the
                    # full configured lang list, not a hard-coded ["en"].
                    self._engine = easyocr.Reader(self.lang, gpu=self.use_gpu)
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
                # Phase 38: paddleocr 2.x returns [ [box, (text, conf)], ... ]
                # but paddleocr 3.x returns a different structure (a list
                # of dicts with 'rec_texts', 'rec_scores', 'dt_polys').
                # Handle both so the pipeline doesn't break when the
                # operator upgrades paddleocr.
                lines = self._normalize_paddle_result(result)
                for entry in lines:
                    box, text, conf = entry
                    if not text:
                        continue
                    if not box:
                        continue
                    x = min(p[0] for p in box)
                    y = min(p[1] for p in box)
                    w = max(p[0] for p in box) - x
                    h = max(p[1] for p in box) - y
                    tokens.append(
                        OCRToken(
                            text=text, confidence=float(conf), bbox=(int(x), int(y), int(w), int(h))
                        )
                    )
            else:
                result = engine.readtext(image)
                for box, text, conf in result:
                    x = min(p[0] for p in box)
                    y = min(p[1] for p in box)
                    w = max(p[0] for p in box) - x
                    h = max(p[1] for p in box) - y
                    tokens.append(
                        OCRToken(
                            text=text, confidence=float(conf), bbox=(int(x), int(y), int(w), int(h))
                        )
                    )
        except Exception:
            return []
        return tokens

    @staticmethod
    def _normalize_paddle_result(result: Any) -> list[tuple[list, str, float]]:
        """Phase 38: paddleocr 2.x vs 3.x compatibility.

        Paddleocr 2.x returns a tuple ``(list_of_lines, None)`` where
        each line is ``[box, (text, conf)]``:

            result = ([
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ("hello", 0.99)],
                ...,
            ], None)

        Paddleocr 3.x changed the return shape to a list of dicts
        (or a single dict):

            result = {
                "rec_texts": ["hello", "world"],
                "rec_scores": [0.99, 0.95],
                "dt_polys": [[[x1,y1],...], [[x1,y1],...]],
            }

        This helper unifies the two into a flat list of
        ``(box, text, conf)`` tuples. Empty / malformed entries are
        dropped.
        """
        out: list[tuple[list, str, float]] = []
        if not result:
            return out
        # Paddleocr 2.x: result is a tuple/list whose first element
        # is the lines list.
        if isinstance(result, (tuple, list)) and len(result) >= 1 and isinstance(result[0], list):
            for line in result[0]:
                try:
                    if isinstance(line, (list, tuple)) and len(line) == 2:
                        box, payload = line
                        if isinstance(payload, (list, tuple)) and len(payload) == 2:
                            text, conf = payload
                        else:
                            # Newer 2.x sometimes returns dict
                            text = payload.get("text", "")
                            conf = payload.get("score", 0.0)
                        out.append((box, str(text), float(conf)))
                except (ValueError, TypeError, AttributeError):
                    continue
            return out
        # Paddleocr 3.x: result is a dict (or list of dicts) with
        # 'rec_texts', 'rec_scores', 'dt_polys' / 'rec_boxes'.
        if isinstance(result, dict):
            rec_texts = result.get("rec_texts") or result.get("texts") or []
            rec_scores = result.get("rec_scores") or result.get("scores") or []
            polys = (
                result.get("dt_polys")
                or result.get("rec_boxes")
                or result.get("boxes")
                or []
            )
            for i, text in enumerate(rec_texts):
                if i >= len(polys):
                    break
                box = polys[i]
                # rec_boxes is a 4-corner flat list [x1,y1,x2,y2,...]
                # dt_polys is a list of 4 points [[x,y], ...].
                # Normalize to 4-point list-of-lists.
                if box and isinstance(box[0], (int, float)):
                    coords = list(box)
                    if len(coords) == 4:
                        box = [[coords[0], coords[1]], [coords[2], coords[1]],
                               [coords[2], coords[3]], [coords[0], coords[3]]]
                    # Phase 54 audit m2: paddleocr 2.x sometimes returns
                    # 8-element flat lists (the 4 corner coordinates
                    # in xy order) instead of 4-element [x, y, x+w, y+h]
                    # bounding boxes. Reassemble into the same 4-point
                    # shape the rest of the code expects.
                    elif len(coords) == 8:
                        box = [[coords[i], coords[i + 1]] for i in (0, 2, 4, 6)]
                conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                out.append((box, str(text), conf))
            return out
        # List of dicts
        if isinstance(result, list):
            for d in result:
                if isinstance(d, dict):
                    out.extend(OCRBackend._normalize_paddle_result(d))
            return out
        return out

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
        ``"auto"`` (try all four corners), or ``"adaptive"`` (same as
        ``"auto"`` — runs all four corners; the two-pass behaviour described
        in older documentation was never implemented). The
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
                    # cv2 already imported at function scope (line 334)
                    sh, sw = sub.shape[:2]
                    up = cv2.resize(sub, (sw * 2, sh * 2), interpolation=cv2.INTER_CUBIC)
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
                                bbox=(
                                    t.bbox[0] // 2,
                                    t.bbox[1] // 2,
                                    t.bbox[2] // 2,
                                    t.bbox[3] // 2,
                                ),
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
                        # Phase 61 Plan 4 (Bug 4.4): stamp every
                        # output of ``recognize_panel_label`` as a
                        # "label" token. The downstream matcher uses
                        # this to prefer label-shaped text over
                        # generic OCR noise when picking the panel id.
                        token_type="label",
                    )
                    for tok in tokens
                ]
        return best_tokens


def extract_species_tokens(tokens: list[OCRToken]) -> list[OCRToken]:
    """Return a NEW list of tokens whose text looks like a radiolarian taxon.

    Each token whose text matches a recognised taxon shape (binomial,
    "Genus cf. species", "Genus sp.", etc.) is stamped with
    ``token_type="species"`` and returned. Tokens that fail the check
    are passed through with their existing ``token_type`` unchanged
    (defaulting to "other") so callers can still filter on
    ``token_type != "species"`` for non-species OCR text.

    The check uses the existing ``_taxon_parts`` decomposition so the
    notion of "species-like" stays consistent with the data-package
    view (Phase 60).
    """
    out: list[OCRToken] = []
    try:
        from .converters import _taxon_parts
    except Exception:
        # If the converters module can't be imported we still pass
        # through tokens unchanged (downstream degrades to legacy
        # behaviour).
        return list(tokens)
    for tok in tokens:
        parts = _taxon_parts(tok.text) or {}
        genus = parts.get("genus")
        epithet = parts.get("specific_epithet")
        qualifier = parts.get("qualifier")
        # Recognise the same shapes _is_valid_species() accepts:
        #   * "Genus species"
        #   * "Genus cf. species"
        #   * "Genus sp." / "Genus spp."
        #   * "Genus indet"
        is_species = False
        if genus and epithet:
            is_species = True
        elif genus and qualifier:
            q = qualifier.strip().rstrip(".").lower()
            if q in {"sp", "spp", "indet", "gr", "group", "subsp", "var", "n", "nom", "cf", "aff"}:
                is_species = True
        if is_species:
            out.append(
                OCRToken(
                    text=tok.text,
                    confidence=tok.confidence,
                    bbox=tok.bbox,
                    metadata=tok.metadata,
                    token_type="species",
                )
            )
        else:
            out.append(tok)
    return out


def normalize_ocr_tokens(tokens: list[OCRToken]) -> list[OCRToken]:
    out: list[OCRToken] = []
    for tok in tokens:
        text = tok.text.strip()
        if not text:
            continue
        out.append(
            OCRToken(text=text, confidence=tok.confidence, bbox=tok.bbox, metadata=tok.metadata)
        )
    return out
