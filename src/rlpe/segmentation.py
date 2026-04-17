from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import PanelCandidate


@dataclass(slots=True)
class SegmentationConfig:
    min_area: int = 2500
    score_threshold: float = 0.8
    use_sam2: bool = True


class PanelSegmenter:
    def __init__(self, config: SegmentationConfig | None = None, checkpoint: str | None = None, model_cfg: str | None = None) -> None:
        self.config = config or SegmentationConfig()
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self._predictor = None

    def _lazy_init_sam2(self):
        if self._predictor is not None:
            return self._predictor
        if not self.config.use_sam2:
            return None
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(self.model_cfg or "sam2_hiera_l.yaml", self.checkpoint or "sam2_hiera_large.pt", device="cuda")
            self._predictor = SAM2ImagePredictor(model)
        except Exception:
            self._predictor = None
        return self._predictor

    def segment(self, image_path: str | Path) -> list[PanelCandidate]:
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        return self.segment_image(image)

    def segment_image(self, image: np.ndarray) -> list[PanelCandidate]:
        predictor = self._lazy_init_sam2()
        if predictor is not None:
            return self._segment_with_sam2(image, predictor)
        return self._segment_with_opencv(image)

    def _segment_with_sam2(self, image: np.ndarray, predictor) -> list[PanelCandidate]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            predictor.set_image(rgb)
            h, w = rgb.shape[:2]
            points = np.array([[w * 0.25, h * 0.25], [w * 0.75, h * 0.25], [w * 0.25, h * 0.75], [w * 0.75, h * 0.75]])
            labels = np.array([1, 1, 1, 1])
            masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
            candidates: list[PanelCandidate] = []
            for mask, score in zip(masks, scores):
                if float(score) < self.config.score_threshold:
                    continue
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                candidates.append(PanelCandidate(panel_id=None, bbox=(x0, y0, x1 - x0, y1 - y0), score=float(score), metadata={"method": "sam2"}))
            return candidates or self._segment_with_opencv(image)
        except Exception:
            return self._segment_with_opencv(image)

    def _segment_with_opencv(self, image: np.ndarray) -> list[PanelCandidate]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        candidates: list[PanelCandidate] = []
        img_area = image.shape[0] * image.shape[1]
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < self.config.min_area:
                continue
            if area > img_area * 0.95:
                continue
            candidates.append(PanelCandidate(panel_id=None, bbox=(int(x), int(y), int(w), int(h)), score=min(1.0, area / img_area), metadata={"method": "opencv"}))
        candidates.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
        return candidates
