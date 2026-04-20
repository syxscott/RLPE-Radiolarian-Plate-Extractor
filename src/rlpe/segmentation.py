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
    grid_size: int = 6
    max_point_prompts: int = 48
    max_box_prompts: int = 24
    dedup_iou_threshold: float = 0.7


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
            point_prompts, box_prompts = self._generate_sam2_prompts(image)
            candidates: list[PanelCandidate] = []

            # 1) 点提示：更适合细碎目标。
            for x, y in point_prompts[: self.config.max_point_prompts]:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[x, y]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=True,
                )
                candidates.extend(self._masks_to_candidates(masks, scores, method="sam2-point"))

            # 2) 框提示：提升对整块panel区域的召回。
            for box in box_prompts[: self.config.max_box_prompts]:
                masks, scores, _ = predictor.predict(
                    box=np.array(box, dtype=np.float32),
                    multimask_output=True,
                )
                candidates.extend(self._masks_to_candidates(masks, scores, method="sam2-box"))

            candidates = self._deduplicate_candidates(candidates)
            candidates.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
            return candidates or self._segment_with_opencv(image)
        except Exception:
            return self._segment_with_opencv(image)

    def _masks_to_candidates(self, masks: np.ndarray, scores: np.ndarray, method: str) -> list[PanelCandidate]:
        out: list[PanelCandidate] = []
        for mask, score in zip(masks, scores):
            score_f = float(score)
            if score_f < self.config.score_threshold:
                continue
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            w, h = x1 - x0, y1 - y0
            if w * h < self.config.min_area:
                continue
            out.append(
                PanelCandidate(
                    panel_id=None,
                    bbox=(x0, y0, w, h),
                    score=score_f,
                    metadata={"method": method},
                )
            )
        return out

    def _generate_sam2_prompts(self, image: np.ndarray) -> tuple[list[tuple[float, float]], list[tuple[float, float, float, float]]]:
        h, w = image.shape[:2]

        # A. 连通域中心点与外接框（针对密集碎片的高召回提示）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(th) > 127:
            th = cv2.bitwise_not(th)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)

        point_prompts: list[tuple[float, float]] = []
        box_prompts: list[tuple[float, float, float, float]] = []
        img_area = h * w
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < self.config.min_area * 0.5:
                continue
            if area > img_area * 0.95:
                continue
            cx, cy = centroids[i]
            point_prompts.append((float(cx), float(cy)))
            box_prompts.append((float(x), float(y), float(x + bw), float(y + bh)))

        # B. 自适应网格点（弥补连通域漏检）
        g = max(2, int(self.config.grid_size))
        for gy in range(1, g + 1):
            for gx in range(1, g + 1):
                px = w * gx / (g + 1)
                py = h * gy / (g + 1)
                point_prompts.append((float(px), float(py)))

        # 去重
        point_prompts = self._dedup_points(point_prompts)
        box_prompts = self._dedup_boxes(box_prompts)
        return point_prompts, box_prompts

    def _deduplicate_candidates(self, candidates: list[PanelCandidate]) -> list[PanelCandidate]:
        if not candidates:
            return []
        kept: list[PanelCandidate] = []
        for c in sorted(candidates, key=lambda x: x.score, reverse=True):
            if any(self._iou(c.bbox, k.bbox) >= self.config.dedup_iou_threshold for k in kept):
                continue
            kept.append(c)
        return kept

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / max(1, union)

    @staticmethod
    def _dedup_points(points: list[tuple[float, float]], eps: float = 8.0) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in points:
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > eps * eps for q in out):
                out.append(p)
        return out

    @staticmethod
    def _dedup_boxes(boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
        out: list[tuple[float, float, float, float]] = []
        seen = set()
        for b in boxes:
            key = tuple(int(v // 4) for v in b)
            if key in seen:
                continue
            seen.add(key)
            out.append(b)
        return out

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
