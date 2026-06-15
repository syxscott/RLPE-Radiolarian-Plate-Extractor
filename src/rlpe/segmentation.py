from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

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
    # Reject connected components whose area is too large to be a
    # single specimen. On dense plates the morphological close in
    # ``_preprocess_enhanced`` merges the entire top half of the image
    # into one giant blob (Bandini 2011 plate 5: 975x507 covering 21
    # specimens). A real specimen is < ~20% of the image, so anything
    # bigger is a multi-specimen blob, not a single panel.
    max_single_panel_area_frac: float = 0.20
    # Reject very thin / wide strips. The enhanced path occasionally
    # returns the entire top row of a grid plate as a single 0.95 x
    # 0.05 strip; those aren't panels, they're "all the panels in
    # this row merged together".
    max_aspect_ratio: float = 4.0
    # Watershed splitting of multi-specimen CCs. After the Otsu /
    # enhanced branches produce their initial CCs, any CC whose
    # area exceeds ``min_area`` and whose area is "non-trivially
    # large" (e.g. >3x min_area) is passed through the watershed
    # splitter. A 3x3 dilate-based seed finder + distance transform
    # identifies local maxima as split seeds; cv2.watershed assigns
    # each non-boundary pixel to a seed. The result is multiple
    # sub-panels in place of the merged blob.
    use_watershed: bool = True
    # Lower bound for the *shape factor* of a CC before watershed
    # is invoked. CCs whose area is barely above ``min_area`` (i.e.
    # just a small isolated specimen, not a merged blob) are not
    # watershed candidates. Set to 3.0 = "if the CC is at least 3x
    # the min_area, try splitting it".
    watershed_area_multiple: float = 3.0
    # Minimum seed area for watershed split. Seeds smaller than
    # this are merged into their nearest neighbor (avoids 1-pixel
    # seeds that would over-segment).
    watershed_min_seed_area: int = 800


class PanelSegmenter:
    def __init__(self, config: SegmentationConfig | None = None, checkpoint: str | None = None, model_cfg: str | None = None) -> None:
        self.config = config or SegmentationConfig()
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self._predictor = None
        self._lock = threading.Lock()

    def _lazy_init_sam2(self):
        if self._predictor is not None:
            return self._predictor
        if not self.config.use_sam2:
            return None
        with self._lock:
            if self._predictor is not None:
                return self._predictor
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

    def _preprocess_gray(self, image: np.ndarray) -> np.ndarray:
        """Shared grayscale+blur+threshold preprocessing.

        Two paths are produced and the caller picks the better one:
          - ``th_otsu``   : plain Otsu (fast; works for sparse plates like pl01)
          - ``th_enhanced``: morphological open→close→erode + adaptive threshold
                             (handles dense plates like pl04 where specimens
                             touch, scale bars merge into the blob, and the
                             background has SEM metadata text strips)
        ``_segment_with_opencv`` runs both and merges the results so that
        dense plates aren't silently under-segmented.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.count_nonzero(th_otsu) > th_otsu.size // 2:
            th_otsu = cv2.bitwise_not(th_otsu)
        return th_otsu

    @staticmethod
    def _preprocess_enhanced(gray: np.ndarray) -> np.ndarray:
        """Morphological pipeline for dense plates with touching specimens.

        Steps (M3-suggested):
          1. Morphological OPEN with 5x5 rect → removes scale bars / labels
          2. Morphological CLOSE with 7x7 ellipse → fills lattice pores
             (reduced from 9x9: the larger kernel merged entire rows of
             hollis2006 pl03 into single CCs, dropping 3 panels per plate)
          3. 3x3 erode → breaks spine-to-spine bridges between specimens
          4. Adaptive Gaussian threshold (block=51, C=5) → handles the
             non-uniform background that defeats global Otsu
        """
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img_open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_open)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        img_closed = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel_close)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_eroded = cv2.erode(img_closed, kernel_erode, iterations=1)
        binary = cv2.adaptiveThreshold(
            img_eroded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 5,
        )
        return binary

    @staticmethod
    def _watershed_split_cc(
        gray: np.ndarray,
        binary: np.ndarray,
        cc_bbox: tuple[int, int, int, int],
        min_seed_area: int,
    ) -> list[tuple[int, int, int, int]]:
        """Split a single connected component via the watershed algorithm.

        Phase A.2 implementation: when the Otsu/enhanced branches return
        a CC that is large enough to be a merged-blob candidate (e.g. 2
        touching specimens that the morphology didn't break apart), the
        watershed algorithm can split it into sub-regions.

        Algorithm (4 steps, all standard watershed):
          1. Crop the binary mask to the CC's bounding box and zero the
             background.
          2. Compute the distance transform of the (cropped) binary
             mask. Local maxima of the distance transform are the
             "centers" of the constituent specimens.
          3. Find the seeds: dilate the distance transform with a 3x3
             kernel so neighbouring maxima collapse to a single point,
             threshold at 0.7 * max, then run connected components.
             Each CC of the thresholded map is a seed.
          4. Apply cv2.watershed with the seed markers (label 1 = first
             seed, label 2 = second seed, etc.). The watershed
             boundaries are marked with -1; positive labels are the
             sub-regions. The label-0 background (i.e. the area outside
             the original CC) is masked out before returning bboxes.

        Returns a list of (x, y, w, h) bboxes in the **original image's
        coordinate system** (not the cropped one — the caller wants
        global coordinates).
        """
        x0, y0, w, h = cc_bbox
        # Crop the binary mask to the CC's bounding box.
        crop_bin = binary[y0:y0 + h, x0:x0 + w]
        if crop_bin.size == 0 or not np.any(crop_bin):
            return []
        # Crop the grayscale image too — cv2.watershed needs a 3-channel
        # 8-bit image (the standard recipe is to convert grayscale to BGR).
        crop_gray = gray[y0:y0 + h, x0:x0 + w]
        if crop_gray.ndim == 2:
            crop_bgr = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
        else:
            crop_bgr = crop_gray

        # Distance transform. The "DIST_L2" mask gives Euclidean
        # distance to the nearest zero pixel; we use the binary CC
        # as the foreground mask.
        dist = cv2.distanceTransform(crop_bin, cv2.DIST_L2, 5)
        # Local-maxima detection. We use a "ridge-extraction"
        # approach: threshold the distance transform at 70% of its
        # max value to get the "high-ridge" region, then connected
        # components that ridge region to find seeds. A single
        # dist==dilate(dist) local-max test only gives 1-pixel peaks
        # because the dist values are smoothly varying; ridge
        # extraction gives a region of ~10-30 pixels per seed.
        dmax = float(dist.max())
        if dmax <= 0:
            return []
        ridge_mask = (dist > 0.7 * dmax).astype(np.uint8) * 255
        n_ridges, ridge_labels, ridge_stats, _ = cv2.connectedComponentsWithStats(
            ridge_mask, connectivity=8,
        )
        if n_ridges <= 1:
            return []  # no real split possible (only 1 ridge = same as 1 blob)
        # Use each ridge CC as a seed. Place the seed marker value
        # (ridge_idx + 1) at the ridge's pixels. CV2.watershed
        # convention:
        #   marker == 0  → unknown (watershed will assign to nearest seed)
        #   marker == 1  → seed 1
        #   marker == 2  → seed 2 (each unique positive value is one seed)
        # The watershed then EXPANDS each seed into the surrounding
        # unknown (marker == 0) region, drawing boundaries (marker == -1)
        # where two seeds meet.
        n_seeds = n_ridges
        seed_labels = ridge_labels
        seed_stats = ridge_stats
        peak_mask = ridge_mask > 0
        # Build a markers image: 0 for unknown (watershed to decide),
        # unique positive value per seed. Background pixels (crop_bin
        # == 0) and unknown foreground (not yet covered by a seed)
        # both start at 0.
        markers = np.zeros(crop_bin.shape, dtype=np.int32)
        for seed_idx in range(1, n_seeds):
            sx, sy, sw, sh, sa = seed_stats[seed_idx]
            if sa < min_seed_area:
                # Tiny seed: skip — the watershed boundary will eat
                # most of it and we'll over-segment.
                continue
            # Place this seed's marker value (seed_idx + 1) at the
            # ridge's pixels.
            markers[seed_labels == seed_idx] = seed_idx + 1
        # Apply watershed.
        cv2.watershed(crop_bgr, markers)
        # Collect sub-region bboxes from the post-watershed labels.
        sub_bboxes: list[tuple[int, int, int, int]] = []
        for label in range(2, n_seeds + 1):
            sub_mask = (markers == label)
            if not sub_mask.any():
                continue
            ys, xs = np.where(sub_mask)
            if len(xs) == 0:
                continue
            sx0, sx1 = int(xs.min()), int(xs.max() + 1)
            sy0, sy1 = int(ys.min()), int(ys.max() + 1)
            sub_bboxes.append((sx0 + x0, sy0 + y0, sx1 - sx0, sy1 - sy0))
        return sub_bboxes

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
        th = self._preprocess_gray(image)
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
        """OpenCV panel segmentation. Uses Otsu as the baseline and the
        M3-suggested morphology+adaptive path as a *supplement* (not a
        duplicate). Anything the enhanced path finds that the Otsu path
        missed is added, provided it doesn't overlap an Otsu panel
        (IoU < 0.1) — otherwise we'd double-count the same specimen.

        The enhanced path is the fix for plates where:
          - specimens touch (e.g. Plate 4 of Feng 2007)
          - scale bars / labels merge with the specimen blob
          - SEM metadata strips overlay the bottom row

        Phase A.2: watershed post-processing. Any "large" CC (area
        >= 3x min_area) that survives the area/aspect filters is
        passed through ``_watershed_split_cc`` and replaced by its
        sub-regions. The threshold of "3x min_area" is below the
        ``max_single_panel_area_frac`` (0.20) so this fires on
        moderate-size merged blobs, not just the giant ones the
        area filter already rejects.
        """
        thresh_otsu = self._preprocess_gray(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh_enh = self._preprocess_enhanced(gray)

        img_area = image.shape[0] * image.shape[1]

        def _ccs(thresh: np.ndarray, source: str) -> list[PanelCandidate]:
            out: list[PanelCandidate] = []
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                if area < self.config.min_area:
                    continue
                if area > img_area * 0.95:
                    continue
                if w * h > img_area * 0.9:
                    continue
                # Reject CCs that are clearly multi-specimen blobs (the
                # enhanced path's morphological close frequently merges
                # the top half of a dense plate into one giant CC).
                if area > img_area * self.config.max_single_panel_area_frac:
                    continue
                if w * h > 0 and max(w, h) > 0:
                    aspect = max(w, h) / max(1, min(w, h))
                    if aspect > self.config.max_aspect_ratio:
                        continue
                # Reject tiny fragments: radiolarian specimens in this corpus
                # are typically 80+ pixels on the short side. CCs smaller than
                # that on either axis are almost always text/labels, scale
                # bars, or partial-specimen noise (Plate 4 of Feng 2007
                # otherwise produces 10-15 such fragments from the enhanced
                # path). 80px is well below the smallest real specimen in
                # the test corpus (pl01 minimum short side is 175 px).
                if min(w, h) < 80:
                    continue
                out.append(
                    PanelCandidate(
                        panel_id=None,
                        bbox=(int(x), int(y), int(w), int(h)),
                        score=min(1.0, area / img_area),
                        metadata={"method": source},
                    )
                )
            return out

        otsu_panels = _ccs(thresh_otsu, "opencv-otsu")
        enh_panels = _ccs(thresh_enh, "opencv-enhanced")

        # Keep all Otsu panels (baseline) + only the enhanced panels that
        # don't overlap any Otsu panel (i.e. genuinely new specimens).
        accepted = list(otsu_panels)
        for ep in enh_panels:
            if all(self._iou(ep.bbox, op.bbox) < 0.1 for op in otsu_panels):
                accepted.append(ep)

        # Phase A.2: watershed split on "large" CCs. For each accepted
        # panel whose area is >= watershed_area_multiple * min_area, run
        # the watershed splitter. If it returns >=2 sub-regions, replace
        # the original panel with the sub-regions. The original is
        # always removed (a CC that we believe to be a merged blob is
        # almost certainly not a single specimen). The sub-regions are
        # re-filtered through the same area / aspect / min-short-side
        # checks as the initial CC pass, so a watershed that "splits"
        # a CC into two half-sized blobs that are still too large will
        # discard both halves (better to over-reject than to keep a
        # merged blob labeled as a panel).
        if self.config.use_watershed:
            new_accepted: list[PanelCandidate] = []
            for c in accepted:
                x, y, w, h = c.bbox
                area = w * h
                if area < self.config.min_area * self.config.watershed_area_multiple:
                    new_accepted.append(c)
                    continue
                # Run watershed on the binary threshold that originally
                # produced this panel. Fall back to the Otsu binary if
                # the source is the enhanced path (which is harder to
                # invert; the Otsu binary contains a superset of the
                # foreground pixels anyway).
                if c.metadata.get("method") == "opencv-otsu":
                    src_bin = thresh_otsu
                else:
                    src_bin = thresh_enh
                sub_bboxes = self._watershed_split_cc(
                    gray, src_bin, (int(x), int(y), int(w), int(h)),
                    self.config.watershed_min_seed_area,
                )
                if len(sub_bboxes) < 2:
                    # Watershed found 0 or 1 sub-regions — not a real
                    # split, keep the original panel as-is.
                    new_accepted.append(c)
                    continue
                # Replace the original panel with the sub-regions,
                # re-applying the area / aspect / min-short-side
                # filters. A "split" that produces two half-bigs is
                # better discarded than kept.
                sub_added = 0
                for sb in sub_bboxes:
                    sx, sy, sw, sh = sb
                    sa = sw * sh
                    if sa < self.config.min_area:
                        continue
                    if sa > img_area * self.config.max_single_panel_area_frac:
                        continue
                    if sw * sh > img_area * 0.9:
                        continue
                    aspect = max(sw, sh) / max(1, min(sw, sh))
                    if aspect > self.config.max_aspect_ratio:
                        continue
                    if min(sw, sh) < 80:
                        continue
                    new_accepted.append(PanelCandidate(
                        panel_id=None,
                        bbox=(int(sx), int(sy), int(sw), int(sh)),
                        score=min(1.0, sa / img_area),
                        metadata={"method": c.metadata.get("method", "opencv") + "+watershed"},
                    ))
                    sub_added += 1
                if sub_added == 0:
                    # All watershed sub-regions were rejected by the
                    # post-filter; keep the original (which is
                    # probably a noisy single specimen, not a merged
                    # blob).
                    new_accepted.append(c)
            accepted = new_accepted

        accepted.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
        return accepted
