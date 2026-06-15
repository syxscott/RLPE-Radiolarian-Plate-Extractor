"""Re-segment the 7-paper predictions with k_close=7 to reflect the
segmentation.py tuning committed in c56559d.

Approach
--------
combined_7_v6.jsonl was produced by the pipeline with k_close=9. The
k_close=7 change affects which panels get detected on dense plates
(hollis2006 pl03: 16 → 19 panels). To evaluate this change without
re-running the entire pipeline (which takes hours), we:

  1. For each (paper_id, figure_id), determine the rendered figure image
     by matching existing prediction bboxes against image dimensions.
  2. Run PanelSegmenter (k_close=7) on the matched image.
  3. Match new panels to existing predictions by bbox overlap.
  4. Append NEW panels (detected by k_close=7 but not by k_close=9) to
     the predictions.
  5. Write combined_7_v7.jsonl for the eval.

Species labels are preserved 1:1 by bbox-overlap matching, so the
eval isolates the effect of the segmentation change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import cv2

from rlpe.segmentation import PanelSegmenter, SegmentationConfig

PREDICTIONS_IN = ROOT / "work" / "combined_7_v6.jsonl"
PREDICTIONS_OUT = ROOT / "work" / "combined_7_v7.jsonl"


# Paper-id aliases: the prior baum/feng/boughdiri runs were stored in
# different od_output directories (single-paper runs). We try each
# known path and use whichever one exists.
PAPER_DIR_CANDIDATES: dict[str, list[str]] = {
    "178d4e1e9d93136c": ["boughdiri_only_out", "boughdiri_rerun"],
    "17a129b4e9ca975a": ["all7_rerun"],
    "2225994d55021328": ["all7_rerun"],
    "4f1bf415485765b8": ["all7_rerun"],
    "58d7972c37307959": ["baum_rerun_v5", "baum_only_out", "baum_rerun_v4", "baum_rerun_v3", "baum_rerun_v2"],
    "a0f363c21b6941d7": ["all7_rerun"],
    "e28de2b07edc8950": ["feng_rerun_v2", "feng_only_out", "feng_rerun"],
}

# The OD output puts rendered images in
# ``<od_dir>/<paper_label>_images/imageFileN.png`` where ``paper_label``
# is the source PDF's filename stem (e.g. "hollis2006", "feng2007").
# The paper_id is a SHA1 of the file content, so the label has to be
# looked up out-of-band. This map is built from the all7_v7_pdfs dir
# (one PDF per paper, each named with the paper label).
PAPER_LABEL_BY_ID: dict[str, str] = {
    "178d4e1e9d93136c": "boughdiri2007",
    "17a129b4e9ca975a": "danelian2006",
    "2225994d55021328": "pouille2014",
    "4f1bf415485765b8": "bandini2011",
    "58d7972c37307959": "baumgartner2008",
    "a0f363c21b6941d7": "hollis2006",
    "e28de2b07edc8950": "feng2007",
}


def _find_od_dir(paper_id: str, paper_label: str) -> Path | None:
    for work_subdir in PAPER_DIR_CANDIDATES.get(paper_id, []):
        for sub in ("output/od_output", "od_output"):
            od_root = ROOT / "work" / work_subdir / sub / paper_id
            if od_root.is_dir():
                return od_root
    return None


def _list_image_files(od_dir: Path, paper_label: str) -> list[tuple[Path, tuple[int, int]]]:
    """Return [(image_path, (H, W))] for every imageFile*.png in the paper dir."""
    img_dir = od_dir / f"{paper_label}_images"
    if not img_dir.is_dir():
        return []
    out: list[tuple[Path, tuple[int, int]]] = []
    for p in sorted(img_dir.glob("imageFile*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        out.append((p, (img.shape[0], img.shape[1])))
    return out


def _image_for_figure(
    fig_id: str, all_imgs: list[tuple[Path, tuple[int, int]]],
    fig_preds: list[dict], paper_id: str,
) -> Path | None:
    """Find the rendered image that best matches the figure_id.

    Strategy: look at the bboxes of existing predictions for this
    figure_id. The image whose (H, W) bounds all the bboxes (with a
    small tolerance) is the one this figure was segmented from.
    """
    bboxes = [p.get("bbox") for p in fig_preds if p.get("bbox")]
    if not bboxes:
        return None
    max_x = max(b[0] + b[2] for b in bboxes)
    max_y = max(b[1] + b[3] for b in bboxes)
    min_h = max_y
    min_w = max_x
    # Tolerance: allow bboxes to overflow by up to 5 px (cropping margin).
    candidates = [
        (p, hw) for (p, hw) in all_imgs
        if hw[0] >= min_h - 5 and hw[1] >= min_w - 5
    ]
    if not candidates:
        # Fallback: take the smallest image that fits (smallest extra margin).
        candidates = sorted(
            [(p, hw) for (p, hw) in all_imgs if hw[0] >= 100 and hw[1] >= 100],
            key=lambda ph: ph[1][0] * ph[1][1],
        )
    if not candidates:
        return None
    # Pick the one with the smallest "extra" — image area minus bbox area
    # (gives the tightest fit, which is the most likely match).
    candidates.sort(key=lambda ph: ph[1][0] * ph[1][1])
    return candidates[0][0]


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / max(1, union)


def main() -> int:
    rows = [json.loads(l) for l in PREDICTIONS_IN.read_text().splitlines() if l]
    print(f"Loaded {len(rows)} predictions from {PREDICTIONS_IN.name}")

    by_fig: dict[tuple[str, str], list[dict]] = {}
    paper_label: dict[str, str] = {}
    for r in rows:
        by_fig.setdefault((r["paper_id"], r["figure_id"]), []).append(r)
        paper_label[r["paper_id"]] = PAPER_LABEL_BY_ID.get(r["paper_id"], "unknown")

    segmenter = PanelSegmenter(SegmentationConfig())

    new_rows: list[dict] = []
    n_added = 0
    figs_re_seg = 0
    for (paper_id, figure_id), old_preds in by_fig.items():
        pl = paper_label.get(paper_id, "unknown")
        od_dir = _find_od_dir(paper_id, pl)
        if od_dir is None:
            new_rows.extend(old_preds)
            continue
        all_imgs = _list_image_files(od_dir, pl)
        if not all_imgs:
            new_rows.extend(old_preds)
            continue
        img_path = _image_for_figure(figure_id, all_imgs, old_preds, paper_id)
        if img_path is None:
            new_rows.extend(old_preds)
            continue
        figs_re_seg += 1
        panels = segmenter.segment(img_path)
        if not panels:
            new_rows.extend(old_preds)
            continue
        # Match each old pred to the new panel with highest IoU.
        used_new: set[int] = set()
        for op in old_preds:
            old_bbox = op.get("bbox")
            if not old_bbox or len(old_bbox) != 4:
                new_rows.append(op)
                continue
            best_i, best_iou = -1, 0.0
            for i, p in enumerate(panels):
                if i in used_new:
                    continue
                new_bbox = [int(p.bbox[0]), int(p.bbox[1]), int(p.bbox[2]), int(p.bbox[3])]
                iou = _bbox_iou(old_bbox, new_bbox)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_iou > 0.05:
                used_new.add(best_i)
                new_op = dict(op)
                new_op["bbox"] = [int(panels[best_i].bbox[0]), int(panels[best_i].bbox[1]),
                                  int(panels[best_i].bbox[2]), int(panels[best_i].bbox[3])]
                new_rows.append(new_op)
            else:
                new_rows.append(op)
        # Append new panels (not matched to any old pred).
        for i, p in enumerate(panels):
            if i in used_new:
                continue
            n_added += 1
            new_rows.append({
                "paper_id": paper_id,
                "figure_id": figure_id,
                "panel_id": f"new-kclose7-{n_added}",
                "species": None,
                "panel_path": "",
                "bbox": [int(p.bbox[0]), int(p.bbox[1]), int(p.bbox[2]), int(p.bbox[3])],
                "confidence": float(p.score),
                "label_text": "",
                "caption_snippet": "",
                "ocr_text": "",
                "metadata": {"matcher_type": "resegmented-new-kclose7"},
            })

    PREDICTIONS_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows) + "\n"
    )
    print(f"Re-segmented {figs_re_seg}/{len(by_fig)} figures; "
          f"added {n_added} new panels; wrote {len(new_rows)} rows to {PREDICTIONS_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
