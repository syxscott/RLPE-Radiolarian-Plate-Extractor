"""Image-label sanity check: re-OCR each prediction's panel image and
compare to the predicted panel_id.

Background
----------
The standard string-match eval (``scripts/evaluate.py``) compares
predictions to gold on ``(paper_id, figure_id, panel_id, species)`` and
*cannot* tell whether the predicted panel_id matches what is actually
visible in the panel image. This was the source of the N10 bug:
predictions had a high string-match F1 (98.19%) but visually the
panel_id was wrong for ~87% of panels (positional fallback, not image
OCR).

This module reads each prediction row, locates its panel image, OCRs
it, and compares the OCR'd label to the predicted panel_id. It
reports:

  - n_checked           : predictions with a resolvable panel image
  - n_ocr_has_label     : OCR returned ≥ 1 numeric token
  - n_image_label_match : pred panel_id == first numeric OCR token
  - image_label_match_rate : n_image_label_match / n_checked

The function is opt-in via the ``--image-label-check`` flag on
``scripts/evaluate.py``; it adds ~5-15 min on a 9-paper corpus because
EasyOCR runs on every panel image.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NUM_RE = re.compile(r"^\d{1,3}$")


@dataclass(slots=True)
class ImageLabelStats:
    paper_id: str
    n_checked: int = 0
    n_ocr_has_label: int = 0
    n_image_label_match: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def image_label_match_rate(self) -> float:
        return self.n_image_label_match / max(1, self.n_checked)

    @property
    def ocr_coverage(self) -> float:
        return self.n_ocr_has_label / max(1, self.n_checked)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "n_checked": self.n_checked,
            "n_ocr_has_label": self.n_ocr_has_label,
            "n_image_label_match": self.n_image_label_match,
            "ocr_coverage": self.ocr_coverage,
            "image_label_match_rate": self.image_label_match_rate,
            "mismatches": self.mismatches[:20],
        }


def _resolve_panel_path(panel_path: str, root: Path) -> Path | None:
    """Locate the panel image. Mirrors the resolver in
    ``scripts/reassign_panel_id_v18.py``: if the relative path in the
    pred file no longer exists, search for the file by tail
    (``<paper_id>/<fig>/panel_NN.png``) under any ``work/*/panels/``
    directory."""
    if not panel_path:
        return None
    p = Path(panel_path)
    if p.exists() and p.is_file():
        return p
    parts = p.parts
    if len(parts) < 3 or "panel_" not in parts[-1]:
        return None
    tail = Path(*parts[-3:])
    candidates = list(root.glob(f"work/*/panels/{tail.parent.parent.name}/{tail.parent.name}/{tail.name}"))
    if not candidates:
        candidates = list(root.glob(f"work/*/panels/{tail.parent.parent.name}/**/{tail.name}"))
    return candidates[0] if candidates else None


def run_image_label_check(
    predictions: list[dict[str, Any]],
    root: Path,
    max_mismatches_per_paper: int = 20,
    reader=None,
) -> dict[str, Any]:
    """Run the OCR-vs-prediction sanity check.

    ``reader`` is an optional pre-initialised EasyOCR reader; if None,
    a CPU EasyOCR reader is lazily created. Pass a reader from outside
    if you want to amortise init across multiple calls (e.g. test
    suite).
    """
    if reader is None:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
    import numpy as np
    from PIL import Image

    by_paper: dict[str, ImageLabelStats] = defaultdict(lambda: ImageLabelStats(paper_id=""))
    for p in predictions:
        pid = p.get("paper_id") or "unknown"
        st = by_paper[pid]
        st.paper_id = pid
        panel_path = p.get("panel_path", "")
        resolved = _resolve_panel_path(panel_path, root)
        if resolved is None:
            continue
        st.n_checked += 1
        try:
            arr = np.array(Image.open(str(resolved)).convert("RGB"))
        except Exception:
            continue
        try:
            tokens = reader.readtext(arr)
        except Exception:
            continue
        nums = [t[1].strip() for t in tokens if _NUM_RE.match(t[1].strip())]
        if not nums:
            continue
        st.n_ocr_has_label += 1
        first = nums[0]
        pred_label = (p.get("panel_id") or "").strip()
        if first == pred_label:
            st.n_image_label_match += 1
        elif len(st.mismatches) < max_mismatches_per_paper:
            st.mismatches.append({
                "figure_id": p.get("figure_id"),
                "panel_path": str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved),
                "pred_panel_id": pred_label,
                "ocr_label": first,
                "all_ocr_numbers": nums,
            })

    total_checked = sum(s.n_checked for s in by_paper.values())
    total_has_label = sum(s.n_ocr_has_label for s in by_paper.values())
    total_match = sum(s.n_image_label_match for s in by_paper.values())
    return {
        "papers": {k: v.to_dict() for k, v in by_paper.items()},
        "aggregate": {
            "n_checked": total_checked,
            "n_ocr_has_label": total_has_label,
            "n_image_label_match": total_match,
            "ocr_coverage": total_has_label / max(1, total_checked),
            "image_label_match_rate": total_match / max(1, total_checked),
        },
    }
