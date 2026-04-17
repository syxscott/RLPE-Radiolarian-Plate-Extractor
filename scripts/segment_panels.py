from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpe.segmentation import PanelSegmenter, SegmentationConfig
from rlpe.utils import ensure_dir, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Segment panels from page or figure images")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-sam2", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-cfg", type=str, default=None)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    segmenter = PanelSegmenter(SegmentationConfig(use_sam2=args.use_sam2), checkpoint=args.checkpoint, model_cfg=args.model_cfg)
    rows = []
    for img_path in sorted(list(args.image_dir.glob("*.png")) + list(args.image_dir.glob("*.jpg")) + list(args.image_dir.glob("*.jpeg"))):
        panels = segmenter.segment(img_path)
        rows.append({
            "image_path": str(img_path),
            "panels": [
                {"panel_id": p.panel_id, "bbox": list(p.bbox), "score": p.score, "metadata": p.metadata}
                for p in panels
            ],
        })
    write_jsonl(args.output_dir / "panels.jsonl", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
