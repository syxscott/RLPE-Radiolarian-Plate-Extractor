"""ML view: JSONL with train/val/test split by paper.

For ML researchers, the natural way to consume the data is a HuggingFace
``datasets``-compatible JSONL with a ``split`` column. The split is
**paper-based** (all panels of a paper go to the same split) so that
train/val/test don't share provenance — preventing data leakage.

The split is deterministic by paper short-name hash, so re-runs of
the same data produce the same splits. Use ``--ml-resplit`` to change
the seed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..schema_models import PanelRecord, RunOutput


@dataclass(slots=True)
class MLOptions:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: str = "rlpe-v1"


def _split_for_paper(paper_id: str, options: MLOptions) -> str:
    """Deterministic paper → split assignment."""
    h = hashlib.sha256(f"{options.seed}:{paper_id}".encode()).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF  # 0..1
    if bucket < options.train_ratio:
        return "train"
    if bucket < options.train_ratio + options.val_ratio:
        return "validation"
    return "test"


def _to_ml_record(panel: PanelRecord, split: str) -> dict:
    """Project a panel into an ML-friendly flat dict."""
    return {
        "paper_id": panel.paper_id,
        "figure_id": panel.figure_id,
        "panel_id": panel.panel_id,
        "species": panel.species,
        "label_text": panel.label_text,
        "confidence": float(panel.confidence),
        "bbox": list(panel.bbox) if panel.bbox else None,
        "caption_snippet": panel.caption_snippet,
        "ocr_text": panel.ocr_text,
        "extraction_source": panel.metadata.extraction_source,
        "matcher_type": panel.metadata.matcher_type,
        "split": split,
    }


def write_ml_split(
    run: RunOutput,
    target_dir: Path,
    options: MLOptions | None = None,
) -> dict[str, int]:
    """Write train/val/test JSONL files into ``target_dir``.

    Returns a dict of split → row count.
    """
    options = options or MLOptions()
    target_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0, "test": 0}
    files = {split: target_dir / f"{split}.jsonl" for split in counts}
    handles = {split: open(files[split], "w", encoding="utf-8") for split in counts}
    try:
        for p in run.panels:
            split = _split_for_paper(p.paper_id, options)
            rec = _to_ml_record(p, split)
            handles[split].write(json.dumps(rec, ensure_ascii=False))
            handles[split].write("\n")
            counts[split] += 1
    finally:
        for h in handles.values():
            h.close()
    return counts
