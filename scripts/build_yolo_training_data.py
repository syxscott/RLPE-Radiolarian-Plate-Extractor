"""Build YOLOv8 training data from RLPE gold panels + per-panel crops.

Walks ``data/gold/<paper>.jsonl`` for the 9 gold papers, locates the
matching per-panel image crops under ``work/<run>/output/panels/``,
and emits a YOLOv8-format dataset::

    data/yolo_dataset/
    ├── images/
    │   ├── train/    (70%)
    │   ├── val/      (15%)
    │   └── test/     (15%)
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── data.yaml
    └── README.md

Each image is a single panel crop; the YOLO label is a single
bounding box that covers the entire image::

    <class_id> 0.5 0.5 1.0 1.0

because the panel crop is already a tight bounding box of the
radiolarian specimen. Class ID is currently always 0
(``radiolarian_panel``) — multi-class support can be added later by
extending :data:`CLASS_NAMES` and passing an explicit class ID per
gold row.

Currently the crop source is the *most recent* work dir that has
panels for that paper (if ``--source-work-dir`` is unset). Filtering
heuristics:

  * Skip gold rows whose ``figure_id`` is not an ``od_plate_*`` or
    ``od_fig_*`` reference (e.g. litholog / paleogeographic-map
    figures which the segmenter skips).
  * Skip non-numeric panel_ids (e.g. ``"1a"`` that the disk-side
    ``panel_NN.png`` naming cannot represent).
  * Take the *first* work dir (most recent, since the caller passes
    ``--source-work-dir`` explicitly, or the freshly scanned list)
    that has a matching panel file.

Usage::

    PYTHONPATH=src python scripts/build_yolo_training_data.py \\
        --output-dir data/yolo_dataset
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES: dict[int, str] = {0: "radiolarian_panel"}
DEFAULT_CLASS_ID = 0
DEFAULT_SPLIT_RATIO: tuple[float, float, float] = (0.70, 0.15, 0.15)
SUPPORTED_PANEL_EXTS = {".png", ".jpg", ".jpeg"}

GOLD_DIR = REPO / "data" / "gold"
WORK_ROOT = REPO / "work"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PanelRecord:
    """A single panel crop intended for the YOLO dataset."""

    paper_id: str
    figure_id: str
    panel_id: str
    species: str
    source_path: Path
    image_relpath: str  # path relative to images/<split>/, e.g. "bandini2011_pl08_panel_03.png"
    label_relpath: str  # path relative to labels/<split>/, e.g. "bandini2011_pl08_panel_03.txt"
    class_id: int = DEFAULT_CLASS_ID


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_panel_files(work_root: Path) -> dict[str, dict[str, list[Path]]]:
    """Return ``{paper_id: {figure_id: [panel_paths]}}``.

    Walks every work dir under ``work_root`` looking for a
    ``output/panels/`` tree. Each ``panel_NN.png`` is a candidate
    crop. The result is deduplicated by paper_id + figure_id +
    panel filename — the *first* work dir that contains a given
    panel wins (caller controls ordering via ``--source-work-dir``
    or dict insertion order).
    """
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for panels_root in work_root.glob("**/output/panels"):
        if not panels_root.is_dir():
            continue
        for paper_dir in panels_root.iterdir():
            if not paper_dir.is_dir():
                continue
            pid = paper_dir.name
            for fig_dir in paper_dir.iterdir():
                if not fig_dir.is_dir():
                    continue
                fig = fig_dir.name
                if not fig.startswith("od_"):
                    continue
                for panel_file in fig_dir.iterdir():
                    if not panel_file.is_file():
                        continue
                    if panel_file.suffix.lower() not in SUPPORTED_PANEL_EXTS:
                        continue
                    if not panel_file.stem.startswith("panel_"):
                        continue
                    out[pid][fig].add(panel_file.name)
    # Convert sets to sorted lists so iteration is deterministic.
    return {pid: {fig: sorted(names) for fig, names in figs.items()} for pid, figs in out.items()}


def load_gold(gold_dir: Path) -> list[dict[str, str]]:
    """Read every ``gold_dir/*.jsonl`` into a flat list of rows."""
    rows: list[dict[str, str]] = []
    for gold_file in sorted(gold_dir.glob("*.jsonl")):
        with open(gold_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def match_panels(
    gold_rows: list[dict[str, str]],
    panel_index: dict[str, dict[str, list[str]]],
) -> tuple[list[PanelRecord], list[dict[str, str]]]:
    """Join gold rows with panel filenames.

    Returns ``(matched, unmatched)``. ``unmatched`` is a list of
    diagnostic dicts (paper_id, figure_id, panel_id, reason) for
    reporting at the end of the run.
    """
    matched: list[PanelRecord] = []
    unmatched: list[dict[str, str]] = []
    for row in gold_rows:
        pid = str(row["paper_id"])
        fig = str(row["figure_id"])
        panel_id = row.get("panel_id")
        if not panel_id:
            unmatched.append({**row, "reason": "missing panel_id"})
            continue
        if not fig.startswith("od_"):
            unmatched.append({**row, "reason": "non-od_figure_id"})
            continue
        try:
            nn = int(panel_id)
        except (ValueError, TypeError):
            unmatched.append({**row, "reason": "non-numeric panel_id"})
            continue
        panel_name = f"panel_{nn:02d}.png"
        if pid not in panel_index or fig not in panel_index[pid]:
            unmatched.append({**row, "reason": "no panel dir for (paper_id, figure_id)"})
            continue
        if panel_name not in panel_index[pid][fig]:
            unmatched.append({**row, "reason": f"panel file {panel_name} missing in panel dir"})
            continue
        matched.append(
            PanelRecord(
                paper_id=pid,
                figure_id=fig,
                panel_id=str(panel_id),
                species=str(row.get("species") or ""),
                source_path=WORK_ROOT
                / "_would_be_built_"
                / pid
                / fig
                / panel_name,  # placeholder; resolved below
                image_relpath="",
                label_relpath="",
                class_id=DEFAULT_CLASS_ID,
            )
        )
    return matched, unmatched


def resolve_source_paths(
    matched: list[PanelRecord],
    work_dirs: list[Path],
) -> list[PanelRecord]:
    """Walk ``work_dirs`` in order and assign each record a real source path.

    Each work_dir is searched for
    ``work_dir/<paper_id>/<figure_id>/panel_NN.png``. The first match
    wins.
    """
    by_paper: dict[str, dict[str, Path]] = {}
    for wd in work_dirs:
        panels_root = wd / "output" / "panels"
        if not panels_root.is_dir():
            continue
        for paper_dir in panels_root.iterdir():
            if not paper_dir.is_dir():
                continue
            pid = paper_dir.name
            for fig_dir in paper_dir.iterdir():
                if not fig_dir.is_dir():
                    continue
                fig = fig_dir.name
                key = (pid, fig)
                if key in by_paper:
                    continue
                for ext in SUPPORTED_PANEL_EXTS:
                    candidate = None
                    # Recover panel_NN from the figure's panel list.
                # We'll just use the first matching file below.
                by_paper[key] = fig_dir  # type: ignore[assignment]
    # Now resolve each record's source path.
    for rec in matched:
        fig_dir = by_paper.get((rec.paper_id, rec.figure_id))
        if fig_dir is None:
            rec.source_path = Path("")
            continue
        try:
            nn = int(rec.panel_id)
        except ValueError:
            rec.source_path = Path("")
            continue
        candidate = fig_dir / f"panel_{nn:02d}.png"
        if candidate.exists():
            rec.source_path = candidate
        else:
            rec.source_path = Path("")
    return matched


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def split_records(
    records: list[PanelRecord],
    ratios: tuple[float, float, float],
    seed: int = 42,
) -> tuple[list[PanelRecord], list[PanelRecord], list[PanelRecord]]:
    """Deterministic 70/15/15 stratified split by paper_id."""
    rng = random.Random(seed)
    by_paper: dict[str, list[PanelRecord]] = defaultdict(list)
    for rec in records:
        by_paper[rec.paper_id].append(rec)
    train: list[PanelRecord] = []
    val: list[PanelRecord] = []
    test: list[PanelRecord] = []
    tr, va, te = ratios
    for pid in sorted(by_paper):
        bucket = sorted(by_paper[pid], key=lambda r: r.image_relpath)
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = max(1, int(round(n * tr))) if n >= 3 else n
        n_val = max(1, int(round(n * va))) if n - n_train >= 2 else 0
        n_test = n - n_train - n_val
        if n_test < 0:
            n_val += n_test
            n_test = 0
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])
    return train, val, test


# ---------------------------------------------------------------------------
# YOLO label normalisation
# ---------------------------------------------------------------------------


def yolo_label_line(
    class_id: int, x_center: float, y_center: float, width: float, height: float
) -> str:
    """Format a single YOLO bbox line with 6-decimal precision.

    Inputs are expected to be in [0, 1]. The convention follows
    the YOLOv8 Ultralytics docs: ``<class> <x_center> <y_center>
    <width> <height>`` all relative to image width/height.
    """
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def emit_yaml(output_dir: Path, classes: dict[int, str]) -> Path:
    """Write ``data.yaml`` for Ultralytics training."""
    yaml_path = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for cid, name in sorted(classes.items()):
        lines.append(f"  {cid}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def emit_readme(output_dir: Path, total: int, splits: dict[str, int], papers: list[str]) -> Path:
    """Write a short README documenting the dataset."""
    readme = output_dir / "README.md"
    body = [
        "# YOLOv8 radiolarian-panel dataset",
        "",
        "Auto-generated by `scripts/build_yolo_training_data.py`.",
        "",
        "Each image is a single per-panel crop extracted from the RLPE",
        "pipeline under `work/<run>/output/panels/`. The YOLO label is",
        "a single bounding box covering the entire image (the panel",
        "crop is already a tight bounding box of the radiolarian).",
        "",
        "## Class names",
        "",
        "| ID | Name |",
        "|----|------|",
        "| 0  | radiolarian_panel |",
        "",
        "## Splits",
        "",
        f"- train: {splits.get('train', 0)} images",
        f"- val:   {splits.get('val', 0)} images",
        f"- test:  {splits.get('test', 0)} images",
        f"- total: {total} images",
        "",
        "## Papers",
        "",
    ]
    for p in sorted(papers):
        body.append(f"- `{p}`")
    body.append("")
    readme.write_text("\n".join(body), encoding="utf-8")
    return readme


def assign_relpaths(records: list[PanelRecord], split: str) -> None:
    """Stamp ``image_relpath`` and ``label_relpath`` on each record."""
    for rec in records:
        # Use a stable name: <paper_short>_<figure_short>_panel_<NN>.png
        paper_short = rec.paper_id[:8] if len(rec.paper_id) >= 8 else rec.paper_id
        fig_short = rec.figure_id.split("_")[-1] if "_" in rec.figure_id else rec.figure_id
        stem = f"{paper_short}_{fig_short}_panel_{int(rec.panel_id):02d}"
        rec.image_relpath = f"{stem}.png"
        rec.label_relpath = f"{stem}.txt"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def pick_work_dirs(work_root: Path, explicit: str | None) -> list[Path]:
    """Order work dirs into a search priority list.

    The first dir whose ``output/panels`` contains the most papers
    wins for ambiguous joins. When ``--source-work-dir`` is given,
    that one dir is used exclusively.
    """
    if explicit:
        wd = Path(explicit)
        if not wd.is_absolute():
            wd = (work_root / wd).resolve()
        return [wd]
    # Otherwise: prefer the dirs that cover the most gold papers.
    candidates = []
    for panels_root in work_root.glob("*/output/panels"):
        if not panels_root.is_dir():
            continue
        wd = panels_root.parent.parent
        n_papers = sum(1 for p in panels_root.iterdir() if p.is_dir())
        candidates.append((n_papers, wd))
    candidates.sort(key=lambda x: (-x[0], x[1].name))
    return [wd for _, wd in candidates]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data" / "yolo_dataset",
        help="Output directory for the YOLO dataset (default: data/yolo_dataset).",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=GOLD_DIR,
        help="Directory containing gold JSONL files (default: data/gold).",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=WORK_ROOT,
        help="Root directory containing work/<run>/output/panels/ (default: work).",
    )
    parser.add_argument(
        "--source-work-dir",
        type=str,
        default=None,
        help=(
            "Restrict the panel search to a single work dir "
            "(e.g. 'v19_run'). Default: scan all work dirs and prefer "
            "the one with the most paper coverage."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val/test split (default: 42).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to deposit panel images into the dataset (default: copy).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing files.",
    )
    args = parser.parse_args()

    work_dirs = pick_work_dirs(args.work_root, args.source_work_dir)
    print(f"Work-dirs to scan: {[str(wd.relative_to(REPO)) for wd in work_dirs]}", file=sys.stderr)

    # ---- 1. Index panels on disk ---------------------------------------
    panel_index = find_panel_files(args.work_root)
    n_papers_indexed = len(panel_index)
    print(f"Indexed panels for {n_papers_indexed} papers", file=sys.stderr)

    # ---- 2. Load gold & join -------------------------------------------
    gold_rows = load_gold(args.gold_dir)
    print(f"Loaded {len(gold_rows)} gold rows from {args.gold_dir}", file=sys.stderr)

    matched, unmatched = match_panels(gold_rows, panel_index)
    matched = [m for m in matched if m.source_path == Path("") or True]  # already filtered
    # Note: matched from match_panels uses placeholder source_path; resolve now.
    matched = resolve_source_paths(matched, work_dirs)
    matched = [m for m in matched if m.source_path != Path("") and m.source_path.exists()]

    print(f"Matched: {len(matched)}  |  Unmatched: {len(unmatched)}", file=sys.stderr)
    if unmatched:
        reasons: dict[str, int] = defaultdict(int)
        for u in unmatched:
            reasons[u.get("reason", "?")] += 1
        for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  unmatch: {r}  ×{n}", file=sys.stderr)

    if not matched:
        print("No panels matched; nothing to do.", file=sys.stderr)
        return 1

    # ---- 3. Split ------------------------------------------------------
    train, val, test = split_records(matched, DEFAULT_SPLIT_RATIO, seed=args.seed)
    n_train, n_val, n_test = len(train), len(val), len(test)
    total = n_train + n_val + n_test
    print(f"Split: train={n_train}  val={n_val}  test={n_test}  total={total}", file=sys.stderr)

    if args.dry_run:
        print("[dry-run] no files written.", file=sys.stderr)
        return 0

    # ---- 4. Materialise -------------------------------------------------
    out = args.output_dir
    for sub in (
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # Assign per-record relpaths
    assign_relpaths(train, "train")
    assign_relpaths(val, "val")
    assign_relpaths(test, "test")

    image_label = "0 0.500000 0.500000 1.000000 1.000000"

    def _deposit(rec: PanelRecord, split: str) -> None:
        img_dst = out / "images" / split / rec.image_relpath
        lbl_dst = out / "labels" / split / rec.label_relpath
        if args.copy_mode == "copy":
            shutil.copy2(rec.source_path, img_dst)
        else:
            img_dst.symlink_to(rec.source_path)
        lbl_dst.write_text(f"{image_label}\n", encoding="utf-8")

    for rec in train:
        _deposit(rec, "train")
    for rec in val:
        _deposit(rec, "val")
    for rec in test:
        _deposit(rec, "test")

    # ---- 5. Emit data.yaml + README ------------------------------------
    papers = sorted({r.paper_id for r in matched})
    yaml_path = emit_yaml(out, CLASS_NAMES)
    readme_path = emit_readme(out, total, {"train": n_train, "val": n_val, "test": n_test}, papers)
    print(f"Wrote {yaml_path}", file=sys.stderr)
    print(f"Wrote {readme_path}", file=sys.stderr)

    # ---- 6. Final summary ----------------------------------------------
    print(f"train: {n_train}  val: {n_val}  test: {n_test}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
