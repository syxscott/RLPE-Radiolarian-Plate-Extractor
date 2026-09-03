#!/usr/bin/env python3
"""3-way YOLO comparison experiment.

Runs 3 YOLO models (yolo11x, yolov8n, radiolarian_yolo_v1) on 5 v19 papers.
Uses DEFAULT config (no threshold tuning). Renders PDFs at 150 DPI, runs YOLO,
records raw detections + inference time, then computes panel-count-proxy
precision/recall vs. gold standard.

Outputs:
  - /tmp/yolo_test/<slug>/<model>_raw.json
  - data/eval/yolo_comparison_3way.csv
  - docs/eval/yolo_comparison_3way.md
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import fitz  # pymupdf
from ultralytics import YOLO

REPO_ROOT = Path("/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor")
PDF_DIR = REPO_ROOT / "data" / "pdfs"
GOLD_DIR = REPO_ROOT / "data" / "gold"
RENDER_DIR = Path("/tmp/yolo_test")
CSV_OUT = REPO_ROOT / "data" / "eval" / "yolo_comparison_3way.csv"
MD_OUT = REPO_ROOT / "docs" / "eval" / "yolo_comparison_3way.md"

PAPERS = ["bandini2011", "baumgartner2008", "beccaro2006", "danelian2006", "hollis2006"]

MODELS = {
    "A_yolo11x": REPO_ROOT / "models" / "yolo11x.pt",
    "B_yolov8n": REPO_ROOT / "yolov8n.pt",
    "C_radio_yolo_v1": REPO_ROOT / "models" / "radiolarian_yolo_v1.pt",
}

DPI = 150


def render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int = DPI) -> list[Path]:
    """Render each page to a PNG. Returns list of PNG paths (page index 0-based)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    page_paths: list[Path] = []
    for i, page in enumerate(doc):
        png_path = out_dir / f"page_{i + 1:03d}.png"
        if not png_path.exists():
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(png_path))
        page_paths.append(png_path)
    doc.close()
    return page_paths


def run_yolo_on_paper(model_path: Path, page_paths: list[Path]) -> dict:
    """Run YOLO on a list of page images. Return per-page detection records + total time."""
    model = YOLO(str(model_path))
    # warm-up to remove one-time CUDA init cost from total
    _ = model(page_paths[:1], verbose=False, save=False)
    if hasattr(model.predictor, "warmup"):
        pass

    per_page: dict[str, dict] = {}
    total_dets = 0
    all_confs: list[float] = []
    t0 = time.perf_counter()
    # batch all pages for one forward pass (more representative)
    results = model(page_paths, verbose=False, save=False)
    elapsed = time.perf_counter() - t0

    for idx, r in enumerate(results, start=1):
        dets = []
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, cls):
                dets.append({
                    "conf": float(cf),
                    "class": int(cl),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                })
                all_confs.append(float(cf))
                total_dets += 1
        per_page[str(idx)] = {"page_num": idx, "detections": dets}
    return {
        "per_page": per_page,
        "total_detections": total_dets,
        "inference_time_total_sec": elapsed,
        "avg_conf": (sum(all_confs) / len(all_confs)) if all_confs else 0.0,
    }


def gold_panel_counts(slug: str) -> tuple[int, Counter]:
    """Return (total_rows, per-figure counts) for a gold file."""
    path = GOLD_DIR / f"{slug}.jsonl"
    per_figure: Counter = Counter()
    total = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            per_figure[row["figure_id"]] += 1
            total += 1
    return total, per_figure


def gold_pages_for_paper(slug: str) -> set[int]:
    """Extract page numbers from figure_id (e.g. *_p013_pl01* → page 13)."""
    path = GOLD_DIR / f"{slug}.jsonl"
    pages: set[int] = set()
    pat = re.compile(r"_p(\d+)_")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            m = pat.search(row["figure_id"])
            if m:
                pages.add(int(m.group(1)))
    return pages


def evaluate_proxy(total_detections: int, gold_total_rows: int) -> tuple[float, float, str]:
    """Approximate precision/recall via panel-count proximity.

    Treat each gold row as a "panel" the detector should fire for.
    - recall_proxy = min(M / N, 1.0)
    - precision_proxy = min(N / M, 1.0)  (a model producing at least as many dets as gold panels
      is treated as having 100% precision against gold; over-detection is not penalized directly)
    - notes describe over/under-detection in human terms.
    Returns (precision_proxy, recall_proxy, notes).
    """
    if gold_total_rows == 0:
        return 0.0, 0.0, "no gold"
    if total_detections == 0:
        return 0.0, 0.0, "no detections"
    recall = min(total_detections / gold_total_rows, 1.0)
    precision = min(gold_total_rows / total_detections, 1.0)
    ratio = total_detections / gold_total_rows
    if ratio < 0.5:
        notes = f"under-detection ({ratio:.2f}x of gold)"
    elif ratio > 2.0:
        notes = f"over-detection ({ratio:.2f}x of gold)"
    else:
        notes = f"in range ({ratio:.2f}x of gold)"
    return round(precision, 4), round(recall, 4), notes


def slug_to_pdf(slug: str) -> Path:
    # hollis2006.pdf, bandini2011.pdf etc.
    return PDF_DIR / f"{slug}.pdf"


def main() -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    detail_rows: list[tuple] = []  # for the markdown table

    for slug in PAPERS:
        pdf_path = slug_to_pdf(slug)
        if not pdf_path.exists():
            print(f"!! PDF missing: {pdf_path}")
            continue
        paper_dir = RENDER_DIR / slug
        print(f"\n=== {slug} ===")
        page_paths = render_pdf_pages(pdf_path, paper_dir)
        total_pages = len(page_paths)
        print(f"  rendered {total_pages} pages -> {paper_dir}")

        gold_total, gold_figs = gold_panel_counts(slug)
        gold_pages = gold_pages_for_paper(slug)
        print(f"  gold rows: {gold_total} ({len(gold_figs)} unique figures, pages {sorted(gold_pages)[:5]}...)")

        for model_label, model_path in MODELS.items():
            if not model_path.exists():
                print(f"  !! model missing: {model_path}")
                continue
            print(f"  -> running {model_label} ({model_path.name})")
            res = run_yolo_on_paper(model_path, page_paths)
            raw_out = paper_dir / f"{model_label}_raw.json"
            raw = {
                "paper_id": slug,
                "model": model_path.name,
                "model_label": model_label,
                "total_pages": total_pages,
                "inference_time_total_sec": round(res["inference_time_total_sec"], 3),
                "total_detections": res["total_detections"],
                "avg_conf": round(res["avg_conf"], 4),
                "gold_total_rows": gold_total,
                "per_page": res["per_page"],
            }
            with open(raw_out, "w") as f:
                json.dump(raw, f, indent=1)
            precision, recall, notes = evaluate_proxy(res["total_detections"], gold_total)
            summary_rows.append({
                "paper_id": slug,
                "model": model_label,
                "model_path": str(model_path),
                "total_pages": total_pages,
                "total_detections": res["total_detections"],
                "avg_conf": round(res["avg_conf"], 4),
                "inference_time_sec": round(res["inference_time_total_sec"], 3),
                "approx_precision_proxy": precision,
                "approx_recall_proxy": recall,
                "notes": notes,
            })
            detail_rows.append((slug, model_label, res["total_detections"],
                                round(res["avg_conf"], 4),
                                round(res["inference_time_total_sec"], 3)))
            print(f"     {res['total_detections']} dets, {res['inference_time_total_sec']:.2f}s, {notes}")

    # write CSV
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nCSV -> {CSV_OUT}")

    # write Markdown summary
    lines: list[str] = []
    lines.append("# 3-Way YOLO Comparison on 5 v19 Papers\n")
    lines.append("**Date:** 2026-09-03  ")
    lines.append("**Setup:** RTX 4090, ultralytics 8.4.106, torch 2.12.0+cu130, 150 DPI PDF render, DEFAULT model config.  ")
    lines.append("**Goal:** find best of-the-shelf detector for radiolarian figure panels.  ")
    lines.append("**Note:** Gold rows are panel observations (no bbox), so precision/recall are *panel-count proxies*, not IoU-based.\n")

    lines.append("## Models\n")
    lines.append("| Label | Weights | Size |")
    lines.append("|---|---|---|")
    for label, p in MODELS.items():
        size_mb = p.stat().st_size / (1024 * 1024)
        lines.append(f"| {label} | `{p.name}` | {size_mb:.1f} MB |")
    lines.append("")

    # cache gold row counts per paper
    gold_count_by_paper: dict[str, int] = {}
    for slug in PAPERS:
        gold_count_by_paper[slug] = sum(1 for _ in open(GOLD_DIR / f"{slug}.jsonl"))

    lines.append("## Per-paper, per-model raw results\n")
    lines.append("| Paper | Model | Detections | Avg Conf | Time (s) | Gold Rows | P (proxy) | R (proxy) | Notes |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in summary_rows:
        gold_total = gold_count_by_paper.get(row["paper_id"], 0)
        lines.append(
            f"| {row['paper_id']} | {row['model']} | {row['total_detections']} | "
            f"{row['avg_conf']:.3f} | {row['inference_time_sec']:.2f} | {gold_total} | "
            f"{row['approx_precision_proxy']:.2f} | {row['approx_recall_proxy']:.2f} | {row['notes']} |"
        )
    lines.append("")

    # comparison table per task spec
    lines.append("## Per-paper A vs B vs C (det / time)\n")
    lines.append("| Paper | A:yolo11x det/time | B:yolov8n det/time | C:radio_yolo_v1 det/time | gold rows |")
    lines.append("|---|---|---|---|---:|")
    for slug in PAPERS:
        gold_total = gold_count_by_paper.get(slug, 0)
        cells = {}
        for label in ["A_yolo11x", "B_yolov8n", "C_radio_yolo_v1"]:
            r = next((x for x in summary_rows if x["paper_id"] == slug and x["model"] == label), None)
            if r:
                cells[label] = f"{r['total_detections']}/{r['inference_time_sec']:.1f}s"
            else:
                cells[label] = "—"
        lines.append(
            f"| {slug} | {cells['A_yolo11x']} | {cells['B_yolov8n']} | {cells['C_radio_yolo_v1']} | {gold_total} |"
        )
    lines.append("")

    # aggregate stats
    lines.append("## Aggregate (across 5 papers)\n")
    lines.append("| Model | Total dets | Avg conf | Total time | Avg P (proxy) | Avg R (proxy) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    agg: dict[str, dict] = {label: {"dets": 0, "time": 0.0, "confs": [], "p": [], "r": []} for label in MODELS}
    for row in summary_rows:
        a = agg[row["model"]]
        a["dets"] += row["total_detections"]
        a["time"] += row["inference_time_sec"]
        a["confs"].append(row["avg_conf"])
        a["p"].append(row["approx_precision_proxy"])
        a["r"].append(row["approx_recall_proxy"])
    for label in MODELS:
        a = agg[label]
        avg_p = statistics.mean(a["p"]) if a["p"] else 0
        avg_r = statistics.mean(a["r"]) if a["r"] else 0
        avg_c = statistics.mean(a["confs"]) if a["confs"] else 0
        lines.append(f"| {label} | {a['dets']} | {avg_c:.3f} | {a['time']:.2f} | {avg_p:.2f} | {avg_r:.2f} |")
    lines.append("")

    lines.append("## Interpretation\n")
    lines.append("- **recall_proxy** = min(detections / gold_rows, 1.0) — proxies how well the model finds expected panels.")
    lines.append("- **precision_proxy** = min(gold_rows / detections, 1.0) — penalizes over-detection; perfect if detections <= gold_rows.")
    lines.append("- Gold rows = panel observations in `data/gold/<slug>.jsonl`; the user-curated ground truth has no bbox, so this is an approximate comparison only.")
    lines.append("- **time** is wall-clock for one YOLO forward pass over ALL rendered pages (single batched call), excluding PDF→PNG rendering.")
    lines.append("- DEFAULT config: no threshold/IoU tuning per model; whatever Ultralytics ships as `model(paths, ...)` defaults.")
    lines.append("")

    with open(MD_OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown -> {MD_OUT}")

    # cleanup
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()