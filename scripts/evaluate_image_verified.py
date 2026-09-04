"""Image-verified panel_id F1 evaluation.

Unlike `scripts/evaluate.py` which compares predicted `(panel_id, species)`
to gold by STRING EQUALITY (which can be inflated by LLM hallucinating
panel_ids from caption text), this script:

  1. Loads the gold panel set
  2. Loads predicted panels (from `llm_first_*.jsonl` or cached `combined_*`)
  3. For each gold panel, EasyOCRs the actual panel image crop
  4. Compares OCR'd printed-number to:
     a) gold panel_id (image-verified ground truth)
     b) pred panel_id (string-match — same as soft F1 baseline)
  5. Reports the gap between string-match F1 and image-verified F1.

Usage
-----
    # Run after a live LLM-first pass has produced panel crops:
    PYTHONPATH=src python scripts/evaluate_image_verified.py \\
        --pred work/v19_run/output/manifests/matches.jsonl \\
        --gold data/gold/ \\
        --panels-root work/v19_run/output/panels/ \\
        --output work/eval_v19_image_verified.json

The script expects panel crops to be at:
    {panels-root}/{paper_id}/{figure_id}/panel_{NN}.png

If a paper has no panel crops on disk, it is reported as "blocked"
rather than counted as zero — this preserves honest accounting of
which papers have been end-to-end verified vs. which are still
pending crops generation.

Output JSON schema:
{
  "papers": {
    "<paper_id>": {
      "n_gold": int,
      "n_pred": int,
      "string_match_panel_id": int,
      "image_verified_panel_id": int,
      "string_match_panel_id_rate": float,
      "image_verified_panel_id_rate": float,
      "gap_pp": float,
      "ocr_coverage": float (fraction of panels with readable printed label),
      "blocked": bool,
    },
    ...
  },
  "aggregate": {
    "n_gold": int,
    "string_match_panel_id_rate": float,
    "image_verified_panel_id_rate": float,
    "gap_pp": float,
    "blocked_papers": [paper_id, ...],
  },
  "panel_id_verification": {
    "status": "measured" | "skipped_no_panels",
    "detail": str,
  },
}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlpe.evaluation.gold import GoldPanel, load_gold  # noqa: E402

logger = logging.getLogger(__name__)


def find_panel_crop(panels_root: Path, paper_id: str, figure_id: str, panel_id: str) -> Path | None:
    """Locate the panel crop PNG. Convention: panels-root/{paper_id}/{figure_id}/panel_{panel_id}.png.

    Falls back to glob if exact match not found (some pipelines pad
    panel_id with leading zeros, e.g. panel_01.png vs panel_1.png).
    """
    fig_dir = panels_root / paper_id / figure_id
    if not fig_dir.exists():
        return None
    # Try exact, then zero-padded 2-digit, then any panel_*.png
    # audit 2026-07-26 M12: panel_id may be non-numeric (e.g. "7a",
    # "A1"); int(panel_id) would raise ValueError and abort the run.
    _pid_zero = panel_id
    try:
        _pid_zero = f"{int(panel_id):02d}"
    except (TypeError, ValueError):
        pass
    candidates = [
        fig_dir / f"panel_{panel_id}.png",
        fig_dir / f"panel_{_pid_zero}.png",
        fig_dir / f"panel_{panel_id}.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def easyocr_panel_label(image_path: Path, reader=None) -> str | None:
    """Run EasyOCR on the panel crop and return the most confident numeric
    label found.

    The crop itself is a single panel — the printed label sits in one
    corner of THIS image (varies by paper: beccaro top-left, bandini
    bottom-right, etc.). EasyOCR on the whole crop is therefore correct;
    cropping to a single corner misses labels in other corners.

    Strategy:
      1. Run EasyOCR on the full panel crop.
      2. Return the highest-confidence token whose text is purely numeric
         (1-3 digits, possibly with a letter suffix like "1a").
    """
    try:
        import easyocr  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    try:
        results = reader.readtext(img, detail=1)
    except Exception as exc:
        logger.debug("EasyOCR failed on %s: %s", image_path, exc)
        return None
    # Pick the highest-confidence token whose text is a numeric label.
    best = None
    for bbox, text, conf in results:
        # Numeric or alphanumeric label (e.g. "1", "12", "1a", "1A").
        # Also accept labels with leading/trailing punctuation that
        # EasyOCR sometimes adds (e.g. ".1", "1.").
        clean = text.strip().strip(".")
        if re.match(r"^\d{1,3}[a-zA-Z]?$", clean):
            if best is None or conf > best[1]:
                best = (clean, conf)
    return best[0] if best else None


def evaluate_image_verified(
    pred_jsonl: Path,
    gold_dir: Path,
    panels_root: Path,
    output: Path | None = None,
    max_panels_per_paper: int | None = None,
) -> dict:
    """Run image-verified F1 across all 9 papers. Returns the report dict.

    Parameters
    ----------
    pred_jsonl:
        Predictions JSONL with (paper_id, figure_id, panel_id, species).
    gold_dir:
        Directory containing `*.jsonl` gold files.
    panels_root:
        Directory containing panel crops at `{paper_id}/{figure_id}/panel_*.png`.
    max_panels_per_paper:
        Cap on panels OCR'd per paper (None = no cap; set to e.g. 50 for
        a quick smoke run).
    """
    # Lazy import EasyOCR only if we'll actually use it
    try:
        import easyocr  # type: ignore[import-not-found]
    except ImportError:
        easyocr = None

    # Load gold
    all_gold: list[GoldPanel] = []
    for gp in sorted(gold_dir.glob("*.jsonl")):
        if gp.name.endswith(".removed"):
            continue
        all_gold.extend(load_gold(gp))
    gold_by_paper: dict[str, list[GoldPanel]] = defaultdict(list)
    for g in all_gold:
        gold_by_paper[g.paper_id].append(g)

    # Load pred
    all_pred: list[dict] = []
    with open(pred_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            all_pred.append(d)
    pred_by_paper: dict[str, list[dict]] = defaultdict(list)
    for p in all_pred:
        pred_by_paper[p["paper_id"]].append(p)

    # Initialise EasyOCR reader once
    reader = None
    if easyocr is not None:
        try:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as exc:
            logger.warning("EasyOCR init failed: %s", exc)
            reader = None

    papers_report: dict[str, dict] = {}
    for pid, gold_list in gold_by_paper.items():
        pred_list = pred_by_paper.get(pid, [])
        n_gold = len(gold_list)
        n_pred = len(pred_list)
        # Build pred lookup keyed on (figure_id, panel_id)
        pred_keys = {(p.get("figure_id", ""), str(p.get("panel_id", ""))) for p in pred_list}
        n_string_match = sum(
            1 for g in gold_list if (g.figure_id or "", str(g.panel_id)) in pred_keys
        )
        # Audit 2026-09-04 eval-7: image-verified F1 used to look up
        # the panel crop by the GOLD panel_id string — but the
        # pipeline writes crops keyed by the pred row's POSITION in
        # the result list (pipeline.py:3163, 5950:
        # ``f"panel_{idx:02d}.png"``), not by the pred panel_id
        # string. When pred.panel_id != gold.panel_id (common — OCR
        # misreads, LLM hallucinates) the gold-driven lookup returns
        # ``None`` and the panel is silently skipped. When they DO
        # match, the OCR re-reads the crop and trivially confirms
        # the pred's already-correct string — i.e. the verification
        # is TAUTOLOGICAL.
        #
        # Fix: iterate PREDS (with valid ``panel_path``) for crop
        # lookup; match pred → gold by (figure_id, position index
        # within the figure — pipeline preserves reading order);
        # compare OCR'd label to the matched GOLD panel_id (real
        # ground truth), not to pred's panel_id.
        n_image_verified = 0
        n_ocr_coverage = 0
        n_checked = 0
        # Index gold panels by figure_id (preserves gold's input
        # order, which mirrors reading order in the source paper).
        gold_by_fig: dict[str, list[GoldPanel]] = defaultdict(list)
        for g in gold_list:
            gold_by_fig[g.figure_id or ""].append(g)
        # Index preds by figure_id (preserves pipeline's emission
        # order, which is also reading order).
        pred_by_fig: dict[str, list[dict]] = defaultdict(list)
        for p in pred_list:
            pred_by_fig[p.get("figure_id", "")].append(p)
        if panels_root.exists() and reader is not None:
            cap = max_panels_per_paper or sum(len(v) for v in pred_by_fig.values())
            checked_so_far = 0
            for fig_id, preds_in_fig in pred_by_fig.items():
                golds_in_fig = gold_by_fig.get(fig_id, [])
                for pred_idx, p in enumerate(preds_in_fig):
                    if checked_so_far >= cap:
                        break
                    panel_path_str = p.get("panel_path")
                    if not panel_path_str:
                        # Pipeline never cropped this pred — skip
                        # honestly rather than fabricate a path from
                        # gold.panel_id.
                        continue
                    crop = Path(panel_path_str)
                    if not crop.exists():
                        # panel_path was written but the file has
                        # since been moved / pruned. Still skip.
                        continue
                    n_checked += 1
                    checked_so_far += 1
                    ocr_label = easyocr_panel_label(crop, reader=reader)
                    if ocr_label is None:
                        continue
                    n_ocr_coverage += 1
                    # Match pred → gold by spatial position. If pred
                    # has no corresponding gold (different figure
                    # length), the verification for that pred is
                    # skipped — it has no ground-truth label to
                    # compare against.
                    if pred_idx >= len(golds_in_fig):
                        continue
                    g = golds_in_fig[pred_idx]
                    if str(ocr_label).strip().lower() == str(g.panel_id).strip().lower():
                        n_image_verified += 1
        # String-match rate (same denominator as image-verified)
        # audit 2026-07-26 M14: was n_string_match / n_gold, but
        # iv_rate uses n_checked as denominator - mixing them makes
        # gap_pp = str_match_rate - iv_rate meaningless. Use n_checked
        # for both so the gap is comparable.
        if n_checked > 0:
            str_match_rate = n_string_match / n_checked if n_checked else 0.0
            iv_rate = n_image_verified / n_checked
        else:
            str_match_rate = n_string_match / n_gold if n_gold else 0.0
            iv_rate = None
        papers_report[pid] = {
            "n_gold": n_gold,
            "n_pred": n_pred,
            "n_string_match": n_string_match,
            "n_image_verified": n_image_verified,
            "n_checked": n_checked,
            "n_ocr_coverage": n_ocr_coverage,
            "string_match_panel_id_rate": str_match_rate,
            "image_verified_panel_id_rate": iv_rate,
            "gap_pp": ((str_match_rate - iv_rate) * 100 if iv_rate is not None else None),
            "blocked": n_checked == 0,
        }

    # Aggregate
    total_gold = sum(p["n_gold"] for p in papers_report.values())
    total_string_match = sum(p["n_string_match"] for p in papers_report.values())
    total_image_verified = sum(p["n_image_verified"] for p in papers_report.values())
    total_checked = sum(p["n_checked"] for p in papers_report.values())
    blocked = [pid for pid, r in papers_report.items() if r["blocked"]]
    sm_rate = total_string_match / total_gold if total_gold else 0.0
    iv_rate = total_image_verified / total_checked if total_checked else None

    report = {
        "papers": papers_report,
        "aggregate": {
            "n_papers": len(papers_report),
            "n_gold": total_gold,
            "n_pred": sum(p["n_pred"] for p in papers_report.values()),
            "n_string_match": total_string_match,
            "n_image_verified": total_image_verified,
            "n_checked": total_checked,
            "string_match_panel_id_rate": sm_rate,
            "image_verified_panel_id_rate": iv_rate,
            "gap_pp": (sm_rate - iv_rate) * 100 if iv_rate is not None else None,
            "blocked_papers": blocked,
        },
        "panel_id_verification": {
            "status": "measured" if reader is not None else "skipped_no_ocr",
            "detail": (
                f"EasyOCR reader {'initialised' if reader else 'NOT initialised'}; "
                f"OCR'd {total_checked}/{total_gold} gold panels across "
                f"{len(papers_report) - len(blocked)}/{len(papers_report)} papers. "
                f"{len(blocked)} paper(s) blocked by missing panel crops."
            ),
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pred", required=True, type=Path, help="predictions JSONL")
    ap.add_argument("--gold", required=True, type=Path, help="gold directory")
    ap.add_argument("--panels-root", required=True, type=Path, help="panel crops root")
    ap.add_argument("--output", type=Path, default=None, help="output JSON path")
    ap.add_argument("--max-panels-per-paper", type=int, default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    report = evaluate_image_verified(
        pred_jsonl=args.pred,
        gold_dir=args.gold,
        panels_root=args.panels_root,
        output=args.output,
        max_panels_per_paper=args.max_panels_per_paper,
    )
    agg = report["aggregate"]
    print()
    print("=" * 70)
    print("Image-verified panel_id F1")
    print("=" * 70)
    print(f"  n_papers:              {agg['n_papers']}")
    print(f"  n_gold:                {agg['n_gold']}")
    print(f"  n_checked (OCR'd):     {agg['n_checked']}")
    print(f"  blocked papers:        {agg['blocked_papers']}")
    print(f"  string-match F1:       {agg['string_match_panel_id_rate']:.4f}")
    if agg["image_verified_panel_id_rate"] is not None:
        print(f"  image-verified F1:     {agg['image_verified_panel_id_rate']:.4f}")
        print(f"  gap (pp):              {agg['gap_pp']:+.2f}")
    else:
        print("  image-verified F1:     N/A (no panels OCR'd)")
    print()
    print("Per paper:")
    for pid, m in report["papers"].items():
        if m["blocked"]:
            print(f"  {pid[:14]:<14}  BLOCKED (no panel crops)")
            continue
        sm = m["string_match_panel_id_rate"]
        iv = m["image_verified_panel_id_rate"]
        if iv is not None:
            print(
                f"  {pid[:14]:<14}  gold={m['n_gold']:>3} pred={m['n_pred']:>3} "
                f"str_match={sm:.3f} iv={iv:.3f} gap={m['gap_pp']:+.1f}pp "
                f"ocr_cov={m['n_ocr_coverage']}/{m['n_checked']}"
            )
        else:
            print(
                f"  {pid[:14]:<14}  gold={m['n_gold']:>3} pred={m['n_pred']:>3} "
                f"str_match={sm:.3f} iv=N/A"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
