"""Re-run v19 9-paper baseline with current prompt for fair comparison.

Uses the same MiniMax-M3 backend + same caption_fixer / prompt / post_process
pipeline as ``scripts/run_research_eval.py`` so the re-measured F1 is directly
comparable to our 9-paper research eval result.

The original v19 SOTA paper (RLPE v19, 2026) reported F1=0.84 on the same
9-paper set. By re-running it under the same harness we can detect whether
the 0.84 was measured under a stronger prompt than the one we use today.

SECURITY: This script never defines or hardcodes the API key. It only reads
``ANTHROPIC_API_KEY`` (and base URL / model) from the environment. The caller
is responsible for setting those vars before invoking this script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import pymupdf
from PIL import Image

# Read API key + base URL + model strictly from the environment. Never define
# any default key string here — if these env vars are missing the script must
# fail loudly at the first real API call rather than silently use a fake key.
os.environ.setdefault("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
os.environ.setdefault("ANTHROPIC_MODEL", "MiniMax-M3")

from caption_fixer import select_caption  # noqa: E402
from post_process import (  # noqa: E402
    dedup_panels,
    filter_low_confidence,
    normalize_panel_id,
    parse_open_nomenclature,
)
from prompts import build_user_prompt, select_prompt  # noqa: E402

from rlpe.utils import stable_id  # noqa: E402

PAPERS_DIR = REPO / "data" / "pdfs"
HOLDOUT_DIR = REPO / "data" / "pdfs_holdout"
GOLD_DIR = REPO / "data" / "gold"
EXTENDED_GOLD_DIR = REPO / "data" / "gold_v19_extended"

# Same 9-paper set used by v19 baseline.
V19_PAPERS = [
    "bandini2011",
    "baumgartner2008",
    "beccaro2006",
    "boughdiri2007",
    "bragin2025",
    "danelian2006",
    "feng2007",
    "hollis2006",
    "pouille2014",
]


def load_gold_for(slug: str) -> list[dict]:
    """Load gold from either legacy or extended gold dir."""
    for d in [GOLD_DIR, EXTENDED_GOLD_DIR]:
        p = d / f"{slug}.jsonl"
        if p.exists():
            return [json.loads(l) for l in open(p) if l.strip()]
    return []


def find_pdf(slug: str) -> Path | None:
    for d in [PAPERS_DIR, HOLDOUT_DIR]:
        for p in d.glob("*.pdf"):
            if p.stem.startswith(slug) or slug in p.stem:
                return p
    return None


def call_m3(backend, img, caption, system_prompt) -> dict | None:
    for attempt in range(3):
        try:
            return backend.infer_panel(
                panel_image=img,
                caption_text=caption,
                ocr_labels=[],
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(caption),
            )
        except Exception as e:
            print(f"  API error attempt {attempt + 1}: {e}")
            time.sleep(5)
    return None


def extract_panels_for_paper(backend, slug: str, gold: list[dict]) -> list[dict]:
    """Same per-paper extraction as run_research_eval.py — see that file for
    the inline rationale. We keep this script independent so the v19
    re-measurement is reproducible without touching the research eval.
    """
    pdf_path = find_pdf(slug)
    if pdf_path is None:
        print(f"  no PDF for {slug}, skip")
        return []
    pid = stable_id(pdf_path)
    print(f"  {slug} paper_id={pid}")

    fig_counts = Counter(g.get("figure_id") for g in gold)
    if not fig_counts:
        return []
    target_fig, _ = fig_counts.most_common(1)[0]
    m = re.search(r"_p(\d{3})_pl(\d+)", target_fig)
    if not m:
        return []
    page_num = int(m.group(1))

    doc = pymupdf.open(str(pdf_path))
    if page_num > len(doc):
        doc.close()
        return []
    full_text = "\n".join(p.get_text() for p in doc)

    plate_anchor = str(int(m.group(2)))
    caption = select_caption(full_text, target_plate=int(plate_anchor))
    if not caption:
        caption = doc[page_num - 1].get_text()
    print(f"    caption: {len(caption)} chars")

    pix = doc[page_num - 1].get_pixmap(dpi=150)
    img_path = f"/tmp/{slug}_v19_p{page_num}.png"
    pix.save(img_path)
    doc.close()

    sys_prompt = select_prompt(caption)

    img = Image.open(img_path)
    r = call_m3(backend, img, caption, sys_prompt)
    if not r or r.get("error") or r.get("fallback_used"):
        return []
    if r.get("_is_multi_panel") and isinstance(r.get("panels"), list):
        panels = r["panels"]
    else:
        panels = [r]
    preds = []
    for p in panels:
        sp_raw = p.get("species")
        sp, qual = parse_open_nomenclature(sp_raw)
        qual_str = f" {qual}" if qual else ""
        preds.append(
            {
                "paper_id": pid,
                "figure_id": target_fig,
                "panel_id": normalize_panel_id(p.get("label", "")),
                "species": f"{sp}{qual_str}" if sp else None,
                "confidence": p.get("confidence", 0.0),
            }
        )
    preds = dedup_panels(preds)
    preds = filter_low_confidence(preds, threshold=0.7)
    print(f"    {len(preds)} preds (after dedup + conf filter)")
    return preds


def micro_f1_with_ci(preds, gold, n_bootstrap: int = 1000):
    """Compute micro-F1 + 95% bootstrap CI on the (pred, gold) pairs.

    Mirrors the aggregation used by ``run_research_eval.py`` but inlined
    here so this script can run standalone without depending on
    ``gold_eval_anchored``'s split logic.
    """
    from rlpe.evaluation.metrics import _norm_species

    def _f1(pp, gp):
        by_paper = {}
        for g in gp:
            pid = g.get("paper_id", "")
            by_paper.setdefault(pid, ([], []))[1].append(g)
        for p in pp:
            pid = p.get("paper_id", "")
            if pid:
                by_paper.setdefault(pid, ([], []))[0].append(p)
        tp = fp = fn = 0
        for pp_, gp_ in by_paper.values():
            pset = {
                (_norm_species(x.get("species")), x.get("figure_id"), x.get("panel_id"))
                for x in pp_
                if x.get("species")
            }
            gset = {
                (_norm_species(x.get("species")), x.get("figure_id"), x.get("panel_id"))
                for x in gp_
                if x.get("species")
            }
            tp += len(pset & gset)
            fp += len(pset - gset)
            fn += len(gset - pset)
        if tp == 0:
            return 0.0
        p_ = tp / (tp + fp)
        r_ = tp / (tp + fn)
        return 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0

    point = _f1(preds, gold)
    if not preds or not gold or n_bootstrap <= 0:
        return point, (point, point)

    # Bootstrap resample at the paper level so CI respects paper boundaries.
    papers = sorted(
        {
            (p.get("paper_id"), g.get("paper_id"))
            for p in preds
            for g in gold
            if p.get("paper_id") and g.get("paper_id") and p.get("paper_id") == g.get("paper_id")
        }
    )
    if not papers:
        return point, (point, point)
    rng = __import__("random").Random(20260902)
    boot = []
    for _ in range(n_bootstrap):
        sample_preds, sample_gold = [], []
        for _ in range(len(papers)):
            pid = rng.choice(papers)[0]
            sample_preds.extend(x for x in preds if x.get("paper_id") == pid)
            sample_gold.extend(x for x in gold if x.get("paper_id") == pid)
        boot.append(_f1(sample_preds, sample_gold))
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[int(0.975 * len(boot)) - 1]
    return point, (lo, hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/snapshot/2026-09-02/v19_baseline_f1.json")
    parser.add_argument(
        "--rate-limit-sec",
        type=int,
        default=60,
        help="Sleep between papers (default 60s — matches run_research_eval.py)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan and exit without calling the API"
    )
    args = parser.parse_args()

    # Hard fail if ANTHROPIC_API_KEY is missing. Never fall back to a default.
    if "ANTHROPIC_API_KEY" not in os.environ or not os.environ["ANTHROPIC_API_KEY"]:
        print("ERROR: ANTHROPIC_API_KEY env var is required.", file=sys.stderr)
        return 2

    from rlpe.llm_backends import MiniMaxM3Backend

    backend = MiniMaxM3Backend(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic"),
        model=os.environ.get("ANTHROPIC_MODEL", "MiniMax-M3"),
        timeout_sec=60,
    )

    if args.dry_run:
        print("Dry run — would extract these 9 papers:")
        for slug in V19_PAPERS:
            print(f"  - {slug}")
        return 0

    all_preds, all_gold = [], []
    for i, slug in enumerate(V19_PAPERS):
        if i > 0:
            time.sleep(args.rate_limit_sec)
        gold = load_gold_for(slug)
        if not gold:
            print(f"  {slug}: no gold, skip")
            continue
        preds = extract_panels_for_paper(backend, slug, gold)
        all_preds.extend(preds)
        all_gold.extend(gold)
        print(f"  {slug}: {len(preds)} preds")

    f1, (lo, hi) = micro_f1_with_ci(all_preds, all_gold)
    print(f"\nv19 re-measured F1: {f1:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    print(f"(n_preds={len(all_preds)}, n_gold={len(all_gold)})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "v19_original_f1": 0.84,
                "v19_re_measured_f1": f1,
                "v19_re_measured_ci_low": lo,
                "v19_re_measured_ci_high": hi,
                "n_preds": len(all_preds),
                "n_gold": len(all_gold),
                "papers": V19_PAPERS,
                "date": "2026-09-02",
            },
            indent=2,
        )
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
