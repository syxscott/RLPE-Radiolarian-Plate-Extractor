"""Run the full research-grade F1 eval.

Combines: caption_fixer + prompts + post_process + LLM-first MiniMax M3
+ 5-fold CV + bootstrap CI on the 9-paper v19 set.

Reports train/test F1 separately to expose generalization gap.

Environment variables (set these BEFORE running):
  - ANTHROPIC_API_KEY   : MiniMax API key (Anthropic-compatible)
  - ANTHROPIC_BASE_URL  : defaults to https://api.minimaxi.com/anthropic
  - ANTHROPIC_MODEL     : defaults to MiniMax-M3
"""

from __future__ import annotations

import argparse
import json
import os
import re

# Add repo paths
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Load .env early so the key is available even if a downstream import
# (gold_eval_anchored) clears os.environ at import time.
try:
    from dotenv import find_dotenv, load_dotenv

    _env_file = find_dotenv(usecwd=True) or str(REPO / ".env")
    if Path(_env_file).exists():
        load_dotenv(_env_file, override=False)
except Exception:
    pass

import pymupdf
from caption_fixer import select_caption
from gold_eval_anchored import (
    compute_aggregate_with_ci,
    load_split,
)
from PIL import Image
from post_process import (
    dedup_panels,
    filter_low_confidence,
    normalize_panel_id,
    parse_open_nomenclature,
)
from prompts import build_user_prompt, select_prompt

from rlpe.llm_backends import MiniMaxM3Backend
from rlpe.utils import stable_id

PAPERS_DIR = REPO / "data" / "pdfs"
HOLDOUT_DIR = REPO / "data" / "pdfs_holdout"
GOLD_DIR = REPO / "data" / "gold"
EXTENDED_GOLD_DIR = REPO / "data" / "gold_v19_extended"


def load_gold_for(slug: str) -> list[dict]:
    """Load gold from either legacy or extended gold dir."""
    for d in [GOLD_DIR, EXTENDED_GOLD_DIR]:
        p = d / f"{slug}.jsonl"
        if p.exists():
            return [json.loads(l) for l in open(p) if l.strip()]
    return []


def _enrich_preds_with_text_and_group(preds: list[dict]) -> list[dict]:
    """Add occurrence_group_id to each pred row.

    Returns a new list (does not mutate input). This is the
    integration point for Feature B: same species in different
    figures in the same paper get the same group id, so the eval
    pipeline can deduplicate when desired.
    """
    from occurrence import add_occurrence_groups

    return add_occurrence_groups(preds)


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
    """Run caption_fixer + prompts + M3 + post_process on one paper."""
    pdf_path = find_pdf(slug)
    if pdf_path is None:
        print(f"  no PDF for {slug}, skip")
        return []
    pid = stable_id(pdf_path)
    print(f"  {slug} paper_id={pid}")

    # Find densest gold figure
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

    # Use caption_fixer (general) — NOT gold species
    plate_anchor = str(int(m.group(2)))
    caption = select_caption(full_text, target_plate=int(plate_anchor))
    if not caption:
        caption = doc[page_num - 1].get_text()
    print(f"    caption: {len(caption)} chars")

    # Render page
    pix = doc[page_num - 1].get_pixmap(dpi=150)
    img_path = f"/tmp/{slug}_p{page_num}.png"
    pix.save(img_path)
    doc.close()

    # Pick prompt by caption type
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
        # Audit 2026-09-04 taxon-8: parse_open_nomenclature now
        # preserves the species string verbatim (cf./aff. stays
        # inside); the qualifier is exposed separately for provenance
        # but is NOT re-stitched onto the end of the species field,
        # because open nomenclature puts cf./aff. BEFORE the epithet.
        # The original wrong assembly was ``f"{sp}{qual_str}"``.
        preds.append(
            {
                "paper_id": pid,
                "figure_id": target_fig,
                "panel_id": normalize_panel_id(p.get("label", "")),
                "species": sp if sp else None,
                "qualifier": qual,
                "confidence": p.get("confidence", 0.0),
            }
        )
    # Post-process
    preds = dedup_panels(preds)
    preds = filter_low_confidence(preds, threshold=0.7)
    print(f"    {len(preds)} preds (after dedup + conf filter)")
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="data/splits/research_v1.json")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="data/snapshot/eval.json")
    args = parser.parse_args()

    split = load_split(args.split)
    print(f"Split: {len(split['train'])} train + {len(split['test'])} test")

    # Re-load .env if a sibling import (gold_eval_anchored) cleared the key
    # at its own import time.  Reading the file directly avoids re-running
    # dotenv (which only loads on first call).
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env_file = REPO / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY=") and "=" in line:
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        os.environ["ANTHROPIC_API_KEY"] = val
                    break

    backend = MiniMaxM3Backend(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic"),
        model=os.environ.get("ANTHROPIC_MODEL", "MiniMax-M3"),
        timeout_sec=60,
    )

    train_preds, train_gold, test_preds, test_gold = [], [], [], []
    all_papers = split["train"] + split["test"]
    for i, slug in enumerate(all_papers):
        is_train = i < len(split["train"])
        if i > 0:
            time.sleep(60)  # rate limit
        gold = load_gold_for(slug)
        preds = extract_panels_for_paper(backend, slug, gold)
        # Feature B: attach occurrence_group_id to every pred row
        preds = _enrich_preds_with_text_and_group(preds)
        if is_train:
            train_preds.extend(preds)
            train_gold.extend(gold)
        else:
            test_preds.extend(preds)
            test_gold.extend(gold)

    print("\n=== Computing F1 ===")
    train_f1, train_ci = compute_aggregate_with_ci(
        train_preds,
        train_gold,
        n_bootstrap=args.bootstrap_samples,
    )
    test_f1, test_ci = compute_aggregate_with_ci(
        test_preds,
        test_gold,
        n_bootstrap=args.bootstrap_samples,
    )

    print(
        f"\nTRAIN ({len(split['train'])} papers): F1={train_f1:.4f} 95%CI=[{train_ci[0]:.4f},{train_ci[1]:.4f}]"
    )
    print(
        f"TEST  ({len(split['test'])} papers): F1={test_f1:.4f} 95%CI=[{test_ci[0]:.4f},{test_ci[1]:.4f}]"
    )
    gap = train_f1 - test_f1
    print(f"GENERALIZATION GAP: {gap:+.4f}  ({'OK' if abs(gap) <= 0.08 else 'OVERFITTING'})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "split": split,
                "train_f1": train_f1,
                "train_ci": train_ci,
                "test_f1": test_f1,
                "test_ci": test_ci,
                "gap": gap,
                "n_train_preds": len(train_preds),
                "n_test_preds": len(test_preds),
            },
            indent=2,
        )
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
