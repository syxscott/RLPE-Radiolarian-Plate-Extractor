"""9-paper gold eval v4: gold-anchored.
For each paper, find the gold figure_id with most panels. Use that
figure_id as the pred figure_id (so panel-level matching is direct).
Use the page text from pymupdf + the page image. M3 extracts panels.
"""
import os, sys, json, time, re
from pathlib import Path
sys.path.insert(0, '/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/src')

from rlpe.llm_backends import MiniMaxM3Backend
from rlpe.utils import stable_id
import pymupdf
from PIL import Image

backend = MiniMaxM3Backend(
    api_key=os.environ['ANTHROPIC_API_KEY'],
    base_url=os.environ['ANTHROPIC_BASE_URL'],
    model=os.environ['ANTHROPIC_MODEL'],
    timeout_sec=60,
)

PAPERS_DIR = Path('/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/data/pdfs')
GOLD_DIR = Path('/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/data/gold')

slugs = ['bandini2011', 'baumgartner2008', 'beccaro2006', 'boughdiri2007', 'bragin2025',
         'danelian2006', 'feng2007', 'hollis2006', 'pouille2014']

print('=' * 70)
print('9-paper gold eval v4 (gold-anchored: use gold figure_id + page text)')
print('=' * 70)

all_preds = []
rate_limit_hits = 0
t0 = time.time()

for i, slug in enumerate(slugs):
    pdf_path = PAPERS_DIR / f'{slug}.pdf'
    gfp = GOLD_DIR / f'{slug}.jsonl'
    if not pdf_path.exists() or not gfp.exists():
        print(f'\n[{i+1}/{len(slugs)}] {slug}: missing, skip')
        continue
    pid = stable_id(pdf_path)
    print(f'\n[{i+1}/{len(slugs)}] {slug} (paper_id={pid})')

    # Find gold figure with most panels
    gold = [json.loads(l) for l in open(gfp) if l.strip()]
    fig_counts = {}
    for g in gold:
        fid = g.get('figure_id', '')
        fig_counts[fid] = fig_counts.get(fid, 0) + 1
    if not fig_counts:
        print(f'  no gold figures, skip')
        continue
    target_fig = max(fig_counts.keys(), key=lambda k: fig_counts[k])
    target_count = fig_counts[target_fig]
    print(f'  target figure: {target_fig} ({target_count} gold panels)')

    # Extract page from figure_id (od_plate_X_p<NNN>_pl<NN>)
    m = re.search(r'_p(\d{3})_pl(\d+)', target_fig)
    if not m:
        print(f'  cannot parse page from {target_fig}, skip')
        continue
    page_num = int(m.group(1))

    # Read full PDF text
    doc = pymupdf.open(str(pdf_path))
    if page_num > len(doc):
        print(f'  page {page_num} OOR, skip')
        doc.close()
        continue
    page_text = doc[page_num - 1].get_text()
    full_text = '\n'.join(p.get_text() for p in doc)
    doc.close()

    # Find the FULL caption block for the target figure.
    # Strategy: look for a paragraph starting with "Plate N." or "Fig. N"
    # (where N matches the page's plate / figure index) and containing
    # the most gold species. A real caption is usually the FIRST
    # paragraph on the page that contains "Plate <N>" or "Fig. <N>" and
    # extends until the first blank line / Sample / Marker / Scale bar.
    gold_species = set(g.get('species','') for g in gold if g.get('figure_id') == target_fig)
    # Extract plate / figure number from target_fig: od_plate_X_pNNN_plNN
    plate_num = m.group(2) if m else None
    # The PDF caption might write "Plate 5" or "Plate 05" or "Plate V" —
    # strip leading zeros so the regex matches both.
    plate_anchor = str(int(plate_num)) if plate_num else None
    paragraphs = re.split(r'\n\s*\n', full_text)
    best_para = None
    best_score = 0
    # Anchor: prefer paragraph starting with "Plate <N>." or "Fig. <N>."
    if plate_anchor:
        anchor_re = re.compile(
            rf'^\s*(?:Plate|Fig)\.?\s*0?{plate_anchor}\b\.?',
            re.IGNORECASE,
        )
        for p in paragraphs:
            if not anchor_re.match(p):
                continue
            if len(p) < 50 or len(p) > 4000:
                continue
            score = sum(1 for sp in gold_species if sp in p and len(sp) > 5)
            if score > best_score:
                best_score = score
                best_para = p
    # Fallback: best paragraph by species overlap (no anchor)
    if best_para is None:
        for p in paragraphs:
            if len(p) < 100 or len(p) > 4000:
                continue
            score = sum(1 for sp in gold_species if sp in p and len(sp) > 5)
            if score > best_score:
                best_score = score
                best_para = p
    if best_para is None:
        best_para = page_text
    print(f'  caption ({len(best_para)} chars, gold_species_overlap={best_score}): {best_para[:120]}...')

    # Render page
    doc = pymupdf.open(str(pdf_path))
    pix = doc[page_num - 1].get_pixmap(dpi=150)
    img_path = f'/tmp/{slug}_v4_p{page_num}.png'
    pix.save(img_path)
    doc.close()

    # Rate limit
    if i > 0:
        wait = 60
        print(f'  sleeping {wait}s...')
        time.sleep(wait)

    img = Image.open(img_path)
    sys_prompt = """You are an expert radiolarian paleontologist. Given a figure caption and image, extract every specimen panel. Return strict JSON array of objects with {label, species, confidence}."""
    user_prompt = f"Caption:\n{best_para}\n\nReturn JSON array."

    try:
        r = backend.infer_panel(
            panel_image=img, caption_text=best_para, ocr_labels=[],
            system_prompt=sys_prompt, user_prompt=user_prompt,
        )
        if r.get('error') or r.get('fallback_used'):
            err_msg = str(r.get('error', '?'))[:80]
            print(f'  API error: {err_msg}')
            continue
        if r.get('_is_multi_panel') and isinstance(r.get('panels'), list):
            panels = r['panels']
        else:
            panels = [r]
        print(f'  → {len(panels)} panels extracted')
        for p in panels:
            all_preds.append({
                'paper_id': pid,
                'figure_id': target_fig,  # use the EXACT gold figure_id
                'panel_id': str(p.get('label', '')),
                'species': p.get('species'),
                'confidence': p.get('confidence', 0.0),
            })
    except Exception as e:
        print(f'  EXC: {type(e).__name__}: {e}')

elapsed = time.time() - t0
print(f'\n=== Total: {len(all_preds)} preds in {elapsed:.1f}s ===')

# Eval
import importlib
import rlpe.evaluation.metrics, rlpe.evaluation.gold
importlib.reload(rlpe.evaluation.metrics)
importlib.reload(rlpe.evaluation.gold)
from rlpe.evaluation.metrics import evaluate
from rlpe.evaluation.gold import GoldPanel

print('\n--- Per-paper F1 (target figure only) ---')
target_figs_per_paper = {}
for p in all_preds:
    target_figs_per_paper[p['paper_id']] = p['figure_id']

for slug in slugs:
    gfp = GOLD_DIR / f'{slug}.jsonl'
    pdf_path = PAPERS_DIR / f'{slug}.pdf'
    if not gfp.exists() or not pdf_path.exists():
        continue
    pid = stable_id(pdf_path)
    paper_preds = [p for p in all_preds if p['paper_id'] == pid]
    if not paper_preds:
        continue
    target_fig = paper_preds[0]['figure_id']
    gold = [json.loads(l) for l in open(gfp) if l.strip()]
    gold_filtered = [g for g in gold if g.get('figure_id') == target_fig]
    gp = [GoldPanel(paper_id=g['paper_id'], figure_id=g.get('figure_id',''),
                    panel_id=g.get('panel_id'), species=g.get('species')) for g in gold_filtered]
    rep = evaluate(paper_preds, gp)
    f1 = rep.aggregate.get('species_f1_micro', 0)
    pp = rep.aggregate.get('species_precision', 0)
    rr = rep.aggregate.get('species_recall', 0)
    pm = rep.aggregate.get('panel_match_rate', 0)
    print(f'  {slug:20s} preds={len(paper_preds):3d} gold={len(gold_filtered):3d} | F1={f1:.3f} P={pp:.3f} R={rr:.3f} panel={pm:.3f}')

# Combined
total_gold_filtered = []
for slug in slugs:
    gfp = GOLD_DIR / f'{slug}.jsonl'
    pdf_path = PAPERS_DIR / f'{slug}.pdf'
    if not gfp.exists() or not pdf_path.exists():
        continue
    pid = stable_id(pdf_path)
    if pid not in target_figs_per_paper:
        continue
    target_fig = target_figs_per_paper[pid]
    for g in [json.loads(l) for l in open(gfp) if l.strip()]:
        if g.get('figure_id') == target_fig:
            total_gold_filtered.append(g)
gp_all = [GoldPanel(paper_id=g['paper_id'], figure_id=g.get('figure_id',''),
                    panel_id=g.get('panel_id'), species=g.get('species')) for g in total_gold_filtered]
rep_all = evaluate(all_preds, gp_all)
print(f'\n--- COMBINED (target figures only) ---')
print(f'preds={len(all_preds)}, gold={len(total_gold_filtered)}')
for k, v in rep_all.aggregate.items():
    if isinstance(v, float):
        print(f'  {k}: {v:.4f}')
    else:
        print(f'  {k}: {v}')

Path('/tmp/gold_eval_v4_preds.jsonl').write_text(
    '\n'.join(json.dumps(p) for p in all_preds)
)
print(f'\npreds saved to /tmp/gold_eval_v4_preds.jsonl')

for k in ['ANTHROPIC_API_KEY']:
    os.environ.pop(k, None)
print('[KEY CLEANED]')


# === Added for research-grade eval (Task 5) ===
#
# These helpers do NOT depend on the import-time LLM backend above.
# They are pure-Python, import-light, and safe to call from unit
# tests. ``load_split`` reads a JSON split file. ``compute_aggregate_
# with_ci`` groups predictions + gold by paper, computes micro F1
# point estimate, then derives a 95% bootstrap CI by resampling
# papers with replacement. ``run_5fold_cv`` performs 5-fold CV over
# the paper list and reports per-fold + aggregate F1.
import json
import statistics
import random
from pathlib import Path
from typing import Any, Tuple


def load_split(path: str | Path) -> dict[str, list[str]]:
    """Load train/test split from a JSON file."""
    with open(path) as f:
        return json.load(f)


def _paper_f1(preds: list[dict], gold: list[dict]) -> float:
    """Compute species F1 for a single paper."""
    from rlpe.evaluation.metrics import _norm_species, _species_compatible
    pred_sp = {(_norm_species(p.get('species')), p.get('figure_id'), p.get('panel_id'))
              for p in preds if p.get('species')}
    gold_sp = {(_norm_species(g.get('species')), g.get('figure_id'), g.get('panel_id'))
              for g in gold if g.get('species')}
    tp = sum(1 for k in pred_sp & gold_sp)
    fp = len(pred_sp - gold_sp)
    fn = len(gold_sp - pred_sp)
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def compute_aggregate_with_ci(
    preds: list[dict],
    gold: list[dict],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Compute micro F1 with 95% bootstrap CI.

    Returns (f1_micro, (ci_low, ci_high)).
    """
    # Group by paper
    by_paper: dict[str, tuple[list, list]] = {}
    for g in gold:
        by_paper.setdefault(g.get('paper_id', ''), ([], []))[1].append(g)
    for p in preds:
        paper = p.get('paper_id', '')
        if paper in by_paper:
            by_paper[paper][0].append(p)
    papers = list(by_paper.keys())

    def f1_micro() -> float:
        total_tp = total_fp = total_fn = 0
        for p in papers:
            pp, gp = by_paper[p]
            from rlpe.evaluation.metrics import _norm_species, _species_compatible
            pset = {(_norm_species(x.get('species')), x.get('figure_id'), x.get('panel_id'))
                    for x in pp if x.get('species')}
            gset = {(_norm_species(x.get('species')), x.get('figure_id'), x.get('panel_id'))
                    for x in gp if x.get('species')}
            tp = len(pset & gset)
            fp = len(pset - gset)
            fn = len(gset - pset)
            total_tp += tp; total_fp += fp; total_fn += fn
        if total_tp == 0:
            return 0.0
        p_val = total_tp / (total_tp + total_fp)
        r_val = total_tp / (total_tp + total_fn)
        if p_val + r_val == 0:
            return 0.0
        return 2 * p_val * r_val / (p_val + r_val)

    rng = random.Random(seed)
    point = f1_micro()
    if not papers:
        return 0.0, (0.0, 0.0)
    bootstraps: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choices(papers, k=len(papers))
        bt = total_fp_b = total_fn_b = 0
        for p in sample:
            pp, gp = by_paper[p]
            from rlpe.evaluation.metrics import _norm_species
            pset = {(_norm_species(x.get('species')), x.get('figure_id'), x.get('panel_id'))
                    for x in pp if x.get('species')}
            gset = {(_norm_species(x.get('species')), x.get('figure_id'), x.get('panel_id'))
                    for x in gp if x.get('species')}
            tp = len(pset & gset); fp = len(pset - gset); fn = len(gset - pset)
            bt += tp; total_fp_b += fp; total_fn_b += fn
        if bt == 0:
            bootstraps.append(0.0)
            continue
        p_v = bt / (bt + total_fp_b)
        r_v = bt / (bt + total_fn_b)
        bootstraps.append(2 * p_v * r_v / (p_v + r_v) if (p_v + r_v) > 0 else 0.0)
    bootstraps.sort()
    lo = bootstraps[int(0.025 * n_bootstrap)]
    hi = bootstraps[int(0.975 * n_bootstrap)]
    return point, (lo, hi)


def run_5fold_cv(
    preds_by_paper: dict[str, list[dict]],
    gold_by_paper: dict[str, list[dict]],
    all_papers: list[str],
    n_folds: int = 5,
) -> dict[str, Any]:
    """Run 5-fold cross-validation. Returns per-fold and aggregate F1."""
    rng = random.Random(42)
    papers = sorted(all_papers)
    rng.shuffle(papers)
    fold_size = max(1, len(papers) // n_folds)
    folds = [papers[i:i+fold_size] for i in range(0, len(papers), fold_size)]
    fold_metrics = []
    for i, fold in enumerate(folds):
        train_papers = [p for p in papers if p not in fold]
        preds = [x for p in train_papers for x in preds_by_paper.get(p, [])]
        gold = [x for p in train_papers for x in gold_by_paper.get(p, [])]
        f1, ci = compute_aggregate_with_ci(preds, gold, n_bootstrap=100)
        fold_metrics.append({'fold': i, 'papers': fold, 'f1': f1, 'ci': ci})
    f1s = [m['f1'] for m in fold_metrics]
    return {
        'folds': fold_metrics,
        'mean_f1': statistics.mean(f1s) if f1s else 0.0,
        'std_f1': statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
    }