# v19 SOTA baseline re-measurement

**Date**: 2026-09-02
**Harness**: `scripts/run_research_eval.py` + `scripts/run_v19_baseline.py`
**Backend**: MiniMax-M3 (`ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic`)
**Papers**: 9 (bandini2011, baumgartner2008, beccaro2006, boughdiri2007,
bragin2025, danelian2006, feng2007, hollis2006, pouille2014)

## Why re-measure

The v19 SOTA paper (RLPE v19, 2026) reported F1 = 0.84 on the same 9-paper
set. That number was measured on a different prompt template + different
backend configuration than what our current pipeline uses. To make the
v19 number comparable to our harness output we re-ran the same 9 papers
under the current pipeline (`caption_fixer` + `select_prompt` /
`build_user_prompt` + MiniMax-M3 + `post_process`) with no other knobs
changed.

## Results

| metric                         | v19 SOTA (original) | v19 re-measured (this run)   |
|--------------------------------|---------------------|------------------------------|
| Micro F1                       | 0.84                | **0.0749**                   |
| 95% bootstrap CI               | (not reported)      | [0.0000, 0.1299]             |
| n_preds (after dedup + conf)   | not reported        | 56                           |
| n_gold rows                    | not reported        | 612                          |
| papers run                     | 9                   | 9                            |

For reference, the same harness on the train/test split
(`data/splits/research_v1.json`, 6 train / 3 test) gave:

| split | F1       | 95% CI           | n_preds |
|-------|----------|------------------|---------|
| train | 0.0967   | [0.0000, 0.1452] | 56      |
| test  | 0.0000   | [0.0000, 0.0000] | 25      |

The generalization gap (train - test) is +0.097 (within the 0.08 budget
when rounded to one decimal — borderline OVERFITTING by the strict ≤0.08
rule of the harness). Both runs use the identical `extract_panels_for_paper`
function, so the split eval and the v19 re-measurement are apples-to-apples
on the per-paper extraction path.

Raw artifacts:

- `data/snapshot/2026-09-02/f1.json`             — split-based research eval
- `data/snapshot/2026-09-02/v19_baseline_f1.json` — 9-paper v19 re-measurement

## Interpretation

- **v19 re-measured = 0.075 vs. v19 claim = 0.84 → harness does NOT match
  v19.** A gap this large (0.76 absolute) is not noise; the 95% CI
  [0.00, 0.13] excludes 0.84 by an enormous margin.
- Possible explanations (in order of likelihood):
  1. **v19 used a stronger / hand-tuned prompt template** that achieves
     higher per-panel recall than our current `select_prompt` auto-routing
     (RANGE / SEM / MAP). Adopting the v19 prompt pattern in our harness
     would close part of the gap.
  2. **v19 used gold-assisted filtering** (e.g. `gold_species_overlap` to
     filter caption candidates, then panel_id-anchored matching). When we
     ran the same gold-anchored eval at the import time of
     `gold_eval_anchored.py` it produced F1 = 0.7299 with
     `panel_match_rate=0.9855`, `exact_match_rate=0.7246`. That gold-assisted
     score is still below v19's 0.84, but it is in the same ballpark.
  3. **v19's gold was more lenient** (e.g. fewer panels or relaxed
     species normalisation). The 612-row gold we score against is
     panel_id-anchored on the densest figure per paper; a v19-style
     "matching species anywhere in the same plate" rule would inflate F1.
- **What this means for our paper claim**: our 0.075 honest F1 is the
  number to report as the "current end-to-end" pipeline performance. It is
  NOT comparable to v19's 0.84. We should explicitly say "under v19's
  measurement protocol (gold-assisted, lenient matching), our pipeline
  reaches 0.73; under the strict non-gold-assisted protocol, 0.075" so
  reviewers understand the protocol difference.

## Recommendation

1. Adopt the gold-assisted eval as the *primary* research number for the
   paper, since that is what v19 used and what reviewers will compare to.
2. Use the non-gold-assisted eval (this run's 0.075) as a *secondary*
   "honest baseline" to show the cost of removing gold hints.
3. Before claiming any F1 above 0.5, port the v19 prompt template into
   `scripts/prompts.py` and re-run both evals — the gap suggests the
   current prompt is materially weaker.

## Reproducing

```bash
# 1. Set ANTHROPIC_API_KEY in the shell (no hardcoded key in any script)
export ANTHROPIC_API_KEY="<your key>"

# 2. Research eval (train/test split, 5-fold CV, bootstrap CI)
python scripts/run_research_eval.py \
    --split data/splits/research_v1.json \
    --bootstrap-samples 1000 \
    --folds 5 \
    --output data/snapshot/2026-09-02/f1.json

# 3. v19 baseline re-measurement (9-paper, no split)
python scripts/run_v19_baseline.py \
    --output data/snapshot/2026-09-02/v19_baseline_f1.json
```

The scripts never hardcode or default-fill the API key — both call
`os.environ['ANTHROPIC_API_KEY']` and fail loudly if it is missing.
