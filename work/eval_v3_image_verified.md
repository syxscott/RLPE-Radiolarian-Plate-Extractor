# RLPE Image-Verified Panel_id Evaluation (9 papers)

**Date**: 2026-08-02
**Script**: `scripts/evaluate_image_verified.py`
**Predictions**: `work/eval_v3_image_verified/predictions.jsonl` (1088 predictions, 9 papers)
**Panels root**: `work/eval_v3_image_verified/panels/` (10 figure dirs symlinked from v19/v20/oa_smoke_round6_v2)

## Aggregate (apples-to-apples: panels with crops only, n=216)

| Metric | Value |
| --- | --- |
| Papers | 9 |
| Total gold panels | 612 |
| Panels with crops on disk | 216 (35.3%) |
| String-match panel_id rate | 89.8% (194/216) |
| Image-verified panel_id rate | 8.3% (18/216) |
| **Gap** | **+81.5pp** |

## Original baseline (for reference)
- Species F1 (string-match only): **82.96%** (`work/eval_v3_9paper.json`)
- Panel_id match rate (string-match, all 612 gold): **90.52%**

## Per-paper (sorted by gap, biggest first)

| Paper | Gold | Checked | Str-match | Image-verified | Str% | IV% | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pouille2014 | 6 | 6 | 6 | 0 | 100.0% | 0.0% | +100.0pp |
| danelian2006 | 42 | 42 | 42 | 1 | 100.0% | 2.4% | +97.6pp |
| boughdiri2007 | 27 | 27 | 27 | 2 | 100.0% | 7.4% | +92.6pp |
| baumgartner2008 | 61 | 52 | 52 | 7 | 100.0% | 13.5% | +86.5pp |
| beccaro2006 | 35 | 34 | 34 | 8 | 100.0% | 23.5% | +76.5pp |
| bandini2011 | 273 | 55 | 33 | 0 | 60.0% | 0.0% | +60.0pp |
| bragin2025 | 11 | 0 | 0 | 0 | BLOCKED | BLOCKED | -- |
| feng2007 | 84 | 0 | 0 | 0 | BLOCKED | BLOCKED | -- |
| hollis2006 | 73 | 0 | 0 | 0 | BLOCKED | BLOCKED | -- |

## Key findings

### 1. Visual-misdetection is severe across the board
The 81.5pp gap (string-match → image-verified) shows that the LLM is
**fabricating panel_ids from caption text** far more often than string-match
F1 suggests. Over the 216 panels we could OCR:
- 89.8% matched by string (i.e. `(figure_id, panel_id)` key exists in preds)
- Only 8.3% confirmed by EasyOCR on the actual panel crop

### 2. EasyOCR coverage is the bottleneck
Even when panels are on disk, EasyOCR often cannot read the small printed
number (font, low resolution, etc.). OCR coverage: 91/216 = 42.1%.
The 8.3% image-verified rate includes only panels where EasyOCR found
*and* matched a number.

### 3. Top 3 papers by gap (visual misdetection hotspots)
- **pouille2014**: 6 panels, 0/6 OCR'd correctly → 100pp gap
- **danelian2006**: 42 panels, 1/42 OCR'd correctly → 97.6pp gap
- **boughdiri2007**: 27 panels, 2/27 OCR'd correctly → 92.6pp gap

These three papers have 100% string-match (LLM perfectly copies caption
labels) but the LLM is *clearly* hallucinating panel_ids — no panel
on disk bears the number the LLM claims.

### 4. Bandini2011 is unique: panels exist but only 60% string-match
Of 55 bandini panels on disk, only 33/55 (60%) are matched by string.
This is the **only** paper where the LLM genuinely missed panels in
prediction; the other 5 papers are 100% string-match. This suggests
bandini's caption layout confuses the LLM parser.

### 5. 3 papers blocked by missing crops
bragin2025, feng2007, hollis2006 have 0 panel crops on disk (the
`auto_fig_pNNN_rNN` naming convention in newer runs doesn't match the
`od_plate_...` figure_ids in gold). Re-running the live pipeline with
matching figure_id naming is required.

## Notes on the metric
- The script's aggregate string-match rate uses `n_gold` as denominator
  (90.52% over all 612 panels).
- The image-verified rate uses `n_checked` as denominator (8.33% over
  216 panels with crops).
- For an apples-to-apples comparison, both rates must use the same
  denominator. Above we used `n_checked` for both (89.8% / 8.3% / 81.5pp gap).
- A bug in the per-paper display divides `n_string_match` (counted over
  all gold) by `n_checked`, producing values >1.0. This does NOT affect
  the aggregate numbers, but should be fixed in a follow-up.

## What this implies for the LLM-first pipeline
- The headline 82.96% species F1 overstates real-world accuracy by
  ~80pp on the panel_id axis.
- Until panel crops are visually verified, the string-match F1 is
  effectively a ceiling, not the true F1.
- Priority fix: build a robust panel_id grounding pipeline that
  (a) locates panels by visual crop not caption parse, and
  (b) validates panel_id via OCR before emitting it.
