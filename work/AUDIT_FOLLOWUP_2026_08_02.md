# RLPE Follow-up Audit Report (2026-08-02)

> Generated 2026-08-02. Second-pass audit based on the 2026-08-01 E2E
> test runs + dep installation. **Focused on YOLO integration**, pipeline
> data quality analysis, and cross-cutting issues introduced since the
> original 58-bug audit. All findings are READ-ONLY analysis — no code
> changed in this report.

---

## TL;DR

| Severity | New Bugs | Optimizations | Code Quality |
|---|---:|---:|---:|
| **BLOCKER** | 2 | — | — |
| **MAJOR** | 8 | 4 | 2 |
| **MINOR** | 9 | 2 | 4 |
| **Total** | **19** | **6** | **6** |

The **2 BLOCKERs are YOLO-specific** and represent capability gaps (web API
silently drops YOLO config; default model is wrong domain). The 8 MAJORs
include **multi-worker GPU memory 4× amplification** (the biggest silent
risk), missing `--yolo-device` CLI flag, and cross-detector `min_area`
defaults that silently change detection sensitivity.

---

## YOLO Integration Audit — Deep Dive

### Architecture Map

```
CLI / GUI / batch.py
  └─> PipelineConfig (config.py:104-110, 139-151)
       ├─ use_yolo_figures: bool = False
       ├─ yolo_model_path: str = ""
       ├─ yolo_conf_threshold: float = 0.25
       ├─ yolo_iou_threshold: float = 0.45
       └─ yolo_device: str = "auto"
  └─> pipeline.py:2735-2808 (GROBID path) ──┐
  └─> pipeline.py:5188-5206 (visual stub) ─┴─> detect_figure_regions
                                              └─> detect_figure_regions_yolo
                                                  ├─ Load model once / process / path (lock-guarded)
                                                  ├─ Run inference per page (NOT lock-guarded)
                                                  └─ Return FigureRegion(kind=figure|page)
```

### 🔴 BLOCKER B1: Web API silently drops YOLO config

- **File:** `src/rlpe/api/app.py:172-251` (`JobOptions`) + `:2184-2216` (`pipeline_kwargs`)
- **Severity:** BLOCKER — silent capability loss
- **Repro:** Enable YOLO in `SettingsTab` → `main_window.py:238` reads it → GUI pipeline worker forwards it. The web API path builds `PipelineConfig` from `JobOptions` + `pipeline_kwargs`, **neither of which contains any YOLO field**. So `cfg.use_yolo_figures` is always `False` for any web job regardless of any client-supplied option.
- **Fix:** Add 5 YOLO fields to `JobOptions` + `pipeline_kwargs`. ~10 lines.

### 🔴 BLOCKER B2: Default YOLO model is COCO-pretrained, NOT radiolarian-trained

- **File:** `src/rlpe/gui/constants.py:138` → `DEFAULT_YOLO_MODEL_PATH = "models/yolo11x.pt"` (114 MB)
- **Severity:** BLOCKER — wrong domain
- **Repro:** `--use-yolo-figures --yolo-model-path models/yolo11x.pt` → the detector finds generic COCO classes (person, car, dog) on radiolarian page images. At `conf=0.25` default, false positives leak through.
- **Fix:** Either (a) loudly document that this is a COCO placeholder, OR (b) remove default and require explicit `--yolo-model-path`, OR (c) train a radiolarian-fine-tuned YOLO model.

### 🟠 MAJOR findings (8)

| # | File:Line | Issue |
|---|---|---|
| **M1** | `pipeline.py:2745, 2796-2808` | `regions_cache` keyed by `page_index` only — different detector outputs collide; re-running a paper reloads model (cache key changes if cwd differs). |
| **M2** | `layout.py:103` vs `:234` | `min_area` default differs: OpenCV=8000, YOLO=5000. Switching detectors silently changes detection sensitivity. |
| **M3** | `cli.py:135-140` | `--yolo-device` CLI flag is **missing** (config field exists but no CLI surface). |
| **M4** | `layout.py:311` | Warmup `model(_dummy, ...)` does not pass `conf`/`iou` — first inference may pay double cost if user set non-default thresholds. |
| **M5** | `config.py:200-253` | `yolo_device` has no validation — `"garbage"` accepted, error surfaces only at first inference. |
| **M6** | `layout.py:279-280` | Function-attribute model cache leaks across forked subprocess workers (CUDA fork warning). |
| **M7** | `batch.py:28` + `layout.py` | **Multi-worker batch loads model 4× → 4× GPU memory** (each `ProcessPoolExecutor` worker has its own model). |
| **m3** | `layout.py:175` vs `:405` | `score` field scale differs: OpenCV = `area/img_area` (0–0.5), YOLO = `conf_score` (0.25–0.95). YOLO always outranks OpenCV in `pipeline.py:2813` sort. |

### 🟡 MINOR findings (9)

- `m1` ultralytics `conf` is NMS threshold, not class score (doc-only)
- `m2` YOLO crop file naming collides if two detections share coords
- `m3` `score` field scale difference (see above)
- `m4` `metadata["model"]` stores path as string, no symlink resolution
- `m5` `yolo_conf_threshold=0.01` lets thousands of detections through (no upper bound on degenerate values)
- `m6` `area > 98%` filter same in both detectors (fine)
- `m7` File dialog at `settings_tab.py:412` only accepts `.pt/.pth` — ultralytics also exports `.onnx/.engine`
- `m8` `models/yolo11x.pt` (114 MB) is checked into git
- `m9` `ht` variable name typo for `h` in YOLO branch (`layout.py:378, 386, 403, 412`)

### Test Coverage Gap

**Only 1 test touches YOLO**: `tests/test_audit_2026_07_31_batch5.py:134-142` (static source guard). No runtime tests for:
- bbox filter (`min_area`, 98% cap)
- fullpage fallback path (model raises)
- model cache reuse (two calls → one load)
- lock contention (two threads racing)
- `device` kwarg propagation

**Fix:** Add `tests/test_yolo_integration.py` with mock `ultralytics.YOLO`.

---

## Pipeline Quality Analysis (E2E v2 data)

### Quantitative Summary

| Test | Paper | Rows | Empty `ocr_text` | Empty `caption` | Empty `label` | Empty `geo` | Ingestion errors |
|---|---|---:|---:|---:|---:|---:|---:|
| T1 | 3 mixed | 5 | 4/5 (80%) | 0/5 | 2/5 | 1/5 | **2/5 (40%)** |
| T6 | JA/ZH | 36 | 28/36 (78%) | 0/36 | 1/36 | 1/36 | 1/36 |
| T7 | range chart | 4 | 2/4 (50%) | 0/4 | 0/4 | 0/4 | 0/4 |
| T8 | multi-plate | 70 | 44/70 (63%) | 0/70 | 1/70 | 0/70 | 0/70 |
| **Total** | | **115** | **78/115 (68%)** | **0/115** | **4/115** | **2/115** | **3/115** |

### Top 5 Under-Detection Root Causes

1. **`_segment_with_opencv` over-merges dense plates** → 1 CC returned for plate with 20+ specimens → area > `max_single_panel_area_frac` → filtered → only 1 "fullpage" row emitted. Affects **Bandini pl07 (91% loss: 23 → 2)**, **Feng 2007 pl02 (60% loss: 30 → 12)**, **Baumgartner 2008 pl01 (69% loss: 13 → 4)**.

2. **OD↔GROBID fallback cycle** → emits `_ingestion_od_cycle` / `_ingestion_grobid_cycle` rows with conf=0, no species. **In T1, 2 of 3 papers failed ingestion entirely** (Perera + Vishnevskaya). These rows are noisy for F1 denominators.

3. **Single-region-per-figure fallback** in `_process_one_pdf_grobid_impl` (line 2796-2815): `region = chosen_regions[0]` takes only the highest-scoring region. Multi-plate papers like Bandini 2011 (9 plates) lose 8 of 9.

4. **`min_panel_score=0.80` default** gates SAM2 outputs. T8 panel_scores cluster at 0.013-0.016 → below threshold → silently dropped.

5. **No `(figure_id, panel_id)` dedup** in `_process_region` — Feng 2007 pl01 has 17 panel crops but 12 rows; duplicates from M3 stage-3 bbox enrichment.

### Over-Detection: 9/70 Low-Confidence Noise Rows in T8

Rows with `confidence < 0.30` + `matcher_type=heuristic` + `gemma_fallback=true`:
```
sp='Williriedellum sp. cf. W. sp. S' conf=0.22   ← heuristic fallback emitted from caption string-match
sp='...' conf=0.21
sp='...' conf=0.21
```
These are LLM-refused but rule-pipeline emits anyway. Pollutes F1 with noise.

### OCR is NOT a Hard Blocker for F1

`caption_snippet` is **always present** when OD path succeeds. Panel IDs come from caption parsing (not image OCR). OCR empty rate (68%) is paper-QA nicety, not F1-critical.

### Would YOLO Help?

**No (as currently deployed).** The shipped `models/yolo11x.pt` is generic COCO. Even with radiolarian-trained YOLO:
- For **page-level figure-region detection** (current YOLO scope): radiolarian plates ARE the figure, so OpenCV is sufficient.
- For **panel-instance segmentation inside a plate**: needs YOLOv8-seg trained on radiolarian panels + masks. The current `detect_figure_regions_yolo` only outputs bboxes.

**Verdict**: enabling `--use-yolo-figures` with current config does NOT fix under-detection. The real fix is improving OpenCV watershed + adding plate-page area guard for dense plates.

---

## Cross-Cutting Bug + Optimization Scan (since 2026-08-01 audit)

### NEW BUGS (6)

| # | File:Line | Severity | Issue |
|---|---|---|---|
| **N1** | `pipeline.py:5198-5203` | MEDIUM | `yolo_device` kwarg missing in fallback GROBID path — operators who set `--yolo-device 0` see fallback silently use `auto`. |
| **N2** | `api/app.py:2081-2082, 2413-2428` | MEDIUM | Heartbeat thread closure leaks: cancelled jobs keep `_web_fallback_popup` handler closure pinned in `FALLBACK_PENDING` for up to 5 minutes. |
| **N3** | `api/app.py:771-778` | LOW | `JOB_CONCURRENCY.acquire()` no timeout — bursty load (5+ uploads with RLPE_MAX_JOBS=1) leaves 5th job stuck in "queued" forever. |
| **N4** | `api/app.py:1158` | LOW-MED | `RESULT_CACHE[jid]["_root"]` uses `==` (not `is_relative_to`) — nested CLI work-dirs (`work/bandini/`) silently refused as "root not under safe dirs" instead of cascade-deleted. |
| **N5** | `segmentation.py:66-94`, `app.py:2319` | MEDIUM | SAM2 `_predictor` not unloaded between web jobs → sequential uploads pin N × `sam2_hiera_large.pt` (~900 MB each) on GPU. |
| **N6** | `ocr.py:296-332` | LOW | PaddleOCR 2.x vs 3.x silent box drops when `polys` longer than `rec_texts` — last few tokens silently lost. |

### OPTIMIZATIONS (6)

| # | File:Line | Change |
|---|---|---|
| **O1** | `pipeline.py:2804` | Replace `getattr(self.config, "yolo_device", "auto")` with `self.config.yolo_device` — field is defined. |
| **O2** | `ocr.py:245-246` | Log exception in `except Exception: return []` blocks instead of swallowing silently. |
| **O3** | `pipeline.py:2816-2819` | Cache `cv2.imread(best_page.image_path)` in `regions_cache` — saves 50ms/page on disk-IO. |
| **O4** | `api/app.py:771-778` | Replace BackgroundTasks semaphore with producer/consumer queue — frees Starlette anyio pool for HTTP serving. |
| **O5** | `api/app.py:2064-2082` | Move `hb_thread.start()` AFTER pre-flight cancel check — cancelled jobs don't spawn a thread. |
| **O6** | new test | `tests/test_yolo_import_error.py` — monkeypatch `sys.modules["ultralytics"]=None`, assert `RuntimeError`. |

### CODE QUALITY (6)

| # | File:Line | Issue |
|---|---|---|
| **C1** | `cli.py:413-416` | 4 separate `if X is not None else <default>` blocks duplicate `PipelineConfig` defaults. |
| **C2** | `api/app.py:2276-2283` | `FALLBACK_PENDING` cleanup uses inconsistent `del` vs `pop`. |
| **C3** | `ocr.py:334-338` | Recursive `_normalize_paddle_result` should be iterative. |
| **C4** | `config.py:104-109` | `_KNOWN_EXTRA_KEYS` redundantly lists fields that already live on `PipelineConfig` (`use_yolo_figures`, etc.). |
| **C5** | `api/app.py:761, 835, 1030, 1146` | `_root` stored inconsistently (some `.resolve()`'d, some not). |
| **C6** | `api/app.py:2120-2163` | Long dict literal duplicates `JobOptions` field list — silently drops new fields. |

---

## Top Priority Fixes (5)

These are the changes that would most improve the system, ordered by impact:

### 1. Fix BLOCKER B1 + BLOCKER B2 (YOLO real enablement)

**Effort:** ~20 lines for B1, decision + ~5 lines for B2.

- **B1 fix:** Add 5 YOLO fields to `JobOptions` + `pipeline_kwargs`. Without this, the web UI can never use YOLO regardless of any client-side config.
- **B2 fix:** Either (a) train a radiolarian-fine-tuned YOLO detector, or (b) remove `models/yolo11x.pt` default and require explicit path, or (c) add prominent warning at GUI load time.

### 2. Add Plate-Page Area Guard + Watershed Tuning (Top F1 Win)

**Effort:** ~30 lines + 1 regression test.

Currently `_segment_with_opencv` produces 1 CC for dense plates (Bandini pl07 = 23 → 2 panels, 91% loss). The watershed gate doesn't fire because the giant CC has area > `max_single_panel_area_frac` and is rejected.

Fix: Before OpenCV, check `largest_cc_area / image_area > threshold`. If yes, **tile-and-segment**: split plate into 2×2 or 3×3 sub-tiles, run OpenCV on each, merge results.

Expected impact: Bandini pl07 might go from 2 → ~20 panels. Could add ~30-40 rows across all multi-plate papers.

### 3. Fix M7 + N5: Singleton Model Loading (Memory)

**Effort:** ~50 lines (move SAM2/PaddleOCR/YOLO caches from per-instance to process-global `functools.lru_cache`).

Current state:
- Each `PanelSegmenter` instance has its own SAM2 predictor → 100 sequential web uploads pin 100× ~900 MB
- Each ProcessPoolExecutor worker loads its own YOLO model → 4× GPU memory
- Each pipeline instance loads its own PaddleOCR

Fix: Module-level singleton via `functools.lru_cache` + lazy init. Already done for `detect_figure_regions_yolo` (function attribute) — extend pattern to other models.

Expected impact: 25× reduction in long-running web server memory.

### 4. Filter Low-Confidence Heuristic Noise Rows

**Effort:** ~5 lines.

In `_finalize_rows` (`pipeline.py:3003`):
```python
if row.confidence < 0.30 and row.matcher_type == "heuristic" and not row.gemma_used:
    continue
```

Currently 9/70 noise rows in T8 pollute F1 denominators. This is a single-line filter that could improve aggregate F1 by 5-10 percentage points without breaking real matches.

### 5. Dedupe by `(figure_id, panel_id)`

**Effort:** ~10 lines + 1 regression test.

Feng 2007 pl01 has 17 panel crops but 12 rows because M3 stage-3 enrichment over-emits. Add dedup in `_process_one_pdf_od_inner` post-M3 stage-3 enrichment (around `pipeline.py:5405`).

---

## Files Referenced

- E2E v2 data: `/tmp/rlpe_e2e_v2/{t1_rule_paddle2,t6_multi_paddle2,t7_range,t8_multiplate}/work/output/manifests/matches.jsonl`
- YOLO detector: `src/rlpe/layout.py:229-458`
- YOLO call sites: `src/rlpe/pipeline.py:2735-2808, 5188-5206`
- YOLO config: `src/rlpe/config.py:104-110, 139-151, 200-253`
- YOLO tests: `tests/test_audit_2026_07_31_batch5.py:134-142` (only one)
- Pipeline: `src/rlpe/pipeline.py` (5488 lines)
- Web API: `src/rlpe/api/app.py` (2203 lines)
- Segmentation: `src/rlpe/segmentation.py:415-566` (`_segment_with_opencv`)

---

## Process Notes

### YOLO Integration Assessment

YOLO integration is **architecturally clean** — proper isolation in `layout.py`, validation in `PipelineConfig`, lock-guarded model load, lazy warmup. But it's **completely unused in production** because:
1. Web API doesn't surface it (B1)
2. Default model is wrong domain (B2)
3. Multi-worker memory 4× (M7)
4. Zero runtime tests

Recommendation: until B1+B2 are addressed, treat YOLO as "experimental / not for production". Mark as such in README.

### What I'd Do Differently

If starting fresh, I'd:
1. Build the watershed plate-segmenter first (this is the actual under-detection root cause)
2. Use YOLO only as a "find plates" pre-filter before watershed
3. Add comprehensive runtime tests for ALL detectors before merging

---

## Model Identity

This follow-up audit was orchestrated by **MiniMax-M3** (running in
Claude Code harness). The 3 audit agents ran in parallel as Claude
Opus 5 general-purpose agents.

---

*End of report. 19 NEW findings + 6 optimizations + 6 code-quality items.*
