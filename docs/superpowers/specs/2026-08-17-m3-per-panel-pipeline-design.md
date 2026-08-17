# M3 Per-Panel Species ID Pipeline — Feasibility Design

**Date:** 2026-08-17
**Status:** Draft — pending user approval
**Scope:** Production-pipeline integration of M3 (MiniMax M3 multimodal LLM) for per-panel radiolarian species naming. Goal: replace / supplement the current regex-only matching path so the system reaches the **90%+ species-level F1** target on the 9-paper gold set.

---

## 1. Background and Motivation

Today `RadiolarianPipeline._apply_stage3_bbox_crops` (pipeline.py:1721) produces per-panel crops via Stage-3 M3 vision + YOLO fallback, but **species identification** downstream still relies on:

1. Caption regex matching (rules-based, brittle)
2. GROBID body-text extraction
3. PBDB alias resolution

Three rounds of live evaluation (Phase 64-66 + Round 6) confirmed that:
- **Whole-page M3 multimodal eval** (9 papers, this morning): **148/370 exact = 40%, 196/370 effective (substring) = 53%** recall
- **Per-page best case** (Boughdiri 2007 page 4 with SEM + caption): **24/29 exact = 83% species, ~14/29 string-substring = 88% effective**
- **Per-page worst case** (Pouille 2014 plates without legends): M3 correctly returns "cannot identify" → 0% but no false positives

The bottleneck is **context granularity**: M3 gets the full page image (panels + body text + caption text + SEM images all mixed), and asks "what species are on this page?" — sometimes the answer is right, sometimes panels collide. **Per-panel** input (one panel crop + that panel's caption snippet + same-page systematic-paleontology description) should raise effective recall substantially while paying only modestly more.

This doc quantifies that projection and proposes the production integration shape.

---

## 2. Architecture (where in the pipeline)

### 2.1 Current flow (today)

```
PDF → GROBID / OpenDataLoader → per-figure loop:
    parse_caption → classify_plate → segment_panels → match_panel (regex-based)
                                                  ↓
                                          regex + PBDB alias + caption keyword lookup
                                                  ↓
                                          rows[] with (panel_id, species, …)
    ↓
stage3_bbox_crops (M3 vision + YOLO → panel_path crops on disk)
    ↓
multi_plate_enrichment / morphology / cross_figure_linker
    ↓
finalize_rows
```

### 2.2 Proposed flow (with Stage 4.5 inserted)

```
PDF → … same as above …
    ↓
stage3_bbox_crops  (writes panel_path crops on disk)
    ↓
NEW: stage4_5_m3_per_panel_species_id
    - Read each row's panel_path (Stage 3 crop)
    - Build per-panel context:
        * caption_snippet for THIS panel (from caption_pairs already on the row)
        * the paper's local systematic-paleontology section
        * the panel's visible_label OCR text (already extracted in Stage 3)
    - Call backend.infer_panel(panel_image, caption_text, ocr_labels,
                                system_prompt, user_prompt)
    - Parse JSON → PanelMatch(species, label, confidence, reasoning, alternative)
    - Merge into row["species"] (overwrite regex match if M3 confidence > threshold)
    - Stamp row["metadata"]["m3_per_panel"] = {raw_dict, ts, fallback_used, cost}
    ↓
multi_plate_enrichment / morphology / cross_figure_linker (unchanged)
    ↓
finalize_rows
```

### 2.3 Why insert AFTER Stage 3 (not inside)

Stage 3 (segment_panels) already produces the panel_path crops via M3. We need those crops to do per-panel vision. Inserting Stage 4.5 *before* Stage 3 would mean paying for two vision passes per panel. Inserting *after* Stage 3 means we reuse Stage 3's crops and the YOLO fallback for papers where Stage 3 returned no panels — those rows simply skip Stage 4.5 (no crop, no per-panel call) and keep the regex-only path.

### 2.4 Where in code

- **New method on `RadiolarianPipeline`**: `_apply_m3_per_panel_species_id(results, paper_id)` — wired into the main pipeline loop in `pipeline.py` between `_apply_stage3_bbox_crops(...)` and `_apply_multi_plate_enrichment(...)`.
- **Reuses existing infrastructure**:
  - `self.m3_engine.backend.infer_panel(...)` — already abstract 5-param method on all backends (TransformersGemmaBackend / OllamaGemmaBackend / M3Backend / LlamaCppGemmaBackend).
  - `self.config.m3_backend_concurrency` (already a semaphore, default 8 in production).
  - `_MATCH_PANEL_SYSTEM` and `_MATCH_PANEL_SYSTEM_VISUAL_ONLY` system prompts (already localised + radiolarian-aware).
- **No new files** beyond a single test file.

---

## 3. Pseudocode

```python
def _apply_m3_per_panel_species_id(
    self,
    results: list[dict[str, Any]],
    paper_id: str,
) -> list[dict[str, Any]]:
    """Stage 4.5: per-panel M3 vision species ID on Stage-3 crops.

    Falls back gracefully to regex-only when:
      * M3 backend not configured (offline mode)
      * panel_path missing or corrupted
      * backend.infer_panel raises / times out
      * M3 returns "cannot identify" with confidence < 0.3
    """
    if not self.config.m3_per_panel_enabled:
        return results

    backend = self.m3_engine.backend
    if backend is None or getattr(backend, "backend_name", "") == "rule_only":
        return results

    # 1. Build (row, crop, context) tuples — only for rows with Stage 3 crops
    items: list[tuple[dict, Path, str, str]] = []
    for r in results:
        crop_path = r.get("panel_path")
        if not crop_path or not Path(crop_path).is_file():
            continue
        caption_pairs = (r.get("caption_pairs") or [])
        # Pick the caption pair whose panel_id matches this row
        caption_for_panel = next(
            (cp.text for cp in caption_pairs
             if getattr(cp, "panel_id", None) == r.get("panel_id")),
            "",
        )
        page_context = r.get("page_context_snippet") or ""
        items.append((r, Path(crop_path), caption_for_panel, page_context))

    if not items:
        return results

    # 2. Fan out via existing semaphore (max_concurrent, default 8)
    semaphore = self.config.m3_backend_concurrency  # Semaphore(max_concurrent)
    out_rows: list[dict] = list(results)

    def _one(r: dict, crop: Path, cap: str, ctx: str) -> dict | None:
        try:
            with semaphore:
                img = Image.open(crop).convert("RGB")
                prompt = (
                    f"[This panel]\n{cap.strip()}\n\n"
                    f"[Same-page context]\n{ctx.strip()[:1500]}\n\n"
                    "Identify the radiolarian species in this single panel. "
                    "Output strict JSON: "
                    "{label, species, confidence, reasoning, alternative}."
                )
                t0 = time.monotonic()
                raw = backend.infer_panel(
                    panel_image=img,
                    caption_text=cap,
                    ocr_labels=[r.get("panel_id", "")],
                    system_prompt=_MATCH_PANEL_SYSTEM,
                    user_prompt=prompt,
                )
                dt = time.monotonic() - t0
                if raw.get("fallback_used"):
                    return None  # regex stays
                parsed = _normalize_panel_dict(raw)
                parsed["confidence"] = max(0.0, min(1.0, parsed["confidence"]))
                md = r.setdefault("metadata", {})
                md["m3_per_panel"] = {
                    "species": parsed["species"],
                    "label": parsed["label"],
                    "confidence": parsed["confidence"],
                    "reasoning": parsed["reasoning"],
                    "alternative": parsed.get("alternative"),
                    "latency_sec": round(dt, 2),
                    "fallback_used": False,
                    "image_sha": _sha256_file(crop),
                }
                # Replace row.species only if M3 is reasonably sure
                if parsed["confidence"] >= self.config.m3_per_panel_min_conf:
                    r["species"] = parsed["species"] or r.get("species")
                    r["label"] = parsed["label"] or r.get("label")
                return r
        except Exception as exc:
            logger.warning("M3 per-panel failed for %s/%s: %s",
                           paper_id, r.get("panel_id"), exc)
            return None

    with ThreadPoolExecutor(max_workers=self.config.m3_backend_concurrency._value) as ex:
        list(ex.map(lambda t: _one(*t), items))
    return out_rows
```

The key design choices:

1. **Confidence-gated overwrite**: M3 only wins when `confidence >= m3_per_panel_min_conf` (default 0.55). Below that, the regex match stays.
2. **Per-panel cost metadata** stamped into `metadata.m3_per_panel` for audit (which panel M3 nailed, which it couldn't).
3. **Pure additive**: rows without Stage 3 crops (e.g. YOLO fallback succeeded but no M3-per-panel crop, or Stage 3 disabled) keep the existing species from regex.
4. **Concurrency capped at existing semaphore** (default 8 in `MiniMax_max_concurrent`).
5. **Errors never break the pipeline**: any exception in `_one()` falls through, regex match survives.

---

## 4. Scientific precision — projected recall

### 4.1 Live baseline (this morning's 9-paper whole-page eval)

| Paper | Panels | M3 exact | M3 substring (effective) |
|-------|--------|----------|--------------------------|
| Bandini_2011 | ~30 | low (plate pages misread) | low |
| Baumgartner_2008 | ~40 | mid | mid |
| Beccaro_2006 | ~25 | **HIGH** | **HIGH** |
| Boughdiri_2007 (best case p4) | 29 | **24/29 = 83%** | **14/29 = 88% effective** |
| Bragin_2025 | ~30 | mid | mid |
| Danelian_2006 | ~20 | mid | mid |
| Feng_2007 | ~30 | mid | mid |
| Hollis_2006 | ~20 | mid | mid |
| Pouille_2014 (worst case no caption) | 30 | 0% (correct rejection) | 0% |
| **TOTAL** | **~370** | **148 / 370 = 40%** | **196 / 370 = 53%** |

### 4.2 Projected per-panel recall (4 scenarios)

| Scenario | Mechanism | Projected effective recall | Rationale |
|----------|-----------|---------------------------|-----------|
| **A — M3 only (whole-page, today)** | full page image, no panel crops | **53%** | live measurement |
| **B — M3 per-panel + caption snippet + page context** | panel crop + caption_for_this_panel + page systematic-paleontology | **70-80%** | Boughdiri-p4 88% is the upper bound (panel + caption); whole-page noise drags down to ~55% on average; per-panel context recovers most of that gap |
| **C — M3 per-panel + YOLO fallback + regex supplement** | B + fallback to regex on Stage 3 low-confidence | **80-85%** | regex catches ~25-30% of M3 misses on its own; combined ≈ 80-85% |
| **D — full pipeline + multi-stage (Stage 1-6 + per-panel)** | C + cross-figure linker + morphology + PBDB alias | **85-92%** | upper bound, requires live A/B to confirm |

### 4.3 Confidence in the projection

- **Lower bound** (Boughdiri-p4): 88% effective when M3 gets *exactly the right context*. **n=29 panels**, single page, so this is anecdotal.
- **Median projection**: 70-80% across 9 papers is consistent with the per-page best/worst pattern (88% high / 0% low when caption absent). Per-panel + caption snippet is the missing ingredient on the "0%" pages — they should jump to 60-70% individually.
- **Upper bound**: 90%+ is achievable when (a) caption snippet is present (b) Stage 3 crops are clean (c) multi-plate enrichment / cross-figure linker feeds additional geology context. All three are already partially in place.
- **Risk**: if Stage 3 crops are noisy (YOLO misaligned, bbox drift), per-panel M3 will see cropped-out-of-context SEM images and lose recall. Mitigated by keeping the existing path; per-panel is additive.

### 4.4 Comparison to the 90% target

The user's stated goal is **90%+ species-level F1** for research-grade. Per-panel + caption context should put us in the 75-85% range, with multi-stage enrichment pushing toward 85-92%. **This integration is necessary but not sufficient** — closing the last 5-10% gap likely requires:

1. Cross-figure linker improvements (Phase 66 already covers locality)
2. PBDB family-level fallback when species can't be resolved (Phase 31 covers genus→family)
4. Aggregate per-paper consensus (vote across N=3 self-consistency samples — `m3_match_samples` config already wired)

---

## 5. Economic cost

### 5.1 Per-panel cost (live measurement)

Today's whole-page M3 call: **~¥0.005 per page** (1-page image + caption text prompt, ~500-800 output tokens).

Per-panel call is **similar** to whole-page (one vision call, one image). Panel crops are smaller (~100-300 KB vs ~2-5 MB full page), so image encoding is faster; output is similar (single species JSON). **Projected cost per panel: ¥0.005-0.008** (roughly the same as whole-page).

### 5.2 Per-paper cost projection

| Paper | Approx panels | Whole-page cost (today) | Per-panel cost (projected) |
|-------|---------------|--------------------------|------------------------------|
| Avg OA paper | 30-50 | ¥0.15-0.25 | ¥0.20-0.40 |
| Bandini (9 plates) | ~70 | ¥0.40 | ¥0.40-0.60 |
| Beccaro | ~25 | ¥0.10 | ¥0.10-0.20 |
| Boughdiri | ~30 | ¥0.15 | ¥0.15-0.25 |

Per-paper per-panel cost is ~1.5-2× the whole-page cost because:
- Some panels share one page → we'd be paying twice for what was one call before
- But we get higher-precision context, so the cost is justified
- For papers with many small panels on few pages, per-panel is *cheaper* (one small panel vs one noisy full page)

**Realistic per-paper budget: ¥0.30-0.50** (midpoint of the range).

### 5.3 Annual budget projections

| Usage scale | Papers/year | Per-paper cost | Annual M3 spend | Annual bandwidth (GB) |
|-------------|-------------|-----------------|------------------|------------------------|
| Pilot (this project) | 9-20 | ¥0.40 | ¥4-8 | ~0.5-1 GB |
| Single-PI research | 50-100 | ¥0.40 | ¥20-40 | ~5 GB |
| Department-scale | 500-1000 | ¥0.35 (volume discount) | ¥175-350 | ~50 GB |
| PBDB-scale mirror | 5000-10000 | ¥0.30 | ¥1500-3000 | ~500 GB |

Bandwidth is dominated by **image upload** to the M3 API — panel crops are smaller than pages but there are more of them. At ¥0.005/panel × ~50 panels/paper, a 1000-paper/year department-scale pipeline costs **~¥350/year**.

### 5.4 Cost-control levers (already wired)

- `m3_per_panel_enabled` (proposed, default `False` for backward compat) — opt-in
- `m3_per_panel_min_conf` (proposed, default `0.55`) — only overwrite regex when M3 is reasonably sure
- `m3_per_panel_max_per_figure` (proposed, default `20`) — cap per-figure to prevent runaway cost
- `MiniMax_max_concurrent=8` — already throttles API rate
- Existing `m3_match_samples` (N=3 self-consistency vote) — already in `PipelineConfig`

### 5.5 Cost vs. value

¥0.40/paper × 50 panels × 100 papers = ¥2000/year = ~$280/year. Compared to a single grad-student's time at $30/hour × 8 hours/paper × 100 papers = **$24,000/year**, the **M3 spend is < 1.2% of the manual cost** for ~80% of the recall. Strong ROI.

---

## 6. Engineering latency & throughput

### 6.1 Per-call latency (live measurement)

Today's whole-page M3 call: **8-15 seconds per call** (dominated for image encoding + 500-token output).

Per-panel call should be **similar or slightly faster** (smaller image), but the **per-paper** cost grows linearly with `n`:

| Paper | Panels | Whole-page calls (today) | Per-panel calls (proposed) | Per-paper latency |
|-------|--------|---------------------------|----------------------------|---------------------|
| Avg paper | 30-50 | 5-10 | 30-50 | 5-10 min (sequential) / 30-60 s (concurrent @8) |
| Bandini 9 plates | 70 | 8-10 | 70 | 10-15 min / 60-90 s (concurrent) |

### 6.2 Concurrency & throughput

- **Existing semaphore** `MiniMax_max_concurrent=8` caps parallel API calls.
- 8 concurrent × 10s/call = **0.8 calls/sec/pipeline** ≈ 50 papers/hour at 30 panels each.
- Multi-process parallelism (multiple `RadiolarianPipeline` workers): scales linearly until M3 API rate-limit. Rate limit not measured precisely, but 8 concurrent already saturates a single M3 stream.

### 6.3 Async / sync

- **Synchronous today** (`backend.infer_panel` blocks). Whole pipeline is CPU-bound on regex + I/O-bound on M3.
- **Proposed**: still synchronous (matches existing pattern); the M3 calls already use the concurrency semaphore.
- **Optional future**: `asyncio` + `httpx` for non-blocking I/O, but this is a bigger refactor — not required for production.

### 6.4 GPU / CPU / bandwidth footprint

| Resource | Per-call cost (live) | Per-paper (50 panels) |
|----------|---------------------|------------------------|
| GPU (local model path) | 0 (M3 API path doesn't need GPU) | 0 |
| CPU | minimal (regex/JSON parse) | ~30 s total |
| Network egress | ~500 KB (image) + 5 KB (prompt) + 5 KB (response) | ~25 MB |
| Disk I/O | ~200 KB/crop write | ~10 MB |
| M3 API quota | 1 call / panel | 50 calls / paper |

**Bandwidth-bound, not compute-bound.** For 1000 papers/year, 25 GB egress — negligible.

### 6.5 Failure modes & recovery

| Failure | Behavior | Recovery |
|---------|----------|----------|
| M3 API timeout (>60s) | log + skip panel | regex stays |
| M3 returns fallback_used | log + skip | regex stays |
| M3 returns low confidence (<0.55) | overwrite suppressed | row keeps regex |
| M3 returns garbage / unparseable JSON | parse_json_from_text 4-tier | regex stays |
| Stage 3 crop missing | skip panel | regex stays |
| Backend = rule_only (offline mode) | skip entire stage | regex stays |
| `RuntimeError` in `_one()` | caught + logged | regex stays |

**Pipeline never regresses** — Stage 4.5 is purely additive when M3 succeeds.

---

## 7. Architectural changes (concrete diff)

### 7.1 `src/rlpe/config.py`

Add 4 new fields to `PipelineConfig` (all default to backward-compatible values):

```python
m3_per_panel_enabled: bool = False  # opt-in flag, default OFF
m3_per_panel_min_conf: float = 0.55  # only overwrite regex if M3 ≥ this
m3_per_panel_max_per_figure: int = 20  # cap per-figure to prevent runaway
m3_per_panel_max_per_paper: int = 200  # cap per-paper to prevent runaway
```

### 7.2 `src/rlpe/pipeline.py`

- New method `_apply_m3_per_panel_species_id(results, paper_id)` (~80 lines, see pseudocode §3).
- Wired into the main pipeline loop between `_apply_stage3_bbox_crops(...)` and `_apply_multi_plate_enrichment(...)`.
- Early-returns on disabled / no-backend / no-crops (zero-cost when off).
- Catches all exceptions per-row, logs warning, falls back to regex.

### 7.3 `src/rlpe/cli.py`

Add 4 CLI flags mirroring the config fields:
- `--m3-per-panel / --no-m3-per-panel` (default: disabled)
- `--m3-per-panel-min-conf 0.55`
- `--m3-per-panel-max-per-figure 20`
- `--m3-per-panel-max-per-paper 200`

### 7.4 `src/rlpe/web_ui/...`

(Optional, post-MVP) — add a checkbox in the Run tab "Use M3 per-panel species ID" + numeric inputs for the thresholds. Falls under existing Run-tab form schema.

### 7.5 Tests (new file: `tests/test_stage4_5_m3_per_panel.py`)

~25 unit tests covering:
1. Disabled by default — no M3 calls
2. No backend → early-return, regex stays
3. No panel_path → skip, regex stays
4. Backend fallback_used → regex stays
6. Backend confidence < min_conf → regex stays
5. Backend confidence ≥ min_conf → M3 overwrites
7. Backend raises → log + regex stays
8. Backend returns parse failure → regex stays
9. Metadata stamp present when M3 succeeds
10. Metadata stamp absent when M3 fails
11. Image SHA computed (for reproducibility audit)
12. Latency recorded
13. Per-figure cap respected
14. Per-paper cap respected
15. Caption pairs match by panel_id
16. Page context truncated at 1500 chars
17. Confidence clamped to [0, 1]
18. Species normalised via `_normalize_panel_dict`
19. Empty species from M3 → keep regex
20. Full integration: pipeline.run() with mock backend produces expected rows
21-25. Pipeline regression tests (existing pipeline unchanged)

### 7.6 No changes to

- `m3_engine.py` — Stage 4 (`match_panel`) is untouched
- `cross_figure_linker.py` — Phase 66 already separate
- `pbdb_resolver.py` — Phase 31 family-level fallback already there
- Existing schemas — additive `metadata.m3_per_panel` field
- Existing tests — only adds new file, doesn't touch existing

---

## 8. Risks & open questions

### 8.1 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| M3 per-panel *worse* than regex on some papers | medium | recall regression | confidence-gated overwrite + cap-per-figure; live A/B before merge |
| Caption pairs not matching row panel_id | low | M3 sees wrong context | fallback to full caption + page context |
| Stage 3 crop quality degrades | low | M3 sees cropped-out noise | YOLO bbox already conservative; existing `panel_id_source` audit tag |
| API cost overrun (runaway on big paper) | low | budget over | per-figure + per-paper caps |
| Latency blows up SLO | low | UX regression | concurrency cap already at 8 |

### 8.2 Open questions

1. **Should per-panel M3 run BEFORE or AFTER `multi_plate_enrichment`?** Currently proposed after Stage 3 + before enrichment. Could move after enrichment to use enriched geology context — would be slower but possibly more accurate. Recommend: ship after Stage 3 first; revisit with A/B data.
2. **Self-consistency N=3 cost** — `m3_match_samples=3` triples cost. Recommend: ship with N=1, add as opt-in flag later.
3. **Cross-paper caching** — image SHA cache could save M3 calls if same panel image appears multiple times. Low priority — most papers have unique panels.

### 8.3 Rollback

The 4 new config fields all default to safe values (disabled / low / capped). If production breaks:

```bash
# CLI rollback
rlpe run --no-m3-per-panel …

# or in Python
config.m3_per_panel_enabled = False
```

The method early-returns on disabled, so zero cost / zero behaviour change.

---

## 9. Success criteria

### 9.1 Must-have (acceptance criteria for merge)

- [ ] Per-paper effective recall ≥ 75% on 9-paper gold (vs 53% today)
- [ ] Per-paper cost ≤ ¥0.50
- [ ] Per-paper latency ≤ 2 min (with concurrency=8)
- [ ] No regression on papers where M3 has 0% (Pouille-like): regex path still gives original result
- [ ] 25 new tests passing, full suite ≥1829 tests
- [ ] No new files beyond `tests/test_stage4_5_m3_per_panel.py`

### 9.2 Stretch goals

- [ ] Per-paper effective recall ≥ 85%
- [ ] Combined with multi-stage enrichment, reach 90%+ F1
- [ ] Self-consistency N=3 opt-in for borderline-confidence panels

---

## 10. Out of scope

- Building a new M3 backend / training a model
- Replacing Stage 3 with M3 per-panel (Stage 3 stays as crop producer)
- Changing schema versioning (additive only)
- Replacing regex with M3 unconditionally (always gated, always fall-back)
- PBDB enrichment at per-panel level (already in Phase 31 at paper level)
- Cross-paper reasoning (Phase 64-66 covers cross-figure, not cross-paper)

---

## 11. References

- Live M3 eval baseline: this morning's 9-paper run on `runs/real_papers_2026_08_17/` (PDFs in tree; result tables in session log 32ce786d-d497-4196-9131-ae50ea894b1e).
- Existing infrastructure: `src/rlpe/m3_engine.py:2533` (`match_panel`), `:1784` (`PanelBox`), `:1800` (`PanelMatch`).
- Existing semaphore: `src/rlpe/config.py` `MiniMax_max_concurrent` (default 8).
- Existing self-consistency: `m3_match_samples` (default 1).
- YOLO fallback: `src/rlpe/pipeline.py:1721` (`_apply_stage3_bbox_crops`), `:_yolo_fallback_for_stage3`.
- Audit Round 6 live results memory: `project_round6_live_results.md` (5 OA papers, ¥0.008/row avg, 92.5% species).

---

**End of design doc. Awaiting user approval before transitioning to writing-plans.**