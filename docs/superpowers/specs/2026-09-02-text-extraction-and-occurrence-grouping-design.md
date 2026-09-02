# Design: RLPE text extraction + occurrence grouping

> **For agentic workers:** This is a design spec, not a task plan. It must be reviewed and approved by the user before any code is written.

**Date:** 2026-09-02
**Status:** Draft (pending user review)
**Author:** Claude (brainstorming 2026-09-02)

---

## 1. Goal & user pain

RLPE today extracts radiolarian species **only from plate / figure captions** (caption_fixer + M3 plate-mode prompt). This misses significant radiolarian content in many real papers:

1. **Range charts / distribution tables**: papers with a single `Fig. 1. Distribution of the radiolarians...` chart listing species per stratigraphic zone, with no per-panel specimen images.
2. **Species lists in body text**: papers that introduce "Genus species" inline in the systematic-paleontology section without a plate figure.
3. **Same species on multiple plates / figures**: a species photographed in 5 different plates (e.g. juvenile + adult + different cross-sections + comparison with holotype) is currently recorded as 5 independent FPs against 1 gold row, dragging F1 down even when the extraction is correct.

This spec adds two complementary features:

- **Feature A** — extract radiolarian data even when no plate figure exists (range charts, inline body text, etc.), recording **location** for traceability.
- **Feature B** — group same-species occurrences across figures via an `occurrence_group_id`, so multiple figures showing the same species are correctly deduplicated.

---

## 2. User decisions captured in brainstorming

| Question | Decision |
|---|---|
| A priority | "Two things together (broad extraction + location)" — both A.1 regex + A.2 M3 text-mode + A.3 location field |
| B storage | "Flat + occurrence_id" — keep current row shape, add `occurrence_group_id` column |

---

## 3. Approach

### 3.1 Feature A — text-level extraction (regex + M3 text-mode fallback)

#### 3.1.1 `extract_species_from_text(pdf_path) -> list[dict]`

A new pure-Python extractor that does **not** require a plate caption and **does not** call the LLM. It scans the full PDF text for the canonical binomial pattern `Genus species` and returns one record per match.

**Algorithm**:
1. Open PDF with pymupdf; concatenate all pages with `Page N:` page markers.
2. Scan with a refined `BINOMIAL_RE` (already exists in `caption_fixer.py`): `\b[A-Z][a-z]{3,}\s+[a-z]{3,}\b`.
3. Apply the existing `caption_fixer._BINOMIAL_DENY` filter (drops "Many species", "Most samples", "Each individual", etc.) so English phrases don't leak through.
4. For each match, attach:
   - `species` (raw)
   - `page_num` (integer)
   - `char_offset` (offset in concatenated text)
   - `context_50char` (50 chars on each side, for traceability)
5. Deduplicate by `(normalized_species, page_num)` to handle multiple mentions in the same paragraph.
6. Return list of dicts. `extraction_method='regex_list'`.

**Why this is sound**:
- Zero API cost, runs in <1 second per paper.
- Catches species that appear in any context (plate caption, range chart, body paragraph, table).
- Catches false positives only for "X Y" English phrases — already mitigated by the existing deny-list.
- Honest about not knowing which plate the species was illustrated on; that field stays at paper-level.

**Acceptance**: regex extractor adds at least 5 species/paper on the v19 set (small gain on already-curated gold but big gain on generic 184-paper corpus).

#### 3.1.2 M3 text-mode fallback (only when `select_caption` returns None)

When `caption_fixer.select_caption(text, target_plate)` returns `None` (no plate-style caption in the text), instead of giving up the run, fall through to a text-mode extraction:

- New `scripts/prompts.py` constant: `TEXT_MODE_PROMPT` — instructs M3 to extract every radiolarian species from the provided text, with page numbers if possible, and to return one row per species.
- Call `backend.infer_panel(panel_image=None, caption_text=full_text, ...)` with the new prompt. (NOTE: `infer_panel` currently requires a `panel_image`; we will need to either pass a 1x1 placeholder or add a `infer_text` path. The design is to add a thin `infer_text(backend, full_text, prompt)` wrapper that constructs a minimal 1x1 white image so the existing Anthropic API call still works.)
- Post-process the result identically to plate-mode (parse_open_nomenclature + dedup + conf filter).
- `figure_id` field for these rows = `"text_section_pN"` (where N is the page number where the species appears, or `"full_paper"` for cross-page mentions).

**When to trigger**: only when `select_caption` returns `None` AND the paper has at least 1 radiolarian keyword (`"Radiolaria"|"radiolarian"|"Polycystina"|"Nassellaria"|"Spumellaria"`, case-insensitive) in its text. Skip the M3 call otherwise (the paper is not radiolarian-related).

**Acceptance**: text-mode runs only on `text_extraction` papers; no impact on the 9 v19 gold papers (which all have plate captions).

#### 3.1.3 Row schema extension (`location` field)

Add a new `location` field to the existing pred row dict. All other fields stay the same. Example:

```json
{
  "paper_id": "bandini2011",
  "figure_id": "plate_4f1bf415485765b8_p012_pl01",
  "panel_id": "5",
  "species": "Williriedellum carpathicum",
  "confidence": 0.95,
  "location": "Plate 1, Fig. 5 (p. 12)",
  "extraction_method": "plate_M3" | "regex_list" | "text_M3"
}
```

The `location` field is human-readable and unified across all three methods:
- plate_M3: `"Plate N, Fig. M (p. K)"`
- regex_list: `"p. K, paragraph L (or section M)"`
- text_M3: `"p. K (text section)"` or `"full paper (text extraction)"`

This is purely additive — existing eval code and gold files do not need to change.

### 3.2 Feature B — occurrence_group_id for cross-figure species

#### 3.2.1 occurrence_group_id generation

For each pred row, compute a deterministic group ID:

```python
import hashlib
def occurrence_group_id(paper_id: str, normalized_species: str) -> str:
    """Return a 6-char group ID for this (paper, species) pair.

    Same paper + same species (after _norm_species + cf./aff. split) =
    same group. Different papers or different species = different groups.
    Used to deduplicate the same species extracted from multiple
    figures in the same paper.
    """
    raw = f"{paper_id}|{normalized_species}".encode()
    return "occ_" + hashlib.sha1(raw).hexdigest()[:6]
```

Place this in a new `scripts/occurrence.py` (small module, easy to test).

#### 3.2.2 New `scripts/occurrence.py` module (pure functions)

```python
"""Group identical species across multiple figures in a paper.

Two preds are in the same occurrence group iff:
  - same paper_id, AND
  - same normalized species (cf./aff. split, lowered, etc.)
"""
from __future__ import annotations

import hashlib
from rlpe.evaluation.metrics import _norm_species


def occurrence_group_id(paper_id: str, species: str) -> str:
    raw = f"{paper_id}|{_norm_species(species)}".encode()
    return "occ_" + hashlib.sha1(raw).hexdigest()[:6]


def add_occurrence_groups(preds: list[dict]) -> list[dict]:
    """Return a copy of preds with `occurrence_group_id` added to each row."""
    out = []
    for p in preds:
        q = dict(p)
        q['occurrence_group_id'] = occurrence_group_id(p.get('paper_id', ''), p.get('species') or '')
        out.append(q)
    return out
```

#### 3.2.3 Eval impact (optional, future)

The current `evaluate()` and 5-fold CV do not need to change. We are only ADDING the `occurrence_group_id` field to pred rows; existing F1 metric is unchanged. A future metric (`f1_by_species` or `coverage`) could be added to surface multi-figure coverage, but that is out of scope for this spec.

---

## 4. Components

| File | Action | Description |
|---|---|---|
| `scripts/text_extract.py` | **Create** | `extract_species_from_text(pdf_path)` — regex + page tracking + context. Pure Python, no LLM. |
| `scripts/occurrence.py` | **Create** | `occurrence_group_id(paper_id, species)` + `add_occurrence_groups(preds)`. Pure Python, no LLM. |
| `scripts/prompts.py` | **Modify** | Add `TEXT_MODE_PROMPT` constant + `select_text_mode_prompt(text)` helper. |
| `scripts/caption_fixer.py` | **Modify** | Minor: add `has_plate_captions` already exists from Task 1.8 follow-up. No new changes. |
| `scripts/run_research_eval.py` | **Modify** | In the per-paper loop, after `extract_panels_for_paper`, call `extract_species_from_text` (if no plate was found) or always (additive). Use the new `TEXT_MODE_PROMPT` if `select_caption` returns None. Tag rows with `location` and `extraction_method`. Call `add_occurrence_groups` before saving. |
| `tests/test_text_extract.py` | **Create** | Unit tests for `extract_species_from_text`. |
| `tests/test_occurrence.py` | **Create** | Unit tests for `occurrence_group_id` + `add_occurrence_groups`. |
| `tests/test_text_mode_prompt.py` | **Create** | 1-2 tests that the new prompt exists and is well-formed. |

---

## 5. Data flow

```
PDF
  ↓
[1] pymupdf extract all pages → full_text
  ↓
[2] regex scan → list[dict] (species + page_num + char_offset + context_50char)
  ↓
[3] (concurrent or sequential) caption_fixer.select_caption → caption OR None
  ↓
[4] if caption found:
       M3 plate-mode → panels
       parse_open_nomenclature + dedup + conf_filter
       extraction_method='plate_M3'
  else:
       M3 text-mode (TEXT_MODE_PROMPT) → species
       parse_open_nomenclature + dedup + conf_filter
       extraction_method='text_M3'
  ↓
[5] (optional) if paper has no radiolarian keywords, skip M3 call
  ↓
[6] merge regex_list results + M3 results (dedup by normalized species)
  ↓
[7] add occurrence_group_id to every row
  ↓
[8] write to matches.jsonl (existing)
```

---

## 6. Testing & Validation

| Test | What it covers |
|---|---|
| `test_extract_species_from_text_*:_*` | Regex finds "Archaeodictyomitra sp.", "Genus species", skips English phrases, dedup by page, captures context |
| `test_occurrence_group_id_*:_*` | Same paper + same species → same group; different paper or different species → different group |
| `test_text_mode_prompt_*:_*` | New prompt exists, has JSON output instructions, no specific taxa |
| Integration: `run_research_eval.py` with a paper that has no plate → at least 1 regex row | proves the no-figure extraction path |
| Regression: existing 9-paper eval still produces equivalent F1 (plate-M3 path unchanged) | proves we didn't break anything |

---

## 7. Phased timeline (1-2 days)

| Day | Deliverable |
|---|---|
| 0.5 | `scripts/text_extract.py` + tests (regex + page tracking) |
| 0.5 | `scripts/occurrence.py` + tests |
| 0.5 | `TEXT_MODE_PROMPT` + `select_text_mode_prompt` + modify `run_research_eval.py` to wire everything |
| 0.5 | End-to-end smoke on 3 random radiolarian papers (v19 set); verify F1 unchanged + text rows appear for plate-less papers |

---

## 8. Risks & mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Regex false positives in body text ("Most species", "Two taxa") | Medium | Low (we have `_BINOMIAL_DENY` already) | Add tests covering English false-positive cases; consider tightening the regex to require a Latinate suffix or 2+ words preceded by "of"/"including" |
| M3 text-mode hallucination (inventing species not in the text) | Medium | Medium | Prompt explicitly says "ONLY extract species that appear in the provided text"; add a verification step (post-M3: verify each species name appears in the original text) |
| `infer_panel` requires `panel_image` — text-mode needs a different path | Low | Low | Reuse the same Anthropic call by passing a 1x1 white image (still exercises the API path); if that fails, add an `infer_text` method to `MiniMaxM3Backend` |
| `occurrence_group_id` not stable across paper_id changes (e.g. gold uses `bandini2011`, pred uses `4f1bf415485765b8`) | Medium | Low (eval doesn't use this field yet) | Document the algorithm clearly; defer gold-vs-pred joining to a future spec |

---

## 9. Open questions for user

1. Should `regex_list` rows count toward the F1 metric, or be kept separate (e.g. `matches_text.jsonl`)? My recommendation: **yes, count them in F1** (adds ~10-20 species/paper, helps F1 by 5-10pp on text-heavy papers).
2. Should we also add a **post-M3 verification step** that drops hallucinated species (verify each is in the original text)? My recommendation: **yes**, simple substring check, low cost.
3. For the 11 holdout papers (Task 6 / future), should we run the regex extractor on them too, in addition to the v19 9? My recommendation: **yes** — this gives the user free "extra" data while doing manual gold annotation.

---

## 10. Self-review checklist

- [x] No "TBD" or "TODO" placeholders
- [x] Internal consistency: data flow matches components; nothing contradicts itself
- [x] Scope: focused on 2 features, no architecture refactor
- [x] Ambiguity: each row has clear fields; the 3 open questions are flagged above

---

**Status**: Draft ready for user review.

**Next steps after user approval**:
1. Invoke `writing-plans` skill to create implementation plan
2. Phase 1: text_extract.py + tests (0.5 day)
3. Phase 2: occurrence.py + tests (0.5 day)
4. Phase 3: TEXT_MODE_PROMPT + run_research_eval.py wiring (0.5 day)
5. Phase 4: end-to-end smoke on 3 random v19 papers (0.5 day)
