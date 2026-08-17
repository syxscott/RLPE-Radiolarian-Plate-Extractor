# Stage 4.5 live smoke test on Bandini 2011

Date: 2026-08-17
Commit (smoke + fix): <filled-after-commit>
Pre-fix smoke commit: `80361aa` (gate-wiring fix, 0/67 high-conf overwrites)
Caption-plumbing follow-on: <see bottom of this file>

## Per-panel run (`--m3-per-panel`), post-caption-plumbing-fix

- Total rows: 78
- m3_per_panel stamped: **57** (73% of rows reached Stage 4.5)
- Rows with overwriting species (M3 confidence >= 0.55): **41**
- High-conf overwrite rate: **41/57 = 72% of stamped rows**
- Avg M3 latency: 4.14s, max 15.44s
- Confidence distribution (stamped rows): min 0.000, mean **0.698**, max 0.990
  (vs pre-fix mean 0.14, max 0.30 — massive shift upward)
- Pipeline exit code: 0
- Sample `metadata.m3_per_panel` payload (panel 1 of pl01):
  ```json
  {
    "species": "Caneta ? sp.",
    "label": "1",
    "confidence": 0.85,
    "reasoning": "Caption directly states 'Caneta ? sp.' for this panel, and the visible numeric label '1' matches. The conical, multi-segmented, highly porous skeleton is consistent with a nassellarian radiolarian.",
    "alternative": null,
    "latency_sec": 4.5,
    "fallback_used": false,
    "image_sha": "..."
  }
  ```

## Baseline run (`--no-m3-per-panel`, pre-fix smoke commit)

- Total rows: 67
- m3_per_panel stamped: 0 (expected)
- Pipeline exit code: 0

## Per-panel run, pre-caption-plumbing-fix (commit `80361aa`)

- Total rows: 86
- m3_per_panel stamped: 67 (78% of rows reached Stage 4.5)
- Rows with overwriting species: **0 / 67** (0%)
- Confidence distribution: min 0.00, mean 0.14, max 0.30
- M3 reasoning text: "No candidate label–species pairs were supplied
  from the caption, and no same-page context is available, so no
  species assignment can be made from the provided metadata."

## Pre-fix bug (commits `654c0fc` and `80361aa`)

Two distinct defects had to be fixed before Stage 4.5 could reach the
target overwrite rate:

### `654c0fc` (Task 8 smoke): flag-wiring defect

The CLI set `m3_per_panel_enabled` as a typed `PipelineConfig`
attribute but the gate in `pipeline.py` read it from
`self.config.extra.get(...)`, which the CLI never populated. The
same defect affected the `m3_stage3` / `m3_multi_plate_enrich` gates.

### `80361aa` (gate-fix follow-on): gate read typed attrs

Gate now reads `self.config.m3_per_panel_enabled` directly. Live
smoke after the fix: 67/86 rows reached Stage 4.5 (vs 0/64 before).
BUT: every row's reasoning was the same "no candidate pairs
supplied" string, and all 67 confidence scores were < 0.30. The gate
was firing but the data flow was empty.

### This commit: caption_pairs + page_context plumbing

The Stage 4.5 worker reads two context keys directly off the row
dict: `caption_pairs` and `page_context_snippet`. Pre-fix:

- `caption_pairs` was consumed upstream by `match_panels()` (which
  used it to build a label→species lookup) but never propagated
  back onto the row. The worker saw an empty list and reported
  "no candidate label–species pairs were supplied".
- `page_context_snippet` was never set on any row. The worker saw
  an empty string and reported "no same-page context is available".

Together those two absences gave M3 zero anchored context per panel,
so it returned uniformly low confidence and the
`m3_per_panel_min_conf` gate (0.55) rejected every overwrite.

The fix plumbs both keys onto every row at figure-build time via
the new `_attach_stage4_5_context` helper, called from the two
`_process_region` return paths (LLM-first and classical).

## Post-fix patch (this commit)

`src/rlpe/pipeline.py` — three changes:

1. **New helper** `_attach_stage4_5_context(rows, *, caption_pairs,
   grobid_sections, figure_page_index)` at ~L2364. Serialises the
   figure-level `CaptionPair` list to plain dicts and stamps the
   same `caption_pairs` value on every row (Stage 4.5 filters by
   panel_id internally). Builds a per-page body-text lookup from
   `grobid_sections` and slices ~1500 chars from the page nearest
   the figure's `page_index` (±2 page window fallback).

2. **LLM-first path** at ~L4849: calls the helper with
   `caption_pairs=llm_caption_pairs or regex_pairs` and the local
   `grobid_sections` / `best_page_index`.

3. **Classical path** at ~L5309: calls the helper with the
   `m3_caption_pairs` produced by the M3 Stage 1 caption parser
   (falling back to whatever the classical matcher built) and
   the same `grobid_sections` / `best_page_index`.

Constraint respected: NO changes to `m3_engine.py`,
`cross_figure_linker.py`, `geology_extraction.py`, or
`llm_backends.py`. Helper lives entirely inside `pipeline.py` and
operates on the dict shape the row already uses.

`tests/test_stage4_5_m3_per_panel.py` — five new regression tests:

- `test_caption_pairs_plumbed_into_m3_prompt` — when a row carries
  a populated `caption_pairs`, the M3 prompt's `[This panel]` block
  contains the species string from the pair whose labels include
  the row's panel_id.
- `test_page_context_snippet_plumbed_into_m3_prompt` — when a row
  carries a populated `page_context_snippet`, the M3 prompt's
  `[Same-page context]` block contains that text.
- `test_empty_context_still_calls_m3` — guards the no-context path:
  empty `caption_pairs` + empty `page_context_snippet` STILL
  produces an M3 call (no silent skip). Existing behaviour.
- `test_attach_stage4_5_context_attaches_caption_pairs_and_page_text` —
  end-to-end on the helper: every row gets the figure-level pairs
  list and the same-page text slice.
- `test_attach_stage4_5_context_handles_noop_inputs` — helper is a
  safe no-op on empty inputs (no rows / no pairs / no body text).

All 33 tests in `test_stage4_5_m3_per_panel.py` pass (28 pre-existing
+ 5 new).

## Confirmations from the live re-run

1. **High-conf overwrite rate jumped from 0/67 to 41/57** — a 72%
   rate among stamped rows. The model now returns
   species-with-citation reasoning like "Caption directly states
   'Caneta ? sp.' for this panel" instead of the "no candidate
   pairs" dead-end string.
2. **Confidence distribution shifted from 0.0-0.30 (mean 0.14) to
   0.0-0.99 (mean 0.70)** — the upper tail of high-confidence
   answers now comfortably clears the 0.55 overwrite gate.
3. **`caption_pairs` is propagated onto 78/78 rows.** The Stage 4.5
   worker filters by panel_id and the model's reasoning now matches
   the caption (e.g. "Caption directly states 'Caneta ? sp.' for
   this panel" — proves the species string flowed through).
4. **`page_context_snippet` is 0/78 in this run.** This is a
   REMAINING CONCERN (see below) — the OpenDataLoader
   `_extract_fulltext_sections` helper does not stamp `page_index`
   on the section dicts it returns, so `_attach_stage4_5_context`
   has no per-page body text to slice. The 41/57 high-conf rate
   was achieved on caption_pairs alone, so this is not blocking,
   but a future enhancement to `_extract_fulltext_sections` to
   track the page index would unlock even higher confidence by
   giving M3 the systematic-paleontology description.

## Remaining concerns

1. **`page_context_snippet` plumbing is dormant for this run**
   because `_extract_fulltext_sections` (in `opendataloader_extractor.py`,
   not protected) doesn't stamp `page_index` on its section dicts.
   The helper falls back gracefully (empty `page_context_snippet`),
   and `caption_pairs` alone was sufficient to drive the 41/57
   high-conf rate, so this is not a blocker. A follow-up commit
   that adds page-index tracking to `_extract_fulltext_sections`
   would unlock the full anchored-context signal.
2. **Stamped-row coverage is 57/78 (73%)** rather than the
   pre-fix 67/86 (78%) because some rows from the multi-plate
   enrichment stub (panel_path=None) and a few zero-confidence
   rows from the second pass don't trigger Stage 4.5. This is by
   design: Stage 4.5 requires a `panel_path` (a crop file) to
   feed into the vision model.

## Artifacts (gitignored, NOT committed)

- `work/output/manifests/matches.jsonl` — per-panel run output (78 rows)
- `work/output/figures/`, `work/output/panels/`, `work/output/tei/`,
  `work/output/od_output/` — per-figure crops + OpenDataLoader output
- `work/output/manifests/run_output.json` — full run record
- `work/output/manifests/llm_usage.json` — LLM cost ledger
- `work/bandini_only/Bandini_2011.pdf` — single-PDF staging dir
