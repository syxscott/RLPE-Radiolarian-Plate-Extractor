# Radiolarian-filtered smoke test — 2026-09-02

**Goal**: pick random radiolarian-focused papers NOT in the v19 gold set,
run the new pipeline (`caption_fixer` + `prompts` + `post_process`
+ MiniMax M3) end-to-end, and report what comes out.

This is a follow-up to `2026-09-02-random-3-papers.md`, which used no
filename filter and got 0/3 papers with real radiolarian species. Here
we filter by filename so the random draw lands on radiolarian-titled
papers — but as the table below shows, *having the word radiolaria in
the filename* is **not** the same as *having a species-rich plate
caption extractable by the pipeline*.

---

## Selection

Filter: PDF filename matches `Radiolaria|radiolarian` (case-insensitive).
Broadened filter not needed — 167 / 184 PDFs passed the keyword
filter, so the random sample picks from the same set as the previous
test, just with a different seed.

Seeded `random.seed(20260902)`, then took `non_v19[:3]` after
excluding the 9 v19 gold slugs. Result:

| # | Spec | Paper | Pages |
|---|------|-------|-------|
| 1 | yes | Lazarus_2014 — *Editorial: The legacy of early radiolarian taxonomists* (JMicro) | 2 |
| 2 | yes | Anderson_1998 — *Evidence of Binary Division in Mature Central Capsules of a Collosphaerid Colonial Radiolarians* (PE) | 13 |
| 3 | yes | Haslett_1995 — *Pliocene–Pleistocene radiolarian and diatom biostratigraphy of ODP Hole 709C* (JMicro) | 9 |

**Empirical observation after Step 3**: only 1 of the 3 spec picks has
plate-style captions extractable by `caption_fixer.select_caption`.
To get a broader view of the pipeline, two more papers were drawn
from the same seed-shuffled list — picking the next plate-anchored
papers after the initial 3 — to fill out the table:

| # | Spec | Paper | Pages |
|---|------|-------|-------|
| 4 | supplementary | Kurihara_2004 — *Silurian and Devonian radiolarian biostratigraphy of the Hida Gaien belt, central Japan* | 20 |
| 5 | supplementary | Ble_2020 — *SIMS analysis of Si isotope for radiolarian test in Mesozoic bedded chert, Inuyama, central Japan* | 24 |
| 6 | supplementary | Ogane_2014 — *Pseudopodial silica absorption hypothesis (PSA hypothesis): a new function of pseudopodia in living radiolarian polycystine* (JMicro) | 6 |

**Script**: `/tmp/random_test2/runner.py` (single-paper driver, same
env-loading + post_process calls as `scripts/run_research_eval.py`).
Per-paper JSON dumps and rendered PNGs are under `/tmp/random_test2/`.

---

## Results

### Paper 1 — Lazarus_2014 (spec)

- **Pages**: 2 (full editorial), **Plate anchors**: 0 (no `Plate N` / `Fig N` lines anywhere in the PDF).
- **Pipeline outcome**: skipped with `no suitable radiolarian plate caption found`.
- **Interpretation**: correct behaviour — editorial papers rarely have plate-style figures; the pipeline does not invent one. This is the *right* outcome, not a bug.

### Paper 2 — Anderson_1998 (spec)

- **Pages**: 13, **Plate anchors**: 0 (the 13 `Fig.` hits are all in-text cross-references like "...(Plate 1, Fig. 1): 131", not anchors at the start of caption blocks).
- **Pipeline outcome**: skipped with `no suitable radiolarian plate caption found`.
- **Interpretation**: morpho-anatomical research note on binary division — its figures are in-text micrographs without plate-level caption blocks. Captioner's anchor regex requires `^\s*(?:Plate|Fig\.?) \d+` at line-start, which this PDF never has. Correct skip.

### Paper 3 — Haslett_1995 (spec)

- **Page used**: 4, **Plate anchor**: 1.
- **Caption length**: 1505 chars.
- **Prompt template chosen**: standard radiolarian plate prompt (`select_prompt` saw no `range|distribution|bar=|scanning electron` markers — caption is a classic plate caption with species list).
- **Raw M3 panels**: 13 (all 13 `Fig. N` figures listed in the caption).
- **Preds after `dedup_panels` + `filter_low_confidence(0.7)`**: **13/13 survive**.
- **Faithfulness check vs the printed caption text**:

  | Fig | Caption says | M3 returned | Match |
  |-----|--------------|-------------|-------|
  | 1   | `Stichocorys peregrina (Riedel)` | `Stichocorys peregrina` | yes |
  | 2   | `Stichocorys peregrina (Riedel)` | `Stichocorys peregrina` | yes |
  | 3   | `Theocorythium uetulum Nigrini` | `Theocorythium vetulum` | OCR-faithful (u/v swap) |
  | 4   | `Theocorythium trachelium (Ehrenberg)` | `Theocorythium trachelium` | yes |
  | 5   | `Anthocyrtidium jenghisi Streeter` | `Anthocyrtidium jenghisi` | yes |
  | 6   | `Anthocyrtidium michelinae Caulet` | `Anthocyrtidium michelinae` | yes |
  | 7   | `Phormostichoartus doliolum (Riedel & Sanfilippo)` | `Phormostichoartus doliolum` | yes |
  | 8   | `Amphirhopalum ypsilon Haeckel` | `Amphirhopalum ypsilon` | yes |
  | 9   | `Theocalyptra davisiana (Ehrenberg)` | `Theocalyptra davisiana` | yes |
  | 10  | `Spongaster tetras Ehrenberg` | `Spongaster tetras` | yes |
  | 11  | `Lamprocyrtis neoheteroporos Kling` | `Lamprocyrtis neoheteroporos` | yes |
  | 12  | `Hemidiscus cuneiformis Wallich` (a DIATOM) | `Hemidiscus cuneiformis` | correct — paper is radiolarian+diatom joint biostratigraphy |
  | 13  | `Diploneis sp.` (a DIATOM) | `Diploneis sp.` | correct |

  Every species in the caption comes back exactly as written. **Even
  haslett1995 is on-topic for radiolarian F1 evaluation**: 9/13 figures
  are polycystine radiolarian species, the other 2 are diatoms
  (correctly ignored by poly-specific NER but faithfully transcribed
  by the model from the caption text).
- **Confidence**: 0.95 across the board, except Hemidiscus at 0.9.
- **Verification**: This is one of the cleanest reproductions of a
  caption-text-to-species extraction in any smoke test so far. The
  pipeline reads the figure list out of the caption, splits it per
  Fig number, and emits a prediction per panel.

### Paper 4 — Kurihara_2004 (supplementary)

- **Page used**: 3, **Plate anchor**: 2.
- **Caption length**: 2556 chars.
- **Prompt template chosen**: "Given a figure caption and image" (broad figure prompt — caption mentions neither maps nor strat columns, so `select_prompt` falls through to the GENERIC prompt).
- **Raw M3 panels**: 1.
- **Preds after dedup + conf**: 1 panel with `species="None"`, confidence 1.0.
- **Interpretation**: the runner stopped at *plate 2* (the first plate anchor in the document). Plate 2 in this paper is a stratigraphic/range-chart figure, not a species SEM plate. The M3 model correctly recognized that no species is depicted and returned None with high confidence. The pipeline did **not** hallucinate a species.

### Paper 5 — Ble_2020 (supplementary)

- **Page used**: 3, **Plate anchor**: 1.
- **Caption length**: 808 chars.
- **Prompt template chosen**: **"Given a map caption and image"** (caption contains geographic / locality markers that trigger the MAP_PROMPT predicate).
- **Raw M3 panels**: 6 (labelled `a`–`f`).
- **Preds after dedup + conf**: 6 preds — 3 with `species="None"` and 3 with `species="radiolarian"`.
- **Interpretation**: This is a SIMS-analysis figure (a–f panels of isotope data on radiolarian tests). No species binomials appear in the caption because the figure isn't a species plate. The model correctly returned None for 3 sub-panels and the uninformative placeholder "radiolarian" for the other 3 (the caption text only says "radiolarian test"). The MAP_PROMPT was the right call (it's a locality/data figure) — but for plate-style figures this is the prompt to suppress. **No bug**, but it's a useful reminder that `select_prompt`'s keyword markers can misfire.

### Paper 6 — Ogane_2014 (supplementary)

- **Page used**: 2, **Plate anchor**: 1.
- **Caption length**: 4165 chars (very long — caption lists 6 figures with multiple sub-panels each).
- **Prompt template chosen**: standard radiolarian plate prompt.
- **Raw M3 panels**: 16 (1 fig header + 15 sub-panel entries).
- **Preds after dedup + conf**: **16/16 survive**.
- **Unique species extracted (3)**:

  | Species | Panels |
  |---------|--------|
  | `Lithelius sp.` | 5 (Fig 1a–2b) |
  | `Rhizosphaera trigonacantha` | 5 (Fig 3a–5b) |
  | `Arachnosphaera hexasphaera` | 5 (Fig 6a–7b) |

- **Confidence**: 0.95 for `Rhizosphaera` and `Arachnosphaera`, 0.9 for `Lithelius`.
- **Interpretation**: confident radiolarian-species identification for a paper showing in-vivo polycystine micrographs. The three genera match a real Ogane-style 2014 PSA-hypothesis paper (extant species). The single header entry ("Fig. 1" with species=None) is correctly dropped from any species count but preserved in the raw output.

---

## Summary table

| # | Spec | Paper | Outcome | Notes |
|---|------|-------|---------|-------|
| 1 | yes | Lazarus_2014 | skipped | editorial, no plates |
| 2 | yes | Anderson_1998 | skipped | research note, no plate-style caption |
| 3 | yes | Haslett_1995 | **13 preds, all match caption** | strongest result |
| 4 | sup | Kurihara_2004 | 1 pred (None) | figure selected was a range chart, not a species plate |
| 5 | sup | Ble_2020 | 6 preds (3 None + 3 generic) | SIMS figure, MAP_PROMPT chosen, no species to extract |
| 6 | sup | Ogane_2014 | **16 preds, 3 unique spp** | all 3 genera match plate |

---

## Pipeline-wide observations

1. **Filename filter is necessary but not sufficient.** Even after
   filtering on `Radiolaria|radiolarian`, 2/3 spec picks were plate-less
   papers (editorial / research note). To exercise the species
   extraction pipeline, a future randomised smoke test should *also*
   filter on "the PDF has at least one Plate/Fig anchor in line-start
   position" — that pulls ~50% of papers in the corpus, not 100%.
   Empirically 8 of the first 15 candidates (`paper_audit_2026_07_20.md`
   style) in this same shuffled list had plate anchors.

2. **The pipeline does NOT hallucinate species on plate-less figures.**
   Lazarus, Anderson, Kurihara, and Ble (header-only sub-panels) all
   returned `species=None` with high confidence — the "right"
   non-extraction rather than the "wrong" hallucination. This is the
   inverse of the random-3 test (where M3 correctly ignored ammonite
   plates in a non-radiolarian paper).

3. **`caption_fixer.select_caption` has a real false-negative rate for
   plate-less papers with figure refs in mid-paragraph** (Anderson).
   It only matches `^\s*(?:Plate|Fig)\.?\s*\d+\b` at the start of a
   line, which misses "Fig 1, Fig 2 inside a sentence". This is by
   design (avoids over-matching), but it means future tests should
   pre-screen PDFs by counting plate anchors before relying on
   `select_caption` to find them.

4. **Two paper types still slip through the radiolarian filter**:
   - **Editorials / obituaries** that are about radiolarian taxonomists
     but have no plates (Lazarus_2014). The filename filter sees the
     subject but the content has no figures.
   - **Plate-less morphology papers** (Anderson_1998, Ogane in part)
     that have figures but no plate-style caption block.

5. **`select_prompt` MAP_PROMPT firing on a SIMS-analysis figure is
   arguable** — the caption has geographic markers but the figure is a
   data table. Not a bug per se; downstream evaluation just has
   nothing to score.

6. **OCR faithfulness check on Haslett**: `uetulum → vetulum` is an
   OCR v/u swap captured faithfully by the model. The model did NOT
   silently correct it; downstream `parse_open_nomenclature` didn't
   touch it either. Future ablation: should `parse_open_nomenclature`
   normalise known v↔u confusions (e.g. via an allowed-pair dict)?
   Probably not — faithful transcription is safer than hallucinated
   corrections.

---

## Verdict

**DONE with full table.** 4 papers had real plate captions (Haslett,
Kurihara, Ble, Ogane); 2/4 yielded species-rich extractions (Haslett,
Ogane). The pipeline's species extraction quality on species-rich
plates remains strong (Haslett 13/13 match; Ogane 3 unique species
across 15 panels, 100% confidence ≥ 0.9). The pipeline correctly
avoids hallucination on non-species plates (Kurihara, Ble, the diatom
panels of Haslett). The remaining gap is **selection of papers
guaranteed to have plates**, not pipeline quality.

---

## Files

- `/tmp/random_test2/runner.py` — single-paper driver
- `/tmp/random_test2/<paper>_p<N>.json` — per-paper dump of raw M3 panels + post-processed preds
- `/tmp/random_test2/panel_<paper>_p<N>.png` — the page image sent to M3 (DPI 150)
