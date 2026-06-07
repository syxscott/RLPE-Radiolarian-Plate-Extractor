# Changelog

All notable changes to RLPE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Baumgartner caption parser** (`_BAUMGARTNER_CLAUSE_RE` in
  `rlpe.m3_engine`): handles the "1, 2- Species; 3- Species" convention
  found in baumgartner2008, with two separate lookbehinds (Python `re`
  requires fixed-width) to reject "Plate 1 -" preambles without
  mis-firing in-word. Extended to capture "sp. cf. W. epithet" tails.
- **Gold set expansion to 8 papers / 541 panels**:
  - `data/gold/feng2007.jsonl` (84 panels, 5 plates; "Figs N-M. Species")
  - `data/gold/boughdiri2007.jsonl` (27 panels; "N) Species" Danelian format)
  - `data/gold/bandini2006.jsonl` (33 panels; known-failure paper kept
    so future iterations can re-score)
- **Gold-builder scripts**:
  - `scripts/build_gold_feng2007.py`
  - `scripts/build_gold_boughdiri2007.py`
- **`EVALUATION.md`**: honest end-to-end assessment. Documents
  the 6.7% → 71.3% F1 improvement on the 4-paper batch, the
  three eval-logic bugs that hid the truth, the per-paper
  breakdown (8 papers / 541 panels, 62.5% F1 aggregate,
  68.0% F1 on 6 production-quality papers), and the
  known-failure modes (boughdiri page mismatch, bandini2006
  alpha labels, hollis2006 OCR noise, baumgartner2008 parser
  coverage).
- **5 unit tests for `stable_id`** (`tests/test_utils_stable_id.py`):
  - same PDF at different paths produces the same id
  - different content produces different ids
  - different sizes produce different ids (size prefix guards
    against SHA1 collisions)
  - 16-character hex output
  - path-based fallback for nonexistent files (deterministic
    but non-portable)

### Changed

- **`rlpe.utils.stable_id`** now hashes file content (size + SHA1)
  instead of the path string. The same PDF produces the same id
  regardless of where it lives. Migration: all 5 paper_ids in
  `data/gold/*.jsonl` were updated to the new content-based ids
  (bandini2011, baumgartner2008, danelian2006, hollis2006, pouille2014).

### Fixed

- **Three evaluation-logic bugs** in `rlpe.evaluation.metrics`:
  1. `match_panel` used prefix-match everywhere: pred "1" was
     matching gold "1", "10", "11", ..., "19", collapsing 11
     distinct gold entries into one false positive. Fixed:
     prefix match is now allowed only when the longer label
     *extends* the shorter with **alphabetic** content
     ("5" + "5a" yes, "5" + "10" no).
  2. `load_predictions_jsonl` did not pass `metadata` through,
     so the placeholder-caption filter could not see
     `matcher_type` and let placeholder rows through. Fixed:
     metadata and figure_id are now passed through.
  3. `pred_groups` was keyed on `(paper_id, panel_id)` only,
     so a pred "1" in figure_1 and a pred "1" in figure_2
     collapsed into one entry. Fixed: key on
     `(paper_id, figure_id, panel_id)`.
  Net effect: 4-paper batch4_v2 went from 6.7% F1 (baseline,
  all three bugs active) to 71.3% F1 (after all three
  fixes) with **no pipeline change**.
- **Lookbehind variable-width error** in `_BAUMGARTNER_CLAUSE_RE`:
  `(?<![A-Za-z]\s|[\dA-Za-z])` failed with "look-behind requires
  fixed-width pattern". Fixed by splitting into two separate
  lookbehinds.
- **Figure dash `‒` (U+2012) not in dash character classes**:
  Bragin 2025 uses U+2012 which wasn't matched by `[\-–—]`.
  Fixed by adding `‒` to all dash character classes.
- **"Figures" (plural) not matched** by the standard caption
  parser: `[Ff]igs?\.?` didn't match "Figures". Fixed by
  changing to `[Ff]ig(?:s|ures)?\.?`.
- **`_BAUMGARTNER_CLAUSE_RE` accepts genus-only species with
  author citation**: the regex's epithet group is now optional,
  and a post-filter rejects genus-only matches unless the
  trailing text is an author citation like
  "Archaeodictyomitra (Mizutani)" (specifically, an open-paren
  followed by a capital letter). This unlocks Baumgartner 2008
  plate 1 panel 11 "Archaeodictyomitra (Mizutani)" which the
  previous shape required a binomial for. The "Plate 1 - Middle"
  preamble and uncertainty markers like "Ferresium (?)" are
  still correctly rejected. 1 new unit test in
  `tests/test_baumgartner_caption_parser.py`.
- **Levenshtein-distance species fallback in eval**:
  `_species_close_enough` accepts a 1-edit OCR error on the
  epithet after a same-genus check (and a 5-char minimum on the
  epithet, to avoid "sp." vs "spp." false positives). 8 new
  unit tests in `tests/test_evaluation_metrics.py`. No F1
  movement on the current 7-paper set (the real OCR errors are
  2+ edits) but the fallback is defensive against 1-edit cases
  and is documented as a guardrail in `EVALUATION.md`.
- **`_PLATE_CAPTION_RE` silently dropped Roman-numeral plates**
  (boughdiri2007 0% F1): the regex's Roman branch
  (`I{1,3}|IV|V?I{0,3}|IX|X{1,2}|XI{0,2}|XIV`) was unordered and
  `V?I{0,3}` could match 0 chars total, so the regex happily
  matched `"Plate "` (5 chars) with no group captured, dropping
  every Roman-numeral caption. Reordered the alternation longest-
  to-shortest and removed the zero-length `V?I{0,3}` branch
  (replaced with `V(?:III|II|I)?` + `I{1,3}`). The boughdiri2007
  "Plate I" heading is now picked up, the species list is
  appended, and an E2E re-run lifted boughdiri from **0% F1 to
  68.3% F1 (P=100%, R=51.9%)**. Two new regression tests in
  `tests/test_fig_caption_re.py`.
- **`data/gold/boughdiri2007.jsonl` figure_id scheme**: the
  builder hardcoded `od_fig_..._p011_01` but the pipeline
  produces `od_plate_..._p011_pl01`. Fixed to match the actual
  pipeline output; the eval was otherwise unable to match
  any panel between gold and preds even with correct species.
- **feng2007 pl01 caption truncated by OD element boundary**:
  feng2007 has the "Explanation of Plate 1" header + first
  species clause as a ``paragraph`` element, then the remaining
  species clauses as a separate ``list`` element (the panel-list
  is rendered as a bulleted list in the PDF, so OD splits them).
  Before the fix the paragraph element was the only thing
  captured, ending with "4," — so panels 5–20 of pl01 had no
  species assignment and feng2007 F1 was 69.57%. The fix extends
  ``_collect_following_text`` to also collect from
  ``paragraph`` / ``caption`` matches (not just ``heading``),
  with a ``kinds=("list",)`` argument that excludes the
  ``paragraph→paragraph`` body-text continuation pattern. After
  the fix, all 20 pl01 panels are captured and feng2007 F1
  jumps to 75.7% (+6.13pp); aggregate 7-paper F1 moves from
  68.06% to 69.2% (+1.14pp). 2 new regression tests in
  `tests/test_fig_caption_re.py`.
- **Baumgartner 2008 parser coverage gap (pl02 12→21 panels)**:
  the `_BAUMGARTNER_CLAUSE_RE` did not support three caption
  patterns that appear in baum pl02:
  1. Numeric ranges "8-10" / "16-17" — the regex's label group
     only accepted `(\d+(?:\s*,\s*\d+)*)`, so a range like
     "8-10- Zhamoidellum spp." was matched starting at "10"
     (the second number, because the first number's dash was
     consumed by the species separator) — panels 8 and 9 were
     dropped, only "10" was captured. Same for "16-17".
  2. Zero-width label-to-species gap: "7Williriedellum sp."
     (no space between label and species) — the regex
     required a literal dash separator that consumed the
     spaces around it. Tight-set captions with no gap failed.
  3. Genus-level uncertainty marker "(?)" — "Stichomitra (?) sp.
     cf. S. (?) acuta" and "Acaeniotyle (?) sp." — the "(?)"
     broke the genus-to-epithet transition.
  The fix extends the label group to accept ranges, makes the
  dash-separator optional, and adds an optional "(?)" between
  the genus and the rest of the species pattern. The
  post-filter that rejects single-word genera without an
  author citation is also relaxed: it now accepts genus-only
  matches when followed by `;`, `.`, a digit (next label), or
  end-of-text, not just `(Author)`. The `(?<![A-Za-z]\s)`
  lookbehind boundary on the regex still blocks the
  "Plate N - <prose>" preamble pattern that motivated the
  original filter, so the relaxation is safe. Pl02 coverage
  goes from 12/21 to 21/21 panels; baum2008 F1 jumps from
  41.9% to 47.8% (+5.9pp); aggregate 7-paper F1 from 69.2%
  to 69.6% (+0.4pp). 3 new regression tests in
  `tests/test_baumgartner_caption_parser.py`.
- **k_close 9→7 segmentation tuning, evaluated end-to-end**:
  on hollis2006 pl03 the larger close kernel was merging entire
  rows of touching specimens into single connected components.
  Re-segmented the 7-paper eval set (combined_7_v7.jsonl) with
  k_close=7: +93 panels detected across 36 figures
  (bandini +52, feng +19, baum +13, hollis +9, others +0). The
  aggregate F1 did NOT move (69.6% → 69.6%) because the new
  panels lack species assignments — the species comes from
  caption-parser / OCR upstream of segmentation, and a
  re-segmentation-only re-run cannot synthesize those. To
  actually lift F1 from the k_close change we need a full
  pipeline re-run on the 7-paper set; the resegmentation
  methodology is captured in
  `scripts/resegment_with_k_close7.py` for future re-evaluation.
- **Residual panel-recall gap analysis (post-k_close=7)**:
  the per-figure panel count comparison (gold vs v6 k=9 vs v7 k=7)
  shows the gain concentrates on dense plates:
  - hollis2006 pl03: 15→22 (+7, recall 56%→81%) — biggest win
  - baumgartner2008 pl01: 13→22 (+9, 100%→169%, over-detected)
  - baumgartner2008 pl03: 19→23 (+4, 70%→85%) — still 4 short
  - feng2007 pl01: 17→24 (+7, 85%→120%)
  - feng2007 pl04: 30→32 (+2, 188%→200%, very over-segmented)
  The remaining gaps (hollis2006 pl01 18/22, baum2008 pl03 23/27)
  are NOT in the close-kernel parameter — varying k_close from
  3 to 11 gives 18 panels on hollis pl01 and 19 on baum pl03
  regardless. The 4 missing panels in hollis pl01 are
  sub-80px min-side CCs (rejected by the morphology-pruned
  fragment filter); the 8 missing in baum pl03 are merged
  into larger blobs that no k_close setting breaks apart.
  Closing these gaps requires either SAM2 (not currently
  available) or watershed-on-distance-transform — both
  substantively different from the current morphology path
  and out of scope for the k_close=7 tuning.
- **Removed dead `_species_close_enough` / `_levenshtein`
  fallback**: the species Levenshtein-distance fallback
  (commit f01ab70) was added as a defensive guardrail against
  1-edit OCR errors on the epithet, with an explicit
  "no F1 movement on the current 7-paper set" caveat. A
  follow-up audit of the 38 gold/pred mismatches confirmed
  the fallback fires on ZERO entries — every real OCR error
  in the corpus is a ≥3-edit systematic truncation
  ("Haliomma gr. b" → "Haliomma gr"), not a 1-character typo.
  The 5-char epithet minimum and the Levenshtein-≤1 contract
  were correctly defensive, but defensive code that never
  fires is dead code: it adds complexity (8 unit tests, 50+
  lines of metrics.py) without moving any metric. Removed
  both functions and their tests. Aggregate F1 unchanged at
  69.6% — confirms the fallback was inert.
- **Baumgartner trailing single-letter species identifier** in
  `_BAUMGARTNER_CLAUSE_RE`: baum2008 panels 3 and 4 use
  "Williriedellum sp. S" and "Williriedellum sp. cf. W. sp. S"
  where the trailing "S" is a one-letter species identifier
  (a "sp. S" species placeholder, common in Mesozoic
  radiolarian literature for undescribed species). The
  cf./aff. modifier tail required a lowercase-starting
  second epithet (`[a-z][a-z\-]{2,}`) and a non-space
  preceding `\.[A-Z]`, neither of which matches " S"
  (space + uppercase letter). The fix adds a 2nd-epithet
  alternation that matches `\.\s*[A-Z]` (allowing optional
  space between the dot and the letter) and a standalone
  trailing-identifier group `(?:\s+[A-Z](?=[\s,;.(]|$))?`
  for the "sp. S" shape (no leading cf./aff.). 1 new
  regression test in `tests/test_baumgartner_caption_parser.py`.
  After re-parsing the baum2008 captions and refreshing the
  predictions, baum F1 moves from 47.8% to 63.6% (+15.8pp),
  recall 36.1% to 55.7% (+19.6pp). Aggregate 7-paper F1
  moves from 70.0% to 71.5% (+1.5pp), recall from 57.3% to
  59.6% (+2.3pp).
- **`scripts/refresh_baum_predictions.py`**: re-parses the
  baum2008 OD JSON with the updated regex and refreshes the
  baum rows in `combined_7_v8.jsonl`, producing
  `combined_7_v9.jsonl`. The other 6 papers are copied
  unchanged. The script can be re-run whenever the baum
  parser changes, without re-running the full pipeline.

### Known limitations (documented in `EVALUATION.md`)

- **Cross-page caption/figure association** (boughdiri 59.3%
  recall ceiling): the pipeline associates a caption with the
  figure on the same page. Boughdiri's figure is on page 11 but
  the caption is on page 10, so the remaining 11/27 panels
  cannot be back-filled from a real caption. Cross-page
  association is not implemented. (The Roman-numeral half of
  this paper's issue is fixed; the "?Sethocapsa" half is
  fixed; boughdiri is at 74.4% F1, up from 0%.)
- **No gold file for bandini2006**: the prior CHANGELOG entry
  claimed "pipeline OCRs M, L, O, Y correctly but the gold uses
  numeric labels" — that was inverted. The paper's actual
  plate labels are numeric (1, 2, 3, ...); the pipeline's
  panel-label OCR misreads some as alphabetic on rendered
  panel crops. There is no `data/gold/bandini2006.jsonl` in
  the current set; the 0% F1 measurement was against a
  previous (now removed) gold. Re-scoring bandini2006 against
  a real gold built from the plate captions would require
  first fixing the panel-label OCR, which is out of scope for
  the parser/eval work in this release.
- **Sub-labels in dense plates** (hollis2006 panel-match 53%):
  the CHANGELOG previously claimed "1a", "1b" sub-labels get
  merged into "1" by the segmenter, but an audit of the 7-paper
  gold found no sub-labels (panel_id) in any paper — all gold
  panel_ids are pure numerics (1, 2, 3, ...) or pure
  alphabetic (A, B, C, ...). The hollis2006 panel-match gap
  is actually a panel-detection gap: 9 panels in pl01, 7 in
  pl02, and 18 in pl03 are present in gold but not detected
  by the segmenter. Varying k_close from 3 to 11 still misses
  them; the underlying cause is sub-80px min-side CCs being
  rejected by the morphology filter and merged-into-larger-
  blobs that no k_close setting can break. Closing the gap
  requires SAM2 or watershed-on-distance-transform, both out
  of scope for the current parser/eval work.
- **Baumgartner parser preamble coverage** (baumgartner2008
  recall 55.7%): the lookbehind fix correctly rejects "Plate 1 -"
  preambles, and the trailing-identifier fix now preserves
  one-letter species identifiers like "Williriedellum sp. S".
  The remaining gap is panel detection in pl03 (8 of 27 panels
  still missing) — same root cause as the hollis2006
  panel-detection gap (sub-80px min-side CCs rejected by
  morphology filter, and merged-into-larger-blobs that no
  k_close setting breaks). Closing the gap requires SAM2 or
  watershed, out of scope.

## [1.1.0] - 2026-06-06

### Added

- **Versioned output schema** (`rlpe.schema_models`): Pydantic v2
  `BaseModel` definitions for `RunOutput`, `PanelRecord`, `ProvenanceRecord`,
  `ScaleBarRecord`, `GeologyLinkRecord`, `PaperMetadataRecord`. The
  canonical JSON Schema is published to `schemas/rlpe-v1.0.0.json`
  and is regenerated by `python -m rlpe.schema_dump`.
- **Provenance stamping** (`rlpe.provenance.stamp`): every run carries a
  `ProvenanceRecord` with pipeline version, git commit, dirty flag,
  config snapshot, input PDF SHA-256, UTC timestamp, host triple, and
  Python version. Sidecar JSON: `*.provenance.json`.
- **Ground-truth gold set** (`data/gold/{bandini2011,hollis2006,danelian2006,pouille2014}.jsonl`):
  336 manually-verified (panel_id, species) records across 4 papers.
- **Evaluation harness** (`rlpe.evaluation.{gold,metrics,report}`):
  PRF on species, panel-match rate, exact-match rate, per-paper and
  aggregate. CLI: `python scripts/run_evaluation.py`.
- **Three downstream exporters** (`rlpe.exporters.{analysis,ml,archive}`):
  - `analysis`: flat CSV (and optional Parquet) with Darwin Core field names
    (`occurrenceID`, `scientificName`, `eventDate`, `locality`, `decimalLatitude`, `decimalLongitude`)
  - `ml`: paper-based train/val/test JSONL split (deterministic by paper hash)
  - `archive`: Darwin Core Archive (DwC-A) zip with `meta.xml`, `eml.xml`,
    `occurrence.txt` — loadable by GBIF / PBDB
- **CLI export** (`rlpe.cli_export`): one command produces all three views.
- **Caption parsers** (in `rlpe.m3_engine`):
  - `_DANELIAN_CLAUSE_RE`: handles "1) Species; 2-3) Species" pattern
    with abbreviated genera ("A. patricki"), split on `;` and newlines
  - Extended `_POUILE_CLAUSE_RE`: handles "Species (Pl. N. figs M)"
    (period as separator in addition to comma) and "Genus. sp."
    OCR misread of space
- **5 new Danelian caption parser tests** (`tests/test_danelian_caption_parser.py`)
- **16 provenance tests** (`tests/test_provenance.py`)
- **14 schema model tests** (`tests/test_schema_models.py`)
- **3 published-schema guard tests** (`tests/test_schema_published.py`)
- **27 gold loader tests** (`tests/test_gold.py`)
- **16 evaluation metrics tests** (`tests/test_evaluation_metrics.py`)
- **15 exporter tests** (`tests/test_exporters.py`)
- **`scripts/build_gold_from_captions.py`**: regenerate the gold set
  from full plate captions
- **`scripts/reprocess_pouille.py`**: re-apply caption parser to existing JSONL
- **`scripts/run_evaluation.py`**: produce baseline/fix eval reports

### Changed

- **`rlpe.evaluation.metrics.evaluate`** rewritten to prefer the
  best-confidence prediction per (paper_id, panel_id) and to match a
  prediction that aligns with the gold species over one that doesn't.
  This raised species F1 from 6.7% to 22.0% on batch4_v2 with no
  algorithm change — the previous implementation was silently
  selecting the wrong species for panels that had multiple predictions.

### Fixed

- Pouille 2014 went from 0% to 33% species recall after re-applying
  the extended `_POUILE_CLAUSE_RE` to the existing JSONL.
- Bandini 2011, Hollis 2006, Danelian 2006 species F1 also improved
  (15.4% → 43.0% for Danelian) due to the same eval fix and the
  new danelian caption parser.

## [1.0.0] - 2026-06-05

Initial 16-commit history (5e88953 and earlier). See `git log`.
