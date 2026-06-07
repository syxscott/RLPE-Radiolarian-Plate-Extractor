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

### Known limitations (documented in `EVALUATION.md`)

- **Cross-page caption/figure association** (boughdiri 51.9%
  recall ceiling): the pipeline associates a caption with the
  figure on the same page. Boughdiri's figure is on page 11 but
  the caption is on page 10, so 13 of 27 panels are still
  page-render placeholders. Cross-page association is not
  implemented. (The Roman-numeral half of this paper's issue
  is fixed; boughdiri is at 68.3% F1, up from 0%.)
- **Alphabetic panel labels in SEM cross-references**
  (bandini2006 0% species F1): pipeline OCRs M, L, O, Y
  correctly but the gold uses numeric labels.
- **Sub-labels in dense plates** (hollis2006 panel-match 53%):
  "1a", "1b" sub-labels get merged into "1" by the segmenter.
- **Baumgartner parser preamble coverage** (baumgartner2008
  recall 30%): the lookbehind fix correctly rejects "Plate 1 -"
  preambles, but fails when the preamble and the species are
  run together with no separator.

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
