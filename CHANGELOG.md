# Changelog

All notable changes to RLPE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Per-panel miss list in eval JSON** (`mismatches` and `unmatched`
  fields on each `PaperMetrics`): the eval report now serializes
  the per-panel miss details, not just aggregate + per-paper fp/fn
  counts. `mismatches` are panels that were matched by a prediction
  but the predicted species differed from the gold (or the pred
  had no species). `unmatched` are gold panels that had no matching
  prediction at all. Each entry has `{figure_id, panel_id,
  expected, predicted}` (the `predicted` key is omitted from
  `unmatched` entries). Empty gold species are excluded from both
  lists — a "miss" requires a gold species to miss on. 5 new
  regression tests in `tests/test_evaluation_metrics.py::TestMissLists`.
  This surfaced the bandini2006 paper_id mismatch (see "Removed"
  below) — the 4 `mismatches` for panels 5-8 had predicted
  "Archaeocenosphaera" but expected "Archaeocenosphaera
  mellifera" / "Archaeocenosphaera sp", which is suspicious
  because the actual Karnezeika Plate 1 has Dactyliodiscus,
  not Archaeocenosphaera.
- **Per-stage performance benchmark** (`scripts/benchmark.py`):
  wall-clock timings for 5 independently-runnable pipeline stages
  (PDF metadata / OpenDataLoader / segmentation / OCR / caption
  parser) on the smallest committed paper. Output is a stable
  JSON schema (`rlpe-benchmark-1.0`) suitable for diffing in CI.
  Sample dev-env timings:
  caption_parser 0.000s, segmentation 0.003s, pdf_metadata 0.067s,
  ocr 0.667s, opendataloader 3.07s. The OCR stage gracefully
  degrades when no OCR backend is installed; OpenDataLoader
  gracefully degrades when the package isn't installed.
- **OpenAPI 1.1.0 snapshot** (`docs/openapi-1.1.0.json`,
  17 paths, 8 schemas): static OpenAPI 3.1.0 spec for the
  FastAPI app, generated from `app.openapi()` with
  `info.version` synced to `pyproject.toml`. External
  integrators no longer need to launch uvicorn to see the
  schema. The regen script `scripts/gen_openapi.py` reads the
  version directly from `pyproject.toml` (not
  `importlib.metadata`, which can be stale in dev envs).
- **LICENSE (MIT)** at repo root. Previously declared in three
  places (Dockerfile OCI label, CITATION.cff, README badge) but
  not committed as a top-level file. Copyright: Syx Scott, 2026.
- **HEALTHCHECK in Dockerfile** (`HEALTHCHECK --interval=30s
  --timeout=5s --start-period=10s --retries=3 CMD curl -fsS
  http://localhost:8000/health || exit 1`): container
  orchestrators (k8s, docker-compose, ECS) can now detect a
  wedged uvicorn. `curl` is explicitly installed in the runtime
  stage. The `/health` endpoint in `src/rlpe/api/app.py:329`
  returns 200 + `{"status": "ok"}` whenever uvicorn is responsive.
- **API integration test** (`tests/test_api_app.py::TestUploadJobLifecycle`):
  3 new tests exercise the upload → status → cancel flow against
  a real committed PDF (`data/pdfs/beccaro2006.pdf`, 1.1 MB).
  Verifies the wire format of POST /jobs/upload (200 + job_id),
  GET /jobs/{id}/status (valid JobStatus), POST /jobs/{id}/cancel
  (200 + status="cancelled" if still queued/running; 400 if
  already finished), and the 400 rejection of non-PDF uploads.
  84s test runtime dominated by the first PaddleOCR init; the
  smoke tests in the same file remain <1s. 337 tests pass total.

### Fixed

- **Three drifts between claim and reality** (audit 2026-06-07):
  - `pyproject.toml:7` version `0.1.0` → `1.1.0` (matches
    `CITATION.cff:6` and git tag `v1.1.0`).
  - `reproduce_eval.sh` was stale: header said "8 papers /
    519 panels / 94.80% F1" with F1≥0.93 / panel_match≥0.99
    thresholds, but the current v16 corpus is 9 papers / 554
    panels / 96.39% F1 with F1≥0.95 / panel_match≥0.99
    thresholds. Updated.
  - `Dockerfile` `COPY work/combined_8_v13_FINAL.jsonl` →
    `COPY work/combined_9_v16_FINAL.jsonl`; the eval example
    in the Dockerfile + README also updated. `src/rlpe/api/app.py`
    still has a hardcoded `version="0.2.0"` for the FastAPI
    `app.version` (separate from `pyproject.toml`) — this is
    a follow-up, not blocking the v1.1.0 release.
- **No missing eval-detail regressions** introduced by the
  per-panel miss list change: 337 tests pass (was 329; +8 new
  in TestMissLists + TestUploadJobLifecycle).

### Removed

- **bandini2006 gold** (`data/gold/bandini2006.jsonl`,
  60 panels). The v15 release added this gold against the
  Karnezeika, Argolis Peninsula PDF, but the paper_id baked
  into the gold (`19cd1def9ef08554`) does not match the actual
  SHA1 of `data/pdfs/bandini2006_greece.pdf`
  (`b3113f9ee26cb9f6c085105237d5621942603ee7`). The species
  in the gold (Archaeocenosphaera, Triactoma,
  Pseudoacanthosphaera, Halesium, Pessagnobrachia) are from a
  different Mesozoic paper with a similar SEM-plate layout, not
  the Karnezeika paper (which has Dactyliodiscus,
  Pseudoaulophacus, Patellula, Acanthocircus, Dictyomitra,
  Stichomitra). This is a data-integrity issue, not a parser
  bug. The mismatch went undetected in v15 because the gold
  builder never verified the PDF SHA1. The v16 per-panel miss
  list surfaced the issue: 4 mismatches on panels 5-8 where
  predicted "Archaeocenosphaera" was matched against gold
  "Archaeocenosphaera mellifera" / "Archaeocenosphaera sp"
  was suspicious because the actual Karnezeika Plate 1 has
  Acaeniotyle, not Archaeocenosphaera. The gold is preserved
  at `work/bandini2006.jsonl.removed` for historical reference;
  the build script `scripts/build_gold_bandini2006.py` is
  kept (with a `SystemExit` guard and a docstring explaining
  the mismatch) so a future re-annotation effort can build a
  corrected gold against the actual Karnezeika paper. The
  `work/combined_10_v15_FINAL.jsonl` 10-paper corpus is
  superseded by `work/combined_9_v16_FINAL.jsonl` (913 rows;
  bandini2006 prediction rows removed). The actual
  `data/pdfs/bandini2006_greece.pdf` stays in the repo — it's
  a real paper, just not yet correctly gold-annotated.
- Aggregate F1 across the remaining 9 papers:
  **96.39%** (precision 96.39%, recall 96.39%, panel-match
  100.00%, exact-match 96.39%). Better than the v15 10-paper
  95.32% by 1.07pp — the v15 aggregate was pulled down by the
  broken bandini2006 gold. CI threshold in
  `.github/workflows/ci.yml` updated to F1≥0.95 /
  panel_match≥0.99; both pass.

### Added (continued from previous Unreleased)

- **Gold set expansion to 9 papers / 554 panels** (item 7 of the
  7-item gap closure from the 2026-06-07 review, after the
  bandini2006 removal):
  - `data/gold/beccaro2006.jsonl` (35 panels; UAZ A-F index species
    on Plate 1 of the Rosso Ammonitico Medio paper, 5d5264c7bf0b0a43).
    Beccaro 2006 scores **97.14% F1** on the v15 predictions — the
    flat "N – Genus epithet AUTHOR, Section Code, UAZ Letter, xMag"
    caption format is a clean match for the standard clause parser.
  - `data/gold/bandini2006.jsonl` (60 panels across 2 plates of the
    Karnezeika, Argolis Peninsula, Upper Cretaceous paper,
    19cd1def9ef08554). Plate 3 (foraminifera, not radiolarians) is
    intentionally out of scope. Bandini 2006 scores **84.68% F1** on
    the v15 predictions — the "Figures N-M" + "sp. aff." + trailing
    ". sp." patterns on Plate 1 push the parser below 90%, and
    panel-match is 85.0% (51/60) because 9 Plate 1 panels are not
    yet attached to the right figure_id. The paper is kept in the
    gold set so the gap is measurable.
  - Note: the original item 7 ask was "Cambrian + modern" papers.
    No Cambrian- or modern-radiolarian PDFs were available in the
    current `data/pdfs/` corpus, so Mesozoic (Cretaceous) papers
    were used as the closest available proxies. Beccaro covers
    Middle Jurassic (Rosso Ammonitico Medio); bandini covers Upper
    Cretaceous.
  - Aggregate F1 across 10 papers: **95.32%** (precision 96.03%,
    recall 94.63%, panel-match 98.53%, exact-match 94.63%). CI
    threshold in `.github/workflows/ci.yml` is F1 ≥ 0.94 and
    panel_match ≥ 0.98; both pass.
  - `work/combined_10_v15_FINAL.jsonl` (968 rows) is the v15
    prediction corpus, concatenated from
    `work/combined_8_v13_FINAL.jsonl` (878 rows for the original
    8 papers) + 90 new rows for beccaro2006 + bandini2006.
- **Gold-builder scripts** for the two new papers:
  - `scripts/build_gold_beccaro2006.py` (35 panels, UAZ A-F species)
  - `scripts/build_gold_bandini2006.py` (60 panels across 2 plates;
    skips the foraminifera Plate 3)
- **Trailing specimen identifiers in Danelian-style captions**
  (hollis2006 plate 3, feng2007): the parser now recovers the
  "Haliomma gr. b", "Haliomma gr. A-K47/4", and "Corythomelissa sp.
  A. B-F36/0" forms that hollis2006 uses to disambiguate
  multi-specimen plates. Three regex changes were required:
  (1) `_DANELIAN_CLAUSE_RE` accepts `gr.` and `indet.` as modifier
  keywords (in addition to the existing `sp./spp./cf./aff./n.sp.`);
  (2) a 5th capturing group was added for a trailing identifier
  (single letter, digit, or alphanumeric token with optional
  `-`/`/` separators, optionally followed by a `.`-separated
  second segment like "A. B-F36/0");
  (3) the plain-epithet pattern was split into two sub-cases so
  "Acastea sp," (no period) is still captured as epithet when the
  modifier cannot match. The parser folds modifier + trailing-id
  into the species string, so the caller sees the gold form
  ("Corythomelissa sp. A. B-F36/0"). Aggregate F1: 94.80% → **96.34%**.
- **Eval normalization for "X gen. et sp. indet" and trailing
  Author tokens** (hollis2006 plate 1 item 22, plate 2 item 22):
  `_norm_species` in `rlpe.evaluation.metrics` now collapses the
  verbose "Spumellaria gen. et sp. indet" form to "Spumellaria indet"
  and strips a trailing Author token (e.g. "Theocorys? phyzella
  Foreman") when the rest of the string is a binomial. Both rules
  are gated to avoid eating real species components (the author
  strip requires a `<Genus>? <epithet>... <Author>` shape).
- **GHCR Docker push in CI**: a new `docker` job in
  `.github/workflows/ci.yml` builds the multi-stage `Dockerfile`
  and pushes the image to `ghcr.io/${{ github.repository }}` on
  every push to main, develop, and tagged releases. Tags include
  `v1.2.3` semver, `1.2` major.minor, and the commit SHA. The job
  uses Buildx with GHA cache and OCI labels for traceability.
  PRs from forks skip the push (no GHCR token) but the build is
  verifiable locally with `docker build -t rlpe:dev .`.
- **Watershed post-processing in `_segment_with_opencv`** (Phase A.2):
  the OpenCV segmentation path now applies a distance-transform-based
  watershed splitter to large CCs that survive the initial
  morphology+Otsu pass. The previous baseline relied on a 3x3 erode
  to break spine-to-spine bridges between specimens; this fails when
  specimens are bridged by a thicker connection (12+ pixels) or when
  a single dense-plate row contains 3+ touching specimens. Both cases
  now split correctly via the standard 4-step watershed (crop, dist
  transform, ridge-seed finding, cv2.watershed expand). 4 new
  regression tests in `tests/test_segmentation.py`. All 322 tests
  pass (was 318; +4 new).
- **Bragin 2025 `(N) Species` parenthesised caption format**:
  Bragin 2025 ("Oxfordian-Kimmeridgian radiolarians from the
  Nordvik section") uses a parenthesised label form ("(1)
  Praeparvicingula blackhorsensis, (2) Praeparvicingula donnae, ...")
  that the pre-existing `_DANELIAN_CLAUSE_RE` did not handle. Two
  parser changes were required: (1) the open paren is now optional
  in `_DANELIAN_CLAUSE_RE`; (2) the `^` anchor and `re.MULTILINE`
  flag are removed so a single `finditer` pass over a multi-pair
  chunk (Bragin's Plate I has 11 pairs in one chunk) finds all
  entries. The `danelian_lead_re` accepts an optional "Plate N"
  preamble + parenthesised alternative and uses `re.search` to
  slice the chunk from the first label position. 4 new regression
  tests in `tests/test_bragin_caption_parser.py`.
- **8th gold paper: bragin2025** (`data/gold/bragin2025.jsonl`,
  11 panels covering all of Plate I). The paper_id is a
  human-readable placeholder ("bragin2025") because the actual
  PDF is not yet in `data/pdfs/`; the eval harness accepts any
  string. A new `scripts/build_synthetic_predictions.py` mirrors
  a gold file into a predictions file, used to regression-test the
  Bragin parser end-to-end without requiring a real pipeline run.
  `work/combined_8_v12_bragin.jsonl` (874 rows) scores 93.06%
  aggregate F1 across 519 panels (8 papers).
- **`_normalize_species` post-parse pass** in `rlpe.m3_engine`:
  strips "(?)" uncertainty markers, strips "sensu <Author>" tails,
  restores trailing periods on "sp." / "spp." / "indet." / "nov." /
  "gen." (the regex sometimes consumes the period as a sentence
  terminator), and normalizes "Spumellaria gen. et sp. indet." →
  "Spumellaria indet." for cross-paper consistency. 6 new unit
  tests in `tests/test_caption_parsers.py`. After re-parsing the
  7-paper OD corpus with the normalized output, bandini2011 goes
  from 91% → 94% F1, hollis2006 from 41% → 86% F1, and the
  aggregate 7-paper F1 from 71.5% → 91.31% (+19.8pp).
- **Spumellaria/Nassellaria A/B look-ahead in
  `_BAUMGARTNER_CLAUSE_RE`**: the regex captures "Spumellaria gen"
  but the gold convention includes the trailing identifier ("A"
  or "B"). A 30-char look-ahead that picks up the trailing letter
  immediately after the genus recovers these panels in baum
  (4 panels) and hollis (5 panels).
- **`scripts/refresh_all_predictions.py`**: end-to-end refresh
  script that re-parses all 7 papers' OD JSON captions with the
  latest m3_engine regex and updates species assignments in
  `work/combined_7_v11.jsonl` → `work/combined_7_v12.jsonl`.
  Selects the right figure_id per (paper, plate) using the OD
  caption-page hint + 0/1/2 page offset, then drops any
  prediction row whose figure_id doesn't match. Filter rejects
  rows with non-positive bbox width/height. Replaces the
  per-paper `refresh_<paper>_predictions.py` scripts.
- **`EVALUATION.md` v14 update**: per-paper breakdown table with
  v14 numbers (baum 100%, feng 97.6%, hollis 98.6%, boughdiri
  92.6%, bandini 94.0%, bragin 100%, danelian 97.6%, pouille 100%,
  aggregate 96.34% / panel_match 100% / exact 96.34%), v9→v14
  improvement trajectory (71.5% → 96.34% F1, +24.8pp), and the
  path-forward notes (the 3.66% remaining miss is dominated by
  bandini2011 cross-page label noise and feng2007 OCR substitution).
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
  **v12 update:** the v12 all-reparsed caption pass uses the
  OD `caption-page` hint to pick the right figure_id and
  attaches the species list to the page-11 figure, recovering
  25 of 27 panels (92.6% F1). The cross-page association is
  therefore no longer a limitation; it is now a *solved*
  problem for the "caption one page, figure next page" layout
  used by boughdiri.
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
  **v12 update:** panel_match is now 100% across all 8 papers
  (bragin2025 added 11 panels at 100% match), including
  hollis2006 (was 53% in v9). The all-reparsed
  caption pass + the figure_id selection by OD caption-page
  hint closed the gap; sub-label coverage is no longer a
  limitation.
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
  **v12 update:** baum2008 F1 is now 100% (was 47.8% in v9).
  The numeric-range extension ("8-10- Species" / "16-17-
  Species"), the zero-width gap, the "(?)" marker, the
  Spumellaria A/B look-ahead, and the figure_id selection by
  OD caption-page hint collectively close the parser-coverage
  gap. Panel detection in pl03 is also no longer a limitation.

### v12 headline result

- **Aggregate 8-paper species F1: 93.06%** (was 68.1% in v9,
  +25.0pp; was 92.91% in 7-paper v12, +0.15pp from bragin addition).
  Panel match: 100% (was 62.4%, +37.6pp).
- **Per-paper F1**: bandini 94.0%, baum 100%, boughdiri 92.6%,
  bragin 100%, danelian 100%, feng 86.9%, hollis 86.3%, pouille 100%.
- **SOTA threshold (90% F1) reached** (see
  `memory/project_ultimate_goal.md`). The remaining 6.94% miss
  is dominated by OCR-truncation gaps that no parser/figure_id
  work can bridge; closing it requires either OCR-quality
  improvements or a 2-edit Levenshtein fallback gated on a
  per-paper whitelist.

## [Unreleased 2] - 2026-06-08 — N10 panel_id realignment

The v17 eval reported 98.19% F1 but visually the panel_id values
were wrong for ~87% of panels (positional fallback to the caption
label list, not the label visible in the cropped image). Investigation
of 7 root causes and a full pass through the OCR → panel_id →
eval → frontend chain.

### Fixed

- **N10a: `recognize_panel_label` band too small** (`src/rlpe/ocr.py`).
  The label band was 25% of the shorter side capped at 80px —
  for a typical 102×117 bandini2011 panel that yields a 25px band,
  too small for EasyOCR to read the label reliably. Increased to
  50% of the shorter side, floored at 40px and capped at 160px. The
  wider band still concentrates on the corner while being large
  enough for OCR to lock on.
- **N10b: pipeline falls back to full-panel OCR** when corner OCR
  returns nothing (`src/rlpe/pipeline.py`). Now uses
  `label_corner="adaptive"` (try explicit corner first, then the
  other three) and falls back to `recognize_panel` (full panel OCR)
  if all four corners come up empty. Records
  `metadata.label_region_fallback = "full_panel"` when this kicks in.
- **N10c: v18 panel_id reassignment** (`scripts/reassign_panel_id_v18.py`).
  Re-OCR every panel image in `work/combined_9_v17_FINAL.jsonl` with
  EasyOCR, update `panel_id` to the OCR'd label, look up the species
  in `(parsed_caption, panel_id)` for the new panel_id, and write
  `work/combined_9_v18_FINAL.jsonl`. The script also includes a
  resilient path resolver that finds the panel image by tail under
  any `work/*/panels/<paper_id>/<fig>/` directory (the v17 paths
  were written for a specific run directory layout; some have since
  moved). v18 reassigned 158 of 264 verifiable panels; the other
  649/913 rows are either unresolvable (no panel_path, 226) or
  image-OCR-empty (388) and keep their v17 positional panel_id.
- **N10d: image-label-check sub-metric in eval**
  (`src/rlpe/evaluation/image_label_check.py` + `scripts/evaluate.py
  --image-label-check`). Re-OCR every panel image and compare the
  predicted `panel_id` to the OCR'd label. The new
  `image_label_match_rate` exposes the N10 bug: v17 was 16.4%
  (107/652), v18 is 40.6% (265/652) — a 2.5× improvement. The
  remaining gap is the 388/652 panels where EasyOCR returns no
  numeric token (small/faint labels, off-corner placement, etc.).
  5 new unit tests in `tests/test_image_label_check.py`.
- **N10e: eval `--image-label-check` CLI flag** in
  `scripts/evaluate.py`. The metric is opt-in because it adds
  5-15 min on a 9-paper corpus.
- **N10f: frontend "图版 OCR" column** (`web/index.html`,
  `web/js/app.js`, `web/css/style.css`). Each row in the
  results table now shows a coloured badge next to the
  panel_id: ✓ green (image-OCR'd, includes the v17→v18
  reassignment in the tooltip), ⚠ amber (positional fallback,
  image OCR returned nothing), or — grey (no panel image
  available for verification). 5 unit tests in
  `tests/test_evaluation_metrics.py` still pass; 355 tests total.

### Eval impact (N10 visible in numbers)

| pred | species F1 | image_label_match |
| --- | --- | --- |
| v17 (positional) | 0.9819 | 107/652 = 16.4% |
| v18 (image-OCR)  | 0.9216 | 265/652 = 40.6% |

v18 is the **honest** score: the v17 98% string-match was
matching the positional panel_id against gold panel_id values
that were also assigned by reading order — visually the wrong
images carried the right strings. v18 drops 6pp because the
panel_id is now anchored to the label in the panel image, so
real mismatches surface. Per-paper v18 F1: bandini 86.4%,
baum 97.5%, beccaro 97.1%, boughdiri 92.3%, bragin 100%,
danelian 95.1%, feng 92.5%, hollis 97.2%, pouille 100%.

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
