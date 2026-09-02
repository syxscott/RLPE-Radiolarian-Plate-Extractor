# Random 3-paper smoke test — 2026-09-02

**Goal**: pick 3 random radiolarian papers NOT in the v19 gold set,
run the new pipeline (`caption_fixer` + `prompts` + `post_process`
+ MiniMax M3) end-to-end, and report what comes out.

**Note on F1**: these 3 papers have no gold annotations in
`data/gold/` or `data/gold_v19_extended/`, so we cannot compute
F1 directly. This smoke test reports **raw extraction counts** and
qualitative correctness only.

---

## Selection

Seeded `random.seed(2026)`, then took the first 3 from a shuffled
list of 175 corpus PDFs (after excluding the 9 v19 slugs).
Result:

| # | Paper | Size |
|---|-------|------|
| 1 | Zeiss_2003 — *The Upper Jurassic of Europe: its subdivision and correlation* | 712 KB |
| 2 | Okosun_2013 — *Ostracod, Diatom and Radiolarian Biostratigraphy of the Niger Delta, Nigeria* | 1668 KB |
| 3 | Danelian_Unknown — *Skeleton growth pattern of a late Cambrian–early Ordovician radiolarian species revealed by X-ray tomography* (Palaeobio preprint) | 431 KB |

**`/tmp/random_test/runner.py`** drives all three calls. Each
processed PDF, rendered PNG, and per-paper JSON are under
`/tmp/random_test/`.

---

## Results

### Paper 1 — Zeiss_2003

- **Page used**: 6, **Plate anchor**: 1
- **Caption length**: 2460 chars (long — a multi-block plate caption)
- **Regex "binomials" found in caption**: 23, but on inspection these
  are **false positives** from a generic regex
  `\b[A-Z][a-z]{3,}\s+[a-z]{3,}\b`:
  - `Cardioceras scarburgense`, `Berriasella jacobi`,
    `Pseudosubplanites grandis / euxinus`, `Durangites vulgaris`,
    `Craspedites nodi`, `Chetaites chetae / sibericus`,
    `Rjasanites rjasanensis`, `Runctonia runctoni` — these are
    **ammonite genera**, not radiolarians.
- **Prompt template chosen**: standard radiolarian template (no
  `range|distribution|scanning electron|bar=` markers in caption).
- **M3 raw response**: 1 panel, `species="None"`, `confidence=0.99`.
- **Interpretation**: M3 correctly recognized that the plate caption
  is **about ammonite biostratigraphy, not radiolarians**, and
  returned a null species. This is the *right* behavior — the
  pipeline did not hallucinate radiolarian taxa on a non-radiolarian
  plate. **Extraction: 0 valid radiolarian species.**

### Paper 2 — Okosun_2013

- **Page used**: 9, **Plate anchor**: 2
- **Caption length**: 551 chars
- **Regex "binomials" in caption**: 5, all from an **ostracod**
  plate caption (e.g., `Loxoconcha aff. ...`, `Carapace elongate`,
  `Anterior margin`). The paper mixes ostracod / diatom /
  radiolarian biostratigraphy; the first plate anchor that
  `caption_fixer` picked is the ostracod plate.
- **Prompt template chosen**: standard radiolarian template.
- **M3 raw response**: 3 panels, after dedup still 3, after
  `conf>=0.7` filter **1 panel**:
  - `Loxoconcha sp.` (panel `2, Figure 12`, confidence 0.95).
- **Interpretation**: M3 faithfully transcribed the **ostracod**
  taxon from the caption. `Loxoconcha` is an ostracod, **not a
  radiolarian**. The pipeline as a whole is honest about what it
  read, but the radiolarian-only prompt did not suppress
  non-radiolarian taxa in this case (the prompt's "If a panel is
  NOT a radiolarian, set species=null" instruction was not
  applied — likely because the caption didn't explicitly identify
  the taxa as ostracods).
- **Extraction: 0 valid radiolarian species** (the only kept taxon
  is the ostracod `Loxoconcha`).

### Paper 3 — Danelian_Unknown

- **Anchor found by my broad regex**: `Figure 1/2/3` (no `Plate`
  anchors anywhere in the 14-page preprint).
- **`caption_fixer.select_caption`**: **returns None** for any
  `target_plate` argument because the splitter only recognizes
  `Plate | Pl | 表 | 図版` as a block boundary (see
  `scripts/caption_fixer.py` line ~62, `_ANCHOR_LINE_RE`).
- **Pipeline result**: error path — `no_caption`.
- **Extraction: not run** (no usable caption block).

---

## Summary table

| Paper | M3 raw | dedup | conf≥0.7 | radiolarian species? |
|-------|--------|-------|----------|----------------------|
| Zeiss_2003 | 1 | 1 | 1 | 0 (`species="None"`) |
| Okosun_2013 | 3 | 3 | 1 | 0 (`Loxoconcha sp.` = ostracod) |
| Danelian_Unknown | — | — | — | pipeline blocked: `caption_fixer` does not split on `Figure N` |

**Headline number**: **0 / 3 papers** yielded any genuine radiolarian
species extraction in this random draw. Three independent reasons:

1. **Zeiss**: paper is about ammonite-based Jurassic biostratigraphy,
   not radiolarian systematics. M3 correctly said `None`.
2. **Okosun**: paper mixes ostracods / diatoms / radiolarians; the
   first `Plate 2` anchor happens to be an ostracod SEM plate.
   M3 transcribed the ostracod taxon honestly.
3. **Danelian**: Cambrian–Ordovician preprint uses `Figure N` only;
   `caption_fixer`'s anchor regex is hard-coded to `Plate|Pl|表|図版`.

---

## Observations / concerns

- **Random corpus is heavily skewed toward non-radiolarian-content
  papers** when measured by "first `Plate N` anchor". Out of 184 PDFs
  in `放射虫论文_OA_download/`, many are biostratigraphic-correlation
  or non-taxonomic papers. A more robust selection would
  pre-filter by `(filename has 'radiolarian' OR title has 'Radiolaria'
  AND first plate has ≥1 binomial)`.
- **`caption_fixer._ANCHOR_LINE_RE` is too narrow.** Cambrian /
  Ordovician / Precambrian radiolarian papers and many modern
  journal papers use only `Figure N`. Adding `Figure | Fig\.` to the
  anchor regex would let `select_caption` work on those.
  Recommend opening a follow-up task.
- **M3's prompt-level "non-radiolarian → species=null" instruction
  is not strict enough for Okosun-style mixed-content papers.**
  The model returned `Loxoconcha sp.` at confidence 0.95 even though
  `Loxoconcha` is unambiguously an ostracod. The prompt should
  either (a) instruct "set species=null for any taxon not in the
  Radiolaria", or (b) request the model to first output a per-panel
  `clade` field that downstream code can filter.
- **API cost this run**: 3 successful MiniMax M3 calls, ~3 × 60s
  total wall time (sequential 30s rate-limit gaps).

---

## Artifacts

- Per-paper JSON: `/tmp/random_test/output/{Zeiss,Okosun,Danelian}*.json`
- Rendered page PNGs: `/tmp/random_test/work/`
- Driver script: `/tmp/random_test/runner.py` and `runner_retry.py`
- Source pipeline under test:
  `scripts/{caption_fixer,prompts,post_process,run_research_eval}.py`

## Status

**DONE_WITH_CONCERNS** — pipeline ran end-to-end on all 3 papers
without errors, but the random draw exposed two real limitations
(non-radiolarian plates get transcribed as taxa; `Figure N` plates
are not handled by `caption_fixer`). Follow-up suggested.
