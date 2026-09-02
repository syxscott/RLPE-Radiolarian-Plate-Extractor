# Text Extraction + Occurrence Grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add radiolarian species extraction from non-plate content (range charts, body text) and occurrence-grouping for same-species across multiple figures in the same paper.

**Architecture:** Two new pure-Python modules (`scripts/text_extract.py`, `scripts/occurrence.py`) plus a new `TEXT_MODE_PROMPT` and minor wiring in `run_research_eval.py`. Zero new API cost for the regex path; M3 text-mode fires only when `caption_fixer` returns None.

**Tech Stack:** Python 3.11, pymupdf, hashlib, existing `rlpe.llm_backends.MiniMaxM3Backend`, existing `caption_fixer` / `prompts` / `post_process` / `gold_eval_anchored` modules.

**Spec:** `docs/superpowers/specs/2026-09-02-text-extraction-and-occurrence-grouping-design.md`

---

## File Structure

### New files
- `scripts/text_extract.py` — pure-Python `extract_species_from_text(pdf_path) -> list[dict]`
- `scripts/occurrence.py` — pure-Python `occurrence_group_id(paper_id, species)` + `add_occurrence_groups(preds)`
- `tests/test_text_extract.py` — unit tests for text extraction
- `tests/test_occurrence.py` — unit tests for occurrence grouping

### Modified files
- `scripts/prompts.py` — add `TEXT_MODE_PROMPT` + `select_text_mode_prompt(text)`
- `scripts/run_research_eval.py` — wire text extract + occurrence group + (optional) M3 text-mode fallback

---

### Task 1: `scripts/text_extract.py` + tests

**Files:**
- Create: `scripts/text_extract.py`
- Create: `tests/test_text_extract.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_text_extract.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from text_extract import extract_species_from_text

_PDF_DIR = Path('/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/data/pdfs')

def _path(slug: str) -> Path:
    for p in _PDF_DIR.glob(f'{slug}*'):
        return p
    raise FileNotFoundError(slug)


def test_extract_finds_binomials():
    rows = extract_species_from_text(_path('bandini2011'))
    assert len(rows) > 0
    for r in rows:
        assert 'paper_id' in r
        assert 'species' in r
        assert 'page_num' in r
        assert r['extraction_method'] == 'regex_list'
        assert isinstance(r['page_num'], int)
        assert r['page_num'] >= 1


def test_extract_filters_english_phrases():
    """Denylist drops 'Many species', 'Each individual', etc."""
    rows = extract_species_from_text(_path('bandini2011'))
    species = {r['species'] for r in rows}
    for forbidden in ['Many species', 'Most samples', 'Each individual']:
        assert forbidden not in species


def test_extract_includes_location():
    """Each row has page_num and char_offset for traceability."""
    rows = extract_species_from_text(_path('bandini2011'))
    for r in rows:
        assert r['page_num'] >= 1
        assert r['char_offset'] >= 0
        assert r['context_50char']  # non-empty string
        assert isinstance(r['context_50char'], str)


def test_extract_dedups_same_species_same_page():
    """Same normalized species on same page appears only once."""
    rows = extract_species_from_text(_path('bandini2011'))
    from collections import Counter
    by_key = Counter((r['paper_id'], r['normalized_species'], r['page_num']) for r in rows)
    max_count = max(by_key.values())
    assert max_count <= 1, f'found duplicate (paper, sp, page) keys: {max_count}'


def test_extract_uses_known_denylist():
    """Same set of English false-positive phrases as caption_fixer._BINOMIAL_DENY."""
    from text_extract import _BINOMIAL_DENY
    expected = {
        'species', 'genera', 'genus', 'sample', 'samples', 'individual',
        'individuals', 'figure', 'figures', 'table', 'caption', 'locality',
        'localities', 'text', 'word', 'words', 'material', 'materials',
        'section', 'plate', 'many', 'most', 'several', 'each',
    }
    assert _BINOMIAL_DENY == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_text_extract.py -v`
Expected: ImportError for `text_extract` module (doesn't exist yet)

- [ ] **Step 3: Implement scripts/text_extract.py**

Create `scripts/text_extract.py`:
```python
"""Pure-Python text-level radiolarian species extractor.

Scans the full PDF text for binomial 'Genus species' patterns. No
LLM call, no gold reference — generic heuristic only. Used as a
fallback / supplement to M3 plate-mode extraction.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pymupdf

# Word-boundary binomial pattern: 'Genus species' (lowercase, 3+ chars each).
# Same regex as caption_fixer so extractor + caption-fixer agree on
# what's a "binomial".
_BINOMIAL_RE = re.compile(r"\b([A-Z][a-z]{3,})\s+([a-z]{3,})\b")

# Deny-list: common English phrases that look like binomials but aren't
# taxa. Same list as caption_fixer._BINOMIAL_DENY (kept in sync — if you
# update one, update the other).
_BINOMIAL_DENY = frozenset({
    'species', 'genera', 'genus', 'sample', 'samples', 'individual',
    'individuals', 'figure', 'figures', 'table', 'caption', 'locality',
    'localities', 'text', 'word', 'words', 'material', 'materials',
    'section', 'plate', 'many', 'most', 'several', 'each',
})


def _normalize_species(genus: str, species: str) -> str:
    """Return a canonicalized form for dedup + occurrence grouping.

    'Williriedellum  carpathicum' → 'Williriedellum carpathicum'
    """
    return f"{genus.strip()} {species.strip()}"


def extract_species_from_text(
    pdf_path: str | Path,
    paper_id: str | None = None,
) -> list[dict]:
    """Return all binomial 'Genus species' matches in the PDF.

    Each row:
        paper_id         : str (inferred from path if not given)
        species          : str  (raw 'Genus species')
        normalized_species: str  ('Genus species', whitespace-stripped)
        page_num         : int
        char_offset      : int   (offset in concatenated text)
        context_50char  : str   (±50 chars around the match)
        extraction_method: 'regex_list'
    """
    pdf_path = Path(pdf_path)
    if paper_id is None:
        paper_id = pdf_path.stem

    doc = pymupdf.open(str(pdf_path))
    # Concatenate all pages with page markers; we record the absolute
    # char_offset of each match so callers can locate it in the source.
    page_offsets: list[int] = []
    chunks: list[str] = []
    cursor = 0
    for page in doc:
        text = page.get_text() or ""
        page_offsets.append(cursor)
        chunks.append(text)
        cursor += len(text) + 1  # +1 for a separator newline we'll add

    full_text = "\n".join(chunks)
    doc.close()

    seen: set[tuple[str, int]] = set()  # (normalized_species, page_num)
    out: list[dict] = []
    for m in _BINOMIAL_RE.finditer(full_text):
        genus = m.group(1)
        species_word = m.group(2)
        if species_word.lower() in _BINOMIAL_DENY:
            continue
        norm = _normalize_species(genus, species_word)
        # Compute page_num from absolute offset.
        abs_start = m.start()
        page_num = 1
        for i, off in enumerate(page_offsets, start=1):
            if abs_start >= off:
                page_num = i
        key = (norm, page_num)
        if key in seen:
            continue
        seen.add(key)
        ctx_start = max(0, abs_start - 50)
        ctx_end = min(len(full_text), abs_start + 50 + len(m.group(0)))
        out.append({
            'paper_id': paper_id,
            'species': m.group(0),
            'normalized_species': norm,
            'page_num': page_num,
            'char_offset': abs_start,
            'context_50char': full_text[ctx_start:ctx_end],
            'extraction_method': 'regex_list',
        })
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_text_extract.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor
git add scripts/text_extract.py tests/test_text_extract.py
git -c user.name="Claude Code" -c user.email="claude@anthropic.com" commit -m "feat(scripts): add text_extract (regex-based radiolarian extraction)"
```

---

### Task 2: `scripts/occurrence.py` + tests

**Files:**
- Create: `scripts/occurrence.py`
- Create: `tests/test_occurrence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_occurrence.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from occurrence import occurrence_group_id, add_occurrence_groups

def test_id_starts_with_occ_8_chars():
    gid = occurrence_group_id('bandini2011', 'Williriedellum carpathicum')
    assert gid.startswith('occ_')
    assert len(gid) == len('occ_') + 6  # 6 hex chars


def test_id_stable_across_calls():
    a = occurrence_group_id('bandini2011', 'Williriedellum carpathicum')
    b = occurrence_group_id('bandini2011', 'Williriedellum carpathicum')
    assert a == b


def test_id_normalizes_species_cf_aff():
    """'X cf. Y' and 'X aff. Y' share a group (the species without qualifier)."""
    a = occurrence_group_id('p1', 'Genus cf. species')
    b = occurrence_group_id('p1', 'Genus species')
    c = occurrence_group_id('p1', 'Genus aff. species')
    # 'Genus cf. species' -> 'Genus species' (after parse_open_nomenclature stripping)
    # The normalized form depends on _norm_species in evaluation.metrics.
    # If the normalized forms match, the ids should match.
    # Just verify they're all deterministic for now.
    assert a == b
    assert b != c  # 'aff.' strips to 'species' too, so a==c is also expected.
    # So really: a == b == c for these inputs. Confirm:
    assert a == c


def test_id_different_paper_different_group():
    a = occurrence_group_id('p1', 'Genus species')
    b = occurrence_group_id('p2', 'Genus species')
    assert a != b


def test_id_different_species_different_group():
    a = occurrence_group_id('p1', 'Genus species')
    b = occurrence_group_id('p1', 'Other species')
    assert a != b


def test_add_occurrence_groups_preserves_rows():
    preds = [
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '1', 'species': 'A sp', 'confidence': 0.9},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '2', 'species': 'A sp', 'confidence': 0.8},
        {'paper_id': 'p1', 'figure_id': 'f2', 'panel_id': '1', 'species': 'B sp', 'confidence': 0.7},
    ]
    out = add_occurrence_groups(preds)
    assert len(out) == 3
    assert all('occurrence_group_id' in r for r in out)
    # First two rows (same paper, same species) share a group
    assert out[0]['occurrence_group_id'] == out[1]['occurrence_group_id']
    # Third row (different species) has a different group
    assert out[2]['occurrence_group_id'] != out[0]['occurrence_group_id']


def test_add_occurrence_groups_handles_missing_fields():
    """Empty paper_id or None species → still produces a deterministic group id."""
    preds = [
        {'paper_id': '', 'species': 'A sp'},
        {'paper_id': 'p1', 'species': None},
    ]
    out = add_occurrence_groups(preds)
    assert all('occurrence_group_id' in r for r in out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_occurrence.py -v`
Expected: ImportError for `occurrence` module (doesn't exist yet)

- [ ] **Step 3: Implement scripts/occurrence.py**

Create `scripts/occurrence.py`:
```python
"""Group identical species across multiple figures in a paper.

Two preds are in the same occurrence group iff:
  - same paper_id, AND
  - same normalized species (cf./aff. split, lowered, etc.)

The group id is deterministic: same input → same output.
"""
from __future__ import annotations

import hashlib
from typing import Any

# Reuse the same normalization as the eval pipeline so occurrences
# match across paper_id (bandini2011 vs 4f1bf415485765b8) and across
# cf./aff. qualifiers.
def _normalize_species_for_occurrence(species: str | None) -> str:
    if not species:
        return ""
    # Use the existing _norm_species for consistency.
    from rlpe.evaluation.metrics import _norm_species
    return _norm_species(species) or ""


def occurrence_group_id(paper_id: str | None, species: str | None) -> str:
    """Return a deterministic 6-char group id for this (paper, species) pair.

    Same paper + same normalized species = same group. Different paper
    or different species = different group.
    """
    pid = (paper_id or "").strip()
    sp = _normalize_species_for_occurrence(species)
    raw = f"{pid}|{sp}".encode()
    return "occ_" + hashlib.sha1(raw).hexdigest()[:6]


def add_occurrence_groups(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of `preds` with `occurrence_group_id` added to each row.

    Does not mutate the input list. Existing fields are preserved.
    """
    out = []
    for p in preds:
        q = dict(p)
        q['occurrence_group_id'] = occurrence_group_id(
            p.get('paper_id', ''), p.get('species'),
        )
        out.append(q)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_occurrence.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor
git add scripts/occurrence.py tests/test_occurrence.py
git -c user.name="Claude Code" -c user.email="claude@anthropic.com" commit -m "feat(scripts): add occurrence grouping (occ_xxxxxx per paper+species)"
```

---

### Task 3: `TEXT_MODE_PROMPT` + `select_text_mode_prompt` + tests

**Files:**
- Modify: `scripts/prompts.py`
- Create: `tests/test_text_mode_prompt.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_text_mode_prompt.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prompts import (
    TEXT_MODE_PROMPT,
    select_text_mode_prompt,
    SEM_PLATE_PROMPT,
    RANGE_CHART_PROMPT,
    MAP_PROMPT,
    GENERIC_PROMPT,
)


def test_text_mode_prompt_exists():
    assert isinstance(TEXT_MODE_PROMPT, str)
    assert len(TEXT_MODE_PROMPT) > 50


def test_text_mode_prompt_no_gold_taxa():
    """No specific taxa leak into the text-mode prompt (anti-overfitting)."""
    for forbidden in ['Archaeodictyomitra', 'Williriedellum', 'Hiscocapsa', 'praeparvicingula']:
        assert forbidden.lower() not in TEXT_MODE_PROMPT.lower()


def test_text_mode_prompt_has_output_format():
    """The prompt must include JSON array output format instructions."""
    assert 'JSON' in TEXT_MODE_PROMPT
    assert 'array' in TEXT_MODE_PROMPT.lower()


def test_text_mode_prompt_requests_location():
    """The prompt should ask for location/page context (per spec Feature A.3)."""
    lower = TEXT_MODE_PROMPT.lower()
    assert any(kw in lower for kw in ['page', 'location', 'context'])


def test_text_mode_prompt_distinct_from_plate_prompts():
    assert TEXT_MODE_PROMPT != SEM_PLATE_PROMPT
    assert TEXT_MODE_PROMPT != RANGE_CHART_PROMPT
    assert TEXT_MODE_PROMPT != MAP_PROMPT
    assert TEXT_MODE_PROMPT != GENERIC_PROMPT


def test_select_text_mode_prompt_returns_text_mode_for_any_caption():
    """select_text_mode_prompt() always returns TEXT_MODE_PROMPT (caller has already
    decided 'text mode' is the right choice)."""
    assert select_text_mode_prompt('any caption here') is TEXT_MODE_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_text_mode_prompt.py -v`
Expected: ImportError for `TEXT_MODE_PROMPT` and `select_text_mode_prompt`

- [ ] **Step 3: Add TEXT_MODE_PROMPT to scripts/prompts.py**

In `scripts/prompts.py`, add this constant + helper **at the end of the file** (just before `_PREDICATE_PATTERNS`):

```python
TEXT_MODE_PROMPT = _build_prompt(
    goal="Given a radiolarian paper's full text (no plate figures available), "
         "extract every radiolarian species mentioned in the text along with its location.",
    special="Output one row per species, with 'location' describing the page or section. "
            "label = the species name; panel_id = the page or section identifier. "
            "If the paper is not about Radiolaria, set species=null.",
)


def select_text_mode_prompt(caption: str) -> str:
    """Always returns TEXT_MODE_PROMPT (caller has already decided to use text mode)."""
    return TEXT_MODE_PROMPT
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_text_mode_prompt.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Plate-Extractor
git add scripts/prompts.py tests/test_text_mode_prompt.py
git -c user.name="Claude Code" -c user.email="claude@anthropic.com" commit -m "feat(scripts): add TEXT_MODE_PROMPT + select_text_mode_prompt"
```

---

### Task 4: Wire text_extract + occurrence into `run_research_eval.py`

**Files:**
- Modify: `scripts/run_research_eval.py`
- Create: `tests/test_run_research_eval_wiring.py` (smoke test that integration works)

- [ ] **Step 1: Write the failing wiring test**

Create `tests/test_run_research_eval_wiring.py`:
```python
"""Smoke test that run_research_eval.py correctly wires text_extract + occurrence."""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

REPO = Path('/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor')
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

# Stub MiniMaxM3Backend so import of run_research_eval doesn't try a real call
os.environ.setdefault('ANTHROPIC_API_KEY', 'dummy')
os.environ.setdefault('ANTHROPIC_BASE_URL', 'https://test.invalid')
os.environ.setdefault('ANTHROPIC_MODEL', 'dummy')

import rlpe.llm_backends
class _StubBackend:
    def __init__(self, *a, **kw): pass
    def infer_panel(self, *a, **kw): return {'error': 'stubbed'}
rlpe.llm_backends.MiniMaxM3Backend = _StubBackend

from run_research_eval import _enrich_preds_with_text_and_group


def test_enrich_attaches_occurrence_group_id():
    preds = [
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '1', 'species': 'A sp', 'confidence': 0.9},
        {'paper_id': 'p1', 'figure_id': 'f2', 'panel_id': '1', 'species': 'A sp', 'confidence': 0.8},
    ]
    out = _enrich_preds_with_text_and_group(preds)
    assert all('occurrence_group_id' in r for r in out)
    assert out[0]['occurrence_group_id'] == out[1]['occurrence_group_id']


def test_enrich_does_not_modify_input():
    preds = [
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '1', 'species': 'A sp', 'confidence': 0.9},
    ]
    snapshot = list(preds)
    _enrich_preds_with_text_and_group(preds)
    assert preds == snapshot  # input not mutated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_run_research_eval_wiring.py -v`
Expected: ImportError for `_enrich_preds_with_text_and_group`

- [ ] **Step 3: Add the enrich helper to run_research_eval.py**

In `scripts/run_research_eval.py`, add this function **between the existing `load_gold_for` and `find_pdf` functions** (or anywhere module-level, but not inside `main` or `extract_panels_for_paper`):

```python
def _enrich_preds_with_text_and_group(preds: list[dict]) -> list[dict]:
    """Add occurrence_group_id to each pred row.

    Returns a new list (does not mutate input). This is the
    integration point for Feature B: same species in different
    figures in the same paper get the same group id, so the eval
    pipeline can deduplicate when desired.
    """
    from occurrence import add_occurrence_groups
    return add_occurrence_groups(preds)
```

- [ ] **Step 4: Modify extract_panels_for_paper to ALSO return text-extract rows when no plate was found**

In `scripts/run_research_eval.py`, locate the `extract_panels_for_paper` function. After the existing M3 call (which returns rows for plate-mode), add a fallback to text-mode. Replace the function body with this:

```python
def extract_panels_for_paper(backend, slug: str, gold: list[dict]) -> list[dict]:
    """Run caption_fixer + prompts + M3 + post_process on one paper.

    Three modes:
    - plate_M3 : M3 with SEM_PLATE/RANGE_CHART/MAP prompt when caption_fixer finds a plate caption
    - text_M3  : M3 with TEXT_MODE_PROMPT when no plate caption is found AND the paper is radiolarian-related
    - regex_list: (always run as supplement) — extract_species_from_text() for ALL papers
    """
    from text_extract import extract_species_from_text
    from occurrence import occurrence_group_id

    pdf_path = find_pdf(slug)
    if pdf_path is None:
        print(f'  no PDF for {slug}, skip')
        return []
    pid = stable_id(pdf_path)
    print(f'  {slug} paper_id={pid}')

    # Always run regex extract first as supplement.
    regex_rows = extract_species_from_text(pdf_path, paper_id=pid)
    if regex_rows:
        print(f'    regex_list: {len(regex_rows)} species')

    # Find densest gold figure
    fig_counts = Counter(g.get('figure_id') for g in gold)
    if not fig_counts:
        return _to_rows(regex_rows, pid) if regex_rows else []
    target_fig, _ = fig_counts.most_common(1)[0]
    m = re.search(r'_p(\d{3})_pl(\d+)', target_fig)
    if not m:
        return _to_rows(regex_rows, pid) if regex_rows else []
    page_num = int(m.group(1))

    doc = pymupdf.open(str(pdf_path))
    if page_num > len(doc):
        doc.close()
        return _to_rows(regex_rows, pid) if regex_rows else []
    full_text = '\n'.join(p.get_text() for p in doc)

    # Try plate caption first
    plate_anchor = str(int(m.group(2)))
    caption = select_caption(full_text, target_plate=int(plate_anchor))
    use_text_mode = caption is None
    if not use_text_mode and not _is_radiolarian_paper(full_text):
        # Not a radiolarian paper, skip M3 call but keep regex rows
        doc.close()
        return _to_rows(regex_rows, pid) if regex_rows else []
    if not use_text_mode and len(caption) < 100:
        use_text_mode = True

    if use_text_mode:
        # M3 text-mode fallback
        sys_prompt = select_text_mode_prompt(caption or full_text)
        img = None  # text-mode doesn't need an image
    else:
        # Render page
        pix = doc[page_num - 1].get_pixmap(dpi=150)
        img_path = f'/tmp/{slug}_p{page_num}.png'
        pix.save(img_path)
        img = Image.open(img_path)
        sys_prompt = select_prompt(caption)

    doc.close()

    try:
        r = call_m3(backend, img, caption or full_text, sys_prompt)
    except Exception as e:
        print(f'  API error: {e}')
        return _to_rows(regex_rows, pid) if regex_rows else []
    if not r or r.get('error') or r.get('fallback_used'):
        return _to_rows(regex_rows, pid) if regex_rows else []
    if r.get('_is_multi_panel') and isinstance(r.get('panels'), list):
        panels = r['panels']
    else:
        panels = [r]
    preds = []
    for p in panels:
        sp_raw = p.get('species')
        sp, qual = parse_open_nomenclature(sp_raw)
        qual_str = f" {qual}" if qual else ""
        preds.append({
            'paper_id': pid,
            'figure_id': target_fig,
            'panel_id': normalize_panel_id(p.get('label', '')),
            'species': f"{sp}{qual_str}" if sp else None,
            'confidence': p.get('confidence', 0.0),
            'location': f'Plate {int(plate_anchor)}, Fig. {p.get("label","?")} (p. {page_num})' if not use_text_mode else f'p. {page_num} (text section)',
            'extraction_method': 'text_M3' if use_text_mode else 'plate_M3',
        })
    # Also add regex rows
    if regex_rows:
        for r in regex_rows:
            preds.append({
                'paper_id': r['paper_id'],
                'figure_id': f'text_section_p{r["page_num"]}',
                'panel_id': None,
                'species': r['species'],
                'confidence': 1.0,  # regex is exact (no LLM hallucination)
                'location': f'p. {r["page_num"]} (regex match: "{r["context_50char"][:50]}...")',
                'extraction_method': 'regex_list',
            })
    preds = dedup_panels(preds)
    preds = filter_low_confidence(preds, threshold=0.7)
    return preds


def _to_rows(regex_rows: list[dict], paper_id: str) -> list[dict]:
    """Convert regex_list rows to the standard pred-row shape."""
    out = []
    for r in regex_rows:
        out.append({
            'paper_id': r['paper_id'],
            'figure_id': f'text_section_p{r["page_num"]}',
            'panel_id': None,
            'species': r['species'],
            'confidence': 1.0,
            'location': f'p. {r["page_num"]} (regex match)',
            'extraction_method': 'regex_list',
        })
    return out


def _is_radiolarian_paper(text: str) -> bool:
    """Heuristic: does the text look radiolarian-themed? Skip non-radio papers."""
    import re as _re
    return bool(_re.search(r'Radiolaria|radiolarian|Polycystina|Nassellaria|Spumellaria',
                            text, _re.IGNORECASE))
```

Also, **add the enrich call to `main()`** — find the line `all_preds.extend(preds)` in `main()` and replace it with:
```python
    for i, slug in enumerate(all_papers):
        is_train = i < len(split['train'])
        if i > 0:
            time.sleep(60)
        gold = load_gold_for(slug)
        preds = extract_panels_for_paper(backend, slug, gold)
        # Feature B: attach occurrence_group_id to every pred row
        preds = _enrich_preds_with_text_and_group(preds)
        if is_train:
            train_preds.extend(preds)
            train_gold.extend(gold)
        else:
            test_preds.extend(preds)
            test_gold.extend(gold)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_run_research_eval_wiring.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Run the full test suite to confirm no regression**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/ -v -k "test_text_extract or test_occurrence or test_text_mode_prompt or test_run_research_eval_wiring" 2>&1 | tail -30`
Expected: All new tests pass (5 + 7 + 5 + 2 = 19 new tests)

- [ ] **Step 6: Commit**

```bash
cd /home/user/shenanxuan/RLPE-Radiolarian-Plate-Extractor
git add scripts/run_research_eval.py tests/test_run_research_eval_wiring.py
git -c user.name="Claude Code" -c user.email="claude@anthropic.com" commit -m "feat(scripts): wire text_extract + occurrence into run_research_eval"
```

---

## Acceptance criteria (whole plan)

- [ ] Task 1: `scripts/text_extract.py` + 5 tests, all pass, 0 API calls
- [ ] Task 2: `scripts/occurrence.py` + 7 tests, all pass, deterministic group id
- [ ] Task 3: `TEXT_MODE_PROMPT` + 5 tests, all pass, no specific taxa leak
- [ ] Task 4: `run_research_eval.py` correctly uses text_extract for all papers + occurrence_group_id on every row
- [ ] End-to-end smoke on 3 random v19 papers: regex_list rows appear; existing plate rows unchanged; F1 within ±0.02 of pre-feature baseline (proves no regression)

## Success criteria

- 19 new tests all pass
- 1 final commit on HEAD per task (4 total)
- `run_research_eval.py` produces preds with `occurrence_group_id` and `location` fields populated
- For plate-less papers: regex_list rows appear with extraction_method='regex_list'
- For plate papers: existing plate_M3 rows unchanged + extra regex_list rows supplement
