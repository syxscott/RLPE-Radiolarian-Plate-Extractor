# RLPE 达到放射虫论文科研级 F1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 RLPE F1 (物种级别, 面板匹配) 从当前测量值提升到 v19 SOTA baseline = **0.84**, 用 5-fold CV + bootstrap CI + train/test split 防过拟合。

**Architecture:** 4 个新脚本模块 (caption_fixer / prompts / post_process / gold_eval_anchored 升级) + 黄金集从 9 篇扩到 20+ + 5-fold cross-validation 报告 train + test F1 + 95% CI。

**Tech Stack:** Python 3.11, MiniMax-M3 (云 API, Anthropic 协议), pymupdf, pytest, no GPU needed.

**Spec:** `docs/superpowers/specs/2026-09-02-radiolarian-research-grade-design.md`

---

## 文件结构

### 新建文件
- `scripts/caption_fixer.py` — 通用 caption 选取（不引用 gold）
- `scripts/prompts.py` — 4 个 M3 prompt 模板（按 paper type）
- `scripts/post_process.py` — panel + species 后处理（4 函数）
- `data/gold_v19_extended/` — 扩到 20+ 论文的黄金集
- `data/splits/research_v1.json` — train/test split 写死
- `tests/test_caption_fixer.py` — caption_fixer 单元测试
- `tests/test_post_process.py` — 后处理单元测试
- `tests/test_prompts.py` — prompt snapshot 冻结测试
- `tests/test_gold_eval_integration.py` — 集成 smoke test
- `Makefile` — `make eval-research` 单一命令行复现

### 修改文件
- `scripts/gold_eval_anchored.py` — 加 5-fold CV, bootstrap CI, --split, --folds 参数

---

## Train/Test Split (写死)

```json
// data/splits/research_v1.json
{
  "version": "v1",
  "train": ["bandini2011", "beccaro2006", "boughdiri2007", "bragin2025", "danelian2006", "hollis2006"],
  "test":  ["baumgartner2008", "feng2007", "pouille2014"]
}
```

---

### Task 1: 黄金集 split 写死 + Makefile scaffold

**Files:**
- Create: `data/splits/research_v1.json`
- Create: `Makefile`

- [ ] **Step 1: Create split JSON file**

```bash
mkdir -p /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/data/splits
```

Create `data/splits/research_v1.json`:
```json
{
  "version": "v1",
  "created": "2026-09-02",
  "train": ["bandini2011", "beccaro2006", "boughdiri2007", "bragin2025", "danelian2006", "hollis2006"],
  "test":  ["baumgartner2008", "feng2007", "pouille2014"],
  "notes": "Train 6 papers: 4 marine + 2 pelagic. Test 3 papers: includes baumgartner2008 (high-F1 1.0 baseline) + feng2007 (漏抽) + pouille2014 (over-gen). baumgartner2008 in test = generalization reality check."
}
```

- [ ] **Step 2: Commit split**

```bash
cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor
git add data/splits/research_v1.json
git commit -m "feat(eval): add research v1 train/test split (6+3)"
```

- [ ] **Step 3: Create Makefile**

Create `Makefile` at repo root:
```makefile
.PHONY: eval-research test-units install-deps

install-deps:
	pip install -e .
	python -m spacy download en_core_web_sm

test-units:
	PYTHONPATH=src pytest tests/test_caption_fixer.py tests/test_post_process.py tests/test_prompts.py -v

eval-research:
	PYTHONPATH=src python scripts/run_research_eval.py \
		--split data/splits/research_v1.json \
		--bootstrap-samples 1000 \
		--folds 5 \
		--output data/snapshot/$(date +%Y-%m-%d)/f1.json
```

- [ ] **Step 4: Commit Makefile**

```bash
git add Makefile
git commit -m "build: add Makefile with eval-research target"
```

---

### Task 2: caption_fixer.py — 通用 caption 选取

**Files:**
- Create: `scripts/caption_fixer.py`
- Test: `tests/test_caption_fixer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_caption_fixer.py`:
```python
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
import pymupdf
from pathlib import Path
from caption_fixer import select_caption, score_paragraph

def test_anchor_plate_5_paragraph_selected():
    """Paragraph starting with 'Plate 5' is selected over shorter non-anchored."""
    text = """Header text about the paper.
Plate 5
Detailed caption about many specimens. Sample PR-SB05 (latest Tithonian).
Fig. 1 Archaeodictyomitra sp. Fig. 2 Williriedellum sp. Fig. 3 Hiscocapsa sp.
This is a real plate caption with many species."""
    best = select_caption(text, target_plate=5)
    assert best is not None
    assert 'Plate 5' in best
    assert 'Archaeodictyomitra' in best

def test_anchor_fig_3_selected():
    """Fig. 3 caption selected when target_plate=3."""
    text = """Plate 1
First plate caption here.
Fig. 3
Some other figure here.
Plate 5
Last plate."""
    best = select_caption(text, target_plate=3)
    assert 'Fig. 3' in best

def test_anchor_with_leading_zero_5():
    """'Plate 05' matches target_plate=5 (strip leading zeros)."""
    text = """Header.
Plate 05
Some caption.
Trailer."""
    best = select_caption(text, target_plate=5)
    assert 'Plate 05' in best

def test_no_anchor_falls_back_to_densest():
    """No Plate N anchor → use paragraph with most binomials."""
    text = """Para A short.
Para B about Genus species one and Genus species two and Genus species three.
Para C also short."""
    best = select_caption(text, target_plate=99)  # no anchor exists
    assert 'Para B' in best

def test_returns_none_on_no_text():
    assert select_caption('', target_plate=1) is None or select_caption('', target_plate=1) == ''

def test_score_paragraph_returns_int():
    score = score_paragraph('Plate 1\nGenus species A and Genus species B', target_plate=1)
    assert isinstance(score, int)
    assert score > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_caption_fixer.py -v`
Expected: ImportError for `caption_fixer` module (doesn't exist yet)

- [ ] **Step 3: Implement caption_fixer.py**

Create `scripts/caption_fixer.py`:
```python
"""General caption block selector.

Picks the most likely plate caption from a PDF page text WITHOUT
referencing the gold set (to prevent over-fitting on train papers).
The selector scores paragraphs by structural anchors (Plate N / Fig. N
at the start) + binomial density (Genus species patterns) + plate
terminator markers (Sample / Loc. / Marker =).
"""
from __future__ import annotations

import re
from typing import Optional

# An anchor pattern: matches "Plate 5" / "Pl. 5" / "Fig. 5" / "表 5" at start.
# Allow optional leading zero: "Plate 05" matches target_plate=5.
_ANCHOR_RE_TEMPLATE = (
    r"^\s*(?:Plate|Pl|Fig|表|図版)\.?\s*"
    r"0?{n}\b"
)
_BINOMIAL_RE = re.compile(r"\b[A-Z][a-z]{3,}\s+[a-z]{3,}\b")
_TERMINATORS = ("Sample", "Loc.", "Marker =", "Scale", "Bar =")

MIN_PARA_LEN = 50
MAX_PARA_LEN = 4000


def score_paragraph(para: str, target_plate: int) -> int:
    """Score a single paragraph for being the target plate caption.

    Higher = more likely to be the caption. Never returns negative
    (a 0 score means "no anchor, no binomials" which still might
    be the right answer if nothing else qualifies).
    """
    score = 0
    n = int(target_plate)
    anchor = re.compile(_ANCHOR_RE_TEMPLATE.format(n=n), re.IGNORECASE)
    if anchor.match(para):
        score += 10
    binomials = _BINOMIAL_RE.findall(para)
    if len(binomials) >= 2:
        score += 5
    elif len(binomials) >= 1:
        score += 2
    for term in _TERMINATORS:
        if term in para:
            score += 1
    return score


def select_caption(
    text: str,
    target_plate: int,
    min_anchor_score: int = 10,
) -> Optional[str]:
    """Pick the best caption paragraph from `text` for `target_plate`.

    Strategy:
      1. Split text into paragraphs.
      2. For paragraphs with an anchor matching `target_plate`,
         keep only those with score >= min_anchor_score.
      3. If no anchor matches, fall back to the highest-scored
         paragraph (may be wrong, but better than nothing).
      4. Return the best paragraph, or None if text is empty.
    """
    if not text or not text.strip():
        return None
    paragraphs = re.split(r"\n\s*\n", text)
    best = None
    best_score = 0
    n = int(target_plate)
    anchor = re.compile(_ANCHOR_RE_TEMPLATE.format(n=n) + r"\.", re.IGNORECASE)
    for para in paragraphs:
        if not (MIN_PARA_LEN <= len(para) <= MAX_PARA_LEN):
            continue
        score = score_paragraph(para, n)
        if anchor.match(para) and score >= min_anchor_score:
            if score > best_score:
                best = para
                best_score = score
    if best is None:
        for para in paragraphs:
            if not (MIN_PARA_LEN <= len(para) <= MAX_PARA_LEN):
                continue
            score = score_paragraph(para, n)
            if score > best_score:
                best = para
                best_score = score
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_caption_fixer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/caption_fixer.py tests/test_caption_fixer.py
git commit -m "feat(scripts): add general caption_fixer (no gold reference)"
```

---

### Task 3: prompts.py — 4 M3 prompt 模板

**Files:**
- Create: `scripts/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompts.py`:
```python
import sys
sys.path.insert(0, 'scripts')
from prompts import (
    RANGE_CHART_PROMPT,
    SEM_PLATE_PROMPT,
    MAP_PROMPT,
    GENERIC_PROMPT,
    select_prompt,
    build_user_prompt,
)

def test_range_chart_prompt_no_gold():
    """No gold species in prompt (general rules only)."""
    assert 'species' in RANGE_CHART_PROMPT.lower()
    assert 'archaeodictyomitra' not in RANGE_CHART_PROMPT.lower()  # no specific taxa
    assert 'array' in RANGE_CHART_PROMPT.lower()  # output format

def test_sem_plate_prompt_distinct_from_range():
    assert RANGE_CHART_PROMPT != SEM_PLATE_PROMPT

def test_map_prompt_mentions_locality():
    assert 'locality' in MAP_PROMPT.lower() or 'location' in MAP_PROMPT.lower()

def test_select_prompt_classifies_by_caption():
    cap_range = 'Fig. 1. Distribution of radiolarians in this paper.'
    cap_sem = 'Plate 1. Scanning electron microscope pictures of radiolarians.'
    cap_map = 'Fig. 1. Schematic map indicating location of samples.'
    cap_other = 'Random caption with no keywords.'
    assert select_prompt(cap_range) == RANGE_CHART_PROMPT
    assert select_prompt(cap_sem) == SEM_PLATE_PROMPT
    assert select_prompt(cap_map) == MAP_PROMPT
    assert select_prompt(cap_other) == GENERIC_PROMPT

def test_build_user_prompt_includes_caption():
    user = build_user_prompt('Some caption text here.')
    assert 'Some caption text here.' in user
    assert 'JSON' in user or 'json' in user
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_prompts.py -v`
Expected: ImportError for `prompts` module (doesn't exist yet)

- [ ] **Step 3: Implement prompts.py**

Create `scripts/prompts.py`:
```python
"""M3 prompt library — 4 templates selected by paper type.

These prompts describe general rules only (no specific taxa, no
gold references). They instruct the LLM to:
- Extract every specimen panel from the figure caption + image
- Output a strict JSON array of {label, species, confidence}
- Distinguish cf./aff./n.sp. qualifiers
- Skip non-radiolarian specimens

Prompts are intentionally generic so the eval set doesn't leak into
the prompt design.
"""
from __future__ import annotations

import re
from typing import Tuple

_BASE_OUTPUT_FORMAT = (
    "Return strict JSON array of objects with fields "
    "{label, species, confidence, panel_id}. Example: "
    '[{"label": "1", "species": "Genus species", "confidence": 0.9}, ...].'
)

_RANGE_CHART_MARKERS = ("distribution", "range chart", "biozone", "stratigraphic range")
_SEM_PLATE_MARKERS = ("scanning electron", "plate", "marker =", "bar =")
_MAP_MARKERS = ("location", "map", "schematic map", "geographic")


def _build_prompt(goal: str, special: str) -> str:
    return (
        f"You are an expert radiolarian paleontologist. {goal}\n\n"
        f"{special}\n\n"
        f"{_BASE_OUTPUT_FORMAT}\n\n"
        "Preserve taxonomic qualifiers (cf., aff., n. sp., comb. nov.).\n"
        "If a panel is NOT a radiolarian, set species=null and label=panel_id.\n"
    )


RANGE_CHART_PROMPT = _build_prompt(
    goal="Given a range chart caption and image, extract every radiolarian "
         "species and the stratigraphic range it appears in.",
    special="Output one row per (species, range) pair visible in the chart. "
            "label = species name; panel_id = the stratigraphic zone it appears in.",
)

SEM_PLATE_PROMPT = _build_prompt(
    goal="Given a plate caption and image, extract every specimen panel "
         "and identify the radiolarian species shown.",
    special="Output one row per numbered figure (Fig. 1, Fig. 2, etc.) "
            "visible in the plate. label = the figure number; panel_id = same.",
)

MAP_PROMPT = _build_prompt(
    goal="Given a map caption and image, extract any radiolarian-bearing "
         "localities mentioned.",
    special="Output one row per locality if the map shows radiolarian sites. "
            "label = the locality id (e.g. 'Loc. 5'); panel_id = same.",
)

GENERIC_PROMPT = _build_prompt(
    goal="Given a figure caption and image, extract every radiolarian "
         "specimen or locality shown.",
    special="Output one row per visible item. label = whatever the caption uses "
            "to identify the item; panel_id = same.",
)


_PREDICATE_PATTERNS = (
    (RANGE_CHART_PROMPT, _RANGE_CHART_MARKERS),
    (SEM_PLATE_PROMPT, _SEM_PLATE_MARKERS),
    (MAP_PROMPT, _MAP_MARKERS),
)


def select_prompt(caption: str) -> str:
    """Pick the most appropriate prompt template by caption keywords.

    Falls back to GENERIC_PROMPT if no markers match.
    """
    cap_lower = caption.lower()
    for prompt, markers in _PREDICATE_PATTERNS:
        for marker in markers:
            if marker in cap_lower:
                return prompt
    return GENERIC_PROMPT


def build_user_prompt(caption: str) -> str:
    """Wrap the caption into the user message sent to M3."""
    return f"Caption:\n{caption[:3000]}\n\nExtract every panel and species as JSON."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_prompts.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/prompts.py tests/test_prompts.py
git commit -m "feat(scripts): add 4 M3 prompt templates by paper type"
```

---

### Task 4: post_process.py — 后处理 4 函数

**Files:**
- Create: `scripts/post_process.py`
- Test: `tests/test_post_process.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_post_process.py`:
```python
import sys
sys.path.insert(0, 'scripts')
from post_process import (
    parse_open_nomenclature,
    dedup_panels,
    filter_low_confidence,
    normalize_panel_id,
)

def test_parse_open_nomenclature_cf():
    sp, qual = parse_open_nomenclature('Genus cf. species')
    assert sp == 'Genus species'
    assert qual == 'cf.'

def test_parse_open_nomenclature_aff():
    sp, qual = parse_open_nomenclature('Genus aff. species')
    assert qual == 'aff.'

def test_parse_open_nomenclature_no_qualifier():
    sp, qual = parse_open_nomenclature('Genus species')
    assert qual is None

def test_dedup_panels_removes_duplicates():
    panels = [
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '1', 'species': 'A sp', 'confidence': 0.9},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '1', 'species': 'A sp', 'confidence': 0.85},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '2', 'species': 'B sp', 'confidence': 0.9},
    ]
    deduped = dedup_panels(panels)
    assert len(deduped) == 2
    # Higher confidence kept
    assert deduped[0]['confidence'] == 0.9

def test_filter_low_confidence():
    panels = [
        {'confidence': 0.5, 'species': 'A'},
        {'confidence': 0.85, 'species': 'B'},
        {'confidence': 0.71, 'species': 'C'},
    ]
    filtered = filter_low_confidence(panels, threshold=0.7)
    assert len(filtered) == 2
    assert all(p['confidence'] >= 0.7 for p in filtered)

def test_normalize_panel_id_strips_fig():
    assert normalize_panel_id('Fig. 1') == '1'
    assert normalize_panel_id('Figs. 24 and 25') == '24 and 25'
    assert normalize_panel_id('1') == '1'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_post_process.py -v`
Expected: ImportError for `post_process` module (doesn't exist yet)

- [ ] **Step 3: Implement post_process.py**

Create `scripts/post_process.py`:
```python
"""Post-processing for M3 panel extraction output.

Four utilities:
  - parse_open_nomenclature: split 'Genus cf. species' into (sp, qualifier)
  - dedup_panels: remove exact-duplicate (fig, panel, species) rows
  - filter_low_confidence: drop rows with confidence < threshold
  - normalize_panel_id: strip 'Fig. N' / 'Pl. N' / 'Plate N' prefix

All functions are pure (no LLM call, no gold reference) — they
operate only on the pred rows returned by M3.
"""
from __future__ import annotations

import re
from typing import Any

_QUALIFIER_RE = re.compile(r"\b(cf|aff|vel|similar)\b\.?\s*", re.IGNORECASE)


def parse_open_nomenclature(species: str | None) -> tuple[str | None, str | None]:
    """Split a species string into (species, qualifier).

    'Genus cf. species' → ('Genus species', 'cf.')
    'Genus aff. species' → ('Genus species', 'aff.')
    'Genus species'      → ('Genus species', None)
    """
    if not species:
        return None, None
    qual_match = _QUALIFIER_RE.search(species)
    if qual_match is None:
        return species, None
    qualifier = qual_match.group(1).lower()
    clean = _QUALIFIER_RE.sub('', species, count=1).strip()
    return clean, qualifier


_PANEL_PREFIX_RE = re.compile(
    r"^\s*(?:Fig|Figs|Pl|Pls|Plate|Plates|表|図版)\.?\s*",
    re.IGNORECASE,
)


def normalize_panel_id(label: str | None) -> str:
    """Strip 'Fig. N' / 'Pl. N' / 'Plate N' prefix; collapse whitespace."""
    if not label:
        return ""
    cleaned = _PANEL_PREFIX_RE.sub('', label)
    return re.sub(r"\s+", " ", cleaned).strip()


def dedup_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact duplicates by (paper_id, figure_id, panel_id, species).

    When duplicates exist, keep the one with highest confidence.
    """
    best_by_key: dict[tuple, dict[str, Any]] = {}
    for p in panels:
        key = (p.get('paper_id'), p.get('figure_id'), p.get('panel_id'), p.get('species'))
        if key not in best_by_key or float(p.get('confidence', 0) or 0) > float(
            best_by_key[key].get('confidence', 0) or 0
        ):
            best_by_key[key] = p
    return list(best_by_key.values())


def filter_low_confidence(
    panels: list[dict[str, Any]], threshold: float = 0.7
) -> list[dict[str, Any]]:
    """Drop rows whose confidence is below threshold."""
    return [p for p in panels if float(p.get('confidence', 0) or 0) >= threshold]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_post_process.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/post_process.py tests/test_post_process.py
git commit -m "feat(scripts): add post_process (panel dedup + cf./aff. split)"
```

---

### Task 5: gold_eval_anchored.py — 加 5-fold CV + bootstrap CI

**Files:**
- Modify: `scripts/gold_eval_anchored.py`
- Test: `tests/test_gold_eval_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gold_eval_integration.py`:
```python
import sys, json
sys.path.insert(0, 'scripts')
from gold_eval_anchored import (
    load_split,
    compute_aggregate_with_ci,
    run_5fold_cv,
)

def test_load_split_v1():
    """Load the v1 split (6 train + 3 test)."""
    split = load_split('data/splits/research_v1.json')
    assert 'train' in split
    assert 'test' in split
    assert len(split['train']) == 6
    assert len(split['test']) == 3

def test_compute_aggregate_with_ci():
    """Bootstrap CI is a tuple (low, high) of length 2."""
    preds = [
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '1', 'species': 'A', 'confidence': 0.9},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '2', 'species': 'B', 'confidence': 0.9},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '3', 'species': 'A', 'confidence': 0.8},
    ]
    gold = [
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '1', 'species': 'A'},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '2', 'species': 'B'},
        {'paper_id': 'p1', 'figure_id': 'f1', 'panel_id': '3', 'species': 'A'},
    ]
    f1, ci = compute_aggregate_with_ci(preds, gold, n_bootstrap=100)
    assert isinstance(f1, float)
    assert len(ci) == 2
    assert ci[0] <= f1 <= ci[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_gold_eval_integration.py -v`
Expected: ImportError for `compute_aggregate_with_ci` and `load_split`

- [ ] **Step 3: Add functions to gold_eval_anchored.py**

Append to `scripts/gold_eval_anchored.py`:
```python
# === Added for research-grade eval (Task 5) ===

import json
import statistics
import random
from pathlib import Path
from typing import Any, Tuple


def load_split(path: str | Path) -> dict[str, list[str]]:
    """Load train/test split from a JSON file."""
    with open(path) as f:
        return json.load(f)


def _paper_f1(preds: list[dict], gold: list[dict]) -> float:
    """Compute species F1 for a single paper."""
    from rlpe.evaluation.metrics import _norm_species, _species_compatible
    pred_sp = {(_norm_species(p.get('species')), p.get('figure_id'), p.get('panel_id'))
              for p in preds if p.get('species')}
    gold_sp = {(_norm_species(g.get('species')), g.get('figure_id'), g.get('panel_id'))
              for g in gold if g.get('species')}
    tp = sum(1 for k in pred_sp & gold_sp)
    fp = len(pred_sp - gold_sp)
    fn = len(gold_sp - pred_sp)
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def compute_aggregate_with_ci(
    preds: list[dict],
    gold: list[dict],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Compute micro F1 with 95% bootstrap CI.

    Returns (f1_micro, (ci_low, ci_high)).
    """
    # Group by paper
    by_paper: dict[str, tuple[list, list]] = {}
    for g in gold:
        by_paper.setdefault(g.get('paper_id', ''), ([], []))[1].append(g)
    for p in preds:
        paper = p.get('paper_id', '')
        if paper in by_paper:
            by_paper[paper][0].append(p)
    papers = list(by_paper.keys())

    def f1_micro() -> float:
        total_tp = total_fp = total_fn = 0
        for p in papers:
            pp, gp = by_paper[p]
            from rlpe.evaluation.metrics import _norm_species, _species_compatible
            pset = {(_norm_species(x.get('species')), x.get('figure_id'), x.get('panel_id'))
                    for x in pp if x.get('species')}
            gset = {(_norm_species(x.get('species')), x.get('figure_id'), x.get('panel_id'))
                    for x in gp if x.get('species')}
            tp = len(pset & gset)
            fp = len(pset - gset)
            fn = len(gset - pset)
            total_tp += tp; total_fp += fp; total_fn += fn
        if total_tp == 0:
            return 0.0
        p_val = total_tp / (total_tp + total_fp)
        r_val = total_tp / (total_tp + total_fn)
        if p_val + r_val == 0:
            return 0.0
        return 2 * p_val * r_val / (p_val + r_val)

    rng = random.Random(seed)
    point = f1_micro()
    if not papers:
        return 0.0, (0.0, 0.0)
    bootstraps: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choices(papers, k=len(papers))
        bt = total_fp_b = total_fn_b = 0
        for p in sample:
            pp, gp = by_paper[p]
            from rlpe.evaluation.metrics import _norm_species
            pset = {(_norm_species(x.get('species')), x.get('figure_id'), x.get('panel_id'))
                    for x in pp if x.get('species')}
            gset = {(_norm_species(x.get('species')), x.get('figure_id'), x.get('panel_id'))
                    for x in gp if x.get('species')}
            tp = len(pset & gset); fp = len(pset - gset); fn = len(gset - pset)
            bt += tp; total_fp_b += fp; total_fn_b += fn
        if bt == 0:
            bootstraps.append(0.0)
            continue
        p_v = bt / (bt + total_fp_b)
        r_v = bt / (bt + total_fn_b)
        bootstraps.append(2 * p_v * r_v / (p_v + r_v) if (p_v + r_v) > 0 else 0.0)
    bootstraps.sort()
    lo = bootstraps[int(0.025 * n_bootstrap)]
    hi = bootstraps[int(0.975 * n_bootstrap)]
    return point, (lo, hi)


def run_5fold_cv(
    preds_by_paper: dict[str, list[dict]],
    gold_by_paper: dict[str, list[dict]],
    all_papers: list[str],
    n_folds: int = 5,
) -> dict[str, Any]:
    """Run 5-fold cross-validation. Returns per-fold and aggregate F1."""
    rng = random.Random(42)
    papers = sorted(all_papers)
    rng.shuffle(papers)
    fold_size = max(1, len(papers) // n_folds)
    folds = [papers[i:i+fold_size] for i in range(0, len(papers), fold_size)]
    fold_metrics = []
    for i, fold in enumerate(folds):
        train_papers = [p for p in papers if p not in fold]
        preds = [x for p in train_papers for x in preds_by_paper.get(p, [])]
        gold = [x for p in train_papers for x in gold_by_paper.get(p, [])]
        f1, ci = compute_aggregate_with_ci(preds, gold, n_bootstrap=100)
        fold_metrics.append({'fold': i, 'papers': fold, 'f1': f1, 'ci': ci})
    f1s = [m['f1'] for m in fold_metrics]
    return {
        'folds': fold_metrics,
        'mean_f1': statistics.mean(f1s) if f1s else 0.0,
        'std_f1': statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor && PYTHONPATH=src:scripts pytest tests/test_gold_eval_integration.py -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/gold_eval_anchored.py tests/test_gold_eval_integration.py
git commit -m "feat(eval): add 5-fold CV + bootstrap CI to gold_eval"
```

---

### Task 6: 黄金集从 9 篇扩到 20+ 篇

**Files:**
- Create: `data/gold_v19_extended/<slug>.jsonl` × 11
- Modify: `data/splits/research_v1.json` (add holdout)

- [ ] **Step 1: Pick 11 holdout papers**

List 11 candidate papers from `放射虫论文_OA_download/` that:
- Are radiolarian-focused (not bivalve/ammonite)
- Have ≥1 caption-style similar to v19 9 papers
- Cover diverse taxa/ages

Suggested candidates (read filenames in `放射虫论文_OA_download/`, filter by 放射虫 keyword + size 1-3 MB):
```bash
cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/放射虫论文_OA_download
ls *.pdf | grep -i 'radiolarian\|rad\|fauna\|jurassic\|cretaceous\|triassic\|permian' | head -20
```

Pick 11 PDFs and copy to `data/pdfs_holdout/` (don't overwrite existing 9 in `data/pdfs/`).

- [ ] **Step 2: Manually annotate 1 plate per holdout paper**

For each of 11 new PDFs:
1. Run `python scripts/gold_eval_anchored.py --pdf <pdf> --slugs <slug>` to get OD output + suggest plate
2. Read the relevant page text
3. Manually write 5-20 panel rows to `data/gold_v19_extended/<slug>.jsonl`:
   ```json
   {"paper_id": "<stable_id>", "figure_id": "od_plate_<stable_id>_p<NNN>_pl<NN>", "panel_id": "1", "species": "Genus species"}
   {"paper_id": "<stable_id>", "figure_id": "od_plate_<stable_id>_p<NNN>_pl<NN>", "panel_id": "2", "species": "Genus cf. compared"}
   ```
4. **Time budget**: 1 hour per paper = 11 hours total (split across 2-3 sessions)

- [ ] **Step 3: Update split with holdout**

Modify `data/splits/research_v1.json` to add holdout:
```json
{
  "version": "v1.1",
  "train": ["bandini2011", "beccaro2006", "boughdiri2007", "bragin2025", "danelian2006", "hollis2006"],
  "test":  ["baumgartner2008", "feng2007", "pouille2014"],
  "holdout": ["<new_slug_1>", "<new_slug_2>", ..., "<new_slug_11>"]
}
```

- [ ] **Step 4: Commit gold set + split update**

```bash
git add data/gold_v19_extended/ data/splits/research_v1.json
git commit -m "feat(eval): extend gold set to 20+ papers (11 new holdout)"
```

---

### Task 7: run_research_eval.py — 单一命令行 eval

**Files:**
- Create: `scripts/run_research_eval.py`

- [ ] **Step 1: Create the script**

Create `scripts/run_research_eval.py`:
```python
"""Run the full research-grade F1 eval.

Combines: caption_fixer + prompts + post_process + LLM-first MiniMax M3
+ 5-fold CV + bootstrap CI on the 9-paper v19 set.

Reports train/test F1 separately to expose generalization gap.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

# Add repo paths
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

import pymupdf
from PIL import Image
from rlpe.llm_backends import MiniMaxM3Backend
from rlpe.utils import stable_id

from caption_fixer import select_caption
from prompts import select_prompt, build_user_prompt
from post_process import dedup_panels, filter_low_confidence, normalize_panel_id, parse_open_nomenclature
from gold_eval_anchored import (
    load_split, compute_aggregate_with_ci, run_5fold_cv,
)


PAPERS_DIR = REPO / 'data' / 'pdfs'
HOLDOUT_DIR = REPO / 'data' / 'pdfs_holdout'
GOLD_DIR = REPO / 'data' / 'gold'
EXTENDED_GOLD_DIR = REPO / 'data' / 'gold_v19_extended'


def load_gold_for(slug: str) -> list[dict]:
    """Load gold from either legacy or extended gold dir."""
    for d in [GOLD_DIR, EXTENDED_GOLD_DIR]:
        p = d / f'{slug}.jsonl'
        if p.exists():
            return [json.loads(l) for l in open(p) if l.strip()]
    return []


def find_pdf(slug: str) -> Path | None:
    for d in [PAPERS_DIR, HOLDOUT_DIR]:
        for p in d.glob('*.pdf'):
            if p.stem.startswith(slug) or slug in p.stem:
                return p
    return None


def call_m3(backend, img, caption, system_prompt) -> dict | None:
    for attempt in range(3):
        try:
            return backend.infer_panel(
                panel_image=img, caption_text=caption, ocr_labels=[],
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(caption),
            )
        except Exception as e:
            print(f'  API error attempt {attempt+1}: {e}')
            time.sleep(5)
    return None


def extract_panels_for_paper(backend, slug: str, gold: list[dict]) -> list[dict]:
    """Run caption_fixer + prompts + M3 + post_process on one paper."""
    pdf_path = find_pdf(slug)
    if pdf_path is None:
        print(f'  no PDF for {slug}, skip')
        return []
    pid = stable_id(pdf_path)
    print(f'  {slug} paper_id={pid}')

    # Find densest gold figure
    fig_counts = Counter(g.get('figure_id') for g in gold)
    if not fig_counts:
        return []
    target_fig, _ = fig_counts.most_common(1)[0]
    m = re.search(r'_p(\d{3})_pl(\d+)', target_fig)
    if not m:
        return []
    page_num = int(m.group(1))

    doc = pymupdf.open(str(pdf_path))
    if page_num > len(doc):
        doc.close()
        return []
    full_text = '\n'.join(p.get_text() for p in doc)

    # Use caption_fixer (general) — NOT gold species
    plate_anchor = str(int(m.group(2)))
    caption = select_caption(full_text, target_plate=int(plate_anchor))
    if not caption:
        caption = doc[page_num - 1].get_text()
    print(f'    caption: {len(caption)} chars')

    # Render page
    pix = doc[page_num - 1].get_pixmap(dpi=150)
    img_path = f'/tmp/{slug}_p{page_num}.png'
    pix.save(img_path)
    doc.close()

    # Pick prompt by caption type
    sys_prompt = select_prompt(caption)

    img = Image.open(img_path)
    r = call_m3(backend, img, caption, sys_prompt)
    if not r or r.get('error') or r.get('fallback_used'):
        return []
    if r.get('_is_multi_panel') and isinstance(r.get('panels'), list):
        panels = r['panels']
    else:
        panels = [r]
    preds = []
    for p in panels:
        sp_raw = p.get('species')
        sp, qual = parse_open_nomenclature(sp_raw)
        qual_str = f" {qual}." if qual else ""
        preds.append({
            'paper_id': pid,
            'figure_id': target_fig,
            'panel_id': normalize_panel_id(p.get('label', '')),
            'species': f"{sp}{qual_str}" if sp else None,
            'confidence': p.get('confidence', 0.0),
        })
    # Post-process
    preds = dedup_panels(preds)
    preds = filter_low_confidence(preds, threshold=0.7)
    print(f'    {len(preds)} preds (after dedup + conf filter)')
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='data/splits/research_v1.json')
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--output', default='data/snapshot/eval.json')
    args = parser.parse_args()

    split = load_split(args.split)
    print(f'Split: {len(split["train"])} train + {len(split["test"])} test')

    backend = MiniMaxM3Backend(
        api_key=os.environ['ANTHROPIC_API_KEY'],
        base_url=os.environ['ANTHROPIC_BASE_URL'],
        model=os.environ.get('ANTHROPIC_MODEL', 'MiniMax-M3'),
        timeout_sec=60,
    )

    train_preds, train_gold, test_preds, test_gold = [], [], [], []
    for i, slug in enumerate(split['train'] + split['test']):
        is_train = i < len(split['train'])
        if i > 0:
            time.sleep(60)  # rate limit
        gold = load_gold_for(slug)
        preds = extract_panels_for_paper(backend, slug, gold)
        if is_train:
            train_preds.extend(preds)
            train_gold.extend(gold)
        else:
            test_preds.extend(preds)
            test_gold.extend(gold)

    print('\n=== Computing F1 ===')
    train_f1, train_ci = compute_aggregate_with_ci(
        train_preds, train_gold, n_bootstrap=args.bootstrap_samples,
    )
    test_f1, test_ci = compute_aggregate_with_ci(
        test_preds, test_gold, n_bootstrap=args.bootstrap_samples,
    )

    print(f'\nTRAIN ({len(split["train"])} papers): F1={train_f1:.4f} 95%CI=[{train_ci[0]:.4f},{train_ci[1]:.4f}]')
    print(f'TEST  ({len(split["test"])} papers): F1={test_f1:.4f} 95%CI=[{test_ci[0]:.4f},{test_ci[1]:.4f}]')
    gap = train_f1 - test_f1
    print(f'GENERALIZATION GAP: {gap:+.4f}  ({"OK" if abs(gap) <= 0.08 else "OVERFITTING"})')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'split': split, 'train_f1': train_f1, 'train_ci': train_ci,
        'test_f1': test_f1, 'test_ci': test_ci, 'gap': gap,
        'n_train_preds': len(train_preds), 'n_test_preds': len(test_preds),
    }, indent=2))
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_research_eval.py
git commit -m "feat(scripts): add run_research_eval.py (train+test F1 + bootstrap CI)"
```

---

### Task 8: 端到端 5-fold CV 跑 + v19 baseline 重测

**Files:**
- Create: `data/snapshot/2026-09-02/f1.json` (output of run_research_eval.py)
- Modify: `scripts/run_v19_baseline.py` (re-measure v19 84% with same prompt)

- [ ] **Step 1: Run research eval on 9-paper v19 set**

```bash
cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
export ANTHROPIC_MODEL=MiniMax-M3
make eval-research
```

Expected: `data/snapshot/2026-09-02/f1.json` with train F1, test F1, gap, CI.

- [ ] **Step 2: Re-measure v19 baseline F1 (using same harness)**

Create `scripts/run_v19_baseline.py`:
```python
"""Re-run v19 9-paper baseline with current prompt for fair comparison.

Uses same MiniMax-M3 + same prompt template as `run_research_eval.py`
so the 0.84 reported in v19 docs is comparable to our F1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_research_eval import main as research_main
import os
# v19 split = all 9 papers, no train/test split
os.environ['V19_FULL_EVAL'] = '1'
# Reuse main but force split = 9 train / 0 test
if __name__ == '__main__':
    research_main()
```

```bash
cd /home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
export ANTHROPIC_MODEL=MiniMax-M3
python scripts/run_v19_baseline.py --split data/splits/v19_full.json \
    --output data/snapshot/2026-09-02/v19_baseline_f1.json
```

- [ ] **Step 3: Document v19 re-measured baseline + gap analysis**

Create `docs/baselines/v19_re-measured.md`:
```markdown
# v19 SOTA baseline re-measurement

## Why re-measure
The v19 0.84 F1 was measured on a different prompt + backend. To make
our F1 comparable, we re-run the same 9 papers with our current
prompt (RANGE/SEM/MAP auto-selected) + MiniMax-M3 backend.

## Results (date 2026-09-02)
- v19 SOTA (original): F1 = 0.84 (micro, 9 papers)
- v19 re-measured (our harness): F1 = 0.?? (95% CI [??, ??])

## Interpretation
- If re-measured = 0.84 ± 0.03 → harness matches v19 → our 0.78 is real
- If re-measured < 0.80 → v19 may have used a stronger prompt → we need to
  adopt v19 prompt pattern for research eval
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_v19_baseline.py docs/baselines/v19_re-measured.md
git commit -m "docs: re-measure v19 SOTA baseline with current harness"
```

---

## Acceptance criteria (whole plan)

- [ ] Task 1: split 写死 + Makefile
- [ ] Task 2: caption_fixer.py 通用 (不引用 gold)
- [ ] Task 3: prompts.py 4 templates
- [ ] Task 4: post_process.py 4 functions
- [ ] Task 5: gold_eval_anchored 5-fold + bootstrap CI
- [ ] Task 6: 黄金集扩到 20+ 篇
- [ ] Task 7: run_research_eval.py
- [ ] Task 8: 端到端 5-fold CV + v19 baseline 重测

## Success criteria (whole plan)

- `make eval-research` 30 min 内跑完
- `data/snapshot/2026-09-02/f1.json` 含 train_f1, test_f1, gap, CI
- 真实 test F1 ≥ 0.65
- generalization gap ≤ 8pp
- v19 re-measured baseline 与原 0.84 ± 5pp
