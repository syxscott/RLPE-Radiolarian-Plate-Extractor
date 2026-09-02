"""Pure-Python text-level radiolarian species extractor.

Scans the full PDF text for binomial 'Genus species' patterns. No
LLM call, no gold reference — generic heuristic only. Used as a
fallback / supplement to M3 plate-mode extraction.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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
) -> list[dict[str, Any]]:
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
    out: list[dict[str, Any]] = []
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