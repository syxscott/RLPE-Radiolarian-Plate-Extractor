"""General caption block selector.

Picks the most likely plate caption from a PDF page text WITHOUT
referencing the gold set (to prevent over-fitting on train papers).
The selector splits text into per-line blocks (anchored by Plate/Fig
prefixes), scores each block by structural anchor + binomial density
+ plate terminator markers, and picks the highest-scored block
whose anchor matches `target_plate`.
"""
from __future__ import annotations

import re
from typing import Optional, Pattern

# Shared binomial pattern + denylist (single source of truth, prevents
# drift vs. text_extract).
from binomial_utils import _BINOMIAL_RE, _BINOMIAL_DENY

# Scoring weights
ANCHOR_SCORE = 10
DENSE_BINOMIAL_SCORE = 5
SPARSE_BINOMIAL_SCORE = 2
TERMINATOR_SCORE = 1
MIN_PARA_LEN = 0
MAX_PARA_LEN = 20000

# Anchor pattern for BLOCK SPLITTING: only Plate-level markers start a
# new block. "Fig. N" inside a caption is just a figure reference, not a
# new caption. Allow optional leading zero: "Plate 05" matches target_plate=5.
# Use re.MULTILINE so ^ matches start of any line (mid-text references
# like "as shown in Fig. 5 above" can also serve as anchors).
_ANCHOR_LINE_RE = re.compile(
    r"^\s*(?:Plate|Pl|表|図版)\.?\s*0?\d+\b",
    re.IGNORECASE | re.MULTILINE,
)
# Anchor pattern: any line containing Plate N / Fig N (for the gold lookup).
_ANCHOR_N_RE_TEMPLATE = (
    r"^\s*(?:Plate|Pl|Fig|表|図版)\.?\s*"
    r"0?{n}\b"
)
# Wider anchor pattern used by pre-screen helpers below; matches Plate /
# Fig / Pl. / Japanese 表・図版 at the start of any line. Mirrors the
# flag set on _ANCHOR_LINE_RE above (IGNORECASE | MULTILINE).
_PLATE_ANCHOR_RE = re.compile(
    r"^\s*(?:Figs?|Plate|Plates|Pl|表|図版)\.?\s*0?\d+\b",
    re.IGNORECASE | re.MULTILINE,
)

# Plate terminator markers (typical end of a real caption).
_TERMINATORS = ("Sample", "Loc.", "Marker =", "Scale", "Bar =")


def _is_real_binomial(span: str) -> bool:
    parts = span.split()
    if len(parts) != 2:
        return False
    return parts[1].lower() not in _BINOMIAL_DENY


def score_paragraph(para: str, target_plate: int, anchor_re: Pattern) -> int:
    """Score a single paragraph for being the target plate caption.

    Higher = more likely to be the caption. The `anchor_re` is a
    pre-compiled regex matching the target plate number — pass it in
    so we don't recompile per paragraph. Uses search() so that anchor
    matches anywhere in the block (e.g. "Fig. 3" inside a "Plate 1"
    block still scores for target_plate=3).
    """
    score = 0
    n = int(target_plate)
    if anchor_re.search(para):
        score += ANCHOR_SCORE
    real_binomials = [m for m in _BINOMIAL_RE.findall(para) if _is_real_binomial(m)]
    if len(real_binomials) >= 2:
        score += DENSE_BINOMIAL_SCORE
    elif len(real_binomials) >= 1:
        score += SPARSE_BINOMIAL_SCORE
    for term in _TERMINATORS:
        if term in para:
            score += TERMINATOR_SCORE
    return score


def _split_into_caption_blocks(text: str) -> list[str]:
    """Split text into per-caption blocks using anchor lines as delimiters.

    PDF text rarely has blank lines between captions, so we group
    consecutive non-anchor lines and start a new block at each anchor.
    """
    lines = text.split('\n')
    blocks: list[str] = []
    cur: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cur:
                blocks.append('\n'.join(cur))
                cur = []
            continue
        if _ANCHOR_LINE_RE.match(stripped):
            if cur:
                blocks.append('\n'.join(cur))
            cur = [stripped]
        else:
            cur.append(stripped)
    if cur:
        blocks.append('\n'.join(cur))
    return blocks


def select_caption(
    text: str,
    target_plate: int,
    min_anchor_score: int = ANCHOR_SCORE,
) -> Optional[str]:
    """Pick the best caption paragraph from `text` for `target_plate`.

    Returns None if no caption block matches the target plate. This is
    safer than guessing — downstream code can fall back to whole-page
    text rather than silently accept a wrong block.
    """
    if not text or not text.strip():
        return None
    blocks = _split_into_caption_blocks(text)
    if not blocks:
        return None
    n = int(target_plate)
    anchor_re = re.compile(_ANCHOR_N_RE_TEMPLATE.format(n=n), re.IGNORECASE | re.MULTILINE)
    # First pass: prefer blocks anchored to target_plate
    best: Optional[str] = None
    best_score = 0
    for block in blocks:
        if len(block) > MAX_PARA_LEN:
            continue
        score = score_paragraph(block, n, anchor_re)
        if anchor_re.search(block) and score >= min_anchor_score:
            if score > best_score:
                best = block
                best_score = score
    return best


def count_plate_anchors(text: str) -> int:
    """Return the number of distinct Plate/Fig anchors in `text`.

    Useful for pre-screening PDFs to skip those with no plate-style
    captions (e.g. editorials, short research notes, range charts that
    don't use Plate N naming).
    """
    return len(set(m.group(0).strip() for m in _PLATE_ANCHOR_RE.finditer(text)))


def has_plate_captions(text: str, min_count: int = 1) -> bool:
    """Return True if `text` contains at least `min_count` plate-style anchors."""
    return count_plate_anchors(text) >= min_count