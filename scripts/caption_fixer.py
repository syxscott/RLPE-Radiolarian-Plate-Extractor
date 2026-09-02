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

MIN_PARA_LEN = 0
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
    best_score = -1
    n = int(target_plate)
    anchor = re.compile(_ANCHOR_RE_TEMPLATE.format(n=n), re.IGNORECASE)
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
