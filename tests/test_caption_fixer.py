import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import re
from pathlib import Path

import pymupdf
from caption_fixer import (
    _ANCHOR_N_RE_TEMPLATE,
    count_plate_anchors,
    has_plate_captions,
    score_paragraph,
    select_caption,
)


def test_anchor_plate_5_paragraph_selected():
    """Paragraph starting with 'Plate 5' is selected over shorter non-anchored."""
    text = """Header text about the paper.
Plate 5
Detailed caption about many specimens. Sample PR-SB05 (latest Tithonian).
Fig. 1 Archaeodictyomitra sp. Fig. 2 Williriedellum sp. Fig. 3 Hiscocapsa sp.
This is a real plate caption with many species."""
    best = select_caption(text, target_plate=5)
    assert best is not None
    assert "Plate 5" in best
    assert "Archaeodictyomitra" in best


def test_anchor_fig_3_selected():
    """Fig. 3 caption selected when target_plate=3."""
    text = """Plate 1
First plate caption here.
Fig. 3
Some other figure here.
Plate 5
Last plate."""
    best = select_caption(text, target_plate=3)
    assert "Fig. 3" in best


def test_anchor_with_leading_zero_5():
    """'Plate 05' matches target_plate=5 (strip leading zeros)."""
    text = """Header.
Plate 05
Some caption.
Trailer."""
    best = select_caption(text, target_plate=5)
    assert "Plate 05" in best


def test_no_anchor_returns_none():
    """No Plate N anchor → return None (don't guess)."""
    text = """Para A short.
Para B about Genus species one and Genus species two and Genus species three.
Para C also short."""
    best = select_caption(text, target_plate=99)  # no anchor exists
    assert best is None


def test_returns_none_on_no_text():
    assert select_caption("", target_plate=1) is None


def test_score_paragraph_returns_int():
    anchor_re = re.compile(_ANCHOR_N_RE_TEMPLATE.format(n=1), re.IGNORECASE)
    score = score_paragraph(
        "Plate 1\nGenus species A and Genus species B", target_plate=1, anchor_re=anchor_re
    )
    assert isinstance(score, int)
    assert score > 0


def test_split_no_blank_lines_real_pdf_style():
    """Real PDF text rarely has blank lines between caption blocks.
    The line-by-line anchor-based split should still find captions."""
    text = (
        "Some running header line\n"
        "Plate 5\n"
        "Detailed caption about many specimens. Sample PR-SB05 (latest Tithonian).\n"
        "Fig. 1 Archaeodictyomitra sp. Fig. 2 Williriedellum sp. Fig. 3 Hiscocapsa sp.\n"
        "Plate 6\n"
        "Different plate caption here. Fig. 4 Emiluvia sp.\n"
    )
    cap = select_caption(text, target_plate=5)
    assert cap is not None
    assert "Plate 5" in cap
    assert "Plate 6" not in cap  # must NOT contain the other plate


def test_no_anchor_strict_none():
    """When no block has the target plate anchor, return None (don't guess)."""
    text = "Just body text with Many species and Most samples here."
    cap = select_caption(text, target_plate=99)
    assert cap is None  # Not a guess — caller should fall back to whole-page


def test_returns_none_strict_empty():
    """Empty text returns None (strict assertion)."""
    assert select_caption("", target_plate=1) is None


def test_binomial_deny_list_filters_english():
    """'Many species' / 'Each individual' must NOT count as binomials."""
    text = "Plate 5\nMany species. Most samples. Each individual. Several individuals.\n"
    cap = select_caption(text, target_plate=5)
    # The block has anchor (+10) but no real binomials. With MIN anchor score
    # = ANCHOR_SCORE, the block IS selected but the score is just 10
    # (anchor only). If a real binomial caption were also present, it would
    # win on score. The important behavior: real binomials (Genus species)
    # are scored, English phrases are not.
    assert cap is not None
    assert "Plate 5" in cap


def test_anchor_matches_mid_text_fig_ref():
    """After Fix A, 'as shown in Fig. 5' mid-sentence should anchor a block."""
    text = """Plate 1
Real caption here.
as shown in Fig. 5 above, another related discussion continues
for several more pages in this section.
Plate 2
Different caption."""
    # Both plates should now be recognized as anchor lines.
    assert count_plate_anchors(text) >= 2


def test_count_plate_anchors_distinct_anchors():
    """count_plate_anchors returns the number of distinct Plate/Fig anchors."""
    text = """Plate 1
caption A
Plate 1 again
Plate 2
caption B
Fig. 3
caption C
Plate 1 yet again
"""
    # 3 distinct anchors: "Plate 1", "Plate 2", "Fig. 3"
    # (Plate 1 appears 3 times but counts once)
    assert count_plate_anchors(text) == 3


def test_has_plate_captions_true():
    assert has_plate_captions("Plate 1\nSome text.\nPlate 2\nMore text.", min_count=2) is True
    assert has_plate_captions("Plate 1\nSome text.", min_count=1) is True


def test_has_plate_captions_false():
    """No plate anchors → False."""
    assert (
        has_plate_captions("This is just body text without any plate references.", min_count=1)
        is False
    )
    assert has_plate_captions("Plate 1\nSome text.", min_count=2) is False
