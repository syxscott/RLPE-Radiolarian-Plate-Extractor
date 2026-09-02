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
