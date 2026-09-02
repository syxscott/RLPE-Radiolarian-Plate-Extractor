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
    """Denylist drops 'Many species', 'Most samples', 'Each individual', etc."""
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