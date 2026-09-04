import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from text_extract import extract_species_from_text

_PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"


def _path(slug: str) -> Path:
    hits = list(_PDF_DIR.glob(f"{slug}*"))
    if not hits:
        # Corpus PDFs are not committed to the repo — CI checkouts
        # (and any fresh clone without ``data/pdfs`` populated) must
        # skip rather than fail (audit 2026-09-04 CI regression).
        pytest.skip(f"{slug}* not present in data/pdfs (corpus not checked out)")
    return hits[0]


def test_extract_finds_binomials():
    rows = extract_species_from_text(_path("bandini2011"))
    assert len(rows) > 0
    for r in rows:
        assert "paper_id" in r
        assert "species" in r
        assert "page_num" in r
        assert r["extraction_method"] == "text_regex"
        assert isinstance(r["page_num"], int)
        assert r["page_num"] >= 1


def test_extract_filters_english_phrases():
    """Denylist drops 'Many species', 'Most samples', 'Each individual', etc."""
    rows = extract_species_from_text(_path("bandini2011"))
    species = {r["species"] for r in rows}
    for forbidden in ["Many species", "Most samples", "Each individual"]:
        assert forbidden not in species


def test_extract_includes_location():
    """Each row has page_num and char_offset for traceability."""
    rows = extract_species_from_text(_path("bandini2011"))
    for r in rows:
        assert r["page_num"] >= 1
        assert r["char_offset"] >= 0
        assert r["context_50char"]  # non-empty string
        assert isinstance(r["context_50char"], str)


def test_extract_dedups_same_species_same_page():
    """Same normalized species on same page appears only once."""
    rows = extract_species_from_text(_path("bandini2011"))
    from collections import Counter

    by_key = Counter((r["paper_id"], r["normalized_species"], r["page_num"]) for r in rows)
    max_count = max(by_key.values())
    assert max_count <= 1, f"found duplicate (paper, sp, page) keys: {max_count}"


# Note: the previous ``test_extract_uses_known_denylist`` drift detection
# test was deleted in Task 1 (2026-09-02). Both consumers now import the
# constants from ``binomial_utils`` — so the constants can never drift
# between the two modules by construction.


def test_extract_nonexistent_file_raises():
    """Nonexistent PDF path raises a FileNotFoundError-derived error.

    pymupdf raises its own ``pymupdf.FileNotFoundError`` (a subclass of
    ``RuntimeError``) when the path doesn't exist. We accept any of the
    three plausible exception classes so the test is robust against the
    underlying library changing its exception hierarchy.
    """
    import pymupdf
    import pytest

    with pytest.raises((FileNotFoundError, OSError, pymupdf.FileNotFoundError)):
        extract_species_from_text("/nonexistent/path/to/missing.pdf")


def test_extract_handles_hyphenated_binomials():
    """'Williriedellum carpathicum-forma' yields 'Williriedellum carpathicum'
    (regex splits on the space, not the hyphen — the '-forma' suffix is
    dropped because it doesn't match the 'lowercase 3+ char' species word)."""
    import os
    import tempfile

    pdf_path = _make_pdf_with_text("Williriedellum carpathicum-forma is a subspecies.")
    try:
        rows = extract_species_from_text(pdf_path, paper_id="test")
        species = {r["species"] for r in rows}
        assert "Williriedellum carpathicum" in species
        # The full hyphenated form does NOT match as a single binomial
        # (the regex needs whitespace-separated words).
        assert "Williriedellum carpathicum-forma" not in species
    finally:
        os.unlink(pdf_path)


def test_extract_handles_unicode_no_match():
    """Non-Latin script (Chinese in this case) does not match the ASCII
    binomial regex — only the Latin 'Williriedellum carpathicum' binomial
    is returned, while the CJK phrase surrounding it is dropped because
    pymupdf's default Helvetica font has no CJK glyphs (renders as ·)
    and the regex is ASCII-only anyway.

    Note: 'species' is in the denylist, so we use a real binomial term
    ('Williriedellum carpathicum') instead.
    """
    import os

    pdf_path = _make_pdf_with_text("放射虫 Williriedellum carpathicum 和 Many samples 都是化石")
    try:
        rows = extract_species_from_text(pdf_path, paper_id="test")
        species = {r["species"] for r in rows}
        # Latin binomial survives the regex + denylist.
        assert "Williriedellum carpathicum" in species
        # The ASCII denylist catches 'Many samples'.
        assert "Many samples" not in species
        # Exactly one binomial comes through (no CJK matches).
        assert len(species) == 1
    finally:
        os.unlink(pdf_path)


def _make_pdf_with_text(body: str) -> str:
    """Build a single-page PDF containing the given text on it."""
    import tempfile

    import pymupdf

    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    import os as _os

    _os.close(fd)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), body)
    doc.save(pdf_path)
    doc.close()
    return pdf_path
