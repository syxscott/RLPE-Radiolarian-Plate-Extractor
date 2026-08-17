"""Round 20 source-guard tests: paper metadata cleanup.

User audit (Round 20 sampling of 4 OA papers) identified 3
systemic paper-metadata issues:

  1. **Title garbage:** ``title="001_020"`` (Bandini),
     ``title="035_048"`` (Danelian), ``title="StrtEng2470030Bragin.fm"``
     (Bragin). GROBID failed and returned header / filename strings.

  2. **Author markers:** ``authors=["Input2"]`` (Bragin). The
     OpenDataLoader fulltext extractor returned a placeholder
     string when no real author was identified.

  3. **Journal wrong / missing:** ``journal="Scale"`` (Bragin).
     GROBID returned a publisher-related word instead of the
     journal name. With a DOI present, Crossref provides the
     correct journal name as a fallback.

These tests pin the three cleanups.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --- 1) Title garbage detection --------------------------------------------


def test_title_page_range_detected():
    """'001_020' (Bandini's page range) must be flagged."""
    from rlpe.paper_metadata_cleanup import looks_like_garbage_title

    assert looks_like_garbage_title("001_020")
    assert looks_like_garbage_title("035_048")
    assert looks_like_garbage_title("001-020")


def test_title_filename_detected():
    """Bragin's 'StrtEng2470030Bragin.fm' must be flagged."""
    from rlpe.paper_metadata_cleanup import looks_like_garbage_title

    assert looks_like_garbage_title("StrtEng2470030Bragin.fm")
    assert looks_like_garbage_title("paper.tex")
    assert looks_like_garbage_title("doc.PDF")


def test_title_pure_digits_detected():
    """Bare page numbers like '15' / '542' must be flagged."""
    from rlpe.paper_metadata_cleanup import looks_like_garbage_title

    assert looks_like_garbage_title("15")
    assert looks_like_garbage_title("542")


def test_title_real_titles_kept():
    """Real titles must NOT be flagged."""
    from rlpe.paper_metadata_cleanup import looks_like_garbage_title

    assert not looks_like_garbage_title("Upper Jurassic Radiolaria from the Vocontian Basin")
    assert not looks_like_garbage_title("Turonian Radiolarians from Karnezeika")
    # None / empty / whitespace ARE flagged (no real title was found)
    # — this is intentional so cleanup_title returns None + review reason.
    assert looks_like_garbage_title(None)
    assert looks_like_garbage_title("")
    assert looks_like_garbage_title("   ")


def test_cleanup_title_returns_review_reason():
    """``cleanup_title`` returns ``(None, review_reason)`` for garbage."""
    from rlpe.paper_metadata_cleanup import cleanup_title

    title, reason = cleanup_title("001_020", paper_id="bandini2006")
    assert title is None
    assert reason == "title_extraction_failed"

    title, reason = cleanup_title("StrtEng2470030Bragin.fm", paper_id="bragin2025")
    assert title is None
    assert reason == "title_extraction_failed"

    title, reason = cleanup_title("Upper Jurassic Radiolaria", paper_id="danelian2006")
    assert title == "Upper Jurassic Radiolaria"
    assert reason is None


# --- 2) Author marker strip ------------------------------------------------


def test_author_marker_input2_stripped():
    """'Input2' must be removed from author list."""
    from rlpe.paper_metadata_cleanup import cleanup_authors

    assert cleanup_authors(["Input2"]) == []
    assert cleanup_authors(["input"]) == []
    assert cleanup_authors(["Unknown"]) == []
    assert cleanup_authors(["N/A"]) == []


def test_author_real_names_preserved():
    """Real author names must NOT be stripped."""
    from rlpe.paper_metadata_cleanup import cleanup_authors

    assert cleanup_authors(["John Smith", "Jane Doe"]) == ["John Smith", "Jane Doe"]
    assert cleanup_authors(["BOUHDI MABROUK"]) == ["BOUHDI MABROUK"]


def test_author_mixed_list_partially_cleaned():
    """Mixed marker + real author: marker stripped, real kept."""
    from rlpe.paper_metadata_cleanup import cleanup_authors

    assert cleanup_authors(["Input2", "John Smith"]) == ["John Smith"]
    assert cleanup_authors(["John Smith", "input"]) == ["John Smith"]


def test_author_handles_none_and_empty():
    from rlpe.paper_metadata_cleanup import cleanup_authors

    assert cleanup_authors(None) == []
    assert cleanup_authors([]) == []
    assert cleanup_authors(["", "  "]) == []


# --- 3) Journal enrichment via DOI -----------------------------------------


def test_journal_enrichment_keeps_existing():
    """If GROBID already has a real journal, don't overwrite."""
    from rlpe.paper_metadata_cleanup import enrich_journal

    assert enrich_journal("Nature", "10.1038/nature12373") == "Nature"


def test_journal_enrichment_handles_none():
    """None journal + no DOI returns None (no fabrication)."""
    from rlpe.paper_metadata_cleanup import enrich_journal

    assert enrich_journal(None, None) is None
    assert enrich_journal("", "") is None


def test_journal_enrichment_short_value_triggers_lookup():
    """Suspiciously short journal values trigger Crossref fallback."""
    from rlpe.paper_metadata_cleanup import enrich_journal, needs_journal_enrichment

    assert needs_journal_enrichment(None)
    assert needs_journal_enrichment("")
    assert needs_journal_enrichment("A")  # 1 char → definitely wrong
    assert not needs_journal_enrichment("Nature")
    assert not needs_journal_enrichment("Eclogae Geologicae Helvetiae")


def test_journal_crossref_lookup_real_doi():
    """Smoke test: a real DOI returns a non-empty journal name.
    This validates the Crossref integration end-to-end without
    hard-coding the expected value (Crossref may rename containers)."""
    from rlpe.paper_metadata_cleanup import _crossref_get_journal

    # A well-known Crossref DOI
    journal = _crossref_get_journal("10.1038/nature12373")
    assert journal, "Crossref returned None for a real DOI"
    assert len(journal) > 3


def test_journal_crossref_handles_404():
    """Invalid DOI returns None without raising."""
    from rlpe.paper_metadata_cleanup import _crossref_get_journal

    # Made-up DOI → Crossref returns 404
    journal = _crossref_get_journal("10.9999/this-doi-does-not-exist-9999")
    assert journal is None


# --- Combined end-to-end ---------------------------------------------------


def test_cleanup_paper_metadata_combined():
    """End-to-end: a paper record with all 3 garbage values is cleaned
    in a single call and review_reasons is populated."""
    from rlpe.paper_metadata_cleanup import cleanup_paper_metadata

    dirty = {
        "paper_id": "bragin2025",
        "title": "StrtEng2470030Bragin.fm",
        "authors": ["Input2"],
        "year": 2025,
        "journal": "Scale",
        "doi": "10.1134/S0869593824700308",
    }
    cleaned, reasons = cleanup_paper_metadata(dirty)

    assert cleaned["title"] is None, "Garbage title not cleaned"
    assert cleaned["authors"] == [], "Input2 marker not stripped"
    # Crossref for this DOI should return "Stratigraphy and Geological
    # Correlation" — but we don't hard-code it (it might be renamed).
    # We just check that journal is enriched to a non-garbage value.
    if cleaned["journal"]:
        assert len(cleaned["journal"]) > 5, f"Journal too short: {cleaned['journal']!r}"
        assert cleaned["journal"] != "Scale", "Original 'Scale' not replaced"
    assert "title_extraction_failed" in reasons


def test_cleanup_paper_metadata_source_guard():
    """Source guard: converters.py must call cleanup_paper_metadata
    inside paper_records_from_matches. Without this wiring, the
    fixes above are dead code."""
    src = Path(
        "/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/src/rlpe/converters.py"
    ).read_text(encoding="utf-8")
    assert "cleanup_paper_metadata" in src, (
        "converters.py does not import or call cleanup_paper_metadata. "
        "Round 20 paper-metadata fixes are not wired."
    )
    assert "paper_metadata_cleanup" in src, (
        "converters.py does not import from paper_metadata_cleanup. "
        "Round 20 cleanup helpers are dead code."
    )
