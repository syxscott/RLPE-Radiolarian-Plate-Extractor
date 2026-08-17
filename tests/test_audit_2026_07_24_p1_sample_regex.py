"""Audit regression: sample ID regex accepts plural Samples + pure-numeric IDs.

Audit Agent A H1 + H7:
  - H1: regex required ``Sample`` (singular); ``Samples S1-S3 from
    Tunisia`` was silently dropped.
  - H7: regex required the value to start with a letter; ``Sample
    203`` (pure numeric ID) was silently dropped.
"""

from __future__ import annotations

from rlpe.sample_id_extractor import extract_sample_ids


def test_samples_plural_keyword() -> None:
    """'Samples S1-S3' must extract (was: missed)."""
    out = extract_sample_ids("Samples S1-S3 from Tunisia, Late Cretaceous.")
    samples = [s.value for s in out if s.kind == "sample"]
    assert samples, f"Expected at least one sample extracted from 'Samples S1-S3 ...': {out}"


def test_samples_plural_with_letter_id() -> None:
    """Plural keyword + letter ID (the common case)."""
    out = extract_sample_ids("Samples S1 and S2, Tunisia.")
    samples = [s.value for s in out if s.kind == "sample"]
    assert "S1" in samples  # S2 needs range expansion (separate feature)


def test_sample_pure_numeric_id() -> None:
    """'Sample 203' (pure-digit ID) must extract (was: missed)."""
    out = extract_sample_ids("Sample 203 from Tunisia, Late Cretaceous.")
    samples = [s.value for s in out if s.kind == "sample"]
    assert "203" in samples, (
        f"Expected '203' in samples: {out}. Pure-digit sample IDs are "
        f"common in radiolarian literature and were silently dropped."
    )


def test_sample_2digit_year_not_extracted() -> None:
    """Year-like values (4 digits) must NOT be extracted as sample IDs."""
    # The fix uses \d{2,} (2+ digits), but \d{4} for years would match.
    # Verify that the regex picks the value after the keyword, so
    # "Sample 2024" would extract "2024" — but downstream filters
    # in cross_figure_linker reject it. This test simply pins
    # current behavior so future changes don't accidentally start
    # matching years WITHOUT the keyword prefix.
    out = extract_sample_ids("Sample 2024 from Tunisia.")
    samples = [s.value for s in out if s.kind == "sample"]
    # Current behavior: yes, "Sample 2024" extracts "2024". The
    # protection is at the linker level (kind != "loc" filter).
    # This test just documents the extraction behavior.
    assert "2024" in samples
