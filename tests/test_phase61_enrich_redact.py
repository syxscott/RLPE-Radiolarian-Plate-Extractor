"""Phase 61 Plan 4 (Bug 4.9): enrichment caption must keep the current plate's content.

Previously ``enrich_plate_panels`` sent the full ``page_caption`` to M3,
which contains captions for *other* plates on the same page. The
``api_redacted`` outbound policy then truncated the entire thing to 200
chars via ``_apply_outbound_policy``, losing the current plate's
species labels.

The fix adds ``_redact_enrichment_caption`` that:
  * identifies the section of ``page_caption`` matching
    ``current_plate_caption`` (the current plate's own caption text),
  * preserves that section in full,
  * redacts (truncates) the surrounding unrelated text.
"""

from __future__ import annotations

import pytest

from rlpe.m3_engine import _redact_enrichment_caption


def test_enrichment_redact_keeps_relevant_plate_caption():
    """The section matching the current plate must survive intact."""
    # Pad the unrelated text so it forces redaction beyond the budget.
    unrelated_padding = "x" * 2000
    page = (
        "PLATE 7: Caption text for plate 7 with species names. "
        + unrelated_padding
        + " PLATE 8: Caption for plate 8. "
        + unrelated_padding
        + " PLATE 9: another plate's caption text."
    )
    current = "Caption text for plate 7 with species names."
    out = _redact_enrichment_caption(page, current)
    # The current plate's caption must appear in full.
    assert "Caption text for plate 7 with species names." in out
    # Distant other plates' captions must be redacted away.
    assert "another plate's caption text." not in out
    # Redacted size is well below the original page size.
    assert len(out) < len(page)


def test_enrichment_redact_no_match_redacts_all():
    """When current plate caption is not found, default to safe redaction."""
    page = "PLATE 1: caption one. PLATE 2: caption two."
    current = "PLATE 7: this caption is not on the page"
    out = _redact_enrichment_caption(page, current)
    # Without a match we can't safely preserve anything; the helper
    # returns a 200-char truncation of the whole page (no current
    # match to highlight), so the output is at most 200 chars.
    assert len(out) <= 200


def test_enrichment_redact_empty_inputs():
    """Empty inputs must not crash."""
    # Empty page → empty output.
    assert _redact_enrichment_caption("", "current") == ""
    # Both empty → empty output.
    assert _redact_enrichment_caption("", "") == ""
    # Empty current caption with a page → safe-truncate to budget.
    out = _redact_enrichment_caption("page", "")
    assert out == "page"


def test_enrichment_redact_truncates_long_unrelated():
    """Long unrelated sections are truncated to a small budget."""
    long_unrelated = "x" * 5000
    page = "PLATE 1: small caption for plate 1. " + long_unrelated
    current = "small caption for plate 1."
    out = _redact_enrichment_caption(page, current)
    assert "small caption for plate 1." in out
    assert len(out) < len(page)
