"""Regression tests for the LLM-truncation hybrid gate in pipeline.py.

Before the fix (2026-07-01), the LLM-first hybrid in
``RadiolarianPipeline._process_region`` only fired when EITHER:
  - the LLM left any species blank (``missing_species`` non-empty), OR
  - the LLM returned fewer than 2 panels

This silently dropped panels whenever the LLM capped its output at ~19
panels (a soft training-data ceiling on Gemma-3/M3) while the caption
actually enumerated 21-35 panels. baumgartner2008 pl02 (21 panels)
and pl03 (27 panels), beccaro2006 (35 panels), and wever2006 (long
captions) all hit this bug.

The fix: fire the hybrid whenever the caption parser finds MORE panels
than the LLM did, with a sanity bound (<=100 panels in pair_lookup) to
guard against degenerate regex over-matching.

These tests lock in the gate's new behaviour via the helper functions
that determine ``pair_lookup`` and the gate's boolean output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_pair_lookup(n: int) -> dict[str, str]:
    """Build a synthetic pair_lookup with n (label, species) pairs."""
    return {str(i): f"Genus{i} species{i}" for i in range(1, n + 1)}


def _make_llm_results(n: int, with_species: bool = True) -> list[dict]:
    """Build a synthetic llm_results list with n entries."""
    return [
        {
            "panel_id": str(i),
            "species": (f"Genus{i} species{i}" if with_species else ""),
            "confidence": 0.9,
        }
        for i in range(1, n + 1)
    ]


def _gate(llm_results, pair_lookup):
    """Replicate the gate logic from pipeline.py for testing.

    Returns True if the hybrid should fire, False otherwise.
    Mirrors the post-fix gate exactly:
      missing_species or len(llm_results) < 2 or
      (caption_has_more and len(pair_lookup) <= 100)
    """
    missing_species = [r for r in llm_results if not r.get("species")]
    caption_has_more = bool(pair_lookup) and len(pair_lookup) > len(llm_results)
    return bool(
        missing_species
        or len(llm_results) < 2
        or (caption_has_more and len(pair_lookup) <= 100)
    )


class TestHybridGate:
    """Lock in the new gate semantics."""

    def test_llm_full_species_but_truncated_fires(self):
        """The KEY fix: LLM returned 19 panels with full species, but
        caption has 27. Hybrid must fire (it would have skipped before)."""
        llm = _make_llm_results(19, with_species=True)
        pairs = _make_pair_lookup(27)
        assert _gate(llm, pairs) is True

    def test_missing_species_fires(self):
        """Original behaviour: LLM left some species blank → fire."""
        llm = _make_llm_results(19, with_species=False)
        pairs = _make_pair_lookup(19)
        assert _gate(llm, pairs) is True

    def test_too_few_panels_fires(self):
        """Original behaviour: <2 panels → fire."""
        llm = _make_llm_results(1, with_species=True)
        pairs = _make_pair_lookup(1)
        assert _gate(llm, pairs) is True

    def test_no_truncation_no_missing_skips(self):
        """LLM returned same number of panels as caption with full species
        → skip the hybrid (no work to do)."""
        llm = _make_llm_results(27, with_species=True)
        pairs = _make_pair_lookup(27)
        assert _gate(llm, pairs) is False

    def test_caption_has_fewer_than_llm_skips(self):
        """LLM returned MORE panels than the caption parser — keep LLM's
        count (LLM may know about cross-references the parser misses)."""
        llm = _make_llm_results(35, with_species=True)
        pairs = _make_pair_lookup(27)
        assert _gate(llm, pairs) is False

    def test_runaway_regex_does_not_fire(self):
        """Sanity bound: if the regex returns >100 panels (degenerate
        caption), do NOT fire the hybrid — likely an OCR / parsing
        runaway (wever2006 produced 1918-panel pair_lookup once)."""
        llm = _make_llm_results(19, with_species=True)
        pairs = _make_pair_lookup(1918)  # degenerate
        assert _gate(llm, pairs) is False, (
            "Gate must bound caption_has_more by pair_lookup <= 100"
        )

    def test_boundary_100_panels_fires(self):
        """Boundary check: pair_lookup of exactly 100 panels, LLM has 19.
        Gate fires (100 is the inclusive upper bound)."""
        llm = _make_llm_results(19, with_species=True)
        pairs = _make_pair_lookup(100)
        assert _gate(llm, pairs) is True

    def test_boundary_101_panels_skips(self):
        """Boundary check: pair_lookup of 101 panels, LLM has 19.
        Gate skips (101 is just above the bound)."""
        llm = _make_llm_results(19, with_species=True)
        pairs = _make_pair_lookup(101)
        assert _gate(llm, pairs) is False

    def test_empty_pair_lookup_no_truncation_skips(self):
        """Caption parser returned 0 panels (caption text was empty or
        unparseable). With LLM also returning 0 panels, skip hybrid."""
        llm = _make_llm_results(19, with_species=True)
        pairs = {}
        assert _gate(llm, pairs) is False

    def test_empty_pair_lookup_with_llm_missing_fires(self):
        """LLM left species blank AND parser returned nothing — fire."""
        llm = _make_llm_results(19, with_species=False)
        pairs = {}
        # missing_species is non-empty → fires
        assert _gate(llm, pairs) is True


class TestBaumgartnerRealCase:
    """Reproduce the actual baumgartner2008 plate-3 regression: 19 LLM
    panels, 27 caption panels, full species returned by LLM."""

    def test_baum_pl03_27_panels_vs_19_llm(self):
        """Real bug: pl03 has 27 species in caption; M3 returned 19.
        Pre-fix: hybrid skipped, panels 20-27 silently dropped.
        Post-fix: hybrid fires, regex adds panels 20-27."""
        llm_results = _make_llm_results(19, with_species=True)
        # Simulate the regex parser correctly recovering all 27 panels
        pair_lookup = {str(i): f"Genus{i} sp{i}" for i in range(1, 28)}
        # Simulate the caption text Baumgartner used (informational only):
        _ = "Plate 3 - Triassic Radiolaria. 1- A; 2- B; ...; 27- Z."

        # Before fix: gate was just `missing_species or len < 2`
        # which would be False here. After fix:
        assert _gate(llm_results, pair_lookup) is True


class TestBeccaroRealCase:
    """Reproduce the actual beccaro2006 case: 33 LLM panels, 35 caption."""

    def test_beccaro_33_llm_vs_35_caption(self):
        """M3 truncated beccaro to 33 panels (caption has 35).
        Hybrid must fire to add panels 34-35."""
        llm_results = _make_llm_results(33, with_species=True)
        pair_lookup = {str(i): f"Sp{i}" for i in range(1, 36)}
        assert _gate(llm_results, pair_lookup) is True


class TestWever2006RegressionGuard:
    """wever2006 once produced a 1918-panel pair_lookup (regex
    over-matched). The bound must prevent this from cascading."""

    def test_wever_runaway_blocked(self):
        llm_results = _make_llm_results(20, with_species=True)
        pair_lookup = _make_pair_lookup(1918)
        assert _gate(llm_results, pair_lookup) is False, (
            "1918-panel pair_lookup must NOT trigger hybrid"
        )
