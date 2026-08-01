"""Regression tests for audit 2026-08-01 batch W2 — scale_bar D17 multi-candidate."""

from __future__ import annotations

from rlpe.scale_bar import ScaleInfo, extract_scale_from_caption


class TestScaleBarMultiCandidate:
    """Audit 2026-08-01 batch W2 — Bug D17.

    ``extract_scale_from_caption`` previously broke out of the
    candidate loop on the FIRST ``_is_real_scale_match`` hit, then
    dropped the result entirely if the parsed value failed the sanity
    range. Captions like "Scale bar 5 cm; scale bar 100 um" returned
    an empty ScaleInfo instead of the later, more precise candidate;
    and "Scale bar 99999 mm; scale bar 50 um" (where the first
    candidate is a catastrophic OCR misread) likewise returned empty
    instead of falling through to the 50 µm bar.

    The new contract:

    * Collect every ``_is_real_scale_match`` candidate.
    * Among those, keep only the ones that survive the sanity range.
    * Pick the survivor with the largest raw ``value``; ties broken
      by later regex occurrence.
    * If no candidate survives sanity, return empty ``ScaleInfo``
      (legacy behaviour).
    """

    def test_later_um_candidate_wins(self):
        """Two real scale-bar mentions in the same caption: the later
        one (100 µm) must be preferred over the earlier "5 cm"
        because the µm reading has the larger raw value and is the
        precise microfossil scale.
        """
        info = extract_scale_from_caption("Scale bar 5 cm; scale bar 100 um")
        assert info is not None
        assert info.value == 100, f"expected value=100, got {info.value!r}"
        assert info.unit == "um", f"expected unit='um', got {info.unit!r}"
        assert info.source == "caption"

    def test_first_sanity_fail_uses_second(self):
        """First candidate is 99999 mm (= 99.999 m, vastly above the
        10 mm sanity ceiling — almost certainly an OCR misread);
        second candidate is the legitimate 50 µm. The 50 µm bar
        must win, not the catastrophic first read.
        """
        info = extract_scale_from_caption("Scale bar 99999 mm; scale bar 50 um")
        assert info is not None
        assert info.value == 50, f"expected value=50, got {info.value!r}"
        assert info.unit == "um", f"expected unit='um', got {info.unit!r}"
        assert info.source == "caption"

    def test_no_candidate_returns_empty(self):
        """No scale-bar mention at all → empty ScaleInfo (value=None)."""
        info = extract_scale_from_caption("no scale info here")
        assert isinstance(info, ScaleInfo)
        assert info.value is None
        assert info.unit is None
        assert info.source == "none"

    def test_single_candidate_unchanged(self):
        """Pre-existing single-caption path must keep working — the
        fix must not regress the simple case where only one scale-bar
        mention exists.
        """
        info = extract_scale_from_caption("Scale bar 200 µm")
        assert info is not None
        assert info.value == 200
        assert info.unit == "um"
        assert info.source == "caption"

    def test_all_candidates_fail_sanity_returns_empty(self):
        """When EVERY candidate fails sanity, the legacy behaviour
        (empty ScaleInfo) is preserved.
        """
        # 99999 mm → 99.999 m, far above the 10 mm ceiling
        info = extract_scale_from_caption("Scale bar 99999 mm; scale bar 50000 mm")
        assert isinstance(info, ScaleInfo)
        assert info.value is None
        assert info.source == "none"

    def test_specimen_size_mention_still_rejected(self):
        """Pre-existing specimen-size filter must still fire — the
        fix must not weaken the left-context guard. A caption that
        mentions a "specimen 250 µm long" but no real scale-bar
        phrasing must continue to return an empty ScaleInfo.
        """
        info = extract_scale_from_caption(
            "Photograph of specimen 250 µm long, showing diagnostic features."
        )
        assert isinstance(info, ScaleInfo)
        assert info.value is None
