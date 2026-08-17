"""Audit regression: _LOC_RE must be case-insensitive.

Audit Agent A C4: OCR often emits ``loc. tunisia`` (lower keyword
plus lower locality) or all-caps headers like ``LOCALITY: TUNISIA``.
The original regex required ``[A-Z]`` first-letter and no
IGNORECASE flag, silently dropping these inputs.

Note: ``_LOC_RE`` is consumed by :func:`extract_sample_ids`
(returns ``SampleID(kind="loc", value=...)``). The standalone
:func:`extract_locality` deliberately ignores Loc. tokens and
uses :data:`_LOCALITY_PHRASE_RE` for the ``from <Name>`` style.
Both regexes need IGNORECASE for the audit fix to take effect
across both code paths.
"""

from __future__ import annotations

from rlpe.sample_id_extractor import extract_locality, extract_sample_ids


def test_loc_kw_lowercase_via_sample_ids() -> None:
    """Lowercase 'loc.' + lowercase locality should yield SampleID(kind='loc')."""
    out = extract_sample_ids("All specimens from loc. tunisia, late cretaceous.")
    locs = [s.value for s in out if s.kind == "loc"]
    assert "tunisia" in locs, (
        f"Expected 'tunisia' in loc-extracts: {locs}. OCR frequently "
        f"emits lowercase locality text after lowercase keyword."
    )


def test_loc_kw_uppercase_all_caps_via_sample_ids() -> None:
    """All-caps header should still extract (case-insensitive)."""
    out = extract_sample_ids("LOCALITY: TUNISIA, sample S1")
    locs = [s.value for s in out if s.kind == "loc"]
    assert "TUNISIA" in locs, (
        f"Expected 'TUNISIA' (preserved case) in loc-extracts: {locs}. "
        f"Section headers in uppercase must still be recognized."
    )


def test_loc_kw_mixed_case_via_sample_ids() -> None:
    """Mixed case should extract and preserve source case."""
    out = extract_sample_ids("Loc. Tunisia, sample S1; Loc. Greece, sample S2")
    locs = [s.value for s in out if s.kind == "loc"]
    assert "Tunisia" in locs
    assert "Greece" in locs


def test_locality_phrase_lowercase_via_extract_locality() -> None:
    """The standalone 'from <Name>' extractor must also be case-insensitive."""
    out = extract_locality("all specimens from tunisia, late cretaceous.")
    assert any("tunisia" in v.casefold() for v in out), (
        f"Lowercase 'from tunisia' should match _LOCALITY_PHRASE_RE: {out}"
    )


def test_negative_lookbehind_still_blocks_inside_word() -> None:
    """The (?<![A-Za-z]) lookbehind must prevent 'alocation' from matching Loc."""
    out = extract_sample_ids("From alocation Tunisia, sample S1.")
    locs = [s.value.casefold() for s in out if s.kind == "loc"]
    # 'alocation' should NOT be misread as a Loc keyword + ation prefix.
    # Only 'Tunisia' (the real locality) should appear.
    assert "alocation" not in locs, (
        f"Negative lookbehind broken: 'alocation' should not match _LOC_RE: {locs}"
    )
