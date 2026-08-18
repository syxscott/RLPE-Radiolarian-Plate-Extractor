"""Round 20 source-guard tests: sample extraction patterns.

User audit: 4 OA papers were sampled. ``run_output.samples`` was 0/4 —
the legacy regex ``Sample\\s+[A-Za-z0-9\\-]+`` only matched the
``Sample N`` shape. Boughdiri 2007's ``CH4, specimen 7`` /
``MB4, specimen 15`` formats were completely missed.

These tests pin the four new patterns and verify they don't
over-match common geological terms.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _build_match(paper_id: str, caption: str):
    from rlpe.types import MatchResult

    return MatchResult(
        paper_id=paper_id,
        figure_id="od_plate_test_p001_pl01",
        panel_id="1",
        species="Genus species",
        panel_path=None,
        bbox=None,
        confidence=0.5,
        caption_snippet=caption,
        metadata={"page_index": 1},
    )


def test_boughdiri_short_codes_extracted():
    """Boughdiri 2007 'CH4, specimen 7' must produce B_CH4 + R_specimen_7."""
    from rlpe.converters import sample_records_from_matches

    matches = [
        _build_match(
            "boughdiri2007",
            "Plate I. Radiolarians from the Jédidi Fm. "
            "1) Ristola, CH4, specimen 7, 550 µm; "
            "2) Palinandromeda, MB4, specimen 15, 200 µm.",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert "B_CH4" in sids, f"B_CH4 missing: {sids}"
    assert "B_MB4" in sids, f"B_MB4 missing: {sids}"
    # Specimen matches embed the literal "specimen N" (with a space)
    # because the regex keeps the whole token. Sample_id is then
    # prefixed with R_ so the operator sees ``R_specimen 7``.
    assert any("specimen 7" in s for s in sids), f"specimen 7 missing: {sids}"
    assert any("specimen 15" in s for s in sids), f"specimen 15 missing: {sids}"


def test_legacy_sample_format_still_works():
    """Regression: 'Sample 12' must still produce *some* sample record.

    Audit 2026-08-18: since the Phase 65 ``extract_sample_ids`` helper
    was wired in, the helper now matches ``Sample N`` first and emits
    ``X_N`` (the helper prefix). The legacy ``S_`` regex still fires
    but is dropped by the cross-prefix dedup because the helper already
    registered ``(paper_id, "12")`` in ``raw_seen``. The invariant we
    still need to defend is that ``Sample N`` produces *some* sample
    record (either ``S_12`` legacy OR ``X_12`` helper); the exact
    prefix depends on which detector fires first. Accept either."""
    from rlpe.converters import sample_records_from_matches

    matches = [
        _build_match(
            "beccaro2006",
            "Plate 1. Radiolarians from Sample 12, locality X.",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert sids, f"'Sample N' produced no record at all: {sids}"
    assert any(s.startswith("S_") for s in sids) or any(s.startswith("X_") for s in sids), (
        f"Legacy 'Sample N' pattern not firing (expected S_ from regex "
        f"path or X_ from extract_sample_ids helper): {sids}"
    )


def test_short_codes_deduped_per_paper():
    """Two matches with the same short code in the same paper must
    dedupe to a single sample record."""
    from rlpe.converters import sample_records_from_matches

    matches = [
        _build_match("p1", "1) Foo, CH4, specimen 7"),
        _build_match("p1", "2) Bar, CH4, specimen 8"),
    ]
    samples = sample_records_from_matches(matches)
    ch4_records = [s for s in samples if s["sample_id"] == "B_CH4"]
    assert len(ch4_records) == 1, f"CH4 dedup failed: {ch4_records}"


def test_short_codes_not_over_matching():
    """False-positive guard: common non-sample text must not match."""
    from rlpe.converters import sample_records_from_matches

    # 'In' is not a Boughdiri-style prefix; '200' isn't a 1-4 digit short code
    matches = [
        _build_match(
            "p1",
            "Plate 1. Found in situ at the Karnezeika section. Length about 200 µm.",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    # No B_ codes (no paper-specific prefix like CH/MB)
    assert not any(s.startswith("B_") for s in sids), f"Short-code detector over-matched: {sids}"


def test_short_code_pattern_source_guard():
    """Source guard: converters.py must contain the Boughdiri short-code
    pattern with the locality prefixes (CH, MB, GA, ...)."""
    src = Path(Path(__file__).resolve().parents[1] / "src" / "rlpe" / "converters.py").read_text(
        encoding="utf-8"
    )
    for needle in (
        "CH|MB|GA|RM|HK|JP",
        "specimen\\s+",
        "B_",
        "R_",
    ):
        assert needle in src, (
            f"converters.py is missing the {needle!r} pattern. "
            "Round 20 sample extraction depends on this."
        )


# --- 6) Regression: span-based dedup (Audit 2026-08-18) ---------------
#
# Before the span-based dedup, the legacy ``S_`` regex and the
# ``extract_sample_ids`` helper each emitted one record per match,
# inflating sample counts. The audit fixed three concrete cases that
# were silently producing 2 records instead of 1:
#   - "Sample ID-203"   helper strips "ID-" → "X_203";
#                        legacy captures full "ID-203" → "S_ID-203"
#   - "Sample 100A"     helper truncates to "100" (was \d{2,});
#                        legacy captures "100A" → "S_100A"
#   - "Sample (12)"     "Sample\s+\(\d+\)" matches "Sample (12)";
#                        bare "\(\d{1,3}\)" also matches "(12)"
# Each case now produces exactly one record.


def test_dedup_sample_id_with_id_prefix():
    """``Sample ID-203`` must produce exactly one record."""
    from rlpe.converters import sample_records_from_matches

    matches = [_build_match("p1", "Plate 1. Radiolarians from Sample ID-203, locality X.")]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert len(samples) == 1, f"double-count on Sample ID-203: {sids}"
    assert "X_203" in sids, f"helper value missing: {sids}"


def test_dedup_sample_alphanumeric_suffix():
    """``Sample 100A`` must produce exactly one record (the ``A`` suffix preserved)."""
    from rlpe.converters import sample_records_from_matches

    matches = [_build_match("p1", "Sample 100A was collected.")]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert len(samples) == 1, f"double-count on Sample 100A: {sids}"
    assert "X_100A" in sids, f"helper should capture full 100A (with A suffix); got: {sids}"


def test_dedup_sample_parenthesized_keeps_specific_pattern():
    """``Sample (12)`` must produce exactly one record with ``Sample`` keyword visible."""
    from rlpe.converters import sample_records_from_matches

    matches = [_build_match("p1", "Section 1. Sample (12), Locality X")]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert len(samples) == 1, f"double-count on Sample (12): {sids}"
    # The more-specific Sample\s+\(\d+\) pattern must fire (not the
    # bare \(\d{1,3}\) one) so the operator sees the "Sample" keyword.
    assert any("Sample (12)" in s for s in sids), (
        f"Sample (12) should retain 'Sample' keyword; got: {sids}"
    )


def test_bare_parenthesized_still_works():
    """A bare ``(1) (2) (3)`` numbered list (no ``Sample`` keyword) still
    matches the ``L_`` pattern — the reorder for ``Sample (12)`` must
    not break the genuine Bragin-style numbered-list case."""
    from rlpe.converters import sample_records_from_matches

    matches = [
        _build_match(
            "p1",
            "(1) Praeparvicingula blackhorsensis, (2) Praeparvicingula donnae",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    l_ids = {s for s in sids if s.startswith("L_")}
    assert len(l_ids) >= 1, f"bare (N) parenthesized lost: {sids}"


# --- 7) Regression: hyphen-suffix IDs (Audit 2026-08-18) ---------------
#
# ``Sample 100-1`` and ``Sample 12-3`` were truncated to ``X_100`` /
# ``X_12`` because the helper regex's digit-led branch only consumed
# trailing alphanumerics (``[A-Za-z0-9]*``), not hyphens. Extended to
# ``[A-Za-z0-9\-]*`` so the helper and legacy regex agree on the value.


def test_dedup_sample_hyphen_suffix():
    """``Sample 100-1`` must produce a single record with the full
    ``100-1`` id (not the truncated ``100``)."""
    from rlpe.converters import sample_records_from_matches

    matches = [_build_match("p1", "Sample 100-1 was collected.")]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert len(samples) == 1, f"double-count on Sample 100-1: {sids}"
    assert "X_100-1" in sids, f"helper should capture full 100-1: {sids}"


def test_dedup_sample_letter_suffix_still_works():
    """Regression: hyphen extension must not break the ``Sample 100A``
    case (no hyphen, just letter suffix)."""
    from rlpe.converters import sample_records_from_matches

    matches = [_build_match("p1", "Sample 100A was collected.")]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert "X_100A" in sids, f"Sample 100A regression: {sids}"


def test_dedup_sample_space_stops_match():
    """``Sample 100 µm`` must stop at the space — ``µm`` is not part
    of the sample id."""
    from rlpe.converters import sample_records_from_matches

    matches = [_build_match("p1", "Sample 100 µm was collected.")]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert "X_100" in sids, f"space should stop match: {sids}"
    assert not any("µm" in s for s in sids), f"µm leaked into id: {sids}"


# --- 8) Source guard: dedup uses span-overlap, not value compare -------
#
# Pins the Audit 2026-08-18 dedup design. A future refactor that
# reverts to value-based comparison will silently re-introduce the
# Sample ID-203 / Sample 100A / Sample (12) double-count bugs. The
# presence of a span-tracking block (``helper_spans`` plus overlap
# check) is the load-bearing structural signal.


def test_sample_records_dedup_uses_span_overlap_not_value_compare():
    """Source guard: ``sample_records_from_matches`` must track
    text spans for cross-prefix dedup, not just compare the captured
    raw values. The helper and the legacy regex normalise the same
    physical sample text differently (helper strips ``ID-``, captures
    only the digits branch in ``Sample 100A``), so value-based dedup
    misses them and silently double-counts.

    Audit 2026-08-18: ``helper_spans`` set + ``any(... < ... and ... <
    ... for ... in helper_spans)`` is the span-overlap check.
    ``raw_seen`` set is also kept as a value-based backstop, but it
    must NOT be the primary dedup mechanism."""
    src = Path(Path(__file__).resolve().parents[1] / "src" / "rlpe" / "converters.py").read_text(
        encoding="utf-8"
    )
    # Span tracking set
    assert "helper_spans" in src, (
        "converters.py must track helper text spans for dedup. "
        "Value-based dedup alone (the pre-sweep-2 implementation) "
        "silently double-counts Sample ID-203 / Sample 100A."
    )
    # Span-overlap check (uses the half-open interval overlap rule
    # ``a < d and c < b``)
    assert "< legacy_end and legacy_start < h_end" in src, (
        "converters.py must use span-overlap (a<d and c<b) to dedup "
        "against the helper, not value equality."
    )
    # Legacy spans must be added to the covered set AFTER insert so a
    # subsequent legacy pattern that overlaps (e.g. Sample\s+\(\d+\)
    # vs \(\d{1,3}\) on "Sample (12)") gets dropped.
    assert "helper_spans.add((m.paper_id, legacy_start, legacy_end))" in src, (
        "converters.py must register each legacy match's span in "
        "helper_spans so the legacy-to-legacy dedup fires."
    )
    # Reorder: Sample\s+\(\d+\) must come BEFORE \(\d{1,3}\) so the
    # more-specific pattern wins. Search for the patterns inside
    # ``re.compile(r"...")`` to avoid matching the comment text that
    # mentions both patterns.
    import re as _re

    pattern_positions: list[tuple[int, str]] = []
    for _m in _re.finditer(r're\.compile\(r"([^"]+)"\)', src):
        pattern_positions.append((_m.start(), _m.group(1)))
    s_par = next((p for p, pat in pattern_positions if r"Sample\s+\(\d+\)" in pat), -1)
    bare_par = next((p for p, pat in pattern_positions if r"\(\d{1,3}\)" in pat), -1)
    assert 0 < s_par < bare_par, (
        f"Sample\\s+\\(\\d+\\) must come BEFORE \\(\\d{{1,3}}\\) in "
        f"_SAMPLE_PATTERNS so 'Sample (12)' keeps the 'Sample' prefix. "
        f"Got positions: Sample_par={s_par}, bare_par={bare_par}"
    )
