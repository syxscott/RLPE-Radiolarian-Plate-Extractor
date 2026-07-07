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
    """Regression: 'Sample 12' must still produce S_Sample 12 / S_12."""
    from rlpe.converters import sample_records_from_matches

    matches = [
        _build_match(
            "beccaro2006",
            "Plate 1. Radiolarians from Sample 12, locality X.",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert any(s.startswith("S_") for s in sids), (
        f"Legacy 'Sample N' pattern not firing: {sids}"
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
            "Plate 1. Found in situ at the Karnezeika section. "
            "Length about 200 µm.",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    # No B_ codes (no paper-specific prefix like CH/MB)
    assert not any(s.startswith("B_") for s in sids), (
        f"Short-code detector over-matched: {sids}"
    )


def test_short_code_pattern_source_guard():
    """Source guard: converters.py must contain the Boughdiri short-code
    pattern with the locality prefixes (CH, MB, GA, ...)."""
    src = Path(
        "/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/src/rlpe/converters.py"
    ).read_text(encoding="utf-8")
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