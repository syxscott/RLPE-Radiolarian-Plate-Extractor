"""Round 21 source-guard tests: extended sample patterns.

User audit (Round 20 sampling): 3 of 4 papers had 0 samples because
their captions used different identifier shapes that the Round 20
patterns didn't match:

  - Bandini 2006: ``"Al74_300"`` (Greek locality code + number +
    underscore + number) — needs ``Al`` prefix added
  - Danelian 2006: ``"Mg-100"`` (Vocontian section code + number)
    — needs ``Mg`` prefix added
  - Bragin 2025: ``"(1) (2) (3)"`` numbered list and ``"pl. 2"``
    abbreviated plate reference — entirely new patterns

Round 21 adds 8 new locality prefixes (Al/Mg/Tr/Pl/BK/OC/WP/CM) to
the Boughdiri-style short-code pattern, plus 3 new pattern types:
parenthesized numbered list, ``pl. N`` plate reference, and
``Sample (N)`` parenthesized form.
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


# --- 1) New locality prefixes (Bandini / Danelian style) ----------------


def test_bandini_al_prefix_extracted():
    """Bandini 2006 'Al74_300' style must produce B_Al74_300."""
    from rlpe.converters import sample_records_from_matches

    matches = [
        _build_match(
            "bandini2006",
            "Plate 1. SEM-illustrations. Figure 3 Acaeniotyle sp. A Al74_300; "
            "Figure 4 Acaeniotyle sp. B Al70_090",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert "B_Al74_300" in sids, f"B_Al74_300 missing: {sids}"
    assert "B_Al70_090" in sids, f"B_Al70_090 missing: {sids}"


def test_danelian_mg_prefix_extracted():
    """Danelian 2006 'Mg-100' style must produce B_Mg-100."""
    from rlpe.converters import sample_records_from_matches

    matches = [
        _build_match(
            "danelian2006",
            "Plate 1. 1) Acastea sp.cf. A. remusa HULL, Mg-100; "
            "2-3) Archaeodictyomitra apiarium (RÜST), Mg-200",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert "B_Mg-100" in sids, f"B_Mg-100 missing: {sids}"


# --- 2) Bragin-style "(N)" numbered list --------------------------------


def test_bragin_parenthesized_numbered_list():
    """Bragin 2025 '(1) (2) (3) ...' style must produce 3 L_-prefixed
    sample_ids. The regex matches the full parenthesized token so
    the resulting sample_id looks like ``L_(1)``."""
    from rlpe.converters import sample_records_from_matches

    matches = [
        _build_match(
            "bragin2025",
            "Plate I. (1) Praeparvicingula blackhorsensis; "
            "(2) Praeparvicingula donnae; "
            "(3) Praeparvicingula excelens",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    # The regex consumes the parentheses too; sample_id is the full token.
    assert "L_(1)" in sids, f"L_(1) missing: {sids}"
    assert "L_(2)" in sids, f"L_(2) missing: {sids}"
    assert "L_(3)" in sids, f"L_(3) missing: {sids}"


# --- 3) Bragin-style "pl. N" plate reference ---------------------------


def test_bragin_pl_dot_pattern():
    """Bragin 2025 'pl. 2' style must produce P_2."""
    from rlpe.converters import sample_records_from_matches

    matches = [
        _build_match(
            "bragin2025",
            "Plate I. See pl. 2 for cross-reference. pl. 4 Fig. 5",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert "P_2" in sids, f"P_2 missing: {sids}"
    assert "P_4" in sids, f"P_4 missing: {sids}"


# --- 4) Sample (N) parenthesized form ----------------------------------


def test_sample_parenthesized_form():
    """'Sample (12)' style must produce S_Sample_(12)."""
    from rlpe.converters import sample_records_from_matches

    matches = [_build_match("p1", "Section 1. Sample (12), Locality X")]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert any("Sample (12)" in s for s in sids), f"Sample (12) missing: {sids}"


# --- 5) Regression: existing Round 20 patterns still work --------------


def test_boughdiri_short_codes_still_work():
    """Round 21 must not break the Round 20 Boughdiri patterns."""
    from rlpe.converters import sample_records_from_matches

    matches = [
        _build_match(
            "boughdiri2007",
            "1) Ristola, CH4, specimen 7; 2) Palinandromeda, MB4, specimen 15",
        )
    ]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert "B_CH4" in sids
    assert "B_MB4" in sids
    assert any("specimen 7" in s for s in sids)
    assert any("specimen 15" in s for s in sids)


def test_legacy_sample_capital_still_works():
    """'Sample 12' (capital S) must still match *some* sample-id pattern.

    Audit 2026-08-18: the Phase 65 ``extract_sample_ids`` helper now
    matches ``Sample N`` first and emits ``X_N`` (the helper prefix).
    The legacy ``S_`` regex still fires but is dropped by the
    cross-prefix dedup because the helper already registered
    ``(paper_id, "12")`` in ``raw_seen``. The invariant we still need
    to defend is that the literal numeric ID ``12`` is captured as
    *some* sample record (either ``S_12`` legacy OR ``X_12`` helper);
    the prefix depends on which detector fires first. Accept either."""
    from rlpe.converters import sample_records_from_matches

    matches = [_build_match("p1", "From Sample 12, locality X")]
    samples = sample_records_from_matches(matches)
    sids = {s["sample_id"] for s in samples}
    assert "S_12" in sids or "X_12" in sids, (
        f"Legacy Sample pattern broken (expected S_12 from regex path "
        f"or X_12 from extract_sample_ids helper): {sids}"
    )


# --- 6) Source guard: pattern list has the new entries -----------------


def test_pattern_list_has_round21_additions():
    """Source guard: converters.py must contain the new prefixes
    and patterns added in Round 21."""
    import re

    # Resolve converters.py relative to this test file so the path
    # works in CI (where the runner cwd is the checkout root, not the
    # developer's /home/user/shenyaxuan/... absolute path).
    src = (Path(__file__).resolve().parent.parent / "src" / "rlpe" / "converters.py").read_text(
        encoding="utf-8"
    )
    # Strip whitespace for substring search so multi-line strings
    # (Python implicit concatenation across lines) match.
    src_compact = re.sub(r"\s+", "", src)
    # New prefixes (Bandini Al, Danelian Mg). The original 13 prefixes
    # are followed by the 8 new ones split across two lines.
    new_prefixes = "Al|Mg|Tr|Pl|BK|OC|WP|CM"
    assert new_prefixes in src_compact, (
        "converters.py missing the new locality prefix list "
        "(Al/Mg/Tr/Pl/BK/OC/WP/CM). Bandini 'Al74_300' and "
        "Danelian 'Mg-100' won't be matched."
    )
    # Parenthesized numbered list — search for the raw regex string
    # (the source contains the literal ``\(\d{1,3}\)``). The regex
    # argument to ``re.search`` must match the source's literal text.
    assert r"\(\d{1,3}\)" in src_compact, (
        "converters.py missing the (N) numbered-list pattern. "
        "Bragin 2025 '(1) (2) (3)' style captions won't be matched."
    )
    # pl. N pattern (the source uses ``pl\.\s*(\d{1,2})`` — search
    # for the unique ``pl\.`` literal since ``\s*`` and ``\d{1,2}``
    # can be followed by different delimiters in different patterns).
    assert r"pl\." in src_compact, (
        "converters.py missing the pl. N abbreviated-plate pattern. "
        "Bragin 2025 'pl. 2' references won't be matched."
    )
    # And the trailing digit block must also be present.
    assert r"\d{1,2}" in src_compact, (
        "converters.py missing the \\d{1,2} digit block — the pl. N pattern's digit run is missing."
    )
