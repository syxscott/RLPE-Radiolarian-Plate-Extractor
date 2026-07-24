"""Audit regression: extract_age_terms must preserve source casing.

Audit Agent A C3 found that ``extract_age_terms`` used ``m.group(1)``
which always returned the lowercased regex alternation string, not
the actual source casing. Docstring always promised "preserve
original casing" — violated since Phase 65.
"""

from __future__ import annotations

import pytest

from rlpe.sample_id_extractor import extract_age_terms


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("Late Cretaceous carbonates of Tunisia", ["Late Cretaceous"]),
        ("LATE CRETACEOUS carbonates of Tunisia", ["LATE CRETACEOUS"]),
        ("late cretaceous carbonates of Tunisia", ["late cretaceous"]),
        ("Late TRIASSIC and EARLY Jurassic", ["Late TRIASSIC", "EARLY Jurassic"]),
        ("Carnian (Late Triassic) of Italy", ["Carnian", "Late Triassic"]),
        ("no age here", []),
    ],
)
def test_extract_age_terms_preserves_source_casing(caption: str, expected: list[str]) -> None:
    assert extract_age_terms(caption) == expected


def test_extract_age_terms_dedup_keeps_first_seen_casing() -> None:
    """If 'Late Cretaceous' appears twice with different casing, keep first."""
    assert extract_age_terms(
        "Late Cretaceous of Tunisia; LATE CRETACEOUS of Greece"
    ) == ["Late Cretaceous"]
