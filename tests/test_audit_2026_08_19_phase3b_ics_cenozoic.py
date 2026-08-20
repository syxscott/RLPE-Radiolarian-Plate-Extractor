"""Regression tests for audit 2026-08-19 Phase 3B — ICS Cenozoic
Tarantian rename + Permian Asselian/Sakmarian boundary precision +
rank-type expansion (epoch -> series/subsystem) + country list
deduplication.

Covers:
  - Tarantian (ICS 2024-09) replaces "Late Pleistocene" with parent
    Quaternary, cn "塔兰期", ma_top 0.012 Ma.
  - Asselian / Sakmarian ICS 2023 boundary at 293.52 Ma (not 295.0).
  - Permian series (Cisuralian / Guadalupian / Lopingian) rank
    "series" (not "epoch").
  - Carboniferous subsystems (Mississippian / Pennsylvanian) rank
    "subsystem".
  - Silurian series (Llandovery / Wenlock / Ludlow / Pridoli) rank
    "series".
  - Country list de-duplicated + United Kingdom / Slovenia / Croatia /
    Tibet added.
  - ``classify_age_string`` still resolves all the renamed rows to the
    right (period, epoch, age) tuple via the middle-rank walk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rlpe.geology_extraction as geo_ext  # noqa: E402
import rlpe.stratigraphy as strat  # noqa: E402
from rlpe.stratigraphy import (  # noqa: E402
    _ICS_ROWS,
    classify_age_string,
    find_ages_in_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_row(name: str) -> dict:
    """Return the _ICS_ROWS dict with name == ``name`` (case-sensitive)."""
    for r in _ICS_ROWS:
        if r["name"] == name:
            return r
    raise AssertionError(f"{name!r} missing from _ICS_ROWS")


def _all_rows_by_name(name: str) -> list[dict]:
    """Return all _ICS_ROWS dicts with name == ``name`` (multiple = duplicate)."""
    return [r for r in _ICS_ROWS if r["name"] == name]


# ---------------------------------------------------------------------------
# B-2 — Tarantian rename (replaces "Late Pleistocene")
# ---------------------------------------------------------------------------


class TestTarantianRename:
    """Bug B-2: ICS 2024-09 chart renamed the final Pleistocene stage
    from "Late Pleistocene" to "Tarantian" (0.012-0.129 Ma), parent
    "Quaternary", Chinese name "塔兰期". The previous row pointed at
    Pleistocene (which is an age, not a period) and used "晚上新世"
    (Late Pliocene) as the Chinese label — a literal mistranslation.
    """

    def test_tarantian_present(self):
        row = _find_row("Tarantian")
        assert row is not None

    def test_late_pleistocene_absent(self):
        # "Late Pleistocene" must not appear anywhere in _ICS_ROWS.
        for r in _ICS_ROWS:
            assert r["name"] != "Late Pleistocene", (
                "Late Pleistocene should have been renamed to Tarantian; "
                f"found duplicate at parent={r.get('parent')!r}"
            )

    def test_tarantian_parent_is_quaternary(self):
        row = _find_row("Tarantian")
        assert row["parent"] == "Quaternary"

    def test_tarantian_chinese_name(self):
        row = _find_row("Tarantian")
        assert row["cn"] == "塔兰期"

    def test_tarantian_ma_top(self):
        row = _find_row("Tarantian")
        assert row["ma_top"] == 0.012

    def test_tarantian_ma_base(self):
        row = _find_row("Tarantian")
        assert row["ma_base"] == 0.129

    def test_tarantian_rank_is_age(self):
        row = _find_row("Tarantian")
        assert row["rank"] == "age"

    def test_classify_tarantian_english(self):
        cls = classify_age_string("Tarantian")
        assert cls.age == "Tarantian"
        assert cls.period == "Quaternary"
        assert cls.rank == "age"
        assert cls.confidence > 0
        assert cls.ma_top == 0.012
        assert cls.ma_base == 0.129

    def test_find_ages_in_text_tarantian(self):
        ages = find_ages_in_text("Tarantian calcareous nannofossils, Sicily")
        tarantian = [a for a in ages if a.age == "Tarantian"]
        assert tarantian, f"Tarantian not surfaced in {ages!r}"
        cls = tarantian[0]
        assert cls.period == "Quaternary"
        assert cls.ma_top == 0.012

    def test_tarantian_chibanian_boundary_monotone(self):
        # Tarantian.ma_top (0.012) must equal Chibanian.ma_base (0.129-1 boundary).
        # i.e. Tarantian sits ABOVE Chibanian: ma_top_tarantian < ma_base_tarantian
        tarantian = _find_row("Tarantian")
        chibanian = _find_row("Chibanian")
        assert tarantian["ma_top"] < tarantian["ma_base"]
        assert chibanian["ma_top"] < chibanian["ma_base"]
        # Tarantian top (0.012) < Chibanian base (0.129): Tarantian
        # is the younger stage.
        assert tarantian["ma_top"] < chibanian["ma_base"]


# ---------------------------------------------------------------------------
# M-1 — Asselian / Sakmarian boundary precision (ICS 2023 = 293.52 Ma)
# ---------------------------------------------------------------------------


class TestPermianBoundaryPrecision:
    """Bug M-1: the previous Asselian / Sakmarian boundary used the
    2004 ICS value of 295.0 Ma. ICS 2023 sets the boundary at
    293.52 Ma — Sakmarian's base and Asselian's top must agree."""

    def test_asselian_ma_top_is_293_52(self):
        row = _find_row("Asselian")
        assert row["ma_top"] == 293.52, (
            f"Asselian.ma_top should be 293.52 Ma (ICS 2023), got {row['ma_top']}"
        )

    def test_sakmarian_ma_base_is_293_52(self):
        row = _find_row("Sakmarian")
        assert row["ma_base"] == 293.52, (
            f"Sakmarian.ma_base should be 293.52 Ma (ICS 2023), got {row['ma_base']}"
        )

    def test_asselian_sakmarian_boundary_is_contiguous(self):
        # Asselian's top = Sakmarian's base (contiguous ICS stage).
        asselian = _find_row("Asselian")
        sakmarian = _find_row("Sakmarian")
        assert asselian["ma_top"] == sakmarian["ma_base"], (
            f"Asselian.ma_top ({asselian['ma_top']}) must equal "
            f"Sakmarian.ma_base ({sakmarian['ma_base']})"
        )

    def test_permian_stages_are_monotone(self):
        # Walk down the Permian stages in order and assert each
        # stage's ma_base >= the next-younger stage's ma_top.
        permian_stages = [
            "Asselian",
            "Sakmarian",
            "Artinskian",
            "Kungurian",
            "Roadian",
            "Wordian",
            "Capitanian",
            "Wuchiapingian",
            "Changhsingian",
        ]
        rows = [_find_row(n) for n in permian_stages]
        for older, younger in zip(rows, rows[1:]):
            assert older["ma_base"] >= younger["ma_top"], (
                f"{older['name']} (ma_base={older['ma_base']}) overlaps "
                f"{younger['name']} (ma_top={younger['ma_top']})"
            )

    def test_classify_asselian_carries_new_boundary(self):
        cls = classify_age_string("Asselian")
        assert cls.age == "Asselian"
        assert cls.period == "Permian"
        assert cls.ma_top == 293.52
        assert cls.ma_base == 298.9

    def test_classify_sakmarian_carries_new_boundary(self):
        cls = classify_age_string("Sakmarian")
        assert cls.age == "Sakmarian"
        assert cls.period == "Permian"
        assert cls.ma_top == 290.1
        assert cls.ma_base == 293.52


# ---------------------------------------------------------------------------
# M-2 — Rank-type expansion (epoch -> series / subsystem)
# ---------------------------------------------------------------------------


class TestRankTypeExpansion:
    """Bug M-2: ICS 2023 formally designates the Permian subdivisions
    as "series", the Carboniferous sub-periods as "subsystem", and
    the Silurian subdivisions as "series" — not "epoch". The change
    must keep the downstream ``classify_age_string`` semantics intact
    (epoch / series / subsystem all populate ``cls.epoch``).
    """

    # --- Permian series ---

    def test_cisuralian_rank_is_series(self):
        assert _find_row("Cisuralian")["rank"] == "series"

    def test_guadalupian_rank_is_series(self):
        assert _find_row("Guadalupian")["rank"] == "series"

    def test_lopingian_rank_is_series(self):
        assert _find_row("Lopingian")["rank"] == "series"

    # --- Carboniferous subsystems ---

    def test_mississippian_rank_is_subsystem(self):
        assert _find_row("Mississippian")["rank"] == "subsystem"

    def test_pennsylvanian_rank_is_subsystem(self):
        assert _find_row("Pennsylvanian")["rank"] == "subsystem"

    # --- Silurian series ---

    def test_llandovery_rank_is_series(self):
        assert _find_row("Llandovery")["rank"] == "series"

    def test_wenlock_rank_is_series(self):
        assert _find_row("Wenlock")["rank"] == "series"

    def test_ludlow_rank_is_series(self):
        assert _find_row("Ludlow")["rank"] == "series"

    def test_pridoli_rank_is_series(self):
        assert _find_row("Pridoli")["rank"] == "series"

    # --- Downstream classify_age_string still populates cls.epoch ---

    def test_late_permian_still_classifies_as_lopingian(self):
        # Backward-compat: even with rank="series", the public
        # classifier must still map "Late Permian" -> epoch =
        # "Lopingian" via _ICS_ALIASES + the middle-rank walk.
        cls = classify_age_string("Late Permian")
        assert cls.epoch == "Lopingian"
        assert cls.period == "Permian"

    def test_early_permian_still_classifies_as_cisuralian(self):
        cls = classify_age_string("Early Permian")
        assert cls.epoch == "Cisuralian"
        assert cls.period == "Permian"

    def test_mississippian_age_walks_to_carboniferous_period(self):
        # Tournaisian (parent=Mississippian, rank=age) must still
        # resolve period=Carboniferous — the parent walk needs to
        # accept "subsystem" parents, not just "epoch".
        cls = classify_age_string("Tournaisian")
        assert cls.age == "Tournaisian"
        assert cls.period == "Carboniferous"
        # epoch is the subsystem itself (Mississippian).
        assert cls.epoch == "Mississippian"

    def test_pennsylvanian_age_walks_to_carboniferous_period(self):
        cls = classify_age_string("Moscovian")
        assert cls.age == "Moscovian"
        assert cls.period == "Carboniferous"
        assert cls.epoch == "Pennsylvanian"

    def test_sakmarian_uses_subsystem_parent_walk(self):
        # Sakmarian.parent = Permian (period), so this is the easy
        # walk; verify the rank="subsystem" change did not regress
        # the easy path either.
        cls = classify_age_string("Sakmarian")
        assert cls.age == "Sakmarian"
        assert cls.period == "Permian"

    def test_chinese_late_permian_alias_unchanged(self):
        # 上二叠统 still maps to Lopingian with the right Ma bounds.
        cls = classify_age_string("上二叠统")
        assert cls.period == "Permian"
        assert cls.epoch == "Lopingian"
        assert cls.ma_top == 251.9
        assert cls.ma_base == 259.51


# ---------------------------------------------------------------------------
# M-9 (optional) — country list deduplication + UK / Slovenia / Croatia
# ---------------------------------------------------------------------------


class TestCountryList:
    """Bug M-9 (optional): the previous ``_COUNTRIES`` tuple had Greece
    and Turkey listed twice (rows 396-397) and was missing UK,
    Slovenia, Croatia, and Tibet."""

    def test_united_kingdom_in_list(self):
        assert "United Kingdom" in geo_ext._COUNTRIES

    def test_slovenia_in_list(self):
        assert "Slovenia" in geo_ext._COUNTRIES

    def test_croatia_in_list(self):
        assert "Croatia" in geo_ext._COUNTRIES

    def test_tibet_in_list(self):
        assert "Tibet" in geo_ext._COUNTRIES

    def test_greece_not_duplicated(self):
        n = geo_ext._COUNTRIES.count("Greece")
        assert n == 1, f"Greece appears {n} times in _COUNTRIES; expected 1"

    def test_turkey_not_duplicated(self):
        n = geo_ext._COUNTRIES.count("Turkey")
        assert n == 1, f"Turkey appears {n} times in _COUNTRIES; expected 1"

    def test_country_list_has_no_duplicates_at_all(self):
        seen: set[str] = set()
        for c in geo_ext._COUNTRIES:
            assert c not in seen, f"Duplicate country in _COUNTRIES: {c!r}"
            seen.add(c)

    def test_country_centroids_cover_new_entries(self):
        # Every newly added country must also have a centroid so the
        # Round 21 country-centroid fallback fires.
        for c in ("United Kingdom", "Slovenia", "Croatia", "Tibet"):
            assert c in geo_ext._COUNTRY_CENTROIDS, (
                f"{c!r} added to _COUNTRIES but missing from _COUNTRY_CENTROIDS"
            )
            lat, lon = geo_ext._COUNTRY_CENTROIDS[c]
            assert -90.0 <= lat <= 90.0, f"{c} centroid latitude out of range: {lat}"
            assert -180.0 <= lon <= 180.0, f"{c} centroid longitude out of range: {lon}"


# ---------------------------------------------------------------------------
# classify_age_string — end-to-end smoke for the renamed/relabelled rows
# ---------------------------------------------------------------------------


class TestClassifyAgeStringEndToEnd:
    """End-to-end checks that ``classify_age_string`` returns the right
    (period, epoch, age) tuple for every row affected by Phase 3B."""

    def test_tarantian_zh_alias_resolves_to_quaternary(self):
        cls = classify_age_string("塔兰期")
        assert cls.age == "Tarantian"
        assert cls.period == "Quaternary"

    def test_sakmarian_asselian_ma_values_via_classifier(self):
        s = classify_age_string("Sakmarian")
        a = classify_age_string("Asselian")
        assert s.ma_base == 293.52
        assert a.ma_top == 293.52
        assert s.ma_base == a.ma_top

    def test_find_ages_in_text_returns_unique_tarantian(self):
        # Regression: find_ages_in_text must not return both
        # "Tarantian" and the obsolete "Late Pleistocene" — only the
        # new name should be present.
        ages = find_ages_in_text("Tarantian-age sediments")
        names = {a.age for a in ages if a.age}
        assert "Tarantian" in names
        assert "Late Pleistocene" not in names


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
