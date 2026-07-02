"""Tests for stratigraphy classification and ICS table lookup."""

from __future__ import annotations

from rlpe.stratigraphy import (
    _ICS_ROWS as ICS_CHART,
)
from rlpe.stratigraphy import (
    classify_age_string,
    find_ages_in_text,
)


class TestClassifyAgeString:
    def test_period_only(self):
        c = classify_age_string("Permian")
        assert c.period == "Permian"
        assert c.rank == "period"
        assert c.confidence > 0

    def test_period_with_early(self):
        c = classify_age_string("Early Jurassic")
        assert c.period == "Jurassic"
        # rank may be "epoch" or "period" depending on how the modifier matches;
        # the period must always be present
        assert c.rank in {"epoch", "period"}
        assert c.confidence > 0

    def test_chinese_period(self):
        c = classify_age_string("二叠纪")
        assert c.period == "Permian"
        assert c.confidence > 0

    def test_chinese_stage(self):
        c = classify_age_string("长兴期")
        assert c.period == "Permian"
        # epoch may be set to the Lopingian parent (when present in the table) or None
        assert c.age == "Changhsingian"
        assert c.rank == "age"
        assert c.confidence > 0

    def test_english_stage(self):
        c = classify_age_string("Changhsingian")
        assert c.age == "Changhsingian"
        assert c.period == "Permian"
        assert c.rank == "age"

    def test_unknown_string(self):
        c = classify_age_string("Foobar")
        assert c.period is None
        assert c.confidence == 0.0

    def test_empty_string(self):
        c = classify_age_string("")
        assert c.period is None


class TestFindAgesInText:
    def test_finds_multiple_periods(self):
        ages = find_ages_in_text(
            "Samples span the Permian to the Jurassic, with a brief Cretaceous interval."
        )
        periods = {a.period for a in ages if a.period}
        assert "Permian" in periods
        assert "Jurassic" in periods
        assert "Cretaceous" in periods

    def test_finds_stage_names(self):
        ages = find_ages_in_text("Dalong Formation, Changhsingian Stage, South China.")
        stage_ages = [a for a in ages if a.rank == "age"]
        assert any(a.age == "Changhsingian" for a in stage_ages)

    def test_finds_early_late_combination(self):
        ages = find_ages_in_text("Late Permian rocks are widespread.")
        # At minimum, the Permian period should be detected
        assert any(a.period == "Permian" for a in ages)

    def test_dedupes(self):
        ages = find_ages_in_text("Permian and Permian and Permian.")
        period_ages = [a for a in ages if a.rank == "period" and a.period == "Permian"]
        assert len(period_ages) == 1

    def test_empty_text(self):
        assert find_ages_in_text("") == []


class TestNormalizeAgeName:
    def test_chinese_to_english_period(self):
        # Chinese names are looked up via classify_age_string
        c = classify_age_string("二叠纪")
        assert c.period == "Permian"

    def test_chinese_to_english_stage(self):
        c = classify_age_string("长兴期")
        assert c.age == "Changhsingian"
        assert c.period == "Permian"

    def test_english_passthrough(self):
        c = classify_age_string("Permian")
        assert c.period == "Permian"


class TestICSChart:
    def test_chart_not_empty(self):
        assert len(ICS_CHART) > 50

    def test_each_entry_has_required_keys(self):
        for row in ICS_CHART[:20]:
            assert "name" in row
            assert "rank" in row
            assert row["rank"] in {"eon", "era", "period", "epoch", "age"}

    def test_canonical_periods_present(self):
        names = {row["name"] for row in ICS_CHART if row["rank"] == "period"}
        for must in (
            "Cambrian",
            "Ordovician",
            "Silurian",
            "Devonian",
            "Carboniferous",
            "Permian",
            "Triassic",
            "Jurassic",
            "Cretaceous",
            "Paleogene",
            "Neogene",
            "Quaternary",
        ):
            assert must in names, f"Missing period {must}"


class TestMaPropagation:
    """Confirm numeric Ma bounds flow from the matched ICS row into
    AgeClassification and onwards into the public GeologyRecord fields.

    This is the deterministic Ma mapping Task 5 wires up; lithology /
    member / group remain for the later MiniMax-assisted extractor.
    """

    def test_changhsingian_carries_numeric_ma(self):
        c = classify_age_string("Changhsingian")
        assert c.age == "Changhsingian"
        assert c.ma_top is not None and c.ma_base is not None
        # ICS values: Changhsingian top ≈ 251.902, base ≈ 254.14.
        assert 251.0 < c.ma_top < 252.5
        assert 253.5 < c.ma_base < 255.0
        # ma_mid is the midpoint of top + base.
        assert c.ma_mid is not None
        assert abs(c.ma_mid - (c.ma_top + c.ma_base) / 2.0) < 1e-9

    def test_period_carries_numeric_ma(self):
        c = classify_age_string("Permian")
        assert c.period == "Permian"
        assert c.ma_top is not None and c.ma_base is not None
        # ICS convention: ma_top is the younger boundary (smaller number),
        # ma_base is the older boundary (larger number). For a period
        # the interval is non-empty so the two differ.
        assert c.ma_top < c.ma_base
        assert c.ma_mid is not None
        assert c.ma_mid == (c.ma_top + c.ma_base) / 2.0

    def test_unknown_string_leaves_ma_none(self):
        c = classify_age_string("definitely-not-a-period")
        assert c.ma_top is None
        assert c.ma_base is None
        assert c.ma_mid is None

    def test_to_dict_includes_ma_keys(self):
        c = classify_age_string("Changhsingian")
        d = c.to_dict()
        for k in ("ma_top", "ma_base", "ma_mid"):
            assert k in d
