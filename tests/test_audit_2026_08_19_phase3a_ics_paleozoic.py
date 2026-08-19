"""Phase 3A (2026-08-19) audit: verify Cambrian / Ordovician / Devonian
series and stages are now present in ``_ICS_ROWS`` and correctly
classified by :func:`classify_age_string`.

Background
----------
The previous ``_ICS_ROWS`` table in :mod:`rlpe.stratigraphy` had
period-level entries for Cambrian, Ordovician and Devonian but no
series (epoch) or stage (age) rows. Paleozoic radiolarian papers
(De Wever, Caridroit, etc.) routinely cite named stages such as
"Wuliuan", "Floian", "Frasnian" — without these rows in the local
table, classification degraded to the parent period (loosing 30-50
Myr of resolution) or fell through to the PBDB network fallback.

This test module asserts the audit-driven additions:

* Cambrian:    4 series (Terreneuvian, Series 2, Miaolingian, Furongian)
               10 stages (Fortunian, Stage 2-4, Wuliuan, Drumian,
               Guzhangian, Paibian, Jiangshanian, Stage 10)
* Ordovician:  3 series (Early/Middle/Late Ordovician)
               7 stages (Tremadocian, Floian, Dapingian, Darriwilian,
               Sandbian, Katian, Hirnantian)
* Devonian:    3 series (Early/Middle/Late Devonian)
               7 stages (Lochkovian, Pragian, Emsian, Eifelian,
               Givetian, Frasnian, Famennian)

Values are taken from the ICS 2023/09 chronostratigraphic chart
(https://stratigraphy.org).
"""

from __future__ import annotations

from rlpe.stratigraphy import (
    _ICS_ROWS as ICS_CHART,
    classify_age_string,
    find_ages_in_text,
)


def _rows_by_rank(rank: str) -> dict[str, dict]:
    """Return all rows of a given rank keyed by their English name."""
    return {r["name"]: r for r in ICS_CHART if r["rank"] == rank}


def _rows_by_name() -> dict[str, dict]:
    """Return all rows keyed by their English name (any rank)."""
    return {r["name"]: r for r in ICS_CHART}


# ---------------------------------------------------------------------------
# Cambrian
# ---------------------------------------------------------------------------


class TestCambrianSeries:
    """All four Cambrian series (ICS 2023/09)."""

    EXPECTED = ("Terreneuvian", "Series 2", "Miaolingian", "Furongian")

    def test_all_present(self):
        epochs = _rows_by_rank("epoch")
        for name in self.EXPECTED:
            assert name in epochs, f"Missing Cambrian series: {name}"

    def test_parent_is_cambrian(self):
        epochs = _rows_by_rank("epoch")
        for name in self.EXPECTED:
            assert epochs[name]["parent"] == "Cambrian", (
                f"{name} should parent to Cambrian, got {epochs[name]['parent']}"
            )

    def test_ma_ranges_monotonic(self):
        """Series must be ordered from oldest (Terreneuvian) to youngest (Furongian)."""
        epochs = _rows_by_rank("epoch")
        # Ma base is the OLDER (larger) boundary. Earlier-in-time series
        # should have LARGER ma_base than later-in-time series.
        bases = [epochs[n]["ma_base"] for n in self.EXPECTED]
        # Terreneuvian (538.8) > Series 2 (521.0) > Miaolingian (509.0) > Furongian (497.0)
        assert bases == sorted(bases, reverse=True), (
            f"Cambrian series ma_base not strictly decreasing: {bases}"
        )

    def test_ma_ranges_span_cambrian_period(self):
        """Union of all series intervals should equal Cambrian period bounds."""
        epochs = _rows_by_rank("epoch")
        # ma_top is the YOUNG (smaller) boundary, ma_base is OLD (larger).
        tops = [epochs[n]["ma_top"] for n in self.EXPECTED]
        bases = [epochs[n]["ma_base"] for n in self.EXPECTED]
        assert min(tops) == 485.4, f"youngest top should be Cambrian ma_top=485.4"
        assert max(bases) == 538.8, f"oldest base should be Cambrian ma_base=538.8"


class TestCambrianStages:
    """All ten Cambrian stages (ICS 2023/09)."""

    EXPECTED = (
        "Fortunian",
        "Stage 2",
        "Stage 3",
        "Stage 4",
        "Wuliuan",
        "Drumian",
        "Guzhangian",
        "Paibian",
        "Jiangshanian",
        "Stage 10",
    )

    def test_all_present(self):
        ages = _rows_by_rank("age")
        for name in self.EXPECTED:
            assert name in ages, f"Missing Cambrian stage: {name}"

    def test_all_have_valid_ma(self):
        """For every stage, ma_top < ma_base (younger boundary < older boundary)."""
        ages = _rows_by_rank("age")
        for name in self.EXPECTED:
            row = ages[name]
            assert row["ma_top"] < row["ma_base"], (
                f"{name} has ma_top={row['ma_top']} >= ma_base={row['ma_base']}"
            )

    def test_stages_parent_to_series_not_period(self):
        """Every Cambrian stage should parent to one of the 4 series,
        NOT directly to "Cambrian". The previous (broken) layout would
        be an age row whose parent == "Cambrian" — that gets caught here."""
        ages = _rows_by_rank("age")
        cambrian_series = set(TestCambrianSeries.EXPECTED)
        for name in self.EXPECTED:
            parent = ages[name]["parent"]
            assert parent in cambrian_series, (
                f"{name} should parent to a Cambrian series "
                f"({cambrian_series}); got {parent!r}"
            )

    def test_fortunian_anchors_terreneuvian(self):
        ages = _rows_by_rank("age")
        row = ages["Fortunian"]
        assert row["parent"] == "Terreneuvian"
        # ICS 2023/09: Fortunian ma_base = 538.8 (GSSP base of Cambrian)
        assert row["ma_base"] == 538.8
        assert row["ma_top"] == 529.0

    def test_wuliuan_anchors_miaolingian(self):
        ages = _rows_by_rank("age")
        row = ages["Wuliuan"]
        assert row["parent"] == "Miaolingian"
        # Wuliuan sits at the base of Miaolingian series
        assert row["ma_base"] == 509.0
        assert row["ma_top"] == 506.5

    def test_stage10_anchors_furongian(self):
        ages = _rows_by_rank("age")
        row = ages["Stage 10"]
        assert row["parent"] == "Furongian"
        # Stage 10 ma_top is the Ordovician-boundary = Cambrian ma_top
        assert row["ma_top"] == 485.4

    def test_stage_ordering_is_monotonic(self):
        """Stages ordered top→bottom (stratigraphic) must have monotonically
        decreasing ma_base (older to younger means smaller number for ma_top)."""
        ages = _rows_by_rank("age")
        # ma_top is the YOUNGER (smaller) boundary, so as we go up the
        # stratigraphic column ma_top DECREASES. Equivalently, walking
        # the EXPECTED list oldest→youngest, ma_base decreases.
        bases = [ages[n]["ma_base"] for n in self.EXPECTED]
        assert bases == sorted(bases, reverse=True), (
            f"Cambrian stage ma_base not monotonically decreasing: {bases}"
        )

    def test_each_stage_falls_in_its_series_interval(self):
        """For every Cambrian stage, the (ma_top, ma_base) tuple must lie
        inside the parent series' (ma_top, ma_base)."""
        epochs = _rows_by_rank("epoch")
        ages = _rows_by_rank("age")
        for name in self.EXPECTED:
            stage = ages[name]
            series = epochs[stage["parent"]]
            assert stage["ma_top"] >= series["ma_top"], (
                f"{name} ma_top={stage['ma_top']} < series ma_top={series['ma_top']}"
            )
            assert stage["ma_base"] <= series["ma_base"], (
                f"{name} ma_base={stage['ma_base']} > series ma_base={series['ma_base']}"
            )


# ---------------------------------------------------------------------------
# Ordovician
# ---------------------------------------------------------------------------


class TestOrdovicianSeries:
    """All three Ordovician series (ICS 2023/09)."""

    EXPECTED = ("Early Ordovician", "Middle Ordovician", "Late Ordovician")

    def test_all_present(self):
        epochs = _rows_by_rank("epoch")
        for name in self.EXPECTED:
            assert name in epochs, f"Missing Ordovician series: {name}"

    def test_parent_is_ordovician(self):
        epochs = _rows_by_rank("epoch")
        for name in self.EXPECTED:
            assert epochs[name]["parent"] == "Ordovician", (
                f"{name} should parent to Ordovician, got {epochs[name]['parent']}"
            )

    def test_ma_ranges_span_ordovician_period(self):
        epochs = _rows_by_rank("epoch")
        tops = [epochs[n]["ma_top"] for n in self.EXPECTED]
        bases = [epochs[n]["ma_base"] for n in self.EXPECTED]
        # Earliest series ma_base = Ordovician ma_base (485.4)
        assert max(bases) == 485.4
        # Latest series ma_top = Ordovician ma_top (443.8)
        assert min(tops) == 443.8

    def test_early_ordovician_ics_2023_values(self):
        """Sanity-check Early Ordovician against ICS 2023/09 (470.0–485.4 Ma)."""
        epochs = _rows_by_rank("epoch")
        row = epochs["Early Ordovician"]
        assert row["ma_top"] == 470.0
        assert row["ma_base"] == 485.4


class TestOrdovicianStages:
    """All seven Ordovician stages (ICS 2023/09)."""

    EXPECTED = (
        "Tremadocian",
        "Floian",
        "Dapingian",
        "Darriwilian",
        "Sandbian",
        "Katian",
        "Hirnantian",
    )

    def test_all_present(self):
        ages = _rows_by_rank("age")
        for name in self.EXPECTED:
            assert name in ages, f"Missing Ordovician stage: {name}"

    def test_all_have_valid_ma(self):
        ages = _rows_by_rank("age")
        for name in self.EXPECTED:
            row = ages[name]
            assert row["ma_top"] < row["ma_base"], (
                f"{name} has ma_top={row['ma_top']} >= ma_base={row['ma_base']}"
            )

    def test_stages_parent_to_series_not_period(self):
        ages = _rows_by_rank("age")
        ordovician_series = set(TestOrdovicianSeries.EXPECTED)
        for name in self.EXPECTED:
            parent = ages[name]["parent"]
            assert parent in ordovician_series, (
                f"{name} should parent to an Ordovician series "
                f"({ordovician_series}); got {parent!r}"
            )

    def test_ordovician_stage_ordering(self):
        ages = _rows_by_rank("age")
        bases = [ages[n]["ma_base"] for n in self.EXPECTED]
        # Tremadocian is oldest (base=485.4), Hirnantian youngest (base=445.2)
        assert bases == sorted(bases, reverse=True)

    def test_hirnantian_anchors_ordovician_top(self):
        """Hirnantian is the top stage of the Ordovician; its ma_top
        should equal Ordovician ma_top."""
        ages = _rows_by_rank("age")
        row = ages["Hirnantian"]
        assert row["parent"] == "Late Ordovician"
        assert row["ma_top"] == 443.8

    def test_each_stage_falls_in_its_series_interval(self):
        epochs = _rows_by_rank("epoch")
        ages = _rows_by_rank("age")
        for name in self.EXPECTED:
            stage = ages[name]
            series = epochs[stage["parent"]]
            assert stage["ma_top"] >= series["ma_top"]
            assert stage["ma_base"] <= series["ma_base"]


# ---------------------------------------------------------------------------
# Devonian
# ---------------------------------------------------------------------------


class TestDevonianSeries:
    """All three Devonian series (ICS 2023/09)."""

    EXPECTED = ("Early Devonian", "Middle Devonian", "Late Devonian")

    def test_all_present(self):
        epochs = _rows_by_rank("epoch")
        for name in self.EXPECTED:
            assert name in epochs, f"Missing Devonian series: {name}"

    def test_parent_is_devonian(self):
        epochs = _rows_by_rank("epoch")
        for name in self.EXPECTED:
            assert epochs[name]["parent"] == "Devonian", (
                f"{name} should parent to Devonian, got {epochs[name]['parent']}"
            )

    def test_ma_ranges_span_devonian_period(self):
        epochs = _rows_by_rank("epoch")
        tops = [epochs[n]["ma_top"] for n in self.EXPECTED]
        bases = [epochs[n]["ma_base"] for n in self.EXPECTED]
        assert max(bases) == 419.2  # Early Devonian ma_base = Devonian ma_base
        assert min(tops) == 358.9   # Late Devonian ma_top = Devonian ma_top

    def test_early_devonian_ics_2023_values(self):
        epochs = _rows_by_rank("epoch")
        row = epochs["Early Devonian"]
        assert row["ma_top"] == 393.3
        assert row["ma_base"] == 419.2


class TestDevonianStages:
    """All seven Devonian stages (ICS 2023/09)."""

    EXPECTED = (
        "Lochkovian",
        "Pragian",
        "Emsian",
        "Eifelian",
        "Givetian",
        "Frasnian",
        "Famennian",
    )

    def test_all_present(self):
        ages = _rows_by_rank("age")
        for name in self.EXPECTED:
            assert name in ages, f"Missing Devonian stage: {name}"

    def test_all_have_valid_ma(self):
        ages = _rows_by_rank("age")
        for name in self.EXPECTED:
            row = ages[name]
            assert row["ma_top"] < row["ma_base"], (
                f"{name} has ma_top={row['ma_top']} >= ma_base={row['ma_base']}"
            )

    def test_stages_parent_to_series_not_period(self):
        ages = _rows_by_rank("age")
        devonian_series = set(TestDevonianSeries.EXPECTED)
        for name in self.EXPECTED:
            parent = ages[name]["parent"]
            assert parent in devonian_series, (
                f"{name} should parent to a Devonian series "
                f"({devonian_series}); got {parent!r}"
            )

    def test_famennian_anchors_devonian_top(self):
        """Famennian is the top stage of the Devonian; its ma_top
        should equal Devonian ma_top."""
        ages = _rows_by_rank("age")
        row = ages["Famennian"]
        assert row["parent"] == "Late Devonian"
        assert row["ma_top"] == 358.9

    def test_lochkovian_anchors_devonian_base(self):
        """Lochkovian is the bottom stage of the Devonian; its ma_base
        should equal Devonian ma_base."""
        ages = _rows_by_rank("age")
        row = ages["Lochkovian"]
        assert row["parent"] == "Early Devonian"
        assert row["ma_base"] == 419.2

    def test_devonian_stage_ordering(self):
        ages = _rows_by_rank("age")
        bases = [ages[n]["ma_base"] for n in self.EXPECTED]
        # Lochkovian is oldest (base=419.2), Famennian youngest (base=358.9)
        assert bases == sorted(bases, reverse=True)

    def test_each_stage_falls_in_its_series_interval(self):
        epochs = _rows_by_rank("epoch")
        ages = _rows_by_rank("age")
        for name in self.EXPECTED:
            stage = ages[name]
            series = epochs[stage["parent"]]
            assert stage["ma_top"] >= series["ma_top"]
            assert stage["ma_base"] <= series["ma_base"]


# ---------------------------------------------------------------------------
# Series → period parent relationships
# ---------------------------------------------------------------------------


class TestSeriesParentRelationships:
    """Every series' parent should be the matching Paleozoic period."""

    def test_cambrian_series_parent_to_cambrian(self):
        epochs = _rows_by_rank("epoch")
        for name in TestCambrianSeries.EXPECTED:
            assert epochs[name]["parent"] == "Cambrian"

    def test_ordovician_series_parent_to_ordovician(self):
        epochs = _rows_by_rank("epoch")
        for name in TestOrdovicianSeries.EXPECTED:
            assert epochs[name]["parent"] == "Ordovician"

    def test_devonian_series_parent_to_devonian(self):
        epochs = _rows_by_rank("epoch")
        for name in TestDevonianSeries.EXPECTED:
            assert epochs[name]["parent"] == "Devonian"


# ---------------------------------------------------------------------------
# Integration with classify_age_string
# ---------------------------------------------------------------------------


class TestClassifyAgeStringIntegration:
    """End-to-end: classify_* should NOT degrade to the period level
    for the newly-added series / stages."""

    # Cambrian stages
    def test_wuliuan_does_not_collapse_to_cambrian(self):
        c = classify_age_string("Wuliuan")
        assert c.age == "Wuliuan"
        assert c.epoch == "Miaolingian"
        assert c.period == "Cambrian"
        assert c.rank == "age"
        assert c.ma_top is not None and c.ma_base is not None
        assert c.ma_top < c.ma_base

    def test_drumian(self):
        c = classify_age_string("Drumian")
        assert c.age == "Drumian"
        assert c.epoch == "Miaolingian"
        assert c.period == "Cambrian"

    def test_guzhangian(self):
        c = classify_age_string("Guzhangian")
        assert c.age == "Guzhangian"
        assert c.epoch == "Miaolingian"
        assert c.period == "Cambrian"

    def test_paibian(self):
        c = classify_age_string("Paibian")
        assert c.age == "Paibian"
        assert c.epoch == "Furongian"
        assert c.period == "Cambrian"

    def test_jiangshanian(self):
        c = classify_age_string("Jiangshanian")
        assert c.age == "Jiangshanian"
        assert c.epoch == "Furongian"
        assert c.period == "Cambrian"

    def test_fortunian(self):
        c = classify_age_string("Fortunian")
        assert c.age == "Fortunian"
        assert c.epoch == "Terreneuvian"
        assert c.period == "Cambrian"

    # Cambrian series (via modifier-stripping fallback)
    def test_terreneuvian_series(self):
        c = classify_age_string("Terreneuvian")
        assert c.epoch == "Terreneuvian"
        assert c.period == "Cambrian"

    def test_miaolingian_series(self):
        c = classify_age_string("Miaolingian")
        assert c.epoch == "Miaolingian"
        assert c.period == "Cambrian"

    def test_furongian_series(self):
        c = classify_age_string("Furongian")
        assert c.epoch == "Furongian"
        assert c.period == "Cambrian"

    # Ordovician series — BEFORE the fix these collapsed to period rank
    def test_early_ordovician_no_longer_collapses_to_period(self):
        """REGRESSION GUARD: "Early Ordovician" used to fall through to
        rank=period because no Early/Middle/Late Ordovician epoch rows
        existed; with the fix it now classifies to epoch rank with
        period=Ordovician."""
        c = classify_age_string("Early Ordovician")
        assert c.rank == "epoch", (
            f"Expected rank=epoch for 'Early Ordovician', got {c.rank}"
        )
        assert c.epoch == "Early Ordovician"
        assert c.period == "Ordovician"

    def test_middle_ordovician(self):
        c = classify_age_string("Middle Ordovician")
        assert c.rank == "epoch"
        assert c.epoch == "Middle Ordovician"
        assert c.period == "Ordovician"

    def test_late_ordovician(self):
        c = classify_age_string("Late Ordovician")
        assert c.rank == "epoch"
        assert c.epoch == "Late Ordovician"
        assert c.period == "Ordovician"

    # Ordovician stages
    def test_tremadocian(self):
        c = classify_age_string("Tremadocian")
        assert c.age == "Tremadocian"
        assert c.epoch == "Early Ordovician"
        assert c.period == "Ordovician"

    def test_floian(self):
        c = classify_age_string("Floian")
        assert c.age == "Floian"
        assert c.epoch == "Early Ordovician"
        assert c.period == "Ordovician"

    def test_darriwilian(self):
        c = classify_age_string("Darriwilian")
        assert c.age == "Darriwilian"
        assert c.epoch == "Middle Ordovician"
        assert c.period == "Ordovician"

    def test_hirnantian(self):
        c = classify_age_string("Hirnantian")
        assert c.age == "Hirnantian"
        assert c.epoch == "Late Ordovician"
        assert c.period == "Ordovician"

    # Devonian series — BEFORE the fix these also collapsed to period rank
    def test_early_devonian_no_longer_collapses_to_period(self):
        """REGRESSION GUARD: "Early Devonian" used to fall through to
        rank=period; the fix promotes it to epoch rank."""
        c = classify_age_string("Early Devonian")
        assert c.rank == "epoch", (
            f"Expected rank=epoch for 'Early Devonian', got {c.rank}"
        )
        assert c.epoch == "Early Devonian"
        assert c.period == "Devonian"

    def test_middle_devonian(self):
        c = classify_age_string("Middle Devonian")
        assert c.rank == "epoch"
        assert c.epoch == "Middle Devonian"
        assert c.period == "Devonian"

    def test_late_devonian(self):
        c = classify_age_string("Late Devonian")
        assert c.rank == "epoch"
        assert c.epoch == "Late Devonian"
        assert c.period == "Devonian"

    # Devonian stages — Famennian is the Hangenberg-Event index
    def test_lochkovian_matches_devonian_early(self):
        """Lochkovian must resolve to a Devonian Early stage (was broken)."""
        c = classify_age_string("Lochkovian")
        assert c.age == "Lochkovian"
        assert c.epoch == "Early Devonian"
        assert c.period == "Devonian"
        assert c.ma_top is not None and c.ma_base is not None
        # ICS 2023: Lochkovian ma_base ≈ 419.2 Ma
        assert 418.5 < c.ma_base < 419.5

    def test_pragian(self):
        c = classify_age_string("Pragian")
        assert c.age == "Pragian"
        assert c.epoch == "Early Devonian"
        assert c.period == "Devonian"

    def test_emsian(self):
        c = classify_age_string("Emsian")
        assert c.age == "Emsian"
        assert c.epoch == "Early Devonian"
        assert c.period == "Devonian"

    def test_eifelian(self):
        c = classify_age_string("Eifelian")
        assert c.age == "Eifelian"
        assert c.epoch == "Middle Devonian"
        assert c.period == "Devonian"

    def test_givetian(self):
        c = classify_age_string("Givetian")
        assert c.age == "Givetian"
        assert c.epoch == "Middle Devonian"
        assert c.period == "Devonian"

    def test_frasnian(self):
        c = classify_age_string("Frasnian")
        assert c.age == "Frasnian"
        assert c.epoch == "Late Devonian"
        assert c.period == "Devonian"

    def test_famennian(self):
        c = classify_age_string("Famennian")
        assert c.age == "Famennian"
        assert c.epoch == "Late Devonian"
        assert c.period == "Devonian"


class TestFindAgesInTextIntegration:
    """``find_ages_in_text`` should pick up the new series / stages
    from raw paper captions without alias plumbing."""

    def test_finds_cambrian_stages(self):
        text = (
            "Wuliuan, Drumian, and Guzhangian are Cambrian Miaolingian "
            "stages; Paibian and Jiangshanian are Furongian."
        )
        ages = find_ages_in_text(text)
        found_stages = {a.age for a in ages if a.rank == "age"}
        for must in ("Wuliuan", "Drumian", "Guzhangian", "Paibian", "Jiangshanian"):
            assert must in found_stages, (
                f"find_ages_in_text missed stage {must!r}; found={found_stages}"
            )

    def test_finds_ordovician_stages(self):
        text = "Tremadocian to Hirnantian, including Floian, Dapingian, Darriwilian, Sandbian, and Katian."
        ages = find_ages_in_text(text)
        found_stages = {a.age for a in ages if a.rank == "age"}
        for must in (
            "Tremadocian",
            "Floian",
            "Dapingian",
            "Darriwilian",
            "Sandbian",
            "Katian",
            "Hirnantian",
        ):
            assert must in found_stages

    def test_finds_devonian_stages(self):
        text = "Lochkovian, Pragian, Emsian, Eifelian, Givetian, Frasnian, Famennian."
        ages = find_ages_in_text(text)
        found_stages = {a.age for a in ages if a.rank == "age"}
        for must in (
            "Lochkovian",
            "Pragian",
            "Emsian",
            "Eifelian",
            "Givetian",
            "Frasnian",
            "Famennian",
        ):
            assert must in found_stages

    def test_finds_late_devonian_modifier(self):
        ages = find_ages_in_text("Late Devonian brachiopods are abundant.")
        # rank=epoch, period=Devonian, epoch="Late Devonian"
        match = [a for a in ages if a.rank == "epoch" and a.period == "Devonian"]
        assert match, "Late Devonian modifier did not resolve to epoch rank"
        assert match[0].epoch == "Late Devonian"


class TestChineseNamesIntegration:
    """Chinese lithostratigraphic names (统 / 期) should also resolve."""

    def test_chinese_cambrian_series(self):
        c = classify_age_string("芙蓉统")
        assert c.epoch == "Furongian"
        assert c.period == "Cambrian"

    def test_chinese_cambrian_stage(self):
        c = classify_age_string("排碧期")
        assert c.age == "Paibian"
        assert c.period == "Cambrian"

    def test_chinese_ordovician_series(self):
        c = classify_age_string("晚奥陶世")
        assert c.epoch == "Late Ordovician"
        assert c.period == "Ordovician"

    def test_chinese_devonian_series(self):
        c = classify_age_string("早泥盆世")
        assert c.epoch == "Early Devonian"
        assert c.period == "Devonian"

    def test_chinese_devonian_stage(self):
        c = classify_age_string("法门期")
        assert c.age == "Famennian"
        assert c.period == "Devonian"


class TestPeriodRowUnchanged:
    """Guard: adding rows must not perturb the existing Cambrian /
    Ordovician / Devonian period-row Ma bounds."""

    def test_cambrian_period_unchanged(self):
        rows = _rows_by_name()
        row = rows["Cambrian"]
        assert row["rank"] == "period"
        assert row["ma_top"] == 485.4
        assert row["ma_base"] == 541.0

    def test_ordovician_period_unchanged(self):
        rows = _rows_by_name()
        row = rows["Ordovician"]
        assert row["rank"] == "period"
        assert row["ma_top"] == 443.8
        assert row["ma_base"] == 485.4

    def test_devonian_period_unchanged(self):
        rows = _rows_by_name()
        row = rows["Devonian"]
        assert row["rank"] == "period"
        assert row["ma_top"] == 358.9
        assert row["ma_base"] == 419.2