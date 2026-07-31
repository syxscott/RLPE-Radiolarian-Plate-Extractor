"""Regression tests for audit 2026-07-31 batch 2 (age classification).

Covers:
  - Chinese lithostratigraphic names (上二叠统 / 早白垩世 / 中侏罗世…)
  - Modifier + period maps to the correct epoch (Late Permian →
    Lopingian) with epoch-level Ma bounds
  - Range forms ("Late Jurassic to Early Cretaceous",
    "Middle to Late Jurassic", "late Valanginian to early Hauterivian")
    resolve to union intervals
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestChineseAgeNames:
    def test_shang_er_die_tong(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("上二叠统")
        assert cls.period == "Permian"
        assert cls.epoch == "Lopingian"
        assert cls.ma_top == 251.9 and cls.ma_base == 259.51

    def test_zhong_er_die_tong(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("中二叠统")
        assert cls.epoch == "Guadalupian"

    def test_zao_bai_e_shi(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("早白垩世")
        assert cls.epoch == "Lower Cretaceous"
        assert cls.period == "Cretaceous"

    def test_zhong_zhu_luo_shi(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("中侏罗世")
        assert cls.epoch == "Middle Jurassic"

    def test_wan_san_die_shi(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("晚三叠世")
        assert cls.epoch == "Late Triassic"


class TestModifierToEpoch:
    def test_late_permian_is_lopingian(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("Late Permian")
        assert cls.epoch == "Lopingian"
        # Lopingian midpoint ≈ 255.7 Ma (was the full-Permian 275.4)
        assert cls.ma_mid is not None and abs(cls.ma_mid - 255.7) < 1.0

    def test_early_permian_is_cisuralian(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("Early Permian")
        assert cls.epoch == "Cisuralian"

    def test_upper_cretaceous_epoch_bounds(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("Upper Cretaceous")
        assert cls.epoch == "Upper Cretaceous"
        assert cls.ma_mid is not None and abs(cls.ma_mid - 83.25) < 1.0

    def test_early_jurassic_epoch_bounds(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("Early Jurassic")
        assert cls.epoch == "Early Jurassic"
        assert cls.ma_top == 174.7 and cls.ma_base == 201.4

    def test_plain_period_unchanged(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("Permian")
        assert cls.rank == "period"
        assert cls.period == "Permian"
        assert cls.epoch is None


class TestAgeRanges:
    def test_jurassic_cretaceous_range(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("Late Jurassic to Early Cretaceous")
        assert cls.rank == "range"
        assert cls.confidence > 0.9
        # union: Late Jurassic (145-161.5) ∪ Early Cretaceous (100.5-145)
        assert cls.ma_top == 100.5
        assert cls.ma_base == 161.5

    def test_middle_to_late_jurassic(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("Middle to Late Jurassic")
        assert cls.rank == "range"
        assert cls.period == "Jurassic"
        assert cls.ma_top == 145.0
        assert cls.ma_base == 174.7

    def test_stage_level_range(self):
        from rlpe.stratigraphy import classify_age_string

        cls = classify_age_string("late Valanginian to early Hauterivian")
        assert cls.rank == "range"
        # union of both stages (no sub-stage data → full stage bounds)
        assert abs(cls.ma_top - 125.77) < 1e-6
        assert abs(cls.ma_base - 139.8) < 1e-6

    def test_find_ages_in_text_handles_range_text(self):
        from rlpe.stratigraphy import find_ages_in_text

        cls_list = find_ages_in_text(
            "Radiolarians from the Late Jurassic to Early Cretaceous interval"
        )
        assert cls_list, "range text should yield classifications"
        # free-text scanning finds both ends as epoch hits
        epochs = {c.epoch for c in cls_list}
        assert "Late Jurassic" in epochs
        assert "Early Cretaceous" in epochs
