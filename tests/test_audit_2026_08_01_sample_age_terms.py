"""Regression tests for audit 2026-08-01 batch W2 — sample_id_extractor D15 Induan fix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.sample_id_extractor import _AGE_TERMS, _AGE_TERMS_SET, extract_age_terms


class TestAgeTerms:
    def test_induan_now_matched(self):
        assert extract_age_terms("Induan-Olenekian boundary beds") == [
            "Induan",
            "Olenekian",
        ]

    def test_induan_alone(self):
        assert extract_age_terms("Induan limestone") == ["Induan"]

    def test_typo_no_longer_present(self):
        assert all("inderbian" not in terms for terms in (_AGE_TERMS, _AGE_TERMS_SET))
