"""Regression tests for audit 2026-08-01 batch W1 — C3 sample_id_extractor.py:85 stopword filter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.sample_id_extractor import extract_sample_ids


class TestSampleStopwordFilter:
    def test_from_skipped(self):
        result = extract_sample_ids("Samples from Tunisia")

        assert all(sample_id.value.casefold() != "from" for sample_id in result)

    def test_at_skipped(self):
        result = extract_sample_ids("Sample at locality X")

        assert all(sample_id.value.casefold() != "at" for sample_id in result)

    def test_in_skipped(self):
        result = extract_sample_ids("Sample in Sicily")

        assert all(sample_id.value.casefold() != "in" for sample_id in result)

    def test_real_id_not_filtered(self):
        result = extract_sample_ids("Sample TUN-12")

        assert any(sample_id.value == "TUN-12" for sample_id in result)

    def test_case_insensitive(self):
        result = extract_sample_ids("Samples FROM Italy")

        assert all(sample_id.value.casefold() != "from" for sample_id in result)
