"""Regression tests for audit 2026-08-05 (Fill Gaps) — Fix 6.

Fix 6: ``extract_sample_ids`` wired into ``sample_records_from_matches``.
  - audit 2026-08-05 verified that ``RunOutput.samples`` was empty
    on Beccaro 2006 because the converter used only its own local
    regex tuple, which doesn't cover the canonical
    ``sample_id_extractor.extract_sample_ids`` patterns.
  - Real downstream consumers want ``samples != []`` whenever the
    caption contains any of the canonical shapes.

The fix: ``sample_records_from_matches`` now invokes
``extract_sample_ids(caption_snippet)`` first and emits a
``SampleRecord`` per result with ``sample_id`` prefixed ``X_``
(distinct from the legacy S_/B_/R_/N_/L_/P_ prefixes emitted by
the regex pass). The legacy _SAMPLE_PATTERNS pass is kept as a
fallback for niche shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestSampleRecordsFromMatchesExtractSampleIds:
    """End-to-end: MatchResult → SampleRecord via sample_records_from_matches
    using extract_sample_ids (X_ prefix) and the legacy regex fallback.
    """

    def _make_match(self, paper_id, figure_id, caption):
        from rlpe.types import MatchResult

        return MatchResult(
            paper_id=paper_id,
            figure_id=figure_id,
            panel_id="1",
            species="Genus species",
            panel_path="/tmp/1.png",
            bbox=None,
            confidence=0.6,
            label_text="1",
            caption_snippet=caption,
            ocr_text=None,
            metadata={"page_index": 1},
            paper_metadata=None,
        )

    def test_legacy_sample_keyword_emitted(self):
        from rlpe.converters import sample_records_from_matches

        m = self._make_match(
            "p1",
            "f1",
            "Plate 1. Sample CH-4. figs 1-2. Genus species",
        )
        recs = sample_records_from_matches([m])
        # ``extract_sample_ids`` matches "Sample CH-4" via the legacy
        # helper shape (line 376-424 in sample_id_extractor.py). The
        # fix wraps it with X_ prefix.
        ids = {r["sample_id"] for r in recs}
        assert any(sid.startswith("X_") for sid in ids), (
            f"expected X_ prefix from extract_sample_ids; got {ids}"
        )

    def test_empty_caption_yields_zero_records(self):
        from rlpe.converters import sample_records_from_matches

        m = self._make_match("p1", "f1", "")
        recs = sample_records_from_matches([m])
        assert recs == []

    def test_pure_species_caption_yields_zero_records(self):
        from rlpe.converters import sample_records_from_matches

        # Beccaro Plate 1: no Sample / Loc / ID tokens — the function
        # should NOT emit spurious records.
        m = self._make_match(
            "p1",
            "f1",
            "Plate 1\n\nScanning electron micrographs of the most "
            "important radiolarians used for the biostratigraphy",
        )
        recs = sample_records_from_matches([m])
        assert recs == [], f"got unexpected records: {recs}"

    def test_dedup_across_multiple_matches(self):
        from rlpe.converters import sample_records_from_matches

        # Two matches with the same sample-id text should produce
        # ONE SampleRecord (keyed by (paper_id, sample_id)).
        m1 = self._make_match(
            "p1",
            "f1",
            "Plate 1. Sample CH-4. figs 1-2.",
        )
        m2 = self._make_match(
            "p1",
            "f1",
            "Plate 1. Sample CH-4. figs 1-2.",
        )
        recs = sample_records_from_matches([m1, m2])
        # At most one record per (paper_id, sample_id) pair.
        keys = [(r["paper_id"], r["sample_id"]) for r in recs]
        assert len(keys) == len(set(keys))

    def test_record_carries_paper_and_figure_id(self):
        from rlpe.converters import sample_records_from_matches

        m = self._make_match(
            "p_paper",
            "f_fig",
            "Plate 1. Sample CH-4.",
        )
        recs = sample_records_from_matches([m])
        assert recs, "expected at least one SampleRecord"
        rec = next(r for r in recs if r["sample_id"].startswith("X_"))
        assert rec["paper_id"] == "p_paper"
        assert rec["figure_id"] == "f_fig"
        assert rec["evidence_text"]
        assert rec["page_index"] == 1


class TestExtractSampleIdsPure:
    """Sanity-check that extract_sample_ids itself still works the way
    the converter now relies on.
    """

    def test_sample_keyword_match(self):
        from rlpe.sample_id_extractor import extract_sample_ids

        out = extract_sample_ids("Sample CH-4")
        assert any(s.value == "CH-4" for s in out)

    def test_sample_with_digit_only(self):
        # Plain "Sample 12" form.
        from rlpe.sample_id_extractor import extract_sample_ids

        out = extract_sample_ids("Sample 12")
        assert any(s.value == "12" for s in out)

    def test_id_dash_match(self):
        from rlpe.sample_id_extractor import extract_sample_ids

        out = extract_sample_ids("ID-203")
        assert any(s.value == "203" for s in out)
