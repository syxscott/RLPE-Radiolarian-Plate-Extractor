"""2026-09-14 — Beccaro Plate-1 regression tests.

Three production failures on Beccaro 2006 (a 35-panel numbered SEM
plate):

1. Crop↔label scramble: Phase 67 bbox recovery paired caption labels
   with OpenCV segments by reading-order RANK distance (rows sorted by
   label vs segments sorted (y, x)); on a jittered SEM grid the topmost
   segment was printed panel 6, so the crop saved as "panel 1" showed
   printed number 6. Fix: pair by the PRINTED number OCR'd on the plate
   (pass 0); rank distance only as a flagged fallback.
2. Missing trailing labels: 35 rows vs 32 segments silently left
   labels 33-35 unpaired.
3. Age mismatch: the caption item "1 – Species AUTHOR, CV 60, UAZ A,
   x250" cites sample + unit; the unit's age lives in the prose
   ("UAZ A is assigned to early?-mid Bathonian - early Callovian pars").
   The proximity geology linker stamped a paper-level "Early Jurassic"
   record onto Middle-Jurassic panels. Fix: per-item sample/unit
   capture + paper-level unit→age resolution prepended to geology_links.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.geology_extraction import (  # noqa: E402
    extract_caption_item_context,
    extract_unit_age_map_regex,
    parse_unit_resolution_response,
)

# ---------------------------------------------------------------------------
# 1. printed-number OCR pairing (pass 0)
# ---------------------------------------------------------------------------


class _Seg:
    def __init__(self, bbox):
        self.bbox = bbox


@pytest.fixture
def pipe(tmp_path):
    from rlpe.config import PipelineConfig
    from rlpe.pipeline import RadiolarianPipeline

    with (
        patch("rlpe.pipeline.GrobidClient"),
        patch("rlpe.pipeline.OCRBackend"),
        patch("rlpe.pipeline.TaxonRecognizer"),
        patch("rlpe.pipeline.PanelSegmenter"),
    ):
        p = RadiolarianPipeline(PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w"))
        yield p


def _rows(labels):
    return [
        {
            "paper_id": "p",
            "figure_id": "f",
            "panel_id": str(lb),
            "species": f"s{lb}",
            "bbox": None,
            "panel_path": None,
            "metadata": {},
        }
        for lb in labels
    ]


class TestPrintedNumberPairing:
    def test_ocr_reads_beat_rank_order(self, pipe):
        """A jittered grid where (y, x) order ≠ label order: the
        topmost segment is printed 6, the second is printed 1. Pass 0
        must pair by the READ number, not by rank."""
        # segments: (x, y, w, h) — seg0 top-RIGHT, seg1 top-LEFT
        segmented = [_Seg((700, 10, 100, 80)), _Seg((20, 12, 100, 80))]
        region = np.zeros((100, 820, 3), dtype=np.uint8)
        tok6 = MagicMock(text="6", confidence=0.9, bbox=(730, 15, 20, 20))
        tok1 = MagicMock(text="1", confidence=0.9, bbox=(30, 18, 15, 20))
        pipe.ocr = MagicMock()
        pipe.ocr.recognize_panel.return_value = [tok6, tok1]

        got = pipe._ocr_printed_panel_assignments(region, segmented, {1, 6})
        assert got[1][0] == 1, "label 1 must take the seg whose printed read is 1 (left)"
        assert got[6][0] == 0, "label 6 must take the seg whose printed read is 6 (right)"

    def test_full_recovery_pairs_rows_by_read_number(self, pipe):
        """End-to-end through _recover_bboxes_via_segmentation: rows
        labelled 1/6 against segments whose visual order would
        rank-mismatch — printed-number pairing wins."""
        segmented = [_Seg((700, 10, 100, 80)), _Seg((20, 12, 100, 80))]
        pipe.segmenter = MagicMock()
        pipe.segmenter._segment_with_opencv.return_value = segmented
        region = np.zeros((100, 820, 3), dtype=np.uint8)
        pipe.ocr = MagicMock()
        pipe.ocr.recognize_panel.return_value = [
            MagicMock(text="6", confidence=0.9, bbox=(730, 15, 20, 20)),
            MagicMock(text="1", confidence=0.9, bbox=(30, 18, 15, 20)),
        ]
        rows = _rows([6, 1])  # row order irrelevant — pass 0 keys on label
        out = pipe._recover_bboxes_via_segmentation(rows, region, "p", "f")
        by_label = {r["panel_id"]: r for r in out}
        assert by_label["1"]["bbox"] == [20, 12, 100, 80], (
            "label 1's crop must sit at the segment whose printed read is 1"
        )
        assert by_label["6"]["bbox"] == [700, 10, 100, 80]
        md1 = by_label["1"]["metadata"]
        assert md1["association_method"] == "printed_number_ocr"
        assert md1["printed_label_read"] == "1"

    def test_fallback_rank_pairing_flagged_for_review(self, pipe):
        """Digits EXIST on the plate but pairing was discarded (coverage
        gate) — the historical rank pairing runs and the rows are
        flagged for review: the order really is unverifiable."""
        pipe.segmenter = MagicMock()
        pipe.segmenter._segment_with_opencv.return_value = [
            _Seg((10 * i, 10, 50, 50)) for i in range(4)
        ]
        region = np.zeros((100, 300, 3), dtype=np.uint8)
        pipe.ocr = MagicMock()
        pipe.ocr.backend = "paddleocr"  # the known-broken primary
        # one stray read for 4 wanted labels -> below the coverage gate
        pipe.ocr.recognize_panel.return_value = [
            MagicMock(text="1", confidence=0.99, bbox=(15, 15, 20, 20))
        ]
        # the EasyOCR fallback construction is gated to real paddle
        # primaries and monkeypatched out here — a real model load in a
        # unit test would be both slow and native-crash-prone.
        pipe._ocr_digit_fallback_backend = False
        rows = _rows([1, 2, 3, 4])
        out = pipe._recover_bboxes_via_segmentation(rows, region, "p", "f")
        for r in out:
            assert r["metadata"]["association_method"] == "positional_fallback"
            assert "positional_panel_association" in r["metadata"]["review_reasons"]

    def test_fallback_no_digits_stamps_caption_order(self, pipe):
        """P2a: the full-plate read (primary + fallback) found NO digit
        tokens at all — the plate carries no printed numbers (Bragin
        2025 shape). Caption/reading order is the only possible pairing:
        expected behaviour, no review alarm."""
        pipe.segmenter = MagicMock()
        pipe.segmenter._segment_with_opencv.return_value = [
            _Seg((10, 10, 50, 50)),
            _Seg((200, 10, 50, 50)),
        ]
        region = np.zeros((100, 300, 3), dtype=np.uint8)
        pipe.ocr = MagicMock()
        pipe.ocr.backend = "paddleocr"
        pipe.ocr.recognize_panel.return_value = []  # nothing readable anywhere
        pipe._ocr_digit_fallback_backend = False
        rows = _rows([1, 2])
        out = pipe._recover_bboxes_via_segmentation(rows, region, "p", "f")
        for r in out:
            assert r["metadata"]["association_method"] == "caption_order_positional"
            assert not r["metadata"].get("needs_review")

    def test_scale_bar_digit_does_not_hijack_pairing(self, pipe):
        """A scale-bar '10 um' token inside another panel's segment must
        not claim label 10 for that segment: the conflicting segment
        (two different wanted labels read inside it) is distrusted."""
        # seg0 = printed panel 7's specimen WITH a scale bar reading 10;
        # seg1 = printed panel 10's specimen with its true label.
        segmented = [_Seg((100, 100, 200, 200)), _Seg((400, 100, 200, 200))]
        region = np.zeros((400, 700, 3), dtype=np.uint8)
        pipe.ocr = MagicMock()
        pipe.ocr.backend = "paddleocr"
        pipe.ocr.recognize_panel.return_value = [
            MagicMock(text="10", confidence=0.99, bbox=(180, 250, 40, 25)),  # scale bar in seg0
            MagicMock(text="7", confidence=0.9, bbox=(110, 110, 25, 25)),  # true label, seg0
            MagicMock(text="10", confidence=0.9, bbox=(410, 110, 40, 25)),  # true label, seg1
        ]
        got = pipe._ocr_printed_panel_assignments(region, segmented, {7, 10})
        # seg0 is conflicted (reads {10, 7}) -> distrusted; label 10 must
        # NOT be assigned to seg0. Label 10 takes its clean seg1.
        assert got[10][0] == 1
        assert 7 not in got, "conflicted segment must not claim label 7"

    def test_coverage_gate_discards_noise_reads(self, pipe):
        """Only 1 of 4 wanted labels read cleanly -> below the 50%
        consistency gate -> pass 0 abandoned entirely (return {}), so
        1-2 stray digit reads cannot scramble a whole plate."""
        segmented = [_Seg((10 * i, 10, 50, 50)) for i in range(4)]
        region = np.zeros((100, 300, 3), dtype=np.uint8)
        pipe.ocr = MagicMock()
        pipe.ocr.backend = "paddleocr"
        pipe.ocr.recognize_panel.return_value = [
            MagicMock(text="1", confidence=0.99, bbox=(15, 15, 20, 20))
        ]
        got = pipe._ocr_printed_panel_assignments(region, segmented, {1, 2, 3, 4})
        assert got == {}

    def test_more_labels_than_segments_keeps_species_rows(self, pipe):
        """35 labels vs 32 segments: unpaired labels keep their rows
        (species + caption) with no bbox — never silently re-ranked or
        dropped."""
        pipe.segmenter = MagicMock()
        pipe.segmenter._segment_with_opencv.return_value = [_Seg((10, 10, 50, 50))]
        region = np.zeros((100, 300, 3), dtype=np.uint8)
        pipe.ocr = MagicMock()
        pipe.ocr.recognize_panel.return_value = [
            MagicMock(text="1", confidence=0.9, bbox=(15, 15, 15, 20))
        ]
        rows = _rows([1, 33, 34, 35])
        out = pipe._recover_bboxes_via_segmentation(rows, region, "p", "f")
        by_label = {r["panel_id"]: r for r in out}
        assert by_label["1"]["bbox"] == [10, 10, 50, 50]
        for lb in ("33", "34", "35"):
            assert by_label[lb]["species"] == f"s{lb}", "row must survive"
            assert by_label[lb]["bbox"] is None


# ---------------------------------------------------------------------------
# 2. caption-item context + unit→age resolution
# ---------------------------------------------------------------------------


BECCARO_CAPTION = (
    "Plate 1\nScanning electron micrographs. The codes are: CV-Coston delle Vette.\n"
    "1 - Eucyrtidiellum unumaense dentatum BAUMGARTNER, CV 60, UAZ A, x250\n"
    "12 - Podobursa vannae BECCARO, CI 4, UAZ C, x100\n"
    "27 - Pseudoeucyrtis sp. B sensu WIDZ 1991, SA 0.35, UAZ E, x100\n"
    "35 - Napora boneti PESSAGNO, IN 30, UAZ F, x150"
)


class TestCaptionItemContext:
    def test_sample_and_unit_captured(self):
        ctx = extract_caption_item_context(BECCARO_CAPTION, "1")
        assert ctx["sample_code"] == "CV 60"
        assert ctx["section_code"] == "CV"
        assert ctx["unit_token"] == "UAZ A"

    def test_decimal_sample_and_unit(self):
        ctx = extract_caption_item_context(BECCARO_CAPTION, "27")
        assert ctx["sample_code"] == "SA 0.35"
        assert ctx["unit_token"] == "UAZ E"

    def test_trailing_item(self):
        ctx = extract_caption_item_context(BECCARO_CAPTION, "35")
        assert ctx["sample_code"] == "IN 30"
        assert ctx["unit_token"] == "UAZ F"

    def test_unknown_label_returns_nones(self):
        ctx = extract_caption_item_context(BECCARO_CAPTION, "99")
        assert ctx == {"sample_code": None, "section_code": None, "unit_token": None}


class TestUnitAgeResolution:
    SECTIONS = [
        {
            "title": "Biostratigraphy",
            "text": "UAZ A is assigned to early?-mid Bathonian - early Callovian pars "
            "thanks to ammonites. UAZ B (early Callovian pars - early Oxfordian) "
            "is present at Cava Vianini.",
        }
    ]

    def test_regex_assigned_pattern(self):
        m = extract_unit_age_map_regex(self.SECTIONS, {"UAZ A", "UAZ B"})
        assert "Bathonian" in m["UAZ A"]["age_text"]
        assert "Callovian" in m["UAZ A"]["age_text"]
        assert "Oxfordian" in m["UAZ B"]["age_text"]

    def test_wanted_filter_drops_uncited_units(self):
        m = extract_unit_age_map_regex(self.SECTIONS, {"UAZ F"})
        assert m == {}

    def test_llm_response_parsed_and_filtered(self):
        out = {
            "units": {
                "UAZ A": {
                    "age_text": "early?-mid Bathonian - early Callovian pars",
                    "chronostratigraphy": "Middle Jurassic",
                    "ma_top": 166.1,
                    "ma_base": 168.2,
                },
                "UAZ ZZZ": {"age_text": "invented"},
            }
        }
        parsed = parse_unit_resolution_response(out, {"UAZ A"})
        assert set(parsed) == {"UAZ A"}
        assert parsed["UAZ A"]["chronostratigraphy"] == "Middle Jurassic"
        assert parsed["UAZ A"]["ma_base"] == 168.2


class TestRowAttachment:
    def test_unit_link_prepended_and_wins_best_link(self, pipe):
        """The synthetic caption-unit link must PREPEND to
        geology_links so _finalize_rows' best-link pick (first link
        with age/locality) resolves the row's age from the unit, not
        from the proximity dump."""
        pipe._paper_unit_geology = {
            "p": {
                "UAZ A": {
                    "age_text": "early?-mid Bathonian - early Callovian pars",
                    "chronostratigraphy": "Middle Jurassic",
                }
            }
        }
        # P1: the sample map is cached alongside the unit map — seed it
        # so the short-circuit fires without an LLM.
        pipe._paper_sample_unit_geology = {"p": {}}
        pipe._paper_sections_cache = {"p": []}
        pipe._paper_unit_geology_sections = {
            "p": {"CV": "Coston delle Vette, Southern Alps, Italy"}
        }
        rows = _rows([1])
        rows[0]["metadata"]["geology_links"] = [
            {"label": "proximity", "age": "Early Jurassic", "ma_top": 174.7, "ma_base": 201.4}
        ]
        out = pipe._attach_caption_unit_geology(rows, "p", {"f": BECCARO_CAPTION})
        links = out[0]["metadata"]["geology_links"]
        assert links[0]["label"] == "caption_unit_geology"
        assert links[0]["age"] == "early?-mid Bathonian - early Callovian pars"
        assert links[0]["chronostratigraphy"] == "Middle Jurassic"
        assert links[0]["biozone"] == "UAZ A"
        assert links[0]["locality"] == "Coston delle Vette, Southern Alps, Italy"
        # the proximity record stays, demoted
        assert links[1]["age"] == "Early Jurassic"

    def test_unresolved_unit_flags_review(self, pipe):
        pipe._paper_unit_geology = {"p": {}}
        pipe._paper_sections_cache = {"p": []}
        pipe._paper_unit_geology_sections = {"p": {}}
        rows = _rows([1])
        out = pipe._attach_caption_unit_geology(rows, "p", {"f": BECCARO_CAPTION})
        md = out[0]["metadata"]
        assert md["geology_links"][0]["label"] == "caption_unit_geology"
        assert "unresolved_unit_age" in md["review_reasons"]
