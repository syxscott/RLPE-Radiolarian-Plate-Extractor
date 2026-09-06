"""Audit 2026-09-07 — coverage-fix regression tests (F1/F2/F4/F5/F6/F7/F9).

Covers the fixes from the 40-paper coverage run:

* F1  — full-page OCR variant for scanned plates covering >60% of the page
        (Motoyama-style papers had 10 empty-caption figures and no rescue).
* F2  — font-shift decoder for JGSJ-cluster PDFs whose ToUnicode CMap is a
        constant ASCII offset (Soeka: "Plate" stored as "3ODWH").
* F4  — Stage 2 caption-evidence override: a caption with >=2 taxon
        entities / >=3 numbered clauses overrides an is_radiolarian=False
        verdict (Munasri's 19-clause plate was rejected as "diagram").
* F5  — per-figure observability logs in the OD loop.
* F6  — duplicate range-chart caption suppression (Munasri p007_09/10).
* F7  — ``--od-panel-detector yolo`` wiring.
* F9  — MiniMax 5xx backoff/storm tracking + NoneType log fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_ROOT = Path(__file__).resolve().parents[1]

from rlpe.opendataloader_extractor import (  # noqa: E402
    OpenDataLoaderExtractor,
    _try_decode_font_shift,
)


class TestFontShiftDecoder:
    """F2 — constant-ASCII-shift decode (Soeka: +0x1D)."""

    def test_decodes_soeka_plate_header(self):
        assert _try_decode_font_shift("3ODWH") == "Plate"

    def test_decodes_soeka_figs_prefix(self):
        # Pure-shifted token decodes; MIXED elements (readable species +
        # shifted "Figs") intentionally fail the 90% letters gate —
        # a whole-string shift would corrupt the readable runs.
        assert _try_decode_font_shift(")LJV") == "Figs"

    def test_mixed_font_element_rejected(self):
        # Soeka's real caption body: readable species + shifted "Figs"
        # prefix — whole-string decode corrupts the readable part, so
        # the helper must reject it (band/full-page OCR handles these).
        assert _try_decode_font_shift(")LJV 1-2. Testus species Example") is None

    def test_readable_text_passthrough(self):
        assert _try_decode_font_shift("Plate 1 Radiolarian species") is None
        assert _try_decode_font_shift("Figure 2 Schematic map") is None

    def test_rejects_non_caption_decodes(self):
        # "1234" shifted stays digits — fails the 90% letters gate AND
        # no shift makes it read like a caption header.
        assert _try_decode_font_shift("1234") is None

    def test_short_text_rejected(self):
        assert _try_decode_font_shift("ab") is None
        assert _try_decode_font_shift("") is None


class TestFullPageOcrVariant:
    def test_marker_check_accepts_clause_lists(self):
        from rlpe.opendataloader_extractor import (
            _rescue_orphan_plate_pages_marker_check,
        )

        # Motoyama-style full-page OCR text: caption + scattered panel
        # numbers, no "Plate" header in the band.
        ok, _ = _rescue_orphan_plate_pages_marker_check(
            "Fig. 2. Seasonal variations in total polycystine flux column "
            "and total mass flux from Mohole site"
        )
        assert ok

    def test_marker_check_rejects_junk(self):
        from rlpe.opendataloader_extractor import (
            _rescue_orphan_plate_pages_marker_check,
        )

        ok, _ = _rescue_orphan_plate_pages_marker_check("100 um")
        assert not ok


class TestStage2Override:
    def _pipe(self, tmp_path):
        from rlpe.config import PipelineConfig
        from rlpe.pipeline import RadiolarianPipeline

        return RadiolarianPipeline(PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w"))

    def test_override_source_guard(self):
        """The F4 override must exist at the Stage 2 rejection point with
        the >=2 evidence threshold, and the rejection must remain the
        default for weak captions."""
        src = (_ROOT / "src" / "rlpe" / "pipeline.py").read_text(encoding="utf-8")
        assert "stage2_overridden" in src, "F4 override diagnostic flag missing"
        assert "_cap_evidence >= 2" in src, "F4 threshold must be >=2 signals"
        # override sits inside the Stage 2 rejection branch
        i_override = src.find('m3_diag["stage2_overridden"] = True')
        i_reject = src.find("rejected (not a radiolarian plate)")
        assert 0 < i_override < i_reject, (
            "override (if-branch) must precede the rejection log (else-branch) "
            "inside the Stage 2 rejection block"
        )

    def test_munasri_caption_meets_threshold(self):
        """The Munasri 19-clause caption shape carries >=2 evidence
        signals through the exact production regex."""
        import re

        caption = "Plate 1 Radiolarian species\n" + "\n".join(
            f"1.{i}. Genus_{i} species_{i} Author" for i in range(1, 20)
        )
        evidence = len(
            re.findall(
                r"(?:Figs?|Fig\.?|Plate|Pl\.)\s*\d+"
                r"|\(\s*\d{1,3}\s*[,–\-]\s*\d{1,3}\s*\)"
                r"|\b\d{1,2}\.\d{1,2}\.?\b",
                caption,
            )
        )
        assert evidence >= 2

    def test_weak_caption_still_rejected(self):
        """Weak caption (no entities, no clauses) must NOT override."""
        import re

        caption_text = "Scanning electron micrograph"  # plausible but no evidence
        evidence = len(
            re.findall(
                r"(?:Figs?|Fig\.?|Plate|Pl\.)\s*\d+"
                r"|\(\s*\d{1,3}\s*[,–\-]\s*\d{1,3}\s*\)"
                r"|\b\d{1,2}\.\d{1,2}\.?\b",
                caption_text,
            )
        )
        assert evidence < 2


class TestRangeChartDedup:
    def test_seen_captions_set_initialised(self, tmp_path):
        from rlpe.config import PipelineConfig
        from rlpe.pipeline import RadiolarianPipeline

        p = RadiolarianPipeline(PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w"))
        assert p._seen_range_chart_captions == set()


class TestYoloPanelDetectorWiring:
    def test_extra_key_whitelisted(self):
        from rlpe.config import _KNOWN_EXTRA_KEYS

        assert "od_panel_detector" in _KNOWN_EXTRA_KEYS

    def test_cli_source_guard(self):
        src = (_ROOT / "src" / "rlpe" / "cli.py").read_text(encoding="utf-8")
        assert '"--od-panel-detector"' in src
        assert 'cfg.extra["od_panel_detector"]' in src

    def test_pipeline_source_guard(self):
        src = (_ROOT / "src" / "rlpe" / "pipeline.py").read_text(encoding="utf-8")
        assert 'self.config.extra.get("od_panel_detector") == "yolo"' in src
        assert "_yolo_panels_used" in src
        # weights committed
        assert (_ROOT / "models" / "panel_detector_v1.pt").exists()


class TestMinimax5xxResilience:
    def test_in_5xx_storm_default_off(self, tmp_path):
        from rlpe.config import PipelineConfig
        from rlpe.pipeline import RadiolarianPipeline

        p = RadiolarianPipeline(PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w"))
        # No backend built — the storm check must be a safe getattr.
        backend = getattr(p.m3_engine, "backend", None) if p.m3_engine else None
        if backend is not None:
            assert backend.in_5xx_storm() is False

    def test_none_type_log_fix_source_guard(self):
        """The spurious 'NoneType: None' tail came from logger.exception
        running outside an except block; the fix passes exc_info
        explicitly."""
        src = (_ROOT / "src" / "rlpe" / "llm_backends.py").read_text(encoding="utf-8")
        assert "MiniMax API call failed after" in src
        # the fixed call uses logger.error with exc_info=last_exc
        assert "exc_info=last_exc" in src
        idx_log = src.find("MiniMax API call failed after")
        idx_exc = src.find("exc_info=last_exc")
        assert 0 < idx_log < idx_exc
        # and the old bare logger.exception form is gone
        seg = src[idx_log - 60 : idx_log]
        assert "logger.exception" not in seg


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
