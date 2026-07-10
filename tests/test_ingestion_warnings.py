"""Tests for ingestion-failure warning injection (Round 6 fix).

The previous pipeline silently dropped failed PDFs:
  - OD failed (e.g. corrupt PDF, OD subprocess crash) → GROBID fallback
  - GROBID failed → 0 rows, 0 warnings
  - User has no diagnostic signal.

The fix: when OD or GROBID fails, inject a stub row carrying
``ingestion_warning=True`` in metadata. The downstream
``run_output_from_provenance`` consumes this via the existing
``warnings_from_matches`` path and emits a ``WarningRecord``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

HAS_CV2 = True
try:
    import cv2  # noqa: F401
except Exception:
    HAS_CV2 = False

requires_cv2 = pytest.mark.skipif(not HAS_CV2, reason="pipeline import requires cv2")


class TestIngestionWarningSourceGuard:
    """Source guard — _process_one_pdf_od must inject an
    ingestion_failed warning row when OD fails."""

    def test_od_failure_injects_warning_stub(self):
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
        text = path.read_text(encoding="utf-8")
        # Locate _process_one_pdf_od
        marker = "def _process_one_pdf_od("
        i = text.find(marker)
        assert i > 0
        next_def = text.find("\n    def ", i + 1)
        assert next_def > 0
        body = text[i:next_def]
        # After OD fails, the function must append a stub with
        # extraction_source="od_failed" so run_output.warnings picks
        # it up.
        assert "od_failed" in body, "OD-failed fallback must inject a warning stub (Audit P1-5)"
        # Specifically the stub is appended (not just logged).
        assert "ingestion_warning" in body, (
            "Ingestion warning stub must carry ingestion_warning=True"
        )

    def test_grobid_failure_injects_warning_stub(self):
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
        text = path.read_text(encoding="utf-8")
        # Phase 29: the GROBID code path was split into
        # ``_process_one_pdf_grobid`` (cycle guard) + an inner helper
        # ``_process_one_pdf_grobid_inner`` (the original body). The
        # ``grobid_failed`` warning stub lives in the inner helper, so
        # the source guard must look there.
        marker = "def _process_one_pdf_grobid_inner("
        i = text.find(marker)
        assert i > 0, (
            "Phase 29 split should leave _process_one_pdf_grobid_inner "
            "as the body that holds the grobid_failed warning stub."
        )
        next_def = text.find("\n    def ", i + 1)
        assert next_def > 0
        body = text[i:next_def]
        # GROBID failure must ALSO inject a warning stub so users see
        # the failure (not just OD failures).
        assert "grobid_failed" in body, "GROBID-failed path must inject a warning stub (Audit P1-5)"
