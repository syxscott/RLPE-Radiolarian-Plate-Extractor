"""Regression tests for audit 2026-07-31 batch 5 (Web/GUI correctness).

Covers:
  - review corrections are actually consumed by the pipeline
  - API sandbox env var (RLPE_API_TEST_TMP) isolates test runs
  - GUI source guards for the batch state machine fixes (PySide6 may
    be absent, so these are static checks of the wiring)
  - settings save validation happens before any writes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_GUI = _SRC / "rlpe" / "gui"


class TestReviewCorrections:
    def test_corrections_applied(self, tmp_path):
        from rlpe.config import PipelineConfig
        from unittest.mock import patch

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
            from rlpe.pipeline import RadiolarianPipeline

            pipe = RadiolarianPipeline(cfg)
            # write a corrections file next to the work dir
            corr_dir = tmp_path / "corrections"
            corr_dir.mkdir()
            (corr_dir / "corrections.jsonl").write_text(
                json.dumps(
                    {
                        "paper_id": "p1",
                        "figure_id": "f1",
                        "panel_path": None,
                        "corrected_species": "Unuma echinatus CORRECTED",
                        "corrected_label": "5",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = [
                {
                    "paper_id": "p1",
                    "figure_id": "f1",
                    "panel_id": "1",
                    "species": "Unuma echinatus",
                    "panel_path": "/x/panel_01.png",
                },
                {
                    "paper_id": "p2",
                    "figure_id": "f1",
                    "panel_id": "1",
                    "species": "Other species",
                    "panel_path": "/x/panel_01.png",
                },
            ]
            out = pipe._apply_review_corrections(rows)
            assert out[0]["species"] == "Unuma echinatus CORRECTED"
            assert out[0]["panel_id"] == "5"
            assert out[0]["metadata"]["review_corrected"] is True
            # non-matching paper untouched
            assert out[1]["species"] == "Other species"
            assert "review_corrected" not in out[1].get("metadata", {})

    def test_no_corrections_file_is_noop(self, tmp_path):
        from rlpe.config import PipelineConfig
        from unittest.mock import patch

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
            from rlpe.pipeline import RadiolarianPipeline

            pipe = RadiolarianPipeline(cfg)
            rows = [{"paper_id": "p1", "figure_id": "f1", "species": "X"}]
            assert pipe._apply_review_corrections(rows) == rows


class TestApiSandbox:
    def test_work_dir_obeys_test_env(self, monkeypatch):
        monkeypatch.setenv("RLPE_API_TEST_TMP", "/tmp/rlpe_api_test_sandbox_x")
        import importlib

        import rlpe.api.app as app_mod

        importlib.reload(app_mod)
        assert str(app_mod.WORK_DIR).startswith("/tmp/rlpe_api_test_sandbox_x")
        assert str(app_mod.UPLOAD_DIR).startswith("/tmp/rlpe_api_test_sandbox_x")
        # restore module state for other tests
        monkeypatch.delenv("RLPE_API_TEST_TMP")
        importlib.reload(app_mod)


class TestGuiSourceGuards:
    def test_batch_placeholder_promotion_wired(self):
        src = (_GUI / "main_window.py").read_text(encoding="utf-8")
        assert "_batch_placeholder_by_stem" in src
        assert "remove_job(ph)" in src

    def test_failed_advances_batch(self):
        src = (_GUI / "main_window.py").read_text(encoding="utf-8")
        # failure path must re-enter the queue when not stopping on error
        assert "not cancelled" in src
        assert "_start_next_batch_job()" in src

    def test_jobs_tab_remove_job_exists(self):
        src = (_GUI / "jobs_tab.py").read_text(encoding="utf-8")
        assert "def remove_job" in src

    def test_cancelled_not_a_failure_dialog(self):
        src = (_GUI / "run_tab.py").read_text(encoding="utf-8")
        assert 'cancelled" in (error or "").lower()' in src

    def test_settings_validate_before_write(self):
        src = (_GUI / "settings_tab.py").read_text(encoding="utf-8")
        # the YOLO validation must sit before the first setValue
        save_start = src.index("def _save")
        first_setvalue = src.index("setValue(QS_KEY_THEME")
        yolo_check = src.index("yolo_enable.isChecked() and not", save_start)
        assert yolo_check < first_setvalue, (
            "YOLO validation must run before any settings write"
        )

    def test_disk_scan_skips_bad_lines(self):
        src = (_GUI / "jobs_tab.py").read_text(encoding="utf-8")
        assert 'errors="replace"' in src
        assert "skipping bad line" in src
        assert "isinstance(row, dict)" in src
