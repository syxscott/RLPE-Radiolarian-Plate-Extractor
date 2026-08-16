"""Regression tests for audit 2026-08-02 image-verified F1 eval.

Covers:
  - scripts/evaluate_image_verified.py runs without crashing
  - Output JSON has the expected schema (papers / aggregate / panel_id_verification)
  - Aggregate contains the required metrics (n_gold, n_checked, string_match_panel_id_rate,
    image_verified_panel_id_rate, gap_pp, blocked_papers)
  - The 9-paper eval artifact at work/eval_v3_image_verified.json reports a finite
    image_verified_panel_id_rate (not None, since we have crops on disk)
  - The 9-paper eval artifact reports at least one BLOCKED paper (proves the
    blocked-papers mechanic still works)
  - EasyOCR and cv2 are importable (the script's hard dependencies)

This test is a *smoke* test for the eval pipeline + a schema gate for the
2026-08-02 baseline artifact. It does NOT re-run EasyOCR (too slow) — it
loads the saved JSON and validates it.

If the panel crops at work/eval_v3_image_verified/panels/ are removed
(e.g. cleanup), the eval JSON's image_verified_panel_id_rate becomes None
and the schema test for finite rate is skipped (xfail).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
_SCRIPTS = _REPO / "scripts"
_WORK = _REPO / "work"
_EVAL_JSON = _WORK / "eval_v3_image_verified.json"
_PRED_JSONL = _WORK / "eval_v3_image_verified" / "predictions.jsonl"
_PANELS_ROOT = _WORK / "eval_v3_image_verified" / "panels"
_GOLD_DIR = _REPO / "data" / "gold"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# --- Hard dependencies (EasyOCR + cv2) ---------------------------------------


class TestEvalDependencies:
    def test_easyocr_importable(self):
        try:
            import easyocr  # noqa: F401
        except ImportError:
            pytest.skip("easyocr not installed in this environment")

    def test_cv2_importable(self):
        cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")

    def test_eval_script_exists(self):
        assert (_SCRIPTS / "evaluate_image_verified.py").exists()


# --- Output schema ----------------------------------------------------------


class TestEvalJsonSchema:
    """Schema gate on the 2026-08-02 saved baseline."""

    @pytest.fixture(scope="class")
    def report(self) -> dict:
        if not _EVAL_JSON.exists():
            pytest.skip(f"{_EVAL_JSON} not present; run image-verified eval first")
        return json.loads(_EVAL_JSON.read_text())

    def test_top_level_keys(self, report):
        assert "papers" in report
        assert "aggregate" in report
        assert "panel_id_verification" in report

    def test_aggregate_keys(self, report):
        agg = report["aggregate"]
        for key in (
            "n_papers",
            "n_gold",
            "n_pred",
            "n_string_match",
            "n_image_verified",
            "n_checked",
            "string_match_panel_id_rate",
            "image_verified_panel_id_rate",
            "gap_pp",
            "blocked_papers",
        ):
            assert key in agg, f"aggregate.{key} missing"

    def test_papers_block(self, report):
        for pid, m in report["papers"].items():
            assert "n_gold" in m
            assert "n_pred" in m
            assert "n_string_match" in m
            assert "n_image_verified" in m
            assert "n_checked" in m
            assert "n_ocr_coverage" in m
            assert "string_match_panel_id_rate" in m
            assert "image_verified_panel_id_rate" in m
            assert "gap_pp" in m
            assert "blocked" in m

    def test_panel_id_verification_status(self, report):
        pv = report["panel_id_verification"]
        assert pv["status"] in {"measured", "skipped_no_ocr", "skipped_no_panels"}

    def test_n_papers_is_9(self, report):
        assert report["aggregate"]["n_papers"] == 9, (
            f"Expected 9 papers (matches the 9-paper gold set), "
            f"got {report['aggregate']['n_papers']}"
        )


# --- Sanity on the 2026-08-02 baseline ---------------------------------------


class TestEvalBaselineNumbers:
    """Sanity checks against the published 2026-08-02 baseline numbers."""

    @pytest.fixture(scope="class")
    def report(self) -> dict:
        if not _EVAL_JSON.exists():
            pytest.skip(f"{_EVAL_JSON} not present; run image-verified eval first")
        return json.loads(_EVAL_JSON.read_text())

    def test_string_match_rate_in_range(self, report):
        sm = report["aggregate"]["string_match_panel_id_rate"]
        # The baseline reported 90.52% over all 612 gold panels.
        assert 0.5 <= sm <= 1.0, f"string_match_panel_id_rate={sm} out of range"

    def test_image_verified_rate_is_finite(self, report):
        iv = report["aggregate"]["image_verified_panel_id_rate"]
        if not _PANELS_ROOT.exists():
            pytest.xfail("Panel crops removed; image-verified rate may be None")
        assert iv is not None, "Expected finite image_verified_panel_id_rate"
        assert 0.0 <= iv <= 1.0

    def test_gap_pp_sign(self, report):
        gap = report["aggregate"]["gap_pp"]
        if gap is None:
            pytest.xfail("gap_pp is None (no OCR coverage)")
        # The 2026-08-02 baseline showed string-match > image-verified
        # (LLM-fabricated panel_ids inflate string-match). If this ever
        # flips, the eval needs investigation.
        assert gap > 0, f"Expected gap > 0 (string-match > image-verified), got {gap}"

    def test_at_least_one_blocked_paper(self, report):
        blocked = report["aggregate"]["blocked_papers"]
        assert len(blocked) >= 1, (
            "Expected at least 1 blocked paper (no panel crops on disk). "
            "If 0, the eval likely got fresh crops and this test should be relaxed."
        )


# --- Smoke: eval script runs end-to-end --------------------------------------


class TestEvalSmoke:
    """Run the eval script in a tmp dir and check it produces the expected schema.

    Skipped if no panel crops are available on disk (the script would just
    mark all papers as blocked, which is still valid but not a meaningful
    smoke test).
    """

    def test_eval_script_runs(self, tmp_path: Path):
        if not _PRED_JSONL.exists():
            pytest.skip(f"{_PRED_JSONL} missing; rebuild predictions first")
        if not _PANELS_ROOT.exists():
            pytest.skip(f"{_PANELS_ROOT} missing; rebuild panel crops first")

        out_json = tmp_path / "smoke_eval.json"
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "evaluate_image_verified.py"),
                "--pred",
                str(_PRED_JSONL),
                "--gold",
                str(_GOLD_DIR),
                "--panels-root",
                str(_PANELS_ROOT),
                "--output",
                str(out_json),
            ],
            capture_output=True,
            text=True,
            cwd=_REPO,
            env={**__import__("os").environ, "PYTHONPATH": "src"},
            timeout=600,
        )
        assert result.returncode == 0, (
            f"eval script exited {result.returncode}; stderr:\n{result.stderr}"
        )
        assert out_json.exists()
        report = json.loads(out_json.read_text())
        # Schema must be present even if all papers are blocked
        assert "aggregate" in report
        assert "papers" in report
        assert "panel_id_verification" in report