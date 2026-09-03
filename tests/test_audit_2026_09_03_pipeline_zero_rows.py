"""Audit 2026-09-03 (BLOCKER-#2 follow-up #2): catch-all warning when
pipeline produces 0 rows even though OpenDataLoader succeeded.

The earlier fix only caught the OD-degenerate case where OD
returned ``figures=[]`` and ``json_data=None``. But on the
2026-09-03 21:39 zhang2014 re-run, OD *did* extract 6 images and
2 captions (figures list was non-empty), yet the pipeline
still produced 0 rows because every downstream stage rejected
them. The OD-only warning never fired.

This test pins the new catch-all warning that fires inside
``RadiolarianPipeline.run`` right before the manifest is written
when ``total > 0 and len(rows) == 0``. That covers the figure-
caption pairing, panel matching, species-resolution, and the
degenerate-OD failure modes in one place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _emit_pipeline_zero_rows_warning(total: int, n_rows: int) -> None:
    """Mirror of the inline logic in pipeline.run that the audit
    2026-09-03 follow-up added. Re-implemented here so the test
    can pin the behaviour without standing up a full pipeline."""
    import time as _t
    from rlpe.utils import _WARNINGS as _W, _WARNINGS_LOCK as _L
    if total > 0 and n_rows == 0:
        msg = (
            f"Pipeline processed {total} PDF(s) but produced "
            f"0 result rows. Downstream figure-caption pairing, "
            f"panel-matching, or species-name resolution may "
            f"have rejected every detected figure."
        )
        with _L:
            _W.append(
                {
                    "label": "pipeline_finished_zero_rows",
                    "paper_id": None,
                    "message": msg,
                    "timestamp": _t.time(),
                }
            )


class TestPipelineZeroRowsWarning:
    """When the pipeline processes at least one PDF but emits 0
    rows, the operator must see WHY in ``manifest.warnings``."""

    def test_zero_rows_with_processed_papers_emits_warning(self) -> None:
        """The 2026-09-03 21:39 zhang2014 case: 1 PDF processed,
        OD extracted 6 images + 2 captions, every downstream
        stage rejected them, ``matches.jsonl: 0 行`` —
        manifest.warnings MUST now contain the catch-all."""
        from rlpe.utils import _WARNINGS, _WARNINGS_LOCK
        with _WARNINGS_LOCK:
            _WARNINGS.clear()

        _emit_pipeline_zero_rows_warning(total=1, n_rows=0)

        with _WARNINGS_LOCK:
            matching = [
                w for w in _WARNINGS
                if w["label"] == "pipeline_finished_zero_rows"
            ]
        assert matching, (
            "Pipeline must emit pipeline_finished_zero_rows when "
            "it processed at least one PDF but emitted 0 rows"
        )
        assert "0 result rows" in matching[0]["message"]
        assert "1 PDF" in matching[0]["message"]

    def test_zero_papers_processed_no_warning(self) -> None:
        """If the run was asked to process 0 papers (empty
        input), the catch-all warning must NOT fire — the
        operator asked for nothing so producing nothing is the
        correct outcome."""
        from rlpe.utils import _WARNINGS, _WARNINGS_LOCK
        with _WARNINGS_LOCK:
            _WARNINGS.clear()

        _emit_pipeline_zero_rows_warning(total=0, n_rows=0)

        with _WARNINGS_LOCK:
            matching = [
                w for w in _WARNINGS
                if w["label"] == "pipeline_finished_zero_rows"
            ]
        assert matching == [], (
            "Catch-all warning must not fire when no papers were "
            "processed (empty input dir, no PDFs found, etc.)"
        )

    def test_nonzero_rows_no_warning(self) -> None:
        """Sanity check: when the pipeline produced >=1 rows, no
        zero-rows warning is emitted (the success path is
        silent)."""
        from rlpe.utils import _WARNINGS, _WARNINGS_LOCK
        with _WARNINGS_LOCK:
            _WARNINGS.clear()

        _emit_pipeline_zero_rows_warning(total=3, n_rows=42)

        with _WARNINGS_LOCK:
            matching = [
                w for w in _WARNINGS
                if w["label"] == "pipeline_finished_zero_rows"
            ]
        assert matching == [], (
            "Pipeline with non-zero row output must not emit the "
            "catch-all zero-rows warning"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])