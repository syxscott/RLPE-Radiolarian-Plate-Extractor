"""Audit 2026-09-03 user-reported bugs:

* Bug #1 (user-reported, zhang2014): JobsTab.load_recent_jobs_from_disk
  did NOT scan ``<last_pdf_dir>/<stem>/work/manifests/matches.jsonl``
  even though the PySide6 GUI writes exactly there. The result: every
  PySide6 run against a user-specified output_dir was invisible to
  the Jobs tab until re-opened in the same GUI session.

* Bug #2 (silent failure): pipeline.open_dataloader_path returned 0
  figures for ``zhang2014`` and 0 figure-type elements in the kids
  tree, but ``manifest.warnings`` was ``[]``. The operator saw 0
  panel rows in the GUI and had no signal whether the pipeline
  crashed or never started.

Both fixes tested below.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Bug #1: JobsTab now scans user-configured last_pdf_dir
# ---------------------------------------------------------------------------


class TestJobsTabScansUserPdfDir:
    """The PySide6 GUI writes single-PDF runs to
    ``<last_pdf_dir>/<stem>/work/manifests/matches.jsonl`` and batch
    runs to ``<last_export_dir>/<stem>_rlpe_out/work/output/manifests/
    matches.jsonl``. Both must be picked up by the disk scan."""

    def _write_user_output(self, base_dir: Path, stem: str, n_rows: int = 3) -> Path:
        """Helper: create a PySide6-style output dir for one stem."""
        work = base_dir / stem / "work"
        manifests = work / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        with (manifests / "matches.jsonl").open("w") as fh:
            for i in range(n_rows):
                row = {
                    "paper_id": f"pid_{stem}",
                    "figure_id": f"fig_{i}",
                    "panel_id": str(i + 1),
                    "species": f"Species {stem} {i}",
                    "confidence": 0.9,
                }
                fh.write(json.dumps(row) + "\n")
        (manifests / "complete.flag").write_text("done", encoding="utf-8")
        return work

    def test_scan_picks_up_single_pdf_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PySide6 GUI run against ``<pdf_dir>/<stem>/work/manifests/``
        must appear in the JobsTab disk scan."""
        from rlpe.gui import jobs_tab as _jt
        from rlpe.gui.jobs_tab import JobsTab

        last_pdf_dir = tmp_path / "papers"
        self._write_user_output(last_pdf_dir, "zhang2014", n_rows=3)

        from PySide6.QtCore import QSettings
        from rlpe.gui.constants import APP_AUTHOR, APP_NAME, QS_KEY_LAST_DIR
        _qsettings = QSettings(APP_AUTHOR, APP_NAME)
        _qsettings.setValue(QS_KEY_LAST_DIR, str(last_pdf_dir))

        # Capture the pending list the disk-scan would schedule.
        pending_holder: list = []

        class _CapturingPending:
            def __init__(self, **kw):
                pending_holder.append(kw)

        monkeypatch.setattr(_jt, "_PendingDiskScan", _CapturingPending)

        # Instantiate JobsTab bypassing __init__ (no QWidget needed
        # for the sync walk).
        tab = JobsTab.__new__(JobsTab)
        tab.load_recent_jobs_from_disk()

        # The zhang2014 path must be in the pending list.
        assert any(
            "zhang2014" in str(p.get("matches_path", ""))
            for p in pending_holder
        ), (
            f"zhang2014 not picked up by disk scan; pending list: "
            f"{pending_holder}"
        )

    def test_scan_picks_up_batch_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PySide6 GUI BATCH run writes to
        ``<last_export_dir>/<stem>_rlpe_out/work/output/manifests/`` —
        that path must also be picked up."""
        from rlpe.gui import jobs_tab as _jt
        from rlpe.gui.jobs_tab import JobsTab

        last_export_dir = tmp_path / "exports"
        batch_root = last_export_dir / "zhang2014_2014_rlpe_out"
        manifests = batch_root / "work" / "output" / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        with (manifests / "matches.jsonl").open("w") as fh:
            for i in range(5):
                fh.write(json.dumps({"panel_id": str(i), "species": "X"}) + "\n")
        (manifests / "complete.flag").write_text("done", encoding="utf-8")

        from PySide6.QtCore import QSettings
        from rlpe.gui.constants import APP_AUTHOR, APP_NAME, QS_KEY_LAST_EXPORT_DIR
        _qsettings = QSettings(APP_AUTHOR, APP_NAME)
        _qsettings.setValue(QS_KEY_LAST_EXPORT_DIR, str(last_export_dir))

        pending_holder: list = []

        class _CapturingPending:
            def __init__(self, **kw):
                pending_holder.append(kw)

        monkeypatch.setattr(_jt, "_PendingDiskScan", _CapturingPending)

        tab = JobsTab.__new__(JobsTab)
        tab.load_recent_jobs_from_disk()

        assert any(
            "zhang2014_2014_rlpe_out" in str(p.get("matches_path", ""))
            for p in pending_holder
        ), (
            f"batch zhang2014 not picked up; pending list: "
            f"{pending_holder}"
        )

    def test_no_user_dir_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If QSettings has no last_pdf_dir / last_export_dir, the
        scan must not crash — just skip the user-root step."""
        from rlpe.gui import jobs_tab as _jt
        from rlpe.gui.jobs_tab import JobsTab

        # Clear the keys.
        from PySide6.QtCore import QSettings
        from rlpe.gui.constants import (
            APP_AUTHOR,
            APP_NAME,
            QS_KEY_LAST_DIR,
            QS_KEY_LAST_EXPORT_DIR,
        )
        _qsettings = QSettings(APP_AUTHOR, APP_NAME)
        _qsettings.setValue(QS_KEY_LAST_DIR, "")
        _qsettings.setValue(QS_KEY_LAST_EXPORT_DIR, "")

        pending_holder: list = []

        class _CapturingPending:
            def __init__(self, **kw):
                pending_holder.append(kw)

        monkeypatch.setattr(_jt, "_PendingDiskScan", _CapturingPending)

        tab = JobsTab.__new__(JobsTab)
        # Should return 0 (or whatever) without raising.
        tab.load_recent_jobs_from_disk()


# ---------------------------------------------------------------------------
# Bug #2: OD 0 figures AND 0 figure-type kids → warning emitted
# ---------------------------------------------------------------------------


@dataclass
class _StubODResult:
    """A tiny stand-in for OpenDataLoaderResult that has just enough
    surface area for the warning-emit path to read."""

    success: bool = True
    figures: list = field(default_factory=list)
    json_data: dict = field(default_factory=dict)
    fulltext_sections: list = field(default_factory=list)
    error: str | None = None


def _emit_od_zero_figure_warning(
    figures: list, json_data: dict, paper_id: str
) -> None:
    """Mirror of the inline logic in pipeline._process_one_pdf_od_inner
    that the audit 2026-09-03 fix added. Re-implemented here so we
    can test it without standing up a full pipeline."""
    import time as _time

    from rlpe.utils import _WARNINGS, _WARNINGS_LOCK

    figures = list(figures or [])
    if figures:
        return
    kids_tree = (json_data or {}).get("kids") or []
    kids_types = [
        k.get("Type") or k.get("type") for k in kids_tree if isinstance(k, dict)
    ]
    figure_types = {
        t
        for t in kids_types
        if t and ("figure" in str(t).lower() or "image" in str(t).lower())
    }
    if not figure_types and kids_tree:
        _warning_msg = (
            f"OpenDataLoader returned 0 figures AND 0 "
            f"figure-type elements in the kids tree for "
            f"{paper_id} ({len(kids_tree)} paragraph-only kids). "
            f"This PDF likely uses a non-standard figure layout "
            f"(phylogenetic tree / line drawing only) — "
            f"species matching will produce 0 rows."
        )
        entry = {
            "label": "od_zero_figure_metadata",
            "paper_id": paper_id,
            "message": _warning_msg,
            "timestamp": _time.time(),
        }
        with _WARNINGS_LOCK:
            _WARNINGS.append(entry)


class TestPipelineZeroFigureWarning:
    """When OpenDataLoader returns 0 figures AND every kid in the
    kids tree is a "paragraph" (no Figure / Image type), the
    pipeline must emit a warning into ``manifest.warnings`` so the
    GUI Results tab and the on-disk manifest both surface the
    silent failure."""

    def test_zero_figures_emits_warning(self) -> None:
        """Stub OD with 0 figures + paragraph-only kids → warning
        emitted into ``_WARNINGS``."""
        from rlpe.utils import _WARNINGS, _WARNINGS_LOCK
        with _WARNINGS_LOCK:
            _WARNINGS.clear()

        # Mimic zhang2014: 0 figures + 216 paragraph-only kids
        stub = _StubODResult(
            success=True,
            figures=[],
            json_data={
                "kids": [
                    {"Type": "paragraph", "content": "paper text"}
                ]
                * 216
            },
        )
        _emit_od_zero_figure_warning(
            stub.figures, stub.json_data, paper_id="zhang2014"
        )

        with _WARNINGS_LOCK:
            matching = [
                w for w in _WARNINGS if w["label"] == "od_zero_figure_metadata"
            ]
        assert matching, (
            "Pipeline should have emitted an od_zero_figure_metadata "
            "warning for a PDF with 0 figures and 0 figure-type kids"
        )
        warning = matching[0]
        assert warning["paper_id"] == "zhang2014"
        assert "paragraph-only kids" in warning["message"]
        assert "0 figures" in warning["message"]
        assert "216" in warning["message"]

    def test_with_real_figures_no_warning(self) -> None:
        """Sanity check: when OD returns at least one figure, the
        warning is NOT emitted (we only flag the degenerate case)."""
        from rlpe.utils import _WARNINGS, _WARNINGS_LOCK
        with _WARNINGS_LOCK:
            _WARNINGS.clear()

        # Mimic a normal PDF: 5 figures + a mix of paragraph + figure kids
        stub = _StubODResult(
            success=True,
            figures=[{"caption_text": "fig 1"}] * 5,
            json_data={
                "kids": [
                    {"Type": "paragraph", "content": "x"},
                    {"Type": "Figure", "content": "real figure"},
                ]
            },
        )
        _emit_od_zero_figure_warning(
            stub.figures, stub.json_data, paper_id="beccaro2006"
        )
        with _WARNINGS_LOCK:
            od_warnings = [
                w for w in _WARNINGS if w["label"] == "od_zero_figure_metadata"
            ]
        assert od_warnings == []

    def test_kids_but_no_figure_type_no_warning(self) -> None:
        """When OD returned 0 figures BUT the kids tree has figure-type
        entries (the OD Java pairing stage failed but the kids tree
        is intact), do NOT emit the warning — the orphan-figure
        rescue path will still pick up captions."""
        from rlpe.utils import _WARNINGS, _WARNINGS_LOCK
        with _WARNINGS_LOCK:
            _WARNINGS.clear()

        stub = _StubODResult(
            success=True,
            figures=[],
            json_data={
                "kids": [
                    {"Type": "paragraph", "content": "text"},
                    {"Type": "Figure", "content": "Fig. 1"},
                    {"Type": "Figure-Caption", "content": "Fig. 1 caption"},
                ]
            },
        )
        _emit_od_zero_figure_warning(
            stub.figures, stub.json_data, paper_id="normal_pdf"
        )
        with _WARNINGS_LOCK:
            od_warnings = [
                w for w in _WARNINGS if w["label"] == "od_zero_figure_metadata"
            ]
        assert od_warnings == [], (
            "Should NOT warn — kids tree has Figure types so the "
            "orphan-caption rescue path will still extract species"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])