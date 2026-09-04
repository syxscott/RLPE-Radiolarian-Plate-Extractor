"""Audit 2026-09-03 (user-reported zhang2014 follow-up):
``JobsTab.mark_done`` must create a placeholder JobRecord when
the ``job_id`` is unknown, not silently early-return.

User-reported scenario: GUI restarted at 21:52:34 (PID 1446524)
loading 143 recent jobs from disk. The user clicked "Run" on
zhang2014 at 21:53:17. By 21:53:27 the pipeline finished
with 0 rows. ``main_window._on_job_finished`` called
``self._jobs_tab.mark_done("215317", [])`` which ``return``-ed
early because the in-memory ``_jobs`` dict didn't contain
"215317" — the disk scan had run before the user's new job.
``_results_tab.load_job(...)`` was also called (so ResultsTab
header updated) but the JobsTab never showed the row, leaving
the operator staring at "Loaded 143 recent jobs" with no sign
that their click actually produced a finished run.

Fix: ``mark_done / mark_failed / mark_cancelled /
update_progress`` now create a placeholder JobRecord when the
job_id is unknown. The operator can immediately see the new
job in the JobsTab with whatever metadata is available
(pdf_path / output_dir may be empty until disk-scan next
re-scans, but the status, row count, and finished_at are
captured).
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from rlpe.gui.jobs_tab import JobsTab

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _build_tab() -> JobsTab:
    """Build a JobsTab-shaped stub with the minimum surface the
    ``mark_*`` / ``update_progress`` helpers need. Avoids
    QWidget.__init__ (no real GUI)."""
    from rlpe.gui.jobs_tab import JobsTab

    tab = JobsTab.__new__(JobsTab)
    tab._jobs = collections.OrderedDict()

    class _StubTable:
        def rowCount(self) -> int:
            return 0

        def columnCount(self) -> int:
            return 0

        def item(self, *_args, **_kwargs):
            return None

        def setItem(self, *_args, **_kwargs):
            pass

        def removeRow(self, *_args, **_kwargs):
            pass

        def setRowCount(self, _n: int) -> None:
            pass

        def insertRow(self, *_args, **_kwargs):
            pass

    tab._table = _StubTable()

    class _StubLabel:
        def setText(self, *_args, **_kwargs):
            pass

        def setObjectName(self, *_args, **_kwargs):
            pass

    tab._count_label = _StubLabel()
    tab._summary = _StubLabel()
    return tab


class TestMarkDoneCreatesPlaceholderForUnknownJob:
    """The 2026-09-03 21:53 zhang2014 GUI run: the JobsTab was
    empty after the run because ``mark_done`` returned early when
    the in-memory dict didn't contain the job_id yet."""

    def test_mark_done_creates_placeholder_when_job_id_unknown(
        self,
    ) -> None:
        from rlpe.gui.constants import STATUS_DONE

        tab = _build_tab()

        tab.mark_done("x-unknown", [{"paper_id": "p", "species": "S"}])

        assert "x-unknown" in tab._jobs, (
            "mark_done must create a placeholder when the id is "
            "unknown — without it the operator sees a finished "
            "INFO log on the pipeline side but the JobsTab row "
            "vanishes."
        )
        job = tab._jobs["x-unknown"]
        assert job.status == STATUS_DONE
        assert job.rows == [{"paper_id": "p", "species": "S"}]
        assert job.finished_at > 0

    def test_mark_failed_creates_placeholder(self) -> None:
        from rlpe.gui.constants import STATUS_FAILED

        tab = _build_tab()
        tab.mark_failed("x-fail", "BOOM")

        assert "x-fail" in tab._jobs
        assert tab._jobs["x-fail"].status == STATUS_FAILED
        assert tab._jobs["x-fail"].error == "BOOM"

    def test_mark_cancelled_creates_placeholder(self) -> None:
        from rlpe.gui.constants import STATUS_CANCELLED

        tab = _build_tab()
        tab.mark_cancelled("x-cancel")

        assert "x-cancel" in tab._jobs
        assert tab._jobs["x-cancel"].status == STATUS_CANCELLED

    def test_update_progress_creates_placeholder(self) -> None:
        from rlpe.gui.constants import STATUS_RUNNING

        tab = _build_tab()
        tab.update_progress("x-live", 5, 100, "Running stage")

        assert "x-live" in tab._jobs
        assert tab._jobs["x-live"].status == STATUS_RUNNING
        assert tab._jobs["x-live"].progress_current == 5
        assert tab._jobs["x-live"].progress_total == 100

    def test_existing_job_is_updated_not_replaced(self) -> None:
        """When the job_id IS in ``_jobs``, ``mark_done`` should
        update the existing record's status / rows / finished_at
        rather than creating a duplicate placeholder."""
        from rlpe.gui.constants import STATUS_RUNNING
        from rlpe.gui.jobs_tab import JobRecord

        tab = _build_tab()
        existing = JobRecord(
            job_id="x",
            pdf_path="/some/path",
            output_dir="/some/output",
            status=STATUS_RUNNING,
        )
        tab._jobs["x"] = existing

        tab.mark_done("x", [{"k": "v"}])

        # Same record, just updated fields.
        assert tab._jobs["x"] is existing
        assert existing.pdf_path == "/some/path"
        assert existing.output_dir == "/some/output"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
