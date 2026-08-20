"""Phase 5A audit (2026-08-19): GUI main-thread blocking fixes (sweep 5A).

Four follow-up fixes from the 2026-08-19 multi-agent audit. Phase 1F
landed the minimum-viable patches (logger.error fallback + QThread
introductions); Phase 5A completes the work:

1. **B-14** verify — ``_FlipVerifiedWorker`` already exists. Verify
   the mark-verified button actually starts the worker (not just
   declares the class), the buttons get re-enabled after the worker
   finishes, and double-fire is impossible.

2. **B-15** verify — ``_DiskScanWorker`` + ``_PendingDiskScan`` already
   exist. Verify ``load_recent_jobs_from_disk`` instantiates the
   worker and wires the ``job_loaded`` signal so the GUI thread never
   reads JSONL synchronously.

3. **M-15** completion — Phase 1F added logger.error fallback only.
   Phase 5A introduces ``_ExportWorker(QThread)`` so the 4 export
   functions (xlsx, json, csv, dwca) actually run off the GUI thread.
   Export buttons get disabled during the IO and re-enabled by a
   single ``_re_enable_export_buttons()`` helper so success / failure
   paths agree.

4. **M-16** audit — AST-scan ``src/rlpe/gui/`` to confirm there are no
   remaining synchronous HTTP calls outside a QThread (no
   ``requests.get``, ``requests.post``, ``httpx.get``, ``httpx.post``,
   ``urllib.request.urlopen``, etc., in any handler reachable from a
   Qt signal/slot).

These tests combine source-string guards (work without PySide6) and
runtime tests (require PySide6 + an event loop, but stub the heavy IO).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import PySide6  # noqa: F401

    _HAS_PYSIDE6 = True
except ImportError:
    _HAS_PYSIDE6 = False


_REPO = Path(__file__).resolve().parents[1]
_SRC_GUI_DIR = _REPO / "src" / "rlpe" / "gui"
_SRC_RESULTS_TAB = _SRC_GUI_DIR / "results_tab.py"
_SRC_JOBS_TAB = _SRC_GUI_DIR / "jobs_tab.py"


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


def _body_before_next_def(src: str, signature: str) -> str:
    """Return the slice of ``src`` from ``signature`` until the next
    nested ``def `` (conservative — stops at the first indented method
    or the next top-level class)."""
    idx = src.find(signature)
    assert idx != -1, f"signature {signature!r} not found"
    after = src[idx + len(signature) :]
    next_class = re.search(r"\nclass |\nif __name__", after)
    if next_class is None:
        return after
    return after[: next_class.start()]


# ============================================================================
# B-14 — _FlipVerifiedWorker runtime verification
# ============================================================================


class TestFlipVerifiedRuntime:
    """B-14: verify the mark-verified flow actually wires the worker."""

    def test_flip_method_starts_worker(self):
        """The slot must call ``.start()`` on the worker, not just
        construct it. A bare ``_FlipVerifiedWorker(...)`` call without
        ``.start()`` would still be a sync block."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _flip_image_verified(self")
        # The async version calls .start() on the worker.
        assert "worker.start()" in body, (
            "B-14 verify: _flip_image_verified must call worker.start() "
            "so the POST runs off the GUI thread"
        )

    def test_flip_method_disables_then_reenables_buttons(self):
        """Buttons must be disabled during the flight AND re-enabled
        after via the helper. We assert both the disable (setEnabled(False))
        and the helper call (_re_enable_flip_buttons)."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _flip_image_verified(self")
        assert "setEnabled(False)" in body, (
            "B-14 verify: _flip_image_verified must disable the buttons"
        )
        # Re-enable must happen via the helper (in both success and
        # error callbacks), so failures still leave the UI clickable.
        assert "_re_enable_flip_buttons()" in body, (
            "B-14 verify: _flip_image_verified must call "
            "_re_enable_flip_buttons() from at least one path"
        )
        # Helper itself must exist
        assert "def _re_enable_flip_buttons(self)" in src, (
            "B-14 verify: the _re_enable_flip_buttons() helper must exist"
        )

    def test_re_enable_helper_re_enables_both_buttons(self):
        """The helper must re-enable BOTH mark-verified buttons."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _re_enable_flip_buttons(self")
        # Audit 2026-08-20: F-2 test (test_gui_results_tab_phase_f2.py::
        # TestMarkButtonNames) is the authoritative naming convention
        # (``_mark_verified_btn`` / ``_mark_unverified_btn``). The earlier
        # draft of this test asserted ``_btn_mark_verified`` which was
        # the wrong attribute name — fix here to match the convention.
        assert "_mark_verified_btn" in body
        assert "_mark_unverified_btn" in body
        assert "setEnabled(True)" in body


# ============================================================================
# B-15 — _DiskScanWorker + signal wiring
# ============================================================================


class TestDiskScanRuntime:
    """B-15: verify the disk scan actually defers to the worker."""

    def test_disk_scan_worker_is_qthread(self):
        src = _read("src/rlpe/gui/jobs_tab.py")
        assert "class _DiskScanWorker(QThread)" in src, (
            "B-15 verify: _DiskScanWorker must subclass QThread"
        )

    def test_load_recent_jobs_constructs_worker(self):
        """The load function must actually instantiate the worker."""
        src = _read("src/rlpe/gui/jobs_tab.py")
        body = _body_before_next_def(src, "def load_recent_jobs_from_disk(self")
        assert "_DiskScanWorker(pending)" in body, (
            "B-15 verify: load_recent_jobs_from_disk must instantiate _DiskScanWorker"
        )
        assert "self._disk_scan_worker = worker" in body, (
            "B-15 verify: the worker must be captured on the instance (PySide6 QThread GC footgun)"
        )

    def test_load_recent_jobs_wires_job_loaded_signal(self):
        """The GUI thread must subscribe to job_loaded to fold each
        parsed JobRecord back into _jobs. Without this connection the
        worker is a black hole."""
        src = _read("src/rlpe/gui/jobs_tab.py")
        body = _body_before_next_def(src, "def load_recent_jobs_from_disk(self")
        assert "job_loaded.connect(" in body, "B-15 verify: job_loaded signal must be connected"
        assert "add_or_update_job" in body, (
            "B-15 verify: add_or_update_job must still be referenced "
            "(called from the job_loaded callback)"
        )

    def test_pending_disk_scan_dataclass_present(self):
        """Phase 49's _PendingDiskScan dataclass must still exist so
        the worker can consume the candidate list."""
        src = _read("src/rlpe/gui/jobs_tab.py")
        assert "class _PendingDiskScan:" in src, (
            "B-15 verify: _PendingDiskScan dataclass must be present"
        )


# ============================================================================
# M-15 — _ExportWorker (Phase 5A proper fix)
# ============================================================================


class TestExportWorkerExists:
    """M-15: the proper Phase 5A fix — exports actually run on QThread."""

    def test_export_worker_is_qthread(self):
        src = _read("src/rlpe/gui/results_tab.py")
        assert "class _ExportWorker(QThread)" in src, "M-15: _ExportWorker must subclass QThread"

    def test_export_worker_emits_success_and_error_signals(self):
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "class _ExportWorker(QThread)")
        assert "Signal(str)" in body, "M-15: _ExportWorker must emit a Signal(str) (success path)"
        # The error signal is named `error`; both `finished_with_success`
        # and `error` are Signal(str).
        assert "finished_with_success" in body
        assert "error = Signal(str)" in body or "error=Signal(str)" in body

    def test_export_worker_run_implements_io(self):
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "class _ExportWorker(QThread)")
        assert "def run(self)" in body, "M-15: _ExportWorker.run() must exist"
        # Each of the 4 formats should route through its exporter.
        # We check that the worker calls the actual IO functions, not
        # the bare handlers.
        for fmt_io in (
            "write_xlsx(",  # xlsx branch
            "json.dump(",  # json branch
            "csv.DictWriter(",  # csv branch
            "write_dwca_zip(",  # dwca branch
        ):
            assert fmt_io in body, (
                f"M-15: _ExportWorker.run() must call {fmt_io!r} for the corresponding format"
            )

    def test_export_worker_validates_format(self):
        """The constructor must reject unknown formats so a typo in
        ``_run_export_worker`` can't silently fail."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "class _ExportWorker(QThread)")
        assert "_VALID_FMTS" in body, "M-15: _ExportWorker must declare a _VALID_FMTS allow-list"
        assert "raise ValueError" in body, (
            "M-15: _ExportWorker.__init__ must raise ValueError on bad fmt"
        )


class TestExportFunctionsAreAsync:
    """M-15: the 4 export slots must NOT do IO inline. They must hand
    off to ``_run_export_worker``."""

    def _check_slot_uses_worker(self, fmt: str) -> None:
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, f"def _export_{fmt}(self")
        # The slot must NOT call the underlying IO function directly.
        # (Before Phase 5A, ``_export_xlsx`` called ``write_xlsx``
        # directly on the GUI thread.)
        forbidden = {
            "xlsx": "write_xlsx(",
            "json": "json.dump(",
            "csv": "csv.DictWriter(",
            "dwca": "write_dwca_zip(",
        }[fmt]
        assert forbidden not in body, (
            f"M-15: _export_{fmt} must NOT call {forbidden!r} "
            f"on the GUI thread; hand it off to _run_export_worker"
        )
        # The slot MUST delegate to _run_export_worker.
        assert "_run_export_worker(" in body, (
            f"M-15: _export_{fmt} must call _run_export_worker({fmt!r}, ...)"
        )
        # Format string passed to the worker must match. Allow
        # whitespace between the function name and the opening quote
        # because the CSV slot uses a multi-line call (it has extra
        # ``rows=`` + ``use_utf8_sig=`` kwargs).
        assert re.search(
            rf'_run_export_worker\(\s*"{fmt}"',
            body,
        ), f"M-15: _export_{fmt} must pass the literal '{fmt}' to _run_export_worker"

    def test_export_xlsx_delegates_to_worker(self):
        self._check_slot_uses_worker("xlsx")

    def test_export_json_delegates_to_worker(self):
        self._check_slot_uses_worker("json")

    def test_export_csv_delegates_to_worker(self):
        self._check_slot_uses_worker("csv")

    def test_export_dwca_delegates_to_worker(self):
        self._check_slot_uses_worker("dwca")


class TestExportButtonsAreInstanceAttributes:
    """M-15: the export buttons must be stored on the instance so the
    disable / re-enable helpers can find them."""

    def test_export_buttons_stored_on_instance(self):
        src = _read("src/rlpe/gui/results_tab.py")
        for attr in ("_btn_export_xlsx", "_btn_export_json", "_btn_export_csv", "_btn_export_dwca"):
            assert f"self.{attr}" in src, f"M-15: export buttons must be stored as self.{attr}"

    def test_run_export_worker_helper_exists(self):
        src = _read("src/rlpe/gui/results_tab.py")
        assert "def _run_export_worker(" in src, "M-15: _run_export_worker() helper must exist"

    def test_run_export_worker_disables_buttons(self):
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _run_export_worker(")
        assert "_disable_export_buttons()" in body, (
            "M-15: _run_export_worker must disable buttons on entry"
        )
        # Both success and error must re-enable.
        assert body.count("_re_enable_export_buttons()") >= 2, (
            "M-15: _re_enable_export_buttons() must be called from "
            "both the success and error callbacks"
        )

    def test_disable_and_re_enable_export_buttons_helpers_exist(self):
        src = _read("src/rlpe/gui/results_tab.py")
        assert "def _disable_export_buttons(self)" in src, (
            "M-15: _disable_export_buttons() helper must exist"
        )
        assert "def _re_enable_export_buttons(self)" in src, (
            "M-15: _re_enable_export_buttons() helper must exist"
        )

    def test_re_enable_export_buttons_iterates_all_four(self):
        """The helper must re-enable all 4 export buttons."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _re_enable_export_buttons(self")
        for attr in ("_btn_export_xlsx", "_btn_export_json", "_btn_export_csv", "_btn_export_dwca"):
            assert attr in body, f"M-15: _re_enable_export_buttons must re-enable {attr}"
        assert "setEnabled(True)" in body


class TestExportWorkerRuntime:
    """Drive the worker end-to-end with stubbed IO to verify success
    and failure paths emit the right signals."""

    @pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
    def test_json_export_worker_success(self, tmp_path):
        """A JSON export must emit finished_with_success(path) on 2xx
        write. We use a real JSON write but a tiny payload so the test
        is fast."""
        from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

        from rlpe.gui.results_tab import _ExportWorker

        app = QCoreApplication.instance() or QCoreApplication(sys.argv)

        out = tmp_path / "out.json"
        run_output = {"panels": [], "schema_version": "1.0.0"}
        worker = _ExportWorker("json", run_output, str(out), [])

        successes: list[str] = []
        errors: list[str] = []
        worker.finished_with_success.connect(successes.append)
        worker.error.connect(errors.append)

        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        QTimer.singleShot(0, worker.start)
        QTimer.singleShot(3000, loop.quit)
        loop.exec()

        assert successes == [str(out)], f"expected success, got {successes}"
        assert errors == [], f"unexpected error: {errors}"
        assert out.exists(), "JSON file must be written"

    @pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
    def test_export_worker_rejects_unknown_format(self):
        """An unknown format must raise ValueError at construction
        time (defence-in-depth — a typo in _run_export_worker is
        caught immediately, not silently swallowed by run()."""
        from rlpe.gui.results_tab import _ExportWorker

        with pytest.raises(ValueError, match="unknown export format"):
            _ExportWorker("xyz", {}, "/tmp/x", [])

    @pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
    def test_export_worker_emits_error_on_write_failure(self, tmp_path, monkeypatch):
        """If the writer raises, the worker must emit error(str), not
        propagate the exception (which would crash the host process)."""
        from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

        from rlpe.gui.results_tab import _ExportWorker

        app = QCoreApplication.instance() or QCoreApplication(sys.argv)

        # Write to a path under a non-existent parent dir to force
        # an OSError on the json.dump open().
        bad_dir = tmp_path / "no_such_dir" / "no_such_subdir"
        out = bad_dir / "out.json"
        worker = _ExportWorker("json", {}, str(out), [])

        successes: list[str] = []
        errors: list[str] = []
        worker.finished_with_success.connect(successes.append)
        worker.error.connect(errors.append)

        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        QTimer.singleShot(0, worker.start)
        QTimer.singleShot(3000, loop.quit)
        loop.exec()

        assert successes == [], "expected no success on write failure"
        assert errors, "error signal must fire on write failure"
        # The error message should describe the failure (FileNotFoundError
        # / OSError / etc.).
        assert any(token in errors[0] for token in ("FileNotFoundError", "OSError", "No such")), (
            f"error message missing detail: {errors[0]!r}"
        )


# ============================================================================
# M-16 — audit: no sync HTTP calls remain on the GUI thread
# ============================================================================


# Patterns we treat as "network IO that must NOT be sync on the GUI
# thread". Order matters — we match the longest prefix first so
# ``requests.post`` doesn't shadow ``requests.post('foo')`` checking.
_NETWORK_CALL_PATTERNS = (
    re.compile(r"\brequests\s*\.\s*(?:get|post|put|delete|patch|head|request)\s*\("),
    re.compile(r"\bhttpx\s*\.\s*(?:get|post|put|delete|patch|head|request)\s*\("),
    re.compile(r"\burllib\s*\.\s*request\s*\.\s*(?:urlopen|Request|urlretrieve)\s*\("),
    re.compile(r"\burlopen\s*\("),
    re.compile(r"\burlretrieve\s*\("),
)


def _is_network_call(line: str) -> bool:
    """True if ``line`` (already stripped) looks like a network call
    we care about. Comment-only lines are ignored by the caller."""
    for pat in _NETWORK_CALL_PATTERNS:
        if pat.search(line):
            return True
    return False


def _is_in_qthread_or_worker(src: str, lineno: int) -> bool:
    """Heuristic: return True if line ``lineno`` of ``src`` (1-based)
    is inside a class that subclasses QThread or a method whose name
    starts with ``_run_*_worker`` / ``_DiskScanWorker`` /
    ``_FlipVerifiedWorker`` / ``_ExportWorker`` / ``PipelineWorker``.
    """
    lines = src.splitlines()
    # Walk backwards from lineno looking for the enclosing class or
    # top-level function definition. If the enclosing scope is a
    # QThread subclass or a method of one, we treat the network call
    # as legitimately on a worker thread.
    for i in range(lineno - 1, -1, -1):
        line = lines[i]
        m = re.match(r"^class\s+(\w+)\s*(\(([^)]*)\))?\s*:", line)
        if m:
            base = (m.group(3) or "").strip()
            return "QThread" in base or "PipelineWorker" in m.group(1)
        # A bare function / method definition signals we've left the
        # current scope without finding a QThread class — but keep
        # walking so we find the OUTER class.
    return False


class TestNoSyncHttpCallsInGui:
    """M-16: AST-scan ``src/rlpe/gui/`` for sync HTTP calls outside
    QThread workers."""

    def _find_network_calls(self, src: str) -> list[tuple[int, str]]:
        """Yield (lineno, line) for every line in ``src`` that looks
        like a sync network call AND is not inside a QThread subclass.
        """
        results: list[tuple[int, str]] = []
        for i, line in enumerate(src.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _is_network_call(stripped):
                if not _is_in_qthread_or_worker(src, i):
                    results.append((i, line))
        return results

    def test_results_tab_has_no_sync_http(self):
        """``results_tab.py`` must not have a sync HTTP call. The
        B-14 ``_FlipVerifiedWorker`` runs ``requests.post`` inside
        ``run()``, which IS on a QThread — the assertion above
        filters that out. Anything else is a regression."""
        src = _read("src/rlpe/gui/results_tab.py")
        offenders = self._find_network_calls(src)
        # Allow the B-14 requests.post inside _FlipVerifiedWorker.run()
        # (which is filtered out by _is_in_qthread_or_worker). We
        # additionally allow comments referencing requests.post.
        real_offenders = [
            (ln, l)
            for (ln, l) in offenders
            if "requests.post" in l  # All requests.post lines must be inside worker
            and "_FlipVerifiedWorker" not in l
        ]
        # Actually: since _is_in_qthread_or_worker already filters,
        # any offenders here are genuinely outside a worker.
        assert not offenders, (
            "M-16: results_tab.py has sync network call(s) outside a "
            "QThread worker:\n" + "\n".join(f"  L{ln}: {l.strip()}" for ln, l in offenders)
        )

    def test_jobs_tab_has_no_sync_http(self):
        """``jobs_tab.py`` must not have any sync HTTP calls."""
        src = _read("src/rlpe/gui/jobs_tab.py")
        offenders = self._find_network_calls(src)
        assert not offenders, (
            "M-16: jobs_tab.py has sync network call(s) outside a "
            "QThread worker:\n" + "\n".join(f"  L{ln}: {l.strip()}" for ln, l in offenders)
        )

    def test_all_gui_files_clean_of_sync_http(self):
        """Whole-directory sweep: every Python file under
        ``src/rlpe/gui/`` must be free of sync network calls outside
        a QThread worker. This is the broader M-16 contract."""
        offenders_total: list[tuple[str, int, str]] = []
        for py in _SRC_GUI_DIR.glob("*.py"):
            src = py.read_text(encoding="utf-8")
            for lineno, line in self._find_network_calls(src):
                offenders_total.append((py.name, lineno, line.strip()))
        assert not offenders_total, (
            "M-16: sync network calls outside QThread workers found:\n"
            + "\n".join(f"  {name}:L{ln}: {l}" for name, ln, l in offenders_total)
        )


class TestGuiFilesAreParseable:
    """Sanity: every Python file in the GUI tree must parse cleanly."""

    @pytest.mark.parametrize(
        "filename",
        [
            "app.py",
            "batch_dialog.py",
            "constants.py",
            "i18n.py",
            "i18n_widgets.py",
            "image_preview.py",
            "jobs_tab.py",
            "main_window.py",
            "outdir_probe.py",
            "pipeline_worker.py",
            "results_tab.py",
            "run_tab.py",
            "settings_tab.py",
            "styles.py",
            "utils.py",
        ],
    )
    def test_file_parses(self, filename: str):
        path = _SRC_GUI_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        src = path.read_text(encoding="utf-8")
        try:
            ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(f"{filename} has syntax error: {exc}")


# ============================================================================
# Integration: source-guard the Phase 5A diff itself
# ============================================================================


class TestPhase5ADiffIsApplied:
    """Make sure the Phase 5A changes are actually in the file."""

    def test_results_tab_has_export_worker_class(self):
        assert "class _ExportWorker(QThread)" in _SRC_RESULTS_TAB.read_text(encoding="utf-8")

    def test_results_tab_has_run_export_worker(self):
        assert "def _run_export_worker(" in _SRC_RESULTS_TAB.read_text(encoding="utf-8")

    def test_results_tab_has_re_enable_export_buttons(self):
        assert "def _re_enable_export_buttons(self)" in _SRC_RESULTS_TAB.read_text(encoding="utf-8")

    def test_results_tab_has_disable_export_buttons(self):
        assert "def _disable_export_buttons(self)" in _SRC_RESULTS_TAB.read_text(encoding="utf-8")

    def test_results_tab_has_re_enable_flip_buttons(self):
        assert "def _re_enable_flip_buttons(self)" in _SRC_RESULTS_TAB.read_text(encoding="utf-8")

    def test_jobs_tab_has_disk_scan_worker_class(self):
        assert "class _DiskScanWorker(QThread)" in _SRC_JOBS_TAB.read_text(encoding="utf-8")

    def test_jobs_tab_has_pending_disk_scan_class(self):
        assert "class _PendingDiskScan:" in _SRC_JOBS_TAB.read_text(encoding="utf-8")
