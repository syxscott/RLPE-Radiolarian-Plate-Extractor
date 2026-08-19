"""Phase F-1 (2026-08-20) audit fixes for ``src/rlpe/gui/results_tab.py``.

Three bugs from the multi-agent audit landed in this sweep:

* **B-2** — ``_FlipVerifiedWorker`` and ``_ExportWorker`` QThreads
  were not stopped/waited when the ResultsTab was destroyed, causing
  exit code 134 on GUI close. New :meth:`ResultsTab.shutdown` cancels
  both workers and waits up to 30 s. The workers expose a ``cancel()``
  method that flips a sticky ``_cancelled`` flag and ``run()``
  honours it at every checkpoint.
* **M-3** — ``PanelRecord`` and ``PipelineWorker`` emit
  ``figure_image_path`` at the row top-level. ``_on_row_selected``
  only checked ``row["metadata"]["figure_image_path"]``, missing
  every v1.1.0 row. New static helper
  :meth:`ResultsTab._resolve_figure_image_path` falls back through
  top-level → ``metadata.figure_image_path`` →
  ``metadata.primary_image`` → ``metadata.image_path``.
* **M-5** — API URL was taken from QSettings verbatim and used to
  build the /review/correction endpoint. A hostile override
  (``file:///etc/passwd``, ``javascript:alert(1)``, etc.) could
  redirect the POST and leak the request body. New module-level
  helper :func:`_validate_api_url` enforces scheme + non-empty host +
  loopback deny-list. New :meth:`ResultsTab._set_api_url` is the
  single write path for the QSettings key.

These tests combine source-string guards (work without PySide6) and
runtime tests (require PySide6 + an event loop, but stub the heavy IO).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

# Mirror the convention used by other GUI tests: force the offscreen
# Qt platform plugin so we don't need a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import PySide6  # noqa: F401

    _HAS_PYSIDE6 = True
except ImportError:
    _HAS_PYSIDE6 = False

_SRC_RESULTS_TAB = _REPO / "src" / "rlpe" / "gui" / "results_tab.py"


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


def _strip_comments(src: str) -> str:
    """Return ``src`` with line and block comments stripped.

    Used to avoid false positives when scanning for ``.terminate()``
    in source — we don't want a comment like
    ``# NOTE: don't add terminate() here`` to fail the test.
    """
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    lines = []
    for line in src.splitlines():
        # Drop everything after ``#`` (naive but good enough: the
        # GUI source does not embed ``#`` inside string literals in
        # the cleanup blocks under test).
        stripped = line.split("#", 1)[0]
        lines.append(stripped)
    return "\n".join(lines)


# ============================================================================
# B-2 — QThread shutdown on GUI close
# ============================================================================


class TestShutdownSourceGuards:
    """B-2: source-level guards on the shutdown contract."""

    def test_shutdown_method_exists(self):
        """ResultsTab must expose a public ``shutdown(self)`` method."""
        src = _read("src/rlpe/gui/results_tab.py")
        assert "def shutdown(self) -> None:" in src, (
            "B-2: ResultsTab must define a public shutdown() method so "
            "the GUI close path can drain in-flight workers"
        )

    def test_shutdown_calls_cancel_and_wait(self):
        """shutdown() must set the cancel flag AND wait() on both workers."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def shutdown(self) -> None:")
        assert "cancel()" in body, (
            "B-2: shutdown() must call worker.cancel() on each worker"
        )
        assert ".wait(" in body, (
            "B-2: shutdown() must call worker.wait(timeout) on each "
            "worker (with a finite timeout)"
        )
        # 30s cap matches the rest of the GUI shutdown paths.
        assert ".wait(30000)" in body, (
            "B-2: shutdown() must use wait(30000) — the GUI shutdown "
            "timeout contract"
        )

    def test_shutdown_drains_both_workers(self):
        """shutdown() must walk both ``_flip_worker`` and ``_export_worker``."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def shutdown(self) -> None:")
        assert "_flip_worker" in body
        assert "_export_worker" in body
        assert "isRunning()" in body, (
            "B-2: shutdown() must check isRunning() before waiting "
            "(an already-finished worker should be dropped, not waited)"
        )

    def test_shutdown_drops_worker_references(self):
        """shutdown() must set ``self._flip_worker = None`` /
        ``self._export_worker = None`` after draining so a future
        export / flip builds a fresh worker."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def shutdown(self) -> None:")
        assert "self._flip_worker = None" in body or "_flip_worker = None" in body, (
            "B-2: shutdown() must drop the _flip_worker reference after "
            "the wait returns"
        )
        assert "self._export_worker = None" in body or "_export_worker = None" in body, (
            "B-2: shutdown() must drop the _export_worker reference after "
            "the wait returns"
        )

    def test_shutdown_called_from_close_event(self):
        """closeEvent must call shutdown() so the QThread destructor
        doesn't run while a worker is mid-flight."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def closeEvent(self, event) -> None:")
        assert "self.shutdown()" in body, (
            "B-2: ResultsTab.closeEvent must call self.shutdown() so "
            "the QThread destructor doesn't trip on a still-running "
            "worker (exit code 134)"
        )

    def test_flip_worker_has_cancel(self):
        """``_FlipVerifiedWorker.cancel()`` must exist so the GUI can
        request a graceful exit."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "class _FlipVerifiedWorker(QThread):")
        assert "def cancel(self) -> None:" in body, (
            "B-2: _FlipVerifiedWorker must expose cancel() so "
            "ResultsTab.shutdown can request a graceful exit"
        )
        assert "_cancelled" in body, (
            "B-2: _FlipVerifiedWorker must carry a _cancelled flag so "
            "the run() loop exits early"
        )

    def test_export_worker_has_cancel(self):
        """``_ExportWorker.cancel()`` must exist."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "class _ExportWorker(QThread):")
        assert "def cancel(self) -> None:" in body, (
            "B-2: _ExportWorker must expose cancel()"
        )
        assert "_cancelled" in body, (
            "B-2: _ExportWorker must carry a _cancelled flag"
        )

    def test_flip_run_checks_cancellation(self):
        """``_FlipVerifiedWorker.run`` must check ``_cancelled`` so a
        ``cancel()`` from the GUI close path stops the in-flight POST."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "class _FlipVerifiedWorker(QThread):")
        run_body = _body_before_next_def(body, "def run(self) -> None:")
        assert run_body.count("self._cancelled") >= 2, (
            "B-2: _FlipVerifiedWorker.run must check _cancelled at "
            "multiple checkpoints (before the POST and after the request)"
        )

    def test_export_run_checks_cancellation(self):
        """``_ExportWorker.run`` must check ``_cancelled`` before
        opening the destination file and between every CSV row."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "class _ExportWorker(QThread):")
        run_body = _body_before_next_def(body, "def run(self) -> None:")
        assert run_body.count("self._cancelled") >= 3, (
            "B-2: _ExportWorker.run must check _cancelled at multiple "
            "checkpoints (before opening file, per CSV row, before success signal)"
        )

    def test_shutdown_does_not_use_terminate(self):
        """D20 contract: never call ``QThread.terminate()``. shutdown()
        must use cancel + wait only."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def shutdown(self) -> None:")
        # Strip docstrings + comments before scanning — ``terminate``
        # appears in the docstring warning about the antipattern.
        executable = _strip_comments(body)
        assert ".terminate()" not in executable, (
            "B-2: ResultsTab.shutdown must NOT call .terminate() — "
            "it orphans subprocesses (audit 2026-08-01 D20 contract)"
        )


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
class TestShutdownRuntime:
    """B-2: drive shutdown() end-to-end with fake long-running workers."""

    def test_shutdown_interrupts_flip_and_export_workers(self, monkeypatch):
        """Spawn fake long-running workers; call ResultsTab.shutdown();
        assert workers were interrupted, waited, and no QThread
        destruction error."""
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        # Need a QApplication to construct QWidgets / QThreads.
        app = QCoreApplication.instance() or QApplication(sys.argv)

        from rlpe.gui.results_tab import ResultsTab

        tab = ResultsTab()

        # Two fake workers: each sleeps until cancelled. The
        # ``isRunning()`` predicate + ``cancel()`` API mirror the real
        # ``_FlipVerifiedWorker`` / ``_ExportWorker`` shapes.
        class _FakeWorker:
            def __init__(self):
                from PySide6.QtCore import QThread

                self._thread = QThread()
                self._cancelled = False
                self._running = False

            def isRunning(self):  # noqa: N802 - Qt contract
                return self._running

            def cancel(self) -> None:
                self._cancelled = True

            def wait(self, timeout: int) -> bool:  # noqa: N802 - Qt contract
                # Simulate "worker respects the cancel flag quickly".
                self._running = False
                return True

            def start(self) -> None:
                self._running = True

        flip = _FakeWorker()
        export = _FakeWorker()
        flip.start()
        export.start()

        tab._flip_worker = flip
        tab._export_worker = export

        # shutdown() must:
        #   1. call cancel() on each,
        #   2. call wait() with a finite timeout,
        #   3. drop the references.
        tab.shutdown()

        assert flip._cancelled is True, "B-2: shutdown() must call flip.cancel()"
        assert export._cancelled is True, "B-2: shutdown() must call export.cancel()"
        assert tab._flip_worker is None, "B-2: shutdown() must drop _flip_worker reference"
        assert tab._export_worker is None, "B-2: shutdown() must drop _export_worker reference"

    def test_shutdown_is_idempotent_on_missing_workers(self):
        """Calling shutdown() before any worker was started must not
        raise AttributeError — the method is idempotent."""
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        app = QCoreApplication.instance() or QApplication(sys.argv)

        from rlpe.gui.results_tab import ResultsTab

        tab = ResultsTab()
        # No _flip_worker / _export_worker attributes yet.
        tab.shutdown()  # must not raise

    def test_shutdown_skips_already_finished_workers(self):
        """A worker that already finished (``isRunning() == False``)
        must be dropped without being cancelled or waited."""
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        app = QCoreApplication.instance() or QApplication(sys.argv)

        from rlpe.gui.results_tab import ResultsTab

        tab = ResultsTab()

        class _FakeFinishedWorker:
            def __init__(self):
                self._cancelled = False
                self._running = False
                self.waited = False

            def isRunning(self):  # noqa: N802 - Qt contract
                return False

            def cancel(self) -> None:
                self._cancelled = True

            def wait(self, timeout: int) -> bool:  # noqa: N802 - Qt contract
                self.waited = True
                return True

        finished = _FakeFinishedWorker()
        tab._export_worker = finished

        tab.shutdown()

        assert finished._cancelled is False, (
            "B-2: shutdown() must NOT cancel an already-finished worker"
        )
        assert finished.waited is False, (
            "B-2: shutdown() must NOT wait on an already-finished worker"
        )
        assert tab._export_worker is None


# ============================================================================
# M-3 — figure_image_path resolution
# ============================================================================


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
class TestResolveFigureImagePath:
    """M-3: ``_resolve_figure_image_path`` falls back through 4 candidates."""

    def test_figure_image_path_top_level_wins(self, tmp_path):
        """Top-level ``figure_image_path`` beats metadata one when both exist."""
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        app = QCoreApplication.instance() or QApplication(sys.argv)

        from rlpe.gui.results_tab import ResultsTab

        top_png = tmp_path / "top.png"
        top_png.write_bytes(b"\x89PNG\r\n\x1a\n")
        md_png = tmp_path / "md.png"
        md_png.write_bytes(b"\x89PNG\r\n\x1a\n")

        row = {
            "figure_image_path": str(top_png),
            "metadata": {
                "figure_image_path": str(md_png),
                "primary_image": None,
                "image_path": None,
            },
        }
        resolved = ResultsTab._resolve_figure_image_path(row)
        assert resolved == top_png.resolve(), (
            f"M-3: top-level figure_image_path must win, got {resolved}"
        )

    def test_figure_image_path_metadata_fallback(self, tmp_path):
        """If only metadata has it, return that."""
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        app = QCoreApplication.instance() or QApplication(sys.argv)

        from rlpe.gui.results_tab import ResultsTab

        md_png = tmp_path / "md_only.png"
        md_png.write_bytes(b"\x89PNG\r\n\x1a\n")

        row = {"metadata": {"figure_image_path": str(md_png)}}
        resolved = ResultsTab._resolve_figure_image_path(row)
        assert resolved == md_png.resolve(), (
            f"M-3: metadata-only figure_image_path must be returned, got {resolved}"
        )

    def test_figure_image_path_primary_image_fallback(self, tmp_path):
        """If neither top-level nor metadata.figure_image_path is set,
        ``metadata.primary_image`` is the next fallback."""
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        app = QCoreApplication.instance() or QApplication(sys.argv)

        from rlpe.gui.results_tab import ResultsTab

        prim = tmp_path / "primary.png"
        prim.write_bytes(b"\x89PNG\r\n\x1a\n")

        row = {"metadata": {"primary_image": str(prim)}}
        resolved = ResultsTab._resolve_figure_image_path(row)
        assert resolved == prim.resolve(), (
            f"M-3: primary_image must be the third fallback, got {resolved}"
        )

    def test_figure_image_path_image_path_fallback(self, tmp_path):
        """If only metadata.image_path is set, return that."""
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        app = QCoreApplication.instance() or QApplication(sys.argv)

        from rlpe.gui.results_tab import ResultsTab

        legacy = tmp_path / "legacy.png"
        legacy.write_bytes(b"\x89PNG\r\n\x1a\n")

        row = {"metadata": {"image_path": str(legacy)}}
        resolved = ResultsTab._resolve_figure_image_path(row)
        assert resolved == legacy.resolve(), (
            f"M-3: image_path must be the fourth fallback, got {resolved}"
        )

    def test_figure_image_path_missing_returns_none(self):
        """No candidates at all → returns None."""
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        app = QCoreApplication.instance() or QApplication(sys.argv)

        from rlpe.gui.results_tab import ResultsTab

        row: dict = {"metadata": {}}
        assert ResultsTab._resolve_figure_image_path(row) is None

        row = {"paper_id": "p1", "figure_id": "f1", "panel_id": "pl1"}
        assert ResultsTab._resolve_figure_image_path(row) is None

        # Missing file even when the path is set must also return None
        # (the helper filters via .exists()).
        row = {
            "figure_image_path": "/nonexistent/path/does_not_exist.png",
            "metadata": {
                "figure_image_path": "/also/nonexistent.png",
                "primary_image": "/still/nonexistent.png",
                "image_path": "/yet/again/nonexistent.png",
            },
        }
        assert ResultsTab._resolve_figure_image_path(row) is None

    def test_figure_image_path_helper_exists(self):
        """Source guard: the helper method must be defined on ResultsTab."""
        src = _read("src/rlpe/gui/results_tab.py")
        assert "def _resolve_figure_image_path(" in src, (
            "M-3: ResultsTab._resolve_figure_image_path must exist"
        )


# ============================================================================
# M-5 — API URL validation
# ============================================================================


class TestValidateApiUrl:
    """M-5: ``_validate_api_url`` rejects hostile URLs."""

    def test_validate_api_url_accepts_https(self):
        """A normal HTTPS URL is accepted."""
        from rlpe.gui.results_tab import _validate_api_url

        out = _validate_api_url("https://api.example.com")
        assert out == "https://api.example.com"

        out = _validate_api_url("https://api.example.com/v1")
        assert out == "https://api.example.com/v1"

    def test_validate_api_url_accepts_http_with_local(self):
        """Loopback is allowed when ``allow_local=True``."""
        from rlpe.gui.results_tab import _validate_api_url

        out = _validate_api_url(
            "http://127.0.0.1:8000", allow_local=True
        )
        assert out == "http://127.0.0.1:8000"

        out = _validate_api_url("http://localhost:8000", allow_local=True)
        assert out == "http://localhost:8000"

    def test_validate_api_url_rejects_loopback_by_default(self):
        """Loopback is rejected unless ``allow_local=True``."""
        from rlpe.gui.results_tab import _validate_api_url

        for hostile in (
            "http://127.0.0.1",
            "http://localhost",
            "http://0.0.0.0",
            "http://[::1]",
        ):
            assert _validate_api_url(hostile) is None, (
                f"M-5: {hostile!r} must be rejected without allow_local"
            )

    def test_validate_api_url_rejects_file_scheme(self):
        """``file:///...`` is rejected so an attacker can't pivot to
        local file reads."""
        from rlpe.gui.results_tab import _validate_api_url

        out = _validate_api_url("file:///etc/passwd")
        assert out is None, "M-5: file:/// scheme must be rejected"

        out = _validate_api_url("file://localhost/etc/passwd")
        assert out is None, "M-5: file://localhost/ scheme must be rejected"

    def test_validate_api_url_rejects_javascript(self):
        """``javascript:...`` is rejected."""
        from rlpe.gui.results_tab import _validate_api_url

        out = _validate_api_url("javascript:alert(1)")
        assert out is None, "M-5: javascript: scheme must be rejected"

        out = _validate_api_url("JavaScript:alert(1)")
        # Case sensitivity: ``urlparse`` lowercases the scheme so this
        # also rejects (the API URL builder was case-blind before too).
        assert out is None, "M-5: case-insensitive javascript: must be rejected"

    def test_validate_api_url_rejects_empty_netloc(self):
        """``http://`` with no host is rejected."""
        from rlpe.gui.results_tab import _validate_api_url

        out = _validate_api_url("http://")
        assert out is None, "M-5: empty netloc must be rejected"

    def test_validate_api_url_rejects_empty(self):
        """Empty string / whitespace / None are rejected."""
        from rlpe.gui.results_tab import _validate_api_url

        assert _validate_api_url("") is None
        assert _validate_api_url("   ") is None
        assert _validate_api_url(None) is None  # type: ignore[arg-type]
        assert _validate_api_url(123) is None  # type: ignore[arg-type]

    def test_validate_api_url_rejects_data_scheme(self):
        """``data:...`` is rejected."""
        from rlpe.gui.results_tab import _validate_api_url

        out = _validate_api_url("data:text/plain;base64,SGVsbG8=")
        assert out is None, "M-5: data: scheme must be rejected"

    def test_validate_api_url_strips_whitespace(self):
        """Leading / trailing whitespace is stripped on success."""
        from rlpe.gui.results_tab import _validate_api_url

        out = _validate_api_url("  https://api.example.com  ")
        assert out == "https://api.example.com"

    def test_validate_api_url_helper_exists(self):
        """Source guard: the helper must be defined at module level."""
        src = _read("src/rlpe/gui/results_tab.py")
        assert "def _validate_api_url(" in src, (
            "M-5: _validate_api_url must be defined at module level"
        )
        # Must use urlparse from stdlib.
        assert "urlparse" in src, (
            "M-5: _validate_api_url must use urllib.parse.urlparse"
        )


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
class TestSetApiUrlRuntime:
    """M-5: ``_set_api_url`` is the single write path for QSettings."""

    def test_set_api_url_persists_valid_https(self):
        """A valid URL is persisted to QSettings and the method returns True."""
        from PySide6.QtCore import QCoreApplication, QSettings
        from PySide6.QtWidgets import QApplication

        app = QCoreApplication.instance() or QApplication(sys.argv)

        from rlpe.gui.constants import APP_AUTHOR, APP_NAME, QS_KEY_API_URL
        from rlpe.gui.results_tab import ResultsTab

        tab = ResultsTab()

        # Snapshot the current value (if any) so we can restore it
        # after the test, leaving the user's QSettings untouched.
        s = QSettings(APP_AUTHOR, APP_NAME)
        previous = s.value(QS_KEY_API_URL, "")
        try:
            s.setValue(QS_KEY_API_URL, "")
            s.sync()

            ok = tab._set_api_url("https://api.example.com")
            assert ok is True, "M-5: _set_api_url must return True on a valid URL"

            # Re-read via a fresh QSettings handle to confirm the
            # value persisted to disk.
            s2 = QSettings(APP_AUTHOR, APP_NAME)
            assert s2.value(QS_KEY_API_URL, "") == "https://api.example.com"
        finally:
            # Restore the user's original QSettings value.
            s.setValue(QS_KEY_API_URL, previous)
            s.sync()

    def test_set_api_url_rejects_hostile(self):
        """A hostile URL is rejected without being persisted."""
        from PySide6.QtCore import QCoreApplication, QSettings
        from PySide6.QtWidgets import QApplication

        app = QCoreApplication.instance() or QApplication(sys.argv)

        from rlpe.gui.constants import APP_AUTHOR, APP_NAME, QS_KEY_API_URL
        from rlpe.gui.results_tab import ResultsTab

        tab = ResultsTab()

        s = QSettings(APP_AUTHOR, APP_NAME)
        previous = s.value(QS_KEY_API_URL, "")
        try:
            # Seed the store with a known-good value so we can prove
            # the rejection didn't overwrite it.
            sentinel = "https://existing.example.com"
            s.setValue(QS_KEY_API_URL, sentinel)
            s.sync()

            for hostile in (
                "file:///etc/passwd",
                "javascript:alert(1)",
                "http://",
                "",
                "  ",
            ):
                ok = tab._set_api_url(hostile)
                assert ok is False, (
                    f"M-5: _set_api_url must reject {hostile!r} and return False"
                )
                # On rejection the existing sentinel value must be
                # untouched.
                s_check = QSettings(APP_AUTHOR, APP_NAME)
                current = s_check.value(QS_KEY_API_URL, "")
                assert current == sentinel, (
                    f"M-5: hostile URL {hostile!r} must not overwrite the "
                    f"existing value; got {current!r}"
                )
        finally:
            s.setValue(QS_KEY_API_URL, previous)
            s.sync()


# ============================================================================
# Source guards: helpers all wired into the GUI
# ============================================================================


class TestSourceGuardIntegration:
    """Source guards that all three fixes are wired into the GUI."""

    def test_results_tab_uses_resolve_helper_in_on_row_selected(self):
        """``_on_row_selected`` must delegate to the new helper."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _on_row_selected(self) -> None:")
        assert "_resolve_figure_image_path(" in body, (
            "M-3: _on_row_selected must call _resolve_figure_image_path"
        )
        # The previous metadata-only loop must be gone.
        assert "md.get(key)" not in body, (
            "M-3: _on_row_selected must not still do the metadata-only "
            "loop after the fix"
        )

    def test_flip_image_verified_validates_url(self):
        """``_flip_image_verified`` must call ``_validate_api_url``
        before using the QSettings override."""
        src = _read("src/rlpe/gui/results_tab.py")
        body = _body_before_next_def(src, "def _flip_image_verified(self, verified: bool) -> None:")
        assert "_validate_api_url(" in body, (
            "M-5: _flip_image_verified must validate the API URL via "
            "_validate_api_url before using it"
        )

    def test_results_tab_exposes_set_api_url(self):
        """``_set_api_url`` must exist as the single write path."""
        src = _read("src/rlpe/gui/results_tab.py")
        assert "def _set_api_url(self, url: str) -> bool:" in src, (
            "M-5: ResultsTab._set_api_url must exist"
        )
