"""Regression tests for audit 2026-08-01 batch W2 — gui D20 QThread.terminate replaced.

Bug D20: ``QThread.terminate()`` was called after a 2-5s wait in both
``run_tab.py`` and ``main_window.py``. ``terminate()`` forcibly kills
the QThread mid-Python execution, which can:

1. Orphan subprocesses (OpenDataLoader JVM, in-flight LLM HTTP requests)
2. Leave partial temp dirs like ``od_output/<paper_id>/``
3. Corrupt Qt signal/slot connections (worker is destroyed while a slot
   is mid-execution)

Fix:
  * Replace ``terminate()`` with ``requestInterruption()`` +
    ``wait(timeout_ms)`` (timeout 30s).
  * After the wait timeout, log a warning — do NOT forcibly kill the
    thread. The OS will reclaim the thread on process exit.
  * Do NOT forcibly kill the OpenDataLoader JVM if it is still alive —
    let it finish.

These tests assert the fix is in place by source-level inspection
(avoids the overhead of spawning a real QThread + offscreen Qt event
loop). They will fail if a future refactor reintroduces ``terminate()``
in either file.
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
pytest.importorskip("PySide6")

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_RUN_TAB = _SRC / "rlpe" / "gui" / "run_tab.py"
_MAIN_WINDOW = _SRC / "rlpe" / "gui" / "main_window.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _slice_cleanup_block(source: str, anchor: str, window: int = 40) -> str:
    """Return ~``window`` lines after ``anchor`` for scoped assertions.

    We don't want to assert against the whole file (would catch
    unrelated ``terminate()`` references in comments); we want the
    block that actually runs when the worker is being shut down.
    """
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if anchor in line:
            start = i
            end = min(len(lines), i + window)
            return "\n".join(lines[start:end])
    raise AssertionError(f"anchor {anchor!r} not found in source")


class TestGuiWorkerShutdown:
    """Source-level regression tests for audit 2026-08-01 D20."""

    def test_uses_requestInterruption_not_terminate(self):
        """Both files must use ``requestInterruption()`` instead of
        ``terminate()`` in the worker-shutdown block."""
        run_src = _read(_RUN_TAB)
        main_src = _read(_MAIN_WINDOW)

        # Locate the cleanup block (right after the signal-disconnect
        # try/except) and assert the *executable* shutdown logic uses
        # ``requestInterruption`` and not ``terminate``.
        run_block = _slice_cleanup_block(run_src, "if worker.isRunning():", window=40)
        main_block = _slice_cleanup_block(main_src, "def _stop_pipeline_worker", window=60)

        # ``requestInterruption(`` MUST appear in the cleanup block.
        # In run_tab.py this is the explicit method call; in
        # main_window.py it is invoked transitively via
        # ``request_cancel()``, which internally calls
        # ``requestInterruption()`` (see pipeline_worker.py:99). We
        # assert the explicit form in run_tab and the transitive form
        # in main_window.
        assert "requestInterruption(" in run_block, (
            "run_tab.py cleanup block must call "
            "``requestInterruption()`` instead of ``terminate()``"
        )
        assert "request_cancel()" in main_block, (
            "main_window.py _stop_pipeline_worker must call "
            "``request_cancel()`` (which internally invokes "
            "``requestInterruption()``)"
        )
        # ``request_cancel`` in pipeline_worker.py delegates to
        # ``self.requestInterruption()`` — verify the transitive call.
        worker_src = _read(_SRC / "rlpe" / "gui" / "pipeline_worker.py")
        assert "self.requestInterruption()" in worker_src, (
            "PipelineWorker.request_cancel must call "
            "``self.requestInterruption()`` so the cancel flag is set"
        )

        # ``terminate()`` MUST NOT appear in either cleanup block.
        # Strip trailing comments / docstrings from the block before
        # scanning — we only care about *executable* calls.
        run_executable = _strip_comments(run_block)
        main_executable = _strip_comments(main_block)
        assert ".terminate()" not in run_executable, (
            "run_tab.py cleanup block must NOT call ``worker.terminate()`` "
            "(audit 2026-08-01 D20). It orphans subprocesses."
        )
        assert ".terminate()" not in main_executable, (
            "main_window.py _stop_pipeline_worker must NOT call "
            "``worker.terminate()`` (audit 2026-08-01 D20)."
        )

    def test_wait_timeout_configurable(self):
        """The ``wait(timeout)`` call must have a finite, non-None
        timeout (otherwise the GUI hangs forever on shutdown)."""
        run_src = _read(_RUN_TAB)
        main_src = _read(_MAIN_WINDOW)

        for label, src in (("run_tab.py", run_src), ("main_window.py", main_src)):
            # Find every ``.wait(<timeout>)`` call. We want at least
            # one in each cleanup block with a numeric timeout.
            matches = re.findall(r"\.wait\(\s*(\d+|None)\s*\)", src)
            assert matches, (
                f"{label} has no ``.wait(timeout)`` call — the GUI shutdown would hang indefinitely"
            )
            for m in matches:
                assert m != "None", (
                    f"{label} has ``.wait(None)`` — this blocks forever "
                    "and is exactly the hang we are trying to prevent"
                )
                timeout_ms = int(m)
                # 30s = 30000ms is the new contract; older calls used
                # 2000/500/5000ms. All finite timeouts are acceptable
                # so long as None / 0 is not used.
                assert timeout_ms > 0, (
                    f"{label} has ``.wait({timeout_ms})`` — zero-timeout is a bug"
                )

    def test_30s_timeout_in_both_cleanup_blocks(self):
        """The new contract is ``wait(30000)`` in both files."""
        run_block = _slice_cleanup_block(_read(_RUN_TAB), "if worker.isRunning():", window=40)
        main_block = _slice_cleanup_block(
            _read(_MAIN_WINDOW), "def _stop_pipeline_worker", window=60
        )
        assert ".wait(30000)" in run_block, (
            "run_tab.py cleanup block must use ``wait(30000)`` (audit 2026-08-01 D20 contract)"
        )
        assert ".wait(30000)" in main_block, (
            "main_window.py _stop_pipeline_worker must use "
            "``wait(30000)`` (audit 2026-08-01 D20 contract)"
        )

    def test_no_lingering_terminate_calls_in_worker_cleanup(self):
        """Source guard: no ``.terminate()`` calls remain anywhere in
        run_tab.py or main_window.py (excluding docstrings/comments).
        ``QThread.terminate()`` is dangerous and must not be used in
        the GUI cleanup path."""
        for label, path in (
            ("run_tab.py", _RUN_TAB),
            ("main_window.py", _MAIN_WINDOW),
        ):
            src = _read(path)
            executable = _strip_comments(src)
            # ``.terminate()`` only matches method calls; not the
            # word "terminate" in a docstring or comment.
            assert ".terminate()" not in executable, (
                f"{label} still has a ``.terminate()`` call. Audit "
                "2026-08-01 D20 forbids this — use "
                "``requestInterruption()`` + ``wait(timeout)`` "
                "instead so subprocesses (JVM, LLM HTTP) are not "
                "orphaned."
            )


def _strip_comments(src: str) -> str:
    """Return ``src`` with line and block comments stripped.

    Used to avoid false positives when scanning for ``.terminate()``
    in source — we don't want a comment like
    ``# NOTE: don't add terminate() here`` to fail the test.
    """
    # Strip block comments (rare in Python but harmless).
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    # Strip full-line comments.
    lines = []
    for line in src.splitlines():
        # Drop everything after ``#`` (naive but good enough: the
        # GUI source does not embed ``#`` inside string literals in
        # the cleanup blocks under test).
        stripped = line.split("#", 1)[0]
        lines.append(stripped)
    return "\n".join(lines)
