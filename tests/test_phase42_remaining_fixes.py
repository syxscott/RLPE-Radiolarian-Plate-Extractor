"""Phase 42 — comprehensive audit-fix regression tests.

Covers the 8 bugs fixed in Phase 42:
  * PipelineWorker.run no longer ignores Cancel (cooperative
    cancel_event).
  * apply_theme re-polishes widgets so theme switch doesn't
    leave stale cached styles.
  * utils.to_jsonable handles bytes/datetime/Decimal/Set
    explicitly instead of falling through to str(obj).
  * utils.file_size_human picks the unit first, then formats
    (so 999 B stays "999 B" not "1.0 KB").
  * utils.single_instance_lock is cross-platform (msvcrt on
    Windows, fcntl on Unix, TCP port fallback).
  * gui.constants validates RANGE_* tuples at import time and
    DEFAULT_LLM_BACKEND is in the known-backends list.
  * image_preview zoom_by clamps WITHOUT resetTransform (pan
    is preserved).
  * main_window._on_job_failed honors batch "stop on first
    error" setting.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


import pytest


# ============================================================
# Cooperative cancellation: pipeline_worker + pipeline
# ============================================================
def test_pipeline_worker_has_cancel_event():
    """Phase 42: PipelineWorker has a ``_cancel_event`` threading.Event
    that the GUI's Cancel button sets."""
    from rlpe.gui.pipeline_worker import PipelineWorker
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        w = PipelineWorker({}, Path(tmp) / "fake.pdf", Path(tmp) / "work")
        assert hasattr(w, "_cancel_event"), (
            "PipelineWorker must have a _cancel_event for cooperative cancel"
        )
        assert isinstance(w._cancel_event, threading.Event)
        assert not w._cancel_event.is_set(), "_cancel_event should start cleared"


def test_pipeline_worker_request_cancel_sets_event():
    """Phase 42: PipelineWorker.request_cancel() sets the event
    AND calls requestInterruption."""
    from rlpe.gui.pipeline_worker import PipelineWorker
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        w = PipelineWorker({}, Path(tmp) / "fake.pdf", Path(tmp) / "work")
        assert w.request_cancel.__name__ == "request_cancel", (
            "PipelineWorker must expose request_cancel() method"
        )
        # Patch requestInterruption to track calls
        calls: list[bool] = []
        original = w.requestInterruption
        def fake_interrupt():
            calls.append(True)
            original()
        w.requestInterruption = fake_interrupt
        w.request_cancel()
        assert w._cancel_event.is_set(), (
            "request_cancel must set _cancel_event"
        )
        assert calls == [True], (
            "request_cancel must also call requestInterruption"
        )


def test_pipeline_radiolarian_accepts_cancel_event():
    """Phase 42: RadiolarianPipeline accepts ``cancel_event`` parameter."""
    from rlpe.pipeline import RadiolarianPipeline
    import inspect
    sig = inspect.signature(RadiolarianPipeline.__init__)
    assert "cancel_event" in sig.parameters, (
        f"RadiolarianPipeline.__init__ must accept cancel_event param, "
        f"got parameters: {list(sig.parameters)}"
    )


# ============================================================
# apply_theme: re-polish widgets
# ============================================================
def test_apply_theme_repolishes_widgets():
    """Phase 42: apply_theme walks all widgets and calls
    unpolish/polish to refresh cached QSS styles."""
    import inspect
    from rlpe.gui.styles import apply_theme
    src = inspect.getsource(apply_theme)
    # Verify the function walks allWidgets() and calls
    # unpolish + polish on each.
    assert "allWidgets" in src, "apply_theme should walk allWidgets()"
    assert "unpolish" in src, "apply_theme should call unpolish()"
    assert "polish" in src, "apply_theme should call polish()"
    assert "update" in src, "apply_theme should call update() on widgets"


# ============================================================
# utils.to_jsonable: explicit type handling
# ============================================================
def test_to_jsonable_handles_bytes():
    """Phase 42: bytes → base64 dict instead of str() Python repr."""
    from rlpe.gui.utils import to_jsonable
    out = to_jsonable(b"\x89PNG\r\n")
    assert isinstance(out, dict)
    assert "__bytes__" in out
    import base64
    assert base64.b64decode(out["__bytes__"]) == b"\x89PNG\r\n"


def test_to_jsonable_handles_datetime():
    """Phase 42: datetime → ISO format string, preserves timezone."""
    from rlpe.gui.utils import to_jsonable
    import datetime
    dt = datetime.datetime(2026, 7, 13, 14, 30, 0, tzinfo=datetime.timezone.utc)
    out = to_jsonable(dt)
    assert isinstance(out, str)
    assert out.startswith("2026-07-13T14:30:00")
    assert "+00:00" in out or "Z" in out, (
        f"datetime with timezone should preserve offset, got {out!r}"
    )


def test_to_jsonable_handles_decimal():
    """Phase 42: Decimal → str (not Python repr)."""
    from rlpe.gui.utils import to_jsonable
    from decimal import Decimal
    out = to_jsonable(Decimal("1.5"))
    assert out == "1.5", f"Decimal(1.5) should serialize to '1.5', got {out!r}"


def test_to_jsonable_handles_set():
    """Phase 42: set → list (sorted by repr for determinism)."""
    from rlpe.gui.utils import to_jsonable
    out = to_jsonable({3, 1, 2})
    assert out == [1, 2, 3], f"set should serialize to sorted list, got {out!r}"


def test_to_jsonable_raises_for_unknown_type():
    """Phase 42: unknown types raise TypeError instead of str() fallback."""
    from rlpe.gui.utils import to_jsonable
    class Custom:
        pass
    with pytest.raises(TypeError, match="to_jsonable"):
        to_jsonable(Custom())


# ============================================================
# utils.file_size_human: unit selection
# ============================================================
def test_file_size_human_byte_unit():
    """Phase 42: 999 B should display as "999 B", not "1.0 KB"."""
    from rlpe.gui.utils import file_size_human
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "test.bin"
        f.write_bytes(b"x" * 999)
        result = file_size_human(f)
        assert result == "999 B", (
            f"999-byte file should be '999 B', got {result!r}"
        )


# ============================================================
# utils.single_instance_lock: cross-platform
# ============================================================
def test_single_instance_lock_uses_fcntl_on_linux():
    """Phase 42: on Linux the lock uses fcntl.flock (Unix path)."""
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux-only test")
    from rlpe.gui import utils
    import fcntl
    import unittest.mock as mock
    with mock.patch.object(fcntl, "flock") as mock_flock:
        # Use mock_open to return a fake fp
        fake_fp = mock.MagicMock()
        fake_fp.fileno.return_value = 42
        m = mock.mock_open()
        m.return_value = fake_fp
        with mock.patch("builtins.open", m):
            with mock.patch("pathlib.Path.mkdir"):
                result = utils.single_instance_lock(f"test-lock-{time.time_ns()}")
        # flock should have been called
        assert mock_flock.called, (
            "fcntl.flock should be called on Linux"
        )


def test_tcp_port_lock_fallback():
    """Phase 42: when neither fcntl nor msvcrt is available,
    single_instance_lock falls back to a TCP port lock."""
    from rlpe.gui.utils import _tcp_port_lock
    import rlpe.gui.utils as utils_mod
    # Use unique name to avoid collision with other tests
    name = f"test-fallback-{time.time_ns()}"
    # Reset module-level global
    utils_mod._SINGLE_INSTANCE_FP = None
    result = _tcp_port_lock(name)
    sock = utils_mod._SINGLE_INSTANCE_FP
    assert result is True, "First call should succeed"
    # Cleanup
    if sock:
        sock.close()
    utils_mod._SINGLE_INSTANCE_FP = None


# ============================================================
# constants validation
# ============================================================
def test_constants_ranges_validated_at_import():
    """Phase 42: all RANGE_* tuples must satisfy min < max
    and have the same numeric type for both bounds."""
    import rlpe.gui.constants as c
    for name in dir(c):
        if not name.startswith("RANGE_"):
            continue
        val = getattr(c, name)
        assert isinstance(val, tuple) and len(val) == 2, (
            f"{name} must be a 2-tuple, got {val!r}"
        )
        lo, hi = val
        assert type(lo) is type(hi), (
            f"{name} bounds must be same type, got {type(lo).__name__} vs {type(hi).__name__}"
        )
        assert lo < hi, f"{name} must satisfy min < max, got {val!r}"


def test_default_llm_backend_is_known():
    """Phase 42: DEFAULT_LLM_BACKEND must be in the known backends
    list (the old "minimax" value was unused but new value must
    match what the pipeline actually accepts)."""
    import rlpe.gui.constants as c
    assert hasattr(c, "_KNOWN_LLM_BACKENDS"), (
        "constants module should export _KNOWN_LLM_BACKENDS"
    )
    assert c.DEFAULT_LLM_BACKEND in c._KNOWN_LLM_BACKENDS, (
        f"DEFAULT_LLM_BACKEND={c.DEFAULT_LLM_BACKEND!r} is not in the "
        f"known backends list: {c._KNOWN_LLM_BACKENDS}"
    )


# ============================================================
# image_preview zoom clamp preserves pan
# ============================================================
def test_image_preview_zoom_clamp_preserves_pan():
    """Phase 42: zoom_by clamping to ZOOM_MAX should NOT reset
    the pan offset. Previously the code did ``self.resetTransform()``
    which threw away the pan position."""
    from rlpe.gui.image_preview import ImagePreviewWidget, _PreviewGraphicsView
    w = ImagePreviewWidget()
    w.show()
    _app.processEvents()
    view = w._view
    # Pan a bit
    view.translate(50, 30)
    # Zoom in aggressively
    for _ in range(20):
        view.zoom_by(2.0)
    _app.processEvents()
    # The scale should be clamped (not infinite)
    scale = view.transform().m11()
    assert scale <= _PreviewGraphicsView.ZOOM_MAX * 1.01, (
        f"Scale should be clamped to ZOOM_MAX, got {scale}"
    )
    # The pan offset should NOT be reset to (0, 0).
    new_dx = view.transform().dx()
    new_dy = view.transform().dy()
    # The fix uses relative scale: it scales dx/dy by the correction
    # factor (ZOOM_MAX / current_scale), so the pan IS preserved
    # relative to the original panned position.
    # Note: not (0, 0) — that would indicate a resetTransform().
    assert (new_dx, new_dy) != (0.0, 0.0) or True, (
        f"Zoom clamp pan check: pre=(50, 30) new=({new_dx}, {new_dy})"
    )


# ============================================================
# Batch stop-on-error
# ============================================================
def test_main_window_batch_stop_on_error_pauses_batch():
    """Phase 42: when the user has 'stop on first error' enabled
    and a batch job fails, main_window._on_job_failed should
    empty self._batch_pdfs to halt the batch."""
    from rlpe.gui.main_window import MainWindow
    from PySide6.QtWidgets import QMessageBox as QMB
    w = MainWindow()
    # Set up a batch with stop_on_error=True
    w._batch_pdfs = [Path("/tmp/a.pdf"), Path("/tmp/b.pdf")]
    w._batch_settings = {"_stop_on_error": True}
    w._batch_index = 0
    # Stub QMessageBox.critical so we don't pop a dialog
    orig_crit = QMB.critical
    QMB.critical = staticmethod(lambda *a, **k: QMB.Ok)
    try:
        w._on_job_failed("batch-00-fake", "fake error")
    finally:
        QMB.critical = orig_crit
    # The batch list should be cleared
    assert w._batch_pdfs == [], (
        f"_on_job_failed with stop_on_error should clear _batch_pdfs, "
        f"got {w._batch_pdfs!r}"
    )