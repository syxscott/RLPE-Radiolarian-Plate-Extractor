"""Phase 41 — comprehensive audit of the remaining GUI files.

The Phase 37 audit covered the most user-visible GUI. Phase 41
goes deeper into:

  * run_tab._on_start — the BLOCKER race (double-click Start
    spawns two workers) and the bare-English labels that
    didn't translate on language switch.
  * batch_dialog — the "Clear all" button only cleared the
    QListWidget but left self._pdfs populated, so the next
    Start would still try to process them. Plus bare-English
    labels.
  * image_preview — set_image() didn't clear the prior
    QGraphicsRectItem overlays, so bboxes piled up across image
    swaps. The audit found this leak.
  * styles.py — the `PROGRESS_BAR_MIN_HEIGHT_PX if False else 18`
    dead-code branch.

These tests pin the fixes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest
pytest.importorskip("PySide6")
from PySide6.QtGui import QColor, QImage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


import pytest


@pytest.fixture(autouse=True)
def _reset_language():
    from rlpe.gui import i18n
    i18n.set_language("zh_CN")
    yield
    i18n.set_language("zh_CN")


# ============================================================
# run_tab._on_start — BLOCKER: double-click race
# ============================================================
def test_run_tab_blocks_double_click_start():
    """Phase 41: clicking Start twice should NOT spawn two
    PipelineWorker threads. The first click starts the job;
    the second is silently ignored (with a log warning)."""
    from unittest.mock import patch, MagicMock
    from rlpe.gui.run_tab import RunTab
    from PySide6.QtWidgets import QMessageBox

    rt = RunTab({})
    # Patch the worker so we can count instances without actually
    # starting a real pipeline.
    workers: list = []
    def fake_worker(*a, **kw):
        m = MagicMock()
        m.isRunning.return_value = True  # so the guard triggers
        m.isRunning = lambda: True
        workers.append(m)
        return m
    with patch("rlpe.gui.run_tab.PipelineWorker", side_effect=fake_worker), \
         patch.object(QMessageBox, "warning", return_value=QMessageBox.Ok), \
         patch.object(rt, "_on_progress", lambda *a, **k: None), \
         patch.object(rt, "_on_finished", lambda *a, **k: None):
        # Pre-set valid inputs
        rt._path_edit.setText("/tmp/dummy.pdf")
        # We need a real path that exists for the validation to pass.
        # Use a tmp file.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name
        rt._path_edit.setText(tmp_path)
        rt._out_edit.setText("/tmp/dummy_out")
        try:
            rt._on_start()
            assert len(workers) == 1, (
                f"First click should spawn 1 worker, got {len(workers)}"
            )
            # Second click while first is "running" must be a no-op
            rt._on_start()
            assert len(workers) == 1, (
                f"Second click while running should NOT spawn another worker; "
                f"got {len(workers)} total"
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ============================================================
# run_tab — bare English replaced with i18n keys
# ============================================================
def test_run_tab_status_label_uses_i18n():
    """Phase 41: _status_label.setText("Cancelling…") replaced
    with i18n._tr(\"runtab.status.cancelling\") so it translates."""
    import inspect
    from rlpe.gui.run_tab import RunTab
    src = inspect.getsource(RunTab)
    # The old bare-English hard-coded strings should be gone
    assert '"Cancelling…"' not in src, (
        "run_tab still has bare English 'Cancelling…' string"
    )
    assert '"Pipeline initialising…"' not in src
    # The i18n calls should be there
    assert 'i18n._tr("runtab.status.cancelling")' in src
    assert 'i18n._tr("runtab.progress.init")' in src


# ============================================================
# batch_dialog — Clear All bug
# ============================================================
def test_batch_dialog_clear_all_empties_internal_list():
    """Phase 41: clicking 'Clear all' must empty both the
    QListWidget AND the self._pdfs list. Previously the lambda
    ``self._file_list.clear()`` only cleared the widget but
    self._pdfs was still populated, so the next Start would
    still process them."""
    from rlpe.gui.batch_dialog import BatchDialog
    from PySide6.QtWidgets import QListWidgetItem
    dlg = BatchDialog({})
    dlg._pdfs = [Path("/tmp/a.pdf"), Path("/tmp/b.pdf")]
    dlg._file_list.addItem(QListWidgetItem("/tmp/a.pdf"))
    dlg._file_list.addItem(QListWidgetItem("/tmp/b.pdf"))
    assert dlg._file_list.count() == 2
    # Call the handler directly
    dlg._on_clear_all()
    assert dlg._file_list.count() == 0, (
        "QListWidget must be empty after _on_clear_all"
    )
    assert dlg._pdfs == [], (
        f"self._pdfs must be empty after _on_clear_all, got {dlg._pdfs!r}"
    )


def test_batch_dialog_buttons_translate():
    """Phase 41: batch dialog buttons use tr_button so they
    translate on language switch."""
    from rlpe.gui.batch_dialog import BatchDialog
    dlg = BatchDialog({})
    from PySide6.QtWidgets import QPushButton
    buttons = dlg.findChildren(QPushButton)
    # All buttons should have an objectName (set by tr_button)
    no_obj = [b.text() for b in buttons if not b.objectName()]
    assert not no_obj, (
        f"Batch dialog has QPushButton without objectName (no i18n): {no_obj}"
    )


# ============================================================
# image_preview — bbox leak on set_image
# ============================================================
def test_image_preview_set_image_clears_old_bboxes():
    """Phase 41: set_image() must clear prior QGraphicsRectItem
    overlays so old bboxes don't pile up across image swaps."""
    from rlpe.gui.image_preview import ImagePreviewWidget
    import tempfile
    from PySide6.QtGui import QPixmap, QImage

    # Create two small test PNGs
    def make_png(path, w=20, h=20, color=(255, 0, 0)):
        img = QImage(w, h, QImage.Format_RGB888)
        img.fill(QColor(*color))
        img.save(str(path), "PNG")

    with tempfile.TemporaryDirectory() as tmp:
        img1 = Path(tmp) / "a.png"
        img2 = Path(tmp) / "b.png"
        make_png(img1, color=(255, 0, 0))
        make_png(img2, color=(0, 255, 0))

        w = ImagePreviewWidget()
        w.set_image(img1)
        bboxes = [{
            "bbox": (2, 2, 16, 16),
            "species": "Test",
            "label_text": "Test",
            "confidence": 0.9,
        }]
        w.set_bboxes(bboxes)
        # After set_bboxes, _bbox_items should have 1 item
        assert len(w._bbox_items) == 1, (
            f"After set_bboxes([1 bbox]), _bbox_items should have 1 item, "
            f"got {len(w._bbox_items)}"
        )
        # Now swap image — _overlay_bboxes re-renders the same
        # bboxes (because self._bboxes is still populated), so the
        # count should STILL be 1 (Phase 41 leak fix), not 2.
        w.set_image(img2)
        assert len(w._bbox_items) == 1, (
            f"After set_image swap, _bbox_items should be 1 (re-rendered), "
            f"NOT 2 (leak). Phase 41: got {len(w._bbox_items)}"
        )
        # Now call set_bboxes with empty list — should clear
        w.set_bboxes([])
        assert len(w._bbox_items) == 0, (
            f"After set_bboxes([]), _bbox_items should be 0, "
            f"got {len(w._bbox_items)}"
        )


# ============================================================
# styles.py — dead code branch removed
# ============================================================
def test_styles_no_dead_if_false_branch():
    """Phase 41: styles.py had `PROGRESS_BAR_MIN_HEIGHT_PX if False else 18`
    which is dead code (always picks 18). Fixed to use the
    constant directly."""
    from rlpe.gui import styles
    import re
    src = open(styles.__file__).read()
    # Strip docstrings + line comments + the import-block context
    # so we only check actual executable code.
    code = re.sub(r'""".*?""""', "", src, flags=re.DOTALL)
    code = re.sub(r"#[^\n]*", "", code)
    assert "if False else" not in code, (
        "styles.py still has the dead 'if False else' branch in code "
        "(docstrings + line comments stripped)"
    )


def test_styles_uses_progress_bar_constant():
    """Phase 41: styles.py must reference the PROGRESS_BAR_MIN_HEIGHT_PX
    constant from constants.py (not hardcode 18)."""
    from rlpe.gui import styles
    from rlpe.gui.constants import PROGRESS_BAR_MIN_HEIGHT_PX
    assert hasattr(styles, "PROGRESS_BAR_MIN_HEIGHT_PX"), (
        "styles module should re-export PROGRESS_BAR_MIN_HEIGHT_PX"
    )
    assert styles.PROGRESS_BAR_MIN_HEIGHT_PX == PROGRESS_BAR_MIN_HEIGHT_PX


# ============================================================
# Round-trip: all GUI files use i18n for user-visible text
# ============================================================
def test_no_bare_english_in_run_tab_qmessagebox():
    """Phase 41: QMessageBox calls in run_tab must use i18n keys."""
    import inspect
    from rlpe.gui.run_tab import RunTab
    src = inspect.getsource(RunTab)
    # Check the warning calls
    assert 'QMessageBox.warning(self, "Missing PDF"' not in src
    assert 'QMessageBox.warning(self, "Missing output dir"' not in src
    # The i18n keys should be there
    assert 'i18n._tr("runtab.prompt.no_pdf.title")' in src
    assert 'i18n._tr("runtab.prompt.no_outdir.title")' in src