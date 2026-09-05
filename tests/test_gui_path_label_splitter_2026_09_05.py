"""Audit 2026-09-05 — results-tab splitter collapse regression.

User report: in the Results tab, clicking the horizontal splitter
handle between the image preview (left) and the detail pane (right)
made the right-hand data disappear.

Root cause: the image preview's toolbar path label received the FULL
image path via ``setText``. With word wrap off,
``QLabel.minimumSizeHint()`` equals the full text width, so a long
output path silently raised ``ImagePreviewWidget.minimumSizeHint()``
to the path's pixel width (reproduced at 945 px vs a 413 px baseline).
The results tab's horizontal QSplitter then held an invalid layout,
and the FIRST user interaction with the divider handle — even a bare
click — snapped the divider to the far left, pinning the detail pane
at its 500 px minimum or pushing it out of the window entirely on
narrower displays (945 + 500 > splitter width at 1280 px).

Fix contract pinned here:
  1. ``ImagePreviewWidget.minimumSizeHint().width()`` must NOT grow
     with the displayed path length (explicit ``setMinimumWidth(1)``
     on the label decouples the layout constraint from the text).
  2. The full path stays reachable via the label's tooltip; the
     displayed text is elided.
  3. End-to-end: with a real image loaded through ``ResultsTab``, a
     bare click on the splitter handle must NOT move the divider.

NOTE: the PySide6 / ``rlpe.gui`` imports are deliberately INSIDE the
test functions so this module is never classified as a Qt-runtime
module by conftest's SIGSEGV guard (which would skip it on the
PySide6>=6.11 + Python 3.11 combo). These tests only construct
widgets offscreen; they never drive ``QEventLoop.exec()``.
"""

from __future__ import annotations

from pathlib import Path


def _long_png(tmp_path: Path) -> Path:
    """Create a real 4x4 PNG whose absolute path is very long (>300 chars).

    The bug only manifests when the path label's text is wider than
    the preview widget's baseline minimum (413 px), so the fixture
    path must be long enough to exceed that.
    """
    from PIL import Image

    deep = tmp_path
    for i in range(12):
        deep = deep / f"level_{i:02d}_{'x' * 24}"
    deep.mkdir(parents=True, exist_ok=True)
    p = deep / ("panel_" + "very_long_output_directory_segment_" * 4 + "01.png")
    Image.new("RGB", (4, 4), color="white").save(p)
    return p


def _qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_preview_min_width_does_not_grow_with_path_length(tmp_path):
    """Contract 1: loading an image must not raise the widget minimum."""
    p = _long_png(tmp_path)
    assert len(str(p)) > 300, "fixture path must be long enough to expose the bug"

    from PySide6.QtTest import QTest  # noqa: E402 — local: keep module off the

    from rlpe.gui.image_preview import ImagePreviewWidget
    # conftest SIGSEGV skip-list (only module-level PySide6 imports count).

    app = _qapp()
    preview = ImagePreviewWidget()
    preview.resize(600, 400)
    # show() is REQUIRED to reproduce: an un-shown widget's layout is
    # never activated, so the inflated QLabel minimumSizeHint does not
    # propagate to minimumSizeHint() (measured: 413 vs 3801 px).
    preview.show()
    QTest.qWaitForWindowExposed(preview)
    app.processEvents()
    baseline = preview.minimumSizeHint().width()
    preview.set_image(p)
    app.processEvents()
    after = preview.minimumSizeHint().width()
    assert after == baseline, (
        f"loading an image raised the preview minimum width "
        f"{baseline} -> {after} px — the path label's minimumSizeHint "
        f"is leaking into the widget constraint and collapses the "
        f"results-tab splitter on first user interaction"
    )
    preview.close()


def test_path_label_elides_and_keeps_tooltip(tmp_path):
    """Contract 2: full path reachable via tooltip; display text elided."""
    p = _long_png(tmp_path)
    from PySide6.QtTest import QTest

    from rlpe.gui.image_preview import ImagePreviewWidget

    app = _qapp()
    preview = ImagePreviewWidget()
    preview.resize(600, 400)
    preview.show()
    QTest.qWaitForWindowExposed(preview)
    app.processEvents()
    preview.set_image(p)
    app.processEvents()
    label = preview._path_label
    assert label.toolTip() == str(p), "full path must stay reachable via tooltip"
    assert len(label.text()) < len(str(p)), "display text should be elided"
    # A short (no-image) text must pass through unelided.
    preview.set_image(None)
    assert label.toolTip() == ""
    assert label.text()  # the "no image" placeholder is shown
    preview.close()


def test_results_tab_splitter_click_does_not_move_divider(tmp_path):
    """Contract 3 (end-to-end): a bare click on the splitter handle must
    not move the divider — pre-fix it snapped [562, 850] -> [912, 500]."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QSplitter

    from rlpe.gui.results_tab import ResultsTab

    p = _long_png(tmp_path)
    app = _qapp()
    row = {
        "paper_id": "p1",
        "figure_id": "fig1",
        "panel_id": "1",
        "species": "Testus species",
        "confidence": 0.9,
        "panel_path": str(p),
        "bbox": [1, 2, 2, 2],
        "caption_snippet": "Plate 1, fig 1",
        "ocr_text": "1",
        "metadata": {},
        "paper_metadata": None,
    }
    tab = ResultsTab()
    tab.resize(1440, 900)
    tab.show()
    QTest.qWaitForWindowExposed(tab)
    app.processEvents()
    # Baseline BEFORE the image loads: at this point the splitter
    # layout is still valid ([~562, ~850] at 1440 px). Pre-fix, merely
    # selecting a row (loading the long path into the label) already
    # invalidated the layout, and the divider snapped to the far left
    # on the next layout pass / first interaction.
    hs = next(s for s in tab.findChildren(QSplitter) if s.orientation() == Qt.Horizontal)
    right = hs.widget(1)
    before = list(hs.sizes())
    tab.load_job("job1", [row], str(tmp_path))
    app.processEvents()
    tab._table.selectRow(0)
    app.processEvents()
    handle = hs.handle(1)
    QTest.mouseClick(handle, Qt.LeftButton)
    app.processEvents()
    after = list(hs.sizes())
    assert after == before, (
        f"loading a row with a long panel path moved the splitter "
        f"{before} -> {after}; the right (detail) pane collapses — "
        f"the path label minimum-width fix has regressed"
    )
    assert right.minimumWidth() <= right.width(), (
        "detail pane squeezed below its minimum — layout went invalid"
    )
    tab.close()
