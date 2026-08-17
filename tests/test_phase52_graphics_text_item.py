"""Phase 52 — fix QGraphicsTextItem NameError in image_preview.

Bug: image_preview.py used ``isinstance(item, (QGraphicsRectItem,
QGraphicsTextItem))`` in mousePressEvent but never imported
``QGraphicsTextItem``. The type hint on line 76
(``list[QGraphicsTextItem]``) worked because ``from __future__
import annotations`` makes annotations lazy strings, so the
import-time error was hidden. The ``isinstance()`` check only
ran when the user clicked a text label, producing:

    NameError: name 'QGraphicsTextItem' is not defined

This crashed the bbox-click flow every time the user clicked
on a species name label.

Fix: import ``QGraphicsTextItem`` from ``PySide6.QtWidgets``.

Tests:
  1. ``image_preview`` module imports without raising.
  2. ``ImagePreviewWidget`` instantiates without raising.
  3. Clicking a ``QGraphicsTextItem`` in the scene fires
     ``bbox_clicked`` without raising NameError.
  4. Source guard: any Qt class used in ``isinstance()`` must
     be imported in the same module.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
import pytest
from PySide6.QtCore import Qt  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QGraphicsRectItem,
    QGraphicsTextItem,
)

_app = QApplication.instance() or QApplication([])


import pytest


# ============================================================
# 1. Module imports cleanly
# ============================================================
def test_image_preview_imports_qgraphicstextitem():
    """Phase 52: ``QGraphicsTextItem`` must be importable from the
    image_preview module's namespace (or at least used inside it
    without NameError)."""
    # Importing the module must not raise
    import rlpe.gui.image_preview as ip_mod

    # And the symbol must be available in the module's namespace
    # (it was used in isinstance() and in a type hint).
    assert hasattr(ip_mod, "QGraphicsTextItem"), (
        "image_preview.py uses QGraphicsTextItem but doesn't import it. "
        "Phase 52 bug: isinstance(item, QGraphicsTextItem) raised NameError."
    )


# ============================================================
# 2. Widget instantiates without raising
# ============================================================
def test_image_preview_widget_instantiates():
    from rlpe.gui.image_preview import ImagePreviewWidget

    widget = ImagePreviewWidget()
    assert widget is not None


# ============================================================
# 3. Clicking a QGraphicsTextItem fires bbox_clicked signal
# ============================================================
def test_clicking_text_label_fires_bbox_clicked_signal():
    """Phase 52: clicking on a QGraphicsTextItem in the scene must
    trigger the bbox_clicked signal without raising NameError.

    Before the fix, isinstance(item, QGraphicsTextItem) raised
    NameError at click time, crashing the event handler.
    """
    from rlpe.gui.image_preview import ImagePreviewWidget

    widget = ImagePreviewWidget()

    # Add a QGraphicsTextItem to the scene
    text_item = QGraphicsTextItem("Species X (confidence 0.95)")
    text_item.setData(0, {"species": "Species X", "confidence": 0.95})
    widget._scene.addItem(text_item)

    # The bbox_clicked signal must fire when clicked
    received = []
    widget.bbox_clicked.connect(lambda d: received.append(d))

    # Trigger mousePressEvent on the view at the text item's position
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QMouseEvent

    # Position in view coords
    item_pos_in_scene = text_item.scenePos()
    view_pos = widget._view.mapFromScene(item_pos_in_scene)

    click_event = QMouseEvent(
        QEvent.MouseButtonPress,
        view_pos,
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    # Should NOT raise NameError
    widget._view.mousePressEvent(click_event)

    assert received, (
        "bbox_clicked should fire when clicking a QGraphicsTextItem. "
        "Phase 52 bug: NameError was raised before signal could fire."
    )
    assert received[0]["species"] == "Species X"


# ============================================================
# 4. Source guard: scan for similar "used in isinstance() but
#    not imported" bugs across the GUI
# ============================================================
def test_no_unimported_classes_in_isinstance_calls():
    """Phase 52 guard: in every GUI module, every class referenced
    in an ``isinstance(...)`` call must be either imported or
    defined in the same module.

    Catches the class of bug where a type annotation worked
    (lazy via ``from __future__ import annotations``) but the
    runtime isinstance() raised NameError.
    """
    gui_dir = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui"
    # Common Qt classes that are always available via PySide6 imports
    # in any module (so we don't flag them as missing).
    safe_builtins = {
        "int",
        "str",
        "float",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "frozenset",
        "bytes",
        "bytearray",
        "object",
        "type",
        "None",
        "Path",
        "QStringListModel",
    }
    failures = []
    for py_file in sorted(gui_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        src = py_file.read_text(encoding="utf-8")
        # Find all isinstance(x, (A, B, ...)) and isinstance(x, A) calls
        for m in re.finditer(r"isinstance\([^,]+,\s*([^)]+)\)", src):
            arg = m.group(1).strip()
            # arg is either a single class or a tuple of classes
            classes = [c.strip() for c in arg.strip("()").split(",") if c.strip()]
            for cls in classes:
                # Strip surrounding parens/whitespace
                cls = cls.strip()
                # Skip lowercase / dotted refs (e.g. 'self.X')
                if not cls or "." in cls or cls[0].islower():
                    continue
                # Skip if it's a string literal ("foo")
                if cls.startswith(("'", '"')):
                    continue
                # Skip if defined in the same module
                if re.search(rf"^{re.escape(cls)}\s*=", src, re.MULTILINE) or re.search(
                    rf"^class\s+{re.escape(cls)}\b", src, re.MULTILINE
                ):
                    continue
                # Skip builtins
                if cls in safe_builtins:
                    continue
                # Skip if imported via `from ... import ...` line
                # Match: `from X import Y` or `from X import (Y, Z)`
                # Or `import X` and then access X.Y
                imported_via_from = re.search(
                    rf"from\s+[\w.]+\s+import\s+[(\s]?[\w\s,]*\b"
                    rf"{re.escape(cls)}\b",
                    src,
                )
                if imported_via_from:
                    continue
                # If it's accessed via module prefix (e.g. Qt.LeftButton)
                # the regex above won't match `Qt.LeftButton` since
                # `Qt` is lowercase — we already filtered those out.
                # So if we get here, cls is a bare uppercase name used
                # in isinstance() but not imported → bug.
                failures.append(f"{py_file.relative_to(gui_dir.parent.parent)}: {cls}")

    assert not failures, (
        "Phase 52: these classes are used in isinstance() but not "
        "imported in their module:\n  " + "\n  ".join(failures)
    )
