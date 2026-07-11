"""Widget-level i18n helpers.

These are small wrappers around Qt widgets that:
  1. Set a deterministic ``objectName`` so the i18n registry can find
     the widget.
  2. Apply the current language immediately.
  3. Wire the widget's text attribute to a translation key so it
     refreshes on every ``set_language()`` call.

Why a wrapper instead of a mixin or QObject subclass?
  - The GUI is built top-down (the MainWindow constructor creates
    each tab). Refactoring every QLineEdit / QLabel to inherit from
    a custom class is invasive.
  - A thin wrapper keeps the call sites readable:
        tr_label(self, "PBDB Family:", key="restab.col.family")
    is clearer than
        lbl = QLabel(); lbl.setObjectName("..."); register_widget_text(...); lbl.setText(_tr(...))

The wrappers are functions, not classes, so we don't need to
register Qt meta-types.

Each wrapper takes ``parent_layout`` (optional) — when supplied
the widget is added to the layout automatically. When None, the
caller is responsible for adding the widget (e.g. for nested
splitter / scrollarea placements).
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from . import i18n


# ============================================================
# Generic translatable text setter
# ============================================================
def _set_text(widget: QWidget, attr: str, key: str) -> None:
    """Set a translation key on the widget + apply the current text."""
    text = i18n._tr(key)
    # Many Qt attrs (text, title, placeholderText) accept strings.
    # setattr works for both QString and str setters in PySide6.
    try:
        if attr == "text":
            widget.setText(text)
        elif attr == "title":
            widget.setTitle(text)
        elif attr == "placeholderText":
            widget.setPlaceholderText(text)
        elif attr == "toolTip":
            widget.setToolTip(text)
        elif attr == "windowTitle":
            widget.setWindowTitle(text)
        else:
            setattr(widget, attr, text)
    except Exception:
        pass
    # The objectName doubles as the registry key. i18n.register_widget_text
    # keys by the object's objectName, NOT by the (widget, attr) pair,
    # because the registry's purpose is to walk allWidgets() in the
    # QApplication and retext.
    widget.setObjectName(key)
    i18n.register_widget_text(key, attr, key)


# ============================================================
# Widget factories
# ============================================================
def tr_label(
    text_key: str,
    *,
    object_name: Optional[str] = None,
    parent: Optional[QWidget] = None,
    align: Qt.AlignmentFlag | None = None,
) -> QLabel:
    """Create a QLabel translated by ``text_key``.

    ``object_name`` defaults to ``text_key``. The widget's
    ``objectName`` is set to the key so the i18n registry can find
    it on language change.
    """
    lbl = QLabel(parent)
    key = object_name or text_key
    if align is not None:
        lbl.setAlignment(align)
    _set_text(lbl, "text", key)
    return lbl


def tr_button(
    text_key: str,
    *,
    object_name: Optional[str] = None,
    parent: Optional[QWidget] = None,
    object_name_attr: str = "default",
    min_height: int = 30,
) -> QPushButton:
    btn = QPushButton(parent)
    key = object_name or text_key
    btn.setObjectName(key)
    btn.setProperty("object_name_attr", object_name_attr)
    btn.setMinimumHeight(min_height)
    i18n.register_widget_text(key, "text", key)
    i18n._tr(key)  # force registration
    # We can't call _set_text because it would re-set the objectName.
    btn.setText(i18n._tr(key))
    return btn


def tr_checkbox(
    text_key: str,
    *,
    object_name: Optional[str] = None,
    parent: Optional[QWidget] = None,
    checked: bool = False,
    min_height: int = 30,
) -> QCheckBox:
    """Create a QCheckBox with translated text.

    Phase 35: default ``min_height=30`` so long checkbox labels
    like "Enable PBDB enrichment (taxonomy + occurrences)" wrap
    cleanly rather than clipping at the QSS 22-px checkbox height."""
    cb = QCheckBox(parent)
    key = object_name or text_key
    cb.setChecked(checked)
    cb.setMinimumHeight(min_height)
    _set_text(cb, "text", key)
    return cb


def tr_groupbox(
    text_key: str,
    *,
    object_name: Optional[str] = None,
    parent: Optional[QWidget] = None,
) -> QGroupBox:
    gb = QGroupBox(parent)
    key = object_name or text_key
    _set_text(gb, "title", key)
    return gb


def tr_lineedit(
    placeholder_key: str,
    *,
    object_name: Optional[str] = None,
    parent: Optional[QWidget] = None,
    min_width: int = 180,
    text: str = "",
    min_height: int = 30,
) -> QLineEdit:
    """Create a QLineEdit with a translated placeholder.

    Note: only the *placeholder* is translated. The user input is
    not (and should not be) re-translated when the language changes
    — that would clobber what the user is typing.

    Phase 35: default ``min_height=30`` so the row in any layout
    matches the 30-px QSpinBox / QComboBox row height and the
    value text isn't clipped.
    """
    le = QLineEdit(text, parent)
    key = object_name or placeholder_key
    le.setMinimumWidth(min_width)
    le.setMinimumHeight(min_height)
    le.setObjectName(key)
    i18n.register_widget_text(key, "placeholderText", key)
    le.setPlaceholderText(i18n._tr(key))
    return le


def tr_spinbox(
    text_key: str,
    *,
    object_name: Optional[str] = None,
    parent: Optional[QWidget] = None,
    min_width: int = 100,
    min_val: int = 0,
    max_val: int = 1000,
    value: int = 0,
    min_height: int = 30,
) -> QSpinBox:
    """Create a QSpinBox. The label above / beside the spinbox is
    a separate ``tr_label`` call — this factory only creates the
    numeric input. We still register a translation key for the
    spinbox's prefix / suffix via the surrounding label.

    Phase 35: default ``min_height=30`` for visual parity with the
    row heights in QFormLayout."""
    sb = QSpinBox(parent)
    sb.setMinimumWidth(min_width)
    sb.setMinimumHeight(min_height)
    sb.setRange(min_val, max_val)
    sb.setValue(value)
    if object_name:
        sb.setObjectName(object_name)
    return sb


def tr_doublespinbox(
    *,
    object_name: str = "",
    parent: Optional[QWidget] = None,
    min_width: int = 100,
    min_val: float = 0.0,
    max_val: float = 1.0,
    value: float = 0.5,
    step: float = 0.05,
    min_height: int = 30,
) -> QDoubleSpinBox:
    sb = QDoubleSpinBox(parent)
    sb.setMinimumWidth(min_width)
    sb.setMinimumHeight(min_height)
    sb.setRange(min_val, max_val)
    sb.setSingleStep(step)
    sb.setValue(value)
    if object_name:
        sb.setObjectName(object_name)
    return sb


def tr_combobox(
    text_key: str,
    *,
    object_name: Optional[str] = None,
    parent: Optional[QWidget] = None,
    min_width: int = 130,
    items: Optional[list[str]] = None,
    current: Optional[str] = None,
    min_height: int = 30,
) -> QComboBox:
    """Create a QComboBox with a translated list of items.

    The ``text_key`` is the registry key. The *items* themselves
    are pulled from a separate per-key item list (not all keys
    have items — only the ones that need translation). If
    ``items`` is provided, those literal strings are inserted.

    Phase 35: default ``min_height=30`` for visual parity."""
    cb = QComboBox(parent)
    key = object_name or text_key
    cb.setMinimumWidth(min_width)
    cb.setMinimumHeight(min_height)
    if items:
        cb.addItems(items)
    if current:
        idx = cb.findText(current)
        if idx >= 0:
            cb.setCurrentIndex(idx)
    cb.setObjectName(key)
    # Placeholder text + items both registered
    if text_key:
        i18n.register_widget_text(key, "placeholderText", text_key)
        cb.setPlaceholderText(i18n._tr(text_key))
    return cb


# ============================================================
# Phase 34: form-row helper + input-height normalisation
# ============================================================
def tr_form_row(
    label_key: str,
    widget: QWidget,
    *,
    label_align: Qt.AlignmentFlag = Qt.AlignRight | Qt.AlignVCenter,
    min_height: int = 30,
) -> tuple[QLabel, QWidget]:
    """Create a (label, widget) pair ready to be passed to
    ``QFormLayout.addRow``. The label is translated via
    ``label_key``; the widget's height is forced to ``min_height``
    so a QSpinBox / QComboBox value isn't visually clipped at
    high DPI or under the QSS dark theme.
    """
    lbl = tr_label(label_key)
    lbl.setAlignment(label_align)
    lbl.setMinimumHeight(min_height)
    widget.setMinimumHeight(min_height)
    return lbl, widget


def normalise_input_height(widget: QWidget, min_height: int = 30) -> None:
    """Force ``min_height`` on an existing input widget.

    Phase 34 fix: QFormLayout with a QSpinBox/QComboBox used to
    collapse the row to ~22 px (the widget's natural height),
    which clipped the value text in the QSS dark theme and at
    150% DPI. Setting a 30-px min height keeps the value visible
    and the row visually balanced with the QGroupBox title.
    """
    widget.setMinimumHeight(min_height)
