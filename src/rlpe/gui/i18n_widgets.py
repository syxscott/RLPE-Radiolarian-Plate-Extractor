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

Note: the caller is always responsible for adding the returned
widget to a layout (``layout.addWidget(w)``). The factories only
construct + configure the widget — they don't attach it to a layout
because that would couple widget construction to layout ownership,
which complicates re-parenting and tab switching.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from . import i18n


# ============================================================
# Phase 36: row-height helpers (QFormLayout / QGridLayout)
# ============================================================
# Note: we do NOT monkey-patch ``sizeHint()`` on Qt widgets because
# overriding a C++ virtual slot on a QObject subclass segfaults under
# PySide6 6.11 when Qt's layout code calls the wrapper. Instead we
# set a robust size policy + bump the QSS to ensure each widget
# reports a height >= 30 px in its layout row.
def _ensure_size_hint(widget: QWidget, patched_height: int) -> None:
    """Phase 36: force a widget to report a row height of at least
    ``patched_height`` in a layout.

    Strategy: set ``QSizePolicy.Fixed`` on the vertical axis and
    bump the widget's minimum height (preserving any explicit
    minimum width the caller already set). Layouts will then use
    ``patched_height`` for the row (Qt treats ``Fixed`` height as
    the canonical row height) without overriding ``sizeHint``.

    Idempotent — multiple calls don't accumulate.
    """
    if getattr(widget, "_phase36_ensured", False):
        return
    try:
        from PySide6.QtWidgets import QSizePolicy
        sp = widget.sizePolicy()
        sp.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        sp.setHorizontalPolicy(QSizePolicy.Policy.Preferred)
        widget.setSizePolicy(sp)
        # Preserve any caller-set minimumWidth, only override the height.
        existing_min_w = widget.minimumWidth()
        existing_min_h = widget.minimumHeight()
        new_min_h = max(existing_min_h, patched_height)
        widget.setMinimumSize(max(existing_min_w, 0), new_min_h)
        widget._phase36_ensured = True
    except Exception:
        pass


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
    except Exception as exc:
        import logging
        logging.getLogger("rlpe.gui").warning(
            "i18n: failed to set %s on %s (%s): %s", attr, key, type(widget).__name__, exc
        )
    # The objectName doubles as the registry key. i18n.register_widget_text
    # keys by the object's objectName, NOT by the (widget, attr) pair,
    # because the registry's purpose is to walk allWidgets() in the
    # QApplication and retext.
    widget.setObjectName(key)
    i18n.register_widget_text(key, attr, key)


# ============================================================
# Widget factories
# ============================================================

# Phase 39: registry of QAction objects whose text is translated
# by i18n. QAction is NOT a QWidget so the global allWidgets() loop
# in i18n._apply_registry misses it — we keep an explicit list and
# update each action's text on language change. See i18n.add_listener
# wiring in main_window._refresh_texts().
_MENU_ACTIONS: list[tuple[Any, str]] = []  # list of (QAction, key)


def tr_action(
    text_key: str,
    *,
    parent: Optional[QWidget] = None,
) -> "QAction":
    """Create a QAction translated by ``text_key``.

    QAction is not a QWidget, so it doesn't show up in
    QApplication.allWidgets() and the i18n registry's auto-rewalk
    won't find it. We register it explicitly via _MENU_ACTIONS so
    the i18n.add_listener(sweep) can update its text on language
    switch.
    """
    from PySide6.QtGui import QAction
    act = QAction(i18n._tr(text_key), parent)
    act.setObjectName(text_key)
    _MENU_ACTIONS.append((act, text_key))
    return act


def tr_menu(
    title_key: str,
    parent: Optional[QWidget] = None,
) -> "QMenu":
    """Create a QMenu translated by ``title_key``.

    Like tr_action, the menu title is registered so it translates
    on language switch.
    """
    from PySide6.QtWidgets import QMenu
    menu = QMenu(i18n._tr(title_key), parent)
    menu.setObjectName(title_key)
    _MENU_ACTIONS.append((menu, title_key))
    return menu


def refresh_all_menu_actions(lang: str) -> None:
    """Phase 39: called by the i18n listener when language changes.

    Walks _MENU_ACTIONS and re-texts each one. QMenu uses
    ``setTitle()`` (not setText), so we dispatch on type. Idempotent
    — items already translated are skipped via the registry.
    """
    from PySide6.QtWidgets import QMenu
    for item, key in _MENU_ACTIONS:
        try:
            text = i18n._tr(key)
            if isinstance(item, QMenu):
                item.setTitle(text)
            else:
                item.setText(text)
        except Exception:
            pass


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
    min_height: int = 32,
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
    min_height: int = 32,
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
    min_height: int = 32,
) -> QLineEdit:
    """Create a QLineEdit with a translated placeholder.

    Note: only the *placeholder* is translated. The user input is
    not (and should not be) re-translated when the language changes
    — that would clobber what the user is typing.

    Phase 36: default ``min_height=32`` (was 30) so a QSpinBox on
    the same row has room for its up/down arrows (16 px each) and
    still looks balanced with adjacent widgets.
    """
    le = QLineEdit(text, parent)
    key = object_name or placeholder_key
    le.setMinimumWidth(min_width)
    le.setMinimumHeight(min_height)
    _ensure_size_hint(le, min_height)
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
    min_height: int = 32,
) -> QSpinBox:
    """Create a QSpinBox. The label above / beside the spinbox is
    a separate ``tr_label`` call — this factory only creates the
    numeric input. We still register a translation key for the
    spinbox's prefix / suffix via the surrounding label.

    Phase 36: default ``min_height=32`` to fit the up/down arrows
    inside the widget rect (each arrow is 16 px, container must
    be 32+ px or arrows escape and overlap adjacent widgets)."""
    sb = QSpinBox(parent)
    sb.setMinimumWidth(min_width)
    sb.setMinimumHeight(min_height)
    _ensure_size_hint(sb, min_height)
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
    min_height: int = 32,
) -> QDoubleSpinBox:
    sb = QDoubleSpinBox(parent)
    sb.setMinimumWidth(min_width)
    sb.setMinimumHeight(min_height)
    _ensure_size_hint(sb, min_height)
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
    item_keys: Optional[list[str]] = None,
    min_height: int = 32,
) -> QComboBox:
    """Create a QComboBox with a translated list of items.

    The ``text_key`` is the registry key for the *placeholder text*.
    The items themselves are inserted in one of two modes:

      * ``items=[str, ...]``  — literal strings, NOT translated.
        Use this for short technical tokens ("en", "ja", "ch_sim")
        that are the same in every language.
      * ``item_keys=[key, ...]`` — each entry is an i18n key whose
        translation is the displayed item. The raw ``items`` are
        stored as ``userData`` so callers can ``currentData()`` to
        get the original token even after language switches.

    Phase 37 audit fix: previously items were literal and never
    translated; now callers can pass either literal items or
    translation keys (or both — ``items`` becomes the fallback
    if ``item_keys`` is not provided).
    """
    cb = QComboBox(parent)
    key = object_name or text_key
    cb.setMinimumWidth(min_width)
    cb.setMinimumHeight(min_height)
    _ensure_size_hint(cb, min_height)
    # Phase 45: block signals during construction so the
    # currentIndexChanged slot doesn't fire on a half-built widget.
    # The slot may try to access other widgets (e.g. _refresh_view
    # in results_tab) that haven't been parented yet, causing
    # AttributeError or worse (segfault on deleted C++ objects).
    cb.blockSignals(True)
    try:
        if item_keys:
            # Translate each item via i18n._tr; store raw token as userData.
            for ix, k in enumerate(item_keys):
                label = i18n._tr(k)
                raw = items[ix] if items and ix < len(items) else label
                cb.addItem(label, userData=raw)
        elif items:
            cb.addItems(items)
        if current:
            # Look up by userData (preferred) or by text (fallback)
            ix = cb.findData(current)
            if ix < 0:
                ix = cb.findText(current)
            if ix >= 0:
                cb.setCurrentIndex(ix)
        cb.setObjectName(key)
        # Phase 37: also register each item_key so the registry re-applies
        # them on language switch.
        if item_keys:
            for ix, k in enumerate(item_keys):
                i18n.register_widget_text(f"{key}:item:{ix}", "comboItem", k)
    finally:
        cb.blockSignals(False)
    if text_key:
        i18n.register_widget_text(key, "placeholderText", text_key)
        cb.setPlaceholderText(i18n._tr(text_key))
    return cb


def populate_friendly_combo(
    combo: QComboBox,
    factory: "Callable[[], list[tuple[str, str]]]",
    default_code: Optional[str] = None,
) -> Callable[[str], None]:
    """Populate ``combo`` from a ``*_friendly_options()`` factory and
    automatically rebuild the displayed items when the language changes.

    The factory must return a list of ``(code, friendly_name)`` tuples
    where ``friendly_name`` is rendered using the language active when
    ``factory()`` is called. Each item is added with ``userData=code``
    so callers can read the raw code via ``currentData()``.

    The current selection is preserved (looked up by userData) across
    rebuilds. A subsequent call on the same combo removes the previous
    listener to avoid accumulating stale closures.
    """
    from . import i18n as _i18n

    def _rebuild(_lang: str) -> None:
        # Guard: skip if combo was deleted by Qt.
        try:
            combo.count()
        except RuntimeError:
            return
        prev_code = combo.currentData() or default_code
        combo.blockSignals(True)
        try:
            combo.clear()
            for code, friendly in factory():
                combo.addItem(friendly, userData=code)
            if prev_code is not None:
                ix = combo.findData(prev_code)
                combo.setCurrentIndex(ix if ix >= 0 else 0)
        finally:
            combo.blockSignals(False)

    old_listener = getattr(combo, "_i18n_listener", None)
    if old_listener is not None:
        try:
            _i18n.remove_listener(old_listener)
        except Exception as exc:
            import logging
            logging.getLogger("rlpe.gui").warning(
                "i18n: failed to remove listener from combo: %s", exc
            )

    _rebuild(_i18n.current_language())
    combo._i18n_listener = _rebuild  # type: ignore[attr-defined]
    _i18n.add_listener(_rebuild)
    return _rebuild


# ============================================================
# Phase 34: form-row helper + input-height normalisation
# ============================================================
def tr_form_row(
    label_key: str,
    widget: QWidget,
    *,
    label_align: Qt.AlignmentFlag = Qt.AlignRight | Qt.AlignVCenter,
    min_height: int = 32,
) -> tuple[QLabel, QWidget]:
    """Create a (label, widget) pair ready to be passed to
    ``QFormLayout.addRow``. The label is translated via
    ``label_key``; the widget's height is forced to ``min_height``
    so a QSpinBox / QComboBox value isn't visually clipped at
    high DPI or under the QSS dark theme.

    Phase 36: also override the widget's ``sizeHint()`` so the
    QFormLayout row uses ``min_height`` rather than the widget's
    intrinsic 22-26 px height."""
    lbl = tr_label(label_key)
    lbl.setAlignment(label_align)
    lbl.setMinimumHeight(min_height)
    widget.setMinimumHeight(min_height)
    _ensure_size_hint(widget, min_height)
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


# ============================================================
# Phase 40: wheel-scroll protection on input widgets
# ============================================================
# Qt's default behaviour: when a QSpinBox / QDoubleSpinBox /
# QComboBox / QAbstractSpinBox / QLineEdit is under the mouse
# cursor, the scroll wheel increments/decrements the value or
# changes the combo selection. This is annoying for the user —
# scrolling over a form to read a tooltip accidentally mutates
# settings. Phase 40 disables wheel events unless the widget has
# keyboard focus.
#
# Implementation: install a global event filter on QApplication
# that filters QEvent::Wheel on QAbstractSpinBox + QComboBox +
# QLineEdit, and consumes the event unless the widget has focus.

def install_wheel_filter(app: QApplication) -> None:
    """Install a global wheel-event filter on ``app``.

    Phase 40: scroll wheels no longer mutate QSpinBox /
    QDoubleSpinBox / QComboBox values unless the widget has
    keyboard focus. Prevents the "I scrolled past a setting and
    it silently changed" footgun.

    Idempotent — calling twice is a no-op.
    """
    from PySide6.QtCore import QObject, QEvent
    if getattr(app, "_phase40_wheel_filter", False):
        return

    class _WheelFilter(QObject):
        def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
            if event.type() != QEvent.Type.Wheel:
                return False
            # Only filter numeric / combo / line-edit widgets
            from PySide6.QtWidgets import (
                QAbstractSpinBox, QComboBox, QLineEdit,
            )
            if not isinstance(obj, (QAbstractSpinBox, QComboBox, QLineEdit)):
                return False
            # If the widget has focus, let the user use the wheel
            # to adjust the value (intentional — they clicked it).
            if obj.hasFocus():
                return False
            # Otherwise, eat the wheel event so the value doesn't
            # change. The wheel scroll still works for parent
            # widgets (e.g. scrolling a QScrollArea).
            event.accept()
            return True

    flt = _WheelFilter(app)
    app.installEventFilter(flt)
    app._phase40_wheel_filter = True
    app._phase40_wheel_filter_obj = flt

