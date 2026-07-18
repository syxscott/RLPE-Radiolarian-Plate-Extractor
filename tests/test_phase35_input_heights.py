"""Phase 35 — input-height enforcement regression tests.

Phase 34 added ``_normalise_input_heights`` on the Settings tab and
bumped ``tr_form_row`` to set ``min_height=30`` on the widget. But:

  * The Settings tab's bare ``QCheckBox(...)`` calls (long English
    labels like "Enable PBDB enrichment (taxonomy + occurrences)")
    were not being normalised because ``_normalise_input_heights``
    only checked SpinBox / DSpinBox / ComboBox / LineEdit / PushButton.
  * The Run tab used ``tr_spinbox`` / ``tr_lineedit`` / ``tr_combobox``
    factories without any min-height enforcement, so the basic-config
    area's QGridLayout rows clipped values at 150% DPI.
  * The QSS forced ``min-height: 22px`` on QPushButton, which on some
    platforms overrode the Python ``setMinimumHeight(30)``.

Phase 35 fixes all three:

  1. ``tr_spinbox``, ``tr_doublespinbox``, ``tr_combobox``,
     ``tr_lineedit``, ``tr_button``, ``tr_checkbox`` all default
     to ``min_height=30``.
  2. Settings-tab bare ``QCheckBox(...)`` calls replaced with
     ``tr_checkbox(key)``.
  3. ``_normalise_input_heights`` now includes QCheckBox.
  4. QSS ``min-height: 22px`` on QPushButton bumped to ``30px``.
  5. QCheckBox QSS gets ``min-height: 22px`` + ``padding: 2px 0``
     so long checkbox labels wrap cleanly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


import pytest


@pytest.fixture(autouse=True)
def _reset_language():
    """Reset to EN before each test so test ordering can't flip the
    language out from under us."""
    from rlpe.gui import i18n
    i18n.set_language("en")
    yield
    i18n.set_language("en")


# ============================================================
# Helpers
# ============================================================
def _user_visible_lineedits(tab) -> list:
    """QLineEdits directly created by the tab (not the internal
    ``qt_spinbox_lineedit`` children inside QSpinBox / QComboBox)."""
    from PySide6.QtWidgets import QLineEdit
    return [
        le for le in tab.findChildren(QLineEdit)
        if le.objectName() != "qt_spinbox_lineedit"
    ]


# ============================================================
# Settings tab
# ============================================================
def test_settings_tab_all_checkboxes_have_min_height_30():
    from PySide6.QtWidgets import QCheckBox
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    for cb in st.findChildren(QCheckBox):
        assert cb.minimumHeight() >= 32, (
            f"Settings QCheckBox {cb.text()!r} has minHeight="
            f"{cb.minimumHeight()}, needs >= 32"
        )


def test_settings_tab_checkboxes_translate():
    """The two bare QCheckBox(...) calls in Phase 33/34 are now
    ``tr_checkbox(key)`` and must translate on language switch."""
    from PySide6.QtWidgets import QCheckBox
    from rlpe.gui import i18n
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    boxes = [c for c in st.findChildren(QCheckBox) if c.text()]
    en = [c.text() for c in boxes]
    i18n.set_language("zh_CN")
    zh = [c.text() for c in boxes]
    unchanged = [(t, z) for t, z in zip(en, zh) if t == z]
    assert len(unchanged) == 0, (
        f"Settings tab checkboxes did NOT translate: {unchanged}"
    )


def test_settings_tab_input_widgets_min_height_30():
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    for le in _user_visible_lineedits(st):
        assert le.minimumHeight() >= 32, (
            f"Settings QLineEdit minHeight={le.minimumHeight()}, needs >= 32"
        )


# ============================================================
# Run tab
# ============================================================
def test_run_tab_all_checkboxes_have_min_height_30():
    from PySide6.QtWidgets import QCheckBox
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    for cb in rt.findChildren(QCheckBox):
        assert cb.minimumHeight() >= 32, (
            f"Run QCheckBox {cb.text()!r} has minHeight="
            f"{cb.minimumHeight()}, needs >= 32"
        )


def test_run_tab_spinboxes_have_min_height_30():
    from PySide6.QtWidgets import QSpinBox
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    for sb in rt.findChildren(QSpinBox):
        assert sb.minimumHeight() >= 32, (
            f"Run QSpinBox minHeight={sb.minimumHeight()}, needs >= 32"
        )


def test_run_tab_comboboxes_have_min_height_30():
    from PySide6.QtWidgets import QComboBox
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    for cb in rt.findChildren(QComboBox):
        assert cb.minimumHeight() >= 32, (
            f"Run QComboBox minHeight={cb.minimumHeight()}, needs >= 32"
        )


def test_run_tab_lineedits_have_min_height_30():
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    for le in _user_visible_lineedits(rt):
        assert le.minimumHeight() >= 32, (
            f"Run QLineEdit minHeight={le.minimumHeight()}, needs >= 32"
        )


def test_run_tab_buttons_have_min_height_30():
    from PySide6.QtWidgets import QPushButton
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    for btn in rt.findChildren(QPushButton):
        assert btn.minimumHeight() >= 32, (
            f"Run QPushButton {btn.text()!r} minHeight="
            f"{btn.minimumHeight()}, needs >= 32"
        )


# ============================================================
# Widget factories default to min_height=30
# ============================================================
def test_tr_checkbox_factory_sets_min_height_30():
    from rlpe.gui.i18n_widgets import tr_checkbox
    from PySide6.QtWidgets import QCheckBox
    cb = tr_checkbox("dummy.checkbox.key")
    assert isinstance(cb, QCheckBox)
    assert cb.minimumHeight() == 32, (
        f"tr_checkbox default minHeight={cb.minimumHeight()}, expected 32"
    )


def test_tr_spinbox_factory_sets_min_height_30():
    from rlpe.gui.i18n_widgets import tr_spinbox
    from PySide6.QtWidgets import QSpinBox
    sb = tr_spinbox("dummy.spinbox.key")
    assert isinstance(sb, QSpinBox)
    assert sb.minimumHeight() == 32


def test_tr_combobox_factory_sets_min_height_30():
    from rlpe.gui.i18n_widgets import tr_combobox
    from PySide6.QtWidgets import QComboBox
    cb = tr_combobox("dummy.combobox.key")
    assert isinstance(cb, QComboBox)
    assert cb.minimumHeight() == 32


def test_tr_lineedit_factory_sets_min_height_30():
    from rlpe.gui.i18n_widgets import tr_lineedit
    from PySide6.QtWidgets import QLineEdit
    le = tr_lineedit("dummy.lineedit.key")
    assert isinstance(le, QLineEdit)
    assert le.minimumHeight() == 32


def test_tr_button_factory_sets_min_height_30():
    from rlpe.gui.i18n_widgets import tr_button
    from PySide6.QtWidgets import QPushButton
    btn = tr_button("dummy.button.key")
    assert isinstance(btn, QPushButton)
    assert btn.minimumHeight() == 32


# ============================================================
# QSS-level: QPushButton / QCheckBox have non-trivial min-height
# ============================================================
def test_qss_pushbutton_min_height_at_least_28():
    """The QSS must not force QPushButton below 28px. Phase 35
    bumped it from 22 → 30 to match QSpinBox row heights."""
    from rlpe.gui.styles import LIGHT_QSS, DARK_QSS
    import re
    # Pull the QPushButton { ... } block from the LIGHT theme
    m = re.search(r"QPushButton\s*\{[^}]*min-height:\s*(\d+)px", LIGHT_QSS)
    assert m, "LIGHT_QSS missing QPushButton min-height"
    assert int(m.group(1)) >= 28, (
        f"LIGHT_QSS QPushButton min-height={m.group(1)}px, expected >= 28"
    )
    m = re.search(r"QPushButton\s*\{[^}]*min-height:\s*(\d+)px", DARK_QSS)
    assert m, "DARK_QSS missing QPushButton min-height"
    assert int(m.group(1)) >= 28


def test_qss_checkbox_min_height_at_least_28():
    """The QSS must give QCheckBox a min-height so long labels
    don't clip. Phase 35 added ``min-height: 30px`` + ``padding: 2px 0``."""
    from rlpe.gui.styles import LIGHT_QSS, DARK_QSS
    import re
    m = re.search(
        r"QCheckBox,\s*QRadioButton\s*\{[^}]*min-height:\s*(\d+)px",
        LIGHT_QSS,
    )
    assert m, "LIGHT_QSS missing QCheckBox/QRadioButton min-height"
    assert int(m.group(1)) >= 28, (
        f"LIGHT_QSS QCheckBox min-height={m.group(1)}px, expected >= 28"
    )
    m = re.search(
        r"QCheckBox,\s*QRadioButton\s*\{[^}]*min-height:\s*(\d+)px",
        DARK_QSS,
    )
    assert m, "DARK_QSS missing QCheckBox/QRadioButton min-height"
    assert int(m.group(1)) >= 28