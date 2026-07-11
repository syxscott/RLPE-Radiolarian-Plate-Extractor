"""Phase 36 — scrollable Settings tab + Chinese-as-default-language tests.

Phase 35 fixed per-widget min heights but the Settings tab still had
two unresolved bugs the user reported:

  1. The tab body overflowed the window on small / 150% DPI screens,
     clipping the "Save settings" button. Phase 36 wraps the body
     in a QScrollArea so the tab is scrollable.

  2. The default language was English. The user wants Chinese
     ("默认语言为中文"). Phase 36 changes ``i18n._CURRENT_LANG``
     to ``zh_CN`` and persists the choice in QSettings under
     ``ui/language``.

These tests pin both behaviours.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

_app = QApplication.instance() or QApplication([])


import pytest


@pytest.fixture(autouse=True)
def _reset_language():
    """Reset to default zh_CN before each test so test ordering can't
    flip the language out from under us. Phase 36 changed the default."""
    from rlpe.gui import i18n
    i18n.set_language("zh_CN")
    yield
    i18n.set_language("zh_CN")


# ============================================================
# Settings tab wraps in QScrollArea
# ============================================================
def test_settings_tab_contains_scrollarea():
    """Phase 36: Settings tab body is wrapped in a QScrollArea so
    the Save button doesn't get clipped on small / 150% DPI windows."""
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    scrolls = st.findChildren(QScrollArea)
    assert len(scrolls) >= 1, (
        "SettingsTab must contain a QScrollArea so the body is scrollable"
    )
    # The scroll area must be resizable (so the body fills the viewport)
    assert scrolls[0].widgetResizable(), (
        "QScrollArea must have widgetResizable=True so the body fills width"
    )


def test_settings_tab_scrollarea_has_body_widget():
    """The scroll area's body widget must be set so it's visible."""
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    scroll = st.findChild(QScrollArea)
    assert scroll is not None
    body = scroll.widget()
    assert body is not None, "QScrollArea must have a body widget set"
    # Body must contain the same widgets the old outer layout had
    from PySide6.QtWidgets import QGroupBox
    boxes = body.findChildren(QGroupBox)
    assert len(boxes) >= 7, (
        f"ScrollArea body should have 7 group boxes (Appearance / Dirs / "
        f"GROBID / OCR / LLM / PBDB / Diagnostics), got {len(boxes)}"
    )


# ============================================================
# Default language is Chinese
# ============================================================
def test_default_language_is_zh_cn():
    """Phase 36: at import time, i18n defaults to zh_CN so first-time
    GUI launches display in Chinese."""
    import importlib
    import rlpe.gui.i18n as i18n_mod
    importlib.reload(i18n_mod)
    assert i18n_mod._CURRENT_LANG == "zh_CN", (
        f"Default language must be zh_CN, got {i18n_mod._CURRENT_LANG!r}"
    )
    assert i18n_mod.current_language() == "zh_CN"


def test_settings_tab_renders_in_chinese_by_default():
    """The freshly-constructed SettingsTab should show Chinese group
    titles because the default language is zh_CN."""
    from rlpe.gui.settings_tab import SettingsTab
    from PySide6.QtWidgets import QGroupBox
    st = SettingsTab({})
    titles = [gb.title() for gb in st.findChildren(QGroupBox)]
    # We removed the emoji so the title is just the Chinese word.
    assert "外观" in titles, (
        f"Settings tab must show '外观' (Appearance in zh_CN), titles={titles!r}"
    )
    assert "GROBID" in titles  # GROBID stays as-is in Chinese too


def test_settings_lang_picker_defaults_to_chinese():
    """The language combo must default to Chinese at startup."""
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    # Find the language combo (its objectName is "settab.lang")
    for i in range(st._lang_combo.count()):
        if st._lang_combo.itemData(i) == "zh_CN":
            assert st._lang_combo.currentIndex() == i, (
                f"Language combo should default to zh_CN (index {i}), "
                f"got currentIndex={st._lang_combo.currentIndex()}"
            )
            return
    pytest.fail("Language combo missing zh_CN entry")


# ============================================================
# Persisted language preference
# ============================================================
def test_lang_picker_persists_choice_to_qsettings():
    """Switching the language picker must write ui/language to QSettings
    so the next launch uses the same language."""
    from rlpe.gui.settings_tab import SettingsTab
    from rlpe.gui import i18n
    from PySide6.QtCore import QSettings
    from rlpe.gui.constants import APP_DOMAIN, APP_NAME

    # Clear any prior setting
    qs = QSettings(APP_DOMAIN, APP_NAME)
    qs.remove("ui/language")
    qs.sync()

    st = SettingsTab({})
    # Find the zh_CN combo item and select en instead
    for i in range(st._lang_combo.count()):
        if st._lang_combo.itemData(i) == "en":
            st._lang_combo.setCurrentIndex(i)
            break

    # Now QSettings should have ui/language = "en"
    qs2 = QSettings(APP_DOMAIN, APP_NAME)
    saved = qs2.value("ui/language", "")
    assert saved == "en", (
        f"Switching lang picker must persist ui/language='en', got {saved!r}"
    )

    # Restore zh_CN for the rest of the suite
    i18n.set_language("zh_CN")


def test_main_window_loads_persisted_language():
    """MainWindow._build_ui must load ui/language from QSettings."""
    from rlpe.gui.main_window import MainWindow
    from rlpe.gui import i18n
    from PySide6.QtCore import QSettings
    from rlpe.gui.constants import APP_DOMAIN, APP_NAME

    qs = QSettings(APP_DOMAIN, APP_NAME)
    qs.setValue("ui/language", "en")
    qs.sync()

    try:
        w = MainWindow()
        assert i18n.current_language() == "en", (
            f"MainWindow must load persisted language 'en', got "
            f"{i18n.current_language()!r}"
        )
    finally:
        # Restore Chinese default for downstream tests
        qs.setValue("ui/language", "zh_CN")
        qs.sync()
        i18n.set_language("zh_CN")


# ============================================================
# QSS no longer uses the broken title-badge subcontrol
# ============================================================
def test_qss_groupbox_title_no_blue_badge():
    """Phase 36 dropped the blue "title badge" subcontrol that was
    clipping titles. Verify the QSS now uses transparent background
    for QGroupBox::title."""
    from rlpe.gui.styles import LIGHT_QSS, DARK_QSS
    import re
    for qss, name in [(LIGHT_QSS, "LIGHT"), (DARK_QSS, "DARK")]:
        m = re.search(
            r"QGroupBox::title\s*\{[^}]*background-color:\s*([^;}]+)",
            qss,
        )
        assert m, f"{name}_QSS missing QGroupBox::title background-color"
        bg = m.group(1).strip()
        assert bg == "transparent", (
            f"{name}_QSS QGroupBox::title must be transparent (was {bg!r}) "
            "so the title text isn't clipped by a badge subcontrol"
        )


# ============================================================
# No emojis in groupbox titles (they caused title clipping)
# ============================================================
def test_groupbox_titles_have_no_emojis():
    """Phase 36: removed leading emoji (🎨/📁/🌐/🔡/🧠/🦴/🛠️)
    from groupbox titles because QFontMetrics.horizontalAdvance on
    the offscreen platform underestimates emoji widths, clipping the
    rest of the title text."""
    from rlpe.gui import strings_en, strings_zh_CN
    title_keys = (
        "settab.appearance", "settab.dirs", "settab.grobid",
        "settab.ocr", "settab.llm", "settab.pbdb", "settab.diag",
    )
    emoji_chars = set("🎨📁🌐🔡🧠🦴🛠️")
    for key in title_keys:
        for mod in (strings_en, strings_zh_CN):
            val = mod.STRINGS.get(key, "")
            for ch in val:
                assert ch not in emoji_chars, (
                    f"{mod.__name__}.STRINGS[{key!r}]={val!r} still "
                    f"contains emoji {ch!r} which caused clipping"
                )


# ============================================================
# Input row height is 32 px (was 30, bumped so spinbox arrows fit)
# ============================================================
def test_widget_factory_default_height_is_32():
    """Phase 36: bumped default min_height from 30 → 32 so spinbox
    up/down arrows fit inside the widget rectangle."""
    from PySide6.QtWidgets import (
        QCheckBox, QComboBox, QLineEdit, QPushButton, QSpinBox,
    )
    from rlpe.gui.i18n_widgets import (
        tr_button, tr_checkbox, tr_combobox, tr_lineedit, tr_spinbox,
    )
    assert tr_button("dummy.btn").minimumHeight() == 32
    assert tr_checkbox("dummy.cb").minimumHeight() == 32
    assert tr_combobox("dummy.combo").minimumHeight() == 32
    assert tr_lineedit("dummy.le").minimumHeight() == 32
    assert tr_spinbox("dummy.sb").minimumHeight() == 32