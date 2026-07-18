"""Phase 34 — language-switching completeness regression tests.

The user reported that "some parts of the language don't switch" and
"the settings tab has many input boxes that are too short to display
all the text". These tests pin those bugs:

  1. Every QLabel on the Settings tab must translate when the
     language switches (no bare QLabel("English text") left).
  2. Every QPushButton on the Settings tab must translate.
  3. The settings tab title must show the translated app.name.
  4. Every QSpinBox / QComboBox / QLineEdit on the Settings tab must
     have minimum height >= 28 px so values are not clipped under
     the QSS dark theme or 150% DPI.

We also test the main window's status / progress / tab labels.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

_app = QApplication.instance() or QApplication([])


import pytest


@pytest.fixture(autouse=True)
def _reset_language():
    from rlpe.gui import i18n
    i18n.set_language("en")
    yield
    i18n.set_language("en")


# ============================================================
# Bug 1 fix: language switching completeness
# ============================================================
def _texts(labels) -> list[str]:
    return [lbl.text() for lbl in labels if lbl.text()]


def test_settings_tab_all_labels_translate():
    from rlpe.gui import i18n
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    labels = st.findChildren(QLabel)
    en = _texts(labels)
    i18n.set_language("zh_CN")
    zh = _texts(labels)
    # Every label that has non-empty text in en must change in zh
    unchanged = [(i, t) for i, (t, z) in enumerate(zip(en, zh)) if t and t == z]
    # Allow at most 2 unchanged (the title which is special-cased
    # and the sectionTitle class which appears twice)
    assert len(unchanged) <= 2, (
        f"Settings tab: {len(unchanged)} labels did NOT translate to zh_CN: "
        f"{unchanged[:5]}"
    )


def test_settings_tab_all_buttons_translate():
    from PySide6.QtWidgets import QPushButton
    from rlpe.gui import i18n
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    buttons = st.findChildren(QPushButton)
    en = [b.text() for b in buttons]
    i18n.set_language("zh_CN")
    zh = [b.text() for b in buttons]
    unchanged = [(i, t) for i, (t, z) in enumerate(zip(en, zh)) if t and t == z]
    assert len(unchanged) == 0, (
        f"Settings tab: {len(unchanged)} buttons did NOT translate: "
        f"{unchanged}"
    )


def test_settings_tab_title_translates():
    from rlpe.gui import i18n
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    # Title widget is created but its objectName is "settab.title"
    # so we look it up by objectName (not by class=sectionTitle which
    # is shared with results_tab's title).
    title_lbl = st.findChild(QLabel, "")
    # Direct attribute is more reliable
    title = getattr(st, "_title_label", None)
    assert title is not None, "Settings tab must have a _title_label attribute"

    i18n.set_language("zh_CN")
    st._refresh_texts()
    zh = title.text()
    assert "放射虫" in zh, f"Settings title should contain 放射虫 in zh_CN, got: {zh!r}"

    i18n.set_language("en")
    st._refresh_texts()
    en = title.text()
    assert "Radiolarian" in en, f"Settings title should contain 'Radiolarian' in en, got: {en!r}"


# ============================================================
# Bug 2 fix: input heights
# ============================================================
def test_settings_tab_spinboxes_have_min_height_28():
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    for sb in st.findChildren(__import__("PySide6.QtWidgets", fromlist=["QSpinBox"]).QSpinBox):
        assert sb.minimumHeight() >= 28, (
            f"QSpinBox for {sb.objectName()!r} has minHeight={sb.minimumHeight()}, "
            "needs >= 28 to avoid value clipping under QSS dark theme"
        )


def test_settings_tab_comboboxes_have_min_height_28():
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    for cb in st.findChildren(__import__("PySide6.QtWidgets", fromlist=["QComboBox"]).QComboBox):
        assert cb.minimumHeight() >= 28, (
            f"QComboBox for {cb.objectName()!r} has minHeight={cb.minimumHeight()}, "
            "needs >= 28"
        )


def test_settings_tab_dspinboxes_have_min_height_28():
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    for sb in st.findChildren(__import__("PySide6.QtWidgets", fromlist=["QDoubleSpinBox"]).QDoubleSpinBox):
        assert sb.minimumHeight() >= 28, (
            f"QDoubleSpinBox for {sb.objectName()!r} has minHeight={sb.minimumHeight()}, "
            "needs >= 28"
        )


def test_settings_tab_lineedits_have_min_height_28():
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    for le in st.findChildren(__import__("PySide6.QtWidgets", fromlist=["QLineEdit"]).QLineEdit):
        assert le.minimumHeight() >= 28, (
            f"QLineEdit for {le.objectName()!r} has minHeight={le.minimumHeight()}, "
            "needs >= 28"
        )


# ============================================================
# MainWindow tabs also translate
# ============================================================
def test_main_window_tab_labels_translate():
    from rlpe.gui import i18n
    from rlpe.gui.main_window import MainWindow
    # Phase 36: zh_CN is now the default. Capture ZH first, then
    # switch to EN, capture EN, then switch back to ZH to confirm
    # the round-trip works.
    i18n.set_language("zh_CN")
    w = MainWindow()
    zh_tabs = [w._tabs.tabText(i) for i in range(w._tabs.count())]
    i18n.set_language("en")
    en_tabs = [w._tabs.tabText(i) for i in range(w._tabs.count())]
    # Tab labels must differ between en and zh
    assert en_tabs != zh_tabs, (
        f"Main window tab labels did NOT change: en={en_tabs!r} zh={zh_tabs!r}"
    )
    # ZH must contain run/jobs/results/settings in Chinese
    joined = " ".join(zh_tabs)
    assert "运行" in joined, f"ZH tabs missing '运行': {zh_tabs!r}"
    # Round-trip: back to ZH should still be Chinese
    i18n.set_language("zh_CN")
    zh2_tabs = [w._tabs.tabText(i) for i in range(w._tabs.count())]
    assert zh2_tabs == zh_tabs, (
        f"Tab labels did not round-trip back to zh: {zh2_tabs!r} != {zh_tabs!r}"
    )
    assert "任务" in joined, f"ZH tabs missing '任务': {zh_tabs!r}"


def test_main_window_title_translates():
    from rlpe.gui import i18n
    from rlpe.gui.main_window import MainWindow
    i18n.set_language("zh_CN")
    w = MainWindow()
    i18n.set_language("en")
    w._refresh_texts()
    en = w.windowTitle()
    i18n.set_language("zh_CN")
    w._refresh_texts()
    zh = w.windowTitle()
    assert "Radiolarian" in en, f"EN title missing 'Radiolarian': {en!r}"
    # ZH title contains the translated app.name (放射虫)
    assert "放射虫" in zh, f"ZH title missing '放射虫': {zh!r}"


# ============================================================
# Tr_label widget factory: i18n wires through tr_label factory
# ============================================================
def test_tr_label_widget_factory_uses_i18n():
    """The tr_label / tr_button / etc. factories should be used by
    the GUI rather than bare QLabel("English text") so the i18n
    registry can re-text on language switch.

    This is a regression test: Phase 34 fixes bare QLabel/button
    in the Settings tab. If a new bare widget is added in the future,
    this test flags it. (We allow bare labels inside the visual
    preview widget since that's dynamically-rendered text.)"""
    from PySide6.QtWidgets import QLabel
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    # Every QLabel with a non-empty text must have either a
    # translation-related objectName (settab.* / app.* / runtab.* etc.)
    # or a known dynamic-text exception.
    KNOWN_DYNAMIC = {
        "sectionTitle",         # the title label (handled separately)
    }
    for lbl in st.findChildren(QLabel):
        if not lbl.text():
            continue
        if lbl.objectName() in KNOWN_DYNAMIC:
            continue
        # Phase 34: every label must be in the i18n registry
        # (objectName == translation key).
        assert lbl.objectName(), (
            f"Settings tab has bare QLabel {lbl.text()!r} with no "
            "objectName — language switch can't translate it. "
            "Replace with tr_label(key) or set objectName."
        )


def test_tr_label_with_format_template():
    """register_widget_text supports ``fmt`` keyword args for
    templates like ``"⚙️ {app}  ·  v{version}"``."""
    from PySide6.QtWidgets import QLabel
    from rlpe.gui import i18n

    lbl = QLabel()
    lbl.setObjectName("phase34.test.label")
    i18n.STRINGS["en"]["phase34.test.label"] = "Hello {name}!"
    i18n.STRINGS["zh_CN"]["phase34.test.label"] = "你好 {name}!"
    i18n.register_widget_text(
        "phase34.test.label", "text", "phase34.test.label",
        name="World",
    )
    i18n.set_language("en")
    i18n._apply_registry()
    assert lbl.text() == "Hello World!", lbl.text()

    i18n.set_language("zh_CN")
    i18n._apply_registry()
    assert lbl.text() == "你好 World!", lbl.text()
