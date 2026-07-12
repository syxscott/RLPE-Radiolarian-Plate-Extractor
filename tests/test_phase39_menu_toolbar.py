"""Phase 39 — menu / toolbar / statusbar i18n regression tests.

User reported that the topmost GUI elements (menu bar, toolbar,
status bar) were not translating to Chinese. Root cause: QAction /
QMenu / QToolBar titles were created with bare English strings; the
i18n registry's allWidgets() loop missed them because:

  * QAction is not a QWidget (it's a QObject).
  * QMenu / QToolBar's titles WERE registered, but the i18n
    _apply_to_one only handles attr="text", not attr="title".

Phase 39 fixes:
  1. New ``tr_action`` / ``tr_menu`` factories in i18n_widgets
     register QAction / QMenu objects in a side-channel registry
     (``_MENU_ACTIONS``) that the i18n.set_language listener walks.
  2. ``refresh_all_menu_actions(lang)`` dispatches QMenu → setTitle
     and QAction → setText.
  3. Status bar QLabel switched to tr_label so it auto-translates.
  4. QToolBar windowTitle registered for re-translation.
  5. Toolbar ``Run / Jobs / Results / Settings`` actions switched
     to tr_action with toolbar.* keys.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402

_app = QApplication.instance() or QApplication([])


import pytest


@pytest.fixture(autouse=True)
def _reset_language():
    from rlpe.gui import i18n
    i18n.set_language("zh_CN")
    yield
    i18n.set_language("zh_CN")


# ============================================================
# Menu bar translates on language switch
# ============================================================
def _file_menu(w):
    """Find the File menu by objectName."""
    for m in w.menuBar().findChildren(QMenu):
        if m.objectName() == "menu.file":
            return m
    raise AssertionError("File menu not found")


def test_file_menu_title_is_chinese_by_default():
    """Phase 39: tr_menu('menu.file', ...) makes the menu
    translate to '文件(&F)' on first construct."""
    from rlpe.gui.main_window import MainWindow
    w = MainWindow()
    title = _file_menu(w).title()
    assert "文件" in title, (
        f"File menu title should be Chinese by default, got {title!r}"
    )


def test_file_menu_title_switches_to_english():
    from rlpe.gui.main_window import MainWindow
    from rlpe.gui import i18n
    w = MainWindow()
    fm = _file_menu(w)
    i18n.set_language("en")
    assert "&File" in fm.title(), (
        f"File menu title should switch to English, got {fm.title()!r}"
    )
    i18n.set_language("zh_CN")
    assert "文件" in fm.title()


def test_file_menu_actions_translate():
    from rlpe.gui.main_window import MainWindow
    from rlpe.gui import i18n
    w = MainWindow()
    fm = _file_menu(w)
    # Default is zh_CN
    zh_texts = [a.text() for a in fm.actions() if a.text()]
    i18n.set_language("en")
    en_texts = [a.text() for a in fm.actions() if a.text()]
    assert zh_texts != en_texts, (
        f"File menu actions did NOT switch: zh={zh_texts!r} en={en_texts!r}"
    )
    # ZH should contain Chinese chars
    joined = " ".join(zh_texts)
    assert "打开" in joined or "批处理" in joined, (
        f"ZH File menu actions should contain Chinese: {zh_texts!r}"
    )


# ============================================================
# All top-level menus translate
# ============================================================
def test_all_top_menus_translate():
    """Phase 39: View, Tools, Help, Theme menus all translate."""
    from rlpe.gui.main_window import MainWindow
    from rlpe.gui import i18n
    w = MainWindow()
    i18n.set_language("en")
    en_menus = [m.title() for m in w.menuBar().findChildren(QMenu)]
    i18n.set_language("zh_CN")
    zh_menus = [m.title() for m in w.menuBar().findChildren(QMenu)]
    assert en_menus != zh_menus, (
        f"Top menus did NOT translate: en={en_menus!r} zh={zh_menus!r}"
    )


# ============================================================
# Toolbar
# ============================================================
def test_toolbar_title_translates():
    from rlpe.gui.main_window import MainWindow
    from rlpe.gui import i18n
    w = MainWindow()
    tb = w.findChild(type(w.menuBar()), "mainToolBar")
    # Or look it up by QToolBar type
    from PySide6.QtWidgets import QToolBar
    tbars = w.findChildren(QToolBar)
    assert len(tbars) >= 1
    tb = tbars[0]
    i18n.set_language("en")
    en_title = tb.windowTitle()
    i18n.set_language("zh_CN")
    zh_title = tb.windowTitle()
    assert en_title != zh_title, (
        f"Toolbar title did NOT translate: en={en_title!r} zh={zh_title!r}"
    )
    assert "主" in zh_title or "工具" in zh_title


def test_toolbar_actions_translate():
    """Phase 39: toolbar actions (Open PDF / Batch / Run / Jobs / ...)
    translate on language switch."""
    from rlpe.gui.main_window import MainWindow
    from rlpe.gui import i18n
    from PySide6.QtWidgets import QToolBar
    w = MainWindow()
    tb = w.findChildren(QToolBar)[0]
    en_actions = [a.text() for a in tb.actions() if a.text()]
    i18n.set_language("en")
    en_actions = [a.text() for a in tb.actions() if a.text()]
    i18n.set_language("zh_CN")
    zh_actions = [a.text() for a in tb.actions() if a.text()]
    assert en_actions != zh_actions, (
        f"Toolbar actions did NOT translate: en={en_actions!r} zh={zh_actions!r}"
    )


# ============================================================
# Status bar
# ============================================================
def test_status_bar_default_text_is_chinese():
    from rlpe.gui.main_window import MainWindow
    from rlpe.gui import i18n
    i18n.set_language("zh_CN")
    w = MainWindow()
    # Phase 39: _status_perm is now tr_label('main.idle')
    text = w._status_perm.text()
    assert "就绪" in text, (
        f"Status bar default text should be '就绪' (Ready) in zh_CN, got {text!r}"
    )


def test_status_bar_text_translates_on_switch():
    from rlpe.gui.main_window import MainWindow
    from rlpe.gui import i18n
    w = MainWindow()
    i18n.set_language("en")
    en = w._status_perm.text()
    i18n.set_language("zh_CN")
    zh = w._status_perm.text()
    assert en != zh, (
        f"Status bar text did NOT translate: en={en!r} zh={zh!r}"
    )


def test_status_bar_running_uses_i18n():
    from rlpe.gui.main_window import MainWindow
    from rlpe.gui import i18n
    w = MainWindow()
    i18n.set_language("zh_CN")
    w._on_job_started("abc123", "/tmp/x")
    text = w._status_perm.text()
    # ZH should contain "任务" (job) and the id
    assert "任务" in text and "abc123" in text, (
        f"ZH status 'Job X running…' should contain '任务 abc123', got {text!r}"
    )
    i18n.set_language("en")
    text = w._status_perm.text()
    assert "Job" in text and "abc123" in text


# ============================================================
# About dialog
# ============================================================
def test_about_dialog_uses_i18n():
    """Phase 39: About dialog should not have bare English."""
    from rlpe.gui.main_window import MainWindow
    from rlpe.gui import i18n
    import inspect
    src = inspect.getsource(MainWindow._on_about)
    # Must use i18n._tr
    assert "i18n._tr" in src, "_on_about should use i18n._tr for its message"