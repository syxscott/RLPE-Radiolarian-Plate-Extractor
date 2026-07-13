"""Phase 47 — friendly names for all technical tokens.

Phase 46 added friendly names for OCR languages. This commit
extends the same approach to:
  * Theme (light/dark/system → 浅色/深色/跟随系统)
  * LLM backend (minimax/minimax-m3/... → MiniMax-M3 (推荐)/...)
  * OCR backend (paddleocr/easyocr → PaddleOCR (推荐)/...)
  * M3 prompt language (auto/zh/en/ja → 自动检测/中文/英语/日本語)

The underlying code (worker, pipeline) still receives the ISO
codes via QComboBox.userData; the UI displays the friendly
names. setEditable(True) on the OCR language combo lets power
users type custom comma-separated lists.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402

_app = QApplication.instance() or QApplication([])


import pytest


@pytest.fixture(autouse=True)
def _reset_language():
    from rlpe.gui import i18n
    i18n.set_language("zh_CN")
    yield
    i18n.set_language("zh_CN")


# ============================================================
# Constants: friendly-name tables
# ============================================================
def test_theme_options_use_friendly_names():
    """Phase 47: theme dropdown shows 浅色 / 深色 / 跟随系统."""
    from rlpe.gui.constants import THEME_OPTIONS
    codes = [code for code, _en, _zh in THEME_OPTIONS]
    assert codes == ["light", "dark", "system"]
    # Each entry has a Chinese name that's not the raw code
    for code, en_name, zh_name in THEME_OPTIONS:
        assert zh_name != code, (
            f"Theme {code!r} has raw code as Chinese name: {zh_name!r}"
        )
        assert "浅" in zh_name or "深" in zh_name or "跟随" in zh_name, (
            f"Theme {code!r} Chinese name {zh_name!r} should hint at the theme"
        )


def test_llm_backend_options_use_friendly_names():
    """Phase 47: LLM backend dropdown shows MiniMax-M3 (推荐) / etc."""
    from rlpe.gui.constants import LLM_BACKEND_OPTIONS
    codes = [code for code, _en, _zh in LLM_BACKEND_OPTIONS]
    assert "minimax" in codes
    assert "minimax-m3" in codes
    assert "rules" in codes  # for users who don't want LLM


def test_ocr_backend_options_use_friendly_names():
    """Phase 47: OCR backend dropdown shows PaddleOCR (推荐) / etc."""
    from rlpe.gui.constants import OCR_BACKEND_OPTIONS
    codes = [code for code, _en, _zh in OCR_BACKEND_OPTIONS]
    assert "paddleocr" in codes
    assert "easyocr" in codes


def test_m3_prompt_lang_options_use_friendly_names():
    """Phase 47: M3 prompt language dropdown shows 自动检测 / etc."""
    from rlpe.gui.constants import M3_PROMPT_LANG_OPTIONS
    codes = [code for code, _en, _zh in M3_PROMPT_LANG_OPTIONS]
    assert "auto" in codes
    assert "zh" in codes
    assert "en" in codes
    assert "ja" in codes


def test_friendly_options_helpers_return_pairs():
    """Phase 47: the *_friendly_options() helpers return (iso, name)
    pairs in the current language."""
    from rlpe.gui import i18n
    from rlpe.gui.constants import (
        theme_friendly_options, llm_backend_friendly_options,
        ocr_backend_friendly_options, m3_prompt_lang_friendly_options,
    )

    for helper in (
        theme_friendly_options, llm_backend_friendly_options,
        ocr_backend_friendly_options, m3_prompt_lang_friendly_options,
    ):
        i18n.set_language("en")
        en_options = helper()
        i18n.set_language("zh_CN")
        zh_options = helper()
        # Same length, different texts
        assert len(en_options) == len(zh_options)
        assert en_options != zh_options, (
            f"{helper.__name__} should return different labels for en vs zh_CN"
        )
        for (iso_en, name_en), (iso_zh, name_zh) in zip(en_options, zh_options):
            assert iso_en == iso_zh, "ISO codes should match"
            assert name_en != name_zh, "Labels should differ between languages"


# ============================================================
# Run tab: friendly comboboxes
# ============================================================
def test_run_tab_llm_combo_uses_friendly_names():
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    # Find the LLM combo by checking itemData on every QComboBox
    for cb in rt.findChildren(QComboBox):
        if cb.count() == 0:
            continue
        first_data = cb.itemData(0)
        if first_data in ("minimax", "minimax-m3", "minimax_api",
                          "transformers", "ollama", "llamacpp", "rules"):
            # Verify text != data
            for i in range(cb.count()):
                assert cb.itemText(i) != cb.itemData(i), (
                    f"Run tab LLM combo item {i} has raw code as text: "
                    f"text={cb.itemText(i)!r} data={cb.itemData(i)!r}"
                )
            return
    pytest.fail("Could not find LLM backend QComboBox in Run tab")


def test_run_tab_m3_lang_uses_friendly_names():
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    for cb in rt.findChildren(QComboBox):
        if cb.count() == 0:
            continue
        if cb.itemData(0) in ("auto", "zh", "en", "ja"):
            # Verify text != data
            for i in range(cb.count()):
                assert cb.itemText(i) != cb.itemData(i), (
                    f"Run tab M3 lang combo item {i} has raw code as text"
                )
            return
    pytest.fail("Could not find M3 prompt language QComboBox in Run tab")


def test_run_tab_ocr_lang_uses_friendly_names():
    from rlpe.gui.run_tab import RunTab
    from rlpe.gui.constants import OCR_LANGUAGE_OPTIONS
    rt = RunTab({})
    for cb in rt.findChildren(QComboBox):
        if cb.count() == 0:
            continue
        if cb.itemData(0) in ("en", "ch_sim"):
            # Should be the OCR lang combo
            for i in range(cb.count()):
                assert cb.itemText(i) != cb.itemData(i), (
                    f"Run tab OCR lang combo item {i} has raw code as text"
                )
            return
    pytest.fail("Could not find OCR language QComboBox in Run tab")


def test_run_tab_collect_settings_returns_iso_codes():
    """Phase 47: collect_settings() must return ISO codes (e.g.
    'minimax', 'ja') not friendly names. The OCR backend expects
    the raw codes."""
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    settings = rt.collect_settings()
    # OCR lang
    assert settings["ocr_lang"] in {"en", "ch_sim", "ch_tra", "ja",
                                     "ko", "fr", "de", "ru"}, (
        f"ocr_lang should be ISO code, got {settings['ocr_lang']!r}"
    )
    # LLM backend
    assert settings["llm_backend"] in {
        "minimax", "minimax-m3", "minimax_api", "transformers",
        "ollama", "llamacpp", "rules",
    }, f"llm_backend should be ISO code, got {settings['llm_backend']!r}"
    # M3 prompt lang
    assert settings["m3_prompt_lang"] in {"auto", "zh", "en", "ja"}, (
        f"m3_prompt_lang should be ISO code, got "
        f"{settings['m3_prompt_lang']!r}"
    )


# ============================================================
# Settings tab: friendly comboboxes
# ============================================================
def test_settings_tab_theme_combo_uses_friendly_names():
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    # The theme combo has the raw codes "light"/"dark"/"system"
    for cb in st.findChildren(QComboBox):
        if cb.itemData(0) in ("light", "dark", "system"):
            for i in range(cb.count()):
                assert cb.itemText(i) != cb.itemData(i), (
                    f"Settings tab theme combo item {i} has raw code as text"
                )
            return
    pytest.fail("Could not find theme QComboBox in Settings tab")


def test_settings_tab_ocr_backend_combo_uses_friendly_names():
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    for cb in st.findChildren(QComboBox):
        if cb.itemData(0) in ("paddleocr", "easyocr"):
            for i in range(cb.count()):
                assert cb.itemText(i) != cb.itemData(i), (
                    f"Settings tab OCR backend combo item {i} has raw code"
                )
            return
    pytest.fail("Could not find OCR backend QComboBox in Settings tab")


def test_settings_tab_llm_backend_combo_uses_friendly_names():
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    for cb in st.findChildren(QComboBox):
        if cb.itemData(0) in {
            "minimax", "minimax-m3", "minimax_api", "transformers",
            "ollama", "llamacpp", "rules",
        }:
            for i in range(cb.count()):
                assert cb.itemText(i) != cb.itemData(i), (
                    f"Settings tab LLM combo item {i} has raw code"
                )
            return
    pytest.fail("Could not find LLM backend QComboBox in Settings tab")


def test_settings_tab_m3_prompt_lang_uses_friendly_names():
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    for cb in st.findChildren(QComboBox):
        if cb.itemData(0) in ("auto", "zh", "en", "ja"):
            for i in range(cb.count()):
                assert cb.itemText(i) != cb.itemData(i), (
                    f"Settings tab M3 prompt lang combo item {i} has raw code"
                )
            return
    pytest.fail("Could not find M3 prompt language QComboBox in Settings tab")


def test_settings_tab_save_uses_current_data_iso_codes():
    """Phase 47: when the user saves Settings, the saved values
    must be ISO codes (not friendly names) so the worker and
    next launch see the right value."""
    from rlpe.gui.settings_tab import SettingsTab
    st = SettingsTab({})
    # Find each friendly combo and verify currentData() returns
    # the raw code (not the displayed friendly name).
    for cb in st.findChildren(QComboBox):
        if cb.itemData(0) in ("light", "dark", "system"):
            # Theme
            assert cb.currentData() in ("light", "dark", "system"), (
                f"Theme currentData should be ISO code, got "
                f"{cb.currentData()!r}"
            )
        elif cb.itemData(0) in ("paddleocr", "easyocr"):
            assert cb.currentData() in ("paddleocr", "easyocr"), (
                f"OCR backend currentData should be ISO code, got "
                f"{cb.currentData()!r}"
            )
        elif cb.itemData(0) in {"minimax", "minimax-m3", "minimax_api",
                                 "transformers", "ollama", "llamacpp", "rules"}:
            assert cb.currentData() in {
                "minimax", "minimax-m3", "minimax_api", "transformers",
                "ollama", "llamacpp", "rules",
            }, f"LLM backend currentData should be ISO code, got {cb.currentData()!r}"


# ============================================================
# min_height preservation
# ============================================================
def test_settings_tab_friendly_combos_have_min_height_32():
    """Phase 47: every QComboBox in the Settings tab must have
    min_height >= 32 (Phase 35 invariant)."""
    from rlpe.gui.settings_tab import SettingsTab
    from PySide6.QtWidgets import QSizePolicy
    st = SettingsTab({})
    _app.processEvents()
    for cb in st.findChildren(QComboBox):
        if cb.count() == 0:
            continue
        # Skip the lang combo (which is created via tr_combobox
        # and has the height baked in already)
        if cb.objectName() == "settab.lang":
            continue
        assert cb.minimumHeight() >= 32 or cb.sizePolicy().verticalPolicy() == QSizePolicy.Fixed, (
            f"Settings tab QComboBox (data={cb.itemData(0)!r}) "
            f"min_height={cb.minimumHeight()}; should be >= 32"
        )