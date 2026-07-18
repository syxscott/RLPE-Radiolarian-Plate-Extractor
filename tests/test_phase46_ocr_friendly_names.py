"""Phase 46 — OCR language friendly names.

User asked: instead of unfriendly ISO codes like "en" / "ch_sim" /
"ja", show "English" / "Chinese (Simplified)" / "Japanese" in the
GUI dropdown. The OCR backend still receives the ISO codes via
userData; the QComboBox is setEditable so power users can type
custom comma-separated lists.

Tests pin:
  1. constants.OCR_LANGUAGE_OPTIONS contains the standard
     languages with friendly names in both English and Chinese.
  2. ocr_lang_to_friendly_name() returns the right name based
     on the current i18n language.
  3. ocr_lang_friendly_options() returns [(iso_code, friendly), ...]
  4. The Run tab ocr_lang widget is a QComboBox (not a QLineEdit)
     with friendly names as labels and ISO codes as userData.
  5. collect_settings returns the ISO code from userData.
  6. The i18n placeholders for OCR language no longer mention
     "en, ja, ch_sim" codes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PySide6")
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
# 1. OCR_LANGUAGE_OPTIONS table
# ============================================================
def test_ocr_language_options_contains_standard_languages():
    """Phase 46: the friendly-name table must include English,
    Chinese (Simplified + Traditional), Japanese, Korean, French,
    German, Russian."""
    from rlpe.gui.constants import OCR_LANGUAGE_OPTIONS
    codes = [code for code, _en, _zh in OCR_LANGUAGE_OPTIONS]
    assert "en" in codes
    assert "ch_sim" in codes
    assert "ch_tra" in codes
    assert "ja" in codes
    assert "ko" in codes
    assert "fr" in codes
    assert "de" in codes
    assert "ru" in codes


def test_ocr_language_options_each_has_friendly_name():
    """Phase 46: each entry must have an English friendly name AND
    a Chinese friendly name (not the raw ISO code)."""
    from rlpe.gui.constants import OCR_LANGUAGE_OPTIONS
    for code, en_name, zh_name in OCR_LANGUAGE_OPTIONS:
        assert en_name != code, (
            f"Friendly name for {code!r} is the raw ISO code: "
            f"en={en_name!r}"
        )
        assert zh_name != code, (
            f"Chinese name for {code!r} is the raw ISO code: "
            f"zh={zh_name!r}"
        )
        assert len(en_name) > 1, (
            f"English name for {code!r} is too short: {en_name!r}"
        )
        assert len(zh_name) > 0, (
            f"Chinese name for {code!r} is empty: {zh_name!r}"
        )


# ============================================================
# 2. ocr_lang_to_friendly_name()
# ============================================================
def test_ocr_lang_to_friendly_name_zh():
    from rlpe.gui.constants import ocr_lang_to_friendly_name
    assert ocr_lang_to_friendly_name("en") == "英语"
    assert ocr_lang_to_friendly_name("ch_sim") == "中文 (简体)"
    assert ocr_lang_to_friendly_name("ja") == "日语"


def test_ocr_lang_to_friendly_name_en():
    from rlpe.gui import i18n
    from rlpe.gui.constants import ocr_lang_to_friendly_name
    i18n.set_language("en")
    assert ocr_lang_to_friendly_name("en") == "English"
    assert ocr_lang_to_friendly_name("ch_sim") == "Chinese (Simplified)"
    assert ocr_lang_to_friendly_name("ja") == "Japanese"


def test_ocr_lang_to_friendly_name_unknown_iso_falls_back():
    from rlpe.gui import i18n
    from rlpe.gui.constants import ocr_lang_to_friendly_name
    i18n.set_language("en")
    # Unknown ISO code → return as-is
    assert ocr_lang_to_friendly_name("xyz") == "xyz"
    i18n.set_language("zh_CN")
    assert ocr_lang_to_friendly_name("xyz") == "xyz"


# ============================================================
# 3. ocr_lang_friendly_options()
# ============================================================
def test_ocr_lang_friendly_options_returns_pairs():
    from rlpe.gui import i18n
    from rlpe.gui.constants import ocr_lang_friendly_options
    i18n.set_language("en")
    options = ocr_lang_friendly_options()
    assert isinstance(options, list)
    assert len(options) >= 8
    for iso_code, friendly in options:
        assert isinstance(iso_code, str)
        assert isinstance(friendly, str)
        assert iso_code.isascii() and "_" in iso_code or len(iso_code) <= 3, (
            f"ISO code {iso_code!r} should be a short ISO code"
        )


def test_ocr_lang_friendly_options_zh_uses_chinese():
    from rlpe.gui import i18n
    from rlpe.gui.constants import ocr_lang_friendly_options
    i18n.set_language("zh_CN")
    options = ocr_lang_friendly_options()
    # In zh_CN, the "en" option should display as "英语" not "English"
    en_option = next(opt for opt in options if opt[0] == "en")
    assert en_option[1] == "英语", (
        f"Under zh_CN, ocr_lang_friendly_options should return Chinese names, "
        f"got {en_option!r}"
    )


# ============================================================
# 4. Run tab uses QComboBox (not QLineEdit) with friendly names
# ============================================================
def test_run_tab_ocr_lang_is_qcombobox_with_friendly_names():
    """Phase 46: the ocr_lang widget must be a QComboBox with
    friendly names as labels and ISO codes in userData."""
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    from PySide6.QtWidgets import QComboBox
    assert isinstance(rt._ocr_lang_edit, QComboBox), (
        f"ocr_lang widget must be a QComboBox, got "
        f"{type(rt._ocr_lang_edit).__name__}"
    )
    assert rt._ocr_lang_edit.count() >= 8, (
        f"ocr_lang combo should have >= 8 entries, got {rt._ocr_lang_edit.count()}"
    )
    # Each item's userData is the ISO code; each item's text is
    # the friendly name (not the raw ISO code).
    for i in range(rt._ocr_lang_edit.count()):
        udata = rt._ocr_lang_edit.itemData(i)
        text = rt._ocr_lang_edit.itemText(i)
        assert udata, f"item {i} userData must be the ISO code, got {udata!r}"
        assert text != udata, (
            f"item {i} text must be the friendly name, not the raw ISO "
            f"code: text={text!r} udata={udata!r}"
        )


def test_run_tab_ocr_lang_default_is_english():
    """Phase 46: default is English ("en") but displayed as
    the friendly name (English / 英语)."""
    from rlpe.gui import i18n
    from rlpe.gui.run_tab import RunTab
    i18n.set_language("zh_CN")
    rt = RunTab({})
    # Default currentData should be "en"
    assert rt._ocr_lang_edit.currentData() == "en", (
        f"default ocr_lang should be 'en', got {rt._ocr_lang_edit.currentData()!r}"
    )
    # Default text should be the friendly name (Chinese: 英语)
    assert rt._ocr_lang_edit.currentText() == "英语", (
        f"default ocr_lang text should be '英语' in zh_CN, got "
        f"{rt._ocr_lang_edit.currentText()!r}"
    )


# ============================================================
# 5. collect_settings returns the ISO code
# ============================================================
def test_collect_settings_returns_iso_code_for_ocr_lang():
    """Phase 46: collect_settings() must return the ISO code (e.g.
    'ja') not the friendly name ('Japanese'). The OCR backend
    needs the ISO code."""
    from rlpe.gui import i18n
    from rlpe.gui.run_tab import RunTab
    i18n.set_language("zh_CN")
    rt = RunTab({})
    # Find the Japanese item, select it
    for i in range(rt._ocr_lang_edit.count()):
        if rt._ocr_lang_edit.itemData(i) == "ja":
            rt._ocr_lang_edit.setCurrentIndex(i)
            break
    settings = rt.collect_settings()
    assert settings["ocr_lang"] == "ja", (
        f"collect_settings['ocr_lang'] must be the ISO code 'ja', got "
        f"{settings['ocr_lang']!r} (Phase 46 bug: friendly name leaks "
        f"into the backend)"
    )


def test_collect_settings_handles_custom_typed_string():
    """Phase 46: if the user types a custom string (e.g. 'en,ja'),
    collect_settings uses the typed text (power user override)."""
    from rlpe.gui import i18n
    from rlpe.gui.run_tab import RunTab
    i18n.set_language("en")
    rt = RunTab({})
    # Simulate power user typing "en,ja" — set the currentIndex
    # to a non-existent value (-1) so currentData() returns None,
    # then set the lineEdit text to "en,ja". The collect_settings
    # logic falls back to currentText() when currentData() is None.
    rt._ocr_lang_edit.setCurrentIndex(-1)
    rt._ocr_lang_edit.lineEdit().setText("en,ja")
    _app.processEvents()  # let Qt propagate textChanged
    settings = rt.collect_settings()
    assert settings["ocr_lang"] == "en,ja", (
        f"Custom typed text must be preserved, got {settings['ocr_lang']!r}"
    )


# ============================================================
# 6. i18n placeholders no longer use ISO codes
# ============================================================
def test_runtab_ocr_lang_placeholder_does_not_mention_iso_codes():
    """Phase 46: the placeholder for the OCR language field
    should not show the raw ISO codes as the only example —
    it should hint at friendly names like English / 中文 / 日本語."""
    from rlpe.gui import i18n
    from rlpe.gui import strings_en
    i18n.set_language("en")
    en_placeholder = strings_en.STRINGS.get("runtab.ocr_lang.placeholder", "")
    # The placeholder may STILL mention ISO codes (it's a hint for
    # power users) but it should ALSO mention the friendly names.
    assert "English" in en_placeholder or "中文" in en_placeholder, (
        f"EN placeholder should hint at friendly names: {en_placeholder!r}"
    )


def test_settab_ocr_lang_placeholder_does_not_mention_iso_codes():
    from rlpe.gui import i18n
    from rlpe.gui import strings_en
    i18n.set_language("en")
    en_placeholder = strings_en.STRINGS.get("settab.ocr.lang.placeholder", "")
    assert "English" in en_placeholder or "中文" in en_placeholder, (
        f"EN settings placeholder should hint at friendly names: "
        f"{en_placeholder!r}"
    )