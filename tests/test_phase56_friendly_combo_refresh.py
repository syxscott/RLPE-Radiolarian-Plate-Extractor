"""Phase 56 regression tests — friendly-name combo refresh on language switch.

Phase 55 audit M8 — the previous code only populated
``*_friendly_options()`` combos once at construction time, so after
the user toggled language the displayed labels stayed in the old
language. The ``populate_friendly_combo`` helper registers an i18n
listener that rebuilds the items on every ``set_language`` call and
preserves the selection by ``userData``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Headless Qt bootstrap — required because PySide6 GUI tests need an
# offscreen QApplication when running under CI / in this venv-less
# environment. The flag MUST be set before importing any Qt module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402

# Make ``rlpe`` importable when pytest is invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication — PySide6 forbids two in one process."""
    app = QApplication.instance() or QApplication(sys.argv)
    yield app
    # Do not quit — leaving it alive lets the test session share it.


# ---------------------------------------------------------------------------
# M8 — friendly combo refresh on language switch
# ---------------------------------------------------------------------------


def test_m8_combo_refreshes_labels_on_language_switch(qapp) -> None:
    """After ``i18n.set_language('en')`` the combo items must show
    English labels, and the previously-selected userData must survive.
    """
    from rlpe.gui import i18n
    from rlpe.gui.constants import ocr_backend_friendly_options
    from rlpe.gui.i18n_widgets import populate_friendly_combo

    # Start in Chinese (the default).
    i18n.set_language("zh_CN")
    cb = QComboBox()
    populate_friendly_combo(cb, ocr_backend_friendly_options, default_code="paddleocr")
    zh_labels = [cb.itemText(i) for i in range(cb.count())]
    assert cb.count() >= 2, "OCR backend combo should have ≥2 items"
    assert cb.currentData() == "paddleocr"
    # At least one Chinese character in the label.
    assert any(ord(c) > 0x2E80 for c in " ".join(zh_labels)), (
        f"Expected Chinese labels in zh_CN mode, got {zh_labels!r}"
    )

    # Switch to English.
    i18n.set_language("en")
    en_labels = [cb.itemText(i) for i in range(cb.count())]
    assert en_labels != zh_labels, (
        "Friendly-name combo did NOT refresh on language switch — Phase 55 M8 regression."
    )
    # userData is preserved.
    assert cb.currentData() == "paddleocr", (
        "Selection lost during language switch; populate_friendly_combo "
        "must restore the previous userData after rebuilding items."
    )

    # Switch back to Chinese — labels re-become Chinese.
    i18n.set_language("zh_CN")
    zh_labels_again = [cb.itemText(i) for i in range(cb.count())]
    assert zh_labels_again == zh_labels, (
        "Friendly-name combo did not return to Chinese on second switch."
    )
    assert cb.currentData() == "paddleocr"


def test_m8_helper_preserves_unknown_selection(qapp) -> None:
    """When the combo has no current userData (first-time build),
    the default_code argument is used and survives subsequent
    language switches.
    """
    from rlpe.gui import i18n
    from rlpe.gui.constants import theme_friendly_options
    from rlpe.gui.i18n_widgets import populate_friendly_combo

    i18n.set_language("zh_CN")
    cb = QComboBox()
    populate_friendly_combo(cb, theme_friendly_options, default_code="light")
    assert cb.currentData() == "light"
    i18n.set_language("en")
    assert cb.currentData() == "light", (
        "Default selection must survive language switch (helper bug)."
    )


def test_m8_helper_does_not_fire_currentIndexChanged(qapp) -> None:
    """populate_friendly_combo blocks signals during the rebuild so
    the page-up/down handlers in the consuming tab don't get a
    half-built combo as their target.
    """
    from rlpe.gui import i18n
    from rlpe.gui.constants import ocr_lang_friendly_options
    from rlpe.gui.i18n_widgets import populate_friendly_combo

    i18n.set_language("zh_CN")
    cb = QComboBox()
    populate_friendly_combo(cb, ocr_lang_friendly_options, default_code="en")
    fired: list[int] = []
    cb.currentIndexChanged.connect(lambda ix: fired.append(ix))
    i18n.set_language("en")
    assert fired == [], (
        f"currentIndexChanged fired {len(fired)} time(s) during "
        "language switch — should have been blocked by the helper."
    )
