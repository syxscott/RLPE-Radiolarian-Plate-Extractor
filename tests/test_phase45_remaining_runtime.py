"""Phase 45 — additional runtime bug fixes.

Covers the audit's next tier:
  1. _ProgressCellDelegate.paint must wrap drawControl in
     painter.save() / restore() so pen/brush changes don't leak
     into the next delegate's paint.
  2. tr_combobox must block signals during construction so
     currentIndexChanged doesn't fire on a half-built widget.
  3. QSettings must use APP_AUTHOR (not APP_DOMAIN) for the
     organization so Windows registry hive matches the set*
     calls. Previously the hive was wrong and the GUI
     would re-create defaults on every launch.
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


# ============================================================
# _ProgressCellDelegate painter save/restore
# ============================================================
def test_progress_cell_delegate_wraps_draw_in_save_restore():
    """Phase 45: drawControl modifies the painter's pen/brush; the
    delegate must call painter.save() / painter.restore() to
    prevent leaks into the next paint call."""
    import inspect

    from rlpe.gui.jobs_tab import _ProgressCellDelegate

    src = inspect.getsource(_ProgressCellDelegate.paint)
    assert "painter.save()" in src, (
        "_ProgressCellDelegate.paint must call painter.save() before drawControl"
    )
    assert "painter.restore()" in src, (
        "_ProgressCellDelegate.paint must call painter.restore() "
        "after drawControl (even on exception)"
    )
    # Verify the save/restore are in a try/finally (defensive)
    assert "finally:" in src, (
        "painter.save/restore must be wrapped in try/finally so "
        "an exception during drawControl doesn't leave the "
        "painter in a leaked state"
    )


# ============================================================
# tr_combobox signal blocking during construction
# ============================================================
def test_tr_combobox_blocks_signals_during_construction():
    """Phase 45: tr_combobox must block currentIndexChanged
    during addItems/setCurrentIndex so slot handlers don't fire
    on a half-built widget."""
    import inspect

    from rlpe.gui.i18n_widgets import tr_combobox

    src = inspect.getsource(tr_combobox)
    # The factory should call blockSignals(True) at the start of
    # construction and blockSignals(False) at the end (in finally).
    assert "blockSignals(True)" in src, "tr_combobox must call blockSignals(True) before addItems"
    assert "blockSignals(False)" in src, (
        "tr_combobox must call blockSignals(False) in a finally block"
    )


# ============================================================
# QSettings hive consistency
# ============================================================
def test_qsettings_uses_app_author_not_app_domain():
    """Phase 45: QSettings(APP_DOMAIN, APP_NAME) writes to a
    DIFFERENT Windows registry hive than setOrganizationName(
    APP_AUTHOR). The GUI would then re-create defaults on every
    launch because the read hive is empty. Fixed: use APP_AUTHOR
    everywhere QSettings is constructed."""
    import inspect

    import rlpe.gui.app as app_mod
    import rlpe.gui.main_window as mw
    import rlpe.gui.settings_tab as st

    for mod in (st, mw, app_mod):
        src = inspect.getsource(mod)
        # We don't have a constant for the value in source; the
        # fix is that the call uses APP_AUTHOR not APP_DOMAIN. We
        # check that the bare APP_DOMAIN string is NOT used as
        # the first arg of any QSettings() call.
        assert "QSettings(APP_DOMAIN," not in src, (
            f"{mod.__name__} still has QSettings(APP_DOMAIN, ...) — "
            "should use APP_AUTHOR so the Windows registry hive "
            "matches setOrganizationName()."
        )


# ============================================================
# Integration: i18n listeners dedupes on re-register
# ============================================================
def test_i18n_add_listener_dedupe_still_works():
    """Regression: Phase 44 dedupe must still work after Phase 45
    changes."""
    from rlpe.gui import i18n

    i18n._LISTENERS.clear()

    calls = []

    def listener(lang):
        calls.append(lang)

    i18n.add_listener(listener)
    i18n.add_listener(listener)  # duplicate
    i18n.add_listener(listener)  # duplicate
    # Switch language
    i18n.set_language("en")
    # Only 1 listener, but 2 in _LISTENERS (one from us, one from
    # jobs_tab, settings_tab, etc.). The dedupe prevents the SAME
    # listener from being added twice.
    # Verify dedupe specifically: the function is in the list once
    listener_count = sum(1 for fn in i18n._LISTENERS if fn is listener)
    assert listener_count == 1, f"add_listener must dedupe by identity; got {listener_count} copies"
    i18n._LISTENERS.clear()
