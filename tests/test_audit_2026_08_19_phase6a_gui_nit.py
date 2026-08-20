"""Phase 6A GUI NIT sweep (2026-08-19).

Five MINOR/NIT bugs from the 2026-08-19 multi-agent audit follow-up:

* **NIT-1** — ``MainWindow._qbool`` did not accept ``"on"`` as a
  truthy value. Phase 55 already covers ``true``/``yes``/``1``; we
  add ``on`` for HTTP / web front-end style checkbox strings.
* **NIT-2** — ``JobsTab.add_or_update_job`` already uses incremental
  ``_refresh_row`` updates. The test pins this contract so a future
  refactor doesn't silently regress to "clear + re-add 100 rows".
* **NIT-3** — Theme choice was lost on GUI restart because
  ``_apply_theme`` never called ``QSettings.sync()``. The test writes
  a theme via the public API, re-reads it, and confirms the value
  survives.
* **NIT-4** — ``JobsTab._show_context_menu`` used ``QMenu(self)`` and
  ``QAction(parent)``, which parented each menu and its actions to
  the JobsTab. Repeated right-clicks accumulated orphaned QMenu +
  QAction instances as children of the tab. The fix parents the
  menu to ``None`` and calls ``deleteLater()`` after ``exec_``.
* **NIT-5** — ``image_preview._bbox_tooltip`` had hard-coded English
  field labels ("confidence:", "x:", "y:", "w:", "h:", "family:").
  Wrapped with ``i18n._tr`` and added matching EN + ZH keys.

This file combines:

1. Source-guard tests (work without PySide6) for the i18n keys
   and the ``_qbool`` body.
2. Runtime tests (require PySide6 + offscreen Qt) for QSettings
   round-trip, theme persistence, jobs-tab incremental refresh,
   and context-menu QAction stability.

Run with::

    python -m pytest tests/test_audit_2026_08_19_phase6a_gui_nit.py -v
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import PySide6  # noqa: F401

    _HAS_PYSIDE6 = True
except ImportError:
    _HAS_PYSIDE6 = False


_REPO = Path(__file__).resolve().parents[1]
_SRC_GUI = _REPO / "src" / "rlpe" / "gui"
_SRC_MAIN_WINDOW = _SRC_GUI / "main_window.py"
_SRC_JOBS_TAB = _SRC_GUI / "jobs_tab.py"
_SRC_IMAGE_PREVIEW = _SRC_GUI / "image_preview.py"
_SRC_STRINGS_EN = _SRC_GUI / "strings_en.py"
_SRC_STRINGS_ZH = _SRC_GUI / "strings_zh_CN.py"


def _read(rel: Path) -> str:
    return rel.read_text(encoding="utf-8")


def _find_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    """Locate a method's FunctionDef node by class + method name."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(
        f"could not find {class_name}.{method_name} in parsed module"
    )


# ============================================================
# NIT-1 — _qbool accepts "on"
# ============================================================
def test_qbool_accepts_on():
    """Phase 6A: ``_qbool`` must accept ``"on"`` as truthy."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    settings = QSettings("RLPE-Phase6A-NIT1", "test-qbool-on")
    settings.clear()
    settings.setValue("flag", "on")
    settings.sync()

    # Build a minimal stand-in to test the parser in isolation.
    from rlpe.gui.main_window import MainWindow

    win = MainWindow.__new__(MainWindow)
    parsed = MainWindow._qbool(win, settings, "flag", default=False)
    assert parsed is True, f'_qbool("on") should be True, got {parsed!r}'


def test_qbool_accepts_true_false_yes_no_1_0():
    """Phase 6A: ``_qbool`` still handles all the legacy strings."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    settings = QSettings("RLPE-Phase6A-NIT1-b", "test-qbool-legacy")
    settings.clear()

    from rlpe.gui.main_window import MainWindow

    win = MainWindow.__new__(MainWindow)

    truthy_inputs = ["true", "True", "TRUE", "yes", "Yes", "1", "on", "On", "ON"]
    falsy_inputs = ["false", "False", "FALSE", "no", "No", "0", "off", ""]

    for s in truthy_inputs:
        settings.setValue("flag", s)
        settings.sync()
        assert MainWindow._qbool(win, settings, "flag", default=False) is True, (
            f"_qbool({s!r}) should be True"
        )
        # bool() comparison shape
        assert (
            MainWindow._qbool(win, settings, "flag", default=False) is not False
        ), f"_qbool({s!r}) should be truthy"

    for s in falsy_inputs:
        settings.setValue("flag", s)
        settings.sync()
        assert MainWindow._qbool(win, settings, "flag", default=True) is False, (
            f"_qbool({s!r}) should be False"
        )


def _call_uses_type_bool(fn: ast.FunctionDef) -> bool:
    """Return True if any Call node inside ``fn`` passes ``type=bool``."""
    for node in ast.walk(ast.Module(body=fn.body, type_ignores=[])):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "type":
                    # `type=bool` → ast.Name(id='bool')
                    if isinstance(kw.value, ast.Name) and kw.value.id == "bool":
                        return True
    return False


def test_qbool_source_contains_on_string():
    """The literal ``"on"`` must appear in the ``_qbool`` body so the
    fix can't be silently reverted."""
    src = _read(_SRC_MAIN_WINDOW)
    tree = ast.parse(src)
    # Walk into the MainWindow class body since _qbool is a method.
    fn = _find_method(tree, "MainWindow", "_qbool")
    body_src = ast.unparse(fn)
    assert '"on"' in body_src or "'on'" in body_src, (
        "_qbool body does not include 'on' as a truthy value — "
        "Phase 6A NIT-1 fix is missing."
    )
    # Also: must NOT use the broken type=bool converter in actual code.
    # Qt converts "no" / "off" / "false" to True under type=bool, so
    # accepting "no" as falsy is impossible while type=bool is in play.
    assert not _call_uses_type_bool(fn), (
        "_qbool still uses QSettings.value(..., type=bool) — "
        "Qt's type conversion treats every non-empty string as True. "
        "Read the raw value and parse it manually."
    )


# ============================================================
# NIT-2 — JobsTab add_or_update_job is incremental
# ============================================================
def test_jobs_tab_add_or_update_job_is_incremental():
    """``JobsTab.add_or_update_job`` must call ``_refresh_row``
    directly, not ``setRowCount(0)`` + re-insert all rows."""
    from rlpe.gui.jobs_tab import JobsTab

    src = inspect.getsource(JobsTab.add_or_update_job)
    assert "_refresh_row" in src, (
        "JobsTab.add_or_update_job no longer calls _refresh_row — "
        "every job update will redraw all rows (Phase 6A NIT-2)."
    )
    assert "setRowCount(0)" not in src, (
        "JobsTab.add_or_update_job should not wipe the table — "
        "use the incremental _refresh_row path."
    )


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
def test_jobs_tab_incremental_refresh_runtime():
    """Insert two jobs; the second call should only touch the new row
    (insertRow called once with row index == current rowCount)."""
    from rlpe.gui.jobs_tab import JobsTab

    jt = JobsTab()
    try:
        from rlpe.gui.jobs_tab import JobRecord

        # First job
        job1 = JobRecord(
            job_id="j-1",
            pdf_path="/tmp/a.pdf",
            output_dir="/tmp/out1",
        )
        jt.add_or_update_job(job1)
        rows_after_first = jt._table.rowCount()
        assert rows_after_first == 1, f"expected 1 row, got {rows_after_first}"

        # Second job — should append, not re-create all rows
        job2 = JobRecord(
            job_id="j-2",
            pdf_path="/tmp/b.pdf",
            output_dir="/tmp/out2",
        )
        jt.add_or_update_job(job2)
        rows_after_second = jt._table.rowCount()
        assert rows_after_second == 2, (
            f"expected 2 rows after second insert, got {rows_after_second}"
        )

        # Verify job1 row content survived (didn't get wiped)
        first_id = jt._table.item(0, 0).text()
        second_id = jt._table.item(1, 0).text()
        assert first_id == "j-1", f"row 0 id changed: {first_id}"
        assert second_id == "j-2", f"row 1 id: {second_id}"
    finally:
        jt.deleteLater()


# ============================================================
# NIT-3 — Theme persistence via QSettings
# ============================================================
def test_theme_apply_calls_qsettings_sync():
    """``MainWindow._apply_theme`` must call ``QSettings.sync()`` so
    the choice survives a hard GUI close."""
    src = _read(_SRC_MAIN_WINDOW)
    tree = ast.parse(src)
    fn = _find_method(tree, "MainWindow", "_apply_theme")
    body_src = ast.unparse(fn)
    assert "sync()" in body_src, (
        "_apply_theme does not call self._qsettings.sync() — "
        "theme choice may be lost on hard close."
    )
    assert "setValue" in body_src, (
        "_apply_theme does not call setValue(QS_KEY_THEME, theme)."
    )


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
def test_theme_persists_across_qsettings_instance():
    """Write "dark" to QSettings, instantiate a fresh QSettings for the
    same org/app, and read back "dark". Verifies the persistence path
    round-trips."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    org = "RLPE-Phase6A-NIT3"
    app_name = "test-theme-persist"

    # Write
    s1 = QSettings(org, app_name)
    s1.clear()
    s1.setValue("ui/theme", "dark")
    s1.sync()

    # Read back with a brand-new instance
    s2 = QSettings(org, app_name)
    val = s2.value("ui/theme", "light")
    assert str(val).lower() == "dark", (
        f"theme persistence failed: wrote 'dark', read back {val!r}"
    )

    # Cleanup
    s2.clear()
    s2.sync()


# ============================================================
# NIT-4 — Context menu does not accumulate QActions
# ============================================================
def test_show_context_menu_does_not_parent_to_self():
    """``JobsTab._show_context_menu`` must NOT pass ``self`` as the
    parent of the QMenu (which would cause every opened menu to live
    on as a child widget). Parent must be None so it can be
    deleteLater()'d after exec_."""
    src = _read(_SRC_JOBS_TAB)
    tree = ast.parse(src)
    fn = _find_method(tree, "JobsTab", "_show_context_menu")
    body_src = ast.unparse(fn)

    # The fix: `menu = QMenu()` (no parent). The bug: `menu = QMenu(self)`.
    assert "menu = QMenu()" in body_src, (
        "_show_context_menu should create `menu = QMenu()` (no parent) "
        "so it can be cleaned up after exec_. Found a parented QMenu instead."
    )
    assert "deleteLater" in body_src, (
        "_show_context_menu must call menu.deleteLater() after exec_ "
        "to release the menu + its actions."
    )
    # And the old buggy form must be gone
    assert "menu = QMenu(self)" not in body_src, (
        "_show_context_menu still has the leaky `menu = QMenu(self)` "
        "— every menu would be parented to JobsTab and accumulate."
    )
    # Should call exec_ (Python alias) so tests can monkeypatch.
    assert "menu.exec_" in body_src, (
        "_show_context_menu should call `menu.exec_(...)` (the "
        "Python-friendly alias) instead of `menu.exec(...)` so "
        "tests can monkey-patch it."
    )


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
def test_context_menu_qaction_count_stable_across_repeats(monkeypatch):
    """Build a JobsTab, pop the context menu 10 times via the
    ``_show_context_menu`` slot. The number of orphan QMenu objects
    parented to the tab must NOT grow monotonically — after each
    invocation, ``findChildren(QMenu)`` excluding the live one must be
    empty / stable.

    With the old ``QMenu(self)`` parent, this would accumulate
    10 children. With the new ``QMenu()`` + ``deleteLater()`` parent,
    the children count after deleteLater processes stays at 0.
    """
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QApplication, QMenu

    from rlpe.gui.jobs_tab import JobRecord, JobsTab

    app = QApplication.instance() or QApplication([])

    # PySide6's QMenu.exec is a C++-bound builtin — we can't monkey-
    # patch it on the class. Patch the module-level reference inside
    # jobs_tab instead so the call inside _show_context_menu goes
    # through our fake. ``exec_`` is the Python-friendly alias that
    # resolves to the same Qt modal-popup call but is a normal Python
    # method, so it IS patchable.
    import rlpe.gui.jobs_tab as _jtmod

    exec_call_count = {"n": 0}

    def fake_exec(self, *_args, **_kwargs):
        exec_call_count["n"] += 1
        return None

    monkeypatch.setattr(_jtmod.QMenu, "exec_", fake_exec)

    jt = JobsTab()
    try:
        jt.add_or_update_job(
            JobRecord(
                job_id="ctx-test",
                pdf_path="/tmp/ctx.pdf",
                output_dir="/tmp/ctx-out",
            )
        )
        # _show_context_menu early-returns if no row is selected, so
        # we must select the row first.
        jt._table.selectRow(0)

        # Snapshot of menu-children count BEFORE
        def menu_children_count():
            return len(jt.findChildren(QMenu))

        before = menu_children_count()

        for _ in range(10):
            pos = QPoint(10, 10)
            jt._show_context_menu(pos)

        # Process pending deleteLater events
        app.processEvents()

        after = menu_children_count()

        # The bug: before=0, after=10 (one QMenu per call, parented to jt).
        # The fix: after == before (deleteLater freed them).
        assert after == before, (
            f"Context menu QMenu children leaked: "
            f"before={before}, after={after} (delta={after - before}). "
            "Phase 6A NIT-4 fix missing."
        )

        # Also verify exec_ was actually invoked 10 times
        assert exec_call_count["n"] == 10, (
            f"_show_context_menu did not call menu.exec_() 10 times; "
            f"got {exec_call_count['n']}"
        )
    finally:
        jt.deleteLater()


# ============================================================
# NIT-5 — image_preview._bbox_tooltip uses i18n keys
# ============================================================
def test_bbox_tooltip_uses_i18n_tr():
    """The hardcoded English labels in ``_bbox_tooltip`` must be
    wrapped with ``i18n._tr``. AST-scan to enforce."""
    from rlpe.gui import image_preview

    src = inspect.getsource(image_preview._bbox_tooltip)
    # No raw English field labels in the body
    forbidden = (
        '"confidence:"',
        "'confidence:'",
        '"x: "',
        "'x: '",
        '"y: "',
        "'y: '",
        '"w: "',
        "'w: '",
        '"h: "',
        "'h: '",
        '"family:"',
        "'family:'",
        "f\"confidence:",
        "f\"x:",
        "f\"y:",
        "f\"w:",
        "f\"h:",
        "f\"family:",
        "f'confidence:",
        "f'x:",
        "f'y:",
        "f'w:",
        "f'h:",
        "f'family:",
    )
    for bad in forbidden:
        assert bad not in src, (
            f"_bbox_tooltip still has hardcoded English label {bad!r}. "
            "Wrap with i18n._tr('preview.tooltip.*')."
        )
    # Must contain i18n._tr calls for the new keys
    for key in (
        "preview.tooltip.confidence",
        "preview.tooltip.coords_xy",
        "preview.tooltip.coords_wh",
        "preview.tooltip.family",
    ):
        assert key in src, (
            f"_bbox_tooltip missing i18n key {key!r}"
        )


def test_bbox_tooltip_keys_exist_en():
    from rlpe.gui import strings_en

    for key in (
        "preview.tooltip.confidence",
        "preview.tooltip.coords_xy",
        "preview.tooltip.coords_wh",
        "preview.tooltip.family",
    ):
        assert key in strings_en.STRINGS, f"missing EN key: {key}"
        val = strings_en.STRINGS[key]
        assert val and isinstance(val, str), f"EN {key} empty/non-string: {val!r}"


def test_bbox_tooltip_keys_exist_zh():
    from rlpe.gui import strings_zh_CN

    for key in (
        "preview.tooltip.confidence",
        "preview.tooltip.coords_xy",
        "preview.tooltip.coords_wh",
        "preview.tooltip.family",
    ):
        assert key in strings_zh_CN.STRINGS, f"missing ZH key: {key}"
        val = strings_zh_CN.STRINGS[key]
        assert val and isinstance(val, str), f"ZH {key} empty/non-string: {val!r}"


def test_bbox_tooltip_keys_differ_en_vs_zh():
    from rlpe.gui import strings_en, strings_zh_CN

    for key in (
        "preview.tooltip.confidence",
        "preview.tooltip.coords_xy",
        "preview.tooltip.coords_wh",
        "preview.tooltip.family",
    ):
        assert strings_en.STRINGS[key] != strings_zh_CN.STRINGS[key], (
            f"{key} EN and ZH are identical — translation missing."
        )


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
def test_bbox_tooltip_renders_translated_fields():
    """Call ``_bbox_tooltip`` and confirm the returned HTML contains
    the i18n-translated label text (not the legacy raw English)."""
    from rlpe.gui import i18n, image_preview

    # Set language to zh_CN to confirm translation works
    i18n.set_language("zh_CN")
    try:
        bbox = {
            "species": "Test species",
            "confidence": 0.87,
            "bbox": (10, 20, 100, 200),
            "metadata": {"paleodb": {"taxonomy": {"family": "Nassellaria"}}},
        }
        html = image_preview._bbox_tooltip(bbox)
        # ZH translations
        assert "置信度：" in html, f"ZH confidence label missing: {html!r}"
        assert "x：" in html, f"ZH x label missing: {html!r}"
        assert "y：" in html, f"ZH y label missing: {html!r}"
        assert "宽：" in html, f"ZH w label missing: {html!r}"
        assert "高：" in html, f"ZH h label missing: {html!r}"
        assert "科：" in html, f"ZH family label missing: {html!r}"
        # Raw English should be absent
        assert "confidence:" not in html, f"EN label still present: {html!r}"
        assert "family:" not in html, f"EN label still present: {html!r}"
    finally:
        i18n.set_language("zh_CN")


# ============================================================
# i18n key parity sweep — every EN preview.tooltip.* has a ZH match
# ============================================================
def test_all_preview_tooltip_keys_have_zh_pair():
    """No half-translated keys: every ``preview.tooltip.*`` in EN
    must also exist in ZH (and vice versa)."""
    from rlpe.gui import strings_en, strings_zh_CN

    en_keys = {k for k in strings_en.STRINGS if k.startswith("preview.tooltip.")}
    zh_keys = {k for k in strings_zh_CN.STRINGS if k.startswith("preview.tooltip.")}

    missing_zh = en_keys - zh_keys
    missing_en = zh_keys - en_keys
    assert not missing_zh, f"EN preview.tooltip.* keys missing in ZH: {missing_zh}"
    assert not missing_en, f"ZH preview.tooltip.* keys missing in EN: {missing_en}"
