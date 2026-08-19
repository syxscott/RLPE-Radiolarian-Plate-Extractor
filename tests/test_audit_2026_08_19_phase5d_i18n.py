"""Phase 5D (2026-08-19): GUI i18n completeness — main window + tabs + dialogs.

The multi-agent audit on 2026-08-19 surfaced several GUI
i18n-coverage bugs:

* **M-17** — ``MainWindow._build_ui`` set tab labels to hard-coded
  English strings ("▶  Run", "📋  Jobs") via ``addTab``. The
  ``_refresh_texts`` method DOES re-translate them on language
  switch, but the initial paint shows the English placeholder
  before the listener fires. Verify the listener is registered and
  re-texts the tabs.

* **M-18** — Several progress / status labels were hard-coded
  English strings:
    - ``run_tab._on_progress`` used ``message or "Working…"`` as a
      hard-coded fallback (now uses ``i18n._tr("runtab.progress.working")``)

* **M-22** — QFileDialog titles already use ``i18n._tr`` (Phase 48);
  this test re-verifies completeness across all four tabs.

* **M-23** — QMessageBox titles/text already use ``i18n._tr`` (Phase 48);
  this test re-verifies completeness across all four tabs.

* **NIT** — i18n key parity: every EN key must exist in ZH (and vice
  versa). ``runtab.status.cancelled`` was missing in both files; the
  typo ``restab.export.xlsx.title`` (in main_window.py:973) used a
  non-existent key instead of ``restab.export.xlsx_title``.

This file is mostly source-guard + key-parity tests because the
runtime i18n behaviour is covered by test_phase48_dialog_i18n.py
and the GUI smoke tests in test_audit_2026_08_19_phase1f_gui.py.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _reset_language():
    from rlpe.gui import i18n

    i18n.set_language("zh_CN")
    yield
    i18n.set_language("zh_CN")


# ============================================================
# 1. M-17 tab labels are i18n-wrapped via _refresh_texts
# ============================================================
def test_main_window_tabs_have_object_names():
    """Each QTabWidget tab should be retexted via ``_refresh_texts``.

    We can't easily inspect ``MainWindow._build_ui`` without spinning
    up a QApplication + a settings file; instead we check the
    method source contains the expected tab-key array AND the
    ``i18n._tr`` call. The actual keys are pinned in test #2 below.
    """
    from rlpe.gui.main_window import MainWindow

    src = inspect.getsource(MainWindow)
    assert '("tab.run", "tab.jobs", "tab.results", "tab.settings")' in src, (
        "MainWindow._refresh_texts missing the tab-key tuple — "
        "tab labels won't translate on language switch."
    )
    assert "i18n._tr(key)" in src, (
        "MainWindow._refresh_texts missing i18n._tr(key) call — "
        "tab labels won't translate on language switch."
    )


def test_main_window_tab_keys_exist_en():
    """Each ``tab.*`` key referenced by ``_refresh_texts`` must exist
    in strings_en.py."""
    from rlpe.gui import strings_en

    for key in ("tab.run", "tab.jobs", "tab.results", "tab.settings"):
        assert key in strings_en.STRINGS, f"missing EN tab key: {key}"
        val = strings_en.STRINGS[key]
        assert val and isinstance(val, str), f"EN {key} is empty/non-string: {val!r}"


def test_main_window_tab_keys_exist_zh():
    from rlpe.gui import strings_zh_CN

    for key in ("tab.run", "tab.jobs", "tab.results", "tab.settings"):
        assert key in strings_zh_CN.STRINGS, f"missing ZH tab key: {key}"
        val = strings_zh_CN.STRINGS[key]
        assert val and isinstance(val, str), f"ZH {key} is empty/non-string: {val!r}"


def test_main_window_tab_keys_differ_en_vs_zh():
    from rlpe.gui import strings_en, strings_zh_CN

    for key in ("tab.run", "tab.jobs", "tab.results", "tab.settings"):
        assert strings_en.STRINGS[key] != strings_zh_CN.STRINGS[key], (
            f"{key} EN and ZH are identical: {strings_en.STRINGS[key]!r}"
        )


def test_main_window_registers_i18n_listener():
    """``_build_ui`` should register ``_on_language_changed`` as an
    i18n listener so language switches refresh tab / window titles."""
    from rlpe.gui.main_window import MainWindow

    # The listener is registered inside ``_build_ui`` (which
    # ``__init__`` delegates to), not in ``__init__`` directly.
    src = inspect.getsource(MainWindow._build_ui)
    assert "i18n.add_listener" in src, (
        "MainWindow._build_ui does not register an i18n listener — "
        "tab labels and window title won't refresh on language switch."
    )
    assert "_refresh_texts" in src, (
        "MainWindow._build_ui does not call _refresh_texts — "
        "initial tab labels may be wrong."
    )


# ============================================================
# 2. M-18 progress labels use i18n keys (not hardcoded English)
# ============================================================
def test_run_tab_progress_uses_i18n_key():
    """``RunTab._on_progress`` must not have a hard-coded English
    fallback string. The "Working…" fallback was the M-18 bug."""
    from rlpe.gui.run_tab import RunTab

    src = inspect.getsource(RunTab._on_progress)
    # The M-18 fix: replace `message or "Working…"` with
    # `message or i18n._tr("runtab.progress.working")`.
    assert '"Working…"' not in src and "'Working…'" not in src, (
        "RunTab._on_progress still has hard-coded English 'Working…' "
        "fallback — must use i18n._tr('runtab.progress.working')."
    )
    assert "i18n._tr" in src, (
        "RunTab._on_progress does not use i18n._tr — progress "
        "fallback string is not localisable."
    )


def test_progress_keys_exist_en():
    from rlpe.gui import strings_en

    for key in (
        "runtab.progress.idle",
        "runtab.progress.working",
        "runtab.progress.starting",
        "runtab.progress.init",
        "runtab.progress.done",
    ):
        assert key in strings_en.STRINGS, f"missing EN progress key: {key}"


def test_progress_keys_exist_zh():
    from rlpe.gui import strings_zh_CN

    for key in (
        "runtab.progress.idle",
        "runtab.progress.working",
        "runtab.progress.starting",
        "runtab.progress.init",
        "runtab.progress.done",
    ):
        assert key in strings_zh_CN.STRINGS, f"missing ZH progress key: {key}"


def test_run_tab_progress_live_progress_message_no_hardcoded_english():
    """``_show_live_progress`` and the live label update paths must
    never hard-code English. Verify by reading the full source."""
    from rlpe.gui.run_tab import RunTab

    src = inspect.getsource(RunTab)
    # Hard-coded "Working…" was the M-18 bug. Catch any other
    # common English fallbacks ("Loading", "Processing") too.
    bad_substrings = ('"Working…"', "'Working…'", '"Loading…"', "'Loading…'", '"Processing…"', "'Processing…'")
    for bad in bad_substrings:
        assert bad not in src, (
            f"RunTab still has hard-coded progress string {bad!r}. "
            "Use i18n._tr('runtab.progress.working') or similar key."
        )


# ============================================================
# 3. M-22 QFileDialog titles use i18n keys
# ============================================================
def _qfiledialog_titles_in(module_path: str) -> list[tuple[int, str]]:
    """Walk the AST and return (line_no, title_str) for every
    ``QFileDialog.{getOpenFileName,getOpenFileNames,getSaveFileName,
    getExistingDirectory}`` call whose title argument is a literal
    string. Returns [] if the title is an ``i18n._tr(...)`` call.
    """
    src = Path(module_path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # func looks like QFileDialog.getOpenFileName
        attr_name = None
        if isinstance(func, ast.Attribute) and func.attr in {
            "getOpenFileName",
            "getOpenFileNames",
            "getSaveFileName",
            "getExistingDirectory",
        }:
            attr_name = func.attr
        if attr_name is None:
            continue
        # The title is typically the SECOND positional arg
        # (parent is first); we accept any positional >=1.
        if len(node.args) >= 2:
            title_arg = node.args[1]
            # Accept literal strings; ignore i18n._tr('...') (Call).
            if isinstance(title_arg, ast.Constant) and isinstance(title_arg.value, str):
                out.append((node.lineno, title_arg.value))
    return out


GUI_PY_FILES = [
    "main_window.py",
    "run_tab.py",
    "jobs_tab.py",
    "results_tab.py",
    "settings_tab.py",
    "batch_dialog.py",
]


@pytest.mark.parametrize("gui_file", GUI_PY_FILES)
def test_qfiledialog_titles_use_i18n(gui_file):
    """Every QFileDialog title in the GUI must be an i18n key
    (i18n._tr(...)) or a Python expression that resolves through
    i18n — never a bare English literal."""
    src_dir = Path(__file__).resolve().parents[1] / "src/rlpe/gui"
    module_path = src_dir / gui_file
    bare = _qfiledialog_titles_in(str(module_path))
    # We accept filter strings ("Excel files (*.xlsx)") as bare — they
    # are file-type descriptors, not user-facing labels.
    for lineno, title in bare:
        assert title.startswith(("PDF files", "Excel files", "JSON files", "CSV files", "Zip files", "All files")), (
            f"{gui_file}:{lineno} QFileDialog has bare title {title!r} — "
            "wrap with i18n._tr('...') for language switch."
        )


# ============================================================
# 4. M-23 QMessageBox text uses i18n keys
# ============================================================
def _qmessagebox_titles_in(module_path: str) -> list[tuple[int, str, str]]:
    """Walk the AST and return (line_no, kind, text_str) for every
    ``QMessageBox.{information,warning,critical,question}`` call whose
    title OR text argument is a literal string. Returns [] if the
    argument is an ``i18n._tr(...)`` call.
    """
    src = Path(module_path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        kind = None
        if isinstance(func, ast.Attribute) and func.attr in {
            "information",
            "warning",
            "critical",
            "question",
            "about",
        }:
            kind = func.attr
        if kind is None:
            continue
        # Title is typically arg index 1, text is arg index 2.
        for idx in (1, 2):
            if len(node.args) > idx:
                arg = node.args[idx]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.append((node.lineno, kind, arg.value))
    return out


@pytest.mark.parametrize("gui_file", GUI_PY_FILES)
def test_qmessagebox_text_uses_i18n(gui_file):
    """Every QMessageBox title/text in the GUI must be an i18n key
    or non-English literal. Acceptable non-i18n literals: app name
    constants, error type names that get formatted into an existing
    i18n template (already i18n-wrapped via .format(error=...))."""
    src_dir = Path(__file__).resolve().parents[1] / "src/rlpe/gui"
    module_path = src_dir / gui_file
    bare = _qmessagebox_titles_in(str(module_path))
    # The M-23 NIT — Phase 48 already wrapped the visible titles.
    # We just want to be sure no NEW bare English strings have crept in.
    for lineno, kind, text in bare:
        # Accept error detail lines that are clearly runtime exception
        # traces (Type:Value) — these get appended to an i18n
        # template via string concat.
        if any(
            text.startswith(prefix)
            for prefix in (
                "f\"{type",
                "f'{type",
                "{type",
                "f\"{error",
                "f'{error",
            )
        ):
            continue
        # Accept technical context that's appended to an i18n body.
        if text in ("error",):
            continue
        # Otherwise flag it.
        assert False, (
            f"{gui_file}:{lineno} QMessageBox.{kind} has bare text {text!r} — "
            "wrap with i18n._tr('...') for language switch."
        )


# ============================================================
# 5. NIT — i18n key parity: every EN key exists in ZH (and vice versa)
# ============================================================
def test_all_en_keys_exist_in_zh():
    """For every key in ``strings_en.STRINGS`` there must be a
    matching key in ``strings_zh_CN.STRINGS``. The i18n._tr fallback
    would otherwise display English in Chinese mode."""
    from rlpe.gui import strings_en, strings_zh_CN

    missing = set(strings_en.STRINGS) - set(strings_zh_CN.STRINGS)
    assert not missing, f"keys in EN but not ZH: {sorted(missing)}"


def test_all_zh_keys_exist_in_en():
    """Inverse parity — ZH keys must exist in EN. The reverse case
    matters because a stale ZH entry referencing a removed EN key
    would render as the raw sentinel `⟦key⟧`."""
    from rlpe.gui import strings_en, strings_zh_CN

    missing = set(strings_zh_CN.STRINGS) - set(strings_en.STRINGS)
    assert not missing, f"keys in ZH but not EN: {sorted(missing)}"


def test_runtab_status_cancelled_key_exists_in_both():
    """Phase 5D: NIT fix — ``runtab.status.cancelled`` was missing
    from both EN and ZH. ``RunTab._on_failed`` uses it twice (the
    user-cancellation branch and the thread-done fallback)."""
    from rlpe.gui import strings_en, strings_zh_CN

    assert "runtab.status.cancelled" in strings_en.STRINGS, (
        "EN strings missing 'runtab.status.cancelled' — "
        "RunTab._on_failed shows ⟦runtab.status.cancelled⟧ in EN mode."
    )
    assert "runtab.status.cancelled" in strings_zh_CN.STRINGS, (
        "ZH strings missing 'runtab.status.cancelled' — "
        "RunTab._on_failed shows ⟦runtab.status.cancelled⟧ in ZH mode."
    )
    assert strings_en.STRINGS["runtab.status.cancelled"] != strings_zh_CN.STRINGS["runtab.status.cancelled"], (
        "EN and ZH translations of 'runtab.status.cancelled' are identical"
    )


def test_restab_export_xlsx_title_key_typo_fixed():
    """Phase 5D: NIT fix — ``MainWindow._export_batch_xlsx`` used
    the wrong key ``restab.export.xlsx.title`` (dots) instead of the
    canonical ``restab.export.xlsx_title`` (underscore). The dot
    version never existed; this produced the ⟦sentinel⟧ in the
    Save As dialog title."""
    from rlpe.gui.main_window import MainWindow

    src = inspect.getsource(MainWindow._export_batch_xlsx)
    assert 'restab.export.xlsx.title' not in src, (
        "MainWindow._export_batch_xlsx still references the typo "
        "'restab.export.xlsx.title' (dots). The correct key is "
        "'restab.export.xlsx_title' (underscore)."
    )
    assert 'restab.export.xlsx_title' in src, (
        "MainWindow._export_batch_xlsx does not use the canonical "
        "'restab.export.xlsx_title' key."
    )


# ============================================================
# 6. Every i18n._tr(key) call resolves to a defined key
# ============================================================
def test_all_i18n_tr_keys_are_defined():
    """Walk the GUI source, extract every ``i18n._tr('...')`` /
    ``i18n._tr("...")`` literal key, and assert each one exists
    in both EN and ZH dicts. This is a static guarantee that
    nothing renders `⟦key⟧` at runtime."""
    import glob

    import re

    from rlpe.gui import strings_en, strings_zh_CN

    en_keys = set(strings_en.STRINGS)
    zh_keys = set(strings_zh_CN.STRINGS)

    src_dir = Path(__file__).resolve().parents[1] / "src/rlpe/gui"
    referenced_keys: set[str] = set()
    for path in glob.glob(str(src_dir / "*.py")):
        text = Path(path).read_text(encoding="utf-8")
        for m in re.finditer(r"""i18n\._tr\(\s*["']([^"']+)["']""", text):
            referenced_keys.add(m.group(1))

    missing_en = referenced_keys - en_keys
    missing_zh = referenced_keys - zh_keys
    assert not missing_en, f"i18n._tr references undefined EN keys: {sorted(missing_en)}"
    assert not missing_zh, f"i18n._tr references undefined ZH keys: {sorted(missing_zh)}"


# ============================================================
# 7. Regression — make sure Phase 48 dialog tests still pass
# ============================================================
def test_phase48_keys_present_for_regression():
    """Spot-check that all the Phase 48 dialog keys still exist.
    If this test fails, the Phase 5D edits regressed Phase 48."""
    from rlpe.gui import strings_en, strings_zh_CN

    phase48_keys = [
        "runtab.browse.title",
        "runtab.out.choose.title",
        "runtab.out.no_outdir.title",
        "runtab.out.no_outdir.body",
        "batch.add.title",
        "batch.add_dir.title",
        "batch.outdir.title",
        "jobstab.export.xlsx_title",
        "jobstab.export.json_title",
        "restab.export.xlsx_title",
        "restab.export.json_title",
        "restab.export.csv_title",
        "restab.export.dwca_title",
    ]
    for key in phase48_keys:
        assert key in strings_en.STRINGS, f"regression: EN key {key} missing"
        assert key in strings_zh_CN.STRINGS, f"regression: ZH key {key} missing"