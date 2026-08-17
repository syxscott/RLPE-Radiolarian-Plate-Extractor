"""Phase 57 regression tests — minor frontend audit sweep.

Covers:
  - MINOR-1: \\n literals in strings_en/zh_CN replaced with real \\n
    (the two-character backslash-n sequence previously rendered as
    the literal characters "\\n" instead of a newline).
  - MINOR-2: rlpe.gui.__version__ is now sourced from constants.APP_VERSION
    so the two can never drift apart.
  - MINOR-3: main_window._open_log_file shows a friendly info dialog
    when the log file doesn't exist yet (fresh install) instead of
    popping a yellow FileNotFoundError warning.
  - MINOR-4: the OCR language combo tooltip is registered in the i18n
    registry and translated on language switch.
  - MINOR-5: image_preview mouse click prefers QGraphicsRectItem hits
    over QGraphicsTextItem (label) hits, so overlapping labels can't
    route the click to a different bbox.
  - MINOR-6: batch_dialog probes the output directory with a tempfile
    + os.W_OK check before letting the batch start.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

# Headless Qt bootstrap — must be set before any PySide6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Make ``rlpe`` importable when pytest is invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication for GUI tests."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    yield app


# ---------------------------------------------------------------------------
# MINOR-1 — \n literals
# ---------------------------------------------------------------------------


def test_minor1_no_escaped_newlines_in_strings() -> None:
    """Both string files must not contain the two-char sequence ``\\n``
    inside a dict value — that sequence renders as the literal
    backslash-n rather than a newline.
    """
    for fname in ("strings_en.py", "strings_zh_CN.py"):
        path = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / fname
        text = path.read_text(encoding="utf-8")
        # Look for the two-character sequence "\\\\n" inside a quoted
        # string literal. The regex skips the docstring header.
        body = text.split('"""', 2)[-1]
        bad = re.findall(r"\\\\n", body)
        assert not bad, (
            f"{fname} still contains {len(bad)} escaped \\\\n sequence(s); "
            "use a real newline character so .format() / Qt widgets "
            "render them correctly."
        )


# ---------------------------------------------------------------------------
# MINOR-2 — __version__ synced with APP_VERSION
# ---------------------------------------------------------------------------


def test_minor2_gui_version_equals_app_version() -> None:
    """rlpe.gui.__version__ must equal constants.APP_VERSION."""
    # Force fresh imports.
    for mod in ("rlpe.gui", "rlpe.gui.constants"):
        if mod in sys.modules:
            del sys.modules[mod]
    import rlpe.gui
    from rlpe.gui.constants import APP_VERSION

    assert rlpe.gui.__version__ == APP_VERSION, (
        f"gui.__version__={rlpe.gui.__version__!r} but "
        f"constants.APP_VERSION={APP_VERSION!r} — keep them in sync."
    )


# ---------------------------------------------------------------------------
# MINOR-3 — log-file pre-check
# ---------------------------------------------------------------------------


def test_minor3_open_log_file_checks_existence(qapp, tmp_path, monkeypatch) -> None:
    """_open_log_file must show an info dialog (not a yellow warning)
    when the log file doesn't exist yet.
    """
    import importlib

    main_window_mod = importlib.import_module("rlpe.gui.main_window")
    MainWindow = main_window_mod.MainWindow

    # Don't actually open a MainWindow — just instantiate _open_log_file
    # logic by patching the QMessageBox to record what was shown.
    from PySide6.QtWidgets import QMessageBox

    shown: list[tuple[str, str]] = []

    def _fake_information(parent, title, body, *args, **kwargs):
        shown.append(("info", title, body))
        return QMessageBox.Ok

    def _fake_warning(parent, title, body, *args, **kwargs):
        shown.append(("warn", title, body))
        return QMessageBox.Ok

    monkeypatch.setattr(main_window_mod.QMessageBox, "information", _fake_information)
    monkeypatch.setattr(main_window_mod.QMessageBox, "warning", _fake_warning)

    # Patch LOG_FILE_NAME to point at a file that doesn't exist.
    from rlpe.gui import utils as utils_mod

    monkeypatch.setattr(utils_mod, "LOG_FILE_NAME", "nonexistent_probe_log.log")
    monkeypatch.setattr(
        main_window_mod.Path,
        "home",
        classmethod(lambda cls: tmp_path),
    )

    win = MainWindow.__new__(MainWindow)  # bypass __init__
    win._open_log_file()

    assert shown, "Expected QMessageBox.information to fire for missing log"
    kind, title, body = shown[0]
    assert kind == "info", (
        f"Expected QMessageBox.information (info dialog) for fresh "
        f"install, got kind={kind!r} (a warning would mean the "
        "pre-check is missing)."
    )
    assert "nonexistent_probe_log.log" in body or str(tmp_path) in body, (
        f"Dialog body should mention the expected log path, got {body!r}"
    )


# ---------------------------------------------------------------------------
# MINOR-4 — OCR-lang tooltip registered
# ---------------------------------------------------------------------------


def test_minor4_ocr_lang_tooltip_translates(qapp) -> None:
    """The OCR language combo tooltip must be a registered i18n key
    in both string files. We check the keys exist; the actual
    language switch behaviour is covered by test_phase56.
    """
    for fname in ("strings_en.py", "strings_zh_CN.py"):
        path = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / fname
        text = path.read_text(encoding="utf-8")
        assert "runtab.ocr_lang.tooltip" in text, (
            f"{fname} is missing runtab.ocr_lang.tooltip — the OCR "
            "lang combo's hardcoded English tooltip would not translate."
        )


# ---------------------------------------------------------------------------
# MINOR-5 — bbox click prefers rect over label
# ---------------------------------------------------------------------------


def test_minor5_bbox_click_prefers_rect(qapp) -> None:
    """Phase 55 audit F-8 — image_preview.mousePressEvent must check
    QGraphicsRectItem hits BEFORE QGraphicsTextItem hits, so an
    overlapping label can't redirect the click to a different bbox.
    """
    src = (
        Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / "image_preview.py"
    ).read_text(encoding="utf-8")
    # Find each ``for item in items:`` loop's start position, then
    # verify the FIRST one contains QGraphicsRectItem and the SECOND
    # contains QGraphicsTextItem. If only one loop exists, the F-8
    # fix is missing.
    loops = [m.start() for m in re.finditer(r"for item in items:", src)]
    assert len(loops) >= 2, (
        "image_preview click handler must have at least two "
        "``for item in items:`` loops (rect-first, then label "
        "fallback). Phase 55 F-8 fix missing — only one pass found."
    )
    # Slice between consecutive loop starts and inspect each block.
    first_block = src[loops[0] : loops[1]]
    second_block = src[loops[1] : loops[1] + 500]  # enough to capture label type check
    assert "QGraphicsRectItem" in first_block, (
        "First loop should test QGraphicsRectItem (rect-first preference)."
    )
    assert "QGraphicsTextItem" in second_block, (
        "Second loop should test QGraphicsTextItem (label fallback)."
    )


# ---------------------------------------------------------------------------
# MINOR-6 — batch output-dir writability probe
# ---------------------------------------------------------------------------


def test_minor6_batch_dialog_probes_writability(qapp, tmp_path) -> None:
    """batch_dialog must probe writability before accepting an output
    directory. audit 2026-07-31: the probe moved to the Qt-free
    outdir_probe module (unit-testable without PySide6); the dialog
    delegates to it.
    """
    probe_src = (
        Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / "outdir_probe.py"
    ).read_text(encoding="utf-8")
    assert "def probe_output_dir_writable" in probe_src, (
        "outdir_probe.py missing the writability probe."
    )
    assert "os.open" in probe_src, "probe must actually create a temp file"
    dialog_src = (
        Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / "batch_dialog.py"
    ).read_text(encoding="utf-8")
    assert "probe_output_dir_writable" in dialog_src, (
        "batch_dialog.py must delegate to probe_output_dir_writable"
    )
    # audit 2026-07-31: the W_OK fallback check lives in the probe
    # module now.
    assert "W_OK" in probe_src, "outdir_probe.py missing the os.access(W_OK) fallback check."
    assert "batch.outdir.not_writable" in dialog_src, (
        "batch_dialog.py doesn't reference the new 'not writable' i18n key."
    )
