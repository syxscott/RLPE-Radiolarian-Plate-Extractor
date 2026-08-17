"""Phase 48 — extend friendly names + i18n to file/export dialogs.

Phase 47 wired up the 4 friendly-name combos (theme / LLM / OCR
backend / M3 prompt lang). Phase 48 finishes the surface by
i18n-wrapping the remaining user-visible strings that were still
bare English:

  * Run tab: QFileDialog titles + QMessageBox "Output / No out dir"
  * Batch dialog: QFileDialog titles + count label
  * Jobs tab: QFileDialog export titles + QMessageBox export dialogs
    + summary "X jobs · running Y · done Z · failed W" label
  * Results tab: QFileDialog export titles (xlsx / json / csv / dwca)
    + status messages
  * Image preview: "(no image)" / "(missing)" / "(failed to load)"
  * Run tab OCR backend combo: friendly name "PaddleOCR (推荐)" /
    "EasyOCR (多语言)" with ISO code in userData

Tests pin the new i18n keys exist, the strings are not bare
English in source, and the widgets return the right data on
language switch.
"""

from __future__ import annotations

import inspect
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
# 1. New i18n keys exist in both EN and ZH
# ============================================================
@pytest.mark.parametrize(
    "key",
    [
        # Run tab
        "runtab.browse.title",
        "runtab.out.choose.title",
        "runtab.out.no_outdir.title",
        "runtab.out.no_outdir.body",
        # Batch dialog
        "batch.add.title",
        "batch.add_dir.title",
        "batch.outdir.title",
        # Jobs tab exports
        "jobstab.export.xlsx_title",
        "jobstab.export.json_title",
        "jobstab.summary.count",
        "jobstab.summary.count_label",
        # Results tab exports
        "restab.export.xlsx_title",
        "restab.export.json_title",
        "restab.export.csv_title",
        "restab.export.dwca_title",
    ],
)
def test_new_i18n_keys_exist_en(key):
    """Phase 48: every new i18n key must be in strings_en.py."""
    from rlpe.gui import strings_en

    assert key in strings_en.STRINGS, f"missing EN key: {key}"
    val = strings_en.STRINGS[key]
    assert val and isinstance(val, str), f"EN {key} is empty/non-string: {val!r}"


@pytest.mark.parametrize(
    "key",
    [
        "runtab.browse.title",
        "runtab.out.choose.title",
        "runtab.out.no_outdir.title",
        "runtab.out.no_outdir.body",
        "batch.add.title",
        "batch.add_dir.title",
        "batch.outdir.title",
        "jobstab.export.xlsx_title",
        "jobstab.export.json_title",
        "jobstab.summary.count",
        "jobstab.summary.count_label",
        "restab.export.xlsx_title",
        "restab.export.json_title",
        "restab.export.csv_title",
        "restab.export.dwca_title",
    ],
)
def test_new_i18n_keys_exist_zh(key):
    from rlpe.gui import strings_zh_CN

    assert key in strings_zh_CN.STRINGS, f"missing ZH key: {key}"
    val = strings_zh_CN.STRINGS[key]
    assert val and isinstance(val, str), f"ZH {key} is empty/non-string: {val!r}"


def test_new_keys_differ_between_en_and_zh():
    """Every new key should have a Chinese translation that is
    materially different from the English one."""
    from rlpe.gui import strings_en, strings_zh_CN

    keys = [
        "runtab.browse.title",
        "runtab.out.choose.title",
        "runtab.out.no_outdir.body",
        "batch.add.title",
        "batch.add_dir.title",
        "batch.outdir.title",
        "jobstab.summary.count",
    ]
    for k in keys:
        assert strings_en.STRINGS[k] != strings_zh_CN.STRINGS[k], (
            f"{k} EN and ZH are identical: {strings_en.STRINGS[k]!r}"
        )


# ============================================================
# 2. Source guards — no bare English in dialog calls
# ============================================================
def test_run_tab_has_no_bare_english_qfiledialog_title():
    """Phase 48: every QFileDialog title in run_tab must use i18n._tr."""
    from rlpe.gui.run_tab import RunTab

    src = inspect.getsource(RunTab)
    # The known bare strings we just removed:
    assert '"Choose a radiolarian paper PDF"' not in src, (
        "run_tab still has bare English 'Choose a radiolarian paper PDF'"
    )
    assert '"Choose an output directory"' not in src, (
        "run_tab still has bare English 'Choose an output directory'"
    )


def test_run_tab_has_no_bare_english_qmessagebox():
    from rlpe.gui.run_tab import RunTab

    src = inspect.getsource(RunTab)
    assert '"Output"' not in src or "runtab.out.no_outdir.title" in src, (
        "run_tab still has bare English QMessageBox 'Output' title"
    )
    assert '"No output directory set yet."' not in src, (
        "run_tab still has bare English 'No output directory set yet.'"
    )


def test_batch_dialog_has_no_bare_english_qfiledialog_title():
    from rlpe.gui.batch_dialog import BatchDialog

    src = inspect.getsource(BatchDialog)
    assert '"Add PDFs"' not in src, (
        "batch_dialog still has bare English 'Add PDFs' QFileDialog title"
    )
    assert '"Choose a directory of PDFs"' not in src, (
        "batch_dialog still has bare English 'Choose a directory of PDFs'"
    )
    assert '"Choose batch output directory"' not in src, (
        "batch_dialog still has bare English 'Choose batch output directory'"
    )


def test_jobs_tab_has_no_bare_english_export_dialog():
    from rlpe.gui.jobs_tab import JobsTab

    src = inspect.getsource(JobsTab)
    # The QFileDialog titles
    assert 'QFileDialog.getSaveFileName(self, "Export xlsx"' not in src, (
        "jobs_tab still has bare English 'Export xlsx' title"
    )
    assert 'QFileDialog.getSaveFileName(self, "Export JSON"' not in src, (
        "jobs_tab still has bare English 'Export JSON' title"
    )
    # The QMessageBox warning title
    assert 'QMessageBox.warning(self, "Export failed"' not in src, (
        "jobs_tab still has bare English 'Export failed' title"
    )


def test_results_tab_has_no_bare_english_export_dialog():
    from rlpe.gui.results_tab import ResultsTab

    src = inspect.getsource(ResultsTab)
    assert 'QFileDialog.getSaveFileName(self, "Export xlsx"' not in src
    assert 'QFileDialog.getSaveFileName(self, "Export JSON"' not in src
    assert 'QFileDialog.getSaveFileName(self, "Export CSV"' not in src
    assert 'QFileDialog.getSaveFileName(self, "Export DwCA"' not in src


def test_jobs_tab_summary_uses_i18n():
    """Phase 48: jobs_tab._update_summary must use i18n._tr, not
    a hard-coded English f-string."""
    from rlpe.gui.jobs_tab import JobsTab

    src = inspect.getsource(JobsTab._update_summary)
    assert "i18n._tr" in src, "jobs_tab._update_summary still uses hard-coded English f-string"
    assert "{total} jobs" not in src, "_update_summary still has hard-coded English '{total} jobs'"


def test_image_preview_no_bare_english_strings():
    """Phase 48: image_preview labels must use i18n._tr."""
    from rlpe.gui.image_preview import ImagePreviewWidget

    src = inspect.getsource(ImagePreviewWidget)
    assert '"(no image)"' not in src, "image_preview still has bare English '(no image)'"
    assert '"(missing) {path.name}"' not in src and 'f"(missing) {path.name}"' not in src, (
        "image_preview still has bare English '(missing) {path.name}'"
    )
    assert 'f"(failed to load) {path.name}"' not in src, (
        "image_preview still has bare English '(failed to load) {path.name}'"
    )


# ============================================================
# 3. Run tab OCR backend combo (friendly name + ISO code in userData)
# ============================================================
def test_run_tab_ocr_backend_uses_friendly_names():
    """Phase 48: OCR backend combo shows friendly names not raw codes."""
    from rlpe.gui.run_tab import RunTab

    rt = RunTab({})
    for cb in rt.findChildren(QComboBox):
        if cb.count() == 0:
            continue
        if cb.itemData(0) in ("paddleocr", "easyocr"):
            for i in range(cb.count()):
                assert cb.itemText(i) != cb.itemData(i), (
                    f"OCR backend combo item {i} has raw code as text: "
                    f"text={cb.itemText(i)!r} data={cb.itemData(i)!r}"
                )
            return
    pytest.fail("Could not find OCR backend QComboBox in Run tab")


def test_run_tab_collect_settings_returns_ocr_backend_iso_code():
    """Phase 48: collect_settings must return the ISO code
    ('paddleocr' / 'easyocr'), not the friendly name."""
    from rlpe.gui.run_tab import RunTab

    rt = RunTab({})
    settings = rt.collect_settings()
    assert settings["ocr_backend"] in {"paddleocr", "easyocr"}, (
        f"ocr_backend should be ISO code, got {settings['ocr_backend']!r}"
    )


def test_run_tab_apply_settings_restores_ocr_backend_by_iso_code():
    """Phase 48: apply_settings should be able to restore the
    OCR backend by ISO code (e.g. 'easyocr')."""
    from rlpe.gui.run_tab import RunTab

    rt = RunTab({})
    rt.apply_settings({"ocr_backend": "easyocr"})
    settings = rt.collect_settings()
    assert settings["ocr_backend"] == "easyocr", (
        f"apply_settings should restore easyocr, got {settings['ocr_backend']!r}"
    )


# ============================================================
# 4. image_preview uses i18n keys
# ============================================================
def test_image_preview_uses_i18n_no_image_key():
    from rlpe.gui import i18n
    from rlpe.gui.image_preview import ImagePreviewWidget

    widget = ImagePreviewWidget()
    i18n.set_language("en")
    widget.set_image(None)
    en_text = widget._path_label.text()
    i18n.set_language("zh_CN")
    widget.clear()
    zh_text = widget._path_label.text()
    assert en_text != zh_text, (
        f"image_preview no-image label did not switch: en={en_text!r} zh={zh_text!r}"
    )
    # zh should contain Chinese characters
    assert "无" in zh_text or "图" in zh_text, (
        f"zh image_preview text should be Chinese: {zh_text!r}"
    )


def test_image_preview_uses_i18n_missing_key():
    from rlpe.gui import i18n
    from rlpe.gui.image_preview import ImagePreviewWidget

    widget = ImagePreviewWidget()
    i18n.set_language("zh_CN")
    widget.set_image("/tmp/does_not_exist_xyz.png")
    text = widget._path_label.text()
    # ZH missing label should be Chinese (not "(missing) ...")
    assert "(missing)" not in text, f"image_preview still shows bare English '(missing)': {text!r}"


# ============================================================
# 5. jobs_tab summary uses i18n format string
# ============================================================
def test_jobs_tab_summary_uses_i18n_format():
    from rlpe.gui import i18n
    from rlpe.gui.jobs_tab import STATUS_QUEUED, JobRecord, JobsTab

    jt = JobsTab()
    # Add a fake job
    job = JobRecord(
        job_id="test-job-1",
        pdf_path=Path("/tmp/x.pdf"),
        output_dir="/tmp/out",
        status=STATUS_QUEUED,
    )
    jt._jobs[job.job_id] = job
    jt._update_summary()
    # Both labels should reflect 1 job (using i18n format)
    i18n.set_language("zh_CN")
    jt._update_summary()
    zh_count = jt._count_label.text()
    i18n.set_language("en")
    jt._update_summary()
    en_count = jt._count_label.text()
    assert en_count != zh_count, f"jobs summary did not switch: en={en_count!r} zh={zh_count!r}"
    assert "1" in en_count and "1" in zh_count, (
        f"both should show '1' job: en={en_count!r} zh={zh_count!r}"
    )


# ============================================================
# 6. Source guards — ensure friendly-name pattern still works
# ============================================================
def test_run_tab_ocr_backend_has_min_height_32():
    """Phase 48: OCR backend combo must keep the Phase 35 min_height
    invariant (32px) just like the other friendly combos."""
    from PySide6.QtWidgets import QSizePolicy

    from rlpe.gui.run_tab import RunTab

    rt = RunTab({})
    for cb in rt.findChildren(QComboBox):
        if cb.itemData(0) in ("paddleocr", "easyocr"):
            assert (
                cb.minimumHeight() >= 32 or cb.sizePolicy().verticalPolicy() == QSizePolicy.Fixed
            ), f"OCR backend combo min_height={cb.minimumHeight()}; should be >= 32"
            return
    pytest.fail("OCR backend combo not found")
