"""Phase F-1 (audit 2026-08-20) — Settings-tab field validators.

The 2026-08-20 frontend audit found two MAJOR bugs in
``src/rlpe/gui/settings_tab.py``:

  * M-1 / M-18 — YOLO model path text field accepted any non-empty
    string. A user could save ``/tmp/foo.abc`` (or a path that
    doesn't exist) and later trigger a ``torch.load()`` on an
    attacker-controlled pickle (RCE risk).
  * M-1 — URL/text validators (GROBID, PBDB) were installed on the
    QLineEdit but ``_save()`` never called ``hasAcceptableInput()``,
    so users could save ``"not-a-url"`` and it stuck.

This file pins the fixes:

  * ``test_yolo_validates_existing_pt_file``
  * ``test_yolo_validates_pth_and_onnx_and_weights``
  * ``test_yolo_rejects_nonexistent_path``
  * ``test_yolo_rejects_unsupported_extension``
  * ``test_yolo_empty_path_allowed``
  * ``test_save_refuses_invalid_url`` (GROBID URL)
  * ``test_save_refuses_invalid_yolo_path``

Plus a couple of defence-in-depth tests for the helpers themselves
(``_validate_api_url`` / ``_validate_ocr_lang``) so the next refactor
can't silently regress the new layer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _reset_language():
    """Pin i18n language across tests so the Phase 47 friendly-name
    logic doesn't race with our field-level assertions."""
    from rlpe.gui import i18n

    i18n.set_language("en")
    yield
    i18n.set_language("en")


# ============================================================
# Module-level helpers: _validate_yolo_model_path
# ============================================================
def test_yolo_validates_existing_pt_file(tmp_path):
    """An existing .pt file returns its absolute path."""
    from rlpe.gui.settings_tab import _validate_yolo_model_path

    p = tmp_path / "yolo11x.pt"
    p.write_bytes(b"")
    result = _validate_yolo_model_path(str(p))
    assert result is not None, f".pt file should validate, got None for {p}"
    assert Path(result).is_file()
    assert Path(result).suffix.lower() == ".pt"


def test_yolo_validates_pth_and_onnx_and_weights(tmp_path):
    """Same rule for .pth, .onnx and legacy .weights extensions."""
    from rlpe.gui.settings_tab import _validate_yolo_model_path

    for ext in (".pth", ".onnx", ".weights"):
        p = tmp_path / f"model{ext}"
        p.write_bytes(b"")
        result = _validate_yolo_model_path(str(p))
        assert result is not None, f"{ext} file should validate, got None for {p}"
        assert Path(result).suffix.lower() == ext


def test_yolo_rejects_nonexistent_path(tmp_path):
    """A path that doesn't exist on disk returns None (no RCE path)."""
    from rlpe.gui.settings_tab import _validate_yolo_model_path

    fake = tmp_path / "does-not-exist.pt"
    assert not fake.exists()
    result = _validate_yolo_model_path(str(fake))
    assert result is None, (
        f"Non-existent .pt path should be rejected, got {result!r}"
    )


def test_yolo_rejects_unsupported_extension(tmp_path):
    """A .abc extension is rejected — YOLO weights are .pt/.pth/.onnx/.weights only."""
    from rlpe.gui.settings_tab import _validate_yolo_model_path

    p = tmp_path / "foo.abc"
    p.write_bytes(b"")
    result = _validate_yolo_model_path(str(p))
    assert result is None, (
        f"Unsupported .abc extension should be rejected, got {result!r}"
    )


def test_yolo_empty_path_allowed():
    """An empty path is treated as 'don't use YOLO' — not an error."""
    from rlpe.gui.settings_tab import _validate_yolo_model_path

    for empty in ("", "   ", None):
        result = _validate_yolo_model_path(empty) if empty is not None else _validate_yolo_model_path("")
        assert result is None, f"Empty path should not validate, got {result!r}"


# ============================================================
# Module-level helpers: _validate_api_url, _validate_ocr_lang
# ============================================================
def test_validate_api_url_accepts_http_and_https():
    from rlpe.gui.settings_tab import _validate_api_url

    assert _validate_api_url("http://localhost:8070") == "http://localhost:8070"
    assert _validate_api_url("https://paleobiodb.org/data1.2") == (
        "https://paleobiodb.org/data1.2"
    )


def test_validate_api_url_rejects_garbage():
    from rlpe.gui.settings_tab import _validate_api_url

    for bad in ("not-a-url", "ftp://example.com", "//missing-scheme", "javascript:alert(1)"):
        assert _validate_api_url(bad) is None, f"{bad!r} should be rejected"


def test_validate_api_url_allows_empty_when_allow_empty_true():
    from rlpe.gui.settings_tab import _validate_api_url

    assert _validate_api_url("", allow_empty=True) == ""
    assert _validate_api_url("   ", allow_empty=True) == ""


def test_validate_api_url_rejects_empty_when_allow_empty_false():
    from rlpe.gui.settings_tab import _validate_api_url

    assert _validate_api_url("") is None
    assert _validate_api_url("   ") is None


def test_validate_ocr_lang_accepts_iso_codes():
    from rlpe.gui.settings_tab import _validate_ocr_lang

    for good in ("en", "en,ja", "ch_sim,en", "en,ja,fr", "de"):
        result = _validate_ocr_lang(good)
        assert result is not None, f"OCR lang {good!r} should validate"


def test_validate_ocr_lang_rejects_unknown_codes():
    from rlpe.gui.settings_tab import _validate_ocr_lang

    for bad in ("xx", "en,xx", "english", "EN" ):
        result = _validate_ocr_lang(bad)
        assert result is None, f"OCR lang {bad!r} should be rejected"


def test_validate_ocr_lang_empty_returns_empty_string():
    from rlpe.gui.settings_tab import _validate_ocr_lang

    assert _validate_ocr_lang("") == ""


# ============================================================
# Save() refuses to persist invalid input
# ============================================================
def test_save_refuses_invalid_url(monkeypatch, tmp_path):
    """M-1: ``_save()`` must NOT update QSettings when GROBID URL is
    ``not-a-url``. A warning popup is shown and the GUI logger records
    the failure."""
    from rlpe.gui.settings_tab import SettingsTab

    # Build a fresh tab with a known cache dict so we can inspect the
    # in-memory mirror without touching the user's real QSettings.
    cache: dict = {}
    st = SettingsTab(cache)
    # Clear YOLO bits so the earlier YOLO validator (which fires BEFORE
    # the GROBID validator in _save()) doesn't intercept this test.
    st._yolo_enable.setChecked(False)
    st._yolo_model_path.setText("")
    st._grobid_url.setText("not-a-url")
    st._pbdb_endpoint.setText("")
    st._ocr_lang.setText("en")

    # Suppress the QMessageBox.warning popup + log call so we can
    # assert on them without the dialog blocking in offscreen mode.
    warnings: list = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **kw: warnings.append((a, kw)) or QMessageBox.Ok),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **kw: QMessageBox.Ok),
    )
    log_calls: list = []
    monkeypatch.setattr(
        st._log,
        "warning",
        lambda msg, *a, **kw: log_calls.append(msg),
    )

    # Snapshot a known pre-save value so we can prove _save() did
    # NOT modify QSettings. Always overwrite so we don't depend on
    # whatever QSettings inherited from a previous test run
    # (QSettings is persistent across processes on the user's disk).
    sentinel = "BEFORE_SAVE_GROBID"
    st._qsettings.setValue("grobid_url", sentinel)
    qs_before = st._qsettings.value("grobid_url", "")

    st._save()

    # The validator must have refused and the QSettings key must
    # NOT be updated to "not-a-url".
    qs_after = st._qsettings.value("grobid_url", "")
    assert qs_after != "not-a-url", (
        f"_save() updated QSettings grobid_url to the invalid value: "
        f"{qs_after!r}. It should have refused to save."
    )
    assert qs_after == qs_before, (
        f"_save() unexpectedly changed grobid_url from {qs_before!r} to {qs_after!r}"
    )
    # The cache mirror (used by Run tab) must also not contain the
    # invalid value.
    assert cache.get("grobid_url") != "not-a-url", (
        f"_save() leaked 'not-a-url' into in-memory cache: {cache.get('grobid_url')!r}"
    )
    # A warning popup must have been shown (call recorded).
    assert len(warnings) >= 1, (
        f"_save() did not display a QMessageBox.warning when the URL was invalid"
    )
    # ...and the GUI logger should have a matching entry.
    assert any("GROBID URL" in str(m) or "URL" in str(m) for m in log_calls), (
        f"_save() did not log a warning about the invalid URL. log_calls={log_calls}"
    )


def test_save_refuses_invalid_yolo_path(monkeypatch, tmp_path):
    """M-1/M-18: ``_save()`` must NOT update QSettings when the YOLO
    path is a file with an unsupported extension (.abc) or points to
    a non-existent file."""
    from rlpe.gui.settings_tab import SettingsTab

    cache: dict = {}
    st = SettingsTab(cache)

    # Plug a YOLO: enabled + path -> invalid path. We use an existing
    # but unsupported extension so the "must exist" check passes but
    # the "must be .pt/.pth/.onnx/.weights" check fails.
    fake_abc = tmp_path / "foo.abc"
    fake_abc.write_bytes(b"")
    st._yolo_enable.setChecked(True)
    st._yolo_model_path.setText(str(fake_abc))
    st._grobid_url.setText("http://localhost:8070")  # keep GROBID valid
    st._pbdb_endpoint.setText("")
    st._ocr_lang.setText("en")

    warnings: list = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **kw: warnings.append((a, kw)) or QMessageBox.Ok),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **kw: QMessageBox.Ok),
    )
    log_calls: list = []
    monkeypatch.setattr(
        st._log,
        "warning",
        lambda msg, *a, **kw: log_calls.append(msg),
    )

    # Snapshot pre-save yolo_model_path.
    st._qsettings.setValue("yolo_model_path", "BEFORE_SAVE_YOLO")
    yolo_before = "BEFORE_SAVE_YOLO"

    st._save()

    yolo_after = st._qsettings.value("yolo_model_path", "")
    assert yolo_after != str(fake_abc), (
        f"_save() persisted invalid YOLO path {str(fake_abc)!r} to QSettings"
    )
    assert yolo_after == yolo_before, (
        f"_save() unexpectedly changed yolo_model_path from {yolo_before!r} to {yolo_after!r}"
    )
    assert cache.get("yolo_model_path") != str(fake_abc), (
        f"_save() leaked invalid YOLO path into cache: {cache.get('yolo_model_path')!r}"
    )
    assert len(warnings) >= 1, (
        f"_save() did not display a QMessageBox.warning when YOLO path was invalid"
    )
    assert any("YOLO" in str(m) for m in log_calls), (
        f"_save() did not log a warning about the invalid YOLO path. log_calls={log_calls}"
    )


def test_save_refuses_nonexistent_yolo_path(monkeypatch, tmp_path):
    """Defence-in-depth: a non-existent path is also refused (the YOLO
    code reaches torch.load(), which would try to open the file and
    later fail; we want it to fail at Save() instead)."""
    from rlpe.gui.settings_tab import SettingsTab

    cache: dict = {}
    st = SettingsTab(cache)
    bogus = tmp_path / "totally-bogus.pt"
    assert not bogus.exists()

    st._yolo_enable.setChecked(True)
    st._yolo_model_path.setText(str(bogus))
    st._grobid_url.setText("http://localhost:8070")
    st._pbdb_endpoint.setText("")
    st._ocr_lang.setText("en")

    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **kw: QMessageBox.Ok),
    )
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **kw: QMessageBox.Ok),
    )

    st._qsettings.setValue("yolo_model_path", "BEFORE_SAVE")
    st._save()
    after = st._qsettings.value("yolo_model_path", "")
    assert after != str(bogus), (
        f"_save() persisted non-existent YOLO path {str(bogus)!r}"
    )


def test_save_accepts_valid_yolo_path(monkeypatch, tmp_path):
    """Sanity: a valid YOLO path IS persisted, with its absolute form."""
    from rlpe.gui.settings_tab import SettingsTab

    cache: dict = {}
    st = SettingsTab(cache)
    real = tmp_path / "real_model.pt"
    real.write_bytes(b"")

    st._yolo_enable.setChecked(True)
    st._yolo_model_path.setText(str(real))
    st._grobid_url.setText("http://localhost:8070")
    st._pbdb_endpoint.setText("")
    st._ocr_lang.setText("en")

    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **kw: QMessageBox.Ok),
    )
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **kw: QMessageBox.Ok),
    )

    st._save()
    after = st._qsettings.value("yolo_model_path", "")
    assert Path(after).is_file(), (
        f"_save() should have persisted the absolute path to the real .pt file, got {after!r}"
    )
    assert Path(after).suffix == ".pt"


def test_save_refuses_invalid_pbdb_endpoint(monkeypatch, tmp_path):
    """PBDB endpoint with garbage text is refused; empty is OK."""
    from rlpe.gui.settings_tab import SettingsTab

    cache: dict = {}
    st = SettingsTab(cache)
    st._yolo_enable.setChecked(False)
    st._grobid_url.setText("http://localhost:8070")
    st._ocr_lang.setText("en")

    # Non-empty invalid PBDB endpoint
    warnings: list = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **kw: warnings.append(1) or QMessageBox.Ok),
    )
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **kw: QMessageBox.Ok),
    )
    st._pbdb_endpoint.setText("not-a-url")
    st._qsettings.setValue("paleodb_endpoint", "BEFORE_SAVE")
    st._save()
    after = st._qsettings.value("paleodb_endpoint", "")
    assert after == "BEFORE_SAVE", (
        f"_save() persisted invalid PBDB endpoint; got {after!r}"
    )
    assert warnings, "PBDB-endpoint rejection should produce a QMessageBox"


def test_save_refuses_invalid_ocr_lang(monkeypatch, tmp_path):
    """OCR lang with unknown codes is refused."""
    from rlpe.gui.settings_tab import SettingsTab

    cache: dict = {}
    st = SettingsTab(cache)
    st._yolo_enable.setChecked(False)
    st._grobid_url.setText("http://localhost:8070")
    st._pbdb_endpoint.setText("")

    warnings: list = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **kw: warnings.append(1) or QMessageBox.Ok),
    )
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **kw: QMessageBox.Ok),
    )

    # Bypass the regex validator by directly setting invalid text;
    # _save() must still reject it.
    st._ocr_lang.setText("xx")
    st._qsettings.setValue("ocr_lang", "BEFORE_SAVE")
    st._save()
    after = st._qsettings.value("ocr_lang", "")
    assert after == "BEFORE_SAVE", (
        f"_save() persisted invalid OCR lang; got {after!r}"
    )
    assert warnings, "OCR-lang rejection should produce a QMessageBox"


# ============================================================
# Visual feedback (text-changed slots): red border on invalid input
# ============================================================
def test_yolo_field_paints_red_border_on_invalid(tmp_path):
    """Typing an invalid YOLO path paints the QLineEdit red."""
    from rlpe.gui.settings_tab import SettingsTab

    st = SettingsTab({})
    bogus = tmp_path / "missing.pt"  # does NOT exist
    st._yolo_model_path.setText(str(bogus))
    # The text-changed handler should have painted the border red.
    assert "border" in st._yolo_model_path.styleSheet(), (
        f"Expected red-border QSS on invalid YOLO path, got styleSheet={st._yolo_model_path.styleSheet()!r}"
    )


def test_yolo_field_clears_red_border_on_valid(tmp_path):
    """Setting a valid YOLO path clears the red-border styling."""
    from rlpe.gui.settings_tab import SettingsTab

    st = SettingsTab({})
    bogus = tmp_path / "missing.pt"
    st._yolo_model_path.setText(str(bogus))
    assert "border" in st._yolo_model_path.styleSheet()

    real = tmp_path / "real.pt"
    real.write_bytes(b"")
    st._yolo_model_path.setText(str(real))
    # Once the path is valid, the border QSS should be cleared
    # (set to "" or to the VALID_BORDER_QSS).
    assert "border: 2px solid #dc3545" not in st._yolo_model_path.styleSheet(), (
        f"Red border should be cleared after setting valid path, "
        f"got styleSheet={st._yolo_model_path.styleSheet()!r}"
    )


def test_grobid_field_paints_red_border_on_garbage():
    from rlpe.gui.settings_tab import SettingsTab

    st = SettingsTab({})
    st._grobid_url.setText("not-a-url")
    assert "border" in st._grobid_url.styleSheet(), (
        f"Expected red-border QSS on invalid GROBID URL, "
        f"got styleSheet={st._grobid_url.styleSheet()!r}"
    )


def test_pbdb_field_red_border_on_garbage_but_not_on_empty():
    """PBDB endpoint: empty is OK (no border); garbage triggers red."""
    from rlpe.gui.settings_tab import SettingsTab

    st = SettingsTab({})

    # Empty initially -> no border.
    st._pbdb_endpoint.setText("")
    assert "border: 2px solid #dc3545" not in st._pbdb_endpoint.styleSheet()

    # Garbage -> red.
    st._pbdb_endpoint.setText("not-a-url")
    assert "border: 2px solid #dc3545" in st._pbdb_endpoint.styleSheet()

    # Back to empty -> cleared.
    st._pbdb_endpoint.setText("")
    assert "border: 2px solid #dc3545" not in st._pbdb_endpoint.styleSheet()
