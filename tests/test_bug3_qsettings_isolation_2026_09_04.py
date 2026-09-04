"""BUG-3 of the 2026-09-04 zero-rows diagnosis.

The pytest suite and the real desktop GUI share the same QSettings
scope (org "RLPE Contributors", app "RLPE - Radiolarian Plate
Extractor" → ``~/.config/RLPE Contributors/…conf``). Any GUI test that
instantiates a tab and saves — or simply constructs QSettings and
writes — lands in the USER's real config file. Empirically the user's
``io/last_pdf_dir`` / ``io/last_export_dir`` were wiped to empty
strings by a test run at 20:01, which broke the GUI's on-restart disk
scan (Phase 49) because the saved user_roots list became empty.

Fixes under test:
  * a session-wide autouse fixture in ``tests/conftest.py`` redirects
    QSettings (IniFormat/UserScope + XDG_CONFIG_HOME) into the pytest
    tmp dir, so test writes can never reach ``~/.config``;
  * ``MainWindow._flush_settings`` writes ONLY the two canonical
    ``io/``-prefixed directory keys — the old ``for key, value in
    self._settings.items(): setValue(key, value)`` dumped every bare
    in-memory key (``last_pdf_dir``, ``llm_backend``, ``theme``, …)
    into the ``[General]`` section where nothing reads them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_HAS_PYSIDE6 = True
try:
    import PySide6  # noqa: F401
except ImportError:
    _HAS_PYSIDE6 = False

pytestmark = [pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")]


# ----------------------------------------------------------------------
# conftest isolation fixture (source guards)
# ----------------------------------------------------------------------
class TestConftestIsolationFixture:
    def test_conftest_redirects_qsettings(self):
        src = (Path(__file__).resolve().parent / "conftest.py").read_text(encoding="utf-8")
        assert "setPath" in src, "conftest must call QSettings.setPath to redirect writes"
        assert "XDG_CONFIG_HOME" in src, "conftest must also redirect XDG_CONFIG_HOME"
        assert "IniFormat" in src

    def test_isolation_fixture_is_autouse(self):
        src = (Path(__file__).resolve().parent / "conftest.py").read_text(encoding="utf-8")
        # The QSettings fixture must be autouse so no test can opt out.
        assert "autouse=True" in src
        fixture_start = src.find("def _isolate_qsettings")
        assert fixture_start != -1, "expected an _isolate_qsettings fixture in conftest"
        head = src[:fixture_start]
        assert "@pytest.fixture(autouse=True)" in head[-100:], (
            "the _isolate_qsettings fixture needs @pytest.fixture(autouse=True)"
        )


# ----------------------------------------------------------------------
# Runtime proof: QSettings writes land in tmp, never in ~/.config
# ----------------------------------------------------------------------
class TestQSettingsRedirected:
    def test_settings_write_lands_in_tmp_not_user_config(self, tmp_path):
        from PySide6.QtCore import QSettings

        real_ini = (
            Path.home()
            / ".config"
            / "RLPE Contributors"
            / "RLPE - Radiolarian Plate Extractor.conf"
        )
        # Snapshot BEFORE writing (a stale marker from an old run must
        # not fail this test — the invariant is "the test write does
        # not modify the user's real config").
        content_before = real_ini.read_text(errors="ignore") if real_ini.exists() else None

        marker = "rlpe-bug3-isolation-marker"
        qs = QSettings("RLPE Contributors", "RLPE - Radiolarian Plate Extractor")
        qs.setValue("io/last_pdf_dir", marker)
        qs.sync()

        # 1. The user's real config must be untouched by the write.
        if content_before is not None:
            assert real_ini.read_text(errors="ignore") == content_before, (
                "test QSettings write leaked into the user's real config file"
            )

        # 2. The write must have landed inside the pytest tmp redirect
        # (a sibling dir issued by tmp_path_factory, NOT the user's
        # home and NOT inside this test's own tmp_path — fixture dirs
        # must not pollute tests that enumerate their tmp_path).
        redirect_root = tmp_path.parent
        conf_files = [
            p for p in redirect_root.rglob("*.conf") if marker in p.read_text(errors="ignore")
        ]
        assert conf_files, "QSettings write must land in the tmp redirect, not the user config"
        assert not any(marker in p.read_text(errors="ignore") for p in tmp_path.rglob("*.conf"))


# ----------------------------------------------------------------------
# MainWindow._flush_settings canonical keys
# ----------------------------------------------------------------------
class _FakeQSettings:
    def __init__(self):
        self.written: dict[str, object] = {}

    def setValue(self, key, value):
        self.written[key] = value

    def sync(self):
        self.synced = True


class TestFlushSettingsCanonicalKeys:
    def _make_window(self, settings: dict):
        from rlpe.gui.main_window import MainWindow

        mw = MainWindow.__new__(MainWindow)
        mw._settings = dict(settings)
        mw._qsettings = _FakeQSettings()
        return mw

    def test_flushes_only_canonical_io_keys(self):
        mw = self._make_window(
            {
                "last_pdf_dir": "/papers",
                "last_export_dir": "/out",
                "llm_backend": "minimax",
                "theme": "dark",
                "ocr_lang": "en",
            }
        )
        mw._flush_settings()
        assert mw._qsettings.written == {
            "io/last_pdf_dir": "/papers",
            "io/last_export_dir": "/out",
        }
        assert getattr(mw._qsettings, "synced", False) is True

    def test_no_bare_key_pollution(self):
        """The old bug: every in-memory key was written under its bare
        name into the [General] section, where no reader ever looks."""
        mw = self._make_window({"last_pdf_dir": "/papers", "llm_backend": "minimax"})
        mw._flush_settings()
        assert "last_pdf_dir" not in mw._qsettings.written
        assert "llm_backend" not in mw._qsettings.written

    def test_empty_settings_flush_nothing(self):
        mw = self._make_window({})
        mw._flush_settings()
        assert mw._qsettings.written == {}

    def test_source_guard_no_bulk_setvalue_loop(self):
        src = (_SRC / "rlpe" / "gui" / "main_window.py").read_text(encoding="utf-8")
        assert "for key, value in self._settings.items():" not in src, (
            "_flush_settings must not bulk-write bare in-memory keys to QSettings"
        )
