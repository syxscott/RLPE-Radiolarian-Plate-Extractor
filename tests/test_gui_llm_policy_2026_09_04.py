"""Audit 2026-09-04 (user-reported Zhang 2014 zero-rows follow-up, BUG-1):

The PySide6 GUI could never use the LLM. ``PipelineWorker._build_config``
hardcoded ``data_outbound_policy`` to ``local_only`` (the setting dict
never carries the key), and the GUI had no API-key / policy controls at
all. ``MiniMaxM3Backend`` short-circuits every ``infer_*`` call under
``local_only``, so Stage-1 caption parsing silently fell back to the
regex parser — which produces a garbage CaptionPair for the
"Explanation of Plate N" caption convention — and the run finished with
0 rows and no warning.

Fixes covered here:
  * ``_resolve_outbound_policy``: auto policy — ``api_redacted`` when a
    MiniMax key is reachable (settings or env), ``local_only`` otherwise;
    an explicit user choice always wins.
  * ``api_full`` without the ``RLPE_DATA_OUTBOUND_OPT_IN`` env var
    downgrades to ``api_redacted`` instead of raising mid-run.
  * ``collect_settings`` forwards ``MiniMax_api_key`` /
    ``data_outbound_policy`` from the shared settings dict.
  * Settings tab exposes the API key (password echo) + policy combo and
    persists both under the bare QSettings keys.
  * i18n keys exist in both string tables.
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

pytestmark = [
    pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed"),
]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _make_worker(settings: dict, tmp_path: Path):
    """Build a PipelineWorker-shaped object without QThread/__init__.

    ``_build_config`` only touches ``self._settings`` and
    ``self._work_dir``; bypassing ``QWidget``/``QThread`` init keeps the
    test off the SIGSEGV (PySide6 6.11 + py3.11) combo and fast.
    A real (dummy) PDF file is needed because ``_build_config`` copies
    it into the work dir's ``input/``.
    """
    from rlpe.gui.pipeline_worker import PipelineWorker

    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%test\n")
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    worker = PipelineWorker.__new__(PipelineWorker)
    worker._settings = dict(settings)
    worker._work_dir = work
    worker._pdf_path = pdf
    return worker


class _StubWidget:
    """Duck-typed stand-in for the Qt widgets ``collect_settings`` reads."""

    def __init__(self, value=None, data=None, text=None):
        self._value = value
        self._data = data
        self._text = text

    def text(self):
        return self._text if self._text is not None else ""

    def value(self):
        return self._value

    def isChecked(self):
        return bool(self._value)

    def currentData(self):
        return self._data

    def currentText(self):
        return self._text if self._text is not None else str(self._data)


def _make_run_tab(settings: dict):
    """Build a RunTab-shaped stub with the widget attributes
    ``collect_settings`` reads, wired so the LLM-auth keys come from
    ``_settings`` exactly like the YOLO keys already do."""
    from rlpe.gui.run_tab import RunTab

    tab = RunTab.__new__(RunTab)
    tab._settings = settings
    tab._ocr_combo = _StubWidget(data="paddleocr")
    tab._ocr_lang_edit = _StubWidget(data="en")
    tab._grobid_edit = _StubWidget(text="http://localhost:8070")
    tab._grobid_retries = _StubWidget(value=3)
    tab._grobid_timeout = _StubWidget(value=300)
    tab._caption_window = _StubWidget(value=2)
    tab._od_caption_window = _StubWidget(value=5)
    tab._workers = _StubWidget(value=1)
    tab._panel_score = _StubWidget(value=0.8)
    tab._gpu_check = _StubWidget(value=False)
    tab._llm_combo = _StubWidget(data="minimax", text="MiniMax M2.5")
    tab._m3_lang = _StubWidget(data="auto")
    tab._m3_model_edit = _StubWidget(text="MiniMax-M3")
    tab._m3_budget = _StubWidget(value=1024)
    tab._m3_output = _StubWidget(value=2048)
    tab._m3_timeout = _StubWidget(value=60)
    tab._m3_max_retries = _StubWidget(value=3)
    tab._paleodb_check = _StubWidget(value=False)
    tab._paleodb_occ = _StubWidget(value=25)
    tab._geo_vision = _StubWidget(value=False)
    tab._m3_stage3 = _StubWidget(value=True)
    tab._m3_multi_plate = _StubWidget(value=True)
    tab._od_fallback = _StubWidget(value=True)
    tab._save_intermediate = _StubWidget(value=False)
    tab._dpi = _StubWidget(value=200)
    return tab


# ----------------------------------------------------------------------
# _resolve_outbound_policy
# ----------------------------------------------------------------------
class TestResolveOutboundPolicy:
    def test_env_key_yields_api_redacted(self, monkeypatch):
        from rlpe.gui.pipeline_worker import _resolve_outbound_policy

        monkeypatch.delenv("MiniMax_API_KEY", raising=False)
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-env")
        assert _resolve_outbound_policy("", None) == "api_redacted"

    def test_no_key_anywhere_yields_local_only(self, monkeypatch):
        from rlpe.gui.pipeline_worker import _resolve_outbound_policy

        monkeypatch.delenv("MiniMax_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        assert _resolve_outbound_policy("", None) == "local_only"

    def test_settings_key_yields_api_redacted(self, monkeypatch):
        from rlpe.gui.pipeline_worker import _resolve_outbound_policy

        monkeypatch.delenv("MiniMax_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        assert _resolve_outbound_policy("", "sk-test-settings") == "api_redacted"

    def test_explicit_policy_wins(self, monkeypatch):
        from rlpe.gui.pipeline_worker import _resolve_outbound_policy

        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-env")
        assert _resolve_outbound_policy("local_only", "sk-x") == "local_only"
        assert _resolve_outbound_policy("api_redacted", None) == "api_redacted"

    def test_garbage_policy_falls_back_to_auto(self, monkeypatch):
        from rlpe.gui.pipeline_worker import _resolve_outbound_policy

        monkeypatch.delenv("MiniMax_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        assert _resolve_outbound_policy("garbage", None) == "local_only"


# ----------------------------------------------------------------------
# _build_config end-to-end policy wiring
# ----------------------------------------------------------------------
class TestBuildConfigPolicy:
    def test_env_key_propagates_api_redacted(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-env")
        worker = _make_worker({"use_gpu": False}, tmp_path)
        cfg = worker._build_config()
        assert cfg.extra["data_outbound_policy"] == "api_redacted"

    def test_no_key_local_only(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MiniMax_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        worker = _make_worker({"use_gpu": False}, tmp_path)
        cfg = worker._build_config()
        assert cfg.extra["data_outbound_policy"] == "local_only"

    def test_settings_key_forwarded_and_policy_redacted(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MiniMax_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        worker = _make_worker({"use_gpu": False, "MiniMax_api_key": "sk-abc"}, tmp_path)
        cfg = worker._build_config()
        assert cfg.extra["MiniMax_api_key"] == "sk-abc"
        assert cfg.extra["data_outbound_policy"] == "api_redacted"

    def test_explicit_local_only_kept_even_with_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-env")
        worker = _make_worker({"use_gpu": False, "data_outbound_policy": "local_only"}, tmp_path)
        cfg = worker._build_config()
        assert cfg.extra["data_outbound_policy"] == "local_only"

    def test_api_full_without_optin_downgrades_to_redacted(self, monkeypatch, tmp_path):
        monkeypatch.delenv("RLPE_DATA_OUTBOUND_OPT_IN", raising=False)
        worker = _make_worker(
            {"use_gpu": False, "data_outbound_policy": "api_full", "MiniMax_api_key": "sk-abc"},
            tmp_path,
        )
        cfg = worker._build_config()
        # api_full is opt-in only; the worker must downgrade instead of
        # letting the backend ValueError kill the run mid-flight.
        assert cfg.extra["data_outbound_policy"] == "api_redacted"

    def test_api_full_with_optin_kept(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RLPE_DATA_OUTBOUND_OPT_IN", "1")
        worker = _make_worker(
            {"use_gpu": False, "data_outbound_policy": "api_full", "MiniMax_api_key": "sk-abc"},
            tmp_path,
        )
        cfg = worker._build_config()
        assert cfg.extra["data_outbound_policy"] == "api_full"


# ----------------------------------------------------------------------
# collect_settings forwarding
# ----------------------------------------------------------------------
class TestCollectSettingsForwardsLlmAuth:
    def test_api_key_and_policy_forwarded(self):
        tab = _make_run_tab({"MiniMax_api_key": "sk-run", "data_outbound_policy": "auto"})
        s = tab.collect_settings()
        assert s["MiniMax_api_key"] == "sk-run"
        assert s["data_outbound_policy"] == "auto"

    def test_missing_keys_default_empty(self):
        tab = _make_run_tab({})
        s = tab.collect_settings()
        assert s["MiniMax_api_key"] == ""
        assert s["data_outbound_policy"] == "auto"  # default, resolved worker-side


# ----------------------------------------------------------------------
# Settings tab controls + persistence (source guards — no QApplication)
# ----------------------------------------------------------------------
_SETTINGS_TAB_SRC = (_SRC / "rlpe" / "gui" / "settings_tab.py").read_text(encoding="utf-8")


class TestSettingsTabControls:
    def test_api_key_widget_exists(self):
        assert "_minimax_api_key" in _SETTINGS_TAB_SRC
        assert "EchoMode.Password" in _SETTINGS_TAB_SRC

    def test_policy_combo_widget_exists(self):
        assert "_data_outbound" in _SETTINGS_TAB_SRC

    def test_save_persists_both_keys(self):
        assert '_qsettings.setValue("MiniMax_api_key"' in _SETTINGS_TAB_SRC
        assert '_qsettings.setValue("data_outbound_policy"' in _SETTINGS_TAB_SRC

    def test_load_restores_both_keys(self):
        assert 'self._qsettings.value("MiniMax_api_key"' in _SETTINGS_TAB_SRC
        assert 'self._qsettings.value("data_outbound_policy"' in _SETTINGS_TAB_SRC

    def test_apply_to_run_settings_carries_both(self):
        assert '"MiniMax_api_key"' in _SETTINGS_TAB_SRC
        assert '"data_outbound_policy"' in _SETTINGS_TAB_SRC


# ----------------------------------------------------------------------
# i18n + friendly options
# ----------------------------------------------------------------------
class TestI18nKeys:
    def test_en_strings_have_new_keys(self):
        src = (_SRC / "rlpe" / "gui" / "strings_en.py").read_text(encoding="utf-8")
        assert '"settab.llm.api_key"' in src
        assert '"settab.llm.outbound"' in src

    def test_zh_strings_have_new_keys(self):
        src = (_SRC / "rlpe" / "gui" / "strings_zh_CN.py").read_text(encoding="utf-8")
        assert '"settab.llm.api_key"' in src
        assert '"settab.llm.outbound"' in src

    def test_friendly_options_defined(self):
        from rlpe.gui.constants import data_outbound_friendly_options

        opts = dict(data_outbound_friendly_options())
        assert opts["auto"]
        assert opts["api_redacted"]
        assert opts["api_full"]
        assert opts["local_only"]


# ----------------------------------------------------------------------
# Worker 0-rows warning wiring (source guards)
# ----------------------------------------------------------------------
class TestZeroRowsWarning:
    def test_worker_warns_on_zero_rows_llm_disabled(self):
        src = (_SRC / "rlpe" / "gui" / "pipeline_worker.py").read_text(encoding="utf-8")
        assert "0 rows" in src and "local_only" in src

    def test_no_unconditional_local_only_default(self):
        src = (_SRC / "rlpe" / "gui" / "pipeline_worker.py").read_text(encoding="utf-8")
        # The old bug: s.get("data_outbound_policy", "local_only") made
        # local_only unreachable-to-override from the GUI.
        assert 's.get("data_outbound_policy", "local_only")' not in src
