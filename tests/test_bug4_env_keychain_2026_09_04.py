"""BUG-4 of the 2026-09-04 zero-rows follow-up (user re-run after the
BUG-1..3 fixes still produced 0 rows in seconds).

Evidence chain from the user's GUI log:

1. ``Pipeline finished: 0 rows`` + the new local_only warning fired,
   so ``_resolve_outbound_policy`` resolved **local_only**;
2. yet pipeline.py logged ``[MiniMax] API error, falling back to rule
   pipeline`` — that is NOT a network error: it is the local_only
   no-op result (``error_type=LocalOnlyPolicy``) flowing through the
   generic FallbackHandler, which mislabels it as an "API error". No
   network call ever happened (hence the run finishing in ~1 s).

Root causes:

* **BUG-4a**: ``gui/app.py`` never calls ``load_env_file`` — only the
  CLI and web server do. The desktop GUI process never sees the
  project .env (``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL`` /
  ``ANTHROPIC_MODEL`` pointing at api.minimaxi.com).
* **BUG-4b**: pipeline.py:289 (Round 18) injects ``ANTHROPIC_API_KEY``
  into ``extra["MiniMax_api_key"]`` — that is the project's documented
  key layout (MiniMax speaks the Anthropic wire protocol) — but the
  BUG-1 worker resolver ``_resolve_outbound_policy`` only checked
  settings + ``MiniMax_API_KEY``/``MINIMAX_API_KEY`` env, so it
  resolved local_only even though the pipeline would have had a key.
  The resolver and the pipeline disagreed; the resolver won and
  disabled the LLM.
* **BUG-4c**: the rules-fallback branch logged only "API error" with
  no detail, hiding that the real reason was the local_only policy.
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

_MINIMAX_ENV_VARS = ("MiniMax_API_KEY", "MINIMAX_API_KEY", "ANTHROPIC_API_KEY")


def _clear_llm_env(monkeypatch):
    for var in _MINIMAX_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Some eval-wiring test modules set this at import time
    # (os.environ.setdefault) — it leaks into every later test, so
    # these tests must always clear it explicitly.
    monkeypatch.delenv("RLPE_DATA_OUTBOUND_OPT_IN", raising=False)


# ----------------------------------------------------------------------
# resolve_minimax_api_key — single source of truth
# ----------------------------------------------------------------------
class TestResolveMinimaxApiKey:
    def test_extra_key_wins(self, monkeypatch):
        from rlpe.llm_backends import resolve_minimax_api_key

        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        assert resolve_minimax_api_key({"MiniMax_api_key": "extra-key"}) == "extra-key"

    def test_minimax_upper_camel_env(self, monkeypatch):
        from rlpe.llm_backends import resolve_minimax_api_key

        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("MiniMax_API_KEY", "camel-key")
        assert resolve_minimax_api_key() == "camel-key"

    def test_minimax_upper_env(self, monkeypatch):
        from rlpe.llm_backends import resolve_minimax_api_key

        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("MINIMAX_API_KEY", "upper-key")
        assert resolve_minimax_api_key() == "upper-key"

    def test_anthropic_api_key_fallback_round18(self, monkeypatch):
        """Round 18 semantics: the project's .env documents
        ANTHROPIC_API_KEY as the user-facing key (MiniMax speaks the
        Anthropic wire protocol) and pipeline.py injects it into
        extra["MiniMax_api_key"]. The resolver must agree."""
        from rlpe.llm_backends import resolve_minimax_api_key

        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-cp-test")
        assert resolve_minimax_api_key() == "sk-cp-test"

    def test_none_when_no_key_anywhere(self, monkeypatch):
        from rlpe.llm_backends import resolve_minimax_api_key

        _clear_llm_env(monkeypatch)
        assert resolve_minimax_api_key() is None


# ----------------------------------------------------------------------
# Worker resolver agrees with the pipeline injection
# ----------------------------------------------------------------------
class TestWorkerResolverSeesAnthropicKey:
    def test_anthropic_key_yields_api_redacted(self, monkeypatch):
        """THE user regression: GUI + .env (ANTHROPIC_API_KEY only) must
        resolve api_redacted, not local_only."""
        from rlpe.gui.pipeline_worker import _resolve_outbound_policy

        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-cp-test")
        assert _resolve_outbound_policy("", None) == "api_redacted"

    def test_no_key_at_all_still_local_only(self, monkeypatch):
        from rlpe.gui.pipeline_worker import _resolve_outbound_policy

        _clear_llm_env(monkeypatch)
        assert _resolve_outbound_policy("", None) == "local_only"

    def test_build_config_end_to_end_with_anthropic_env(self, monkeypatch, tmp_path):
        """_build_config must resolve api_redacted so the backend the
        pipeline builds makes real API calls (key injected by
        pipeline.py:289)."""
        from rlpe.gui.pipeline_worker import PipelineWorker

        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-cp-test")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")

        pdf = tmp_path / "in.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        work = tmp_path / "work"
        work.mkdir()
        worker = PipelineWorker.__new__(PipelineWorker)
        worker._settings = {"use_gpu": False}
        worker._work_dir = work
        worker._pdf_path = pdf
        cfg = worker._build_config()
        assert cfg.extra["data_outbound_policy"] == "api_redacted"


# ----------------------------------------------------------------------
# GUI loads the project .env (BUG-4a)
# ----------------------------------------------------------------------
class TestGuiLoadsProjectEnv:
    def test_run_app_calls_env_loader(self):
        src = (_SRC / "rlpe" / "gui" / "app.py").read_text(encoding="utf-8")
        assert "load_env_file" in src, "gui/app.py must load the project .env"
        assert "_load_project_env" in src

    def test_load_project_env_sets_keys_from_dotenv(self, monkeypatch, tmp_path):
        """Functional: the extracted helper loads a given .env via the
        shared env_loader (override rules included)."""
        from rlpe.gui import app as gui_app

        env_file = tmp_path / ".env"
        env_file.write_text("RLPE_BUG4_PROBE_KEY=probe-value\n")
        monkeypatch.delenv("RLPE_BUG4_PROBE_KEY", raising=False)
        loaded = gui_app._load_project_env(env_file)
        assert loaded >= 1
        import os

        assert os.environ["RLPE_BUG4_PROBE_KEY"] == "probe-value"

    def test_load_project_env_missing_file_returns_zero(self, tmp_path):
        from rlpe.gui import app as gui_app

        assert gui_app._load_project_env(tmp_path / "nope.env") == 0


# ----------------------------------------------------------------------
# Pipeline fallback log carries the real error (BUG-4c)
# ----------------------------------------------------------------------
class TestFallbackLogTransparency:
    def test_rules_branch_logs_error_text(self):
        src = (_SRC / "rlpe" / "pipeline.py").read_text(encoding="utf-8")
        assert 'error_info.get("error")' in src, (
            "the rules-fallback warning must include the actual error text "
            "(it previously hid local_only short-circuits as 'API error')"
        )
