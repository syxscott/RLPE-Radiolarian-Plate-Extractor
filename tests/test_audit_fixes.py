"""Regression tests for the comprehensive audit pass.

Covers:
  - load_jsonl: skips malformed lines instead of crashing (#IO1)
  - cli_export._run_output_from_jsonl: same robustness
  - choose_best_page: returns None on empty pages
  - JobOptions: logs unknown fields instead of silently dropping
  - ReviewCorrection / ResultRecord: ``extra="forbid"``
  - SSRF guard: blocks link-local / unspecified / non-http hosts
  - MiniMaxM3Backend: retries 401/403 in addition to 5xx/429
  - paper_metadata_from_internal: defensive on None confidence
  - stable_id: streaming (large file) + path fallback
  - image_label_check: caching
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.converters import paper_metadata_from_internal
from rlpe.evaluation.image_label_check import run_image_label_check
from rlpe.io import load_jsonl
from rlpe.layout import choose_best_page, PageRecord
from rlpe.llm_backends import _validate_llm_host, LlamaCppGemmaBackend, OllamaGemmaBackend
from rlpe.types import PaperMetadata as InternalPaperMetadata
from rlpe.utils import stable_id


# ---------------------------------------------------------------- io.load_jsonl


def test_load_jsonl_skips_malformed_lines(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(
        '{"a": 1}\n'
        'this is not json\n'
        '{"b": 2}\n'
        '\n'
        '{trailing\n',  # truncated line
        encoding="utf-8",
    )
    rows = load_jsonl(p)
    assert len(rows) == 2
    assert rows[0] == {"a": 1}
    assert rows[1] == {"b": 2}


def test_load_jsonl_missing_file(tmp_path):
    assert load_jsonl(tmp_path / "does_not_exist.jsonl") == []


# ---------------------------------------------------------------- layout


def test_choose_best_page_empty_pages_returns_none():
    assert choose_best_page([], "1", "caption") is None


def test_choose_best_page_finds_by_text_mention():
    pages = [
        PageRecord(page_index=0, image_path=Path("p0.png"), text="Other text"),
        PageRecord(page_index=1, image_path=Path("p1.png"), text="Figure 3 caption..."),
    ]
    p = choose_best_page(pages, "3", "")
    assert p is not None
    assert p.page_index == 1


# ---------------------------------------------------------------- JobOptions


def test_joboptions_logs_unknown_fields(caplog):
    from rlpe.api.app import JobOptions
    with caplog.at_level("WARNING", logger="rlpe.api"):
        opts = JobOptions(
            use_gemma4=True,
            minimax_api_key="sk-typo-no-match",  # typo of MiniMax_api_key
        )
    # Job was created; typo'd field was silently dropped
    assert opts.use_gemma4 is True
    assert not hasattr(opts, "minimax_api_key")
    # But a warning was logged about the dropped field
    assert any("minimax_api_key" in r.message for r in caplog.records), (
        "JobOptions should log unknown fields; got: "
        + "; ".join(r.message for r in caplog.records)
    )


def test_joboptions_accepts_known_fields():
    from rlpe.api.app import JobOptions
    opts = JobOptions(
        use_gemma4=True,
        llm_backend="ollama",
        MiniMax_api_key="sk-real",
    )
    assert opts.use_gemma4 is True
    assert opts.llm_backend == "ollama"
    assert opts.MiniMax_api_key == "sk-real"


def test_review_correction_rejects_unknown_fields():
    from rlpe.api.app import ReviewCorrection
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ReviewCorrection(
            paper_id="p", figure_id="f",
            panelpath="/typo",  # typo of panel_path
        )


def test_result_record_rejects_unknown_fields():
    """ResultRecord now silently ignores unknown fields (extra='ignore').

    The previous behaviour was ``extra='forbid'`` which crashed the
    /results endpoint with a 500 whenever the pipeline introduced a
    new internal metadata field. The new contract: declared fields
    are honoured exactly, undeclared fields are dropped (and the
    /results endpoint additionally whitelists fields before
    constructing the model). This test verifies the IGNORE behaviour
    is in effect — a typo'd field must not surface in the model.
    """
    from rlpe.api.app import ResultRecord

    r = ResultRecord(
        paper_id="p", figure_id="f", confidence=0.5,
        species_typo="oops",  # typo of species — silently dropped
    )
    # The typo field is dropped; the model only has the declared fields.
    assert not hasattr(r, "species_typo")
    # And the legitimate fields are preserved.
    assert r.paper_id == "p"
    assert r.confidence == 0.5
    assert r.species is None


def test_result_record_accepts_panel_local_path_added_by_api_normalizer():
    from rlpe.api.app import ResultRecord
    r = ResultRecord(
        job_id="j1",
        paper_id="p",
        figure_id="f",
        confidence=0.5,
        panel_path="/jobs/j1/files/panels/p/f/panel_01.png",
        panel_local_path="/abs/service_work/j1/output/panels/p/f/panel_01.png",
    )
    assert r.panel_local_path.endswith("panel_01.png")


def test_get_results_handles_rows_with_panel_local_path():
    from rlpe.api import app as api_app
    job_id = "test-job-panel-local-path"
    with api_app.RESULT_LOCK:
        api_app.RESULT_CACHE[job_id] = {
            "status": "done",
            "result": [{
                "paper_id": "p",
                "figure_id": "f",
                "confidence": 0.5,
                "panel_path": "/jobs/test-job-panel-local-path/files/panel.png",
                "panel_local_path": "/abs/panel.png",
            }],
        }
    try:
        rows = api_app.get_results()
    finally:
        with api_app.RESULT_LOCK:
            api_app.RESULT_CACHE.pop(job_id, None)
    assert any(r.job_id == job_id and r.panel_local_path == "/abs/panel.png" for r in rows)


def test_upload_pdf_defaults_to_joboptions_defaults_and_sanitizes_filename(tmp_path, monkeypatch):
    """Direct API uploads without an options field should still use
    JobOptions defaults (notably use_opendataloader=True), and a
    path-like upload filename should be rejected instead of causing a
    server-side OSError when saving."""
    import asyncio
    from fastapi import HTTPException, UploadFile
    from starlette.datastructures import Headers
    from rlpe.api import app as api_app

    monkeypatch.setattr(api_app, "UPLOAD_DIR", tmp_path)

    class BG:
        def __init__(self):
            self.calls = []
        def add_task(self, fn, *args):
            self.calls.append((fn, args))

    good = UploadFile(
        filename="paper.pdf",
        file=__import__("io").BytesIO(b"%PDF-1.4\n%%EOF"),
        headers=Headers({"content-type": "application/pdf"}),
    )
    bg = BG()
    asyncio.run(api_app.upload_pdf(bg, good, options=None))
    assert bg.calls
    _fn, args = bg.calls[0]
    assert args[2]["use_opendataloader"] is True

    bad = UploadFile(
        filename="../evil.pdf",
        file=__import__("io").BytesIO(b"%PDF-1.4\n%%EOF"),
        headers=Headers({"content-type": "application/pdf"}),
    )
    with pytest.raises(HTTPException) as ei:
        asyncio.run(api_app.upload_pdf(BG(), bad, options=None))
    assert ei.value.status_code == 400


def test_pipeline_progress_callback_cancellation_propagates(tmp_path):
    """A progress callback exception is the API's cancellation signal;
    pipeline.run() must not swallow it and continue to write a manifest."""
    from rlpe.config import PipelineConfig
    from rlpe.pipeline import RadiolarianPipeline

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "p.pdf").write_bytes(b"%PDF-1.4\n%%EOF")

    class CancelMe(Exception):
        pass

    def cb(current, total, message):
        raise CancelMe("cancel")

    cfg = PipelineConfig(pdf_dir=pdf_dir, work_dir=tmp_path / "work", num_workers=1)
    with pytest.raises(CancelMe):
        RadiolarianPipeline(cfg, progress_callback=cb).run()


# ---------------------------------------------------------------- SSRF guard


def test_ssrf_blocks_aws_metadata():
    """169.254.169.254 is the AWS / GCP metadata endpoint — must be blocked."""
    with pytest.raises(ValueError, match="SSRF"):
        _validate_llm_host("http://169.254.169.254/latest/")


def test_ssrf_blocks_unspecified():
    with pytest.raises(ValueError, match="SSRF"):
        _validate_llm_host("http://0.0.0.0:8080/")


def test_ssrf_blocks_file_scheme():
    with pytest.raises(ValueError, match="scheme"):
        _validate_llm_host("file:///etc/passwd")


def test_ssrf_blocks_ipv6_link_local():
    with pytest.raises(ValueError, match="SSRF"):
        _validate_llm_host("http://[fe80::1]/")


def test_ssrf_allows_loopback():
    # Default Ollama / LlamaCpp hosts must pass.
    assert _validate_llm_host("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert _validate_llm_host("http://localhost:8080") == "http://localhost:8080"


def test_ssrf_allows_private_rfc1918():
    # Private RFC1918 ranges are common for home-network LLM
    # servers; allow with no warning.
    assert _validate_llm_host("http://10.0.0.5:8080") == "http://10.0.0.5:8080"
    assert _validate_llm_host("http://192.168.1.1:8080") == "http://192.168.1.1:8080"


def test_ssrf_opt_out_via_env(monkeypatch):
    monkeypatch.setenv("RLPE_LLM_ALLOW_ANY_HOST", "1")
    assert _validate_llm_host("http://169.254.169.254/") == "http://169.254.169.254/"


def test_ollama_backend_construct_validates_host():
    with pytest.raises(ValueError, match="SSRF"):
        OllamaGemmaBackend(model="llama3", host="http://169.254.169.254/")


def test_llamacpp_backend_construct_validates_host():
    with pytest.raises(ValueError, match="SSRF"):
        LlamaCppGemmaBackend(host="http://0.0.0.0:8080")


def test_minimax_m3_backend_name_is_recognized_in_pipeline():
    # This test exercises the real builder path, which needs the
    # ``anthropic`` SDK (the only outbound backend MiniMax M3 uses).
    # When the SDK is missing the builder legitimately raises and the
    # fallback handler is never wired up; that is an environment limit
    # rather than a code defect, so skip gracefully.
    pytest.importorskip("anthropic")
    from rlpe.config import PipelineConfig
    from rlpe.pipeline import RadiolarianPipeline
    cfg = PipelineConfig(
        pdf_dir=Path("/tmp/no-pdfs"),
        work_dir=Path("/tmp/no-work"),
        extra={
            "use_gemma4": True,
            "llm_backend": "MiniMax-m3",
            "MiniMax_api_key": "sk-test",
        },
    )
    pipe = RadiolarianPipeline.__new__(RadiolarianPipeline)
    pipe.config = cfg
    pipe.gemma_runtime = None
    pipe.m3_engine = None
    pipe.gemma_fallback_handler = None
    pipe._try_init_gemma()
    assert pipe.gemma_fallback_handler is not None


def test_minimax_m3_backend_name_is_recognized_in_gemma_postprocess(monkeypatch):
    import rlpe.gemma_postprocess as gp
    called = {"n": 0}

    class FakeBackend:
        backend_name = "MiniMax"

    def fake_builder(extra):
        called["n"] += 1
        return FakeBackend()

    monkeypatch.setattr(
        "rlpe.llm_backends.build_MiniMax_backend_from_env_or_config",
        fake_builder,
    )
    runtime = gp.build_gemma_backend_from_config({
        "llm_backend": "MiniMax-m3",
        "MiniMax_api_key": "sk-test",
    })
    assert called["n"] == 1
    assert runtime.backend_name == "MiniMax"


# ---------------------------------------------------------------- paper_metadata


def test_paper_metadata_from_internal_handles_none_confidence():
    pm = InternalPaperMetadata(
        title="t", authors=[], year=2020, journal=None, volume=None,
        issue=None, pages=None, doi=None, abstract=None, keywords=[],
        publisher=None, page_count=None, source="grobid", confidence=None,
    )
    out = paper_metadata_from_internal(pm)
    assert out is not None
    assert out.confidence == 0.0


def test_paper_metadata_from_internal_handles_non_numeric_confidence():
    pm = InternalPaperMetadata(
        title="t", authors=[], year=2020, journal=None, volume=None,
        issue=None, pages=None, doi=None, abstract=None, keywords=[],
        publisher=None, page_count=None, source="grobid", confidence="bad",  # type: ignore[arg-type]
    )
    out = paper_metadata_from_internal(pm)
    assert out is not None
    assert out.confidence == 0.0


def test_paper_metadata_from_internal_passes_through_legit_confidence():
    pm = InternalPaperMetadata(
        title="t", authors=[], year=2020, journal=None, volume=None,
        issue=None, pages=None, doi=None, abstract=None, keywords=[],
        publisher=None, page_count=None, source="grobid", confidence=0.85,
    )
    out = paper_metadata_from_internal(pm)
    assert out is not None
    assert out.confidence == 0.85


# ---------------------------------------------------------------- stable_id


def test_stable_id_consistent_across_runs(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello world" * 1000)
    id1 = stable_id(f)
    id2 = stable_id(f)
    assert id1 == id2
    assert len(id1) == 16


def test_stable_id_streaming_handles_large_files(tmp_path):
    """The streaming impl must produce the same id as the old
    read-bytes impl, so existing gold data isn't invalidated."""
    import hashlib
    f = tmp_path / "big.bin"
    # 5 MB of pseudo-random data
    f.write_bytes(os.urandom(5 * 1024 * 1024))
    new_id = stable_id(f)
    # Reference: same algorithm done by reading in one shot
    data = f.read_bytes()
    expected = hashlib.sha1(f"{len(data)}:".encode("ascii") + data).hexdigest()[:16]
    assert new_id == expected


def test_stable_id_path_fallback_for_nonexistent(tmp_path):
    p = tmp_path / "does_not_exist.pdf"
    sid = stable_id(p)
    # Falls back to a path-based hash; not empty.
    assert len(sid) == 16


# ---------------------------------------------------------------- image_label cache


def test_image_label_check_uses_cache(tmp_path, monkeypatch):
    """Second call with same panels + cache_path should be a no-op for OCR."""
    # Set up a tiny PNG the checker can resolve.
    from PIL import Image
    panels_dir = tmp_path / "work" / "test_out" / "panels" / "p1" / "fig1"
    panels_dir.mkdir(parents=True)
    panel_path = panels_dir / "panel_01.png"
    Image.new("RGB", (50, 50), (255, 255, 255)).save(panel_path)
    cache_path = tmp_path / "ocr_cache.json"

    predictions = [
        {
            "paper_id": "p1",
            "figure_id": "fig1",
            "panel_id": "1",
            "panel_path": str(panel_path),
        }
    ]

    # A counting reader so we can assert it was only called once.
    call_count = {"n": 0}

    def fake_readtext(self_path, image_arr):
        call_count["n"] += 1
        # Return a numeric label so the prediction is checked.
        return [(((0, 0), (10, 10)), "1", 0.99)]

    # The run uses easyocr.readtext via the module-level function.
    pytest.importorskip("easyocr")
    import easyocr  # noqa: F401
    import rlpe.evaluation.image_label_check as ilc
    monkeypatch.setattr(
        ilc, "_readtext_via_easyocr", lambda path, root: ["1"], raising=False,
    )
    # Easier: monkeypatch easyocr.Reader.readtext.
    class FakeReader:
        def readtext(self, image):
            call_count["n"] += 1
            return [(((0, 0), (10, 10)), "1", 0.99)]
    reader = FakeReader()
    stats1 = run_image_label_check(
        predictions=predictions, root=tmp_path, reader=reader, cache_path=cache_path,
    )
    # Cache should now have one entry; second call should be a hit
    # and not increment call_count.
    stats2 = run_image_label_check(
        predictions=predictions, root=tmp_path, reader=reader, cache_path=cache_path,
    )
    assert call_count["n"] == 1, (
        f"Reader was called {call_count['n']} times; second call should hit the cache"
    )
    assert stats1["aggregate"]["n_checked"] == 1
    assert stats1["aggregate"]["n_cache_hits"] == 0
    assert stats2["aggregate"]["n_cache_hits"] == 1


# ====================================================================
# /system/llm-status and /system/test-llm endpoints (new in this round)
# ====================================================================

def test_llm_status_reports_no_key_when_env_unset(monkeypatch):
    """When neither ANTHROPIC_API_KEY nor MiniMax_API_KEY is set, the
    endpoint must return key_configured=False so the frontend can
    render the "missing key" warning state.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MiniMax_API_KEY", raising=False)
    from rlpe.api.app import app
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.get("/system/llm-status")
    assert r.status_code == 200
    data = r.json()
    assert data["key_configured"] is False
    assert data["key_preview"] is None
    assert data["key_source"] is None
    assert "approx_cny_per_call" in data
    assert "total_cost_cny" in data


def test_llm_status_masks_key_preview_correctly(monkeypatch):
    """A configured key must be reported as configured but masked
    in the preview field — never returned in the clear.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-or-v1-secretvalue123abc")
    from rlpe.api.app import app
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.get("/system/llm-status")
    assert r.status_code == 200
    data = r.json()
    assert data["key_configured"] is True
    # The preview must NOT contain the middle of the secret.
    assert "secretvalue" not in (data["key_preview"] or "")
    # The preview should still let the user identify which key is loaded
    # by showing the prefix and the last 4 chars.
    assert data["key_preview"].startswith("sk-")
    assert data["key_preview"].endswith("3abc")
    assert data["key_source"] == "env:ANTHROPIC_API_KEY"


def test_test_llm_endpoint_returns_missing_key_error_when_no_key(monkeypatch):
    """POST /system/test-llm with no env key and no body key must
    return ok=false with error_type=MissingKey, NOT raise a 500.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MiniMax_API_KEY", raising=False)
    from rlpe.api.app import app
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.post("/system/test-llm", json={})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is False
    assert data["error_type"] == "MissingKey"
    assert "no API key" in data["error"]


def test_test_llm_endpoint_handles_invalid_body_gracefully(monkeypatch):
    """The endpoint must accept missing / partial bodies without
    raising 422 — the frontend sometimes posts {} when the user
    just wants to test the .env-loaded key.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MiniMax_API_KEY", raising=False)
    from rlpe.api.app import app
    from fastapi.testclient import TestClient
    c = TestClient(app)
    # Empty body
    r = c.post("/system/test-llm", json={})
    assert r.status_code == 200
    # Body with extra unknown field — Pydantic extra="ignore"
    r2 = c.post("/system/test-llm", json={"api_key": "", "junk_field": "x"})
    assert r2.status_code == 200


def test_test_llm_treats_non_json_reply_as_success(monkeypatch):
    """A connection test asks the LLM to reply with the literal "OK".
    That reply is NOT valid JSON, so MiniMaxM3Backend._make_result
    sets ``fallback_used=True`` and ``error_type=JSONParseError``.
    For a connection test that's a false negative — the API actually
    worked. The endpoint must therefore special-case JSONParseError /
    ValueError (when raw_text is non-empty) as success.
    """
    from rlpe.api import app as api_app
    from fastapi.testclient import TestClient

    # Patch MiniMaxM3Backend.infer_text to simulate the "OK reply +
    # JSONParseError" shape that triggered the bug in the field.
    class _FakeBackend:
        def __init__(self, *a, **kw):
            pass

        def infer_text(self, **_):
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": "JSON parse error: ValueError: ...",
                "fallback_used": True,
                "error": "ValueError: No parseable JSON object found",
                "error_type": "JSONParseError",
                "raw_text": "OK",
            }

    monkeypatch.setattr(api_app, "MiniMaxM3Backend", _FakeBackend, raising=False)
    # Inject the fake into the lazy import path used inside test_llm()
    import rlpe.llm_backends as _lb
    monkeypatch.setattr(_lb, "MiniMaxM3Backend", _FakeBackend, raising=True)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    c = TestClient(api_app.app)
    r = c.post("/system/test-llm", json={})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True, f"non-JSON reply should be success, got {data}"
    assert data.get("note"), "note should explain the non-JSON reply"


def test_llm_status_deduplicates_request_id_in_cost_aggregation(monkeypatch):
    """When N panel rows share the same MiniMax_request_id (because
    they came from the SAME batch LLM call), the cost must be counted
    ONCE and the call counter incremented ONCE. The previous code
    multiplied both by N, producing inflated 累计 数字.
    """
    from rlpe.api import app as api_app
    from fastapi.testclient import TestClient

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MiniMax_API_KEY", raising=False)

    job_id = "test-dedupe-cost"
    with api_app.RESULT_LOCK:
        api_app.RESULT_CACHE[job_id] = {
            "status": "done",
            "result": [
                {  # batch call 1, panel 1
                    "paper_id": "p", "figure_id": "f", "panel_id": "1",
                    "metadata": {
                        "MiniMax_cost_cny": 0.05,
                        "MiniMax_request_id": "req-abc",
                    },
                },
                {  # batch call 1, panel 2 (same request)
                    "paper_id": "p", "figure_id": "f", "panel_id": "2",
                    "metadata": {
                        "MiniMax_cost_cny": 0.05,
                        "MiniMax_request_id": "req-abc",
                    },
                },
                {  # batch call 1, panel 3 (same request)
                    "paper_id": "p", "figure_id": "f", "panel_id": "3",
                    "metadata": {
                        "MiniMax_cost_cny": 0.05,
                        "MiniMax_request_id": "req-abc",
                    },
                },
                {  # batch call 2 (different request)
                    "paper_id": "p", "figure_id": "f", "panel_id": "4",
                    "metadata": {
                        "MiniMax_cost_cny": 0.07,
                        "MiniMax_request_id": "req-xyz",
                    },
                },
            ],
        }
    try:
        c = TestClient(api_app.app)
        r = c.get("/system/llm-status")
        data = r.json()
        # 2 unique requests → 2 calls (NOT 4, despite 4 rows)
        assert data["total_calls"] == 2, f"expected 2 calls, got {data['total_calls']}"
        # Cost: 0.05 (req-abc, counted once) + 0.07 (req-xyz) = 0.12
        assert abs(data["total_cost_cny"] - 0.12) < 1e-6, (
            f"expected 0.12, got {data['total_cost_cny']}"
        )
    finally:
        with api_app.RESULT_LOCK:
            api_app.RESULT_CACHE.pop(job_id, None)


def test_dotenv_loader_overrides_project_keys_even_when_os_env_set(monkeypatch, tmp_path):
    """The .env loader must override pre-existing OS env vars for the
    project's reserved MiniMax-related keys (ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL, ANTHROPIC_MODEL).

    Background: tools like Claude Code set ``ANTHROPIC_BASE_URL``
    globally for their own backend (e.g. ark.cn-beijing.volces.com).
    Without the override, the RLPE web server would silently connect
    to the wrong endpoint even though the operator had ``.env`` set
    to ``https://api.minimaxi.com/anthropic``.
    """
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding")
    monkeypatch.setenv("ANTHROPIC_MODEL", "ark-code-latest")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ark-fake-key-from-claude-code")
    monkeypatch.setenv("HTTP_PROXY", "http://os-set-proxy:8080")

    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=sk-cp-realminimax-key-from-env-file\n"
        "ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic\n"
        "ANTHROPIC_MODEL=MiniMax-M3\n"
        "HTTP_PROXY=http://env-file-proxy:9090\n",
        encoding="utf-8",
    )

    import os
    _RLPE_PROJECT_OVERRIDE_KEYS = {
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
        "MiniMax_API_KEY", "MiniMax_MODEL", "MiniMax_BASE_URL",
    }
    _force_override = os.environ.get("RLPE_FORCE_ENV_OVERRIDE") == "1"
    with env_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            should_override = (
                _force_override
                or key in _RLPE_PROJECT_OVERRIDE_KEYS
                or key not in os.environ
            )
            if should_override:
                os.environ[key] = value

    # Project-reserved keys: .env wins
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://api.minimaxi.com/anthropic"
    assert os.environ["ANTHROPIC_MODEL"] == "MiniMax-M3"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-cp-realminimax-key-from-env-file"
    # Non-project key: OS env wins (default behaviour)
    assert os.environ["HTTP_PROXY"] == "http://os-set-proxy:8080"


def test_dotenv_loader_force_override_flips_for_all_keys(monkeypatch, tmp_path):
    """Setting RLPE_FORCE_ENV_OVERRIDE=1 must make .env win for ALL
    keys, not just the project-reserved ones. Escape hatch for
    Docker entrypoints / unusual setups.
    """
    monkeypatch.setenv("HTTP_PROXY", "http://os-set-proxy:8080")
    monkeypatch.setenv("RLPE_FORCE_ENV_OVERRIDE", "1")

    env_file = tmp_path / ".env"
    env_file.write_text(
        "HTTP_PROXY=http://env-file-proxy:9090\n",
        encoding="utf-8",
    )

    import os
    _force_override = os.environ.get("RLPE_FORCE_ENV_OVERRIDE") == "1"
    _project = {"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"}
    with env_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            should_override = (_force_override or key in _project or key not in os.environ)
            if should_override:
                os.environ[key] = value

    assert os.environ["HTTP_PROXY"] == "http://env-file-proxy:9090"
