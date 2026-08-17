"""Phase 29 — GROBID retry + OD fallback regression suite.

Round 25 final 5-paper live test exposed a structural gap: GROBID
failure silently fell through to a visual-only stub path that
produced "Auto-generated figure for page N" for every panel. OD
failure had a GROBID fallback path, but GROBID failure had no OD
fallback. Phase 29 fixes this by:

1. Adding retry+backoff to ``GrobidClient.process_pdf`` (default
   3 attempts, exponential cap 30s).
2. Distinguishing error types (``connection_refused`` / ``timeout``
   / ``http_5xx`` / ``http_4xx`` / ``parse_error`` / ``unknown``).
3. When GROBID fails, the pipeline falls back to OpenDataLoader
   instead of the visual-only stub.
4. Adding ``--grobid-max-retries`` + ``--grobid-timeout`` CLI flags.

The scaffolding pattern follows ``test_round27_japanese_extraction.py``
and ``test_round28_caption_page_distance.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.config import PipelineConfig  # noqa: E402
from rlpe.grobid import GrobidClient, GrobidResult  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ============================================================================
# Source-guard tests
# ============================================================================


def test_grobid_max_retries_field_in_config_extras():
    """PipelineConfig extras whitelist must include ``grobid_max_retries``."""
    src = _read("src/rlpe/config.py")
    assert '"grobid_max_retries"' in src


def test_grobid_timeout_field_in_config_extras():
    """PipelineConfig extras whitelist must include ``grobid_timeout``."""
    src = _read("src/rlpe/config.py")
    assert '"grobid_timeout"' in src


def test_cli_grobid_max_retries_flag_exists():
    """``--grobid-max-retries`` CLI flag must be wired."""
    cli = _read("src/rlpe/cli.py")
    assert "--grobid-max-retries" in cli
    assert "args.grobid_max_retries" in cli


def test_cli_grobid_timeout_flag_exists():
    """``--grobid-timeout`` CLI flag must be wired."""
    cli = _read("src/rlpe/cli.py")
    assert "--grobid-timeout" in cli
    assert "args.grobid_timeout" in cli


def test_grobid_client_accepts_max_retries_kwarg():
    """``GrobidClient.__init__`` must accept the new kwargs."""
    import inspect

    params = list(inspect.signature(GrobidClient.__init__).parameters.keys())
    assert "max_retries" in params
    assert "retry_backoff" in params


def test_grobid_result_has_retry_count_and_error_type():
    """``GrobidResult`` dataclass must expose retry_count + error_type."""
    fields = GrobidResult.__dataclass_fields__.keys()
    assert "retry_count" in fields
    assert "error_type" in fields


def test_grobid_error_types_constant_defined():
    """``_ERROR_TYPES`` class constant lists all valid error categories."""
    assert "none" in GrobidClient._ERROR_TYPES
    assert "connection_refused" in GrobidClient._ERROR_TYPES
    assert "timeout" in GrobidClient._ERROR_TYPES
    assert "http_5xx" in GrobidClient._ERROR_TYPES
    assert "parse_error" in GrobidClient._ERROR_TYPES


def test_pipeline_forwards_retry_and_timeout_to_grobid_client():
    """pipeline.py must forward ``grobid_max_retries`` + ``grobid_timeout``
    from config.extra into the GrobidClient constructor."""
    src = _read("src/rlpe/pipeline.py")
    assert 'max_retries=int(self.config.extra.get("grobid_max_retries"' in src
    assert 'timeout=int(self.config.extra.get("grobid_timeout"' in src


def test_pipeline_od_fallback_block_present():
    """The OD-fallback path inside ``_process_one_pdf_grobid`` must exist."""
    src = _read("src/rlpe/pipeline.py")
    assert "self._process_one_pdf_od" in src
    assert "extraction_source" in src  # the row-tagging
    assert "od_after_grobid_failed" in src


def test_pipeline_cycle_guard_set_present():
    """The ``_grobid_in_progress`` set must be declared + used."""
    src = _read("src/rlpe/pipeline.py")
    assert "_grobid_in_progress" in src
    assert "_grobid_in_progress.add" in src
    assert "_grobid_in_progress.discard" in src


def test_pipeline_od_skips_recursive_grobid_call():
    """``_process_one_pdf_od`` must skip the recursive GROBID call when
    paper_id is already in ``_grobid_in_progress`` to break the cycle."""
    src = _read("src/rlpe/pipeline.py")
    assert "in self._grobid_in_progress" in src
    # Comment must explain the cycle
    assert "cycle" in src.lower()


# ============================================================================
# Behavioural tests — _classify_exception
# ============================================================================


def test_classify_exception_connection_refused():
    """requests.ConnectionError wraps a ConnectionRefusedError → 'connection_refused'."""
    import requests

    inner = ConnectionRefusedError(111, "Connection refused")
    outer = requests.ConnectionError("refused")
    outer.__cause__ = inner
    assert GrobidClient._classify_exception(outer) == "connection_refused"


def test_classify_exception_timeout():
    """requests.Timeout → 'timeout'."""
    import requests

    assert GrobidClient._classify_exception(requests.Timeout("timed out")) == "timeout"


def test_classify_exception_http_5xx():
    """requests.HTTPError with 500 → 'http_5xx'."""
    import requests

    resp = requests.Response()
    resp.status_code = 502
    err = requests.HTTPError("bad gateway")
    err.response = resp
    assert GrobidClient._classify_exception(err) == "http_5xx"


def test_classify_exception_http_4xx():
    """requests.HTTPError with 400 → 'http_4xx'."""
    import requests

    resp = requests.Response()
    resp.status_code = 404
    err = requests.HTTPError("not found")
    err.response = resp
    assert GrobidClient._classify_exception(err) == "http_4xx"


def test_classify_exception_parse_error():
    """xml.etree.ElementTree.ParseError → 'parse_error'."""
    import xml.etree.ElementTree as ET

    assert GrobidClient._classify_exception(ET.ParseError("bad xml")) == "parse_error"


def test_classify_exception_unknown():
    """Anything else → 'unknown'."""
    assert GrobidClient._classify_exception(ValueError("weird")) == "unknown"


# ============================================================================
# Behavioural tests — retry + backoff
# ============================================================================


def test_grobid_client_default_max_retries_is_3():
    """Default matches legacy single-attempt × 3 retry-loop behaviour."""
    assert GrobidClient().max_retries == 3


def test_grobid_client_default_timeout_is_300():
    """Default timeout matches legacy 300s."""
    assert GrobidClient().timeout == 300


def test_grobid_client_max_retries_clamped_to_at_least_1():
    """``max_retries=0`` must clamp to 1 (must attempt at least once)."""
    assert GrobidClient(max_retries=0).max_retries == 1


def test_grobid_result_default_retry_count_is_0():
    """Default value before any attempt is 0."""
    r = GrobidResult(
        paper_id="t",
        pdf_path=Path("/tmp/x.pdf"),
        tei_path=None,
        tei_xml=None,
        captions=[],
        fulltext_sections=[],
        success=True,
    )
    assert r.retry_count == 0
    assert r.error_type == "none"


def test_grobid_process_pdf_returns_retry_count_after_failure(monkeypatch):
    """When all retries fail, the GrobidResult carries ``retry_count=3``
    and the classified ``error_type``."""
    import requests

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fake_post)
    c = GrobidClient(
        server_url="http://localhost:1",
        timeout=1,
        max_retries=3,
        retry_backoff=0.0,  # no real sleep in tests
    )
    pdf = Path("/tmp/__nonexistent__.pdf")
    if not pdf.exists():
        pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    try:
        r = c.process_pdf(pdf, Path("/tmp/__grobid_out__"))
        assert r.success is False
        assert r.retry_count == 3
        assert r.error_type == "connection_refused"
    finally:
        pdf.unlink(missing_ok=True)


def test_grobid_process_pdf_succeeds_on_third_attempt(monkeypatch):
    """``retry_count=2`` when first two attempts fail and the third
    succeeds."""
    from unittest.mock import MagicMock

    import requests

    call_count = {"n": 0}

    def fake_post(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise requests.ConnectionError("transient")
        # third attempt: return a fake 200 with empty TEI
        resp = MagicMock()
        resp.text = (
            '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text><body/></text></TEI>'
        )
        resp.raise_for_status = lambda: None
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    c = GrobidClient(
        server_url="http://localhost:1",
        timeout=1,
        max_retries=3,
        retry_backoff=0.0,
    )
    pdf = Path("/tmp/__nonexistent2__.pdf")
    if not pdf.exists():
        pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    try:
        r = c.process_pdf(pdf, Path("/tmp/__grobid_out2__"))
        assert r.success is True
        assert r.retry_count == 2  # 0-indexed: 2 failed + 1 success
        assert r.error_type == "none"
        assert call_count["n"] == 3
    finally:
        pdf.unlink(missing_ok=True)


# ============================================================================
# Backward-compat
# ============================================================================


def test_pipelineconfig_legacy_grobid_url_still_works():
    """Default PipelineConfig has grobid_url unchanged."""
    cfg = PipelineConfig(pdf_dir=Path("/tmp"), work_dir=Path("/tmp"))
    assert cfg.grobid_url == "http://localhost:8070"


def test_disable_od_fallback_in_config_extras():
    """``disable_od_fallback`` opt-out must be in the whitelist."""
    src = _read("src/rlpe/config.py")
    assert '"disable_od_fallback"' in src
