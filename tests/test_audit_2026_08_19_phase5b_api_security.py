"""Regression tests for audit 2026-08-19 phase 5b — FastAPI security hardening.

Covers five real issues an audit found in ``rlpe.api.app``:

* M-1 — Traceback / site-packages path leak: when an endpoint raises
  an uncaught ``Exception``, the response must not contain the
  Python traceback, file paths, line numbers, or module names. The
  fix installs a global ``@app.exception_handler(Exception)`` that
  returns a sanitised ``{"detail": "Internal server error",
  "type": "<ExceptionTypeName>"}`` body and logs the full
  traceback server-side via ``logger.exception``.

* M-2 — Upload size cap: the default cap was lowered from 256 MB
  to 100 MB, configurable via ``RLPE_MAX_UPLOAD_MB``. Streaming
  + hard cap remain in place. Oversized uploads must return 413
  both via the chunked read loop AND via an explicit
  ``Content-Length`` header.

* M-3 — Content-Disposition header injection (CWE-93): the
  ``paper_id`` is sanitised through ``re.sub(r"[^\\w.\\-]", "_", ...)``
  before being spliced into ``Content-Disposition``. A paper_id
  containing ``\\r\\n`` (CR/LF) must NOT inject a second header,
  and the filename portion must contain only safe characters.

* M-4 — API key auth: when ``RLPE_API_KEY`` is set in the server
  environment, every state-changing endpoint requires the same
  string in the ``X-API-Key`` header. Wrong / missing keys get a
  403 with ``WWW-Authenticate: ApiKey``. When the env var is NOT
  set, the dependency is a no-op so existing same-origin SPA
  flows keep working.

* M-5 — Security response headers: every response carries
  ``X-Content-Type-Options: nosniff``, ``X-Frame-Options: DENY``,
  and ``Referrer-Policy: no-referrer``.

These tests are pure white-box; they don't run a real PDF pipeline
or talk to GROBID / EasyOCR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ``app.py`` reads ``RLPE_API_TEST_TMP`` at import time to redirect
# the upload / work directories; we set that env var inside a fixture
# AFTER clearing the module from ``sys.modules``. The patterns here
# mirror tests/test_audit_2026_08_19_phase1e_api.py.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Import rlpe.api.app with a fresh ``RLPE_API_TEST_TMP`` and a
    clean (no) ``RLPE_API_KEY`` env var so auth is disabled by default
    for tests that aren't explicitly testing auth."""
    monkeypatch.setenv("RLPE_API_TEST_TMP", str(tmp_path))
    monkeypatch.delenv("RLPE_API_KEY", raising=False)
    for mod_name in list(sys.modules):
        if mod_name == "rlpe.api.app" or mod_name.startswith("rlpe.api."):
            sys.modules.pop(mod_name, None)
    import rlpe.api.app as app_mod  # noqa: E402

    return app_mod


@pytest.fixture
def client(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


@pytest.fixture
def authed_app_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Import rlpe.api.app with ``RLPE_API_KEY=secret123`` so the
    sensitive endpoints require the matching ``X-API-Key`` header."""
    monkeypatch.setenv("RLPE_API_TEST_TMP", str(tmp_path))
    monkeypatch.setenv("RLPE_API_KEY", "secret123")
    for mod_name in list(sys.modules):
        if mod_name == "rlpe.api.app" or mod_name.startswith("rlpe.api."):
            sys.modules.pop(mod_name, None)
    import rlpe.api.app as app_mod  # noqa: E402

    return app_mod


@pytest.fixture
def authed_client(authed_app_module):
    from fastapi.testclient import TestClient

    return TestClient(authed_app_module.app)


# ---------------------------------------------------------------------------
# M-1 — Traceback / file path leak in 500 responses
# ---------------------------------------------------------------------------
class TestTracebackNotLeaked:
    """The global ``@app.exception_handler(Exception)`` must catch
    any uncaught exception and return a sanitised payload. The
    response body must not contain a Python traceback, file paths,
    or line numbers."""

    def test_unhandled_exception_returns_sanitised_500(
        self, app_module: object
    ) -> None:
        """Inject a route that raises ``RuntimeError("boom")``, hit
        it, and assert the response is 500 with only a sanitised
        body — no traceback, no file paths, no module names."""
        from fastapi.testclient import TestClient

        app = app_module.app  # type: ignore[attr-defined]

        @app.get("/__test/boom")
        def _boom():  # noqa: ANN202 — intentional raise
            raise RuntimeError("boom — sensitive internal detail")

        # ``raise_server_exceptions=False`` so the TestClient
        # returns the actual 500 response rather than re-raising
        # in the test (Starlette's default is True for ease of
        # debugging, but we want to assert on the wire response).
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/__test/boom")
        assert r.status_code == 500
        body_text = r.text
        # Forbidden substrings (case-sensitive).
        for needle in ("Traceback", "File \"", "site-packages", "/__test/", "RuntimeError: boom"):
            assert needle not in body_text, (
                f"Traceback leak detected (needle={needle!r}): {body_text[:400]}"
            )
        # Sanitised payload contract.
        body = r.json()
        assert body.get("detail") == "Internal server error"
        # The exception TYPE name is allowed (operators need it to
        # correlate with log lines), but the exception MESSAGE is not.
        assert body.get("type") == "RuntimeError"

    def test_value_error_message_not_leaked(self, app_module: object) -> None:
        """An exception whose ``str()`` contains a file path must
        not have the path returned to the client."""
        from fastapi.testclient import TestClient

        app = app_module.app  # type: ignore[attr-defined]

        @app.get("/__test/leak_path")
        def _leak_path():  # noqa: ANN202 — intentional raise
            # Path that looks like ``/Users/.../site-packages/...``
            raise ValueError(
                "Could not open /Users/alice/.venv/lib/python3.13/site-packages/foo.py"
            )

        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/__test/leak_path")
        assert r.status_code == 500
        body_text = r.text
        assert "/Users/alice" not in body_text
        assert "site-packages" not in body_text
        assert ".venv" not in body_text
        # And the type is preserved.
        assert r.json().get("type") == "ValueError"


# ---------------------------------------------------------------------------
# M-2 — Upload size cap (default 100 MB, env-tunable)
# ---------------------------------------------------------------------------
class TestUploadSizeCap:
    """Phase 5B: default cap lowered from 256 MB to 100 MB.
    Operators can raise it via ``RLPE_MAX_UPLOAD_MB``. Oversized
    uploads must return 413 both via the chunked read loop AND via
    an explicit ``Content-Length`` header."""

    def _make_pdf(self, tmp_path: Path, size_bytes: int) -> Path:
        path = tmp_path / "tiny.pdf"
        path.write_bytes(b"%PDF-1.4\n" + b"%pad\n" * (size_bytes // 5 + 1))
        with path.open("r+b") as f:
            f.truncate(size_bytes)
        return path

    def test_default_cap_is_100_mb(self, app_module: object) -> None:
        """The constant defaults to 100 MB (audit 2026-08-19 phase 5b)."""
        assert app_module.MAX_UPLOAD_SIZE_MB == 100, (
            f"default upload cap should be 100 MB, got {app_module.MAX_UPLOAD_SIZE_MB}"
        )

    def test_env_override_lowers_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``RLPE_MAX_UPLOAD_MB`` overrides the default."""
        monkeypatch.setenv("RLPE_MAX_UPLOAD_MB", "50")
        for mod_name in list(sys.modules):
            if mod_name == "rlpe.api.app" or mod_name.startswith("rlpe.api."):
                sys.modules.pop(mod_name, None)
        import rlpe.api.app as app_mod  # noqa: E402

        assert app_mod.MAX_UPLOAD_SIZE_MB == 50

    def test_50mb_upload_succeeds(
        self, app_module: object, client, tmp_path: Path
    ) -> None:
        """A 50 MB PDF must upload successfully under the 100 MB
        default cap."""
        # Force the cap to a known value to keep the test fast.
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(app_module, "MAX_UPLOAD_SIZE_MB", 100)
        try:
            pdf = self._make_pdf(tmp_path, 50 * 1024 * 1024)
            with pdf.open("rb") as f:
                r = client.post(
                    "/jobs/upload",
                    files={"file": (pdf.name, f, "application/pdf")},
                )
            assert r.status_code in (200, 202), r.text[:500]
        finally:
            monkeypatch.undo()

    def test_200mb_upload_rejected_with_413(
        self, app_module: object, client, tmp_path: Path
    ) -> None:
        """A 200 MB PDF must be rejected with 413 under the 100 MB
        default cap — BEFORE the full body is buffered (the chunked
        read loop must short-circuit)."""
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(app_module, "MAX_UPLOAD_SIZE_MB", 100)
        try:
            pdf = self._make_pdf(tmp_path, 200 * 1024 * 1024)
            with pdf.open("rb") as f:
                r = client.post(
                    "/jobs/upload",
                    files={"file": (pdf.name, f, "application/pdf")},
                )
            assert r.status_code == 413, (
                f"expected 413 for 200 MB upload, got {r.status_code}: {r.text[:300]}"
            )
        finally:
            monkeypatch.undo()


# ---------------------------------------------------------------------------
# M-3 — Content-Disposition header injection (CWE-93)
# ---------------------------------------------------------------------------
class TestContentDispositionSecurity:
    """The export endpoint splices ``paper_id`` into a
    ``Content-Disposition: attachment; filename="..."`` header.
    A paper_id containing CR/LF (``\\r\\n``) must NOT inject a
    second header. The whitelist ``[\\w.-]`` must scrub any
    forbidden character."""

    @pytest.fixture
    def setup(self, app_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Stub the xlsx writer + seed a done job whose paper_id
        is the malicious payload."""
        from rlpe.api.app import RESULT_CACHE, RESULT_LOCK

        RESULT_CACHE.clear()

        def _fake_write_xlsx(run_output, panel_filter=None):  # noqa: ARG001
            return b"X"

        try:
            import rlpe.exporters.xlsx as xlsx_mod

            monkeypatch.setattr(xlsx_mod, "write_xlsx", _fake_write_xlsx)
        except Exception:
            pass

        jid = "cwe93-test"
        malicious = "abc\r\nSet-Cookie: x=1"
        panels: list[dict] = [{"paper_id": malicious, "panel_id": "p1"}]
        with RESULT_LOCK:
            RESULT_CACHE[jid] = {
                "status": "done",
                "result": panels,
                "error": None,
                "detail": None,
                "created_at": "2026-08-19T00:00:00",
                "filename": "x.pdf",
                "progress": 100,
            }
        return jid

    def test_crlf_paper_id_does_not_inject_second_header(
        self, app_module, setup: str, client
    ) -> None:
        """paper_id with ``\\r\\n`` must NOT introduce a
        ``Set-Cookie`` header. The Content-Disposition filename
        must contain only whitelist characters."""
        r = client.get(f"/jobs/{setup}/export.xlsx")
        assert r.status_code == 200, r.text

        # 1. Set-Cookie must NOT be present in any form.
        set_cookie = r.headers.get("set-cookie")
        assert set_cookie is None, (
            f"CR/LF injection succeeded — Set-Cookie header present: {set_cookie!r}"
        )

        # 2. Content-Disposition must be a single, valid header.
        cd = r.headers.get("content-disposition", "")
        assert cd, "Content-Disposition missing"
        assert "\r" not in cd, f"CR in Content-Disposition: {cd!r}"
        assert "\n" not in cd, f"LF in Content-Disposition: {cd!r}"

        # 3. Filename portion only contains safe characters.
        assert 'filename="' in cd
        start = cd.index('filename="') + len('filename="')
        end = cd.index('"', start)
        filename = cd[start:end]
        # No control characters, no whitespace, no path separators.
        for forbidden in ("\r", "\n", " ", "/", "\\", ";", ":", '"', "'"):
            assert forbidden not in filename, (
                f"forbidden char {forbidden!r} in filename: {filename!r}"
            )
        # Filename must start with ``rlpe_`` and end with ``.xlsx``.
        assert filename.startswith("rlpe_"), filename
        assert filename.endswith(".xlsx"), filename

    def test_raw_headers_in_response_have_no_crlf(
        self, app_module, setup: str, client
    ) -> None:
        """Iterate ``response.headers.raw`` and assert every header
        line is single-line (no embedded CR/LF). The HTTP spec
        forbids header splitting as it enables response-splitting
        attacks (CWE-93)."""
        r = client.get(f"/jobs/{setup}/export.xlsx")
        assert r.status_code == 200
        # ``response.headers.raw`` is a list of (bytes, bytes) tuples
        # in httpx; we check both header name and value.
        for name, value in r.headers.raw:
            assert b"\r" not in name and b"\n" not in name, (
                f"CR/LF in header name: {name!r}"
            )
            assert b"\r" not in value and b"\n" not in value, (
                f"CR/LF in header value ({name!r}): {value!r}"
            )


# ---------------------------------------------------------------------------
# M-4 — API key auth (optional, env-gated)
# ---------------------------------------------------------------------------
class TestApiKeyAuth:
    """When ``RLPE_API_KEY`` is set in the server environment, every
    state-changing endpoint must require the matching ``X-API-Key``
    header. Wrong / missing keys get a 403 with a ``WWW-Authenticate:
    ApiKey`` response header. When the env var is NOT set, the
    dependency is a no-op."""

    def test_no_key_when_env_unset_allows_request(
        self, app_module, client
    ) -> None:
        """With ``RLPE_API_KEY`` unset, the same-origin SPA workflow
        (POST /review/correction) must succeed without any header."""
        payload = {
            "paper_id": "test_paper",
            "figure_id": "test_figure",
            "corrected_species": "Genus species",
        }
        r = client.post("/review/correction", json=payload)
        assert r.status_code in (200, 202, 404), (
            f"expected success without auth, got {r.status_code}: {r.text[:300]}"
        )

    def test_missing_x_api_key_returns_403(
        self, authed_app_module, authed_client
    ) -> None:
        """With ``RLPE_API_KEY=secret123`` set, a POST without the
        ``X-API-Key`` header must return 403."""
        payload = {
            "paper_id": "test_paper",
            "figure_id": "test_figure",
            "corrected_species": "Genus species",
        }
        r = authed_client.post("/review/correction", json=payload)
        assert r.status_code == 403, (
            f"expected 403, got {r.status_code}: {r.text[:300]}"
        )
        assert r.headers.get("www-authenticate") == "ApiKey"

    def test_wrong_x_api_key_returns_403(
        self, authed_app_module, authed_client
    ) -> None:
        """With ``RLPE_API_KEY=secret123`` set, a POST with the
        wrong ``X-API-Key`` header must return 403."""
        payload = {
            "paper_id": "test_paper",
            "figure_id": "test_figure",
            "corrected_species": "Genus species",
        }
        r = authed_client.post(
            "/review/correction",
            json=payload,
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 403, (
            f"expected 403, got {r.status_code}: {r.text[:300]}"
        )

    def test_correct_x_api_key_returns_success(
        self, authed_app_module, authed_client
    ) -> None:
        """With ``RLPE_API_KEY=secret123`` set, a POST with the
        matching ``X-API-Key`` header must succeed."""
        payload = {
            "paper_id": "test_paper",
            "figure_id": "test_figure",
            "corrected_species": "Genus species",
        }
        r = authed_client.post(
            "/review/correction",
            json=payload,
            headers={"X-API-Key": "secret123"},
        )
        assert r.status_code in (200, 202, 404), (
            f"expected success with valid key, got {r.status_code}: {r.text[:300]}"
        )

    def test_upload_endpoint_requires_auth(
        self, authed_app_module, authed_client, tmp_path: Path
    ) -> None:
        """The /jobs/upload endpoint must also be auth-protected
        when the env var is set — the operator's paid MiniMax key
        is the highest-value target."""
        pdf = tmp_path / "tiny.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%pad\n")
        # No header → 403.
        with pdf.open("rb") as f:
            r = authed_client.post(
                "/jobs/upload",
                files={"file": (pdf.name, f, "application/pdf")},
            )
        assert r.status_code == 403, (
            f"expected 403 for upload without key, got {r.status_code}"
        )

        # With the right header → 200/202.
        with pdf.open("rb") as f:
            r = authed_client.post(
                "/jobs/upload",
                files={"file": (pdf.name, f, "application/pdf")},
                headers={"X-API-Key": "secret123"},
            )
        assert r.status_code in (200, 202), (
            f"expected success with valid key, got {r.status_code}: {r.text[:300]}"
        )

    def test_read_endpoint_does_not_require_auth(
        self, authed_app_module, authed_client
    ) -> None:
        """GET endpoints stay open — the SPA fetches them from
        loopback without an auth header."""
        r = authed_client.get("/health")
        assert r.status_code == 200, (
            f"GET /health should not require auth, got {r.status_code}"
        )
        r = authed_client.get("/system/info")
        assert r.status_code == 200

    def test_require_api_key_uses_constant_time_compare(
        self, app_module: object
    ) -> None:
        """The dependency must use ``hmac.compare_digest`` so the
        check doesn't leak length / position via timing. This is a
        source-level assertion — cheap and deterministic."""
        import inspect

        src = inspect.getsource(app_module.require_api_key)
        assert "hmac.compare_digest" in src, (
            "require_api_key should use hmac.compare_digest for "
            "constant-time comparison"
        )


# ---------------------------------------------------------------------------
# M-5 — Security response headers
# ---------------------------------------------------------------------------
class TestSecurityHeaders:
    """Every response must carry ``X-Content-Type-Options: nosniff``,
    ``X-Frame-Options: DENY``, and ``Referrer-Policy: no-referrer``."""

    def test_health_response_has_security_headers(self, client) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.headers.get("x-content-type-options") == "nosniff", (
            f"missing nosniff header: {dict(r.headers)!r}"
        )
        assert r.headers.get("x-frame-options") == "DENY", (
            f"missing DENY frame-options: {dict(r.headers)!r}"
        )
        assert r.headers.get("referrer-policy") == "no-referrer", (
            f"missing no-referrer: {dict(r.headers)!r}"
        )

    def test_error_response_has_security_headers(self, client) -> None:
        """Even on a 404 the security headers must be present."""
        r = client.get("/nonexistent-path-for-test")
        assert r.status_code in (404, 405)
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert r.headers.get("x-frame-options") == "DENY"
        assert r.headers.get("referrer-policy") == "no-referrer"

    def test_500_response_has_security_headers(self, app_module: object) -> None:
        """A 500 from the global exception handler must STILL carry
        the security headers — the middleware runs around all
        responses, including those from the exception handler."""
        from fastapi.testclient import TestClient

        app = app_module.app  # type: ignore[attr-defined]

        @app.get("/__test/headers-on-500")
        def _boom():  # noqa: ANN202
            raise RuntimeError("kaboom")

        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/__test/headers-on-500")
        assert r.status_code == 500
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert r.headers.get("x-frame-options") == "DENY"
        assert r.headers.get("referrer-policy") == "no-referrer"
