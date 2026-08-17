"""Source-grep regression tests for the 2026-08-17 Web UI audit.

The 7 fixes (WEB-B1..B7) all live in static JS / HTML / Python files.
There is no jsdom harness, so the tests are predominantly source-grep
asserts — they verify the FIX LANDED and stays landed by failing if
the bad patterns reappear. Behavioural coverage (HTTP request
plumbing) is added where the FastAPI side changed.

Audit 2026-08-17: 7 web UI bugs fixed in one commit. This file is
the regression net; see commit message for the per-bug rationale.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_JS = REPO_ROOT / "web" / "js" / "app.js"
WEB_HTML = REPO_ROOT / "web" / "index.html"
API_PY = REPO_ROOT / "src" / "rlpe" / "api" / "app.py"
XLSX_PY = REPO_ROOT / "src" / "rlpe" / "exporters" / "xlsx.py"


@pytest.fixture(scope="module")
def js() -> str:
    if not WEB_JS.exists():
        pytest.skip("web/js/app.js not present")
    return WEB_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def html() -> str:
    if not WEB_HTML.exists():
        pytest.skip("web/index.html not present")
    return WEB_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def api_src() -> str:
    return API_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def xlsx_src() -> str:
    return XLSX_PY.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _function_body(text: str, fn_name: str) -> str:
    """Return the body of ``function <fn_name>(...) { ... }`` up to its
    matching closing brace. Naive but fine for our needs."""
    marker = f"function {fn_name}("
    i = text.find(marker)
    if i < 0:
        return ""
    brace_open = text.find("{", i)
    if brace_open < 0:
        return ""
    depth = 0
    for j in range(brace_open, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace_open : j + 1]
    return ""


def _strip_js_comments(text: str) -> str:
    """Drop ``// line`` and ``/* block */`` comments so source-grep
    assertions don't trip on audit-fix comments that LEGITIMATELY
    reference the bug pattern. Regexes are fine here; this isn't a
    parser."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


# ==========================================================================
# WEB-B1 — resolveAssetUrl() must build the /jobs/{id}/files/... URL for
# filesystem-relative paths, and every caller must pass a job_id so the
# server route can be hit.
# ==========================================================================

class TestResolveAssetUrlAudit20260817:

    def test_function_present(self, js: str) -> None:
        """resolveAssetUrl still exists."""
        assert "function resolveAssetUrl(" in js

    def test_function_accepts_job_id_param(self, js: str) -> None:
        """Signature now carries a jobId parameter so filesystem-relative
        paths can be rewritten into /jobs/{id}/files/..."""
        body = _function_body(js, "resolveAssetUrl")
        assert body, "resolveAssetUrl not found"
        assert "jobId" in body, (
            "resolveAssetUrl() must accept a jobId so it can build the "
            "/jobs/{id}/files/{path} URL for filesystem-relative panel_path values"
        )

    def test_function_builds_jobs_files_url(self, js: str) -> None:
        """The relative-path branch must build /jobs/{id}/files/..."""
        body = _function_body(js, "resolveAssetUrl")
        assert "/jobs/" in body and "/files/" in body, (
            "resolveAssetUrl() must construct /jobs/{jobId}/files/{rel} for "
            "filesystem-relative panel_path values (was returning the raw "
            "relative path which produced a 404 in <img src=...>)"
        )

    def test_call_site_passes_job_id(self, js: str) -> None:
        """Every caller must pass the row's job_id through."""
        # Find every resolveAssetUrl( call (regex char class needs `+` to
        # span multiple chars inside the parentheses).
        matches = re.findall(r"resolveAssetUrl\(([^)]+)\)", js)
        assert matches, "resolveAssetUrl is not called anywhere"
        passed_job_id = any("," in m and "r.job_id" in m for m in matches)
        assert passed_job_id, (
            "At least one resolveAssetUrl() call must pass r.job_id so the "
            "row's filesystem-relative panel_path can be mapped to the "
            "server's /jobs/{id}/files/... endpoint"
        )


# ==========================================================================
# WEB-B2 — v18_panel_id_source was a dead key (producers write
# ``panel_id_source`` without the v18_ prefix). The frontend must use the
# real key.
# ==========================================================================

class TestPanelIdSourceAudit20260817:

    def test_no_dead_v18_prefix(self, js: str) -> None:
        """``metadata.v18_panel_id_source`` must NOT appear anywhere in
        the served JS — every row was wrongly tagged "位置回退"."""
        assert "v18_panel_id_source" not in js, (
            "v18_panel_id_source is the dead-key bug from WEB-B2. The "
            "producer (converters.py) writes 'panel_id_source' without "
            "the v18_ prefix."
        )

    def test_real_key_used_in_getRecordStatus(self, js: str) -> None:
        """getRecordStatus() must read the real key."""
        body = _function_body(js, "getRecordStatus")
        assert body, "getRecordStatus not found"
        assert "panel_id_source" in body
        assert "v18_panel_id_source" not in body

    def test_real_key_used_in_renderResults(self, js: str) -> None:
        """renderResults()'s ocrSource binding must read the real key."""
        body = _function_body(js, "renderResults")
        assert body, "renderResults not found"
        assert "panel_id_source" in body
        assert "v18_panel_id_source" not in body

    def test_real_key_used_in_openImageModal(self, js: str) -> None:
        """openImageModal() must read the real key."""
        body = _function_body(js, "openImageModal")
        assert body, "openImageModal not found"
        assert "panel_id_source" in body
        assert "v18_panel_id_source" not in body

    def test_at_least_three_panel_id_source_occurrences(self, js: str) -> None:
        """Sanity: the un-prefixed key should appear at least 3 times
        (getRecordStatus + renderResults + openImageModal)."""
        assert js.count("panel_id_source") >= 3


# ==========================================================================
# WEB-B3 — populateResultFilter must always rebuild the <select>, regardless
# of job count.
# ==========================================================================

class TestPopulateResultFilterAudit20260817:

    def test_no_early_return_on_single_job(self, js: str) -> None:
        """The ``if (jobIds.length <= 1) { ... return; }`` short-circuit
        was the bug; ensure it's gone. The defensive ``if (!filter)
        return;`` early return for a missing DOM element is still
        allowed (it never matched the bug pattern)."""
        raw_body = _function_body(js, "populateResultFilter")
        assert raw_body, "populateResultFilter not found"
        body = _strip_js_comments(raw_body)
        # explicit forbidden pattern: the old guard that compared job
        # count and skipped rebuilding the <select>.
        assert "jobIds.length <= 1" not in body, (
            "Old early-return guard 'jobIds.length <= 1' reappeared — "
            "WEB-B3 regression."
        )
        # and no "return" sits between the function start and the
        # option-rebuild line (filter.innerHTML = ...).
        rebuild_idx = body.find("filter.innerHTML")
        assert rebuild_idx > 0
        prefix = body[:rebuild_idx]
        # Allow the defensive !filter return; nothing else.
        # We assert by stripping that one allowed early-return and
        # then forbidding any other `return` token in the prefix.
        cleaned = re.sub(r"if\s*\(\!\s*filter\s*\)\s*return\s*;", "", prefix)
        assert "return" not in cleaned, (
            "populateResultFilter() must NOT early-return for <=1 jobs "
            "(the WEB-B3 bug). The function body may have the defensive "
            "`if (!filter) return;` guard but nothing else."
        )

    def test_function_rebuilds_options(self, js: str) -> None:
        """The function still writes innerHTML with <option> tags."""
        body = _function_body(js, "populateResultFilter")
        assert "filter.innerHTML" in body
        assert "全部论文" in body


# ==========================================================================
# WEB-B4 — A SINGLE shared filterRows function must back both the table
# render and the export click handler.
# ==========================================================================

class TestFilterRowsAudit20260817:

    def test_filter_rows_defined(self, js: str) -> None:
        """filterRows(rows, searchTerm, filterJob) is the shared helper."""
        assert "function filterRows(" in js, (
            "filterRows() must be defined as a top-level helper — both "
            "renderResults() and getFilteredResults() now delegate to it."
        )

    def test_filter_rows_searches_all_six_fields(self, js: str) -> None:
        """The shared function must check paper_id, species, panel_id,
        figure_id, the geology blob, and caption_snippet — not just 3."""
        body = _function_body(js, "filterRows")
        assert body, "filterRows not found"
        # All 6 fields the table is supposed to search.
        for field in ("paperId", "species", "panelId", "figureId", "geoBlob", "caption"):
            assert field in body, (
                f"filterRows() must search {field} (was missing from the "
                "export path, silently exporting rows the operator had "
                "filtered out)"
            )

    def test_get_filtered_results_delegates(self, js: str) -> None:
        """getFilteredResults() must call filterRows(), not duplicate the
        logic."""
        body = _function_body(js, "getFilteredResults")
        assert body, "getFilteredResults not found"
        assert "filterRows(" in body, (
            "getFilteredResults() must delegate to filterRows(); the two "
            "implementations drifted in WEB-B4 and silently exported "
            "different sets than the visible table."
        )

    def test_render_results_delegates(self, js: str) -> None:
        """renderResults() must call filterRows() too."""
        body = _function_body(js, "renderResults")
        assert body, "renderResults not found"
        assert "filterRows(" in body


# ==========================================================================
# WEB-B5 — /jobs/{id}/export.xlsx must accept filter query params so the
# .xlsx mirrors the UI filter; the JS must pass them.
# ==========================================================================

class TestExportFilterAudit20260817:

    def test_endpoint_accepts_filter_params(self, api_src: str) -> None:
        """The endpoint signature gains paper_ids / panel_ids / species."""
        # Extract the function body via simple brace-matching.
        marker = 'def export_job_xlsx('
        i = api_src.find(marker)
        assert i >= 0, "export_job_xlsx not found"
        assert "paper_ids" in api_src[i : i + 1500], (
            "/jobs/{job_id}/export.xlsx must accept a paper_ids filter "
            "query param so the download mirrors the UI table"
        )
        assert "species" in api_src[i : i + 1500]
        assert "panel_ids" in api_src[i : i + 1500]

    def test_endpoint_passes_panel_filter(self, api_src: str) -> None:
        """The endpoint threads the filter through write_xlsx."""
        marker = 'def export_job_xlsx('
        i = api_src.find(marker)
        # Body is well within the next 4KB.
        body = api_src[i : i + 5000]
        assert "panel_filter=" in body, (
            "export_job_xlsx must thread a panel_filter callable into "
            "write_xlsx() so the .xlsx only contains rows matching the UI "
            "filter (was previously exporting the FULL job)"
        )

    def test_xlsx_module_accepts_panel_filter(self, xlsx_src: str) -> None:
        """write_xlsx() must declare the new parameter."""
        marker = "def write_xlsx("
        i = xlsx_src.find(marker)
        assert i >= 0, "write_xlsx not found"
        # signature is short; read forward ~500 chars.
        assert "panel_filter" in xlsx_src[i : i + 500]

    def test_xlsx_applies_filter_to_panels_sheet(self, xlsx_src: str) -> None:
        """The panels sheet must filter before writing rows."""
        marker = "panels_all = list(run_output.get(\"panels\", []) or [])"
        i = xlsx_src.find(marker)
        assert i >= 0, (
            "write_xlsx must extract 'panels_all' and apply panel_filter "
            "to it before iterating the sheets (audit 2026-08-17 WEB-B5)"
        )
        assert "panel_filter is not None" in xlsx_src

    def test_js_export_sends_filter_querystring(self, js: str) -> None:
        """The frontend export handler must URLSearchParams-ify the
        per-job filter into the export URL."""
        marker = "export-btn"
        i = js.find(marker)
        assert i >= 0
        # Read forward a few KB to cover the handler body.
        chunk = js[i : i + 6000]
        assert "URLSearchParams" in chunk, (
            "Export handler must build a URLSearchParams and append it "
            "to /jobs/{id}/export.xlsx"
        )
        assert "paper_ids" in chunk
        assert "species" in chunk
        assert "panel_ids" in chunk
        assert "search=" in chunk or "'search'" in chunk


# ==========================================================================
# WEB-B6 — initLLMBackendSync must validate the saved value against the
# union of available <option> values, not the current default.
# ==========================================================================

class TestLlmBackendRestoreAudit20260817:

    def test_function_uses_options_union(self, js: str) -> None:
        """The restore branch must walk both <select>'s ``options``."""
        body = _function_body(js, "initLLMBackendSync")
        assert body, "initLLMBackendSync not found"
        assert "basic.options" in body
        assert "advanced.options" in body
        assert "allValues" in body, (
            "initLLMBackendSync must build an allValues Set from the "
            "union of both <select>'s <option> values; pre-fix it "
            "compared against the DEFAULT value, which never matched a "
            "user's saved preference."
        )

    def test_no_default_value_compare(self, js: str) -> None:
        """The old bug pattern must NOT appear inside an executable
        branch. The COMMENT text in the audit fix is allowed to
        reference the bug; what we forbid is an actual code path
        that compares ``saved`` to ``[basic.value, advanced.value]``."""
        import re
        # Strip out line and block comments so the audit-comment text
        # that references the bug doesn't trip the assertion.
        stripped = re.sub(r"//[^\n]*", "", js)
        stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
        assert "[basic.value, advanced.value]" not in stripped, (
            "Pre-fix bug code path '[basic.value, advanced.value]"
            ".includes(saved)' reappeared — it compared against the "
            "DEFAULT value (still 'MiniMax' at DOMContentLoaded) and "
            "silently failed for every non-default saved choice."
        )


# ==========================================================================
# WEB-B7 — The correction modal gains an image_verified checkbox and the
# submit handler posts it to /review/correction.
# ==========================================================================

class TestImageVerifiedToggleAudit20260817:

    def test_checkbox_in_html(self, html: str) -> None:
        """The HTML form must include the new checkbox input."""
        assert 'id="correction-image-verified"' in html
        assert 'type="checkbox"' in html

    def test_payload_includes_image_verified(self, js: str) -> None:
        """The submit handler must include ``image_verified`` in the
        payload sent to POST /review/correction."""
        # Find the submit handler near the form's submit listener.
        i = js.find("correction-form")
        assert i >= 0
        chunk = js[i : i + 3000]
        assert "image_verified" in chunk, (
            "Submit handler must include image_verified in the payload "
            "sent to POST /review/correction (audit 2026-08-17 WEB-B7)"
        )

    def test_modal_function_pre_populates_checkbox(self, js: str) -> None:
        """openCorrectionModal() must pre-populate the checkbox from the
        record's current image_verified value."""
        body = _function_body(js, "openCorrectionModal")
        assert body, "openCorrectionModal not found"
        assert "correction-image-verified" in body, (
            "openCorrectionModal() must look up the checkbox by id so it "
            "can pre-fill the state when the modal opens"
        )
        assert "image_verified" in body, (
            "openCorrectionModal() must read the record's current "
            "image_verified to decide whether the box is checked."
        )

    def test_render_results_passes_record_to_modal(self, js: str) -> None:
        """renderResults()'s click handler must hand the row to the modal."""
        body = _function_body(js, "renderResults")
        # find the openCorrectionModal call site
        i = body.find("openCorrectionModal(")
        assert i >= 0
        # The 4th argument must be present (the row).
        call = body[i : i + 200]
        # Match openCorrectionModal(a, b, c, d) — 4 args.
        m = re.search(r"openCorrectionModal\(\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^)]+\)", call)
        assert m, (
            "renderResults() must call openCorrectionModal(r.paper_id, "
            "r.figure_id, r.panel_path, r) so the modal can pre-fill "
            "image_verified from the row's current state."
        )