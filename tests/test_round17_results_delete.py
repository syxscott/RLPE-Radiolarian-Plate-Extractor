"""Round 17 source-guard tests: results-tab one-click + batch delete.

Locks in the new ``DELETE /results`` (clear all) and
``DELETE /results/batch`` endpoints plus the frontend wiring
(checkboxes + 一键删除 / 批量删除 buttons). Each test reads the
production source and verifies the contract so a future refactor
can't silently remove the feature.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(path: str) -> str:
    return Path(__file__).resolve().parents[1].joinpath(path).read_text(encoding="utf-8")


# --- 1) Backend endpoints exist with correct shape -----------------------


def test_delete_all_results_endpoint_exists():
    """``DELETE /results`` must exist and clear every done job's result list."""
    api_src = _read("src/rlpe/api/app.py")
    assert '@app.delete("/results")' in api_src, (
        "src/rlpe/api/app.py is missing @app.delete('/results') — the 一键删除 button would 404."
    )
    # The handler must touch RESULT_CACHE under the lock so concurrent
    # uploads don't race a clear.
    assert "RESULT_LOCK" in api_src.split('@app.delete("/results")')[1][:600], (
        "DELETE /results handler must acquire RESULT_LOCK before clearing each job's result list."
    )


def test_delete_results_batch_endpoint_exists():
    """``DELETE /results/batch`` must exist with a ``row_ids`` payload."""
    api_src = _read("src/rlpe/api/app.py")
    assert '@app.delete("/results/batch")' in api_src, (
        "src/rlpe/api/app.py is missing @app.delete('/results/batch') — "
        "the 批量删除 button would 404."
    )
    # Must accept a row_ids list.
    assert "row_ids" in api_src.split('@app.delete("/results/batch")')[1][:600], (
        "DELETE /results/batch must accept a row_ids payload."
    )


def test_row_id_field_in_result_record():
    """ResultRecord must declare ``row_id`` so the frontend can address rows."""
    api_src = _read("src/rlpe/api/app.py")
    # Find the ResultRecord class body
    idx = api_src.find("class ResultRecord")
    assert idx > 0
    end = api_src.find("\nclass ", idx + 1)
    body = api_src[idx:end]
    assert "row_id:" in body, (
        "ResultRecord is missing the row_id field. The frontend can't "
        "address rows for /results/batch DELETE without it."
    )


def test_row_id_synthesised_in_get_results():
    """get_results() must inject ``row_id`` into each returned row."""
    api_src = _read("src/rlpe/api/app.py")
    # The get_results function must call _row_id or set filtered["row_id"].
    assert 'filtered["row_id"]' in api_src, (
        "get_results() must synthesise row_id per row. Frontend receives "
        "rows without row_id if this is missing."
    )


# --- 2) Frontend HTML has buttons + checkbox column ---------------------


def test_results_delete_buttons_in_html():
    """The results toolbar must have 一键删除 + 批量删除 buttons + select-all."""
    html = _read("web/index.html")
    assert 'id="results-delete-all-btn"' in html, (
        "Missing 一键删除 button (id=results-delete-all-btn) in web/index.html."
    )
    assert 'id="results-delete-selected-btn"' in html, (
        "Missing 批量删除 button (id=results-delete-selected-btn) in web/index.html."
    )
    assert 'id="results-delete-selected-count"' in html, (
        "Missing selected-count span (id=results-delete-selected-count) in web/index.html."
    )
    assert 'id="results-select-all"' in html, (
        "Missing select-all checkbox (id=results-select-all) in web/index.html."
    )


def test_results_checkbox_column_in_table():
    """The results table header must include a checkbox column."""
    html = _read("web/index.html")
    assert 'class="col-check"' in html, (
        "Results table missing the checkbox column (class=col-check)."
    )
    # The placeholder colspan must be updated to 10 (was 8, +1 for
    # checkbox +1 for the Round 18 geology column).
    assert 'colspan="10"' in html, (
        "Empty-results placeholder colspan must be 10 (was 8 before "
        "the checkbox + geology columns were added)."
    )


# --- 3) Frontend JS wires the new controls ------------------------------


def test_results_delete_handlers_in_js():
    """app.js must define deleteAllResults + deleteSelectedResults + the init."""
    js = _read("web/js/app.js")
    for fn in ("deleteAllResults", "deleteSelectedResults", "initResultsDeleteButtons"):
        assert f"function {fn}" in js, f"web/js/app.js is missing function {fn}()."
    # The init must be called on DOMContentLoaded.
    assert "initResultsDeleteButtons()" in js, (
        "initResultsDeleteButtons() must be called on DOMContentLoaded."
    )


def test_results_delete_calls_correct_endpoints():
    """deleteAllResults → DELETE /results; deleteSelectedResults → DELETE /results/batch."""
    js = _read("web/js/app.js")
    # deleteAllResults → DELETE /results
    delete_all_block = js.split("async function deleteAllResults", 1)[1].split("async function", 1)[
        0
    ]
    assert 'method: "DELETE"' in delete_all_block or "method: 'DELETE'" in delete_all_block, (
        "deleteAllResults must call fetch with method DELETE"
    )
    assert "/results" in delete_all_block, "deleteAllResults must call DELETE /results"
    # deleteSelectedResults → DELETE /results/batch
    delete_sel_block = js.split("async function deleteSelectedResults", 1)[1].split(
        "async function", 1
    )[0]
    assert "/results/batch" in delete_sel_block, (
        "deleteSelectedResults must call DELETE /results/batch"
    )
    assert "row_ids" in delete_sel_block, (
        "deleteSelectedResults must include row_ids in the request body"
    )


def test_results_persistent_selection_set():
    """selectedResultRowIds must be a module-level Set so the user's
    selection survives search/filter re-renders."""
    js = _read("web/js/app.js")
    assert "let selectedResultRowIds" in js, (
        "selectedResultRowIds Set is missing — search/filter would silently "
        "drop the user's checkbox selection on every render."
    )


def test_load_results_prunes_stale_selection():
    """loadResults must prune row_ids that no longer exist in /results."""
    js = _read("web/js/app.js")
    block = js.split("async function loadResults")[1].split("\nasync function", 1)[0]
    assert "selectedResultRowIds" in block, (
        "loadResults() doesn't prune selectedResultRowIds. Rows deleted "
        "elsewhere (CLI, another tab) would accumulate in the set forever."
    )
