"""Round 18 source-guard tests: non-specimen popup suppression.

When MiniMax M3 sees a figure that isn't a radiolarian specimen
(bar chart, table, map, publication-count graph), it returns a
deliberate refusal like:

    "该panel为图表（柱状/表格形式），展示1834至2007年放射虫相关
    出版物年度数量统计，并非放射虫标本显微图像。caption明确说明
    Fig.1为出版物数量统计图，不涉及具体属/种拉丁学名，也无字母
    标签对应物种。无标签与物种可判定。"

The previous code surfaced this as a MiniMaxAPIError popup,
forcing the operator to click through a no-op decision (retry /
rules / gemma4 / stop) for a figure the model correctly refused.

The fix:
  1. Backend: _collect_fallback_error_info flags the refusal with
     ``is_non_specimen_figure=True``.
  2. Backend: _apply_gemma_with_fallback short-circuits to a silent
     skip when that flag is set.
  3. Frontend: showMiniMaxFallbackModal has a defensive double-check
     that pattern-matches the same phrases in case a stale server
     doesn't send the flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(path: str) -> str:
    return Path(__file__).resolve().parents[1].joinpath(path).read_text(encoding="utf-8")


# --- 1) Backend: detection helper + flag ---------------------------


def test_non_specimen_helper_recognises_known_phrases():
    """``_looks_like_non_specimen_error`` must flag the standard
    refusal text that M3 returns for non-specimen figures."""
    from rlpe.pipeline import RadiolarianPipeline

    helper = RadiolarianPipeline._looks_like_non_specimen_error
    refusals = [
        # The exact message from the Suzuki 2011 Fig 1 failure
        "该panel为图表（柱状/表格形式），展示1834至2007年放射虫相关"
        "出版物年度数量统计，并非放射虫标本显微图像。无标签与物种可判定。",
        # Bare chart phrases
        "This is a bar chart of publication counts.",
        "This figure is a table of results.",
        "Image is a map of study area.",
        "is not a radiolarian specimen plate",
        "no panels found",
        # Sufficient markers
        "统计图 - 非标本",
    ]
    for r in refusals:
        assert helper(r, ""), (
            f"_looks_like_non_specimen_error should match: {r!r}"
        )


def test_non_specimen_helper_does_not_flag_real_errors():
    """Real API errors (rate limit, timeout, malformed JSON) must NOT
    be misclassified as non-specimen refusals."""
    from rlpe.pipeline import RadiolarianPipeline

    helper = RadiolarianPipeline._looks_like_non_specimen_error
    real_errors = [
        "anthropic.RateLimitError: 429 too many requests",
        "anthropic.APIConnectionError: connection refused",
        "JSON parse error: unexpected token at position 42",
        "Invalid API key",
        "Internal server error",
        "",  # empty
    ]
    for r in real_errors:
        assert not helper(r, ""), (
            f"_looks_like_non_specimen_error wrongly flagged real error: {r!r}"
        )


def test_collect_fallback_error_info_includes_non_specimen_flag():
    """``_collect_fallback_error_info`` must include
    ``is_non_specimen_figure`` in the returned dict so the popup
    handler can short-circuit."""
    src = _read("src/rlpe/pipeline.py")
    assert '"is_non_specimen_figure"' in src or "'is_non_specimen_figure'" in src, (
        "_collect_fallback_error_info no longer includes the "
        "is_non_specimen_figure flag in the returned error_info dict."
    )


# --- 2) Backend: short-circuit before popup ----------------------


def test_apply_gemma_with_fallback_silently_skips_non_specimen():
    """``_apply_gemma_with_fallback`` must check the non-specimen
    flag and skip silently — never call ``gemma_fallback_handler``
    (which would trigger the popup) for these cases."""
    src = _read("src/rlpe/pipeline.py")
    # Locate the _apply_gemma_with_fallback function
    idx = src.find("def _apply_gemma_with_fallback(")
    assert idx > 0
    # Take a wide window — the function body is ~150 lines.
    window = src[idx : idx + 8000]
    assert "is_non_specimen_figure" in window, (
        "_apply_gemma_with_fallback doesn't consult is_non_specimen_figure. "
        "Non-specimen figures will still trigger the popup."
    )
    # The check must happen BEFORE the gemma_fallback_handler call.
    flag_pos = window.find("is_non_specimen_figure")
    handler_pos = window.find("gemma_fallback_handler(error_info)")
    assert 0 < flag_pos, "is_non_specimen_figure not found in function body"
    assert 0 < handler_pos, "gemma_fallback_handler call not found"
    assert flag_pos < handler_pos, (
        "The is_non_specimen_figure check must happen BEFORE the "
        "gemma_fallback_handler() call. Otherwise the popup still fires."
    )


def test_non_specimen_skip_records_action_metadata():
    """When skipping a non-specimen figure, the match metadata must
    record ``MiniMax_fallback_action='skipped_non_specimen'`` so
    audit tools can see WHY a figure produced no rows."""
    src = _read("src/rlpe/pipeline.py")
    assert '"skipped_non_specimen"' in src or "'skipped_non_specimen'" in src, (
        "Pipeline doesn't stamp MiniMax_fallback_action='skipped_non_specimen' "
        "on matches that were silently dropped for non-specimen content."
    )


# --- 3) Frontend: defensive popup suppression ----------------------


def test_frontend_has_non_specimen_patterns():
    """``web/js/app.js`` must define a list of refusal patterns so
    a stale server build (no is_non_specimen_figure flag) doesn't
    push the popup for known refusal text."""
    src = _read("web/js/app.js")
    assert "_NON_SPECIMEN_REFUSAL_PATTERNS" in src, (
        "web/js/app.js is missing the _NON_SPECIMEN_REFUSAL_PATTERNS list. "
        "A stale server build could still surface non-specimen refusals "
        "as popups."
    )
    assert "looksLikeNonSpecimenRefusal" in src, (
        "web/js/app.js is missing the looksLikeNonSpecimenRefusal helper."
    )


def test_frontend_popup_checks_non_specimen_first():
    """``showMiniMaxFallbackModal`` must consult is_non_specimen_figure
    or the pattern helper BEFORE creating / showing the modal DOM."""
    src = _read("web/js/app.js")
    # Find the function
    idx = src.find("function showMiniMaxFallbackModal")
    assert idx > 0
    window = src[idx : idx + 1500]
    assert "is_non_specimen_figure" in window, (
        "showMiniMaxFallbackModal doesn't consult is_non_specimen_figure "
        "from the server's error_info."
    )
    assert "looksLikeNonSpecimenRefusal" in window, (
        "showMiniMaxFallbackModal doesn't run looksLikeNonSpecimenRefusal "
        "as a defensive fallback."
    )
    # Both checks must run BEFORE the modal DOM is created.
    server_flag_pos = window.find("is_non_specimen_figure")
    pattern_pos = window.find("looksLikeNonSpecimenRefusal")
    modal_create_pos = window.find("createElement")
    assert 0 < server_flag_pos < modal_create_pos, (
        "Server is_non_specimen_figure check must run before modal createElement."
    )
    assert 0 < pattern_pos < modal_create_pos, (
        "Pattern-matcher check must run before modal createElement."
    )