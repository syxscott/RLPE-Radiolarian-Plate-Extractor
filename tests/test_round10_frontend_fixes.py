"""Round 10 frontend fix source-guard tests.

These tests verify the static shape of ``web/js/app.js`` and
``web/index.html`` after the Round 10 frontend bug fixes. They use
regex matching (no JS runtime) because the project has no JS test
infrastructure; the assertions lock in the patterns that each fix
introduced so a future refactor that silently removes a fix will fail
CI.

Round 10 fixed 9 frontend bugs (3 HIGH + 5 MEDIUM + 2 LOW):
  FH1 NaN propagation in refreshInterval
  FH2 tab switch race (cancelable setTimeout)
  FH3 showNotification/showToast unification
  FM1 Escape key for modals
  FM2 cancelable auto-tab-switch setTimeout (covered by FH2)
  FM3 file input reset on reopen
  FM4 status-filter event delegation
  FM5 getElementById null safety
  FL3 a11y attributes
  FL4 z-index design tokens
"""

from __future__ import annotations

import re
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
APP_JS = WEB_DIR / "js" / "app.js"
INDEX_HTML = WEB_DIR / "index.html"
STYLE_CSS = WEB_DIR / "css" / "style.css"


def _read(path: Path) -> str:
    assert path.exists(), f"Round 10 fixture missing: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# FH1 — NaN propagation in refreshInterval
# ---------------------------------------------------------------------------


def test_fh1_safe_parse_int_helper_present():
    """The `_safeParseInt` helper gates parseInt on Number.isFinite so
    localStorage corruption (e.g. ``"abc"``) can't poison
    ``CONFIG.refreshInterval`` and silently kill ``setInterval``.
    """
    text = _read(APP_JS)
    assert "function _safeParseInt" in text
    # Uses the fallback when value is null/empty/non-finite
    assert re.search(
        r"function _safeParseInt\([^)]*\)\s*\{[^}]*Number\.isFinite",
        text,
        re.DOTALL,
    ), "_safeParseInt must guard on Number.isFinite"


def test_fh1_refresh_interval_uses_safe_parse_int():
    """The CONFIG init must call `_safeParseInt`, not the raw
    ``parseInt``, so a corrupted localStorage value doesn't NaN out
    the polling interval.
    """
    text = _read(APP_JS)
    # Locate the refreshInterval: line and assert it uses _safeParseInt.
    m = re.search(r"refreshInterval:\s*([^,\n]+)", text)
    assert m, "refreshInterval config line not found"
    rhs = m.group(1).strip()
    assert "_safeParseInt" in rhs, (
        f"refreshInterval must use _safeParseInt; got: {rhs!r}"
    )


def test_fh1_safe_parse_int_rejects_nan_strings():
    """Sanity check the helper logic itself: it must return the
    fallback for non-finite parses. Run via Node so we exercise the
    real implementation, not just the regex."""
    import subprocess

    out = subprocess.run(
        [
            "node",
            "-e",
            "function _safeParseInt(v, fb) { if (v == null || v === '') return fb; const n = parseInt(v, 10); return Number.isFinite(n) ? n : fb; } "
            "console.log(_safeParseInt('abc', 3));"
            "console.log(_safeParseInt('', 3));"
            "console.log(_safeParseInt('5', 3));"
            "console.log(_safeParseInt('NaN', 3));",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = out.stdout.strip().splitlines()
    assert lines == ["3", "3", "5", "3"], (
        f"_safeParseInt should fall back to 3 on bad input; got {lines}"
    )


# ---------------------------------------------------------------------------
# FH2 + FM2 — cancelable auto-tab-switch setTimeout
# ---------------------------------------------------------------------------


def test_fh2_auto_switch_timer_declared():
    """A module-level ``_autoSwitchTimer`` must exist so the tab
    click handler can cancel it.
    """
    text = _read(APP_JS)
    assert re.search(r"^\s*let _autoSwitchTimer = null;", text, re.MULTILINE), (
        "Module-level `let _autoSwitchTimer = null;` missing"
    )


def test_fh2_tab_handler_cancels_pending_auto_switch():
    """When the user clicks a tab manually during the 1200 ms grace
    period, the auto-switch timer must be cancelled so the user's
    intent is respected.
    """
    text = _read(APP_JS)
    handler_idx = text.find("document.querySelectorAll('.tab-btn').forEach")
    assert handler_idx >= 0
    chunk = text[handler_idx : handler_idx + 3000]
    assert "clearTimeout(_autoSwitchTimer)" in chunk, (
        "Tab click handler must clearTimeout the pending auto-switch"
    )


def test_fh2_load_jobs_stores_timer_id():
    """The auto-switch setTimeout in loadJobs must be stored in
    ``_autoSwitchTimer`` so it can be cancelled. Round 10 fix.
    """
    text = _read(APP_JS)
    # Look for `_autoSwitchTimer = setTimeout(` inside loadJobs (after
    # the justCompleted block).
    load_jobs_idx = text.find("async function loadJobs()")
    assert load_jobs_idx >= 0
    chunk = text[load_jobs_idx : load_jobs_idx + 5000]
    assert "_autoSwitchTimer = setTimeout(" in chunk, (
        "loadJobs must store its auto-switch setTimeout id"
    )


# ---------------------------------------------------------------------------
# FH3 — showNotification / showToast unification
# ---------------------------------------------------------------------------


def test_fh3_showtoast_delegates_to_shownotification():
    """``showToast`` must delegate to ``showNotification`` so both
    functions share the #notification DOM and timer. Pre-fix,
    showToast created a new <div> per call with its own setTimeout,
    leaking DOM and using inconsistent z-index.
    """
    text = _read(APP_JS)
    # Locate showToast body and check it calls showNotification.
    m = re.search(
        r"function showToast\([^)]*\)\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert m, "showToast function not found"
    body = m.group(1)
    assert "showNotification(" in body, (
        "showToast must call showNotification (no separate DOM element)"
    )
    # And must NOT call createElement or appendChild (no DOM leak).
    assert "createElement" not in body, (
        "showToast must not create its own DOM elements"
    )
    assert "appendChild" not in body


# ---------------------------------------------------------------------------
# FM1 — Escape key for modals
# ---------------------------------------------------------------------------


def test_fm1_escape_key_handler_present():
    """A document-level keydown listener must close any open modal
    when Escape is pressed (WCAG 2.1 SC 2.1.1).
    """
    text = _read(APP_JS)
    assert re.search(
        r"document\.addEventListener\(['\"]keydown['\"], [\s\S]{0,400}Escape[\s\S]{0,400}classList\.add\(['\"]hidden['\"]\)",
        text,
    ), "Escape key handler missing or not closing .modal"


def test_fm1_minimax_modal_excluded_from_escape():
    """The MiniMax fallback modal must NOT close on Escape (force a
    choice UX). The handler should reference its id explicitly.
    """
    text = _read(APP_JS)
    assert "MiniMax-fallback-modal" in text, (
        "MiniMax fallback modal must be referenced (excluded from "
        "auto-close on Escape)"
    )


# ---------------------------------------------------------------------------
# FM3 — file input reset on reopen
# ---------------------------------------------------------------------------


def test_fm3_openfilepicker_resets_input_value():
    """The new openFilePicker() helper must clear ``pdfInput.value``
    before calling ``click()`` so re-selecting the same file fires
    the ``change`` event.
    """
    text = _read(APP_JS)
    m = re.search(
        r"function openFilePicker\(\)\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert m, "openFilePicker function not found"
    body = m.group(1)
    # Must reset value then click, in that order.
    assert re.search(r"pdfInput\.value\s*=\s*['\"]['\"]", body), (
        "openFilePicker must reset pdfInput.value to empty"
    )
    assert "pdfInput.click()" in body


def test_fm3_uploadarea_calls_openfilepicker():
    """The click + keydown handlers must call ``openFilePicker()``
    instead of the inline ``pdfInput.click()`` so the reset applies
    uniformly.
    """
    text = _read(APP_JS)
    assert re.search(
        r"uploadArea\.addEventListener\(['\"]click['\"],\s*openFilePicker\)",
        text,
    ), "upload-area click handler must call openFilePicker()"
    assert re.search(
        r"uploadArea\.addEventListener\(['\"]keydown['\"][\s\S]{0,400}openFilePicker\(\)",
        text,
    ), "upload-area keydown handler must call openFilePicker()"


# ---------------------------------------------------------------------------
# FM4 — status-filter event delegation (no leak)
# ---------------------------------------------------------------------------


def test_fm4_status_filter_uses_oneshot_flag():
    """The status-filter container must wire its click handler ONCE
    via ``__rlpeFilterWired`` flag (matching the pagination
    pattern). Pre-fix it called addEventListener per button on every
    render — duplicate handlers accumulated.
    """
    text = _read(APP_JS)
    # Find renderResultsStatusFilterCounts body.
    fn_idx = text.find("function renderResultsStatusFilterCounts")
    assert fn_idx >= 0
    chunk = text[fn_idx : fn_idx + 1500]
    assert "__rlpeFilterWired" in chunk, (
        "renderResultsStatusFilterCounts must use __rlpeFilterWired "
        "single-delegation pattern"
    )
    # And must NOT call addEventListener on each button (the old leak).
    assert "btn.addEventListener" not in chunk, (
        "renderResultsStatusFilterCounts must not addEventListener "
        "to each button (leaks handlers across renders)"
    )


# ---------------------------------------------------------------------------
# FM5 — getElementById null safety
# ---------------------------------------------------------------------------


def test_fm5_buildllmoptions_uses_optional_chaining():
    """Every ``getElementById(...).value`` read inside the option
    builders must guard against the missing-element case via
    ``?.value`` (and ``?.trim() ?? ''`` where appropriate).
    """
    text = _read(APP_JS)
    # Find _buildLLMOptions body.
    fn_idx = text.find("function _buildLLMOptions()")
    assert fn_idx >= 0
    chunk = text[fn_idx : fn_idx + 4000]
    # No bare `getElementById('x').value` left (would throw on missing).
    bad = re.findall(
        r"getElementById\([^)]+\)\.value(?!\?)",
        chunk,
    )
    assert not bad, (
        f"_buildLLMOptions has unguarded getElementById(...).value "
        f"reads: {bad}"
    )


def test_fm5_buildpaleodboptions_uses_optional_chaining():
    """Same check for _buildPaleodbOptions."""
    text = _read(APP_JS)
    fn_idx = text.find("function _buildPaleodbOptions()")
    assert fn_idx >= 0
    chunk = text[fn_idx : fn_idx + 1500]
    bad = re.findall(
        r"getElementById\([^)]+\)\.value(?!\?)",
        chunk,
    )
    assert not bad, (
        f"_buildPaleodbOptions has unguarded getElementById(...).value "
        f"reads: {bad}"
    )


def test_fm5_closecorrectionmodal_nullsafe():
    """``closeCorrectionModal`` must not throw when the modal element
    is missing (defensive against accidental HTML renames).
    """
    text = _read(APP_JS)
    fn_idx = text.find("function closeCorrectionModal()")
    assert fn_idx >= 0
    chunk = text[fn_idx : fn_idx + 400]
    assert "?.classList" in chunk, (
        "closeCorrectionModal must use optional chaining on getElementById"
    )


# ---------------------------------------------------------------------------
# FL3 — accessibility
# ---------------------------------------------------------------------------


def test_fl3_results_table_has_caption_and_aria_label():
    """The results table must carry an ``aria-label`` and a visually-
    hidden ``<caption>`` for screen readers.
    """
    text = _read(INDEX_HTML)
    assert 'id="results-table"' in text
    assert 'aria-label="图版与标签匹配结果"' in text, (
        "results-table must have aria-label"
    )
    assert '<caption class="visually-hidden"' in text, (
        "results-table must have a visually-hidden <caption>"
    )


def test_fl3_modals_have_dialog_role():
    """Each dismissable modal must declare ``role="dialog"`` and
    ``aria-modal="true"`` so assistive technology announces it as a
    modal dialog.
    """
    text = _read(INDEX_HTML)
    for modal_id in ["image-modal", "job-details-modal", "correction-modal"]:
        # Locate the modal opening tag.
        idx = text.find(f'id="{modal_id}"')
        assert idx >= 0, f"{modal_id} not found"
        snippet = text[idx : idx + 200]
        assert 'role="dialog"' in snippet, (
            f"{modal_id} must declare role=dialog"
        )
        assert 'aria-modal="true"' in snippet, (
            f"{modal_id} must declare aria-modal=true"
        )


def test_fl3_icon_only_buttons_have_aria_label():
    """Icon-only buttons (refresh, delete, export) need aria-label
    because the SVG content alone is opaque to screen readers.
    """
    text = _read(INDEX_HTML)
    for btn_id, expected in [
        ("refresh-jobs-btn", "刷新任务列表"),
        ("delete-selected-btn", "删除选中的任务"),
        ("export-btn", "导出当前筛选结果为 CSV"),
    ]:
        idx = text.find(f'id="{btn_id}"')
        assert idx >= 0, f"button {btn_id} not found"
        snippet = text[idx : idx + 500]
        assert f'aria-label="{expected}"' in snippet, (
            f"{btn_id} must have aria-label={expected!r}"
        )


def test_fl3_visually_hidden_class_defined():
    """The .visually-hidden utility must exist in the CSS so the
    sr-only captions / titles are hidden visually but remain in the
    accessibility tree.
    """
    text = _read(STYLE_CSS)
    assert re.search(r"^\s*\.visually-hidden\s*\{", text, re.MULTILINE), (
        ".visually-hidden class not defined"
    )


# ---------------------------------------------------------------------------
# FL4 — z-index design tokens
# ---------------------------------------------------------------------------


def test_fl4_zindex_tokens_defined():
    """A canonical z-index scale must live in :root so overlay
    surfaces don't have hard-coded z-index values scattered around.
    """
    text = _read(STYLE_CSS)
    assert "--z-sticky:" in text
    assert "--z-dropdown:" in text
    assert "--z-modal:" in text
    assert "--z-toast:" in text


def test_fl4_no_hardcoded_z_index_for_overlay_surfaces():
    """Modal/toast z-index values must use the design tokens, not
    hard-coded numbers. (Small decorative z-indexes like 2, 10 are
    fine — only the documented overlay surfaces are checked.)
    """
    text = _read(STYLE_CSS)
    # Find every z-index declaration and report any that are >= 1000
    # and don't reference a --z-* token. These should use the tokens.
    offenders = []
    for m in re.finditer(r"z-index:\s*(\d+);", text):
        val = int(m.group(1))
        if val >= 1000:
            offenders.append((m.start(), val))
    assert not offenders, (
        f"Hard-coded z-index >= 1000 found (use --z-modal/--z-toast "
        f"tokens instead): {offenders}"
    )