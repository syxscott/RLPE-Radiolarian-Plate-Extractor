"""Static checks for the web/js/app.js fetch pattern.

The frontend historically had a class of bugs where
``resp.json()`` was called before ``resp.ok`` was checked, which
crashed on non-2xx responses (HTML error pages from proxies, empty
bodies, etc.). These tests assert the safe pattern
``if (resp.ok) { ... await resp.json() ... } else { ... await resp.json().catch(...) ... }``
is present at the three call sites flagged in the 2026-07-03 audit:
``confirmDelete`` (single-delete path), ``cancelJob``, and
``submitMiniMaxFallback``.

The frontend has no jsdom test harness, so these are grep-based
static assertions on the served JS file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_JS = REPO_ROOT / "web" / "js" / "app.js"

# Ensure scripts/ is importable so the L6 test can probe smoke_oa_corpus
# without requiring PYTHONPATH to be set externally.
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture(scope="module")
def js() -> str:
    if not WEB_JS.exists():
        pytest.skip("web/js/app.js not present")
    return WEB_JS.read_text(encoding="utf-8")


def _function_body(text: str, fn_name: str) -> str:
    """Return the body of ``function <fn_name>(... ) { ... }`` up to its
    matching closing brace. Greedy enough for our needs."""
    marker = f"function {fn_name}("
    i = text.find(marker)
    if i < 0:
        pytest.fail(f"{fn_name} not found in web/js/app.js")
    brace_open = text.find("{", i)
    if brace_open < 0:
        pytest.fail(f"{fn_name} has no opening brace")
    depth = 0
    for j in range(brace_open, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace_open : j + 1]
    pytest.fail(f"{fn_name} has no matching closing brace")


class TestConfirmDeleteChecksOkBeforeJson:
    """H1: single-delete path in confirmDelete must check resp.ok BEFORE
    calling resp.json()."""

    def test_single_delete_uses_ok_guard(self, js: str) -> None:
        body = _function_body(js, "confirmDelete")
        # Locate the single-delete branch: ``if (jobIds.length === 1)``.
        branch_start = body.find("if (jobIds.length === 1)")
        assert branch_start > 0, "confirmDelete: single-delete branch not found"
        branch_end = body.find("}", branch_start)
        # Find next '}' that closes this block — the outer if's closing brace.
        # Easier: take the next 400 chars and assert pattern.
        snippet = body[branch_start : branch_start + 800]
        # Find the fetch and the next .json() call.
        assert "method: 'DELETE'" in snippet
        # The bug pattern is: ``resp = await fetch(...); data = await resp.json();``
        # (no resp.ok between fetch and json).
        # The fix pattern is one of:
        #   if (resp.ok) { data = await resp.json(); }
        #   data = resp.ok ? await resp.json() : { ... };
        fetch_pos = snippet.find("await fetch(")
        json_pos = snippet.find(".json()", fetch_pos)
        # Look at the wider window between fetch and json: either the
        # resp.ok guard precedes json (if/ternary), or resp.json() is
        # nested inside an `if (resp.ok)` block higher up. Both shapes
        # are correct; the only failing pattern is bare ``data = await resp.json()``.
        # Find the first occurrence of `data =` between fetch and json.
        data_assign = snippet.find("data =", fetch_pos)
        if data_assign > 0 and data_assign < json_pos:
            # There's a direct data = ... assignment before the first json.
            # Must be guarded: ``data = resp.ok ? ...`` or the json() comes
            # before any data = assignment.
            assignment = snippet[data_assign : data_assign + 60]
            assert "resp.json" in assignment or "resp.ok" in assignment, (
                "confirmDelete: data = assignment before .json() must be "
                "guarded by resp.ok (audit H1). Got: " + assignment
            )
        # Also assert the broader pattern: the file does not contain
        # ``data = await resp.json();`` directly inside the single-delete
        # branch. Allow the line ``data = resp.ok ? await resp.json() :`` shape.
        forbidden = "data = await resp.json();"
        assert forbidden not in snippet, (
            f"confirmDelete single-delete path has unguarded {forbidden!r}; (2026-07-03 audit H1)"
        )


class TestCancelJobChecksOkBeforeJson:
    """H2: cancelJob must check response.ok BEFORE calling response.json()."""

    def test_cancel_uses_ok_guard(self, js: str) -> None:
        body = _function_body(js, "cancelJob")
        assert "if (response.ok)" in body, (
            "cancelJob must check response.ok before response.json() (2026-07-03 audit H2)"
        )
        # Verify the ok check precedes the json parse on the success path.
        ok_pos = body.find("if (response.ok)")
        json_pos = body.find("response.json()", ok_pos)
        assert json_pos > ok_pos, "cancelJob: response.json() must come AFTER response.ok check"


class TestSubmitMiniMaxFallbackChecksOkBeforeJson:
    """H3: submitMiniMaxFallback must check r.ok BEFORE calling r.json()."""

    def test_submit_uses_ok_guard(self, js: str) -> None:
        body = _function_body(js, "submitMiniMaxFallback")
        assert "if (!r.ok)" in body, (
            "submitMiniMaxFallback must check r.ok before r.json() (2026-07-03 audit H3)"
        )
        ok_pos = body.find("if (!r.ok)")
        json_pos = body.find("r.json()", ok_pos)
        assert json_pos > ok_pos, "submitMiniMaxFallback: r.json() must come AFTER r.ok check"


class TestCorpusPathValidation:
    """L6: scripts/smoke_oa_corpus.py must fail fast on a non-existent
    --corpus path instead of silently producing a zero-row summary.
    """

    def test_main_validates_corpus_path(self) -> None:
        # We can't actually call main() in tests because it calls argparse's
        # exit() on errors; instead, parse args and verify _parse_args
        # validates the path before run_smoke is called.
        from smoke_oa_corpus import _parse_args

        old_argv = sys.argv
        try:
            sys.argv = ["smoke_oa_corpus", "--corpus", "/nonexistent/path/abc123", "--reset"]
            with pytest.raises(SystemExit) as exc:
                _parse_args()
            # argparse exits with code 2 on argument errors.
            assert exc.value.code == 2, (
                "smoke_oa_corpus must reject a non-existent --corpus path; got exit code != 2"
            )
        finally:
            sys.argv = old_argv


class TestEventDelegationUsed:
    """M8 + M9: the old code called addEventListener on freshly-
    recreated button / img elements inside the results table tbody
    and pagination container — duplicate listeners accumulated on every
    render. The fix uses event delegation on stable parent elements.

    We assert that the old buggy patterns are gone from app.js.
    """

    def test_no_addEventListener_in_renderResults_thumbnails(self, js: str) -> None:
        # Locate renderResults body and assert no inner addEventListener
        # calls. Delegation uses one outer listener wired once.
        body = _function_body(js, "renderResults")
        # Strip out the delegation block — it's a legitimate listener
        # at the top of the function. Look for forbidden patterns:
        # forEach(img => { img.addEventListener  OR
        # forEach(btn => { btn.addEventListener
        # within the renderResults body.
        forbidden_forEach = "querySelectorAll('.thumbnail-img').forEach"
        assert forbidden_forEach not in body, (
            "renderResults must NOT use querySelectorAll('.thumbnail-img').forEach "
            "to wire click handlers (audit M8). Use event delegation."
        )
        forbidden_corr = "querySelectorAll('[data-correct-index]').forEach"
        assert forbidden_corr not in body, (
            "renderResults must NOT use querySelectorAll('[data-correct-index]').forEach "
            "to wire click handlers (audit M8). Use event delegation."
        )

    def test_no_addEventListener_in_renderResultsPagination_buttons(self, js: str) -> None:
        # renderResultsPagination should NOT call addEventListener on
        # the freshly-recreated button elements. The fix uses delegation
        # via container.addEventListener (allowed), but never
        # ``btn.addEventListener`` or ``ps.addEventListener`` directly.
        body = _function_body(js, "renderResultsPagination")
        # Drop block + line comments so we don't false-positive on
        # docstrings that mention ``addEventListener`` as context.
        code_only = "\n".join(
            line for line in body.splitlines() if not line.strip().startswith(("//", "/*", "*"))
        )
        forbidden_direct = ["btn.addEventListener", "ps.addEventListener"]
        for pat in forbidden_direct:
            assert pat not in code_only, (
                f"renderResultsPagination must NOT call {pat!r} directly "
                f"(audit M9). Use delegation."
            )

    def test_tbody_has_delegation_guard(self, js: str) -> None:
        # The fix uses an instance-property guard (__rlpeListenersWired)
        # to wire the tbody listener exactly once.
        body = _function_body(js, "renderResults")
        assert "__rlpeListenersWired" in body, (
            "renderResults must guard the tbody delegated listener with "
            "__rlpeListenersWired (audit M8)"
        )

    def test_pagination_container_has_delegation_guard(self, js: str) -> None:
        body = _function_body(js, "renderResultsPagination")
        assert "__paginationWired" in body, (
            "renderResultsPagination must guard the container delegated "
            "listener with __paginationWired (audit M9)"
        )
