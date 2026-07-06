"""Round 13 source-guard tests: full-codebase audit fixes.

Locks in the bug fixes found in the 2026-07-06 full audit pass.
The 3 bugs are:
  - llm_backends API-key redaction (Bug A): a 401/403 from the
    Anthropic SDK embeds the raw key in ``str(exc)`` and that string
    used to land in ``match.metadata["gemma_error"]`` and the
    output JSONL. The fix adds ``_redact_api_keys`` and routes the
    error through it.
  - llm_backends config coercion (Bug B): ``int(extra.get(...))``
    with a non-numeric string raises ValueError deep in the
    constructor. The fix adds ``_coerce_int/_coerce_float/_coerce_bool``
    with safe fallbacks.
  - taxon TaxoNERD shape guard (Bug C): TaxoNERD occasionally
    returns a string or a list containing non-dict elements on
    model-mismatch errors. ``item.get(...)`` then raises
    AttributeError, which the bare ``except Exception: pass``
    silently swallowed. The fix adds explicit type guards.
  - web/addFiles case-insensitive PDF (Bug D): ``f.name.endsWith('.pdf')``
    silently dropped files with upper-case ``.PDF`` extension.
  - web/querySelector tab click (Bug E): missing jobs-tab button
    threw TypeError, bypassing uploadedFiles reset.
  - web/localStorage try/catch (Bug F): Safari private mode and
    quota-exceeded throws on localStorage.setItem/getItem.
  - api/usage falsy check (Bug G): ``result.get("usage") or {}``
    treated ``{"input_tokens": 0, ...}`` as falsy and crashed the
    downstream .get("input_tokens") call.
  - scripts/simulate_v20/21 hard_species_f1 KeyError (Bug H): the
    aggregate dict doesn't include these legacy keys.
  - exporters/archive EML XML escaping (Bug I): paper title /
    authors were injected raw, allowing XSS or malformed XML when
    a paper title contained ``<``/``&``/``]]>``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --- Bug A: API key redaction in _make_error_result -----------------------

def test_redact_api_keys_basic():
    """``_redact_api_keys`` must replace sk-... style keys with [REDACTED]."""
    from rlpe.llm_backends import _redact_api_keys

    # Anthropic-style: ``sk-ant-api03-`` + 48+ alnum
    assert _redact_api_keys("sk-ant-api03-" + "a" * 48) == "[REDACTED]"
    # Generic: ``sk-`` + 16+ alnum (MiniMax, OpenAI, etc.)
    assert _redact_api_keys("sk-" + "a" * 32) == "[REDACTED]"
    # MiniMax cp-style: ``sk-cp-`` + 16+ alnum
    assert _redact_api_keys("sk-cp-" + "X" * 32) == "[REDACTED]"
    # Non-key text passes through unchanged
    assert _redact_api_keys("API key not valid: 401 Unauthorized") == (
        "API key not valid: 401 Unauthorized"
    )
    # Mixed: the key is stripped, the rest survives. The trailing
    # non-key text must not be eaten by the redaction. Use a realistic
    # 48-char Anthropic-style body so the regex is forced to terminate
    # cleanly at the end of the key.
    assert _redact_api_keys(
        "prefix sk-ant-api03-" + "a" * 48 + " tail"
    ) == "prefix [REDACTED] tail"


def test_make_error_result_redacts():
    """``MiniMaxM3Backend._make_error_result`` must never echo raw keys."""
    from rlpe.llm_backends import MiniMaxM3Backend

    # Construct a fake exception that includes an API key (real-shape
    # Anthropic key: 48 alnum chars after ``sk-ant-api03-``).
    class FakeAuthError(Exception):
        def __str__(self) -> str:
            return "API key not valid: sk-ant-api03-" + "a" * 48 + " (request-id: abc)"

    backend = MiniMaxM3Backend(api_key="placeholder")
    res = backend._make_error_result(FakeAuthError())
    assert "sk-ant-" not in res["error"], f"API key leaked in error: {res['error']!r}"
    assert "sk-ant-" not in res["reasoning"], (
        f"API key leaked in reasoning: {res['reasoning']!r}"
    )
    assert "[REDACTED]" in res["error"]


# --- Bug B: config coercion -----------------------------------------------

def test_coerce_int_valid():
    from rlpe.llm_backends import _coerce_int

    assert _coerce_int("2048", default=0, name="x") == 2048
    assert _coerce_int(2048, default=0, name="x") == 2048
    assert _coerce_int("0", default=10, name="x") == 0


def test_coerce_int_invalid_fallback():
    """A non-numeric string must fall back to default, not raise."""
    from rlpe.llm_backends import _coerce_int

    assert _coerce_int("abc", default=2048, name="x") == 2048
    assert _coerce_int(None, default=2048, name="x") == 2048
    assert _coerce_int("", default=2048, name="x") == 2048


def test_coerce_float_invalid_fallback():
    from rlpe.llm_backends import _coerce_float

    assert _coerce_float("0.1", default=0.5, name="x") == 0.1
    assert _coerce_float("not-a-float", default=0.5, name="x") == 0.5
    assert _coerce_float(None, default=0.5, name="x") == 0.5


def test_coerce_bool_strings():
    from rlpe.llm_backends import _coerce_bool

    assert _coerce_bool("true", default=False) is True
    assert _coerce_bool("false", default=True) is False
    assert _coerce_bool("yes", default=False) is True
    assert _coerce_bool("no", default=True) is False
    assert _coerce_bool("on", default=False) is True
    assert _coerce_bool("off", default=True) is False
    assert _coerce_bool("0", default=True) is False
    assert _coerce_bool("1", default=False) is True
    assert _coerce_bool(None, default=True) is True
    assert _coerce_bool(True, default=False) is True
    assert _coerce_bool(False, default=True) is False


def test_build_backend_from_config_rejects_garbage_ints():
    """The MiniMax backend builder must not crash on a non-numeric
    ``MiniMax_max_output_tokens`` value."""
    from rlpe.llm_backends import build_MiniMax_backend_from_env_or_config

    # No API key + local_only policy so we don't actually call the API
    cfg = {
        "data_outbound_policy": "local_only",
        "MiniMax_max_output_tokens": "not-a-number",
        "MiniMax_thinking_budget_tokens": "abc",
        "gemma_temperature": "extremely-hot",
        "gemma_top_p": "yes",
        "MiniMax_max_retries": "ten",
        "MiniMax_max_concurrent": "many",
    }
    backend = build_MiniMax_backend_from_env_or_config(cfg)
    assert backend.max_output_tokens == 2048
    assert backend.thinking_budget_tokens == 1024
    assert backend.temperature == 0.1
    assert backend.top_p == 0.9
    assert backend.max_retries == 3
    assert backend.max_concurrent == 8


# --- Bug C: taxon TaxoNERD shape guard -----------------------------------

def test_taxon_skips_non_dict_items():
    """``TaxonRecognizer.predict`` must not crash when TaxoNERD returns
    a non-dict element. The bare ``except Exception: pass`` previously
    swallowed the AttributeError, masking the bug."""
    from rlpe.taxon import TaxonRecognizer

    rec = TaxonRecognizer(model="dummy")
    # Inject a fake engine whose .predict() returns a malformed result
    class FakeEngine:
        def predict(self, text: str):
            return [
                "not-a-dict",  # string
                None,  # None
                {"text": "Genus species", "start": 0, "end": 13, "label": "taxon", "score": 0.9},
            ]

    rec._engine = FakeEngine()
    rec._lazy_init_done = True
    entities = rec.predict("Some text with Genus species in it")
    # The dict entry should survive; non-dict entries should be skipped silently.
    assert any(e.text == "Genus species" for e in entities), (
        f"Valid dict entry was dropped; got {entities}"
    )


# --- Bug D: case-insensitive PDF accept ----------------------------------

def test_addFiles_accepts_uppercase_pdf():
    """``addFiles`` should accept ``.PDF`` (uppercase) extensions, not just
    ``.pdf``. macOS / iOS Finder and many academic repos use uppercase."""
    web_js = Path(__file__).resolve().parents[1] / "web" / "js" / "app.js"
    src = web_js.read_text(encoding="utf-8")
    # Extract the addFiles function body line-by-line, ignoring lines that
    # are part of a ``// ...`` comment block at the top of the function.
    # Comments mentioning the old case-sensitive pattern are allowed (they
    # document the bug); the live filter expression is what we want to check.
    code_lines = []
    in_addfiles = False
    for line in src.splitlines():
        if re.match(r"\s*function addFiles\s*\(", line):
            in_addfiles = True
            continue
        if in_addfiles:
            if re.match(r"\s*\}\s*$", line):
                break
            # Strip inline ``// ...`` comments so a docstring mention of
            # the old bug pattern doesn't trigger the assertion.
            stripped = re.sub(r"//.*$", "", line)
            code_lines.append(stripped)
    body = "\n".join(code_lines)
    assert "endsWith('.pdf')" not in body, (
        "Case-sensitive '.pdf' check still present in addFiles() — uppercase "
        ".PDF files will be silently dropped."
    )
    # The replacement pattern must use a case-insensitive regex.
    # The literal JS source is ``/\.pdf$/i`` which in a Python string is
    # ``"/\\.pdf$/i"``.
    assert "/\\.pdf$/i" in body, (
        "Case-insensitive PDF regex not found in addFiles(); the fix is "
        "either missing or the pattern is malformed. body was: " + body
    )


# --- Bug E: querySelector tab click fallback -----------------------------

def test_jobs_tab_click_has_fallback():
    """The job-upload handler must NOT throw when the jobs tab button is
    missing — that previously bypassed ``uploadedFiles = []`` and confused
    users. The fix adds a ``?.click()`` plus a fallback tab-pane show."""
    web_js = Path(__file__).resolve().parents[1] / "web" / "js" / "app.js"
    src = web_js.read_text(encoding="utf-8")
    # The unsafe pattern: ``document.querySelector('[data-tab="jobs"]').click()``
    # has no guard against a missing button. The fix must use ?. or a null check.
    bad_pattern = "document.querySelector('[data-tab=\"jobs\"]').click()"
    assert bad_pattern not in src, (
        f"Unsafe bare .click() still present at the upload handler — "
        f"a missing jobs-tab button would throw and leave uploadedFiles uncleared."
    )
    # Verify the safer pattern is in place
    assert "if (jobsTab)" in src or "jobsTab?.click" in src, (
        "Expected the new fallback ``if (jobsTab)`` guard to be present."
    )


# --- Bug F: localStorage try/catch ---------------------------------------

def test_safe_storage_helpers_used():
    """All non-test localStorage calls in app.js must go through
    ``_safeStorageGet`` / ``_safeStorageSet`` / ``_safeStorageRemove``
    so Safari private-mode and quota-exceeded throws don't break click
    handlers."""
    web_js = Path(__file__).resolve().parents[1] / "web" / "js" / "app.js"
    src = web_js.read_text(encoding="utf-8")
    # Strip the helper bodies — they're allowed to call raw localStorage
    # inside their try/except wrappers. Each helper is a one-liner of the
    # form ``function _safeStorageFoo(x) { try { localStorage.... } catch (_) {...} }``.
    outside_lines = []
    in_helper = False
    brace_depth = 0
    for line in src.splitlines():
        if not in_helper and re.match(r"\s*function _safeStorage\w+\(", line):
            in_helper = True
            brace_depth = line.count("{") - line.count("}")
            continue
        if in_helper:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                in_helper = False
            continue
        outside_lines.append(line)
    outside = "\n".join(outside_lines)
    # Find any direct localStorage calls outside the helpers
    raw_calls = re.findall(r"\blocalStorage\.(getItem|setItem|removeItem)\b", outside)
    assert not raw_calls, (
        "Found " + str(len(raw_calls)) + " raw localStorage calls outside _safeStorage*: "
        + repr(raw_calls) + ". Safari private mode would break these."
    )


# --- Bug G: api/app.py usage falsy check ---------------------------------

def test_api_app_usage_uses_isinstance():
    """``/api/MiniMax/test-connection`` must check usage is a dict, not
    rely on ``or {}`` which treats ``{"input_tokens": 0}`` as falsy."""
    api_app = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "api" / "app.py"
    src = api_app.read_text(encoding="utf-8")
    # The bug pattern was ``result.get("usage") or {}`` on a single line —
    # multiline comments that mention the old pattern are allowed (they
    # document why the fix is correct).
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        bad_pat = 'result.get("usage") or {}'
        assert bad_pat not in stripped, (
            "Found live (non-comment) usage of `" + bad_pat + "` on line: "
            + repr(stripped) + ". Use `isinstance(...)` check."
        )
    # Confirm the fix is in place
    assert "isinstance(usage_raw, dict)" in src, (
        "Expected the isinstance() guard around usage_raw to be present."
    )


# --- Bug H: simulate_v20 / v21 hard_species_f1 KeyError ------------------

def test_simulate_v20_no_hard_species_f1():
    src = Path(__file__).resolve().parents[1] / "scripts" / "simulate_v20_fix.py"
    text = src.read_text(encoding="utf-8")
    assert "agg['hard_species_f1']" not in text, (
        "simulate_v20_fix.py still references agg['hard_species_f1'] which "
        "does not exist in evaluate().aggregate."
    )
    assert "agg['normalisation_gap']" not in text, (
        "simulate_v20_fix.py still references agg['normalisation_gap'] which "
        "does not exist in evaluate().aggregate."
    )


def test_simulate_v21_no_hard_species_f1():
    src = Path(__file__).resolve().parents[1] / "scripts" / "simulate_v21.py"
    text = src.read_text(encoding="utf-8")
    assert "agg['hard_species_f1']" not in text, (
        "simulate_v21.py still references agg['hard_species_f1'] which "
        "does not exist in evaluate().aggregate."
    )
    assert "agg['normalisation_gap']" not in text, (
        "simulate_v21.py still references agg['normalisation_gap'] which "
        "does not exist in evaluate().aggregate."
    )


def test_simulate_v20_has_main_guard():
    """The script must wrap its body in main() with __name__ guard so
    ``import simulate_v20_fix`` doesn't execute the entire pipeline."""
    src = Path(__file__).resolve().parents[1] / "scripts" / "simulate_v20_fix.py"
    text = src.read_text(encoding="utf-8")
    assert "def main() -> int" in text, "simulate_v20_fix.py missing main()"
    assert 'if __name__ == "__main__"' in text, (
        "simulate_v20_fix.py missing __main__ guard"
    )


# --- Bug I: archive XML escaping -----------------------------------------

def test_archive_eml_escapes_user_fields():
    """The DwC-A EML builder must XML-escape paper title and authors —
    a paper with ``&``/``<``/``]]>`` in its title would otherwise break
    the archive or allow XSS in GBIF."""
    archive_py = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "exporters" / "archive.py"
    src = archive_py.read_text(encoding="utf-8")
    # The fix uses html.escape; verify the import exists
    assert "from html import escape" in src, (
        "archive.py is not importing html.escape — paper-controlled XML "
        "fields (title, authors) are not being escaped."
    )
    # The fix wraps pm.title in _xml_escape(...) inside _build_eml_xml
    assert "_xml_escape(pm.title)" in src, (
        "_xml_escape(pm.title) not found in _build_eml_xml; the XSS fix is missing."
    )
    assert "_xml_escape(author)" in src, (
        "_xml_escape(author) not found in _build_eml_xml; the XSS fix is missing."
    )
