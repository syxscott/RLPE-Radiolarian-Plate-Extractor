"""Round 16 source-guard tests: project default = LLM, with safety.

After the 2026-07-06 audit found LLM hardcoded as the project
default throughout the UI and pipeline, the user chose to keep
LLM as the *project default* (their preferred cloud backend)
but tighten the safety rails around it:

  Project-default-Minimax (UI defaults):
    - HTML: both <select> elements pre-select LLM with 🌟 + "推荐"
    - HTML: onboarding banner sells LLM on first visit
    - HTML: success-green badge "默认 LLM LLM"
    - JS:  ``?? 'MiniMax'`` fallback when the dropdown is missing
    - JS:  model-name fallback hardcoded as "MiniMax-M3"

  Safety rails (independent of vendor choice):
    - Extended-thinking checkbox defaults to UNCHECKED (avoid
      surprise API cost; user must opt in)
    - data_outbound_policy defaults to "api_redacted" (privacy)
    - SemanticEngine requires explicit llm_enhanced_mode=True for ALL
      backends (no vendor gets a privileged default)
    - ANTHROPIC_API_KEY is NOT silently consumed for LLM
    - LLM backend choice persists to localStorage (so a user who
      switches off LLM keeps that choice across reloads)

These tests pin both: the project default (LLM-first UX) AND
the safety improvements (cost / privacy / opt-in).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --- 1) HTML: LLM must NOT be the default option ---------------------


def _read(path: str) -> str:
    return Path(__file__).resolve().parents[1].joinpath(path).read_text(encoding="utf-8")


def test_basic_view_default_is_anthropic():
    """``<select id="llm-backend-basic">`` MUST pre-select the
    Anthropic-compatible cloud backend with the 🌟 + '推荐' label —
    this is the project's chosen default LLM backend (F17 rename).
    Tests pin the project default so it doesn't accidentally drift."""
    src = _read("web/index.html")
    idx = src.find('id="llm-backend-basic"')
    assert idx > 0
    end = src.find("</select>", idx)
    block = src[idx : end + len("</select>")]
    assert 'value="anthropic" selected' in block, (
        "Basic-view <select> does not pre-select the cloud backend. "
        "Project default is the Anthropic-compatible API."
    )
    assert "🌟" in block, "Basic-view option should carry the 🌟 marker"


def test_advanced_view_default_is_anthropic():
    """``<select id="llm-backend">`` (advanced view) MUST pre-select
    the Anthropic-compatible cloud backend (F17)."""
    src = _read("web/index.html")
    idx = src.find('id="llm-backend"')
    assert idx > 0
    end = src.find("</select>", idx)
    block = src[idx : end + len("</select>")]
    assert 'value="anthropic" selected' in block, (
        "Advanced-view <select> does not pre-select the cloud backend."
    )


def test_onboarding_banner_mentions_anthropic():
    """The onboarding banner copy MUST mention LLM as the
    pre-configured choice (project default)."""
    src = _read("web/index.html")
    # Find the onboarding-banner block via the class attribute
    marker = 'class="onboarding-banner'
    idx = src.find(marker)
    assert idx > 0, "Onboarding banner block not found"
    # Walk forward to find the matching </div> using a depth counter
    # that respects opening/closing tags inside attributes (rare in
    # this banner).
    depth = 0
    cursor = idx
    end = idx
    while cursor < len(src):
        if src[cursor : cursor + 5] == "<div>":
            depth += 1
        elif src[cursor : cursor + 5] == "<div ":
            depth += 1
        elif src[cursor : cursor + 6] == "</div>":
            depth -= 1
            if depth == 0:
                end = cursor + 6
                break
        cursor += 1
    block = src[idx:end]
    assert "Anthropic 兼容" in block, (
        "Onboarding banner no longer mentions the Anthropic-compatible "
        "cloud API as the pre-configured choice (F17) — first-visit UX "
        "should reflect that."
    )


def test_recommended_badge_mentions_anthropic():
    """The LLM status card carries an 'Anthropic 兼容' badge — pinning
    the vendor-agnostic posture (F17)."""
    src = _read("web/index.html")
    assert "Anthropic 兼容" in src, (
        "The 'Anthropic 兼容' badge is missing from the LLM status "
        "card — the vendor-agnostic default should be visible."
    )


def test_thinking_checkbox_defaults_off():
    """The LLM extended-thinking checkbox must default to unchecked.
    Default-on means every job spends extra tokens (more API cost)."""
    src = _read("web/index.html")
    # Find the checkbox line
    idx = src.find('id="llm-enable-thinking"')
    assert idx > 0
    line = src[idx : src.find(">", idx) + 1]
    assert "checked" not in line, (
        f"LLM extended-thinking checkbox defaults to checked: {line!r}. "
        f"Default must be unchecked to avoid surprise API cost."
    )


# --- 2) JS: fallback chain must not hardcode LLM ---------------------


def test_js_backend_fallback_defaults_to_anthropic():
    """web/js/app.js: the LLM backend fallback chain ends with
    'anthropic' (F17 rename) — when no dropdown is found AND no
    localStorage entry exists, the cloud backend is used. Model names
    are provider-specific now, so no hard-coded model fallback."""
    src = _read("web/js/app.js")
    code_lines = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        if "//" in line:
            line = line.split("//", 1)[0]
        code_lines.append(line)
    code_only = "\n".join(code_lines)
    assert "|| 'anthropic'" in code_only or "?? 'anthropic'" in code_only, (
        "web/js/app.js no longer falls back to 'anthropic'. Pin the "
        "cloud backend as the ultimate fallback."
    )
    assert "|| 'MiniMax-M3'" not in code_only, (
        "Model-name fallbacks still reference a vendor model (F17 "
        "removed hard-coded vendor defaults)."
    )


def test_js_persists_llm_backend_choice():
    """The LLM backend selector must persist to localStorage so the
    user's choice survives page reloads. Previously every reload
    silently reverted to LLM."""
    src = _read("web/js/app.js")
    # Must define a storage key constant
    assert "LLM_BACKEND_KEY" in src, (
        "web/js/app.js is missing LLM_BACKEND_KEY — the user's LLM "
        "backend choice is not persisted to localStorage."
    )
    # Must use _safeStorageSet on change
    assert "_safeStorageSet(LLM_BACKEND_KEY" in src, (
        "LLM backend change handlers must persist the choice via "
        "_safeStorageSet(LLM_BACKEND_KEY, ...)."
    )


# --- 3) Backend defaults: thinking OFF, policy=redacted, LLM opt-in -----


def test_job_options_thinking_defaults_off():
    """api/app.py JobOptions.llm_enable_thinking must default to False.
    Default-on means every web job spends extra tokens."""
    api_src = _read("src/rlpe/api/app.py")
    # Find the field declaration. Don't import JobOptions — the api
    # module requires FastAPI, which isn't installed in all envs.
    idx = api_src.find("llm_enable_thinking")
    assert idx > 0, "llm_enable_thinking field not found in api/app.py"
    # Take the surrounding 200 chars
    window = api_src[idx : idx + 200]
    assert "= False" in window, (
        f"JobOptions.llm_enable_thinking is not defaulted to False: "
        f"{window!r}. Default-on means every web job spends extra tokens."
    )


def test_dataclass_data_outbound_policy_is_api_redacted():
    """Audit 2026-09-03 (BLOCKER-#2 fix): the default for
    ``data_outbound_policy`` flipped from ``api_full`` to
    ``api_redacted`` so a fresh pipeline run does NOT silently
    ship full-resolution images + verbatim captions to LLM.
    Operators who need the historical full-resolution behaviour
    must opt in explicitly via the new CLI flag
    ``--i-understand-data-leaves-my-machine`` (which sets
    ``RLPE_DATA_OUTBOUND_OPT_IN=1``).
    """
    import inspect

    from rlpe.llm_backends import AnthropicCompatBackend

    field = AnthropicCompatBackend.__dataclass_fields__["data_outbound_policy"]
    assert field.default == "api_redacted", (
        f"AnthropicCompatBackend.data_outbound_policy defaults to {field.default!r}; "
        f"Audit 2026-09-03 (BLOCKER-#2 fix) requires 'api_redacted' as "
        f"the safe default. Operators opt in to 'api_full' via "
        f"--i-understand-data-leaves-my-machine (which sets "
        f"RLPE_DATA_OUTBOUND_OPT_IN=1)."
    )


def test_dataclass_enable_thinking_defaults_off():
    """llm_backends.py AnthropicCompatBackend.enable_thinking must default to False."""
    import inspect

    from rlpe.llm_backends import AnthropicCompatBackend

    field = AnthropicCompatBackend.__dataclass_fields__["enable_thinking"]
    assert field.default is False, (
        f"AnthropicCompatBackend.enable_thinking defaults to {field.default!r}; "
        f"must default to False to avoid surprise API cost."
    )


def test_m3engine_is_symmetric_opt_in():
    """pipeline.py: SemanticEngine must require explicit opt-in (``llm_enhanced_mode=True``)
    for ALL backends. The previous code auto-enabled for LLM only."""
    src = _read("src/rlpe/pipeline.py")
    # Skip past the docstring/class-header occurrences of
    # ``llm_enhanced_mode`` to find the actual code that decides
    # whether to build SemanticEngine. The first occurrence in the
    # class-level docstring is documentation, not code.
    code_start = src.find("# Build the LLM semantic engine")
    if code_start < 0:
        # Fallback: search for the explicit assignment
        code_start = src.find('want_m3 = self.config.extra.get("llm_enhanced_mode"')
    assert code_start > 0, "Could not locate LLM init code in pipeline.py"
    # Take the next 800 chars
    window = src[code_start : code_start + 800]
    # Must NOT contain the asymmetric auto-enable pattern
    assert "want_m3 = backend_name in minimax_backends" not in window, (
        "pipeline.py still auto-enables LLM for LLM backend only. "
        "LLM must require explicit llm_enhanced_mode=True for all backends "
        "so no vendor gets a privileged default."
    )
    # Must use the explicit-default-False form
    assert (
        'want_m3 = self.config.extra.get("llm_enhanced_mode", False)' in window
        or "want_m3 = self.config.extra.get('llm_enhanced_mode', False)" in window
    ), "LLM auto-enable must read llm_enhanced_mode with default False."


def test_pipeline_resolves_key_through_shared_chain():
    """pipeline.py (F17): the cloud-backend heuristic must resolve the
    key through ``resolve_llm_api_key`` — the single chain that covers
    extra, the saved settings file and the env (ANTHROPIC_* canonical,
    legacy vendor names as read-only fallbacks). The old heuristic
    checked a vendor-branded env var directly, which silently missed
    keys stored in the settings file."""
    src = _read("src/rlpe/pipeline.py")
    assert "resolve_llm_api_key(self.config.extra) is not None" in src, (
        "pipeline.py heuristic no longer uses resolve_llm_api_key — "
        "key resolution must stay centralised in the shared chain."
    )
