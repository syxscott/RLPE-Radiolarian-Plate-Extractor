"""Round 16 source-guard tests: project default = MiniMax, with safety.

After the 2026-07-06 audit found MiniMax hardcoded as the project
default throughout the UI and pipeline, the user chose to keep
MiniMax as the *project default* (their preferred cloud backend)
but tighten the safety rails around it:

  Project-default-Minimax (UI defaults):
    - HTML: both <select> elements pre-select MiniMax with 🌟 + "推荐"
    - HTML: onboarding banner sells MiniMax on first visit
    - HTML: success-green badge "默认 MiniMax M3"
    - JS:  ``?? 'MiniMax'`` fallback when the dropdown is missing
    - JS:  model-name fallback hardcoded as "MiniMax-M3"

  Safety rails (independent of vendor choice):
    - Extended-thinking checkbox defaults to UNCHECKED (avoid
      surprise API cost; user must opt in)
    - data_outbound_policy defaults to "api_redacted" (privacy)
    - M3Engine requires explicit m3_enhanced_mode=True for ALL
      backends (no vendor gets a privileged default)
    - ANTHROPIC_API_KEY is NOT silently consumed for MiniMax
    - LLM backend choice persists to localStorage (so a user who
      switches off MiniMax keeps that choice across reloads)

These tests pin both: the project default (MiniMax-first UX) AND
the safety improvements (cost / privacy / opt-in).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --- 1) HTML: MiniMax must NOT be the default option ---------------------


def _read(path: str) -> str:
    return Path(__file__).resolve().parents[1].joinpath(path).read_text(encoding="utf-8")


def test_basic_view_default_is_minimax():
    """``<select id="llm-backend-basic">`` MUST pre-select MiniMax with
    the 🌟 + '推荐' label — this is the project's chosen default LLM
    backend. Tests pin the project default so it doesn't accidentally
    drift."""
    src = _read("web/index.html")
    idx = src.find('id="llm-backend-basic"')
    assert idx > 0
    end = src.find("</select>", idx)
    block = src[idx : end + len("</select>")]
    assert 'value="MiniMax" selected' in block, (
        "Basic-view <select> does not pre-select MiniMax. Project default is MiniMax M3 (云端 API)."
    )
    assert "🌟" in block, "Basic-view option should carry the 🌟 marker"


def test_advanced_view_default_is_minimax():
    """``<select id="llm-backend">`` (advanced view) MUST pre-select MiniMax."""
    src = _read("web/index.html")
    idx = src.find('id="llm-backend"')
    assert idx > 0
    end = src.find("</select>", idx)
    block = src[idx : end + len("</select>")]
    assert 'value="MiniMax" selected' in block, (
        "Advanced-view <select> does not pre-select MiniMax."
    )


def test_onboarding_banner_mentions_minimax():
    """The onboarding banner copy MUST mention MiniMax as the
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
    assert "MiniMax M3" in block, (
        "Onboarding banner no longer mentions MiniMax M3 as the "
        "pre-configured choice. The project default is MiniMax M3 — "
        "first-visit UX should reflect that."
    )


def test_recommended_badge_mentions_minimax():
    """The LLM status card carries a '默认 MiniMax M3' success-green
    badge — pinning the project default."""
    src = _read("web/index.html")
    assert "默认 MiniMax M3" in src, (
        "The '默认 MiniMax M3' success-green badge is missing from the "
        "LLM status card — project default should be visible at a glance."
    )


def test_thinking_checkbox_defaults_off():
    """The MiniMax extended-thinking checkbox must default to unchecked.
    Default-on means every job spends extra tokens (more API cost)."""
    src = _read("web/index.html")
    # Find the checkbox line
    idx = src.find('id="MiniMax-enable-thinking"')
    assert idx > 0
    line = src[idx : src.find(">", idx) + 1]
    assert "checked" not in line, (
        f"MiniMax extended-thinking checkbox defaults to checked: {line!r}. "
        f"Default must be unchecked to avoid surprise API cost."
    )


# --- 2) JS: fallback chain must not hardcode MiniMax ---------------------


def test_js_backend_fallback_defaults_to_minimax():
    """web/js/app.js: the LLM backend fallback chain ends with 'MiniMax'.
    This is the project's chosen default — when no dropdown is found
    AND no localStorage entry exists, MiniMax is used."""
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
    # The fallback chain must end with 'MiniMax' (project default).
    assert "|| 'MiniMax'" in code_only or "?? 'MiniMax'" in code_only, (
        "web/js/app.js no longer falls back to 'MiniMax'. The project "
        "default LLM backend is MiniMax M3 — pin it as the ultimate fallback."
    )
    # The model-name fallbacks should reference MiniMax-M3 (project model).
    assert "|| 'MiniMax-M3'" in code_only, (
        "Model-name fallbacks no longer reference 'MiniMax-M3'. The "
        "project default model name should pin to MiniMax-M3."
    )


def test_js_persists_llm_backend_choice():
    """The LLM backend selector must persist to localStorage so the
    user's choice survives page reloads. Previously every reload
    silently reverted to MiniMax."""
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


# --- 3) Backend defaults: thinking OFF, policy=redacted, M3 opt-in -----


def test_job_options_thinking_defaults_off():
    """api/app.py JobOptions.MiniMax_enable_thinking must default to False.
    Default-on means every web job spends extra tokens."""
    api_src = _read("src/rlpe/api/app.py")
    # Find the field declaration. Don't import JobOptions — the api
    # module requires FastAPI, which isn't installed in all envs.
    idx = api_src.find("MiniMax_enable_thinking")
    assert idx > 0, "MiniMax_enable_thinking field not found in api/app.py"
    # Take the surrounding 200 chars
    window = api_src[idx : idx + 200]
    assert "= False" in window, (
        f"JobOptions.MiniMax_enable_thinking is not defaulted to False: "
        f"{window!r}. Default-on means every web job spends extra tokens."
    )


def test_dataclass_data_outbound_policy_is_api_full():
    """Phase 61 Plan 4 (Bug 4.11): the default for
    ``data_outbound_policy`` flipped from ``api_redacted`` to
    ``api_full`` so M3 vision gets full-resolution morphology details.
    Operators working with sensitive preprints can still opt back in
    via ``--data-outbound-policy api_redacted``.
    """
    import inspect

    from rlpe.llm_backends import MiniMaxM3Backend

    field = MiniMaxM3Backend.__dataclass_fields__["data_outbound_policy"]
    assert field.default == "api_full", (
        f"MiniMaxM3Backend.data_outbound_policy defaults to {field.default!r}; "
        f"Phase 61 Plan 4 requires 'api_full' so M3 vision sees full-res images."
    )


def test_dataclass_enable_thinking_defaults_off():
    """llm_backends.py MiniMaxM3Backend.enable_thinking must default to False."""
    import inspect

    from rlpe.llm_backends import MiniMaxM3Backend

    field = MiniMaxM3Backend.__dataclass_fields__["enable_thinking"]
    assert field.default is False, (
        f"MiniMaxM3Backend.enable_thinking defaults to {field.default!r}; "
        f"must default to False to avoid surprise API cost."
    )


def test_m3engine_is_symmetric_opt_in():
    """pipeline.py: M3Engine must require explicit opt-in (``m3_enhanced_mode=True``)
    for ALL backends. The previous code auto-enabled for MiniMax only."""
    src = _read("src/rlpe/pipeline.py")
    # Skip past the docstring/class-header occurrences of
    # ``m3_enhanced_mode`` to find the actual code that decides
    # whether to build M3Engine. The first occurrence in the
    # class-level docstring is documentation, not code.
    code_start = src.find("# Build the M3 semantic engine")
    if code_start < 0:
        # Fallback: search for the explicit assignment
        code_start = src.find('want_m3 = self.config.extra.get("m3_enhanced_mode"')
    assert code_start > 0, "Could not locate M3 init code in pipeline.py"
    # Take the next 800 chars
    window = src[code_start : code_start + 800]
    # Must NOT contain the asymmetric auto-enable pattern
    assert "want_m3 = backend_name in minimax_backends" not in window, (
        "pipeline.py still auto-enables M3 for MiniMax backend only. "
        "M3 must require explicit m3_enhanced_mode=True for all backends "
        "so no vendor gets a privileged default."
    )
    # Must use the explicit-default-False form
    assert (
        'want_m3 = self.config.extra.get("m3_enhanced_mode", False)' in window
        or "want_m3 = self.config.extra.get('m3_enhanced_mode', False)" in window
    ), "M3 auto-enable must read m3_enhanced_mode with default False."


def test_pipeline_does_not_silently_consume_anthropic_key():
    """pipeline.py: ``has_minimax_key`` must NOT fall back to
    ANTHROPIC_API_KEY. Previously a Claude Code user with an Anthropic
    key got silently routed to MiniMax with no warning."""
    src = _read("src/rlpe/pipeline.py")
    # Locate the has_minimax_key block
    idx = src.find("has_minimax_key")
    assert idx > 0
    end = src.find("\n", src.find(")", idx) + 1)
    window = src[idx:end]
    assert "ANTHROPIC_API_KEY" not in window, (
        "pipeline.py has_minimax_key still checks ANTHROPIC_API_KEY. "
        "Vendor-specific key (MINIMAX_API_KEY or MiniMax_api_key) must "
        "be required; ANTHROPIC_API_KEY is Claude Code's key and should "
        "not silently route to MiniMax."
    )
