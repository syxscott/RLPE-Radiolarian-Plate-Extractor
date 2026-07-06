"""Round 16 source-guard tests: vendor neutrality / no-default-MiniMax.

The 2026-07-06 user audit asked: is the frontend defaulting to MiniMax
as the LLM backend? Four parallel agents confirmed the lock-in:

  - HTML: both <select> elements pre-select MiniMax with 🌟 + "推荐"
  - HTML: onboarding banner sells MiniMax on first visit
  - HTML: success-green badge "默认 MiniMax M3"
  - HTML: extended-thinking checkbox checked by default
  - JS:  ``?? 'MiniMax'`` fallback when the dropdown is missing
  - JS:  LLM backend not persisted to localStorage — every reload
         reverts to MiniMax regardless of past selection
  - API: JobOptions.MiniMax_enable_thinking defaults to True
  - API: backend dataclass defaults base_url=https://api.minimaxi.com
  - Backend: data_outbound_policy='api_full' (least private default)
  - Pipeline: M3Engine auto-enables for MiniMax backend only
         (asymmetric opt-in vs opt-out across vendors)
  - Pipeline: ANTHROPIC_API_KEY (Claude Code's key) silently routes
         to MiniMax

The fixes below are pinned by source-guard tests so a future refactor
can't silently restore the lock-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --- 1) HTML: MiniMax must NOT be the default option ---------------------


def _read(path: str) -> str:
    return Path(__file__).resolve().parents[1].joinpath(path).read_text(encoding="utf-8")


def test_basic_view_default_is_not_minimax():
    """``<select id="llm-backend-basic">`` must NOT pre-select MiniMax.
    A privacy-conscious / offline-first user landing on the page should
    see a local backend by default."""
    src = _read("web/index.html")
    # Locate the basic-view <select> block
    idx = src.find('id="llm-backend-basic"')
    assert idx > 0
    # Find the next closing </select>
    end = src.find("</select>", idx)
    block = src[idx : end + len("</select>")]
    # The 'selected' attribute must NOT appear on a MiniMax option.
    assert 'value="MiniMax" selected' not in block, (
        "Basic-view <select> still pre-selects MiniMax. "
        "Default must be a local backend (llamacpp / ollama / transformers)."
    )


def test_advanced_view_default_is_not_minimax():
    """``<select id="llm-backend">`` (advanced view) must NOT pre-select MiniMax."""
    src = _read("web/index.html")
    idx = src.find('id="llm-backend"')
    assert idx > 0
    end = src.find("</select>", idx)
    block = src[idx : end + len("</select>")]
    assert 'value="MiniMax" selected' not in block, (
        "Advanced-view <select> still pre-selects MiniMax. Default must be a local backend."
    )


def test_onboarding_banner_is_vendor_neutral():
    """The onboarding banner copy must NOT name MiniMax as the
    pre-configured / recommended choice. First-visit UX should be
    neutral so users with no MiniMax account are not funnelled into
    one."""
    src = _read("web/index.html")
    # Find the onboarding-banner block
    idx = src.find("onboarding-banner")
    assert idx > 0
    end = src.find("</div>", idx)
    # Walk through nested </div>s to find the block end
    depth = 0
    cursor = idx
    while cursor < len(src):
        if src[cursor : cursor + 6] == "<div " or src[cursor : cursor + 5] == "<div>":
            depth += 1
        elif src[cursor : cursor + 6] == "</div>":
            depth -= 1
            if depth == 0:
                end = cursor + 6
                break
        cursor += 1
    block = src[idx:end]
    # Banner must not contain the marketing phrasings
    for phrase in (
        "已为您预设最佳配置",
        "MiniMax M3 云端 API",
        "MiniMax Token Plan",
        "platform.minimaxi.com",
    ):
        assert phrase not in block, (
            f"Onboarding banner still contains '{phrase}' — the first-visit UX "
            f"must not funnel users toward MiniMax."
        )


def test_recommended_badge_not_minimax_lockin():
    """The LLM status card must not carry a '默认 MiniMax M3' success-green
    badge — it implied MiniMax was the canonical/recommended choice."""
    src = _read("web/index.html")
    assert "默认 MiniMax M3" not in src, (
        "The '默认 MiniMax M3' success-green badge is still in the "
        "LLM status card — replace with a vendor-neutral backend label."
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


def test_js_backend_fallback_is_vendor_neutral():
    """web/js/app.js: the LLM backend fallback must not hardcode 'MiniMax'.
    The previous fallback silently routed every job to the cloud vendor
    when the dropdown was missing."""
    src = _read("web/js/app.js")
    # Strip comments and string literals from each line so a docstring
    # mention of 'MiniMax' (which is legitimate) doesn't trigger the
    # assertion. We only care about code paths.
    code_lines = []
    for line in src.splitlines():
        # Drop full-line comments
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        # Drop inline // comments
        if "//" in line:
            line = line.split("//", 1)[0]
        code_lines.append(line)
    code_only = "\n".join(code_lines)
    # Look for the fallback pattern in code paths
    bad_patterns = [
        "?.value ?? 'MiniMax'",
        "|| 'MiniMax'",
        "?? 'MiniMax'",
    ]
    for pat in bad_patterns:
        assert pat not in code_only, (
            f"web/js/app.js still contains the hardcoded MiniMax fallback "
            f"{pat!r} in code. Replace with a vendor-neutral local default."
        )
    # The model-name fallbacks (data.active_model || ... || 'MiniMax-M3')
    # also need to stop naming a vendor product. They should fall back
    # to '' or 'N/A' instead.
    assert "|| 'MiniMax-M3'" not in code_only, (
        "Model-name fallbacks (e.g. data.active_model || ... || 'MiniMax-M3') "
        "still name a vendor product. Replace with '' or 'N/A' so local-backend "
        "users don't see 'MiniMax-M3' as their model."
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


def test_dataclass_data_outbound_policy_is_redacted():
    """llm_backends.py MiniMaxM3Backend.data_outbound_policy must default
    to 'api_redacted', not 'api_full' (which sends full PDF text)."""
    import inspect

    from rlpe.llm_backends import MiniMaxM3Backend

    field = MiniMaxM3Backend.__dataclass_fields__["data_outbound_policy"]
    assert field.default == "api_redacted", (
        f"MiniMaxM3Backend.data_outbound_policy defaults to {field.default!r}; "
        f"must default to 'api_redacted' (privacy-by-default)."
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
