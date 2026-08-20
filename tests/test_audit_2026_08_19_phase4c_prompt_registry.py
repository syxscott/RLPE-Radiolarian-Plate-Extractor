"""Regression tests for audit 2026-08-19 Phase 4C — unified M3 / Gemma
prompt registry + Gemma postprocess field-name fallback.

Bug fixes covered:
- M-10 (Gemma fallback receives stale M3 prompt): Gemma
  ``apply_gemma_to_matches`` / ``batch_gemma_postprocess_rows`` used to
  re-define their own per-panel system prompt inline. When M3's prompt
  was updated upstream (Phase 27 added the Japanese parse_caption
  prompt; Phase 64/65/66 added new prompts to the registry) the inline
  Gemma copy silently drifted, so the Gemma fallback path after an M3
  failure used a STALE prompt that no longer matched M3's JSON
  contract. Fix: unify through ``rlpe.m3_engine.get_prompt_registry()``
  and let Gemma pull from the registry.

- M-11 (Gemma schema drift): M3's per-stage output has used several
  field-name variants across migration cycles —
  ``confidence`` / ``conf_score`` / ``c_score`` / ``score`` for the
  probability field, and ``verbatim_name`` / ``raw_name`` / ``name`` /
  ``taxon`` for the raw species string. Gemma hard-coded ``out.get(
  "confidence")`` and ``out.get("species")``, so a successful M3 call
  with ``conf_score`` would silently be mapped to ``gemma_conf = 0.0``
  and the row marked fallback. Fix: add a field-name fallback list and
  a shared ``_pick_field`` helper.

- M-12 (M3 prompt duplicates): ``m3_engine`` and ``gemma_postprocess``
  used to each maintain their own copy of the per-panel prompt.
  ``get_prompt_registry()`` is now the single source of truth; Gemma
  imports it instead of re-defining the prompt.

These tests are read-only against the live source so they catch
prompt drift, accidental removal of the registry helper, and removal
of the field-name fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ===========================================================================
# M-10 / M-12: prompt registry exists and exposes the canonical M3 prompts
# ===========================================================================


class TestPromptRegistry:
    """``m3_engine.get_prompt_registry()`` must be importable and
    return a tuple ``(dict, version_str)`` containing the canonical
    5+ stage system prompts (audit Phase 4E / Phase 4C)."""

    def test_get_prompt_registry_importable(self):
        from rlpe.m3_engine import get_prompt_registry

        assert callable(get_prompt_registry)

    def test_get_prompt_registry_returns_tuple(self):
        """Phase 4E: registry now returns ``(dict, version_str)`` so
        callers can pin a result to a known prompt version."""
        from rlpe.m3_engine import get_prompt_registry

        result = get_prompt_registry()
        assert isinstance(result, tuple)
        assert len(result) == 2
        registry, version = result
        assert isinstance(registry, dict)
        assert isinstance(version, str)
        assert len(version) > 0, "version stamp must be non-empty"

    def test_get_prompt_registry_has_expected_stages(self):
        """The registry must cover at least the 5 vision stages plus
        the Japanese parse_caption variant (audit 2026-08-19 Bug
        M-12: registry completeness guard)."""
        from rlpe.m3_engine import get_prompt_registry

        registry, _version = get_prompt_registry()
        expected_keys = {
            "parse_caption",
            "parse_caption_ja",
            "classify_plate",
            "segment_panels",
            "match_panel",
            "match_panel_visual_only",
            "critique_matches",
        }
        missing = expected_keys - set(registry.keys())
        assert not missing, (
            f"prompt registry missing stages: {missing}; found: {set(registry.keys())}"
        )
        assert len(registry) >= 5, (
            f"registry should expose at least 5 stage keys, got {len(registry)}"
        )

    def test_each_prompt_is_substantive(self):
        """No empty / placeholder prompts in the registry — each is a
        multi-stage JSON contract several-hundred tokens long."""
        from rlpe.m3_engine import get_prompt_registry

        registry, _version = get_prompt_registry()
        for key, prompt in registry.items():
            assert isinstance(prompt, str), f"{key} prompt is not a str"
            assert len(prompt) > 200, (
                f"{key} prompt suspiciously short ({len(prompt)} chars); "
                "should be a multi-stage vision JSON contract"
            )

    def test_registry_keys_match_module_constants(self):
        """Sanity: the registry keys must mirror the module-level
        constants in m3_engine (no accidental renaming)."""
        from rlpe.m3_engine import (
            _CLASSIFY_PLATE_SYSTEM,
            _CRITIQUE_SYSTEM,
            _MATCH_PANEL_SYSTEM,
            _MATCH_PANEL_SYSTEM_VISUAL_ONLY,
            _PARSE_CAPTION_SYSTEM,
            _PARSE_CAPTION_SYSTEM_JA,
            _SEGMENT_PANELS_SYSTEM,
            get_prompt_registry,
        )

        registry, _version = get_prompt_registry()
        assert registry["parse_caption"] == _PARSE_CAPTION_SYSTEM
        assert registry["parse_caption_ja"] == _PARSE_CAPTION_SYSTEM_JA
        assert registry["classify_plate"] == _CLASSIFY_PLATE_SYSTEM
        assert registry["segment_panels"] == _SEGMENT_PANELS_SYSTEM
        assert registry["match_panel"] == _MATCH_PANEL_SYSTEM
        assert registry["match_panel_visual_only"] == _MATCH_PANEL_SYSTEM_VISUAL_ONLY
        assert registry["critique_matches"] == _CRITIQUE_SYSTEM

    def test_registry_returns_independent_dict(self):
        """Mutating the returned dict MUST NOT poison the cached
        module-level constants (defensive copy contract)."""
        from rlpe.m3_engine import get_prompt_registry

        registry1, _v1 = get_prompt_registry()
        original_len = len(registry1)
        registry1["_poison_key_for_test"] = "should not persist"
        registry2, _v2 = get_prompt_registry()
        assert "_poison_key_for_test" not in registry2
        assert len(registry2) == original_len

    def test_registry_version_constant_exported(self):
        """Phase 4E: the ``PROMPT_REGISTRY_VERSION`` constant must be
        importable and follow the ``vMAJOR.MINOR.PATCH`` convention."""
        from rlpe.m3_engine import PROMPT_REGISTRY_VERSION

        assert isinstance(PROMPT_REGISTRY_VERSION, str)
        assert PROMPT_REGISTRY_VERSION.startswith("v")
        # Bump semantics: major for schema change, minor for wording,
        # patch for typos. We only check the general shape here.
        parts = PROMPT_REGISTRY_VERSION.lstrip("v").split(".")
        assert len(parts) == 3, f"version {PROMPT_REGISTRY_VERSION!r} must be vMAJOR.MINOR.PATCH"
        for part in parts:
            assert part.isdigit(), f"non-numeric version segment {part!r}"


# ===========================================================================
# M-10: Gemma postprocess uses M3's canonical prompts (no drift)
# ===========================================================================


class TestGemmaUsesM3Prompts:
    """Gemma postprocess must pull per-panel prompts from the M3
    registry, not re-define them inline (audit 2026-08-19 Bug M-10)."""

    def test_gemma_module_imports_m3_registry(self):
        """Static guard: ``gemma_postprocess`` must import
        ``get_prompt_registry`` from ``m3_engine`` so a refactor that
        re-defines the prompt inline breaks the test loudly."""
        import rlpe.gemma_postprocess as mod

        assert hasattr(mod, "get_prompt_registry"), (
            "gemma_postprocess must import get_prompt_registry from m3_engine; "
            "if this test fails, someone re-introduced the inline duplicate prompt"
        )

    def test_gemma_get_system_prompt_matches_m3_registry(self):
        """Compare ``gemma._get_system_prompt('match_panel')`` with
        ``m3.get_prompt_registry()[0]['match_panel']`` — they must be
        the SAME STRING. A drift here means Gemma would emit a
        different JSON contract than M3 (the original M-10 bug)."""
        import rlpe.gemma_postprocess as gemma
        from rlpe.m3_engine import get_prompt_registry

        registry, _version = get_prompt_registry()
        m3_prompt = registry["match_panel"]
        gemma_prompt = gemma._get_system_prompt("match_panel")
        assert gemma_prompt == m3_prompt, (
            "Gemma's match_panel prompt has drifted from M3's. "
            "This means Gemma fallback will silently emit a different "
            "JSON shape than the M3 outputs it is supposed to replace."
        )

    def test_gemma_get_system_prompt_visual_only_matches(self):
        import rlpe.gemma_postprocess as gemma
        from rlpe.m3_engine import get_prompt_registry

        registry, _version = get_prompt_registry()
        m3_prompt = registry["match_panel_visual_only"]
        gemma_prompt = gemma._get_system_prompt("match_panel_visual_only")
        assert gemma_prompt == m3_prompt

    def test_gemma_get_system_prompt_unknown_returns_none(self):
        """Unknown stages return None so callers fall back to the
        legacy inline prompt (graceful degradation)."""
        import rlpe.gemma_postprocess as gemma

        assert gemma._get_system_prompt("nonexistent_stage_xyz") is None

    def test_gemma_stage_aliases_resolve(self):
        """``zh`` / ``en`` / ``match`` aliases are accepted so legacy
        callers keep working without code changes."""
        import rlpe.gemma_postprocess as gemma
        from rlpe.m3_engine import get_prompt_registry

        registry, _version = get_prompt_registry()
        assert gemma._get_system_prompt("zh") == registry["match_panel"]
        assert gemma._get_system_prompt("match") == registry["match_panel"]
        assert gemma._get_system_prompt("en") == registry["match_panel_visual_only"]

    def test_gemma_handles_tuple_and_legacy_dict_registries(self):
        """Phase 4E: ``_get_m3_prompts`` must transparently accept both
        the new tuple-returning registry and a legacy dict for
        backward compatibility during the migration."""
        import rlpe.gemma_postprocess as gemma

        # Force a re-load with a stubbed tuple-shaped registry.
        gemma._PROMPTS_CACHE = None
        gemma._PROMPTS_VERSION = None
        with gemma._PROMPTS_LOCK if hasattr(gemma, "_PROMPTS_LOCK") else _noop_lock():
            original = gemma.get_prompt_registry
            try:
                gemma.get_prompt_registry = lambda: ({"match_panel": "Z"}, "v9.9.9")
                gemma._PROMPTS_CACHE = None
                gemma._PROMPTS_VERSION = None
                prompts = gemma._get_m3_prompts()
            finally:
                gemma.get_prompt_registry = original
        assert prompts.get("match_panel") == "Z"

        # Now with a legacy dict-shaped registry.
        gemma._PROMPTS_CACHE = None
        gemma._PROMPTS_VERSION = None
        with gemma._PROMPTS_LOCK if hasattr(gemma, "_PROMPTS_LOCK") else _noop_lock():
            original = gemma.get_prompt_registry
            try:
                gemma.get_prompt_registry = lambda: {"match_panel": "Y"}
                gemma._PROMPTS_CACHE = None
                gemma._PROMPTS_VERSION = None
                prompts = gemma._get_m3_prompts()
            finally:
                gemma.get_prompt_registry = original
        assert prompts.get("match_panel") == "Y"


from contextlib import contextmanager


@contextmanager
def _noop_lock():
    """Tiny contextmanager so the import block above doesn't need a
    try/except around an internal lock in gemma_postprocess."""
    yield


# ===========================================================================
# M-11: Gemma postprocess field-name fallback (confidence / species)
# ===========================================================================


class TestConfidenceFieldFallback:
    """Gemma postprocess must read the first present field out of
    ``{confidence, conf_score, c_score, score}`` (audit M-11)."""

    def test_pick_field_prefers_confidence(self):
        from rlpe.gemma_postprocess import _CONFIDENCE_FIELD_FALLBACK, _pick_field

        # ``confidence`` is the canonical M3 name and must be preferred.
        payload = {"confidence": 0.9, "conf_score": 0.5, "score": 0.1}
        assert _pick_field(payload, _CONFIDENCE_FIELD_FALLBACK) == 0.9

    def test_pick_field_falls_back_to_conf_score(self):
        from rlpe.gemma_postprocess import _CONFIDENCE_FIELD_FALLBACK, _pick_field

        # Older prompts / YOLO paths emit ``conf_score`` instead.
        payload = {"conf_score": 0.95}
        assert _pick_field(payload, _CONFIDENCE_FIELD_FALLBACK) == 0.95

    def test_pick_field_falls_back_to_score(self):
        from rlpe.gemma_postprocess import _CONFIDENCE_FIELD_FALLBACK, _pick_field

        # Layout / pipeline paths use the bare ``score`` key.
        payload = {"score": 0.85}
        assert _pick_field(payload, _CONFIDENCE_FIELD_FALLBACK) == 0.85

    def test_pick_field_falls_back_to_c_score(self):
        from rlpe.gemma_postprocess import _CONFIDENCE_FIELD_FALLBACK, _pick_field

        # Stage 3 bbox/crop paths use ``c_score``.
        payload = {"c_score": 0.7}
        assert _pick_field(payload, _CONFIDENCE_FIELD_FALLBACK) == 0.7

    def test_pick_field_missing_returns_none(self):
        from rlpe.gemma_postprocess import _CONFIDENCE_FIELD_FALLBACK, _pick_field

        assert _pick_field({}, _CONFIDENCE_FIELD_FALLBACK) is None
        assert _pick_field({"unrelated": 1.0}, _CONFIDENCE_FIELD_FALLBACK) is None

    def test_pick_field_skips_null(self):
        """A present-but-null entry must not be returned (caller
        treated as missing)."""
        from rlpe.gemma_postprocess import _CONFIDENCE_FIELD_FALLBACK, _pick_field

        payload = {"confidence": None, "conf_score": 0.8}
        assert _pick_field(payload, _CONFIDENCE_FIELD_FALLBACK) == 0.8

    def test_pick_field_with_non_dict_payload(self):
        """Defensive: ``_pick_field`` should not crash on a non-dict."""
        from rlpe.gemma_postprocess import _CONFIDENCE_FIELD_FALLBACK, _pick_field

        assert _pick_field(None, _CONFIDENCE_FIELD_FALLBACK) is None
        assert _pick_field("string", _CONFIDENCE_FIELD_FALLBACK) is None
        assert _pick_field(42, _CONFIDENCE_FIELD_FALLBACK) is None


class TestNameFieldFallback:
    """Gemma postprocess must read the first present field out of
    ``{verbatim_name, raw_name, name, taxon}`` (audit M-11)."""

    def test_pick_field_prefers_verbatim_name(self):
        from rlpe.gemma_postprocess import _NAME_FIELD_FALLBACK, _pick_field

        # ``verbatim_name`` is the canonical 2026-08-19 B-2 schema.
        payload = {"verbatim_name": "Entactinia", "raw_name": "other"}
        assert _pick_field(payload, _NAME_FIELD_FALLBACK) == "Entactinia"

    def test_pick_field_falls_back_to_raw_name(self):
        from rlpe.gemma_postprocess import _NAME_FIELD_FALLBACK, _pick_field

        payload = {"raw_name": "Entactinia"}
        assert _pick_field(payload, _NAME_FIELD_FALLBACK) == "Entactinia"

    def test_pick_field_falls_back_to_name(self):
        from rlpe.gemma_postprocess import _NAME_FIELD_FALLBACK, _pick_field

        # Older M3 prompts emitted ``name``.
        payload = {"name": "Entactinia"}
        assert _pick_field(payload, _NAME_FIELD_FALLBACK) == "Entactinia"

    def test_pick_field_falls_back_to_taxon(self):
        from rlpe.gemma_postprocess import _NAME_FIELD_FALLBACK, _pick_field

        # ``taxon`` is used by some converter paths.
        payload = {"taxon": "Entactinia"}
        assert _pick_field(payload, _NAME_FIELD_FALLBACK) == "Entactinia"

    def test_pick_field_missing_returns_none(self):
        from rlpe.gemma_postprocess import _NAME_FIELD_FALLBACK, _pick_field

        assert _pick_field({}, _NAME_FIELD_FALLBACK) is None


# ===========================================================================
# M-11: confidence coercion robustness (catches TypeError / ValueError)
# ===========================================================================


class TestConfidenceCoercion:
    """The confidence coercion must not crash on a non-numeric M3
    payload — auditing claim: a malformed ``conf_score`` string should
    become ``gemma_conf = 0.0`` rather than raise."""

    def test_string_numeric_coerced(self):
        from rlpe.gemma_postprocess import _CONFIDENCE_FIELD_FALLBACK, _pick_field

        # ``conf_score`` arrives as a string sometimes (provider side).
        payload = {"conf_score": "0.92"}
        raw = _pick_field(payload, _CONFIDENCE_FIELD_FALLBACK)
        try:
            val = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            val = 0.0
        assert val == pytest.approx(0.92)

    def test_garbage_value_becomes_zero(self):
        from rlpe.gemma_postprocess import _CONFIDENCE_FIELD_FALLBACK, _pick_field

        # Garbage payload must NOT raise — pattern used in
        # ``apply_gemma_to_matches`` wraps the conversion in
        # try/except (TypeError, ValueError).
        payload = {"conf_score": "not-a-number"}
        raw = _pick_field(payload, _CONFIDENCE_FIELD_FALLBACK)
        try:
            val = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            val = 0.0
        assert val == 0.0


# ===========================================================================
# Few-shot format guards (the original M3 prompts must keep their examples)
# ===========================================================================


class TestM3PromptsRetainFewShotFormat:
    """The prompts we now share with Gemma must remain formal few-shot
    examples — the audit guard against accidental prompt-text removal
    that would silently degrade the LLM output."""

    def test_match_panel_prompt_has_fewshot_example(self):
        """``_MATCH_PANEL_SYSTEM`` must contain at least one complete
        input->output example after the refactor."""
        from rlpe.m3_engine import _MATCH_PANEL_SYSTEM

        assert "Example" in _MATCH_PANEL_SYSTEM
        # The example must reference the JSON shape Gemma now reads
        # (label, species, confidence, reasoning).
        for key in ("label", "species", "confidence", "reasoning"):
            assert key in _MATCH_PANEL_SYSTEM, f"_MATCH_PANEL_SYSTEM missing output key {key!r}"

    def test_critique_system_has_fewshot_example(self):
        from rlpe.m3_engine import _CRITIQUE_SYSTEM

        assert "Example" in _CRITIQUE_SYSTEM

    def test_classify_plate_has_fewshot_example(self):
        from rlpe.m3_engine import _CLASSIFY_PLATE_SYSTEM

        assert "Example" in _CLASSIFY_PLATE_SYSTEM

    def test_segment_panels_has_fewshot_example(self):
        from rlpe.m3_engine import _SEGMENT_PANELS_SYSTEM

        assert "Example" in _SEGMENT_PANELS_SYSTEM

    def test_visual_only_has_structured_output(self):
        """``_MATCH_PANEL_SYSTEM_VISUAL_ONLY`` is a concise prompt and
        may not embed an ``Example`` header literally, but it MUST
        document the JSON output schema Gemma reads."""
        from rlpe.m3_engine import _MATCH_PANEL_SYSTEM_VISUAL_ONLY

        for key in ("label", "species", "confidence", "reasoning"):
            assert key in _MATCH_PANEL_SYSTEM_VISUAL_ONLY, (
                f"_MATCH_PANEL_SYSTEM_VISUAL_ONLY missing output key {key!r}"
            )


# ===========================================================================
# End-to-end smoke: Gemma wire format unchanged at the API surface
# ===========================================================================


class TestGemmaAPISurfaceUnchanged:
    """The Phase 4C refactor MUST NOT change the public Gemma API:
    every existing import path must keep working so other tests don't
    regress."""

    def test_public_functions_still_importable(self):
        import rlpe.gemma_postprocess as mod

        # Each public surface used by other tests / scripts. The
        # Phase 4C migration dropped the legacy ``GEMMA_SYSTEM_PROMPT_ZH``
        # / ``_EN`` module constants in favour of ``_get_system_prompt``
        # (the M3 registry), so those two are NOT expected to remain.
        for name in (
            "GemmaRuntime",
            "set_global_seed",
            "load_gemma4_model",
            "load_gemma4_ollama",
            "load_gemma4_llamacpp",
            "build_gemma_backend_from_config",
            "gemma_match_panel",
            "gemma_extract_text_json",
            "apply_gemma_to_matches",
            "batch_gemma_postprocess_rows",
            "get_prompt_registry",
            "_get_system_prompt",
            "_get_m3_prompts",
            "_get_m3_prompt_version",
            "_pick_field",
            "_CONFIDENCE_FIELD_FALLBACK",
            "_NAME_FIELD_FALLBACK",
        ):
            assert hasattr(mod, name), f"gemma_postprocess missing public attr {name!r}"

    def test_field_fallback_constants_are_tuples(self):
        """The fallback lists must be tuples (immutable) so a caller
        can't accidentally mutate them."""
        from rlpe.gemma_postprocess import (
            _CONFIDENCE_FIELD_FALLBACK,
            _NAME_FIELD_FALLBACK,
        )

        assert isinstance(_CONFIDENCE_FIELD_FALLBACK, tuple)
        assert isinstance(_NAME_FIELD_FALLBACK, tuple)

    def test_field_fallback_constants_include_legacy_keys(self):
        """Defensive: the fallback lists MUST contain the canonical
        current key AND the legacy aliases (regression guard against
        someone deleting one)."""
        from rlpe.gemma_postprocess import (
            _CONFIDENCE_FIELD_FALLBACK,
            _NAME_FIELD_FALLBACK,
        )

        for key in ("confidence", "conf_score", "c_score", "score"):
            assert key in _CONFIDENCE_FIELD_FALLBACK

        for key in ("verbatim_name", "raw_name", "name", "taxon"):
            assert key in _NAME_FIELD_FALLBACK
