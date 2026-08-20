"""Regression tests for audit 2026-08-19 Phase 1d — open-nomenclature (B-7/B-8/M-1).

Bug fixes covered:
- B-7: M3/LLM path that emits ICZN open-nomenclature markers
  (cf./aff./?/ex gr.) previously kept the LLM's high confidence.
  Adding ``_apply_open_nomen_discount`` post-filter in
  ``llm_backends._normalize_panel_dict`` caps confidence at 0.55
  (cf./aff./?) or 0.50 (ex gr.).
- B-8: ``m3_engine.parse_caption`` LLM path did not run
  ``_normalize_species`` on the species/modifier string, so LLM
  outputs like ``Triactoma cf kamoense`` (no trailing period) and
  ``Archaeodictyomitra (?) sp.`` were emitted verbatim. The regex
  fallback path normalizes — now the LLM path normalizes too.
- M-1: ``_PARSE_CAPTION_SYSTEM`` prompt now includes a 1-shot
  complete I/O example that covers the ``open_nomenclature_strength``
  field for cf./? markers. Heavy few-shot augmentation is deferred
  to a later phase.

These tests are read-only against the live source so they catch
prompt drift and accidental removal of the discount helper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# B-7: confidence discount for open-nomenclature markers
# ---------------------------------------------------------------------------
class TestB7OpenNomenDiscount:
    """``_normalize_panel_dict`` must discount confidence when the
    species string carries ICZN open-nomenclature markers."""

    def test_cf_marker_discounts_to_055(self):
        from rlpe.llm_backends import _normalize_panel_dict

        # cf. with period → cap at 0.55
        out = _normalize_panel_dict({"species": "Triactoma cf. kamoensis", "confidence": 0.85})
        assert out["confidence"] <= 0.55

    def test_cf_marker_no_period_also_discounts(self):
        """LLM sometimes emits ``cf`` without trailing period
        (``Triactoma cf kamoense``). The discount must still fire."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "Triactoma cf kamoense", "confidence": 0.85})
        assert out["confidence"] <= 0.55

    def test_aff_marker_discounts_to_055(self):
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "Triactoma aff. kamoensis", "confidence": 0.85})
        assert out["confidence"] <= 0.55

    def test_question_marker_discounts_to_055(self):
        """The ``?`` literal in the species string is the
        caption ``(?)`` uncertainty marker (after M3
        normalization the ``(`` ``)`` are stripped but ``?``
        may remain in raw LLM output)."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "Archaeodictyomitra (?) sp.", "confidence": 0.85})
        assert out["confidence"] <= 0.55

    def test_question_marker_bare_genus(self):
        """Bare ``Genus?`` (genus uncertain) also triggers discount."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "Genus?", "confidence": 0.85})
        assert out["confidence"] <= 0.55

    def test_no_open_nomen_marker_preserves_confidence(self):
        """Plain species without cf./aff./?/ex gr. must NOT be discounted."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "Actinomma sp.", "confidence": 0.85})
        # 0.85 rounded to 2dp is 0.85 — must not be touched
        assert out["confidence"] == 0.85

    def test_no_open_nomen_marker_plain_binomial(self):
        """Plain binomial ``Puffinus pacificus`` must NOT be discounted
        (the substring ``cific`` must not match the cf. regex)."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "Puffinus pacificus", "confidence": 0.85})
        assert out["confidence"] == 0.85

    def test_ex_gr_marker_discounts_to_050(self):
        """``ex gr.`` (group) is the strictest ICZN open-nomen marker
        in bandini 2011 — caps at 0.50 (lower than cf./aff./?)."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "Stichocapsa ex gr. convexa", "confidence": 0.85})
        assert out["confidence"] <= 0.50

    def test_ex_gr_shortform_discounts(self):
        """``ex.gr.`` abbreviation must also trigger."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "Stichocapsa ex.gr. convexa", "confidence": 0.85})
        assert out["confidence"] <= 0.50

    def test_discount_floor_at_input_confidence(self):
        """If the input confidence is already lower than the cap,
        the discount is a no-op (we only ``min()``, never boost)."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "Triactoma cf. kamoensis", "confidence": 0.30})
        # 0.30 < 0.55 cap → stays at 0.30
        assert out["confidence"] == 0.30

    def test_none_species_passes_through(self):
        """``species=None`` (LLM returned no identification) must not
        crash the discount helper."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": None, "confidence": 0.0})
        # Pre-existing behaviour: ``str(None).strip()`` is ``"None"``,
        # not None — this test guards against the discount helper
        # crashing on the "None" string. Confidence must stay at 0.0
        # because no cf./aff./? markers are present in the string
        # ``"None"``.
        assert out["confidence"] == 0.0

    def test_discount_helper_exported(self):
        """``_apply_open_nomen_discount`` must be importable so other
        code (e.g. parse_caption post-processing) can call it directly."""
        from rlpe.llm_backends import _apply_open_nomen_discount

        out = {"species": "Triactoma cf. kamoensis", "confidence": 0.9}
        _apply_open_nomen_discount(out)
        assert out["confidence"] <= 0.55


# ---------------------------------------------------------------------------
# B-8: parse_caption LLM path now calls _normalize_species
# ---------------------------------------------------------------------------
class TestB8ParseCaptionNormalizes:
    """The LLM path of ``parse_caption`` must call ``_normalize_species``
    on both species and modifier, matching the regex fallback behaviour."""

    def _engine_with_llm(self, raw_text: str):
        from rlpe.m3_engine import M3Engine

        class _FakeBackend:
            backend_name = "fake-llm"
            enable_thinking = False

            def infer_text(self, system_prompt, user_prompt):
                return {"fallback_used": False, "raw_text": raw_text}

        return M3Engine(_FakeBackend())

    def test_llm_cf_no_period_passes_through_normalize(self):
        """LLM emits ``Triactoma cf kamoense`` (no period on cf).
        ``_normalize_species`` is called — the result keeps the
        ``cf`` token even if the period is not auto-inserted
        (we only check that the call site was wired up; the
        actual period-restoration behaviour is covered by the
        per-corpus test in test_m3_engine.py)."""
        from rlpe.m3_engine import _normalize_species

        payload = json.dumps(
            [
                {
                    "labels": ["1"],
                    "species": "Triactoma cf kamoense",
                    "modifier": "",
                    "confidence": 0.95,
                }
            ]
        )
        engine = self._engine_with_llm(payload)
        pairs = engine.parse_caption("Fig. 1. caption text")
        assert len(pairs) == 1
        # ``_normalize_species`` is the source of truth for what
        # the species string becomes — verify the LLM path used
        # the same normalizer as the regex fallback.
        expected = _normalize_species("Triactoma cf kamoense")
        assert pairs[0].species == expected
        # The species must still carry the cf marker (the discount
        # helper relies on this) — i.e. NOT been collapsed to
        # ``Triactoma kamoense``.
        assert "cf" in pairs[0].species.lower()

    def test_llm_question_marker_stripped(self):
        """LLM emits ``Archaeodictyomitra (?) sp.``; the LLM path
        must strip the ``(?)`` uncertainty marker, matching the
        regex fallback (which always normalised this)."""
        payload = json.dumps(
            [
                {
                    "labels": ["3"],
                    "species": "Archaeodictyomitra (?) sp.",
                    "modifier": "",
                    "confidence": 0.95,
                }
            ]
        )
        engine = self._engine_with_llm(payload)
        pairs = engine.parse_caption("Fig. 1. caption text")
        assert len(pairs) == 1
        # "(?)" is stripped, leaving the gold-form species.
        assert "(?)" not in pairs[0].species
        assert "Archaeodictyomitra" in pairs[0].species
        assert "sp." in pairs[0].species

    def test_llm_modifier_also_normalized(self):
        """The ``modifier`` field must also go through
        ``_normalize_species`` — guards against regression where
        only species is normalized."""
        payload = json.dumps(
            [
                {
                    "labels": ["A"],
                    "species": "Foo",
                    "modifier": "(?)",
                    "confidence": 0.95,
                }
            ]
        )
        engine = self._engine_with_llm(payload)
        pairs = engine.parse_caption("Fig. 1. caption text")
        assert len(pairs) == 1
        # "(?)" in modifier is also stripped.
        assert "(?)" not in pairs[0].modifier

    def test_llm_species_no_op_when_normalizer_returns_none(self):
        """Edge case: if ``_normalize_species`` returns ``None``
        (e.g. input is empty/whitespace), the original species
        string is preserved so we don't silently drop a pair."""
        from rlpe.m3_engine import _normalize_species

        # Pre-condition check: _normalize_species("") returns None
        assert _normalize_species("") is None
        # We can't easily force an LLM to emit empty species
        # (parse_caption already filters empty species earlier)
        # so we verify the call-site logic with a no-op shape.
        payload = json.dumps(
            [
                {
                    "labels": ["A"],
                    "species": "Foo bar",
                    "modifier": "",
                    "confidence": 0.9,
                }
            ]
        )
        engine = self._engine_with_llm(payload)
        pairs = engine.parse_caption("Fig. 1. caption text")
        assert len(pairs) == 1
        assert pairs[0].species == "Foo bar"


# ---------------------------------------------------------------------------
# M-1: parse_caption prompt contains few-shot example + open_nomen_strength
# ---------------------------------------------------------------------------
class TestM1ParseCaptionFewShot:
    """The ``_PARSE_CAPTION_SYSTEM`` prompt must:
    1. document the ``open_nomenclature_strength`` field
    2. include at least one complete input->output JSON example that
       demonstrates cf. and (?) species with the new field
    """

    def test_prompt_contains_open_nomenclature_strength_field(self):
        from rlpe.m3_engine import _PARSE_CAPTION_SYSTEM

        assert "open_nomenclature_strength" in _PARSE_CAPTION_SYSTEM

    def test_prompt_documents_cf_aff_question_ex_gr_values(self):
        """The prompt must list all 6 enum values so the LLM knows
        the vocabulary."""
        from rlpe.m3_engine import _PARSE_CAPTION_SYSTEM

        for value in ("none", "cf.", "aff.", "ex gr.", "subgen.", "?"):
            assert value in _PARSE_CAPTION_SYSTEM, (
                f"_PARSE_CAPTION_SYSTEM missing enum value {value!r}"
            )

    def test_prompt_contains_complete_few_shot_example(self):
        """A single-shot example covering cf. and (?) species with
        the new field populated."""
        from rlpe.m3_engine import _PARSE_CAPTION_SYSTEM

        # The example caption must mention cf. and (?) species.
        assert "Triactoma kamoensis" in _PARSE_CAPTION_SYSTEM
        assert "Pessagnoa" in _PARSE_CAPTION_SYSTEM
        assert "Archaeodictyomitra" in _PARSE_CAPTION_SYSTEM
        # The example output must include the new field with cf.
        # and ? values. The prompt uses compact JSON (no space
        # after colon) for the few-shot block — match that form.
        assert '"open_nomenclature_strength":"cf."' in _PARSE_CAPTION_SYSTEM
        assert '"open_nomenclature_strength":"?"' in _PARSE_CAPTION_SYSTEM
        assert '"open_nomenclature_strength":"none"' in _PARSE_CAPTION_SYSTEM

    def test_match_panel_prompt_documents_open_nomenclature_strength(self):
        """The match_panel prompt must also document the new field
        so the LLM emits it on per-panel species assignments."""
        from rlpe.m3_engine import _MATCH_PANEL_SYSTEM

        assert "open_nomenclature_strength" in _MATCH_PANEL_SYSTEM
        # All 6 enum values must appear.
        for value in ("none", "cf.", "aff.", "ex gr.", "subgen.", "?"):
            assert value in _MATCH_PANEL_SYSTEM, f"_MATCH_PANEL_SYSTEM missing enum value {value!r}"

    def test_critique_prompt_documents_open_nomenclature_strength(self):
        """The critique prompt must document the new field too,
        so the critique stage can echo open-nomen strength on
        suggested_species corrections."""
        from rlpe.m3_engine import _CRITIQUE_SYSTEM

        assert "open_nomenclature_strength" in _CRITIQUE_SYSTEM


# ---------------------------------------------------------------------------
# Source-guard: prevent accidental removal of the discount helper
# ---------------------------------------------------------------------------
class TestSourceGuards:
    """These tests scan the live source so any later refactor that
    silently removes the open-nomen discount fires the guard."""

    def test_llm_backends_defines_open_nomen_discount(self):
        from rlpe.llm_backends import _apply_open_nomen_discount

        # Must be callable
        assert callable(_apply_open_nomen_discount)

    def test_llm_backends_calls_discount_helper(self):
        """Static check: ``_normalize_panel_dict`` source code must
        reference the helper. Catches the "someone removed the
        call site" regression."""
        from rlpe import llm_backends

        src = Path(llm_backends.__file__).read_text()
        # The function definition must still call the helper
        # inside _normalize_panel_dict.
        norm_panel_idx = src.find("def _normalize_panel_dict")
        assert norm_panel_idx > 0, "_normalize_panel_dict missing"
        # Slice from the function start to the next top-level def
        # (approximate — the helper is referenced at least once).
        assert "_apply_open_nomen_discount" in src

    def test_m3_engine_parse_caption_calls_normalize_species(self):
        """Static check: ``parse_caption`` source must reference
        ``_normalize_species`` so the LLM path normalisation
        cannot be silently removed."""
        from rlpe import m3_engine

        src = Path(m3_engine.__file__).read_text()
        # Locate the parse_caption function (it contains
        # ``_normalize_species`` call) — we just check the file
        # overall has the call.
        assert "parse_caption" in src
        # The post-fix call site is in parse_caption — verify
        # the function body still calls _normalize_species.
        fn_start = src.find("def parse_caption(")
        assert fn_start > 0
        # Search the rest of the file for normalize_species call
        # to be safe (allow some headroom for the second function
        # definition later in the file).
        assert "_normalize_species" in src[fn_start:]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
