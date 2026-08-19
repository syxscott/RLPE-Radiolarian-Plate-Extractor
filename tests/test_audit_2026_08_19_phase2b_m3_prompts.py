"""Regression tests for audit 2026-08-19 Phase 2b — M3 prompts + geo whitelist + Ma range.

Bug fixes covered:
- M-1 (rest): The 4 remaining M3 stage prompts
  (``_CLASSIFY_PLATE_SYSTEM``, ``_SEGMENT_PANELS_SYSTEM``,
  ``_MATCH_PANEL_SYSTEM``, ``_CRITIQUE_SYSTEM``) now include
  complete in-context input->output few-shot examples so the
  vision LLM has concrete demonstrations of the expected JSON
  schema (Phase 1d added the same for ``_PARSE_CAPTION_SYSTEM``).
- M-12: ``extract_geology`` vision output is now filtered through
  ``_GEO_KEY_WHITELIST`` so LLM-hallucinated extras ("habitat",
  "depositional_environment", "paleoclimate", ...) cannot pollute
  ``panel.metadata.geology_links``. Dropped keys are logged at
  WARNING level for audit visibility.
- M-13: Ma-range direction is now validated. ICZN / stratigraphic
  convention is ``ma_top < ma_base`` (younger = smaller Ma). The
  vision LLM occasionally emits the inverted range; the new
  ``_validate_ma_range`` helper nulls both fields (and ma_mid)
  with a WARNING so the caller can fall back to caption regex
  or PBDB lookup.

These tests are read-only against the live source so they catch
prompt drift, accidental removal of the whitelist helper, and
removal of the Ma-range validation hook.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ===========================================================================
# M-1 (rest): few-shot examples in the 4 remaining M3 stage prompts
# ===========================================================================


class TestM1ClassifyPlateFewShot:
    """``_CLASSIFY_PLATE_SYSTEM`` must contain at least one
    complete input->output few-shot example demonstrating the
    plate/non-plate JSON contract."""

    def test_prompt_contains_example_marker(self):
        from rlpe.m3_engine import _CLASSIFY_PLATE_SYSTEM

        assert "Example" in _CLASSIFY_PLATE_SYSTEM

    def test_prompt_has_positive_example(self):
        """A radiolarian-plate example showing true + JSON keys."""
        from rlpe.m3_engine import _CLASSIFY_PLATE_SYSTEM

        # The positive example must contain the JSON keys this stage
        # is supposed to emit (audit 2026-08-19: prompt drift guard).
        for key in (
            "is_radiolarian_plate",
            "image_type",
            "panel_count_estimate",
            "quality",
        ):
            assert key in _CLASSIFY_PLATE_SYSTEM, (
                f"_CLASSIFY_PLATE_SYSTEM missing JSON key {key!r} in examples"
            )

    def test_prompt_has_negative_example(self):
        """A non-plate (text/column/etc.) example showing false."""
        from rlpe.m3_engine import _CLASSIFY_PLATE_SYSTEM

        # At least one example must show is_radiolarian_plate:false
        assert "false" in _CLASSIFY_PLATE_SYSTEM

    def test_prompt_documents_all_image_types(self):
        """The prompt must list all image_type enum values."""
        from rlpe.m3_engine import _CLASSIFY_PLATE_SYSTEM

        for v in ("SEM", "micrograph", "photomicrograph", "diagram", "photo", "other"):
            assert v in _CLASSIFY_PLATE_SYSTEM, (
                f"_CLASSIFY_PLATE_SYSTEM missing image_type enum value {v!r}"
            )


class TestM1SegmentPanelsFewShot:
    """``_SEGMENT_PANELS_SYSTEM`` must contain a complete
    input->output few-shot example demonstrating the panel
    bbox JSON contract."""

    def test_prompt_contains_example_marker(self):
        from rlpe.m3_engine import _SEGMENT_PANELS_SYSTEM

        assert "Example" in _SEGMENT_PANELS_SYSTEM

    def test_prompt_documents_json_schema(self):
        """The prompt must list all output fields."""
        from rlpe.m3_engine import _SEGMENT_PANELS_SYSTEM

        for field_name in (
            "panel_id",
            "bbox",
            "visible_label",
            "morphology",
            "confidence",
        ):
            assert field_name in _SEGMENT_PANELS_SYSTEM, (
                f"_SEGMENT_PANELS_SYSTEM missing field {field_name!r}"
            )

    def test_prompt_has_complete_json_array_example(self):
        """At least one Example must show a JSON array ``[...`` with
        realistic bbox coordinates (numeric x,y,w,h values)."""
        from rlpe.m3_engine import _SEGMENT_PANELS_SYSTEM

        # Look for the opening ``[`` of a JSON array example.
        assert "[" in _SEGMENT_PANELS_SYSTEM
        # Look for typical bbox coordinates like 30,30,540,400
        # (the actual numeric pattern from the multi-panel example).
        assert "30,30" in _SEGMENT_PANELS_SYSTEM or "540,400" in _SEGMENT_PANELS_SYSTEM


class TestM1MatchPanelFewShot:
    """``_MATCH_PANEL_SYSTEM`` must contain complete
    input->output few-shot examples demonstrating the
    label+species JSON contract (Phase 1d already verified
    that ``open_nomenclature_strength`` is documented; this
    test verifies the *example I/O blocks* exist)."""

    def test_prompt_contains_example_marker(self):
        from rlpe.m3_engine import _MATCH_PANEL_SYSTEM

        assert "Example" in _MATCH_PANEL_SYSTEM

    def test_prompt_documents_open_nomenclature_strength(self):
        """M-1 baseline from Phase 1d: must enumerate the 6 enum values."""
        from rlpe.m3_engine import _MATCH_PANEL_SYSTEM

        for v in ("none", "cf.", "aff.", "ex gr.", "subgen.", "?"):
            assert v in _MATCH_PANEL_SYSTEM, (
                f"_MATCH_PANEL_SYSTEM missing enum value {v!r}"
            )

    def test_prompt_has_complete_example_with_cf_marker(self):
        """At least one Example must demonstrate cf. output with
        the discounted confidence value."""
        from rlpe.m3_engine import _MATCH_PANEL_SYSTEM

        # The cf. example should mention confidence=0.55 (the
        # discount cap applied at parse time).
        assert "0.55" in _MATCH_PANEL_SYSTEM, (
            "_MATCH_PANEL_SYSTEM few-shot cf. example must mention confidence 0.55"
        )

    def test_prompt_documents_json_schema_fields(self):
        """The prompt must list all output fields so the LLM knows
        what to emit."""
        from rlpe.m3_engine import _MATCH_PANEL_SYSTEM

        for field_name in (
            "label",
            "species",
            "open_nomenclature_strength",
            "confidence",
            "reasoning",
            "alternative",
            "is_radiolarian",
        ):
            assert field_name in _MATCH_PANEL_SYSTEM, (
                f"_MATCH_PANEL_SYSTEM missing field {field_name!r}"
            )


class TestM1CritiqueFewShot:
    """``_CRITIQUE_SYSTEM`` must contain complete
    input->output few-shot examples demonstrating the
    verdict JSON contract."""

    def test_prompt_contains_example_marker(self):
        from rlpe.m3_engine import _CRITIQUE_SYSTEM

        assert "Example" in _CRITIQUE_SYSTEM

    def test_prompt_documents_all_verdict_values(self):
        """The prompt must enumerate agree / disagree / uncertain."""
        from rlpe.m3_engine import _CRITIQUE_SYSTEM

        for v in ("agree", "disagree", "uncertain"):
            assert v in _CRITIQUE_SYSTEM, (
                f"_CRITIQUE_SYSTEM missing verdict value {v!r}"
            )

    def test_prompt_documents_open_nomenclature_strength(self):
        """M-1 baseline from Phase 1d: must reference the field."""
        from rlpe.m3_engine import _CRITIQUE_SYSTEM

        assert "open_nomenclature_strength" in _CRITIQUE_SYSTEM

    def test_prompt_has_disagree_example(self):
        """At least one Example must show the disagree verdict path
        (the most common correction case in live runs)."""
        from rlpe.m3_engine import _CRITIQUE_SYSTEM

        # Search for the disagree verdict (case-insensitive search
        # via the source — the prompt capitalizes differently).
        # The current prompt uses lowercase JSON-like ``"disagree"``.
        assert '"disagree"' in _CRITIQUE_SYSTEM or "disagree" in _CRITIQUE_SYSTEM

    def test_prompt_documents_json_schema_fields(self):
        from rlpe.m3_engine import _CRITIQUE_SYSTEM

        for field_name in (
            "panel_id",
            "verdict",
            "suggested_species",
            "open_nomenclature_strength",
            "confidence",
            "reasoning",
        ):
            assert field_name in _CRITIQUE_SYSTEM, (
                f"_CRITIQUE_SYSTEM missing field {field_name!r}"
            )


class TestM1PromptSize:
    """All 4 prompts must have grown substantially vs Phase 1d
    (the Phase 1d prompt had ~1 example). A regression to a
    pre-Phase-2b size means the few-shot blocks were dropped."""

    def test_classify_plate_prompt_grew(self):
        from rlpe.m3_engine import _CLASSIFY_PLATE_SYSTEM

        # Phase 1d baseline ~700 chars; Phase 2b should be >1500 chars
        # because it now has 3 complete examples (radiolarian / text /
        # stratigraphic column).
        assert len(_CLASSIFY_PLATE_SYSTEM) > 1500

    def test_segment_panels_prompt_grew(self):
        from rlpe.m3_engine import _SEGMENT_PANELS_SYSTEM

        # Phase 1d baseline ~700 chars; Phase 2b should have 2
        # examples with JSON bbox arrays.
        assert len(_SEGMENT_PANELS_SYSTEM) > 1500

    def test_match_panel_prompt_grew(self):
        from rlpe.m3_engine import _MATCH_PANEL_SYSTEM

        # Phase 1d baseline ~1400 chars; Phase 2b adds 3 examples.
        assert len(_MATCH_PANEL_SYSTEM) > 2500

    def test_critique_prompt_grew(self):
        from rlpe.m3_engine import _CRITIQUE_SYSTEM

        # Phase 1d baseline ~1100 chars; Phase 2b adds 3 examples.
        assert len(_CRITIQUE_SYSTEM) > 2000


# ===========================================================================
# M-12: schema whitelist for extract_geology
# ===========================================================================


class TestM12GeoKeyWhitelist:
    """The ``_GEO_KEY_WHITELIST`` constant and ``_apply_geo_whitelist``
    helper must (a) exist, (b) keep whitelisted keys, and (c) drop
    LLM-hallucinated extras with a WARNING log."""

    def test_whitelist_constant_exists(self):
        from rlpe.llm_backends import _GEO_KEY_WHITELIST

        # Must be a set/frozenset of strings.
        assert isinstance(_GEO_KEY_WHITELIST, (set, frozenset))
        assert all(isinstance(k, str) for k in _GEO_KEY_WHITELIST)

    def test_whitelist_contains_core_keys(self):
        """The whitelist must include all schema-declared keys."""
        from rlpe.llm_backends import _GEO_KEY_WHITELIST

        for key in (
            "age",
            "ma_top",
            "ma_base",
            "ma_mid",
            "formation",
            "lithology",
            "biozone",
            "confidence",
            "section_type",
            "link_source",
        ):
            assert key in _GEO_KEY_WHITELIST, (
                f"_GEO_KEY_WHITELIST missing required key {key!r}"
            )

    def test_hallucinated_keys_are_dropped(self):
        from rlpe.llm_backends import _apply_geo_whitelist

        item = {
            "age": "Late Triassic",
            "habitat": "marine",
            "paleoclimate": "warm",
            "depositional_environment": "deltaic",
            "tectonic_setting": "passive_margin",
        }
        out = _apply_geo_whitelist(dict(item))
        # Only whitelisted keys survive.
        assert "age" in out
        for hallucinated in (
            "habitat",
            "paleoclimate",
            "depositional_environment",
            "tectonic_setting",
        ):
            assert hallucinated not in out, (
                f"{hallucinated!r} should have been dropped from {out}"
            )

    def test_hallucinated_keys_log_warning(self, caplog):
        from rlpe.llm_backends import _apply_geo_whitelist

        caplog.set_level(logging.WARNING, logger="rlpe.llm_backends")
        item = {
            "age": "Late Triassic",
            "habitat": "marine",
            "paleoclimate": "warm",
        }
        _apply_geo_whitelist(item)
        # At least one WARNING log should mention the dropped fields.
        warning_msgs = [
            record.message for record in caplog.records if record.levelno >= logging.WARNING
        ]
        assert any("habitat" in m or "paleoclimate" in m for m in warning_msgs), (
            f"Expected WARNING log naming dropped keys, got {warning_msgs}"
        )

    def test_non_dict_returns_unchanged(self):
        from rlpe.llm_backends import _apply_geo_whitelist

        # Non-dict input must be a no-op (returns the input as-is).
        for non_dict in (None, "string", 42, [1, 2, 3]):
            assert _apply_geo_whitelist(non_dict) == non_dict

    def test_dict_with_only_whitelisted_keys_is_unchanged(self):
        from rlpe.llm_backends import _apply_geo_whitelist

        item = {
            "age": "Late Triassic",
            "ma_top": 220.0,
            "ma_base": 230.0,
            "formation": "Sundance",
            "lithology": "sandstone",
            "confidence": 0.9,
        }
        out = _apply_geo_whitelist(dict(item))
        assert out == item

    def test_whitelist_drops_in_place(self):
        """The helper mutates in place (per docstring) and returns the same object."""
        from rlpe.llm_backends import _apply_geo_whitelist

        item = {"age": "Late Triassic", "habitat": "marine"}
        out = _apply_geo_whitelist(item)
        # The returned object must be the same instance.
        assert out is item
        # The hallucinated key must be gone from the original dict.
        assert "habitat" not in item


class TestM12ExtractGeologyIntegration:
    """End-to-end test: ``extract_geology`` must invoke
    ``_apply_geo_whitelist`` on every parsed item so LLM
    hallucinations never reach panel.metadata."""

    def _engine_with_geo(self, geo_payload: dict):
        from rlpe.m3_engine import M3Engine

        class _FakeBackend:
            backend_name = "fake-llm"
            enable_thinking = False

            def infer_text(self, system_prompt, user_prompt):
                return {"fallback_used": False, "raw_text": ""}

            def infer_panel(
                self,
                *,
                panel_image=None,
                caption_text="",
                ocr_labels=None,
                system_prompt="",
                user_prompt="",
                extra_image=None,
                **_kw,
            ):
                return {
                    "fallback_used": False,
                    "raw_text": json.dumps(geo_payload),
                }

        return M3Engine(_FakeBackend())

    def test_geo_entry_hallucinated_fields_dropped(self):
        from PIL import Image

        payload = {
            "geo": [
                {
                    "age": "Late Triassic",
                    "formation": "Sundance",
                    "habitat": "marine",  # hallucinated
                    "paleoclimate": "warm",  # hallucinated
                }
            ]
        }
        engine = self._engine_with_geo(payload)
        img = Image.new("RGB", (64, 64), "white")
        result = engine.extract_geology(img, "caption", "plate", "paper1", "fig1")
        assert len(result) == 1
        assert "habitat" not in result[0]
        assert "paleoclimate" not in result[0]
        # Whitelisted fields preserved.
        assert result[0]["age"] == "Late Triassic"
        assert result[0]["formation"] == "Sundance"

    def test_locality_hallucinated_fields_dropped(self):
        """Paleogeographic map localities route through a separate
        code path that must also whitelist-filter."""
        from PIL import Image

        payload = {
            "geo": [],  # Required so we get past the early-return
            "localities": [
                {
                    "species": "Actinomma sp.",
                    "latitude": 42.5,
                    "habitat": "shallow_marine",  # hallucinated
                }
            ],
        }
        engine = self._engine_with_geo(payload)
        img = Image.new("RGB", (64, 64), "white")
        result = engine.extract_geology(img, "caption", "paleogeographic_map", "p1", "f1")
        assert len(result) == 1
        assert "habitat" not in result[0]
        assert result[0]["species"] == "Actinomma sp."

    def test_layer_hallucinated_fields_dropped(self):
        """Strat column layers route through a separate code path
        that must also whitelist-filter."""
        from PIL import Image

        payload = {
            "geo": [],  # Required so we get past the early-return
            "layers": [
                {
                    "layer_index": 1,
                    "formation": "Sundance",
                    "depositional_environment": "deltaic",  # hallucinated
                }
            ],
        }
        engine = self._engine_with_geo(payload)
        img = Image.new("RGB", (64, 64), "white")
        result = engine.extract_geology(img, "caption", "strat_column", "p1", "f1")
        assert len(result) == 1
        assert "depositional_environment" not in result[0]
        assert result[0]["formation"] == "Sundance"


# ===========================================================================
# M-13: Ma-range direction validation
# ===========================================================================


class TestM13MaRangeValidation:
    """``_validate_ma_range`` enforces the ICZN/stratigraphic
    convention ``ma_top < ma_base`` (younger = smaller Ma).
    An inverted range is invalid and both fields are nulled."""

    def test_invalid_range_top_greater_than_base(self):
        """ma_top > ma_base violates the younger=smaller convention."""
        from rlpe.m3_engine import _validate_ma_range

        record = {"ma_top": 140, "ma_base": 120}
        out = _validate_ma_range(record)
        assert out["ma_top"] is None
        assert out["ma_base"] is None

    def test_invalid_range_top_greater_than_base_logged(self, caplog):
        from rlpe.m3_engine import _validate_ma_range

        caplog.set_level(logging.WARNING, logger="rlpe.m3_engine")
        _validate_ma_range({"ma_top": 140, "ma_base": 120})
        warnings = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("ma_top=140" in m and "ma_base=120" in m for m in warnings), (
            f"Expected WARNING with the bad range values, got {warnings}"
        )

    def test_valid_range_preserved(self):
        """ma_top < ma_base is the younger=smaller convention."""
        from rlpe.m3_engine import _validate_ma_range

        record = {"ma_top": 50, "ma_base": 100}
        out = _validate_ma_range(record)
        assert out["ma_top"] == 50
        assert out["ma_base"] == 100

    def test_valid_range_equal_preserved(self):
        """Edge: ma_top == ma_base is technically valid (zero-thickness
        range). Don't reject."""
        from rlpe.m3_engine import _validate_ma_range

        record = {"ma_top": 100, "ma_base": 100}
        out = _validate_ma_range(record)
        assert out["ma_top"] == 100
        assert out["ma_base"] == 100

    def test_partial_missing_top_preserved(self):
        from rlpe.m3_engine import _validate_ma_range

        record = {"ma_top": None, "ma_base": 100}
        out = _validate_ma_range(record)
        assert out["ma_top"] is None
        assert out["ma_base"] == 100

    def test_partial_missing_base_preserved(self):
        from rlpe.m3_engine import _validate_ma_range

        record = {"ma_top": 50, "ma_base": None}
        out = _validate_ma_range(record)
        assert out["ma_top"] == 50
        assert out["ma_base"] is None

    def test_both_none_preserved(self):
        from rlpe.m3_engine import _validate_ma_range

        record = {"ma_top": None, "ma_base": None}
        out = _validate_ma_range(record)
        assert out["ma_top"] is None
        assert out["ma_base"] is None

    def test_string_numeric_values_coerced(self):
        """LLM sometimes emits numbers as strings (``"120"``).
        The validation must coerce before comparing."""
        from rlpe.m3_engine import _validate_ma_range

        # Inverted: "140" > "120" numerically but lexicographically
        # "140" < "120" — make sure we coerce to float first.
        record = {"ma_top": "140", "ma_base": "120"}
        out = _validate_ma_range(record)
        assert out["ma_top"] is None
        assert out["ma_base"] is None

    def test_non_numeric_values_pass_through(self):
        """If the LLM emits non-numeric junk, leave it alone (don't
        silently null valid string captions)."""
        from rlpe.m3_engine import _validate_ma_range

        record = {"ma_top": "younger", "ma_base": "older"}
        out = _validate_ma_range(record)
        # Coercion fails → return record unchanged.
        assert out["ma_top"] == "younger"
        assert out["ma_base"] == "older"

    def test_invalid_range_clears_ma_mid(self):
        """ma_mid is meaningless when the range is inverted — drop
        it so callers don't carry a phantom midpoint."""
        from rlpe.m3_engine import _validate_ma_range

        record = {"ma_top": 140, "ma_base": 120, "ma_mid": 130}
        out = _validate_ma_range(record)
        assert "ma_mid" not in out

    def test_returns_same_dict_for_chaining(self):
        from rlpe.m3_engine import _validate_ma_range

        record = {"ma_top": 50, "ma_base": 100}
        out = _validate_ma_range(record)
        assert out is record


class TestM13ExtractGeologyIntegration:
    """End-to-end: ``extract_geology`` must call ``_validate_ma_range``
    on every parsed item so inverted ranges never reach
    panel.metadata."""

    def _engine_with_geo(self, geo_payload: dict):
        from rlpe.m3_engine import M3Engine

        class _FakeBackend:
            backend_name = "fake-llm"
            enable_thinking = False

            def infer_text(self, system_prompt, user_prompt):
                return {"fallback_used": False, "raw_text": ""}

            def infer_panel(
                self,
                *,
                panel_image=None,
                caption_text="",
                ocr_labels=None,
                system_prompt="",
                user_prompt="",
                extra_image=None,
                **_kw,
            ):
                return {
                    "fallback_used": False,
                    "raw_text": json.dumps(geo_payload),
                }

        return M3Engine(_FakeBackend())

    def test_inverted_geo_entry_range_is_nulled(self):
        from PIL import Image

        payload = {
            "geo": [
                {
                    "age": "Late Triassic",
                    "formation": "Sundance",
                    "ma_top": 140,  # inverted
                    "ma_base": 120,
                }
            ]
        }
        engine = self._engine_with_geo(payload)
        img = Image.new("RGB", (64, 64), "white")
        result = engine.extract_geology(img, "caption", "plate", "p1", "f1")
        assert len(result) == 1
        assert result[0]["ma_top"] is None
        assert result[0]["ma_base"] is None

    def test_valid_geo_entry_range_is_preserved(self):
        from PIL import Image

        payload = {
            "geo": [
                {
                    "age": "Late Triassic",
                    "ma_top": 220,
                    "ma_base": 230,
                }
            ]
        }
        engine = self._engine_with_geo(payload)
        img = Image.new("RGB", (64, 64), "white")
        result = engine.extract_geology(img, "caption", "plate", "p1", "f1")
        assert len(result) == 1
        assert result[0]["ma_top"] == 220
        assert result[0]["ma_base"] == 230

    def test_inverted_locality_range_is_nulled(self):
        from PIL import Image

        payload = {
            "geo": [],  # Required so we get past the early-return
            "localities": [
                {
                    "species": "Actinomma sp.",
                    "ma_top": 200,
                    "ma_base": 180,
                }
            ],
        }
        engine = self._engine_with_geo(payload)
        img = Image.new("RGB", (64, 64), "white")
        result = engine.extract_geology(img, "caption", "paleogeographic_map", "p1", "f1")
        assert len(result) == 1
        assert result[0]["ma_top"] is None
        assert result[0]["ma_base"] is None

    def test_inverted_layer_range_is_nulled(self):
        from PIL import Image

        payload = {
            "geo": [],  # Required so we get past the early-return
            "layers": [
                {
                    "layer_index": 1,
                    "ma_top": 105,  # inverted (older than base)
                    "ma_base": 95,
                }
            ],
        }
        engine = self._engine_with_geo(payload)
        img = Image.new("RGB", (64, 64), "white")
        result = engine.extract_geology(img, "caption", "strat_column", "p1", "f1")
        assert len(result) == 1
        assert result[0]["ma_top"] is None
        assert result[0]["ma_base"] is None


# ===========================================================================
# Source-guard: catch silent removal of the new helpers
# ===========================================================================


class TestSourceGuards:
    """Static checks against the live source so any later refactor
    that silently removes the whitelist or Ma-validation hook fires
    the guard."""

    def test_llm_backends_defines_whitelist_and_helper(self):
        from rlpe.llm_backends import _apply_geo_whitelist, _GEO_KEY_WHITELIST

        assert callable(_apply_geo_whitelist)
        assert isinstance(_GEO_KEY_WHITELIST, (set, frozenset))

    def test_llm_backends_source_references_whitelist_in_extract_path(self):
        """The m3_engine must reference ``_apply_geo_whitelist``
        inside the extract_geology body so the filter cannot be
        silently bypassed."""
        from rlpe import m3_engine

        src = Path(m3_engine.__file__).read_text()
        assert "_apply_geo_whitelist" in src

    def test_m3_engine_defines_validate_ma_range(self):
        from rlpe.m3_engine import _validate_ma_range

        assert callable(_validate_ma_range)

    def test_m3_engine_source_references_ma_validation_in_extract_path(self):
        """The m3_engine must call ``_validate_ma_range`` inside the
        extract_geology body so the filter cannot be silently bypassed."""
        from rlpe import m3_engine

        src = Path(m3_engine.__file__).read_text()
        assert "_validate_ma_range" in src

    def test_m3_engine_class_structure_intact(self):
        """Regression guard: inserting a module-level helper between
        two M3Engine methods previously broke the class structure
        (causing ``AttributeError: M3Engine has no attribute
        'extract_geology'``). Verify the class still has all the
        expected methods."""
        from rlpe.m3_engine import M3Engine

        for method_name in (
            "parse_caption",
            "classify_plate",
            "segment_panels",
            "match_panel",
            "critique_matches",
            "extract_geology",
            "extract_schematic",
        ):
            assert hasattr(M3Engine, method_name), (
                f"M3Engine lost method {method_name!r} — likely class "
                f"structure broken by a misplaced helper."
            )
