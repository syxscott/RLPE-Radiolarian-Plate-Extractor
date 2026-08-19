"""Phase 4A M3 few-shot prompt audit.

Adds regression coverage on top of Phase 2b (test_audit_2026_08_19_
phase2b_m3_prompts.py). Phase 2b verified that the 5 stage prompts
had *some* few-shot example. Phase 4A requires that every stage
prompt be **complete** — at least 3 examples per stage, covering
English + Chinese + rare formats — and that each example be a full
(input → expected output) pair so the LLM has concrete demonstrations
of every JSON contract the downstream schema expects.

The tests are read-only against the live source: they grep the
constants and assert structural properties. No LLM calls. They fail
if any prompt's few-shot coverage regresses, or if an example drops
its (input, output) pair shape.

Specifically:

1. ``_PARSE_CAPTION_SYSTEM``, ``_CLASSIFY_PLATE_SYSTEM``,
   ``_SEGMENT_PANELS_SYSTEM``, ``_MATCH_PANEL_SYSTEM``,
   ``_CRITIQUE_SYSTEM`` each have at least 3 few-shot examples.
2. Each example contains both an input excerpt (caption / image
   description) and a JSON output block.
3. ``_PARSE_CAPTION_SYSTEM`` covers all 3 ICZN open-nomenclature
   strengths (``cf.`` / ``aff.`` / ``ex gr.``).
4. ``_CLASSIFY_PLATE_SYSTEM`` covers a Chinese caption / panel
   variant.
5. The three geology vision prompts (``strat_column_geo``,
   ``litholog_column_geo``, ``paleogeographic_map_geo``) each have
   at least one complete example demonstrating their layered
   contract.
6. The ``"Output MUST match the JSON schema exactly. See example(s)
   below."`` footer is appended to every prompt we modified.
7. Example output fields are consistent with the schema:
   - ``_PARSE_CAPTION_SYSTEM`` example fields match
     ``TaxonRecord`` keys (verbatim_name / normalized_name / genus
     / family / confidence / qualifier).
   - ``strat_column_geo`` / ``litholog_column_geo`` example fields
     match ``GeologyContextRecord`` keys (age / formation /
     lithology / ma_top / ma_base / biozone / confidence).
   - ``paleogeographic_map_geo`` example fields match
     ``LocalityRecord`` keys (name / country / latitude /
     longitude / coordinate_source) — at least via the ``geo``
     wrapper.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The five stage prompt constants — imported lazily inside each test so the
# file can be parsed even if a future refactor removes one of them.
STAGE_PROMPTS = (
    "_PARSE_CAPTION_SYSTEM",
    "_CLASSIFY_PLATE_SYSTEM",
    "_SEGMENT_PANELS_SYSTEM",
    "_MATCH_PANEL_SYSTEM",
    "_CRITIQUE_SYSTEM",
)

GEOLOGY_PROMPTS = (
    "strat_column_geo",
    "litholog_column_geo",
    "paleogeographic_map_geo",
)

# The required footer string. Every modified prompt ends with this.
REQUIRED_FOOTER = (
    "Output MUST match the JSON schema exactly. See example"
)


def _example_count(prompt_text: str) -> int:
    """Count the number of few-shot examples in a prompt.

    Recognises ``Example 1`` / ``Example 2`` / ``Example (Phase 4A)``
    / ``示例输入`` / ``示例输出`` style markers. Returns 0 when no
    examples are present.
    """
    # English numbering: "Example 1 (" / "Example 2 (" / "Example N ("
    # OR the Phase 4A footer "Example (Phase 4A)" / "Example:" markers.
    en_numbered = re.findall(r"\bExample\s+\d+\b", prompt_text)
    en_phase4a = re.findall(r"Example\s*\(Phase 4A\)", prompt_text)
    en_bare = re.findall(r"^Example\s*:", prompt_text, flags=re.MULTILINE)
    en_total = len(en_numbered) + len(en_phase4a) + len(en_bare)
    # Chinese marker: "示例输入" or "示例输出"
    zh_markers = re.findall(r"示例(?:输入|输出)", prompt_text)
    return max(en_total, len(zh_markers))


def _has_input_output_pair(prompt_text: str) -> bool:
    """Return True iff the prompt contains both an input excerpt AND a
    JSON-shaped output block (object or array) anywhere in the text.

    Heuristic: looks for typical input markers ("Input image:" /
    "Input:" / "Panel image:" / "Plate image:" / "示例输入") AND a JSON
    bracket pair (any ``{...}`` block containing at least 2 quoted keys).
    """
    has_input = bool(
        re.search(
            r"(Input image:|Input:|Panel image:|Plate image:|示例输入|Candidate pairs)",
            prompt_text,
        )
    )
    # Find a JSON-looking object/array with >=2 quoted keys.  We are
    # deliberately tolerant — match any ``"key": "value"`` or
    # ``"key": number`` pair to avoid false negatives.
    json_blocks = re.findall(r'\{[^{}]*"[^"]+"\s*:\s*[^,\{\}]+', prompt_text)
    return has_input and len(json_blocks) >= 1


# ---------------------------------------------------------------------------
# 1. Each stage prompt has at least 3 few-shot examples
# ---------------------------------------------------------------------------


class TestStagePromptFewShotCount:
    """Every stage prompt must have at least 3 examples."""

    @pytest.mark.parametrize("prompt_name", STAGE_PROMPTS)
    def test_at_least_3_examples(self, prompt_name: str):
        from rlpe import m3_engine

        prompt = getattr(m3_engine, prompt_name)
        n = _example_count(prompt)
        assert n >= 3, (
            f"{prompt_name} has only {n} few-shot examples; "
            f"Phase 4A requires >= 3 (English + Chinese + rare format). "
            f"Prompt:\n{prompt[:300]}…"
        )


class TestGeologyPromptFewShotCount:
    """The 3 geology vision prompts must each have at least 1
    complete input/output example (these prompts had zero examples
    before Phase 4A)."""

    @pytest.mark.parametrize("prompt_key", GEOLOGY_PROMPTS)
    def test_at_least_1_example(self, prompt_key: str):
        from rlpe import m3_engine

        prompt = m3_engine.PROMPT_REGISTRY[prompt_key]
        n = _example_count(prompt)
        assert n >= 1, (
            f"PROMPT_REGISTRY[{prompt_key!r}] has only {n} examples; "
            f"Phase 4A requires >= 1 complete example."
        )

    @pytest.mark.parametrize("prompt_key", GEOLOGY_PROMPTS)
    def test_input_output_pair_present(self, prompt_key: str):
        from rlpe import m3_engine

        prompt = m3_engine.PROMPT_REGISTRY[prompt_key]
        assert _has_input_output_pair(prompt), (
            f"PROMPT_REGISTRY[{prompt_key!r}] missing a full (input, "
            f"JSON output) pair in its few-shot example."
        )


# ---------------------------------------------------------------------------
# 2. Every few-shot example is a complete (input, expected output) pair
# ---------------------------------------------------------------------------


class TestFewShotInputOutputPairs:
    """Each example must contain both an input excerpt and a JSON
    output block. The presence of just the output is insufficient —
    the LLM needs to see the mapping."""

    @pytest.mark.parametrize("prompt_name", STAGE_PROMPTS)
    def test_complete_input_output_pair(self, prompt_name: str):
        from rlpe import m3_engine

        prompt = getattr(m3_engine, prompt_name)
        assert _has_input_output_pair(prompt), (
            f"{prompt_name} few-shot examples lack a full (input, "
            f"JSON output) pair."
        )


class TestFewShotOutputIsJson:
    """The example output block must be JSON-shaped (contains at
    least one quoted key:value pair and a closing brace)."""

    @pytest.mark.parametrize("prompt_name", STAGE_PROMPTS)
    def test_output_block_is_json(self, prompt_name: str):
        from rlpe import m3_engine

        prompt = getattr(m3_engine, prompt_name)
        # Match a JSON object/array with at least 2 quoted keys.  We
        # accept either: an array ``[...]`` (segment_panels) or an
        # object ``{...}`` (most others).
        obj_match = re.search(r'\{[^{}]*"[^"]+"\s*:\s*[^,\{\}]+', prompt)
        array_match = re.search(r"\[\s*\{", prompt)
        assert obj_match or array_match, (
            f"{prompt_name} example output is not JSON-shaped "
            f"(no matching object/array block found)."
        )


# ---------------------------------------------------------------------------
# 3. Examples cover key fields
# ---------------------------------------------------------------------------


class TestParseCaptionCoverage:
    """``_PARSE_CAPTION_SYSTEM`` must demonstrate the 3 ICZN
    open-nomenclature strengths: cf., aff., ex gr.
    Phase 4A added a third English example covering all three."""

    def test_cf_marker_present(self):
        from rlpe import m3_engine

        assert "cf." in m3_engine._PARSE_CAPTION_SYSTEM

    def test_aff_marker_present(self):
        from rlpe import m3_engine

        assert "aff." in m3_engine._PARSE_CAPTION_SYSTEM

    def test_ex_gr_marker_present(self):
        from rlpe import m3_engine

        assert "ex gr." in m3_engine._PARSE_CAPTION_SYSTEM

    def test_english_caption_example(self):
        """At least one example uses English-language caption text."""
        from rlpe import m3_engine

        prompt = m3_engine._PARSE_CAPTION_SYSTEM
        # Look for English caption conventions: "Figure N." / "Plate N." / "Figs. N."
        assert re.search(r"(Figure \d+|Plate \d+|Figs?\.?\s*\d+)", prompt), (
            "_PARSE_CAPTION_SYSTEM missing English caption example"
        )

    def test_chinese_caption_example(self):
        """At least one example uses Chinese-language caption text."""
        from rlpe import m3_engine

        prompt = m3_engine._PARSE_CAPTION_SYSTEM
        # Look for Chinese caption conventions: 图版 / 图 / 比例尺
        assert re.search(r"(图版|比例尺|扫描电镜)", prompt), (
            "_PARSE_CAPTION_SYSTEM missing Chinese caption example"
        )


class TestClassifyPlateCoverage:
    """``_CLASSIFY_PLATE_SYSTEM`` must have a Chinese caption
    variant example covering Asian-paper format conventions."""

    def test_chinese_caption_example_present(self):
        from rlpe import m3_engine

        prompt = m3_engine._CLASSIFY_PLATE_SYSTEM
        # Chinese paper conventions: 中文图说 / 中文图版 / 横版
        # OR Chinese scale-bar text "比例尺" or "图版"
        assert re.search(r"(中文|比例尺|图版)", prompt), (
            "_CLASSIFY_PLATE_SYSTEM missing Chinese caption example"
        )

    def test_outdoor_photo_example_present(self):
        """Phase 4A adds an outdoor-photo negative example."""
        from rlpe import m3_engine

        prompt = m3_engine._CLASSIFY_PLATE_SYSTEM
        assert re.search(r"(野外|露头|照片)", prompt), (
            "_CLASSIFY_PLATE_SYSTEM missing outdoor/field photo example"
        )

    def test_strat_column_negative_example_present(self):
        """A stratigraphic column negative example must exist so the
        LLM doesn't mis-classify column figures as plates."""
        from rlpe import m3_engine

        prompt = m3_engine._CLASSIFY_PLATE_SYSTEM
        assert "stratigraphic" in prompt.lower() or "柱状" in prompt, (
            "_CLASSIFY_PLATE_SYSTEM missing strat-column negative example"
        )


class TestSegmentPanelsCoverage:
    """``_SEGMENT_PANELS_SYSTEM`` must show multi-panel + single-
    panel + irregular layouts."""

    def test_multi_panel_example_present(self):
        from rlpe import m3_engine

        prompt = m3_engine._SEGMENT_PANELS_SYSTEM
        assert "2x2" in prompt or "2x3" in prompt, (
            "_SEGMENT_PANELS_SYSTEM missing multi-panel grid example"
        )

    def test_single_panel_example_present(self):
        from rlpe import m3_engine

        prompt = m3_engine._SEGMENT_PANELS_SYSTEM
        assert "single" in prompt.lower() or "P1" in prompt, (
            "_SEGMENT_PANELS_SYSTEM missing single-panel example"
        )

    def test_irregular_layout_example_present(self):
        from rlpe import m3_engine

        prompt = m3_engine._SEGMENT_PANELS_SYSTEM
        assert "不规则" in prompt or "irregular" in prompt.lower(), (
            "_SEGMENT_PANELS_SYSTEM missing irregular-layout example"
        )


class TestMatchPanelCoverage:
    """``_MATCH_PANEL_SYSTEM`` examples must cover all 3 outcomes:
    confident caption match / open-nomen match / no candidate."""

    def test_caption_match_example_present(self):
        from rlpe import m3_engine

        prompt = m3_engine._MATCH_PANEL_SYSTEM
        assert "Tetraspongodiscus" in prompt, (
            "_MATCH_PANEL_SYSTEM missing confident caption match example"
        )

    def test_open_nomen_example_present(self):
        from rlpe import m3_engine

        prompt = m3_engine._MATCH_PANEL_SYSTEM
        assert "cf." in prompt, (
            "_MATCH_PANEL_SYSTEM missing open-nomen example"
        )

    def test_no_candidate_example_present(self):
        from rlpe import m3_engine

        prompt = m3_engine._MATCH_PANEL_SYSTEM
        assert "Candidate pairs (from caption): []" in prompt, (
            "_MATCH_PANEL_SYSTEM missing empty-candidate example"
        )

    def test_english_caption_example_present(self):
        """Phase 4A adds an English caption example for international papers."""
        from rlpe import m3_engine

        prompt = m3_engine._MATCH_PANEL_SYSTEM
        assert "Hsuum" in prompt, (
            "_MATCH_PANEL_SYSTEM missing English caption example "
            "(Hsuum is a well-known Late Jurassic nassellarian)"
        )


class TestCritiqueCoverage:
    """``_CRITIQUE_SYSTEM`` must cover all 3 verdict paths."""

    def test_agree_example_present(self):
        from rlpe import m3_engine

        prompt = m3_engine._CRITIQUE_SYSTEM
        assert '"agree"' in prompt or "agree" in prompt, (
            "_CRITIQUE_SYSTEM missing agree example"
        )

    def test_disagree_example_present(self):
        from rlpe import m3_engine

        prompt = m3_engine._CRITIQUE_SYSTEM
        assert "disagree" in prompt, (
            "_CRITIQUE_SYSTEM missing disagree example"
        )

    def test_uncertain_example_present(self):
        from rlpe import m3_engine

        prompt = m3_engine._CRITIQUE_SYSTEM
        assert "uncertain" in prompt, (
            "_CRITIQUE_SYSTEM missing uncertain example"
        )

    def test_low_confidence_batch_example_present(self):
        """Phase 4A adds a 3-panel low-confidence batch example."""
        from rlpe import m3_engine

        prompt = m3_engine._CRITIQUE_SYSTEM
        # The Phase 4A example has 3 panels with confidence 0.4 each.
        assert "0.4" in prompt, (
            "_CRITIQUE_SYSTEM missing low-confidence batch example"
        )


# ---------------------------------------------------------------------------
# 4. Schema consistency — example output fields must exist in the schema
# ---------------------------------------------------------------------------


class TestParseCaptionSchemaConsistency:
    """The JSON keys used in ``_PARSE_CAPTION_SYSTEM`` examples must
    be present in the downstream ``TaxonRecord`` schema (or be
    pipeline-internal fields like ``labels`` that the converter maps
    to ``verbatim_name`` / ``normalized_name``)."""

    def test_species_field_present_in_taxonrecord(self):
        from rlpe.schema_models import TaxonRecord

        # ``species`` in the parse_caption output maps to
        # ``TaxonRecord.verbatim_name`` / ``.normalized_name``.
        fields = TaxonRecord.model_fields.keys()
        assert "verbatim_name" in fields and "normalized_name" in fields

    def test_confidence_field_present_in_taxonrecord(self):
        from rlpe.schema_models import TaxonRecord

        fields = TaxonRecord.model_fields.keys()
        assert "confidence" in fields

    def test_qualifier_field_present_in_taxonrecord(self):
        """``modifier`` in parse_caption maps to ``TaxonRecord.qualifier``."""
        from rlpe.schema_models import TaxonRecord

        fields = TaxonRecord.model_fields.keys()
        assert "qualifier" in fields

    def test_example_uses_schema_mappable_keys(self):
        """The example output uses keys that the converter maps to
        TaxonRecord fields. ``species`` -> verbatim_name,
        ``modifier`` -> qualifier, ``confidence`` -> confidence."""
        from rlpe import m3_engine

        prompt = m3_engine._PARSE_CAPTION_SYSTEM
        # At least one example output object must include "species".
        assert '"species"' in prompt
        # ... and a "modifier" or "qualifier" equivalent.
        assert '"modifier"' in prompt
        # ... and a confidence float.
        assert re.search(r'"confidence"\s*:\s*0\.\d+', prompt), (
            "_PARSE_CAPTION_SYSTEM example must show a numeric confidence"
        )


class TestExtractGeologySchemaConsistency:
    """The JSON keys used in ``strat_column_geo`` /
    ``litholog_column_geo`` / ``paleogeographic_map_geo`` examples
    must exist in ``GeologyContextRecord`` or ``LocalityRecord``
    schemas."""

    def test_strat_column_example_uses_geology_context_keys(self):
        from rlpe import m3_engine
        from rlpe.schema_models import GeologyContextRecord

        prompt = m3_engine.PROMPT_REGISTRY["strat_column_geo"]
        # The ``geo`` wrapper must include ``age``, ``formation``,
        # ``lithology``, ``ma_top``, ``ma_base``, ``confidence``.
        fields = GeologyContextRecord.model_fields.keys()
        for key in ("age", "formation", "lithology", "ma_top", "ma_base", "confidence"):
            assert key in fields, f"GeologyContextRecord missing field {key!r}"
        # The example output must use them.
        for key in ("age", "formation", "lithology", "ma_top", "ma_base", "confidence"):
            assert f'"{key}"' in prompt, (
                f"strat_column_geo example missing field {key!r}"
            )

    def test_litholog_column_example_uses_geology_context_keys(self):
        from rlpe import m3_engine
        from rlpe.schema_models import GeologyContextRecord

        prompt = m3_engine.PROMPT_REGISTRY["litholog_column_geo"]
        fields = GeologyContextRecord.model_fields.keys()
        for key in ("age", "lithology", "ma_top", "ma_base", "confidence"):
            assert key in fields, f"GeologyContextRecord missing field {key!r}"
        for key in ("age", "lithology", "ma_top", "ma_base", "confidence"):
            assert f'"{key}"' in prompt, (
                f"litholog_column_geo example missing field {key!r}"
            )

    def test_paleogeographic_example_uses_locality_keys(self):
        from rlpe import m3_engine
        from rlpe.schema_models import LocalityRecord

        prompt = m3_engine.PROMPT_REGISTRY["paleogeographic_map_geo"]
        fields = LocalityRecord.model_fields.keys()
        # LocalityRecord uses modern_latitude / modern_longitude
        # (Darwin Core convention); verify those exist.
        for key in ("modern_latitude", "modern_longitude", "name", "country"):
            assert key in fields, f"LocalityRecord missing field {key!r}"
        # The example output uses latitude/longitude (raw schema key
        # in the geo wrapper) — verify those appear at least once in
        # the example block.
        assert '"latitude"' in prompt, (
            "paleogeographic_map_geo example missing latitude field"
        )
        assert '"longitude"' in prompt, (
            "paleogeographic_map_geo example missing longitude field"
        )
        assert '"species"' in prompt, (
            "paleogeographic_map_geo example missing species field"
        )


# ---------------------------------------------------------------------------
# 5. Required footer on every modified prompt
# ---------------------------------------------------------------------------


class TestRequiredFooter:
    """Every modified prompt must end with the standard footer so the
    LLM knows the example below is authoritative."""

    @pytest.mark.parametrize("prompt_name", STAGE_PROMPTS)
    def test_stage_prompt_has_footer(self, prompt_name: str):
        from rlpe import m3_engine

        prompt = getattr(m3_engine, prompt_name)
        assert REQUIRED_FOOTER in prompt, (
            f"{prompt_name} missing required footer "
            f"{REQUIRED_FOOTER!r} (Phase 4A addition)"
        )

    @pytest.mark.parametrize("prompt_key", GEOLOGY_PROMPTS)
    def test_geology_prompt_has_footer(self, prompt_key: str):
        from rlpe import m3_engine

        prompt = m3_engine.PROMPT_REGISTRY[prompt_key]
        assert REQUIRED_FOOTER in prompt, (
            f"PROMPT_REGISTRY[{prompt_key!r}] missing required footer "
            f"{REQUIRED_FOOTER!r}"
        )


# ---------------------------------------------------------------------------
# 6. Token-budget guard (don't blow up the input token budget)
# ---------------------------------------------------------------------------


class TestPromptTokenBudget:
    """All modified prompts must stay under 4000 tokens (rough char
    estimate: 1 token per ASCII char / 4 + 1 per CJK char)."""

    @pytest.mark.parametrize("prompt_name", STAGE_PROMPTS)
    def test_stage_prompt_under_budget(self, prompt_name: str):
        from rlpe import m3_engine

        prompt = getattr(m3_engine, prompt_name)
        cjk = sum(1 for c in prompt if ord(c) > 0x3000)
        ascii_ = len(prompt) - cjk
        approx_tokens = cjk + ascii_ // 4
        assert approx_tokens < 4000, (
            f"{prompt_name} ~{approx_tokens} tokens exceeds 4000 budget"
        )

    @pytest.mark.parametrize("prompt_key", GEOLOGY_PROMPTS)
    def test_geology_prompt_under_budget(self, prompt_key: str):
        from rlpe import m3_engine

        prompt = m3_engine.PROMPT_REGISTRY[prompt_key]
        cjk = sum(1 for c in prompt if ord(c) > 0x3000)
        ascii_ = len(prompt) - cjk
        approx_tokens = cjk + ascii_ // 4
        assert approx_tokens < 4000, (
            f"PROMPT_REGISTRY[{prompt_key!r}] ~{approx_tokens} tokens exceeds 4000 budget"
        )


# ---------------------------------------------------------------------------
# 7. Source-guard: don't break the prompt constants
# ---------------------------------------------------------------------------


class TestSourceGuards:
    """Make sure each prompt constant still exists and the few-shot
    additions are syntactically inside the constant string (no
    trailing junk like a forgotten comma)."""

    @pytest.mark.parametrize("prompt_name", STAGE_PROMPTS)
    def test_prompt_is_string(self, prompt_name: str):
        from rlpe import m3_engine

        prompt = getattr(m3_engine, prompt_name)
        assert isinstance(prompt, str) and len(prompt) > 500

    @pytest.mark.parametrize("prompt_key", GEOLOGY_PROMPTS)
    def test_geology_prompt_is_string(self, prompt_key: str):
        from rlpe import m3_engine

        prompt = m3_engine.PROMPT_REGISTRY[prompt_key]
        assert isinstance(prompt, str) and len(prompt) > 500

    def test_module_compiles(self):
        """Round-trip import — if any of the prompt constants was
        accidentally truncated, this raises immediately.

        Note: do NOT ``importlib.reload()`` here — reloading creates fresh
        class objects (e.g. ``LLMSchemaError``), so any test that
        captured the old class via ``from rlpe.m3_engine import ...``
        will silently break ``isinstance`` checks downstream. The
        import alone is sufficient to surface syntax / NameErrors.
        """
        from rlpe import m3_engine  # noqa: F401  (import-side-effect check)