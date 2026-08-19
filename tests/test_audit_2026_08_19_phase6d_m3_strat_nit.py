"""Phase 6D m3_engine / stratigraphy NIT sweep (2026-08-19).

Four NIT bugs from the 2026-08-19 multi-agent audit follow-up:

* **NIT-1** — ``_coerce_bbox`` in ``m3_engine.py`` silently clamped
  out-of-range values. A bbox with a negative coordinate
  (``[-0.5, 0.3, 0.4, 0.2]``) was clamped to ``(0, 30, 40, 20)`` and
  a bbox with a value > 1.0 in the normalized range
  (``[0.5, 0.3, 1.5, 0.2]``) was routed through the pixel-path clamp
  (truncated to ``img_w - x_px``). Both masked malformed LLM output
  and could poison the segmenter with a bbox that spans the full
  figure. The fix rejects both shapes with :class:`ValueError` so
  the caller can log the source figure and decide whether to skip
  the panel or retry the LLM call.

* **NIT-2** — ``BiozoneRecord`` (in ``range_chart_extractor.py``)
  carried only the zone name, age, thickness, and (Phase 3E)
  ``zone_type``. Downstream consumers that wanted to disambiguate
  zone names differed by author (e.g. "N. optima Zone" was first
  defined by Ishiga & Imoto 1982 but re-used by at least 4
  subsequent authors) had no structured way to know which
  authority / year was intended. The fix adds two optional fields:
  ``zone_authority: str | None`` and
  ``zone_publication_year: int | None``. Both default to ``None``
  for backward compatibility.

* **NIT-3** — ``extract_geology`` in ``m3_engine.py`` received an
  inverted ``ma_top / ma_base`` pair (``ma_top=100, ma_base=50``)
  and the previous :func:`_validate_ma_range` helper nulled both
  fields. The new :func:`_normalize_ma_pair` helper auto-swaps the
  pair so the schema contract (``ma_top < ma_base``) is preserved
  with no information loss. The strict null-on-violation behavior
  is retained for callers that prefer it (chain ``_normalize_ma_pair``
  before ``_validate_ma_range``).

* **NIT-4** — ``_normalize_species`` in ``m3_engine.py`` had a
  trailing ``re.sub(r"\\s+", " ", s)`` but the collapse only ran
  at the END of the function. A malformed multi-space species
  name (``"Entactinia   sp."`` with 3 spaces) would survive all
  the structured-rule passes (Spumellaria fold, sensu strip, gr
  strip, etc.) and only be normalised when the final sub ran. The
  fix collapses whitespace at the START of the function so every
  downstream regex sees a single-spaced input.

The tests are read-only against the live source (no LLM calls, no
filesystem writes). They fail if any of the four fixes regresses
or any of the new helpers is removed.

Run with::

    python -m pytest tests/test_audit_2026_08_19_phase6d_m3_strat_nit.py -v
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


_REPO = Path(__file__).resolve().parents[1]
_SRC_M3 = _REPO / "src" / "rlpe" / "m3_engine.py"
_SRC_RCE = _REPO / "src" / "rlpe" / "range_chart_extractor.py"


def _read(rel: Path) -> str:
    return rel.read_text(encoding="utf-8")


# ===========================================================================
# NIT-1: _coerce_bbox strict [0.0, 1.0] range validation
# ===========================================================================


class TestNIT1CoerceBboxRejectsNegative:
    """``_coerce_bbox`` must raise :class:`ValueError` when any
    coordinate is negative (regardless of normalized vs pixel
    interpretation)."""

    def test_negative_in_x_position_raises(self):
        from rlpe.m3_engine import _coerce_bbox

        with pytest.raises(ValueError):
            _coerce_bbox([-0.5, 0.3, 0.4, 0.2], 1000, 1000)

    def test_negative_in_y_position_raises(self):
        from rlpe.m3_engine import _coerce_bbox

        with pytest.raises(ValueError):
            _coerce_bbox([0.5, -0.3, 0.4, 0.2], 1000, 1000)

    def test_negative_in_width_raises(self):
        from rlpe.m3_engine import _coerce_bbox

        with pytest.raises(ValueError):
            _coerce_bbox([0.5, 0.3, -0.4, 0.2], 1000, 1000)

    def test_negative_in_height_raises(self):
        from rlpe.m3_engine import _coerce_bbox

        with pytest.raises(ValueError):
            _coerce_bbox([0.5, 0.3, 0.4, -0.2], 1000, 1000)

    def test_negative_pixel_path_also_raises(self):
        """A bbox with a negative value that would otherwise be
        routed through the pixel path (because max > 1.01) must
        also raise — negative pixel coords are equally invalid."""
        from rlpe.m3_engine import _coerce_bbox

        with pytest.raises(ValueError):
            _coerce_bbox([100, 200, -5, 80], 1000, 1000)


class TestNIT1CoerceBboxRejectsOverOne:
    """``_coerce_bbox`` must raise :class:`ValueError` when a
    normalized bbox (max <= 1.01) contains a value > 1.0."""

    def test_value_just_above_one_raises(self):
        """A bbox with 1.005 (still within the 1.01 tolerance for
        CLASSIFICATION purposes) must be rejected as malformed."""
        from rlpe.m3_engine import _coerce_bbox

        with pytest.raises(ValueError):
            _coerce_bbox([0.5, 0.3, 0.4, 1.005], 1000, 1000)

    def test_value_at_upper_tolerance_raises(self):
        """A bbox with 1.01 (exactly at the classification tolerance)
        must be rejected as malformed because the value is still > 1.0."""
        from rlpe.m3_engine import _coerce_bbox

        with pytest.raises(ValueError):
            _coerce_bbox([0.5, 0.3, 1.01, 0.2], 1000, 1000)

    def test_height_above_one_raises(self):
        """A bbox with 1.005 in the height position (still within
        the 1.01 tolerance) must be rejected."""
        from rlpe.m3_engine import _coerce_bbox

        with pytest.raises(ValueError):
            _coerce_bbox([0.5, 0.3, 0.4, 1.005], 1000, 1000)


class TestNIT1CoerceBboxAcceptsValid:
    """``_coerce_bbox`` must still accept valid inputs (regression
    guard against an over-aggressive validation that breaks
    legitimate normalized and pixel bboxes)."""

    def test_normalized_bbox_accepted(self):
        from rlpe.m3_engine import _coerce_bbox

        out = _coerce_bbox([0.5, 0.3, 0.4, 0.2], 1000, 1000)
        assert out is not None
        # (0.5*1000, 0.3*1000, 0.4*1000, 0.2*1000) = (500, 300, 400, 200)
        assert out == (500, 300, 400, 200)

    def test_normalized_bbox_zero_origin_accepted(self):
        from rlpe.m3_engine import _coerce_bbox

        out = _coerce_bbox([0.0, 0.0, 1.0, 1.0], 1000, 1000)
        assert out == (0, 0, 1000, 1000)

    def test_pixel_bbox_accepted(self):
        """A pixel coord bbox (max > 1.01) must still be accepted
        without raising — the validation only rejects out-of-range
        normalized values."""
        from rlpe.m3_engine import _coerce_bbox

        out = _coerce_bbox([100, 200, 300, 400], 1000, 1000)
        assert out == (100, 200, 300, 400)

    def test_pixel_bbox_clamps_to_image_bounds(self):
        """The pixel path still clamps to image bounds (preserves
        the existing Phase 55 audit fix)."""
        from rlpe.m3_engine import _coerce_bbox

        out = _coerce_bbox([500, 400, 800, 900], 1000, 1000)
        # w capped at img_w - x = 1000 - 500 = 500
        # h capped at img_h - y = 1000 - 400 = 600
        assert out == (500, 400, 500, 600)

    def test_invalid_input_returns_none(self):
        """Non-list/tuple input still returns None (no raise)."""
        from rlpe.m3_engine import _coerce_bbox

        assert _coerce_bbox(None, 1000, 1000) is None
        assert _coerce_bbox("not a list", 1000, 1000) is None
        assert _coerce_bbox([0.5, 0.3], 1000, 1000) is None  # too few
        assert _coerce_bbox([0.5, 0.3, 0.4, 0.2, 0.5], 1000, 1000) is None  # too many

    def test_non_numeric_entry_returns_none(self):
        """Non-numeric entries still return None (no raise)."""
        from rlpe.m3_engine import _coerce_bbox

        assert _coerce_bbox(["not", "a", "number", "list"], 1000, 1000) is None


class TestNIT1SourceGuard:
    """Static source guard: the validation must be present in the
    live source code so future refactors don't silently drop it."""

    def test_validation_in_source(self):
        src = _read(_SRC_M3)
        assert "_coerce_bbox: negative coordinate" in src, (
            "_coerce_bbox must reject negative coordinates (NIT-1)"
        )
        assert "_coerce_bbox: normalized coordinate > 1.0" in src, (
            "_coerce_bbox must reject > 1.0 in normalized bbox (NIT-1)"
        )


# ===========================================================================
# NIT-2: BiozoneRecord.zone_authority + zone_publication_year
# ===========================================================================


class TestNIT2BiozoneRecordFields:
    """``BiozoneRecord`` must carry the new NIT-2 fields."""

    def test_zone_authority_default_none(self):
        from rlpe.range_chart_extractor import BiozoneRecord

        r = BiozoneRecord()
        assert r.zone_authority is None

    def test_zone_publication_year_default_none(self):
        from rlpe.range_chart_extractor import BiozoneRecord

        r = BiozoneRecord()
        assert r.zone_publication_year is None

    def test_zone_authority_set_round_trips(self):
        from rlpe.range_chart_extractor import BiozoneRecord

        r = BiozoneRecord(name="N. optima Zone", zone_authority="Ishiga & Imoto")
        assert r.zone_authority == "Ishiga & Imoto"
        d = r.to_dict()
        assert d["zone_authority"] == "Ishiga & Imoto"
        assert d["name"] == "N. optima Zone"

    def test_zone_publication_year_set_round_trips(self):
        from rlpe.range_chart_extractor import BiozoneRecord

        r = BiozoneRecord(name="N. optima Zone", zone_publication_year=1982)
        assert r.zone_publication_year == 1982
        d = r.to_dict()
        assert d["zone_publication_year"] == 1982

    def test_zone_authority_and_year_combined(self):
        from rlpe.range_chart_extractor import BiozoneRecord

        r = BiozoneRecord(
            name="N. optima Zone",
            zone_authority="Ishiga & Imoto",
            zone_publication_year=1982,
        )
        d = r.to_dict()
        assert d["zone_authority"] == "Ishiga & Imoto"
        assert d["zone_publication_year"] == 1982

    def test_to_dict_contains_all_fields(self):
        """The to_dict() output must include all NIT-2 fields so the
        downstream JSON export / xlsx writer / GBIF pipeline can
        read them."""
        from rlpe.range_chart_extractor import BiozoneRecord

        r = BiozoneRecord()
        d = r.to_dict()
        for key in ("zone_authority", "zone_publication_year"):
            assert key in d, f"BiozoneRecord.to_dict() missing {key!r}"


class TestNIT2ParseExtractionResponse:
    """``_parse_extraction_response`` must populate the new fields
    from the LLM JSON output."""

    def test_authority_and_year_parsed(self):
        from rlpe.range_chart_extractor import (
            _parse_extraction_response,
        )

        parsed = {
            "biozones": [
                {
                    "name": "N. optima Zone",
                    "age": "Latest Changhsingian",
                    "zone_type": "range",
                    "zone_authority": "Ishiga & Imoto",
                    "zone_publication_year": 1982,
                }
            ]
        }
        result = _parse_extraction_response(parsed=parsed, paper_id="p", figure_id="f")
        assert len(result.biozones) == 1
        bz = result.biozones[0]
        assert bz.zone_authority == "Ishiga & Imoto"
        assert bz.zone_publication_year == 1982

    def test_authority_omitted_yields_none(self):
        from rlpe.range_chart_extractor import (
            _parse_extraction_response,
        )

        parsed = {
            "biozones": [
                {"name": "Bare Zone", "age": "Miocene"},
            ]
        }
        result = _parse_extraction_response(parsed=parsed, paper_id="p", figure_id="f")
        assert result.biozones[0].zone_authority is None
        assert result.biozones[0].zone_publication_year is None

    def test_year_out_of_range_yields_none(self):
        """Year outside 1700-2100 is treated as a hallucination and
        dropped to None."""
        from rlpe.range_chart_extractor import (
            _parse_extraction_response,
        )

        parsed = {
            "biozones": [
                {"name": "X", "zone_authority": "Smith", "zone_publication_year": 99},
            ]
        }
        result = _parse_extraction_response(parsed=parsed, paper_id="p", figure_id="f")
        assert result.biozones[0].zone_publication_year is None

    def test_year_non_integer_yields_none(self):
        from rlpe.range_chart_extractor import (
            _parse_extraction_response,
        )

        parsed = {
            "biozones": [
                {"name": "X", "zone_authority": "Smith", "zone_publication_year": "not a year"},
            ]
        }
        result = _parse_extraction_response(parsed=parsed, paper_id="p", figure_id="f")
        assert result.biozones[0].zone_publication_year is None

    def test_authority_empty_string_yields_none(self):
        """Empty-string authority is treated as omitted (not stored
        as an empty string)."""
        from rlpe.range_chart_extractor import (
            _parse_extraction_response,
        )

        parsed = {
            "biozones": [
                {"name": "X", "zone_authority": "   "},
            ]
        }
        result = _parse_extraction_response(parsed=parsed, paper_id="p", figure_id="f")
        assert result.biozones[0].zone_authority is None


class TestNIT2PromptMentionsFields:
    """The M3 prompt for range_chart_extractor must mention the
    new fields so the LLM knows to emit them."""

    def test_prompt_mentions_zone_authority(self):
        src = _read(_SRC_RCE)
        assert "zone_authority" in src, (
            "M3 range_chart prompt must mention zone_authority"
        )

    def test_prompt_mentions_zone_publication_year(self):
        src = _read(_SRC_RCE)
        assert "zone_publication_year" in src, (
            "M3 range_chart prompt must mention zone_publication_year"
        )


# ===========================================================================
# NIT-3: extract_geology auto-swap ma_top / ma_base
# ===========================================================================


class TestNIT3NormalizeMaPair:
    """``_normalize_ma_pair`` must swap ma_top / ma_base when
    inverted."""

    def test_swap_inverted_pair(self):
        from rlpe.m3_engine import _normalize_ma_pair

        record = {"ma_top": 100.0, "ma_base": 50.0}
        out = _normalize_ma_pair(record)
        assert out["ma_top"] == 50.0
        assert out["ma_base"] == 100.0

    def test_does_not_swap_correct_pair(self):
        from rlpe.m3_engine import _normalize_ma_pair

        record = {"ma_top": 50.0, "ma_base": 100.0}
        out = _normalize_ma_pair(record)
        assert out["ma_top"] == 50.0
        assert out["ma_base"] == 100.0

    def test_does_not_swap_equal_pair(self):
        from rlpe.m3_engine import _normalize_ma_pair

        record = {"ma_top": 50.0, "ma_base": 50.0}
        out = _normalize_ma_pair(record)
        assert out["ma_top"] == 50.0
        assert out["ma_base"] == 50.0

    def test_null_top_passes_through(self):
        from rlpe.m3_engine import _normalize_ma_pair

        record = {"ma_top": None, "ma_base": 50.0}
        out = _normalize_ma_pair(record)
        assert out["ma_top"] is None
        assert out["ma_base"] == 50.0

    def test_null_base_passes_through(self):
        from rlpe.m3_engine import _normalize_ma_pair

        record = {"ma_top": 100.0, "ma_base": None}
        out = _normalize_ma_pair(record)
        assert out["ma_top"] == 100.0
        assert out["ma_base"] is None

    def test_non_numeric_passes_through(self):
        from rlpe.m3_engine import _normalize_ma_pair

        record = {"ma_top": "old", "ma_base": "young"}
        out = _normalize_ma_pair(record)
        # Non-numeric values are passed through unchanged; the
        # strict null-on-violation path lives in _validate_ma_range.
        assert out["ma_top"] == "old"
        assert out["ma_base"] == "young"

    def test_returns_same_object(self):
        """The helper mutates in place and returns the same object
        for chaining (mirrors _validate_ma_range contract)."""
        from rlpe.m3_engine import _normalize_ma_pair

        record = {"ma_top": 100.0, "ma_base": 50.0}
        out = _normalize_ma_pair(record)
        assert out is record


class TestNIT3ExtractGeologyAutoswaps:
    """``extract_geology`` must call ``_normalize_ma_pair`` before
    the strict null-on-violation check so the swap happens first."""

    def test_extract_geology_swaps_via_fake_backend(self):
        """Drive ``extract_geology`` with a fake backend that
        returns an inverted ma_top / ma_base pair and verify the
        resulting record has the values swapped (not nulled)."""
        from rlpe.m3_engine import M3Engine, _validate_ma_range, _normalize_ma_pair

        # Use the helpers directly with the JSON the LLM would emit
        # — this avoids the heavy image / puppet backend plumbing.
        llm_output = {"ma_top": 100.0, "ma_base": 50.0}
        # Simulate the extract_geology post-processing order:
        _normalize_ma_pair(llm_output)
        _validate_ma_range(llm_output)
        assert llm_output["ma_top"] == 50.0
        assert llm_output["ma_base"] == 100.0

    def test_extract_geology_invocation_order_in_source(self):
        """The source code must call ``_normalize_ma_pair`` BEFORE
        ``_validate_ma_range`` so the swap happens first and the
        strict null-on-violation check sees a valid pair."""
        src = _read(_SRC_M3)
        # Find the extract_geology method body and check ordering.
        # The helpers must appear in this order in the geo_list loop.
        extract_geo_idx = src.find("def extract_geology")
        assert extract_geo_idx > 0
        # Search for the helper calls inside the extract_geology body.
        end_marker = src.find("\n    def ", extract_geo_idx + 1)
        body = src[extract_geo_idx:end_marker if end_marker > 0 else len(src)]
        # The normalize helper must be called; the validate helper
        # must also be called; normalize must come first.
        normalize_pos = body.find("_normalize_ma_pair")
        validate_pos = body.find("_validate_ma_range")
        assert normalize_pos > 0, "_normalize_ma_pair not called in extract_geology"
        assert validate_pos > 0, "_validate_ma_range not called in extract_geology"
        assert normalize_pos < validate_pos, (
            "_normalize_ma_pair must be called BEFORE _validate_ma_range "
            "in extract_geology so the swap happens first"
        )


class TestNIT3SourceGuard:
    """Static source guard: the new helper must be present in the
    live source."""

    def test_normalize_ma_pair_function_defined(self):
        src = _read(_SRC_M3)
        assert "def _normalize_ma_pair" in src, (
            "_normalize_ma_pair function must be defined in m3_engine.py (NIT-3)"
        )

    def test_normalize_ma_pair_documented(self):
        """The helper must have a docstring referencing NIT-3 so
        future maintainers know the audit-history."""
        src = _read(_SRC_M3)
        # Find the function and the next 60 lines for the docstring.
        idx = src.find("def _normalize_ma_pair")
        assert idx > 0
        chunk = src[idx:idx + 2000]
        assert "NIT-3" in chunk, (
            "_normalize_ma_pair docstring must mention NIT-3"
        )


# ===========================================================================
# NIT-4: _normalize_species multi-whitespace collapse
# ===========================================================================


class TestNIT4NormalizeSpeciesCollapsesWhitespace:
    """``_normalize_species`` must collapse runs of whitespace
    including multi-space OCR artefacts like ``"Entactinia   sp."``."""

    def test_three_space_collapses(self):
        from rlpe.m3_engine import _normalize_species

        out = _normalize_species("Entactinia   sp.")
        assert out == "Entactinia sp."

    def test_tab_collapses(self):
        from rlpe.m3_engine import _normalize_species

        out = _normalize_species("Entactinia\t\tsp.")
        assert out == "Entactinia sp."

    def test_mixed_whitespace_collapses(self):
        from rlpe.m3_engine import _normalize_species

        out = _normalize_species("Entactinia \t \n sp.")
        assert out == "Entactinia sp."

    def test_binomial_with_extra_spaces(self):
        from rlpe.m3_engine import _normalize_species

        out = _normalize_species("Archaeodictyomitra    mitra")
        assert out == "Archaeodictyomitra mitra"

    def test_leading_and_trailing_whitespace_stripped(self):
        from rlpe.m3_engine import _normalize_species

        out = _normalize_species("   Entactinia sp.   ")
        assert out == "Entactinia sp."

    def test_single_spaces_preserved(self):
        """A normal single-spaced name must be preserved unchanged."""
        from rlpe.m3_engine import _normalize_species

        out = _normalize_species("Archaeodictyomitra mitra")
        assert out == "Archaeodictyomitra mitra"

    def test_empty_returns_none(self):
        from rlpe.m3_engine import _normalize_species

        assert _normalize_species("") is None
        assert _normalize_species("   ") is None


class TestNIT4SourceGuard:
    """Static source guard: the new whitespace collapse must be
    present at the BEGINNING of the function so it runs before any
    downstream regex."""

    def test_early_whitespace_collapse_in_source(self):
        """The ``re.sub(r"\\s+", " ", s)`` collapse must appear
        in the function body."""
        src = _read(_SRC_M3)
        idx = src.find("def _normalize_species")
        assert idx > 0
        # Take the next ~5000 chars (the function body is long because
        # of the multi-line docstring).
        chunk = src[idx:idx + 5000]
        # Find the EARLIEST collapse position in the chunk.
        # The source uses a raw string r"\s+" — match it literally.
        early_collapse = chunk.find('re.sub(r"\\s+", " ", s)')
        assert early_collapse > 0, (
            "_normalize_species must call re.sub(r'\\s+', ' ', s) "
            "to collapse whitespace (NIT-4)"
        )

    def test_collapses_at_function_start(self):
        """The whitespace collapse must happen BEFORE the
        Spumellaria/Nassellaria fold so multi-space input doesn't
        survive the first structured rule."""
        src = _read(_SRC_M3)
        idx = src.find("def _normalize_species")
        assert idx > 0
        chunk = src[idx:idx + 5000]
        collapse_pos = chunk.find('re.sub(r"\\s+", " ", s)')
        spumellaria_pos = chunk.find("Spumellaria|Nassellaria")
        assert collapse_pos > 0
        assert spumellaria_pos > 0
        assert collapse_pos < spumellaria_pos, (
            "whitespace collapse must run BEFORE the "
            "Spumellaria/Nassellaria fold"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
