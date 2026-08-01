"""Regression tests for audit 2026-08-01 batch W4 — converters M9 coord_source None default."""

from __future__ import annotations

import re
from pathlib import Path

from rlpe.converters import _coordinate_uncertainty_for

CONVERTERS_PATH = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "converters.py"


class TestCoordinateUncertaintyFor:
    """Bug M9 (audit 2026-08-01): unknown coord_source must return None,
    not silently fall back to 1000.0 (which would masquerade as 1-km
    precision in GBIF/PBDB submissions).
    """

    def test_known_source_returns_mapped_uncertainty(self) -> None:
        # ``regex`` is mapped to ~1000m per the docstring.
        result = _coordinate_uncertainty_for("regex")
        assert result is not None
        assert isinstance(result, float)
        assert result == 1000.0

    def test_unknown_source_returns_none(self) -> None:
        # Spelling drift / new source name must NOT silently default
        # to 1000.0.  That would be indistinguishable from real
        # 1-km-precision coordinates downstream.
        assert _coordinate_uncertainty_for("regex_strict") is None

    def test_empty_string_returns_none(self) -> None:
        assert _coordinate_uncertainty_for("") is None

    def test_docstring_promise_locked(self) -> None:
        # Source guard: the ``return table.get(coord_source, ...)``
        # line must NOT carry a numeric default.  A numeric default
        # silently mislabels unknown sources as 1-km precise.
        src = CONVERTERS_PATH.read_text(encoding="utf-8")
        bad = re.search(r"return\s+table\.get\(\s*coord_source\s*,\s*[^)]+\)", src)
        assert bad is None, (
            f"converters.py must NOT default unknown coord_source "
            f"to a numeric value; found: {bad.group(0) if bad else ''}"
        )
        # And confirm the fixed line is present.
        assert "return table.get(coord_source)" in src
