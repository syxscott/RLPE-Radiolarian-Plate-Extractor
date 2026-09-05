"""Regression: GUI crashed on rows whose ``metadata.paleodb.taxonomy`` is null.

2026-09-04 user report (first *successful* LLM run after the BUG-1..4
fixes — 13 rows on Zhang 2014): loading the job into the Results tab
raised

    results_tab.py, in _extract_column
    ((row.get("metadata") or {}).get("paleodb") or {}).get("taxonomy", {}).get("family")
    AttributeError: 'NoneType' object has no attribute 'get'

and selecting a row raised the same inside image_preview._bbox_tooltip.

Root cause: ``dict.get(k, default)`` applies ``default`` only when the
key is MISSING. Real PBDB rows carry ``"taxonomy": null`` (reverse
genus→family fallback miss writes an explicit None), so every
``.get("taxonomy", {})`` hop in a chained lookup returned None and the
next ``.get(...)`` blew up. Two of the five chain sites crashed in the
user's run; two more were latent (search box / family filter — both
guarded by earlier ``if`` checks that merely delayed the crash); one
was guarded but fragile.

Fix: every hop now uses ``.get(k) or {}`` (matches the pre-existing
safe style in results_tab._format_pbdb_cell and jobs_tab). Tests cover
the two crashed paths, the two latent ones, and a source guard so no
new ``.get(..., {})`` mid-chain hop can land in the GUI again.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The row shape that triggered the crash (verbatim from the user's job
# output: 12 rows with a taxonomy dict, 1 with taxonomy=None).
ROW_TAXONOMY_NULL: dict[str, Any] = {
    "species": "Follicucullus sp.",
    "paper_id": "p",
    "figure_id": "f",
    "panel_path": "x.png",
    "metadata": {
        "paleodb": {
            "taxonomy": None,
            "occurrences": [],
            "occurrence_count": 0,
            "looked_up": True,
        }
    },
}
ROW_TAXONOMY_OK: dict[str, Any] = {
    "species": "Pseudoalbaillella sp.",
    "metadata": {
        "paleodb": {
            "taxonomy": {
                "name": "Pseudoalbaillella",
                "family": "Follicucullidae",
                "genus": "Parafollicucullus",
            },
            "occurrences": [{"country": "Japan"}],
            "looked_up": True,
        }
    },
}


# ----------------------------------------------------------------------
# _extract_column (crashed in the user's run during table build)
# ----------------------------------------------------------------------
class TestExtractColumnNullTaxonomy:
    def _tab(self):
        from rlpe.gui.results_tab import ResultsTab

        return ResultsTab.__new__(ResultsTab)  # no QWidget init (SIGSEGV combo)

    def test_null_taxonomy_returns_none(self):
        assert self._tab()._extract_column(ROW_TAXONOMY_NULL, "family") is None

    def test_dict_taxonomy_returns_family(self):
        assert self._tab()._extract_column(ROW_TAXONOMY_OK, "family") == "Follicucullidae"

    def test_missing_paleodb_returns_none(self):
        assert self._tab()._extract_column({"metadata": {}}, "family") is None

    def test_paleodb_null_returns_none(self):
        assert self._tab()._extract_column({"metadata": {"paleodb": None}}, "family") is None

    def test_metadata_null_returns_none(self):
        assert self._tab()._extract_column({"metadata": None}, "family") is None


# ----------------------------------------------------------------------
# _bbox_tooltip (crashed in the user's run on row selection)
# ----------------------------------------------------------------------
class TestBboxTooltipNullTaxonomy:
    def test_null_taxonomy_no_crash(self):
        from rlpe.gui.image_preview import _bbox_tooltip

        tip = _bbox_tooltip(dict(ROW_TAXONOMY_NULL, confidence=0.9, bbox=[1, 2, 3, 4]))
        assert "Follicucullidae" not in tip
        assert "Follicucullus" in tip  # species still rendered

    def test_dict_taxonomy_renders_family(self):
        from rlpe.gui.image_preview import _bbox_tooltip

        tip = _bbox_tooltip(dict(ROW_TAXONOMY_OK))
        assert "Follicucullidae" in tip

    def test_paleodb_null_no_crash(self):
        from rlpe.gui.image_preview import _bbox_tooltip

        assert _bbox_tooltip({"metadata": {"paleodb": None}}) == ""


# ----------------------------------------------------------------------
# _filter_rows latent crash paths (search box / family filter)
# ----------------------------------------------------------------------
class _Stub:
    def __init__(self, value: Any = None, text: str = ""):
        self._value = value
        self._text = text

    def text(self) -> str:
        return self._text

    def currentData(self) -> Any:
        return self._value

    # QComboBox surface used by _refresh_filter_options
    def blockSignals(self, _b: bool) -> None:
        pass

    def clear(self) -> None:
        self._items: list[Any] = []

    def addItem(self, label: str, userData: Any = None) -> None:
        self._items.append(userData if userData is not None else label)

    def addItems(self, labels: list[str]) -> None:
        self._items.extend(labels)

    def setCurrentIndex(self, _i: int) -> None:
        pass

    def count(self) -> int:
        return len(getattr(self, "_items", []))


class TestFilterRowsNullTaxonomy:
    def _tab(self, rows: list[dict[str, Any]], search: str = "", family: str = ""):
        from rlpe.gui.results_tab import ResultsTab

        tab = ResultsTab.__new__(ResultsTab)
        tab._all_rows = rows
        tab._search_edit = _Stub(text=search)
        tab._species_filter = _Stub(value="__ALL__")
        tab._family_filter = _Stub(value=(family or "__ALL__"))
        tab._has_pbdb = _Stub(value="__ANY__")
        return tab

    def test_search_over_null_taxonomy_row(self):
        # Latent crash #1: typing anything into the search box built a
        # blob from the same broken chain. "follicucullus" matches the
        # null-taxonomy row via species but not the OK row (its family
        # "Follicucullidae" does not contain that substring).
        rows = self._tab(
            [ROW_TAXONOMY_NULL, ROW_TAXONOMY_OK], search="follicucullus"
        )._filter_rows()
        assert [r["figure_id"] for r in rows] == ["f"]

    def test_family_filter_over_null_taxonomy_row(self):
        # Latent crash #2: selecting a family compared every row's
        # family through the same broken chain.
        rows = self._tab(
            [ROW_TAXONOMY_NULL, ROW_TAXONOMY_OK], family="Follicucullidae"
        )._filter_rows()
        assert [r["species"] for r in rows] == ["Pseudoalbaillella sp."]

    def test_refresh_filter_options_with_null_taxonomy(self):
        # The guarded comprehension must still yield the family exactly
        # once with a null-taxonomy row present.
        from rlpe.gui.results_tab import ResultsTab

        tab = ResultsTab.__new__(ResultsTab)
        tab._all_rows = [ROW_TAXONOMY_NULL, ROW_TAXONOMY_OK]
        tab._species_filter = _Stub()
        tab._family_filter = _Stub()
        tab._refresh_filter_options()
        assert tab._family_filter.count() == 2  # "(all)" + 1 family


# ----------------------------------------------------------------------
# Source guard: no more literal-default dict hops in GUI chains
# ----------------------------------------------------------------------
class TestNoLiteralDefaultChainHops:
    def test_no_get_with_empty_dict_default_in_gui(self):
        """``.get(k, {})`` on a dynamic data hop returns None (not {})
        when the key exists with a null value — the exact trap that
        crashed the user's run. Data-field hops in the GUI must use
        ``.get(k) or {}``. (Static config tables like i18n's
        ``STRINGS.get("en", {})`` are exempt: their values are literal
        dicts, never null.)"""
        offenders = []
        for path in sorted((_SRC / "rlpe" / "gui").rglob("*.py")):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re_default_hop.search(stripped):
                    offenders.append(f"{path.relative_to(_SRC)}:{lineno}: {stripped}")
        assert not offenders, (
            ".get(k, {}) on a dynamic data field returns None when the "
            "key holds null; use .get(k) or {} instead:\n" + "\n".join(offenders)
        )


# Dynamic per-row data fields whose values may legitimately be null in
# persisted matches.jsonl rows (proven by the user's real job output).
re_default_hop = re.compile(
    r'\.get\("(?:metadata|paleodb|taxonomy|geology_links|occurrences|bbox|bounding_box)",'
    r"\s*\{\}\)"
)
