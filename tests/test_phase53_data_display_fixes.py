"""Phase 53 — fix data display + matching bugs in ResultsTab / RunTab.

Phase 33-52 fixed engineering bugs (layout / i18n / disk scan /
NameError). Phase 53 fixes bugs in the data display and
matching layer — the part the user actually interacts with
to inspect their extracted data.

Bugs found in audit:

  * BLOCKER-1: results_tab.py _filter_rows had
      if has_pbdb != "(any)":
    but has_pbdb is the userData sentinel ("__ANY__" / "yes" /
    "no"), never the display string "(any)". So the condition
    was ALWAYS true → the PBDB filter always ran → when user
    selected "(any)" (show all), rows WITH PBDB data were
    silently dropped.

  * BLOCKER-2: _extract_column("page_index") returned
    row.get("page_index") which is None — the pipeline stores
    page_index at metadata.page_index, not at the row top level.
    The Page column showed "—" for every row.

  * MAJOR-1: apply_settings used findText("minimax") to find
    the LLM backend combo entry, but the combo stores friendly
    names as itemText and ISO codes as userData (Phase 47
    pattern). findText("minimax") always failed to match
    "MiniMax-M3 (推荐)". Same bug for m3_prompt_lang.

  * MAJOR-2: Column headers set from hardcoded English on init
    ("Species (Latin)", "Panel ID", ...). The _refresh_texts
    listener applied i18n later, but first-paint users saw
    English regardless of locale.

(Note: the audit also flagged BLOCKER-3 about `family` always
being empty. Investigating showed the data contract is correct
— ``metadata.paleodb.taxonomy.family`` is the real path. The
"empty" rows simply have use_paleodb=False (it's opt-in). No
fix needed.)

Tests:
  1. BLOCKER-1: "(any)" sentinel keeps both rows-with and rows-without PBDB
  2. BLOCKER-1: "yes" filter keeps only rows with PBDB data
  3. BLOCKER-1: "no" filter keeps only rows without PBDB data
  4. BLOCKER-2: page_index is read from metadata.page_index
  5. MAJOR-1: apply_settings restores llm_backend by ISO code
  6. MAJOR-1: apply_settings restores m3_prompt_lang by ISO code
  7. MAJOR-2: column headers are in zh_CN on first paint (init)
  8. MAJOR-2: column headers are in en on first paint
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QTableWidgetItem  # noqa: E402

_app = QApplication.instance() or QApplication([])


import pytest


@pytest.fixture(autouse=True)
def _reset_language():
    from rlpe.gui import i18n
    i18n.set_language("zh_CN")
    yield
    i18n.set_language("zh_CN")


# ============================================================
# Test data: realistic rows that mimic pipeline output
# ============================================================
@pytest.fixture
def sample_rows():
    """A small batch of rows that mimic real pipeline output.

    Includes:
      - Row with PBDB data (looked_up=True, taxonomy.family present)
      - Row without PBDB data (no metadata.paleodb)
      - Row with page_index in metadata
    """
    return [
        {
            "species": "Species A",
            "panel_id": "p1",
            "caption_snippet": "Fig. 1 caption",
            "label_text": "1",
            "metadata": {
                "page_index": 5,
                "paleodb": {
                    "looked_up": True,
                    "taxonomy": {"family": "FamilyX", "order": "OrderY"},
                    "occurrence_count": 3,
                },
            },
        },
        {
            "species": "Species B",
            "panel_id": "p2",
            "caption_snippet": "Fig. 2 caption",
            "label_text": "2",
            "metadata": {
                "page_index": 7,
                # No paleodb — use_paleodb was False
            },
        },
        {
            "species": "Species C",
            "panel_id": "p3",
            "caption_snippet": "Fig. 3 caption",
            "label_text": "3",
            "metadata": {
                "page_index": 12,
                "paleodb": {
                    "looked_up": False,  # looked_up but no taxonomy
                    "taxonomy": {},
                },
            },
        },
    ]


# ============================================================
# BLOCKER-1: has_pbdb filter
# ============================================================
def test_pbdb_any_keeps_all_rows(sample_rows):
    """Phase 53 BLOCKER-1 fix: when user selects "(any)" / "__ANY__",
    all rows must be kept (regardless of PBDB status).

    Before the fix, the comparison ``has_pbdb != "(any)"`` was always
    True because ``has_pbdb`` is the sentinel "__ANY__" / "yes" / "no",
    never the display string "(any)". The filter block always ran,
    and rows WITH PBDB data were dropped when the user wanted "(any)".
    """
    from rlpe.gui.results_tab import ResultsTab
    rt = ResultsTab()
    rt._all_rows = sample_rows
    # Default selection on the has_pbdb combo is "__ANY__"
    assert rt._has_pbdb.currentData() == "__ANY__"
    filtered = rt._filter_rows()
    assert len(filtered) == 3, (
        f"BLOCKER-1: with '(any)' selected, all 3 rows must be kept, "
        f"got {len(filtered)} (rows-with-PBDB were being silently dropped)"
    )


def test_pbdb_yes_keeps_only_rows_with_pbdb(sample_rows):
    """Phase 53: 'yes' filter keeps only rows where paleodb was
    successfully looked up AND has taxonomy data."""
    from rlpe.gui.results_tab import ResultsTab
    rt = ResultsTab()
    rt._all_rows = sample_rows
    # Find the "yes" item (userData "yes")
    for i in range(rt._has_pbdb.count()):
        if rt._has_pbdb.itemData(i) == "yes":
            rt._has_pbdb.setCurrentIndex(i)
            break
    filtered = rt._filter_rows()
    # Only row 0 (Species A) has looked_up=True AND taxonomy
    assert len(filtered) == 1
    assert filtered[0]["species"] == "Species A"


def test_pbdb_no_keeps_only_rows_without_pbdb(sample_rows):
    """Phase 53: 'no' filter keeps only rows without PBDB data."""
    from rlpe.gui.results_tab import ResultsTab
    rt = ResultsTab()
    rt._all_rows = sample_rows
    for i in range(rt._has_pbdb.count()):
        if rt._has_pbdb.itemData(i) == "no":
            rt._has_pbdb.setCurrentIndex(i)
            break
    filtered = rt._filter_rows()
    # Rows 1 (no paleodb) and 2 (looked_up=False) should remain
    species_kept = [r["species"] for r in filtered]
    assert "Species B" in species_kept
    assert "Species C" in species_kept
    assert "Species A" not in species_kept


# ============================================================
# BLOCKER-2: page_index extraction
# ============================================================
def test_extract_column_page_index_reads_from_metadata(sample_rows):
    """Phase 53 BLOCKER-2 fix: page_index must read from metadata.page_index.

    Before the fix, the code did ``row.get("page_index")`` which
    returned None because the pipeline stores page_index at
    ``metadata.page_index``. The Page column always showed "—".
    """
    from rlpe.gui.results_tab import ResultsTab
    rt = ResultsTab()
    # Row 0 has page_index=5 in metadata
    assert rt._extract_column(sample_rows[0], "page_index") == 5, (
        "BLOCKER-2: page_index should read from metadata.page_index"
    )
    assert rt._extract_column(sample_rows[1], "page_index") == 7
    assert rt._extract_column(sample_rows[2], "page_index") == 12


def test_extract_column_page_index_handles_missing_metadata():
    """Phase 53: if metadata is missing entirely, page_index returns None."""
    from rlpe.gui.results_tab import ResultsTab
    rt = ResultsTab()
    row = {"species": "X"}  # no metadata at all
    assert rt._extract_column(row, "page_index") is None


# ============================================================
# MAJOR-1: apply_settings uses findData not findText
# ============================================================
def test_run_tab_apply_settings_restores_llm_backend_by_iso_code():
    """Phase 53 MAJOR-1 fix: apply_settings must use findData for the
    LLM backend combo (ISO code "minimax-m3" in userData). Before
    the fix, findText("minimax-m3") failed to match the friendly
    label "MiniMax-M3 (备用接入点)"."""
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    rt.apply_settings({"llm_backend": "minimax-m3"})
    settings = rt.collect_settings()
    assert settings["llm_backend"] == "minimax-m3", (
        f"apply_settings should restore llm_backend='minimax-m3' by ISO code, "
        f"got {settings['llm_backend']!r} (MAJOR-1 bug: findText vs findData)"
    )


def test_run_tab_apply_settings_restores_m3_prompt_lang_by_iso_code():
    """Phase 53: same fix for M3 prompt language."""
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    rt.apply_settings({"m3_prompt_lang": "ja"})
    settings = rt.collect_settings()
    assert settings["m3_prompt_lang"] == "ja", (
        f"apply_settings should restore m3_prompt_lang='ja' by ISO code, "
        f"got {settings['m3_prompt_lang']!r}"
    )


def test_run_tab_apply_settings_falls_back_to_findtext_for_legacy():
    """Phase 53: if a user has a legacy config with the friendly name
    in settings (e.g. migrated from an older GUI version), apply_settings
    should still work via findText fallback."""
    from rlpe.gui.run_tab import RunTab
    rt = RunTab({})
    # Try with the friendly zh_CN label (would only match zh_CN lang)
    from rlpe.gui import i18n
    i18n.set_language("zh_CN")
    # friendly label for "rules" in zh_CN
    from rlpe.gui.constants import llm_backend_friendly_options
    rules_label = next(
        label for code, label in llm_backend_friendly_options() if code == "rules"
    )
    rt.apply_settings({"llm_backend": rules_label})
    settings = rt.collect_settings()
    assert settings["llm_backend"] == "rules", (
        f"apply_settings should fall back to findText for legacy configs "
        f"with friendly label, got {settings['llm_backend']!r}"
    )


# ============================================================
# MAJOR-2: column headers i18n on init
# ============================================================
def test_results_tab_column_headers_translated_on_init_zh():
    """Phase 53 MAJOR-2 fix: column headers must be translated on
    FIRST PAINT, not just after a language switch.

    Before the fix, headers were set from ``c.label`` (hardcoded
    English constants in constants.py). A user with zh_CN as their
    default language saw English headers until they manually toggled
    languages.
    """
    from rlpe.gui import i18n
    from rlpe.gui.results_tab import ResultsTab
    i18n.set_language("zh_CN")
    rt = ResultsTab()
    headers = [
        rt._table.horizontalHeaderItem(i).text()
        for i in range(rt._table.columnCount())
    ]
    # Headers should contain Chinese characters (zh_CN translation)
    joined = " ".join(headers)
    assert any('一' <= c <= '鿿' for c in joined), (
        f"Phase 53 MAJOR-2: zh_CN column headers should contain Chinese "
        f"characters, got {headers!r}"
    )


def test_results_tab_column_headers_translated_on_init_en():
    """Phase 53: EN headers also translate on init."""
    from rlpe.gui import i18n
    from rlpe.gui.results_tab import ResultsTab
    i18n.set_language("en")
    rt = ResultsTab()
    headers = [
        rt._table.horizontalHeaderItem(i).text()
        for i in range(rt._table.columnCount())
    ]
    joined = " ".join(headers).lower()
    # English headers should mention species/panel/caption
    assert "species" in joined or "panel" in joined, (
        f"EN column headers should mention species/panel, got {headers!r}"
    )


def test_results_tab_headers_use_i18n_key_not_hardcoded_label():
    """Phase 53 source guard: ensure setHorizontalHeaderLabels on init
    uses i18n._tr(...) and NOT the hardcoded c.label."""
    import inspect
    from rlpe.gui.results_tab import ResultsTab
    # The fix lives in _build_ui (called from __init__), not __init__
    # directly. Check both.
    src_init = inspect.getsource(ResultsTab.__init__)
    src_build = inspect.getsource(ResultsTab._build_ui)
    combined = src_init + src_build
    # The fix uses f"restab.col.{c.key}" inside i18n._tr(...)
    assert 'restab.col.{c.key}' in combined, (
        "ResultsTab._build_ui should use i18n._tr(f\"restab.col.{c.key}\") "
        "for column headers, not the hardcoded c.label (MAJOR-2 bug)."
    )
    # And the bad pattern (c.label directly) should not appear in
    # the setHorizontalHeaderLabels call
    assert '[c.label for c in RESULT_COLUMNS]' not in combined, (
        "ResultsTab still uses [c.label for c in RESULT_COLUMNS] — "
        "should use [i18n._tr(f'restab.col.{c.key}') for c in RESULT_COLUMNS]"
    )
