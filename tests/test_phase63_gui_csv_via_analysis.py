"""Tests for Phase 63 Plan 6 — Bug 6.5: GUI CSV export must route through
``analysis.write_csv``.

Before: ``ResultsTab._export_csv`` invoked ``csv.DictWriter`` directly
with ``encoding="utf-8"``, no formula sanitisation, no UTF-8 BOM, no
DwC terms. Excel on Windows mangled Greek letters / CJK; a paper
title like ``=cmd|'/c calc'!A1`` would execute on open.

After: the GUI CSV uses ``analysis.write_csv`` semantics — formula
sanitisation (Round 15), UTF-8 BOM (Phase 63 Plan 6.10), DwC-style
columns.

The GUI still uses its tabular column layout (``RESULT_COLUMNS``) as
the human-readable view but pipes the cell values through the
analysis sanitiser, so a runaway formula from a paper caption can
never make it into the exported CSV.

These tests pin the source-code behaviour so a future refactor that
re-introduces the direct ``csv.DictWriter`` path will fail loudly.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read_results_tab_csv_writer():
    """Find the CSV-writing code path inside results_tab.py."""
    src = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / "results_tab.py"
    return src.read_text(encoding="utf-8")


def test_results_tab_csv_uses_analysis_sanitiser():
    """``_export_csv`` must call ``_sanitise_csv_cell`` (analysis.py) for
    every cell. Source-guard: a direct ``csv.DictWriter`` call without
    the analysis-level sanitiser fails the test.

    The previous bug was a ``csv.DictWriter(fh, ...)`` call whose
    writerow used ``self._extract_column(...)`` directly — formula
    injection was possible.
    """
    src = _read_results_tab_csv_writer()
    # The fix: the writerow loop must call _sanitise_csv_cell.
    assert "csv.DictWriter" in src, "results_tab.py still defines CSV export"
    # The sanitiser call must appear in the same method (heuristic:
    # _export_csv runs from "# Phase 63" until the next blank-line
    # boundary).
    fn_idx = src.find("def _export_csv")
    assert fn_idx > 0
    # Slice forward until the next def or end-of-file
    end = src.find("\n    def ", fn_idx + 1)
    fn_block = src[fn_idx:end if end > 0 else len(src)]
    assert "_sanitise_csv_cell" in fn_block, (
        "results_tab._export_csv is writing cells without "
        "_sanitise_csv_cell — Excel formula injection is possible. "
        "Phase 63 Plan 6.5 fix regressed?"
    )


def test_results_tab_csv_uses_utf8_sig_bom():
    """The file open() in ``_export_csv`` must use ``encoding='utf-8-sig'``
    (Phase 63 Plan 6.10). Without the BOM, Excel on Windows misreads
    Greek / CJK characters."""
    src = _read_results_tab_csv_writer()
    fn_idx = src.find("def _export_csv")
    end = src.find("\n    def ", fn_idx + 1)
    fn_block = src[fn_idx:end if end > 0 else len(src)]
    assert "utf-8-sig" in fn_block, (
        "results_tab._export_csv is not writing a UTF-8 BOM — Excel "
        "on Windows will mangle Greek / CJK. Phase 63 Plan 6.10 fix "
        "regressed?"
    )


def test_results_tab_csv_handles_nan_inf():
    """CSV cells must replace ``float('nan')`` / ``float('inf')`` with
    empty strings."""
    src = _read_results_tab_csv_writer()
    fn_idx = src.find("def _export_csv")
    end = src.find("\n    def ", fn_idx + 1)
    fn_block = src[fn_idx:end if end > 0 else len(src)]
    # Either the analysis sanitiser is called (which handles NaN/Inf)
    # or the GUI performs an inline replacement. The csv module
    # doesn't natively handle NaN/Inf — pandas writes "nan"/"inf"
    # strings by default. We accept either pattern.
    has_nan_handler = (
        "_sanitise_csv_cell" in fn_block
        or "_csv_cell" in fn_block
        or ("nan" in fn_block and "inf" in fn_block)
    )
    assert has_nan_handler, (
        "results_tab._export_csv has no NaN/Inf handler — float "
        "values such as NaN/Inf will write the python repr to CSV. "
        "Phase 63 Plan 6.8 fix regressed?"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
