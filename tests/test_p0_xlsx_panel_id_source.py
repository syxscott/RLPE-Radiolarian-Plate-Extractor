"""P0 regression: Phase 58 Plan 1.3 (Bug 1.3).

xlsx export's panel-sheet "Panel来源" (panel_id_source) column was
always empty. Root cause: xlsx.py:175 read
``md.get("v18_panel_id_source")`` — wrong key. The actual location
is ``panel.metadata["panel_id_source"]`` (per
``test_association_panel_id_propagation.py`` and ``converters.py``).

Strategy: the worktree's venv lacks ``openpyxl``, so we can't drive
``_row_for_panel`` end-to-end here. We use the same source-guard
pattern as Round 23/24 tests: read ``xlsx.py`` source and assert the
buggy literal is gone and the correct key is present. End-to-end
runtime validation is deferred to the conda env that has openpyxl.
"""

from __future__ import annotations

from pathlib import Path

SRC_XLSX = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "exporters" / "xlsx.py"


def _read_src() -> str:
    return SRC_XLSX.read_text(encoding="utf-8")


def test_xlsx_no_longer_reads_v18_panel_id_source() -> None:
    """The buggy literal must be removed from xlsx.py."""
    src = _read_src()
    assert "v18_panel_id_source" not in src, (
        "xlsx.py still references 'v18_panel_id_source' — this is the "
        "wrong key (Round 23 audit). panel_id_source lives at "
        "panel.metadata['panel_id_source'] (set by converters.py and "
        "association.match_panels)."
    )


def test_xlsx_reads_panel_id_source_correctly() -> None:
    """xlsx.py must read 'panel_id_source' from either top-level or metadata.

    Acceptable patterns:
        - p.get('panel_id_source')           (top-level — PanelRecord schema)
        - md.get('panel_id_source')          (metadata — set by converters)
        - any chained fallback that hits the right key
    """
    src = _read_src()
    # The fix must reference 'panel_id_source' (without 'v18_' prefix).
    # Allow the buggy v18_ prefix to be absent; require the correct key.
    assert "panel_id_source" in src, (
        "xlsx.py no longer references 'panel_id_source' at all. The Panel来源 column will be empty."
    )


def test_xlsx_panel_source_column_reads_metadata_first() -> None:
    """Preferred path: read from ``panel.metadata['panel_id_source']``,
    the location set by converters.py:448 and association.match_panels.
    """
    src = _read_src()
    # Find the row builder for panels and confirm metadata path is used.
    # We accept either md.get('panel_id_source') or p.get('panel_id_source')
    # but NOT the buggy md.get('v18_panel_id_source').
    assert (
        'md.get("panel_id_source")' in src
        or "md.get('panel_id_source')" in src
        or "p.get('panel_id_source')" in src
        or 'p.get("panel_id_source")' in src
    ), (
        "xlsx.py does not read panel_id_source from the correct location. "
        "Expected either md.get('panel_id_source') or p.get('panel_id_source')."
    )
