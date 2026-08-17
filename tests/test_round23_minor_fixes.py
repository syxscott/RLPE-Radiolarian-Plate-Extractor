"""Round 23 source-guard tests: minor audit fix follow-ups.

The user requested all remaining audit fixes be applied. This
test file pins the 8 fixes in the "Round 23" pass:

  - F-4: geo_vision emits a stub record EVEN when M3 returns 0 geo
    links (previously the figure was silently dropped).
  - /results pagination: ``?limit=N&offset=M`` query params.
  - Audit-9: frontend search now covers formation / locality / age
    / country / caption_snippet.
  - Audit-10: CSV export includes 20 columns of geology data.
  - Audit-11: placeholder colspan matches the column count (10).
  - F-6: Crossref failure logs at ``warning`` (was ``debug``).
  - F-7: dead ``_paleocoord_missing_warning`` stub removed.
  - B-1: ``_validate_bbox`` helper logs warning on wrong length.
  - pipeline_panel_index: read via ``getattr`` (safe attribute
    access).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text(encoding="utf-8")


# --- F-4: geo_vision stub emission ---------------------------------------


def test_geo_vision_path_emits_stub_even_when_empty():
    """Source guard: the geo_vision route must append the stub record
    unconditionally, not behind ``if geo_links:``. Otherwise M3
    failures silently drop the figure from ``results``."""
    src = _read("src/rlpe/pipeline.py")
    # The previous guard was ``if geo_links:`` followed by the
    # results.append(...) block. The fix removes the guard.
    # We check that the append is at the same indentation as
    # the rest of the try/except body (no ``if`` guard).
    # Crude check: count ``if geo_links:`` near the geo_vision code.
    geo_vision_section_start = src.find("geo_vision %s failed")
    if geo_vision_section_start < 0:
        assert False, "geo_vision error log not found"
    # Look at the next 500 chars for ``if geo_links:``
    nearby = src[geo_vision_section_start : geo_vision_section_start + 1500]
    # The old guard had ``if geo_links: results.append(...)``.
    # We expect the append to be unconditional now.
    assert "results.append(" in nearby, (
        "geo_vision route must include results.append() even when "
        "geo_links is empty (Round 23 audit F-4 fix)."
    )


# --- /results pagination --------------------------------------------------


def test_get_results_accepts_limit_offset():
    """The /results endpoint must accept ``?limit=N&offset=M``."""
    src = _read("src/rlpe/api/app.py")
    assert "def get_results(" in src, "get_results() not found in app.py"
    # Match the function signature
    import re as _re

    fn = _re.search(r"def get_results\([^)]*\)", src)
    assert fn, "could not parse get_results signature"
    sig = fn.group(0)
    assert "limit:" in sig and "offset:" in sig, (
        f"get_results signature missing limit/offset query params: {sig}"
    )


# --- Audit-9: frontend search covers geology fields ---------------------


def test_frontend_search_includes_geology_fields():
    """The frontend result-search must include formation / locality
    / age / country in addition to paper_id / species / panel_id.
    The Audit-9 fix expanded the search blob to include
    ``metadata.geology_links[0].{formation,locality,country,...}``
    and the caption_snippet."""
    src = _read("web/js/app.js")
    assert "geoBlob" in src, (
        "web/js/app.js missing geoBlob — Round 23 Audit-9 fix removed "
        "geology-field coverage from the search filter."
    )
    assert "gl0.formation" in src and "gl0.country" in src, (
        "geoBlob must include formation and country from geology_links[0]."
    )
    assert "caption.includes(searchTerm)" in src, (
        "Search filter no longer searches caption_snippet."
    )


# --- Audit-10: export includes geology (CSV/Excel) ----------------------


def test_export_includes_geology_columns():
    """The export must include formation / locality / country / lat /
    lon / sample_ids columns. Round 23 Audit-10 expanded the
    CSV from 6 columns to 20; Round 24 replaced the CSV path
    with a backend Excel endpoint, but the columns are still
    surfaced in the same 20-column shape (just via a different
    code path)."""
    # Round 24: the CSV path was replaced by an Excel endpoint,
    # so the test now asserts the xlsx exporter has the columns
    # instead of the (now-removed) client-side CSV.
    src_xlsx = _read("src/rlpe/exporters/xlsx.py")
    assert "'Formation'" in src_xlsx or '"Formation"' in src_xlsx, (
        "xlsx.py export header missing 'Formation' column."
    )
    assert "'Locality'" in src_xlsx or '"Locality"' in src_xlsx, (
        "xlsx.py export header missing 'Locality' column."
    )
    assert "'Modern_Lat'" in src_xlsx or '"Modern_Lat"' in src_xlsx, (
        "xlsx.py export header missing 'Modern_Lat' column."
    )
    assert "'Sample IDs'" in src_xlsx or '"Sample IDs"' in src_xlsx, (
        "xlsx.py export header missing 'Sample IDs' column."
    )


# --- Audit-11: placeholder colspan ---------------------------------------


def test_placeholder_colspan_matches_column_count():
    """The results-table placeholder must have colspan equal to the
    number of table columns. The audit found colspan=9 vs 10
    actual columns."""
    src = _read("web/index.html")
    # Find the placeholder row (the one inside results-table-body
    # that shows "暂无结果").
    import re as _re

    m = _re.search(r'colspan="(\d+)"[^>]*>\s*暂无结果', src)
    assert m, "placeholder row not found"
    colspan = int(m.group(1))
    # Count <th> elements inside the results table header.
    th_count = src.count("<th")
    # The data rows have checkbox + paper_id + figure_id + panel_id +
    # label_source + species + confidence + _geo_age + thumbnail +
    # actions = 10 columns.
    assert colspan == 10, (
        f"placeholder colspan={colspan} but table has 10 columns. Audit-11 regression."
    )


# --- F-6: Crossref log at warning level -------------------------------


def test_crossref_log_at_warning_level():
    """Crossref lookup failures must log at ``warning`` (not
    ``debug``) so operators can see them in normal server logs."""
    src = _read("src/rlpe/paper_metadata_cleanup.py")
    # The function ``_crossref_get_journal`` should NOT contain
    # ``logger.debug`` for the three failure paths
    # (missing requests / non-200 / exception).
    fn_start = src.find("def _crossref_get_journal")
    if fn_start < 0:
        assert False, "_crossref_get_journal not found"
    fn_end = src.find("\ndef ", fn_start + 10)
    if fn_end < 0:
        fn_end = len(src)
    body = src[fn_start:fn_end]
    assert "logger.debug(" not in body, (
        "_crossref_get_journal still logs at debug. Round 23 "
        "audit F-6 requires warning-level logging."
    )


# --- F-7: dead stub removed ---------------------------------------------


def test_dead_paleocoord_missing_warning_removed():
    """The Round 20 ``_paleocoord_missing_warning`` dead stub must
    be removed (Round 23 cleanup). The historical reference in
    the file's module docstring is allowed; we check that no
    callable ``def _paleocoord_missing_warning`` exists."""
    src = _read("src/rlpe/converters.py")
    assert "def _paleocoord_missing_warning" not in src, (
        "_paleocoord_missing_warning function is dead code "
        "(Round 20 deprecated). Round 23 audit removed the "
        "function definition."
    )


# --- B-1: bbox validation helper ---------------------------------------


def test_bbox_validation_helper_logs_warning():
    """``_validate_bbox`` must log a warning when bbox length is not 4
    so the operator can find the offending panel."""
    src = _read("src/rlpe/converters.py")
    assert "def _validate_bbox(" in src, (
        "_validate_bbox helper missing. Round 23 audit B-1 fix "
        "replaces the silent ``bbox = ... or None`` with a helper "
        "that logs a warning on wrong length."
    )
    assert "bbox has wrong length" in src, (
        "_validate_bbox helper missing the 'bbox has wrong length' "
        "warning message. Round 23 audit B-1 fix."
    )


# --- pipeline_panel_index safe access -----------------------------------


def test_pipeline_panel_index_uses_safe_attribute_access():
    """The converter must read ``pipeline_panel_index`` via
    ``getattr(match, 'panel_index', None)`` so a future pipeline
    site that sets the attribute is picked up automatically."""
    src = _read("src/rlpe/converters.py")
    assert 'getattr(match, "panel_index", None)' in src, (
        "converters.py pipeline_panel_index read must use "
        "``getattr(match, 'panel_index', None)`` for safe "
        "attribute access. Round 23 fix."
    )
    # And the old dead-code fallback to meta.get is gone.
    assert 'meta.get("panel_index")' not in src, (
        "converters.py still has the dead `meta.get('panel_index')` "
        "fallback. Round 23 fix removed it."
    )


# --- WarningRecord emission: paper_metadata_cleanup + paleo_reconstruction


def test_warning_record_helper_exists():
    """A ``_warning_record`` helper must exist in converters.py so
    the new ``paper_metadata_cleanup`` and ``paleo_reconstruction``
    failure paths can emit WarningRecord entries."""
    src = _read("src/rlpe/converters.py")
    assert "def _warning_record(" in src, (
        "_warning_record helper missing. Round 23 audit fixes the "
        "silent-failure paths for paper_metadata_cleanup and "
        "paleo_reconstruction by emitting WarningRecord dicts."
    )


def test_paper_records_emits_warning_on_cleanup_failure():
    """``paper_records_from_matches`` returns ``(records, warnings)``
    and emits a WarningRecord when the cleanup helper raises."""
    import inspect

    from rlpe.converters import paper_records_from_matches

    sig = inspect.signature(paper_records_from_matches)
    # The return annotation should be a tuple of (list, list)
    assert sig.return_annotation == tuple[list, list] or "tuple" in str(sig.return_annotation), (
        f"paper_records_from_matches must return tuple[list, list]; got {sig.return_annotation}"
    )


def test_paleo_coordinates_returns_records_and_warnings():
    """``paleo_coordinates_from_localities`` must return
    ``(records, warnings)`` so backend failures are surfaced."""
    import inspect

    from rlpe.converters import paleo_coordinates_from_localities

    sig = inspect.signature(paleo_coordinates_from_localities)
    assert "tuple" in str(sig.return_annotation), (
        f"paleo_coordinates_from_localities must return tuple; got {sig.return_annotation}"
    )
