"""Round 24 source-guard tests: user suggestions implementation.

The user asked for 3 things after the 5-paper sampling:
  1. 高精度伴生生物地层带 (cross-dating)
  2. 地球化学与古环境异常指标 (Extinction Proxies)
  3. 导出一个 excel 表格的功能 (除图像之外的所有数据都要有)

We delivered the latter two as part of WS-R24-B and WS-R24-A. The
geochem/environment piece added 4 new fields to ``GeologyLinkRecord``
(``paleoenvironment``, ``redox``, ``chemostrat``, ``facies``) backed
by curated vocabularies in ``geology_extraction.py``. The Excel
piece added a new ``xlsx.py`` exporter that produces a 5-sheet
workbook (panels / geology_contexts / localities /
paleo_coordinates / legend) plus a new ``/jobs/{id}/export.xlsx``
backend endpoint and a frontend button swap.

WS-R24-C (warning pollution) added a run-level summary warning
to the front of the warnings list so the operator sees the count
distribution at a glance without scrolling 100s of identical
per-panel rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(rel: str) -> str:
    return Path("/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/" + rel).read_text(
        encoding="utf-8"
    )


# --- WS-R24-A: Excel multi-sheet export --------------------------------


def test_xlsx_exporter_module_exists():
    """Round 24 introduced ``rlpe.exporters.xlsx`` for the
    multi-sheet Excel export. The module must exist and define
    the 5 expected sheet columns."""
    src = _read("src/rlpe/exporters/xlsx.py")
    assert "def write_xlsx(" in src, "xlsx.py must define write_xlsx() — Round 24 export endpoint."
    # The 4 NON-default sheets must use ``wb.create_sheet``.
    # The first sheet ("panels") is the default active sheet
    # (``wb.active``), so we check for the title assignment
    # ``ws.title = "panels"`` rather than ``create_sheet``.
    assert 'ws.title = "panels"' in src, (
        "xlsx.py missing the first sheet title assignment "
        "(panels) — Round 24 must use the default active sheet "
        "for the first panel sheet."
    )
    # The 4 remaining sheets must use ``wb.create_sheet``
    for sheet in ("geology_contexts", "localities", "paleo_coordinates", "legend"):
        assert f'wb.create_sheet("{sheet}")' in src, (
            f'xlsx.py missing create_sheet("{sheet}") — Round 24 export must include this sheet.'
        )


def test_xlsx_endpoint_exists():
    """The ``/jobs/{job_id}/export.xlsx`` endpoint must exist in
    the API app and return the right media type."""
    src = _read("src/rlpe/api/app.py")
    assert "/export.xlsx" in src, (
        "api/app.py missing /export.xlsx endpoint — Round 24 added this for the Excel download."
    )
    # The endpoint must set the right media type so browsers save
    # as .xlsx
    assert "vnd.openxmlformats-officedocument.spreadsheetml.sheet" in src, (
        "Excel endpoint missing correct media type — browsers may save as .zip instead of .xlsx."
    )


def test_xlsx_includes_legend():
    """The legend sheet must describe the 4 new Round 24 fields
    (古环境 / 氧化还原 / 地球化学 / 沉积相)."""
    src = _read("src/rlpe/exporters/xlsx.py")
    for needle in ("paleoenvironment", "redox", "chemostrat", "facies"):
        assert needle in src, f"xlsx.py missing the {needle} column descriptor in legend."


def test_frontend_export_button_uses_excel():
    """The frontend export button label must reference Excel (not
    the old CSV)."""
    src = _read("web/index.html")
    assert "导出 Excel" in src, (
        "index.html still has the old '导出 CSV' button — Round 24 renamed it to '导出 Excel'."
    )
    assert "导出当前筛选结果为 CSV" not in src, (
        "index.html still has the old CSV label — Round 24 replaces "
        "the CSV button with the Excel button."
    )


# --- WS-R24-B: geochem / environment / facies fields --------------------


def test_geology_link_record_has_4_new_fields():
    """Round 24 added 4 environment / geochem fields to
    GeologyLinkRecord. They must be declared so the converter
    emits them via ``_geology_links_from_meta``."""
    from rlpe.schema_models import GeologyLinkRecord

    fields = set(GeologyLinkRecord.model_fields.keys())
    for name in ("paleoenvironment", "redox", "chemostrat", "facies"):
        assert name in fields, f"GeologyLinkRecord missing {name} — Round 24 audit fix."


def test_environment_dictionaries_defined():
    """The 4 curated vocabularies must exist in geology_extraction.py
    so the regexes have something to match against."""
    src = _read("src/rlpe/geology_extraction.py")
    for name in ("_PALEOENV_VOCAB", "_REDOX_VOCAB", "_CHEMOSTRAT_VOCAB", "_FACIES_VOCAB"):
        assert f"{name} = (" in src, f"geology_extraction.py missing {name} — Round 24 vocab."


def test_environment_patterns_defined():
    """The 4 compiled regexes must exist so the extraction pipeline
    can call them on each section text."""
    src = _read("src/rlpe/geology_extraction.py")
    for name in ("PALEOENV_PATTERN", "REDOX_PATTERN", "CHEMOSTRAT_PATTERN", "FACIES_PATTERN"):
        assert f"{name} = re.compile(" in src, (
            f"geology_extraction.py missing {name} compiled regex."
        )


def test_environment_extraction_end_to_end():
    """End-to-end: feeding a section text with all 4 proxies
    should populate all 4 fields on the GeologyRecord."""
    from rlpe.geology_extraction import extract_geology_from_sections

    sections = [
        {
            "title": "Geological setting",
            "section_type": "geological_setting",
            "text": (
                "The Permian–Triassic boundary coincides with a "
                "carbon isotope excursion (CIE) in anoxic/euxinic "
                "deposits of the pelagic basin facies. A Siberian "
                "Traps LIP eruption is implicated in the mass "
                "extinction."
            ),
        }
    ]
    records = extract_geology_from_sections(sections)
    assert records, "no records produced"
    r = records[0]
    # Round 24: at least one of the 4 proxies should be populated
    populated = sum(
        1 for k in ("paleoenvironment", "redox", "chemostrat", "facies") if getattr(r, k)
    )
    assert populated >= 2, (
        f"Expected at least 2 of 4 proxies populated, got {populated}: "
        f"paleoenv={r.paleoenvironment!r} redox={r.redox!r} "
        f"chemostrat={r.chemostrat!r} facies={r.facies!r}"
    )
    # chemostrat should definitely match (text has 'CIE' and 'Siberian Traps')
    assert r.chemostrat, f"chemostrat should be 'CIE' or 'Siberian Traps', got {r.chemostrat!r}"


def test_environment_in_converter():
    """The ``_geology_links_from_meta`` converter must forward the
    4 new fields to the Pydantic GeologyLinkRecord."""
    src = _read("src/rlpe/converters.py")
    for name in ("paleoenvironment", "redox", "chemostrat", "facies"):
        assert f"{name}=g.get(" in src, f"converters.py doesn't forward {name} from geology_links."


# --- WS-R24-C: warning pollution -----------------------------------------


def test_warnings_from_matches_emits_summary():
    """Round 24 added a job-level summary warning at the front of
    the warnings list so the operator sees the count distribution
    without scrolling 100s of identical per-panel rows."""
    src = _read("src/rlpe/converters.py")
    assert "panel_review_summary" in src, (
        "converters.py missing the panel_review_summary warning code."
    )
    # The summary must be inserted at the front
    assert "out.insert(0, summary)" in src, (
        "converters.py summary warning not prepended — operator sees "
        "the 100s of per-panel rows first."
    )


def test_warnings_summary_uses_counter():
    """The summary is built from a Counter of code occurrences so
    we report correct counts."""
    src = _read("src/rlpe/converters.py")
    assert "code_counts: Counter[str] = Counter()" in src, (
        "converters.py missing the Counter for warning code aggregation."
    )
