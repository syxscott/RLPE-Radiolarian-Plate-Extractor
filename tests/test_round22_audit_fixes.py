"""Round 22 source-guard tests: comprehensive code audit fixes.

User: "请从头到尾彻底检查代码有没有问题，尤其是前端数据的展示".
The audit found 4 CRITICAL + multiple MAJOR issues. These tests
pin the fixes:

  CRITICAL-1: ``panel_record_from_match`` and ``geology_contexts_from_matches``
              used different ``geology_context_id`` schemes, so the
              cross-table join always failed. Fix: shared
              ``_geology_context_id`` helper.

  CRITICAL-2: Frontend modal read ``g.latitude`` / ``g.longitude`` but
              the API emits ``g.modern_latitude`` / ``g.modern_longitude``.
              Paleo coordinates never rendered. Fix: read modern_*.

  CRITICAL-3: Frontend ``_geo_age`` sort key was broken — the column
              header sort indicator was active but the comparator
              read ``a._geo_age`` which was always undefined. Fix:
              compute age client-side.

  MAJOR-A1: ``coord_source`` (Round 21 centroid fallback marker) was
              never added to ``GeologyLinkRecord`` schema and was
              silently dropped at the converter. Fix: schema field.

  MAJOR-A5: ``PaperRecord.review_reasons`` was being injected into
              the dumped dict, bypassing ``extra=forbid``. Fix: declare
              the field on the schema.

  MAJOR-A6: ``LocalityRecord.coordinate_source`` was hardcoded to
              ``"caption"`` regardless of centroid fallback. Fix: read
              ``g.get("coord_source")`` from the geology link.

  FRONTEND: The modal now displays ``geology_scope``, ``coord_source``,
              paper metadata (title / authors / journal with review
              reason badges), and sample IDs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(rel: str) -> str:
    return Path(
        "/home/user/shenyaxuan/RLPE-Radiolarian-Plate-Extractor/" + rel
    ).read_text(encoding="utf-8")


# --- CRITICAL-1: shared geology_context_id helper -----------------------


def test_panel_geology_context_id_matches_geo_contexts_list():
    """End-to-end: a panel's geology_context_id must equal the
    geology_context_id in the geology_contexts list. The audit found
    panel used prefix ``"geo"`` while the list used ``"geoctx"``."""
    from rlpe.converters import (
        _geology_context_id,
        geology_contexts_from_matches,
        panel_record_from_match,
    )
    from rlpe.types import MatchResult

    geo = {
        "age": "Late Jurassic",
        "chronostratigraphy": "Kimmeridgian",
        "formation": "Fonzaso Formation",
        "member": "Lower Member",
        "locality": "Italy",
        "evidence_text": "Sample A from Fonzaso Fm, Italy.",
    }
    expected_id = _geology_context_id(geo)

    # 1. The helper itself produces a stable id.
    assert expected_id.startswith("geoctx_"), (
        f"_geology_context_id must use the geoctx prefix; got {expected_id!r}"
    )

    # 2. The same geology dict, when emitted via the panel path AND
    #    the geo_contexts path, must produce the SAME id.
    match = MatchResult(
        paper_id="p1", figure_id="f1", panel_id="1",
        species="Genus sp.", panel_path=None, bbox=None,
        confidence=0.9, caption_snippet="Sample A from Fonzaso Fm, Italy.",
        metadata={"geology_links": [geo]},
    )
    panel = panel_record_from_match(match)
    contexts = geology_contexts_from_matches([match])
    assert panel.geology_context_id == expected_id, (
        f"panel.geology_context_id ({panel.geology_context_id}) != "
        f"helper output ({expected_id})"
    )
    assert contexts[0]["geology_context_id"] == expected_id, (
        f"contexts[0].geology_context_id ({contexts[0]['geology_context_id']}) != "
        f"helper output ({expected_id})"
    )
    # And — the original bug — they must agree with each other:
    assert panel.geology_context_id == contexts[0]["geology_context_id"], (
        "Round 22 audit F-1 regression: panel and geo_contexts "
        "geology_context_id do not match — cross-table join is broken."
    )


def test_geology_context_id_helper_exists():
    """Source guard: the ``_geology_context_id`` shared helper must
    exist in converters.py so both ``panel_record_from_match`` and
    ``geology_contexts_from_matches`` use the same scheme."""
    src = _read("src/rlpe/converters.py")
    assert "def _geology_context_id(" in src
    # Both call sites must use the helper, not the inline form.
    assert src.count("_stable_id(\n                \"geo\"") == 0, (
        "converters.py still has the inline `_stable_id(\"geo\", ...)` "
        "call that broke the join. Use the shared helper."
    )


# --- CRITICAL-2: frontend uses modern_latitude/modern_longitude --------


def test_frontend_modal_uses_modern_latitude():
    """The frontend's modal renders ``g.modern_latitude`` /
    ``g.modern_longitude`` (the schema's canonical names), not the
    legacy ``g.latitude`` / ``g.longitude`` which were always None
    in API responses."""
    src = _read("web/js/app.js")
    # The legacy field reads must be gone.
    assert "g.latitude).toFixed" not in src, (
        "web/js/app.js still reads g.latitude — Round 22 audit "
        "Audit-1 regression. Use g.modern_latitude / g.modern_longitude."
    )
    assert "g.longitude).toFixed" not in src, (
        "web/js/app.js still reads g.longitude — Round 22 audit "
        "Audit-1 regression. Use g.modern_latitude / g.modern_longitude."
    )
    # The new reads must be present.
    assert "g.modern_latitude" in src and "g.modern_longitude" in src, (
        "web/js/app.js missing g.modern_latitude / g.modern_longitude — "
        "modern coordinates won't render in the modal."
    )
    # And the new paleo coordinates too.
    assert "g.paleo_latitude" in src and "g.paleo_longitude" in src, (
        "web/js/app.js missing g.paleo_latitude / g.paleo_longitude — "
        "paleo coordinates won't render in the modal."
    )


# --- CRITICAL-3: _geo_age sort actually sorts --------------------------


def test_frontend_geo_age_sort_comparator():
    """The ``_geo_age`` sort key must produce a working comparator
    that derives the age client-side. The audit found the sort
    read ``a._geo_age`` (always undefined)."""
    src = _read("web/js/app.js")
    assert "sortKey === '_geo_age'" in src, (
        "web/js/app.js missing the _geo_age special-case sort "
        "comparator. Without this, sorting by _geo_age is random."
    )
    assert "_ageOf" in src, (
        "web/js/app.js missing the _ageOf helper that derives the "
        "age from metadata.geology_links[0]."
    )


# --- MAJOR-A1: coord_source in schema ---------------------------------


def test_geology_link_record_has_coord_source():
    """``GeologyLinkRecord`` must declare ``coord_source`` so the
    Round 21 centroid fallback marker reaches the API output."""
    from rlpe.schema_models import GeologyLinkRecord

    assert "coord_source" in GeologyLinkRecord.model_fields, (
        "GeologyLinkRecord missing coord_source — Round 21 centroid "
        "fallback provenance is silently dropped at the API boundary."
    )


def test_coord_source_in_json_schema():
    """The published JSON Schema must include coord_source."""
    import json

    schema = json.loads(_read("schemas/rlpe-v1.0.0.json"))
    defs = schema.get("$defs", {})
    geo_link = defs.get("GeologyLinkRecord", {})
    props = geo_link.get("properties", {})
    assert "coord_source" in props, (
        "Published JSON Schema missing coord_source in GeologyLinkRecord. "
        "Re-run emit_json_schema() after schema changes."
    )


def test_coord_source_forwarded_in_converter():
    """The ``_geology_links_from_meta`` converter must read
    ``coord_source`` from the geology dict."""
    src = _read("src/rlpe/converters.py")
    # The wire-through must include coord_source=
    assert "coord_source=g.get" in src, (
        "converters.py doesn't forward coord_source to "
        "GeologyLinkRecord. The Round 21 centroid marker is lost."
    )


# --- MAJOR-A5: PaperRecord.review_reasons declared ------------------


def test_paper_record_has_review_reasons():
    """``PaperRecord`` must declare ``review_reasons`` so the
    Round 20 cleanup's flags actually reach the frontend."""
    from rlpe.schema_models import PaperRecord

    fields = set(PaperRecord.model_fields.keys())
    assert "review_reasons" in fields, (
        "PaperRecord missing review_reasons — Round 20 cleanup "
        "flags (e.g. title_extraction_failed) are silently dropped."
    )
    assert "needs_review" in fields, (
        "PaperRecord missing needs_review — operator cannot see "
        "whether the paper has parse issues."
    )


def test_paper_records_from_matches_uses_setattr():
    """Source guard: the cleanup wiring must assign via setattr
    on the model (not silent extra-key injection into the dumped
    dict, which bypassed extra=forbid)."""
    src = _read("src/rlpe/converters.py")
    # The silent injection code path is gone.
    assert "dumped[\"review_reasons\"] = review_reasons" not in src, (
        "converters.py still has the silent `dumped[...] = ...` "
        "injection that bypassed Pydantic extra=forbid."
    )
    # The proper setattr path is in place.
    assert "rec.review_reasons = list(review_reasons)" in src, (
        "converters.py missing the setattr-based review_reasons "
        "wiring. Cleanup flags would silently fail to reach frontend."
    )
    assert "rec.needs_review = True" in src, (
        "converters.py missing the needs_review setattr. "
        "Round 20 cleanup wouldn't surface."
    )


# --- MAJOR-A6: locality coordinate_source ---------------------------


def test_locality_coordinate_source_reads_coord_source():
    """``locality_records_from_geology`` must read ``coord_source``
    from the geology link, not hardcode ``"caption"``."""
    src = _read("src/rlpe/converters.py")
    # The hardcoded ``coordinate_source="caption"`` line is gone.
    assert 'coordinate_source="caption" if g.get("latitude")' not in src, (
        "converters.py still hardcodes coordinate_source='caption'. "
        "Round 21 centroid fallback provenance is lost."
    )
    # And the read-from-coord_source path is present.
    assert "g.get(\"coord_source\")" in src, (
        "converters.py doesn't read g.get('coord_source'). "
        "Locality records always claim 'caption' regardless of source."
    )


# --- Frontend display of paper metadata + scope ---------------------


def test_frontend_modal_shows_paper_metadata():
    """The frontend modal must display ``paper_metadata`` (title /
    authors / journal) so operators can see Round 20 cleanup
    decisions (e.g. why title=None)."""
    src = _read("web/js/app.js")
    assert "record.paper_metadata" in src, (
        "web/js app.js doesn't access record.paper_metadata in the "
        "modal — paper title / authors / journal are invisible."
    )
    assert "paperMeta.title" in src, (
        "Modal doesn't render paper title (paperMeta.title)."
    )
    assert "paperMeta.authors" in src, (
        "Modal doesn't render paper authors (paperMeta.authors)."
    )
    assert "paperMeta.journal" in src, (
        "Modal doesn't render paper journal (paperMeta.journal)."
    )
    assert "paperReviewReasons" in src, (
        "Modal doesn't render paper review_reasons (cleanup flags)."
    )


def test_frontend_modal_shows_geology_scope():
    """The frontend modal must display ``geology_scope`` (Round 19)
    so operators know whether geology is panel-specific, figure-
    anchored, or missing."""
    src = _read("web/js/app.js")
    assert "md.geology_scope" in src, (
        "web/js/app.js doesn't access md.geology_scope — Round 19 "
        "panel / figure_anchor / none marker is invisible."
    )
    assert "scopeBadge" in src, (
        "Modal doesn't render the scopeBadge derived from "
        "geology_scope (panel / figure_anchor / none)."
    )
    assert "'figure_anchor'" in src or '"figure_anchor"' in src, (
        "Modal must handle the figure_anchor scope value."
    )


def test_frontend_modal_shows_sample_ids():
    """The frontend modal must display sample IDs from geology_links
    (Round 21 prefix-tagged: S_ / B_ / R_ / N_ / L_ / P_)."""
    src = _read("web/js/app.js")
    assert "geoSampleIds" in src, (
        "web/js/app.js doesn't extract sample_ids from "
        "geology_links — Round 21 prefixes (S_, B_, R_, N_, L_, P_) "
        "are invisible in the UI."
    )
    assert "Sample IDs:" in src, (
        "Modal doesn't render the Sample IDs row."
    )


def test_frontend_shows_coord_source_badge():
    """The frontend modal must show a ``[centroid]`` badge when
    coordinates came from the Round 21 country-centroid fallback."""
    src = _read("web/js/app.js")
    assert "country_centroid" in src, (
        "web/js/app.js doesn't render the country_centroid badge. "
        "Operators can't tell which coords came from a fallback."
    )
    css = _read("web/css/style.css")
    assert ".modal-geo-source" in css, (
        "web/css/style.css missing .modal-geo-source for the "
        "centroid badge style."
    )


def test_paper_meta_block_css():
    """CSS for the paper metadata block in the modal must exist."""
    css = _read("web/css/style.css")
    assert ".modal-paper-meta" in css, (
        "web/css/style.css missing .modal-paper-meta styles."
    )