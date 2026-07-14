"""Round 18 source-guard tests: ABCD complete coverage.

Locks in the four-part Round 18 user-requested enrichment:

  A — Regex extraction fills member / group / lithology / country /
      biozone / paleo vs modern coordinate split.
  B — M3Engine.extract_geology vision path is enabled by default and
      the plate figure_type is in the allowlist.
  C — PBDB / GPlates-style paleo reconstruction fills paleo_lat /
      paleo_lon / plate_id / reconstruction_model / reconstruction_age_ma.
  D — Frontend modal renders evidence_text as a collapsible block and
      the paleo / modern / plate chips.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(path: str) -> str:
    return Path(__file__).resolve().parents[1].joinpath(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A — Regex enrichment
# ---------------------------------------------------------------------------


def test_geology_extraction_splits_member_group_formation():
    """The formation regex output must split into ``group``,
    ``formation``, and ``member`` instead of collapsing all three
    into a single ``formation`` field."""
    src = _read("src/rlpe/geology_extraction.py")
    assert "_GROUP_RE" in src, "Group regex missing"
    assert "_FORMATION_RE" in src, "Formation regex missing"
    assert "_MEMBER_RE" in src, "Member regex missing"
    # The legacy greedy FORMATION_PATTERN was retired.
    assert 'groups[0]' in src or "groups[0] if groups" in src, (
        "groups[0] must be threaded into the new field path"
    )
    assert 'members[0]' in src, "members[0] must be threaded into the new field path"


def test_geology_extraction_has_lithology_and_biozone():
    """``LITHOLOGY_PATTERN`` and ``_BIOZONE_RE`` must be defined
    so the rock-type and biozone fields are populated."""
    src = _read("src/rlpe/geology_extraction.py")
    assert "LITHOLOGY_PATTERN" in src, "LITHOLOGY_PATTERN missing"
    assert "limestone" in src, "limestone keyword missing in lithology list"
    assert "chert" in src, "chert keyword missing in lithology list"
    assert "_BIOZONE_RE" in src, "_BIOZONE_RE missing"


def test_geology_extraction_classifies_paleo_vs_modern():
    """The coordinate classifier must distinguish paleo vs modern
    coords based on context keywords like ``"during the "`` /
    ``"at that time"`` vs ``"today"`` / ``"present-day"``."""
    src = _read("src/rlpe/geology_extraction.py")
    assert "_classify_coordinate_age" in src, (
        "_classify_coordinate_age helper missing"
    )
    assert "paleo_latitude" in src, "paleo_latitude must be populated"
    assert "modern_latitude" in src, "modern_latitude must be populated"
    assert "_PALEO_KEYWORDS" in src, "_PALEO_KEYWORDS missing"
    assert "_MODERN_KEYWORDS" in src, "_MODERN_KEYWORDS missing"


def test_geology_record_dataclass_has_new_fields():
    """``GeologyRecord`` must declare the new fields so the
    converter can read them off the producer's output."""
    src = _read("src/rlpe/geology_extraction.py")
    for f in ("group:", "member:", "lithology:", "country:", "biozone:",
              "modern_latitude:", "modern_longitude:",
              "paleo_latitude:", "paleo_longitude:"):
        assert f in src, f"GeologyRecord missing field {f!r}"


# ---------------------------------------------------------------------------
# B — Vision path enabled
# ---------------------------------------------------------------------------


def test_use_geo_vision_default_true_in_joboptions():
    """``JobOptions.use_geo_vision`` must default to True so web-UI
    users get all 25 geology fields without flipping a flag."""
    src = _read("src/rlpe/api/app.py")
    assert "use_geo_vision: bool = True" in src, (
        "JobOptions.use_geo_vision default must be True. The Round 18 "
        "enrichment is opt-in otherwise and most users never see the "
        "geology fields."
    )


def test_plate_figure_type_in_geo_vision_allowlist():
    """``plate`` must be in the default ``geo_vision_figure_types``
    allowlist so per-plate figures (the bulk of radiolarian papers)
    are extracted via the vision path."""
    src = _read("src/rlpe/pipeline.py")
    # Locate the default list and confirm "plate" is the first entry
    # (it should be — plate figures are the common case).
    idx = src.find('"plate",\n                "range_chart"')
    assert idx > 0, (
        "plate_geo must be in the default geo_vision_figure_types "
        "allowlist. Without this, vision is silently skipped for "
        "specimen plates."
    )


def test_geo_vision_options_propagate_to_extra():
    """``use_geo_vision`` and ``geo_vision_figure_types`` must be
    copied from JobOptions into ``cfg.extra`` so the pipeline
    actually reads them."""
    src = _read("src/rlpe/api/app.py")
    # The propagation block must list both keys.
    assert '"use_geo_vision"' in src, "use_geo_vision not propagated to extra"
    assert '"geo_vision_figure_types"' in src, (
        "geo_vision_figure_types not propagated to extra"
    )


# ---------------------------------------------------------------------------
# C — PBDB / GPlates-style paleo reconstruction
# ---------------------------------------------------------------------------


def test_paleo_reconstruction_module_exists():
    src = _read("src/rlpe/paleo_reconstruction.py")
    # Core public API
    for sym in (
        "def infer_plate_id",
        "def reconstruct_paleo_position",
        "def enrich_geology_record",
        "EULER_POLES",
        "COUNTRY_PLATE",
    ):
        assert sym in src, f"paleo_reconstruction missing {sym}"


def test_paleo_enrichment_wired_into_pipeline():
    """``_process_one_pdf`` must call ``enrich_geology_record`` for each
    result row so paleo_lat/lon + plate_id are populated after the
    regex / vision extractors finish."""
    src = _read("src/rlpe/pipeline.py")
    assert "enrich_geology_record" in src, (
        "Pipeline doesn't call enrich_geology_record. The C fix is "
        "staged in the module but never invoked."
    )
    # Must iterate over geology_links inside each result row's metadata.
    idx = src.find("for gl in md.get")
    assert idx > 0, (
        "Pipeline must iterate `for gl in md.get('geology_links', [])` "
        "to enrich each link in place."
    )


def test_euler_pole_tables_extend_to_250_ma():
    """Common plates (Adria, Iberia, Eurasia, etc.) must have poles
    at 200 Ma AND 250 Ma so Late Triassic (~226 Ma) age strings
    resolve to a paleo position. Otherwise the C fix is a no-op for
    the radiolarian era."""
    src = _read("src/rlpe/paleo_reconstruction.py")
    for plate in ("Adria", "Iberia", "Eurasia", "North China", "South China", "Africa", "Anatolia"):
        # Each plate's tuple list should have an entry whose age is
        # >= 250 Ma. Find the plate's table and check the last age.
        # Crude text-grep is enough for the source-guard.
        block_idx = src.find(f'"{plate}":')
        assert block_idx > 0, f"Plate {plate!r} not in EULER_POLES"
        # Find the closing ']' for this plate's list.
        end_idx = src.find("]", block_idx)
        block = src[block_idx:end_idx]
        # Look for a 250.0 entry in the plate's tuples.
        assert "250.0" in block or "250," in block, (
            f"Plate {plate!r} doesn't extend to 250 Ma. Late-Triassic "
            f"papers (~226 Ma) would fall outside the table and "
            f"return no paleo position."
        )


def test_paleo_reconstruction_does_not_crash_on_legacy_data():
    """``enrich_geology_record`` must be a no-op (no exception) when
    fields are missing — it's called on every result row in the
    pipeline. A single bad row mustn't blow up the whole job."""
    from rlpe.paleo_reconstruction import enrich_geology_record

    # All-empty record
    enrich_geology_record({})
    # Half-empty record
    enrich_geology_record({"latitude": None, "longitude": None})
    # Latitude only
    enrich_geology_record({"latitude": 38.0, "longitude": None})
    # All the rest is silently ignored
    assert True  # no exception


# ---------------------------------------------------------------------------
# D — evidence_text + paleo chips in modal
# ---------------------------------------------------------------------------


def test_frontend_modal_renders_evidence_collapsible():
    """``web/js/app.js`` must render ``g.evidence_text`` as a
    ``<details class="modal-geo-evidence">`` block so the operator
    can see WHICH sentence the extractor pulled the data from."""
    src = _read("web/js/app.js")
    assert "modal-geo-evidence" in src, (
        "Frontend missing .modal-geo-evidence collapsible block"
    )
    assert "evidence_text" in src, (
        "Frontend not referencing g.evidence_text from geology links"
    )
    # The summary label should be the 📄 icon + a Chinese phrase.
    assert "提取证据" in src, (
        "Evidence summary label missing the Chinese 📄 提取证据 marker"
    )


def test_frontend_modal_renders_paleo_and_modern_chips():
    """The modal must render the paleo coordinate chip with plate ID
    and reconstruction age, and the modern coordinate chip, when
    the data is present."""
    src = _read("web/js/app.js")
    assert "modal-geo-paleo" in src, "paleo chip class missing"
    assert "modal-geo-modern" in src, "modern chip class missing"
    assert "plate_id" in src, "plate_id not rendered"
    assert "reconstruction_age_ma" in src, "reconstruction_age_ma not rendered"


def test_frontend_modal_css_classes_match_js():
    """The CSS must define the modal-geo-paleo / modal-geo-modern /
    modal-geo-evidence classes referenced by the JS."""
    src = _read("web/css/style.css")
    assert ".modal-geo-paleo" in src, ".modal-geo-paleo CSS missing"
    assert ".modal-geo-modern" in src, ".modal-geo-modern CSS missing"
    assert ".modal-geo-evidence" in src, ".modal-geo-evidence CSS missing"
    # The evidence block must have a pre-formatted styling.
    assert "modal-geo-evidence pre" in src, (
        "CSS for the evidence <pre> block missing"
    )