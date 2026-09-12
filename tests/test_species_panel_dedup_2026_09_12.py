"""2026-09-12: same species + identical geology linked to many panel
images collapses into ONE table row — first image stays the
association, the rest move to metadata.additional_panel_paths (files
are untouched on disk)."""

from __future__ import annotations

from rlpe.pipeline import RadiolarianPipeline


def _row(species: str | None, path: str | None, panel_id: str = "1", age: str | None = "Lower Permian") -> dict:
    md: dict = {"figure_type": "plate"}
    if age:
        md["geology_summary"] = {"age": age, "locality": "Kondurovka"}
    return {
        "paper_id": "pap1",
        "figure_id": "fig1",
        "panel_id": panel_id,
        "species": species,
        "panel_path": path,
        "bbox": None,
        "confidence": 0.7,
        "label_text": panel_id,
        "caption_snippet": "Plate 3",
        "ocr_text": None,
        "paper_metadata": None,
        "metadata": md,
    }


def _dedup(rows):
    return RadiolarianPipeline._dedup_species_panels(rows)


class TestSpeciesPanelDedup:
    def test_same_species_same_geology_merges(self):
        rows = [
            _row("Glomeropyle algidum", "/p/panel_12.png", "12"),
            _row("Glomeropyle algidum", "/p/panel_13.png", "13"),
            _row("Glomeropyle algidum", "/p/panel_14.png", "14"),
        ]
        out = _dedup(rows)
        assert len(out) == 1
        md = out[0]["metadata"]
        assert out[0]["panel_path"] == "/p/panel_12.png"
        assert md["additional_panel_paths"] == ["/p/panel_13.png", "/p/panel_14.png"]
        assert md["additional_panel_ids"] == ["13", "14"]

    def test_different_geology_not_merged(self):
        rows = [
            _row("Sp. a", "/p/1.png", age="Asselian"),
            _row("Sp. a", "/p/2.png", age="Sakmarian"),
        ]
        assert len(_dedup(rows)) == 2

    def test_different_species_not_merged(self):
        rows = [
            _row("Sp. a", "/p/1.png"),
            _row("Sp. b", "/p/2.png"),
        ]
        assert len(_dedup(rows)) == 2

    def test_no_species_rows_pass_through(self):
        rows = [_row(None, "/p/1.png"), _row(None, "/p/2.png")]
        assert len(_dedup(rows)) == 2

    def test_rows_without_panel_image_pass_through(self):
        """A caption-only row (no panel image) is NOT a specimen image —
        it passes through unmerged, and the image-bearing row for the
        same species stays a separate record (nothing to preserve on
        the imageless row, so merging would only blur provenance)."""
        rows = [
            _row("Sp. a", None, "1"),
            _row("Sp. a", "/p/2.png", "2"),
        ]
        out = _dedup(rows)
        assert len(out) == 2
        assert not any(
            (r["metadata"] or {}).get("additional_panel_paths") for r in out
        )

    def test_stub_rows_untouched(self):
        stub = {
            "paper_id": "pap1",
            "figure_id": "_ingestion_od_failed",
            "panel_id": None,
            "species": None,
            "panel_path": None,
            "metadata": {},
        }
        assert _dedup([stub]) == [stub]

    def test_review_flags_survive_merge(self):
        r2 = _row("Sp. a", "/p/2.png", "2")
        r2["metadata"]["needs_review"] = True
        r2["metadata"]["review_reasons"] = ["low_confidence"]
        out = _dedup([_row("Sp. a", "/p/1.png", "1"), r2])
        md = out[0]["metadata"]
        assert md.get("needs_review") is True
        assert "low_confidence" in md.get("review_reasons", [])
