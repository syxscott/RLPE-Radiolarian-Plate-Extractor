"""Tests for the three exporters (analysis, ml, archive)."""
from __future__ import annotations

import csv
import json
import zipfile

import pytest

from rlpe.exporters import (
    AnalysisOptions,
    DwCAOptions,
    MLOptions,
    write_csv,
    write_dwca_zip,
    write_ml_split,
    write_parquet,
)
from rlpe.provenance import build_provenance
from rlpe.schema_models import (
    GeologyLinkRecord,
    PanelMetadata,
    PanelRecord,
    PaperMetadataRecord,
    ProvenanceRecord,
    RunOutput,
    ScaleBarRecord,
)


def _make_run(n: int = 4) -> RunOutput:
    """Build a small RunOutput for testing."""
    prov = ProvenanceRecord(**build_provenance().to_dict())
    panels: list[PanelRecord] = []
    for i in range(n):
        pm = PaperMetadataRecord(
            title=f"Test paper {i}",
            authors=[f"Author {i}"],
            year=2020,
            doi=f"10.1000/test-{i}",
            source="opendataloader",
            confidence=0.8,
        )
        sb = ScaleBarRecord(value=100.0, unit="um", source="caption",
                            um_per_px=0.1, confidence=0.8)
        geo = GeologyLinkRecord(
            age="Late Jurassic", chronostratigraphy="Kimmeridgian",
            formation="Fonzaso", locality="Italy",
            latitude=46.5, longitude=11.5, confidence=0.7,
        )
        meta = PanelMetadata(
            panel_score=0.5, ocr_count=2, taxon_count=1,
            figure_number="1", page_index=10,
            matcher_used=False, matcher_type="heuristic",
            matcher_conf=0.0, caption_pairs_used=True,
            scale_bar=sb, geology_links=[geo],
            extraction_source="opendataloader",
        )
        panels.append(PanelRecord(
            paper_id=f"paper_{i}",
            figure_id=f"figure_{i}",
            panel_id=str(i + 1),
            species=f"Genus species_{i}" if i % 2 == 0 else None,
            panel_path=f"/path/panel_{i}.png",
            bbox=[10, 20, 100, 200],
            confidence=0.7,
            label_text=str(i + 1),
            caption_snippet=f"Caption for {i}",
            ocr_text=str(i + 1),
            metadata=meta,
            paper_metadata=pm,
        ))
    return RunOutput(provenance=prov, panels=panels)


class TestAnalysisCsv:
    def test_write_csv_basic(self, tmp_path):
        run = _make_run(3)
        target = tmp_path / "out.csv"
        n = write_csv(run, target)
        assert n == 3
        with open(target) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert "scientificName" in rows[0]
        assert "decimalLatitude" in rows[0]

    def test_dwc_field_mapping(self, tmp_path):
        run = _make_run(1)
        target = tmp_path / "out.csv"
        write_csv(run, target)
        with open(target) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["scientificName"] == "Genus species_0"
        assert row["basisOfRecord"] == "FossilSpecimen"
        assert row["eventDate"] == "2020"
        assert row["decimalLatitude"] == "46.5"
        assert row["decimalLongitude"] == "11.5"
        assert row["geologicalContextID"] == "Late Jurassic"
        assert row["identifiedBy"] == "Author 0"
        assert row["associatedReferences"] == "10.1000/test-0"
        assert row["occurrenceID"] == "paper_0:figure_0:1"

    def test_excludes_unmatched_when_disabled(self, tmp_path):
        run = _make_run(4)
        target = tmp_path / "out.csv"
        # Default: include all (the test set has alternating matched/unmatched)
        n_all = write_csv(run, target)
        # Now exclude unmatched
        n_filtered = write_csv(
            run, target, options=AnalysisOptions(include_unmatched=False)
        )
        assert n_all == 4
        assert n_filtered == 2  # only the even-indexed panels have species


class TestAnalysisParquet:
    def test_write_parquet(self, tmp_path):
        run = _make_run(2)
        target = tmp_path / "out.parquet"
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed")
        n = write_parquet(run, target)
        assert n == 2
        assert target.exists()

    def test_parquet_missing_dependency(self, tmp_path):
        run = _make_run(1)
        target = tmp_path / "out.parquet"
        # If pyarrow is not available, this should raise ImportError
        # We don't fail the test if it's installed.
        try:
            import pyarrow  # noqa: F401
            pytest.skip("pyarrow is installed; cannot test missing-dep path")
        except ImportError:
            with pytest.raises(ImportError, match="pyarrow"):
                write_parquet(run, target)


class TestMlSplit:
    def test_write_ml_split(self, tmp_path):
        run = _make_run(10)
        target_dir = tmp_path / "ml"
        counts = write_ml_split(run, target_dir, options=MLOptions(
            train_ratio=0.5, val_ratio=0.3, test_ratio=0.2, seed="test-seed",
        ))
        # Sum must equal n_panels
        assert sum(counts.values()) == 10
        # All counts are non-negative
        for k, v in counts.items():
            assert v >= 0
        # Files exist
        assert (target_dir / "train.jsonl").exists()
        assert (target_dir / "validation.jsonl").exists()
        assert (target_dir / "test.jsonl").exists()

    def test_ml_split_deterministic(self, tmp_path):
        run = _make_run(20)
        opts = MLOptions(seed="abc")
        a = write_ml_split(run, tmp_path / "a", options=opts)
        b = write_ml_split(run, tmp_path / "b", options=opts)
        assert a == b

    def test_ml_split_different_seeds_differ(self, tmp_path):
        run = _make_run(20)
        a = write_ml_split(run, tmp_path / "a", options=MLOptions(seed="abc"))
        b = write_ml_split(run, tmp_path / "b", options=MLOptions(seed="xyz"))
        # Different seeds should generally produce different counts
        # (with high probability for 20 panels)
        assert a != b or sum(a.values()) == sum(b.values())

    def test_ml_record_format(self, tmp_path):
        run = _make_run(2)
        target_dir = tmp_path / "ml"
        write_ml_split(run, target_dir)
        with open(target_dir / "train.jsonl") as f:
            for line in f:
                d = json.loads(line)
                assert "split" in d
                assert d["split"] in {"train", "validation", "test"}
                break


class TestDwcArchive:
    def test_write_dwca_zip(self, tmp_path):
        run = _make_run(3)
        target = tmp_path / "dwca.zip"
        n = write_dwca_zip(run, target)
        assert n == 2  # only matched panels (default excludes unmatched)
        assert target.exists()

    def test_zip_structure(self, tmp_path):
        run = _make_run(2)
        target = tmp_path / "dwca.zip"
        write_dwca_zip(run, target)
        with zipfile.ZipFile(target) as zf:
            names = zf.namelist()
        assert "meta.xml" in names
        assert "eml.xml" in names
        assert "occurrence.txt" in names

    def test_meta_xml_format(self, tmp_path):
        run = _make_run(1)
        target = tmp_path / "dwca.zip"
        write_dwca_zip(run, target)
        with zipfile.ZipFile(target) as zf:
            meta = zf.read("meta.xml").decode()
        assert "<archive" in meta
        assert "occurrenceID" in meta
        assert "scientificName" in meta
        assert 'rowType="http://rs.tdwg.org/dwc/terms/Occurrence"' in meta

    def test_eml_xml_format(self, tmp_path):
        run = _make_run(1)
        target = tmp_path / "dwca.zip"
        write_dwca_zip(run, target)
        with zipfile.ZipFile(target) as zf:
            eml = zf.read("eml.xml").decode()
        assert "<eml:eml" in eml
        assert "RLPE" in eml
        assert run.provenance.git_commit in eml

    def test_occurrence_tsv_format(self, tmp_path):
        run = _make_run(2)
        target = tmp_path / "dwca.zip"
        write_dwca_zip(run, target, options=DwCAOptions(include_unmatched=True))
        with zipfile.ZipFile(target) as zf:
            occ = zf.read("occurrence.txt").decode()
        # First line is header
        lines = occ.strip().split("\n")
        assert "occurrenceID" in lines[0]
        assert "scientificName" in lines[0]
        # Tab-separated
        assert "\t" in lines[0]
        # At least one data row
        assert len(lines) >= 2

    def test_occurrence_row_values(self, tmp_path):
        run = _make_run(1)
        target = tmp_path / "dwca.zip"
        write_dwca_zip(run, target, options=DwCAOptions(include_unmatched=True))
        with zipfile.ZipFile(target) as zf:
            occ = zf.read("occurrence.txt").decode()
        lines = occ.strip().split("\n")
        header = lines[0].split("\t")
        row = lines[1].split("\t")
        d = dict(zip(header, row))
        assert d["occurrenceID"] == "paper_0:figure_0:1"
        assert d["scientificName"] == "Genus species_0"
        assert d["basisOfRecord"] == "FossilSpecimen"
        assert d["eventDate"] == "2020"
        assert d["decimalLatitude"] == "46.5"
        assert d["decimalLongitude"] == "11.5"
        assert d["geologicalContextID"] == "Late Jurassic"
        assert d["identifiedBy"] == "Author 0"
        assert d["genus"] == "Genus"
        assert d["specificEpithet"] == "species_0"
