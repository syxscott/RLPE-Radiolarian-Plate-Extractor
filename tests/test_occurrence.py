import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from occurrence import add_occurrence_groups, occurrence_group_id


def test_id_starts_with_occ_10_chars():
    gid = occurrence_group_id("bandini2011", "Williriedellum carpathicum")
    assert gid.startswith("occ_")
    assert len(gid) == len("occ_") + 6


def test_id_stable_across_calls():
    a = occurrence_group_id("bandini2011", "Williriedellum carpathicum")
    b = occurrence_group_id("bandini2011", "Williriedellum carpathicum")
    assert a == b


def test_id_normalizes_species_cf_aff():
    a = occurrence_group_id("p1", "Genus cf. species")
    b = occurrence_group_id("p1", "Genus species")
    c = occurrence_group_id("p1", "Genus aff. species")
    assert a == b
    assert b == c


def test_id_different_paper_different_group():
    a = occurrence_group_id("p1", "Genus species")
    b = occurrence_group_id("p2", "Genus species")
    assert a != b


def test_id_different_species_different_group():
    a = occurrence_group_id("p1", "Genus species")
    b = occurrence_group_id("p1", "Other species")
    assert a != b


def test_id_handles_empty_inputs():
    a = occurrence_group_id("", "Genus species")
    b = occurrence_group_id(None, "Genus species")
    assert a == b
    c = occurrence_group_id("p1", None)
    d = occurrence_group_id("p1", "")
    assert c == d
    e = occurrence_group_id("p1", "   ")
    assert c == e


def test_add_occurrence_groups_preserves_rows():
    preds = [
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "1",
            "species": "A sp",
            "confidence": 0.9,
        },
        {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "2",
            "species": "A sp",
            "confidence": 0.8,
        },
        {
            "paper_id": "p1",
            "figure_id": "f2",
            "panel_id": "1",
            "species": "B sp",
            "confidence": 0.7,
        },
    ]
    out = add_occurrence_groups(preds)
    assert len(out) == 3
    assert all("occurrence_group_id" in r for r in out)
    assert out[0]["occurrence_group_id"] == out[1]["occurrence_group_id"]
    assert out[2]["occurrence_group_id"] != out[0]["occurrence_group_id"]


def test_add_occurrence_groups_handles_missing_fields():
    preds = [
        {"paper_id": "", "species": "A sp"},
        {"paper_id": "p1", "species": None},
    ]
    out = add_occurrence_groups(preds)
    assert all("occurrence_group_id" in r for r in out)
