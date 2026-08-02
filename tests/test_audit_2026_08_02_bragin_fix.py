"""Regression tests for the Bragin 2025 figure-id schema variant."""

from __future__ import annotations

from rlpe.evaluation import GoldPanel, evaluate
from rlpe.evaluation.metrics import _figure_id_logical_key


PAPER = "bragin2025"
GOLD_FIGURE = "od_plate_bragin2025_p001_pl01"
PRED_FIGURE = "od_plate_2e85364a3c605326_p006_pl01"


def _pred(panel_id: str, figure_id: str = PRED_FIGURE) -> dict:
    return {
        "paper_id": PAPER,
        "figure_id": figure_id,
        "panel_id": panel_id,
        "species": "Pantanellium moscowiense",
        "metadata": {},
    }


def test_bragin_gold_and_raw_prediction_keys_are_logically_equal():
    assert _figure_id_logical_key(GOLD_FIGURE) == "bragin2025_pl01"
    assert _figure_id_logical_key(PRED_FIGURE) == "bragin2025_pl01"


def test_bragin_schema_variant_matches_panel_and_species():
    gold = [GoldPanel(PAPER, GOLD_FIGURE, "11", "Pantanellium moscowiense")]
    report = evaluate([_pred("11")], gold)
    metrics = report.papers[PAPER]
    assert metrics.panel_match == 1
    assert metrics.panel_match_rate == 1.0
    assert metrics.species_tp == 1


def test_bragin_plate_number_remains_distinct():
    assert _figure_id_logical_key(
        "od_plate_2e85364a3c605326_p007_pl02"
    ) == "bragin2025_pl02"
    assert _figure_id_logical_key(
        "od_plate_2e85364a3c605326_p007_pl02"
    ) != _figure_id_logical_key(PRED_FIGURE)


def test_non_bragin_hash_keeps_page_guard():
    # Do not broaden the Bragin exception to every OD document hash.
    assert _figure_id_logical_key(
        "od_plate_deadbeefdeadbeef_p006_pl01"
    ) == "deadbeefdeadbeef_p006"


def test_bragin_wrong_panel_does_not_match():
    gold = [GoldPanel(PAPER, GOLD_FIGURE, "1", "Genus species")]
    report = evaluate([_pred("2")], gold)
    assert report.papers[PAPER].panel_match == 0
