"""Phase 64 Plan B Task 1: 4 new figure types in ``classify_figure_type``.

This locks the keyword-matching contract for schematic / diagram /
reconstruction / phylogenetic figure types so callers can rely on
the routing decisions. The earlier classifier only handled plate,
range_chart, map, strat_column, litholog_column, paleogeographic_map,
photo. Phase 64 Plan B adds four conceptual figure types.

Detection order (set in range_chart_extractor.classify_figure_type):

  1. plate (with range_chart override)
  2. paleogeographic_map (before map)
  3. map (before range_chart)
  4. strat_column / litholog_column (before range_chart)
  5. range_chart
  6. photo
  7. phylogenetic (most specific of the 4 new types)
  8. schematic / diagram / reconstruction
  9. other

The ``paleogeographic reconstruction`` / ``palaeogeographic
reconstruction`` keywords remain in the paleogeographic_map list so a
"paleogeographic reconstruction" caption still routes to the existing
paleogeographic-map vision path (not to the plain "reconstruction"
type). This mirrors the historic ``paleogeographic_map`` vs ``map``
ordering.

Each new type has at least 2 keyword-matching test cases covering
single-word and multi-word forms so the keyword lists don't silently
shrink during future edits.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rlpe.range_chart_extractor import classify_figure_type


class TestSchematicClassifier:
    """Phase 64 Plan B: 4 new figure types in classify_figure_type."""

    @pytest.mark.parametrize(
        "caption",
        [
            "Figure 5. Schematic of the paleoceanographic model.",
            "Schematic diagram showing ocean circulation pathways.",
            "Conceptual diagram of the radiolarian evolution model.",
            "Schematic reconstruction of late Jurassic ocean currents.",
        ],
    )
    def test_schematic_classification(self, caption: str) -> None:
        assert classify_figure_type(caption) == "schematic"

    @pytest.mark.parametrize(
        "caption",
        [
            "Figure 2. Diagram of radiolarian skeletal growth stages.",
            "Block diagram illustrating lithospheric movements.",
            "Flow diagram of the radiolarian extraction workflow.",
        ],
    )
    def test_diagram_classification(self, caption: str) -> None:
        assert classify_figure_type(caption) == "diagram"

    @pytest.mark.parametrize(
        "caption",
        [
            "Figure 3. Reconstruction of the Tethys Ocean at 100 Ma.",
            "Artistic reconstruction of a radiolarian swarm in life.",
            "Life reconstruction of Jurassic nassellarian radiolarians.",
        ],
    )
    def test_reconstruction_classification(self, caption: str) -> None:
        assert classify_figure_type(caption) == "reconstruction"

    @pytest.mark.parametrize(
        "caption",
        [
            "Figure 6. Phylogenetic tree of Nassellaria.",
            "Cladogram showing the evolutionary relationships of Cenozoic radiolarians.",
            "Phylogeny of Spumellaria based on ribosomal DNA.",
            "Evolutionary tree of Mesozoic radiolarian families.",
            "Cladistic analysis of late Paleozoic radiolarian genera.",
        ],
    )
    def test_phylogenetic_classification(self, caption: str) -> None:
        assert classify_figure_type(caption) == "phylogenetic"

    def test_paleogeographic_reconstruction_routes_to_paleogeographic_map(self) -> None:
        """The plain ``reconstruction`` keywords must NOT capture
        paleogeographic-reconstruction captions — those go to the
        existing paleogeographic_map vision path."""
        assert (
            classify_figure_type(
                "Paleogeographic reconstruction of Pangea at 250 Ma."
            )
            == "paleogeographic_map"
        )
        assert (
            classify_figure_type(
                "Palaeogeographic reconstruction of the Tethyan margin."
            )
            == "paleogeographic_map"
        )

    def test_schematic_does_not_collide_with_plate(self) -> None:
        """A "schematic diagram" caption must NOT be classified as
        ``plate`` even though plates can include diagram-like inset
        figures. We accept either ``schematic`` or ``diagram`` here
        (the classifier tries schematic first, then diagram)."""
        result = classify_figure_type("Figure 1. Schematic diagram of the extraction flow.")
        assert result in {"schematic", "diagram"}

    def test_phylogenetic_beats_diagram_when_both_match(self) -> None:
        """"A phylogenetic cladogram" must classify as phylogenetic,
        not diagram — phylogenetic is the more specific keyword."""
        assert (
            classify_figure_type(
                "Figure 7. Phylogenetic cladogram of Mesozoic nassellarians."
            )
            == "phylogenetic"
        )

    def test_classifier_preserves_legacy_types(self) -> None:
        """Adding the 4 new types must NOT silently break the
        legacy 7 types. Lock the existing behavior so a future
        edit to the new keyword lists doesn't accidentally
        re-order the existing routing."""
        assert (
            classify_figure_type(
                "Plate 1. figs 1-5. Actinomma leptodermum"
            )
            == "plate"
        )
        assert (
            classify_figure_type(
                "Stratigraphic distribution of radiolarians from the Late Permian of South China."
            )
            == "range_chart"
        )
        assert (
            classify_figure_type("Location map of studied sections in Tunisia.")
            == "map"
        )

    def test_captionless_returns_other(self) -> None:
        """A None / empty caption still falls back to ``other``."""
        assert classify_figure_type(None) == "other"
        assert classify_figure_type("") == "other"
