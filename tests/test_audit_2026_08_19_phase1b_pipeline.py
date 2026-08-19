"""Regression tests for audit 2026-08-19 Phase 1B pipeline fixes.

Covers three BLOCKER bugs found by the multi-agent audit:

B-1 — ``_apply_geo_vision`` allowlist typo
    Default allowlist contained ``"stratigraphic_column"`` but
    ``classify_figure_type`` returns ``"strat_column"``. GROBID-path
    strat_column figures were silently skipped.

B-5 — ``_recover_bboxes_via_segmentation`` duplicate panel_id
    Old code keyed a lookup dict by panel_id; duplicate panel_ids all
    shared the LAST row's bbox. Fixed to preserve each row's own
    bbox by walking sorted-position → original-position via a
    queue of original indices per panel_id.

MAJOR-1 — ``_exit_od_grobid_guard`` default symmetry
    Enter defaults to 0; exit used to default to 1, so a fresh
    thread-local that never called enter would set depth = 1 - 1 = 0
    by accident, masking real bugs. Both ends now default to 0.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.pipeline import RadiolarianPipeline
from rlpe.types import PanelCandidate


_PIPELINE_PATH = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"


# ---------------------------------------------------------------------------
# B-1 — ``_apply_geo_vision`` allowlist typo
# ---------------------------------------------------------------------------


class TestApplyGeoVisionAllowlist:
    """B-1: default allowlist must use ``"strat_column"`` (the actual
    value returned by ``classify_figure_type``), NOT the misspelled
    ``"stratigraphic_column"``.
    """

    def _extract_default_allowlist(self) -> str:
        """Return the slice of pipeline.py covering the
        ``geo_vision_figure_types`` default-list literal so each
        assertion only looks inside the right scope."""
        text = _PIPELINE_PATH.read_text(encoding="utf-8")
        # Locate the ``allowed: list[str] = list(...)`` block.
        marker = 'allowed: list[str] = list('
        i = text.find(marker)
        assert i > 0, "could not locate _apply_geo_vision allowlist literal"
        # End at the matching closing paren of the outer ``list(...)`` call.
        # The block is short (~10 lines); 800 chars is more than enough.
        end = text.find("\n        )", i)
        assert end > 0, "could not find end of allowlist literal"
        return text[i:end]

    def test_default_allowlist_uses_strat_column(self):
        """The misspelled ``stratigraphic_column`` would silently
        filter out every strat-column figure on the GROBID path."""
        block = self._extract_default_allowlist()
        assert '"strat_column"' in block, (
            "_apply_geo_vision default allowlist must include "
            "'strat_column' (the actual value classify_figure_type returns). "
            "Misspelled 'stratigraphic_column' would silently skip every "
            "strat-column figure on the GROBID path."
        )

    def test_default_allowlist_does_not_contain_misspelled_key(self):
        """Guard: the typo must not silently come back."""
        block = self._extract_default_allowlist()
        assert '"stratigraphic_column"' not in block, (
            "_apply_geo_vision default allowlist still contains the "
            "misspelled 'stratigraphic_column' — typo from audit B-1 "
            "has regressed."
        )

    def test_default_allowlist_includes_other_expected_types(self):
        """The default must keep the other figure types that were
        already correctly listed."""
        block = self._extract_default_allowlist()
        for expected in (
            '"plate"',
            '"range_chart"',
            '"litholog_column"',
            '"paleogeographic_map"',
        ):
            assert expected in block, (
                f"default allowlist missing expected type {expected!r}"
            )


# ---------------------------------------------------------------------------
# B-5 — ``_recover_bboxes_via_segmentation`` duplicate panel_id
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_path: Path) -> RadiolarianPipeline:
    from rlpe.config import PipelineConfig

    cfg = PipelineConfig(
        pdf_dir=tmp_path,
        work_dir=tmp_path,
        save_intermediate=False,
        min_panel_score=0.0,
    )
    return RadiolarianPipeline(cfg)


def _build_plate(w: int = 768, h: int = 256) -> np.ndarray:
    """Wide plate to host three side-by-side 256x256 panels."""
    return np.full((h, w, 3), 240, dtype=np.uint8)


class TestRecoverBboxesDuplicatePanelId:
    """B-5: duplicate panel_id rows must each get their own bbox.

    Before the fix the helper built
    ``{r.get("panel_id"): r for r in sorted_results}`` so any duplicate
    panel_id collapsed to the LAST sorted row. Then ``id()`` index
    lookup gave every duplicate the same ``idx`` → every duplicate got
    the same segmented bbox.
    """

    def test_duplicate_panel_ids_get_distinct_bboxes(self, tmp_path):
        """Three rows with panel_id="1" must each be paired with a
        different segmented bbox (positions 0, 1, 2), not all three
        sharing the last bbox."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = MagicMock()
        pipe.segmenter._segment_with_opencv = MagicMock(
            return_value=[
                PanelCandidate(panel_id="seg0", bbox=(10, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg1", bbox=(200, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg2", bbox=(400, 10, 60, 80), score=0.9),
            ]
        )

        plate = _build_plate()
        # Three rows that all share panel_id="1". This is the
        # pathological case the old dict-keyed lookup could not handle.
        results = [
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
        ]

        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="pl01"
        )

        # Each row's bbox must equal one of the three segmented bboxes.
        expected = [
            [10, 10, 60, 80],
            [200, 10, 60, 80],
            [400, 10, 60, 80],
        ]
        assigned = [r["bbox"] for r in out]
        assert assigned == expected, (
            f"duplicate panel_id rows must each get a distinct bbox. "
            f"Old code made all three share the last bbox. Got {assigned}"
        )

        # Each row's panel_id_source must be stamped (none skipped).
        for r in out:
            assert r["metadata"]["panel_id_source"] == "phase67_segmentation_recovery"

    def test_mixed_unique_and_duplicate_panel_ids(self, tmp_path):
        """Mixed case: one unique row + two duplicates. Each row gets
        a distinct bbox in reading order, no shared bboxes."""
        pipe = _make_pipeline(tmp_path)
        pipe.segmenter = MagicMock()
        pipe.segmenter._segment_with_opencv = MagicMock(
            return_value=[
                PanelCandidate(panel_id="seg0", bbox=(10, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg1", bbox=(200, 10, 60, 80), score=0.9),
                PanelCandidate(panel_id="seg2", bbox=(400, 10, 60, 80), score=0.9),
            ]
        )

        plate = _build_plate()
        results = [
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "2", "bbox": None, "panel_path": None, "metadata": {}},
            {"panel_id": "1", "bbox": None, "panel_path": None, "metadata": {}},
        ]

        out = pipe._recover_bboxes_via_segmentation(
            results, plate, paper_id="p1", figure_id="pl01"
        )

        # Sort puts the two "1"s first (sorted by panel_id numeric
        # value): sorted_results = [orig_0("1"), orig_2("1"), orig_1("2")].
        # Pairing by sorted-position → segmented-position:
        #   sorted_idx 0 (orig_0, "1") → seg0
        #   sorted_idx 1 (orig_2, "1") → seg1
        #   sorted_idx 2 (orig_1, "2") → seg2
        expected = [
            [10, 10, 60, 80],   # orig_0  ("1") gets seg0
            [400, 10, 60, 80],  # orig_1  ("2") gets seg2
            [200, 10, 60, 80],  # orig_2  ("1") gets seg1
        ]
        assigned = [r["bbox"] for r in out]
        assert assigned == expected, (
            f"mixed unique+duplicate panel_ids must each get a distinct "
            f"bbox. Got {assigned}"
        )


# ---------------------------------------------------------------------------
# MAJOR-1 — ``_exit_od_grobid_guard`` default symmetry
# ---------------------------------------------------------------------------


class TestOdGrobidGuardDefaultSymmetry:
    """MAJOR-1: ``_exit_od_grobid_guard`` must default to 0 (matching
    ``_enter_od_grobid_guard``), not 1.

    With the old default of 1, ``_exit`` on a fresh thread-local (never
    entered) would compute ``max(1 - 1, 0) = 0`` — same answer — but a
    thread that entered once and exited twice would compute
    ``max(1 - 1, 0) = 0`` instead of ``max(0 - 1, 0) = 0`` which is
    silently identical only because of the floor. The bug is structural:
    any future change to ``enter``'s initial value would break the
    invariant. Pin the symmetry.
    """

    @pytest.fixture
    def pipe(self, tmp_path):
        from unittest.mock import patch

        from rlpe.config import PipelineConfig

        with (
            patch("rlpe.pipeline.GrobidClient"),
            patch("rlpe.pipeline.OCRBackend"),
            patch("rlpe.pipeline.TaxonRecognizer"),
            patch("rlpe.pipeline.PanelSegmenter"),
        ):
            cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "work")
            return RadiolarianPipeline(cfg)

    def test_enter_and_exit_default_to_zero(self):
        """Source guard: both ``enter`` and ``exit`` must default to 0
        so a fresh thread-local starts and ends at depth 0."""
        text = _PIPELINE_PATH.read_text(encoding="utf-8")
        # Enter default
        assert 'getattr(self._od_grobid_depth, "depth", 0)' in text, (
            "_enter_od_grobid_guard must default to 0"
        )
        # Exit default — find the _exit function specifically.
        marker = "def _exit_od_grobid_guard"
        i = text.find(marker)
        assert i > 0
        body = text[i : i + 400]
        assert 'getattr(self._od_grobid_depth, "depth", 0)' in body, (
            "_exit_od_grobid_guard must default to 0 (matching _enter). "
            "Audit MAJOR-1: asymmetric default (1) hides depth bugs."
        )
        # And the old buggy default of 1 must NOT be present in exit.
        assert 'getattr(self._od_grobid_depth, "depth", 1)' not in body, (
            "_exit_od_grobid_guard still has the asymmetric default of 1"
        )

    def test_exit_before_enter_is_noop(self, pipe):
        """Calling _exit on a fresh thread-local (never entered) must
        leave depth at 0, not silently go negative or be clamped weird.

        The old buggy default ``getattr(..., "depth", 1)`` made this
        work only because ``max(1 - 1, 0) = 0`` — but that's incidental.
        With both ends defaulting to 0, the test pins the structural
        invariant.
        """
        # Fresh thread-local — depth attribute does not exist yet.
        # _exit must not raise and must leave depth at 0.
        pipe._exit_od_grobid_guard()
        assert getattr(pipe._od_grobid_depth, "depth", 0) == 0, (
            "_exit_od_grobid_guard on a fresh thread-local must leave "
            "depth at 0"
        )

    def test_enter_exit_cycle_balances(self, pipe):
        """A single enter→exit pair must return depth to 0, allowing a
        subsequent enter to succeed (i.e. the recursion budget is
        restored)."""
        assert pipe._enter_od_grobid_guard("p1", "OD") is True
        assert getattr(pipe._od_grobid_depth, "depth", 0) == 1
        pipe._exit_od_grobid_guard()
        assert getattr(pipe._od_grobid_depth, "depth", 0) == 0, (
            "enter→exit must return depth to 0"
        )

    def test_enter_exit_loop_accumulates_and_releases(self, pipe):
        """Multiple enter→exit cycles must accumulate and release
        depth cleanly without leaking.

        Note: the guard refuses the 3rd nested enter (depth >= 3
        returns False) — so each cycle does 2 enters + 2 exits to stay
        within the budget.
        """
        for _ in range(5):
            for _ in range(2):
                assert pipe._enter_od_grobid_guard("p1", "OD") is True
            assert getattr(pipe._od_grobid_depth, "depth", 0) == 2
            for _ in range(2):
                pipe._exit_od_grobid_guard()
            assert getattr(pipe._od_grobid_depth, "depth", 0) == 0, (
                "depth must return to 0 after a balanced enter→exit loop"
            )

    def test_depth_never_goes_negative(self, pipe):
        """Exiting more than entering must clamp to 0, never go negative."""
        for _ in range(5):
            pipe._exit_od_grobid_guard()
        depth = getattr(pipe._od_grobid_depth, "depth", 0)
        assert depth == 0, f"depth must clamp to 0, got {depth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
