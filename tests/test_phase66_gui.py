"""Phase 66 Plan C.6 — GUI shows cross_figure_visual_links tests.

PySide6 is required to import ``rlpe.gui.results_tab``. When running
in an env without Qt (the worktree venv), we skip with a clear
message; full source-guard tests run in the conda env that ships
PySide6.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

try:
    import PySide6  # noqa: F401
    _HAS_PYSIDE6 = True
except ImportError:
    _HAS_PYSIDE6 = False

# Source-guard fallback so the contract is also pinned in envs without
# PySide6 installed.
_SRC_RESULTS_TAB = (
    Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / "results_tab.py"
)
_SRC_STRINGS_EN = (
    Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / "strings_en.py"
)
_SRC_STRINGS_ZH = (
    Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / "strings_zh_CN.py"
)


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
class TestVisualLinksGuiRender:
    def test_render_detail_emits_visual_section(self):
        """When a panel has cross_figure_visual_links, _render_detail
        must include a '视觉坐标关联' / 'Visual coordinate links' section
        listing each link's target figure, layer, age, formation, and
        confidence."""
        from rlpe.gui.results_tab import _render_detail

        md = {
            "link_source": "locality_match",
            "link_confidence": 0.7,
            "cross_figure_visual_links": [
                {
                    "target_figure_id": "strat1",
                    "target_layer": 3,
                    "target_age": "Late Triassic",
                    "target_formation": "Scaglia Fm",
                    "confidence": 0.9,
                    "source": "m3_visual",
                }
            ],
        }
        html = _render_detail(panel={"metadata": md}, i18n_keys={})
        assert any("Visual" in s or "视觉" in s for s in html), (
            "expected a visual-coordinate link section in the rendered detail; "
            f"got: {html}"
        )
        assert any("strat1" in s for s in html)
        assert any("Late Triassic" in s for s in html)


class TestVisualLinksSourceGuard:
    """Even without PySide6, the GUI source must contain the Phase C
    visual-coordinates section so the operator's manual UI walkthrough
    succeeds. These tests scan the source files for the strings."""

    def test_results_tab_mentions_visual_section(self):
        if not _SRC_RESULTS_TAB.exists():
            pytest.skip("results_tab.py not found")
        text = _SRC_RESULTS_TAB.read_text(encoding="utf-8")
        assert "cross_figure_visual_links" in text

    def test_results_tab_visual_section_uses_i18n_key(self):
        if not _SRC_RESULTS_TAB.exists():
            pytest.skip("results_tab.py not found")
        text = _SRC_RESULTS_TAB.read_text(encoding="utf-8")
        # The visual-coordinates section should pull its label from the
        # i18n module so it shows in EN + ZH.
        assert "restab.detail.visual_links" in text

    def test_strings_en_has_visual_key(self):
        if not _SRC_STRINGS_EN.exists():
            pytest.skip("strings_en.py not found")
        text = _SRC_STRINGS_EN.read_text(encoding="utf-8")
        assert "restab.detail.visual_links" in text

    def test_strings_zh_has_visual_key(self):
        if not _SRC_STRINGS_ZH.exists():
            pytest.skip("strings_zh_CN.py not found")
        text = _SRC_STRINGS_ZH.read_text(encoding="utf-8")
        assert "restab.detail.visual_links" in text


if __name__ == "__main__":
    pytest.main([__file__, "-q"])