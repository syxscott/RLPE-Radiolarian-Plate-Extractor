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
        # audit 2026-07-31: _render_detail is a ResultsTab method, not
        # a module function; the test env now has PySide6 so construct
        # a real instance.
        from PySide6.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication([])
        from rlpe.gui.results_tab import ResultsTab
        _rt = ResultsTab()

        class _FakeBrowser:
            def __init__(self):
                self.last_html = ""

            def setHtml(self, html):
                self.last_html = html

        _rt._detail_browser = _FakeBrowser()

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
        _rt._render_detail({"metadata": md})
        html = _rt._detail_browser.last_html
        assert ("Visual" in html or "视觉" in html), (
            "expected a visual-coordinate link section in the rendered detail; "
            f"got: {html[:300]}"
        )
        assert "strat1" in html
        assert "Late Triassic" in html


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