"""Phase 65 Plan A.6 — GUI linker badge tests.

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
_SRC_RESULTS_TAB = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / "results_tab.py"
_SRC_STRINGS_EN = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / "strings_en.py"
_SRC_STRINGS_ZH = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "gui" / "strings_zh_CN.py"


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
class TestLinkSourceBadge:
    def test_sample_match_emits_chip(self):
        from rlpe.gui.results_tab import _emit_link_source_badge

        html: list[str] = []
        _emit_link_source_badge(html, "cross_figure_linker:sample_match")
        assert any("link: " in s and ("Sample ID match" in s or "样品号匹配" in s) for s in html)

    def test_locality_match_emits_chip(self):
        from rlpe.gui.results_tab import _emit_link_source_badge

        html: list[str] = []
        _emit_link_source_badge(html, "cross_figure_linker:locality_match")
        assert any("Locality match" in s or "产地匹配" in s for s in html)

    def test_m3_emits_amber_chip(self):
        from rlpe.gui.results_tab import _emit_link_source_badge

        html: list[str] = []
        _emit_link_source_badge(html, "cross_figure_linker:m3_inference")
        assert any("M3 inference" in s or "M3 推理" in s for s in html)
        # Amber chip class
        assert any("badge-warn" in s for s in html)

    def test_unlinked_emits_grey_chip(self):
        from rlpe.gui.results_tab import _emit_link_source_badge

        html: list[str] = []
        _emit_link_source_badge(html, "cross_figure_linker:unlinked")
        assert any("Unlinked" in s or "未关联" in s for s in html)
        assert any("badge-muted" in s for s in html)

    def test_non_linker_no_chip(self):
        from rlpe.gui.results_tab import _emit_link_source_badge

        html: list[str] = []
        _emit_link_source_badge(html, "geo_vision")
        assert html == []

    def test_empty_no_chip(self):
        from rlpe.gui.results_tab import _emit_link_source_badge

        html: list[str] = []
        _emit_link_source_badge(html, "")
        assert html == []


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
class TestLinkSummaryBadge:
    def test_sample_match_with_conf(self):
        from rlpe.gui.results_tab import _emit_link_summary_badge

        html: list[str] = []
        _emit_link_summary_badge(html, "sample_match", 1.0)
        assert any(("Sample ID match" in s or "样品号匹配" in s) and "100%" in s for s in html)

    def test_unlinked_zero_conf(self):
        from rlpe.gui.results_tab import _emit_link_summary_badge

        html: list[str] = []
        _emit_link_summary_badge(html, "unlinked", 0.0)
        assert any(("Unlinked" in s or "未关联" in s) and "0%" in s for s in html)

    def test_unknown_source_uses_raw_label(self):
        from rlpe.gui.results_tab import _emit_link_summary_badge

        html: list[str] = []
        _emit_link_summary_badge(html, "future_strategy", 0.5)
        assert any("future_strategy" in s for s in html)


@pytest.mark.skipif(not _HAS_PYSIDE6, reason="PySide6 not installed")
class TestRenderDetailIntegration:
    """End-to-end smoke test: run _render_detail on a row that has a
    linker link and verify the badge appears in the HTML."""

    def test_render_includes_link_source(self):
        from PySide6.QtWidgets import QApplication

        # Ensure a QApplication exists (PySide6 requires one).
        app = QApplication.instance() or QApplication([])

        from rlpe.gui.results_tab import ResultsTab

        class FakeBrowser:
            def __init__(self):
                self.last_html = ""

            def setHtml(self, html):
                self.last_html = html

        row = {
            "paper_id": "p1",
            "figure_id": "f1",
            "panel_id": "p1",
            "species": "Genus species",
            "bbox": None,
            "confidence": 0.5,
            "needs_review": False,
            "review_reasons": [],
            "caption_snippet": "All from Sample S1",
            "metadata": {
                "link_source": "sample_match",
                "link_confidence": 1.0,
                "link_figure_id": "strat1",
                "geology_links": [
                    {
                        "age": "Late Cretaceous",
                        "formation": "Scaglia",
                        "locality": "Italy",
                        "confidence": 0.95,
                        "section_type": "cross_figure_link",
                        "coord_source": "cross_figure_linker:sample_match",
                    }
                ],
            },
        }

        # audit 2026-07-31: __new__ bypassing __init__ raises in
        # PySide6 6.11 ("base class __init__ not called"). Construct
        # for real (offscreen) and swap the browser.
        from PySide6.QtWidgets import QApplication

        _app = QApplication.instance() or QApplication([])
        rt = ResultsTab()
        rt._detail_browser = FakeBrowser()
        rt._render_detail(row)
        html = rt._detail_browser.last_html
        # audit 2026-07-31: accept either language (the test env's
        # default language is zh_CN; the string table renders the
        # equivalent Chinese labels)
        assert "Sample ID match" in html or "样品号匹配" in html, html
        assert "Cross-figure link" in html or "跨图关联" in html, html


@pytest.mark.skipif(_HAS_PYSIDE6, reason="source-guard only")
class TestSourceGuard:
    """Source-level assertions for when PySide6 isn't installed."""

    def test_results_tab_uses_link_helpers(self):
        src = _SRC_RESULTS_TAB.read_text(encoding="utf-8")
        assert "_emit_link_source_badge" in src
        assert "_emit_link_summary_badge" in src
        assert "Cross-figure link" in src
        # Per-link chip pattern (the cross_figure_linker: prefix)
        assert "cross_figure_linker:" in src

    def test_strings_en_has_link_source_keys(self):
        src = _SRC_STRINGS_EN.read_text(encoding="utf-8")
        for key in (
            "restab.detail.link_source.sample_match",
            "restab.detail.link_source.locality_match",
            "restab.detail.link_source.m3_inference",
            "restab.detail.link_source.unlinked",
        ):
            assert key in src, f"missing i18n key: {key}"

    def test_strings_zh_has_link_source_keys(self):
        src = _SRC_STRINGS_ZH.read_text(encoding="utf-8")
        for key in (
            "restab.detail.link_source.sample_match",
            "restab.detail.link_source.locality_match",
            "restab.detail.link_source.m3_inference",
            "restab.detail.link_source.unlinked",
        ):
            assert key in src, f"missing zh-CN i18n key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
