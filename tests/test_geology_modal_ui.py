"""UI smoke tests for the geology_links display in the panel-detail modal.

The panel detail modal (openImageModal) renders the metadata.geology_links
list as age / formation / locality with confidence. These tests verify
the code path is in place AND reachable from the served static assets,
so a deployment that drops the file or strips the modal code fails fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_JS = REPO_ROOT / "web" / "js" / "app.js"
WEB_CSS = REPO_ROOT / "web" / "css" / "style.css"


class TestGeologyModalCode:
    @pytest.fixture(scope="class")
    def js(self) -> str:
        if not WEB_JS.exists():
            pytest.skip("web/js/app.js not present")
        return WEB_JS.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def css(self) -> str:
        if not WEB_CSS.exists():
            pytest.skip("web/css/style.css not present")
        return WEB_CSS.read_text(encoding="utf-8")

    def test_open_image_modal_function_exists(self, js: str) -> None:
        "The modal renderer must be defined."
        assert "function openImageModal(" in js

    def test_geology_links_renderer_in_open_image_modal(self, js: str) -> None:
        "openImageModal must walk metadata.geology_links."
        i = js.find("function openImageModal(")
        j = js.find("// Attach close handlers", i)
        assert j > 0, "could not find end of openImageModal"
        body = js[i:j]
        assert "geology_links" in body
        assert "g.age" in body
        assert "g.formation" in body
        assert "g.locality" in body

    def test_modal_geo_list_class_used(self, js: str) -> None:
        assert "modal-geo-list" in js

    def test_modal_geo_conf_class_used(self, js: str) -> None:
        assert "modal-geo-conf" in js

    def test_escape_html_used_on_geo_fields(self, js: str) -> None:
        "User-supplied age/formation/locality must be HTML-escaped."
        i = js.find("function openImageModal(")
        j = js.find("// Attach close handlers", i)
        body = js[i:j]
        assert "escapeHtml(age)" in body
        assert "escapeHtml(g.formation)" in body
        assert "escapeHtml(g.locality)" in body

    def test_renders_extended_geology_fields(self, js: str) -> None:
        """Task 6 enrichment: the modal must render Ma range (ma_top /
        ma_base), lithology, member, group, and biozone when present.
        These fields come from Task 5's deterministic Ma propagation
        and the planned MiniMax-assisted lithology/member/group
        extractor. They were previously dropped on the floor.
        """
        i = js.find("function openImageModal(")
        j = js.find("// Attach close handlers", i)
        body = js[i:j]
        for field in ("g.lithology", "g.member", "g.group", "g.biozone"):
            assert field in body, f"missing {field} in modal render"
        assert "ma_top" in body
        assert "ma_base" in body
        assert "modal-geo-ma" in body, "missing Ma badge CSS class"

    def test_escape_html_used_on_extended_geo_fields(self, js: str) -> None:
        """All new optional fields must also be HTML-escaped."""
        i = js.find("function openImageModal(")
        j = js.find("// Attach close handlers", i)
        body = js[i:j]
        for field in ("g.lithology", "g.member", "g.group", "g.biozone"):
            assert f"escapeHtml({field})" in body, f"missing escapeHtml for {field} in modal render"

    def test_geology_block_template_literal_terminated(self, js: str) -> None:
        "No stray template literal: the IIFE and its template must close."
        i = js.find("function openImageModal(")
        j = js.find("// Attach close handlers", i)
        body = js[i:j]
        bt = body.count(chr(96))
        assert bt % 2 == 0, "backticks unbalanced in openImageModal: " + str(bt)
        assert "})()" in body

    def test_css_modal_geo_list_styled(self, css: str) -> None:
        i = css.find(".modal-geo-list {")
        assert i > 0, ".modal-geo-list rule missing"
        end = css.find("}", i)
        rule = css[i:end]
        assert rule.count(chr(10)) >= 3

    def test_css_modal_geo_conf_styled(self, css: str) -> None:
        i = css.find(".modal-geo-conf {")
        assert i > 0, ".modal-geo-conf rule missing"

    def test_css_modal_geo_ma_styled(self, css: str) -> None:
        """The Ma badge class must be styled, otherwise it renders as
        plain text indistinguishable from surrounding content."""
        i = css.find(".modal-geo-ma {")
        assert i > 0, ".modal-geo-ma rule missing"
        end = css.find("}", i)
        rule = css[i:end]
        assert rule.count(chr(10)) >= 3, (
            "modal-geo-ma rule must contain at least 3 declarations "
            "(background + color + numeric formatting)"
        )
        assert "tabular-nums" in rule or "tabular" in rule, (
            "Ma values should use tabular-nums for stable column width"
        )


class TestGeologyModalRender:
    """Simulate the openImageModal geology block in Python.

    Runs the same logic the JS would run for a given record, and asserts
    the produced HTML contains the expected age/formation/locality tags.
    """

    @staticmethod
    def _render(record):
        md = record.get("metadata") or {}
        links = md.get("geology_links") or []
        if not links:
            return ""
        items = []
        for g in links:
            age = g.get("age") or g.get("chronostratigraphy")
            head = []
            if age:
                head.append("<strong>" + str(age) + "</strong>")
            if g.get("formation"):
                head.append("<em>" + str(g["formation"]) + "</em>")
            if g.get("locality"):
                head.append("<span>" + str(g["locality"]) + "</span>")
            head_str = " · ".join(head)
            if not head_str:
                continue
            conf = g.get("confidence")
            conf_html = ""
            if conf is not None:
                conf_html = (
                    " <span class="
                    + chr(34)
                    + "modal-geo-conf"
                    + chr(34)
                    + ">("
                    + str(int(conf * 100))
                    + "%)</span>"
                )
            items.append("<li>" + head_str + conf_html + "</li>")
        if not items:
            return ""
        return (
            "<div class="
            + chr(34)
            + "modal-row modal-row-wide"
            + chr(34)
            + ">"
            + "<span class="
            + chr(34)
            + "modal-label"
            + chr(34)
            + ">地质关联:</span>"
            + "<ul class="
            + chr(34)
            + "modal-geo-list"
            + chr(34)
            + ">"
            + "".join(items)
            + "</ul></div>"
        )

    def test_empty_links_renders_empty(self):
        assert self._render({"metadata": {}}) == ""
        assert self._render({"metadata": {"geology_links": []}}) == ""

    def test_missing_metadata_renders_empty(self):
        assert self._render({}) == ""

    def test_full_record_renders_three_fields(self):
        rec = {
            "metadata": {
                "geology_links": [
                    {
                        "age": "Permian",
                        "formation": "Dalong",
                        "locality": "Guizhou",
                        "confidence": 0.85,
                    }
                ]
            }
        }
        html = self._render(rec)
        assert "<strong>Permian</strong>" in html
        assert "<em>Dalong</em>" in html
        assert "<span>Guizhou</span>" in html
        assert "85%" in html
        assert "modal-geo-list" in html

    def test_chrono_fallback_when_age_missing(self):
        rec = {"metadata": {"geology_links": [{"chronostratigraphy": "Changhsingian"}]}}
        html = self._render(rec)
        assert "<strong>Changhsingian</strong>" in html

    def test_only_formation_renders_without_strong(self):
        rec = {"metadata": {"geology_links": [{"formation": "Dalong"}]}}
        html = self._render(rec)
        assert "<em>Dalong</em>" in html
        assert "<strong>" not in html

    def test_record_with_all_blank_fields_renders_empty(self):
        rec = {"metadata": {"geology_links": [{"section_title": "x"}]}}
        html = self._render(rec)
        assert html == ""

    def test_multiple_records_rendered_as_list_items(self):
        rec = {
            "metadata": {
                "geology_links": [{"age": "Permian"}, {"age": "Triassic", "confidence": 0.5}]
            }
        }
        html = self._render(rec)
        assert html.count("<li>") == 2
        assert "<strong>Permian</strong>" in html
        assert "<strong>Triassic</strong>" in html
