"""2026-09-12: content-evidence section typing — an "other"-typed
section whose BODY independently carries 2+ distinct geology signal
families (ICS ages, stratigraphic ranks, coordinates, lithology) is
retyped geological_setting, regardless of its heading."""

from __future__ import annotations

from rlpe.opendataloader_extractor import (
    _extract_fulltext_sections,
    _geology_content_signals,
    _infer_section_type,
)


GEO_TEXT = (
    "The Khivach River section exposes the Halika Formation near "
    "64.5 N 170.2 E. Norian and Carnian radiolarians occur in grey "
    "chert and siliceous mudstone of the Pvantunskaya Formation."
)
PLAIN_TEXT = (
    "The specimens were collected during the expedition of 1994 and "
    "deposited in the museum collection of the institute."
)


class TestContentSignals:
    def test_geology_text_scores_high(self):
        assert _geology_content_signals(GEO_TEXT) >= 2

    def test_plain_text_scores_zero(self):
        assert _geology_content_signals(PLAIN_TEXT) == 0


class TestSectionRetyping:
    def _tree(self, title: str, text: str) -> dict:
        return {
            "kids": [
                {"type": "heading", "content": title, "page number": 1},
                {"type": "paragraph", "page number": 1, "content": text},
            ]
        }

    def test_other_typed_section_upgraded_on_content(self):
        secs = _extract_fulltext_sections(self._tree("Background", GEO_TEXT))
        assert secs, "section must exist"
        assert secs[0]["section_type"] == "geological_setting"

    def test_plain_other_section_stays_other(self):
        secs = _extract_fulltext_sections(self._tree("Background", PLAIN_TEXT))
        assert secs[0]["section_type"] == "other"

    def test_short_geology_text_not_upgraded(self):
        # Below the 200-char trust threshold: heading keywords only.
        secs = _extract_fulltext_sections(
            self._tree("Notes", "Halika Formation, Norian chert, 64.5 N.")
        )
        assert secs[0]["section_type"] == "other"

    def test_heading_keyword_still_wins(self):
        secs = _extract_fulltext_sections(self._tree("Geological Setting", PLAIN_TEXT))
        assert secs[0]["section_type"] == "geological_setting"
