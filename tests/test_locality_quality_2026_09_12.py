"""2026-09-12: locality quality — author-line sections fabricated a
"locality" (Isakova, a cited author) plus the Russia country centroid
(60N/100E, conf 0.3); section_name carried the author byline
"M. S. Afanasieva*". Author-shaped sections must not be geology
sources, and failed paleo reconstructions must report a precise
status instead of one generic bucket."""

from __future__ import annotations

from rlpe.geology_extraction import (
    _is_author_line_title,
    extract_geology_from_sections,
)


class TestAuthorLineGate:
    def test_dotted_initial_byline_detected(self):
        assert _is_author_line_title("M. S. Afanasieva*") is True
        assert _is_author_line_title("M.S. Afanasieva") is True

    def test_footnote_name_detected(self):
        assert _is_author_line_title("Corresponding author*") is True

    def test_real_section_headings_pass(self):
        assert _is_author_line_title("Kondurovka Section") is False
        assert _is_author_line_title("Geological Setting") is False
        assert _is_author_line_title("Materials and Methods") is False
        assert _is_author_line_title("Systematic paleontology") is False

    def test_long_titles_exempt(self):
        assert _is_author_line_title(
            "Revision of the genera Spinodeflandrella Kozur 1981 and Holdsworthella" * 2
        ) is False


class TestAuthorSectionSkipped:
    def test_author_section_yields_no_records(self):
        sections = [
            {
                "title": "M. S. Afanasieva*",
                "section_type": "other",
                "text": (
                    "in Russia by Isakova and colleagues at the institute. "
                    "The studied material came from Russia."
                ),
            }
        ]
        assert extract_geology_from_sections(sections) == []

    def test_real_geology_section_unaffected(self):
        sections = [
            {
                "title": "Geological Setting",
                "section_type": "geological_setting",
                "text": (
                    "The Kondurovka Section is located in Russia, "
                    "in the South Urals (51.5N, 57.5E)."
                ),
            }
        ]
        records = extract_geology_from_sections(sections)
        assert records, "real geology section must still produce records"


class TestPaleoStatusPrecision:
    def test_stable_plate_status(self):
        from rlpe.paleo_reconstruction import explain_paleo_status

        assert (
            explain_paleo_status("Siberia", 286.8) == "stable_plate_no_rotation"
        )

    def test_age_and_plate_unknown(self):
        from rlpe.paleo_reconstruction import explain_paleo_status

        assert explain_paleo_status("Siberia", None) == "age_unknown"
        assert explain_paleo_status(None, 286.8) == "plate_unknown"
        assert explain_paleo_status("NotAPlate", 286.8) == "plate_unknown"


class TestRangeChartBridgeGate:
    """2026-09-12: the Stage-2 verdict bridge routes figures the vision
    classifier rejects as plates but types as chart/diagram (with a
    species-bearing caption) to the range-chart extractor. The gate
    logic mirrors the pipeline block; this pins the predicate."""

    def test_predicate_signals(self):
        import re

        caption = (
            "Fig. 2. Asselian and Sakmarian radiolarians of the Lower "
            "Permian South Urals in the Kondurovka (1\u20133)"
        )
        img_type = "diagram"
        # epithet >= 4 chars so the header tail "2. Asselian and"
        # ("and" = 3 chars) is not counted as a species clause.
        sp_signals = len(
            re.findall(r"\b\d{1,2}\s*\.\s+[A-Z][a-z]{3,}\s+[a-z]{4,}", caption)
        )
        assert sp_signals == 0
        # A real numbered clause list does count.
        real = (
            "Plate 4. Figs. 1-11. 1. Pseudoalbaillella sakmarensis "
            "2. Holdsworthella permica"
        )
        assert (
            len(re.findall(r"\b\d{1,2}\s*\.\s+[A-Z][a-z]{3,}\s+[a-z]{4,}", real)) >= 2
        )

    def test_image_type_whitelist(self):
        whitelist = {"diagram", "chart", "table", "graph"}
        assert "diagram" in whitelist and "photo" not in whitelist
