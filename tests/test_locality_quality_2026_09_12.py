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
