"""Round 20 source-guard tests: geology field correctness.

User audit (Round 20 sampling of 4 OA papers) identified 4 systemic
geology-field correctness issues:

  1. AGE_PATTERN matches phrases like ``"lower part"`` and
     ``"upper reaches"`` because the regex ``(?:Early|Middle|Late|...)\\s+
     [A-Z][a-z]+`` accepts any Capitalised noun. These leak into the
     ``age`` column of GeologyLinkRecord. Fix: validate each match
     through ``classify_age_string`` (ICS lexicon) and drop
     non-stratigraphic phrases.

  2. LOCALITY_PATTERN captures period names (e.g. ``"from Upper
     Cretaceous formations"`` → locality=``"Upper Cretaceous"``).
     Fix: reject any locality string that the ICS lexicon
     recognises as a real period / epoch / age.

  3. The formation regex's name prefix can absorb tokens like
     ``"Karnezeika-19 Formation"`` where the digit "19" makes the
     match invalid (it's a page reference, not a formation).
     Fix: post-filter matches whose name contains any digit.

  4. References / bibliography sections leak OTHER papers' geology
     into the current paper. Danelian 2006's References section
     mentioned "Fonzaso Formation" (a Beccaro 2002 title) and
     "Japan" (a Palaeopacific citation), and both leaked into
     Danelian's geology records. Fix: GROBID ``infer_section_type``
     recognises references / bibliography; ``extract_geology_from_sections``
     skips them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text(
        encoding="utf-8"
    )


# --- 1) Age whitelist -------------------------------------------------------


def test_age_matches_lower_part_dropped():
    """Bragin 2025's 'lower part of the succession' must NOT appear
    in the age column. The raw regex would match 'lower part' because
    of the ``Lower\\s+[A-Z][a-z]+`` alternative; the Round 20 fix
    rejects it via classify_age_string."""
    from rlpe.geology_extraction import extract_geology_from_sections

    sections = [
        {
            "title": "Geological setting",
            "text": (
                "Boreal radiolarians from the lower part of the section are "
                "from the Upper Jurassic, and the upper reaches contain "
                "Lower Cretaceous taxa. The Chulkovskaya Formation is "
                "phosphorite."
            ),
            "section_type": "geological_setting",
        }
    ]
    records = extract_geology_from_sections(sections)
    ages_seen = [r.age for r in records if r.age]
    # "lower part" / "upper reaches" must NOT be in ages
    assert "lower part" not in ages_seen, (
        f"Age leaked: 'lower part' survived whitelist. Got ages: {ages_seen}"
    )
    assert "upper reaches" not in ages_seen, (
        f"Age leaked: 'upper reaches' survived whitelist. Got ages: {ages_seen}"
    )
    # But the real periods must be kept
    assert any("Jurassic" in a or "Cretaceous" in a for a in ages_seen), (
        f"Real ages were dropped — whitelist too aggressive. Got: {ages_seen}"
    )


def test_age_pattern_late_jurassic_kept():
    """Sanity: 'Late Jurassic' (a real age) must survive validation."""
    from rlpe.geology_extraction import extract_geology_from_sections

    sections = [
        {
            "title": "Geological setting",
            "text": "Samples come from Late Jurassic strata of the basin.",
            "section_type": "geological_setting",
        }
    ]
    records = extract_geology_from_sections(sections)
    ages = [r.age for r in records if r.age]
    assert any("Late Jurassic" in a for a in ages), f"Lost real age: {ages}"


def test_age_validation_uses_stratigraphy_lexicon():
    """Source guard: geology_extraction must import classify_age_string
    and use it to filter raw age matches."""
    src = _read("src/rlpe/geology_extraction.py")
    assert "classify_age_string" in src, (
        "geology_extraction.py does not use classify_age_string. "
        "The age whitelist depends on this function."
    )


# --- 2) Locality filter -----------------------------------------------------


def test_locality_upper_cretaceous_rejected():
    """Bandini 2006's 'from Upper Cretaceous formations' must NOT
    produce locality='Upper Cretaceous'."""
    from rlpe.geology_extraction import extract_geology_from_sections

    sections = [
        {
            "title": "Geological setting",
            "text": (
                "The radiolarian taxa described here are from Upper Cretaceous "
                "formations of the Karnezeika area in Greece."
            ),
            "section_type": "geological_setting",
        }
    ]
    records = extract_geology_from_sections(sections)
    localities = [r.locality for r in records if r.locality]
    assert "Upper Cretaceous" not in localities, f"Period used as locality: {localities}"


def test_locality_karnezeika_kept():
    """Real locality name 'Karnezeika' must still be captured."""
    from rlpe.geology_extraction import extract_geology_from_sections

    sections = [
        {
            "title": "Geological setting",
            "text": (
                "The radiolarian taxa described here are from Karnezeika, "
                "Greece. The formations are Upper Cretaceous."
            ),
            "section_type": "geological_setting",
        }
    ]
    records = extract_geology_from_sections(sections)
    localities = [r.locality for r in records if r.locality]
    assert "Karnezeika" in localities, f"Real locality lost: {localities}"


# --- 3) Formation digit filter ---------------------------------------------


def test_formation_with_digits_rejected():
    """'Karnezeika-19 Formation' (page reference) must be rejected;
    'Fonzaso Formation' (real) must be kept."""
    from rlpe.geology_extraction import _FORMATION_RE

    text = (
        "See page 19 in Beccaro for the Karnezeika-19 Formation details. "
        "The Fonzaso Formation is siliceous limestone."
    )
    matches = [m.group(1) for m in _FORMATION_RE.finditer(text)]
    # The formation regex still matches both — the post-filter is in
    # extract_geology_from_sections, not the regex itself. We assert
    # the post-filter via the higher-level extractor.
    from rlpe.geology_extraction import extract_geology_from_sections

    sections = [
        {
            "title": "Geological setting",
            "text": text,
            "section_type": "geological_setting",
        }
    ]
    records = extract_geology_from_sections(sections)
    formations = [r.formation for r in records if r.formation]
    # The digit-containing match should be filtered out
    assert not any("19" in f for f in formations), f"Formation with digit leaked: {formations}"
    # The real formation should still be present
    assert any("Fonzaso" in f for f in formations), f"Real formation lost: {formations}"


def test_formation_filter_helper_exists():
    """Source guard: _formation_name_ok must exist and reject digits."""
    import rlpe.geology_extraction as ge
    from rlpe.geology_extraction import extract_geology_from_sections

    # The function name may be local; check by source inspection
    src = _read("src/rlpe/geology_extraction.py")
    assert "_formation_name_ok" in src, (
        "geology_extraction.py is missing the _formation_name_ok "
        "post-filter that rejects matches with digits in the prefix."
    )


# --- 4) References section skip --------------------------------------------


def test_references_section_skipped():
    """Danelian 2006 had a References section mentioning 'Fonzaso
    Formation' (Beccaro 2002 citation) and 'Japan' (Palaeopacific).
    These must NOT leak into the records as if they were Danelian's
    own geology."""
    from rlpe.geology_extraction import extract_geology_from_sections

    sections = [
        {
            "title": "Geological setting",
            "text": "The Vocontian Basin of SE France is Upper Jurassic.",
            "section_type": "geological_setting",
        },
        {
            "title": "References",
            "text": (
                "BECCARO P. 2002: Radiolarian biostratigraphy of the Fonzaso "
                "Formation. Palaeopacific (Japan and elsewhere)."
            ),
            "section_type": "references",
        },
    ]
    records = extract_geology_from_sections(sections)
    formations = [r.formation for r in records if r.formation]
    localities = [r.locality for r in records if r.locality]
    countries = [r.country for r in records if r.country]
    # The references-section leaks must be gone
    assert "Fonzaso Formation" not in formations, f"Reference leaked into formation: {formations}"
    assert "Japan" not in countries, f"Reference country leaked: {countries}"


def test_grobid_recognises_references_section():
    """Source guard: grobid.infer_section_type must classify a
    'References' heading as 'references' so the filter above works."""
    from rlpe.grobid import infer_section_type

    for title in ["References", "REFERENCES", "Bibliography", "Literature cited"]:
        got = infer_section_type(title)
        assert got == "references", (
            f"infer_section_type({title!r}) returned {got!r}; expected 'references'"
        )


def test_references_skipped_even_without_section_type():
    """Even if section_type is missing (e.g. older pipelines), a title
    containing 'reference' / 'bibliograph' must skip the section."""
    from rlpe.geology_extraction import extract_geology_from_sections

    sections = [
        {
            "title": "Bibliography",
            "text": "Fonzaso Formation is mentioned in Beccaro 2002.",
            # No section_type set — simulates older pipeline output
        }
    ]
    records = extract_geology_from_sections(sections)
    formations = [r.formation for r in records if r.formation]
    assert "Fonzaso Formation" not in formations, (
        f"Bibliography section leaked into formation: {formations}"
    )


# --- 5) Systematic Palaeontology citation leakage -------------------------


def test_systematic_palaeontology_does_not_leak_into_panel_fallback():
    """Round 20 sampling followup: Danelian 2006 panel 6 had
    country=Japan / formation="Fonzaso Formation" from the
    "Systematic Palaeontology" section (which cites Beccaro 2002
    in synonymy lists). ``link_panels_to_geology`` must NOT pull
    candidates from systematic_paleontology sections — they're
    citation-heavy and leak other papers' geology.
    """
    from rlpe.geology_extraction import link_panels_to_geology

    fallback_sections = [
        {
            "title": "Geological setting",
            "text": "The Vocontian Basin of SE France is Upper Jurassic.",
            "section_type": "geological_setting",
        },
        {
            "title": "Systematic Palaeontology",
            "text": (
                "Genus Acastea. Type species from Fonzaso Formation, "
                "previously known only from Palaeopacific (Japan and "
                "elsewhere)."
            ),
            "section_type": "systematic_paleontology",
        },
    ]
    captions = {"panel_X": "Plate 1. Auto-generated figure for page 1."}
    out = link_panels_to_geology(captions, fallback_sections=fallback_sections)
    panel_geo = out.get("panel_X", [])
    formations = [g.get("formation") for g in panel_geo if g.get("formation")]
    countries = [g.get("country") for g in panel_geo if g.get("country")]
    assert "Fonzaso Formation" not in formations, (
        f"Systematic Palaeontology leaked Fonzaso: {formations}"
    )
    assert "Japan" not in countries, f"Systematic Palaeontology leaked Japan: {countries}"


def test_systematic_palaeontology_filter_source_guard():
    """Source guard: link_panels_to_geology must filter fallback
    sections by section_type to exclude systematic_paleontology."""
    src = _read("src/rlpe/geology_extraction.py")
    assert "systematic_paleontology" in src, (
        "geology_extraction.py doesn't filter systematic_paleontology "
        "sections. Round 20 sampling showed Danelian panel 6 leaking "
        "Japan / Fonzaso Formation through this path."
    )
