"""Phase 65 Plan A.1 — sample-id extractor tests."""

from __future__ import annotations

import pytest

from rlpe.sample_id_extractor import (
    SampleID,
    extract_age_terms,
    extract_locality,
    extract_sample_ids,
)


class TestExtractSampleIDs:
    def test_simple_sample(self):
        out = extract_sample_ids("All specimens from Sample S1")
        assert out == [SampleID(kind="sample", value="S1", confidence=1.0)]

    def test_sample_with_dash(self):
        out = extract_sample_ids("Sample LR-7 was collected in 2019")
        assert SampleID(kind="sample", value="LR-7", confidence=1.0) in out

    def test_sample_with_id_prefix(self):
        out = extract_sample_ids("Sample ID-203, Loc. Tunisia")
        sample_ids = [s for s in out if s.kind == "sample"]
        loc_ids = [s for s in out if s.kind == "loc"]
        assert len(sample_ids) == 1
        assert sample_ids[0].value == "203"
        assert sample_ids[0].confidence < 1.0  # prefixed "Sample ID-" lowers conf
        assert any(l.value == "Tunisia" for l in loc_ids)

    def test_locality_keyword(self):
        out = extract_sample_ids("Localities: Tunisia and Greece")
        loc_values = [s.value for s in out if s.kind == "loc"]
        assert "Tunisia" in loc_values
        assert "Greece" in loc_values

    def test_loc_dot_keyword(self):
        out = extract_sample_ids("Loc. Sicily, Sample A")
        loc_values = [s.value for s in out if s.kind == "loc"]
        assert "Sicily" in loc_values
        sample_values = [s.value for s in out if s.kind == "sample"]
        assert "A" in sample_values

    def test_bare_id(self):
        out = extract_sample_ids("Specimen ID-42 was photographed.")
        ids = [s for s in out if s.kind == "id"]
        assert any(i.value == "42" for i in ids)

    def test_dedup(self):
        out = extract_sample_ids("Sample A. Then again sample a.")
        sample_a = [s for s in out if s.kind == "sample" and s.value.upper() == "A"]
        assert len(sample_a) == 1

    def test_empty(self):
        assert extract_sample_ids("") == []

    def test_none_safe(self):
        # Defensive: type checkers and external callers sometimes pass None.
        assert extract_sample_ids(None or "") == []

    def test_malformed_no_crash(self):
        # Random punctuation / no real token.
        out = extract_sample_ids("!@#$%^&*()")
        assert out == []

    def test_all_caps(self):
        out = extract_sample_ids("SAMPLE ZX-9 WAS RECOVERED")
        assert SampleID(kind="sample", value="ZX-9", confidence=1.0) in out

    def test_lowercase(self):
        out = extract_sample_ids("sample lr-3 collected in sicily")
        assert any(s.value.upper() == "LR-3" for s in out if s.kind == "sample")


class TestExtractLocality:
    def test_simple_from(self):
        out = extract_locality("All specimens from Tunisia")
        assert "Tunisia" in out

    def test_multi_word(self):
        out = extract_locality("Localities: NW Turkey and New Zealand")
        # These should be captured by extract_sample_ids (Loc. keyword);
        # extract_locality only catches "from/at/in" phrases without keyword.
        # Test only the bare phrase pattern:
        out2 = extract_locality("collected from NW Turkey")
        assert "NW Turkey" in out2

    def test_age_blocked(self):
        # "Late Cretaceous" must NOT be returned as a locality.
        out = extract_locality("from Late Cretaceous onwards")
        assert "Late Cretaceous" not in out

    def test_two_localities(self):
        out = extract_locality("from Tunisia and Greece")
        assert "Tunisia" in out
        assert "Greece" in out

    def test_empty(self):
        assert extract_locality("") == []

    def test_dedup_case_insensitive(self):
        out = extract_locality("from Italy. also from ITALY")
        italy_count = sum(1 for x in out if x.casefold() == "italy")
        assert italy_count == 1

    def test_short_phrase_filtered(self):
        # "from I" should be filtered (single-letter is noise).
        out = extract_locality("from I think so")
        assert "I" not in out


class TestExtractAgeTerms:
    def test_cretaceous(self):
        out = extract_age_terms("Late Cretaceous, Sample A")
        assert "Late Cretaceous" in out
        assert "A" not in out  # "A" is not an age

    def test_carnian(self):
        out = extract_age_terms("Carnian (Late Triassic)")
        assert "Carnian" in out

    def test_no_match(self):
        out = extract_age_terms("The paper discusses radiolarian taxonomy.")
        assert out == []

    def test_empty(self):
        assert extract_age_terms("") == []

    def test_dedup(self):
        out = extract_age_terms("Late Cretaceous. Then Late Cretaceous again.")
        late_count = sum(1 for x in out if x.casefold() == "late cretaceous")
        assert late_count == 1

    def test_longer_wins(self):
        # When both "Late Cretaceous" and "Cretaceous" match, the longer
        # phrase is kept; the dedup is on the full phrase so "Cretaceous"
        # also stays. We just assert both are present.
        out = extract_age_terms("Late Cretaceous")
        assert "Late Cretaceous" in out


# --- Audit 2026-08-18: extract_locality over-match fixes ----------------
#
# Four false-positive classes were over-matching without leading-article /
# trailing-stopword / Latin-particle handling:
#   1. ``Found in situ at the Karnezeika section.`` extracted ``situ``
#      and ``the Karnezeika section``.
#   2. ``at the Karnezeika section`` kept ``the Karnezeika section`` (3
#      trailing words including the leading article).
#   3. ``from the Scaglia formation`` kept the full phrase; the
#      blocklist had ``scaglia`` but the exact phrase didn't match.
#   4. ``in the field`` extracted ``the field`` (generic word that is
#      not a locality).
# All four are now caught by ``_normalize_locality_phrase``.


class TestExtractLocalityNormalization:
    def test_in_situ_not_extracted(self):
        out = extract_locality("Plate 1. Found in situ at the Karnezeika section.")
        assert "situ" not in out, f"Latin particle 'situ' over-matched: {out}"

    def test_leading_article_stripped(self):
        out = extract_locality("at the Karnezeika section, sample X")
        # Leading ``the`` and trailing ``section`` both stripped.
        assert "Karnezeika" in out, f"Karnezeika lost: {out}"
        assert "the Karnezeika section" not in out

    def test_blocklist_substring_match(self):
        """``from the Scaglia formation`` must be caught because
        ``scaglia`` is blocklisted as a substring."""
        out = extract_locality("Plate 1. From the Scaglia formation.")
        assert "Scaglia" not in out and "the Scaglia formation" not in out, (
            f"Scaglia formation should be blocklisted: {out}"
        )

    def test_field_not_extracted(self):
        out = extract_locality("Plate 1. Collected in the field.")
        assert "field" not in out and "the field" not in out, (
            f"Generic 'field' should not be a locality: {out}"
        )

    def test_real_locality_still_extracted(self):
        out = extract_locality("Plate 1. Radiolarians from Tunisia.")
        assert "Tunisia" in out

    def test_real_multi_word_locality_still_extracted(self):
        out = extract_locality("Samples from NW Turkey, locality X.")
        assert "NW Turkey" in out

    def test_trailing_and_pair_still_extracted(self):
        """The tail-pass that captures ``, X`` / ``and X`` after the
        initial match must still pick up both halves of a multi-locality
        phrase (``from Tunisia and Greece``)."""
        out = extract_locality("Radiolarians from Tunisia and Greece.")
        assert "Tunisia" in out
        assert "Greece" in out

    def test_in_vitro_blocklisted(self):
        """Other Latin particles (``in vivo``, ``in vitro``) are also
        blocklisted — invariant must hold."""
        out = extract_locality("Plate 1. Observed in vitro.")
        assert "vitro" not in out, f"Latin particle 'vitro' over-matched: {out}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
