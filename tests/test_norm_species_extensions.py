"""Tests for the species normalisation rules.

The soft norm (_norm_species) is intentionally conservative: it should
never silently equate two biologically-distinct taxon strings. The
historical aff./cf. qualifier stripping was REMOVED in 2026-07-01
because it inverted the F1-inflation problem (was hiding real gold/
pred mismatches instead of exposing them).

These tests cover both the rules that DO apply and the rules that
DO NOT apply (adversarial regression tests so a future patch that
re-introduces aff./cf. stripping will fail loudly).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.evaluation.metrics import _norm_species, _strict_norm_species  # noqa: E402


# ---------------------------------------------------------------------------
# Strict norm — should NOT touch cf./aff./trademarks, only whitespace
# ---------------------------------------------------------------------------
def test_strict_norm_preserves_cf_qualifier():
    """Strict norm must keep 'X cf. Y' intact — these are different taxa."""
    assert _strict_norm_species("X cf. Y") == "X cf. Y"


def test_strict_norm_preserves_aff_qualifier():
    assert _strict_norm_species("X aff. Y") == "X aff. Y"


def test_strict_norm_preserves_sample_code():
    """The '-K47/4' suffix is a type-specimen identifier; do not strip."""
    assert _strict_norm_species("Haliomma gr. A-K47/4") == "Haliomma gr. A-K47/4"


def test_strict_norm_drops_leading_question_only():
    """Strict norm should only drop leading '?' (which is whitespace-like),
    never mid-string '?' or aff./cf."""
    assert _strict_norm_species("?Theocorys phyzella") == "Theocorys phyzella"
    assert _strict_norm_species("Theocorys? phyzella") == "Theocorys? phyzella"


# ---------------------------------------------------------------------------
# Soft norm — what gets normalised and what stays
# ---------------------------------------------------------------------------
def test_trailing_question_after_genus_collapsed():
    """hollis2006 gold 'Theocorys? phyzella' vs pred 'Theocorys phyzella'.
    Rule: drop the in-line '?' so they match."""
    assert _norm_species("Theocorys? phyzella") == _norm_species("Theocorys phyzella")
    assert _norm_species("Theocorys? phyzella") == "Theocorys phyzella"


def test_cf_qualifier_preserved_adversarial():
    """ADVERSARIAL: the old (overfit) rule stripped ' aff. <epithet>' and
    ' cf. <epithet>'. The current norm must NOT do this — these are
    legitimate open-nomenclature signals."""
    # 'cf.' preserved with epithet (note: Archaeo→Archeo canonicalization
    # applies; the cf. preservation is the property we're testing).
    assert _norm_species("Hiscocapsa cf. kaminogoensis") == "Hiscocapsa cf. kaminogoensis"
    assert _norm_species("Archaeodictyomitra cf. tumandae") == "Archeodictyomitra cf. tumandae"
    assert _norm_species("Williriedellum cf. carpathicum") == "Williriedellum cf. carpathicum"


def test_aff_qualifier_preserved_adversarial():
    """ADVERSARIAL: same reasoning as cf. above."""
    assert _norm_species("Axoprunum aff. bispiculum") == "Axoprunum aff. bispiculum"
    assert _norm_species("Hiscocapsa aff. asseni") == "Hiscocapsa aff. asseni"


def test_cf_qualifier_collapsed_to_binomial_NOT():
    """ADVERSARIAL: the old rule used to collapse 'X cf. Y' to 'X' which
    is biologically wrong. The current rule must NOT do this."""
    # Old (broken) behaviour would give: "Hiscocapsa"
    # Current (correct) behaviour gives: "Hiscocapsa cf. kaminogoensis"
    assert _norm_species("Hiscocapsa cf. kaminogoensis") != "Hiscocapsa"


def test_sample_code_preserved_adversarial():
    """ADVERSARIAL: the old rule stripped '-[A-Z]?\\d+/\\d+' suffixes
    on the gold side. The current rule must NOT do this."""
    # Old (broken) behaviour would give: "Haliomma gr. A"
    # Current (correct) behaviour gives: "Haliomma gr. A-K47/4"
    assert _norm_species("Haliomma gr. A-K47/4") == "Haliomma gr. A-K47/4"


def test_trinomial_collapse_real_subspecies():
    """audit 2026-07-31: a real subspecies ('Eucyrtidiellum unumaense
    pustulatum') is a DISTINCT taxon and is preserved by _norm_species.
    It remains COMPATIBLE with the species-level name via
    _species_compatible (subspecies is a refinement of the species
    determination), so a species-level gold still matches."""
    from rlpe.evaluation.metrics import _species_compatible

    norm = _norm_species("Eucyrtidiellum unumaense pustulatum")
    assert norm == "Eucyrtidiellum unumaense pustulatum", norm
    assert _species_compatible(norm, "Eucyrtidiellum unumaense")
    # a DIFFERENT subspecies must NOT match
    assert not _species_compatible(norm, "Eucyrtidiellum unumaense dentatum")


def test_trinomial_collapse_stops_at_qualifier():
    """A 3-word string that contains a qualifier (cf./aff./gr./subsp.) is
    NOT a real trinomial — it's open-nomenclature and must be preserved."""
    assert _norm_species("X cf. Y") == "X cf. Y"
    assert _norm_species("X aff. Y") == "X aff. Y"
    assert _norm_species("X gr. A") == "X gr. A"


def test_trinomial_collapse_stops_at_gr_marker():
    """'Haliomma gr. A' has 3 words but 'gr.' is a paper-specific
    grouping marker, not an epithet — must NOT be collapsed."""
    assert _norm_species("Haliomma gr. A") == "Haliomma gr. A"


def test_archaeo_canonicalised():
    """Archaeo/Archeo spelling variants are interchangeable."""
    assert _norm_species("Archaeodictyomitra sp.") == _norm_species("Archeodictyomitra sp.")


def test_spumellaria_long_form_collapses():
    """hollis2006 gold 'Spumellaria gen. et sp. indet. A' (pred drops
    the verbose form)."""
    assert _norm_species("Spumellaria gen. et sp. indet. A") == "Spumellaria indet A"


def test_spumellaria_short_form_collapses():
    assert _norm_species("Spumellaria indet. A") == "Spumellaria indet A"


def test_spumellaria_long_and_short_match():
    a = _norm_species("Spumellaria gen. et sp. indet. A")
    b = _norm_species("Spumellaria indet. A")
    assert a == b


def test_spumellaria_long_form_B():
    a = _norm_species("Spumellaria gen. et sp. indet. B")
    b = _norm_species("Spumellaria indet. B")
    assert a == b


def test_paren_question_marker_stripped():
    """hollis2006 'Stichomitra (?) sp.' should match 'Stichomitra sp.'."""
    assert _norm_species("Stichomitra (?) sp.") == _norm_species("Stichomitra sp.")


def test_bare_sp_stripped_at_end():
    """'X sp' (parser-added) and 'X' (gold dropped) should match."""
    assert _norm_species("Theocampe sp") == _norm_species("Theocampe")


def test_sp_with_letter_preserved():
    """'X sp. B' has a meaningful morphotype identifier — must NOT collapse.
    Note: Archaeo→Archeo canonicalization applies."""
    assert _norm_species("Archaeodictyomitra sp. A") == "Archeodictyomitra sp. A"
    assert _norm_species("Archaeodictyomitra sp. B") == "Archeodictyomitra sp. B"


def test_genus_initial_unchanged_under_strict():
    """Strict norm preserves 'A. patricki' (it's a gold abbreviation convention)."""
    assert _strict_norm_species("A. patricki") == "A. patricki"
    assert _strict_norm_species("Archaeodictyomitra patricki") == "Archaeodictyomitra patricki"


# ---------------------------------------------------------------------------
# ADVERSARIAL: rules that should NEVER apply
# ---------------------------------------------------------------------------
def test_bare_genus_no_false_collapse():
    """The rule 'spp' → '' must not eat any non-spp suffix."""
    assert _norm_species("X spp") == "X"
    assert _norm_species("X sp") == "X"


def test_cf_qualifier_doesnt_trigger_sp_strip():
    """'X cf. Y sp.' — the cf. is preserved AND sp. at end is stripped
    (because 'cf. Y sp.' is not a real pattern, cf. Y is the species).
    Actually: 'sp.' trailing is stripped first, then 'X cf. Y' remains.
    This is what hollis2006 wants for open-nomenclature."""
    # sp. trailing gets stripped (matches \s+sp\.?$)
    # then cf. is preserved (no rule touches it)
    result = _norm_species("X cf. Y sp.")
    # 'sp.' is at the very end → stripped → "X cf. Y"
    assert result == "X cf. Y"


def test_no_false_positive_sp_in_middle():
    """'Genus species' (binomial) must not be reduced to 'Genus'."""
    assert _norm_species("Genus species") == "Genus species"


def test_no_false_positive_trinomial_with_capitalized_third():
    """'Genus epithet Authority' (author citation) — third word is
    capitalized, so the all-lowercase trinomial rule does NOT apply."""
    assert _norm_species("Theocorys phyzella Foreman") == "Theocorys phyzella Foreman"


def test_trinomial_collapse_skips_uncertainty_marker():
    """REGRESSION (2026-07-01): a trinomial with trailing '?' on the
    epithet (e.g. 'X unumaense pustulatum?') was being collapsed to
    'X unumaense', silently dropping the uncertainty marker. The
    uncertainty marker is a real taxonomic signal and must be kept."""
    assert _norm_species("X unumaense pustulatum?") == "X unumaense pustulatum?"
    assert _norm_species("X unumaense pustulatum ?") == "X unumaense pustulatum ?"


# ---------------------------------------------------------------------------
# Symmetric — same rule on both sides
# ---------------------------------------------------------------------------
def test_sp_strip_symmetric():
    """Both 'X' and 'X sp' should collapse to 'X' (parser asymmetry both ways)."""
    assert _norm_species("X") == "X"
    assert _norm_species("X sp") == "X"
    assert _norm_species("X sp.") == "X"
