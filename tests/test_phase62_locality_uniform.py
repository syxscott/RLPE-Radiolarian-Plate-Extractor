"""Phase 62 Plan 5 (Bug 5.17): LOCALITY_PATTERN punctuation normalisation.

The ``LOCALITY_PATTERN`` regex captures a Capitalised word +
optionally 1-3 more Capitalised tokens, then looks ahead for
either a sentence-boundary punctuation (``.``, ``;``, ``:``, ``(``,
``)``) or a stop-word (``and``, ``the``, ``of``, ``a``, ``an``,
``is``, ``are``, ``was``, ``were``, ``in``, ``at``).

The current shape handles "from <Caps> the" and "from <Caps>,"
uniformly, but not the "from <Caps>, in <LOCATION>" form
("from Favignana, in Sicily"). The lookahead ``[,.;:()]|\s+(?:and|the|...)``
allows the comma + space + stopword form, but a comma followed by
``in`` (a preposition used to introduce the next locality) is
sometimes rejected depending on the regex engine.

The fix: normalise the punctuation — collapse "from <X>, in <Y>"
into two separate locality matches. We do this by:

  1. The ``, in`` form is already handled by the lookahead
     (stopword ``in`` is in the list). We confirm this works.
  2. Trim trailing stop-words from the captured group so a match
     like "from The Italy, the" produces "Italy" not "Italy, the".

This test asserts:
  * ``, in <Loc>`` form produces two locality matches.
  * Trailing stop-words ("Italy, the") are trimmed off the
    captured locality.
"""
from __future__ import annotations

from rlpe.geology_extraction import (
    LOCALITY_PATTERN,
    extract_geology_from_sections,
)


def test_locality_handles_comma_in():
    """'from Favignana, in Sicily' must produce both Favignana and
    Sicily as localities."""
    text = "samples from Favignana, in Sicily were collected"
    matches = [m.group(1).strip(" .,;") for m in LOCALITY_PATTERN.finditer(text)]
    # Both localities should be captured.
    assert "Favignana" in matches, f"Favignana not in {matches}"
    assert "Sicily" in matches, f"Sicily not in {matches}"


def test_locality_trims_trailing_stopword():
    """'from Italy, the' must produce 'Italy' (trailing stopword
    stripped), not 'Italy, the'."""
    text = "from Italy, the type section is described"
    matches = [m.group(1).strip(" .,;") for m in LOCALITY_PATTERN.finditer(text)]
    # The match should be just "Italy" without trailing stopword.
    assert "Italy" in matches
    # And NOT a polluted form like "Italy, the".
    assert not any("Italy, the" in m for m in matches), (
        f"trailing stopword 'the' not stripped; matches={matches}"
    )


def test_locality_trims_trailing_the():
    """'from New York, the' → 'New York' (just 'New York', not
    'New York, the')."""
    text = "from New York, the samples were processed"
    matches = [m.group(1).strip(" .,;") for m in LOCALITY_PATTERN.finditer(text)]
    assert "New York" in matches
    assert not any("New York, the" in m for m in matches), matches


def test_locality_handles_comma_in_real_sections():
    """End-to-end via extract_geology_from_sections: a section with
    'from Favignana, in Sicily' must yield at least one record with
    locality = one of the two names."""
    sections = [
        {
            "text": "Samples from Favignana, in Sicily were collected from the Rosso Ammonitico Formation in the Jurassic.",
            "title": "test",
            "section_type": "geological_setting",
        }
    ]
    records = extract_geology_from_sections(sections)
    localities = {r.locality for r in records if r.locality}
    # At least one of the two names must be captured.
    assert localities & {"Favignana", "Sicily"}, (
        f"neither Favignana nor Sicily captured; got: {localities}"
    )


def test_locality_regression_simple():
    """Regression: 'from Italy' (no trailing punctuation) still works."""
    text = "from Italy we describe new species"
    matches = [m.group(1).strip(" .,;") for m in LOCALITY_PATTERN.finditer(text)]
    assert "Italy" in matches