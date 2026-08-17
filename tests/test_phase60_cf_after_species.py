"""Tests for Phase 60 Plan 3 — Bug 3.1: ``cf./aff.`` after species epithet.

The previous ``TAXON_LIKE_PATTERN`` in ``rlpe.association`` only allowed
``cf./aff.`` BEFORE the species epithet (the inner group sat between the
genus and the epithet). Real radiolarian captions (e.g. Bandini 2011 pl08
and pl09) write the comparison species AFTER the epithet:

    "Genus species cf. S. excelsa n. sp."

The old regex matched ``"Genus species"`` and dropped the rest, losing the
``cf. S. excelsa`` context. The fix relaxes the pattern so the trailing
``cf. <Compared>`` block is preserved as a separate taxon entity.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.association import extract_taxa_from_caption  # noqa: E402

# ---------------------------------------------------------------------------
# Bug 3.1: cf./aff. AFTER the epithet must not be silently dropped
# ---------------------------------------------------------------------------


def test_extract_taxa_handles_cf_after_species():
    """Caption "figs 1-3. Genus species cf. S. excelsa n. sp." should yield
    the canonical binomial "Genus species" plus the comparison reference
    "S. excelsa" — both as taxon strings."""
    caption = "figs 1-3. Entactinia compacta cf. S. excelsa n. sp."
    taxa = extract_taxa_from_caption(caption)
    # The main binomial must still be extracted.
    assert "Entactinia compacta" in taxa, taxa
    # The compared species should also be present (the bare name token
    # without the leading "S." author initial). We assert the substring
    # "excelsa" appears somewhere — exact shape depends on regex
    # iteration order, but the species epithet must not be lost.
    assert any("excelsa" in t for t in taxa), taxa


def test_extract_taxa_handles_aff_after_species():
    """Same shape but with "aff." qualifier."""
    caption = "figs 4-5. Pseudoalbaillella scalprata aff. P. longicornis"
    taxa = extract_taxa_from_caption(caption)
    assert "Pseudoalbaillella scalprata" in taxa, taxa
    assert any("longicornis" in t for t in taxa), taxa


def test_extract_taxa_handles_cf_after_species_keeps_main_binomial():
    """The main binomial must still come through even when there's a
    trailing cf. clause — regression guard against the cf. swallowing
    the epithet."""
    caption = "figs 1-3. Archaeodictyomitra apiarium cf. A. mullerriedi"
    taxa = extract_taxa_from_caption(caption)
    assert "Archaeodictyomitra apiarium" in taxa, taxa


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
