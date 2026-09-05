"""Regression: audit 2026-09-04 taxon-5 — ``PAPER_WHITELIST`` rules that
rewrite stored species names destroy valid nomenclature.

Three forcing rules in ``ocr_corrections.PAPER_WHITELIST`` were
proved to contradict the project's own gold labels and to silently
rename valid published determinations:

  hollis2006 #6
    "Corythomelissa sp. A" → "Corythomelissa sp. A. B-F36/0"
    injects a sample code (B-F36/0) into the species field. A
    label plus a sample code is NOT a taxon name.

  feng2007
    "Trilonche cimelia" → "Trilonche pseudocimelia"
    production output contains 6 rows of the real "Trilonche
    cimelia" determination; if the layer is ever wired into
    production (it has been inert since 2026-09-04 only because
    nothing calls it), those 6 rows are silently renamed to a
    different species.

  beccaro2006
    "Pseudoeucyrtis sp." → "Pseudoeucyrtis sp. B"
    the genuine open-nomenclature label "Pseudoeucyrtis sp."
    exists in ``data/gold/beccaro2006.jsonl``; forcing it to
    "sp. B" destroys a real nomenclature distinction
    (undetermined species vs named informal morphogroup).

Fix contract: the three forcing rules are REMOVED from
``PAPER_WHITELIST``. A source guard fails if any of them ever
returns. The downstream m3_engine soft-norm rules that strip
"sp. A" / "sp. B" letter-group markers stay intact (they are a
gold-shape accommodation, not a name-rewriter) — but they cannot
produce the species name "Pseudoeucyrtis sp. B" from an input
"Pseudoeucyrtis sp." because the input no longer carries the B.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe import ocr_corrections


class TestForcingRulesRemoved:
    def test_hollis2006_sample_code_no_longer_injected(self):
        # The rule that pasted the sample code into the species
        # field is gone: input must survive untouched.
        out = ocr_corrections.apply_corrections("Corythomelissa sp. A", "hollis2006")
        assert out == "Corythomelissa sp. A", out
        assert "B-F36/0" not in out

    def test_feng2007_cimelia_not_renamed(self):
        out = ocr_corrections.apply_corrections("Trilonche cimelia", "feng2007")
        assert out == "Trilonche cimelia", out
        assert "pseudocimelia" not in out

    def test_beccaro2006_bare_sp_not_renamed_to_sp_b(self):
        out = ocr_corrections.apply_corrections("Pseudoeucyrtis sp.", "beccaro2006")
        assert out == "Pseudoeucyrtis sp.", out
        assert "sp. B" not in out


class TestSafeWhitelistRulesStillActive:
    """Pin that the SURVIVING (non-forcing) whitelist rules keep
    working — the fix removed three rules, not the whole layer."""

    def test_hollis2006_author_suffix_strip(self):
        out = ocr_corrections.apply_corrections("Theocorys? phyzella Foreman", "hollis2006")
        assert "Foreman" not in out, out

    def test_axoprunum_aff_recovery(self):
        out = ocr_corrections.apply_corrections("Axoprunum bispiculum", "hollis2006")
        assert out == "Axoprunum aff. bispiculum", out


class TestSourceGuardRulesRemoved:
    def test_no_forcing_pattern_in_ocr_corrections(self):
        """If any of the three forcing patterns ever re-appears in
        ``ocr_corrections.py``, this guard fires so the rule must
        be re-justified rather than silently re-added."""
        src = (_SRC / "rlpe" / "ocr_corrections.py").read_text(encoding="utf-8")
        forbidden = [
            r"Corythomelissa sp\\\. A\(?!",
            r"Trilonche cimelia\(?!",
            r"Pseudoeucyrtis sp\\\.\(?!",
        ]
        offenders = [pat for pat in forbidden if re.search(pat, src)]
        assert not offenders, (
            "one of the forcing PAPER_WHITELIST rules has returned; "
            f"offending patterns: {offenders}"
        )
