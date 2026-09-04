"""Regression: audit 2026-09-04 taxon-8 — ``parse_open_nomenclature``
deleted the cf./aff. qualifier and fused the compared species into a
pseudo-trinomial, so research-eval drivers wrote corrupted names.

The previous behaviour:

    parse_open_nomenclature('Archaeodictyomitra mitra cf. S. excelsa')
        → ('Archaeodictyomitra mitra S. excelsa', 'cf.')

fused the compared species "S. excelsa" onto the canonical binomial
as if it were a subspecies — a name that exists in no literature.

    parse_open_nomenclature('Zhamoidellum cf. testatum')
        → ('Zhamoidellum testatum', 'cf.')

promoted an uncertain determination to the definite species, and
the caller then re-emitted ``f"{sp}{qual_str}" = 'Zhamoidellum
testatum cf.'`` — open nomenclature puts cf. BEFORE the epithet, not
after, so the position was wrong too. Every F1 produced by
``scripts/run_research_eval.py`` and ``scripts/run_v19_baseline.py``
flowed through this corruption.

Fix contract: ``parse_open_nomenclature`` returns the original species
string unchanged (only the qualifier label is reported separately);
the callers use the species field as-is instead of re-stitching the
qualifier onto the end. The original label-vs-content distinction
("X cf. Y") and the comparison form ("X species cf. S. excelsa")
both survive intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
# scripts/ is not normally on PYTHONPATH; add it for direct import of
# the callers so we can pin they no longer re-stitch qualifiers.
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from post_process import parse_open_nomenclature  # noqa: E402


class TestParseOpenNomenclaturePreservesContent:
    def test_definite_species_unchanged(self):
        assert parse_open_nomenclature("Follicucullus scholasticus") == (
            "Follicucullus scholasticus",
            None,
        )

    def test_cf_label_preserved(self):
        # No deletion, no fusion — cf. testatum survives verbatim.
        assert parse_open_nomenclature("Zhamoidellum cf. testatum") == (
            "Zhamoidellum cf. testatum",
            "cf.",
        )

    def test_aff_label_preserved(self):
        assert parse_open_nomenclature("Pseudoeucyrtis aff. hannai") == (
            "Pseudoeucyrtis aff. hannai",
            "aff.",
        )

    def test_comparison_form_with_abbreviated_genus_preserved(self):
        # The crucial real shape: 'cf. S. excelsa' must NOT collapse
        # into 'mitra S. excelsa'.
        assert parse_open_nomenclature("Archaeodictyomitra mitra cf. S. excelsa") == (
            "Archaeodictyomitra mitra cf. S. excelsa",
            "cf.",
        )

    def test_open_nom_sp_passes_through(self):
        assert parse_open_nomenclature("Archaeodictyomitra sp.") == (
            "Archaeodictyomitra sp.",
            None,
        )

    def test_bare_genus_passes_through(self):
        assert parse_open_nomenclature("Stichomitra") == ("Stichomitra", None)

    def test_none_and_empty_inputs(self):
        # The historical contract: pure-whitespace / empty / None all
        # collapse to (None, None) so the caller can detect "no
        # species" with a single ``if sp is None`` check.
        assert parse_open_nomenclature(None) == (None, None)
        assert parse_open_nomenclature("") == (None, None)
        assert parse_open_nomenclature("   ") == (None, None)


class TestCallerNoLongerRestitchesQualifier:
    """The two research-eval callers (``scripts/run_research_eval.py``
    and ``scripts/run_v19_baseline.py``) used to assemble the output
    row's species field as ``f"{sp}{qual_str}"``, which re-attached
    the qualifier to the END of the species even when it was already
    inside (or should be there at all). Pin that the surviving row
    keeps the exact input species string."""

    def test_research_eval_caller_does_not_restitch(self):
        import importlib

        # Importing the module triggers module-level references; if
        # the old ``f"{sp}{qual_str}"`` assembly is still in source we
        # need to know — assert by AST scan that the bad pattern is
        # gone.
        import ast

        path = _ROOT / "scripts" / "run_research_eval.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                src = ast.unparse(node)
                if "qual_str" in src and '"{sp}' in src:
                    raise AssertionError(
                        "run_research_eval.py still re-stitches the "
                        "qualifier onto the species field: " + src
                    )

    def test_v19_baseline_caller_does_not_restitch(self):
        import ast

        path = _ROOT / "scripts" / "run_v19_baseline.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                src = ast.unparse(node)
                if "qual_str" in src and '"{sp}' in src:
                    raise AssertionError(
                        "run_v19_baseline.py still re-stitches the "
                        "qualifier onto the species field: " + src
                    )