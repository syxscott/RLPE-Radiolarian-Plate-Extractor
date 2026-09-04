"""Regression: audit 2026-09-04 taxon-1 — ``_is_valid_species`` rejected
the real open-nomenclature shapes that make up ~22% of the gold rows
(134/612), so ``pipeline.py``'s hybrid-fill loop silently dropped them
(``skipped_invalid``).

Root cause: the open-nomenclature set membership test compared the
WHOLE qualifier phrase against single words — ``"cf. tumandae"``
never equals ``"cf"`` — so every ``Genus cf. species`` /
``Genus gr. x`` / ``Genus sp. N`` row failed the shape check. Bare
genera (a legitimate radiolarian citation form, e.g. "Stichomitra")
and the bandini 2011 ``"Genus gen"`` convention failed too.

The hallucination defences must NOT regress: author-surname genera
("Foreman 1995"), placeholder tokens ("Dubious", "indeterminate"),
and bare "n." without a paired epithet stay invalid.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.taxon import _is_valid_species


class TestGoldShapesNowValid:
    """Shapes taken directly from gold rows the pipeline used to drop."""

    def test_cf_comparison_reference(self):
        assert _is_valid_species("Archaeodictyomitra cf. tumandae") is True

    def test_aff_comparison_reference(self):
        assert _is_valid_species("Pseudoeucyrtis aff. hannai") is True

    def test_group_marker_with_epithet_letter(self):
        assert _is_valid_species("Haliomma gr. b") is True

    def test_numbered_open_nomenclature(self):
        assert _is_valid_species("Entactinia sp. 1") is True

    def test_bare_genus(self):
        assert _is_valid_species("Stichomitra") is True

    def test_bandini_gen_convention(self):
        assert _is_valid_species("Spumellaria gen") is True

    def test_cf_with_abbreviated_reference(self):
        # bandini 2011 pl08/pl09 real shape: "Genus species cf. S. excelsa"
        assert _is_valid_species("Archaeodictyomitra sp. cf. S. excelsa") is True


class TestHallucinationDefencesIntact:
    def test_author_surname_genus_still_invalid(self):
        assert _is_valid_species("Foreman 1995") is False

    def test_placeholder_still_invalid(self):
        assert _is_valid_species("Dubious") is False
        assert _is_valid_species("indeterminate") is False
        assert _is_valid_species("unknown") is False

    def test_bare_n_dot_still_invalid(self):
        # "n." without a paired "sp." is incomplete ICZN nomenclature —
        # including when _taxon_parts parses it into the epithet slot.
        assert _is_valid_species("Archaeodictyomitra n.") is False


class TestPipelineIntegration:
    def test_pipeline_hybrid_loop_uses_validity_check(self):
        """The drop path (skipped_invalid) must still route through
        _is_valid_species — the fix relaxes the shape check, it does
        not remove the guard."""
        import inspect

        from rlpe import pipeline

        src = inspect.getsource(pipeline)
        assert "skipped_invalid" in src
        assert "_is_valid_species" in src
