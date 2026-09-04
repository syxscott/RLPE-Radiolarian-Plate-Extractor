"""Regression: audit 2026-09-04 taxon-6 — the ``Archaeo`` ↔ ``Archeo``
spelling fold in :func:`rlpe.evaluation.metrics._norm_species` ran in
the WRONG direction. It mapped ``Archaeo`` → ``Archeo`` (treating the
orthographic variant as canonical) but never the reverse, so a gold
row in the standard ``Archaeo`` form did not match a pred row in the
variant ``Archeo`` form.

Both spellings are legitimate in the literature (De Wever 2001 uses
both) but the accepted form for nomenclatural comparison is
``Archaeo`` (the Greek prefix is ``archaîos``, transliterated
``archaeo-``). Folding to that accepted form — in either direction —
is what F1 comparison should do.

Fix contract: both inputs (``Archaeo`` and ``Archeo``) are normalised
to ``Archaeo`` so any pair — pred=Archeo vs gold=Archaeo, pred=Archaeo
vs gold=Archeo, both sides same, case variations — compare equal after
the lowercasing step.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.evaluation.metrics import _norm_species  # noqa: E402


class TestArchaeoArcheoBidirectionalFold:
    def test_pred_archeological_to_gold_archaeological(self):
        # Bandini pl02 pred (LLM-fused): "Archeodictyomitra vulgaris"
        # Gold: "Archaeodictyomitra vulgaris"
        # Both should normalise to the same canonical form.
        a = _norm_species("Archeodictyomitra vulgaris")
        b = _norm_species("Archaeodictyomitra vulgaris")
        assert a == b
        assert a == "Archaeodictyomitra vulgaris"

    def test_pred_archaeological_to_gold_archeological(self):
        # Inverse direction: pred in the standard form, gold in the
        # variant. This is the case the audit 2026-09-04 taxon-6
        # flagged — the previous one-way fold only matched when the
        # pred was the variant, biasing the comparison.
        a = _norm_species("Archaeodictyomitra vulgaris")
        b = _norm_species("Archeodictyomitra vulgaris")
        assert a == b

    def test_qualifier_preserved_through_fold(self):
        # "Archeo… cf. X" must still match "Archaeo… cf. X"
        a = _norm_species("Archeodictyomitra cf. vulgaris")
        b = _norm_species("Archaeodictyomitra cf. vulgaris")
        assert a == b

    def test_unrelated_genus_not_affected(self):
        # "Theocorys" must not be touched.
        a = _norm_species("Theocorys phyzella")
        b = _norm_species("Theocorys phyzella")
        assert a == b
        assert a == "Theocorys phyzella"
