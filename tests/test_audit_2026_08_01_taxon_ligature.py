"""Regression tests for audit 2026-08-01 batch W2 — taxon C11 ligature offset."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.taxon import TaxonRecognizer  # noqa: E402


class TestTaxonLigatureOffset:
    @staticmethod
    def _entity_for(text: str, expected: str):
        entities = TaxonRecognizer()._fallback_predict(text)
        return next(entity for entity in entities if expected in entity.text)

    def test_ae_uppercase_offset(self):
        text = "Æquilon sp. nov."

        entity = self._entity_for(text, "quilon")

        assert entity.text == "Æquilon sp. nov."
        assert entity.start == 0
        assert entity.end == len(text)
        assert text[entity.start : entity.end] == entity.text

    def test_oe_lowercase_offset(self):
        text = "Cenosphaera cœlenterate"

        entity = self._entity_for(text, "lenterate")

        assert entity.text == text
        assert entity.start == 0
        assert entity.end == len(text)
        assert text[entity.start : entity.end] == entity.text

    def test_no_ligature_unchanged(self):
        text = "Entactinia compacta"

        entity = self._entity_for(text, text)

        assert entity.text == text
        assert entity.start == 0
        assert entity.end == len(text)
        assert text[entity.start : entity.end] == entity.text
