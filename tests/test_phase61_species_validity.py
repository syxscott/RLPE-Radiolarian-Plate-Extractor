"""Phase 61 Plan 4 (Bug 4.2): LLM-first hybrid must NOT overwrite correct
rule-extracted species with invalid LLM hallucinated genus names.

The hybrid fill path in ``pipeline._process_region`` overwrites an LLM
row's species with the matched caption-parser pair when present. But the
caption-parser pair can itself be a hallucinated "Genus dubious" that
passes ``_taxon_parts`` shape but the genus name is a known author
surname (e.g. "Foreman 1995"). The new ``_is_valid_species`` guard
rejects those before they pollute downstream eval.
"""
from __future__ import annotations

import pytest

from rlpe.taxon import _is_valid_species


def test_valid_species_passes():
    assert _is_valid_species("Genus species") is True


def test_author_surname_genus_rejected():
    """"Foreman" is a known author surname → must be rejected as genus."""
    assert _is_valid_species("Foreman species") is False


def test_dubious_genus_rejected():
    assert _is_valid_species("Dubious species") is False


def test_none_passes_through():
    """None / empty / non-string must NOT be flagged as an invalid species
    (those are normal "no species yet" states the rule extractor emits)."""
    assert _is_valid_species(None) is True
    assert _is_valid_species("") is True
    assert _is_valid_species("   ") is True


def test_shape_failure_rejected():
    """A single-token string cannot be a valid binomial species."""
    assert _is_valid_species("Genus") is False
    # Lower-case "genus" looks like a normal English word, not a proper
    # Latin genus name; reject (conservative behaviour).
    assert _is_valid_species("genus speciesa extra") is False
