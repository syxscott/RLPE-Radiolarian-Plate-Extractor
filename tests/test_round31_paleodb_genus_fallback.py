"""Phase 31 — PBDB genus→family reverse fallback regression suite.

Round 25 final 5-paper live test showed the PBDB family hit rate at
0.8% (1/128 unique species). Cause: the 5 papers are Cenozoic
extant species; PBDB indexes them at the genus level but not the
species level. Phase 31 adds a genus-level fallback so the
family/order/class_ fields are populated even when the species
match returns None.

This test module follows the ``tests/test_round25_pbdb_integration.py``
pattern (synthetic PBDB JSON, ``sys.path.insert`` + ``_read`` source
guard).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.paleodb import PaleoDB  # noqa: E402
from rlpe.types import TaxonomyMatch  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ============================================================================
# Source-guard tests
# ============================================================================


def test_lookup_genus_method_exists():
    """``PaleoDB.lookup_genus`` must exist."""
    assert hasattr(PaleoDB, "lookup_genus")


def test_paleodb_has_genus_fallback_source_tag():
    """The source="genus_fallback" tag must appear in the PBDB module."""
    src = _read("src/rlpe/paleodb.py")
    assert '"genus_fallback"' in src


def test_pipeline_calls_lookup_genus():
    """``pipeline._attach_paleodb_metadata`` must call ``lookup_genus``
    on species-level miss."""
    src = _read("src/rlpe/pipeline.py")
    assert "lookup_genus" in src
    # And it must be the Phase 31 fallback path: only invoked when
    # ``lookup_species`` returned None
    assert "tax = client.lookup_species(name)" in src
    assert "tax is None and" in src


# ============================================================================
# lookup_genus unit tests (offline mode)
# ============================================================================


def test_lookup_genus_empty_string_returns_none():
    """Empty input must return None (no PBDB call)."""
    c = PaleoDB(offline=True)
    assert c.lookup_genus("") is None


def test_lookup_genus_whitespace_only_returns_none():
    """Whitespace-only input is treated as empty."""
    c = PaleoDB(offline=True)
    assert c.lookup_genus("   ") is None


def test_lookup_genus_offline_returns_none(tmp_path):
    """When offline=True, no network call is made; returns None.

    Uses a tmp cache dir + a unique genus name to avoid any chance
    of hitting a pre-existing cache entry from prior runs."""
    c = PaleoDB(offline=True, cache_dir=tmp_path)
    # Should not raise; returns None
    assert c.lookup_genus(f"UniqueGenus_{tmp_path.name}") is None


def test_lookup_genus_miss_returns_none(monkeypatch, tmp_path):
    """When PBDB returns 0 records, lookup_genus returns None."""
    import json

    class FakeResp:
        def __init__(self, body: bytes):
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):
        return FakeResp(b'{"records": []}')

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    # tmp_path cache_dir ensures this test doesn't collide with
    # any real cache entries.
    c = PaleoDB(offline=False, min_interval=0.0, cache_dir=tmp_path)
    assert c.lookup_genus(f"NonexistentGenus_{tmp_path.name}") is None


def test_lookup_genus_hit_returns_taxonomy_match(monkeypatch, tmp_path):
    """When PBDB returns a genus record, lookup_genus returns a
    TaxonomyMatch tagged with source='genus_fallback'."""
    import json

    # Use a unique genus name that's not in the PBDB cache so the
    # mock urlopen is actually called.
    unique_genus = f"TestGenus_{tmp_path.name}"
    payload = {
        "records": [
            {
                "name": unique_genus,
                "rank": "genus",
                "status": "valid",
                "kingdom": "Chromista",
                "phylum": "Radiozoa",
                "class": ["Polycystinea", 100],
                "order": ["Entactinaria", 200],
                "family": ["Actinommidae", 300],
                "genus": [unique_genus, 1],
                "match_score": 0.95,
            }
        ]
    }

    class FakeResp:
        def __init__(self, body: bytes):
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    body = json.dumps(payload).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        return FakeResp(body)

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    # Use a tmp cache dir so the unique genus doesn't collide with
    # any real cache entries from previous runs.
    c = PaleoDB(offline=False, min_interval=0.0, cache_dir=tmp_path)
    result = c.lookup_genus(unique_genus)
    assert result is not None
    assert isinstance(result, TaxonomyMatch)
    assert result.family == "Actinommidae"
    assert result.order == "Entactinaria"
    assert result.class_ == "Polycystinea"
    assert result.phylum == "Radiozoa"
    assert result.source == "genus_fallback"


def test_lookup_genus_distinct_cache_from_species():
    """``lookup_genus`` and ``lookup_species`` must use distinct cache
    keys so the same string doesn't poison both lookups."""
    import inspect

    src = _read("src/rlpe/paleodb.py")
    # Find the cache keys used in each method
    genus_method_start = src.find("def lookup_genus")
    species_method_start = src.find("def lookup_species")
    # Each method has its own cache key prefix
    assert 'f"genus|{clean.lower()}"' in src, "lookup_genus must use a distinct cache prefix"
    assert 'f"taxa|{clean.lower()}"' in src, "lookup_species must keep its existing taxa| prefix"


# ============================================================================
# Pipeline wiring
# ============================================================================


def test_pipeline_genus_fallback_only_when_species_misses():
    """The genus fallback must ONLY fire when ``lookup_species`` returns
    None. If species match exists but is missing fields, genus
    fallback is not currently wired (Phase 32+ candidate)."""
    src = _read("src/rlpe/pipeline.py")
    # Find the fallback block
    fallback_idx = src.find("lookup_genus")
    # The condition immediately preceding must check tax is None
    pre = src[max(0, fallback_idx - 500) : fallback_idx]
    assert "tax is None" in pre, (
        "Genus fallback must be conditional on tax is None (no species match)"
    )


def test_pipeline_genus_fallback_skips_genus_only_input():
    """A genus-only name (no space) must NOT trigger the genus fallback
    because there's no genus to extract."""
    src = _read("src/rlpe/pipeline.py")
    # The fallback must check for a space
    assert '" " in name' in src, "Genus fallback must check for a space (binomial format)"


def test_pipeline_genus_fallback_does_not_call_lookup_occurrences():
    """Bug-fix M-1: when the species lookup misses and genus fallback
    fills the taxonomy, ``lookup_occurrences`` must NOT be called
    because occurrences are species-specific and would return wrong
    biozone.

    The pipeline gates the occurrence lookup on ``not tax_from_genus``:
    ``occs = client.lookup_occurrences(name, max_n=max_occ) if (tax and
    not tax_from_genus) else []``. We pin that gate here.
    """
    src = _read("src/rlpe/pipeline.py")
    # Find the fallback block. Start a bit before ``lookup_genus`` so
    # the leading comment ("occurrences are species-specific") is
    # included.
    fallback_idx = max(0, src.find("lookup_genus") - 800)
    fallback_end = fallback_idx + 3500
    block = src[fallback_idx:fallback_end]
    # Bug-fix M-1: occurrences must be gated by ``not tax_from_genus``.
    assert "tax_from_genus" in block, (
        "Pipeline must track ``tax_from_genus`` to gate occurrence lookup"
    )
    assert "if tax and not tax_from_genus" in block, (
        "Occurrence lookup must be gated by ``not tax_from_genus`` (M-1 fix)"
    )
    # The comment explaining the design decision may use any of:
    has_comment = any(
        phrase in block
        for phrase in (
            "occurrences are species-specific",
            "do NOT look up occurrences",
            "NOT looked up",
            "are species-specific",
        )
    )
    assert has_comment, (
        f"Genus fallback block must explain why occurrences aren't looked up. Block:\n{block[:800]}"
    )


# ============================================================================
# Backward compatibility
# ============================================================================


def test_lookup_species_unchanged_by_phase31():
    """``lookup_species`` signature + behaviour must NOT change."""
    import inspect

    sig = inspect.signature(PaleoDB.lookup_species)
    assert list(sig.parameters.keys()) == ["self", "name"]
    src = _read("src/rlpe/paleodb.py")
    # The species source tag is still "paleodb", not "genus_fallback"
    assert 'source=payload.get("_source", "paleodb")' in src


def test_paleodb_class_unchanged():
    """``PaleoDB.__init__`` signature must be unchanged (Phase 31 only
    adds ``lookup_genus``)."""
    import inspect

    sig = inspect.signature(PaleoDB.__init__)
    params = list(sig.parameters.keys())
    assert params == ["self", "endpoint", "cache_dir", "min_interval", "timeout", "offline"]
