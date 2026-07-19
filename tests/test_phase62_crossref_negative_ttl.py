"""Phase 62 Plan 5 (Bug 5.14): Crossref negative cache TTL = 60s.

The Crossref DOI → journal cache used a single 1-hour TTL for both
positive and negative entries. A transient Crossref outage at the
start of a batch run would tag every paper in that batch with
``journal=None`` for the next hour — even after Crossref recovered.

The fix: split the TTLs.
  * Positive cache (real journal name) → 1 hour (unchanged).
  * Negative cache (None / network error / non-200) → 60 seconds.

The negative TTL is short enough that a transient outage clears
within one minute, but long enough to dedupe a single paper's 5
species of bad DOIs into 1 network call.

The test asserts:
  * The two TTL constants exist with the expected values.
  * A cache entry stored as ``None`` (negative) expires after 60s.
  * A cache entry stored as a real journal string (positive)
    expires after 1 hour.
"""
from __future__ import annotations

import time
from unittest.mock import patch

from rlpe.paper_metadata_cleanup import (
    _CROSSREF_CACHE,
    _CROSSREF_NEGATIVE_TTL_SEC,
    _CROSSREF_POSITIVE_TTL_SEC,
    _crossref_get_journal,
)


class _FakeResponse:
    """Minimal stand-in for requests.Response used by the
    _crossref_get_journal mock."""

    def __init__(self, status_code: int, json_payload: dict):
        self.status_code = status_code
        self._json = json_payload

    def json(self):
        return self._json


def setup_function(_):
    """Clear the cache before each test."""
    _CROSSREF_CACHE.clear()


def test_negative_ttl_is_60_seconds():
    """The negative cache TTL must be 60 seconds."""
    assert _CROSSREF_NEGATIVE_TTL_SEC == 60


def test_positive_ttl_is_1_hour():
    """The positive cache TTL must be 1 hour (3600 seconds)."""
    assert _CROSSREF_POSITIVE_TTL_SEC == 3600


def test_negative_cache_expires_after_60s():
    """A None cache entry must expire after 60 seconds — long enough
    to dedupe, short enough to recover from outages."""
    import sys
    fake_requests = type(sys)("fake_requests")
    fake_requests.get = lambda *a, **kw: _FakeResponse(
        200, {"message": {"container-title": ["Test Journal"]}}
    )
    _CROSSREF_CACHE["10.1234/test-doi"] = (None, time.time() - 61)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _crossref_get_journal("10.1234/test-doi")
    assert out == "Test Journal", (
        f"negative cache did not expire after 60s; "
        f"got out={out!r}"
    )


def test_positive_cache_expires_after_1_hour():
    """A positive cache entry must still be valid after 60s but
    expired after 3601s."""
    doi = "10.1234/positive-doi"
    # Stored 100 seconds ago (within negative TTL window but also
    # within positive TTL window).
    _CROSSREF_CACHE[doi] = ("Real Journal", time.time() - 100)
    out = _crossref_get_journal(doi)
    assert out == "Real Journal", (
        f"positive cache hit lost too early; got {out!r}"
    )


def test_positive_cache_expires_after_1_hour_full():
    """A positive cache entry stored 3601s ago must miss the cache."""
    import sys
    fake_requests = type(sys)("fake_requests")
    fake_requests.get = lambda *a, **kw: _FakeResponse(
        200, {"message": {"container-title": ["Fresh Journal"]}}
    )
    doi = "10.1234/positive-doi-old"
    _CROSSREF_CACHE[doi] = ("Real Journal", time.time() - 3601)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _crossref_get_journal(doi)
    assert out == "Fresh Journal", (
        f"positive cache did not expire after 3601s; got {out!r}"
    )