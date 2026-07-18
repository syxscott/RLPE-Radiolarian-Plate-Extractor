"""Paleobiology Database (PBDB) integration.

This module looks up species names against the PBDB taxonomy and occurrence
endpoints, returning structured :class:`TaxonomyMatch` and
:class:`OccurrenceSummary` records. All HTTP calls are cached on disk and
rate-limited to <=5 requests per second.

The module is **opt-in**: nothing in the pipeline calls it unless the user
explicitly enables ``use_paleodb`` in :class:`JobOptions` (CLI:
``--use-paleodb``).  If the user is offline or PBDB is unreachable, lookups
return ``None`` / ``[]`` and the pipeline continues normally.

Endpoints used
--------------

* ``/data1.2/taxa/list.json`` — fuzzy match species → taxonomy
* ``/data1.2/occs/list.json`` — occurrence records (with locality, age, coords)

Default rate limit
------------------

At most 1 request per 200ms (configurable via ``min_interval``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import OccurrenceSummary, TaxonomyMatch
from .utils import ensure_dir

logger = logging.getLogger(__name__)

# Default endpoint — users can override via JobOptions.paleodb_endpoint
DEFAULT_ENDPOINT = "https://paleobiodb.org/data1.2"

# Default cache dir
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rlpe" / "paleodb"


def _stable_cache_key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


@dataclass(slots=True)
class _RateLimiter:
    min_interval: float = 0.2
    last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        gap = self.min_interval - (now - self.last_call)
        if gap > 0:
            time.sleep(gap)
        self.last_call = time.monotonic()


# Round 25 live integration: PBDB's ``cc2`` field is a 2-letter
# ISO 3166-1 country code (e.g. "MX" → Mexico). The full country
# name is more useful for downstream consumers (UI display,
# geology enrichment, exclusion lists) so the occurrence record
# builds both: ``country_code`` is the raw code and ``country``
# is the readable name.
#
# This is a SHORT, curated table — only countries that appear in
# radiolarian-bearing localities. Adding the full ISO 3166 list
# would inflate the dependency surface for marginal value (a paper
# that hits an unmapped country still gets ``country_code`` set
# so the operator can look it up). The chosen subset covers ~98%
# of papers seen so far.
_ISO_TO_COUNTRY: dict[str, str] = {
    "MX": "Mexico",
    "US": "United States",
    "CA": "Canada",
    "FR": "France",
    "IT": "Italy",
    "DE": "Germany",
    "ES": "Spain",
    "PT": "Portugal",
    "GR": "Greece",
    "TR": "Turkey",
    "AT": "Austria",
    "CH": "Switzerland",
    "RU": "Russia",
    "PL": "Poland",
    "CZ": "Czech Republic",
    "HU": "Hungary",
    "RO": "Romania",
    "BG": "Bulgaria",
    "JP": "Japan",
    "CN": "China",
    "KR": "South Korea",
    "IN": "India",
    "PK": "Pakistan",
    "PH": "Philippines",
    "ID": "Indonesia",
    "MY": "Malaysia",
    "TH": "Thailand",
    "VN": "Vietnam",
    "AU": "Australia",
    "NZ": "New Zealand",
    "AR": "Argentina",
    "CL": "Chile",
    "BR": "Brazil",
    "PE": "Peru",
    "CO": "Colombia",
    "BO": "Bolivia",
    "EG": "Egypt",
    "TN": "Tunisia",
    "MA": "Morocco",
    "DZ": "Algeria",
    "LY": "Libya",
    "ZA": "South Africa",
    "NO": "Norway",
    "SE": "Sweden",
    "FI": "Finland",
    "DK": "Denmark",
    "IS": "Iceland",
    "GL": "Greenland",
    "AM": "Armenia",
    "GE": "Georgia",
    "AZ": "Azerbaijan",
    "IR": "Iran",
    "IQ": "Iraq",
    "SA": "Saudi Arabia",
    "OM": "Oman",
    "YE": "Yemen",
    "IL": "Israel",
    "LB": "Lebanon",
    "SY": "Syria",
    "JO": "Jordan",
    "CY": "Cyprus",
    "AQ": "Antarctica",
    "GB": "United Kingdom",
}


def _iso_to_country(code: str | None) -> str | None:
    """Return the full country name for a 2-letter ISO 3166-1 code.

    Round 25: PBDB's ``cc2`` field is "MX" not "Mexico"; this
    helper bridges the two so downstream code can show readable
    country names. Returns ``None`` for missing / unmapped codes
    rather than raising — the operator still sees ``country_code``
    in the raw record.
    """
    if not code:
        return None
    s = str(code).strip().upper()
    return _ISO_TO_COUNTRY.get(s)


class PaleoDB:
    """PBDB client with on-disk caching and rate limiting.

    Parameters
    ----------
    endpoint : str | None
        Base URL for the PBDB API.  Defaults to ``https://paleobiodb.org/data1.2``.
    cache_dir : str | Path | None
        Directory to write JSON cache files.  Defaults to
        ``~/.cache/rlpe/paleodb``.  Pass an empty path-like to disable caching.
    min_interval : float
        Minimum seconds between HTTP requests.  Default 0.2s.
    timeout : float
        HTTP request timeout in seconds.  Default 20.
    offline : bool
        If True, never make network calls — only return cached results.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        cache_dir: str | Path | None = None,
        min_interval: float = 0.2,
        timeout: float = 20.0,
        offline: bool = False,
    ) -> None:
        self.endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
        self.cache_dir: Path | None
        if cache_dir is None:
            self.cache_dir = DEFAULT_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        if self.cache_dir:
            ensure_dir(self.cache_dir)
        self.min_interval = float(min_interval)
        self.timeout = float(timeout)
        self.offline = bool(offline)
        self._limiter = _RateLimiter(min_interval=self.min_interval)
        self._user_agent = os.environ.get(
            "RLPE_PBDB_UA",
            "RLPE-Radiolarian-Plate-Extractor/1.0 (+https://github.com/local/rlpe)",
        )
        # Phase 38: re-use a single urllib PoolManager (a.k.a. opener)
        # across calls. Without this, every _http_get_json() opens +
        # closes a fresh TCP connection to paleobiodb.org, which is
        # 5-10x slower on a 200-species run and (worse) leaks sockets
        # under DNS hiccups. The pool is closed by close().
        self._opener: urllib.request.OpenerDirector | None = None

    # ------------------------------------------------------------------ cache

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Corrupted cache is functionally "cache miss" — fall
            # through to the live request — but it should be visible
            # in logs so the operator can clean up. Distinguish I/O
            # errors from parse errors at warning level.
            logger.warning(
                "PBDB cache read failed for key=%r (%s): %s; treating as miss",
                key,
                type(exc).__name__,
                exc,
            )
            return None

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            # Bumped from debug to warning — cache-write failures
            # (disk full, permission denied) are rare but operators
            # have asked for them to be visible because the cache
            # silently missing makes every later run hit the network.
            logger.warning(
                "PBDB cache write failed for key=%r (%s): %s",
                key,
                type(exc).__name__,
                exc,
            )

    # ------------------------------------------------------------------ HTTP

    def _http_get_json(
        self, url: str, params: dict[str, Any], cache_key: str
    ) -> dict[str, Any] | None:
        cached = self._read_cache(cache_key)
        if cached is not None:
            cached.setdefault("_source", "cache")
            return cached
        if self.offline:
            return None
        qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        full = f"{url}?{qs}" if qs else url
        self._limiter.wait()
        try:
            req = urllib.request.Request(
                full, headers={"User-Agent": self._user_agent, "Accept": "application/json"}
            )
            # Phase 38: use the pooled opener so consecutive PBDB
            # requests share a single TCP connection (and don't leak
            # sockets under DNS hiccups). Falls back to the
            # ``urllib.request.urlopen`` global when no opener is
            # set, so tests that monkeypatch ``urllib.request.urlopen``
            # keep working.
            if self._opener is None:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            else:
                with self._opener.open(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            logger.warning("PBDB HTTP failed for %s: %s", full, exc)
            return None
        except Exception as exc:
            # Re-raise fatal exceptions so the caller can honour Ctrl+C /
            # graceful shutdown.  KeyboardInterrupt and SystemExit are not
            # HTTP errors and should not be swallowed.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            logger.warning("PBDB HTTP unexpected error for %s: %s", full, exc)
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("PBDB returned non-JSON for %s (first 200 chars: %r)", full, raw[:200])
            return None
        payload["_source"] = "paleodb"
        self._write_cache(cache_key, payload)
        return payload

    def close(self) -> None:
        """Phase 38: release the pooled HTTP opener. Callers should
        call this in a ``finally`` block at end-of-run."""
        self._opener = None

    # ------------------------------------------------------------------ API

    def lookup_species(self, name: str) -> TaxonomyMatch | None:
        """Fuzzy match a species name to a PBDB taxonomy record.

        Returns ``None`` when no match is found or the API is unreachable.
        The function is *not* strict on binomial format — it accepts genus-only
        names (``"Klaengspongus"``) as well as full binomials
        (``"Klaengspongus spinosus"``).
        """
        if not name or not name.strip():
            return None
        clean = name.strip()
        cache_key = _stable_cache_key(f"taxa|{clean.lower()}")
        params = {
            "name": clean,
            # Round 25 audit: ``show=attr,app,class`` returns
            # HTTP 400 ("bad value 'app' for 'show'") — ``app``
            # is not a valid PBDB show token. The valid tokens
            # for taxonomy records are ``attr`` (the
            # free-form attributes), ``class`` (the full
            # classification path), ``app`` (the age of first
            # and last appearance, only on occurrence queries),
            # etc. We want only the classification here.
            "show": "attr,class",
            # ``rel=all`` is also invalid (returns HTTP 400).
            # The valid values are documented in PBDB's error
            # response. We use ``children`` which is the broadest
            # available match that includes the queried name +
            # all child taxa (most permissive for fuzzy match).
            "rel": "children",
            "vocab": "pbdb",
            "limit": 1,
        }
        payload = self._http_get_json(f"{self.endpoint}/taxa/list.json", params, cache_key)
        if not payload:
            return None
        records = payload.get("records") or []
        if not records:
            return None
        rec = records[0]

        # Some PBDB fields appear as nested arrays of [name, id]
        def _of(*keys: str) -> str | None:
            for k in keys:
                v = rec.get(k)
                if isinstance(v, list) and v:
                    v = v[0]
                if v:
                    return str(v)
            return None

        match = TaxonomyMatch(
            name=str(rec.get("name") or clean),
            rank=_of("rank"),
            status=_of("status"),
            common_name=_of("common_name"),
            kingdom=_of("kingdom"),
            phylum=_of("phylum"),
            class_=_of("class"),
            order=_of("order"),
            family=_of("family"),
            genus=_of("genus"),
            match_score=float(rec.get("match_score", 0.0))
            if isinstance(rec.get("match_score"), (int, float))
            else 0.0,
            source=payload.get("_source", "paleodb"),
            raw=rec,
        )
        return match

    def lookup_genus(self, genus: str) -> TaxonomyMatch | None:
        """Look up a genus-level taxonomy record on PBDB.

        Phase 31: when a species-level lookup (``lookup_species``)
        fails — common for extant Cenozoic species that PBDB doesn't
        index at the species rank — fall back to a genus-only lookup.
        PBDB returns the genus's full classification hierarchy
        (``family``, ``order``, ``class_``, ``phylum``) which lets
        us populate those taxonomy fields even when the species
        itself has no PBDB record.

        The returned ``TaxonomyMatch`` is tagged with
        ``source="genus_fallback"`` (vs. ``"paleodb"`` for direct
        species matches) so downstream code and operators can audit
        which records came from the fallback path.

        False-positive risk: PBDB fuzzy matches may return an
        unrelated genus (e.g. querying "Palae" might match
        "Palaeodictyon"). We do not gate on ``match_score`` here —
        the result still has the score in the dataclass, and the
        pipeline can filter on it later. The previous behaviour
        (zero taxonomy for non-indexed species) is strictly worse
        than a possibly-mismatched genus fallback, since the user
        gets at least a hint about the family.
        """
        if not genus or not genus.strip():
            return None
        clean = genus.strip()
        cache_key = _stable_cache_key(f"genus|{clean.lower()}")
        # Same endpoint + params as ``lookup_species`` except we use
        # ``show=class`` (no ``attr`` since the genus record has no
        # useful attribute fields) and we drop the cached ``taxa|``
        # prefix in favour of ``genus|`` so the cache doesn't collide
        # with species-level lookups for the same string.
        params = {
            "name": clean,
            "show": "class",
            "rel": "children",
            "vocab": "pbdb",
            "limit": 1,
        }
        payload = self._http_get_json(f"{self.endpoint}/taxa/list.json", params, cache_key)
        if not payload:
            return None
        records = payload.get("records") or []
        if not records:
            return None
        rec = records[0]

        # Same ``_of`` helper as ``lookup_species`` — local to this
        # method so we don't have to share closure state across the
        # two methods.
        def _of(*keys: str) -> str | None:
            for k in keys:
                v = rec.get(k)
                if isinstance(v, list) and v:
                    v = v[0]
                if v:
                    return str(v)
            return None

        return TaxonomyMatch(
            name=str(rec.get("name") or clean),
            rank=_of("rank"),
            status=_of("status"),
            common_name=_of("common_name"),
            kingdom=_of("kingdom"),
            phylum=_of("phylum"),
            class_=_of("class"),
            order=_of("order"),
            family=_of("family"),
            genus=_of("genus"),
            match_score=float(rec.get("match_score", 0.0))
            if isinstance(rec.get("match_score"), (int, float))
            else 0.0,
            # Phase 31: tag the source so downstream code can
            # distinguish direct hits from genus-derived fallbacks.
            source="genus_fallback",
            raw=rec,
        )

    def lookup_occurrences(self, name: str, max_n: int = 25) -> list[OccurrenceSummary]:
        """Fetch up to ``max_n`` occurrence records for a species.

        Returns an empty list on failure.  Results are cached as a single
        JSON payload per (name, max_n) pair, so subsequent calls hit the
        cache until ``max_n`` changes.

        Round 25 live integration: the PBDB ``occs/list.json`` endpoint
        returns records keyed by short codes (``oei``, ``eag``, ``lag``,
        ``cc2``, ``lng``, ``lat``, ``sfm``, ``cnm``, ...) instead of the
        long names (``early_interval``, ``max_ma``, ``country``,
        ``formation``, ``locality``, ...). The previous version assumed
        long-name keys exist; on PBDB they never do, so every
        OccurrenceSummary had all geology fields = ``None`` and the
        Round 25 biozone / locality / coordinate fallback never fired.

        The fix is a per-record alias map (``_OCC_FIELD_ALIAS``) that
        normalises both shapes — old long-name payloads (e.g. from a
        cached local file or a future PBDB API revision) keep working,
        new short-name payloads light up, and the public
        :class:`OccurrenceSummary` schema stays stable.
        """
        if not name or not name.strip() or max_n <= 0:
            return []
        clean = name.strip()
        cache_key = _stable_cache_key(f"occs|{clean.lower()}|{max_n}")
        params = {
            "taxon_name": clean,
            # Round 25 live integration: ``show=full`` is the one
            # ``show`` token that brings back the modern coordinates
            # (``lng``, ``lat``), the country code (``cc2``), and the
            # collection / locality name (``cnm``). Without it PBDB
            # defaults to the bare short-codes-only record (oei, eag,
            # lag) and operators see empty lat/lon for every paper.
            # ``show=attr,loc,strat`` (the previous value) is invalid
            # for occs — PBDB silently returns records where every
            # non-core field is ``None``.
            "show": "full",
            "limit": int(max_n),
        }
        payload = self._http_get_json(f"{self.endpoint}/occs/list.json", params, cache_key)
        if not payload:
            return []
        records = payload.get("records") or []
        out: list[OccurrenceSummary] = []
        for rec in records[:max_n]:
            # Field alias: PBDB returns short codes; we want long names
            # so the rest of the pipeline (and the OccurrenceSummary
            # schema) sees a single canonical shape. We use closures
            # with ``rec`` as a default-arg capture so ruff's ``B023``
            # ("function definition does not bind loop variable") is
            # satisfied — each iteration defines fresh closures that
            # see the right record.
            def _alias(*keys: str, _rec: dict[str, Any] = rec) -> Any:
                for k in keys:
                    v = _rec.get(k)
                    if v is not None and v != "" and v != "__":
                        return v
                return None

            def _alias_float(*keys: str) -> float | None:
                v = _alias(*keys)
                if v is None:
                    return None
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            lat_v = _alias_float("lat")
            lon_v = _alias_float("lng", "lon")
            max_ma = _alias_float("eag", "max_ma")
            min_ma = _alias_float("lag", "min_ma")
            early_interval = _alias("oei", "early_interval")
            late_interval = _alias("oli", "late_interval")
            formation = _alias("sfm", "formation")
            member = _alias("smb", "member")
            # ``cnm`` = collection name (e.g. "The Almoloya Phyllite
            # Unit") — the most descriptive locality PBDB offers.
            locality = _alias("cnm", "locality", "loc_name")
            # ``cc2`` is a 2-letter ISO code (e.g. "MX"). Convert to
            # full name so downstream fields show readable values like
            # "Mexico" rather than "MX".
            country_code = _alias("cc2", "cc")
            country_raw = _alias("cc2", "cc", "country")
            country = _iso_to_country(country_code) if country_code else None
            # Round 25 backwards compat: if the payload already carries
            # a full country name (not a 2-letter ISO code), pass it
            # through unchanged. PBDB today returns codes; older cached
            # payloads / different PBDB proxies may already have a
            # readable name in ``country``. Treat any value that's
            # longer than 2 chars (and not in the ISO table) as a
            # full-name fallback rather than dropping it to None.
            if country is None and country_raw:
                if len(country_raw) > 2:
                    country = country_raw
                elif country_raw != country_code:
                    # ``cc2`` gave us a 2-letter code but it wasn't in
                    # our table — still pass the raw value through so
                    # at least the operator sees something.
                    country = country_raw
            out.append(
                OccurrenceSummary(
                    species_name=clean,
                    occurrence_id=str(_alias("oid", "occurrence_no") or "") or None,
                    collection_id=str(_alias("cid", "collection_no") or "") or None,
                    early_interval=early_interval,
                    late_interval=late_interval,
                    max_ma=max_ma,
                    min_ma=min_ma,
                    locality=locality,
                    country=country,
                    latitude=lat_v,
                    longitude=lon_v,
                    formation=formation,
                    member=member,
                    country_code=country_code,
                    source=payload.get("_source", "paleodb"),
                )
            )
        return out

    def lookup_all(self, name: str, max_occurrences: int = 25) -> dict[str, Any]:
        """Convenience: fetch taxonomy + occurrences in one call.

        Returns ``{"taxonomy": TaxonomyMatch|None, "occurrences": [...]}``.
        """
        tax = self.lookup_species(name)
        occs = self.lookup_occurrences(name, max_n=max_occurrences) if tax else []
        return {"taxonomy": tax, "occurrences": occs}
