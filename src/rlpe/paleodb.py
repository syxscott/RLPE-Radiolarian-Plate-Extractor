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
                key, type(exc).__name__, exc,
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
                key, type(exc).__name__, exc,
            )

    # ------------------------------------------------------------------ HTTP

    def _http_get_json(self, url: str, params: dict[str, Any], cache_key: str) -> dict[str, Any] | None:
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
            req = urllib.request.Request(full, headers={"User-Agent": self._user_agent, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            logger.warning("PBDB HTTP failed for %s: %s", full, exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive
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
            "show": "attr,app,class",
            "rel": "all",
            "vocab": "pbdb",
            "limit": 1,
        }
        payload = self._http_get_json(
            f"{self.endpoint}/taxa/list.json", params, cache_key
        )
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
            match_score=float(rec.get("match_score", 0.0)) if isinstance(rec.get("match_score"), (int, float)) else 0.0,
            source=payload.get("_source", "paleodb"),
            raw=rec,
        )
        return match

    def lookup_occurrences(self, name: str, max_n: int = 25) -> list[OccurrenceSummary]:
        """Fetch up to ``max_n`` occurrence records for a species.

        Returns an empty list on failure.  Results are cached as a single
        JSON payload per (name, max_n) pair, so subsequent calls hit the
        cache until ``max_n`` changes.
        """
        if not name or not name.strip() or max_n <= 0:
            return []
        clean = name.strip()
        cache_key = _stable_cache_key(f"occs|{clean.lower()}|{max_n}")
        params = {
            "taxon_name": clean,
            "show": "attr,loc,strat",
            "limit": int(max_n),
        }
        payload = self._http_get_json(
            f"{self.endpoint}/occs/list.json", params, cache_key
        )
        if not payload:
            return []
        records = payload.get("records") or []
        out: list[OccurrenceSummary] = []
        for rec in records[:max_n]:
            lat = rec.get("lat")
            lon = rec.get("lng") if "lng" in rec else rec.get("lon")
            try:
                lat_v: float | None = float(lat) if lat is not None else None
            except (TypeError, ValueError):
                lat_v = None
            try:
                lon_v: float | None = float(lon) if lon is not None else None
            except (TypeError, ValueError):
                lon_v = None
            try:
                max_ma = float(rec["max_ma"]) if rec.get("max_ma") is not None else None
            except (TypeError, ValueError, KeyError):
                max_ma = None
            try:
                min_ma = float(rec["min_ma"]) if rec.get("min_ma") is not None else None
            except (TypeError, ValueError, KeyError):
                min_ma = None
            out.append(
                OccurrenceSummary(
                    species_name=clean,
                    occurrence_id=str(rec.get("oid") or rec.get("occurrence_no") or "") or None,
                    collection_id=str(rec.get("cid") or rec.get("collection_no") or "") or None,
                    early_interval=rec.get("early_interval"),
                    late_interval=rec.get("late_interval"),
                    max_ma=max_ma,
                    min_ma=min_ma,
                    locality=rec.get("locality") or rec.get("loc_name"),
                    country=rec.get("cc") or rec.get("country"),
                    latitude=lat_v,
                    longitude=lon_v,
                    formation=rec.get("formation"),
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
