"""Round 20 paper-metadata cleanup helpers.

Three systemic issues were identified in the 4-paper sampling:

  1. **Title extraction returns garbage.**
     Bandini 2006 → ``title="001_020"`` (page numbers)
     Danelian 2006 → ``title="035_048"`` (page numbers)
     Bragin 2025 → ``title="StrtEng2470030Bragin.fm"`` (filename)
     The GROBID TEI parser failed silently and returned whatever
     text happened to be in the ``<title>`` element. We detect the
     three known garbage patterns and flag the paper for review.

  2. **Author markers leak into the author list.**
     Bragin 2025 → ``authors=["Input2"]``
     The OpenDataLoader fulltext extraction returns placeholder
     strings when it cannot identify a real author. We strip the
     known markers so they don't pollute the author list.

  3. **Journal is missing or wrong.**
     Bragin 2025 → ``journal="Scale"`` (a stray word from the
     publisher field, not the journal name). When GROBID misses the
     journal and we have a DOI, we fall back to the Crossref public
     API (``https://api.crossref.org/works/{doi}``) which returns
     ``container-title[0]``. The lookup is cached per-process.

These helpers are imported by ``paper_records_from_matches`` in
``converters.py`` so the fixes apply to every export.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# Phase 54 audit m19 — TTL constant for the Crossref cache.
# 1 hour: long enough to dedupe a single run's repeated DOI lookups
# (one paper with 5 species of bad DOIs hits the network once instead
# of 5 times), short enough that transient Crossref outages self-heal
# without a process restart.
_CROSSREF_CACHE_TTL_SEC: int = 3600

# Phase 62 Plan 5 (Bug 5.14): split the Crossref TTL into positive
# (real journal) and negative (None / non-200 / network error)
# buckets. A transient Crossref outage at the start of a batch
# run was tagging every paper with ``journal=None`` for the next
# hour. The negative TTL of 60s is short enough to recover from
# outages within one minute but long enough to dedupe a single
# paper's 5 bad-DOI retries into 1 network call.
_CROSSREF_POSITIVE_TTL_SEC: int = 3600
_CROSSREF_NEGATIVE_TTL_SEC: int = 60


# --- 1) Title garbage detection --------------------------------------------

# Title extracted by GROBID is one of these three patterns when the
# TEI parse failed. They are NOT real titles.
_TITLE_GARBAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "001_020" / "035_048" — page-range strings from the running
    # header. 2-3 digits + underscore + 2-3 digits, with no letters.
    re.compile(r"^\d{1,3}[_-]\d{1,3}$"),
    # "StrtEng2470030Bragin.fm" — filename ends in .fm / .tex / .pdf.
    re.compile(r"^.*\.(fm|tex|pdf|dvi)$", re.IGNORECASE),
    # Pure page numbers like "542" or "15" (1-4 digits, no letters).
    re.compile(r"^\d{1,4}$"),
)

# A real title is usually longer than 8 characters and contains at
# least one alphabetic character. We use this as a final filter
# rather than a primary check (some short titles like "GIS data" are
# legitimate).
_MIN_TITLE_LEN = 8

# Phase 62 Plan 5 (Bug 5.13): apply _MIN_TITLE_LEN as a final
# alphanumeric-only filter. Titles shorter than this that contain
# NO alphabetic characters (or that have no run of 3+ consecutive
# alphabetic characters) are flagged as parse artifacts.
# Examples caught:
#   - "1234567" (digits only, 7 chars)
#   - "a1b2c3" (alternating single letters + digits, no real word)
# Examples NOT caught (real titles):
#   - "GIS data" (8 chars, contains "GIS" — 3 consecutive letters)
#   - "A 2-D map" (contains real words)
_RUN_OF_LETTERS_RE = re.compile(r"[A-Za-z]{3,}")


def looks_like_garbage_title(title: str | None) -> bool:
    """Return True if ``title`` looks like a parse-failure artifact.

    Detected patterns (Round 20 sampling):
      - Page-range strings (``"001_020"``, ``"035_048"``)
      - Filenames ending in ``.fm`` / ``.tex`` / ``.pdf`` / ``.dvi``
      - Pure digits (``"15"``, ``"542"``)
      - Phase 62 Plan 5 (Bug 5.13): alphanumeric gibberish. Titles
        shorter than ``_MIN_TITLE_LEN`` characters that contain no
        run of 3+ consecutive letters (e.g. ``"a1b2c3"``,
        ``"1234567"``) are flagged. Real short titles like
        ``"GIS data"`` are preserved because they contain
        consecutive-letter runs.

    Returns False for ``None``, empty string, or any plausible
    title that does not match the patterns above.
    """
    if not title:
        return True
    t = title.strip()
    if not t:
        return True
    for pat in _TITLE_GARBAGE_PATTERNS:
        if pat.match(t):
            return True
    # Phase 62 Plan 5 (Bug 5.13): short alphanumeric titles with
    # no real word are garbage. Real titles always contain at
    # least one run of 3+ consecutive letters (e.g. "Late",
    # "Triassic", "GIS", "Italy").
    if len(t) < _MIN_TITLE_LEN and not _RUN_OF_LETTERS_RE.search(t):
        return True
    return False


def cleanup_title(
    title: str | None, *, paper_id: str | None = None
) -> tuple[str | None, str | None]:
    """Return ``(cleaned_title, review_reason_or_None)``.

    If the title looks like a parse artifact, returns
    ``(None, "title_extraction_failed")`` so the consumer can flag
    the paper for review. Otherwise returns the title unchanged.
    """
    if looks_like_garbage_title(title):
        logger.info(
            "paper %s: title %r looks like a parse artifact; flagging for review",
            paper_id,
            title,
        )
        return (None, "title_extraction_failed")
    return (title, None)


# --- 2) Author marker strip ------------------------------------------------

# OpenDataLoader returns placeholder strings when it cannot identify
# a real author. We strip them so they don't pollute the author list.
_AUTHOR_MARKERS: frozenset[str] = frozenset(
    {
        "input",
        "input2",
        "input3",
        "unknown",
        "unknown author",
        "n/a",
        "na",
        "no author",
        "anonymous",
        "[no author]",
    }
)


def cleanup_authors(authors: list[str] | None) -> list[str]:
    """Strip placeholder markers from a list of author names.

    The check is case-insensitive and ignores surrounding
    whitespace. The original list is not mutated; a new list is
    returned.
    """
    if not authors:
        return []
    cleaned: list[str] = []
    for a in authors:
        if not a:
            continue
        norm = a.strip().lower()
        if not norm:
            continue
        if norm in _AUTHOR_MARKERS:
            logger.debug("Stripping author marker %r", a)
            continue
        cleaned.append(a.strip())
    return cleaned


# --- 3) Crossref DOI → journal ---------------------------------------------

# In-memory cache so repeated lookups for the same DOI don't hit
# the network. The cache is intentionally per-process (not
# persistent) — the journal name rarely changes once published.
# Phase 54 audit m19 — value + timestamp tuple for TTL support.
_CROSSREF_CACHE: dict[str, tuple[str | None, float]] = {}


def _crossref_get_journal(doi: str, *, timeout_sec: float = 5.0) -> str | None:
    """Fetch the journal name for a DOI via the Crossref public API.

    Returns ``container-title[0]`` from the Crossref response, or
    ``None`` on any failure (network error, 404, malformed JSON).

    Round 20 sampling showed that GROBID sometimes returns the
    publisher name in the journal field (e.g. ``journal="Scale"``
    for Bragin 2025, which is actually the publisher "Pleiades
    Publishing"). When GROBID's journal is None, empty, or
    suspiciously short, this function provides a fallback.

    Phase 62 Plan 5 (Bug 5.14): the cache TTL is split into
    positive (real journal name → 1 hour) and negative (None,
    non-200, network error → 60s) buckets. A transient Crossref
    outage at the start of a batch run no longer tags every paper
    with ``journal=None`` for the next hour.
    """
    if doi in _CROSSREF_CACHE:
        cached_value, cached_at = _CROSSREF_CACHE[doi]
        # Phase 54 audit m19 — TTL on negative (and positive) cache
        # entries. Previously a 404 from Crossref was cached forever,
        # so a paper with 5 species of bad DOIs hit the network 5
        # times per process AND a previously-networked failure kept
        # re-warning the operator on every call. 1 hour is long
        # enough to dedupe within a single run (one paper's 5
        # retries collapse to 1 network call) and short enough that
        # transient Crossref outages self-heal without a process
        # restart.
        #
        # Phase 62 Plan 5 (Bug 5.14): negative entries (None) use
        # the much shorter 60s TTL so a transient outage recovers
        # within a minute rather than an hour.
        ttl = (
            _CROSSREF_NEGATIVE_TTL_SEC
            if cached_value is None
            else _CROSSREF_POSITIVE_TTL_SEC
        )
        if (time.time() - cached_at) < ttl:
            return cached_value
    try:
        import requests  # local import to keep cold-import cheap
    except ImportError:
        # Round 23 audit: missing requests library is operationally
        # important (papers will keep GROBID's null journal) so we
        # upgrade to warning. The cached ``None`` still avoids
        # retrying on every record.
        logger.warning("requests not available; cannot call Crossref")
        _CROSSREF_CACHE[doi] = (None, time.time())
        return None
    url = f"https://api.crossref.org/works/{doi}"
    try:
        resp = requests.get(url, timeout=timeout_sec)
        if resp.status_code != 200:
            # Round 23 audit: non-200 from Crossref is also
            # operationally relevant (the DOI may be invalid or the
            # service is degraded); upgrade to warning.
            logger.warning(
                "Crossref returned HTTP %s for DOI=%s; journal stays empty",
                resp.status_code,
                doi,
            )
            _CROSSREF_CACHE[doi] = (None, time.time())
            return None
        data = resp.json().get("message", {})
        container = data.get("container-title") or []
        title = container[0] if container else None
        _CROSSREF_CACHE[doi] = (title, time.time())
        return title
    except Exception as exc:
        # Round 23 audit: network / JSON / TLS errors should be
        # visible to operators (not silent debug-level).
        logger.warning(
            "Crossref lookup failed for DOI=%s: %s", doi, exc
        )
        # Phase 54 audit m19 — TTL the network failure too. Without
        # the cache entry the next call would hit the network again
        # and the operator would see the same warning in a tight
        # loop. The TTL keeps both positive and negative entries on
        # the same self-healing 1-hour schedule.
        _CROSSREF_CACHE[doi] = (None, time.time())
        return None


# Treat GROBID's journal field as "needs enrichment" when it's
# None, empty, or shorter than 4 characters. These are the cases
# where Crossref is most likely to help.
def needs_journal_enrichment(journal: str | None) -> bool:
    """Return True if the GROBID-extracted journal is missing,
    suspiciously short, or contains publisher-name vocabulary that
    Round-20 sampling showed leaked into the journal field.

    Round 20 cases that triggered enrichment:
      - ``journal=None`` (Boughdiri)
      - ``journal="Scale"`` (Bragin — that's a publisher name, not
        a journal; the real journal is "Stratigraphy and Geological
        Correlation")
      - ``journal="Pleiades"`` (a publisher name)
      - ``journal=""`` (empty)

    The threshold is 6 characters (most legitimate journal names
    are at least that long — "Nature", "Science", "Geology") plus
    a blacklist of common publisher words.
    """
    if not journal:
        return True
    j = journal.strip()
    if len(j) < 6:
        return True
    lower = j.lower()
    publisher_words = (
        "scale",
        "pleiades",
        "publishing",
        "press",
        "elsevier",
        "springer",
        "wiley",
        "taylor",
        "francis",
        "mdpi",
        "hindawi",
    )
    if any(pw in lower for pw in publisher_words):
        return True
    return False


def enrich_journal(journal: str | None, doi: str | None) -> str | None:
    """Return the journal name, falling back to Crossref when GROBID's
    value is missing or implausibly short (or a known publisher name).

    Returns ``None`` when both GROBID and Crossref have no answer.
    """
    if not needs_journal_enrichment(journal):
        return journal
    if not doi:
        return journal or None
    crossref_journal = _crossref_get_journal(doi)
    if crossref_journal and not needs_journal_enrichment(crossref_journal):
        logger.info(
            "enriched journal for DOI=%s: %r (was GROBID=%r)",
            doi,
            crossref_journal,
            journal,
        )
        return crossref_journal
    return journal or None


# --- Combined entry point --------------------------------------------------


def cleanup_paper_metadata(paper_dict: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Apply all three cleanups to a paper record.

    Returns ``(cleaned_dict, review_reasons)``. The ``review_reasons``
    list contains any flags raised (e.g. ``"title_extraction_failed"``)
    so the caller can stamp them onto the paper record or surface
    them in the UI.
    """
    review_reasons: list[str] = []
    cleaned = dict(paper_dict)

    # Title
    title, title_reason = cleanup_title(cleaned.get("title"), paper_id=cleaned.get("paper_id"))
    if title_reason:
        review_reasons.append(title_reason)
        cleaned["title"] = title  # None
    else:
        cleaned["title"] = title

    # Authors
    cleaned["authors"] = cleanup_authors(cleaned.get("authors") or [])

    # Journal via DOI
    cleaned["journal"] = enrich_journal(cleaned.get("journal"), cleaned.get("doi"))
    # audit 2026-07-31: a real run shipped journal="Explanation of
    # Plate" — the caption header leaked into the journal field and
    # went straight into CSV/DwC/EML exports. Drop caption-fragment
    # values here.
    if _is_garbage_journal(cleaned.get("journal")):
        cleaned["journal"] = None
        if "journal_extraction_failed" not in review_reasons:
            review_reasons.append("journal_extraction_failed")

    return cleaned, review_reasons


# audit 2026-07-31: journal values polluted by caption fragments —
# "Explanation of Plate 1", "Plate 2", "Fig. 1", page-range strings.
# These are never journal names.
_JOURNAL_GARBAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^explanation\s+of\s+plate", re.IGNORECASE),
    re.compile(r"^plates?\s+\d+", re.IGNORECASE),
    re.compile(r"^figs?\.?\s*\d+", re.IGNORECASE),
    re.compile(r"^\d{1,3}[_-]\d{1,3}$"),
)


def _is_garbage_journal(journal: Any) -> bool:
    if journal is None:
        return True
    s = str(journal).strip()
    if not s or len(s) < 3:
        return True
    return any(p.match(s) for p in _JOURNAL_GARBAGE_PATTERNS)
