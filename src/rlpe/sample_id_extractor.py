"""Sample ID / Locality / Age extractor for cross-figure linking.

Phase 65 Plan A.1: regular-expression extractors that turn caption text
into structured ``SampleID`` / locality / age triples which the
``cross_figure_linker`` uses to build plate→strat-column / plate→map
associations.

Three pure functions, no I/O, no LLM, fully deterministic so they can
be unit-tested without any external services:

* :func:`extract_sample_ids` — regex pulls ``Sample X`` / ``Loc. Y`` /
  ``ID-N`` tokens out of a caption.
* :func:`extract_locality` — pulls capitalized locality phrases
  (``from Tunisia``, ``at Mt. Etna``, …) using a pattern similar to
  ``geology_extraction.LOCALITY_PATTERN`` but kept narrower to limit
  false positives.
* :func:`extract_age_terms` — pulls geological-period / epoch / age
  phrases (``Late Cretaceous``, ``Carnian``, …) using a curated lexicon.

All extractors are case-insensitive, deduplicate, and preserve the
original casing of the matched token in the returned value.

The module is intentionally self-contained — it does not import from
``geology_extraction`` so it can run even when that module's heavy
import chain is unavailable (e.g. tiny smoke harness).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SampleID:
    """A sample / locality / numeric-id token extracted from a caption.

    Attributes
    ----------
    kind : str
        ``"sample"`` for ``Sample S1``-style tokens, ``"loc"`` for
        ``Loc. Tunisia``-style tokens, ``"id"`` for ``ID-203``-style
        numeric IDs.
    value : str
        The matched token (e.g. ``"S1"``, ``"Tunisia"``, ``"203"``).
        The casing is preserved from the source text.
    confidence : float
        Heuristic confidence score in ``[0.0, 1.0]``. Direct tokens
        (e.g. ``Sample S1``) score 1.0; tokens where the regex had to
        infer the prefix score lower (0.7-0.9).
    """

    kind: str  # "sample" | "loc" | "id"
    value: str
    confidence: float


# ---------------------------------------------------------------------------
# Sample / Loc / ID regex patterns
# ---------------------------------------------------------------------------

# ``Sample S1``, ``sample A12``, ``Sample ID-203`` (with optional ID- prefix),
# ``sample LR-7``, ``Sample ZX-9``, ``Sample 100A`` (digit + letter suffix).
# We deliberately anchor on the keyword so random single-letter strings
# elsewhere in the caption don't get picked up. Three branches:
# (a) tokens with a digit, letter-led (handles ZX-9, LR-7, A12),
# (b) pure-letter tokens of 1-6 chars (handles A, LR, etc.),
# (c) digit-led tokens (handles "100", "100A", "100-2").
#
# NOTE: We use ``(?<![A-Za-z])`` / ``(?![A-Za-z])`` rather than ``\b``
# because the default Python ``\b`` only matches at ASCII word
# boundaries, which do NOT fire between e.g. ``e`` and ``Z`` in
# ``Sample ZX-9`` (both word chars). The lookarounds give us proper
# case-insensitive keyword isolation.
_SAMPLE_RE = re.compile(
    # Audit fix 2026-07-24 (Agent A H1 + H7):
    #   - H1: accept ``Samples`` (plural) keyword too — real captions
    #     say "Samples S1–S3 from Tunisia" where the regex previously
    #     missed the whole phrase.
    #   - H7: also accept purely numeric sample IDs like "Sample 203".
    #     The original required the value to start with a letter
    #     (``[A-Za-z]+[-]?[A-Za-z]*\d``), silently dropping bare-digit
    #     IDs. Added a third alternation branch ``\d{2,}`` (2+ digits
    #     to avoid matching years like 2024).
    # Audit fix 2026-08-18:
    #   - Allow trailing alphanumeric chars after a digit-led branch
    #     (``\d{2,}[A-Za-z0-9\-]*``). Captions like ``Sample 100A``
    #     were truncated to ``"100"``, then the legacy ``Sample\s+``
    #     regex in the converter would emit a separate ``S_100A``
    #     record that the cross-prefix dedup had to drop — silently
    #     losing the ``A`` suffix. Allowing the helper to consume the
    #     alphanumeric + hyphen tail keeps both detectors in agreement
    #     on values like ``100A``, ``100-1``, ``12-3``. Slash is NOT
    #     included here because ``sample 14/2`` is a separate legacy
    #     pattern (Round 21 ``N_`` prefix).
    r"(?<![A-Za-z])Samples?\s+(?:ID[-:]\s*)?"
    r"([A-Za-z]+[-]?[A-Za-z]*\d[A-Za-z0-9\-]*|[A-Za-z]{1,6}|\d{2,}[A-Za-z0-9\-]*)",
    re.IGNORECASE,
)

# ``Loc. Tunisia``, ``Loc Tunisia``, ``Locality: Greece``,
# ``Localities: Tunisia and Greece`` (the ``and <Name>`` tail is handled
# by an optional non-capturing group below).
#
# Audit fix 2026-07-24 (Agent A C4): the regex was case-sensitive on
# the locality phrase. OCR frequently emits ``loc. tunisia`` (lower
# keyword kept upper but locality normalized to lower), or all-caps
# headers like ``LOCALITY: TUNISIA``. The original required the
# first letter to be ``[A-Z]``, silently dropping lowercased outputs.
# re.IGNORECASE on the whole regex fixes both the keyword prefix
# (Loc. vs loc. vs LOC.) and the locality body.
_LOC_RE = re.compile(
    r"(?<![A-Za-z])(?:Loc\.?|Localit(?:y|ies))\s*[:\-]?\s+"
    # audit 2026-07-26: greedy match (was non-greedy {0,3}? which
    # truncated multi-word locality names like "Monte San Gottardo" to
    # just "Monte"), but exclude the word "and" so "Tunisia and Greece"
    # still splits into two localities via the optional and-group below.
    r"((?!and\b)[A-Za-z][A-Za-z\-]+(?:\s+(?!and\b)[A-Za-z][A-Za-z\-]+){0,3})"
    r"(?:\s*,?\s*and\s+((?!and\b)[A-Za-z][A-Za-z\-]+(?:\s+(?!and\b)[A-Za-z][A-Za-z\-]+){0,3}))?",
    re.IGNORECASE,
)

# Bare ``ID-203``, ``ID:42`` when not already captured by the sample regex.
# We intentionally exclude the ``Sample ID-`` prefix so we don't double-count.
_ID_RE = re.compile(r"(?<![Ss]ample\s)\bID[-:]\s*([A-Z0-9][A-Za-z0-9\-]{0,15})\b")


# ---------------------------------------------------------------------------
# Locality phrase patterns (separate from Sample/Loc)
# ---------------------------------------------------------------------------

# Standalone locality phrases that do not follow "Sample/Loc/Loc."
# keywords (e.g. "from Tunisia", "at Mt. Etna"). The lookahead stop set
# mirrors geology_extraction.LOCALITY_PATTERN so the two extractors stay
# consistent.
_LOCALITY_PHRASE_RE = re.compile(
    r"\b(?:from|at|in|of|near)\s+"
    r"([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,3}?)"
    r"(?=\s*[,.;:()]|\s+(?:and|the|of|a|an|is|are|was|were|in|at|"
    r"we|to|by|for|on|as|which|that|where)\b|$)",
    re.IGNORECASE,
)

# Sample values that are grammatically attached to the ``Sample`` keyword but
# are not identifiers (e.g. ``Samples from Tunisia``). Lowercase comparison.
_SAMPLE_STOPWORDS: frozenset[str] = frozenset(
    {"from", "at", "in", "of", "near", "the", "and", "to", "for", "with"}
)

# Reject common false-positive locality phrases (chronostratigraphy terms,
# paper-grammar stop words, etc.). Lowercase comparison.
_LOCALITY_BLOCKLIST: frozenset[str] = frozenset(
    {
        "late cretaceous",
        "early cretaceous",
        "middle cretaceous",
        "early jurassic",
        "middle jurassic",
        "late jurassic",
        "early triassic",
        "middle triassic",
        "late triassic",
        "early devonian",
        "middle devonian",
        "late devonian",
        "early permian",
        "middle permian",
        "late permian",
        "early carboniferous",
        "late carboniferous",
        "early cambrian",
        "middle cambrian",
        "late cambrian",
        "early ordovician",
        "middle ordovician",
        "late ordovician",
        "early silurian",
        "middle silurian",
        "late silurian",
        "early paleocene",
        "middle paleocene",
        "late paleocene",
        "early eocene",
        "middle eocene",
        "late eocene",
        "early miocene",
        "middle miocene",
        "late miocene",
        "early oligocene",
        "late oligocene",
        "early pliocene",
        "late pliocene",
        "this paper",
        "the paper",
        "this study",
        "the study",
        "figure",
        "plate",
        "section",
        "sample",
        "locality",
        "localities",
        # Lithostratigraphic formation names (Italian papers): shape like
        # locality names but are rock units, not places. Audit 2026-08-01 M7.
        "scaglia",
        "rosso ammonitico",
        "maiolica",
        "biancone",
        "fonzaso",
        "sicani",
        "radiolarian chert",
        # Latin particles that match the preposition+word pattern but
        # are NOT localities (audit 2026-08-18). ``in situ`` / ``in vivo``
        # / ``in vitro`` all fire the ``in <X>`` locality regex and would
        # otherwise emit ``situ`` / ``vivo`` / ``vitro`` as fake
        # localities.
        "situ",
        "vivo",
        "vitro",
        # Generic single-word "place" terms that are not actual
        # localities on their own. ``collected in the field`` /
        # ``found in the area`` would otherwise emit ``field`` /
        # ``area`` as fake localities.
        "field",
        "area",
        "region",
        "site",
    }
)

# Trailing modifiers that, when stripped from the END of a captured
# locality phrase, leave the actual locality behind. ``at the Karnezeika
# section`` -> ``Karnezeika``; ``from the Scaglia formation`` ->
# ``Scaglia`` (which is then caught by the blocklist substring check).
_LOCALITY_TRAILING_STOPWORDS: frozenset[str] = frozenset(
    {
        "section",
        "formation",
        "sample",
        "locality",
        "figure",
        "plate",
        "area",
        "field",
        "region",
        "site",
    }
)

# Leading articles that may be captured as the first word of the phrase
# (the locality regex's capture group allows ``[A-Za-z]`` as the start,
# so ``the`` can sneak in). Stripped from the front before the blocklist
# check.
_LOCALITY_LEADING_ARTICLES: frozenset[str] = frozenset({"the", "a", "an"})


# ---------------------------------------------------------------------------
# Age / period lexicon
# ---------------------------------------------------------------------------

# Curated ICS period + epoch names (case-insensitive match). Order
# matters: longer phrases first so "Late Cretaceous" wins over
# "Cretaceous" when both are present.
_AGE_TERMS: tuple[str, ...] = (
    # Series / epoch combinations first
    "early cambrian",
    "middle cambrian",
    "late cambrian",
    "lower cambrian",
    "middle cambrian",
    "upper cambrian",
    "early ordovician",
    "middle ordovician",
    "late ordovician",
    "lower ordovician",
    "upper ordovician",
    "early silurian",
    "middle silurian",
    "late silurian",
    "lower silurian",
    "upper silurian",
    "pridoli",
    "ludlow",
    "wenlock",
    "llandovery",
    "early devonian",
    "middle devonian",
    "late devonian",
    "lower devonian",
    "middle devonian",
    "upper devonian",
    "early carboniferous",
    "late carboniferous",
    "mississippian",
    "pennsylvanian",
    "early permian",
    "middle permian",
    "late permian",
    "cisuralian",
    "guadalupian",
    "lopingian",
    "early triassic",
    "middle triassic",
    "late triassic",
    "lower triassic",
    "middle triassic",
    "upper triassic",
    "induan",
    "olenekian",
    "anisian",
    "ladinian",
    "carnian",
    "norian",
    "rhaetian",
    "early jurassic",
    "middle jurassic",
    "late jurassic",
    "lower jurassic",
    "middle jurassic",
    "upper jurassic",
    "hettangian",
    "sinemurian",
    "pliensbachian",
    "toarcian",
    "aalenian",
    "bajocian",
    "bathonian",
    "callovian",
    "oxfordian",
    "kimmeridgian",
    "tithonian",
    "early cretaceous",
    "middle cretaceous",
    "late cretaceous",
    "lower cretaceous",
    "upper cretaceous",
    "berriasian",
    "valanginian",
    "hauterivian",
    "barremian",
    "aptian",
    "albian",
    "cenomanian",
    "turonian",
    "coniacian",
    "santonian",
    "campanian",
    "maastrichtian",
    "early paleocene",
    "middle paleocene",
    "late paleocene",
    "paleocene",
    "eocene",
    "oligocene",
    "early eocene",
    "middle eocene",
    "late eocene",
    "early oligocene",
    "late oligocene",
    "miocene",
    "pliocene",
    "early miocene",
    "middle miocene",
    "late miocene",
    "early pliocene",
    "late pliocene",
    "pleistocene",
    "holocene",
    # Period roots (catch-all; longer phrases above win when matched first)
    "cambrian",
    "ordovician",
    "silurian",
    "devonian",
    "carboniferous",
    "permian",
    "triassic",
    "jurassic",
    "cretaceous",
)

_AGE_TERMS_SET: frozenset[str] = frozenset(_AGE_TERMS)
# Compile the period roots separately so they match after the longer
# phrases above (we'll dedupe later).
_AGE_ROOT_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _AGE_TERMS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _dedupe_preserve(values: Iterable[str]) -> list[str]:
    """Return ``values`` deduplicated, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        key = v.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def extract_sample_ids(caption: str) -> list[SampleID]:
    """Extract ``Sample`` / ``Loc.`` / ``ID-N`` tokens from a caption.

    The function is order-preserving — tokens are returned in the order
    they appear in the caption — and deduplicated on ``(kind, value)``
    (case-insensitive on ``value``).
    """
    if not caption:
        return []

    out: list[SampleID] = []
    seen: set[tuple[str, str]] = set()

    for m in _SAMPLE_RE.finditer(caption):
        value = m.group(1).strip()
        if not value:
            continue
        if value.casefold() in _SAMPLE_STOPWORDS:
            continue
        # When the prefix was "Sample ID-N", strip the leading "ID-" so
        # we report just the numeric part as the canonical token. This
        # keeps the public API uniform: ``extract_sample_ids("Sample ID-203")``
        # returns ``SampleID(value="203")`` instead of ``"ID-203"``.
        matched_text = m.group(0)
        if re.match(r"\s*[Ss]ample\s+ID[-:]\s*", matched_text):
            value = re.sub(r"^ID[-:]\s*", "", value, flags=re.IGNORECASE)
            if not value:
                continue
        key = ("sample", value.casefold())
        if key in seen:
            continue
        seen.add(key)
        # Sample tokens with explicit "Sample" keyword get full confidence.
        # The optional "ID-" prefix lowers it slightly because the regex
        # is more permissive there.
        conf = 0.95 if matched_text.lower().startswith("sample id") else 1.0
        out.append(SampleID(kind="sample", value=value, confidence=conf))

    for m in _LOC_RE.finditer(caption):
        # ``_LOC_RE`` optionally captures a 2nd ``and <Name>`` group; we
        # iterate through both so "Localities: Tunisia and Greece"
        # returns both.
        for group_idx in (1, 2):
            value = m.group(group_idx)
            if not value:
                continue
            value = value.strip()
            if not value:
                continue
            # Reject the blocklist (chronostratigraphy terms would otherwise
            # leak through, e.g. "Locality: Late Cretaceous").
            if value.casefold() in _LOCALITY_BLOCKLIST:
                continue
            key = ("loc", value.casefold())
            if key in seen:
                continue
            seen.add(key)
            out.append(SampleID(kind="loc", value=value, confidence=1.0))

    for m in _ID_RE.finditer(caption):
        value = m.group(1).strip()
        if not value:
            continue
        key = ("id", value.casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(SampleID(kind="id", value=value, confidence=0.9))

    return out


def _normalize_locality_phrase(phrase: str) -> str | None:
    """Normalize a raw locality capture: strip leading articles, strip
    trailing stopwords, then check the blocklist (exact + substring).

    Returns the normalized phrase or ``None`` if it should be dropped
    (blocklisted, too short, or reduced to nothing by stripping).

    Audit 2026-08-18: four false-positive classes were over-matching
    without normalization:

    * ``Found in situ at the Karnezeika section.`` -> ``situ`` (Latin
      particle, blocklisted).
    * ``from the Scaglia formation`` -> ``the Scaglia formation``
      (Scaglia is blocklisted but the exact phrase wasn't checked;
      substring check fixes this once ``formation`` is stripped from the
      end).
    * ``at the Karnezeika section`` -> ``the Karnezeika section``
      (leading ``the`` + trailing ``section`` both stripped).
    * ``Collected in the field.`` -> ``the field`` (leading article
      stripped, then ``field`` alone is blocklisted).
    """
    if not phrase:
        return None

    # Strip leading articles.
    words = phrase.split()
    while words and words[0].casefold() in _LOCALITY_LEADING_ARTICLES:
        words.pop(0)
    if not words:
        return None
    # Strip trailing stopwords (e.g. ``section``, ``formation``).
    while words and words[-1].casefold() in _LOCALITY_TRAILING_STOPWORDS:
        words.pop()
    if not words:
        return None

    normalized = " ".join(words)
    if len(normalized) < 3:
        return None

    pc = normalized.casefold()
    # Exact-match blocklist.
    if pc in _LOCALITY_BLOCKLIST:
        return None
    # Substring blocklist (audit 2026-08-18: ``the Scaglia formation``
    # passed the exact-match check but should be caught because
    # ``scaglia`` is a substring).
    for blocked in _LOCALITY_BLOCKLIST:
        if blocked in pc:
            return None
    return normalized


def extract_locality(caption: str) -> list[str]:
    """Extract capitalized locality phrases from a caption.

    Returns phrases like ``"Tunisia"``, ``"NW Turkey"``, ``"Mt. Etna"``
    in the order they appear in the caption, deduplicated. The function
    does NOT extract tokens preceded by ``Loc.`` / ``Locality:`` —
    those are returned by :func:`extract_sample_ids` instead.
    """
    if not caption:
        return []

    raw: list[str] = []
    # Run a second-pass over the caption that also captures trailing
    # ``and <Locality>`` and ``, <Locality>`` so we get both halves of
    # "from Tunisia and Greece".
    tail_re = re.compile(
        r"(?:,|and)\s+([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,3})", re.IGNORECASE
    )
    for m in _LOCALITY_PHRASE_RE.finditer(caption):
        phrase = _normalize_locality_phrase(m.group(1).strip())
        if phrase is None:
            continue
        raw.append(phrase)
        # Look for a trailing ", X" or "and X" after this match.
        tail_start = m.end()
        tail_end = min(len(caption), tail_start + 60)
        tail_section = caption[tail_start:tail_end]
        for tm in tail_re.finditer(tail_section):
            extra = _normalize_locality_phrase(tm.group(1).strip())
            if extra is None:
                continue
            raw.append(extra)
    return _dedupe_preserve(raw)


def extract_age_terms(caption: str) -> list[str]:
    """Extract geological-period / epoch names from a caption.

    Returns phrases like ``"Late Cretaceous"``, ``"Carnian"`` in the
    **original source casing** (sliced from the caption via
    ``caption[m.start(1):m.end(1)]``, NOT taken from the lowercased
    regex alternation). Deduplicated case-insensitively while
    preserving the first-seen casing.

    Audit fix 2026-07-24: the previous implementation used
    ``m.group(1)`` which always returned the lowercased string from
    the regex alternation (since ``_AGE_TERMS`` is all lowercase).
    The docstring always promised "preserve original casing" but the
    output never honored it.
    """
    if not caption:
        return []

    raw: list[str] = []
    for m in _AGE_ROOT_RE.finditer(caption):
        phrase = caption[m.start(1) : m.end(1)].strip()
        if not phrase:
            continue
        raw.append(phrase)
    return _dedupe_preserve(raw)


__all__ = [
    "SampleID",
    "extract_sample_ids",
    "extract_locality",
    "extract_age_terms",
]
