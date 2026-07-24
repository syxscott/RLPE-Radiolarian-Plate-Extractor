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
from dataclasses import dataclass
from typing import Iterable


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
# ``sample LR-7``, ``Sample ZX-9``. We deliberately anchor on the keyword so
# random single-letter strings elsewhere in the caption don't get picked up.
# Two branches: (a) tokens with a digit (handles ZX-9, LR-7, A12), and
# (b) pure-letter tokens of 1-6 chars (handles A, LR, etc.).
#
# NOTE: We use ``(?<![A-Za-z])`` / ``(?![A-Za-z])`` rather than ``\b``
# because the default Python ``\b`` only matches at ASCII word
# boundaries, which do NOT fire between e.g. ``e`` and ``Z`` in
# ``Sample ZX-9`` (both word chars). The lookarounds give us proper
# case-insensitive keyword isolation.
_SAMPLE_RE = re.compile(
    r"(?<![A-Za-z])Sample\s+(?:ID[-:]\s*)?"
    r"([A-Za-z]+[-]?[A-Za-z]*\d[A-Za-z0-9\-]*|[A-Za-z]{1,6})",
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
    r"([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,3})"
    r"(?:\s*,?\s*and\s+([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,3}))?",
    re.IGNORECASE,
)

# Bare ``ID-203``, ``ID:42`` when not already captured by the sample regex.
# We intentionally exclude the ``Sample ID-`` prefix so we don't double-count.
_ID_RE = re.compile(
    r"(?<![Ss]ample\s)\bID[-:]\s*([A-Z0-9][A-Za-z0-9\-]{0,15})\b"
)


# ---------------------------------------------------------------------------
# Locality phrase patterns (separate from Sample/Loc)
# ---------------------------------------------------------------------------

# Standalone locality phrases that do not follow "Sample/Loc/Loc."
# keywords (e.g. "from Tunisia", "at Mt. Etna"). The lookahead stop set
# mirrors geology_extraction.LOCALITY_PATTERN so the two extractors stay
# consistent.
_LOCALITY_PHRASE_RE = re.compile(
    r"\b(?:from|at|in|of|near)\s+"
    r"([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,3})"
    r"(?=\s*[,.;:()]|\s+(?:and|the|of|a|an|is|are|was|were|in|at|"
    r"we|to|by|for|on|as|which|that|where)\b|$)",
    re.IGNORECASE,
)

# Reject common false-positive locality phrases (chronostratigraphy terms,
# paper-grammar stop words, etc.). Lowercase comparison.
_LOCALITY_BLOCKLIST: frozenset[str] = frozenset({
    "late cretaceous", "early cretaceous", "middle cretaceous",
    "early jurassic", "middle jurassic", "late jurassic",
    "early triassic", "middle triassic", "late triassic",
    "early devonian", "middle devonian", "late devonian",
    "early permian", "middle permian", "late permian",
    "early carboniferous", "late carboniferous",
    "early cambrian", "middle cambrian", "late cambrian",
    "early ordovician", "middle ordovician", "late ordovician",
    "early silurian", "middle silurian", "late silurian",
    "early paleocene", "middle paleocene", "late paleocene",
    "early eocene", "middle eocene", "late eocene",
    "early miocene", "middle miocene", "late miocene",
    "early oligocene", "late oligocene",
    "early pliocene", "late pliocene",
    "this paper", "the paper", "this study", "the study",
    "figure", "plate", "section", "sample", "locality", "localities",
})


# ---------------------------------------------------------------------------
# Age / period lexicon
# ---------------------------------------------------------------------------

# Curated ICS period + epoch names (case-insensitive match). Order
# matters: longer phrases first so "Late Cretaceous" wins over
# "Cretaceous" when both are present.
_AGE_TERMS: tuple[str, ...] = (
    # Series / epoch combinations first
    "early cambrian", "middle cambrian", "late cambrian",
    "lower cambrian", "middle cambrian", "upper cambrian",
    "early ordovician", "middle ordovician", "late ordovician",
    "lower ordovician", "upper ordovician",
    "early silurian", "middle silurian", "late silurian",
    "lower silurian", "upper silurian", "pridoli", "ludlow", "wenlock", "llandovery",
    "early devonian", "middle devonian", "late devonian",
    "lower devonian", "middle devonian", "upper devonian",
    "early carboniferous", "late carboniferous",
    "mississippian", "pennsylvanian",
    "early permian", "middle permian", "late permian",
    "cisuralian", "guadalupian", "lopingian",
    "early triassic", "middle triassic", "late triassic",
    "lower triassic", "middle triassic", "upper triassic",
    "inderbian", "olenekian", "anisian", "ladinian", "carnian", "norian", "rhaetian",
    "early jurassic", "middle jurassic", "late jurassic",
    "lower jurassic", "middle jurassic", "upper jurassic",
    "hettangian", "sinemurian", "pliensbachian", "toarcian",
    "aalenian", "bajocian", "bathonian", "callovian",
    "oxfordian", "kimmeridgian", "tithonian",
    "early cretaceous", "middle cretaceous", "late cretaceous",
    "lower cretaceous", "upper cretaceous",
    "berriasian", "valanginian", "hauterivian", "barremian",
    "aptian", "albian", "cenomanian", "turonian", "coniacian",
    "santonian", "campanian", "maastrichtian",
    "early paleocene", "middle paleocene", "late paleocene",
    "paleocene", "eocene", "oligocene",
    "early eocene", "middle eocene", "late eocene",
    "early oligocene", "late oligocene",
    "miocene", "pliocene",
    "early miocene", "middle miocene", "late miocene",
    "early pliocene", "late pliocene",
    "pleistocene", "holocene",
    # Period roots (catch-all; longer phrases above win when matched first)
    "cambrian", "ordovician", "silurian", "devonian", "carboniferous",
    "permian", "triassic", "jurassic", "cretaceous",
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
    tail_re = re.compile(r"(?:,|and)\s+([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,3})", re.IGNORECASE)
    for m in _LOCALITY_PHRASE_RE.finditer(caption):
        phrase = m.group(1).strip()
        if not phrase:
            continue
        if phrase.casefold() in _LOCALITY_BLOCKLIST:
            continue
        if len(phrase) < 3:
            continue
        raw.append(phrase)
        # Look for a trailing ", X" or "and X" after this match.
        tail_start = m.end()
        tail_end = min(len(caption), tail_start + 60)
        tail_section = caption[tail_start:tail_end]
        for tm in tail_re.finditer(tail_section):
            extra = tm.group(1).strip()
            if not extra or extra.casefold() in _LOCALITY_BLOCKLIST:
                continue
            if len(extra) < 3:
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
        phrase = caption[m.start(1):m.end(1)].strip()
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
