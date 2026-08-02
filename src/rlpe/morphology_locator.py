"""Morphology context locator (audit 2026-08-02).

Stage 6 of the RLPE pipeline needs a chunk of body text describing one
species — typically the Description / Diagnosis / Remarks / Dimensions
section. The fulltext TEI (GROBID) and OpenDataLoader extracted sections
both expose a ``fulltext_sections`` list with the same shape::

    {"section_id": str, "title": str, "section_type": str, "text": str}

This module finds, for a given species, the smallest span of section
text that contains the species anchor and ends before the next species
heading or the "Occurrence / Distribution / Stratigraphic range" cue.

Design rules
------------
* Pure function (no I/O, no backend calls). Trivial to unit-test.
* Anchor: species name (with authority stripped) must appear as a word
  boundary inside the section text. The authority stripping is best-
  effort — we accept either the bare binomial or the binom + "(Author, 1900)"
  pattern. Anything we can't anchor on returns None.
* Cut point: stop at the next species heading (any of the title
  patterns in ``_SPECIES_HEADING_PATTERNS``) OR the first occurrence of
  one of ``_STOP_PATTERNS`` (Occurrence / Distribution / Stratigraphic
  range). If neither is found within ``max_chars``, the whole remainder
  of the section is returned (truncated to ``max_chars``).
* Unicode hyphens: any of ASCII hyphen, en-dash, em-dash, minus sign,
  figure dash, hyphen-minus, non-breaking hyphen are normalised to a
  single ASCII hyphen BEFORE the anchor search so the cut at the next
  species heading does not miss a "stauracan- something" hyphenated
  shape that should be collapsed.
* Whitespace is collapsed (newlines and runs of spaces → single space)
  so the returned ``source_text`` is a single readable paragraph for
  the LLM prompt.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Any of: ASCII hyphen-minus, en-dash, em-dash, minus sign, figure dash,
# non-breaking hyphen, hyphen (U+2010). These all appear in OCR'd
# morphology text and need to be normalised so "stauracan- X" can be
# matched the same way as "stauracan X".
_UNICODE_HYPHENS = "-‐‑‒–—−―­"


def _strip_authority(species_name: str) -> str:
    """Remove the trailing ``(Author, 1900)`` or ``Author, 1900`` style authority.

    We keep the first three tokens (genus + specific_epithet + optional
    qualifier like ``"sp."`` / ``"cf."``). Anything beyond is dropped.
    The author/year pair is detected by looking for a token matching
    ``[A-Z][a-z]+,`` followed by a 4-digit year (or just the comma
    alone — many papers omit the year).
    """
    if not species_name:
        return ""
    # Drop parenthetical authority.
    cleaned = re.sub(r"\([^)]*\)", " ", species_name)
    cleaned = cleaned.strip()
    # Normalise Unicode hyphens (same set as ``_normalise_text``) so
    # ``Genus‐alpha`` (U+2010) matches the normalised text that the
    # locator compares against.
    for hy in _UNICODE_HYPHENS:
        if hy in cleaned:
            cleaned = cleaned.replace(hy, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Drop trailing author/year pair: "Smith, 1900" or just "Smith,".
    cleaned = re.sub(
        r"\s+[A-Z][a-zA-Z\.]+,\s*\d{0,4}\s*$",
        "",
        cleaned,
    )
    # If a stray comma remains (because year was missing), drop it.
    cleaned = cleaned.rstrip(",").rstrip()
    # Keep genus + specific + optional qualifier.
    parts = cleaned.split()
    if len(parts) <= 3:
        return " ".join(parts)
    return " ".join(parts[:3])


# Sentinel string used by ``_normalise_text`` to mask digit-digit
# hyphens so the hyphen-normalisation pass below doesn't touch them.
# We use a string rather than a control char so it round-trips through
# Python string literals without ``\xNN`` escape headaches.
_DIGIT_HYPHEN_SENTINEL = "RANGE"


def _normalise_text(text: str) -> str:
    """Collapse whitespace and normalise Unicode hyphens.

    The pipeline frequently sees text from OCR or TEI extraction that
    mixes line-broken words (``"stauracan-\\nthauma"``) with inconsistent
    whitespace. We collapse runs of whitespace to a single space, strip
    NFKC-normalised variants, and replace Unicode hyphens with ASCII so
    the anchor search is deterministic.

    Exception: hyphens between two digits are preserved (so a
    measurement range like ``180-220 µm`` stays ``180-220 µm`` — the
    hyphen is meaningful, not a line break).
    """
    if not text:
        return ""
    # NFKC normalisation collapses fullwidth / compatibility forms.
    text = unicodedata.normalize("NFKC", text)
    # First, mask digit-digit ranges with a placeholder so the
    # hyphen-normalisation step below doesn't touch them. Use
    # ``\d+`` on each side so ``180-220 µm`` (a full range) is
    # masked as a unit, not just ``0-2``. The hyphen character
    # class is built from ``_UNICODE_HYPHENS`` (which includes
    # ASCII ``-`` plus the Unicode variants) so both ``180-220``
    # and ``180‐220`` (U+2010) are caught.
    hyphen_class = "[" + re.escape(_UNICODE_HYPHENS) + "]"
    digit_dash = re.compile(r"(\d+)\s*" + hyphen_class + r"\s*(\d+)")
    text = digit_dash.sub(
        lambda m: m.group(1) + _DIGIT_HYPHEN_SENTINEL + m.group(2),
        text,
    )
    # Replace any remaining Unicode hyphen with a space. This is
    # what radiolarian papers actually mean when they break a word
    # across a line; collapsing without spacing would change
    # ``stauracanthidium`` to ``stauracanth idium`` and miss the
    # authority matching.
    for hy in _UNICODE_HYPHENS:
        if hy in text:
            text = text.replace(hy, " ")
    # Restore the digit-digit placeholder back to a hyphen.
    text = text.replace(_DIGIT_HYPHEN_SENTINEL, "-")
    # Collapse runs of whitespace to a single space.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Heading patterns that mark the start of a NEW species description
# (i.e. the NEXT species after the one we just anchored on). We
# intentionally do NOT match bare ``Description:`` because the
# current species' own Description heading sits AFTER the anchor
# and would otherwise truncate the very text we're trying to
# extract.
_SPECIES_HEADING_PATTERNS = (
    # "Genus species" or "Genus species Smith, 1900" — a NEW species
    # heading is a capitalised Genus followed by a lowercase specific
    # epithet, followed by EITHER authority / heading word OR a
    # sentence terminator (``.`` / ``.\\n``).
    r"\b[A-Z][a-z]{2,}\s+[a-z]{3,}(?:\s+(?:[A-Z][a-z\.]+,\s*\d{0,4}|Description|Diagnosis|Remarks))?[.:\n]",
)

# Patterns that mark the END of a species description. We cut as soon
# as we see one of these so the returned context doesn't bleed into
# the next species' geographic/strat info.
_STOP_PATTERNS = (
    r"\b(?:Occurrence|Distribution|Stratigraphic\s+range|Stratigraphic\s+distribution|Geographic\s+range)\s*[\.:]",
)


def locate_morphology_context(
    species_name: str,
    fulltext_sections: list[dict[str, Any]],
    *,
    max_chars: int = 6000,
) -> dict[str, Any] | None:
    """Find the Description / Diagnosis section for the given species.

    Returns a dict with keys::

        {
          "source_text": str,         # normalised, cut, ≤ max_chars
          "section_id": str,
          "section_title": str,
          "section_type": str,
          "evidence_span": str,       # short verbatim anchor context
          "anchor_species": str,      # the stripped name we matched on
        }

    Returns ``None`` if no reliable anchor can be found.

    Logic:
      1. Filter ``fulltext_sections`` to ``systematic_paleontology``
         sections OR sections whose title contains
         ``Description / Diagnosis / Remarks / Dimensions`` (case-
         insensitive).
      2. For each candidate section, normalise the text (hyphens +
         whitespace) and search for the species anchor at a word
         boundary. The anchor is the authority-stripped species name.
      3. From the anchor position, cut at the first ``_SPECIES_HEADING_
         PATTERNS`` match OR the first ``_STOP_PATTERNS`` match — which-
         ever comes first.
      4. If no section contains the anchor, return None.
    """
    if not species_name or not species_name.strip():
        return None
    if not fulltext_sections:
        return None
    anchor = _strip_authority(species_name)
    if not anchor:
        return None

    # 1. Filter candidate sections.
    candidates: list[dict[str, Any]] = []
    title_pattern = re.compile(
        r"\b(?:systematic\s+paleontology|description|diagnosis|remarks|dimensions)\b",
        re.IGNORECASE,
    )
    for sec in fulltext_sections:
        if not isinstance(sec, dict):
            continue
        stype = str(sec.get("section_type") or "").lower()
        title = str(sec.get("title") or "")
        text = sec.get("text") or ""
        if not text:
            continue
        if stype == "systematic_paleontology":
            candidates.append(sec)
        elif title_pattern.search(title):
            candidates.append(sec)
    if not candidates:
        return None

    # 2. Search each candidate for the species anchor.
    # We split into words for the anchor search so a 3-token name
    # like "Triassocampe sp." still matches when "Triassocampe" and
    # "sp." are separated by a hyphen that we've normalised away.
    anchor_tokens = [t for t in anchor.split() if t]
    anchor_re = re.compile(
        r"\b" + r"\s+".join(re.escape(tok) for tok in anchor_tokens) + r"\b",
        re.IGNORECASE,
    )
    for sec in candidates:
        text = sec.get("text") or ""
        norm_text = _normalise_text(text)
        m = anchor_re.search(norm_text)
        if not m:
            continue
        start = m.start()
        # 3. Find the first stop / next-heading cut point after the anchor.
        # We use the normalised text for matching so the regex
        # spacing matches the text we're slicing. The heading
        # pattern is case-SENSITIVE — we want to detect the start
        # of a NEW species whose Genus is capitalised (mid-sentence
        # lowercase "spumellarian with cortical shell" is NOT a new
        # species heading). The stop pattern stays case-insensitive.
        head_pat = re.compile("|".join(_SPECIES_HEADING_PATTERNS))
        stop_pat = re.compile("|".join(_STOP_PATTERNS), re.IGNORECASE)
        # Walk past the anchor itself to find the next cut.
        tail = norm_text[start:]
        # Skip past the matched species name itself when scanning for the
        # next heading (otherwise the heading pattern can fire on the
        # "Genus species Description" preamble).
        post_anchor = tail[len(m.group(0)) :]
        # Find earliest cut point.
        cut_offset = len(post_anchor)  # default: take the rest
        head_match = head_pat.search(post_anchor)
        if head_match:
            cut_offset = min(cut_offset, head_match.start())
        stop_match = stop_pat.search(post_anchor)
        if stop_match:
            cut_offset = min(cut_offset, stop_match.start())
        # Reconstruct the source text from the (normalised) section text.
        source_text = tail[: len(m.group(0)) + cut_offset].strip()
        if len(source_text) > max_chars:
            source_text = source_text[:max_chars] + "..."
        # evidence_span: short verbatim quote around the anchor.
        ev_start = max(0, m.start() - 40)
        ev_end = min(len(norm_text), m.end() + 80)
        evidence_span = norm_text[ev_start:ev_end].strip()
        return {
            "source_text": source_text,
            "section_id": str(sec.get("section_id") or ""),
            "section_title": str(sec.get("title") or ""),
            "section_type": str(sec.get("section_type") or ""),
            "evidence_span": evidence_span,
            "anchor_species": anchor,
        }
    return None


__all__ = ["locate_morphology_context"]
