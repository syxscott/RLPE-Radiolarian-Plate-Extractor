"""Range chart (stratigraphic distribution diagram) extractor.

A range chart is a figure type that shows the stratigraphic distribution of
species across measured sections — it pairs with a radiolarian paper's
plate figures and is the single richest source of geological context for
each extracted panel.

Pipeline integration
--------------------
This module runs AFTER ``OpenDataLoaderExtractor`` (which gives us figure
images + captions) and AFTER ``panel detection`` (which gives us per-figure
panel records with species). The role here is:

  1. Classify each figure as ``plate`` (SEM images) / ``range_chart`` /
     ``map`` / ``photo`` / ``other`` — a range chart gets special handling.
  2. Send the range-chart image + caption to MiniMax M3 vision and extract
     sections, species ranges, biozones, and other fossil occurrences as
     a strict JSON contract.
  3. Link the extracted geology back to the per-panel records: each panel
     that has a species matching a ``species_ranges`` entry inherits that
     entry's section, age_range, biozone, and range_top/range_base as a
     new ``GeologyLinkRecord`` attached to the panel's ``geology_links``.

The linking step is what makes this useful in practice: the panel records
already drive the downstream DwC export (see ``exporters/analysis.py``),
and a per-panel geology_links entry now carries stratigraphic context
that was previously only available in the raw text via the regex-based
``geology_extraction.extract_geology_from_sections``.

Design constraints
-----------------
  - Pure-vision: we rely on the LLM to read the chart, not on a brittle
    rule-based parser of the chart's column layout. Range-chart layouts
    vary wildly between papers (sections side-by-side vs stacked, with or
    without a lithology column, with or without numeric depth scales) and
    a layout parser would overfit to one paper's style.
  - Honest confidence: every record carries a ``confidence`` (0..1). The
    downstream merge step records the provenance of the link
    (``extraction_source="range_chart_vision"``) so an operator can
    filter out low-confidence links without losing the data.
  - Best-effort OCR: M3's text reading on low-resolution charts is noisy;
    we accept spelling errors in species names and recover via the
    existing ``_normalize_species`` against gold/caption-parser output.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)


# --- Figure type classification ------------------------------------------------

# audit 2026-07-31: whole-word "plate"/"plates" matcher for the plate
# keyword, plus the common words that contain "plate" as a substring
# and must NOT count ("pelagic plateau", "carbonate platform").
_PLATE_WORD_RE = re.compile(r"\bplates?\b")
_PLATE_SUBSTRING_VETO_WORDS = ("plateau", "platform")

_FIGURE_TYPE_PROMPT_KEYWORDS = {
    # Map: keyword -> figure type. Checked in order; first hit wins.
    "plate": (
        # NOTE: "plate" must be first so that "Plate N" captions
        # match before any range_chart keyword (e.g. "distribution
        # of radiolarians" in a plate caption). The word "plate"
        # alone is specific enough because radiolarian papers
        # consistently use "Plate N" for SEM figures.
        "plate",
        "scanning electron micrograph",
        "sem images",
        "sem micrograph",
        "transmitted light",
        "secondary electron",
        "back-scattered",
    ),
    "range_chart": (
        # NOTE: keywords here must be SPECIFIC enough to NOT match
        # generic radiolarian-plate captions. Beccaro2006 Plate 1's
        # caption mentions "stratigraphic sections" — single-word
        # "section" / "column" / "range" would over-match and send
        # the plate down the range_chart path. We require multi-word
        # context (stratigraphic column, range chart, etc.).
        "distribution of",
        "stratigraphic range",
        "range chart",
        "species range",
        "occurrence chart",
        "range distribution",
        "stratigraphic distribution",
        "species distribution",
        "biozone",
        "biostratigraphy",
        "conodont zone",
        "ammonoid zone",
        "bed-by-bed",
        "bed distribution",
        # Multi-word only — removed single-word "column" / "section"
        # because they over-match plate captions that mention
        # "stratigraphic sections" in passing.
        "columnar section",
        "chronostratigraphic column",
        "stratigraphic column",
        "fossil range",
        "taxon range",
        "biostratigraphic",
    ),
    "map": (
        # NOTE: "paleogeographic map" is intentionally NOT here — it
        # goes to the more specific "paleogeographic_map" type below.
        # Geographic place names ("Range" in "Great Dividing Range")
        # can over-match range-chart keywords, so we keep map
        # detection conservative first.
        "location map",
        "geographic distribution",
        "geological map",
        "study area",
        "index map",
        "map of",
        "sketch map",
        "location map of",
        # Less obvious map captions
        "route map",
        "outcrop map",
        "distribution map",
        "locations of outcrop",
        "location of outcrop",
        "outcrop location",
        "sample location",
        # Round 20 sampling: Boughdiri 2007 Fig 1 "Location of
        # studied sections..." missed the bare "location map"
        # pattern. Add "location of studied" / "location of section"
        # so generic location-map captions route correctly.
        "location of studied",
        "location of section",
        "location of the studied",
        "study area map",
    ),
    # Stratigraphic column / litholog column — more specific than
    # "range_chart" so we route them to the proper vision prompt.
    "strat_column": (
        # Round 20 sampling: the previous list missed 3 of the 4
        # strat-column captions we found in real papers (overview
        # captions, composite column captions, zone columns). Add
        # the missing phrases below. ``overview of.*strat`` and
        # ``strat.*overview`` are matched as two substrings; the
        # classifier uses plain ``in`` so each is checked
        # independently and the order doesn't matter.
        "stratigraphic column",
        "columnar section",
        "measured section",
        "generalized stratigraphy",
        "stratigraphic log",
        "composite column",
        "composite strat",
        "lithostratigraphic column",
        "composite lithostrat",
        "zones and beds",
        "zone column",
        "stratigraphic overview",
        "overview of the strat",
        "overview of",
        "tunisian jurassic strat",
        "jurassic stratigraphy",
        "jurassic strat",
        "cretaceous stratigraphy",
        "cretaceous strat",
        "stratigraphic framework",
        "chronostratigraphic",
    ),
    "litholog_column": (
        # Round 20 sampling: Boughdiri 2007's "Lithological sections
        # from Jebels Jédidi..." was missed by the old list. Add
        # the multi-word variants below. We keep multi-word only
        # because single-word "section" over-matches plate captions.
        "litholog column",
        "litholog log",
        "lithology column",
        "lithologic log",
        "lithology log",
        "lithological section",
        "lithological log",
        "lithologic column",
        "lithologic sections",
    ),
    "paleogeographic_map": (
        # Round 20 sampling: Bragin 2025's "The most important
        # localities of Oxfordian-Valanginian Boreal radiolarians
        # in Russia" was missed (no map keyword). Add locality-map
        # patterns. We keep "paleogeographic" / "palaeogeographic"
        # intact for the strict paleogeographic-map type, plus
        # locality/distribution patterns that often co-occur.
        "paleogeographic map",
        "palaeogeographic",
        "paleogeographic reconstruction",
        "palaeogeographic map",
        "locality map",
        "localities of",
        "localities in",
        "important localities",
        "main localities",
        "distribution of the",
    ),
    "photo": (
        "field photograph",
        "outcrop photograph",
        "field photo",
        # Round 20 sampling: Boughdiri 2007 Fig 5 "Exposures from
        # Oued Tazega section. a) General view..." was classified as
        # "other" because the photo keyword list only had three
        # variants. Add broader patterns so outcrop / field exposure
        # photos route correctly.
        "exposures from",
        "exposure of",
        "general view",
        "outcrop view",
        "outcrop photo",
        "field exposure",
    ),
    # Phase 64 Plan B: schematic / diagram / reconstruction /
    # phylogenetic figure types. These four new types route to a
    # dedicated M3 prompt (``PROMPT_REGISTRY["schematic_geo"]``) that
    # extracts text elements + concept relationships (e.g. "evolved
    # into") + extracted facts (ages, geography, taxa). They are
    # distinct from the existing map / range_chart / strat_column /
    # litholog_column / paleogeographic_map types because their
    # primary content is *conceptual* — boxes, arrows, and labels —
    # not measured stratigraphic sections or geographic shapes.
    #
    # Detection order matters: ``paleogeographic_map`` and ``map``
    # are checked BEFORE ``reconstruction`` so a "paleogeographic
    # reconstruction" caption routes to the existing
    # paleogeographic-map vision path (it carries geographic /
    # continent context that ``reconstruction`` would lose). The
    # plain "reconstruction" keywords below fire only when none of
    # the geographic-map / stratigraphic-column keywords match.
    "schematic": (
        "schematic",
        "schematic diagram",
        "conceptual diagram",
        "schematic reconstruction",
    ),
    "diagram": (
        "diagram",
        "block diagram",
        "flow diagram",
        "schematic diagram",
    ),
    "reconstruction": (
        "reconstruction",
        # Note: "paleogeographic reconstruction" / "palaeogeographic
        # reconstruction" intentionally NOT listed here — those
        # route to the more specific ``paleogeographic_map`` type.
        "artistic reconstruction",
        "life reconstruction",
    ),
    "phylogenetic": (
        "phylogenetic tree",
        "cladogram",
        "phylogeny",
        "evolutionary tree",
        "phylogenetic",
        "cladistic",
    ),
}


def classify_figure_type(caption: str | None, image_path: str | None = None) -> str:
    """Heuristically classify a figure's type from its caption text.

    Returns one of: ``plate``, ``range_chart``, ``map``,
    ``strat_column``, ``litholog_column``, ``paleogeographic_map``,
    ``photo``, ``schematic``, ``diagram``, ``reconstruction``,
    ``phylogenetic``, ``other``.

    The classifier is caption-only (no vision) and intentionally
    conservative — the default for any caption that doesn't clearly match
    one of the keyword lists is ``other`` (which is treated as ``plate``
    by the downstream pipeline, preserving the existing SEM-image flow).
    The vision-based range-chart extraction only runs when the caption
    explicitly mentions distribution/range/biozone keywords, which is
    the safest gate: false positives are cheap (an extra API call), but
    false negatives silently lose the geological linkage.

    Detection order matters:
      1. plate (with range_chart override)
      2. map (before range_chart to avoid place-name false positives)
      3. strat_column / litholog_column / paleogeographic_map (more
         specific than range_chart, so they take priority)
      4. range_chart (catch-all for distribution/range/biozone)
      5. photo
      6. schematic / diagram / reconstruction / phylogenetic
         (Phase 64 Plan B: conceptual / non-geographic figures)
      7. other (fallback → treated as plate downstream)
    """
    if not caption:
        return "other"
    low = caption.lower()
    # audit 2026-07-31: "plate" is a bare-substring keyword, so
    # "pelagic PLATEau of the Trapanese Domain" / "carbonate
    # PLATform" (both routine in geology texts) routed paleogeographic
    # maps / locality figures into the plate path — their captions
    # then got species-mined ("An attempt of" shipped as a species).
    # Match "plate"/"plates" as a whole word and veto the words that
    # merely contain the substring.
    if not any(w in low for w in _PLATE_SUBSTRING_VETO_WORDS) and _PLATE_WORD_RE.search(low):
        # Even if plate-like, check if the caption ALSO mentions
        # range/distribution — that overrides.
        for rc_kw in _FIGURE_TYPE_PROMPT_KEYWORDS["range_chart"]:
            if rc_kw in low:
                return "range_chart"
        return "plate"
    # Other plate-family keywords stay as substring matches.
    for kw in _FIGURE_TYPE_PROMPT_KEYWORDS["plate"][1:]:
        if kw in low:
            return "plate"
    # 2. Check the more specific paleogeographic_map BEFORE the
    # generic map because "paleogeographic map of..." contains the
    # substring "map of" which would otherwise match the map list.
    # The longer, more specific keyword must win.
    for kw in _FIGURE_TYPE_PROMPT_KEYWORDS["paleogeographic_map"]:
        if kw in low:
            return "paleogeographic_map"
    # 3. Check map BEFORE range_chart because map captions frequently
    # contain words like "Range" as geographic place names ("Nadanhada
    # Range", "Great Dividing Range") that have nothing to do with
    # species stratigraphic ranges. A "map" keyword match is a much
    # stronger signal than a bare "Range" in a place name.
    for mk in _FIGURE_TYPE_PROMPT_KEYWORDS["map"]:
        if mk in low:
            return "map"
    # 4. Check the more specific column types BEFORE the generic
    # range_chart catch-all so they route to the right vision prompt.
    for specific in ("strat_column", "litholog_column"):
        for kw in _FIGURE_TYPE_PROMPT_KEYWORDS[specific]:
            if kw in low:
                return specific
    # 4. Generic range_chart catch-all.
    for kw in _FIGURE_TYPE_PROMPT_KEYWORDS["range_chart"]:
        if kw in low:
            return "range_chart"
    # 5. Photo.
    for kw in _FIGURE_TYPE_PROMPT_KEYWORDS["photo"]:
        if kw in low:
            return "photo"
    # 6. Phase 64 Plan B: schematic / diagram / reconstruction /
    # phylogenetic. These are CONCEPTUAL figures — boxes, arrows,
    # cladograms — distinct from the geographic / stratigraphic /
    # photographic types above. We check phylogenetic first because
    # the keyword "phylogeny" is the most specific signal of the
    # group ("a phylogenetic diagram" should still classify as
    # phylogenetic, not diagram). Then reconstruction / schematic /
    # diagram in order of specificity.
    for specific in ("phylogenetic", "schematic", "diagram", "reconstruction"):
        for kw in _FIGURE_TYPE_PROMPT_KEYWORDS[specific]:
            if kw in low:
                return specific
    # 7. Fallback.
    return "other"


# --- Range chart extraction ----------------------------------------------------


@dataclass(slots=True)
class RangeChartSection:
    """A stratigraphic section in the chart."""

    name: str = ""
    age_range: str = ""
    formations: list[str] = field(default_factory=list)
    formation_thickness_m: str = ""
    coordinates: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpeciesRange:
    """One species' stratigraphic range within a section."""

    species: str = ""
    section: str = ""
    range_top: str = ""  # free-text bed/level name (e.g. "Bed 9")
    range_base: str = ""
    # Phase 3E audit 2026-08-19 (Bug M-11): the previous fields were
    # both free-text strings like "Bed 9" — unreadable for
    # biostratigraphy FAD/LAD analysis without a section-specific
    # legend. The new ``range_top_ma`` / ``range_base_ma`` carry the
    # numeric Ma values derived from a chart's Ma axis (if any);
    # ``None`` when the chart has no Ma axis or the model could not
    # read it. These are populated from the M3 prompt JSON
    # ``"range_top_ma"`` / ``"range_base_ma"`` fields (see the
    # extract_range_chart PROMPT) and never re-derived from bed
    # numbers.
    range_top_ma: float | None = None
    range_base_ma: float | None = None
    biozone: str = ""
    # Phase 62 Plan 5 (Bug 5.7): per-species confidence (0..1).
    # Distinct from the chart-wide ``RangeChartResult.confidence``:
    # a species can have a confident identity but an uncertain
    # chart-position reading (or vice versa). ``0.0`` is the
    # dataclass default and is also the sentinel used by the parser
    # to mean "JSON omitted this field — inherit chart-wide".
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BiozoneRecord:
    """A biozone identified in the chart."""

    name: str = ""
    age: str = ""
    thickness_m: str = ""
    # Phase 3E audit 2026-08-19 (Bug M-10): the previous fields were
    # zone-name-only. Real biostratigraphy distinguishes zone TYPES:
    #   * taxon-range zone ("N. optima Range Zone") — total range of
    #     a single marker taxon
    #   * concurrent-range zone ("N. optima – F. prisca Concurrent
    #     Range Zone") — overlap of two marker taxa
    #   * interval zone — bounded by two FAD/LAD events
    #   * assemblage zone — defined by an assemblage of species, no
    #     strict FAD/LAD
    # These four types govern how the zone is plotted on the Ma axis
    # (range zones plot at the FAD–LAD span; interval zones plot at
    # the boundary FADs; assemblage zones plot at the assemblage
    # span). The free-text ``name`` already encodes the type via
    # the "Range Zone" / "Concurrent Range Zone" / "Interval Zone" /
    # "Assemblage Zone" suffix, but downstream consumers want a
    # structured field. ``None`` when the chart label is ambiguous
    # or the model cannot determine the type.
    zone_type: str | None = None
    # Phase 6D audit 2026-08-19 NIT-2: structured citation for the
    # zone definition. Many published biozone names are ambiguous
    # without provenance — e.g. "N. optima Zone" was first defined
    # by Ishiga & Imoto (1982) but has been re-defined (and re-used
    # for different FAD/LAD pairs) by at least 4 subsequent authors.
    # Embed an optional ``zone_authority`` (surname string, e.g.
    # "Ishiga & Imoto") and ``zone_publication_year`` (4-digit int)
    # so downstream consumers can disambiguate when the same zone
    # name appears in two papers with different Ma bounds. Both
    # default to ``None`` for backward compatibility with existing
    # callers and gold-annotation files that only carry the name.
    zone_authority: str | None = None
    zone_publication_year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RangeChartResult:
    """Full extraction result for a single range-chart figure."""

    figure_id: str = ""
    paper_id: str = ""
    image_path: str = ""
    caption: str = ""
    sections: list[RangeChartSection] = field(default_factory=list)
    species_ranges: list[SpeciesRange] = field(default_factory=list)
    biozones: list[BiozoneRecord] = field(default_factory=list)
    other_fossils: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""
    # M21: surface API/parse failures to callers. Previously the
    # function returned an empty ``RangeChartResult`` for any HTTP
    # error, JSON parse error, or transport error — downstream code
    # couldn't distinguish "the API said no range data here" from
    # "we never got a response". ``status`` is ``"ok"`` on success
    # and ``"error"`` on any failure; ``error`` carries the human-
    # readable reason; ``error_message`` carries exception class +
    # message for programmatic consumption (audit 2026-08-01 M21).
    status: str = "ok"
    error: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "paper_id": self.paper_id,
            "image_path": self.image_path,
            "caption": self.caption,
            "sections": [s.to_dict() for s in self.sections],
            "species_ranges": [r.to_dict() for r in self.species_ranges],
            "biozones": [b.to_dict() for b in self.biozones],
            "other_fossils": list(self.other_fossils),
            "confidence": self.confidence,
            # Phase 55 audit: M21 — status and error were added to the dataclass
            # but to_dict() was hand-written and missed them. Without these,
            # callers cannot distinguish 'API returned 200 with no data' from
            # 'HTTP 500 / JSON decode error'.
            "status": self.status,
            "error": self.error,
            # Audit 2026-08-01 M21: programmatic error_message (exception
            # class + message) for callers that need structured failure
            # context (e.g. UI banners, log aggregators).
            "error_message": self.error_message,
        }


def _extract_balanced_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` JSON object substring of ``text``,
    or None if no balanced object exists. Handles nested braces correctly
    (the previous ``re.search(r"\\{.*\\}", ...)` regex was greedy and
    would swallow from the outermost ``{`` to the outermost ``}``, so
    ``{"a": 1, "b": {"c": 2}}`` produced ``{"a": 1, "b": {"c": 2}}``
    correctly only by luck; ``{"a": 1} extra {`` would match the whole
    string).
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        # No balanced close for this start; try the next "{"
        start = text.find("{", start + 1)
    return None


def _safe_json_loads(text: str) -> dict[str, Any]:
    """Lenient JSON object extraction. Same contract as ``m3_engine._safe_json_loads``
    but exposed locally to keep this module self-contained.

    Audit M3: M3 occasionally emits trailing fences / extra prose that
    breaks the strict ``json.loads`` first pass. The fallback
    ``_extract_balanced_json_object`` already handles the common case
    (it scans for the first ``{`` and matches the closing ``}`` while
    tracking brace depth + JSON string state). We rely on that
    fallback for trailing-junk handling instead of stripping fences
    aggressively (which would risk eating ``\\````` inside JSON string
    literals).
    """
    if not text:
        raise ValueError("empty text")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        # Audit 2026-09-01 BL-19: M3 sometimes wraps the whole response
        # in a JSON array (e.g. ``[{"sections": [...]}]``) instead of an
        # object. The previous code returned ``parsed`` unchanged and
        # the downstream ``dict``-only consumer then crashed. Wrap a
        # single-element array as a dict so the rest of the parser keeps
        # working; if the LLM emitted >1 element we keep only the first
        # and surface a debug log.
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                return first
    except Exception:
        pass
    candidate = _extract_balanced_json_object(text)
    if candidate is not None:
        return json.loads(candidate)
    raise ValueError(f"no JSON object found in {text[:120]!r}")


_RANGE_CHART_SYSTEM_PROMPT = """You are an expert in radiolarian biostratigraphy reading a stratigraphic range chart (also called a species distribution chart).

The chart shows measured sections on the vertical axis (top = young / Triassic, bottom = old / Permian) and species on the horizontal axis. Each species has a vertical line/bar showing its range across one or more sections.

Extract every piece of geological information visible in the chart as strict JSON with these fields:

{
  "sections": [
    {
      "name": "Pingdingshan" or "Majiashan" etc. (string),
      "age_range": "Late Permian (Late Changhsingian) – Early Triassic" (string),
      "formations": ["Talung Formation", "Yinkeng Formation"] (array of strings),
      "formation_thickness_m": "Talung: ~9 m; Yinkeng: 2 m" (string, free text),
      "coordinates": "31°N, 117°E" or "Not visible in chart" (string)
    }
  ],
  "species_ranges": [
    {
      "species": "Neoalbaillella optima" (string, full Latin binomial),
      "section": "Pingdingshan" (string, must match a section.name),
      "range_top": "Bed 9 (Yinkeng Fm base)" (string, the YOUNG/upper limit),
      "range_base": "Bed 7 (top Talung Fm)" (string, the OLD/lower limit),
      "range_top_ma": 251.9 (number, optional — if the chart has a Ma
        axis and you can read the top boundary, give the Ma here;
        null/omit when no Ma axis or unreadable),
      "range_base_ma": 252.5 (number, optional — old/lower limit in Ma,
        null/omit when no Ma axis or unreadable),
      "biozone": "N. optima Zone (latest Changhsingian)" (string, optional),
      "confidence": 0.0-1.0 reflecting your certainty in THIS ROW specifically
        (species identity + chart position). Distinct from the chart-wide
        "confidence" below.
    }
  ],
  "biozones": [
    {
      "name": "Neoalbaillella optima Zone" (string),
      "age": "Latest Changhsingian (Late Permian)" (string),
      "thickness_m": "Pingdingshan: ~3 m" (string),
      "zone_type": "range" (string, optional — one of
        "range" / "concurrent range" / "interval" / "assemblage";
        pick the closest match to the chart label, or null if unclear),
      "zone_authority": "Ishiga & Imoto" (string, optional — the
        surname(s) of the author(s) who first defined the zone, e.g.
        "Ishiga & Imoto", "Sanfilippo & Nigrini", "Riedel"; only
        emit when the chart label or legend explicitly names the
        defining authority; null if unclear),
      "zone_publication_year": 1982 (int, optional — 4-digit year
        of the original defining publication; matches the
        ``zone_authority``; null if no authority was named or the
        year is not visible in the chart)
    }
  ],
  "other_fossils": [
    "Ammonoid: Pleuronodoceras sp. (Pingdingshan, bed 9, Yinkeng Fm)" (free text entries)
  ],
  "confidence": 0.0-1.0 reflecting your certainty in the extraction overall
}

Rules:
- Read the species names carefully — chart text is often small and OCR can
  produce slight misreads (e.g. "Pteruridonrceras" should be corrected
  to the real ammonoid genus if the chart is clearly showing Pleuronodoceras).
- Only extract what you can READ from the chart. Do not invent.
- If a species appears in multiple sections, emit one entry per section.
- Preserve bed/level numbers exactly as printed.
- Return JSON only, no markdown fences, no commentary."""


def extract_range_chart(
    *,
    paper_id: str,
    figure_id: str,
    caption: str,
    image_path: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout_sec: int = 120,
) -> RangeChartResult:
    """Extract geological info from a stratigraphic range chart image.

    Parameters
    ----------
    paper_id, figure_id : str
        Identifiers propagated into the result so downstream code can
        join range-chart records to per-panel records by figure_id.
    caption : str
        Figure caption; included in the prompt for grounding.
    image_path : str
        Absolute path to the figure image (PNG/JPG). Must be readable.
    api_key, base_url, model : str
        MiniMax-Anthropic-compatible API configuration. The caller is
        responsible for sourcing these from the active environment
        (typically loaded from ``.env``).
    timeout_sec : int
        Request timeout for the API call.

    Returns
    -------
    RangeChartResult with the parsed fields. On error, returns a result
    with empty lists and confidence=0.0; never raises — the caller
    decides how to handle the failure (typically: log + skip).
    """
    import base64
    import mimetypes

    result = RangeChartResult(
        figure_id=figure_id,
        paper_id=paper_id,
        image_path=image_path,
        caption=caption or "",
    )

    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
    except OSError as exc:
        logger.warning("range_chart: cannot open %s: %s", image_path, exc)
        # M21 (audit 2026-08-01): previously this returned a
        # ``status="ok"`` empty result — indistinguishable from
        # "API said no data here". Mark as error and surface the
        # exception class + message so callers can distinguish
        # transport failures from genuine empty extractions.
        result.status = "error"
        result.error = f"OSError: cannot open {image_path}: {exc}"
        result.error_message = f"OSError: {exc}"
        return result

    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/png"
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    user_prompt = (
        f"Paper: {paper_id}\nFigure: {figure_id}\n\nCaption:\n{caption or '(no caption)'}\n\n"
        "Extract the geological information as the strict JSON contract."
    )

    resp = None
    try:
        # Audit H4: use ``with`` so the response is closed even if a
        # non-RequestException (e.g. MemoryError, urllib3 internal
        # error) escapes ``requests.post``. Pre-fix, ``resp`` could
        # remain undefined when the post raised; the trailing
        # ``finally`` block then raised NameError, both masking the
        # real error and leaking the connection.
        try:
            with requests.post(
                f"{base_url.rstrip('/')}/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 4000,
                    "system": _RANGE_CHART_SYSTEM_PROMPT,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime,
                                        "data": img_b64,
                                    },
                                },
                                {"type": "text", "text": user_prompt},
                            ],
                        }
                    ],
                },
                timeout=timeout_sec,
            ) as _resp:
                resp = _resp
                if resp.status_code != 200:
                    logger.warning(
                        "range_chart API returned %d for %s/%s: %s",
                        resp.status_code,
                        paper_id,
                        figure_id,
                        resp.text[:200],
                    )
                    # M21: don't return a silent empty result. Mark
                    # the result as an error and attach a short
                    # description so callers / the UI can surface
                    # the actual failure mode.
                    result.status = "error"
                    result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    result.error_message = f"HTTPError: status={resp.status_code}"
                    return result
                try:
                    payload = resp.json()
                except ValueError as exc:
                    # M21: JSON-decode failures used to be silent
                    # (empty result, no error). Mark them as errors
                    # with the underlying message.
                    result.status = "error"
                    result.error = f"JSON decode error: {exc}"
                    result.error_message = f"ValueError: {exc}"
                    return result
        except requests.RequestException as exc:
            logger.warning("range_chart API call failed for %s/%s: %s", paper_id, figure_id, exc)
            # M21: transport-level failures (DNS, connect, timeout)
            # used to return an empty result indistinguishable from
            # success-with-no-data. Mark as error.
            result.status = "error"
            exc_cls = exc.__class__.__name__
            result.error = str(exc)
            result.error_message = f"{exc_cls}: {exc}"
            return result
    finally:
        # Belt-and-braces: ``with`` already closed, but if the
        # assignment to ``resp`` itself failed before ``with`` entered,
        # we still need to drop the attribute to allow GC.
        resp = None

    raw_text = ""
    for c in payload.get("content", []):
        if c.get("type") == "text":
            raw_text = c.get("text", "")
            break
    result.raw_response = raw_text

    try:
        parsed = _safe_json_loads(raw_text)
    except ValueError as exc:
        logger.warning("range_chart JSON parse failed for %s/%s: %s", paper_id, figure_id, exc)
        # M21 (audit 2026-08-01): previously this returned a
        # ``status="ok"`` empty result — indistinguishable from
        # "API said no data here". Mark as error and surface the
        # exception class + message so callers can distinguish
        # JSON-parse failures from genuine empty extractions.
        result.status = "error"
        result.error = f"JSON parse error: {exc}"
        result.error_message = f"ValueError: {exc}"
        return result

    return _parse_extraction_response(
        parsed=parsed, paper_id=paper_id, figure_id=figure_id, base_result=result
    )


def _parse_extraction_response(
    *,
    parsed: dict[str, Any],
    paper_id: str,
    figure_id: str,
    base_result: RangeChartResult | None = None,
) -> RangeChartResult:
    """Populate ``RangeChartResult`` fields from the parsed JSON dict.

    Phase 62 Plan 5 (Bug 5.7): each ``SpeciesRange`` now carries its
    own ``confidence`` field (the LLM's per-row certainty), distinct
    from the chart-wide ``result.confidence``. When the JSON omits
    the per-row value, the species inherits the chart-wide value
    (backward-compat fallback).

    Extracted from ``extract_range_chart`` so tests can drive the
    JSON → dataclass conversion without making a real API call.
    """
    result = base_result or RangeChartResult(figure_id=figure_id, paper_id=paper_id)
    for sec in parsed.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        result.sections.append(
            RangeChartSection(
                name=str(sec.get("name", "")),
                age_range=str(sec.get("age_range", "")),
                formations=list(sec.get("formations") or []),
                formation_thickness_m=str(sec.get("formation_thickness_m", "")),
                coordinates=str(sec.get("coordinates", "")),
            )
        )
    for sp in parsed.get("species_ranges") or []:
        if not isinstance(sp, dict):
            continue
        # Phase 62 Plan 5 (Bug 5.7): per-species confidence. Read
        # ``sp["confidence"]`` if present; otherwise fall back to the
        # chart-wide confidence parsed below. We stash the raw value
        # first and rewrite it after parsing the chart-wide conf so
        # the fallback uses the same post-clamp value.
        try:
            raw_sp_conf = float(sp.get("confidence", 0.0))
        except (TypeError, ValueError):
            raw_sp_conf = 0.0
        # Phase 3E audit 2026-08-19 (Bug M-11): numeric Ma bounds
        # parsed from the JSON's ``range_top_ma`` / ``range_base_ma``
        # fields. These default to ``None`` when the chart has no Ma
        # axis, the model omitted the field, or the value is not
        # finite. We do NOT silently derive Ma from bed numbers —
        # "Bed 9" without a section-specific legend is unreadable,
        # and pretending otherwise would corrupt the biostratigraphy
        # column on the Web UI / xlsx export.
        try:
            raw_top_ma = sp.get("range_top_ma")
            raw_top_ma = float(raw_top_ma) if raw_top_ma is not None else None
            if raw_top_ma is not None and not math.isfinite(raw_top_ma):
                raw_top_ma = None
        except (TypeError, ValueError):
            raw_top_ma = None
        try:
            raw_base_ma = sp.get("range_base_ma")
            raw_base_ma = float(raw_base_ma) if raw_base_ma is not None else None
            if raw_base_ma is not None and not math.isfinite(raw_base_ma):
                raw_base_ma = None
        except (TypeError, ValueError):
            raw_base_ma = None
        result.species_ranges.append(
            SpeciesRange(
                species=str(sp.get("species", "")),
                section=str(sp.get("section", "")),
                range_top=str(sp.get("range_top", "")),
                range_base=str(sp.get("range_base", "")),
                range_top_ma=raw_top_ma,
                range_base_ma=raw_base_ma,
                biozone=str(sp.get("biozone", "")),
                confidence=raw_sp_conf,
            )
        )
    for bz in parsed.get("biozones") or []:
        if not isinstance(bz, dict):
            continue
        # Phase 3E audit 2026-08-19 (Bug M-10): parse the optional
        # ``zone_type`` from the JSON. Accepted values mirror the
        # standard ICS / Salvador biozone typology: "range",
        # "concurrent range", "interval", "assemblage". Any other
        # value is stored verbatim (forward compat) so unusual
        # labels from an LLM are not silently dropped. ``None`` when
        # the JSON omitted the field — downstream code can then fall
        # back to parsing the ``name`` suffix.
        zt_raw = bz.get("zone_type")
        if isinstance(zt_raw, str) and zt_raw.strip():
            zone_type = zt_raw.strip()
        else:
            zone_type = None
        # Phase 6D audit 2026-08-19 NIT-2: parse the optional
        # ``zone_authority`` and ``zone_publication_year``. Both
        # fields default to ``None`` when omitted / non-string /
        # non-integer. The year is clamped to a 4-digit positive
        # int (1700-2100) to defend against LLM hallucinations like
        # 198 (truncated), 0, or 99999 that would otherwise reach
        # the GBIF / Darwin Core export.
        za_raw = bz.get("zone_authority")
        if isinstance(za_raw, str) and za_raw.strip():
            zone_authority: str | None = za_raw.strip()
        else:
            zone_authority = None
        zy_raw = bz.get("zone_publication_year")
        zone_publication_year: int | None = None
        if zy_raw is not None:
            try:
                zy_int = int(zy_raw)
                if 1700 <= zy_int <= 2100:
                    zone_publication_year = zy_int
            except (TypeError, ValueError):
                zone_publication_year = None
        result.biozones.append(
            BiozoneRecord(
                name=str(bz.get("name", "")),
                age=str(bz.get("age", "")),
                thickness_m=str(bz.get("thickness_m", "")),
                zone_type=zone_type,
                zone_authority=zone_authority,
                zone_publication_year=zone_publication_year,
            )
        )
    of = parsed.get("other_fossils") or []
    if isinstance(of, list):
        result.other_fossils = [str(x) for x in of if str(x).strip()]
    try:
        result.confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        result.confidence = 0.0
    # M21: clamp confidence into [0.0, 1.0]. The LLM occasionally
    # emits NaN, Inf, or out-of-range floats (e.g. 1.5 or -0.1);
    # downstream sort/display code then does the wrong thing
    # (NaN sort order is undefined, 1.5 breaks colour-bar binning).
    # Treat non-finite values as 0.0 and clip the rest into range.
    if not math.isfinite(result.confidence):
        result.confidence = 0.0
    result.confidence = max(0.0, min(1.0, result.confidence))

    # Phase 62 Plan 5 (Bug 5.7): any species whose JSON omitted a
    # per-row confidence value inherits the (now clamped) chart-wide
    # value. Species that DID provide a per-row value keep theirs
    # (clamped to [0,1] for safety).
    for sr in result.species_ranges:
        if not math.isfinite(sr.confidence):
            sr.confidence = 0.0
        sr.confidence = max(0.0, min(1.0, sr.confidence))
        # 0.0 is the sentinel we wrote when JSON omitted the field.
        # Promote to chart-wide confidence in that case so the
        # emitted link is honest about the model's uncertainty.
        if sr.confidence == 0.0 and parsed.get("confidence") is not None:
            # Only promote if the JSON had a chart-wide confidence.
            # If the chart-wide is also 0.0, leave as 0.0 (model gave
            # us nothing — don't pretend otherwise).
            sr.confidence = result.confidence

    logger.info(
        "range_chart %s/%s: %d sections, %d species_ranges, %d biozones, conf=%.2f",
        paper_id,
        figure_id,
        len(result.sections),
        len(result.species_ranges),
        len(result.biozones),
        result.confidence,
    )
    return result


# --- Linking range chart results to panel records -------------------------------

# A loose species-name matching strategy. We accept a match when the
# normalized range-chart species shares its first 2 words with the panel's
# normalized species (or the first word when the panel species is just a
# genus name like "Entactinia"). This is intentionally permissive so a
# minor OCR noise ("Follicucullus sp." vs "Follicucullus charveti")
# doesn't drop the link; downstream can filter low-confidence links.


_QUALIFIER_RE = re.compile(r"\b(cf|aff)\b\.?", re.IGNORECASE)


def _norm(s: str | None) -> str:
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+sp\.?$", "", s, flags=re.IGNORECASE).strip()
    # Audit 2026-09-04 taxon-4: cf. and aff. are now PRESERVED in the
    # normalised form so ``_species_match`` can distinguish them. The
    # previous deletion collapsed "cf. jamesi" onto "jamesi" and made
    # the range-chart linker attribute a panel's uncertain
    # determination to the definite chart species — corrupting the
    # biozone/age metadata and downstream PBDB/GBIF submissions.
    return s.lower()


def _qualifier(s: str) -> str | None:
    """Return "cf" | "aff" | None from a normalised species string.

    The qualifier is what differentiates ICZN's two open-nomenclature
    assertions — cf. (tentative identification) and aff. (similar but
    distinct species) — from a definite determination. Range-chart
    linkage must never fuse one qualifier into another.
    """
    m = _QUALIFIER_RE.search(s)
    if not m:
        return None
    return m.group(1).lower()


def _species_match(rc_species: str, panel_species: str) -> bool:
    """Loose match: same genus (first word) + compatible epithet.

    Audit 2026-09-04 taxon-4: the qualifier (cf./aff./none) on each
    side must agree. cf./aff. are different ICZN assertions and must
    not collapse onto the definite species — see the BLOCKER audit
    note on biozone attribution below.
    """
    a = _norm(rc_species)
    b = _norm(panel_species)
    if not a or not b:
        return False
    # Qualifier must match (or both be absent). cf vs aff is a
    # mismatch; either vs definite is a mismatch.
    a_q = _qualifier(a)
    b_q = _qualifier(b)
    if a_q != b_q:
        return False
    # Strip the qualifier for the rest of the comparison.
    a_clean = _QUALIFIER_RE.sub("", a).strip()
    b_clean = _QUALIFIER_RE.sub("", b).strip()
    if a_clean == b_clean:
        return True
    a_parts = a_clean.split()
    b_parts = b_clean.split()
    if not a_parts or not b_parts:
        return False
    if a_parts[0] != b_parts[0]:
        return False
    # Same genus. If one is just the genus name, accept.
    if len(a_parts) == 1 or len(b_parts) == 1:
        return True
    # If epithets are very close (allow OCR 1-char typos), accept.
    if len(a_parts) >= 2 and len(b_parts) >= 2:
        e1, e2 = a_parts[1], b_parts[1]
        if e1 == e2:
            return True
        # Audit 2026-09-03 (BLOCKER-#8): the previous 1-char tolerance
        # (``diffs <= 1``) silently mis-paired sibling species that
        # differ by exactly one letter, e.g. ``Cryptocapsa tecta``
        # vs. ``Cryptocapsa texta`` (De Wever 2001 separates these
        # as distinct species) and ``Parvicingula jamesi`` vs.
        # ``Parvicingula jonesi`` (Bandini 2011). When range-chart
        # noise created a "link" between the chart species and the
        # wrong sibling on the panel, the wrong occurrence was
        # attributed to the chart species' sample — corrupting
        # downstream PBDB / GBIF submissions.
        # The same-genus disambiguation in
        # ``build_geology_links_for_panels`` already handles the
        # case where the panel is a bare genus with >=2 chart
        # siblings, so removing this tolerance does NOT regress
        # the bare-genus case. Tightened to exact epithet match.
        # (If OCR noise dominates, the upstream
        # ``_normalize_ocr_chars`` in ocr_corrections handles
        # the l/1 / I/l / long-vowel class of confusions before
        # the string ever reaches here.)
    return False


def build_geology_links_for_panels(
    chart: RangeChartResult,
    panel_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build ``GeologyLinkRecord`` dicts to attach to per-panel records.

    For each (panel, range-chart species) pair where ``_species_match``
    is True, emit a link carrying the chart's section, age_range, biozone,
    formation_top/base, and the range-chart's confidence. The link's
    ``evidence_text`` includes the raw range/bed strings so an operator
    can audit the linkage.
    """
    if not chart.species_ranges:
        return []

    links: list[dict[str, Any]] = []
    # Index by (section, species) for faster lookup, with a fallback to
    # (section, None) so a panel that has no section info can still link
    # to a species-wide range.
    by_section_species: dict[tuple[str, str], SpeciesRange] = {}
    by_species: dict[str, list[SpeciesRange]] = {}
    for sr in chart.species_ranges:
        by_section_species[(sr.section.strip().lower(), _norm(sr.species))] = sr
        by_species.setdefault(_norm(sr.species), []).append(sr)

    # Section metadata lookup
    section_meta: dict[str, RangeChartSection] = {}
    for s in chart.sections:
        section_meta[s.name.strip().lower()] = s

    for pr in panel_records:
        pspecies = str(pr.get("species") or "").strip()
        if not pspecies:
            continue
        ps_norm = _norm(pspecies)
        # Direct hit: panel species exact-matches a range species
        candidates = by_species.get(ps_norm, [])
        if not candidates and len(ps_norm.split()) == 1:
            # Panel has only a genus name — collect ALL range-chart
            # species in that genus. Audit 2026-08-01 (M3): previously
            # we took the FIRST genus-prefix match silently, which
            # linked bare-genus panels to whichever species happened
            # to be encountered first in the chart (e.g. Bandini 2006
            # ``Archaeodictyomitra`` → ``A. rigida`` when the chart
            # actually carries 5 species of the genus). Disambiguation
            # requires species-level matching: if the chart has ≥ 2
            # distinct species under this genus, we cannot pick one
            # without guessing, so we skip the link entirely.
            genus_candidates: list[SpeciesRange] = []
            for sp_key, sp_list in by_species.items():
                if sp_key.startswith(ps_norm + " "):
                    genus_candidates.extend(sp_list)
            distinct_species = {sr.species for sr in genus_candidates}
            if len(distinct_species) >= 2:
                logger.info(
                    "range_chart: panel species %r is bare genus with "
                    "%d distinct species in chart %s — skipping ambiguous link",
                    pspecies,
                    len(distinct_species),
                    chart.figure_id,
                )
                continue
            candidates = genus_candidates
        for sr in candidates:
            if not _species_match(sr.species, pspecies):
                continue
            sec_meta = section_meta.get(sr.section.strip().lower())
            link = {
                "age": sec_meta.age_range if sec_meta else None,
                "chronostratigraphy": sr.biozone or None,
                "chronostratigraphy_rank": "biozone" if sr.biozone else None,
                "formation": (
                    ", ".join(sec_meta.formations) if sec_meta and sec_meta.formations else None
                ),
                "locality": sr.section or None,
                "latitude": None,
                "longitude": None,
                "section_type": "stratigraphic_column",
                # P1-5 fix: use actual section name, not figure ID
                "section_title": sr.section or chart.figure_id,
                "evidence_text": (
                    f"range_chart_vision[{chart.figure_id}]: {sr.species} "
                    f"in section {sr.section or '?'}, range {sr.range_base} → {sr.range_top}, "
                    f"biozone={sr.biozone or '?'}"
                ),
                # Phase 62 Plan 5 (Bug 5.7): per-species confidence.
                # Previously this stamped chart-wide confidence on
                # every link; now each species carries its own row
                # confidence (or inherits the chart-wide value when
                # the LLM omitted it). The species name is included
                # so downstream consumers can join species records
                # back to panels.
                "confidence": sr.confidence,
                "species": sr.species,
            }
            links.append(link)
    return links
