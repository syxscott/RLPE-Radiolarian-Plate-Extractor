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
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)


# --- Figure type classification ------------------------------------------------

_FIGURE_TYPE_PROMPT_KEYWORDS = {
    # Map: keyword -> figure type. Checked in order; first hit wins.
    "plate": (
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
        "location map",
        "geographic distribution",
        "paleogeographic map",
        "palaeogeographic",
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
    ),
    "photo": (
        "field photograph",
        "outcrop photograph",
        "field photo",
    ),
}


def classify_figure_type(caption: str | None, image_path: str | None = None) -> str:
    """Heuristically classify a figure's type from its caption text.

    Returns one of: ``plate``, ``range_chart``, ``map``, ``photo``, ``other``.

    The classifier is caption-only (no vision) and intentionally
    conservative — the default for any caption that doesn't clearly match
    one of the keyword lists is ``other`` (which is treated as ``plate``
    by the downstream pipeline, preserving the existing SEM-image flow).
    The vision-based range-chart extraction only runs when the caption
    explicitly mentions distribution/range/biozone keywords, which is
    the safest gate: false positives are cheap (an extra API call), but
    false negatives silently lose the geological linkage.
    """
    if not caption:
        return "other"
    low = caption.lower()
    # Check plate first because plates often co-occur with range charts
    # in a paper but the keyword "distribution of radiolarians" is what
    # marks the range chart, not a plate caption that mentions
    # "distribution" in passing.
    for kw in _FIGURE_TYPE_PROMPT_KEYWORDS["plate"]:
        if kw in low:
            # Even if plate-like, check if the caption ALSO mentions
            # range/distribution — that overrides.
            for rc_kw in _FIGURE_TYPE_PROMPT_KEYWORDS["range_chart"]:
                if rc_kw in low:
                    return "range_chart"
            return "plate"
    # Check map BEFORE range_chart because map captions frequently
    # contain words like "Range" as geographic place names ("Nadanhada
    # Range", "Great Dividing Range") that have nothing to do with
    # species stratigraphic ranges. A "map" keyword match is a much
    # stronger signal than a bare "Range" in a place name.
    for mk in _FIGURE_TYPE_PROMPT_KEYWORDS["map"]:
        if mk in low:
            return "map"
    for ftype, keywords in _FIGURE_TYPE_PROMPT_KEYWORDS.items():
        if ftype in ("plate", "map"):
            continue
        for kw in keywords:
            if kw in low:
                return ftype
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
    biozone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BiozoneRecord:
    """A biozone identified in the chart."""

    name: str = ""
    age: str = ""
    thickness_m: str = ""

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
    but exposed locally to keep this module self-contained."""
    if not text:
        raise ValueError("empty text")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
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
      "biozone": "N. optima Zone (latest Changhsingian)" (string, optional)
    }
  ],
  "biozones": [
    {
      "name": "Neoalbaillella optima Zone" (string),
      "age": "Latest Changhsingian (Late Permian)" (string),
      "thickness_m": "Pingdingshan: ~3 m" (string)
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
        return result

    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/png"
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    user_prompt = (
        f"Paper: {paper_id}\nFigure: {figure_id}\n\nCaption:\n{caption or '(no caption)'}\n\n"
        "Extract the geological information as the strict JSON contract."
    )

    try:
        resp = requests.post(
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
        )
    except requests.RequestException as exc:
        logger.warning("range_chart API call failed for %s/%s: %s", paper_id, figure_id, exc)
        return result

    # Always close the HTTP connection. ``requests`` will GC it, but
    # for a long-running pipeline that processes many figures an
    # explicit close is safer — leaked connections can exhaust the
    # OS's per-process FD limit. Use try/finally so the close runs
    # on ALL exit paths (status_code != 200, payload parse failure,
    # success).
    try:
        if resp.status_code != 200:
            logger.warning(
                "range_chart API returned %d for %s/%s: %s",
                resp.status_code,
                paper_id,
                figure_id,
                resp.text[:200],
            )
            return result
        try:
            payload = resp.json()
        except ValueError:
            return result
    finally:
        try:
            resp.close()
        except Exception:
            pass

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
        return result

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
        result.species_ranges.append(
            SpeciesRange(
                species=str(sp.get("species", "")),
                section=str(sp.get("section", "")),
                range_top=str(sp.get("range_top", "")),
                range_base=str(sp.get("range_base", "")),
                biozone=str(sp.get("biozone", "")),
            )
        )
    for bz in parsed.get("biozones") or []:
        if not isinstance(bz, dict):
            continue
        result.biozones.append(
            BiozoneRecord(
                name=str(bz.get("name", "")),
                age=str(bz.get("age", "")),
                thickness_m=str(bz.get("thickness_m", "")),
            )
        )
    of = parsed.get("other_fossils") or []
    if isinstance(of, list):
        result.other_fossils = [str(x) for x in of if str(x).strip()]
    try:
        result.confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        result.confidence = 0.0

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


def _norm(s: str | None) -> str:
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+sp\.?$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+cf\.?\s+", " ", s, flags=re.IGNORECASE).strip()
    return s.lower()


def _species_match(rc_species: str, panel_species: str) -> bool:
    """Loose match: same genus (first word) + compatible epithet."""
    a = _norm(rc_species)
    b = _norm(panel_species)
    if not a or not b:
        return False
    if a == b:
        return True
    a_parts = a.split()
    b_parts = b.split()
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
        # 1-char Levenshtein-ish tolerance for short epithets.
        if len(e1) >= 5 and len(e2) >= 5:
            diffs = sum(1 for x, y in zip(e1, e2) if x != y)
            if diffs <= 1 and abs(len(e1) - len(e2)) <= 1:
                return True
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
            # Panel has only a genus name — accept any range-chart
            # species in the same genus.
            for sp_key, sp_list in by_species.items():
                if sp_key.startswith(ps_norm + " "):
                    candidates = sp_list
                    break
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
                "section_title": chart.figure_id,
                "evidence_text": (
                    f"range_chart_vision[{chart.figure_id}]: {sr.species} "
                    f"in section {sr.section or '?'}, range {sr.range_base} → {sr.range_top}, "
                    f"biozone={sr.biozone or '?'}"
                ),
                "confidence": chart.confidence,
            }
            links.append(link)
    return links
