from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any

AGE_PATTERN = re.compile(
    r"\b(?:Early|Middle|Late|Lower|Upper)\s+[A-Z][a-z]+|"
    r"\b(?:Precambrian|Cambrian|Ordovician|Silurian|Devonian|Carboniferous|Permian|"
    r"Triassic|Jurassic|Cretaceous|Paleocene|Eocene|Oligocene|Miocene|Pliocene|"
    r"Pleistocene|Holocene)\b",
    re.IGNORECASE,
)
FORMATION_PATTERN = re.compile(r"\b([A-Z][A-Za-z\-\s]+(?:Formation|Member|Group|Fm\.|Mb\.|Gp\.))\b")
# Locality phrase: "from/at/in <Capitalised name>" — bounded so the
# match doesn't run past the immediate locality token. The previous
# pattern ``[A-Za-z\-\s]{2,80}`` was too permissive: ``\s`` allowed
# arbitrary cross-sentence runs (e.g. "from California is the type"
# matched "California is the type"). Now: capture a Capitalised word
# optionally followed by 1-3 more Capitalised tokens (proper-noun
# locality phrase like "Southwest Japan", "New South Wales"); bail
# out at the first lowercase word, end-of-sentence punctuation, or
# numeric token.
LOCALITY_PATTERN = re.compile(
    r"\b(?:from|at|in)\s+"
    r"([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3})"
    r"(?=\s*[,.;:()]|\s+(?:and|the|of|a|an|is|are|was|were|in|at)\b|$)"
)
COORDINATE_PATTERN = re.compile(
    r"\b(\d{1,3}(?:\.\d+)?)\s*°?\s*([NSns])?[,\s]+(\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])?\b"
)


@dataclass(slots=True)
class GeologyRecord:
    age: str | None = None                              # period name (e.g. "Permian")
    chronostratigraphy: str | None = None               # most specific stage (e.g. "Changhsingian")
    chronostratigraphy_rank: str | None = None          # "period" | "epoch" | "age"
    formation: str | None = None
    locality: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    section_type: str | None = None
    section_title: str | None = None
    evidence_text: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_geology_from_sections(sections: list[dict[str, str]]) -> list[GeologyRecord]:
    out: list[GeologyRecord] = []
    # Lazy import to avoid circular
    try:
        from .stratigraphy import classify_age_string, find_ages_in_text
    except Exception:
        find_ages_in_text = None
        classify_age_string = None
    for sec in sections:
        text = sec.get("text", "")
        if not text:
            continue
        ages = [m.group(0).strip() for m in AGE_PATTERN.finditer(text)]
        forms = [m.group(1).strip() for m in FORMATION_PATTERN.finditer(text)]
        locs = [m.group(1).strip(" .,;") for m in LOCALITY_PATTERN.finditer(text)]

        # Stratigraphy enrichment — find stage names (Changhsingian, Wuchiapingian, …)
        chrono = None
        chrono_rank = None
        if find_ages_in_text is not None:
            try:
                cls_list = find_ages_in_text(text)
                # Pick the most specific one (rank="age" > "epoch" > "period")
                rank_order = {"age": 3, "epoch": 2, "period": 1}
                best = max(
                    [c for c in cls_list if c.confidence > 0],
                    key=lambda c: rank_order.get(c.rank or "", 0),
                    default=None,
                )
                if best is not None:
                    chrono = best.age or best.period
                    chrono_rank = best.rank
            except Exception:
                pass
        # Coordinate parsing. ``_extract_first_coord`` already validates
        # the latitude/longitude ranges (rejects e.g. lat=200, lon=400)
        # and applies hemisphere flips. The previous code re-ran the
        # regex and overwrote lat/lon with values that BYPASSED the
        # range check, which let invalid coordinates leak into
        # GeologyRecord.latitude/longitude.
        lat, lon = _extract_first_coord(text)

        if not ages and not forms and not locs and chrono is None and lat is None:
            continue

        # 以句子级片段做证据，先走规则抽取。
        # If we have chrono from stratigraphy, use that as age.
        primary_age = chrono if chrono else (ages[0] if ages else None)
        for age in (ages or [None]):
            rec = GeologyRecord(
                age=age or primary_age,
                chronostratigraphy=chrono,
                chronostratigraphy_rank=chrono_rank,
                formation=forms[0] if forms else None,
                locality=locs[0] if locs else None,
                latitude=lat,
                longitude=lon,
                section_type=sec.get("section_type"),
                section_title=sec.get("title"),
                evidence_text=text[:300],
                confidence=0.7 if chrono else 0.55,
            )
            out.append(rec)
        # If no ages were found but chrono was, still emit a record
        if not ages and chrono:
            rec = GeologyRecord(
                age=primary_age,
                chronostratigraphy=chrono,
                chronostratigraphy_rank=chrono_rank,
                formation=forms[0] if forms else None,
                locality=locs[0] if locs else None,
                latitude=lat,
                longitude=lon,
                section_type=sec.get("section_type"),
                section_title=sec.get("title"),
                evidence_text=text[:300],
                confidence=0.7,
            )
            # Compare on a stable key tuple instead of the full dataclass.
            # `dataclass(slots=True)` gives an auto-generated __eq__ that
            # works today, but ``rec not in out`` does linear scans and
            # would silently fail (always True) if a future field was
            # excluded from eq=. The key tuple is the same one
            # ``dedup_geology_records`` uses below, so the two paths are
            # consistent.
            key = (
                rec.age, rec.chronostratigraphy, rec.formation,
                rec.locality, rec.section_title,
            )
            existing_keys = {
                (r.age, r.chronostratigraphy, r.formation,
                 r.locality, r.section_title)
                for r in out
            }
            if key not in existing_keys:
                out.append(rec)
    return dedup_geology_records(out)


def _extract_first_coord(text: str) -> tuple[float | None, float | None]:
    """Best-effort coordinate extraction. Returns ``(lat, lon)`` or ``(None, None)``.

    Requires at least one hemisphere indicator (N/S/E/W) or a degree symbol
    (°) in the match. Without this filter, bare number pairs like "45, 90"
    (page numbers, specimen dimensions, figure counts) would be falsely
    parsed as coordinates.
    """
    if not text:
        return None, None
    for m in COORDINATE_PATTERN.finditer(text):
        # Require at least one hemisphere indicator or degree symbol to
        # reduce false positives from bare number pairs.
        has_hemisphere = bool(m.group(2) or m.group(4))
        has_degree = "°" in m.group(0)
        if not has_hemisphere and not has_degree:
            continue
        try:
            lat = float(m.group(1))
            if m.group(2) and m.group(2).upper() == "S":
                lat = -lat
            lon = float(m.group(3))
            if m.group(4) and m.group(4).upper() == "W":
                lon = -lon
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            return lat, lon
        except Exception:
            continue
    return None, None


def link_species_to_geology(
    species_names: list[str],
    sections: list[dict[str, str]],
    llm_runtime: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Link species to geology records.
    If llm_runtime is provided, use LLM for relation refinement; else use proximity heuristics.
    """
    geology = extract_geology_from_sections(sections)
    links: dict[str, list[dict[str, Any]]] = {s: [] for s in species_names}
    if not species_names:
        return links

    if llm_runtime is None:
        # Heuristic linking: a species is linked to a geology record
        # ONLY when the species name actually appears in the same
        # section as the record. The previous implementation also
        # appended a fallback record to every unmatched species,
        # which produced the same 5-record dump for every panel of
        # a paper that uses generic any-age captions. The new
        # behaviour is honest: if the operator has no LLM and no
        # OCR, a panel with no detectable species ends up with NO
        # geology links rather than fabricated ones.
        for s in species_names:
            s_lower = s.lower()
            for sec in sections:
                text = (sec.get("text") or "").lower()
                if s_lower in text:
                    for rec in geology:
                        if rec.section_title == sec.get("title"):
                            links[s].append(rec.to_dict())
        return links

    # 使用 LLM 关系链接。
    from .gemma_postprocess import gemma_extract_text_json

    system_prompt = (
        "You are a scientific IE assistant. Given species name and section text, "
        "extract linked geology fields as strict JSON with keys: label,species,confidence,reasoning."
    )
    for s in species_names:
        best_records: list[dict[str, Any]] = []
        for sec in sections:
            text = sec.get("text", "")
            if not text:
                continue
            user_prompt = (
                f"Species: {s}\nSection title: {sec.get('title')}\n"
                f"Section type: {sec.get('section_type')}\n"
                f"Text: {text[:1500]}\n"
                "Return JSON: {\"label\":\"geo_link\",\"species\":\"...\",\"confidence\":0-1,\"reasoning\":\"age=...,formation=...,locality=...\"}"
            )
            out = gemma_extract_text_json(llm_runtime, system_prompt, user_prompt)
            conf = float(out.get("confidence", 0.0))
            if conf < 0.4:
                continue
            reasoning = str(out.get("reasoning", ""))
            rec = GeologyRecord(
                age=_extract_first(AGE_PATTERN, reasoning),
                formation=_extract_first(FORMATION_PATTERN, reasoning),
                locality=_extract_first(LOCALITY_PATTERN, reasoning),
                section_type=sec.get("section_type"),
                section_title=sec.get("title"),
                evidence_text=text[:300],
                confidence=conf,
            )
            best_records.append(rec.to_dict())
        links[s] = best_records[:5]
    return links


def link_panels_to_geology(
    captions: dict[str, str],
    fallback_sections: list[dict[str, str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Link each panel to the geology facts MENTIONED IN ITS OWN CAPTION.

    This is the panel-scoped counterpart of
    :func:`link_species_to_geology`. Where the species-level version
    asks "which species lives in which section", this function asks
    "which age/formation/locality does THIS panel's caption mention".

    Parameters
    ----------
    captions : dict[str, str]
        Mapping ``panel_id -> caption_text``. The panel_id is free-form
        (e.g. ``"auto_fig_p004_r01:panel_01"``) and is preserved in the
        returned dict so the caller can wire it back into PanelRecord.
    fallback_sections : list[dict[str, str]] | None
        Optional list of fulltext sections, used as a SOURCE of geology
        records (NOT as the search text). The previous implementation
        searched fulltext for the panel's caption; this version keeps
        the search local to the caption and only pulls the candidate
        records (with their proper evidence_text) from the fulltext.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        ``{panel_id: [geology_record_dict, ...]}``. Panels whose
        caption does not mention any age/formation/locality get an
        empty list -- this is intentional and is the fix for the
        "every panel inherits the whole paper's geology" bug.
    """
    # Build the candidate record pool from the fulltext once. The
    # fulltext is used to populate `evidence_text` and the section
    # metadata, but the AGE/FORMATION/LOCALITY facts must appear in
    # the panel's own caption -- not in some other paragraph of
    # the body text.
    candidates: list[GeologyRecord] = []
    if fallback_sections:
        candidates = extract_geology_from_sections(fallback_sections)
    out: dict[str, list[dict[str, Any]]] = {}
    # Memoise extraction results per unique caption text. In the
    # typical case (all panels in one figure), every panel_id maps
    # to the same caption string, so we only need to run the regex
    # pipeline once rather than N times (once per panel). The
    # cache key is the raw caption string; panel_id is not used
    # inside ``extract_geology_from_sections`` so two calls with
    # the same caption always produce the same records.
    _cap_cache: dict[str, list[GeologyRecord]] = {}
    for panel_id, caption in captions.items():
        if not caption:
            out[panel_id] = []
            continue
        records: list[GeologyRecord] = []
        # 1) Always try to extract FROM the caption first -- the
        # panel's own text is the authoritative source.
        # Use a per-caption cache so the regex pipeline runs once
        # per unique caption string rather than once per panel
        # (panels in the same figure share the same caption text;
        # the panel_id is cosmetic and does not affect extraction).
        if caption not in _cap_cache:
            _cap_cache[caption] = extract_geology_from_sections(
                [{"title": f"panel:{panel_id}", "text": caption, "section_type": "panel_caption"}]
            )
        # Deep-copy so per-panel metadata tweaks (confidence,
        # evidence_text) don't pollute the cached originals.
        for r in _cap_cache[caption]:
            r2 = replace(r)
            r2.confidence = max(r2.confidence, 0.6)
            r2.evidence_text = caption[:300]
            records.append(r2)
        # 2) If the caption is short ("Auto-generated figure for
        # page 1") or matches nothing, fall back to the
        # fulltext-derived candidates BUT only for facts the
        # caption actually mentions. This is the safety net
        # for papers where the caption is non-informative but
        # the body text mentions the same age the plate is
        # illustrating.
        if not records and candidates:
            cap_l = caption.lower()
            for c in candidates:
                # A candidate is "echoed in the caption" only when
                # at least one of its key fields is a substring.
                keys = (c.age, c.formation, c.locality, c.chronostratigraphy)
                if any(k and k.lower() in cap_l for k in keys):
                    records.append(c)
        # 3) If the caption is a known placeholder string (very
        # short, contains "auto-generated" / "placeholder" /
        # "page N") AND we have fulltext sections, attach the
        # most relevant fulltext geology record by section
        # title. We pick the FIRST section (typically the
        # "Geological setting" or "Introduction" section that
        # discusses the paper's stratigraphy and locality) --
        # this gives the operator a sensible per-paper default
        # without dumping every age in the paper onto every
        # panel.
        if not records and candidates and _is_placeholder_caption(caption):
            # Pick candidates whose section_title is the FIRST
            # one in the fulltext. This anchors all panels in
            # the paper to the same introductory stratigraphy
            # context, which is the most common real-world case.
            first_title = fallback_sections[0].get("title") if fallback_sections else None
            for c in candidates:
                if first_title and c.section_title == first_title:
                    records.append(c)
            # If no candidates matched the first section (e.g.
            # the fulltext had no usable age mentions), fall
            # back to the first candidate. This is the only
            # path that uses a "global default" -- the previous
            # implementation applied this to every panel, the
            # new code applies it only to placeholders.
            if not records and candidates:
                records.append(candidates[0])
        out[panel_id] = [r.to_dict() for r in records]
    return out


def build_knowledge_graph(links: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[tuple[str, str]] = set()
    for species, records in links.items():
        species_node = ("species", species)
        if species_node not in seen_nodes:
            seen_nodes.add(species_node)
            nodes.append({"id": f"species:{species}", "type": "species", "name": species})

        for idx, rec in enumerate(records):
            for field in ("age", "formation", "locality"):
                value = rec.get(field)
                if not value:
                    continue
                node_key = (field, value)
                if node_key not in seen_nodes:
                    seen_nodes.add(node_key)
                    nodes.append({"id": f"{field}:{value}", "type": field, "name": value})
                edges.append(
                    {
                        "source": f"species:{species}",
                        "target": f"{field}:{value}",
                        "relation": f"has_{field}",
                        "confidence": rec.get("confidence", 0.0),
                    }
                )
    return {"nodes": nodes, "edges": edges}


def dedup_geology_records(records: list[GeologyRecord]) -> list[GeologyRecord]:
    out: dict[tuple[Any, ...], GeologyRecord] = {}
    for rec in records:
        key = (rec.age, rec.chronostratigraphy, rec.formation, rec.locality, rec.section_title)
        old = out.get(key)
        if old is None or rec.confidence > old.confidence:
            out[key] = rec
    return list(out.values())


def _extract_first(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text or "")
    if not m:
        return None
    if m.lastindex:
        return str(m.group(1)).strip()
    return str(m.group(0)).strip()


# Inline placeholder-caption detector. Mirrors
# rlpe.text_filters.looks_like_placeholder_caption. We delegate to
# text_filters as the single source of truth so the two paths cannot
# drift (the previous local regex required a digit after "page" while
# text_filters did not, so a caption like "page auto-generated"
# matched here but not there). Keep the local function name so
# existing callers don't change.
from .text_filters import looks_like_placeholder_caption as _is_placeholder_caption  # noqa: E402,F401
