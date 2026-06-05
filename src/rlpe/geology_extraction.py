from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any


AGE_PATTERN = re.compile(
    r"\b(?:Early|Middle|Late|Lower|Upper)\s+[A-Z][a-z]+|"
    r"\b(?:Precambrian|Cambrian|Ordovician|Silurian|Devonian|Carboniferous|Permian|"
    r"Triassic|Jurassic|Cretaceous|Paleocene|Eocene|Oligocene|Miocene|Pliocene|"
    r"Pleistocene|Holocene)\b",
    re.IGNORECASE,
)
FORMATION_PATTERN = re.compile(r"\b([A-Z][A-Za-z\-\s]+(?:Formation|Member|Group|Fm\.|Mb\.|Gp\.))\b")
LOCALITY_PATTERN = re.compile(r"\b(?:from|at|in)\s+([A-Z][A-Za-z\-\s]{2,80})\b")
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
        from .stratigraphy import find_ages_in_text, classify_age_string
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
        # Coordinate parsing
        lat, lon = _extract_first_coord(text)
        if lat is None and lon is None and COORDINATE_PATTERN.search(text):
            m = COORDINATE_PATTERN.search(text)
            if m:
                try:
                    lat = float(m.group(1))
                    if m.group(2) and m.group(2).upper() == "S":
                        lat = -lat
                    lon = float(m.group(3))
                    if m.group(4) and m.group(4).upper() == "W":
                        lon = -lon
                except Exception:
                    pass

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
            if rec not in out:
                out.append(rec)
    return dedup_geology_records(out)


def _extract_first_coord(text: str) -> tuple[float | None, float | None]:
    """Best-effort coordinate extraction. Returns ``(lat, lon)`` or ``(None, None)``."""
    if not text:
        return None, None
    m = COORDINATE_PATTERN.search(text)
    if not m:
        return None, None
    try:
        lat = float(m.group(1))
        if m.group(2) and m.group(2).upper() == "S":
            lat = -lat
        lon = float(m.group(3))
        if m.group(4) and m.group(4).upper() == "W":
            lon = -lon
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None, None
        return lat, lon
    except Exception:
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
        # 简单启发式：在同一章节中，若物种名出现，则链接该章节地质属性。
        for s in species_names:
            s_lower = s.lower()
            for sec in sections:
                if s_lower in (sec.get("text", "").lower()):
                    for rec in geology:
                        if rec.section_title == sec.get("title"):
                            links[s].append(rec.to_dict())
            # 若未命中章节，退化为全局最可能地质记录。
            if not links[s] and geology:
                links[s].append(geology[0].to_dict())
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
