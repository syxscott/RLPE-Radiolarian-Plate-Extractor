from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


AGE_PATTERN = re.compile(
    r"\b(Precambrian|Cambrian|Ordovician|Silurian|Devonian|Carboniferous|Permian|Triassic|Jurassic|Cretaceous|Paleocene|Eocene|Oligocene|Miocene|Pliocene|Pleistocene|Holocene|Early\s+[A-Z][a-z]+|Middle\s+[A-Z][a-z]+|Late\s+[A-Z][a-z]+)\b",
    re.IGNORECASE,
)
FORMATION_PATTERN = re.compile(r"\b([A-Z][A-Za-z\-\s]+(?:Formation|Member|Group|Fm\.|Mb\.|Gp\.))\b")
LOCALITY_PATTERN = re.compile(r"\b(?:from|at|in)\s+([A-Z][A-Za-z\-\s]{2,80})\b")


@dataclass(slots=True)
class GeologyRecord:
    age: str | None = None
    formation: str | None = None
    locality: str | None = None
    section_type: str | None = None
    section_title: str | None = None
    evidence_text: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_geology_from_sections(sections: list[dict[str, str]]) -> list[GeologyRecord]:
    out: list[GeologyRecord] = []
    for sec in sections:
        text = sec.get("text", "")
        if not text:
            continue
        ages = [m.group(1) for m in AGE_PATTERN.finditer(text)]
        forms = [m.group(1).strip() for m in FORMATION_PATTERN.finditer(text)]
        locs = [m.group(1).strip(" .,;") for m in LOCALITY_PATTERN.finditer(text)]

        if not ages and not forms and not locs:
            continue

        # 以句子级片段做证据，先走规则抽取。
        for age in (ages or [None]):
            rec = GeologyRecord(
                age=age,
                formation=forms[0] if forms else None,
                locality=locs[0] if locs else None,
                section_type=sec.get("section_type"),
                section_title=sec.get("title"),
                evidence_text=text[:300],
                confidence=0.55,
            )
            out.append(rec)
    return dedup_geology_records(out)


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
    out: dict[tuple[str | None, str | None, str | None, str | None], GeologyRecord] = {}
    for rec in records:
        key = (rec.age, rec.formation, rec.locality, rec.section_title)
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
