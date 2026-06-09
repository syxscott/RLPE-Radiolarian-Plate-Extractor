"""Converters between internal dataclasses and the published Pydantic schema.

The internal pipeline uses ``rlpe.types.MatchResult`` (a dataclass) and
``rlpe.types.PaperMetadata`` (a dataclass). At export time we convert
each ``MatchResult`` to a :class:`rlpe.schema_models.PanelRecord` and
each paper metadata to :class:`rlpe.schema_models.PaperMetadataRecord`.

The conversion is intentionally a single, well-tested function pair so
the rest of the codebase keeps its lightweight dataclass types.
"""
from __future__ import annotations

from typing import Any

from .schema_models import (
    GeologyLinkRecord,
    PanelMetadata,
    PanelRecord,
    PaperMetadataRecord,
    ProvenanceRecord,
    ScaleBarRecord,
)
from .types import MatchResult
from .types import PaperMetadata as InternalPaperMetadata


def _scale_bar_from_meta(meta: dict[str, Any]) -> ScaleBarRecord | None:
    sb = meta.get("scale_bar")
    if not isinstance(sb, dict):
        return None
    return ScaleBarRecord(
        value=sb.get("value"),
        unit=sb.get("unit"),
        source=sb.get("source"),
        pixel_length=sb.get("pixel_length"),
        um_per_px=sb.get("um_per_px"),
        confidence=sb.get("confidence", 0.0) or 0.0,
    )


def _geology_links_from_meta(meta: dict[str, Any]) -> list[GeologyLinkRecord]:
    out: list[GeologyLinkRecord] = []
    for g in meta.get("geology_links", []) or []:
        if not isinstance(g, dict):
            continue
        out.append(GeologyLinkRecord(
            age=g.get("age"),
            chronostratigraphy=g.get("chronostratigraphy"),
            chronostratigraphy_rank=g.get("chronostratigraphy_rank"),
            formation=g.get("formation"),
            locality=g.get("locality"),
            latitude=g.get("latitude"),
            longitude=g.get("longitude"),
            section_type=g.get("section_type"),
            section_title=g.get("section_title"),
            evidence_text=g.get("evidence_text"),
            confidence=g.get("confidence", 0.0) or 0.0,
        ))
    return out


def panel_metadata_from_match(match: MatchResult) -> PanelMetadata:
    meta = match.metadata or {}
    return PanelMetadata(
        panel_score=meta.get("panel_score"),
        ocr_count=int(meta.get("ocr_count", 0) or 0),
        taxon_count=int(meta.get("taxon_count", 0) or 0),
        figure_number=meta.get("figure_number"),
        page_index=meta.get("page_index"),
        matcher_used=bool(meta.get("matcher_used", False)),
        matcher_type=str(meta.get("matcher_type", "heuristic")),
        matcher_conf=float(meta.get("matcher_conf", 0.0) or 0.0),
        caption_pairs_used=bool(meta.get("caption_pairs_used", False)),
        scale_bar=_scale_bar_from_meta(meta),
        geology_links=_geology_links_from_meta(meta),
        m3_diagnostic=dict(meta.get("m3_diagnostic", {}) or {}),
        extraction_source=str(meta.get("extraction_source", "") or ""),
    )


def paper_metadata_from_internal(pm: InternalPaperMetadata | None) -> PaperMetadataRecord | None:
    if pm is None:
        return None
    # Defensive: ``InternalPaperMetadata.confidence`` is typed as float
    # but in practice can be None when GROBID TEI parsing failed partway
    # through (e.g. PDF rendered but no DOI / no abstract → no confidence
    # populated). Passing None through to the Pydantic model raises a
    # validation error and breaks the whole export; coerce to 0.0 here
    # so a partially-populated paper metadata record still makes it into
    # the JSONL. Matches the defensive pattern used in
    # ``_scale_bar_from_meta`` and ``_geology_links_from_meta`` above.
    confidence_val: float
    if pm.confidence is None:
        confidence_val = 0.0
    else:
        try:
            confidence_val = float(pm.confidence)
        except (TypeError, ValueError):
            confidence_val = 0.0
    return PaperMetadataRecord(
        title=pm.title,
        authors=list(pm.authors or []),
        year=pm.year,
        journal=pm.journal,
        volume=pm.volume,
        issue=pm.issue,
        pages=pm.pages,
        doi=pm.doi,
        abstract=pm.abstract,
        keywords=list(pm.keywords or []),
        publisher=pm.publisher,
        page_count=pm.page_count,
        source=pm.source,
        confidence=confidence_val,
    )


def panel_record_from_match(match: MatchResult) -> PanelRecord:
    """Convert an internal ``MatchResult`` to a published ``PanelRecord``."""
    return PanelRecord(
        paper_id=match.paper_id,
        figure_id=match.figure_id,
        panel_id=match.panel_id,
        species=match.species,
        panel_path=match.panel_path,
        bbox=list(match.bbox) if match.bbox is not None else None,
        confidence=float(match.confidence),
        label_text=match.label_text,
        caption_snippet=match.caption_snippet,
        ocr_text=match.ocr_text,
        metadata=panel_metadata_from_match(match),
        paper_metadata=paper_metadata_from_internal(match.paper_metadata),
    )


def run_output_from_provenance(
    provenance: ProvenanceRecord,
    matches: list[MatchResult],
) -> dict[str, Any]:
    """Build a JSON-serializable RunOutput dict from a provenance and a
    list of ``MatchResult`` instances. Use this before writing the
    canonical JSONL.
    """
    return {
        "schema_version": provenance.schema_version,
        "provenance": provenance.model_dump(),
        "panels": [panel_record_from_match(m).model_dump() for m in matches],
    }
