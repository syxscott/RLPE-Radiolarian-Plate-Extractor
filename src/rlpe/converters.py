"""Converters between internal dataclasses and the published Pydantic schema.

The internal pipeline uses ``rlpe.types.MatchResult`` (a dataclass) and
``rlpe.types.PaperMetadata`` (a dataclass). At export time we convert
each ``MatchResult`` to a :class:`rlpe.schema_models.PanelRecord` and
each paper metadata to :class:`rlpe.schema_models.PaperMetadataRecord`.

The conversion is intentionally a single, well-tested function pair so
the rest of the codebase keeps its lightweight dataclass types.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .schema_models import (
    FigureRecord,
    GeologyContextRecord,
    GeologyLinkRecord,
    LocalityRecord,
    PanelMetadata,
    PanelRecord,
    PaperRecord,
    PaperMetadataRecord,
    ProvenanceRecord,
    SampleRecord,
    ScaleBarRecord,
    TaxonRecord,
    WarningRecord,
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
        out.append(
            GeologyLinkRecord(
                age=g.get("age"),
                chronostratigraphy=g.get("chronostratigraphy"),
                chronostratigraphy_rank=g.get("chronostratigraphy_rank"),
                ma_top=g.get("ma_top"),
                ma_base=g.get("ma_base"),
                ma_mid=g.get("ma_mid"),
                formation=g.get("formation"),
                member=g.get("member"),
                group=g.get("group"),
                lithology=g.get("lithology"),
                locality=g.get("locality"),
                country=g.get("country"),
                latitude=g.get("latitude"),
                longitude=g.get("longitude"),
                modern_latitude=g.get("modern_latitude") or g.get("latitude"),
                modern_longitude=g.get("modern_longitude") or g.get("longitude"),
                paleo_latitude=g.get("paleo_latitude"),
                paleo_longitude=g.get("paleo_longitude"),
                plate_id=g.get("plate_id"),
                reconstruction_model=g.get("reconstruction_model"),
                reconstruction_age_ma=g.get("reconstruction_age_ma"),
                sample_id=g.get("sample_id"),
                section_type=g.get("section_type"),
                section_title=g.get("section_title"),
                evidence_text=g.get("evidence_text"),
                confidence=g.get("confidence", 0.0) or 0.0,
                biozone=g.get("biozone"),
            )
        )
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
        extraction_method=str(meta.get("extraction_method", "") or ""),
        needs_review=bool(meta.get("needs_review", False)),
        review_reasons=list(meta.get("review_reasons", []) or []),
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



def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(p or "") for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _normalise_species_name(species: str | None) -> str | None:
    if not species:
        return None
    return " ".join(str(species).split()).strip(" .,;") or None


def _taxon_parts(species: str | None) -> dict[str, str | None]:
    """Best-effort, conservative taxon string decomposition.

    This is not a nomenclatural authority parser. It only exposes the
    obvious genus/epithet/qualifier fields needed by the first data-package
    view and leaves uncertain parts in ``verbatim_name``.
    """
    name = _normalise_species_name(species)
    if not name:
        return {"genus": None, "specific_epithet": None, "qualifier": None}
    parts = name.split()
    genus = parts[0] if parts else None
    qualifier = None
    for token in parts[1:]:
        bare = token.rstrip(".,;?").lower()
        if bare in {"cf", "aff", "sp", "spp", "indet", "gr", "group"} or "?" in token:
            qualifier = token
            break
    epithet = None
    for token in parts[1:]:
        bare = token.rstrip(".,;?")
        if bare.lower() in {"cf", "aff", "sp", "spp", "indet", "gr", "group"}:
            continue
        if bare and bare[0].islower():
            epithet = bare
            break
    return {"genus": genus, "specific_epithet": epithet, "qualifier": qualifier}


def _panel_review_reasons(match: MatchResult) -> list[str]:
    reasons: list[str] = []
    meta = match.metadata or {}
    if not match.species:
        reasons.append("missing_species")
    if not match.panel_path:
        reasons.append("missing_panel_image")
    if match.bbox is None:
        reasons.append("missing_bbox")
    if not (meta.get("printed_panel_id") or meta.get("image_panel_id")):
        reasons.append("missing_printed_panel_id")
    if meta.get("extraction_method") == "llm_first" and not match.panel_path:
        reasons.append("llm_first_without_visual_evidence")
    return reasons


def panel_record_from_match(match: MatchResult) -> PanelRecord:
    """Convert an internal ``MatchResult`` to a published ``PanelRecord``."""
    meta = match.metadata or {}
    review_reasons = list(meta.get("review_reasons", []) or [])
    for reason in _panel_review_reasons(match):
        if reason not in review_reasons:
            review_reasons.append(reason)
    caption_panel_id = meta.get("caption_panel_id") or match.panel_id
    printed_panel_id = meta.get("printed_panel_id") or meta.get("image_panel_id")
    canonical_panel_id = meta.get("canonical_panel_id") or match.panel_id
    panel_id_source = meta.get("panel_id_source") or (
        "image_ocr" if printed_panel_id else "legacy"
    )
    taxon_id = _stable_id("taxon", _normalise_species_name(match.species)) if match.species else None
    geology_context_id = None
    geos = meta.get("geology_links") or []
    if geos and isinstance(geos[0], dict):
        g0 = geos[0]
        geology_context_id = _stable_id(
            "geo",
            g0.get("age"),
            g0.get("chronostratigraphy"),
            g0.get("formation"),
            g0.get("locality"),
            g0.get("evidence_text"),
        )
    return PanelRecord(
        paper_id=match.paper_id,
        figure_id=match.figure_id,
        panel_id=match.panel_id,
        caption_panel_id=str(caption_panel_id) if caption_panel_id is not None else None,
        printed_panel_id=str(printed_panel_id) if printed_panel_id is not None else None,
        pipeline_panel_index=meta.get("pipeline_panel_index") or meta.get("panel_index"),
        canonical_panel_id=str(canonical_panel_id) if canonical_panel_id is not None else None,
        panel_id_source=str(panel_id_source),
        species=match.species,
        taxon_id=taxon_id,
        sample_id=meta.get("sample_id"),
        geology_context_id=geology_context_id,
        panel_path=match.panel_path,
        figure_image_path=meta.get("figure_image_path") or meta.get("image_path"),
        bbox=list(match.bbox) if match.bbox is not None else None,
        confidence=float(match.confidence),
        label_text=match.label_text,
        caption_snippet=match.caption_snippet,
        ocr_text=match.ocr_text,
        extraction_method=str(meta.get("extraction_method", "") or ""),
        needs_review=bool(review_reasons) or bool(meta.get("needs_review", False)),
        review_reasons=review_reasons,
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

    The first-stage product data package includes the panels plus the
    deduped papers/figures/taxa/samples/geology_contexts/localities
    views. The paleocoordinates view is empty in the first stage
    because no paleocoord backend is wired yet. ``warnings`` are emitted
    only from rule-based detection of review-worthy records; richer
    warnings will land with the geo/paleo and sample sub-pipelines.
    """
    panels = [panel_record_from_match(m) for m in matches]
    panel_dump = [p.model_dump() for p in panels]
    paper_dump = paper_records_from_matches(matches)
    figure_dump = figure_records_from_matches(matches)
    taxon_dump = taxon_records_from_matches(matches)
    sample_dump = sample_records_from_matches(matches)
    geology_dump = geology_contexts_from_matches(matches)
    locality_dump = locality_records_from_geology(matches)
    warnings_dump = warnings_from_matches(matches)
    return {
        "schema_version": provenance.schema_version,
        "provenance": provenance.model_dump(),
        "papers": paper_dump,
        "figures": figure_dump,
        "panels": panel_dump,
        "taxa": taxon_dump,
        "samples": sample_dump,
        "geology_contexts": geology_dump,
        "localities": locality_dump,
        "paleo_coordinates": [],
        "warnings": warnings_dump,
    }


def paper_records_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for m in matches:
        pid = m.paper_id
        if not pid or pid in seen:
            continue
        pm = paper_metadata_from_internal(m.paper_metadata)
        rec = PaperRecord(
            paper_id=pid,
            title=pm.title if pm else None,
            authors=pm.authors if pm else [],
            year=pm.year if pm else None,
            journal=pm.journal if pm else None,
            volume=pm.volume if pm else None,
            issue=pm.issue if pm else None,
            pages=pm.pages if pm else None,
            doi=pm.doi if pm else None,
            abstract=pm.abstract if pm else None,
            keywords=pm.keywords if pm else [],
            publisher=pm.publisher if pm else None,
            page_count=pm.page_count if pm else None,
            source_pdf=None,
            pdf_sha256=None,
            source=pm.source if pm else "",
            confidence=pm.confidence if pm else 0.0,
        )
        seen[pid] = rec.model_dump()
    return list(seen.values())


def figure_records_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for m in matches:
        meta = m.metadata or {}
        key = (m.paper_id, m.figure_id)
        if not m.figure_id or key in seen:
            continue
        rec = FigureRecord(
            figure_id=m.figure_id,
            paper_id=m.paper_id,
            figure_number=str(meta.get("figure_number")) if meta.get("figure_number") is not None else None,
            figure_type=meta.get("figure_type"),
            page_index=meta.get("page_index"),
            caption=m.caption_snippet,
            caption_source=meta.get("caption_source"),
            image_path=meta.get("image_path") or meta.get("figure_image_path"),
            bbox=list(meta.get("bbox")) if isinstance(meta.get("bbox"), list) else None,
            scale_bar=_scale_bar_from_meta(meta),
            panel_ids=list(meta.get("panel_ids") or []),
            confidence=float(m.confidence),
            needs_review=bool(meta.get("needs_review", False)),
            review_reasons=list(meta.get("review_reasons", []) or []),
        )
        seen[key] = rec.model_dump()
    return list(seen.values())


def taxon_records_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for m in matches:
        sp = _normalise_species_name(m.species)
        if not sp:
            continue
        taxon_id = _stable_id("taxon", sp)
        if taxon_id in seen:
            continue
        parts = _taxon_parts(sp)
        rec = TaxonRecord(
            taxon_id=taxon_id,
            verbatim_name=sp,
            normalized_name=sp,
            genus=parts["genus"],
            specific_epithet=parts["specific_epithet"],
            qualifier=parts["qualifier"],
            authority=None,
            rank="species" if parts["specific_epithet"] else "genus_or_other",
            family=None,
            order=None,
            class_name=None,
            source=(m.metadata or {}).get("extraction_method") or None,
            confidence=float(m.confidence),
            needs_review=bool((m.metadata or {}).get("needs_review", False)),
            review_reasons=list((m.metadata or {}).get("review_reasons", []) or []),
        )
        seen[taxon_id] = rec.model_dump()
    return list(seen.values())


def sample_records_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    sample_pat = re.compile(r"Sample\s+([A-Za-z0-9\-]+)")
    for m in matches:
        text = m.caption_snippet or ""
        if not text:
            continue
        for sm in sample_pat.finditer(text):
            sid = sm.group(1)
            if sid in seen:
                continue
            rec = SampleRecord(
                sample_id=sid,
                paper_id=m.paper_id,
                figure_id=m.figure_id,
                caption_panel_range=None,
                locality_id=None,
                geology_context_id=None,
                evidence_text=text[:300],
                page_index=(m.metadata or {}).get("page_index"),
                confidence=0.5,
            )
            seen[sid] = rec.model_dump()
    return list(seen.values())


def geology_contexts_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for m in matches:
        geos = (m.metadata or {}).get("geology_links") or []
        if not isinstance(geos, list):
            continue
        for g in geos:
            if not isinstance(g, dict):
                continue
            key = _stable_id(
                "geoctx",
                g.get("age"),
                g.get("chronostratigraphy"),
                g.get("formation"),
                g.get("member"),
                g.get("locality"),
                g.get("evidence_text"),
            )
            if key in seen:
                continue
            rec = GeologyContextRecord(
                geology_context_id=key,
                sample_id=g.get("sample_id"),
                age=g.get("age"),
                chronostratigraphy=g.get("chronostratigraphy"),
                chronostratigraphy_rank=g.get("chronostratigraphy_rank"),
                ma_top=g.get("ma_top"),
                ma_base=g.get("ma_base"),
                ma_mid=g.get("ma_mid"),
                formation=g.get("formation"),
                member=g.get("member"),
                group=g.get("group"),
                lithology=g.get("lithology"),
                biozone=g.get("biozone"),
                locality_id=_stable_id(
                    "loc",
                    g.get("locality"),
                    g.get("latitude"),
                    g.get("longitude"),
                ) if g.get("locality") else None,
                evidence_text=g.get("evidence_text"),
                confidence=float(g.get("confidence", 0.0) or 0.0),
            )
            seen[key] = rec.model_dump()
    return list(seen.values())


def locality_records_from_geology(matches: list[MatchResult]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for m in matches:
        geos = (m.metadata or {}).get("geology_links") or []
        if not isinstance(geos, list):
            continue
        for g in geos:
            if not isinstance(g, dict):
                continue
            locality = g.get("locality")
            if not locality:
                continue
            loc_id = _stable_id("loc", locality, g.get("latitude"), g.get("longitude"))
            if loc_id in seen:
                continue
            rec = LocalityRecord(
                locality_id=loc_id,
                name=locality,
                country=g.get("country"),
                region=None,
                section_name=g.get("section_title"),
                modern_latitude=g.get("modern_latitude") or g.get("latitude"),
                modern_longitude=g.get("modern_longitude") or g.get("longitude"),
                coordinate_source="caption" if g.get("latitude") is not None else None,
                geocoding_source=None,
                confidence=float(g.get("confidence", 0.0) or 0.0),
            )
            seen[loc_id] = rec.model_dump()
    return list(seen.values())


def warnings_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        for code in _panel_review_reasons(m):
            warning_id = _stable_id("warn", i, m.paper_id, m.figure_id, m.panel_id, code)
            wr = WarningRecord(
                warning_id=warning_id,
                level="warning",
                code=code,
                message=code.replace("_", " "),
                entity_type="panel",
                entity_id=f"{m.paper_id}/{m.figure_id}/{m.panel_id}",
                evidence_text=(m.caption_snippet or "")[:200],
            )
            out.append(wr.model_dump())
    return out
