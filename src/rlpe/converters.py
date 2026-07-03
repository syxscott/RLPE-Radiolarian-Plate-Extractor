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
    PaperMetadataRecord,
    PaperRecord,
    ProvenanceRecord,
    SampleRecord,
    ScaleBarRecord,
    TaxonRecord,
    WarningRecord,
)
from .types import MatchResult
from .types import PaperMetadata as InternalPaperMetadata


def match_result_from_dict(d: dict[str, Any]) -> MatchResult:
    """Reconstruct a :class:`MatchResult` dataclass from a dict row.

    Used by the pipeline after a ``matches.jsonl`` round-trip and by
    ``cli_export`` to feed rows into ``run_output_from_provenance``.
    Unknown keys are dropped (MatchResult is a plain dataclass, not
    pydantic) and ``None`` falls back for any field that the JSONL
    did not populate.

    ``paper_metadata`` is reconstructed only if the value is a
    mapping; a non-mapping (e.g. ``None`` after a partial run) is
    treated as missing.
    """
    pm = None
    pm_raw = d.get("paper_metadata")
    if isinstance(pm_raw, dict):
        pm = InternalPaperMetadata(
            title=pm_raw.get("title"),
            authors=list(pm_raw.get("authors", []) or []),
            year=pm_raw.get("year"),
            journal=pm_raw.get("journal"),
            volume=pm_raw.get("volume"),
            issue=pm_raw.get("issue"),
            pages=pm_raw.get("pages"),
            doi=pm_raw.get("doi"),
            abstract=pm_raw.get("abstract"),
            keywords=list(pm_raw.get("keywords", []) or []),
            publisher=pm_raw.get("publisher"),
            page_count=pm_raw.get("page_count"),
            source=pm_raw.get("source", ""),
            confidence=pm_raw.get("confidence", 0.0),
        )
    return MatchResult(
        paper_id=d.get("paper_id", ""),
        figure_id=d.get("figure_id", ""),
        panel_id=d.get("panel_id"),
        species=d.get("species"),
        panel_path=d.get("panel_path"),
        bbox=d.get("bbox"),
        confidence=d.get("confidence", 0.0),
        label_text=d.get("label_text"),
        caption_snippet=d.get("caption_snippet"),
        ocr_text=d.get("ocr_text"),
        metadata=d.get("metadata", {}) or {},
        paper_metadata=pm,
    )


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
                modern_latitude=_resolve_modern_coord(g.get("modern_latitude"), g.get("latitude")),
                modern_longitude=_resolve_modern_coord(
                    g.get("modern_longitude"), g.get("longitude")
                ),
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


def _resolve_modern_coord(modern: Any, legacy: Any) -> Any:
    """Pick the modern coordinate value with explicit None handling.

    The earlier ``modern or legacy`` chain silently collapsed legacy
    ``0.0`` or ``None`` to ``None`` while the dedup key still used
    the legacy field — leaving the key and the record disagreeing
    for the same physical locality. This helper picks the first
    present numeric value, preferring the modern field when both
    are non-null.
    """
    if modern is not None:
        return modern
    if legacy is not None:
        return legacy
    return None


def _taxon_parts(species: str | None) -> dict[str, str | None]:
    """Conservative taxon string decomposition for the data-package view.

    The earlier dual-loop heuristic broke on nested-author citations
    such as ``"Genus species cf. S. excelsa"`` (bandini2011 pl08 / pl09)
    because the second loop would treat ``S.`` (a single-letter author
    initial) as the start of a qualifier and emit
    ``qualifier="S. excelsa"`` while overwriting the real epithet.

    This implementation is a single left-to-right scan that recognises
    the well-known micropalaeontology shapes and prefers silence
    (None) over invention when the shape is ambiguous:

      * ``"Genus species"``                       → genus + epithet, no qualifier
      * ``"Genus cf. species"``                    → genus only, qualifier="cf. species"
      * ``"Genus species cf. S. excelsa"``         → genus + epithet, qualifier="cf. S. excelsa"
      * ``"Theocorys? phyzella"``                  → genus="Theocorys", epithet="phyzella", qualifier="?"
      * ``"Genus sp."`` / ``"Genus spp."``          → genus only, qualifier="sp."/"spp."

    Anything that does not match one of these shapes falls back to the
    safest projection: genus = first token (if capitalised), epithet
    = the first lower-cased token after the genus, qualifier = None.
    The full input string is preserved in ``TaxonRecord.verbatim_name``
    so no information is lost on the research side.
    """
    name = _normalise_species_name(species)
    if not name:
        return {"genus": None, "specific_epithet": None, "qualifier": None}

    def _is_author_initial(token: str) -> bool:
        bare = token.rstrip(".")
        return len(bare) == 1 and bare.isalpha() and bare.isupper()

    qualifier_starts = {
        "cf",
        "aff",
        "sp",
        "spp",
        "indet",
        "gr",
        "group",
        "subsp",
        "var",
        "f",
        "nom",
    }

    def _is_qualifier_token(token: str) -> bool:
        bare = token.rstrip(".,;?").lower()
        return bare in qualifier_starts or "?" in token

    tokens = name.split()
    if not tokens:
        return {"genus": None, "specific_epithet": None, "qualifier": None}

    # Genus heuristic: only accept the first token if it starts with an
    # upper-case letter. Strip a trailing ``?`` so "Theocorys?" produces
    # genus "Theocorys" with the ``?`` carried separately as an
    # open-nomenclature marker. Reject strings like "cf." with no real
    # genus to avoid emitting genus="cf." in the data package.
    first = tokens[0]
    if not (first[:1].isalpha() and first[:1].isupper()):
        return {"genus": None, "specific_epithet": None, "qualifier": None}
    genus = first.rstrip("?")
    if not genus:
        return {"genus": None, "specific_epithet": None, "qualifier": None}

    # If the genus itself ended in "?", capture the marker as a
    # separate qualifier. The remaining tokens (if any) are treated as
    # the binomial/trinomial continuation.
    if first.endswith("?"):
        rest = tokens[1:]
        # If the rest starts with a qualifier token, absorb it into
        # the qualifier field; otherwise the ``?`` itself is the
        # qualifier and the rest is the epithet (or empty).
        if rest and _is_qualifier_token(rest[0]):
            return {
                "genus": genus,
                "specific_epithet": None,
                "qualifier": " ".join(rest),
            }
        if not rest:
            return {"genus": genus, "specific_epithet": None, "qualifier": "?"}
        # Locate the next qualifier boundary inside rest. The first
        # lower-cased token (after the optional initial author-prefix)
        # becomes the epithet; everything from the next qualifier /
        # author-initial onwards is the qualifier. We use scoped
        # ``__`` names here so they do not collide with the
        # function-scope ``epithet``/``qualifier`` declared below.
        gen_epithet: str | None = None
        for i, tok in enumerate(rest):
            if _is_qualifier_token(tok) or _is_author_initial(tok):
                break
            bare = tok.rstrip(".,;?")
            if bare and bare[0].islower():
                gen_epithet = bare
                break
        gen_qualifier = "?"
        if rest and (rest[0] != "?"):
            # The "?" is on the genus; no separate qualifier token to
            # emit beyond the marker itself.
            gen_qualifier = "?"
        return {
            "genus": genus,
            "specific_epithet": gen_epithet,
            "qualifier": gen_qualifier,
        }

    # If the second token is itself a qualifier (e.g. "cf."), the
    # binomial is incomplete; emit the qualifier as everything from
    # there onward and leave the epithet empty.
    if len(tokens) >= 2 and _is_qualifier_token(tokens[1]):
        return {
            "genus": genus,
            "specific_epithet": None,
            "qualifier": " ".join(tokens[1:]),
        }

    # Walk the rest. The qualifier is everything from the first
    # qualifying token onward; the epithet is the contiguous lowercase
    # block right after the genus, stopping at the qualifier or at an
    # author-initial boundary.
    epithet: str | None = None
    qualifier: str | None = None
    qualifier_idx: int | None = None
    for i, token in enumerate(tokens[1:], start=1):
        if _is_qualifier_token(token) or _is_author_initial(token):
            qualifier_idx = i
            break
        bare = token.rstrip(".,;?")
        if bare and bare[0].islower():
            epithet = bare
            # Continue scanning to see if a qualifier follows.
            continue
        # Token is capitalised but is not a single-letter initial and
        # not a qualifier. Stop scanning; this is not a canonical
        # binomial/trinomial shape and we should not invent one.
        break

    if qualifier_idx is not None:
        qualifier = " ".join(tokens[qualifier_idx:])

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
    printed_id = meta.get("printed_panel_id") or meta.get("image_panel_id")
    panel_id_source = meta.get("panel_id_source") or ("image_ocr" if printed_id else "legacy")
    # ``llm_first`` and ``caption`` are honest label-provenance tags
    # that signal "this id is NOT from pixel-level evidence" — they
    # intentionally leave printed_panel_id unset. Flagging them as
    # ``missing_printed_panel_id`` would (a) mislabel legitimate work
    # as a defect and (b) make every LLM-first / hybrid row trigger
    # review UI noise. The flag is reserved for true visual-evidence
    # paths (image_ocr / image_panel_label) where the absence of a
    # pixel-read label genuinely indicates a missing OCR step.
    if not printed_id and panel_id_source not in (
        "image_ocr",
        "image_panel_label",
        "llm_first",
        "caption",
    ):
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
    panel_id_source = meta.get("panel_id_source") or ("image_ocr" if printed_panel_id else "legacy")
    taxon_id = (
        _stable_id("taxon", _normalise_species_name(match.species)) if match.species else None
    )
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
        bbox=(list(match.bbox) if match.bbox is not None and len(match.bbox) == 4 else None),
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


def _paleocoord_missing_warning(locality_dump: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Emit a single warning when localities exist but the
    paleocoordinates view is empty.

    This makes the empty paleo_coordinates list self-explanatory:
    consumers can tell the field is reserved, not forgotten. The
    warning is intentionally not emitted when there are no localities,
    because that would be noise for papers that have no geographic
    data at all.
    """
    if not locality_dump:
        return None
    return {
        "warning_id": _stable_id("warn", "paleocoord_backend_missing", str(len(locality_dump))),
        "level": "warning",
        "code": "paleocoord_backend_missing",
        "message": (
            "Paleocoordinate reconstruction not yet wired; the "
            "paleo_coordinates list is empty by design. Records will "
            "appear once the backend is implemented."
        ),
        "entity_type": "run",
        "entity_id": None,
        "evidence_text": f"{len(locality_dump)} locality record(s) detected",
    }


def run_output_from_provenance(
    provenance: ProvenanceRecord,
    matches: list[MatchResult] | None,
) -> dict[str, Any]:
    """Build a JSON-serializable RunOutput dict from a provenance and a
    list of ``MatchResult`` instances. Use this before writing the
    canonical JSONL.

    The first-stage product data package includes the panels plus the
    deduped papers/figures/taxa/samples/geology_contexts/localities
    views. The paleocoordinates view is empty in the first stage
    because no paleocoord backend is wired yet; when localities are
    present we emit a single ``paleocoord_backend_missing`` warning
    so consumers can tell the empty list is reserved, not forgotten.

    Audit L1: ``matches=None`` is now treated as an empty list. The
    previous signature had no None guard, so a caller that passed
    ``None`` (e.g. an early-exit pipeline branch with no matches yet)
    raised ``TypeError: object of type 'NoneType' has no len()``
    inside the ``*_records_from_matches`` helpers. An empty list
    produces an empty RunOutput with the provenance still attached —
    which is the correct degraded behavior.
    """
    if matches is None:
        matches = []
    panels = [panel_record_from_match(m) for m in matches]
    panel_dump = [p.model_dump() for p in panels]
    paper_dump = paper_records_from_matches(matches)
    figure_dump = figure_records_from_matches(matches)
    taxon_dump = taxon_records_from_matches(matches)
    sample_dump = sample_records_from_matches(matches)
    geology_dump = geology_contexts_from_matches(matches)
    locality_dump = locality_records_from_geology(matches)
    warnings_dump = warnings_from_matches(matches)
    paleo_warn = _paleocoord_missing_warning(locality_dump)
    if paleo_warn is not None:
        warnings_dump = warnings_dump + [paleo_warn]
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


def _blank_to_none(v: Any) -> Any:
    """Coerce a blank/whitespace string to None.

    JSON deserialisation often produces ``""`` or ``"   "`` for
    ``None`` fields. The strict ``extra=forbid`` schema accepts
    either, but the data package is more useful downstream if empty
    strings and missing values are the same thing. This helper is
    intentionally narrow: it only treats pure whitespace / empty
    strings as missing; numeric 0 stays 0.
    """
    if v is None:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return v


def figure_records_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for m in matches:
        meta = m.metadata or {}
        key = (m.paper_id, m.figure_id)
        if not m.figure_id or key in seen:
            continue
        fig_no = _blank_to_none(meta.get("figure_number"))
        rec = FigureRecord(
            figure_id=m.figure_id,
            paper_id=m.paper_id,
            figure_number=str(fig_no) if fig_no is not None else None,
            figure_type=_blank_to_none(meta.get("figure_type")),
            page_index=meta.get("page_index"),
            caption=_blank_to_none(m.caption_snippet),
            caption_source=_blank_to_none(meta.get("caption_source")),
            image_path=_blank_to_none(meta.get("image_path") or meta.get("figure_image_path")),
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
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    sample_pat = re.compile(r"Sample\s+([A-Za-z0-9\-]+)")
    for m in matches:
        text = m.caption_snippet or ""
        if not text or not m.paper_id:
            continue
        for sm in sample_pat.finditer(text):
            sid = sm.group(1)
            key = (m.paper_id, sid)
            if key in seen:
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
            seen[key] = rec.model_dump()
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
                )
                if g.get("locality")
                else None,
                evidence_text=g.get("evidence_text"),
                confidence=float(g.get("confidence", 0.0) or 0.0),
            )
            seen[key] = rec.model_dump()
    return list(seen.values())


def locality_records_from_geology(matches: list[MatchResult]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, object, object], dict[str, Any]] = {}
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
            key = (
                m.paper_id,
                locality,
                g.get("latitude"),
                g.get("longitude"),
            )
            if key in seen:
                continue
            loc_id = _stable_id(
                "loc",
                m.paper_id,
                locality,
                g.get("latitude"),
                g.get("longitude"),
            )
            rec = LocalityRecord(
                locality_id=loc_id,
                name=locality,
                country=g.get("country"),
                region=None,
                section_name=g.get("section_title"),
                modern_latitude=_resolve_modern_coord(g.get("modern_latitude"), g.get("latitude")),
                modern_longitude=_resolve_modern_coord(
                    g.get("modern_longitude"), g.get("longitude")
                ),
                coordinate_source="caption" if g.get("latitude") is not None else None,
                geocoding_source=None,
                confidence=float(g.get("confidence", 0.0) or 0.0),
            )
            seen[key] = rec.model_dump()
    return list(seen.values())


def warnings_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in matches:
        for code in _panel_review_reasons(m):
            # The warning_id is content-derived only (no loop index) so
            # it is stable when matches are re-ordered. Two runs over
            # the same logical panel emit the same warning_id.
            warning_id = _stable_id("warn", m.paper_id, m.figure_id, m.panel_id, code)
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
