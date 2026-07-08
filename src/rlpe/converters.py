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
import logging
import re
from typing import Any

from .schema_models import (
    FigureRecord,
    GeologyContextRecord,
    GeologyLinkRecord,
    LocalityRecord,
    PaleoCoordinateRecord,
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

logger = logging.getLogger(__name__)


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
                # Round 22 audit: forward the coord_source marker
                # so the frontend can distinguish regex-extracted
                # coords from country-centroid fallbacks.
                coord_source=g.get("coord_source", "") or "",
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
        geology_scope=str(meta.get("geology_scope", "") or ""),
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


def _geology_context_id(geo: dict[str, Any]) -> str:
    """Build a stable ``geology_context_id`` from a geology-link dict.

    Round 22 audit: ``panel_record_from_match`` previously used
    ``_stable_id("geo", age, chrono, formation, locality, evidence_text)``
    while ``geology_contexts_from_matches`` used
    ``_stable_id("geoctx", age, chrono, formation, member, locality,
    evidence_text)``. The two IDs NEVER matched for the same
    underlying geology fact, so ``PanelRecord.geology_context_id``
    could never reference a real ``GeologyContextRecord.geology_context_id``.
    The audit also showed the panel version was missing the ``member``
    field, so a panel whose first geology link's formation was the
    same but member differed would point to the wrong context.

    This helper produces ONE stable_id scheme used by both call
    sites. We deliberately include ``member`` so the join is unique
    on the full rank triad (Group / Formation / Member), matching
    ``geology_contexts_from_matches`` (Round 18 split).
    """
    return _stable_id(
        "geoctx",
        geo.get("age"),
        geo.get("chronostratigraphy"),
        geo.get("formation"),
        geo.get("member"),
        geo.get("locality"),
        geo.get("evidence_text"),
    )


def _locality_id(geo: dict[str, Any], paper_id: str) -> str:
    """Build a stable ``locality_id`` from a geology-link dict.

    Round 22 audit: ``geology_contexts_from_matches`` and
    ``locality_records_from_geology`` previously each built the
    locality_id inline with the same fields but slightly different
    orderings, which could diverge on edge cases. This helper
    centralises the schema to ``(paper_id, locality, lat, lon)`` so
    the two lists always join.
    """
    return _stable_id(
        "loc",
        paper_id,
        geo.get("locality"),
        geo.get("latitude"),
        geo.get("longitude"),
    )


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
        # Round 22 audit: use the shared helper so this ID matches
        # the keys emitted by ``geology_contexts_from_matches`` (which
        # uses prefix ``"geoctx"`` and includes the ``member`` field).
        # The previous inline ``_stable_id("geo", ...)`` call used a
        # different prefix and dropped ``member``, so the join was
        # always broken.
        geology_context_id = _geology_context_id(geos[0])
    return PanelRecord(
        paper_id=match.paper_id,
        figure_id=match.figure_id,
        panel_id=match.panel_id,
        caption_panel_id=str(caption_panel_id) if caption_panel_id is not None else None,
        printed_panel_id=str(printed_panel_id) if printed_panel_id is not None else None,
        # Round 23 audit: ``pipeline_panel_index`` is declared on the
        # schema but is never populated by the pipeline (the two
        # MatchResult construction sites in pipeline.py don't pass
        # it). Use ``getattr`` with a default so a future pipeline
        # site that DOES set ``match.panel_index`` (e.g. via
        # PanelCandidate) is picked up automatically. Until then,
        # the field stays ``None`` — the schema correctly declares
        # it as optional.
        pipeline_panel_index=getattr(match, "panel_index", None),
        canonical_panel_id=str(canonical_panel_id) if canonical_panel_id is not None else None,
        panel_id_source=str(panel_id_source),
        species=match.species,
        taxon_id=taxon_id,
        sample_id=meta.get("sample_id"),
        geology_context_id=geology_context_id,
        panel_path=match.panel_path,
        figure_image_path=meta.get("figure_image_path") or meta.get("image_path"),
        bbox=_validate_bbox(match.bbox, paper_id=getattr(match, "paper_id", None), figure_id=getattr(match, "figure_id", None), panel_id=match.panel_id),
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


# Round 23 audit: removed the deprecated ``_paleocoord_missing_warning``
# stub. It was Round-20 dead code (always returned ``None``) and
# had no callers in the codebase. Removing it eliminates the
# "is this safe to call?" ambiguity for future readers. The
# Round 20 ``paleo_coordinates_from_localities`` helper emits
# ``paleo_reconstruction_unavailable`` warnings via the
# ``(records, warnings)`` return shape, so the warning channel
# is preserved.


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
    # Initialise warnings_dump BEFORE the helpers that may append
    # their own warnings (paper_metadata_cleanup failure,
    # paleo_reconstruction backend failure). The earlier order —
    # define warnings_dump at the end — caused an F821 undefined-
    # name error in the Round 23 cleanup pass.
    warnings_dump = warnings_from_matches(matches)
    paper_dump, paper_warns = paper_records_from_matches(matches)
    if paper_warns:
        warnings_dump = warnings_dump + paper_warns
    figure_dump = figure_records_from_matches(matches)
    taxon_dump = taxon_records_from_matches(matches)
    sample_dump = sample_records_from_matches(matches)
    geology_dump = geology_contexts_from_matches(matches)
    locality_dump = locality_records_from_geology(matches)
    # Round 20: wire GPlates-style paleocoordinate reconstruction.
    # Previously the ``paleo_coordinates`` view was hard-coded empty
    # with a ``paleocoord_backend_missing`` warning. The backend
    # exists in ``rlpe.paleo_reconstruction`` (Euler pole table for
    # 14 plates, 0-250 Ma range) but was never connected. Now we
    # call ``paleo_coordinates_from_localities`` which pairs each
    # locality with its associated geology context (by locality_id),
    # runs ``reconstruct_paleo_position`` for the context's
    # ``ma_mid``, and emits a ``PaleoCoordinateRecord`` per locality.
    # Localities without coordinates are skipped silently. The
    # warning is no longer emitted since the backend is live.
    paleo_dump, paleo_warns = paleo_coordinates_from_localities(locality_dump, geology_dump)
    if paleo_warns:
        warnings_dump = warnings_dump + paleo_warns
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
        "paleo_coordinates": paleo_dump,
        "warnings": warnings_dump,
    }


def paper_records_from_matches(
    matches: list[MatchResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build paper-level records from matches.

    Round 23 audit: returns ``(records, warnings)``. The warnings
    list contains ``WarningRecord`` dicts for failures the
    post-processing cleanup encountered (e.g. requests library
    missing for Crossref). These flow into ``RunOutput.warnings``
    so the operator sees backend failures in the UI rather than
    only in server logs.
    """
    seen: dict[str, dict[str, Any]] = {}
    warnings_out: list[dict[str, Any]] = []
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
        # Round 20: post-process the record to fix the three systemic
        # paper-metadata issues identified in the 4-paper sampling:
        # garbage titles (page numbers / filenames), author markers
        # like "Input2", and missing / wrong journal names. The
        # helper returns a cleaned dict + any review reasons raised;
        # review_reasons are merged into the paper record's
        # review_reasons list so the operator sees the flag in the
        # UI. Without this pass, papers with parse-failure titles
        # (Bandini / Danelian / Bragin) would silently retain
        # garbage values.
        try:
            from .paper_metadata_cleanup import cleanup_paper_metadata

            cleaned, review_reasons = cleanup_paper_metadata(rec.model_dump())
            # Round 22 audit: ``PaperRecord`` now declares
            # ``review_reasons`` and ``needs_review`` so we can apply
            # them via setattr on the Pydantic model (no more
            # silent extra-key injection into the dumped dict, which
            # previously bypassed ``extra=forbid`` and was not
            # validated downstream).
            allowed_fields = set(PaperRecord.model_fields.keys())
            for k, v in cleaned.items():
                if k in allowed_fields:
                    setattr(rec, k, v)
            if review_reasons:
                rec.review_reasons = list(review_reasons)
                rec.needs_review = True
        except Exception as exc:
            # Cleanup must never block export. If the helper raises
            # (e.g. requests library missing), we fall back to the
            # raw record but emit a WarningRecord so the operator
            # sees the failure in the UI rather than only in
            # server logs.
            logger.warning(
                "paper_metadata_cleanup failed for %s: %s; using raw values",
                pid,
                exc,
            )
            warnings_out.append(
                _warning_record(
                    code="paper_metadata_cleanup_failed",
                    message=(
                        f"paper_metadata_cleanup failed for paper_id="
                        f"{pid!r}; falling back to raw GROBID values. "
                        "title / authors / journal may be missing or "
                        "wrong."
                    ),
                    entity_type="paper",
                    evidence_text=str(exc)[:300],
                )
            )
        seen[pid] = rec.model_dump()
    return list(seen.values()), warnings_out


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
    # Round 20 sampling: Boughdiri 2007 captions use the formats
    # ``CH4, specimen 7``, ``MB4, specimen 15`` — the old regex
    # ``Sample\\s+[A-Za-z0-9\\-]+`` matched none of these, leaving
    # samples = [] for the whole paper. The new pattern set
    # covers four common radiolarian-caption shapes:
    #
    #   1. ``Sample 12`` / ``Sample CH-4``  (legacy)
    #   2. ``CH4, MB4, GA7, RM3, ...``     (Boughdiri short codes)
    #   3. ``specimen 7``                  (Boughdiri long form)
    #   4. ``sample 14-2``                 (some Bandini plates)
    #
    # Each pattern is paired with a single-letter prefix that goes
    # into ``sample_id`` so the operator can tell which detector
    # fired (S_ legacy, B_ Boughdiri-style short code, R_ specimen,
    # N_ numeric-only).
    # Round 23 fix: the tuple type is ``tuple[re.Pattern[str], str]``
    # (compiled regex + sample-id prefix), not ``tuple[str, str]``.
    # The previous annotation was incorrect and tripped mypy.
    _SAMPLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        # (compiled regex, sample-id prefix)
        (re.compile(r"Sample\s+([A-Za-z0-9][A-Za-z0-9\-]*)"), "S_"),
        # Boughdiri 2007-style short codes (CH4, MB4, GA7, RM3, etc.).
        # Round 21: extended prefix list to cover Bandini / Danelian /
        # other Mediterranean / Russian papers. The original 13 prefixes
        # (CH|MB|GA|RM|HK|JP|BS|TS|TR|SP|TK|DF|DP|MS|AS|RS) covered
        # Boughdiri; the additions cover:
        #   Al  — Bandini 2006 ("Al74_300" Greek locality codes)
        #   Mg  — Danelian 2006 ("Mg-100" Vocontian-Basin samples)
        #   Tr  — Tatric / Triassic / common 2-letter locality prefix
        #   Pl  — Polino / Polish / Plutonic papers
        #   BK  — Boughdiri Korbeek-Lo (less common but seen)
        #   OC  — Oceanic Core (IODP-style)
        #   WP  — West Pacific (deep-sea papers)
        #   CM  — Central Morocco / Mediterranean
        # The regex pattern uses three non-capturing tricks:
        #   1. ``(?<![A-Za-z0-9])`` — a NEGATIVE lookbehind so we
        #      don't match inside longer alphanumeric tokens. This
        #      replaces the leading ``\b`` which fails when the
        #      preceding char is itself alphanumeric (rare but seen
        #      in species names like ``NannAl74``).
        #   2. ``[-_]?\d{1,4}(?:[-_]\d{1,4})?`` — optionally a
        #      second ``-`` or ``_`` separated digit block, so
        #      Bandini's ``Al74_300`` (year_sample) matches fully
        #      rather than truncating to ``Al74``.
        #   3. ``(?!\d)`` — a NEGATIVE lookahead after the digit
        #      block so we don't truncate matches in the middle of
        #      longer digit runs.
        # Together these let us match ``Al74_300`` as a single
        # identifier (the full locality code + sample number).
        (
            re.compile(
                r"(?<![A-Za-z0-9])"
                r"(?:CH|MB|GA|RM|HK|JP|BS|TS|TR|SP|TK|DF|DP|MS|AS|RS|"
                r"Al|Mg|Tr|Pl|BK|OC|WP|CM)"
                r"[-_]?\d{1,4}(?:[-_]\d{1,4})?(?!\d)"
            ),
            "B_",
        ),
        # "specimen 7" / "specimen 15" — Boughdiri long form. The
        # sample_id embeds the full word so the operator sees
        # ``R_specimen_7`` (not just ``R_7``) and can distinguish
        # specimen numbers from any other numbered identifier.
        (re.compile(r"(specimen\s+\d{1,4})\b"), "R_"),
        # "sample 14-2" style (some Bandini captions)
        (
            re.compile(r"\bsample\s+(\d{1,4}[-/]?\d{0,4})\b", re.IGNORECASE),
            "N_",
        ),
        # Round 21: parenthesized numbered list "(1) (2) (3) ..." —
        # common in Bragin 2025 captions ("(1) Praeparvicingula
        # blackhorsensis, (2) Praeparvicingula donnae ..."). Tagged
        # with prefix ``L_`` because the operator must verify each
        # match isn't a figure number reference. The regex is narrow
        # (``\d{1,3}`` max 3 digits, in parentheses) so false positives
        # are rare.
        (re.compile(r"\(\d{1,3}\)"), "L_"),
        # Round 21: "pl. N" abbreviated plate reference — Bragin 2025
        # and other Russian / older papers. ``pl.`` is unambiguous in
        # plate captions. Tagged with prefix ``P_``.
        (re.compile(r"pl\.\s*(\d{1,2})\b"), "P_"),
        # Round 21: "Sample (12)" parenthesized form (rare but seen
        # in some Bandini captions). Tagged ``S_`` to fold into the
        # legacy sample bucket.
        (re.compile(r"Sample\s+\(\d+\)"), "S_"),
    )
    for m in matches:
        text = m.caption_snippet or ""
        if not text or not m.paper_id:
            continue
        for pat, prefix in _SAMPLE_PATTERNS:
            for sm in pat.finditer(text):
                # group(1) is the captured id; fall back to group(0)
                # for patterns without a capture group (the Boughdiri
                # short-code pattern uses an alternation that doesn't
                # capture, so the whole match is the id).
                sid_raw = sm.group(1) if sm.lastindex else sm.group(0)
                sid = f"{prefix}{sid_raw}"
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
            key = _geology_context_id(g)
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
                locality_id=_locality_id(g, m.paper_id)
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
            loc_id = _locality_id(g, m.paper_id)
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
                # Round 22 audit: read the actual coord_source from
                # the geology link. The previous code hardcoded
                # ``"caption"`` regardless of whether the coordinate
                # came from regex extraction or the country-centroid
                # fallback (Round 21). The fallback signal is
                # preserved here so the operator can distinguish.
                coordinate_source=(
                    g.get("coord_source")
                    or ("caption" if g.get("latitude") is not None else None)
                ),
                geocoding_source=None,
                confidence=float(g.get("confidence", 0.0) or 0.0),
            )
            seen[key] = rec.model_dump()
    return list(seen.values())


def _validate_bbox(
    bbox: list[int] | tuple[int, ...] | None,
    *,
    paper_id: str | None = None,
    figure_id: str | None = None,
    panel_id: str | None = None,
) -> list[int] | None:
    """Validate that ``bbox`` is a 4-element integer list.

    Round 23 audit: the previous code silently coerced a malformed
    bbox (length != 4) to ``None``. The Pydantic model would
    reject it (``min_length=4, max_length=4``), so the silent
    coercion was masking a data-quality bug. Now we log a warning
    and let ``None`` flow through only when the bbox is truly
    absent (``None``). The empty / wrong-length case emits a
    warning so the operator can find the offending panel.

    Returns the bbox list when valid, ``None`` when the bbox is
    absent. Raises no exception: malformed bbox still produces a
    valid ``PanelRecord`` (with ``bbox=None``) but logs the issue
    so the operator can investigate.
    """
    if bbox is None:
        return None
    if len(bbox) != 4:
        logger.warning(
            "bbox has wrong length %d (expected 4) for paper=%s figure=%s panel=%s; "
            "storing as None. This usually indicates an upstream panel-"
            "detector bug; please file an issue with the offending "
            "paper_id so we can fix the source.",
            len(bbox),
            paper_id,
            figure_id,
            panel_id,
        )
        return None
    return list(bbox)


def _warning_record(
    code: str,
    message: str,
    entity_type: str = "run",
    evidence_text: str | None = None,
) -> dict[str, Any]:
    """Build a WarningRecord dict for emission to ``RunOutput.warnings``.

    Round 23 audit: the conversion helpers previously returned
    ``None`` or empty lists when their backend (Crossref, paleo
    reconstruction, paper_metadata cleanup) failed. Operators had
    no UI surface for these failures. This helper lets the
    converters emit a structured warning so the failures reach the
    ``/results`` response and the frontend's warnings tab.

    The shape matches ``WarningRecord.model_dump()`` so it can be
    merged into ``RunOutput.warnings`` directly.
    """
    return {
        "warning_id": _stable_id("warn", code, message[:80]),
        "level": "warning",
        "code": code,
        "message": message,
        "entity_type": entity_type,
        "entity_id": None,
        "evidence_text": evidence_text,
    }


def paleo_coordinates_from_localities(
    localities: list[dict[str, Any]],
    geology_contexts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build PaleoCoordinateRecords by pairing localities with their
    associated geology context and running GPlates-style reconstruction.

    Round 20 wiring: ``run_output_from_provenance`` calls this
    helper to populate the previously-empty ``paleo_coordinates``
    view. For each locality that has a non-None
    ``modern_latitude`` / ``modern_longitude`` we look up its
    geology context (by ``locality_id``), take the context's
    ``ma_mid`` as the reconstruction age, and call
    ``reconstruct_paleo_position`` from ``rlpe.paleo_reconstruction``
    to compute the rotated paleo coordinate. The plate is inferred
    from the locality's country via ``infer_plate_id``.

    Localities without coordinates are skipped silently — no fake
    records are emitted, matching the no-fabrication policy used
    throughout Round 18-20.

    Round 23 audit: returns ``(records, warnings)`` so the caller
    can surface backend-import failures or per-locality
    reconstruction failures as ``WarningRecord`` entries in the
    run output. Previously the import-failure was logged-only
    (no UI surface); now the warning reaches the operator.

    Returns a 2-tuple ``(records, warnings)``:
      * ``records`` — list of dicts ready to be JSON-serialised
        into ``RunOutput.paleo_coordinates``.
      * ``warnings`` — list of WarningRecord dicts (already in
        ``WarningRecord.model_dump()`` shape) to merge into
        ``RunOutput.warnings``.
    """
    warnings_out: list[dict[str, Any]] = []
    if not localities:
        return [], warnings_out
    # Build a locality_id → geology-context lookup so we can find the
    # ma_mid for each locality in O(1).
    geo_by_loc: dict[str, dict[str, Any]] = {}
    for g in geology_contexts:
        loc_id = g.get("locality_id")
        if loc_id:
            geo_by_loc[loc_id] = g
    out: list[dict[str, Any]] = []
    # Local import keeps converters.py free of paleo_reconstruction at
    # import time (avoids circular imports and keeps the unit-test
    # dependency surface small).
    try:
        from .paleo_reconstruction import (
            infer_plate_id,
            reconstruct_paleo_position,
        )
    except Exception as exc:
        # Round 23 audit: surface this as a WarningRecord so the
        # operator sees the backend is unavailable (was previously
        # only a server-log warning). The paleocoordinates view will
        # be empty for this run, which is the correct degraded
        # behaviour.
        logger.warning(
            "paleo_reconstruction import failed; paleo_coordinates will be empty",
            exc_info=True,
        )
        warnings_out.append(
            _warning_record(
                code="paleo_reconstruction_unavailable",
                message=(
                    "Paleocoordinate reconstruction backend "
                    "(paleo_reconstruction.py) failed to import. "
                    "paleo_coordinates will be empty for this run."
                ),
                entity_type="run",
                evidence_text=str(exc)[:300],
            )
        )
        return [], warnings_out
    for loc in localities:
        mod_lat = loc.get("modern_latitude")
        mod_lon = loc.get("modern_longitude")
        if mod_lat is None or mod_lon is None:
            continue
        # Find the geology context (by locality_id) and read its ma_mid.
        loc_id = loc.get("locality_id")
        geo = geo_by_loc.get(loc_id or "")
        age_ma: float | None = None
        if geo is not None:
            age_ma = geo.get("ma_mid")
            # ma_mid is optional in the schema; fall back to ma_top
            # if ma_mid is missing.
            if age_ma is None:
                age_ma = geo.get("ma_top")
        # Infer the plate from country / locality name. The fallback
        # order in ``infer_plate_id`` is country > locality > coords.
        plate_id = infer_plate_id(
            country=loc.get("country"),
            locality=loc.get("name"),
            modern_lat=float(mod_lat),
            modern_lon=float(mod_lon),
        )
        paleo_lat, paleo_lon = reconstruct_paleo_position(
            modern_lat=float(mod_lat),
            modern_lon=float(mod_lon),
            age_ma=age_ma,
            plate_id=plate_id,
        )
        rec = PaleoCoordinateRecord(
            paleo_coordinate_id=_stable_id(
                "paleo",
                loc_id,
                mod_lat,
                mod_lon,
                age_ma,
                plate_id,
            ),
            locality_id=loc_id,
            modern_latitude=float(mod_lat),
            modern_longitude=float(mod_lon),
            reconstruction_age_ma=float(age_ma) if age_ma is not None else None,
            paleo_latitude=float(paleo_lat) if paleo_lat is not None else None,
            paleo_longitude=float(paleo_lon) if paleo_lon is not None else None,
            plate_id=plate_id,
            reconstruction_model="Seton2012",
            method="euler_pole_rotation",
            confidence=0.7 if paleo_lat is not None else 0.0,
            backend_status="ok" if paleo_lat is not None else "plate_or_age_unknown",
        )
        out.append(rec.model_dump())
    return out, warnings_out


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
