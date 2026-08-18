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
from collections import Counter
from typing import Any

from .evaluation.metrics import wilson_score_interval
from .schema_models import (
    FigureRecord,
    GeologyContextRecord,
    GeologyLinkRecord,
    LocalityRecord,
    MorphologyRecord,
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
                # Round 24: environment / geochem / facies fields
                # (see GeologyLinkRecord docstring in schema_models.py).
                paleoenvironment=g.get("paleoenvironment"),
                redox=g.get("redox"),
                chemostrat=g.get("chemostrat"),
                facies=g.get("facies"),
                # Phase 63 Plan 6.15 (Bug 6.15): GBIF requires
                # ``coordinateUncertaintyInMeters``. We map
                # ``coord_source`` to a representative radius (see
                # ``_coordinate_uncertainty_for``). ``None`` when the
                # coord_source isn't one of the recognised values.
                coordinate_uncertainty_in_meters=_coordinate_uncertainty_for(
                    g.get("coord_source", "") or ""
                ),
            )
        )
    return out


def panel_metadata_from_match(match: MatchResult) -> PanelMetadata:
    meta = match.metadata or {}
    # Phase 64 Plan B (Task B.5): forward the schematic extraction
    # payload from the match metadata onto the exported PanelMetadata.
    # We store the value verbatim (the JSON shape comes from the M3
    # prompt contract in extract_schematic) so downstream consumers
    # see the same structure they would see if they called the M3
    # engine directly.
    schematic_data = meta.get("figure_schematic_data")
    if not isinstance(schematic_data, dict):
        schematic_data = None
    # Phase 65 Plan A.5: forward cross-figure linker provenance so the
    # export chain (JSONL / xlsx / DwC-A) can surface the link strategy
    # + confidence to operators. Defaults stay None / 0.0 so legacy
    # records (no linker run) remain valid.
    link_source_val = meta.get("link_source")
    if link_source_val is not None and not isinstance(link_source_val, str):
        link_source_val = None
    link_figure_id_val = meta.get("link_figure_id")
    if link_figure_id_val is not None and not isinstance(link_figure_id_val, str):
        link_figure_id_val = None
    try:
        link_confidence_val = float(meta.get("link_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        link_confidence_val = 0.0
    link_confidence_val = max(0.0, min(1.0, link_confidence_val))
    # Phase 66 Plan C.5: forward cross_figure_visual_links from the
    # match metadata. Each entry is the dict shape the linker emits
    # (target_figure_id / target_layer / target_age / target_formation
    # / confidence / source). We filter to dicts only so a corrupted
    # list doesn't break the export.
    raw_visual_links = meta.get("cross_figure_visual_links")
    visual_links: list[dict[str, Any]] = []
    if isinstance(raw_visual_links, list):
        for entry in raw_visual_links:
            if isinstance(entry, dict):
                visual_links.append(entry)
    # audit 2026-07-31: forward PBDB taxonomy onto the exported
    # metadata so the DwC-A exporter can fill kingdom…family (the
    # columns were hard-coded empty because PanelRecord never carried
    # the PBDB payload).
    pbdb_tax = (meta.get("paleodb") or {}).get("taxonomy")
    if not isinstance(pbdb_tax, dict):
        pbdb_tax = None
    return PanelMetadata(
        paleodb_taxonomy=pbdb_tax,
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
        figure_schematic_data=schematic_data,
        link_source=link_source_val,
        link_confidence=link_confidence_val,
        link_figure_id=link_figure_id_val,
        cross_figure_visual_links=visual_links,
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

    Phase 63 Plan 6.14 (Bug 6.14): use modern_latitude / modern_longitude
    when present (Round 25+ convention), falling back to legacy
    latitude / longitude. The previous formula relied solely on the
    legacy ``latitude/longitude`` fields, which Round 25+ leaves as
    ``None`` for derived (centroid, paleo-reconstructed) coords —
    so two physically distinct localities at the same name with
    different modern coords collapsed onto the SAME hash and the
    export silently dropped one.
    """
    lat = _resolve_modern_coord(geo.get("modern_latitude"), geo.get("latitude"))
    lon = _resolve_modern_coord(geo.get("modern_longitude"), geo.get("longitude"))
    return _stable_id(
        "loc",
        paper_id,
        geo.get("locality"),
        lat,
        lon,
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
        return {
            "genus": None,
            "specific_epithet": None,
            "qualifier": None,
            "authority": None,
            "generic_name": None,
        }

    # Audit 2026-08-01 M1/M2: pre-extract authority and subgenus from
    # the raw name string before tokenization, so they do not get
    # swept into the qualifier field.  Authority lives in
    # ``authority`` (parens preserved), subgenus in ``generic_name``.
    # The remaining (cleaned) name flows through the existing token
    # loop below.  Order matters: postfix-subgenus must be checked
    # BEFORE authority, otherwise ``(Podocyrtites)`` would match the
    # authority surname pattern and steal the subgenus field.
    working = name
    authority: str | None = None
    generic_name: str | None = None

    # M1 (postfix subgenus): ``Podocyrtis amphora (Podocyrtites)``
    # → ``generic_name="Podocyrtites"``.  Anchored at end of string
    # so we don't grab the epithet's own parentheses.  Require a
    # binomial (genus + epithet) before the paren — bare "X (Smith)"
    # is an authority citation, not a subgenus.
    postfix_m = re.search(r"\s\(([A-Z][\w\-']+)\)\s*$", working)
    if postfix_m:
        prefix_tokens = working[: postfix_m.start()].strip().split()
        if len(prefix_tokens) >= 2:
            generic_name = postfix_m.group(1).strip()
            working = working[: postfix_m.start()].rstrip()

    # M2 (authority in parens): ``(Haeckel, 1887)`` or ``(Smith)``.
    # Require either a 4-digit year OR a leading capitalised
    # surname, so that ``(Podocyrtites)`` (now stripped above) and
    # ``(?)`` (qualifier marker) do not match.  Parens are kept in
    # the ``authority`` value to mirror ICZN's parenthesised-author
    # convention.
    auth_m = re.search(
        r"\(([^()]*\d{4}[a-z]?|[A-Z][a-z]+(?:[,\s]+\d{4}[a-z]?)?)\)\s*$",
        working,
    )
    if auth_m:
        authority = auth_m.group(0).strip()
        working = (working[: auth_m.start()] + working[auth_m.end() :]).strip()

    # Prefix subgenus (already worked for the authorship helper;
    # mirror here so ``generic_name`` is populated for the prefix
    # shape too): ``Podocyrtis (Podocyrtites) amphora``.
    if generic_name is None:
        prefix_m = re.match(r"^([A-Z][\w\-']+)\s+\(([A-Z][\w\-']+)\)\s+", working)
        if prefix_m:
            generic_name = prefix_m.group(2).strip()
            working = (prefix_m.group(1) + " " + working[prefix_m.end() :]).strip()

    # The cleaned name drives the token loop below.
    name = working

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
        # audit 2026-07-31: "gen" is the ICZN "genus" abbreviation
        # (LLM truncation shape "Spumellarian gen"). Without it the
        # token was eaten as an epithet and the validity guard let
        # the truncated form through.
        "gen",
    }

    def _is_qualifier_token(token: str) -> bool:
        bare = token.rstrip(".,;?").lower()
        return bare in qualifier_starts or "?" in token

    tokens = name.split()
    if not tokens:
        return {
            "genus": None,
            "specific_epithet": None,
            "qualifier": None,
            "authority": None,
            "generic_name": None,
        }

    # Genus heuristic: only accept the first token if it starts with an
    # upper-case letter. Strip a trailing ``?`` so "Theocorys?" produces
    # genus "Theocorys" with the ``?`` carried separately as an
    # open-nomenclature marker. Reject strings like "cf." with no real
    # genus to avoid emitting genus="cf." in the data package.
    first = tokens[0]
    if not (first[:1].isalpha() and first[:1].isupper()):
        return {
            "genus": None,
            "specific_epithet": None,
            "qualifier": None,
            "authority": None,
            "generic_name": None,
        }
    genus = first.rstrip("?")
    if not genus:
        return {
            "genus": None,
            "specific_epithet": None,
            "qualifier": None,
            "authority": None,
            "generic_name": None,
        }

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
                "authority": authority,
                "generic_name": generic_name,
            }
        if not rest:
            return {
                "genus": genus,
                "specific_epithet": None,
                "qualifier": "?",
                "authority": authority,
                "generic_name": generic_name,
            }
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
            "authority": authority,
            "generic_name": generic_name,
        }

    # If the second token is itself a qualifier (e.g. "cf."), the
    # binomial is incomplete; emit the qualifier as everything from
    # there onward and leave the epithet empty.
    # audit 2026-07-31: EXCEPT when the qualifier is a parenthesised
    # marker — "Sethoconus (?) amphora" — where the epithet follows
    # and must not be dropped. (The "(?)" sets the qualifier start and
    # the walk below keeps scanning for the epithet.)
    if len(tokens) >= 2 and _is_qualifier_token(tokens[1]) and not tokens[1].startswith("("):
        return {
            "genus": genus,
            "specific_epithet": None,
            "qualifier": " ".join(tokens[1:]),
            "authority": authority,
            "generic_name": generic_name,
        }

    # Walk the rest. The qualifier is everything from the first
    # qualifying token onward; the epithet is the contiguous lowercase
    # block right after the genus, stopping at the qualifier or at an
    # author-initial boundary.
    epithet: str | None = None
    qualifier: str | None = None
    qualifier_idx: int | None = None
    for i, token in enumerate(tokens[1:], start=1):
        # audit 2026-07-31: parenthesised tokens are subgenus /
        # uncertainty markers — "Podocyrtis (Podocyrtites) amphora",
        # "Sethoconus (?) amphora" (Haeckel's classic subgenus shape).
        # They start the qualifier but must NOT stop the scan: the
        # epithet comes after the closing paren.
        if token.startswith("(") and token.endswith(")"):
            if qualifier_idx is None:
                qualifier_idx = i
            continue
        if _is_qualifier_token(token) or _is_author_initial(token):
            qualifier_idx = i
            break
        bare = token.rstrip(".,;?")
        # P1-2 fix: "n. sp." is ICZN "new species" — token "n" followed
        # by "sp" (stripped) forms a unit qualifier.  Do NOT treat bare="n"
        # as an epithet; instead emit the whole "n. sp." as the qualifier.
        if bare.lower() == "n" and i + 1 < len(tokens):
            next_bare = tokens[i + 1].rstrip(".,;?").lower()
            if next_bare == "sp":
                qualifier_idx = i
                break
        if bare and bare[0].islower():
            epithet = bare
            # Continue scanning to see if a qualifier follows.
            continue
        # Token is capitalised but is not a single-letter initial and
        # not a qualifier. Stop scanning; this is not a canonical
        # binomial/trinomial shape and we should not invent one.
        break

    if qualifier_idx is not None:
        if tokens[qualifier_idx].startswith("("):
            # audit 2026-07-31: parenthesised subgenus/uncertainty
            # markers are qualifiers that END at the closing paren —
            # the epithet after them was already captured and must not
            # be re-included ("(Podocyrtites) amphora" → qualifier
            # "(Podocyrtites)", epithet "amphora").
            end = qualifier_idx + 1
            while end < len(tokens) and tokens[end].startswith("("):
                end += 1
            qualifier = " ".join(tokens[qualifier_idx:end])
        else:
            qualifier = " ".join(tokens[qualifier_idx:])

    return {
        "genus": genus,
        "specific_epithet": epithet,
        "qualifier": qualifier,
        "authority": authority,
        "generic_name": generic_name,
    }


def _extract_authorship(species: str | None) -> tuple[str | None, str | None, str | None]:
    """Best-effort split of an ICZN authorship out of a species string.

    Returns ``(genus, subgenus, authorship)``. The input shape is::

        ``Genus species (Smith, 1900)``
        ``Genus (Subgenus) species Smith, 1900``
        ``Podocyrtis (Podocyrtites) species Haeckel, 1887``

    Phase 63 Plan 6.17/6.18 (Bugs 6.17/6.18) decompose this shape into
    DwC-compatible fields: ``genericName`` (subgenus),
    ``scientificNameAuthorship`` (the author/year string).

    Returns ``(None, None, None)`` when no clear authority is found
    — caller stores ``None`` and downstream reviewers see the
    verbatim_name carrying the full string.
    """
    name = _normalise_species_name(species)
    if not name:
        return None, None, None

    # Look for an authorship block in parentheses ``(Smith, 1900)``
    # or as a trailing Smith, 1900. We scan from the right.
    authorship: str | None = None
    subgenus: str | None = None
    rest = name

    # 1. Parenthesised authority e.g. ``(Smith, 1900)``
    m = re.search(r"\(([^()]+(?:,\s*\d{4}[a-z]?))\)\s*$", name)
    if m:
        authorship = m.group(1).strip()
        rest = name[: m.start()].strip()

    # 2. Trailing ``, 1900`` style (Smith, 1900) without parens — only
    # when there's no parenthesised match above.
    if authorship is None:
        m = re.search(r"([A-Z][\w\-']*(?:,\s*\d{4}[a-z]?))\s*$", name)
        if m:
            authorship = m.group(1).strip()
            rest = name[: m.start()].strip()

    # 3. Subgenus in parentheses e.g. ``Podocyrtis (Podocyrtites) species``
    if rest:
        sm = re.match(r"^([A-Z][\w\-']+)\s+\(([A-Z][\w\-']+)\)\s+", rest)
        if sm:
            subgenus = sm.group(2).strip()

    # 4. Audit 2026-08-01 M1: POSTFIX subgenus e.g.
    # ``Podocyrtis amphora (Podocyrtites)`` — the parenthesised
    # subgenus trails the epithet instead of following the genus
    # (Sanfilippo & Riedel 1985, O'Dogherty 1994, De Wever et al.
    # 2001 routinely use this shape).  Only fires when no prefix
    # subgenus was matched above, and only on the trailing
    # parenthetical (anchored at end-of-string with whitespace
    # before the opening paren).  Require at least 2 tokens before
    # the paren — a single-token prefix like ``X (Smith)`` is an
    # authority citation, not a postfix subgenus.
    if subgenus is None and rest:
        pm = re.search(r"\s\(([A-Z][\w\-']+)\)\s*$", rest)
        if pm:
            prefix_tokens = rest[: pm.start()].strip().split()
            if len(prefix_tokens) >= 2:
                subgenus = pm.group(1).strip()

    return None, subgenus, authorship


def _coordinate_uncertainty_for(coord_source: str | None) -> float | None:
    """Map ``coord_source`` to a GBIF-compatible
    ``coordinateUncertaintyInMeters`` value.

    GBIF guidelines:
      * ``regex`` / ``caption``: textual, ~1000m uncertainty
      * ``paleodb``: derived from PBDB centroid, ~5000m
      * ``country_centroid``: large fallback, ~25000m
      * ``paleo_reconstructed``: paleocoord, ~10000m
      * None: missing
    """
    if not coord_source:
        return None
    coord_source = coord_source.strip()
    if not coord_source:
        return None
    table = {
        "regex": 1000.0,
        "caption": 1000.0,
        "paleodb": 5000.0,
        "country_centroid": 25000.0,
        "paleo_reconstructed": 10000.0,
    }
    return table.get(coord_source)


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


# Reasons that flag a panel as **critical** for the review queue —
# missing any of these means downstream consumers can't trust the row
# without human eyes. Critical reasons map to priority 2; any other
# reason maps to 1; no reason maps to 0. Producers that want to
# override (e.g. an LLM-verified row with no review reasons) can
# still set ``meta["review_priority"]`` directly and the converter
# will respect it.
_CRITICAL_REVIEW_REASONS: frozenset[str] = frozenset(
    {
        "missing_species",
        "missing_bbox",
        "missing_printed_panel_id",
        "missing_panel_image",
    }
)


def _review_priority_from_reasons(reasons: list[str]) -> int:
    """Map review reasons to a 0/1/2 priority bucket.

    Audit 2026-08-05 (Fill Gaps): ``PanelRecord.review_priority``
    was previously always 0 because no producer wrote it. The Web UI
    review queue (``restab.review_priority`` filter) sorts on this
    field, so panels with critical review needs should surface at
    the top of the human queue.

    Bucketing rules:
      2 (high)   — any critical reason (missing species / bbox /
                   printed_panel_id). These rows cannot be used
                   downstream without verification.
      1 (medium) — at least one non-critical review reason.
      0 (low)    — no review reasons; row is publishable as-is.
    """
    if any(r in _CRITICAL_REVIEW_REASONS for r in reasons):
        return 2
    if reasons:
        return 1
    return 0


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
    # audit 2026-08-05 (Fill Gaps): compute Wilson 95% CI from
    # ``confidence`` and an evidence-count hint
    # (``metadata["matcher_evidence_count"]``, default 5) when the
    # producer did not stamp its own CI bounds. Likewise compute
    # review_priority from review_reasons when the producer did not
    # stamp it directly. Both fall back to the meta-pass-through for
    # producers that already populated the fields (e.g. live review
    # UI, replay scripts, audit scripts).
    _p_hat = float(meta.get("confidence", match.confidence))
    _n_ev = int(meta.get("matcher_evidence_count", 5))
    _ci_low, _ci_high = wilson_score_interval(_p_hat, _n_ev)
    _ci_low_meta = meta.get("confidence_interval_low")
    _ci_high_meta = meta.get("confidence_interval_high")
    _priority_meta = meta.get("review_priority")
    _ci_low_final = float(_ci_low_meta) if _ci_low_meta is not None else _ci_low
    _ci_high_final = float(_ci_high_meta) if _ci_high_meta is not None else _ci_high
    _priority_final = (
        int(_priority_meta)
        if _priority_meta is not None
        else _review_priority_from_reasons(review_reasons)
    )
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
        bbox=_validate_bbox(
            match.bbox,
            paper_id=getattr(match, "paper_id", None),
            figure_id=getattr(match, "figure_id", None),
            panel_id=match.panel_id,
        ),
        confidence=float(match.confidence),
        label_text=match.label_text,
        caption_snippet=match.caption_snippet,
        ocr_text=match.ocr_text,
        extraction_method=str(meta.get("extraction_method", "") or ""),
        needs_review=bool(review_reasons) or bool(meta.get("needs_review", False)),
        review_reasons=review_reasons,
        # audit 2026-08-02 (Schema v1.1.0): forward the three new
        # optional fields from match metadata onto the published
        # PanelRecord. All three are producer-side hints: the
        # defaults (None / False / 0) keep legacy matches valid and
        # the Pydantic model enforces the [0,1] / [0,2] ranges.
        # audit 2026-08-05 (Fill Gaps): the v1.1.0 fields are now
        # COMPUTED locally (Wilson CI / priority heuristic) when the
        # upstream pipeline didn't stamp them — see the variables
        # computed just before the ``return PanelRecord(`` call.
        confidence_interval_low=_ci_low_final,
        confidence_interval_high=_ci_high_final,
        image_verified=bool(meta.get("image_verified", False)),
        review_priority=_priority_final,
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


def _coerce_provenance(provenance: ProvenanceRecord | dict[str, Any]) -> ProvenanceRecord:
    """Accept either a :class:`ProvenanceRecord` or a plain dict and
    return a fully-validated :class:`ProvenanceRecord`.

    Used by ``run_output_from_provenance`` (Phase 63 Plan 6.1, Bug 6.1)
    so callers (GUI, web ``Job.rows``, CLI export) can pass either
    form. Mirrors ``_coerce_run_output_from_dict`` in
    ``exporters.archive``: a partial dict (``{job_id, source}`` from
    the GUI) is backfilled with harmless stub values rather than
    rejected. This keeps the legacy ``ProvenanceRecord``-only callers
    fully compatible.
    """
    if isinstance(provenance, ProvenanceRecord):
        return provenance
    if not isinstance(provenance, dict):
        raise TypeError(
            f"provenance must be ProvenanceRecord or dict, got {type(provenance).__name__}"
        )
    prov = dict(provenance)
    allowed_keys = {
        "pipeline_version",
        "schema_version",
        "git_commit",
        "git_dirty",
        "config_snapshot",
        "input_sha256",
        "timestamp_utc",
        "host",
        "python_version",
    }
    prov = {k: v for k, v in prov.items() if k in allowed_keys}
    prov.setdefault("pipeline_version", "unknown")
    prov.setdefault("schema_version", "1.0.0")
    prov.setdefault("git_commit", "unknown")
    prov.setdefault("git_dirty", False)
    prov.setdefault("timestamp_utc", "1970-01-01T00:00:00Z")
    prov.setdefault("host", "unknown")
    prov.setdefault("python_version", "unknown")
    return ProvenanceRecord(**prov)


def run_output_from_provenance(
    provenance: ProvenanceRecord | dict[str, Any],
    matches: list[MatchResult] | None,
    *,
    paper_morphologies: list[dict[str, Any]] | None = None,
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

    Phase 63 Plan 6.1 (Bug 6.1): ``provenance`` may now be a plain
    ``dict`` (mirrors ``write_dwca_zip``'s accept-dict behaviour).
    A partial dict is backfilled with stub fields so the GUI's
    truncated provenance (``{job_id, source}``) does not blow up
    ``ProvenanceRecord.model_validate``.

    Audit 2026-08-02: ``paper_morphologies`` carries the Stage-6
    MorphologyRecord dicts produced by
    ``RadiolarianPipeline._apply_morphology_enrichment`` for the
    paper. Merged with per-row ``metadata["morphology"]`` fallbacks
    so existing tests keep working.
    """
    provenance = _coerce_provenance(provenance)
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
    # Audit 2026-08-02: Stage 6 morphology records. ``paper_morphologies``
    # is a list of dicts the pipeline helper built per-paper; the
    # ``matches`` parameter is also checked so per-row ``metadata[
    # "morphology"]`` fallbacks keep working (tests + legacy paths).
    morphology_dump = morphology_records_from_matches(
        matches, paper_morphologies=paper_morphologies
    )
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
        "morphologies": morphology_dump,
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
    """Build one TaxonRecord per unique species.

    Round 25: when the pipeline has PBDB metadata attached
    (``m.metadata["paleodb"]["taxonomy"]``), the family / order /
    class_name fields are populated from PBDB. Without PBDB, these
    three fields stay ``None`` — the rest of the fields
    (verbatim_name / genus / specific_epithet / qualifier) are
    always populated from the species string itself.

    PBDB lookups are rate-limited (30 req/min on the public
    endpoint); the pipeline caches results to disk and a
    second run is instant. See ``rlpe.paleodb.PaleoDB``.
    """
    seen: dict[str, dict[str, Any]] = {}
    for m in matches:
        sp = _normalise_species_name(m.species)
        if not sp:
            continue
        taxon_id = _stable_id("taxon", sp)
        if taxon_id in seen:
            continue
        parts = _taxon_parts(sp)
        # Round 25: read PBDB taxonomy from match metadata. The
        # pipeline (``_attach_paleodb_metadata``) attaches the
        # full payload (``taxonomy`` / ``occurrences`` /
        # ``occurrence_count``) when ``use_paleodb=True`` is set
        # in the job options. Without PBDB, all three are None.
        meta = m.metadata or {}
        pbdb = meta.get("paleodb") or {}
        pbdb_tax = pbdb.get("taxonomy") or {}
        # Audit 2026-08-02: collect morphology_ids attached to this
        # species by the Stage-6 morphology enrichment. The pipeline
        # helper attaches morphology records at the paper level
        # (``_apply_morphology_enrichment``) but we also accept
        # per-row ``m.metadata["morphology_ids"]`` for backwards
        # compatibility with tests that wire morphology at the
        # MatchResult level.
        morph_ids = list(meta.get("morphology_ids") or [])
        # Phase 63 Plan 6.17/6.18 (Bugs 6.17/6.18): extract the
        # authority/year and subgenus from the verbatim species
        # string. ``_extract_authorship`` recognises both
        # parenthesised ``(Smith, 1900)`` and trailing ``Smith, 1900``
        # shapes; ``subgenus`` is the parenthetical after the genus
        # (``Podocyrtis (Podocyrtites) species Haeckel``).
        _, subgenus, authorship = _extract_authorship(sp)
        # Phase 63 Plan 6.19 (Bug 6.19): taxon_remarks captures the
        # extraction method so DwC reviewers can see how the taxon
        # was determined. The Round-1 default was ``source`` on
        # ``TaxonRecord`` for the same purpose; ``taxon_remarks`` is
        # the explicit DwC term so the export self-documents.
        extraction_method = meta.get("extraction_method") or ""
        taxon_remarks = f"extraction_method={extraction_method}" if extraction_method else None
        # P3-4 fix: when PBDB provides taxonomy, tag source as "paleodb"
        pbdb_provided_taxonomy = bool(
            pbdb_tax.get("family") or pbdb_tax.get("order") or pbdb_tax.get("class")
        )
        rec = TaxonRecord(
            taxon_id=taxon_id,
            verbatim_name=sp,
            normalized_name=sp,
            genus=parts["genus"],
            specific_epithet=parts["specific_epithet"],
            qualifier=parts["qualifier"],
            authority=pbdb_tax.get("authority") or authorship,
            rank=(
                pbdb_tax.get("rank")
                or ("species" if parts["specific_epithet"] else "genus_or_other")
            ),
            # P3-4 fix: when PBDB provides taxonomy, tag source as "paleodb"
            # so consumers can distinguish PBDB-sourced family/order from
            # caption-parsed values.
            family=pbdb_tax.get("family"),
            order=pbdb_tax.get("order"),
            class_name=pbdb_tax.get("class"),
            source="paleodb" if pbdb_provided_taxonomy else (meta.get("extraction_method") or None),
            confidence=float(m.confidence),
            needs_review=bool(meta.get("needs_review", False)),
            review_reasons=list(meta.get("review_reasons", []) or []),
            # Phase 63 Plan 6.16 (Bug 6.16): ICZN is the default;
            # explicit so the export self-documents.
            nomenclatural_code="ICZN",
            # Phase 63 Plan 6.17 (Bug 6.17)
            scientific_name_authorship=authorship,
            # Phase 63 Plan 6.18 (Bug 6.18): subgenus extracted from
            # the parenthetical shape ``Podocyrtis (Podocyrtites)``.
            generic_name=subgenus,
            # Phase 63 Plan 6.19 (Bug 6.19)
            taxon_remarks=taxon_remarks,
            # Audit 2026-08-02: link to MorphologyRecord entries
            # produced by Stage 6. May be empty when Stage 6 is off,
            # the species had no anchorable description, or M3
            # returned an empty dict.
            morphology_ids=morph_ids,
        )
        seen[taxon_id] = rec.model_dump()
    return list(seen.values())


def morphology_records_from_matches(
    matches: list[MatchResult] | None,
    *,
    paper_morphologies: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build MorphologyRecord dicts ready for ``RunOutput.morphologies``.

    Audit 2026-08-02 — Stage 6 morphology enrichment (opt-in).

    Two input paths are supported and merged:

    1. ``paper_morphologies``: list of dicts the pipeline's
       ``_apply_morphology_enrichment`` produced for a single paper
       (each entry is already a dict-shaped MorphologyRecord payload).
       These are validated through ``MorphologyRecord.model_validate``
       so unknown fields are rejected (extra="forbid" on the schema).

    2. ``matches``: legacy / test fallback — any MatchResult whose
       metadata carries ``metadata["morphology"]`` (a dict shaped like
       a MorphologyRecord) is converted to a record. Same shape as
       ``paper_morphologies`` but reads from per-row metadata.

    The function is idempotent: dedup by ``morphology_id`` so a paper
    that produces records via both paths keeps the first occurrence.
    Invalid entries (missing required fields, wrong types) are skipped
    with a logged warning so a single malformed record doesn't break
    the entire export.
    """
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in paper_morphologies or []:
        if not isinstance(entry, dict):
            continue
        rec = _safe_morphology_record(entry)
        if rec is None:
            continue
        if rec["morphology_id"] in seen_ids:
            continue
        seen_ids.add(rec["morphology_id"])
        out.append(rec)
    for m in matches or []:
        meta = m.metadata or {}
        entry = meta.get("morphology")
        if not isinstance(entry, dict):
            continue
        rec = _safe_morphology_record(entry)
        if rec is None:
            continue
        if rec["morphology_id"] in seen_ids:
            continue
        seen_ids.add(rec["morphology_id"])
        out.append(rec)
    return out


def _safe_morphology_record(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a morphology dict through ``MorphologyRecord``.

    Returns the dumped dict on success, ``None`` on any validation
    failure. Logs a warning so the operator can see which entry was
    dropped.
    """
    try:
        rec = MorphologyRecord.model_validate(entry)
    except Exception as exc:
        logger.warning(
            "morphology_records_from_matches: dropping malformed entry (%s): %s",
            exc,
            {k: entry.get(k) for k in ("morphology_id", "taxon_id", "paper_id")},
        )
        return None
    return rec.model_dump()


def sample_records_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    # Audit 2026-08-18: separate index for cross-prefix dedup.
    # ``seen`` indexes by the full prefixed sample_id (X_PR-SB28, S_PR-SB28)
    # so the function returns exactly one record per distinct prefixed id.
    # ``raw_seen`` indexes by the raw sample value (PR-SB28) so when the
    # legacy ``S_`` regex fires after the helper already inserted X_PR-SB28,
    # the legacy hit is dropped. Both indexes are dicts; the function only
    # returns ``seen.values()`` so the raw_seen entries never surface as
    # duplicate records.
    raw_seen: set[tuple[str, str]] = set()
    # audit 2026-08-05 (Fill Gaps): try the canonical
    # ``extract_sample_ids`` helper from
    # ``src/rlpe/sample_id_extractor.py`` first. It already covers
    # the most common shapes ("Sample 12", "Loc. 5",
    # "ID-N", Boughdiri short codes) and returns a list of
    # typed ``SampleID`` dataclasses with kind + value + confidence.
    # The legacy _SAMPLE_PATTERNS tuple below is kept as a fallback
    # because it covers some niche patterns (parenthesised
    # numbered lists, "pl. N" abbreviated plate refs) that the
    # helper does not. Using ``X_`` prefix so the operator can
    # tell the extract_sample_ids source from the legacy regex
    # sources (S_/B_/R_/N_/L_/P_).
    try:
        from .sample_id_extractor import (
            _ID_RE,
            _LOC_RE,
            _SAMPLE_RE,
            extract_sample_ids,
        )
    except Exception:  # pragma: no cover - module is shipped
        extract_sample_ids = None  # type: ignore[assignment]
        _SAMPLE_RE = _LOC_RE = _ID_RE = None  # type: ignore[assignment]
    # Audit 2026-08-18: text-span dedup. The value-based ``raw_seen``
    # check misses cases where the helper normalises the captured text
    # differently from the legacy regex (e.g. ``"Sample ID-203"``: the
    # helper strips the ``ID-`` prefix → value ``"203"``, but the legacy
    # ``Sample\s+`` regex keeps the full ``"ID-203"``; ``"Sample 100A"``:
    # the helper's ``\d{2,}`` branch captures just ``"100"`` but the
    # legacy regex captures ``"100A"``). Two records get inserted for
    # the same physical sample, inflating the count. The robust signal
    # is the text span: if the helper and the legacy regex match
    # overlapping character offsets in the caption, they refer to the
    # same physical sample, regardless of how each side normalises the
    # captured text.
    helper_spans: set[tuple[str, int, int]] = set()
    if extract_sample_ids is not None:
        for m in matches:
            text = m.caption_snippet or ""
            if not text or not m.paper_id:
                continue
            # Track text spans covered by the helper's regexes so the
            # legacy pass can skip overlapping matches below. We run
            # the helper's compiled regexes directly (instead of just
            # looking at the SampleID value list) because the span is
            # the only reliable overlap signal.
            for helper_re in (_SAMPLE_RE, _LOC_RE, _ID_RE):
                if helper_re is None:
                    continue
                for sm in helper_re.finditer(text):
                    helper_spans.add((m.paper_id, sm.start(), sm.end()))
            try:
                sample_ids = extract_sample_ids(text)
            except Exception:
                sample_ids = []
            for sid in sample_ids:
                key = (m.paper_id, f"X_{sid.value}")
                if key in seen:
                    continue
                rec = SampleRecord(
                    sample_id=f"X_{sid.value}",
                    paper_id=m.paper_id,
                    figure_id=m.figure_id,
                    caption_panel_range=None,
                    locality_id=None,
                    geology_context_id=None,
                    evidence_text=text[:300],
                    page_index=(m.metadata or {}).get("page_index"),
                    confidence=sid.confidence,
                )
                seen[key] = rec.model_dump()
                # Audit 2026-08-18: register the raw value so the
                # legacy ``S_`` regex path's cross-prefix dedup
                # check (see ``raw_key in raw_seen`` below) drops
                # hits for ids the helper already covered. Kept for
                # backstop — the primary dedup now uses text spans
                # (see ``helper_spans`` check in the legacy loop).
                raw_seen.add((m.paper_id, sid.value))
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
        # Round 21: "Sample (12)" parenthesized form (rare but seen
        # in some Bandini captions). Tagged ``S_`` to fold into the
        # legacy sample bucket.
        #
        # Audit 2026-08-18: this pattern must come BEFORE the bare
        # ``\(\d{1,3}\)`` pattern below. The previous order fired the
        # bare-parenthesized detector first and emitted ``L_(12)``,
        # then the ``Sample\s+\(\d+\)`` detector dropped via the
        # legacy-to-legacy span dedup. Operator saw ``L_(12)`` with
        # no indication that this was actually a ``Sample (12)``
        # reference, losing semantic information. By ordering the
        # more-specific pattern first, the operator sees the more
        # informative ``S_Sample (12)`` (the bare ``L_`` pattern
        # still fires for genuinely-bare parenthesized numbers like
        # Bragin's ``(1) (2) (3)`` caption panel lists).
        (re.compile(r"Sample\s+\(\d+\)"), "S_"),
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
                sid_str = f"{prefix}{sid_raw}"
                # Audit 2026-08-18: span-overlap dedup against the helper
                # AND against any previously-recorded legacy span for the
                # same paper. Two spans ``[a, b)`` and ``[c, d)`` overlap
                # iff ``a < d and c < b``. The helper and the legacy
                # regex might normalise the same physical sample text
                # differently (e.g. helper strips ``ID-`` from
                # ``Sample ID-203`` → captures ``Sample ID-203`` span
                # but extracts ``"203"``; legacy captures ``Sample ID-203``
                # span and extracts ``"ID-203"``. Same span → dedup.
                # ``Sample 100A``: helper captures ``Sample 100`` span
                # (the ``\d{2,}`` branch stops at the trailing ``A``),
                # legacy captures ``Sample 100A`` span. Different spans
                # but they overlap → dedup. ``Sample (12)``: the
                # ``Sample\s+\(\d+\)`` pattern matches ``Sample (12)``
                # and the bare ``\(\d{1,3}\)` pattern matches ``(12)``
                # — both fire on the same physical sample. The first
                # one to fire inserts a record; the second is dropped
                # because its span overlaps the first's.
                legacy_start, legacy_end = sm.start(), sm.end()
                if any(
                    h_start < legacy_end and legacy_start < h_end
                    for h_paper, h_start, h_end in helper_spans
                    if h_paper == m.paper_id
                ):
                    continue
                # Audit 2026-08-18: dedupe across extractor prefixes.
                # The ``X_`` (extract_sample_ids helper) and ``S_``
                # (legacy regex) both match "Sample PR-SB28" — without
                # cross-prefix dedup the data package carried 2 records
                # per unique id, inflating the sample count. Compare
                # the raw value as a secondary key: if either the
                # helper or a previous regex already recorded this
                # sample value, skip the legacy hit. (Kept as a
                # backstop; the span check above is the primary dedup.)
                key = (m.paper_id, sid_str)
                if key in seen:
                    continue
                raw_key = (m.paper_id, sid_raw)
                if raw_key in raw_seen:
                    continue
                rec = SampleRecord(
                    sample_id=sid_str,
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
                raw_seen.add(raw_key)
                # Register this legacy span so a subsequent legacy
                # pattern that overlaps (e.g. ``\(\d{1,3}\)`` vs
                # ``Sample\s+\(\d+\)`` both matching ``Sample (12)``)
                # is dropped. Mirrors the helper-span tracking above.
                helper_spans.add((m.paper_id, legacy_start, legacy_end))
    return list(seen.values())


def _pbdb_enrich_geology(
    matches: list[MatchResult],
) -> None:
    """In-place: fill missing geology-link fields from PBDB occurrences.

    For each unique species with a ``paleodb.occurrences`` payload
    (the PBDB occurrences list), aggregate the most-common
    ``early_interval`` (used as biozone fallback), ``formation``,
    ``locality``, ``country``, and ``latitude`` / ``longitude``.
    When a panel's first geology link is missing any of these,
    fill from the PBDB aggregation. The cross-dating use case
    (user's suggestion 1) — e.g. "did this species also occur in
    the P/T boundary?  find its biostratigraphic range" — is
    the primary motivation.

    The fill is **only** into empty fields; the original
    extracted text is preserved. The ``coord_source`` is set
    to ``"paleodb"`` for coords filled this way so the
    operator can distinguish them from regex / centroid sources.

    Round 25: this function is no-op if no ``paleodb`` payload
    is attached to any match (i.e. ``use_paleodb=False`` in
    JobOptions).

    Phase 60 Plan 3 (Bug 3.7): the function is also a no-op when
    every match's metadata already carries ``pbdb_enriched=True``.
    The flag is flipped at the end of the pass so a second call
    (e.g. the export converter invoking it again after a re-run)
    does not re-aggregate and rewrite the already-populated
    geology-link fields. Without this guard, each subsequent
    pass appended another ``[PBDB first-occurrence: ...]`` suffix
    to ``evidence_text`` and re-averaged ``ma_top`` / ``ma_base``
    (drifting them away from the original PBDB values).
    """
    # Phase 60 Plan 3 (Bug 3.7): idempotency guard.
    if matches and all((m.metadata or {}).get("pbdb_enriched") for m in matches):
        return
    # Aggregate per species: most-common non-None value per field.
    species_agg: dict[str, dict[str, Any]] = {}
    for m in matches:
        pbdb = (m.metadata or {}).get("paleodb") or {}
        occs = pbdb.get("occurrences") or []
        sp = (m.species or "").strip()
        if not sp or not occs:
            continue
        agg = species_agg.setdefault(
            sp,
            {
                "early_interval": {},
                "formation": {},
                "locality": {},
                "country": {},
                "ma_top": [],
                "ma_base": [],
                "lat": [],
                "lon": [],
            },
        )
        for o in occs:
            for f in ("early_interval", "formation", "locality", "country"):
                v = o.get(f)
                if v:
                    agg[f][v] = agg[f].get(v, 0) + 1
            mt = o.get("max_ma")
            mb = o.get("min_ma")
            if isinstance(mt, (int, float)):
                agg["ma_top"].append(float(mt))
            if isinstance(mb, (int, float)):
                agg["ma_base"].append(float(mb))
            lat = o.get("latitude")
            lon = o.get("longitude")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                agg["lat"].append(float(lat))
                agg["lon"].append(float(lon))

    # Per-species most-common value.
    species_top: dict[str, dict[str, Any]] = {}
    for sp, agg in species_agg.items():
        top: dict[str, Any] = {}
        for f in ("early_interval", "formation", "locality", "country"):
            if agg[f]:
                top[f] = max(agg[f].items(), key=lambda x: x[1])[0]
        # Phase 63 Plan 6.21 (Bug 6.21): use ``statistics.median`` for
        # biostratigraphic range bounds. Mean of a bimodal range has
        # no biostratigraphic meaning (a Carboniferous-Cambrian mean
        # is just a number, not a real range). Median selects the
        # centre of the largest cluster and is robust to outliers.
        if agg["ma_top"]:
            import statistics as _stats

            top["ma_top"] = _stats.median(agg["ma_top"])
        if agg["ma_base"]:
            import statistics as _stats

            top["ma_base"] = _stats.median(agg["ma_base"])
        if agg["lat"]:
            top["lat"] = sum(agg["lat"]) / len(agg["lat"])
            top["lon"] = sum(agg["lon"]) / len(agg["lon"])
        species_top[sp] = top

    # Fill missing fields on each geology link.
    for m in matches:
        sp = (m.species or "").strip()
        if not sp or sp not in species_top:
            continue
        top = species_top[sp]
        for g in (m.metadata or {}).get("geology_links") or []:
            if not isinstance(g, dict):
                continue
            if not g.get("biozone") and top.get("early_interval"):
                # Round 25: cross-dating — PBDB's first-occurrence
                # interval becomes a biozone proxy. Tagged with
                # ``_paleodb_biozone`` so the operator can tell
                # the source.
                g["biozone"] = top["early_interval"]
                g.setdefault("evidence_text", "")
                g["evidence_text"] = (
                    (g.get("evidence_text") or "")[:120]
                    + f" [PBDB first-occurrence: {top['early_interval']}]"
                )[:300]
            if not g.get("formation") and top.get("formation"):
                g["formation"] = top["formation"]
            if not g.get("locality") and top.get("locality"):
                g["locality"] = top["locality"]
            if not g.get("country") and top.get("country"):
                g["country"] = top["country"]
            # Modern coords: only fill if BOTH lat AND lon missing.
            if (
                g.get("latitude") is None
                and g.get("longitude") is None
                and top.get("lat") is not None
                and top.get("lon") is not None
            ):
                g["latitude"] = round(top["lat"], 4)
                g["longitude"] = round(top["lon"], 4)
                g["modern_latitude"] = round(top["lat"], 4)
                g["modern_longitude"] = round(top["lon"], 4)
                g["coord_source"] = "paleodb"
                g["paleo_latitude"] = None
                g["paleo_longitude"] = None
            # Ma bounds from PBDB: only if the regex didn't
            # extract them.
            if g.get("ma_top") is None and top.get("ma_top") is not None:
                g["ma_top"] = round(top["ma_top"], 2)
            if g.get("ma_base") is None and top.get("ma_base") is not None:
                g["ma_base"] = round(top["ma_base"], 2)

    # Phase 60 Plan 3 (Bug 3.7): mark every touched match as
    # already-enriched so a second call to this function short-
    # circuits via the early-return at the top. The flag lives on
    # ``match.metadata`` so the export pipeline / future callers can
    # also short-circuit on the marker without re-invoking.
    for m in matches:
        meta = m.metadata or {}
        if meta.get("paleodb") and not meta.get("pbdb_enriched"):
            meta["pbdb_enriched"] = True
            m.metadata = meta


def geology_contexts_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    # Round 25: in-place enrich missing geology fields from
    # PBDB occurrences. The enrichment is a no-op when no
    # ``paleodb.occurrences`` payload is attached, so the
    # behaviour matches the pre-R25 path for jobs that
    # don't enable ``use_paleodb``.
    _pbdb_enrich_geology(matches)
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
                locality_id=_locality_id(g, m.paper_id) if g.get("locality") else None,
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
            # Phase 63 Plan 6.14 (Bug 6.14): the dedup key now uses
            # modern_latitude / modern_longitude (Round 25+ convention)
            # so two distinct localities with the same name but
            # different coords are kept separate. ``_resolve_modern_coord``
            # picks the modern field when present, falling back to legacy.
            lat = _resolve_modern_coord(g.get("modern_latitude"), g.get("latitude"))
            lon = _resolve_modern_coord(g.get("modern_longitude"), g.get("longitude"))
            key = (
                m.paper_id,
                locality,
                lat,
                lon,
            )
            if key in seen:
                continue
            loc_id = _locality_id(g, m.paper_id)
            rec = LocalityRecord(
                locality_id=loc_id,
                # audit 2026-08-17 (EXP-1): stamp paper_id so xlsx's
                # "论文ID" column isn't blank. Previously the
                # LocalityRecord schema didn't declare paper_id at all
                # and xlsx fell back to "".
                paper_id=m.paper_id or "",
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
                    g.get("coord_source") or ("caption" if g.get("latitude") is not None else None)
                ),
                geocoding_source=None,
                confidence=float(g.get("confidence", 0.0) or 0.0),
                # Phase 63 Plan 6.15 (Bug 6.15)
                coordinate_uncertainty_in_meters=_coordinate_uncertainty_for(
                    g.get("coord_source", "") or ""
                ),
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
            # audit 2026-08-17 (EXP-1): stamp paper_id so xlsx's
            # "论文ID" column isn't blank for paleo rows. The parent
            # locality came from this same paper.
            paper_id=(loc.get("paper_id") or "") if isinstance(loc, dict) else "",
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
            # Phase 63 Plan 6.20 (Bug 6.20): PBDB-style paleocoord
            # uncertainty. Seton2012 Euler-pole reconstructions carry
            # ~50 km of plate motion uncertainty at typical ages.
            coordinate_uncertainty_in_meters=50000.0,
        )
        out.append(rec.model_dump())
    return out, warnings_out


def warnings_from_matches(matches: list[MatchResult]) -> list[dict[str, Any]]:
    """Build WarningRecord dicts for each panel's review reasons.

    Round 24: prepend a single JOB-LEVEL summary warning that
    aggregates the per-panel counts. Without the summary, a paper
    with 78 panels missing a printed_panel_id produces 78 nearly
    identical warning rows in the UI, drowning out the actionable
    signals (e.g. one rare "bbox invalid" warning). The summary
    has ``entity_type="run"`` and ``level="info"`` so it shows in
    the warnings tab without triggering review workflows.

    The per-panel warnings are still emitted (with
    ``entity_type="panel"``) so the operator can drill down to
    individual panels. The summary is purely additive.
    """
    out: list[dict[str, Any]] = []
    code_counts: Counter[str] = Counter()
    for m in matches:
        for code in _panel_review_reasons(m):
            code_counts[code] += 1
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
    if code_counts:
        # Prepend a single run-level summary so the operator sees
        # the count distribution at a glance without scrolling
        # through 100s of per-panel rows.
        summary = {
            "warning_id": _stable_id("warn", "summary", "round24", *sorted(code_counts.keys())),
            "level": "info",
            "code": "panel_review_summary",
            "message": (
                "Per-panel review reasons: "
                + ", ".join(f"{c}={n}" for c, n in sorted(code_counts.items(), key=lambda x: -x[1]))
            ),
            "entity_type": "run",
            "entity_id": None,
            "evidence_text": (
                f"Total panels: {len(matches)}; "
                f"panels with at least one review reason: "
                f"{sum(1 for m in matches if _panel_review_reasons(m))}."
            ),
        }
        out.insert(0, summary)
    return out
