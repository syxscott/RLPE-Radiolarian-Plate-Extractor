"""Darwin Core Archive (DwC-A) export.

A DwC-A is a zip containing:
    - meta.xml           declares the structure of occurrence.txt
    - eml.xml            Ecological Metadata Language: paper metadata
    - occurrence.txt     tab-separated rows, one per occurrence

The fields map directly from :class:`PanelRecord`. The archive is
loadable by GBIF, PBDB, and most biodiversity informatics tools.

We deliberately do **not** include the panel image crops in the
archive; DwC-A is for occurrence records, not media. (Media can be
referenced via ``associatedMedia`` if desired.)
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from html import escape as _xml_escape
from pathlib import Path
from typing import Any

from ..converters import _extract_authorship, _taxon_parts
from ..schema_models import PanelRecord, RunOutput


@dataclass(slots=True)
class DwCAOptions:
    """Options for the Darwin Core Archive export."""

    include_unmatched: bool = False
    encoding: str = "utf-8"
    # audit 2026-08-17 (EXP-2): the previous defaults were the literal
    # two-char strings ``"\\t"`` / ``"\\n"``. meta.xml then wrote those
    # two-char strings as-is (backslash + t), and occurrence.txt was
    # emitted with a real ``\t``. Strict GBIF validators compared the
    # declared delimiter to the file's actual byte and rejected the
    # archive. Defaults are now the real one-char tab / newline.
    fields_terminated_by: str = "\t"
    lines_terminated_by: str = "\n"
    fields_enclosed_by: str = ""
    # DwC term URI prefix; GBIF uses http://rs.tdwg.org/dwc/terms/
    term_ns: str = "http://rs.tdwg.org/dwc/terms/"


# DwC term URIs we emit. The order is also the column order in
# occurrence.txt. Keep this list aligned with GBIF's occurrence core.
DWC_FIELDS: list[tuple[str, str]] = [
    ("occurrenceID", "http://rs.tdwg.org/dwc/terms/occurrenceID"),
    ("basisOfRecord", "http://rs.tdwg.org/dwc/terms/basisOfRecord"),
    ("scientificName", "http://rs.tdwg.org/dwc/terms/scientificName"),
    ("kingdom", "http://rs.tdwg.org/dwc/terms/kingdom"),
    ("phylum", "http://rs.tdwg.org/dwc/terms/phylum"),
    ("class", "http://rs.tdwg.org/dwc/terms/class"),
    ("order", "http://rs.tdwg.org/dwc/terms/order"),
    ("family", "http://rs.tdwg.org/dwc/terms/family"),
    ("genus", "http://rs.tdwg.org/dwc/terms/genus"),
    ("specificEpithet", "http://rs.tdwg.org/dwc/terms/specificEpithet"),
    # P3-1 fix: scientificNameAuthorship was extracted (Phase 63) but never
    # exported — add it to DwC-A so GBIF consumers can see the authority/year.
    ("scientificNameAuthorship", "http://rs.tdwg.org/dwc/terms/scientificNameAuthorship"),
    ("eventDate", "http://rs.tdwg.org/dwc/terms/eventDate"),
    ("year", "http://rs.tdwg.org/dwc/terms/year"),
    ("locality", "http://rs.tdwg.org/dwc/terms/locality"),
    ("country", "http://rs.tdwg.org/dwc/terms/country"),
    ("decimalLatitude", "http://rs.tdwg.org/dwc/terms/decimalLatitude"),
    ("decimalLongitude", "http://rs.tdwg.org/dwc/terms/decimalLongitude"),
    ("geologicalContextID", "http://rs.tdwg.org/dwc/terms/geologicalContextID"),
    ("formation", "http://rs.tdwg.org/dwc/terms/formation"),
    ("identifiedBy", "http://rs.tdwg.org/dwc/terms/identifiedBy"),
    ("associatedReferences", "http://rs.tdwg.org/dwc/terms/associatedReferences"),
    ("associatedMedia", "http://rs.tdwg.org/dwc/terms/associatedMedia"),
    # Phase 64 Plan B (Task B.5): dynamicProperties carries the
    # schematic / diagram / reconstruction / phylogenetic
    # extraction JSON as a single-string blob. DwC-A's
    # ``dynamicProperties`` is the canonical extension point for
    # non-DwC terms; consumers like GBIF accept it unchanged and
    # preserve the original JSON shape. Empty when the row has
    # no figure_schematic_data so the column stays clean.
    ("dynamicProperties", "http://rs.tdwg.org/dwc/terms/dynamicProperties"),
]


def _occurrence_row(panel: PanelRecord) -> dict[str, str]:
    """Build a single occurrence row, all values as strings."""
    pm = panel.paper_metadata
    geo = panel.metadata.geology_links[0] if panel.metadata.geology_links else None
    # Audit 2026-09-01 BL-31: the previous occurrenceID construction
    # joined ``paper_id`` / ``figure_id`` / ``panel_id`` with ":", but
    # dropped any empty value silently. When ``paper_id=""`` AND
    # ``figure_id=""`` (the legacy / "figure-only" paper metadata path),
    # the resulting ID collapsed to just ``panel_id`` — two panels on
    # different papers with the same panel_id then collided and the
    # DwC-A archive rejected the second one. Replace each empty
    # component with the literal ``"(unknown)"`` so every
    # occurrenceID is globally unique.
    parts = [
        panel.paper_id or "(unknown)",
        panel.figure_id or "(unknown)",
        panel.panel_id or "_",
    ]
    occ_id = ":".join(parts)
    media = panel.panel_path or ""
    # Phase 58 Plan 1.2 (Bug 1.2): prefer modern_latitude/longitude when
    # present, fall back to legacy latitude/longitude. Round 25+
    # converters populate modern_* (used by GBIF/PBDB), while legacy
    # fields exist for backwards compat with older extraction runs.
    lat = (
        geo.modern_latitude
        if geo and geo.modern_latitude is not None
        else (geo.latitude if geo and geo.latitude is not None else None)
    )
    lon = (
        geo.modern_longitude
        if geo and geo.modern_longitude is not None
        else (geo.longitude if geo and geo.longitude is not None else None)
    )
    # Phase 63 Plan 6.7 (Bug 6.7): use ``_taxon_parts`` (centralised in
    # ``rlpe.converters``) instead of naive ``species.split()``. The
    # naive split mis-classified ``Genus cf. species`` -> genus="Genus",
    # specificEpithet="cf." and similarly broke on ``aff.``, ``sp.``,
    # ``spp.``, trinomial names (``Genus species subspecies``),
    # and author citations (``Genus species (Smith, 1900)``).
    # ``_taxon_parts`` returns ``None``/empty when the binomial is
    # incomplete (cf./aff./sp./spp.) — GBIF/PBDB reject those as
    # authoritative entries; open-nomenclature rows carry only the
    # ``scientificName`` string.
    taxon = _taxon_parts(panel.species)
    # audit 2026-07-31: the old authorship path read
    # ``panel.taxa[0].scientific_name_authorship`` — PanelRecord has
    # NO ``taxa`` field, so the column was ALWAYS empty (dead code).
    # Use the centralised ``_extract_authorship`` (Phase 63) which
    # parses "Genus species (Smith, 1900)" / "… Smith, 1900" shapes.
    _, _subgenus, _authorship = _extract_authorship(panel.species)
    # audit 2026-07-31: the higher-rank DwC columns (kingdom …
    # family) were hard-coded empty even though the pipeline attaches
    # PBDB taxonomy to the match metadata. It is now forwarded onto
    # PanelMetadata.paleodb_taxonomy; join it here so GBIF uploads
    # carry the full classification.
    _pbdb_tax = panel.metadata.paleodb_taxonomy or {}
    # P3-2 fix: getattr guard for geology_context_id.
    _geo_ctx_id = getattr(panel, "geology_context_id", None)
    return {
        "occurrenceID": occ_id,
        "basisOfRecord": "FossilSpecimen" if panel.species else "",
        "scientificName": panel.species or "",
        "kingdom": str(_pbdb_tax.get("kingdom") or ""),
        "phylum": str(_pbdb_tax.get("phylum") or ""),
        "class": str(_pbdb_tax.get("class") or ""),
        "order": str(_pbdb_tax.get("order") or ""),
        "family": str(_pbdb_tax.get("family") or ""),
        "genus": (taxon.get("genus") or ""),
        "specificEpithet": (taxon.get("specific_epithet") or ""),
        "scientificNameAuthorship": (_authorship or ""),
        "eventDate": str(pm.year) if pm and pm.year else "",
        "year": str(pm.year) if pm and pm.year else "",
        "locality": (geo.locality if geo and geo.locality else ""),
        # Phase 38: read country from geo (set by paleodb's
        # _iso_to_country resolution) instead of hard-coded "".
        # Previously every DwCA export had an empty country column
        # even when the PBDB occurrence record had country data.
        "country": (geo.country if geo and geo.country else ""),
        "decimalLatitude": (str(lat) if lat is not None else ""),
        "decimalLongitude": (str(lon) if lon is not None else ""),
        "geologicalContextID": (
            _geo_ctx_id if _geo_ctx_id else (geo.age if geo and geo.age else "")
        ),
        "formation": (geo.formation if geo and geo.formation else ""),
        "identifiedBy": ("; ".join(pm.authors) if pm and pm.authors else ""),
        "associatedReferences": (pm.doi if pm and pm.doi else ""),
        "associatedMedia": media,
        # Phase 64 Plan B (Task B.5): schematic / diagram /
        # reconstruction / phylogenetic extractions ride on
        # DwC's ``dynamicProperties`` extension term as a
        # JSON-encoded blob. We serialise the same prompt-
        # contract shape M3 produced so GBIF / DwC consumers
        # can re-parse the JSON without needing our schema.
        # Phase 65 Plan A.5: cross-figure linker metadata is
        # merged into the same dynamicProperties blob so the
        # DwC-A file carries one self-describing JSON payload
        # rather than two parallel columns. The linker payload
        # is added as a separate top-level key
        # ``"cross_figure_link"`` so existing schematic
        # consumers (which only know the figure_type /
        # text_elements keys) keep working unchanged.
        "dynamicProperties": _merged_dynamic_properties(panel.metadata),
    }


def _schematic_dynamic_properties(schematic_data: Any) -> str:
    """Phase 64 Plan B (Task B.5): serialise the schematic
    extraction as a single JSON blob for DwC's
    ``dynamicProperties`` column.

    Returns an empty string when no schematic data is present
    (regular plate row). The JSON shape mirrors the M3 prompt
    contract (figure_type / text_elements / relationships /
    extracted_facts / confidence) so a downstream consumer can
    re-parse it identically.
    """
    if not isinstance(schematic_data, dict):
        return ""
    if not schematic_data.get("figure_type"):
        return ""
    try:
        import json as _json

        # sort_keys=True keeps the output deterministic so two
        # equal dicts produce byte-identical blobs (helps test
        # snapshots and idempotent exports).
        return _json.dumps(schematic_data, ensure_ascii=False, sort_keys=True)
    except Exception:
        return ""


def _linker_dynamic_properties(metadata: Any) -> str:
    """Phase 65 Plan A.5: serialise cross-figure linker provenance
    as a JSON blob for DwC's ``dynamicProperties`` column.

    Returns an empty string when the linker didn't run for this
    row (legacy rows, or the linker flag was off). The JSON shape
    is intentionally flat:
        ``{
            "source": "sample_match" | "locality_match" |
                      "m3_inference" | "unlinked",
            "confidence": float,
            "figure_id": str | null,
        }``
    so a downstream consumer can read it without consulting the
    RLPE schema. When paired with the schematic blob via
    ``_merged_dynamic_properties``, the linker block sits under
    ``cross_figure_link`` to avoid colliding with the schematic
    keys (figure_type / text_elements / ...).
    """
    if metadata is None:
        return ""
    src = getattr(metadata, "link_source", None)
    conf = getattr(metadata, "link_confidence", 0.0) or 0.0
    fig_id = getattr(metadata, "link_figure_id", None)
    if not src:
        return ""
    try:
        import json as _json

        return _json.dumps(
            {
                "source": str(src),
                "confidence": float(conf),
                "figure_id": fig_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    except Exception:
        return ""


def _merged_dynamic_properties(metadata: Any) -> str:
    """Phase 65 Plan A.5 + Phase 66 Plan C.5: combine schematic +
    Phase A linker + Phase C visual-linker payloads into a single
    DwC ``dynamicProperties`` blob.

    Behaviour:
    * If only the schematic block is present -> serialise it alone
      (back-compat for Phase 64 Plan B).
    * If only the linker block is present -> serialise a small
      wrapper ``{"cross_figure_link": {...}}``.
    * If only the visual-links block is present -> serialise a small
      wrapper ``{"cross_figure_visual_links": [...]}`` (Phase C).
    * If any combination is present -> merge them under one JSON
      object so the row carries a single self-describing payload
      (preferred for downstream consumers).
    * If none is present -> empty string.

    Empty string on any serialisation error so the export never
    crashes mid-row.
    """
    sch = getattr(metadata, "figure_schematic_data", None)
    link_src = getattr(metadata, "link_source", None)
    visual_links = getattr(metadata, "cross_figure_visual_links", None)

    if not sch and not link_src and not visual_links:
        return ""
    try:
        import json as _json
    except ImportError:  # pragma: no cover
        return ""

    payload: dict[str, Any] = {}
    # Schematic block: only include when figure_type is present
    # (matches _schematic_dynamic_properties' contract).
    if isinstance(sch, dict) and sch.get("figure_type"):
        payload.update(sch)
    # Linker block: only include when the linker actually ran.
    if link_src:
        try:
            conf = float(getattr(metadata, "link_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        payload["cross_figure_link"] = {
            "source": str(link_src),
            "confidence": conf,
            "figure_id": getattr(metadata, "link_figure_id", None),
        }
    # Phase 66 Plan C.5: visual-linker block — only include when the
    # list is non-empty AND each entry is a dict. Same shape as the
    # panel_metadata field, so downstream consumers can re-parse it.
    if isinstance(visual_links, list) and visual_links:
        clean = [e for e in visual_links if isinstance(e, dict)]
        if clean:
            payload["cross_figure_visual_links"] = clean
    if not payload:
        return ""
    try:
        return _json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        return ""


def _build_meta_xml(opts: DwCAOptions) -> str:
    """Build the meta.xml describing the occurrence core.

    audit 2026-08-17 (EXP-2): previously this wrote the literal
    two-char strings ``\\t`` / ``\\n`` (backslash + t/n) into
    ``fieldsTerminatedBy`` / ``linesTerminatedBy``, while
    ``occurrence.txt`` was actually emitted with a real ``\t``. GBIF
    validators that compare the declared delimiter against the file's
    real bytes would reject the archive. We now read the values off
    ``opts`` (whose defaults are real ``\t`` / ``\n`` since EXP-2) and
    serialise them via ``xml.sax.saxutils.quoteattr`` so any tab /
    newline character is correctly escaped into XML attribute syntax
    (e.g. ``&#9;`` / ``&#10;``) — this is what a strict validator
    expects.
    """
    from xml.sax.saxutils import quoteattr

    fields_xml_lines: list[str] = []
    for idx, (_field, uri) in enumerate(DWC_FIELDS, start=1):
        fields_xml_lines.append(f'    <field index="{idx}" term="{uri}"/>')
    fields_xml = "\n".join(fields_xml_lines)
    ftb = quoteattr(opts.fields_terminated_by)
    ltb = quoteattr(opts.lines_terminated_by)
    feb = quoteattr(opts.fields_enclosed_by)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<archive xmlns="http://rs.tdwg.org/dwc/text/" metadata="eml.xml">\n'
        f'  <core encoding="UTF-8" fieldsTerminatedBy={ftb} '
        f"linesTerminatedBy={ltb} fieldsEnclosedBy={feb} "
        'ignoreHeaderLines="1"'
        ' rowType="http://rs.tdwg.org/dwc/terms/Occurrence">\n'
        "    <files><location>occurrence.txt</location></files>\n"
        f"{fields_xml}\n"
        "  </core>\n"
        "</archive>\n"
    )


def _build_eml_xml(run: RunOutput) -> str:
    """Build a minimal EML describing the dataset as a whole.

    All paper-controlled fields (title, authors) are XML-escaped via
    :func:`html.escape`. Without this, a paper title containing ``&``,
    ``<``, ``>``, or ``]]>`` would produce malformed XML; a paper with a
    maliciously-crafted title could inject arbitrary XML (XSS in
    downstream consumers like GBIF that render the EML in a browser).
    """
    prov = run.provenance
    papers = {p.paper_id: p.paper_metadata for p in run.panels if p.paper_metadata}
    title_parts = []
    for pm in papers.values():
        if pm and pm.title:
            title_parts.append(_xml_escape(pm.title))
    dataset_title = (
        f"RLPE {_xml_escape(prov.pipeline_version or '')} extraction of "
        f"{len(papers)} paper(s), {len(run.panels)} panel(s)"
    )
    abstract = (
        "Automatically extracted radiolarian specimen records from "
        "published literature. Provenance: pipeline "
        f"{_xml_escape(prov.pipeline_version or '')} (commit {_xml_escape(prov.git_commit or '')}), "
        f"schema {_xml_escape(prov.schema_version or '')}, run at {_xml_escape(prov.timestamp_utc or '')}."
    )
    creators_xml = ""
    for pm in papers.values():
        if not pm or not pm.authors:
            continue
        for author in pm.authors:
            creators_xml += (
                f"    <creator><individualName><surName>{_xml_escape(author)}</surName>"
                f"</individualName></creator>\n"
            )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<eml:eml xmlns:eml="https://eml.ecoinformatics.org/eml-2.1.1"\n'
        '         xmlns:dc="http://purl.org/dc/terms/" packageId="rlpe-v1" '
        f'system="rlpe" scope="system" xml:lang="en">\n'
        "  <dataset>\n"
        f"    <title>{dataset_title}</title>\n"
        f"    <abstract><para>{abstract}</para></abstract>\n"
        "    <pubDate>" + (prov.timestamp_utc or "")[:10] + "</pubDate>\n"
        "    <language>eng</language>\n"
        f"    <creatorList>\n{creators_xml}    </creatorList>\n"
        "  </dataset>\n"
        "</eml:eml>\n"
    )


def write_dwca_zip(
    run: RunOutput | dict,
    target: Path,
    options: DwCAOptions | None = None,
) -> int:
    """Write a DwC-A zip file. Returns the row count.

    Phase 58 Plan 1.1 (Bug 1.1): the GUI's ``_build_run_output`` returns a
    plain ``dict``, not a :class:`RunOutput`. We accept either and coerce
    transparently. If ``provenance`` is incomplete (e.g. GUI only supplies
    ``job_id``/``source``), we fill in stub fields rather than rejecting
    the export outright — DwC-A consumers care more about the occurrence
    rows than a perfect provenance stamp.
    """
    if isinstance(run, dict):
        run = _coerce_run_output_from_dict(run)
    options = options or DwCAOptions()
    target.parent.mkdir(parents=True, exist_ok=True)

    panels = run.panels
    if not options.include_unmatched:
        panels = [p for p in panels if p.species]

    field_names = [f for f, _ in DWC_FIELDS]
    rows = [_occurrence_row(p) for p in panels]

    # audit 2026-08-17 (EXP-3): occurrenceID MUST be unique within a
    # Darwin Core Archive. Previously ``PanelRecord.panel_id`` was
    # allowed to be None / duplicated, and ``_occurrence_row`` would
    # happily produce two rows with the same ``occurrenceID``
    # (e.g. paper_id:figure_id:_), producing a non-GBIF-compliant
    # archive. We now collect every occ_id and raise ``ValueError``
    # with a clear message listing the duplicates, so the operator
    # can fix the input data (or pass ``include_unmatched=False`` to
    # drop the empties). We prefer raising over silent suffixing
    # because silent ``:dupN`` suffixes would silently hide the
    # underlying data-quality problem.
    seen_occ_ids: set[str] = set()
    duplicates: dict[str, int] = {}
    for r in rows:
        occ_id = r.get("occurrenceID") or ""
        if occ_id in seen_occ_ids:
            duplicates[occ_id] = duplicates.get(occ_id, 0) + 1
        seen_occ_ids.add(occ_id)
    if duplicates:
        sample = sorted(duplicates.items())[:10]
        raise ValueError(
            "Duplicate occurrenceID values in DwC-A export: "
            + ", ".join(f"{oid!r} x{cnt + 1}" for oid, cnt in sample)
            + (f" (and {len(duplicates) - 10} more)" if len(duplicates) > 10 else "")
            + ". Fix the panel data (panel_id collisions / missing panel_id) "
            "or pass DwCAOptions(include_unmatched=False) to drop empty rows."
        )

    # Build occurrence.txt in-memory
    buf = io.StringIO()
    w = csv.DictWriter(
        buf,
        fieldnames=field_names,
        delimiter="\t",
        extrasaction="ignore",
        lineterminator="\n",
    )
    w.writeheader()
    for r in rows:
        w.writerow(r)
    occurrence_txt = buf.getvalue()

    # Build meta.xml and eml.xml
    meta_xml = _build_meta_xml(options)
    eml_xml = _build_eml_xml(run)

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.xml", meta_xml)
        zf.writestr("eml.xml", eml_xml)
        zf.writestr("occurrence.txt", occurrence_txt)
    return len(rows)


def _coerce_run_output_from_dict(run: dict) -> RunOutput:
    """Coerce a GUI-built dict into a fully-validated :class:`RunOutput`.

    The GUI's :meth:`ResultsTab._build_run_output` (and ``Job.rows``
    loaded from ``matches.jsonl``) supplies only a minimal ``provenance``
    (``job_id`` + ``source``). To keep ``ProvenanceRecord`` happy, we
    backfill any missing required fields with harmless stub values
    (``pipeline_version="unknown"``, ``host="unknown"``, etc.).
    """
    prov = dict(run.get("provenance") or {})
    # Phase 58 Plan 1.1: backwards-compat shim — GUI exports carried
    # GUI-only keys (job_id/source) and lacked full provenance. Strip
    # unknown keys and backfill stubs so RunOutput.model_validate accepts.
    allowed_prov_keys = {
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
    prov = {k: v for k, v in prov.items() if k in allowed_prov_keys}
    prov.setdefault("pipeline_version", "unknown")
    prov.setdefault("schema_version", run.get("schema_version", "1.0.0"))
    prov.setdefault("git_commit", "unknown")
    prov.setdefault("git_dirty", False)
    prov.setdefault("timestamp_utc", "1970-01-01T00:00:00Z")
    prov.setdefault("host", "unknown")
    prov.setdefault("python_version", "unknown")
    # audit 2026-07-31: stub provenance must not be SILENT. A GUI/Web
    # export with a missing git_commit or timestamp used to ship with
    # the fake values ("unknown", "1970-01-01") and the consumer could
    # not tell it apart from a fully-provenanced export. Warn loudly
    # and stamp the incompleteness on the payload.
    _stubbed = [
        k
        for k in ("git_commit", "timestamp_utc", "pipeline_version")
        if prov.get(k) in ("unknown", "1970-01-01T00:00:00Z")
    ]
    if _stubbed:
        import logging as _logging

        # audit 2026-07-31: the stub values themselves (git_commit=
        # "unknown", timestamp_utc="1970-01-01…") remain in the
        # export — consumers can detect them — but the operator is
        # now WARNED instead of silently shipping fake provenance.
        _logging.getLogger(__name__).warning(
            "Exporting run WITHOUT full provenance (stubbed: %s) — "
            "the output is not FAIR-traceable to a source commit",
            ", ".join(_stubbed),
        )
    payload = dict(run)
    payload["provenance"] = prov
    return RunOutput.model_validate(payload, context={"skip_dedup": True})
