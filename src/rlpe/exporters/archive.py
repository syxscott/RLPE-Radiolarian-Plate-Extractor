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

from ..schema_models import PanelRecord, RunOutput


@dataclass(slots=True)
class DwCAOptions:
    """Options for the Darwin Core Archive export."""

    include_unmatched: bool = False
    encoding: str = "utf-8"
    fields_terminated_by: str = "\\t"
    lines_terminated_by: str = "\\n"
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
]


def _occurrence_row(panel: PanelRecord) -> dict[str, str]:
    """Build a single occurrence row, all values as strings."""
    pm = panel.paper_metadata
    geo = panel.metadata.geology_links[0] if panel.metadata.geology_links else None
    parts = [panel.paper_id, panel.figure_id, panel.panel_id or "_"]
    occ_id = ":".join(p for p in parts if p)
    media = panel.panel_path or ""
    return {
        "occurrenceID": occ_id,
        "basisOfRecord": "FossilSpecimen" if panel.species else "",
        "scientificName": panel.species or "",
        "kingdom": "",
        "phylum": "",
        "class": "",
        "order": "",
        "family": "",
        "genus": (panel.species.split()[0] if panel.species else ""),
        "specificEpithet": (
            panel.species.split()[1] if panel.species and len(panel.species.split()) >= 2 else ""
        ),
        "eventDate": str(pm.year) if pm and pm.year else "",
        "year": str(pm.year) if pm and pm.year else "",
        "locality": (geo.locality if geo and geo.locality else ""),
        # Phase 38: read country from geo (set by paleodb's
        # _iso_to_country resolution) instead of hard-coded "".
        # Previously every DwCA export had an empty country column
        # even when the PBDB occurrence record had country data.
        "country": (geo.country if geo and geo.country else ""),
        "decimalLatitude": (str(geo.latitude) if geo and geo.latitude is not None else ""),
        "decimalLongitude": (str(geo.longitude) if geo and geo.longitude is not None else ""),
        "geologicalContextID": (geo.age if geo and geo.age else ""),
        "formation": (geo.formation if geo and geo.formation else ""),
        "identifiedBy": ("; ".join(pm.authors) if pm and pm.authors else ""),
        "associatedReferences": (pm.doi if pm and pm.doi else ""),
        "associatedMedia": media,
    }


def _build_meta_xml(opts: DwCAOptions) -> str:
    """Build the meta.xml describing the occurrence core."""
    fields_xml_lines: list[str] = []
    for idx, (_field, uri) in enumerate(DWC_FIELDS, start=1):
        fields_xml_lines.append(f'    <field index="{idx}" term="{uri}"/>')
    fields_xml = "\n".join(fields_xml_lines)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<archive xmlns="http://rs.tdwg.org/dwc/text/" metadata="eml.xml">\n'
        '  <core encoding="UTF-8" fieldsTerminatedBy="\\t" '
        'linesTerminatedBy="\\n" fieldsEnclosedBy="" ignoreHeaderLines="1"'
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
    payload = dict(run)
    payload["provenance"] = prov
    return RunOutput.model_validate(payload)
