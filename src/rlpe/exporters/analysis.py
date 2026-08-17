"""Analysis view: flat CSV/Parquet with DwC field names.

The CSV columns map directly to Darwin Core terms where possible:

- ``occurrenceID``     = unique row ID (paper_id + figure_id + panel_id)
- ``scientificName``   = species
- ``basisOfRecord``    = "FossilSpecimen"
- ``eventDate``        = publication year from paper_metadata
- ``locality``         = first geology link's locality
- ``decimalLatitude``  = first geology link's latitude
- ``decimalLongitude`` = first geology link's longitude
- ``geologicalContextID`` = first geology link's age
- ``identifiedBy``     = paper authors (joined with "; ")
- ``associatedReferences`` = DOI

Plus RLPE-specific columns (``panel_id``, ``figure_id``, ``paper_id``,
``confidence``, ``label_text``) for traceability.

Parquet uses the same column names with proper types (no string-only
fallback for numeric lat/long).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..export import _sanitise_csv_cell as _shared_sanitise_csv_cell
from ..schema_models import PanelRecord, RunOutput


def _sanitise_csv_cell(value: Any) -> Any:
    """Thin re-export of :func:`rlpe.export._sanitise_csv_cell`.

    audit 2026-08-17 (EXP-4): the canonical sanitiser now lives in
    ``rlpe.export`` so the CLI ``--export-csv`` path and the
    analysis-view ``write_csv`` share the same CWE-1236 defence.
    Pre-existing imports of ``_sanitise_csv_cell`` from this module
    keep working unchanged.
    """
    return _shared_sanitise_csv_cell(value)


@dataclass(slots=True)
class AnalysisOptions:
    """Options for the analysis-view export."""

    include_unmatched: bool = True
    # Phase 63 Plan 6.10 (Bug 6.10): default to utf-8-sig so Excel on
    # Windows detects the UTF-8 encoding and renders Greek / CJK
    # scientificName / locality chars verbatim. The 3-byte BOM is
    # transparent to csv.DictReader / Pandas (``utf-8-sig`` strips
    # it on read).
    csv_encoding: str = "utf-8-sig"
    csv_delimiter: str = ","


CSV_COLUMNS: list[str] = [
    "occurrenceID",
    "paper_id",
    "figure_id",
    "panel_id",
    "scientificName",
    # P3-1 fix: scientificNameAuthorship was extracted (Phase 63) but never
    # exported in analysis view — add it so CSV consumers see the authority/year.
    "scientificNameAuthorship",
    "basisOfRecord",
    "eventDate",
    "locality",
    "decimalLatitude",
    "decimalLongitude",
    "geologicalContextID",
    "formation",
    "identifiedBy",
    "associatedReferences",
    "scale_bar_value",
    "scale_bar_unit",
    "scale_bar_um_per_px",
    "label_text",
    "confidence",
    "matcher_type",
    "extraction_source",
    "panel_path",
]


def _to_analysis_row(panel: PanelRecord) -> dict[str, Any]:
    """Transform a :class:`PanelRecord` into a flat dict with DwC column names."""
    pm = panel.paper_metadata
    geo = panel.metadata.geology_links[0] if panel.metadata.geology_links else None
    sb = panel.metadata.scale_bar
    occurrence_id_parts = [panel.paper_id, panel.figure_id, panel.panel_id or "_"]
    occurrence_id = ":".join(p for p in occurrence_id_parts if p)
    # Phase 58 Plan 1.2 (Bug 1.2): prefer modern_latitude/longitude when
    # present, fall back to legacy latitude/longitude (Round 25+ convention).
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

    # P3-1 fix: pull authorship from TaxonRecord if present.
    # Use getattr since PanelRecord may be constructed without taxa field.
    _taxa = getattr(panel, "taxa", None) or []
    # audit 2026-07-26 M11: pick the taxon matching panel.taxon_id
    # rather than blindly using _taxa[0] - a panel may carry multiple
    # TaxonRecords (synonyms/children) and [0] is not guaranteed to be
    # the one panel.species refers to, so authorship was often empty.
    _authorship = ""
    if _taxa:
        _panel_tid = getattr(panel, "taxon_id", None)
        _match = (
            next((t for t in _taxa if getattr(t, "taxon_id", None) == _panel_tid), None)
            if _panel_tid
            else None
        )
        if _match is None:
            _match = _taxa[0]
        _authorship = _match.scientific_name_authorship or ""
    return {
        "occurrenceID": occurrence_id,
        "paper_id": panel.paper_id,
        "figure_id": panel.figure_id,
        "panel_id": panel.panel_id or "",
        "scientificName": panel.species or "",
        "scientificNameAuthorship": _authorship,
        "basisOfRecord": "FossilSpecimen" if panel.species else "",
        "eventDate": str(pm.year) if pm and pm.year else "",
        "locality": (geo.locality if geo and geo.locality else "") or "",
        "decimalLatitude": lat,
        "decimalLongitude": lon,
        "geologicalContextID": (geo.age if geo and geo.age else "") or "",
        "formation": (geo.formation if geo and geo.formation else "") or "",
        "identifiedBy": ("; ".join(pm.authors) if pm and pm.authors else ""),
        "associatedReferences": (pm.doi if pm and pm.doi else "") or "",
        "scale_bar_value": (sb.value if sb and sb.value is not None else None),
        "scale_bar_unit": (sb.unit if sb and sb.unit else "") or "",
        "scale_bar_um_per_px": (sb.um_per_px if sb and sb.um_per_px is not None else None),
        "label_text": (panel.label_text or "") or "",
        "confidence": panel.confidence,
        "matcher_type": panel.metadata.matcher_type,
        "extraction_source": panel.metadata.extraction_source,
        "panel_path": (panel.panel_path or "") or "",
    }


def panels_to_rows(run: RunOutput, options: AnalysisOptions | None = None) -> list[dict[str, Any]]:
    """Project all panels of a RunOutput into analysis-view rows."""
    options = options or AnalysisOptions()
    rows: list[dict[str, Any]] = []
    for p in run.panels:
        if not options.include_unmatched and not p.species:
            continue
        rows.append(_to_analysis_row(p))
    return rows


def write_csv(
    run: RunOutput,
    target: Path,
    options: AnalysisOptions | None = None,
) -> int:
    """Write the analysis view to a CSV file. Returns the row count."""
    options = options or AnalysisOptions()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = panels_to_rows(run, options)
    with open(target, "w", encoding=options.csv_encoding, newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            delimiter=options.csv_delimiter,
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            # Round 15 audit: sanitise CSV cells against formula
            # injection (CWE-1236). Excel/LibreOffice treat a cell
            # starting with =, +, -, @, or tab as a formula; a paper
            # title like ``=cmd|'/c calc'!A1`` would execute on open.
            # Prefixing with a single quote neutralises the formula.
            w.writerow({k: _sanitise_csv_cell(v) for k, v in r.items()})
    return len(rows)


def write_parquet(
    run: RunOutput,
    target: Path,
    options: AnalysisOptions | None = None,
) -> int:
    """Write the analysis view to a Parquet file. Returns the row count.

    Requires the optional ``pyarrow`` dependency. If missing, raises
    :class:`ImportError` with an installation hint.
    """
    options = options or AnalysisOptions()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = panels_to_rows(run, options)
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "Parquet export requires pyarrow. Install with: pip install pyarrow"
        ) from e
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, target)
    return len(rows)
