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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..schema_models import PanelRecord, RunOutput


@dataclass(slots=True)
class AnalysisOptions:
    """Options for the analysis-view export."""
    include_unmatched: bool = True
    csv_encoding: str = "utf-8"
    csv_delimiter: str = ","


CSV_COLUMNS: list[str] = [
    "occurrenceID",
    "paper_id",
    "figure_id",
    "panel_id",
    "scientificName",
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

    return {
        "occurrenceID": occurrence_id,
        "paper_id": panel.paper_id,
        "figure_id": panel.figure_id,
        "panel_id": panel.panel_id or "",
        "scientificName": panel.species or "",
        "basisOfRecord": "FossilSpecimen" if panel.species else "",
        "eventDate": str(pm.year) if pm and pm.year else "",
        "locality": (geo.locality if geo and geo.locality else "") or "",
        "decimalLatitude": (geo.latitude if geo and geo.latitude is not None else ""),
        "decimalLongitude": (geo.longitude if geo and geo.longitude is not None else ""),
        "geologicalContextID": (geo.age if geo and geo.age else "") or "",
        "formation": (geo.formation if geo and geo.formation else "") or "",
        "identifiedBy": ("; ".join(pm.authors) if pm and pm.authors else ""),
        "associatedReferences": (pm.doi if pm and pm.doi else "") or "",
        "scale_bar_value": (sb.value if sb and sb.value is not None else ""),
        "scale_bar_unit": (sb.unit if sb and sb.unit else "") or "",
        "scale_bar_um_per_px": (sb.um_per_px if sb and sb.um_per_px is not None else ""),
        "label_text": (panel.label_text or "") or "",
        "confidence": panel.confidence,
        "matcher_type": panel.metadata.matcher_type,
        "extraction_source": panel.metadata.extraction_source,
        "panel_path": (panel.panel_path or "") or "",
    }


def panels_to_rows(
    run: RunOutput, options: AnalysisOptions | None = None
) -> list[dict[str, Any]]:
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
            f, fieldnames=CSV_COLUMNS, delimiter=options.csv_delimiter,
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
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
