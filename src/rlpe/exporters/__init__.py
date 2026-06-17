"""Exporters — produce downstream-friendly views of a RunOutput.

Three views are produced:

- :mod:`rlpe.exporters.analysis`: flat CSV/Parquet with DwC field names,
  for paleontologists analysing the data in Excel/R/Python.
- :mod:`rlpe.exporters.ml`: JSONL with train/val/test split by paper,
  for ML researchers who want to load the data via HuggingFace datasets.
- :mod:`rlpe.exporters.archive`: Darwin Core Archive (DwC-A) zip with
  meta.xml + occurrence.txt, for PBDB/GBIF ingest.

All exporters are pure functions: ``(RunOutput, options) -> output``.
They do not modify the input.
"""

from .analysis import AnalysisOptions, write_csv, write_parquet
from .archive import DwCAOptions, write_dwca_zip
from .ml import MLOptions, write_ml_split

__all__ = [
    "AnalysisOptions",
    "write_csv",
    "write_parquet",
    "DwCAOptions",
    "write_dwca_zip",
    "MLOptions",
    "write_ml_split",
]
