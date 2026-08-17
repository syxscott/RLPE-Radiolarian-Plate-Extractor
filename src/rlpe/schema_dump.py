"""CLI: regenerate the published JSON Schema.

Usage::

    python -m rlpe.schema_dump                # writes to schemas/rlpe-v1.0.0.json
    python -m rlpe.schema_dump --out /tmp/x.json

The schema file is what downstream consumers (and the schema-test in
``tests/test_schema_published.py``) pin to. Re-run this whenever a
field in ``rlpe.schema_models`` is added, removed, or has its
semantics changed, and bump ``SCHEMA_VERSION``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import schema_models as _schema_models
from .schema_models import SCHEMA_VERSION, emit_json_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = REPO_ROOT / "schemas" / f"rlpe-v{SCHEMA_VERSION}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit the RLPE JSON Schema")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_TARGET, help="Where to write the schema file"
    )
    args = parser.parse_args()
    # Phase 61 Plan 4 fix: Pydantic v2 caches model schemas across
    # imports. After adding a field like ScaleBarRecord.warning, the
    # cached schema lags behind the source. Force-rebuild every BaseModel
    # in this module so the emitted schema matches the live model.
    for name in dir(_schema_models):
        obj = getattr(_schema_models, name)
        if isinstance(obj, type) and issubclass(obj, _schema_models.BaseModel):
            try:
                obj.model_rebuild(force=True)
            except Exception:
                pass
    out = emit_json_schema(args.out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
