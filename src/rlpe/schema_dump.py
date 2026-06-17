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

from .schema_models import SCHEMA_VERSION, emit_json_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = REPO_ROOT / "schemas" / f"rlpe-v{SCHEMA_VERSION}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit the RLPE JSON Schema")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_TARGET, help="Where to write the schema file"
    )
    args = parser.parse_args()
    out = emit_json_schema(args.out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
