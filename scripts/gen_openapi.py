"""Regenerate the committed OpenAPI snapshot for the RLPE API.

This script is the source of truth for `docs/openapi-1.1.0.json`.
It loads the FastAPI app and writes the spec to disk with the
package version (from `importlib.metadata`) substituted into
`info.version` so the snapshot version stays in sync with
pyproject.toml.

Run from the repo root:

    PYTHONPATH=src python scripts/gen_openapi.py

Or via the module form:

    PYTHONPATH=src python -m rlpe.api.gen_openapi

The output path is hardcoded so the snapshot is reproducible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "openapi-1.1.0.json"


def _pkg_version() -> str:
    """Read the package version. Prefer pyproject.toml over the
    installed-package metadata so the snapshot version matches what
    `pip install -e .` will publish (and so the script works in
    dev environments where `pip install -e .` hasn't been re-run
    since the last version bump)."""
    import re
    pyproject = REPO_ROOT / "pyproject.toml"
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    if m:
        return m.group(1)
    try:
        from importlib.metadata import version
        return version("rlpe-radiolarian-plate-extractor")
    except Exception:
        return "1.1.0"


def main() -> int:
    from rlpe.api.app import app
    spec = app.openapi()
    spec["info"]["version"] = _pkg_version()
    spec["info"]["description"] = (
        f"RLPE v{spec['info']['version']} — Radiolarian Literature "
        "Plate Extractor. This OpenAPI snapshot is committed at the "
        f"v{spec['info']['version']} release; regenerate via "
        "`PYTHONPATH=src python scripts/gen_openapi.py` when the API "
        "changes."
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {OUT_PATH.relative_to(REPO_ROOT)} "
        f"({OUT_PATH.stat().st_size} bytes; "
        f"{len(spec['paths'])} paths, "
        f"{len(spec['components']['schemas'])} schemas)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
