from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    src_str = str(src)
    if src.exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)
