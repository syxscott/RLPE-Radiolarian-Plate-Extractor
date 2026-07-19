"""Tests for Phase 63 Plan 6 — Bug 6.11: ``_run_job`` panel_path URL
must be URL-encoded.

Before: ``panel_path = f"/jobs/{job_id}/files/{rel.as_posix()}"``
emitted a raw relative path. If the panel crop lives in a
sub-directory whose name contains a space, percent, or non-ASCII
char (Japanese paper figure folder ``図版1``), the URL is malformed
and the browser / curl / requests client sends a broken request.

After: ``rel.as_posix()`` is run through ``urllib.parse.quote`` with
``safe='/'`` (so the path separators stay readable) and the URL is
valid for any client.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


APP_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "rlpe" / "api" / "app.py"
).read_text(encoding="utf-8")


def test_run_job_panel_path_url_encoded():
    """The line that constructs ``panel_path`` must URL-encode ``rel``."""
    # The raw f-string `f"/jobs/{job_id}/files/{rel.as_posix()}"` no
    # longer exists; the new line must call urllib.parse.quote.
    assert 'rel.as_posix()' not in APP_SRC.split('from ..pipeline')[0] or (
        'urllib.parse.quote' in APP_SRC
    ), (
        "api/app.py still constructs the URL with rel.as_posix() "
        "without URL-encoding. Phase 63 Plan 6.11 fix regressed?"
    )


def test_url_encode_import():
    """``urllib.parse.quote`` is imported in app.py."""
    assert "urllib.parse" in APP_SRC, (
        "api/app.py does not import urllib.parse — "
        "Phase 63 Plan 6.11 URL-encoding fix regressed?"
    )


def test_panel_path_construction_uses_quote():
    """The line that constructs panel_path must use quote(..., safe='/')."""
    # Look for the actual patched line with ``safe='/'`` and ``rel``
    idx = APP_SRC.find('normalized["panel_path"]')
    assert idx > 0
    block = APP_SRC[idx:idx + 600]
    assert "quote(" in block and "safe=" in block, (
        f"panel_path URL construction does not use urllib.parse.quote "
        f"with safe='/'. Got block:\n{block[:200]!r}"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
