"""Audit regression test for Phase B smoke_schematic matcher typo.

Original (Phase 64 commit 6a003cd + eb3c728) had a matcher that
checked ``"schematic_geo" in sp``, but PROMPT_REGISTRY uses
``schematic_extract`` (renamed in Phase 64 fixup commit eb3c728
to avoid collision with test_m3_geology_extraction's every-_geo-
prompt-must-mention-formation assertion). The old matcher NEVER
fired, so the 5-paper smoke test was a silent no-op.

This test pins the matcher to ``schematic_extract`` so the typo
can't be reintroduced.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts" / "smoke_schematic.py"


def _read_script() -> str:
    return SCRIPTS.read_text(encoding="utf-8")


def test_matcher_uses_schematic_extract_key():
    """The matcher in _make_engine() must check the real PROMPT_REGISTRY key."""
    src = _read_script()
    # Find the matcher lambda
    match = re.search(r'"match":\s*lambda sp:\s*"([^"]+)"\s+in sp', src)
    assert match is not None, "Could not find FakeM3Backend matcher lambda in smoke_schematic.py"
    matched_key = match.group(1)
    # The actual PROMPT_REGISTRY key for schematic extraction is
    # "schematic_extract" (renamed from "schematic_geo" in Phase 64
    # fixup commit eb3c728).
    assert matched_key == "schematic_extract", (
        f"smoke_schematic matcher checks '{matched_key}' but "
        f"PROMPT_REGISTRY uses 'schematic_extract'. This means the "
        f"canned M3 response never fires and the smoke test is a no-op. "
        f"Fix: change the lambda to use 'schematic_extract'."
    )


def test_matcher_does_not_use_legacy_schematic_geo_key():
    """The typo'd key 'schematic_geo' must not appear in any matcher."""
    src = _read_script()
    # The legacy key should not appear anywhere except possibly in comments
    # explaining why we changed it.
    legacy_matches = re.findall(r'"\s*schematic_geo\s*"\s*,?\s*\)?', src, re.MULTILINE)
    # Allow zero matches in non-comment lines (we removed the lambda body).
    # Strip both # comments and triple-quoted docstrings before counting
    # active code occurrences.
    import re as _re

    code_only = _re.sub(r'"""[\s\S]*?"""', "", src)
    code_only = _re.sub(r"'''[\s\S]*?'''", "", code_only)
    code_only = "\n".join(
        line for line in code_only.splitlines() if not line.lstrip().startswith("#")
    )
    code_occurrences = re.findall(r'"\s*schematic_geo\s*"', code_only)
    assert code_occurrences == [], (
        f"Active code still references the legacy 'schematic_geo' key: "
        f"{code_occurrences}. This is the typo that made the smoke test a no-op."
    )
    # Sanity: ensure the comment that documents the typo exists, otherwise
    # this test loses its teaching value for future readers.
    assert "schematic_geo" in src, (
        "Expected a comment mentioning the legacy 'schematic_geo' key as "
        "anti-pattern documentation; remove this assertion if intentionally "
        "scrubbing comments."
    )
