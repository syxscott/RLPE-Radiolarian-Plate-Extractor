"""Regression: audit 2026-09-04 llm-4 — the M3 prompt few-shot
examples in :mod:`rlpe.m3_engine` taught the model to INVENT
coordinates from locality / country strings:

  m3_engine.py:438-439   Mt. Wilmot, Wyoming  →  44.0, -107.9
  m3_engine.py:535-536   Lombardy Basin, Italy → 45.4, 9.5
  m3_engine.py:649       Italy (filled circle) → 45.4, 12.0
  m3_engine.py:656       Tunisia (filled circle) → 34.0, 9.0
  m3_engine.py:663       Spain (filled circle) → 40.0, -3.0

The few-shot examples show the model: "given a country / locality
string in the caption, emit specific decimal lat/lon". The model
learns to do this for ANY future input — generating
pseudo-precise DwC ``decimalLatitude`` / ``decimalLongitude``
values from just a name. These values then get exported as
scientific facts (Darwin Core) with no provenance flag.

Real failure mode: a paper caption mentions "Gulf of Mexico
Miocene locality" — the model emits ``decimalLatitude=23.5,
decimalLongitude=-90.0`` (Mexico centroid? Gulf centroid? Guessing).
The DwC export puts those numbers into a biodiversity database as
if they were measured coordinates. No provenance flag distinguishes
"model invented from locality name" from "read off the figure".

Fix contract:
  * Every few-shot example uses ``"latitude": null, "longitude":
    null`` UNLESS the source explicitly shows coordinates in the
    image (e.g. a map with graticule ticks).
  * The prompt instructions explicitly forbid inventing coordinates
    from locality / country strings.
  * No more "model-trained-on-fabrication" pattern in the M3 prompt
    library.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))


# Locale strings the audit identified. These appear in few-shot
# examples paired with a fabricated lat/lon. The fix must remove the
# fabrication (latitude: null / longitude: null).
_AUDIT_LOCALES = [
    ("Mt. Wilmot", "44.0"),  # Wyoming outcrop
    ("Lombardy Basin", "45.4"),  # Italy basin
    ("Italy", None),  # appears in paleogeographic + litholog
    ("Tunisia", "34.0"),
    ("Spain", "40.0"),
]


def _find_block_around(src: str, anchor: str, window: int = 800) -> str | None:
    """Return up to ``window`` characters of source text starting at
    ``anchor`` — enough to capture the few-shot block."""
    i = src.find(anchor)
    if i == -1:
        return None
    return src[i : i + window]


class TestM3PromptsDoNotTeachCoordinateInvention:
    @pytest.mark.parametrize(
        "locale,fab_value", [(l, v) for l, v in _AUDIT_LOCALES], ids=[l for l, _ in _AUDIT_LOCALES]
    )
    def test_few_shot_block_for_locale_uses_null_lat(self, locale, fab_value):
        """For every locale the audit flagged, the surrounding
        few-shot block must NOT contain a non-null latitude paired
        with that locale.

        Strategy: locate the few-shot block by its locale string,
        then scan forward up to ~1500 chars (long enough to capture
        the example output JSON) and check whether any non-null
        ``"latitude": <number>`` appears.
        """
        src = (_SRC / "rlpe" / "m3_engine.py").read_text(encoding="utf-8")
        idx = src.find(f'"{locale}"')
        if idx == -1:
            # Locale not in the file — fix already applied, nothing
            # to check. (Audit flagged it; fix removed it.)
            pytest.skip(f"{locale!r} no longer appears in m3_engine.py")
        # Look at a window starting from the locale mention. The
        # few-shot example output block is typically within 1500
        # chars of the input caption mention.
        window = src[idx : idx + 1500]
        # Within the example output block, scan for "latitude": <number>.
        # If found and non-null, the bug is present.
        bad = re.search(r'"latitude"\s*:\s*-?[0-9]+(\.[0-9]+)?', window)
        assert bad is None, (
            f"audit 2026-09-04 llm-4: few-shot example for locale "
            f"{locale!r} still teaches the model to fabricate "
            f"coordinates — found {bad.group(0)!r} within 1500 "
            f"chars of the locale mention. The fix must change "
            f'this to "latitude": null, "longitude": null. '
            f"Expected: the model should only emit coordinates when "
            f"they are explicitly printed on the figure."
        )


class TestPromptForbidsCoordinateInvention:
    def _prompt_body(self, prompt_name: str) -> str:
        src = (_SRC / "rlpe" / "m3_engine.py").read_text(encoding="utf-8")
        m = re.search(
            rf'"{prompt_name}"\s*:\s*\(\s*"(?P<body>.*?)"\s*,\s*"',
            src,
            re.DOTALL,
        )
        assert m, f"prompt {prompt_name!r} not found in m3_engine.py"
        return m.group("body")

    def test_strat_column_prompt_explicitly_says_no_invention(self):
        """The strat_column prompt must explicitly tell the model
        not to invent coordinates from locality names."""
        body = self._prompt_body("strat_column_geo")
        # Acceptable phrases. The wording can vary but must forbid
        # the act of invention.
        phrases = [
            "do not invent",
            "do not fabricate",
            "never invent",
            "must not invent",
            "must be null unless",
            "explicitly stated",
            "only if printed",
            "only if coordinates are",
        ]
        body_lower = body.lower()
        matched = any(p in body_lower for p in phrases)
        assert matched, (
            f"audit 2026-09-04 llm-4: strat_column prompt must "
            f"explicitly forbid inventing coordinates from locality "
            f"strings. None of the expected phrases found in the "
            f"prompt body. Searched: {phrases}"
        )

    def test_litholog_column_prompt_explicitly_says_no_invention(self):
        body = self._prompt_body("litholog_column_geo")
        phrases = [
            "do not invent",
            "do not fabricate",
            "never invent",
            "must not invent",
            "must be null unless",
            "explicitly stated",
            "only if printed",
            "only if coordinates are",
        ]
        body_lower = body.lower()
        matched = any(p in body_lower for p in phrases)
        assert matched, (
            "audit 2026-09-04 llm-4: litholog_column prompt must "
            "explicitly forbid inventing coordinates."
        )

    def test_paleogeographic_prompt_explicitly_says_no_invention(self):
        body = self._prompt_body("paleogeographic_map_geo")
        phrases = [
            "do not invent",
            "do not fabricate",
            "never invent",
            "must not invent",
            "must be null unless",
            "explicitly stated",
            "only if printed",
            "only if coordinates are",
            "only if visible",
        ]
        body_lower = body.lower()
        matched = any(p in body_lower for p in phrases)
        assert matched, (
            "audit 2026-09-04 llm-4: paleogeographic_map prompt "
            "must explicitly forbid inventing coordinates."
        )
