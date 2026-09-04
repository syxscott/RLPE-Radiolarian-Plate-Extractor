"""Regression: audit 2026-09-04 llm-1 — prompt-compliant ``{"panels":
[...]}`` answers were silently discarded.

The LLM-first system prompt (pipeline.py ``_LLM_FIRST_SYSTEM_PROMPT``)
demands: "The JSON must be an object with a single key 'panels' whose
value is an array of objects" — and even the placeholder case must be
``{"panels": []}``. But ``parse_json_from_text`` handed that dict to
``_normalize_panel_dict``, whose schema whitelist does not contain
"panels", so the key was dropped and a single empty row
(label=None, species=None) came back. The more faithfully a model
followed the prompt, the more of its paid answer was destroyed. The
same payload as a bare top-level array parsed fine (2 panels), which
is why tests that stub the model with arrays never caught this — the
same stub-the-parser blind spot that produced the 2026-09 F1 0.84 vs
0.075 embarrassment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.llm_backends import parse_json_from_text

_COMPLIANT_OBJECT = """{
  "panels": [
    {"label": "1", "species": "Follicucullus scholasticus", "confidence": 0.9},
    {"label": "2", "species": "Pseudoalbaillella globosa", "confidence": 0.8},
    {"label": "3", "species": null, "confidence": 0.4}
  ]
}"""


class TestPromptCompliantPanelsObject:
    def test_top_level_panels_object_yields_all_panels(self):
        res = parse_json_from_text(_COMPLIANT_OBJECT)
        assert res.get("_is_multi_panel") is True
        panels = res["panels"]
        assert len(panels) == 3
        assert panels[0]["species"] == "Follicucullus scholasticus"
        assert panels[1]["species"] == "Pseudoalbaillella globosa"

    def test_compliant_object_beats_single_empty_row(self):
        # The exact historical failure: one row with species=None.
        res = parse_json_from_text(_COMPLIANT_OBJECT)
        if res.get("_is_multi_panel"):
            rows = res["panels"]
        else:
            rows = [res]
        assert not (len(rows) == 1 and rows[0].get("species") is None), (
            "prompt-compliant panels object collapsed to a single empty row"
        )

    def test_panels_object_with_prose_preamble(self):
        text = "Here is the extraction result:\n```json\n" + _COMPLIANT_OBJECT + "\n```"
        res = parse_json_from_text(text)
        assert res.get("_is_multi_panel") is True
        assert len(res["panels"]) == 3

    def test_placeholder_panels_empty_list(self):
        # Prompt says the placeholder caption case returns {"panels": []}
        res = parse_json_from_text('{"panels": []}')
        # Must NOT become a fake empty row; either multi-panel empty or
        # an empty-ish single dict is acceptable, but no fabricated row.
        if res.get("_is_multi_panel"):
            assert res["panels"] == []
        else:
            assert res.get("species") is None

    def test_bare_array_still_works(self):
        res = parse_json_from_text(
            '[{"label": "1", "species": "Albaillella excelsus", "confidence": 0.9}]'
        )
        assert res.get("_is_multi_panel") is True
        assert res["panels"][0]["species"] == "Albaillella excelsus"

    def test_plain_single_panel_dict_still_works(self):
        res = parse_json_from_text(
            '{"label": "2", "species": "Nazarovella gracilis", "confidence": 0.85}'
        )
        assert res.get("species") == "Nazarovella gracilis"
        assert res.get("_is_multi_panel") is None

    def test_panels_key_not_in_field_whitelist_is_fine(self):
        # The whitelist still exists to drop hallucinated per-panel
        # fields; "panels" itself is handled structurally before the
        # whitelist ever sees a panel dict.
        from rlpe.llm_backends import _normalize_panel_dict

        cleaned = _normalize_panel_dict({"label": "1", "species": "X", "panels": [1, 2]})
        assert "panels" not in cleaned
