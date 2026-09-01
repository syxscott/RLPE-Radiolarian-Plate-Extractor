"""M3-Centric Semantic Figure Understanding Engine.

This module turns MiniMax M3 (a multimodal LLM with extended thinking) from
"末端匹配器" into the *semantic engine* of the radiolarian plate pipeline.
The pipeline is decomposed into 5 cascading M3 stages, each producing
structured JSON consumed by the next stage.  Any stage can fail gracefully
and the next-best classical method takes over.

Stages
------
1. ``parse_caption``         text ->  ``CaptionPair[]``  (label -> species mapping)
2. ``classify_plate``        vision -> ``PlateClassification``  (filter non-radiolarian)
3. ``segment_panels``        vision -> ``PanelBox[]``  (panel bboxes + visible labels)
4. ``match_panel``           vision+text -> ``PanelMatch``  (per-panel species assignment)
5. ``critique_matches``      vision+text -> ``Critique[]``  (cross-panel consistency)

The novelty: M3's multimodal extended-thinking is used as a *joint* document
understanding engine, not a per-panel classifier.  Each stage is small and
focused, the JSON contract is strict, and stages can be turned on/off via
``PipelineConfig.extra['m3_stage_<n>'] = True/False`` for ablation.

Cost (rough): ~¥0.10/figure with 10 panels; 4 figures ≈ ¥0.40.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import Condition, RLock, get_ident
from typing import Any

from PIL import Image

from .grobid import PipelineCancelledError  # noqa: E402

# Import FallbackRecommendedError to detect when backend recommends switching
from .llm_backends import (  # noqa: E402
    FallbackRecommendedError,
    LLMAuthenticationError,
    LLMRateLimitError,
    _apply_geo_whitelist,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM error classification (Phase 4E Task 1)
# ---------------------------------------------------------------------------
#
# Before Phase 4E, the engine caught every M3 exception in one bucket
# (`except Exception`) and only logged "infer_panel failed" — operators
# had no way to distinguish an auth failure (which requires a key rotation)
# from a rate-limit (which is transient) from a timeout (which is
# load-related) from a parse error (which is a model-quality issue).
# The fix adds a typed ``except`` chain in ``_infer_text`` /
# ``_infer_vision`` that maps each stub exception class to a short
# string code stored on ``_telemetry.llm_error``:
#
#   * ``"auth"``        → 401 / 403 → key rotation
#   * ``"rate_limit"``  → 429 → back off
#   * ``"timeout"``     → call timed out → server load
#   * ``"parse"``       → JSON decode / schema violation → model quality
#   * ``"other"``       → unclassified (catch-all)
#
# ``LLMAuthenticationError`` / ``LLMRateLimitError`` live in
# ``llm_backends`` (Phase 4B M-23 introduced them as a hierarchy under
# ``LLMHTTPError``). ``TimeoutError`` is the Python built-in (covers
# socket-read timeouts). The engine-specific ``LLMSchemaError`` covers
# schema violations after a successful JSON parse so audit can tell
# the two parse-stage failure modes apart.


class LLMSchemaError(Exception):
    """LLM output failed schema validation (after a successful parse).

    Phase 4E Task 1: maps to ``_telemetry.llm_error == "parse"``.
    Distinguished from a raw JSONDecodeError so the audit can tell the
    two parse-stage failure modes apart.
    """


def _classify_llm_error(exc: BaseException) -> str:
    """Map an LLM exception instance to a short string code.

    Returns one of: ``"auth"``, ``"rate_limit"``, ``"timeout"``,
    ``"parse"``, ``"other"``. Order matters — more specific
    exception types must be checked before ``Exception`` /
    ``ValueError`` so a custom subclass of ``ValueError`` for
    schema validation isn't silently swallowed by the JSON path.
    """
    if isinstance(exc, LLMAuthenticationError):
        return "auth"
    if isinstance(exc, LLMRateLimitError):
        return "rate_limit"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, (LLMSchemaError, json.JSONDecodeError)):
        return "parse"
    if isinstance(exc, ValueError):
        # ``_safe_json_loads`` raises ValueError when no JSON object
        # can be found in the LLM's output; group with the parse
        # failures so callers can treat the whole parse-class
        # together.
        return "parse"
    return "other"


# ---------------------------------------------------------------------------
# JSON parsing helpers (more lenient than the backend's default)
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _safe_json_loads(text: str) -> Any:
    """Parse JSON from text, tolerating ```json fences and preamble.

    Tries the whole text first, then a JSON array, then a JSON object. If
    parsing fails on a top-level array (e.g. M3 produced a syntax error
    inside), falls back to extracting *individual* balanced objects from
    the text so we can recover as much structure as possible.
    """
    if not text:
        raise ValueError("empty text")
    text = text.strip()
    # Strip common code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # 1) Whole text
    try:
        parsed = json.loads(text)
        # Audit 2026-09-01: also wrap a top-level array into its first
        # dict element so callers that expect a single-object return
        # value keep working. M3 occasionally emits ``[{"sections":
        # [...]}]`` at the top level; previously the caller got a
        # ``list`` and crashed on ``result["sections"]``.
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                return first
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # 2) First array match
    arr_match = _JSON_ARRAY_RE.search(text)
    # Track whether step 2 actually *attempted* a parse (i.e. the regex
    # matched a [...] substring) so step 4 (balanced-object recovery)
    # only runs as a recovery for a *broken array*. Without this gate,
    # M20 made step 4 run for every non-array input too, which turned
    # a single-object preamble like ``"Here is the result: {\"a\": 1}
    # Done."`` into a list ``[{"a": 1}]`` and broke the
    # ``_safe_json_loads_with_preamble`` regression test.
    arr_attempted_and_failed = False
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except Exception:
            arr_attempted_and_failed = True
    # M20: when the array regex matched but failed to parse (typical
    # for LLM output like "[obj, obj, obj]" missing a comma or with
    # trailing junk), try the balanced-objects recovery *before*
    # falling back to the first-object regex. The first-object regex
    # often grabs an *interior* object of the broken array (e.g. the
    # second element), hiding the rest of the data. The balanced-
    # objects pass returns every valid object and lets the caller
    # decide which is the "real" answer.
    # 4) Best-effort: find every balanced {...} block in the text and
    #    parse them individually. Useful when the LLM emits a malformed
    #    array (missing comma, extra brace) but each object is valid.
    if arr_attempted_and_failed:
        items = _extract_balanced_objects(text)
        if items:
            return items
    # 3) First object match. audit 2026-07-31: the greedy
    # ``_JSON_OBJECT_RE`` (``\{.*\}``) spans MULTIPLE concatenated
    # objects ("{"a": 1} {"b": 2}") and fails to parse the whole
    # span, discarding valid output. Try each balanced object
    # non-greedily and return the first that parses.
    for obj_match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            continue
    raise ValueError(f"No JSON object/array found in text: {text[:200]!r}")


def _extract_balanced_objects(text: str) -> list[Any]:
    """Find every balanced ``{...}`` substring in ``text`` and parse each."""
    out: list[Any] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c != "{":
            i += 1
            continue
        # Try to find the matching closing brace, tracking nesting.
        depth = 0
        j = i
        in_str = False
        escape = False
        while j < n:
            ch = text[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        # Check bounds BEFORE using j: when j >= n the loop exited
        # because we ran off the end without finding a closing brace
        # (depth != 0). In that case text[i:j+1] would be text[i:n+1]
        # which Python slices gracefully but produces garbage.
        if j >= n:
            break
        if depth != 0:
            break
        snippet = text[i : j + 1]
        try:
            out.append(json.loads(snippet))
        except Exception:
            # Try the "balanced to the closest closing brace" approach.
            # If this also fails, skip and continue.
            pass
        i = j + 1
    return out


_TELEMETRY_KEYS = (
    "MiniMax_request_id",
    "MiniMax_cost_cny",
    "MiniMax_model_version",
    "MiniMax_usage",
)


def _telemetry_subset(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Pick MiniMax telemetry fields from a backend raw result.

    M3 stage callers propagate ``raw`` into ``PanelMatch.raw``; pipeline
    stage-4 then copies these into MatchResult metadata so /system/llm-
    status can aggregate cost across all stages.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    rid = raw.get("request_id")
    if rid:
        out["MiniMax_request_id"] = str(rid)
    cost = raw.get("cost_cny")
    if cost is not None:
        try:
            out["MiniMax_cost_cny"] = float(cost)
        except (TypeError, ValueError):
            pass
    mv = raw.get("model_version")
    if mv:
        out["MiniMax_model_version"] = str(mv)
    usage = raw.get("usage")
    if isinstance(usage, dict):
        # M12: whitelist only safe usage fields. The raw ``usage`` dict from
        # the provider can contain PII or internal fields (cache TTL,
        # invocation id, organization id, routing data, etc.). We only
        # forward the well-known token accounting fields.
        safe_usage: dict[str, Any] = {}
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            if key in usage:
                safe_usage[key] = usage[key]
        if safe_usage:
            out["MiniMax_usage"] = safe_usage
    return out


# ---------------------------------------------------------------------------
# Multi-modal geology extraction
# ---------------------------------------------------------------------------
#
# ``extract_geology()`` below sends a figure image + caption to the
# MiniMax-M3 backend and asks for structured geology fields (lithology,
# formation, member, group, country, biozone, Ma range, coordinates).
# One system prompt per figure_type keeps each prompt ~150 tokens and
# focused on what to look for. The JSON contract is identical across
# all six prompts so downstream callers can treat the output uniformly.
#
# Range-chart vision extraction is NOT routed through ``extract_geology``;
# ``range_chart_extractor.extract_range_chart()`` keeps its richer per-
# species-range / section / biozone JSON shape (committed earlier in
# 720df73). This module deliberately complements, not replaces, that
# path — different figure types have different data shapes and forcing
# them into a single schema would lose information.

PROMPT_REGISTRY: dict[str, str] = {
    "plate_geo": (
        "You are a geology assistant. Given a SEM plate image of "
        "radiolarians and its caption, extract any geological context "
        "visible in the caption (formation, member, group, lithology, "
        "locality, country). Return strict JSON only:\n\n"
        '{"geo": [{"age": str|null, "chronostratigraphy": str|null, '
        '"chronostratigraphy_rank": str|null, "ma_top": float|null, '
        '"ma_base": float|null, "ma_mid": float|null, '
        '"formation": str|null, "member": str|null, "group": str|null, '
        '"lithology": str|null, "locality": str|null, "country": str|null, '
        '"latitude": float|null, "longitude": float|null, '
        '"biozone": str|null, "confidence": 0.0-1.0}]}\n\n'
        "Use only information visible in the caption. Output JSON only."
    ),
    "range_chart_geo": (
        "You are a geology assistant reading a stratigraphic range chart "
        "or species-distribution diagram. The chart shows species ranges "
        "across measured sections on a vertical axis (top = young, base = "
        "old). Read the age axis carefully and emit numeric Ma bounds "
        "(ma_top = younger boundary, ma_base = older boundary, ma_mid = "
        "(top+base)/2). Also emit biozone names, formation, member, group, "
        "lithology, locality, country when visible. Return strict JSON only:\n\n"
        '{"geo": [{"age": str|null, "chronostratigraphy": str|null, '
        '"chronostratigraphy_rank": str|null, "ma_top": float|null, '
        '"ma_base": float|null, "ma_mid": float|null, '
        '"formation": str|null, "member": str|null, "group": str|null, '
        '"lithology": str|null, "locality": str|null, "country": str|null, '
        '"latitude": float|null, "longitude": float|null, '
        '"biozone": str|null, "confidence": 0.0-1.0}]}\n\n'
        "If the age axis is unreadable, set ma_top/ma_base/ma_mid to null. "
        "Do NOT guess. Output JSON only."
    ),
    "map_geo": (
        "You are a geology assistant reading a geographic / location map. "
        "Extract every place name, country, latitude, longitude, and any "
        "geological context visible (formation, lithology, locality). "
        "Return strict JSON only:\n\n"
        '{"geo": [{"age": str|null, "chronostratigraphy": str|null, '
        '"chronostratigraphy_rank": str|null, "ma_top": float|null, '
        '"ma_base": float|null, "ma_mid": float|null, '
        '"formation": str|null, "member": str|null, "group": str|null, '
        '"lithology": str|null, "locality": str|null, "country": str|null, '
        '"latitude": float|null, "longitude": float|null, '
        '"biozone": str|null, "confidence": 0.0-1.0}]}\n\n'
        "Prioritize country and locality fields; only fill others if the "
        "map legend / labels show them. Output JSON only."
    ),
    "strat_column_geo": (
        "You are a geology assistant reading a stratigraphic column "
        "(columnar section, measured section). The figure shows layered "
        "rock units on a vertical axis. Extract TWO kinds of information:\n\n"
        "1. GLOBAL CONTEXT (one entry in ``geo``): formation / member / group "
        "names, overall age, and Ma range for the entire column.\n\n"
        "2. PER-LAYER (one entry per visible layer in ``layers``): "
        "the lithology, formation, member, age, and Ma range of each "
        "distinct layer visible in the column (from top to bottom).\n\n"
        "Return strict JSON only, no markdown fences:\n\n"
        "{\n"
        '  "geo": [{"age": str|null, "chronostratigraphy": str|null, '
        '"chronostratigraphy_rank": str|null, "ma_top": float|null, '
        '"ma_base": float|null, "ma_mid": float|null, '
        '"formation": str|null, "member": str|null, "group": str|null, '
        '"lithology": str|null, "locality": str|null, "country": str|null, '
        '"latitude": float|null, "longitude": float|null, '
        '"biozone": str|null, "confidence": 0.0-1.0}],\n'
        '  "layers": [\n'
        "    {\n"
        '      "layer_index": int,\n'
        '      "y_top_normalized": float|null,\n'
        '      "y_base_normalized": float|null,\n'
        '      "lithology": str|null,\n'
        '      "formation": str|null,\n'
        '      "member": str|null,\n'
        '      "age": str|null,\n'
        '      "ma_top": float|null,\n'
        '      "ma_base": float|null,\n'
        '      "biozone": str|null,\n'
        '      "thickness_m": float|null,\n'
        '      "evidence": str|null,\n'
        '      "confidence": 0.0-1.0\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- ``y_top_normalized`` and ``y_base_normalized`` are floats 0.0–1.0, "
        "where 0.0 = top of the column and 1.0 = base. Estimate from the "
        "relative heights of layers as printed.\n"
        "- ``layer_index`` starts at 0 (topmost layer).\n"
        "- ``thickness_m`` is the estimated thickness in metres if printed on "
        "the figure; otherwise null.\n"
        "- Include all visible layers even if unlithified or poorly printed.\n"
        "- Output JSON only, no markdown fences.\n\n"
        # Phase 4A: few-shot example covering the strat_column contract.
        "Example (Phase 4A):\n"
        "Input image: Tall stratigraphic column (figure caption: 'Fig. 4. "
        "Composite measured section of the Sundance Fm. showing five "
        "lithostratigraphic units; total thickness 27 m; outcrop at Mt. "
        "Wilmot, Wyoming, USA.'). Three layers visible from top to bottom: "
        "(L0) thin grey-green shale, ~2 m, age 'Late Jurassic'; (L1) "
        "yellowish limestone bed, ~8 m, age 'Middle Jurassic (Bajocian)'; "
        "(L2) red sandstone, ~17 m, age 'Early Jurassic'.\n"
        "Output:\n"
        "{\n"
        '  "geo": [{\n'
        '    "age": "Middle Jurassic",\n'
        '    "chronostratigraphy": null,\n'
        '    "chronostratigraphy_rank": null,\n'
        '    "ma_top": 170.0,\n'
        '    "ma_base": 174.0,\n'
        '    "ma_mid": 172.0,\n'
        '    "formation": "Sundance Fm",\n'
        '    "member": null,\n'
        '    "group": null,\n'
        '    "lithology": "mixed shale/limestone/sandstone",\n'
        '    "locality": "Mt. Wilmot",\n'
        '    "country": "USA",\n'
        '    "latitude": 44.0,\n'
        '    "longitude": -107.9,\n'
        '    "biozone": null,\n'
        '    "confidence": 0.92\n'
        "  }],\n"
        '  "layers": [\n'
        '    {"layer_index": 0, "y_top_normalized": 0.0, '
        '"y_base_normalized": 0.08, "lithology": "shale", '
        '"formation": "Sundance Fm", "member": null, '
        '"age": "Late Jurassic", "ma_top": 163.0, "ma_base": 168.0, '
        '"biozone": null, "thickness_m": 2.0, '
        '"evidence": "greenish shale at top of column", '
        '"confidence": 0.88},\n'
        '    {"layer_index": 1, "y_top_normalized": 0.08, '
        '"y_base_normalized": 0.38, "lithology": "limestone", '
        '"formation": "Sundance Fm", "member": null, '
        '"age": "Middle Jurassic", "ma_top": 170.0, "ma_base": 174.0, '
        '"biozone": null, "thickness_m": 8.0, '
        '"evidence": "yellowish limestone middle bed", '
        '"confidence": 0.9},\n'
        '    {"layer_index": 2, "y_top_normalized": 0.38, '
        '"y_base_normalized": 1.0, "lithology": "sandstone", '
        '"formation": "Sundance Fm", "member": null, '
        '"age": "Early Jurassic", "ma_top": 174.0, "ma_base": 200.0, '
        '"biozone": null, "thickness_m": 17.0, '
        '"evidence": "red sandstone at base", '
        '"confidence": 0.85}\n'
        "  ]\n"
        "}\n\n"
        "Output MUST match the JSON schema exactly. See example below."
    ),
    "litholog_column_geo": (
        "You are a geology assistant reading a lithological log "
        "(litholog column). The figure is a vertical strip showing rock "
        "type patterns and brief annotations. Extract TWO kinds of "
        "information:\n\n"
        "1. GLOBAL CONTEXT (one entry in ``geo``): the overall formation, "
        "member, age, and Ma range visible in the column header or legend.\n\n"
        "2. PER-LAYER (one entry per visible layer in ``layers``): "
        "the lithology, age, formation, and Ma range of each layer "
        "(from top to bottom).\n\n"
        "Return strict JSON only, no markdown fences:\n\n"
        "{\n"
        '  "geo": [{"age": str|null, "chronostratigraphy": str|null, '
        '"chronostratigraphy_rank": str|null, "ma_top": float|null, '
        '"ma_base": float|null, "ma_mid": float|null, '
        '"formation": str|null, "member": str|null, "group": str|null, '
        '"lithology": str|null, "locality": str|null, "country": str|null, '
        '"latitude": float|null, "longitude": float|null, '
        '"biozone": str|null, "confidence": 0.0-1.0}],\n'
        '  "layers": [\n'
        "    {\n"
        '      "layer_index": int,\n'
        '      "y_top_normalized": float|null,\n'
        '      "y_base_normalized": float|null,\n'
        '      "lithology": str|null,\n'
        '      "formation": str|null,\n'
        '      "member": str|null,\n'
        '      "age": str|null,\n'
        '      "ma_top": float|null,\n'
        '      "ma_base": float|null,\n'
        '      "biozone": str|null,\n'
        '      "thickness_m": float|null,\n'
        '      "evidence": str|null,\n'
        '      "confidence": 0.0-1.0\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- ``y_top_normalized`` and ``y_base_normalized`` are floats 0.0–1.0, "
        "where 0.0 = top and 1.0 = base of the column.\n"
        "- ``layer_index`` starts at 0 (topmost layer).\n"
        "- Include all visible lithology patterns even if unlabelled.\n"
        "- Output JSON only, no markdown fences.\n\n"
        # Phase 4A: few-shot example covering the litholog_column contract.
        "Example (Phase 4A):\n"
        "Input image: Narrow litholog column (figure caption: 'Fig. 7. "
        "Lithological log of borehole BH-12 from the Lombardy Basin, "
        "northern Italy; Pleistocene-Holocene fluvial sediments; total "
        "depth 35 m.'). Three layers visible: (L0) light brown clay, ~5 m; "
        "(L1) grey silt with sand lenses, ~12 m; (L2) coarse gravel with "
        "rounded pebbles, ~18 m.\n"
        "Output:\n"
        "{\n"
        '  "geo": [{\n'
        '    "age": "Pleistocene-Holocene",\n'
        '    "chronostratigraphy": null,\n'
        '    "chronostratigraphy_rank": null,\n'
        '    "ma_top": 0.0,\n'
        '    "ma_base": 2.58,\n'
        '    "ma_mid": 1.29,\n'
        '    "formation": null,\n'
        '    "member": null,\n'
        '    "group": null,\n'
        '    "lithology": "mixed clay/silt/gravel",\n'
        '    "locality": "Lombardy Basin",\n'
        '    "country": "Italy",\n'
        '    "latitude": 45.4,\n'
        '    "longitude": 9.5,\n'
        '    "biozone": null,\n'
        '    "confidence": 0.85\n'
        "  }],\n"
        '  "layers": [\n'
        '    {"layer_index": 0, "y_top_normalized": 0.0, '
        '"y_base_normalized": 0.14, "lithology": "clay", '
        '"formation": null, "member": null, '
        '"age": "Holocene", "ma_top": 0.0, "ma_base": 0.012, '
        '"biozone": null, "thickness_m": 5.0, '
        '"evidence": "light brown clay at top", '
        '"confidence": 0.82},\n'
        '    {"layer_index": 1, "y_top_normalized": 0.14, '
        '"y_base_normalized": 0.49, "lithology": "silt with sand lenses", '
        '"formation": null, "member": null, '
        '"age": "Late Pleistocene", "ma_top": 0.012, "ma_base": 0.13, '
        '"biozone": null, "thickness_m": 12.0, '
        '"evidence": "grey silt middle", '
        '"confidence": 0.80},\n'
        '    {"layer_index": 2, "y_top_normalized": 0.49, '
        '"y_base_normalized": 1.0, "lithology": "gravel", '
        '"formation": null, "member": null, '
        '"age": "Middle Pleistocene", "ma_top": 0.13, "ma_base": 0.78, '
        '"biozone": null, "thickness_m": 18.0, '
        '"evidence": "coarse gravel at base", '
        '"confidence": 0.78}\n'
        "  ]\n"
        "}\n\n"
        "Output MUST match the JSON schema exactly. See example below."
    ),
    "paleogeographic_map_geo": (
        "You are a geology assistant reading a paleogeographic map. The "
        "figure shows reconstructed continents at a specific geological age, "
        "and may contain labelled sampling localities (each labelled with a "
        "species name and/or number). Extract TWO kinds of information:\n\n"
        "1. GLOBAL CONTEXT (one entry in ``geo``): the overall map age, "
        "continent/plate names, and any formation/lithology/country visible "
        "in the map legend or labels.\n\n"
        "2. POINT LOCALITIES (one entry per labelled point in ``localities``): "
        "read EVERY labelled locality on the map — each has a label (number "
        "or symbol), a species name (Latin binomial), and coordinates "
        "(latitude/longitude or relative position). Also extract the age, "
        "formation, and lithology associated with each point if visible.\n\n"
        "Return strict JSON only, no markdown fences:\n\n"
        "{\n"
        '  "geo": [{"age": str|null, "chronostratigraphy": str|null, '
        '"chronostratigraphy_rank": str|null, "ma_top": float|null, '
        '"ma_base": float|null, "ma_mid": float|null, '
        '"formation": str|null, "member": str|null, "group": str|null, '
        '"lithology": str|null, "locality": str|null, "country": str|null, '
        '"latitude": float|null, "longitude": float|null, '
        '"biozone": str|null, "confidence": 0.0-1.0}],\n'
        '  "localities": [\n'
        "    {\n"
        '      "species": str|null,\n'
        '      "label": str|null,\n'
        '      "latitude": float|null,\n'
        '      "longitude": float|null,\n'
        '      "paleo_latitude": float|null,\n'
        '      "paleo_longitude": float|null,\n'
        '      "age": str|null,\n'
        '      "ma_top": float|null,\n'
        '      "ma_base": float|null,\n'
        '      "formation": str|null,\n'
        '      "lithology": str|null,\n'
        '      "biozone": str|null,\n'
        '      "evidence": str|null,\n'
        '      "confidence": 0.0-1.0\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- For ``localities``: read EVERY visible point label (1, 2, 3… or "
        "symbols). Include all species names even if partial or hand-written.\n"
        "- Coordinates: use any printed lat/lon. If only relative position is "
        "shown (e.g. dots on a continent outline), record ``latitude: null`` "
        "``longitude: null`` and note the appearance in ``evidence``.\n"
        "- ``age`` per locality overrides the global ``geo.age`` when different.\n"
        "- ``confidence`` per locality reflects how clearly the species name "
        "is legible (not the taxonomic correctness).\n"
        "- Output JSON only, no markdown fences.\n\n"
        # Phase 4A: few-shot example covering the paleogeographic_map contract.
        "Example (Phase 4A):\n"
        "Input image: Paleogeographic map showing the Late Cretaceous "
        "(Maastrichtian, ~70 Ma) reconstruction of the Western Tethys "
        "(caption: 'Fig. 9. Late Cretaceous paleogeographic map of the "
        "Western Tethys region with sampling localities (filled circles) "
        "of radiolarian-bearing sediments.'). Three labelled localities: "
        "(1) Italy, ~45.4 N, 12.0 E, species 'Cretaceous radiolarian sp. A'; "
        "(2) Tunisia, ~34.0 N, 9.0 E, species 'Pseudocrucella sp.'; "
        "(3) Spain, ~40.0 N, -3.0 E, no species label printed.\n"
        "Output:\n"
        "{\n"
        '  "geo": [{\n'
        '    "age": "Late Cretaceous",\n'
        '    "chronostratigraphy": null,\n'
        '    "chronostratigraphy_rank": null,\n'
        '    "ma_top": 66.0,\n'
        '    "ma_base": 72.0,\n'
        '    "ma_mid": 69.0,\n'
        '    "formation": null,\n'
        '    "member": null,\n'
        '    "group": null,\n'
        '    "lithology": "marine chalk/limestone",\n'
        '    "locality": "Western Tethys",\n'
        '    "country": null,\n'
        '    "latitude": null,\n'
        '    "longitude": null,\n'
        '    "biozone": null,\n'
        '    "confidence": 0.9\n'
        "  }],\n"
        '  "localities": [\n'
        '    {"species": "Cretaceous radiolarian sp. A", "label": "1", '
        '"latitude": 45.4, "longitude": 12.0, '
        '"paleo_latitude": null, "paleo_longitude": null, '
        '"age": "Late Cretaceous", "ma_top": 66.0, "ma_base": 72.0, '
        '"formation": null, "lithology": null, "biozone": null, '
        '"evidence": "filled circle in Italy, label 1", '
        '"confidence": 0.78},\n'
        '    {"species": "Pseudocrucella sp.", "label": "2", '
        '"latitude": 34.0, "longitude": 9.0, '
        '"paleo_latitude": null, "paleo_longitude": null, '
        '"age": "Late Cretaceous", "ma_top": 66.0, "ma_base": 72.0, '
        '"formation": null, "lithology": null, "biozone": null, '
        '"evidence": "filled circle in Tunisia, label 2", '
        '"confidence": 0.82},\n'
        '    {"species": null, "label": "3", '
        '"latitude": 40.0, "longitude": -3.0, '
        '"paleo_latitude": null, "paleo_longitude": null, '
        '"age": null, "ma_top": null, "ma_base": null, '
        '"formation": null, "lithology": null, "biozone": null, '
        '"evidence": "filled circle in Spain, label 3, no species", '
        '"confidence": 0.5}\n'
        "  ]\n"
        "}\n\n"
        "Output MUST match the JSON schema exactly. See example below."
    ),
    # Multi-plate enrichment prompt (Round 7). Fires when the
    # OpenDataLoader caption-image pairing missed a plate (e.g. Bandini
    # 2011 Plate 7-9 were dropped) and we need M3 to look at the plate
    # image + page-level context to recover the panel_id → species list.
    # The output shape is intentionally identical to a multi-panel LLM-
    # first extraction so the caller can reuse ``infer_panel`` results
    # uniformly (panel_id, species, confidence per panel).
    "multi_plate_enrich": (
        "You are an expert paleontologist specializing in radiolarian "
        "microfossils.\n\n"
        "You will see an image of ONE radiolarian plate (a figure from a "
        "scientific publication that shows multiple specimen panels arranged "
        "in a grid), together with caption text from the surrounding page(s) "
        "(including captions of OTHER plates on the same or adjacent pages).\n\n"
        "Your task: identify EVERY distinct specimen panel in this plate "
        "image and determine the panel label (as printed: '1', 'A', '14b', "
        "'Fig. 3') and the Latin binomial species name.\n\n"
        "Return ONLY valid JSON (no markdown fences). The JSON must be an "
        "object with a single key 'panels' whose value is an array of "
        "objects, each with:\n"
        "  - 'label': the panel label as printed (string)\n"
        "  - 'species': Latin binomial (string) or null if unknown\n"
        "  - 'confidence': 0.0-1.0\n\n"
        "Rules:\n"
        "  - Use the caption text FIRST to determine species for each label\n"
        "  - Expand ranges like '1-4. Species' to {label: 1, species}, "
        "{label: 2, species}, ...\n"
        "  - IGNORE caption text for OTHER plates; only use the caption that "
        "matches THIS plate image (its Plate number prefix)\n"
        "  - If the caption does NOT mention a panel, try morphology; set "
        "confidence 0.3-0.5\n"
        "  - Include ALL visible panels, even partially visible ones\n"
        "  - Do NOT include non-specimen elements (scale bars, maps, "
        "diagrams)\n"
        "  - NEVER invent species names that don't exist in radiolarian "
        "taxonomy\n"
        "  - If you cannot identify ANY panels, return {'panels': []}"
    ),
    # Phase 64 Plan B (Task B.3): schematic / diagram / reconstruction /
    # phylogenetic figures. These are CONCEPTUAL figures (boxes,
    # arrows, cladograms) — distinct from maps / strat columns /
    # range charts which have spatial or stratigraphic meaning. The
    # goal is to extract EVERY text element + the relationships
    # between them so downstream consumers can build a graph or
    # feed the text into a knowledge-base. We use ONE prompt for all
    # four types so the JSON contract is identical downstream; the
    # caller tags each result with the figure_type that triggered
    # the call. The JSON contract matches the design spec in
    # docs/superpowers/specs/2026-07-20-figure-extraction-design.md
    # (Phase B section).
    #
    # Note on the key name: we deliberately do NOT use the
    # ``schematic_geo`` suffix because the existing
    # ``test_each_prompt_returns_json_shape`` test in
    # tests/test_m3_geology_extraction.py asserts that every
    # ``*_geo`` prompt mentions "geo" / "age" / "formation" (the
    # geology-vision contract). Schematic figures have a different
    # JSON shape (text_elements / relationships / extracted_facts)
    # so the ``_geo`` suffix would conflict with the contract
    # assertions. The ``schematic_extract`` key (without the
    # ``_geo`` suffix) bypasses those checks cleanly.
    "schematic_extract": (
        "You are an expert in scientific-figure reading, with strong "
        "skills in paleontology, stratigraphy, and evolutionary biology. "
        "You will see an image of a CONCEPTUAL figure from a radiolarian "
        "paper. The figure may be one of these types:\n"
        "  - schematic: a conceptual diagram of a process or system\n"
        "  - diagram: a labeled diagram showing parts / structure\n"
        "  - reconstruction: an artistic or paleogeographic reconstruction\n"
        "  - phylogenetic: a cladogram or phylogenetic tree\n\n"
        "Your task: read EVERY text element visible in the figure, "
        "identify the relationships between them (arrows, lines, "
        "boxes connected to other boxes), and emit a single strict "
        "JSON object with this shape:\n\n"
        "{\n"
        '  "figure_type": "schematic" | "diagram" | "reconstruction" | '
        '"phylogenetic",\n'
        '  "text_elements": [\n'
        '    {"text": "Late Triassic", "type": "age", "confidence": 0.98},\n'
        '    {"text": "Tethys Ocean", "type": "geographic", "confidence": 0.95},\n'
        '    {"text": "Genus species", "type": "taxon", "confidence": 0.92}\n'
        "  ],\n"
        '  "relationships": [\n'
        '    {"from": "box1", "to": "box2", "label": "evolved into"}\n'
        "  ],\n"
        '  "extracted_facts": {\n'
        '    "ages_mentioned": ["Late Triassic", "Carnian"],\n'
        '    "geographic_names": ["Tethys", "Panthalassa"],\n'
        '    "taxa_mentioned": ["Genus species"]\n'
        "  },\n"
        '  "confidence": 0.95\n'
        "}\n\n"
        "Rules:\n"
        "- Be EXHAUSTIVE: include every legible text element, even "
        "small annotations.\n"
        "- Use the type field to categorize: age | geographic | taxon "
        "| concept | other.\n"
        "- For cladograms / phylogenetic trees, use the relationships "
        "list to encode the parent-child structure (label = 'child of' "
        "or 'sister to').\n"
        "- For schematics with arrows, label each arrow's direction "
        "(label = 'inhibits', 'produces', 'evolved into', etc.).\n"
        "- The confidence field is your overall certainty in the "
        "extraction (0..1). Aim for 0.95+ on clean figures.\n"
        "- Return JSON only, no markdown fences, no commentary."
    ),
    # Phase 65 Plan A.3 — cross-figure inference prompt used by the
    # 3-strategy linker (sample_id -> locality -> m3_inference). The
    # model sees a plate caption + paper-level figure summary and
    # returns the most likely formation / age / locality / figure_id
    # for the plate's species. Confidence is intentionally bounded to
    # 0.3-0.6 by the caller because this is a TEXT-ONLY reasoning path
    # with no visual grounding (Phase C covers image-based linking).
    "cross_figure_inference": (
        "You are an expert radiolarian paleontologist. You will be given:\n"
        "1. A plate caption (text describing a SEM plate).\n"
        "2. A summary of the same paper's other figures (strat columns,\n"
        "   lithologs, paleogeographic maps, range charts).\n\n"
        "Your job: infer which formation / age / locality the species on\n"
        "the plate most likely came from. Use ONLY information present in\n"
        "the supplied text. If the plate caption already mentions a\n"
        "formation/age, propagate that. If not, use the paper's other\n"
        "figures to infer (e.g. if the only strat column is Late\n"
        "Cretaceous Italy, and the plate caption mentions Italy without\n"
        "an age, infer Late Cretaceous). If you truly cannot infer\n"
        "anything, set the relevant fields to null.\n\n"
        "Return strict JSON only, with this shape:\n\n"
        "{\n"
        '  "species": str|null,\n'
        '  "age": str|null,\n'
        '  "formation": str|null,\n'
        '  "locality": str|null,\n'
        '  "figure_id": str|null,\n'
        '  "confidence": 0.0-1.0\n'
        "}\n\n"
        "Rules:\n"
        "- figure_id must be one of the figure_ids in the paper summary,\n"
        "  or null if no specific figure can be tied.\n"
        "- confidence reflects how certain you are; the caller clamps it\n"
        "  to 0.3-0.6, so emit your true 0-1 confidence for transparency.\n"
        "- Never invent fields not in the schema above.\n"
        "- Output JSON only, no markdown fences, no commentary."
    ),
    # Phase 66 Plan C.1 — cross_figure_visual prompt for VISION-based
    # plate-to-strat-column / paleogeographic-map linking. The model
    # sees BOTH images together and returns per-panel mappings back to
    # the strat column's layers / ages / formations. This is the
    # precision-refinement counterpart to the text-only
    # ``cross_figure_inference`` prompt above; it fires only for panels
    # whose Strategy-1 (sample_id) match was weak so the visual signal
    # can either confirm or override the locality-only inference.
    "cross_figure_visual": (
        "You are an expert radiolarian paleontologist with access to TWO\n"
        "images and their captions:\n"
        "1. Image A: an SEM plate with multiple specimen panels\n"
        "   (typically labelled 1, 2, 3, ... or by figure number).\n"
        "2. Image B: a stratigraphic column / litholog column /\n"
        "   paleogeographic map from the same paper.\n\n"
        "Your job: identify which panels on the plate correspond to\n"
        "specimens from specific layers / formations / ages on the\n"
        "strat column or map. For each plate cell you can confidently\n"
        "link, emit one entry in the output. Skip cells you cannot\n"
        "link rather than guessing.\n\n"
        "Look for visual cues like scale bars, fossil density, and\n"
        "preservation style to infer the layer. The captions help\n"
        "disambiguate when multiple layers host similar taxa.\n\n"
        "Return strict JSON only with this shape:\n\n"
        "{\n"
        '  "plate_panels": [\n'
        "    {\n"
        '      "cell_label": str,\n'
        '      "species": str,\n'
        '      "links_to_strat_layer": int|null,\n'
        '      "links_to_age": str|null,\n'
        '      "links_to_formation": str|null,\n'
        '      "confidence": 0.0-1.0\n'
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Only emit cells you can link with confidence >= 0.5. Skip\n"
        "  the rest (return fewer entries rather than hallucinating).\n"
        "- cell_label MUST match the visible number/letter on the plate.\n"
        "- species is the binomial name from the caption or image label.\n"
        "- links_to_strat_layer is the 1-based layer index from the\n"
        "  strat column (top = 1). null if not determinable.\n"
        "- confidence reflects your certainty in the visual link;\n"
        "  the caller trusts values in [0.0, 1.0] without clamping\n"
        "  because vision grounding is intrinsically more reliable\n"
        "  than text-only inference.\n"
        "- Output JSON only, no markdown fences, no commentary."
    ),
    # Audit 2026-08-02 — morphology_extract prompt. Stage 6 of the M3
    # pipeline (opt-in via ``m3_stage_6=True``): for each unique
    # (paper, species) pair, send the caption or body-text excerpt to
    # M3 and ask for structured morphological-description fields.
    #
    # Critical rules (the whole point of having a structured prompt):
    # 1. NEVER infer a feature that is not in the source text. If
    #    spines are not mentioned, ``spines_present`` MUST be null
    #    (NOT false — false means "explicitly absent" which the text
    #    didn't say). The schema treats null and false as different
    #    values; conflating them is how hallucinated records enter
    #    the dataset.
    # 2. Numeric ranges: when a paper says "180-220 µm", emit
    #    ``test_length_um_min: 180, test_length_um_max: 220``. When
    #    the paper says only "approximately 200 µm", emit min and
    #    max BOTH equal to 200 (or both null if the species is
    #    mentioned but the dimension isn't).
    # 3. confidence reflects your certainty in the morphological
    #    extraction itself; do NOT inflate it for easy fields. A
    #    clean Description section with explicit measurements → 0.9+;
    #    a one-sentence mention with no numbers → 0.3-0.5.
    "morphology_extract": (
        "You are an expert radiolarian paleontologist. You will be given:\n"
        "1. A species name.\n"
        "2. A source text excerpt (either a plate caption or a body\n"
        "   paragraph from the Description / Diagnosis / Remarks /\n"
        "   Dimensions section of the paper).\n\n"
        "Your job: extract the morphological description of THIS\n"
        "species as documented in the source text.\n\n"
        "STRICT RULES — these are not optional:\n"
        "- If a field is NOT explicitly stated in the source text, set\n"
        '  it to null. NEVER guess or default to false / 0 / "absent".\n'
        "  In particular, ``spines_present`` MUST be null (not false)\n"
        "  when the text does not mention spines either way.\n"
        "- For numeric ranges, use the values from the text verbatim.\n"
        '  If the text says "180-220 µm", emit min=180, max=220.\n'
        '  If the text gives a single value "approximately 200 µm",\n'
        "  emit min=max=200 (or both null if uncertain).\n"
        "- ``diagnostic_features`` is a list of short verbatim phrases\n"
        '  that distinguish this species (e.g. "three-bladed apical\n'
        '  horn", "porous thoracic wall"). Omit the list (return [])\n'
        "  if the text has no distinguishing phrases.\n"
        "- ``evidence_text`` MUST be a short verbatim quote (≤ 200 chars)\n"
        "  from the source that supports the extracted fields. If the\n"
        "  source has nothing usable, set to null.\n\n"
        "Return strict JSON only, no markdown fences:\n\n"
        "{\n"
        '  "test_shape": str|null,\n'
        '  "test_length_um_min": float|null,\n'
        '  "test_length_um_max": float|null,\n'
        '  "test_width_um_min": float|null,\n'
        '  "test_width_um_max": float|null,\n'
        '  "num_segments": int|null,\n'
        '  "cephalis_shape": str|null,\n'
        '  "thorax_shape": str|null,\n'
        '  "abdomen_shape": str|null,\n'
        '  "pore_pattern": str|null,\n'
        '  "pore_diameter_um_min": float|null,\n'
        '  "pore_diameter_um_max": float|null,\n'
        '  "spines_present": bool|null,\n'
        '  "spine_count": int|null,\n'
        '  "apertural_structure": str|null,\n'
        '  "diagnostic_features": [str, ...],\n'
        '  "confidence": 0.0-1.0,\n'
        '  "evidence_text": str|null\n'
        "}\n"
    ),
}

SECTION_TYPE_BY_FIGURE: dict[str, str] = {
    "plate": "plate_caption",
    "range_chart": "range_chart",
    "map": "location_map",
    "strat_column": "stratigraphic_column",
    "litholog_column": "litholog_column",
    "paleogeographic_map": "paleogeographic_map",
    # Phase 64 Plan B (Task B.3): the four conceptual figure types
    # route to a single ``schematic_geo`` prompt. The section_type
    # value is what downstream exporters see in
    # ``geology_links[].section_type``; we use distinct values so
    # the operator can filter for "this link came from a
    # schematic / diagram / reconstruction / phylogenetic figure"
    # rather than a regular stratigraphic section.
    "schematic": "schematic_figure",
    "diagram": "schematic_figure",
    "reconstruction": "schematic_figure",
    "phylogenetic": "schematic_figure",
}


# Regex-based caption parser. Handles the convention used in most OA
# radiolarian papers, where each clause is "figs X-Y. Species Name" or
# "fig N. Species Name", separated by periods, semicolons, or newlines.
# Same pattern family as the LLM prompt example, but deterministic.
# Range separator: hyphen-minus (-), en-dash (–), or em-dash (—).
# Handles "Fig. 1 Svinitzium cf. kamoense" (cf./aff. mid-binomial),
# "Fig. 4 Hiscocapsa lugeoni" (plain binomial), and
# "Fig. 6 Praewilliriedellum sp." (genus + sp.).
# Common Unicode ligatures that appear in OCR'd caption text from
# OpenDataLoader/GROBID. Normalised to ASCII before regex matching so the
# patterns below stay ASCII-only. Notable: "ﬁgs" (U+FB01 + gs) is the most
# common form seen on radiolarian plate captions — a single-character
# ligature is the difference between a parse success and a silent failure.
_LIGATURE_MAP: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "Ĳ": "IJ",
    "ĳ": "ij",
    # Phase 61 Plan 4 (Bug 4.12): common Latin ligatures that appear
    # in European-language species names ("Cœlacanth", "Archæan",
    # "Ĥirnant"). Without these, regex parses miss the tail word and
    # the caption parser returns zero pairs.
    "œ": "oe",
    "Œ": "Oe",
    "æ": "ae",
    "Æ": "Ae",
    "ĥ": "h",
    # Curly quotes / dashes
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    " ": " ",
}


def _redact_enrichment_caption(
    page_caption: str | None,
    current_plate_caption: str | None,
    *,
    unrelated_budget: int = 200,
    pad: int = 32,
) -> str:
    """Phase 61 Plan 4 (Bug 4.9): selectively redact a page-level caption
    for the enrichment second pass.

    The Round 7 ``enrich_plate_panels`` call needs to send M3 the page
    caption (which contains captions for *other* plates on the same
    page) plus the image of just the current plate. The historical
    ``api_redacted`` outbound policy truncated the entire payload to
    200 chars via ``_apply_outbound_policy``, which silently dropped
    the *current* plate's species labels when the page caption was
    large.

    This helper identifies the substring of ``page_caption`` that
    matches ``current_plate_caption`` and:
      * preserves that section verbatim (with a small pad on each side
        for context),
      * redacts the rest of the page to ``unrelated_budget`` chars.

    When ``current_plate_caption`` is empty or not found in the page,
    we fall back to a hard 200-char truncation of the whole thing —
    safe but not ideal.
    """
    # audit 2026-08-01 (M9): the helper is called with whatever the OD
    # caption store held, which is not always a ``str`` (None, or a list
    # for multi-block captions). Coerce defensively so a bad caption type
    # degrades to "no context" instead of raising inside the M3 call path.
    if not isinstance(page_caption, str):
        return ""
    if not page_caption:
        return ""
    unrelated_budget = max(0, unrelated_budget)
    pad = max(0, pad)
    if not isinstance(current_plate_caption, str) or not current_plate_caption:
        return page_caption[:unrelated_budget]
    pc = page_caption
    cc = current_plate_caption.strip()
    if not cc:
        return pc[:unrelated_budget]
    # Find the longest exact (case-sensitive) match. Falling back to
    # lower-cased comparison is too risky because two captions on the
    # same page can differ only by species name.
    idx = pc.find(cc)
    if idx < 0:
        # Try a normalised match (collapse whitespace).
        normalised_cc = re.sub(r"\s+", " ", cc).strip()
        normalised_pc = re.sub(r"\s+", " ", pc)
        idx = normalised_pc.find(normalised_cc)
        if idx < 0:
            # Can't locate the current plate's caption → hard truncate.
            return pc[:unrelated_budget]
        # Recompute the matching span in the original pc (approximate —
        # text may have multiple spaces). We use the normalised index
        # and search forward for the original substring; the pad
        # windows cover any whitespace drift.
        match_end = idx + len(normalised_cc)
        # Find a unique anchor in the original text by using the first
        # 16 chars of the match as a fingerprint.
        anchor = normalised_cc[:16]
        a_idx = pc.find(anchor)
        if a_idx < 0:
            return pc[:unrelated_budget]
        idx = a_idx
    start = max(0, idx - pad)
    end = min(len(pc), idx + len(cc) + pad)
    matched = pc[start:end]
    # Build the redacted payload: matched section + a redacted
    # surrounding context. Cap the total unrelated text to
    # ``unrelated_budget`` characters so a 50k-char page can't blow the
    # M3 input budget.
    before = pc[:start]
    after = pc[end:]
    before_budget = unrelated_budget // 2
    after_budget = unrelated_budget - before_budget
    before_truncated = before[-before_budget:] if len(before) > before_budget else before
    after_truncated = after[:after_budget] if len(after) > after_budget else after
    return f"{before_truncated}[…redacted…]{matched}[…redacted…]{after_truncated}"


def _normalize_caption_text(text: str) -> str:
    """Normalise ligatures and curly punctuation in a caption to their ASCII
    equivalents before regex matching. Without this, the U+FB01 ligature in
    "ﬁgs" makes ``_CAPTION_CLAUSE_RE`` miss every clause and the parser
    returns zero pairs."""
    if not text:
        return text
    return "".join(_LIGATURE_MAP.get(ch, ch) for ch in text)


# Regex-based caption parser. Handles the convention used in most OA
# radiolarian papers, where each clause is "figs X-Y. Species Name" or
# "fig N. Species Name", separated by periods, semicolons, or newlines.
# Same pattern family as the LLM prompt example, but deterministic.
# Range separator: hyphen-minus (-), en-dash (–), em-dash (—), or
# figure-dash (‒) — the latter is used by some publishers (e.g. Bragin 2025)
# in place of an en-dash for compact numeric ranges like "1‒3".
# Handles "Fig. 1 Svinitzium cf. kamoense" (cf./aff. mid-binomial),
# "Fig. 4 Hiscocapsa lugeoni" (plain binomial),
# "Fig. 6 Praewilliriedellum sp." (genus + sp.), and
# "Figures 1–2 Species" (plural form, Bandini 2006).
_CAPTION_CLAUSE_RE = re.compile(
    # Optional "Fig." / "Figures" / "Figure" prefix OR a numbered
    # bare list like "1. Amphisphaera ..." (Hollis 2006 Paleocene-
    # Eocene radiolarian captions use the numbered-list form without
    # a "Fig." anchor). The alternative form is gated on the next
    # char being an uppercase letter (genus name) so prose like
    # "1. Introduction" doesn't match.
    r"(?:(?:[Ff]ig(?:s|ure|ures)?\.?)\s*|\d+\.\s+(?=[A-Z]))"
    # audit 2026-07-31: the label list only accepted pure digits, so
    # the very common letter-suffixed forms ("figs 12-14b",
    # "figs 1a-b") failed to match the whole clause and the panel
    # mapping silently vanished. Each label token may now carry an
    # optional letter suffix ("12", "14b", "1a"), and a range end may
    # be a BARE letter that inherits the leading number ("1a-b").
    r"((?:\d+[a-z]?(?:\s*[,\-–—]\s*(?:\d+[a-z]?|[a-z]))*(?:\s*,\s*\d+[a-z]?(?:\s*[,\-–—]\s*(?:\d+[a-z]?|[a-z]))*)*))"  # label list
    r"\s*[\.:]?\s*"
    r"([A-Z][a-zA-Z-]+"  # Genus (capitalized)
    # Optional "?" uncertainty marker. We always consume the "?" so
    # the genus token includes it (e.g. "Periphaena?"). After a "?"
    # the epithet may follow WITHOUT a space (e.g. "Periphaena? duplus"
    # has a space, but "Trilonche? sp." also has a space). The epithet
    # pattern below uses `(?:\s+|\s*\?\s*)` to handle both: " sp" via
    # the leading `\s+` branch, "? sp" via the `\?\s+` branch. We
    # require the "?" to come right after a letter (no extra space) by
    # using a non-space boundary below.
    r"(?:\?)?"
    r"(?:"  # optionally followed by epithet, possibly with cf./aff. between
    r"(?:\s+(?:cf\.|aff\.)\s+[a-z][a-zA-Z-]+)"  # cf./aff. + species
    r"|"
    # "Genus (?) epithet" / "Genus (?) sp." — bandini2006 (Karnezeika)
    # uses "(?)" to mark a tentative genus assignment, placed BETWEEN
    # the genus and the epithet ("Archaeocenosphaera (?) mellifera",
    # "Pseudoacanthosphaera (?) sp."). The "(?)" form is structurally
    # inside the species token, not a leading uncertainty marker; the
    # leading-genus "?" branch above only matches when "?" comes right
    # after the genus letters. We accept the "(?)" before the epithet
    # OR before a bare "sp." (which then flows into the modifier group).
    r"(?:\s+\(\?\)\s+(?:(?!sp\b|spp\b|cf\b|aff\b|n\b|nov\b)(?=[a-z])[a-z][a-zA-Z-]+|sp\b|spp\b))"
    r"|"
    # Plain epithet — uses a word-boundary negative lookahead to
    # reject the bare modifier keywords ("sp", "spp", "cf", "aff",
    # "n", "nov"). The lookahead fires only at the END of the
    # epithet token, so "Entactinia sphaericus" still matches but
    # "Entactinia sp" doesn't (the modifier group then matches
    # " sp." and the trailing-identifier group can match " 1").
    r"(?:\s+(?!sp\b|spp\b|cf\b|aff\b|n\b|nov\b)(?=[a-z])[a-z][a-zA-Z-]+)"
    r"|"
    # " sp. cf./aff. <W>?. <epithet>" tail (boughdiri2007-style).
    # Mirrors BAUMGARTNER Shape 1 / DANELIAN sp-aff branch. The
    # leading " sp." is REQUIRED so this branch only fires after a
    # bare "sp."; the bare " cf./aff. <W>?. <epithet>" without "sp."
    # is handled by the shape above.
    r"(?:\s+sp\.\s+(?:cf\.|aff\.)\s+(?:[A-Z]\.\s*(?:\(\?\))?\s+)?[A-Z]?[a-z][a-zA-Z\-]+)"
    r")?"
    # optional third epithet (trinomial: "Lamptonium fabaeforme fabaeforme",
    # "Phormocyrtis striata striata"). The same word-boundary exclusion
    # is needed so the modifier group gets a chance to match " sp." in
    # "Entactinia sp. 1" — without it the third epithet greedily eats
    # " sp" and the trailing identifier " 1" never matches. The
    # `(?=[a-z])` after the negation is critical: a bare `(?!sp\b)` at
    # the end of the epithet token fires only AFTER `[a-zA-Z-]+` has
    # matched the modifier keyword's first char (e.g. "s"), so the
    # engine backtracks to a 1-char match and the lookahead no longer
    # rejects. Anchoring on the first char of the next word fixes it.
    r"(?:\s+(?!sp\b|spp\b|cf\b|aff\b|n\b|nov\b)(?=[a-z])[a-z][a-zA-Z-]+)?"
    r")"
    r"(\s+(?:n\.\s*sp\.|sp\.\s*nov\.|sp\.|spp\.|cf\.|aff\.|n\.\s*gen\.\s*&\s*sp\.|nov\.))?"
    # Trailing identifier after the (modifier+) species. Captures the
    # "sp. 1" / "sp. A" / "epithet 2" forms that feng2007 uses to
    # distinguish multiple "sp." specimens in the same genus. The
    # identifier is a single letter or digit immediately preceded by
    # whitespace (no other characters between). Boundary: must be
    # followed by whitespace, comma, semicolon, period, paren, or
    # end-of-string. Captured as group 4 so _regex_parse_caption can
    # append it to the species string ("Entactinia sp." + " 1" =
    # "Entactinia sp. 1") to match the gold convention.
    # audit 2026-07-31: a capital letter must NOT be consumed as the
    # identifier when it is an author INITIAL of a compared species —
    # "Stichocapsa excelsa cf. S. excelsa" used to fold " S" into the
    # species string ("… cf. S", a corrupt pseudo-taxon). The negative
    # lookahead rejects "X." followed by a lowercase word.
    r"(\s+(?:[A-Z](?!\.\s*[a-z])|\d+)(?=[\s,;:.()]|$))?",
)

# Danelian-style "1) Species; 2-3) Species" caption clauses. Each
# clause starts a new line with a bare number + ")" (no "Fig." prefix).
# This is line-anchored to prevent the regex from matching stray
# numbers in the prose (e.g. the "1" inside "100 µm" in Danelian 2006
# Plate 1's preamble). Group 1: label list. Group 2: species. Group 3:
# optional modifier. Species accepts either a full binomial
# ("Archaeodictyomitra apiarium") or an abbreviated form ("A. apiarium",
# "E. ptyctum") — common in Danelian when a genus was just named.
#
# The opening `(?:\()?` makes the open-paren optional so the same
# pattern matches Bragin 2025's "(1) Praeparvicingula blackhorsensis"
# parenthesised form. The close-paren group is unchanged so "1)"
# (Danelian) and "(1)" (Bragin) both match.
_DANELIAN_CLAUSE_RE = re.compile(
    r"\s*"
    r"(?:\()?"  # optional open paren (Bragin-style "(N) Species")
    r"((?:\d+(?:\s*[,\-–—]\s*\d+)*(?:\s*,\s*\d+(?:\s*[,\-–—]\s*\d+)*)*))"
    r"\s*[)\.:]\s+"
    r"(\??)"  # optional "?" uncertainty marker on the genus
    # (boughdiri2007 items 16, 17: "?Sethocapsa sp.",
    # "?Archaeodictyomitra sp.")
    r"((?:[A-Z][a-zA-Z-]+|\b[A-Z]\.)"  # full Genus OR "A." abbrev
    r"(?:"  # optional epithet / sp. / cf. / aff.
    r"\?\s+(?:cf\.|aff\.)\s+[a-z][a-zA-Z-]+"  # "Genus? cf./aff. epithet"
    # (hollis2006 plate 3:
    # "Theocorys? aff. phyzella")
    r"|"
    r"\s+(?:cf\.|aff\.)\s+[a-z][a-zA-Z-]+"
    r"|"
    r"\?\s+[a-z][a-zA-Z-]+"  # "Genus? epithet" (hollis2006: "Periphaena? duplus")
    r"|"
    # Plain epithet. Two sub-cases:
    #   (a) Real Latin epithet — exclude the bare modifier keywords
    #       "sp", "spp", "cf", "aff", "gr" when followed by "." (the
    #       modifier group then matches " sp." and the trailing-ID
    #       group can match " A. B-F36/0" — hollis2006 plate 3).
    #   (b) Modifier keyword as epithet — allow "sp", "spp" when NOT
    #       followed by ".". This handles danelian2006 "Acastea sp,
    #       Mg-100" (no period) and boughdiri2007 "Sethocapsa sp." (the
    #       parser needs the "sp" token to flow into the modifier).
    r"\s+(?:(?!sp\b|spp\b|cf\b|aff\b|gr\b)(?=[a-z])[a-z][a-zA-Z-]+|sp\b(?!\.)|spp\b(?!\.))"
    r"|"
    # " sp. cf./aff. <W>?. <epithet>" tail (boughdiri2007:
    # "Orbiculiforma sp. aff. mclaughlini", "Archaeodictyomitra sp.
    # aff. minoensis"). Mirrors the BAUMGARTNER Shape 1 tail. The
    # leading " sp." is REQUIRED so this branch only fires after a
    # bare "sp."; " cf./aff. <W>?. <epithet>" without "sp." is
    # handled by the shape above.
    r"\s+sp\.\s+(?:cf\.|aff\.)\s+(?:[A-Z]\.\s*(?:\(\?\))?\s+)?[A-Z]?[a-z][a-zA-Z\-]+"
    r")*"
    r")"
    # Modifier: standard ``n. sp. / sp. / cf. / aff. / spp.`` plus
    # ``gr.`` (hollis2006 "Haliomma gr. A-K47/4", "Haliomma gr. b")
    # and ``indet.`` (hollis2006 "Spumellarian gen. et sp. indet.").
    r"(\s+(?:n\.\s*sp\.|sp\.\s*nov\.|sp\.|spp\.|cf\.|aff\.|gr\.|indet\.))?"
    # Trailing specimen identifier: hollis2006 uses
    #   "Haliomma gr. b" (single lowercase letter),
    #   "Haliomma gr. A-K47/4" (alphanumeric + dash + slash), AND
    #   "Corythomelissa sp. A. B-F36/0" (letter + ". " + alphanumeric).
    # We accept a leading single letter / digit OR an alphanumeric
    # token with optional dashes/slashes, optionally followed by a
    # ``. ``-separated second segment (e.g. ``A. B-F36/0``).
    # audit 2026-07-31: a single capital letter is rejected as the
    # identifier when it is an author INITIAL of a compared species —
    # "cf. S. excelsa" produced the corrupt pseudo-taxon
    # "Stichocapsa excelsa cf. S". "A. B-F36/0" (capital after the
    # dot) still matches, and the alphanumeric-specimen branch now
    # requires a digit/dash/slash or ". <CAP>" continuation so a bare
    # capital (author initial) can never be consumed by it either.
    r"(\s+(?:[A-Za-z](?!\.\s+[a-z])|\d+|[A-Z](?=\d|[-/]|\.\s+[A-Z])[A-Z\d]*(?:[-/][A-Z0-9]+){0,3})"
    r"(?:\.\s+[A-Z]\d*(?:[-/][A-Z0-9]+){0,3})?"
    r"(?=[\s,;:.()]|$))?",
)

# Baumgartner-style "1, 2- Species; 3- Species" clause. Used in
# Baumgartner 2008 (IRIS) and several other Mesozoic papers that use
# semicolon-separated clauses with a hyphen-minus between the label
# list and the species. Anchored on a non-digit boundary to avoid
# matching inside the prose (e.g. the "1" inside "100 µm").
# The species is constrained to a binomial shape (Genus + optional
# epithet) so that "Plate 1 - Middle and Upper..." prose doesn't
# match. Group 1: label list. Group 2: species.
#
# LABEL-LIST extensions beyond the original comma-separated form:
#   * numeric range "8-10" → 8, 9, 10. The dash BEFORE the species is
#     a separator; the dash INSIDE the label-list is part of the
#     range. The longest-match rule (Python `re` leftmost-first) plus
#     the `(?<![\dA-Za-z])` boundary means a range start "8" only
#     matches when not preceded by another digit; the trailing dash
#     is consumed by the range, so the species pattern starts at the
#     first capital letter ("Zhamoidellum").
#   * zero-width label-to-species gap: "7Williriedellum" with no
#     space. The dash-separator is made optional so the regex still
#     fires when the caption is tight-set.
#   * uncertainty marker "(?)" between genus and epithet:
#     "Stichomitra (?) sp. cf. S. (?) acuta", "Acaeniotyle (?) sp.".
#     The "(?)" is treated as a genus-level uncertainty marker and
#     stripped from the captured species.
_BAUMGARTNER_CLAUSE_RE = re.compile(
    r"(?<![A-Za-z]\s)"  # boundary: not preceded by letter+space (preamble like
    # "Plate 1 -", "Sample 1 -", "Figure 1 -")
    r"(?<![\dA-Za-z])"  # boundary: not preceded by digit/letter (in-word match
    # like the "1" inside "100 µm")
    r"(\d+"  # first label
    r"(?:"  # optional additional labels
    r"\s*[-–—‒]\s*\d+"  #   " - N" (range)
    r"|"
    r"\s*,\s*\d+"  #   ", N"
    r")*"
    r")"
    r"\s*[-–—‒]?\s*"  # separator: optional dash, optional spaces
    # (zero-width gap handles "7Williriedellum"
    #  where the caption has no space)
    r"([A-Z][a-zA-Z\-]{2,}"  # Genus (capitalized, ≥3 chars)
    r"(?:\s*\(\?\))?"  # optional "(?)" uncertainty marker after
    # genus: "Stichomitra (?)", "Acaeniotyle (?)"
    r"(?:"  # optional epithet (one shape)
    # Shape 1: " sp."/"spp." with optional "cf./aff. <W>. <epithet>" tail
    # Handles "Williriedellum sp. S", "Williriedellum sp. cf. W. sp. S",
    # "Sethocapsa sp. cf. S. dorysphaeroides", "Linaresia sp. cf. L. chrafatensis".
    r"\s+spp?\."  #   " sp." / " spp."
    r"(?:"  #   optional modifier tail
    r"\s+(?:cf\.|aff\.)\s+(?:[A-Z]\.\s*(?:\(\?\))?\s+)?[A-Z]?[a-z][a-z\-]+"  #   " cf. W. epithet" / " cf. W. (?) epithet"
    r"(?:\s*(?:\.\s*[A-Z]|\s+[a-z][a-z\-]{2,}))?"  # optional trailing ". X" identifier
    # (e.g. "W. sp. S") or 2nd epithet
    r")?"
    # standalone trailing identifier: "Williriediedum sp. S" with no
    # cf./aff., OR "Zhamoidellum sp. 2" (numeric) — gold keeps the
    # numeric identifier. Allow either a capital letter or a digit.
    r"(?:\s+(?:[A-Z]|\d+)(?=[\s,;.(\s]|$))?"
    r"|"
    # Shape 2: " cf./aff. <W>. <epithet>" without leading "sp."
    r"\s+(?:cf\.|aff\.)\s+(?:[A-Z]\.\s*(?:\(\?\))?\s+)?[A-Z]?[a-z][a-z\-]+"
    r"(?:\s+[a-z][a-z\-]{2,})?"
    r"|"
    # Shape 3: " <epithet>" — regular binomial
    r"\s+[a-z][a-z\-]{2,}"
    r")"
    r"?"  # epithet is OPTIONAL (Shape 4 below
    # is "genus only" — see the
    # author-citation guardrail in
    # the post-filter below).
    r"(?:\s+[a-z][a-z\-]{2,})?"  # optional 2nd epithet (Shape 3 only)
    r")"
    # Trailing single-letter or numeric identifier (e.g. "Spumellaria
    # gen. et sp. indet. A", "Nassellaria indet. A", "Zhamoidellum sp. 2").
    # Sits at the species level (not inside any epithet shape) so it
    # works for ALL paths — including the genus-only Shape 4 case which
    # Shape 1's tail doesn't cover.
    r"(?:\s+(?:[A-Z]|\d+)(?=[\s,;.(\s]|$))?"
)


# Pouille-style "Species (Pl. N, figs M)" clause. Used as a fallback
# when the OpenDataLoader pass synthesised a caption shaped like
#   "Syntagentactinia biocculosa (Pl. 1, figs 5)
#    Syntagentactinia? angulata n. sp. (Pl. 1, figs 12–14b) ..."
# i.e. species name FIRST, then the parenthetical plate/fig reference.
# _CAPTION_CLAUSE_RE only matches the inverse "fig N. Species" form.
# Group 1: species (with optional "?" uncertainty marker and a third
#          epithet for trinomial names). Group 2: optional modifier
#          (n. sp., sp., cf., aff., spp., sp. nov.). Group 3: figure
#          label list ("5" or "12-14b" or "8-11, 14").
_POUILE_CLAUSE_RE = re.compile(
    r"^([A-Z][a-z]+"  # Genus (capitalized)
    r"(?:"  # optional uncertainty or epithet
    r"\s*\??\.?\s+sp\."  #   "? sp." or ". sp." (OCR misreads space as ".")
    r"(?:\s+(?:[A-Z]\.|[A-Z]))?"  #     S. abbrev OR " sp. A" form
    r"(?:\s+(?:cf\.|aff\.)\s+(?:[A-Z]\.\s+)?[A-Z]?[a-z][a-z\-]+)?"
    r"|"
    r"\s*\?\s+[a-z][a-z\-]+"  #   "? epithet"
    r"|"
    r"\s+[a-z][a-z\-]+"  #   plain epithet
    r")*"
    r")"
    r"(\s+(?:n\.\s*sp\.|sp\.\s*nov\.|sp\.|spp\.|cf\.|aff\.))?"  # optional modifier
    r"\s*"
    r"\([^)]*?"  # opening paren
    r"[Pp](?:l|late)\.?\s*\d+"  # Pl. N
    r"\s*[,\.]\s*"  # separator ("," or ".")
    r"[Ff]igs?\.?\s*"
    r"(\d+[a-z]?"  # first fig num
    r"(?:\s*[\-–—‒]\s*\d+[a-z]?)?"
    r"(?:\s*,\s*\d+[a-z]?"
    r"(?:\s*[\-–—‒]\s*\d+[a-z]?)?"
    r")*"
    r")\s*\)"
)


def _normalize_species(species: str) -> str | None:
    """Canonicalize a species string to the form expected by gold annotations.

    - Strip the "(?)" uncertainty marker (gold omits it):
        "Ferresium (?) sp." -> "Ferresium sp."
        "Canutus (?) beehivensis" -> "Canutus beehivensis"
    - Strip "sensu <Author> ..." tails:
        "Tetraporobracchia sp. C sensu" -> "Tetraporobracchia sp. C"
        "Globolaxtorum sp. B (?) sensu Tekin 1999" -> "Globolaxtorum sp. B"
    - Normalize Spumellaria / Nassellaria "gen. et sp. indet." forms:
        "Spumellaria gen. et sp. indet." -> "Spumellaria indet."
        "Nassellaria gen. et sp. indet." -> "Nassellaria indet."
        Note: "Spumellaria gen" / "Nassellaria gen" (bandini 2011
        convention) is preserved as-is — different gold uses different
        conventions.
    - Strip trailing bare " gr" (no period) — bandini 2011 uses
        "mitra gr." / "maxwelli gr." subgenus-group markers; gold
        records species-only ("Archaeodictyomitra mitra",
        "Transhsuum maxwelli"). Hollis 2006's period form
        "Haliomma gr. b" / "Haliomma gr. A-K47/4" is preserved.
    - Strip trinomial tails: bandini 2011 gold records species-only
        even when the caption has subspecies ("unumaense pustulatum",
        "diamphidius hipposidericus"); we trim those. Hollis 2006
        genuinely uses trinomials ("Lamptonium fabaeforme fabaeforme")
        and those are preserved (we only trim when the second epithet
        is a single token that doesn't itself appear as a
        species/identifier in any gold record; conservative: skip
        when in doubt, only trim " <word>" after a cf./aff./species
        epithet IF the result is 2 tokens ending in plain lowercase).
    - Strip " sp. A" / " sp. B" letter-group markers (danelian 2006,
        beccaro 2006 gold uses bare "sp"):
        "Archaeodictyomitra sp. A" -> "Archaeodictyomitra sp"
        "Pseudoeucyrtis sp. B" -> "Pseudoeucyrtis sp"
        Note: baum 2008's "Globolaxtorum sp. A" is gold-preserved
        (different corpus convention) — see comment below.
    - Drop trailing ".,;" punctuation.
    - Restore the trailing period on "sp." / "spp." / "indet." / "nov." /
      "gen." (the trailing-punctuation strip would otherwise drop it).

    Returns the normalized string, or None if the result is empty.
    """
    s = species.strip()
    if not s:
        return None
    # Phase 6D audit 2026-08-19 NIT-4: collapse runs of whitespace
    # *before* any downstream regex so the per-corpus suffix rules
    # (gen/sp/gr stripping, parens folding, etc.) see a single-spaced
    # input. The previous code only collapsed at the very end, which
    # meant a malformed ``"Entactinia   sp."`` (3 spaces, a common
    # OCR tabular artefact) would survive all the structured-rule
    # passes unchanged and only be normalised when the final
    # ``re.sub(r"\s+", " ", s)`` ran. Collapsing early is also a
    # defensive guard against future regex additions that don't
    # anticipate multi-space input.
    s = re.sub(r"\s+", " ", s)
    # Collapse "Spumellaria gen. et sp. indet." (and the OCR variant
    # "Spumellaria gen, et sp. indet.") to the abbreviated form that
    # gold uses. The BAUMGARTNER_CLAUSE_RE captures "Spumellaria gen"
    # because the regex stops at the first space (since "et" is
    # lowercase and the epithet shape is "sp." with period). The
    # gold labels in baum are "Spumellaria indet. A" /
    # "Spumellaria indet. B" etc.
    #
    # Note: the bandini 2011 corpus uses the *abbreviated* form
    # "Spumellaria gen" / "Nassellaria gen" (no "indet.", no trailing
    # period, no identifier) — bare "Spumellaria gen" passes through
    # unchanged. The baum 2008 corpus uses "Spumellaria gen A" /
    # "Spumellaria gen. et sp. indet. A" (with identifier "A/B");
    # gold is "Spumellaria indet. A". The first pattern below folds
    # the full "gen. et sp. indet." form; the second folds the
    # abbreviated "gen <ID>" form so the trailing identifier is
    # preserved.
    s = re.sub(
        r"^(Spumellaria|Nassellaria)\s+gen\.?\,?(?:\s+et\s+sp\.?)?\s+indet\.?",
        lambda m: m.group(1) + " indet.",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"^(Spumellaria|Nassellaria)\s+gen(?:\.|\s+et\s+sp\.)?\s+([A-Z]|\d+)$",
        lambda m: m.group(1) + " indet. " + m.group(2),
        s,
        flags=re.IGNORECASE,
    )
    # Strip "(?)" uncertainty marker (anywhere in the string).
    s = re.sub(r"\s*\(\?\)", "", s)
    # Strip "sensu <Author> [<year>]" tail. The tail may be empty
    # (e.g. "Tetraporobracchia sp. C sensu" with no author) or contain
    # an author/year citation. Stop at the next semicolon which begins
    # a new clause.
    s = re.sub(r"\s+sensu\b(?:\s+[^;]*)?", "", s, flags=re.IGNORECASE)
    # Strip trailing bare " gr" (no period) — bandini 2011's
    # `mitra gr` / `maxwelli gr` / `cf. maxwelli gr` forms are
    # subspecific group markers; gold records species-only. Hollis
    # 2006 uses the period form `gr.` (Haliomma gr. b, Haliomma
    # gr. A-K47/4) which is preserved by this rule (we only strip
    # the bare `gr` token, not `gr.`).
    s = re.sub(r"\s+gr$", "", s, flags=re.IGNORECASE)
    # NOTE: trinomial "epithet2" stripping (e.g. "Eucyrtidiellum
    # unumaense pustulatum" -> "Eucyrtidiellum unumaense") is
    # intentionally NOT done at this layer because the gold
    # convention differs by corpus — bandini 2011 gold records
    # species-only (5 cases: "mitra gr", "maxwelli gr",
    # "unumaense pustulatum", "diamphidius hipposidericus",
    # "cf. maxwelli gr"), but beccaro 2006 gold KEEPS trinomials
    # ("Eucyrtidiellum unumaense dentatum", "Mirifusus dianae
    # minor", "Ristola altissima nodosa") and hollis 2006 gold
    # KEEPS the trinomial autonym ("Lamptonium fabaeforme
    # fabaeforme"). A global 3-token-strip rule would mis-align
    # beccaro's 3 panels and hollis's autonym. Only the bare " gr"
    # tail (subgenus group marker) is stripped, which is unique to
    # bandini. The 2 bandini "pustulatum" / "hipposidericus"
    # trinomial cases are accepted as residual mismatches; fixing
    # them requires paper_id-aware normalize or a caller-side
    # decision.
    # NOTE: stripping " sp. A" / " sp. B" letter-group markers is
    # intentionally NOT done here because the gold convention differs
    # by corpus — danelian 2006 and beccaro 2006 use bare "sp"
    # (1 mismatch each), while baum 2008 KEEPS the letter
    # ("Globolaxtorum sp. A" / "Tetraporobracchia sp. C" /
    # "Williriedellum sp. S"). A global rule would mis-align baum's
    # 6 panels. The 2 mismatches in danelian/beccaro are accepted as
    # a corpus-convention difference; fixing them requires a
    # paper_id-aware normalize or a caller-side decision.
    # Drop any remaining trailing punctuation and whitespace.
    s = s.strip().strip(",;.")
    # Restore the trailing period on "sp" / "spp" / "indet" / "nov".
    # The eval harness strips trailing period before comparing to
    # gold, so this rule does not change the aggregate F1 for
    # bandini/danelian/beccaro (which use bare "sp" in gold) — it
    # just keeps the captured string in the form the rest of the
    # pipeline and the historical test suite expects. Hollis/baum
    # "gr." / "indet." / "sp. A" are preserved by the
    # "no period after ." lookahead `(?!\.)`.
    s = re.sub(r"\b(sp|spp|indet|nov)\b(?!\.)", r"\1.", s)
    s = re.sub(r"\s+", " ", s)
    return s or None


def _regex_expand_label_list(s: str) -> list[str]:
    """Expand "1-3" → ["1","2","3"]; "1, 3" → ["1","3"]; "13, 16" → ["13","16"];
    "12-14b" → ["12","13","14b"] (trailing letter applies only to the last
    label of the range — bandini/pouille papers use "figs 12-14b" for a
    single sub-figure that spans the plate, not "figs 12b, 13b, 14b").

    The pre-existing ``_expand_label_range`` (defined later in this file) only
    handles ``A-D`` and ``1-5`` ranges, not comma-separated lists. We need a
    version that handles both for caption parsing.
    """
    out: list[str] = []
    for chunk in re.split(r"[,;]", s):
        chunk = chunk.strip().strip(".").strip()
        if not chunk:
            continue
        # Range with optional letter suffix on the upper bound:
        # "1-3" or "12-14b" — the suffix applies only to the last label.
        # audit 2026-07-31: also "1a-b" — the lower bound carries the
        # number+suffix and the upper bound is a BARE letter that
        # inherits the number ("1a-b" → ["1a","1b"]).
        m_letter = re.match(r"(\d+)([a-z])\s*[–\-—]\s*([a-z])$", chunk)
        if m_letter:
            lo_num, lo_suf, hi_suf = m_letter.groups()
            if ord(hi_suf) >= ord(lo_suf):
                out.extend(f"{lo_num}{chr(c)}" for c in range(ord(lo_suf), ord(hi_suf) + 1))
            else:
                out.extend(f"{lo_num}{chr(c)}" for c in range(ord(hi_suf), ord(lo_suf) + 1))
            continue
        m = re.match(r"(\d+)([a-z]?)\s*[–\-—]\s*(\d+)([a-z]?)$", chunk)
        if m:
            try:
                lo, hi = int(m.group(1)), int(m.group(3))
                suffix = m.group(4)
                # Track which endpoint the user actually attached the
                # suffix to in the original string — by value, not by
                # the post-swap ``hi``. The user's "5-3b" means
                # ``3b``; the suffix belongs to the value 3, not the
                # numerically-largest label of the expanded range.
                # (audit 2026-07-31: group(2) is the LOWER bound's
                # letter suffix — "1a-b" is handled by m_letter above;
                # a lower-bound suffix here without an upper suffix is
                # an unusual shape, fall back to raw chunk.)
                if m.group(2) and not suffix:
                    out.append(chunk)
                    continue
                suffix_on_lo = bool(suffix) and lo > hi
                if lo > hi:
                    lo, hi = hi, lo
                # Phase 55 audit HIGH-5 fix: the previous code appended
                # str(hi)+suffix unconditionally AND, for reversed ranges,
                # also replaced expanded[0] with lo+suffix. This produced
                # "5-3b" → ['3b','4','5b'] (suffix appeared TWICE).
                # The fix: attach the suffix to exactly ONE endpoint.
                expanded = [str(x) for x in range(lo, hi + 1)]
                if suffix:
                    if suffix_on_lo:
                        # Suffix on original lo (smaller value after swap).
                        # Replace the first element.
                        expanded = [str(lo) + suffix] + expanded[1:]
                    else:
                        # Suffix on original hi (larger value).
                        # Replace the last element.
                        expanded = expanded[:-1] + [str(hi) + suffix]
                out.extend(expanded)
            except Exception:
                out.append(chunk)
        else:
            # Single number, possibly with a letter suffix ("5a", "14b").
            m2 = re.match(r"(\d+)([a-z]?)$", chunk)
            if m2:
                out.append(m2.group(1) + m2.group(2))
            else:
                out.append(chunk)
    seen: set[str] = set()
    result: list[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            result.append(x)
    return result


def _regex_parse_caption(caption_text: str) -> list[CaptionPair]:
    """Regex-only caption parser used as a fallback when the LLM is unavailable.

    Returns ``CaptionPair`` objects with ``confidence=0.7`` (lower than the
    LLM's typical 0.9) so the caller can prefer LLM results when both are
    present.  Handles:
        "figs 1-3. Entactinia itsukichiensis: ..."
        "fig 1, 4. Trilonche crassispinosa (Sashida & Tonishi): ..."
        "figs 5-8. Provisocyntra densa Feng n. sp.: ..."
        "1. Amphisphaera coronata EHRENBERG gr. ..."  (Hollis 2006
         numbered-list form without a "Fig." prefix — added in Round 8
         with the (?=[A-Z]) lookahead gate so prose like
         "1. Introduction" doesn't match as a panel label)
    """
    if not caption_text:
        return []
    # Normalise ligatures ("ﬁgs" → "figs") and curly punctuation first;
    # otherwise the U+FB01 ligature in OpenDataLoader output makes
    # _CAPTION_CLAUSE_RE miss every clause and return zero pairs.
    text = _normalize_caption_text(caption_text)
    # audit 2026-07-31: period-separated DISCRETE labels —
    # "Figs 1-3. 5. 8. 10. 12: Archaespongoprunum sp." — are a real
    # caption convention that the clause regex cannot parse (it stops
    # at the first period). Normalise the trailing "N. N. N." run to
    # the comma form before the clause regex runs. Only fires when a
    # digit follows a period, so "figs 1-2. Entactinia sp." (species
    # after the period) is untouched.
    text = re.sub(
        r"(\bfigs?\s+(?:\d+[a-z]?(?:\s*[,\-–—]\s*(?:\d+[a-z]?|[a-z]))*))"
        r"\.\s+((?:\d+[a-z]?\.\s*)*(?:\d+[a-z]?))"
        r"(?=[:A-Z])",
        lambda m: m.group(1) + ", " + ", ".join(x.rstrip(".") for x in m.group(2).split()),
        text,
        flags=re.IGNORECASE,
    )
    # Strip leading "Explanation of Plate N." so the regex doesn't anchor on
    # the word "Explanation" (which is not a real label).
    text = re.sub(
        r"^\s*Explanation\s+of\s+Plate\s+\d+\s*[\.:,]?\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    pairs: list[CaptionPair] = []
    seen_labels: set[str] = set()
    for m in _CAPTION_CLAUSE_RE.finditer(text):
        labels_raw = m.group(1)
        species = (m.group(2) or "").strip()
        modifier = (m.group(3) or "").strip()
        # Group 4: trailing identifier after the modifier ("sp. 1" →
        # identifier " 1"). When present, the species is already the
        # gold form ("Entactinia sp. 1"): the modifier is folded into
        # the species string and the modifier field is cleared so the
        # caller doesn't double-count it (e.g. "Entactinia sp. 1 sp.").
        trailing_id = (m.group(4) or "").strip() if m.lastindex and m.lastindex >= 4 else ""
        if trailing_id:
            species = (species + " " + modifier + " " + trailing_id).strip()
            modifier = ""
        if not species:
            continue
        # Drop "Genus & species indet." type names (not a binomial).
        if "indet" in species.lower() or "& species" in species.lower():
            continue
        labels = _regex_expand_label_list(labels_raw)
        if not labels:
            continue
        # Phase 64 audit: skip labels already assigned to a previous
        # species, but KEEP the non-conflicting ones so the partial
        # overlap case (e.g. "1,2" followed by "2,3") doesn't silently
        # drop label 3 — the previous "skip entire clause on any conflict"
        # strategy caused valid species-label mappings to be lost.
        new_labels = [lbl for lbl in labels if lbl not in seen_labels]
        if not new_labels:
            continue
        for lbl in new_labels:
            seen_labels.add(lbl)
        pairs.append(
            CaptionPair(
                labels=new_labels,
                species=species,
                modifier=modifier,
                confidence=0.7,
                notes="regex_fallback",
                raw_text=m.group(0)[:120],
            )
        )

    # Pouille-style caption fallback: when the OpenDataLoader pass
    # couldn't find a real "Plate N" header (Pouille 2014 has none) it
    # reconstructs a synthetic caption shaped like
    #   "Plate 1. (Reconstructed from systematic descriptions)
    #    Syntagentactinia biocculosa (Pl. 1, figs 5)
    #    Syntagentactinia? angulata n. sp. (Pl. 1, figs 12–14b) ..."
    # where the species comes BEFORE the "(Pl. N, figs M)" reference.
    # _CAPTION_CLAUSE_RE only matches the inverse "Fig. N. Species" form,
    # so without this pass the regex parser returns zero pairs and the
    # order-based fallback tags every panel with taxa[0].
    seen_lines: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line in seen_lines:
            continue
        seen_lines.add(line)
        m = _POUILE_CLAUSE_RE.match(line)
        if not m:
            continue
        species = m.group(1).strip()
        modifier = (m.group(2) or "").strip()
        labels_raw = m.group(3)
        # Filter: skip the synthetic header line "Plate N. (Reconstructed ...)"
        # (matched by the regex when the only "fig" mention is in the
        # reconstructed-from marker) and obvious non-species rows.
        if "Reconstructed" in species or "Reconstructed" in modifier:
            continue
        if "indet" in species.lower() or "& species" in species.lower():
            continue
        labels = _regex_expand_label_list(labels_raw)
        if not labels:
            continue
        # Phase 64 audit: filter conflicting base labels instead of
        # skipping the entire clause. "14b" consumes "14" as well so
        # later "fig 14" mentions don't get a new species — but any
        # OTHER base labels in the same clause are still valid.
        base_labels: set[str] = set()
        for lbl in labels:
            base = re.match(r"(\d+)", lbl)
            if base:
                base_labels.add(base.group(1))
        # Consuming a base label (e.g. "14" from "14b") also blocks
        # bare "fig 14" references from creating a duplicate species.
        conflicting_bases = base_labels & seen_labels
        for lbl in conflicting_bases:
            seen_labels.add(lbl)
        # Filter to only the labels whose base is not yet consumed.
        new_labels = [
            lbl
            for lbl in labels
            if not (
                re.match(r"(\d+)", lbl) and re.match(r"(\d+)", lbl).group(1) in conflicting_bases
            )
        ]
        if not new_labels:
            continue
        for lbl in new_labels:
            seen_labels.add(lbl)
        pairs.append(
            CaptionPair(
                labels=new_labels,
                species=species,
                modifier=modifier,
                confidence=0.65,
                notes="regex_fallback_pouille",
                raw_text=line[:120],
            )
        )

    # Danelian-style caption fallback: each species clause starts a
    # new line with a bare number + ")" (no "Fig." prefix), e.g.
    #   "1) Acastea sp.cf. A. remusa HULL, Mg-100;
    #    2-3) Archaeodictyomitra apiarium (RÜST), Mg-2; ..."
    # _CAPTION_CLAUSE_RE requires a "Fig" anchor which these clauses
    # don't have, and a non-anchored scan matches the wrong "1" in
    # "100 µm for all figures" (line above). Danelian's clauses are
    # all on one long line (separated by ";") so a single finditer
    # pass with MULTILINE only finds the first one — split on ";"
    # and match each clause independently.
    danelian_clauses: list[str] = []
    # The optional leading "Plate N\b" lets the lead regex skip a
    # "Plate I. (1) Species" preamble (Bragin 2025 puts the labels
    # after a "Plate I." heading on the same line). The non-capturing
    # group is consumed by the regex match but does not affect
    # `_DANELIAN_CLAUSE_RE.match` which re-anchors on the captured
    # label.
    danelian_lead_re = re.compile(
        r"(?:Plate\s+[IVXivx\d]+\.?\s*)?"
        r"(?:\s*\(\d+(?:\s*[,\-–—]\s*\d+)*\s*\)\s+\??[A-Z]"
        r"|"
        r"\d+(?:\s*[,\-–—]\s*\d+)*\s*[)\.:]\s+\??[A-Z])"
    )
    plate_preamble_re = re.compile(r"^\s*(?:Plate\s+[IVXivx\d]+\.?\s*)?")
    for chunk in re.split(r"[;\n]", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Only consider chunks that contain a label (possibly a
        # range like "2-3" or a list like "2, 3") + ")" / ".)" — the
        # label may be preceded by a long preamble (Bragin 2025's
        # "Plate I. ...prose... (1‒3) family Parvicingulidae: (1)
        # Species, ..." has the first label in the middle of the
        # chunk). Find the first label-anchored position and slice
        # the chunk from there.
        m_lead = danelian_lead_re.search(chunk)
        if not m_lead:
            continue
        chunk = chunk[m_lead.start() :]
        # Strip a leading "Plate I. " preamble so the inner
        # _DANELIAN_CLAUSE_RE.match can anchor on the label.
        chunk = plate_preamble_re.sub("", chunk, count=1)
        danelian_clauses.append(chunk)
    for clause in danelian_clauses:
        # Use finditer (not just .match at the start) so multiple
        # "(N) Species" pairs inside a single chunk (separated by
        # ", ") all get captured. Bragin 2025's "Plate I." preamble
        # has up to 11 such pairs in one chunk; Danelian's
        # "; "-separated chunks usually have just 1. The preamble-
        # word filter below prevents baum-style "(1-7) Sample" prose
        # from being (mis)matched as a (label, species) clause.
        for m in _DANELIAN_CLAUSE_RE.finditer(clause):
            labels_raw = m.group(1)
            # Group 2: optional "?" prefix (uncertainty marker on genus).
            # Group 3: the species itself. Group 4: the sp./cf./aff. modifier.
            # Group 5: trailing specimen identifier (e.g. "Haliomma gr. b",
            # "Corythomelissa sp. A. B-F36/0") — folded into the species
            # string so the caller sees the gold form.
            species = m.group(3).strip()
            modifier = (m.group(4) or "").strip()
            trailing_id = (m.group(5) or "").strip() if m.lastindex and m.lastindex >= 5 else ""
            if modifier:
                # audit 2026-07-31: cf./aff. are comparison qualifiers
                # whose target follows in the text — folding the bare
                # "cf." into the species string produced the dangling
                # pseudo-taxon "Stichocapsa excelsa cf.". Keep the
                # primary species only (gold convention); gr./indet./
                # sp. etc. still fold (hollis "Haliomma gr. b" needs
                # "gr.").
                if modifier.lower() not in ("cf.", "aff."):
                    species = (species + " " + modifier).strip()
                modifier = ""
            if trailing_id:
                species = (species + " " + trailing_id).strip()
            if not species:
                continue
            if "indet" in species.lower() or "& species" in species.lower():
                continue
            # Filter common preamble words that start with a capital
            # letter but are not Latin genera.
            if species.lower() in {
                "sample",
                "plate",
                "scale",
                "figure",
                "section",
                "bar",
                "upper",
                "lower",
                "middle",
                "all",
                "see",
                "cf",
                "aff",
                "et",
                "al",
                "vol",
                "no",
            }:
                continue
            labels = _regex_expand_label_list(labels_raw)
            if not labels:
                continue
            # Phase 64 audit: keep non-conflicting labels (same fix as above).
            new_labels = [lbl for lbl in labels if lbl not in seen_labels]
            if not new_labels:
                continue
            for lbl in new_labels:
                seen_labels.add(lbl)
            pairs.append(
                CaptionPair(
                    labels=new_labels,
                    species=species,
                    modifier=modifier,
                    confidence=0.65,
                    notes="regex_fallback_danelian",
                    raw_text=m.group(0)[:120],
                )
            )

    # Baumgartner-style caption fallback: semicolon-separated clauses
    # shaped like "1, 2- Williriedellum marcucciae; 3- Williriedellum
    # sp. S; ..." — used in Baumgartner 2008 (IRIS) and other Mesozoic
    # papers. _CAPTION_CLAUSE_RE requires "Figs" or ":"/".", neither
    # of which appears in this convention. Run finditer over the full
    # text; the regex's lookbehind boundary already prevents matches
    # inside prose like "100 µm for all illustrations".
    for m in _BAUMGARTNER_CLAUSE_RE.finditer(text):
        labels_raw = m.group(1)
        species = m.group(2).strip()
        if not species:
            continue
        # Filter obvious non-species: location words, lithology, etc.
        if "indet" in species.lower() or "& species" in species.lower():
            continue
        if species.lower() in {"sample", "plate", "scale"}:
            continue
        # Genus-only matches (no epithet and no "sp.") are accepted if:
        #   1. The species is followed by an author citation in
        #      parentheses — e.g. "Archaeodictyomitra (Mizutani)".
        #   2. The species is followed by an uncertainty marker — e.g.
        #      "Triactoma" followed by ";" (next clause) or
        #      "(?)" / "(?) sp." patterns.
        #   3. The species is preceded by a species-list preamble
        #      (i.e. NOT the "Plate N - <prose>" preamble, which is
        #      already blocked by the regex's lookbehind boundary).
        # The original filter only accepted option (1), which
        # over-rejected real single-word genera like "Triactoma" in
        # the baum pl02 caption ("1, 2- Triactoma; 8-10- ..."). Adding
        # the additional accept paths lifts that without re-enabling
        # the original "Plate 1 - Middle" false positive (the
        # lookbehind `(?<![A-Za-z]\s)` on the regex already blocks
        # that preamble pattern).
        tokens = species.split()
        if len(tokens) == 1 and tokens[0][0].isupper():
            tail = text[m.end() : m.end() + 8].lstrip()
            tail_first = tail[:1]
            # Accept if the next non-space character is one of:
            #   "(" — author citation or "(?)" uncertainty marker
            #   ";" — next clause
            #   "." — sentence end
            #   digit — next label
            #   end-of-text — final clause
            # Reject otherwise (prose continuation like
            # "Middle and Upper..." or "Ferresium in Baumgartner...").
            if tail_first not in ("(", ";", ".", ""):
                # Special case: a digit at the start of the tail is
                # the start of the next label.
                if not (tail_first.isdigit()):
                    continue
        labels = _regex_expand_label_list(labels_raw)
        if not labels:
            continue
        # Phase 64 audit: keep non-conflicting labels (same fix as above).
        new_labels = [lbl for lbl in labels if lbl not in seen_labels]
        if not new_labels:
            continue
        for lbl in new_labels:
            seen_labels.add(lbl)
        # Patch up missing A/B identifier for Spumellaria / Nassellaria
        # "gen. et sp. indet." forms. The regex stops at "gen" (because
        # "et" is 2 chars and doesn't match the epithet shape), so the
        # "A" / "B" suffix at the end of the clause is unreachable.
        # We look ahead up to 30 chars in the caption for a single
        # capital letter identifier; if found we append it to the
        # species so the post-normalize step yields "Spumellaria
        # indet. A" (matching gold). The previous pattern ``\b([A-Z])\b``
        # was too loose — it could grab the leading "P" of "Plate", the
        # "M" of "Mizutani", or any other capital letter that happened
        # to be in the next 30 chars. Tighten the boundary to require
        # the identifier to be at the end of the clause: followed by
        # a clause terminator (``;``, ``.``, or end-of-text), never
        # by a word that would indicate we matched the wrong capital.
        if re.match(r"^(Spumellaria|Nassellaria)\s+gen\.?$", species, flags=re.IGNORECASE):
            tail_window = text[m.end() : m.end() + 30]
            m_id = re.search(r"\b([A-Z])\b(?=\s*[;.,]|\s*$)", tail_window)
            if m_id:
                species = species + " " + m_id.group(1)
        pairs.append(
            CaptionPair(
                labels=new_labels,
                species=species,
                modifier="",
                confidence=0.65,
                notes="regex_fallback_baumgartner",
                raw_text=m.group(0)[:120],
            )
        )
    # Normalize species strings across all parsers to strip "(?)" markers,
    # "sensu <Author>" tails, and normalize Spumellaria / Nassellaria
    # "gen. et sp. indet." forms to "indet." (the gold convention).
    normalized: list[CaptionPair] = []
    for p in pairs:
        sp = _normalize_species(p.species)
        if not sp:
            continue
        if sp != p.species:
            p2 = CaptionPair(
                labels=p.labels,
                species=sp,
                modifier=p.modifier,
                confidence=p.confidence,
                notes=p.notes,
                raw_text=p.raw_text,
            )
            normalized.append(p2)
        else:
            normalized.append(p)
    return normalized


# ---------------------------------------------------------------------------
# Data classes for stage I/O
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CaptionPair:
    """A parsed (label-set -> species) clause from a caption."""

    labels: list[str]  # e.g. ["A", "B"] or ["3", "4"]
    species: str  # canonical Latin name
    modifier: str = ""  # "sp.", "cf.", "aff.", "?", "n. sp."
    confidence: float = 0.9  # M3's self-assessed parse confidence
    notes: str = ""  # optional parsing notes
    raw_text: str = ""  # original caption span that produced this pair

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlateClassification:
    """M3's view of the entire plate."""

    is_radiolarian_plate: bool = True
    image_type: str = (
        "micrograph"  # "micrograph" | "SEM" | "photomicrograph" | "diagram" | "photo" | "other"
    )
    panel_count_estimate: int | None = None
    specimen_count_estimate: int | None = None
    quality: str = "ok"  # "good" | "ok" | "poor"
    dominant_taxa: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PanelBox:
    """M3's view of an individual panel within the plate."""

    panel_id: str
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in plate pixel coordinates
    visible_label: str | None = None  # e.g. "A" if M3 sees the letter on the panel
    morphology: str = ""  # one-line morphology hint
    confidence: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


@dataclass(slots=True)
class PanelMatch:
    """M3's per-panel species assignment."""

    panel_id: str
    label: str | None
    species: str | None
    confidence: float
    reasoning: str
    alternative: str | None = None
    is_radiolarian: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Critique:
    """M3 self-critique of an existing per-panel match."""

    panel_id: str
    verdict: str  # "agree" | "disagree" | "uncertain"
    suggested_species: str | None = None
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage prompts (Chinese; designed for extended-thinking M3)
# ---------------------------------------------------------------------------

_PARSE_CAPTION_SYSTEM = """你是放射虫古生物学专家，专长是从图版说明（caption）中抽取"图版label-拉丁学名"映射。

任务：给定一段非结构化的图版说明文字，输出所有 (label集合 → 物种) 配对。

输出规则：
1. 严格 JSON 数组，每项一个配对。
2. labels: 字符串数组，按图版印刷顺序排。处理 A,B / A-D / 3,4 / 3-5 / A-C, 4 等形式。
3. species: 拉丁学名（属 + 种加词，可含 sp. / cf. / aff. / n. sp. 等修饰）。
4. modifier: 单独的修饰标记（"sp."/"cf."/"aff."/"n. sp."/"?"/""）。
5. confidence: 0-1，反映你对自己解析的把握。
6. notes: 简短的解析说明（"括注=scale 50μm, 忽略"等），没有则空串。
7. raw_text: 产生该配对的原文片段；找不到则空串。
8. open_nomenclature_strength: "none" | "cf." | "aff." | "ex gr." | "subgen." | "?" — ICZN 开放命名强度。species 含 cf./aff./ex gr./? 标记时填相应值，否则填 "none"。**该字段决定下游 confidence 折扣**。

示例输入（中文图说）：
"图3 扫描电镜照片。A-D: Tetraspongodiscus stauracanthus n. sp.; E, F: Falcispongus scalaris sp. nov. Scale bars = 50 μm in A, C; 30 μm in B, D-F."

示例输出：
[{"labels":["A","B","C","D"],"species":"Tetraspongodiscus stauracanthus","modifier":"n. sp.","confidence":0.97,"notes":"","raw_text":"A-D: Tetraspongodiscus stauracanthus n. sp.","open_nomenclature_strength":"none"},{"labels":["E","F"],"species":"Falcispongus scalaris","modifier":"sp. nov.","confidence":0.95,"notes":"","raw_text":"E, F: Falcispongus scalaris sp. nov.","open_nomenclature_strength":"none"}]

完整输入→输出示例（含开放命名）：
输入：Fig. 1. 1, Triactoma kamoensis; 2, cf. Pessagnoa sp.; 3, Archaeodictyomitra (?) sp.
输出：
[
  {"labels":["1"],"species":"Triactoma kamoensis","modifier":"","confidence":0.95,"notes":"","raw_text":"1, Triactoma kamoensis","open_nomenclature_strength":"none"},
  {"labels":["2"],"species":"Pessagnoa sp.","modifier":"cf.","confidence":0.55,"notes":"caption 写 cf.，是 ICZN confer 标记","raw_text":"cf. Pessagnoa sp.","open_nomenclature_strength":"cf."},
  {"labels":["3"],"species":"Archaeodictyomitra sp.","modifier":"(?)","confidence":0.55,"notes":"caption 含 (?) 不确定标记","raw_text":"Archaeodictyomitra (?) sp.","open_nomenclature_strength":"?"}
]

示例输入（英文图说，Phase 4A 补全）：
Input: "Figure 3. SEM images of Entactinia sp. from the Late Devonian (Frasnian) of the Canning Basin, Western Australia. Scale bars = 100 μm."
Output:
[
  {"labels":["3"],"species":"Entactinia sp.","modifier":"","confidence":0.85,"notes":"单 panel 英文图说，sp. 标记","raw_text":"Figure 3. SEM images of Entactinia sp.","open_nomenclature_strength":"none"}
]

示例输入（中文图版含 sp. nov. 列表，Phase 4A 补全）：
Input: "图版 5. 西藏南部晚三叠世放射虫化石。1-3. Triassocampe deweveri (Nakaseko &amp; Nishimura); 4-6. Archaeosemantis pteropus Haeckel. 比例尺 = 50 μm。"
Output:
[
  {"labels":["1","2","3"],"species":"Triassocampe deweveri","modifier":"","confidence":0.94,"notes":"范围 1-3，作者置于括号","raw_text":"1-3. Triassocampe deweveri (Nakaseko & Nishimura)","open_nomenclature_strength":"none"},
  {"labels":["4","5","6"],"species":"Archaeosemantis pteropus","modifier":"","confidence":0.94,"notes":"范围 4-6，作者 Haeckel","raw_text":"4-6. Archaeosemantis pteropus Haeckel","open_nomenclature_strength":"none"}
]

示例输入（含 cf./aff./ex gr. 三种开放命名，Phase 4A 补全）：
Input: "Plate 2. Various radiolarians from the Lower Jurassic (Pliensbachian) of the Budva zone. 1. cf. Parahsuum sp. A; 2. aff. Hsuum sp.; 3. Praecanadium ex gr. aptum Blome. Sample BB-14."
Output:
[
  {"labels":["1"],"species":"Parahsuum sp. A","modifier":"cf.","confidence":0.5,"notes":"cf. confer 标记，sp. A 标本编号","raw_text":"1. cf. Parahsuum sp. A","open_nomenclature_strength":"cf."},
  {"labels":["2"],"species":"Hsuum sp.","modifier":"aff.","confidence":0.5,"notes":"aff. affinis 标记","raw_text":"2. aff. Hsuum sp.","open_nomenclature_strength":"aff."},
  {"labels":["3"],"species":"Praecanadium aptum","modifier":"ex gr.","confidence":0.5,"notes":"ex gr. ex grege 居群标记，作者 Blome","raw_text":"3. Praecanadium ex gr. aptum Blome","open_nomenclature_strength":"ex gr."}
]

Output MUST match the JSON schema exactly. See examples below.

只输出 JSON 数组，不要任何解释文本。"""


# Phase 27: Japanese system prompt for parse_caption. Mirrors the
# Chinese prompt above byte-for-byte on the JSON output schema so the
# downstream ``_safe_json_loads`` and ``CaptionPair`` field-mapping
# code paths work unchanged. Only the language of the *system
# instructions* is Japanese — the actual caption text in the user
# prompt can be any language (JA, bilingual JA+EN, or even ZH).
#
# Triggered automatically by ``_detect_caption_lang`` when the caption
# contains Hiragana / Katakana / CJK ideographs, OR explicitly via
# ``--m3-prompt-lang ja``.
_PARSE_CAPTION_SYSTEM_JA = """あなたは放散虫古生物学の専門家で、図版キャプションから「ラベル集合 → ラテン学名」のマッピングを抽出することが専門です。

タスク: 非構造化の図版説明文を受け取り、すべての (label集合 → 種) ペアを出力する。

出力規則 (strict JSON array; 各要素 1 ペア):
1. labels: 文字列配列。図版印刷順に並べる。A,B / A-D / 3,4 / 3-5 / A-C, 4 など。
2. species: ラテン学名 (属 + 種小名; sp. / cf. / aff. / n. sp. 等を含む)。
3. modifier: 単独の修飾語 ("sp." / "cf." / "aff." / "n. sp." / "?" / "")。
4. confidence: 0-1。解析の確信度。
5. notes: 簡潔な解析メモ (例: "括注 = scale 50μm, 無視"); なければ空文字列。
6. raw_text: ペア生成元の原文抜粋。なければ空文字列。

入力例 (和文キャプション):
"図版1 走査電子顕微鏡写真。A-D: Tetraspongodiscus stauracanthus n. sp.; E, F: Falcispongus scalaris sp. nov. Scale bars = 50 μm in A, C; 30 μm in B, D-F."

出力例:
[{"labels":["A","B","C","D"],"species":"Tetraspongodiscus stauracanthus","modifier":"n. sp.","confidence":0.97,"notes":"","raw_text":"A-D: Tetraspongodiscus stauracanthus n. sp."},{"labels":["E","F"],"species":"Falcispongus scalaris","modifier":"sp. nov.","confidence":0.95,"notes":"","raw_text":"E, F: Falcispongus scalaris sp. nov."}]

JSON 配列のみを出力し、説明文は付けないこと。"""


def _detect_caption_lang(text: str) -> str:
    """Heuristic language detector for caption text.

    Returns ``"ja"`` if the text contains Hiragana, Katakana, or any
    CJK Unified Ideograph (the latter intentionally covers both JA
    kanji and ZH hanzi — for our routing purposes both map to the
    JA-aware parse_caption prompt, because the JA prompt is also the
    closest match for ZH bilingual JA+EN papers; the ZH-only prompt
    remains the fallback). Returns ``"zh"`` otherwise so English-only
    captions and ASCII text fall through to the legacy Chinese
    system prompt unchanged.

    Phase 27: this is a tiny O(n) char scan, no external deps. It's
    deliberately conservative — false-positives on Hiragana are
    extremely rare in radiolarian papers (the surrounding text is
    either Latin species binomials, CJK running text, or English).
    """
    if not text:
        return "zh"
    for ch in text:
        code = ord(ch)
        # Hiragana (0x3040-0x309F), Katakana (0x30A0-0x30FF),
        # CJK Unified Ideographs (0x4E00-0x9FFF — covers both JA kanji
        # and ZH hanzi).
        if 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF:
            return "ja"
    return "zh"


_CLASSIFY_PLATE_SYSTEM = """你是放射虫图版审查员。请判断一张给定的图像是否是一张"含放射虫标本的图版（plate）"。

任务：观察图像并返回严格 JSON。

判定要点：
1. is_radiolarian_plate: true/false。
   - true  = 主要内容是放射虫扫描电镜/光镜照片，含 1 个或多个标本 panel。
   - false = 是普通正文页、参考文献、目录、章节标题、方程、地图、柱状图、相图、表格、流程图、纯文字页等。
2. image_type: "SEM" / "micrograph" / "photomicrograph" / "diagram" / "photo" / "other"。
3. panel_count_estimate: 你能看到的独立 panel（带 A/B/C 标签的小图）数量，估不出则 null。
4. specimen_count_estimate: 估计的标本总数；估不出则 null。
5. quality: "good" / "ok" / "poor" — 图像清晰度/对比度/裁剪质量。
6. dominant_taxa: 若可识别，列出 1-3 个最可能的属名（识别不出则空数组）。
7. reasoning: 1-2 句解释你为什么这么判断。

Example 1 (radiolarian SEM plate):
Input image: SEM micrograph arranged in a 2x3 grid of 6 panels labeled "A" through "F" in the lower-left of each panel; spherical to subspherical radiolarian specimens with latticed walls; scale bar "50 μm" visible in panel A; high contrast, black background.
Output:
{"is_radiolarian_plate":true,"image_type":"SEM","panel_count_estimate":6,"specimen_count_estimate":6,"quality":"good","dominant_taxa":["Tetraspongodiscus"],"reasoning":"看到 6 个带 A-F 标签的 SEM 标本 panel，对比度高，壳壁格状结构清晰。"}

Example 2 (text-only page — NOT a plate):
Input image: Single-column black-on-white text page with the centered bold heading "Applications of Radiolarian Studies in Biostratigraphy" at the top; below are three paragraphs of running text; no figures, no SEM imagery.
Output:
{"is_radiolarian_plate":false,"image_type":"other","panel_count_estimate":null,"specimen_count_estimate":null,"quality":"ok","dominant_taxa":[],"reasoning":"这是一段正文的标题页，'Applications' 字样，无标本图像。"}

Example 3 (stratigraphic column figure — NOT a plate):
Input image: Tall narrow vertical column with stacked colored rectangles (yellow = sandstone, grey = shale, blue = limestone), depth scale in meters on left, formation labels on right ("Sundance Fm", "Morrison Fm"), 5 numbered beds; no radiolarian specimens.
Output:
{"is_radiolarian_plate":false,"image_type":"diagram","panel_count_estimate":null,"specimen_count_estimate":null,"quality":"good","dominant_taxa":[],"reasoning":"图是柱状地层剖面，含岩性色块和地层名，没有任何放射虫标本 panel。"}

Example 4 (野外露头照片 — NOT a plate，Phase 4A 补全):
Input image: 横版彩色照片（1600x900 px），中央是一处山地露头，可见灰色层状灰岩夹泥岩，地表有稀疏植被；右下角有指南针和地质锤作为比例参考；左下角贴有红色样本袋标注 "Loc-14"。无任何 SEM 标本。
Output:
{"is_radiolarian_plate":false,"image_type":"photo","panel_count_estimate":null,"specimen_count_estimate":null,"quality":"good","dominant_taxa":[],"reasoning":"是野外露头照片，含岩层 + 比例物（锤子/罗盘）+ 样本袋标注；无显微标本 panel。判定为非放射虫图版。"}

Example 5 (中文图说 + SEM 标本，Phase 4A 补全):
Input image: 中文图版，2x2 共 4 个 panel，标签 "1"、"2"、"3"、"4" 在每个 panel 左上角；每个 panel 内有一颗钟形放射虫（campanulate），可见顶角和胸腹节；比例尺 "30 μm" 在右下角；高对比度 SEM 图。
Output:
{"is_radiolarian_plate":true,"image_type":"SEM","panel_count_estimate":4,"specimen_count_estimate":4,"quality":"good","dominant_taxa":["Archaeodictyomitra"],"reasoning":"看到 4 个带 1-4 数字标签的 SEM 标本 panel；钟形壳 + 顶角形态指向 Archaeodictyomitra 类群；中文期刊常见此类版式。"}

Output MUST match the JSON schema exactly. See examples below.

只输出严格 JSON，不要解释。"""


_SEGMENT_PANELS_SYSTEM = """你是放射虫图版版面分析专家。任务：从一张放射虫图版中识别出每个独立 panel（每个 panel 通常带一个 A/B/C/D...字母标签）的边界框。

输出规则（严格 JSON 数组）：
1. panel_id: 字符串，从可见标签读取，例如 "A"、"B"；若没有标签，用 "P1"/"P2"...
2. bbox: [x, y, w, h]，像素坐标，原点在左上。
3. visible_label: 看到的标签字符串；看不到则 null。
4. morphology: 一句话描述这个 panel 里标本的形态特征（"球形壳，三主轴，格状壁孔"等）。
5. confidence: 0-1，反映你对自己框选和形态描述的把握。

重要：
- 框必须紧贴标本内容（包括 scale bar 旁边的内容，但裁掉纯白边）。
- 多个 panel 之间不要重叠。
- 如果整图就一个 panel，输出单元素数组覆盖整图（bbox=[0,0,W,H]）。
- 若图版里有箭头、字母标签、scale bar，把它们一起包进对应 panel 框内。
- 不要包含图说文字（caption 文字在图版外面）。

Example 1 (multi-panel SEM plate):
Input image: SEM plate, 1200x900 px. 4 panels in a 2x2 grid: "A" (top-left, ~30,30,540,400), "B" (top-right, ~620,30,540,400), "C" (bottom-left, ~30,470,540,400), "D" (bottom-right, ~620,470,540,400). Each panel shows one spherical radiolarian specimen with scale bar "50 μm" in the lower-right of the panel.
Output:
[
  {"panel_id":"A","bbox":[30,30,540,400],"visible_label":"A","morphology":"球形壳，约 250 μm 直径，壁孔呈规则六边形排列","confidence":0.95},
  {"panel_id":"B","bbox":[620,30,540,400],"visible_label":"B","morphology":"球形壳，三根对称主刺从极点伸出，格状壁孔","confidence":0.93},
  {"panel_id":"C","bbox":[30,470,540,400],"visible_label":"C","morphology":"椭球形壳，两极略尖，壁孔小而密","confidence":0.92},
  {"panel_id":"D","bbox":[620,470,540,400],"visible_label":"D","morphology":"球形壳，外壁薄，内骨架可见","confidence":0.90}
]

Example 2 (single-panel figure with no visible letter label):
Input image: Single high-resolution SEM photo of one radiolarian specimen filling 90% of the frame (1024x1024 px), no letter labels visible, white background margin ~50 px on each side.
Output:
[
  {"panel_id":"P1","bbox":[50,50,924,924],"visible_label":null,"morphology":"钟形壳，顶端有小孔，底部 3 根细刺","confidence":0.88}
]

Example 3 (不规则布局，Phase 4A 补全):
Input image: 不规则 SEM 图版（1600x1200 px），共 5 个 panel，布局如下：(1) 大 panel A 在左侧中央（约 40,200,720,720），含一个特写标本；(2) 中 panel B 在右上（约 800,40,420,420）；(3) 小 panel C 在右下（约 1280,40,300,300）；(4) 中 panel D 在左下（约 40,1000,700,180，含 2 个并列标本视图，被一个细色带分开为 D-left 和 D-right，但整体仍属 panel D 的一个 bbox，标注写在最左）；(5) 微 panel E 在 panel D 右侧（约 760,1000,180,180）。每个 panel 都有清晰字母标签。
Output:
[
  {"panel_id":"A","bbox":[40,200,720,720],"visible_label":"A","morphology":"大特写单标本，球形壳 + 6 根对称主刺 + 格状壁孔","confidence":0.96},
  {"panel_id":"B","bbox":[800,40,420,420],"visible_label":"B","morphology":"椭球形壳，壁孔小而密，无主刺","confidence":0.93},
  {"panel_id":"C","bbox":[1280,40,300,300],"visible_label":"C","morphology":"钟形壳，顶端有明显顶角","confidence":0.91},
  {"panel_id":"D","bbox":[40,1000,700,180],"visible_label":"D","morphology":"水平并列两枚标本（左：球形，右：椭球），中间色带分隔","confidence":0.85},
  {"panel_id":"E","bbox":[760,1000,180,180],"visible_label":"E","morphology":"微型 panel，1 根主刺局部","confidence":0.80}
]

Output MUST match the JSON schema exactly. See examples below.

只输出 JSON 数组，不要解释。"""


_MATCH_PANEL_SYSTEM_VISUAL_ONLY = """你是放射虫古生物学专家，负责为单个 panel 标本做形态学鉴定（visual-only 模式）。

场景：图版说明（caption）未能被提取，**没有候选物种清单**。请完全依靠 panel 图像本身的形态特征做最可能的分类。

任务：观察一个 panel 图像 + 整张图版（用于上下文） + 可选可见字母标签，输出最可能的属/种鉴定。

输出（严格 JSON）：
1. label: 该 panel 的字母标签（如有 "A"），否则用 "?"。
2. species: 拉丁学名（属 + 种加词）；若是新种、未知种或仅可鉴定到属，给出属名 + " sp." 或 "sp. indet."；若完全无法鉴定，设为 null。
3. confidence: 0-1，反映判定的把握。**visual-only 模式请保守**：conf >= 0.6 仅在你对形态非常确定时给出。
4. reasoning: 1-2 句说明形态依据（"球形壳 + 6 根对称主刺 + 格状壳壁 → Archaeosemantis 形态类型" 等）。
5. alternative: 第二可能的物种；无则 null。
6. is_radiolarian: true/false — 这真的是放射虫吗？若 false 则 species 设为 null。

注意：
- 不要虚构从未在公开文献中记录的物种。
- 形态不清晰时倾向给属名 + sp.，并降低 confidence。
- 若图中可见 A/B/C 字母标签，原样输出。

只输出严格 JSON。"""


_MATCH_PANEL_SYSTEM = """你是放射虫古生物学专家，负责为单个 panel 标本指定其最可能的分类（label + 拉丁学名）。

任务：观察一个 panel 图像 + 候选标签-物种配对列表 + 整张图版说明，输出最可能的配对及推理。

输入：
- 图像：一个 panel 标本的裁剪图。
- 候选配对：从图版说明中解析出的 (label集合 → 物种) 列表（已去重、按字母顺序排好）。
- 提示标签：M3 之前在图版上看到的可见字母（如 "A"），可能为 null。
- 完整图说：作为辅助上下文。

输出（严格 JSON）：
1. label: 你判定的 panel 对应的字母（A/B/C...）；若该 panel 不属于任何候选配对，给出最可能的字母或 null。
2. species: 拉丁学名。
3. open_nomenclature_strength: "none" | "cf." | "aff." | "ex gr." | "subgen." | "?" — ICZN 开放命名的强度：
   - "none" = 完全鉴定到种（默认）。
   - "cf." = 形态相似，可能但不确认是该种（confer）。
   - "aff." = 形态接近但有差异（affinis）。
   - "ex gr." = 属于某一 group / 居群（ex grege，bandini 2011 的 subgenus-group 标记）。
   - "subgen." = 亚属标记。
   - "?" = 鉴定不确定（caption 里的 "(?)" 标记）。
   当 species 含 cf./aff./ex gr./? 任一标记时必须填，否则填 "none"。
4. confidence: 0-1，反映判定的把握。**注意：open_nomenclature_strength 不为 "none" 时，confidence 上限自动折扣为 0.55（cf./aff./?）或 0.50（ex gr.）；请你在 confidence 上自行反映这种不确定性**。
5. reasoning: 1-2 句解释（"caption 中 A-B 配 X；图上标签为 A；形态与 X 一致" 等）。
6. alternative: 第二可能的物种；无则 null。
7. is_radiolarian: true/false — 这真的是放射虫吗？若 false 则 species 设为 null。

判定优先级：
1) 图上可见字母标签（最高）。
2) caption 中明确写出的 label-物种 子句。
3) 形态/语义一致性（最低；只能用于在多个候选间 tie-break）。

注意：
- 引用候选配对中的物种名时，**保持原始拼写**（包括大小写、空格）。
- 候选配对为空时，species 设为 null 并降低 confidence。
- 不要凭空编造从未在 caption 出现过的物种名。

Example 1 (clear caption match, label visible):
Panel image: spherical radiolarian with 3 polar spines, latticed cortical shell, ~250 μm diameter, label "B" visible in lower-left.
Candidate pairs (from caption): [{"labels":["A","B"],"species":"Tetraspongodiscus stauracanthus","modifier":"n. sp."},{"labels":["C","D"],"species":"Falcispongus scalaris","modifier":"sp. nov."}]
Visible label hint: "B"
Caption: "A, B: Tetraspongodiscus stauracanthus n. sp.; C, D: Falcispongus scalaris sp. nov."
Output:
{"label":"B","species":"Tetraspongodiscus stauracanthus","open_nomenclature_strength":"none","confidence":0.97,"reasoning":"图上可见字母标签 B；caption 中 A-B 配 Tetraspongodiscus stauracanthus；球形壳+3 主刺形态一致。","alternative":null,"is_radiolarian":true}

Example 2 (open-nomenclature match — confidence must be lower):
Panel image: subspherical radiolarian, cortical shell with small regular pores, no visible label.
Candidate pairs (from caption): [{"labels":["1"],"species":"Pessagnoa sp.","modifier":"cf."},{"labels":["2"],"species":"Archaeodictyomitra sp.","modifier":"(?)"}]
Visible label hint: null
Caption: "1, cf. Pessagnoa sp.; 2, Archaeodictyomitra (?) sp."
Output:
{"label":"1","species":"Pessagnoa sp.","open_nomenclature_strength":"cf.","confidence":0.55,"reasoning":"caption 写 cf.，是 ICZN confer 标记；形态亚球形+小孔与 Pessagnoa 一致；open-nomen 上限 0.55。","alternative":"Archaeodictyomitra sp.","is_radiolarian":true}

Example 3 (no candidate, species = null):
Panel image: blurry object, cannot determine if radiolarian.
Candidate pairs (from caption): []
Visible label hint: null
Caption: "(caption not extracted)"
Output:
{"label":null,"species":null,"open_nomenclature_strength":"none","confidence":0.2,"reasoning":"候选配对为空且图像模糊无法鉴定；降低 confidence 反映高度不确定性。","alternative":null,"is_radiolarian":false}

Example 4 (英文 caption + 多候选，Phase 4A 补全):
Panel image: nassellarian with 4 segments, distinct apical horn, thorax pores arranged in rows.
Candidate pairs (from caption): [{"labels":["1","2"],"species":"Parahsuum sp.","modifier":"cf."},{"labels":["3","4","5"],"species":"Hsuum sp.","modifier":""},{"labels":["6"],"species":"Lamptonium sp.","modifier":"aff."}]
Visible label hint: "3"
Caption: "Figs. 1, 2. cf. Parahsuum sp.; 3-5. Hsuum sp.; 6. aff. Lamptonium sp."
Output:
{"label":"3","species":"Hsuum sp.","open_nomenclature_strength":"none","confidence":0.93,"reasoning":"图上可见标签 3，落在 caption 范围 3-5 → Hsuum sp.；多节 nassellarian 形态与 Hsuum 一致。","alternative":null,"is_radiolarian":true}

Output MUST match the JSON schema exactly. See examples below.

只输出严格 JSON。"""


_CRITIQUE_SYSTEM = """你是放射虫分类学审查员。任务：交叉验证一组已经做出的 panel→物种配对，纠正明显错误。

输入：
- 完整图版图像。
- 已配对结果：每个 panel 的 (panel_id, label, species, confidence, reasoning)。
- 图版说明：caption 原文。

请对**每个 panel** 给出评判（严格 JSON 数组，每项一个）：
1. panel_id: 字符串（与输入一致）。
2. verdict: "agree" / "disagree" / "uncertain"。
3. suggested_species: 若 disagree/uncertain 且有更合理候选，给出拉丁名；否则 null。
4. open_nomenclature_strength: "none" | "cf." | "aff." | "ex gr." | "subgen." | "?" — suggested_species 含 cf./aff./ex gr./? 标记时填相应值，否则 "none"。**该字段用于在 F1 评分时折扣开放命名匹配**。
6. confidence: 0-1，反映你的判断把握。
7. reasoning: 1 句解释。

判定规则：
- 如果原配对 species 与 caption 中同一 label 的候选一致 → agree。
- 如果 caption 中该 label 应是 X，但你看到图上形态明显属于 Y（如球形 vs 钟形）→ disagree 并给 Y。
- 如果信息不足判断 → uncertain，suggested_species 可给一个第二可能。
- 若 panel 不是放射虫（is_radiolarian=false）→ 直接 agree 不要改。

Example 1 (agree — caption + morphology both consistent):
Plate image: 4-panel SEM, panels A-D each showing one spherical radiolarian with 3 polar spines.
Existing pairings: [{panel_id:"A",label:"A",species:"Tetraspongodiscus stauracanthus",confidence:0.95}, {panel_id:"B",label:"B",species:"Tetraspongodiscus stauracanthus",confidence:0.94}, {panel_id:"C",label:"C",species:"Falcispongus scalaris",confidence:0.92}, {panel_id:"D",label:"D",species:"Falcispongus scalaris",confidence:0.91}]
Caption: "A, B: Tetraspongodiscus stauracanthus n. sp.; C, D: Falcispongus scalaris sp. nov."
Output:
[
  {"panel_id":"A","verdict":"agree","suggested_species":null,"open_nomenclature_strength":"none","confidence":0.97,"reasoning":"caption A 配 Tetraspongodiscus，图上球形+3 主刺形态一致。"},
  {"panel_id":"B","verdict":"agree","suggested_species":null,"open_nomenclature_strength":"none","confidence":0.97,"reasoning":"同 A，形态与 caption 一致。"},
  {"panel_id":"C","verdict":"agree","suggested_species":null,"open_nomenclature_strength":"none","confidence":0.95,"reasoning":"caption C 配 Falcispongus；图上椭球形壳无主刺，与 Falcispongus 形态一致。"},
  {"panel_id":"D","verdict":"agree","suggested_species":null,"open_nomenclature_strength":"none","confidence":0.95,"reasoning":"同 C，形态与 caption 一致。"}
]

Example 2 (disagree — morphology contradicts caption assignment):
Plate image: panel E shows a clear bell-shaped (campanulate) radiolarian with apical horn.
Existing pairings: [{panel_id:"E",label:"E",species:"Cenosphaera sp.",confidence:0.88}]
Caption: "E: Archaocenosphaera campanula n. sp."
Output:
[
  {"panel_id":"E","verdict":"disagree","suggested_species":"Archaocenosphaera campanula","open_nomenclature_strength":"none","confidence":0.92,"reasoning":"caption 明确配 Archaocenosphaera campanula，图上钟形壳+顶角形态也支持钟形属；原配对 Cenosphaera（球形属）与形态矛盾。"}
]

Example 3 (uncertain — info insufficient):
Plate image: panel F is partially out of focus, shows a fragment.
Existing pairings: [{panel_id:"F",label:"F",species:"GenusA sp.",confidence:0.55}]
Caption: "F: GenusA sp. or GenusB sp."
Output:
[
  {"panel_id":"F","verdict":"uncertain","suggested_species":"GenusB sp.","open_nomenclature_strength":"none","confidence":0.45,"reasoning":"图像模糊且 caption 同时列两个候选；形态不足判别，建议 GenusB 作为第二可能。"}
]

Example 4 (低 confidence 整体审查，Phase 4A 补全):
Plate image: 6-panel SEM plate; panels A-F each with one specimen; however the SEM stage drift left all panels slightly blurred, scale bars barely readable.
Existing pairings: [{panel_id:"A",label:"A",species:"GenusA sp.",confidence:0.62},{panel_id:"B",label:"B",species:"GenusB sp.",confidence:0.58},{panel_id:"C",label:"C",species:"GenusA sp.",confidence:0.61}]
Caption: "A, C, E: GenusA sp.; B, D, F: GenusB sp."
Output:
[
  {"panel_id":"A","verdict":"agree","suggested_species":null,"open_nomenclature_strength":"none","confidence":0.4,"reasoning":"caption 配 GenusA sp. 一致；图像模糊仅依 caption 验证；confidence 反映图像质量折扣。"},
  {"panel_id":"B","verdict":"agree","suggested_species":null,"open_nomenclature_strength":"none","confidence":0.4,"reasoning":"caption 配 GenusB sp. 一致；图像模糊仅依 caption 验证；confidence 反映图像质量折扣。"},
  {"panel_id":"C","verdict":"agree","suggested_species":null,"open_nomenclature_strength":"none","confidence":0.4,"reasoning":"caption 配 GenusA sp. 一致；图像模糊仅依 caption 验证；confidence 反映图像质量折扣。"}
]

Output MUST match the JSON schema exactly. See examples below.

只输出严格 JSON 数组。"""


# ---------------------------------------------------------------------------
# Prompt registry (audit 2026-08-19 Phase 4C)
# ---------------------------------------------------------------------------
#
# The 5 stage system prompts above are the canonical sources of truth
# for the JSON contract each stage emits. Historically
# ``gemma_postprocess.py`` re-defined its own copies of the per-panel
# prompt and would silently drift when M3 prompts were updated — a
# real bug (audit 2026-08-19 Bug M-10) because after M3 fails the
# Gemma fallback would use a STALE prompt that no longer matches the
# format M3 actually emits.
#
# ``get_prompt_registry()`` aggregates the 5 stage system prompts
# (plus the Japanese variant of ``parse_caption``) into a single
# dict keyed by stage name so other modules can pull the canonical
# prompt instead of re-implementing it. Tests import this function
# to verify Gemma really uses M3's prompt (audit Bug M-12).
#
# Note: callers should treat the returned dict as read-only. We
# deliberately return a NEW dict each call so a caller can mutate it
# locally without poisoning the cached prompts inside ``m3_engine``.


# Audit 2026-08-19 Phase 4E (Task 3): version stamp on the prompt
# registry. Every change to ``_PARSE_CAPTION_SYSTEM`` /
# ``_CLASSIFY_PLATE_SYSTEM`` / ``_SEGMENT_PANELS_SYSTEM`` /
# ``_MATCH_PANEL_SYSTEM`` / ``_MATCH_PANEL_SYSTEM_VISUAL_ONLY`` /
# ``_CRITIQUE_SYSTEM`` MUST bump this string so downstream audit can
# correlate schema drift / quality regressions with a known prompt
# revision. The convention is ``"vMAJOR.MINOR.PATCH"``; major for any
# change that alters the JSON output schema, minor for wording changes
# that don't change the schema, patch for typo / formatting fixes.
PROMPT_REGISTRY_VERSION: str = "v1.3.0"


def get_prompt_registry() -> tuple[dict[str, str], str]:
    """Return ``(prompt_registry_dict, version)`` for the M3 stage prompts.

    Audit 2026-08-19 Phase 4E: the function now returns a 2-tuple so
    callers can pin a result to a known prompt version. The first
    element is the prompt map (a NEW dict on every call, so callers
    can mutate it locally without poisoning the module cache); the
    second element is the registry's version stamp.

    Keys (audit 2026-08-19 Phase 4C):
      - ``parse_caption``            : Stage 1 (text-only, ZH)
      - ``parse_caption_ja``         : Stage 1 (text-only, JA)
      - ``classify_plate``           : Stage 2 (vision)
      - ``segment_panels``           : Stage 3 (vision)
      - ``match_panel_visual_only``  : Stage 4 fallback (vision)
      - ``match_panel``              : Stage 4 (vision + text)
      - ``critique_matches``         : Stage 5 (vision + text)

    Other modules (notably ``gemma_postprocess``) MUST call this
    function rather than re-define the prompt inline.
    """
    return (
        {
            "parse_caption": _PARSE_CAPTION_SYSTEM,
            "parse_caption_ja": _PARSE_CAPTION_SYSTEM_JA,
            "classify_plate": _CLASSIFY_PLATE_SYSTEM,
            "segment_panels": _SEGMENT_PANELS_SYSTEM,
            "match_panel_visual_only": _MATCH_PANEL_SYSTEM_VISUAL_ONLY,
            "match_panel": _MATCH_PANEL_SYSTEM,
            "critique_matches": _CRITIQUE_SYSTEM,
        },
        PROMPT_REGISTRY_VERSION,
    )


def get_prompt_registry_version() -> str:
    """Return just the version string of the M3 prompt registry.

    Convenience accessor so callers (e.g. ``_make_telemetry``) don't
    have to unpack the full tuple when they only need the version.
    Equivalent to ``get_prompt_registry()[1]`` but avoids building
    the prompt dict when only the version is wanted.
    """
    return PROMPT_REGISTRY_VERSION


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class _ThinkingFlagGate:
    """Reader/writer gate protecting ``backend.enable_thinking``.

    audit 2026-08-01 (D2): the retry path in ``_infer_text`` /
    ``_infer_vision`` temporarily flips ``backend.enable_thinking`` to
    ``False`` while holding ``_thinking_retry_lock``, but a *first*
    attempt on another worker thread read the flag lock-free — the
    backend itself reads ``self.enable_thinking`` when it builds its
    request kwargs. So worker A's paid first call could silently run
    with thinking disabled purely because worker B happened to be
    inside its retry window: no error, just a quietly degraded answer.

    Serialising every M3 call behind the retry lock would fix that but
    throw away worker concurrency (M3 vision calls are seconds long).
    Instead first attempts take the *read* side — many at a time,
    blocked only while a retry is actually in flight — and the retry
    path takes the *write* side (exclusive). Writers take priority over
    newly arriving readers so a steady stream of first calls can't
    starve the retry path.

    Both sides are reentrant for the owning thread (mirroring the
    ``RLock`` choice for ``_thinking_retry_lock``) so a backend that
    re-enters the engine from inside its own handler cannot deadlock.
    """

    def __init__(self) -> None:
        self._cond = Condition()
        # thread ident -> nesting depth, so the writer can ignore reads
        # held by itself when it waits for readers to drain.
        self._readers: dict[int, int] = {}
        self._writer_thread: int | None = None
        self._writer_depth = 0
        self._writers_waiting = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        """Hold the shared side for the duration of a first-attempt call."""
        me = get_ident()
        with self._cond:
            while self._writer_thread not in (None, me) or (
                self._writer_thread is None and self._writers_waiting > 0
            ):
                self._cond.wait()
            self._readers[me] = self._readers.get(me, 0) + 1
        try:
            yield
        finally:
            with self._cond:
                depth = self._readers.get(me, 0) - 1
                if depth > 0:
                    self._readers[me] = depth
                else:
                    self._readers.pop(me, None)
                self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        """Hold the exclusive side across a save→flip→call→restore retry."""
        me = get_ident()
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_thread not in (None, me) or (
                    self._writer_thread is None and any(tid != me for tid in self._readers)
                ):
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer_thread = me
            self._writer_depth += 1
        try:
            yield
        finally:
            with self._cond:
                self._writer_depth -= 1
                if self._writer_depth == 0:
                    self._writer_thread = None
                self._cond.notify_all()


def _validate_ma_range(record: dict[str, Any]) -> dict[str, Any]:
    """Validate that ``ma_top < ma_base`` (younger = smaller Ma).

    Audit 2026-08-19 Phase 2b M-13: Ma conventions in stratigraphy
    follow *younger-up*: ma_top (top of section) is the younger
    boundary, ma_base is the older boundary, so numerically
    ``ma_top < ma_base``. The vision LLM occasionally flips this
    (e.g. reads a chart with old-at-top axis convention and emits
    ``ma_top > ma_base``) — but our schema contract is fixed. We
    can't auto-fix (we don't know which number is the truth), so we
    null out both fields and warn so the caller can fall back to
    caption-text regex or PBDB lookup.

    Mutates ``record`` in place; returns the same object for chaining.
    """
    top = record.get("ma_top")
    base = record.get("ma_base")
    if top is None or base is None:
        return record
    try:
        top_f = float(top)
        base_f = float(base)
    except (TypeError, ValueError):
        return record
    if top_f > base_f:
        logger.warning(
            "Invalid Ma range in biozone: ma_top=%s > ma_base=%s; setting both to null",
            top,
            base,
        )
        record["ma_top"] = None
        record["ma_base"] = None
        # ma_mid is meaningless when the range is invalid — clear it
        # too so callers don't carry a phantom midpoint.
        record.pop("ma_mid", None)
    return record


def _normalize_ma_pair(record: dict[str, Any]) -> dict[str, Any]:
    """Auto-swap ``ma_top`` and ``ma_base`` when the LLM emits an inverted range.

    Phase 6D audit 2026-08-19 NIT-3: ICZN / stratigraphic convention is
    ``ma_top < ma_base`` (younger = smaller Ma, top of section is the
    younger boundary). The vision LLM occasionally emits the inverted
    range — e.g. reads a stratigraphic column with old-at-top axis
    convention and reports ``ma_top=100, ma_base=50`` (thinking that
    "top" means the visually upper label on the figure rather than
    the stratigraphically younger boundary).

    The previous :func:`_validate_ma_range` helper nulled both fields
    on inversion, which threw away data we could have salvaged. This
    helper AUTO-SWAPS the values so the schema contract is preserved
    with no information loss. The schema layer
    (:class:`rlpe.schema_models.GeologyLinkRecord`) already accepts
    ``None`` for the raw pair, so callers that prefer strict null-on-
    violation can keep using :func:`_validate_ma_range`; callers that
    prefer auto-fix should chain ``_normalize_ma_pair`` BEFORE
    :func:`_validate_ma_range` so the swap happens first.

    Mutates ``record`` in place; returns the same object for chaining.
    """
    top = record.get("ma_top")
    base = record.get("ma_base")
    if top is None or base is None:
        return record
    try:
        top_f = float(top)
        base_f = float(base)
    except (TypeError, ValueError):
        return record
    if top_f > base_f:
        logger.warning(
            "Ma range inverted (ma_top=%r > ma_base=%r); auto-swapping",
            top,
            base,
        )
        record["ma_top"] = base_f
        record["ma_base"] = top_f
        # ma_mid is the midpoint of the pair; if it was provided we
        # assume the LLM got the midpoint right (since the midpoint of
        # (a, b) equals the midpoint of (b, a)) and leave it untouched.
        # If the LLM only emitted ma_top / ma_base and we compute ma_mid
        # downstream, that code uses the (now swapped) fields anyway.
    return record


class M3Engine:
    """M3-Centric 5-stage engine.

    Usage
    -----
    >>> engine = M3Engine(backend, config={"m3_stage_1": True, ...})
    >>> pairs = engine.parse_caption(caption_text)
    >>> cls = engine.classify_plate(plate_image)
    >>> if not cls.is_radiolarian_plate:
    ...     return []
    >>> panels = engine.segment_panels(plate_image)
    >>> matches = [engine.match_panel(p.image, pairs, p.visible_label) for p in panels]
    >>> critiques = engine.critique_matches(plate_image, matches, caption_text)
    >>> final = engine.apply_critiques(matches, critiques)
    """

    def __init__(
        self,
        backend: Any,
        config: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.backend = backend
        self.config = dict(config or {})
        # audit 2026-08-01 (M16): cooperative cancellation hook.
        # The pipeline's ``pool.shutdown(wait=False, cancel_futures=True)``
        # only drops QUEUED futures — in-flight LLM calls keep running
        # and rack up token spend. By accepting a shared ``threading.Event``
        # here, the engine can short-circuit its own retry-loop sleeps
        # and the "retry without thinking" path: when the pipeline
        # detects user cancellation it ``.set()``s the event, and the
        # engine's ``_call_api`` (and ``_wait_or_cancel`` helper) return
        # immediately rather than waiting out a 30s+ retry. ``None``
        # preserves the legacy behaviour (plain ``time.sleep``) so this
        # is purely additive.
        self._cancel_event = cancel_event
        # Audit 2026-09-01 CR-21: per-engine lock for the
        # read-modify-write of the backend's sampling attributes
        # (temperature / max_output_tokens / thinking_budget_tokens).
        # Multiple worker threads may call ``_apply_config_sampling_params``
        # concurrently; without this lock, two threads can stomp on
        # each other's temperature setting mid-flight, producing
        # nondeterministic output. See ``_apply_config_sampling_params``.
        self._sampling_lock = threading.Lock()
        # Stage toggles. Default: all on.
        for i in range(1, 6):
            self.config.setdefault(f"m3_stage_{i}", True)
        # Temperature overrides for stages that need more creative reasoning.
        self.config.setdefault("m3_temperature", 0.1)
        # Self-consistency: re-sample stage 4 N times and majority-vote.
        self.config.setdefault("m3_match_samples", 1)
        # Thinking budget for vision stages (more thinking for harder visual reasoning).
        self.config.setdefault("m3_thinking_budget", 1024)
        # audit 2026-07-31: the m3_temperature / m3_thinking_budget
        # keys were setdefault'd here but never READ anywhere — the
        # user's knobs were dead. Push both onto the backend so the
        # sampling parameters actually reach the API call.
        self._apply_config_sampling_params()

    def _apply_config_sampling_params(self) -> None:
        """Forward ``m3_temperature`` / ``m3_thinking_budget`` from
        the config dict onto the backend's sampling attributes.

        The backend classes (llm_backends) expose ``temperature``,
        ``max_output_tokens`` and ``thinking_budget_tokens`` fields;
        setting them here is the only place the config knobs are
        consumed (audit 2026-07-31: previously dead keys).

        Audit 2026-09-01 CR-21: the previous implementation wrote
        ``self.backend.temperature = ...`` / ``max_output_tokens = ...``
        directly on the shared backend instance. The M3Engine is
        invoked concurrently from per-panel worker threads (the
        ``_apply_m3_per_panel_species_id`` enrichment loop), so two
        panels running side-by-side could each set ``temperature`` to
        different values — the result was nondeterministic sampling
        (one panel would see temperature=0.4, the next would see
        temperature=0.7 mid-flight). Wrap each setter in
        ``self._sampling_lock`` so the read-modify-write of the
        backend's sampling attributes is atomic per-call.
        """
        with self._sampling_lock:
            temp = self.config.get("m3_temperature")
            if temp is not None and hasattr(self.backend, "temperature"):
                try:
                    self.backend.temperature = float(temp)
                except (TypeError, ValueError):
                    pass
            thinking = self.config.get("m3_thinking_budget")
            if thinking is not None and hasattr(self.backend, "thinking_budget_tokens"):
                try:
                    self.backend.thinking_budget_tokens = int(thinking)
                except (TypeError, ValueError):
                    pass
            max_out = self.config.get("m3_max_output_tokens")
            if max_out is not None and hasattr(self.backend, "max_output_tokens"):
                try:
                    self.backend.max_output_tokens = int(max_out)
                except (TypeError, ValueError):
                    pass
        # Skip stage-4 per-panel matching if caption parser found zero pairs.
        # Default True: when no caption pairs were extracted, M3 stage 4 has
        # no candidate species list to choose from, so its visual-only mode
        # tends to hallucinate. Skipping is safer. Set this to False to
        # enable visual-only morphology identification (lower confidence).
        # NOTE: this default MUST stay in sync with the read in
        # ``pipeline._apply_m3_stage4`` (which also defaults to True).
        self.config.setdefault("m3_skip_match_on_empty_caption", True)
        # Diagnostic dump: also save M3 raw output to this directory (None = off).
        self.config.setdefault("m3_diagnostic_dir", None)
        self._diagnostic_counter = 0
        # Lock for the "retry without thinking" path in _infer_text /
        # _infer_vision. The retry mutates ``backend.enable_thinking``
        # (a MiniMax-specific attribute) before the second call and
        # restores it afterwards. When multiple pipeline workers call
        # M3 concurrently, one thread's toggle can race another's
        # save/restore, leaving ``enable_thinking`` in the wrong state
        # for the first thread's original call.
        #
        # Round 9 (Bug-M3): use ``RLock`` (reentrant) instead of ``Lock``
        # so the save→flip→call→restore sequence can be held inside a
        # single critical section. The previous code released the lock
        # around ``infer_panel()`` (claiming a deadlock risk with
        # backends that re-enter _infer_vision), but that opened a
        # race window: another thread could flip ``enable_thinking``
        # in between, and the first thread's restore would overwrite
        # the other thread's setup. RLock sidesteps the deadlock
        # concern entirely (re-entry by the same thread is fine) and
        # the entire retry+call+restore now happens atomically.
        self._thinking_retry_lock = RLock()
        # audit 2026-08-01 (D2): ``_thinking_retry_lock`` only excludes
        # other *retries*; first attempts still read ``enable_thinking``
        # (inside the backend) with no synchronisation at all. The gate
        # below lets concurrent first calls run together but keeps them
        # out of the window where a retry has the flag flipped off.
        self._thinking_gate = _ThinkingFlagGate()

    # ------------------------------------------------------------------ helpers
    def _wait_or_cancel(self, seconds: float) -> bool:
        """Sleep ``seconds`` (or until ``self._cancel_event`` is set).

        audit 2026-08-01 (M16): thin wrapper used by the engine's retry
        loop so user cancellation can short-circuit a multi-second
        ``time.sleep``. Returns ``True`` if the event was set (caller
        should bail), ``False`` if the wait timed out normally.

        With no event (``self._cancel_event is None``) the behaviour
        is identical to the previous ``time.sleep(seconds)`` — backward
        compatible.
        """
        evt = self._cancel_event
        if evt is None:
            time.sleep(max(0.0, float(seconds)))
            return False
        # ``Event.wait(timeout=...)`` returns True iff the event was
        # set during the wait, False on timeout. ``max(0.0, ...)``
        # guards against a stray negative arg from upstream.
        return evt.wait(timeout=max(0.0, float(seconds)))

    def _call_api(
        self,
        kind: str,
        *args: Any,
        max_retries: int = 1,
        retry_wait: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """Wrapper around the backend API call with a cancel-aware retry loop.

        audit 2026-08-01 (M16): centralises the engine-side retry path
        so cancellation (via ``self._cancel_event``) can short-circuit
        the back-off between attempts. The first attempt always runs;
        any retry sleep uses ``_wait_or_cancel`` and aborts early when
        the event is set. ``kind`` is one of ``"text"`` / ``"vision"``
        and selects which backend method to invoke.

        This method exists so the engine owns a hook that the
        ``test_cancel_event_set_short_circuits_retry`` test can
        monkeypatch and observe. The concrete retry semantics for the
        MiniMax backends (status-code routing, jitter, etc.) live in
        ``llm_backends._call_api``; here we just want a place where
        ``self._cancel_event`` is honoured on every retry back-off.
        """
        if self.backend is None:
            return {"fallback_used": True, "error": "no backend"}
        if kind == "text":
            invoke = getattr(self.backend, "infer_text", None)
        elif kind == "vision":
            invoke = getattr(self.backend, "infer_panel", None)
        else:
            raise ValueError(f"M3Engine._call_api: unknown kind={kind!r}")
        if invoke is None:
            return {"fallback_used": True, "error": f"backend has no {kind} method"}
        attempt = 0
        last_exc: Exception | None = None
        while attempt < max(1, int(max_retries)):
            try:
                return invoke(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — surface to caller
                last_exc = exc
                # Don't sleep before the last attempt: a cancel that
                # arrives after the final try should propagate via
                # the exception, not be masked by an extra wait.
                if attempt + 1 >= max(1, int(max_retries)):
                    break
                if self._wait_or_cancel(retry_wait):
                    # Event set during the back-off — re-raise the
                    # last exception but with a clearer message so
                    # callers can distinguish cancel vs. genuine error.
                    raise PipelineCancelledError(
                        f"cancelled during {kind} retry (attempt {attempt + 1})"
                    ) from exc
                attempt += 1
        # Exhausted retries — surface the last exception.
        if last_exc is not None:
            raise last_exc
        return None

    # ------------------------------------------------------------------ stage 1
    def parse_caption(
        self,
        caption_text: str,
        lang: str | None = None,
    ) -> list[CaptionPair]:
        """Stage 1: caption text -> structured (label, species) pairs.

        Tries the LLM first; if the LLM returns nothing (rate-limited, low
        quality, model errors), falls back to a regex-based parser that
        handles the most common caption formats:
            "fig 1. Species A" / "figs 1-3. Species B" / "fig 1, 4. Species C"

        Phase 27: ``lang`` selects the system prompt. ``"ja"`` uses
        ``_PARSE_CAPTION_SYSTEM_JA`` (Japanese instructions, same JSON
        output schema); ``"zh"`` / ``"en"`` / ``None`` + detector returns
        ``"zh"`` use the existing Chinese prompt. ``None`` triggers
        auto-detection via ``_detect_caption_lang`` — Hiragana /
        Katakana / CJK characters in the caption text switch to JA.
        """
        if not self._stage_enabled(1) or not caption_text or not caption_text.strip():
            return []
        # Configurable: skip the LLM and go straight to the regex parser.
        # Useful for tests and for cost-sensitive runs where the regex is
        # accurate enough for the caption convention at hand.
        if self.config.get("m3_caption_regex_only", False):
            fallback = _regex_parse_caption(caption_text)
            if fallback:
                logger.info(
                    "Stage 1 parse_caption -> %d pairs (regex only, m3_caption_regex_only=True)",
                    len(fallback),
                )
            return fallback
        # Phase 27: language dispatch. Explicit ``lang`` from the caller
        # wins; otherwise auto-detect from the caption text. Only ``"ja"``
        # switches prompts — everything else keeps the legacy Chinese
        # prompt so English-only papers are unaffected.
        if lang is None:
            lang = _detect_caption_lang(caption_text)
        if lang == "ja":
            system_prompt = _PARSE_CAPTION_SYSTEM_JA
        else:
            system_prompt = _PARSE_CAPTION_SYSTEM
        prompt = (
            "请解析下列图版说明，输出严格的 JSON 数组（label->物种 配对列表）。"
            "\n\n[Caption]\n" + caption_text.strip() + "\n\n[输出 JSON]"
        )
        raw = self._infer_text(system_prompt, prompt)
        if not raw or raw.get("fallback_used"):
            return []
        try:
            data = _safe_json_loads(raw.get("raw_text", ""))
        except Exception as exc:
            logger.warning("Stage 1 JSON parse failed: %s", exc)
            return []
        pairs: list[CaptionPair] = []
        if not isinstance(data, list):
            return []
        for item in data:
            if not isinstance(item, dict):
                continue
            labels = item.get("labels") or []
            species = str(item.get("species") or "").strip()
            if not labels or not species:
                continue
            if isinstance(labels, str):
                # Audit 2026-09-01 BL-20: the LLM sometimes emits a
                # **comma-separated string** like "1-3, 5, 8". The
                # previous code fed the entire string into
                # ``_expand_label_range`` which only handles a single
                # "A-Z" / "1-9" range — the result was a single label
                # "1-3, 5, 8" that matched every panel on the plate,
                # inflating image-verified F1 by counting one
                # caption-pair as several. Split on commas first, then
                # expand each segment individually.
                labels = [
                    lab
                    for seg in labels.split(",")
                    for lab in _expand_label_range(seg)
                    if lab.strip()
                ]
            elif not isinstance(labels, list):
                continue
            labels = [str(x).strip() for x in labels if str(x).strip()]
            if not labels:
                continue
            try:
                conf = float(item.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            # audit 2026-08-19 Phase 1d (B-8): the LLM may emit
            # species without a trailing period on cf./aff. ("Triactoma
            # cf kamoense") or with the "(?)" uncertainty marker
            # ("Archaeodictyomitra (?) sp."). The regex fallback path
            # already calls _normalize_species before emitting
            # CaptionPair, but the LLM path was leaving the species
            # string verbatim — causing mismatches against gold.
            # Normalize both species and modifier so the LLM path
            # behaves consistently with the regex fallback.
            normalized_species = _normalize_species(species)
            if normalized_species:
                species = normalized_species
            modifier_raw = str(item.get("modifier") or "").strip()
            # ``_normalize_species`` returns ``None`` when the input
            # reduces to whitespace-only (e.g. literal "(?)" — the
            # "(?)" uncertainty marker is stripped entirely). Map
            # that to an empty modifier so the LLM path behaves
            # consistently with the regex fallback (which never
            # emits a modifier containing "(?)").
            normalized_modifier = _normalize_species(modifier_raw)
            if normalized_modifier is None:
                modifier_raw = ""
            else:
                modifier_raw = normalized_modifier
            pairs.append(
                CaptionPair(
                    labels=labels,
                    species=species,
                    modifier=modifier_raw,
                    confidence=max(0.0, min(1.0, conf)),
                    notes=str(item.get("notes") or "").strip(),
                    raw_text=str(item.get("raw_text") or "").strip(),
                )
            )
        logger.info("Stage 1 parse_caption -> %d pairs", len(pairs))
        if pairs:
            return pairs
        # Fallback: regex-based caption parser. Avoids the cost (and flakiness)
        # of an LLM call when the caption is well-formed enough to parse
        # deterministically. Handles the convention used in most OA radiolarian
        # papers: ``figs 1-3. Species A. figs 4-5. Species B.``.
        fallback = _regex_parse_caption(caption_text)
        if fallback:
            logger.info("Stage 1 parse_caption -> %d pairs (regex fallback)", len(fallback))
        return fallback

    # ------------------------------------------------------------------ stage 2
    def classify_plate(self, plate_image: Image.Image) -> PlateClassification:
        """Stage 2: plate image -> semantic classification (filter non-plate)."""
        if not self._stage_enabled(2) or plate_image is None:
            return PlateClassification()
        prompt = (
            "观察这张图像，判断它是否是放射虫图版。"
            "严格输出 JSON（is_radiolarian_plate / image_type / panel_count_estimate / "
            "specimen_count_estimate / quality / dominant_taxa / reasoning）。"
        )
        raw = self._infer_vision(_CLASSIFY_PLATE_SYSTEM, prompt, plate_image)
        if not raw or raw.get("fallback_used"):
            return PlateClassification()
        try:
            data = _safe_json_loads(raw.get("raw_text", ""))
        except Exception as exc:
            logger.warning("Stage 2 JSON parse failed: %s", exc)
            return PlateClassification()
        if not isinstance(data, dict):
            return PlateClassification()
        cls = PlateClassification(
            is_radiolarian_plate=_safe_bool(data.get("is_radiolarian_plate"), default=True),
            image_type=str(data.get("image_type") or "micrograph"),
            panel_count_estimate=_safe_int(data.get("panel_count_estimate")),
            specimen_count_estimate=_safe_int(data.get("specimen_count_estimate")),
            quality=str(data.get("quality") or "ok"),
            dominant_taxa=[str(x) for x in (data.get("dominant_taxa") or []) if str(x).strip()],
            reasoning=str(data.get("reasoning") or "").strip(),
        )
        logger.info(
            "Stage 2 classify_plate -> is_radiolarian=%s type=%s panels≈%s taxa=%s",
            cls.is_radiolarian_plate,
            cls.image_type,
            cls.panel_count_estimate,
            cls.dominant_taxa,
        )
        return cls

    # ------------------------------------------------------------------ stage 3
    def segment_panels(
        self, plate_image: Image.Image, hint_count: int | None = None
    ) -> list[PanelBox]:
        """Stage 3: plate image -> panel bboxes + visible labels."""
        if not self._stage_enabled(3) or plate_image is None:
            return []
        w, h = plate_image.size
        hint = f"\n\n提示：预期 panel 数量约 {hint_count}。\n" if hint_count else ""
        prompt = (
            f"观察这张放射虫图版（{w}x{h} 像素），识别每个独立 panel 的边界框 + 可见字母标签。"
            f"{hint}"
            "严格输出 JSON 数组（panel_id / bbox / visible_label / morphology / confidence）。"
        )
        raw = self._infer_vision(_SEGMENT_PANELS_SYSTEM, prompt, plate_image)
        if not raw or raw.get("fallback_used"):
            return []
        try:
            data = _safe_json_loads(raw.get("raw_text", ""))
        except Exception as exc:
            logger.warning("Stage 3 JSON parse failed: %s", exc)
            return []
        if not isinstance(data, list):
            return []
        panels: list[PanelBox] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            bbox = _coerce_bbox(item.get("bbox"), w, h)
            if bbox is None:
                continue
            try:
                conf = float(item.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            panels.append(
                PanelBox(
                    panel_id=str(item.get("panel_id") or f"P{len(panels) + 1}"),
                    bbox=bbox,
                    # Phase 38: visible_label may be a list (e.g. M3
                    # returns ["A", "B"] when it sees two labels).
                    # Previously ``str(item.get("visible_label")).strip()``
                    # would produce the Python repr ``"['A', 'B']"``
                    # which then broke panel_id assignment downstream
                    # because every panel ended up with a list-repr
                    # "label". Join list values with a comma; coerce
                    # numbers / None to a clean string.
                    visible_label=_coerce_label(item.get("visible_label")),
                    morphology=str(item.get("morphology") or "").strip(),
                    confidence=max(0.0, min(1.0, conf)),
                )
            )
        # Sort top-to-bottom, left-to-right and re-assign sequential ids if none.
        panels.sort(key=lambda p: (p.bbox[1], p.bbox[0]))
        # Two-pass id assignment so duplicates from the LLM (e.g. it
        # emitted "P1" three times) get unique sequential ids. The
        # previous one-pass loop only renamed empty / placeholder ids
        # and would happily leave three panels all called "P1", which
        # broke downstream lookups (any dict keyed on panel_id silently
        # collapsed those rows together).
        used_ids: set[str] = set()
        for i, p in enumerate(panels, start=1):
            cur = (p.panel_id or "").strip()
            if not cur or cur in used_ids:
                # Find a free Pn name. We start from i so the natural
                # case (no collisions) keeps the canonical numbering.
                k = i
                while f"P{k}" in used_ids:
                    k += 1
                p.panel_id = f"P{k}"
            used_ids.add(p.panel_id)
        logger.info("Stage 3 segment_panels -> %d panels", len(panels))
        return panels

    # ------------------------------------------------------------------ stage 4
    def match_panel(
        self,
        panel_image: Image.Image,
        caption_pairs: list[CaptionPair],
        caption_text: str = "",
        suggested_label: str | None = None,
    ) -> PanelMatch:
        """Stage 4: panel image + caption pairs -> (label, species, confidence).

        Two modes:
          - With caption pairs: standard M3 matcher with constrained candidates.
          - Visual-only (no caption): M3 does morphology-based identification
            with a separate prompt and conservative confidence.
        """
        if not self._stage_enabled(4) or panel_image is None:
            return PanelMatch(
                panel_id="?",
                label=None,
                species=None,
                confidence=0.0,
                reasoning="stage 4 disabled",
            )
        visual_only = not caption_pairs
        if visual_only:
            system_prompt = _MATCH_PANEL_SYSTEM_VISUAL_ONLY
            hint = f"\n提示标签（来自 M3 阶段 3）：{suggested_label}\n" if suggested_label else ""
            caption_block = (
                f"\n[完整图说（仅供参考，可能为空）]\n{caption_text.strip()}\n"
                if caption_text
                else ""
            )
            prompt = (
                f"{caption_block}"
                f"{hint}\n"
                "模式：visual-only。无候选物种清单，请完全依靠形态特征鉴定。\n"
                "请为该 panel 选出最可能的属/种，严格输出 JSON。"
            )
        else:
            system_prompt = _MATCH_PANEL_SYSTEM
            pairs_json = json.dumps(
                [p.to_dict() for p in caption_pairs], ensure_ascii=False, indent=2
            )
            hint = f"\n提示标签（来自 M3 阶段 3）：{suggested_label}\n" if suggested_label else ""
            caption_block = f"\n[完整图说]\n{caption_text.strip()}\n" if caption_text else ""
            prompt = (
                "[候选配对（caption 解析）]\n"
                f"{pairs_json}\n"
                f"{caption_block}"
                f"{hint}\n"
                "请为该 panel 选出最可能的 label + 物种，严格输出 JSON。"
            )
        # Optional self-consistency: sample N times at higher temperature and vote.
        n_samples = max(1, int(self.config.get("m3_match_samples", 1)))
        results: list[dict[str, Any]] = []
        last_error: str | None = None
        last_raw_kept: dict[str, Any] | None = None
        for _ in range(n_samples):
            raw = self._infer_vision(system_prompt, prompt, panel_image)
            if not raw or raw.get("fallback_used"):
                # Capture the backend error so we can distinguish a real API
                # failure from "M3 said this isn't a radiolarian". The pipeline
                # uses raw["error"] to route through the FallbackHandler.
                if raw and raw.get("error"):
                    last_error = str(raw.get("error"))
                continue
            try:
                data = _safe_json_loads(raw.get("raw_text", ""))
            except Exception:
                continue
            if isinstance(data, dict):
                results.append(data)
                # Accumulate backend telemetry across all self-consistency
                # samples so the winning PanelMatch reports the *total* cost
                # and merged token usage, not just the last sample's worth
                # (M13: previously ``last_raw_kept`` only took the final
                # sample, undercounting actual spend in ``/system/llm-
                # status`` for multi-sample self-consistency). ``cost_cny``
                # is summed (each call is a separate request); ``usage``
                # is merged with int fields summed and list fields
                # concatenated so per-call breakdowns are preserved.
                if last_raw_kept is None:
                    last_raw_kept = dict(raw)
                else:
                    # Phase 55 audit HIGH-6 fix: handle mixed-type usage values.
                    # The old code assumed merged[k] is always the same type as v,
                    # but a previous sample may have written a float/int while
                    # the new sample writes a list, or vice-versa. We now check
                    # both existing and new types before merging, and overwrite
                    # rather than crash when types mismatch.
                    try:
                        old_cost = last_raw_kept.get("cost_cny")
                        new_cost = raw.get("cost_cny")
                        last_raw_kept["cost_cny"] = (
                            float(old_cost) if old_cost is not None else 0.0
                        ) + (float(new_cost) if new_cost is not None else 0.0)
                    except (TypeError, ValueError):
                        # One side is not numeric — give up on accumulating cost.
                        # Don't silently overwrite with 0 which would hide the field.
                        pass
                    prev_usage = last_raw_kept.get("usage")
                    new_usage = raw.get("usage")
                    if isinstance(prev_usage, dict) and isinstance(new_usage, dict):
                        merged: dict[str, Any] = dict(prev_usage)
                        for k, v in new_usage.items():
                            existing = merged.get(k)
                            if isinstance(v, list) and isinstance(existing, list):
                                # Both are lists — concatenate per-call breakdowns.
                                merged[k] = existing + v
                            elif (
                                isinstance(v, (int, float))
                                and type(v) is not bool
                                and isinstance(existing, (int, float))
                                and type(existing) is not bool
                            ):
                                # Both are numeric (excluding bool) — sum token counts.
                                # bool is a subclass of int so isinstance(True, int)
                                # is True; we must exclude it explicitly or bool
                                # flags get summed as integers (True+True=2).
                                merged[k] = existing + v
                            else:
                                # Type mismatch: don't crash, just take the newer value.
                                # This is defensible because mixed-type usage keys
                                # indicate a backend change mid-self-consistency run,
                                # which is not a supported configuration.
                                merged[k] = v
                        last_raw_kept["usage"] = merged
                    elif isinstance(new_usage, dict):
                        last_raw_kept["usage"] = dict(new_usage)
        if not results:
            # Two distinct failure modes that the pipeline treats very
            # differently. The previous code conflated them by setting
            # ``is_radiolarian=False`` for both:
            #   1. ``last_error`` is set → real API / runtime failure
            #      (network blip, M3 rejected the image, quota exceeded,
            #      ...). Pipeline routes this through the FallbackHandler
            #      so the user can retry or pick a local fallback.
            #   2. ``last_error`` is None → M3 returned a response but
            #      nothing parseable survived ``_safe_json_loads`` across
            #      N self-consistency samples. This is NOT the same as
            #      "M3 said this isn't a radiolarian" — M3 may simply
            #      have produced malformed JSON. Leaving
            #      ``is_radiolarian=True`` (the PanelMatch default) plus
            #      a ``raw["unparseable"]`` flag lets the pipeline treat
            #      this as a soft fallback (rule-based species preserved)
            #      instead of a hard rejection.
            if last_error:
                return PanelMatch(
                    panel_id="?",
                    label=None,
                    species=None,
                    confidence=0.0,
                    reasoning=f"M3 error: {last_error}",
                    is_radiolarian=False,
                    raw={"error": last_error},
                )
            return PanelMatch(
                panel_id="?",
                label=None,
                species=None,
                confidence=0.0,
                reasoning="M3 returned no parseable output",
                is_radiolarian=True,
                raw={"unparseable": True},
            )

        # Majority-vote on (label, species); keep best confidence.
        # Force both elements to be hashable strings (or None). The LLM
        # occasionally returns a list/dict for ``species`` (e.g. when it
        # decides the panel covers multiple species and emits an array);
        # ``votes.setdefault(key, [])`` would then raise
        # ``TypeError: unhashable type``.
        def _hashable(v: Any) -> str | None:
            if v is None:
                return None
            if isinstance(v, (str, int, float, bool)):
                return str(v).strip() or None
            try:
                return json.dumps(v, ensure_ascii=False, sort_keys=True)
            except Exception:
                return repr(v)

        votes: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
        for r in results:
            key = (_hashable(r.get("label")), _hashable(r.get("species")))
            votes.setdefault(key, []).append(r)
        # m14: tie-break votes deterministically. ``max(..., key=len)`` alone
        # picks the first-inserted group when two groups have the same
        # vote count — that depends on dict insertion order (i.e. the
        # order in which the LLM emitted samples) and is not reproducible
        # across runs. Add a secondary key that prefers (1) the group
        # with the most votes *and* the highest single-sample confidence
        # within it, so the tie is broken by a stable signal.
        best_key, best_group = max(
            votes.items(),
            key=lambda kv: (
                len(kv[1]),
                max(
                    (_safe_float(r.get("confidence")) for r in kv[1]),
                    default=0.0,
                ),
            ),
        )
        best = max(best_group, key=lambda r: _safe_float(r.get("confidence")))
        conf = _safe_float(best.get("confidence"))
        # If there's a runner-up, surface as alternative.
        runner_up: str | None = None
        if len(votes) > 1:
            sorted_groups = sorted(votes.values(), key=lambda g: -len(g))
            if len(sorted_groups) > 1:
                ru = max(sorted_groups[1], key=lambda r: _safe_float(r.get("confidence")))
                ru_sp = str(ru.get("species") or "").strip() or None
                if ru_sp and ru_sp != best.get("species"):
                    runner_up = ru_sp
        return PanelMatch(
            panel_id="?",
            label=(str(best.get("label") or "").strip() or None),
            species=(str(best.get("species") or "").strip() or None),
            confidence=max(0.0, min(1.0, conf)),
            reasoning=str(best.get("reasoning") or "").strip(),
            alternative=runner_up,
            is_radiolarian=_safe_bool(best.get("is_radiolarian"), default=True),
            raw={
                "votes": len(results),
                "agreement": len(best_group) / max(1, len(results)),
                **_telemetry_subset(last_raw_kept),
            },
        )

    # ------------------------------------------------------------------ stage 5
    def critique_matches(
        self,
        plate_image: Image.Image,
        matches: list[PanelMatch],
        caption_text: str = "",
        caption_pairs: list[CaptionPair] | None = None,
    ) -> list[Critique]:
        """Stage 5: full plate + per-panel matches -> critique / override list."""
        if not self._stage_enabled(5) or plate_image is None or not matches:
            return []
        matches_json = json.dumps(
            [
                {
                    "panel_id": m.panel_id,
                    "label": m.label,
                    "species": m.species,
                    "confidence": m.confidence,
                    "reasoning": m.reasoning,
                }
                for m in matches
            ],
            ensure_ascii=False,
            indent=2,
        )
        cap_block = f"\n[图说]\n{caption_text.strip()}\n" if caption_text else ""
        pairs_block = ""
        if caption_pairs:
            pairs_block = (
                "\n[图说解析出的候选配对]\n"
                + json.dumps([p.to_dict() for p in caption_pairs], ensure_ascii=False, indent=2)
                + "\n"
            )
        prompt = (
            "[当前配对结果]\n"
            f"{matches_json}\n"
            f"{cap_block}"
            f"{pairs_block}\n"
            "请逐个 panel 评判（agree / disagree / uncertain），必要时给出 suggested_species。严格 JSON 数组。"
        )
        raw = self._infer_vision(_CRITIQUE_SYSTEM, prompt, plate_image)
        if not raw or raw.get("fallback_used"):
            return []
        try:
            data = _safe_json_loads(raw.get("raw_text", ""))
        except Exception as exc:
            logger.warning("Stage 5 JSON parse failed: %s", exc)
            return []
        if not isinstance(data, list):
            return []
        out: list[Critique] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            # M14: drop critiques with empty/missing panel_id. The previous
            # ``str(item.get("panel_id") or "")`` silently turned None into
            # "" and then matched *every* panel in downstream code — that
            # cascade-applied the verdict to the whole plate. Skip empty
            # ids so the critique either names a panel or is rejected.
            if not item.get("panel_id"):
                continue
            verdict = str(item.get("verdict") or "").strip().lower()
            if verdict not in {"agree", "disagree", "uncertain"}:
                verdict = "uncertain"
            try:
                conf = float(item.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            out.append(
                Critique(
                    panel_id=str(item.get("panel_id") or ""),
                    verdict=verdict,
                    suggested_species=(str(item.get("suggested_species") or "").strip() or None),
                    confidence=max(0.0, min(1.0, conf)),
                    reasoning=str(item.get("reasoning") or "").strip(),
                )
            )
        logger.info("Stage 5 critique_matches -> %d critiques", len(out))
        return out

    # ------------------------------------------------------------------ apply
    @staticmethod
    def apply_critiques(
        matches: list[PanelMatch],
        critiques: list[Critique],
        override_threshold: float = 0.6,
    ) -> list[PanelMatch]:
        """Apply critiques to matches in place. Returns the same list (mutated).

        When multiple matches share a ``panel_id`` (e.g. OCR duplicates), each
        critique is applied to the FIRST unmatched match with that id, so no
        match receives a critique intended for a different panel. The previous
        ``by_id`` dict approach silently dropped duplicate critiques and could
        apply the wrong critique to the wrong match.
        """
        # Build a multimap: panel_id -> list of critiques (in order).
        by_id: dict[str, list[Critique]] = {}
        for c in critiques:
            by_id.setdefault(c.panel_id, []).append(c)
        # Track which critique indices have been consumed so each critique
        # is applied at most once.
        consumed: dict[str, int] = {}
        for m in matches:
            cands = by_id.get(m.panel_id)
            if not cands:
                continue
            pos = consumed.get(m.panel_id, 0)
            if pos >= len(cands):
                continue
            consumed[m.panel_id] = pos + 1
            c = cands[pos]
            if c.verdict == "agree":
                m.raw.setdefault("critique", {"verdict": "agree", "confidence": c.confidence})
                continue
            if c.suggested_species and c.confidence >= override_threshold:
                m.raw["critique"] = {
                    "verdict": c.verdict,
                    "from": m.species,
                    "to": c.suggested_species,
                    "confidence": c.confidence,
                    "reasoning": c.reasoning,
                }
                m.species = c.suggested_species
                # Lower confidence since we overrode
                m.confidence = min(m.confidence, max(0.3, c.confidence))
        return matches

    # ------------------------------------------------------------------ helpers
    def _stage_enabled(self, n: int) -> bool:
        return bool(self.config.get(f"m3_stage_{n}", True))

    def _make_telemetry(
        self,
        *,
        start: float,
        llm_error: str | None = None,
    ) -> dict[str, Any]:
        """Build the ``_telemetry`` dict for an M3 call result.

        Phase 4E Task 2 (audit 2026-08-19): every M3 result now carries
        a ``_telemetry`` sub-dict with:

        * ``model``         - ``backend.model`` string (the model that
                              produced the response).
        * ``prompt_version``- the ``PROMPT_REGISTRY_VERSION`` string
                              so audit can correlate output drift with
                              a known prompt revision.
        * ``latency_ms``    - wall-clock milliseconds since ``start``
                              was sampled, rounded to int.
        * ``timestamp``     - ISO-8601 UTC time the call completed.
        * ``llm_error``     - OPTIONAL short code for the failure
                              reason (``"auth"`` / ``"rate_limit"`` /
                              ``"timeout"`` / ``"parse"`` / ``"other"``);
                              only present when the call did NOT
                              return a usable response.

        This helper centralises the field shape so success and failure
        paths in ``_infer_text`` / ``_infer_vision`` always stamp the
        same keys — including the ones downstream consumers expect
        (e.g. ``/system/llm-status`` dashboards).
        """
        tel: dict[str, Any] = {
            "model": getattr(self.backend, "model", None) if self.backend is not None else None,
            "prompt_version": get_prompt_registry_version(),
            "latency_ms": int(max(0, (time.time() - float(start)) * 1000)),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if llm_error is not None:
            tel["llm_error"] = llm_error
        return tel

    def _infer_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.backend is None:
            return {
                "fallback_used": True,
                "error": "no backend",
                "_telemetry": self._make_telemetry(
                    start=time.time(),
                    llm_error="other",
                ),
            }
        # Phase 55 audit CRITICAL-1 fix: snapshot enable_thinking BEFORE any
        # concurrent worker can mutate it. The previous code read
        # ``self.backend.enable_thinking`` inside the retry condition below,
        # which races with another thread's retry path that flips the flag
        # to False. By capturing it now (single-threaded entry point) and
        # using the snapshot in both the first-call and the retry decision,
        # each call consistently uses the value that was active when the
        # call started — no more silent quality degradation or doubled cost
        # from a race mid-flight.
        #
        # audit 2026-08-01 (D2): the snapshot alone wasn't enough — the
        # backend re-reads ``enable_thinking`` itself when it builds the
        # request, so the read gate keeps this first attempt out of any
        # concurrent retry's flip window.
        with self._thinking_gate.read():
            enable_thinking_snapshot = getattr(self.backend, "enable_thinking", False)
            start = time.time()
            try:
                res = self.backend.infer_text(system_prompt=system_prompt, user_prompt=user_prompt)
            except FallbackRecommendedError:
                # audit 2026-07-31: mirror _infer_vision — the backend
                # asked us to switch to the configured fallback; swallowing
                # it here meant the pipeline never saw the recommendation
                # and the fallback feature stayed dead.
                raise
            except LLMAuthenticationError as exc:
                # Phase 4E Task 1: 401/403 — operator must rotate the API
                # key. Surface as error-level so dashboards flag it
                # immediately.
                logger.error("LLM auth failed: %s", exc)
                return {
                    "fallback_used": True,
                    "error": str(exc),
                    "_telemetry": self._make_telemetry(start=start, llm_error="auth"),
                }
            except LLMRateLimitError as exc:
                # Phase 4E Task 1: 429 — transient, may succeed on retry.
                logger.warning("LLM rate limited: %s", exc)
                return {
                    "fallback_used": True,
                    "error": str(exc),
                    "_telemetry": self._make_telemetry(start=start, llm_error="rate_limit"),
                }
            except TimeoutError as exc:
                # Phase 4E Task 1: socket / upstream timeout — load-related.
                logger.warning("LLM timeout: %s", exc)
                return {
                    "fallback_used": True,
                    "error": str(exc),
                    "_telemetry": self._make_telemetry(start=start, llm_error="timeout"),
                }
            except (LLMSchemaError, json.JSONDecodeError, ValueError) as exc:
                # Phase 4E Task 1: JSONDecodeError + ValueError come
                # from _safe_json_loads when the response body isn't
                # parseable; LLMSchemaError covers schema violations
                # after a successful parse.
                logger.warning("LLM parse error: %s", exc)
                return {
                    "fallback_used": True,
                    "error": str(exc),
                    "_telemetry": self._make_telemetry(start=start, llm_error="parse"),
                }
            except Exception as exc:
                logger.exception("M3 text inference failed")
                return {
                    "fallback_used": True,
                    "error": str(exc),
                    "_telemetry": self._make_telemetry(start=start, llm_error="other"),
                }
        # Stamp telemetry on the successful first-attempt response so
        # downstream code can see model / prompt_version / latency /
        # timestamp without re-walking the raw dict.
        res = dict(res)  # copy so we don't mutate the backend's dict
        res["_telemetry"] = self._make_telemetry(start=start)
        # Retry without thinking if the response is empty.
        if (
            self.config.get("m3_retry_without_thinking", True)
            and (res.get("fallback_used") or not (res.get("raw_text") or "").strip())
            and enable_thinking_snapshot
        ):
            logger.info("M3 text returned empty; retrying with thinking disabled")
            with self._thinking_retry_lock, self._thinking_gate.write():
                saved = self.backend.enable_thinking
                try:
                    self.backend.enable_thinking = False
                    res2 = self.backend.infer_text(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                except Exception as exc:
                    logger.warning("M3 text retry failed: %s", exc)
                    res2 = res
                finally:
                    self.backend.enable_thinking = saved
            if (res2.get("raw_text") or "").strip():
                res = dict(res2)
                # Phase 4E: re-stamp telemetry to reflect the total
                # wall-clock (first attempt + retry sleep + retry).
                res["_telemetry"] = self._make_telemetry(start=start)
        return res

    def _infer_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Image.Image,
        extra_image: Image.Image | None = None,
    ) -> dict[str, Any]:
        """Vision inference helper with retry-without-thinking.

        Parameters
        ----------
        system_prompt : str
        user_prompt : str
        image : PIL.Image.Image
            The primary image (e.g. SEM plate).
        extra_image : PIL.Image.Image | None, default ``None``
            Audit M-14: optional SECOND image (e.g. strat column /
            paleogeographic map). Forwarded to ``backend.infer_panel``
            via the ``extra_image`` keyword argument. Backends that
            support multi-image (e.g. ``MiniMaxM3Backend``) receive
            BOTH images as separate content blocks; single-image
            backends (e.g. ``LlamaCppGemmaBackend``) silently drop the
            second image after recording an explanatory prompt note.
        """
        if self.backend is None:
            return {
                "fallback_used": True,
                "error": "no backend",
                "_telemetry": self._make_telemetry(
                    start=time.time(),
                    llm_error="other",
                ),
            }
        # Phase 55 audit CRITICAL-1 fix: snapshot enable_thinking BEFORE any
        # concurrent worker can mutate it (same pattern as _infer_text).
        # The snapshot is used for both the first-call decision (pass to
        # backend) and the retry condition — ensuring consistent behaviour
        # throughout the lifetime of this call regardless of other workers.
        #
        # audit 2026-08-01 (D2): hold the gate's read side across the call
        # so this (paid) first attempt cannot land inside another worker's
        # retry window, where ``enable_thinking`` is flipped off.
        with self._thinking_gate.read():
            enable_thinking_snapshot = getattr(self.backend, "enable_thinking", False)
            start = time.time()
            # First attempt — with thinking enabled (the default).
            try:
                res = self.backend.infer_panel(
                    panel_image=image,
                    caption_text="",  # we put context in user_prompt
                    ocr_labels=[],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    extra_image=extra_image,
                )
            except FallbackRecommendedError:
                # Phase 61 Plan 4 (Bug 4.10): FallbackRecommendedError carries
                # ``recommended_backend`` - let it propagate to the pipeline
                # so it can switch to the configured fallback backend.
                raise
            except LLMAuthenticationError as exc:
                # Phase 4E Task 1: 401/403 — operator must rotate the
                # API key.
                logger.error("LLM auth failed: %s", exc)
                return {
                    "fallback_used": True,
                    "error": str(exc),
                    "_telemetry": self._make_telemetry(start=start, llm_error="auth"),
                }
            except LLMRateLimitError as exc:
                logger.warning("LLM rate limited: %s", exc)
                return {
                    "fallback_used": True,
                    "error": str(exc),
                    "_telemetry": self._make_telemetry(start=start, llm_error="rate_limit"),
                }
            except TimeoutError as exc:
                logger.warning("LLM timeout: %s", exc)
                return {
                    "fallback_used": True,
                    "error": str(exc),
                    "_telemetry": self._make_telemetry(start=start, llm_error="timeout"),
                }
            except (LLMSchemaError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("LLM parse error: %s", exc)
                return {
                    "fallback_used": True,
                    "error": str(exc),
                    "_telemetry": self._make_telemetry(start=start, llm_error="parse"),
                }
            except Exception as exc:
                logger.exception("M3 vision inference failed")
                return {
                    "fallback_used": True,
                    "error": str(exc),
                    "_telemetry": self._make_telemetry(start=start, llm_error="other"),
                }
        # Stamp telemetry on the first-attempt success path so callers
        # see model / prompt_version / latency / timestamp without
        # re-walking the raw dict. Phase 4E Task 2.
        res = dict(res)  # don't mutate the backend's returned dict
        res["_telemetry"] = self._make_telemetry(start=start)
        self._maybe_dump_diagnostic(image, system_prompt, user_prompt, res)
        # Retry without thinking if the first attempt produced no text
        # (known M3 issue when thinking exhausts the output budget).
        # P2-10 fix: skip retry if thinking block has content — the valid
        # structured output may be in the thinking block and should not be
        # discarded by retrying without thinking.
        if (
            self.config.get("m3_retry_without_thinking", True)
            and (res.get("fallback_used") or not (res.get("raw_text") or "").strip())
            and enable_thinking_snapshot
            and not (res.get("thinking") or "").strip()
        ):
            logger.info("M3 returned empty text; retrying with thinking disabled")
            # Round 9 (Bug-M3): hold the RLock for the entire
            # save → flip → call → restore sequence. RLock is reentrant
            # so a backend that re-enters ``_infer_vision`` (e.g. a
            # custom subclass that calls M3 again inside its handler)
            # won't deadlock — the same thread can re-acquire the
            # lock cleanly. The whole retry is now atomic from the
            # perspective of other workers: no other thread can flip
            # ``enable_thinking`` in between our save and restore.
            #
            # audit 2026-08-01 (D2): the gate's write side additionally
            # keeps *first* attempts on other workers out of this window.
            with self._thinking_retry_lock, self._thinking_gate.write():
                saved = self.backend.enable_thinking
                self.backend.enable_thinking = False
                try:
                    res2 = self.backend.infer_panel(
                        panel_image=image,
                        caption_text="",
                        ocr_labels=[],
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        extra_image=extra_image,
                    )
                except Exception as exc:
                    logger.warning("M3 retry without thinking failed: %s", exc)
                    res2 = res
                finally:
                    self.backend.enable_thinking = saved
            if (res2.get("raw_text") or "").strip():
                res = dict(res2)
                # Phase 4E: re-stamp telemetry to reflect the total
                # wall-clock (first attempt + retry sleep + retry).
                res["_telemetry"] = self._make_telemetry(start=start)
        return res

    # ------------------------------------------------------------- stage 6 (geo)

    def extract_geology(
        self,
        image: Image.Image,
        caption: str,
        figure_type: str,
        paper_id: str,
        figure_id: str,
    ) -> list[dict[str, Any]]:
        """Run multi-modal MiniMax-M3 vision extraction on a figure image.

        Returns a list of dicts shaped like ``GeologyLinkRecord`` so the
        caller can append them straight into ``panel.metadata.geology_links``.
        Skipped silently when ``figure_type`` has no prompt registered or
        the image is too small to be meaningful.

        Parameters
        ----------
        image : PIL.Image.Image
            Figure image (RGB or convertible).
        caption : str
            Figure caption text. Always included in the prompt.
        figure_type : str
            One of ``plate``, ``range_chart``, ``map``, ``paleogeographic_map``,
            ``stratigraphic_column``, ``litholog_column``. ``extract_geology``
            uses ``SECTION_TYPE_BY_FIGURE`` to fill the ``section_type`` field.
        paper_id, figure_id : str
            Provenance IDs; stamped into ``evidence_text`` so downstream
            audit can trace each link back to its source.

        Notes
        -----
        Range-chart vision extraction is intentionally NOT routed through
        this method (see ``range_chart_extractor.extract_range_chart()``)
        because the range-chart schema carries richer per-species / per-
        section information than this generic schema.
        """
        prompt_key = f"{figure_type}_geo"
        if prompt_key not in PROMPT_REGISTRY:
            return []
        # Skip tiny images — MiniMax-M3 vision on a 16×16 thumbnail is
        # pure noise and burns cost without producing real signal.
        # Audit Bug 10: narrow the except to AttributeError/TypeError so
        # unrelated exceptions in the size check are not silently
        # swallowed. A non-PIL image raises AttributeError (no .width);
        # a PIL subclass with a buggy .width descriptor raises TypeError.
        try:
            if image.width < 32 or image.height < 32:
                return []
        except (AttributeError, TypeError):
            return []

        system_prompt = PROMPT_REGISTRY[prompt_key]
        user_prompt = (
            f"Paper: {paper_id}\nFigure: {figure_id}\n\n"
            f"Caption:\n{caption or '(no caption)'}\n\n"
            "Return strict JSON only, no markdown fences."
        )

        res = self._infer_vision(system_prompt, user_prompt, image)
        if res.get("fallback_used"):
            return []
        raw_text = res.get("raw_text") or ""
        try:
            parsed = _safe_json_loads(raw_text)
        except ValueError as exc:
            # Malformed JSON (truncated, no balanced braces) — log and
            # return [] rather than propagating. The caller treats this
            # the same as fallback_used.
            logger.warning(
                "extract_geology: failed to parse JSON for %s/%s: %s",
                paper_id,
                figure_id,
                exc,
            )
            return []
        if not isinstance(parsed, dict):
            logger.warning(
                "extract_geology: backend returned non-dict JSON for %s/%s",
                paper_id,
                figure_id,
            )
            return []
        geo_list = parsed.get("geo")
        if not isinstance(geo_list, list):
            return []

        section_type = SECTION_TYPE_BY_FIGURE.get(figure_type, "figure_caption")
        out: list[dict[str, Any]] = []

        # --- Phase X: parse global geo entries (backward-compatible) ---
        for item in geo_list:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            # Audit 2026-08-19 Phase 2b M-12: strip LLM-hallucinated
            # extras so they never reach panel.metadata.geology_links.
            _apply_geo_whitelist(item)
            # Phase 6D audit 2026-08-19 NIT-3: auto-swap ma_top / ma_base
            # when the LLM emits an inverted range. Run BEFORE the strict
            # null-on-violation check so the swap happens first and the
            # data is preserved.
            _normalize_ma_pair(item)
            # Audit 2026-08-19 Phase 2b M-13: enforce ma_top < ma_base.
            _validate_ma_range(item)
            item.setdefault("section_type", section_type)
            # Stamp provenance into evidence_text for audit. Only override
            # if not already set by the model.
            if not item.get("evidence_text"):
                item["evidence_text"] = (
                    f"geo_vision[{figure_type}]: paper={paper_id} "
                    f"figure={figure_id} conf={item.get('confidence')}"
                )
            out.append(item)

        # --- Phase X: parse per-locality entries for paleogeographic maps ---
        # Each locality carries its own species + coords + age, so it becomes
        # a fully-independent GeologyLinkRecord with link_source="geo_vision_point".
        if figure_type == "paleogeographic_map":
            localities = parsed.get("localities")
            if isinstance(localities, list):
                for loc in localities:
                    if not isinstance(loc, dict):
                        continue
                    loc = dict(loc)
                    # Audit 2026-08-19 Phase 2b M-12/M-13
                    _apply_geo_whitelist(loc)
                    # Phase 6D audit 2026-08-19 NIT-3: auto-swap before
                    # the strict null-on-violation check.
                    _normalize_ma_pair(loc)
                    _validate_ma_range(loc)
                    species = loc.get("species")
                    if not species:
                        # Skip entries without a species name — no useful link.
                        continue
                    evidence = (
                        loc.get("evidence")
                        or f"paleogeographic_map_vision[{figure_id}] "
                        f"point '{loc.get('label')}' conf={loc.get('confidence')}"
                    )
                    out.append(
                        {
                            "species": species,
                            "section_type": "paleogeographic_map",
                            "link_source": "geo_vision_point",
                            "age": loc.get("age"),
                            "ma_top": loc.get("ma_top"),
                            "ma_base": loc.get("ma_base"),
                            "formation": loc.get("formation"),
                            "lithology": loc.get("lithology"),
                            "biozone": loc.get("biozone"),
                            "latitude": loc.get("latitude"),
                            "longitude": loc.get("longitude"),
                            "paleo_latitude": loc.get("paleo_latitude"),
                            "paleo_longitude": loc.get("paleo_longitude"),
                            "evidence_text": evidence,
                            "confidence": loc.get("confidence", 0.0),
                            "figure_id": figure_id,
                        }
                    )

        # --- Phase X: parse per-layer entries for strat / litholog columns ---
        # Each layer carries its own formation/lithology/age/ma, forming
        # independent GeologyLinkRecords so downstream can associate species
        # to specific layers via the ma_top/ma_base range.
        if figure_type in ("strat_column", "litholog_column"):
            layers = parsed.get("layers")
            if isinstance(layers, list):
                for layer in layers:
                    if not isinstance(layer, dict):
                        continue
                    layer = dict(layer)
                    # Audit 2026-08-19 Phase 2b M-12/M-13
                    _apply_geo_whitelist(layer)
                    # Phase 6D audit 2026-08-19 NIT-3: auto-swap before
                    # the strict null-on-violation check.
                    _normalize_ma_pair(layer)
                    _validate_ma_range(layer)
                    evidence = (
                        layer.get("evidence")
                        or f"{figure_type}_vision[{figure_id}] "
                        f"layer {layer.get('layer_index')} conf={layer.get('confidence')}"
                    )
                    out.append(
                        {
                            "section_type": section_type,
                            "link_source": "geo_vision_layer",
                            "age": layer.get("age"),
                            "ma_top": layer.get("ma_top"),
                            "ma_base": layer.get("ma_base"),
                            "formation": layer.get("formation"),
                            "member": layer.get("member"),
                            "lithology": layer.get("lithology"),
                            "biozone": layer.get("biozone"),
                            "evidence_text": evidence,
                            "confidence": layer.get("confidence", 0.0),
                            # layer index is informational provenance, not a formal field
                            "_layer_index": layer.get("layer_index"),
                            "_y_top_normalized": layer.get("y_top_normalized"),
                            "_y_base_normalized": layer.get("y_base_normalized"),
                            "_thickness_m": layer.get("thickness_m"),
                            "figure_id": figure_id,
                        }
                    )

        return out

    # ------------------------------------------------------------- stage 7 (schematic)

    def extract_schematic(
        self,
        image: Image.Image,
        caption: str,
        figure_type: str,
        paper_id: str,
        figure_id: str,
    ) -> dict[str, Any] | None:
        """Run MiniMax-M3 vision extraction on a CONCEPTUAL figure.

        Used for schematic / diagram / reconstruction / phylogenetic
        figures (Phase 64 Plan B Task B.3). The output JSON matches the
        prompt contract declared in ``PROMPT_REGISTRY["schematic_geo"]``:

          {
            "figure_type": "schematic" | "diagram" | "reconstruction" | "phylogenetic",
            "text_elements": [{"text": str, "type": str, "confidence": float}, ...],
            "relationships": [{"from": str, "to": str, "label": str}, ...],
            "extracted_facts": {
                "ages_mentioned": [str, ...],
                "geographic_names": [str, ...],
                "taxa_mentioned": [str, ...],
            },
            "confidence": float,
          }

        Returns ``None`` when the figure type isn't one of the four
        supported, when the image is too small, when the backend
        returns no parseable text, or when the JSON shape is wrong.
        The caller treats ``None`` identically to an empty result
        (i.e. it falls through to downstream stubs).

        The returned dict is also stamped with two provenance fields:
          - ``_paper_id`` / ``_figure_id`` so audit can trace each
            extraction back to its source
          - ``_source`` = "schematic_geo" so downstream code can
            filter schematic extractions from regular geology links
        These leading-underscore fields are not part of the prompt
        contract; downstream code can strip them when projecting the
        data into the JSONL export.
        """
        if figure_type not in {
            "schematic",
            "diagram",
            "reconstruction",
            "phylogenetic",
        }:
            return None
        if "schematic_extract" not in PROMPT_REGISTRY:
            return None
        # Skip tiny images — same threshold as extract_geology. We
        # narrow the except to AttributeError/TypeError so unrelated
        # exceptions in the size check are not silently swallowed.
        try:
            if image.width < 32 or image.height < 32:
                return None
        except (AttributeError, TypeError):
            return None

        system_prompt = PROMPT_REGISTRY["schematic_extract"]
        user_prompt = (
            f"Paper: {paper_id}\nFigure: {figure_id}\n"
            f"figure_type: {figure_type}\n\n"
            f"Caption:\n{caption or '(no caption)'}\n\n"
            "Return strict JSON only, no markdown fences."
        )

        res = self._infer_vision(system_prompt, user_prompt, image)
        if res.get("fallback_used"):
            return None
        raw_text = res.get("raw_text") or ""
        try:
            parsed = _safe_json_loads(raw_text)
        except ValueError as exc:
            logger.warning(
                "extract_schematic: failed to parse JSON for %s/%s: %s",
                paper_id,
                figure_id,
                exc,
            )
            return None
        if not isinstance(parsed, dict):
            logger.warning(
                "extract_schematic: backend returned non-dict JSON for %s/%s",
                paper_id,
                figure_id,
            )
            return None

        # Normalize the figure_type field. The classifier's caption-
        # grounded value is the more reliable signal (the LLM might
        # mis-categorize "schematic" as "diagram" since the prompt
        # contract is identical), so we always overwrite with the
        # caller's ``figure_type`` argument. This keeps the four
        # categories from blurring into each other downstream and
        # means the test/audit can trust ``figure_schematic_data.
        # figure_type`` to match the figure_type the pipeline chose.
        parsed["figure_type"] = figure_type

        # Stamp provenance for audit. Leading-underscore fields are
        # filtered out by the JSONL export but kept here so the
        # pipeline can pass them through unchanged.
        parsed["_paper_id"] = paper_id
        parsed["_figure_id"] = figure_id
        parsed["_source"] = "schematic_extract"
        return parsed

    # ------------------------------------------------------------- phase 65 plan a.3
    def infer_species_age_formation(
        self,
        panel_caption: str,
        paper_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Cross-figure inference (Strategy 3 of the Phase 65 linker).

        Given a plate's caption snippet and a paper-level summary of
        every non-plate figure (strat column / litholog / paleogeographic
        map / range chart), ask MiniMax-M3 to infer which formation /
        age / locality the plate's species most likely came from.

        Parameters
        ----------
        panel_caption : str
            The caption text for the plate (or panel) we are trying to
            link. Usually the ``MatchResult.caption_snippet`` field.
        paper_context : dict, optional
            Free-form paper-level context. The linker passes
            ``{"figures": [...]}`` where each entry has at minimum
            ``figure_id``, ``figure_type``, ``caption``, ``formation``,
            ``age``, ``locality``. May also be ``{}`` or ``None``.

        Returns
        -------
        dict
            Shape:
              ``{
                  "species": str | None,
                  "age": str | None,
                  "formation": str | None,
                  "locality": str | None,
                  "figure_id": str | None,
                  "confidence": float (0.3-0.6),
              }``

            Empty / fallback dict (``{"confidence": 0.0, ...}``) on
            backend failure so the caller can distinguish "M3 said no"
            from "M3 didn't run".

        Notes
        -----
        * Text-only call (no image). The cross-figure reasoning is over
          the paper's figure caption summary, not the figures' pixels;
          sending images would dramatically increase cost without much
          accuracy gain at this stage (Phase C covers image-based
          cross-figure linking).
        * Falls back to ``{"confidence": 0.0}`` (not raises) when the
          backend is unavailable / returns malformed JSON, so the
          cross_figure_linker can drop back to the ``unlinked`` source.
        """
        if self.backend is None:
            return {
                "species": None,
                "age": None,
                "formation": None,
                "locality": None,
                "figure_id": None,
                "confidence": 0.0,
            }
        paper_context = paper_context or {}
        figures = paper_context.get("figures") or []
        # Truncate each caption so the prompt stays within budget.
        # ~5 figures × 200 chars + plate caption (~400 chars) ≈ 1.4KB,
        # well within M3's text window.
        figure_lines: list[str] = []
        for fig in figures[:8]:  # cap at 8 to keep prompt small
            fid = str(fig.get("figure_id") or "?")
            ftype = str(fig.get("figure_type") or "?")
            cap = str(fig.get("caption") or "")
            if len(cap) > 200:
                cap = cap[:200] + "..."
            formation = fig.get("formation")
            age = fig.get("age")
            locality = fig.get("locality")
            bits = [f"[{fid} type={ftype}]"]
            if cap:
                bits.append(f"caption={cap!r}")
            if formation:
                bits.append(f"formation={formation}")
            if age:
                bits.append(f"age={age}")
            if locality:
                bits.append(f"locality={locality}")
            figure_lines.append(" ".join(bits))
        figures_blob = "\n".join(figure_lines) if figure_lines else "(none)"
        panel_blob = (panel_caption or "").strip()
        if len(panel_blob) > 400:
            panel_blob = panel_blob[:400] + "..."

        system_prompt = PROMPT_REGISTRY["cross_figure_inference"]
        user_prompt = (
            f"Plate caption:\n{panel_blob or '(no caption)'}\n\n"
            f"Paper figures:\n{figures_blob}\n\n"
            "Return strict JSON only, no markdown fences."
        )

        try:
            res = self._infer_text(system_prompt, user_prompt)
        except Exception:
            logger.exception("infer_species_age_formation: backend call failed")
            return {
                "species": None,
                "age": None,
                "formation": None,
                "locality": None,
                "figure_id": None,
                "confidence": 0.0,
            }
        if res.get("fallback_used") or not (res.get("raw_text") or "").strip():
            return {
                "species": None,
                "age": None,
                "formation": None,
                "locality": None,
                "figure_id": None,
                "confidence": 0.0,
            }
        try:
            parsed = _safe_json_loads(res.get("raw_text") or "")
        except ValueError:
            logger.warning("infer_species_age_formation: failed to parse JSON")
            return {
                "species": None,
                "age": None,
                "formation": None,
                "locality": None,
                "figure_id": None,
                "confidence": 0.0,
            }
        if not isinstance(parsed, dict):
            return {
                "species": None,
                "age": None,
                "formation": None,
                "locality": None,
                "figure_id": None,
                "confidence": 0.0,
            }
        # Clamp confidence to the spec band [0.3, 0.6] for this strategy.
        conf = parsed.get("confidence", 0.4)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.4
        conf = max(0.3, min(0.6, conf))
        return {
            "species": parsed.get("species"),
            "age": parsed.get("age"),
            "formation": parsed.get("formation"),
            "locality": parsed.get("locality"),
            "figure_id": parsed.get("figure_id"),
            "confidence": conf,
        }

    # ------------------------------------------------------------------ audit 2026-08-02
    def infer_morphology(
        self,
        species_name: str,
        source_text: str,
        *,
        source: str = "body_text",
        paper_id: str | None = None,
        max_chars: int = 6000,
    ) -> dict[str, Any]:
        """Stage 6: per-species morphological-description extraction.

        For a single species and a caption or body-text excerpt, ask
        MiniMax-M3 to emit a structured morphological-description
        record. This is the audit-2026-08-02 Stage-6 MVP: opt-in via
        ``m3_stage_6=True``, never modifies existing species/panel
        fields, and never raises (any failure → ``{}`` so the caller
        can distinguish "M3 said no" from "M3 didn't run").

        Parameters
        ----------
        species_name : str
            The binomial or open-nomenclature name of the species
            (``"Triassocampe sp."``, ``"Podocyrtis sinuosa"``). The
            prompt does not gate on this — it's echoed back into the
            user message so the model knows which species to scope the
            extraction to.
        source_text : str
            Either the plate caption (when ``source="caption"``) or a
            body-text excerpt from the Description / Diagnosis /
            Remarks section (when ``source="body_text"``). Truncated
            to ``max_chars`` characters.
        source : str
            Provenance label forwarded into the returned dict so
            callers can round-trip the source kind without reading
            ``evidence_text``. One of ``"caption"``, ``"body_text"``,
            ``"m3_vision"``. Default ``"body_text"``.
        paper_id : str, optional
            For logging only — does not affect the prompt.
        max_chars : int
            Upper bound on the size of ``source_text`` included in the
            prompt. Default 6000 (matches the locator's default).

        Returns
        -------
        dict
            The parsed morphology record with the schema declared in
            ``PROMPT_REGISTRY["morphology_extract"]``. Any failure
            (no backend, empty response, malformed JSON) yields
            ``{}`` and emits a warning. Empty dict lets the caller
            distinguish "M3 ran but extracted nothing" from "M3 did
            not run".

        Notes
        -----
        * Null vs false: the prompt explicitly forbids ``false`` for
          unmentioned features. The caller is responsible for
          round-tripping null / false correctly when it stores the
          record.
        * Per-paper dedup: callers must dedup on
          ``(paper_id, normalized_species)`` BEFORE calling this
          method (the locator + pipeline helper already do).
        """
        if self.backend is None:
            return {}
        if not species_name or not species_name.strip():
            return {}
        if not source_text or not source_text.strip():
            return {}
        # Trim source_text to keep the prompt within budget. We add an
        # ellipsis to signal truncation to the model (it should not
        # invent features that were cut off).
        trimmed = source_text.strip()
        if len(trimmed) > max_chars:
            trimmed = trimmed[:max_chars] + "..."
        system_prompt = PROMPT_REGISTRY["morphology_extract"]
        user_prompt = (
            f"Species: {species_name.strip()}\n"
            f"Source ({source}):\n{trimmed}\n\n"
            "Return strict JSON only, no markdown fences."
        )
        try:
            res = self._infer_text(system_prompt, user_prompt)
        except Exception:
            logger.exception(
                "infer_morphology: backend call failed (paper=%s, species=%s)",
                paper_id,
                species_name,
            )
            return {}
        if res.get("fallback_used") or not (res.get("raw_text") or "").strip():
            return {}
        try:
            parsed = _safe_json_loads(res.get("raw_text") or "")
        except ValueError:
            logger.warning(
                "infer_morphology: failed to parse JSON (paper=%s, species=%s)",
                paper_id,
                species_name,
            )
            return {}
        if not isinstance(parsed, dict):
            return {}
        # Clamp confidence to [0.0, 1.0]. The prompt asks for 0-1 but a
        # misbehaving model can emit 1.5 or -0.2 — round-trip safely.
        try:
            conf = float(parsed.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        parsed["confidence"] = conf
        # Stamp the source kind onto the returned dict so the caller
        # doesn't have to remember which flavor of context produced
        # the record. Provenance is then carried into MorphologyRecord.
        parsed["_source"] = source
        return parsed

    # ------------------------------------------------------------- phase 66 plan c.1
    def cross_figure_visual_inference(
        self,
        plate_image: Image.Image,
        strat_image: Image.Image | None = None,
        plate_caption: str = "",
        strat_caption: str = "",
        *,
        litholog_image: Image.Image | None = None,
        paleogeographic_image: Image.Image | None = None,
        litholog_caption: str = "",
        paleogeographic_caption: str = "",
    ) -> dict[str, Any]:
        """VISION-based cross-figure inference (Phase 66 Plan C.1).

        Given an SEM plate image AND a strat column / paleogeographic
        map / litholog column image (with their captions), ask
        MiniMax-M3 to identify which plate panels correspond to which
        strat layers / formations / ages. This is the precision-
        refinement counterpart to ``infer_species_age_formation``
        (text-only, confidence 0.3-0.6): vision grounding is
        intrinsically more reliable, so the caller trusts the returned
        confidence (0.0-1.0) without clamping.

        Parameters
        ----------
        plate_image : PIL.Image.Image
            The SEM plate figure image (RGB or convertible).
        strat_image : PIL.Image.Image | None, default ``None``
            The strat column / paleogeographic map / litholog column
            figure image from the same paper (the legacy single-
            secondary-image entry point). When supplied, takes
            priority over the keyword-only ``litholog_image`` and
            ``paleogeographic_image`` parameters so existing callers
            keep working unchanged.
        plate_caption : str
            Caption text for the plate.
        strat_caption : str
            Caption text for the strat column / map (legacy).
        litholog_image : PIL.Image.Image | None, default ``None``
            Optional litholog column image. Used as the secondary
            image only when ``strat_image`` is ``None``.
        paleogeographic_image : PIL.Image.Image | None, default ``None``
            Optional paleogeographic map image. Used as the secondary
            image only when BOTH ``strat_image`` and ``litholog_image``
            are ``None`` (lowest priority of the three).
        litholog_caption : str, default ``""
            Caption for the litholog column. Folded into the user
            prompt when ``litholog_image`` is the selected secondary.
        paleogeographic_caption : str, default ``""
            Caption for the paleogeographic map. Folded into the user
            prompt when ``paleogeographic_image`` is the selected
            secondary.

        Returns
        -------
        dict
            Shape::

              {
                "plate_panels": [
                  {
                    "cell_label": str,
                    "species": str,
                    "links_to_strat_layer": int | None,
                    "links_to_age": str | None,
                    "links_to_formation": str | None,
                    "confidence": float (0.0-1.0, unclamped),
                  },
                  ...
                ]
              }

            Returns ``{"plate_panels": []}`` (NOT raises) on any
            failure path: no backend, fallback_used, malformed JSON,
            tiny images, or any exception during inference. This lets
            the caller treat "M3 said nothing" identically to "M3
            didn't run".

        Notes
        -----
        * The BOTH images (``plate_image`` AND the selected
          secondary) are forwarded to the backend when the backend
          supports it. The Anthropic-backed ``MiniMaxM3Backend``
          accepts multiple image blocks in a single Messages API call
          — it receives both images as separate content blocks so
          the model can ground plate panels against strat-column
          layers directly. Local backends (llama.cpp) are single-
          image and only see the plate image; ``infer_panel``
          injects a prompt note that the secondary image was
          dropped. (Audit M-14 / Phase 4D.)
        * Selection priority for the secondary image (Phase 4D):
          ``strat_image`` > ``litholog_image`` >
          ``paleogeographic_image``. The first non-``None`` value
          (with width/height ``>= 32``) wins. Callers that only
          have one figure type can pass it positionally; callers
          that have all three can use the keyword arguments so
          future multi-image backends can use them all.
        * Panel entries missing ``cell_label`` OR ``species`` are
          dropped before returning; the linker requires both keys to
          attach a visual link to a specific panel.
        * Confidence is clamped to ``[0.0, 1.0]`` for safety but
          otherwise NOT clamped to any narrower band — the visual
          signal can support > 0.6 confidence legitimately.
        """
        empty: dict[str, list[Any]] = {"plate_panels": []}
        if self.backend is None:
            return empty
        # Tiny images produce only noise; bail early without burning
        # a vision call. Same threshold as extract_geology / extract_schematic.
        try:
            if plate_image.width < 32 or plate_image.height < 32:
                return empty
        except (AttributeError, TypeError):
            return empty

        # Phase 4D: pick the first usable secondary image. ``_infer_vision``
        # today only forwards ONE extra_image, so we materialise the
        # priority chain (strat > litholog > paleo) into the existing
        # ``extra_image`` slot. When a future multi-extra-image backend
        # lands, callers can still pass all three and the engine will
        # at least surface them as prompt text via the captions below.
        candidates: list[tuple[str, Image.Image | None, str]] = [
            ("strat", strat_image, strat_caption),
            ("litholog", litholog_image, litholog_caption),
            ("paleogeographic", paleogeographic_image, paleogeographic_caption),
        ]
        chosen_label = ""
        chosen_image: Image.Image | None = None
        chosen_caption = ""
        for label, img, cap in candidates:
            if img is None:
                continue
            try:
                if img.width < 32 or img.height < 32:  # type: ignore[attr-defined]
                    continue
            except (AttributeError, TypeError):
                continue
            chosen_label = label
            chosen_image = img
            chosen_caption = cap or ""
            break
        if chosen_image is None:
            # No usable secondary image → silently bail with empty
            # result. The visual cross-figure contract requires both
            # images to be present; the trigger check is enforced
            # upstream (see ``cross_figure_linker._has_plate_and_anchor``).
            return empty

        system_prompt = PROMPT_REGISTRY["cross_figure_visual"]
        # Truncate captions to keep the prompt budget-bounded; long
        # captions mostly add noise for visual linking.
        plate_blob = (plate_caption or "").strip()
        if len(plate_blob) > 600:
            plate_blob = plate_blob[:600] + "..."
        secondary_blob = (chosen_caption or "").strip()
        if len(secondary_blob) > 600:
            secondary_blob = secondary_blob[:600] + "..."
        secondary_kind = {
            "strat": "Strat column / map",
            "litholog": "Litholog column",
            "paleogeographic": "Paleogeographic map",
        }.get(chosen_label, "Secondary figure")

        user_prompt = (
            f"Plate caption:\n{plate_blob or '(no caption)'}\n\n"
            f"{secondary_kind} caption:\n{secondary_blob or '(no caption)'}\n\n"
            "Return strict JSON only, no markdown fences."
        )

        # Audit M-14: forward BOTH images to the backend. The Anthropic
        # backend natively supports multi-image content blocks; local
        # backends (llama.cpp / Ollama) accept only one and the
        # ``infer_panel`` implementations inject a prompt note that
        # the second image was dropped.
        try:
            res = self._infer_vision(
                system_prompt, user_prompt, plate_image, extra_image=chosen_image
            )
        except Exception:
            logger.exception("cross_figure_visual_inference: backend call failed")
            return empty
        if res.get("fallback_used") or not (res.get("raw_text") or "").strip():
            return empty
        try:
            parsed = _safe_json_loads(res.get("raw_text") or "")
        except ValueError:
            logger.warning("cross_figure_visual_inference: failed to parse JSON")
            return empty
        if not isinstance(parsed, dict):
            return empty
        panels_raw = parsed.get("plate_panels")
        if not isinstance(panels_raw, list):
            return empty

        # Filter + normalize each panel entry. We require cell_label
        # and species; everything else is optional and passes through
        # after a light type-coerce + confidence clamp.
        cleaned: list[dict[str, Any]] = []
        for entry in panels_raw:
            if not isinstance(entry, dict):
                continue
            cell_label = entry.get("cell_label")
            species = entry.get("species")
            if not cell_label or not species:
                continue
            # Confidence clamp to [0.0, 1.0]; never narrow the band.
            try:
                conf = float(entry.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            cleaned.append(
                {
                    "cell_label": str(cell_label),
                    "species": str(species),
                    "links_to_strat_layer": entry.get("links_to_strat_layer"),
                    "links_to_age": entry.get("links_to_age"),
                    "links_to_formation": entry.get("links_to_formation"),
                    "confidence": conf,
                }
            )

        return {"plate_panels": cleaned}

    def enrich_plate_panels(
        self,
        image: Image.Image,
        *,
        page_caption: str,
        paper_id: str,
        figure_id: str,
        expected_plate_label: str | None = None,
    ) -> list[dict[str, Any]]:
        """Round 7 multi-plate enrichment: ask MiniMax-M3 to extract the full
        panel list from a plate image + page-level caption context.

        Used as a SECOND PASS when the first-pass extraction (OD caption-
        image pairing + LLM-first per-figure) produced fewer panels than
        the caption parser claims exist. Common failure mode: OpenDataLoader
        dropped the caption for some plates (e.g. Bandini 2011 Plate 7-9
        sit on pages with multiple figures and OD's pairing logic misses
        them). In that case ``enrich_plate_panels`` is called with the
        page-level caption (which includes captions for ALL plates on the
        page) and the model is told to only emit panels for the matching
        plate (``expected_plate_label`` like "Plate 7").

        Returns a list of dicts shaped like:
          ``{"label": "1", "species": "...", "confidence": 0.9, ...}``
        or ``[]`` when the model returns nothing usable (fallback_used,
        tiny image, malformed JSON, etc.).

        Cost: one M3 vision call (~¥0.01-0.02 per plate). Callers should
        gate this on observed panel-count loss to avoid wasted spend.
        """
        try:
            if image.width < 32 or image.height < 32:
                return []
        except (AttributeError, TypeError):
            return []

        system_prompt = PROMPT_REGISTRY["multi_plate_enrich"]
        constraint = (
            f" This image is plate '{expected_plate_label}'." if expected_plate_label else ""
        )
        user_prompt = (
            f"Paper: {paper_id}\n"
            f"Figure: {figure_id}{constraint}\n\n"
            f"Caption text from page(s):\n{page_caption or '(no caption)'}\n\n"
            "Identify all specimen panels in the plate image. Return JSON."
        )

        res = self._infer_vision(system_prompt, user_prompt, image)
        if res.get("fallback_used") or res.get("error"):
            logger.debug(
                "enrich_plate_panels %s/%s: M3 returned fallback/error",
                paper_id,
                figure_id,
            )
            return []

        # Parse JSON response. M3 sometimes wraps in ```json fences;
        # _safe_json_loads handles that, and we accept either {"panels": [...]}
        # at top level (model contract) or a bare list (lenient fallback).
        raw = res.get("raw_text") or ""
        # audit 2026-08-01 (M8): every sibling M3 method
        # (``extract_geology``, ``extract_schematic``,
        # ``cross_figure_visual_inference``, ``infer_species_age_formation``)
        # wraps this call and returns ``[]`` on unparseable output. This one
        # didn't, so a malformed response escaped as an exception and the
        # caller logged it as a generic "multi_plate_enrich failed".
        try:
            parsed = _safe_json_loads(raw)
        except (ValueError, AttributeError):
            logger.warning(
                "enrich_plate_panels %s/%s: could not parse M3 JSON response",
                paper_id,
                figure_id,
            )
            return []
        panels_data: list[Any] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("panels"), list):
            panels_data = list(parsed.get("panels") or [])
        elif isinstance(parsed, list):
            # audit 2026-08-01 (M10): the balanced-object recovery inside
            # ``_safe_json_loads`` returns a LIST even when the model emitted
            # a single ``{"panels": [...]}`` object (typical after a fence /
            # preamble rescue). Unwrap that shape before treating the list as
            # a bare panel array — otherwise every recovered response yielded
            # zero panels because ``{"panels": ...}`` has no ``label`` key.
            if parsed and isinstance(parsed[0], dict) and "panels" in parsed[0]:
                inner = parsed[0].get("panels")
                panels_data = list(inner) if isinstance(inner, list) else []
            else:
                panels_data = list(parsed)
        if not panels_data:
            return []

        out: list[dict[str, Any]] = []
        for p in panels_data:
            if not isinstance(p, dict):
                continue
            label = str(p.get("label", "")).strip()
            species = p.get("species")
            conf = p.get("confidence")
            try:
                conf_f = float(conf) if conf is not None else 0.7
            except (TypeError, ValueError):
                conf_f = 0.7
            if not label:
                continue
            out.append(
                {
                    "label": label,
                    "species": species if species else None,
                    "confidence": conf_f,
                }
            )
        return out

    def _maybe_dump_diagnostic(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        result: dict[str, Any],
    ) -> None:
        out_dir = self.config.get("m3_diagnostic_dir")
        if not out_dir:
            return
        try:
            from pathlib import Path

            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            self._diagnostic_counter += 1
            idx = self._diagnostic_counter
            image.save(p / f"img_{idx:04d}.png")
            with (p / f"call_{idx:04d}.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "result": result,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
        except Exception as exc:
            logger.debug("Diagnostic dump failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LABEL_RANGE_RE = re.compile(r"^\s*([A-Za-z0-9]+)\s*[-–~]\s*([A-Za-z0-9]+)\s*$")


def _expand_label_range(s: str) -> list[str]:
    """Expand 'A-D' / '3-5' to ['A','B','C','D']; pass through if not a range.

    Audit 2026-09-01 BL-21: the previous implementation used
    ``range(int(a), int(b) + 1)`` which returns an empty list when
    ``int(a) > int(b)`` — silently dropping the entire panel's species
    association. The LLM sometimes emits "Z-A" / "9-3" (reverse order).
    Normalise to ascending order before iterating.
    """
    m = _LABEL_RANGE_RE.match(s)
    if not m:
        return [s.strip()] if s.strip() else []
    a, b = m.group(1), m.group(2)
    # Letters
    if len(a) == 1 and len(b) == 1 and a.isalpha() and b.isalpha():
        if a.isupper() and b.isupper():
            lo, hi = sorted([ord(a), ord(b)])
            return [chr(c) for c in range(lo, hi + 1)]
        if a.islower() and b.islower():
            lo, hi = sorted([ord(a), ord(b)])
            return [chr(c) for c in range(lo, hi + 1)]
    # Digits
    if a.isdigit() and b.isdigit():
        ia, ib = int(a), int(b)
        lo, hi = sorted([ia, ib])
        return [str(i) for i in range(lo, hi + 1)]
    return [s]


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Parse a numeric confidence the LLM may have emitted as a string
    ("0.8") or as a non-numeric label ("high"). Returns ``default`` on
    anything unparseable (audit 2026-07-31: a bare ``float()`` here
    crashed match_panel and voided the paid M3 judgement)."""
    if isinstance(v, (int, float)):
        return float(v)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _safe_bool(v: Any, default: bool = False) -> bool:
    """Parse a JSON boolean that the LLM may have emitted as a STRING.

    audit 2026-07-31: ``bool("false")`` is True in Python — M3
    returning ``"is_radiolarian_plate": "false"`` used to pass the
    stage-2 gate as a real radiolarian plate (and vice versa). Accepts
    real bools and the common string spellings.
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"true", "yes", "1", "y", "t"}:
        return True
    if s in {"false", "no", "0", "n", "f", ""}:
        return False
    return default


def _coerce_label(value: Any) -> str | None:
    """Phase 38: M3 sometimes returns visible_label as a list (e.g.
    ``["A", "B"]``) when it sees two labels on the same panel. The
    old code did ``str(value).strip()`` which produced the Python
    repr ``"['A', 'B']"`` and broke downstream panel_id assignment.
    Join list values; coerce numbers / None to a clean string."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, (list, tuple)):
        parts = [_coerce_label(v) for v in value]
        joined = ", ".join(p for p in parts if p)
        return joined or None
    if isinstance(value, (int, float)):
        s = str(value).strip()
        return s or None
    # Last resort: stringify but avoid the "['x', 'y']" repr trap.
    s = str(value).strip()
    return s or None


def _coerce_bbox(v: Any, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """Coerce M3 bbox output (often normalized 0-1) to absolute pixel coords.

    Detection rule: ``max(nums) <= 1.01`` (1.0 + 0.01 float tolerance) means
    normalized, regardless of image size. The previous code additionally
    required ``img_w > 100 or img_h > 100`` — when M3 returned normalized
    coordinates for a thumbnail-sized figure (e.g. a 80x80 plate image),
    the size guard silently routed the bbox through the pixel path, which
    truncated the four values to ``(0, 0, 1, 1)`` and broke Stage 3 /
    multi-plate enrichment crops. Real pixel values ≤ 1 are exceedingly
    rare in any meaningful figure (a 1x1 bbox is useless to the
    segmenter), so the 1.01 tolerance alone is sufficient to disambiguate.

    Phase 6D audit 2026-08-19 NIT-1: enforce strict ``[0.0, 1.0]`` range
    on normalized bboxes. The previous code silently clamped negative
    values to 0 and values > 1.0 to ``img_w / img_h`` in the pixel path,
    which masked malformed LLM output (e.g. ``[-0.5, 0.3, 0.4, 0.2]``
    or ``[0.5, 0.3, 1.5, 0.2]``). We now raise :class:`ValueError` so
    the caller can log the source figure / caption and decide whether
    to skip the panel or retry the LLM call.
    """
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        nums = [float(x) for x in v]
    except Exception:
        return None
    # NIT-1: reject any negative coordinate. Negative values are
    # meaningless for both normalized (axis convention) and pixel
    # (image-origin) interpretations — no clamping can recover them.
    if any(n < 0.0 for n in nums):
        raise ValueError(
            f"_coerce_bbox: negative coordinate in {v!r} (img_w={img_w}, img_h={img_h})"
        )
    # NIT-1: for normalized bbox (max <= 1.01), every value must be
    # in [0.0, 1.0]. Silently clamping > 1.0 to the image edge would
    # collapse a malformed bbox into a full-image rectangle, which
    # then poisons the segmenter with a panel that spans the whole
    # figure. Use the 1.01 tolerance only to CLASSIFY the bbox as
    # normalized, not to excuse out-of-range values within it.
    if max(nums) <= 1.01 and any(n > 1.0 for n in nums):
        raise ValueError(
            f"_coerce_bbox: normalized coordinate > 1.0 in {v!r} (img_w={img_w}, img_h={img_h})"
        )
    # P4-9 fix: require at least one value > 1.0 to classify as
    # pixel coords. A 1x1 pixel bbox at the origin has max=1.0, which
    # would incorrectly match the normalized threshold and get scaled
    # by image dimensions (e.g. 4000x3000 → 40x30, wrong).
    if max(nums) <= 1.01 and min(nums) >= 0:
        x, y, w, h = nums
        x_px = max(0, int(x * img_w))
        y_px = max(0, int(y * img_h))
        # Width: pixel value, minimum 1, capped at available distance to right edge
        w_px = max(1, min(int(w * img_w), img_w - x_px))
        h_px = max(1, min(int(h * img_h), img_h - y_px))
        return (x_px, y_px, w_px, h_px)
    # Pixel path: clamp each coord to image bounds (Phase 55 audit fix)
    x = max(0, min(int(nums[0]), img_w))
    y = max(0, min(int(nums[1]), img_h))
    w = max(1, min(int(nums[2]), img_w - x))
    h = max(1, min(int(nums[3]), img_h - y))
    return (x, y, w, h)
