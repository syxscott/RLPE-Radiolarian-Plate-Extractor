from __future__ import annotations

import base64
import io
import ipaddress
import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_JSON_ARR_RE = re.compile(r"\[.*?\]", re.DOTALL)

# Phase 61 Plan 4 (Bug 4.3): deterministic run knob. When a caller passes
# ``deterministic=True`` to ``resolve_deterministic_kwargs`` the resulting
# sampling params drive every LLM backend to temperature=0 + greedy
# decode + a fixed Python / numpy / torch seed. The default is the
# stochastic behaviour from before Phase 61 (do_sample=True, temperature
# = 0.1) so production runs are unchanged unless ``--deterministic`` is
# passed.
DEFAULT_DETERMINISTIC_SEED: int = 42


def resolve_deterministic_kwargs(
    base: dict[str, Any] | None = None,
    *,
    deterministic: bool = False,
    seed: int = DEFAULT_DETERMINISTIC_SEED,
    seed_python: bool = True,
) -> dict[str, Any]:
    """Return a copy of ``base`` overwritten for deterministic decode.

    When ``deterministic`` is True, sets ``temperature=0.0``,
    ``do_sample=False``, and ``seed=<int>``. Also seeds the standard
    library ``random`` + ``numpy.random`` modules so any non-LLM
    randomness (random choice for NMS tie-breaks, etc.) is reproducible
    across runs. ``torch`` is seeded lazily (only if torch is imported).

    Returns the merged dict. Returns ``base`` unchanged when
    ``deterministic`` is False.
    """
    out: dict[str, Any] = dict(base or {})
    if not deterministic:
        return out
    out["temperature"] = 0.0
    out["do_sample"] = False
    out["seed"] = int(seed)
    if seed_python:
        try:
            import random as _random

            _random.seed(int(seed))
        except Exception:
            pass
        try:
            import numpy as _np

            _np.random.seed(int(seed))
        except Exception:
            pass
        try:
            import torch as _torch

            _torch.manual_seed(int(seed))
        except Exception:
            pass
    return out


# Phase 61 Plan 4 (Bug 4.1): re-export the token-aware caption truncation
# helper from ``_llm_caption`` so callers can ``from rlpe.llm_backends
# import _truncate_caption_for_llm``. We keep the actual implementation in
# its own module to avoid dragging the Anthropic SDK / heavy dataclass
# imports into places that just need the helper.
from ._llm_caption import (  # noqa: E402,F401  (intentional re-export)
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TOKENS,
    _truncate_caption_for_llm,
)


class FallbackRecommendedError(Exception):
    """Phase 61 Plan 4 (Bug 4.10): raised when a 4xx error suggests switching to the fallback backend.

    The ``recommended_backend`` attribute carries the name of the backend
    to switch to (from ``PipelineConfig.extra["fallback_llm_backend"]``).
    """

    def __init__(self, message: str, recommended_backend: str | None = None):
        super().__init__(message)
        self.recommended_backend = recommended_backend


class LLMHTTPError(Exception):
    """Base class for HTTP errors raised by LLM backends.

    Audit 2026-08-19 Phase 4B (M-23): the llama.cpp backend used to
    silently fall back to the ``/completion`` text-only endpoint on
    ANY exception — including non-transient 4xx errors that signal a
    real configuration problem (401 unauthorized, 403 forbidden,
    404 wrong model, 429 rate limit, 413 payload too large, ...).
    Callers could not tell "endpoint missing the multimodal route"
    from "API key invalid". The new exception hierarchy lets callers
    distinguish auth / not-found / rate-limit failures so they can
    surface the real reason to the user instead of swapping to a
    degraded path.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LLMAuthenticationError(LLMHTTPError):
    """HTTP 401 / 403 — credentials rejected or forbidden.

    Audit M-23: surfaces authentication failures so the caller can
    prompt the user to refresh their API key instead of silently
    swapping to a degraded text-only path.
    """


class LLMNotFoundError(LLMHTTPError):
    """HTTP 404 — wrong model name, missing endpoint, or unknown route.

    Audit M-23: surfaces "model not found" so the caller knows the
    configuration is wrong, not that the multimodal path is broken.
    """


class LLMRateLimitError(LLMHTTPError):
    """HTTP 429 — too many requests; the server has asked us to back off.

    Audit M-23: distinct from a generic HTTP error so the retry loop
    can apply a real rate-limit backoff (Retry-After header +
    exponential jitter) instead of failing fast like a 4xx client
    error.
    """


def select_backend_after_4xx(
    current_backend: str,
    configured_fallback: str | None,
    attempts_made: int,
) -> str:
    """Phase 61 Plan 4 (Bug 4.10): pick the backend for the next retry.

    Pure helper so tests can verify the policy without spinning up an
    LLM client. Behaviour:
      * If no fallback is configured → return ``current_backend``.
      * If we are already on the fallback backend → return it (no loop).
      * If ``attempts_made < 2`` → return ``current_backend`` (we still
        owe one retry to the primary backend; only after the first
        retry fails do we switch).
      * Otherwise → return ``configured_fallback``.
    """
    if not configured_fallback:
        return current_backend
    if configured_fallback == current_backend:
        return current_backend
    if attempts_made < 2:
        return current_backend
    return configured_fallback


# Match Anthropic / MiniMax / OpenAI style API keys (sk-ant-..., sk-...,
# plus generic 40+ char sk- prefixes). Anthropic's actual key shape is
# ``sk-ant-api03-<48 alnum>``; MiniMax / OpenAI use ``sk-<30+ alnum>`` or
# ``sk-proj-<...>``. We use a conservative pattern that:
#   - requires a non-key character (or start) immediately before ``sk-``,
#   - requires the key body to be at least 16 alnum characters long,
#   - allows hyphens in the key body but treats any run of
#     ``-ascii_lower`` as a potential suffix (so a stray ``-suffix`` is
#     not consumed as part of the key).
_API_KEY_PATTERNS = (
    # Generic ``sk-<16+ chars>``: stops at any non-alnum, non-underscore,
    # non-trailing-hyphen boundary.
    re.compile(r"(?<![A-Za-z0-9_])sk-(?=[A-Za-z0-9]{16})[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+){0,3}"),
    re.compile(r"(?<![A-Za-z0-9_])sk-ant-api03-[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_])sk-ant-(?!api03-)[A-Za-z0-9]{16,}"),
    re.compile(r"(?<![A-Za-z0-9_])sk-proj-[A-Za-z0-9]{16,}"),
    re.compile(r"(?<![A-Za-z0-9_])sk-cp-[A-Za-z0-9]{16,}"),
    # Audit 2026-09-01 CR-17: extend the redaction set to cover the
    # cloud-provider credentials that the operator may inject into
    # ``extra`` when routing MiniMax via AWS Bedrock / Vertex /
    # Azure. Without these patterns, an AKIA / ya29 / Azure key
    # embedded in the run would be persisted verbatim into
    # ``matches.jsonl`` and exposed via the public-facing
    # ``/jobs/{id}/results`` endpoint — a cloud-credential leak
    # masquerading as a science-data file.
    # AWS access key id (20 uppercase alnum chars, starts with AKIA/ASIA/AROA/AIDA/ANPA/ANVA/AGPA).
    re.compile(r"\b(?:AKIA|ASIA|AROA|AIDA|ANPA|ANVA|AGPA)[A-Z0-9]{16}\b"),
    # Google OAuth refresh token (``ya29.<base64>``).
    re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}\b"),
    # Azure OpenAI / Cognitive Services key (32 lowercase hex chars).
    re.compile(r"\b[a-f0-9]{32}\b(?=.*Azure)"),
    # Stripe live key (sk_live_<24+>).
    re.compile(r"\bsk_live_[A-Za-z0-9]{20,}\b"),
)


def _redact_api_keys(text: str) -> str:
    """Replace any API-key-looking substrings with ``[REDACTED]``.

    The Anthropic SDK embeds the offending key in its 401/403 messages
    and that string flows into ``str(exc)`` and ultimately into
    ``match.metadata["gemma_error"]``. Without this redaction, the user's
    raw key would land in ``matches.jsonl`` and the Web UI. We never want
    secrets in the manifest even if the SDK leaks them.
    """
    if not text:
        return text
    for pat in _API_KEY_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def _last_balanced_json_object(text: str) -> str | None:
    """Return the LAST balanced JSON object substring, or None.

    Uses a brace counter that respects string boundaries so that braces
    inside a string literal (e.g. ``"key": "{not really json}"``) do
    not throw the counter off. Audit 2026-08-17: when a thinking block
    contains an EXAMPLE JSON object (e.g. ``{"species": "species name"}``)
    and the real answer comes after ``</think>`` as a SECOND JSON
    object, ``parse_json_from_text``'s "load whole text" path used to
    pick the FIRST one. By returning the LAST balanced object we let
    callers default to the real answer.

    Returns ``None`` if no balanced object is found.
    """
    if not text:
        return None
    last: str | None = None
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if c == "{":
            depth = 1
            j = i + 1
            in_string = False
            escape_next = False
            while j < n and depth > 0:
                cj = text[j]
                if in_string:
                    if escape_next:
                        escape_next = False
                    elif cj == "\\":
                        escape_next = True
                    elif cj == '"':
                        in_string = False
                else:
                    if cj == '"':
                        in_string = True
                    elif cj == "{":
                        depth += 1
                    elif cj == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = text[i : j + 1]
                            # Must contain at least one colon to be
                            # plausibly a JSON object (not a Python set
                            # literal or empty ``{}``).
                            if ":" in candidate:
                                last = candidate
                            break
                j += 1
            i = j
        else:
            i += 1
    return last


# ----------------------------------------------------------------------
# SSRF guard for user-supplied LLM hosts
# ----------------------------------------------------------------------
# The Ollama and LlamaCpp backends accept an arbitrary ``host`` URL
# from the operator. Without a guard, a malicious or accidental
# ``host`` (e.g. ``http://169.254.169.254/latest/meta-data/`` on AWS)
# would make the service fetch cloud metadata server-side, or probe
# internal hosts the operator didn't intend to expose. The guard
# below is intentionally permissive for the common case (loopback
# and RFC1918 private — local LLM servers usually live there) but
# blocks link-local addresses and non-HTTP(S) schemes that are the
# usual SSRF payloads.
#
# Set ``RLPE_LLM_ALLOW_ANY_HOST=1`` to disable the check entirely
# (e.g. when running on a hardened network where the operator
# controls the LLM host). The default is safe-by-default.
def _validate_llm_host(host: str) -> str:
    """Return ``host`` unchanged if it passes the SSRF guard, else raise.

    The check covers:
      - scheme: must be ``http`` or ``https`` (rejects ``file://``,
        ``gopher://``, ``ftp://`` etc.)
      - link-local addresses (169.254.0.0/16, fe80::/10) — these
        include the AWS / GCP / Azure metadata endpoints and IPv6
        link-local; almost never a legitimate LLM host.
      - the unspecified address (0.0.0.0, ::) — would route to a
        local interface the operator doesn't expect.
    """
    if os.environ.get("RLPE_LLM_ALLOW_ANY_HOST") == "1":
        return host
    parsed = urlparse(host)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"LLM host must use http or https scheme, got {parsed.scheme!r} "
            f"in {host!r}. Set RLPE_LLM_ALLOW_ANY_HOST=1 to override."
        )
    hostname = parsed.hostname or ""
    # Reject URLs without a hostname (e.g. "http:///metadata"). The
    # previous code fell through to ``ipaddress.ip_address("")`` which
    # raised ValueError and was caught as "treat as DNS name" — the
    # request then went to whatever the OS resolver decided "" meant
    # (usually localhost), bypassing the SSRF guard. Empty hostnames
    # are never legitimate for an LLM host so reject explicitly.
    if not hostname:
        raise ValueError(
            f"LLM host {host!r} has no hostname; refusing to connect "
            f"(SSRF guard). Set RLPE_LLM_ALLOW_ANY_HOST=1 to override."
        )
    # Audit 2026-09-01 CR-18: reject IPv6 zone-id suffixes (``%eth0``,
    # ``%12``) — Python's ``ipaddress.ip_address`` raises on the raw
    # form, but downstream DNS resolution libraries may strip the
    # zone and route the request to a link-local / cloud-metadata
    # endpoint on the *original* interface. Block at the boundary so
    # the operator never silently hits the AWS / GCP metadata service.
    if "%" in hostname:
        raise ValueError(
            f"LLM host {host!r} carries an IPv6 zone-id ('%'-suffix); "
            f"refusing to connect (SSRF guard). Use the bare IPv6 address "
            f"or its DNS hostname instead. Set RLPE_LLM_ALLOW_ANY_HOST=1 "
            f"to override."
        )
    # Try to parse as an IP literal. If it isn't, the host is a DNS
    # name and we can't enumerate its addresses without a DNS lookup
    # (which itself can be an SSRF vector) — accept it and let the
    # outbound connection handle the resolution. Operators who don't
    # trust their DNS should set RLPE_LLM_ALLOW_ANY_HOST=0 explicitly
    # (which is the default).
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        return host
    # Audit M10: also block IPv4-mapped IPv6 addresses whose IPv4
    # payload is itself non-routable (e.g. ``::ffff:169.254.169.254``
    # which is the AWS / GCP / Azure metadata endpoint). The previous
    # check only flagged ``is_link_local`` / ``is_unspecified`` /
    # ``is_multicast`` directly on the IPv6 object, which leaves a
    # window for an IPv6 wrapper around a dangerous IPv4 address.
    # We intentionally do NOT add ``is_private`` to the predicate —
    # loopback (127.0.0.0/8) and RFC1918 (10/8, 172.16/12, 192.168/16)
    # are legitimate local LLM hosts and the docstring promises
    # they're allowed. ``is_reserved`` is also excluded from the
    # main predicate because it would over-block ``[::1]`` (IPv6
    # loopback); the reserved-range check is still applied to the
    # inner IPv4 of an IPv4-mapped IPv6 wrapper below.
    ipv4_mapped = getattr(addr, "ipv4_mapped", None)
    if ipv4_mapped is not None:
        # The host is an IPv4-mapped IPv6 literal. Reject if the
        # inner IPv4 is one of the SSRF payloads (link-local,
        # unspecified, multicast, or reserved).
        if (
            ipv4_mapped.is_link_local
            or ipv4_mapped.is_unspecified
            or ipv4_mapped.is_multicast
            or ipv4_mapped.is_reserved
        ):
            raise ValueError(
                f"LLM host {host!r} resolves to a non-routable address "
                f"({addr} -> {ipv4_mapped}); refusing to connect (SSRF guard). "
                f"Set RLPE_LLM_ALLOW_ANY_HOST=1 to override."
            )
    if addr.is_link_local or addr.is_unspecified or addr.is_multicast:
        raise ValueError(
            f"LLM host {host!r} resolves to a non-routable address "
            f"({addr}); refusing to connect (SSRF guard). "
            f"Set RLPE_LLM_ALLOW_ANY_HOST=1 to override."
        )
    return host


def parse_json_from_text(text: str) -> dict[str, Any]:
    """Parse the first balanced JSON object from ``text``.

    Tries, in order: the whole text (after stripping code fences), then the
    first ``[...]`` array element, then the first ``{...}`` object.  This
    is robust to LLMs that emit a JSON array when asked for one.

    Returns a normalized dict with at minimum: ``label``, ``species``,
    ``confidence``, ``reasoning``.  Raises ``ValueError`` if no parseable
    JSON object can be found.
    """
    if not text:
        raise ValueError("empty text")
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # 1) Try the whole cleaned text first
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            # Audit 2026-09-01 (live end-to-end test on Bandini_2011
            # via MiniMax-M3): M3 returns the FULL multi-panel output
            # as a top-level JSON array — e.g. ``[{"label":"1",
            # "species":"Genus species A"}, {"label":"2", ...}, ...]``.
            # The previous implementation normalised only the FIRST
            # element of the array (line "Use the first object in the
            # array" below), discarding all subsequent panels. That
            # silently capped the LLM-first pipeline at exactly 1 row
            # per figure — collapsing the F1 ceiling to the proportion
            # of plates that contain a Fig. 1 species. Detect the array
            # path FIRST and normalise every element so the multi-
            # panel contract flows through.
            norm: list[dict[str, Any]] = []
            for item in obj:
                if isinstance(item, dict):
                    norm.append(_normalize_panel_dict(item))
            if norm:
                return {"_is_multi_panel": True, "panels": norm}
        elif isinstance(obj, dict):
            # Audit 2026-09-04 (llm-1): the LLM-first system prompt
            # demands exactly this shape — ``{"panels": [...]}`` — so a
            # fully prompt-compliant answer must unwrap to the multi-
            # panel contract instead of being fed to
            # ``_normalize_panel_dict`` (whose whitelist drops the
            # "panels" key and collapses the answer to one empty row).
            unwrapped = _unwrap_panels_object(obj)
            if unwrapped is not None:
                return unwrapped
            return _normalize_panel_dict(obj)
    except Exception:
        pass

    # 2) Try first JSON array element (when the LLM emits a list)
    arr_match = _JSON_ARR_RE.search(cleaned)
    if arr_match:
        try:
            arr = json.loads(arr_match.group(0))
            if isinstance(arr, list) and arr:
                # Audit 2026-09-01 (live Bandini test): the previous
                # implementation normalised only the FIRST element of
                # the array (line "Use the first object in the array")
                # — silently dropping every subsequent panel. Now
                # normalise every element so a 31-panel Bandini_2011
                # caption round-trips as 31 MatchResults rather than 1.
                items = [x for x in arr if isinstance(x, dict)]
                if items:
                    if len(items) == 1:
                        return _normalize_panel_dict(items[0])
                    norm = [_normalize_panel_dict(it) for it in items]
                    return {"_is_multi_panel": True, "panels": norm}
        except Exception:
            pass

    # 3) Fall back to first {...} match (non-greedy to avoid swallowing).
    #    Audit 2026-08-17 BUG-C: when prose preamble contains inline
    #    JSON-like placeholders (e.g. ``{foo: bar}`` quoting a dictionary
    #    shape), the non-greedy ``_JSON_RE`` matches the FIRST object
    #    — which is not the real answer. We accept the match only if
    #    it carries at least one of the schema's expected keys
    #    (``species`` / ``label``); otherwise fall through to path 4
    #    which uses the LAST balanced object (the real answer).
    obj_match = _JSON_RE.search(cleaned)
    if obj_match:
        try:
            obj = json.loads(obj_match.group(0))
            if isinstance(obj, dict):
                unwrapped = _unwrap_panels_object(obj)
                if unwrapped is not None:
                    return unwrapped
                if any(k in obj for k in ("species", "label")):
                    return _normalize_panel_dict(obj)
                # else: not a panel-shape JSON, keep looking
        except Exception:
            pass
    # 4) Audit 2026-08-17: brace-balanced scan. Real M3 / Qwen3
    #    responses sometimes emit a prose preamble that itself
    #    contains ``{...}`` placeholders (e.g. "The relevant context is
    #    {locality: Tunisia, age: Late Cretaceous}") followed by a
    #    JSON object inside a ```json``` fence but the fence is
    #    unclosed (truncated output, max_tokens hit). The non-greedy
    #    ``_JSON_RE`` above matches the FIRST prose block — which is
    #    NOT valid JSON — and json.loads raises. ``_last_balanced_json_object``
    #    scans for balanced braces and returns the LAST one, which is
    #    almost always the real JSON answer (the model puts prose
    #    first and JSON last).
    last_obj = _last_balanced_json_object(cleaned)
    if last_obj is not None:
        try:
            obj = json.loads(last_obj)
            if isinstance(obj, dict):
                unwrapped = _unwrap_panels_object(obj)
                if unwrapped is not None:
                    return unwrapped
                return _normalize_panel_dict(obj)
        except Exception:
            pass
    raise ValueError("No parseable JSON object found in LLM output.")


# ---------------------------------------------------------------------------
# Schema whitelist for ``_normalize_panel_dict`` (audit 2026-08-19
# Phase 4B M-21). The species-identification vision prompt declares a
# JSON contract with a fixed set of canonical keys plus a handful of
# optional structured extras (see ``_MATCH_PANEL_SYSTEM`` in
# ``m3_engine.py``). Without an explicit whitelist, the LLM occasionally
# emits hallucinated fields ("ocr_confidence", "valid_name",
# "fake_field", ...) which — even when filtered by the canonical-key
# loop below — pollute log output, downstream record schemas, and the
# export pipeline. The whitelist enumerates every key that downstream
# code is allowed to see; anything else is dropped with a debug log so
# audit can spot prompt drift.
# ---------------------------------------------------------------------------
_ALLOWED_PANEL_FIELDS = frozenset(
    {
        # Canonical output schema (the four fields the function
        # always emits).
        "label",
        "species",
        "confidence",
        "reasoning",
        # Optional structured fields documented in M3 match-panel
        # prompts (``_MATCH_PANEL_SYSTEM``); the LLM may emit any
        # subset of these per response.
        "open_nomenclature_strength",
        "alternative",
        "is_radiolarian",
        "extraction_method",
        "notes",
        # Caller-managed structural extras forwarded to downstream
        # code (added by ``infer_panel`` / ``infer_text`` after the
        # raw JSON is parsed).
        "raw_text",
        "fallback_used",
        "multimodal_degraded",
        "extra_image_unsupported",
        "error",
        # Lists / nested dicts occasionally emitted when the prompt
        # explicitly asks for them (e.g. ``species_list`` for
        # "list every species in this plate").
        "species_list",
        # Arbitrary structured extras preserved for downstream consumers
        # that understand them (e.g. ``morphology`` shell descriptors,
        # ``stratigraphy`` age/formation dicts).
        "morphology",
        "stratigraphy",
    }
)


def _unwrap_panels_object(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Return the multi-panel contract for a prompt-compliant
    ``{"panels": [...]}`` answer, or ``None`` when ``obj`` is not that
    shape.

    audit 2026-09-04 (llm-1): the LLM-first system prompt requires the
    model to answer ``{"panels": [...]}`` — but the per-panel whitelist
    in ``_normalize_panel_dict`` drops the "panels" key, so a compliant
    answer collapsed to a single empty row. Unwrapping happens at the
    parse layer, before the whitelist ever sees a panel dict.
    """
    panels_val = obj.get("panels")
    if not isinstance(panels_val, list):
        return None
    items = [p for p in panels_val if isinstance(p, dict)]
    if not items:
        # Placeholder contract ("{"panels": []}" for auto captions) —
        # return an explicitly empty multi-panel result rather than a
        # fabricated empty row.
        return {"_is_multi_panel": True, "panels": []}
    return {
        "_is_multi_panel": True,
        "panels": [_normalize_panel_dict(p) for p in items],
    }


def _normalize_panel_dict(obj: dict[str, Any]) -> dict[str, Any]:
    # Audit M-21 (Phase 4B 2026-08-19): schema whitelist post-filter.
    # The LLM occasionally emits hallucinated extras ("ocr_confidence",
    # "valid_name", "fake_field", ...) which, without filtering, would
    # either leak through as canonical fields or be preserved as
    # "structural extras" (list/dict values) downstream. The whitelist
    # explicitly enumerates every key that downstream code is allowed
    # to see — anything else is dropped with a debug log so audit can
    # spot prompt drift.
    if obj:
        dropped = [k for k in list(obj.keys()) if k not in _ALLOWED_PANEL_FIELDS]
        if dropped:
            logger.debug("Dropped hallucinated panel fields: %s", sorted(dropped))
            for k in dropped:
                obj.pop(k, None)

    # Audit M12: the LLM may return ``confidence`` as a string like
    # ``"high"`` / ``"medium"`` / ``"0.8"`` / ``"low"`` instead of a
    # number. The previous code unconditionally called
    # ``float(obj.get("confidence", 0.0))`` which raised ValueError
    # on those non-numeric strings and broke the whole parse. We
    # now coerce safely: numeric types pass through, numeric
    # strings are parsed, anything else (including None) defaults
    # to 0.0.
    raw_conf = obj.get("confidence", 0.0)
    if isinstance(raw_conf, bool):
        # ``bool`` is a subclass of ``int`` in Python — treat True as
        # 1.0 and False as 0.0 so ``True`` doesn't leak through.
        conf_value = 1.0 if raw_conf else 0.0
    elif isinstance(raw_conf, (int, float)):
        conf_value = float(raw_conf)
    elif isinstance(raw_conf, str):
        try:
            conf_value = float(raw_conf.strip())
        except (TypeError, ValueError):
            conf_value = 0.0
    else:
        # None, list, dict, object — none are sensible confidence
        # values, so default to 0.0 instead of crashing.
        conf_value = 0.0
    out = {
        "label": (str(obj.get("label", "")).strip() or None),
        "species": (str(obj.get("species", "")).strip() or None),
        "confidence": conf_value,
        "reasoning": str(obj.get("reasoning", "")).strip() or "No reasoning provided.",
    }
    out["confidence"] = max(0.0, min(1.0, round(out["confidence"], 2)))
    # Audit 2026-08-17 BUG-E: preserve structural extras (e.g.
    # ``species_list`` returned when the prompt asks for "list every
    # species in this plate"). The previous implementation returned
    # only the 4 normalized keys, silently dropping any list/dict
    # fields the caller explicitly asked for. We keep any list-typed
    # or dict-typed values verbatim so callers can consume them
    # downstream (e.g. for per-panel species matching against gold).
    #
    # Audit 2026-08-19 Phase 4B M-21: extend the preservation rule
    # to cover ALL whitelisted keys (scalar + structural). The
    # pre-filter above already drops everything that isn't in
    # ``_ALLOWED_PANEL_FIELDS``; the structural-extras loop below
    # was previously limited to ``list``/``dict`` only. With the
    # whitelist authoritative, any whitelisted value (including
    # ``notes: "..."`` or ``alternative: "X"``) is preserved
    # verbatim — the only remaining drops are the hallucinated
    # fields the pre-filter already removed.
    for k, v in obj.items():
        if k in out:
            continue
        if k in _ALLOWED_PANEL_FIELDS:
            out[k] = v
    # audit 2026-08-19 Phase 1d (B-7): open-nomenclature discount.
    # ICZN open-nomenclature markers (cf./aff./?/ex gr.) mean the
    # identification is *not* at species level. Previously the LLM
    # could return confidence=0.9 for "Triactoma cf. kamoensis" and
    # the F1 would credit a gold match against "Triactoma kamoensis"
    # as a true positive even though the LLM explicitly said
    # "compare-with" the species. Discount the confidence here so
    # downstream scoring treats open-nomenclature rows as lower-trust.
    if isinstance(out.get("species"), str) and out["species"]:
        _apply_open_nomen_discount(out)
    return out


# Confidence caps for open-nomenclature markers. Mirrors the
# ``open_nomenclature_strength`` field we ask the LLM to emit on
# match_panel output (audit 2026-08-19 Phase 1d). The discount is
# applied as a *post-filter* on the normalized species string so it
# works regardless of whether the LLM returns the
# ``open_nomenclature_strength`` field or not.
_OPEN_NOMEN_CF_AFF_CAP = 0.55  # cf. / aff. / ?
_OPEN_NOMEN_EX_GR_CAP = 0.50  # ex gr. (group)


def _apply_open_nomen_discount(out: dict[str, Any]) -> None:
    """Discount ``out['confidence']`` when the species carries ICZN
    open-nomenclature markers.

    audit 2026-08-19 Phase 1d (B-7): cf./aff./? -> cap at 0.55,
    ex gr. -> cap at 0.50. Detection is regex-based on the
    already-normalized species string (catches both with-period
    and bare-word forms). Detection runs *after* the species has
    been str-stripped but *before* any further normalization, so a
    literal "cf" / "aff" without trailing period still triggers.

    Mutates ``out`` in place; no return value.
    """
    species = out.get("species") or ""
    if not species:
        return
    # "cf." / "aff." / "cf" / "aff" as standalone tokens. The
    # leading/trailing boundary guard avoids matching the
    # substring inside longer words (e.g. "pacificus" doesn't
    # match "cf"). Case-insensitive — gold uses lowercase but
    # OCR-derived text sometimes uppercases.
    has_cf_aff = bool(re.search(r"\b(?:cf|aff)\.?\b", species, flags=re.IGNORECASE))
    # "?" literal — gold/caption convention is "(?)" before sp.
    # but raw LLM output may emit "?" anywhere. Only treat as
    # open-nomen when "?" appears between the genus and the
    # epithet/sp. marker (i.e. the question-mark-in-binomial
    # convention). A bare "?" at end of "Genus?" without sp. is
    # also valid (genus uncertain).
    has_question = "?" in species
    # "ex gr." — ICZN group marker. Match "ex gr." with optional
    # whitespace; also catch the rarer "ex.gr." abbreviation.
    has_ex_gr = bool(re.search(r"\bex\.?\s*gr\.?\b", species, flags=re.IGNORECASE))
    if has_ex_gr:
        out["confidence"] = min(out["confidence"], _OPEN_NOMEN_EX_GR_CAP)
    elif has_cf_aff or has_question:
        out["confidence"] = min(out["confidence"], _OPEN_NOMEN_CF_AFF_CAP)


# ---------------------------------------------------------------------------
# Schema whitelist for ``extract_geology`` vision outputs (audit 2026-08-19
# M-12). The geology vision prompt declares a JSON contract with ~14
# well-known keys (age/formation/ma_top/...). Without a whitelist, the LLM
# occasionally emits hallucinated extras ("habitat", "depositional_environment",
# "paleoclimate", ...) which become formal fields in panel.metadata.geology_links
# and pollute downstream filtering/scoring. Whitelist filters the dict
# *in place* and logs a warning listing the dropped keys so audit can spot
# prompt drift.
# ---------------------------------------------------------------------------
_GEO_KEY_WHITELIST = frozenset(
    {
        # Identification / chronostratigraphy
        "age",
        "chronostratigraphy",
        "chronostratigraphy_rank",
        "ma_top",
        "ma_base",
        "ma_mid",
        "biozone",
        "stage",
        # Lithostratigraphy
        "formation",
        "member",
        "group",
        "lithology",
        "thickness",
        "thickness_m",
        # Geography / locality
        "locality",
        "country",
        "latitude",
        "longitude",
        "paleo_latitude",
        "paleo_longitude",
        # Species linkage
        "species",
        # Provenance / quality
        "confidence",
        "source",
        "notes",
        "evidence",
        "evidence_text",
        "section_type",
        "link_source",
        "figure_id",
        # Layer index fields (for strat_column / litholog_column)
        "layer_index",
        "y_top_normalized",
        "y_base_normalized",
        # Internal markers stamped by extract_geology post-processing
        "_layer_index",
        "_y_top_normalized",
        "_y_base_normalized",
        "_thickness_m",
    }
)


def _validate_ma_range(ma_top: Any, ma_base: Any) -> tuple[Any, Any]:
    """Validate (and auto-swap) an inverted Ma-range pair.

    Audit 2026-08-19 Phase 4B M-22: ICZN / stratigraphic convention is
    ``ma_top < ma_base`` (younger = smaller Ma, top of section is the
    younger boundary). The vision LLM occasionally emits the inverted
    range — e.g. reads a stratigraphic column with old-at-top axis
    convention and reports ``ma_top > ma_base``.

    This helper AUTO-SWAPS an inverted pair with a WARNING log so the
    caller still has *some* usable range to work with (better than
    dropping both values when one of them is plausibly correct).

    Returns ``(ma_top, ma_base)`` — either the input unchanged or the
    swapped pair. Non-numeric values are passed through unchanged
    (the strict null-on-violation path lives in
    ``m3_engine._validate_ma_range``; this helper just fixes the
    ordering when both numbers are present and comparable).

    This helper is intentionally DIFFERENT from
    ``rlpe.m3_engine._validate_ma_range``: the engine helper enforces
    the schema contract (null on violation); the llm_backends helper
    fixes the value (swap on violation). Both helpers coexist — the
    engine one runs in the strict ``extract_geology`` pipeline, this
    one is exposed here for callers who prefer auto-fix over rejection.
    Note: ``_apply_geo_whitelist`` does NOT auto-call this helper; that
    pipeline relies on the strict engine helper downstream, so a
    swap here would mask bad ranges from the engine's null branch.
    Callers that want lenient swap-on-violation behavior should call
    this helper explicitly.
    """
    if ma_top is None or ma_base is None:
        return ma_top, ma_base
    try:
        top_f = float(ma_top)
        base_f = float(ma_base)
    except (TypeError, ValueError):
        return ma_top, ma_base
    if top_f > base_f:
        logger.warning(
            "Ma range reversed (ma_top=%r > ma_base=%r); swapping",
            ma_top,
            ma_base,
        )
        return ma_base, ma_top
    return ma_top, ma_base


def _apply_geo_whitelist(item: dict[str, Any]) -> dict[str, Any]:
    """Filter ``item`` to keys in :data:`_GEO_KEY_WHITELIST`.

    Audit 2026-08-19 M-12: the geology vision prompt asks for a fixed
    schema, but LLMs occasionally invent extras ("habitat",
    "depositional_environment", "paleoclimate", ...). Those extras were
    being persisted as first-class fields in
    ``panel.metadata.geology_links`` and silently breaking downstream
    filtering. This helper strips them, logging a warning so audit can
    catch prompt drift.

    Note: ``_apply_geo_whitelist`` does NOT call
    :func:`_validate_ma_range` — the strict null-on-violation policy
    is owned by ``m3_engine._validate_ma_range``, which runs downstream
    of this helper inside ``extract_geology``. Adding a swap here
    would mask bad ranges from the engine's null branch and silently
    break the M-13 regression tests in
    ``tests/test_audit_2026_08_19_phase2b_m3_prompts.py``. Callers
    wanting lenient swap-on-violation should call
    ``_validate_ma_range`` explicitly.

    Returns the filtered dict (same object, mutated in place for
    convenience). Returns ``item`` unchanged if it is not a dict.
    """
    if not isinstance(item, dict):
        return item
    extras = set(item.keys()) - _GEO_KEY_WHITELIST
    if extras:
        logger.warning(
            "LLM output dropped non-whitelisted geology fields: %s",
            sorted(extras),
        )
        for k in extras:
            item.pop(k, None)
    return item


class BaseLLMBackend:
    backend_name = "base"

    def __init__(self) -> None:
        # Audit 2026-09-01 (PERF-18): shared HTTP session per backend
        # instance. The previous implementation called
        # ``requests.post(...)`` directly, so every M3 / Anthropic /
        # Ollama inference rebuilt a fresh TCP connection (DNS +
        # TLS + HTTP handshake) — typically 80-200 ms per call. A
        # typical paper issues 200+ LLM calls; that's 16-40 seconds
        # of pure connection overhead per paper. The session is
        # pool-backed via ``HTTPAdapter`` so concurrent workers can
        # share connections without serialization.
        import requests as _requests
        from requests.adapters import HTTPAdapter as _HTTPAdapter

        self._session = _requests.Session()
        _adapter = _HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0,  # backend classes handle retry policy themselves
        )
        self._session.mount("http://", _adapter)
        self._session.mount("https://", _adapter)

    def close(self) -> None:
        """Release the HTTP session's connection pool.

        Audit 2026-09-01 (systemic #2 follow-up): called from
        :meth:`RadiolarianPipeline.close` so the connection pool's
        sockets are returned to the OS on ``run()`` exit (otherwise
        the pool keeps the file descriptors alive until process
        exit).
        """
        if hasattr(self, "_session") and self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def infer_panel(
        self,
        panel_image,
        caption_text: str,
        ocr_labels: list[str],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def infer_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(slots=True)
class TransformersGemmaBackend(BaseLLMBackend):
    model: Any
    processor: Any
    tokenizer: Any
    is_multimodal: bool
    backend_name: str = "transformers"
    max_new_tokens: int = 220
    temperature: float = 0.1
    top_p: float = 0.9

    def infer_panel(
        self,
        panel_image,
        caption_text: str,
        ocr_labels: list[str],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        import torch

        try:
            with torch.inference_mode():
                if self.is_multimodal and panel_image is not None:
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": panel_image},
                                {"type": "text", "text": user_prompt},
                            ],
                        },
                    ]
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    generated = output[:, inputs["input_ids"].shape[-1] :]
                    text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
                else:
                    full_prompt = system_prompt + "\n\n" + user_prompt
                    tokens = self.tokenizer(
                        full_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=4096,
                    ).to(self.model.device)
                    output = self.model.generate(
                        **tokens,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    gen = output[0][tokens["input_ids"].shape[-1] :]
                    text = self.tokenizer.decode(gen, skip_special_tokens=True)
            parsed = parse_json_from_text(text)
            parsed["raw_text"] = text
            parsed["fallback_used"] = False
            return parsed
        except Exception as exc:
            # Log at error level so operators can distinguish hardware failures
            # (CUDA OOM, model crash) from semantic parse failures.  Without
            # this log, a CUDA OOM surfaces only as ``fallback_used=True`` with
            # no other trace — making debugging very difficult.
            logger.error(
                "TransformersGemmaBackend inference failed (fallback_used=True): %s: %s",
                type(exc).__name__,
                _redact_api_keys(str(exc)),
            )
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": f"Transformers inference error: {type(exc).__name__}",
                "fallback_used": True,
                # M11b: redact API keys that may appear in exception text.
                "error": _redact_api_keys(str(exc)),
            }

    def infer_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        # 文本关系抽取可以复用 panel 推理逻辑（不传图像）。
        return self.infer_panel(None, "", [], system_prompt, user_prompt)


@dataclass(slots=True)
class OllamaGemmaBackend(BaseLLMBackend):
    model: str
    host: str = "http://127.0.0.1:11434"
    timeout_sec: int = 120
    temperature: float = 0.1
    top_p: float = 0.9
    backend_name: str = "ollama"

    def __post_init__(self) -> None:
        # SSRF guard: reject link-local / unspecified / non-http hosts.
        # See ``_validate_llm_host`` for the policy. The default
        # ``http://127.0.0.1:11434`` passes the check (loopback is
        # always allowed).
        self.host = _validate_llm_host(self.host)

    def infer_panel(
        self,
        panel_image,
        caption_text: str,
        ocr_labels: list[str],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        images = []
        if panel_image is not None:
            images.append(_encode_image_base64(panel_image))
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "images": images,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        try:
            resp = requests.post(
                f"{self.host.rstrip('/')}/api/generate", json=payload, timeout=self.timeout_sec
            )
            resp.raise_for_status()
            data = resp.json()
            text = str(data.get("response", ""))
            parsed = parse_json_from_text(text)
            parsed["raw_text"] = text
            parsed["fallback_used"] = False
            return parsed
        except Exception as exc:
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": f"Ollama inference error: {type(exc).__name__}",
                "fallback_used": True,
                # M11b: redact API keys that may appear in exception text.
                "error": _redact_api_keys(str(exc)),
            }

    def infer_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        try:
            resp = requests.post(
                f"{self.host.rstrip('/')}/api/generate", json=payload, timeout=self.timeout_sec
            )
            resp.raise_for_status()
            data = resp.json()
            text = str(data.get("response", ""))
            parsed = parse_json_from_text(text)
            parsed["raw_text"] = text
            parsed["fallback_used"] = False
            return parsed
        except Exception as exc:
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": f"Ollama text inference error: {type(exc).__name__}",
                "fallback_used": True,
                # M11b: redact API keys that may appear in exception text.
                "error": _redact_api_keys(str(exc)),
            }


@dataclass(slots=True)
class LlamaCppGemmaBackend(BaseLLMBackend):
    """llama.cpp server backend.

    Prefer OpenAI-compatible /v1/chat/completions, and fallback to /completion.
    The default local server address in this project is http://127.0.0.1:8080.
    """

    host: str = "http://127.0.0.1:8080"
    model: str | None = None
    timeout_sec: int = 120
    temperature: float = 0.1
    top_p: float = 0.9
    backend_name: str = "llamacpp"

    def __post_init__(self) -> None:
        # SSRF guard — see ``_validate_llm_host`` for the policy.
        self.host = _validate_llm_host(self.host)

    def infer_panel(
        self,
        panel_image,
        caption_text: str,
        ocr_labels: list[str],
        system_prompt: str,
        user_prompt: str,
        extra_image: Any = None,
    ) -> dict[str, Any]:
        """Run the llama.cpp backend on a vision request.

        Parameters
        ----------
        panel_image : PIL.Image.Image | None
            Primary image (e.g. SEM plate). ``None`` means text-only.
        caption_text : str
            Optional caption text (unused by llama.cpp but kept for
            backend signature parity with ``MiniMaxM3Backend``).
        ocr_labels : list[str]
            Optional OCR-detected panel labels (also kept for parity).
        system_prompt : str
            System prompt.
        user_prompt : str
            User prompt.
        extra_image : PIL.Image.Image | None, default ``None``
            Audit M-14: optional SECOND image. llama.cpp's
            ``/v1/chat/completions`` endpoint accepts only ONE image
            per request (OpenAI-compatible single-image contract), so
            we cannot pass both. We therefore inject a clear prompt
            note that the second image is NOT used by this backend and
            the caller must rely on the strat-column caption text. The
            primary image is still passed to the multimodal endpoint
            so the model sees at least ONE image.
        """
        try:
            text, multimodal_degraded = self._chat_completion(
                panel_image, system_prompt, user_prompt, extra_image=extra_image
            )
            parsed = parse_json_from_text(text)
            parsed["raw_text"] = text
            parsed["fallback_used"] = False
            # Audit M7: propagate the multimodal-degraded flag so the
            # caller knows the model only saw text, not the image.
            # This is critical for radiolarian species ID: if the
            # multimodal path failed and we degraded to text-only,
            # the model's confidence is necessarily lower and the
            # caller may want to mark the result as "vision-fallback".
            parsed["multimodal_degraded"] = multimodal_degraded
            # Audit M-14: flag when a second image was supplied but
            # could not be forwarded (single-image backend) so the
            # caller knows the strat column visual signal is missing.
            if extra_image is not None:
                parsed["extra_image_unsupported"] = True
            return parsed
        except Exception as exc:
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": f"llama.cpp inference error: {type(exc).__name__}",
                "fallback_used": True,
                # M11b: redact API keys that may appear in exception text.
                "error": _redact_api_keys(str(exc)),
            }

    def infer_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            text, _ = self._chat_completion(None, system_prompt, user_prompt)
            parsed = parse_json_from_text(text)
            parsed["raw_text"] = text
            parsed["fallback_used"] = False
            # text-only call — multimodal was never attempted.
            parsed["multimodal_degraded"] = False
            return parsed
        except Exception as exc:
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": f"llama.cpp text inference error: {type(exc).__name__}",
                "fallback_used": True,
                # M11b: redact API keys that may appear in exception text.
                "error": _redact_api_keys(str(exc)),
            }

    def _chat_completion(
        self,
        panel_image,
        system_prompt: str,
        user_prompt: str,
        extra_image: Any = None,
    ) -> tuple[str, bool]:
        """Return ``(text, multimodal_degraded)``.

        Audit M7: ``multimodal_degraded`` is True when the multimodal
        path failed and we degraded to text-only via the
        ``/completion`` endpoint. False in every other case (text-only
        call, successful multimodal call, fallback when no image was
        provided). The caller can use this to flag the result so
        downstream code knows the model only saw text.

        Audit B-4: only TRANSIENT errors (5xx, connection failures,
        schema mismatches) are allowed to fall back to the text-only
        ``/completion`` endpoint. NON-TRANSIENT 4xx errors (401/403
        auth, 402 quota, 404 wrong model, 413 payload too large, etc.)
        are RE-RAISED so the caller sees the real failure instead of
        silently swapping to a degraded path. Silently degrading on
        auth failure hides a misconfigured API key for the lifetime of
        the session.

        Audit M-14: when ``extra_image`` is provided, llama.cpp's
        single-image multimodal contract means we cannot send BOTH
        images. We inject a clear note in the user prompt that the
        second image is not used by this backend — the caller is then
        responsible for relying on the strat-column caption text.
        """
        # Audit M-14: if a secondary image is supplied, inject a clear
        # note into the user prompt so the model knows the strat
        # column image is NOT being forwarded. We do this BEFORE the
        # text-only short-circuit because text-only callers also
        # benefit from the explicit note (so the model's response can
        # mention it in ``reasoning`` for downstream observability).
        if (
            extra_image is not None
            and "strat column image not used by this backend" not in user_prompt
        ):
            user_prompt = (
                "[Note: strat column image not used by this backend — caption-only path]\n\n"
                + user_prompt
            )

        # Text-only calls never attempt the multimodal path.
        if panel_image is None:
            prompt = self._build_text_prompt(system_prompt, user_prompt)
            completion_payload = {
                "prompt": prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": False,
            }
            if self.model:
                completion_payload["model"] = self.model
            resp = requests.post(
                self.host.rstrip("/") + "/completion",
                json=completion_payload,
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = str(data.get("content") or data.get("response") or "")
            return self._clean_response_text(raw), False
        # 1) 优先尝试 OpenAI-compatible chat/completions
        payload = {
            "model": self.model or "default",
            "messages": [
                self._system_message(system_prompt),
                self._user_message(user_prompt, panel_image),
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }
        url = self.host.rstrip("/") + "/v1/chat/completions"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            return self._clean_response_text(self._extract_chat_text(data)), False
        except Exception as exc:
            # Audit B-4: NON-TRANSIENT 4xx errors (401/403 auth,
            # 402 quota, 404 wrong model, 413 payload too large, etc.)
            # are NOT a sign of "multimodal degraded" — they're real
            # client / configuration failures. Re-raise so the
            # caller's ``infer_panel`` exception path surfaces the
            # real reason instead of silently swapping to text-only.
            # Only network / 5xx / parse failures are treated as
            # transient and eligible for the ``/completion`` fallback.
            #
            # Audit 2026-08-19 Phase 4B M-23: distinguish the
            # *category* of 4xx so the caller can route the error
            # correctly. 401/403 → ``LLMAuthenticationError``
            # (refresh the API key); 404 → ``LLMNotFoundError``
            # (fix the model name); 429 → ``LLMRateLimitError``
            # (back off and retry); all other 4xx → re-raise the
            # raw ``HTTPError`` (caller sees the real status code).
            status_code = self._extract_status_code(exc)
            if status_code is not None and 400 <= status_code < 500:
                if status_code in (401, 403):
                    logger.debug(
                        "llama.cpp /v1/chat/completions %d auth error; "
                        "raising LLMAuthenticationError",
                        status_code,
                    )
                    raise LLMAuthenticationError(
                        f"HTTP {status_code} from llama.cpp; check credentials",
                        status_code=status_code,
                    ) from exc
                if status_code == 404:
                    logger.debug("llama.cpp /v1/chat/completions 404; raising LLMNotFoundError")
                    raise LLMNotFoundError(
                        "HTTP 404 from llama.cpp; check model name or endpoint",
                        status_code=status_code,
                    ) from exc
                if status_code == 429:
                    logger.debug(
                        "llama.cpp /v1/chat/completions 429 rate-limited; raising LLMRateLimitError"
                    )
                    raise LLMRateLimitError(
                        "HTTP 429 from llama.cpp; rate-limited",
                        status_code=status_code,
                    ) from exc
                logger.debug(
                    "llama.cpp /v1/chat/completions 4xx (status=%d); not degrading to /completion",
                    status_code,
                )
                raise
            # 2) 回退到 llama.cpp /completion 接口（纯文本）
            # Audit M7: signal the degraded mode so callers can tell
            # "model saw the image" apart from "image was dropped
            # silently".
            logger.debug(
                "llama.cpp /v1/chat/completions failed; falling back to /completion (text-only): %s",
                type(exc).__name__,
            )
            prompt = self._build_text_prompt(system_prompt, user_prompt)
            completion_payload = {
                "prompt": prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": False,
            }
            if self.model:
                completion_payload["model"] = self.model
            resp = requests.post(
                self.host.rstrip("/") + "/completion",
                json=completion_payload,
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = str(data.get("content") or data.get("response") or "")
            return self._clean_response_text(raw), True

    @staticmethod
    def _extract_status_code(exc: BaseException) -> int | None:
        """Extract an HTTP status code from a ``requests`` /
        ``HTTPError`` exception if present.

        Returns ``None`` when the exception has no recognizable
        status code (connection errors, schema mismatches, etc.) so
        the caller treats the failure as transient and eligible for
        the ``/completion`` fallback.

        Audit B-4: this helper exists so the 4xx-not-degrade logic
        works for both ``requests.exceptions.HTTPError`` (which
        attaches ``response.status_code``) and exceptions that expose
        a top-level ``status_code`` attribute (some SDKs do).
        """
        # requests.HTTPError path: ``exc.response.status_code``.
        resp = getattr(exc, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
            if isinstance(code, int):
                return code
        # Top-level ``status_code`` (some custom error wrappers).
        code = getattr(exc, "status_code", None)
        if isinstance(code, int):
            return code
        return None

    def _system_message(self, system_prompt: str) -> dict[str, Any]:
        return {"role": "system", "content": system_prompt}

    def _user_message(self, user_prompt: str, panel_image) -> dict[str, Any]:
        if panel_image is None:
            return {"role": "user", "content": user_prompt}
        data_uri = "data:image/png;base64," + _encode_image_base64(panel_image)
        # OpenAI-compatible multimodal format
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": user_prompt},
            ],
        }

    def _build_text_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Build the /completion prompt.

        Audit 2026-08-17 (live llama.cpp + Qwen3.8-27B): the previous
        implementation worked for Gemma-era backends; for Qwen3 the
        chat template injects a <think>...</think> block whenever the
        prompt ends with ``assistant\\n<think>\\n`` — which we used to
        pre-pend — so the model copied the template back verbatim and
        emitted the real JSON only AFTER the closing ``</think>``.
        ``parse_json_from_text`` then loaded the FIRST JSON object
        (a placeholder / template residue) instead of the real answer.
        Live probe returned ``{"species": "species name"}`` instead of
        ``{"species": "Ceratartia"}``.

        Fix: do NOT prepend the Gemma-style template. /completion has
        no chat-template auto-injection; Qwen3 emits a thinking block
        only when its chat template asks for one (gated by the request's
        ``enable_thinking`` field, which we don't set).
        """
        return f"{system_prompt}\n\n{user_prompt}\n\nPlease output strict JSON only."

    @staticmethod
    def _clean_response_text(text: str) -> str:
        """Strip thinking blocks + fences + whitespace.

        Audit 2026-08-17 (live llama.cpp + Qwen3.8-27B): even with the
        fixed ``_build_text_prompt``, real backends occasionally emit a
        ``<think>...</think>`` segment (especially when the chat
        template auto-injects one). Leaving it in causes
        ``parse_json_from_text`` to load the FIRST JSON object — which
        inside a thinking block is usually a placeholder example like
        ``{"species": "species name"}`` — instead of the real answer
        that comes after the closing ``</think>``.

        Strategy:
        1. Strip ``<think>...</think>`` (non-greedy, case-insensitive)
        2. Strip ``<answer>...</answer>`` (Qwen alternative tag)
        3. Strip ``` fences
        4. Return the LAST non-empty JSON object if more than one
           survived the fence strip (the model occasionally emits a
           "draft" object first and the real answer second). We pick
           last because the real answer typically comes after the draft
           in Qwen3 output streams.
        5. Fallback to the raw stripped text if no JSON object is found
           (so callers see the model's prose instead of an empty string)
        """
        if not text:
            return text
        # Strip think/answer segments
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # <answer> is a wrapper (Qwen uses it to delimit the final
        # answer); KEEP the contents, drop the tags only.
        cleaned = re.sub(
            r"</?answer>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Strip ``` fences (markdown code blocks)
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()
        # If we still see multiple JSON objects, keep the LAST one.
        # We scan balanced braces so we don't get tripped up by braces
        # inside strings.
        last_obj = _last_balanced_json_object(cleaned)
        if last_obj is not None:
            return last_obj
        return cleaned

    def _extract_chat_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("No choices in llama.cpp response")
        choice = choices[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if content is None:
            content = choice.get("text")
        return str(content or "")


def _encode_image_base64(image) -> str:
    from PIL import Image

    if isinstance(image, Image.Image):
        im = image
    else:
        im = Image.fromarray(image)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_image_anthropic_block(image) -> dict[str, Any]:
    """Encode PIL image as Anthropic multimodal content block."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    # Resize if too large (Anthropic recommends < 1568px on long edge for speed)
    max_long_edge = 1568
    w, h = image.size
    if max(w, h) > max_long_edge:
        if w >= h:
            new_w = max_long_edge
            new_h = int(h * max_long_edge / w)
        else:
            new_h = max_long_edge
            new_w = int(w * max_long_edge / h)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64,
        },
    }


# =============================================================================
# MiniMax M3 backend (Anthropic-compatible API)
# =============================================================================

# Cost per million tokens (CNY), reference 2026-06
MiniMax_PRICE_INPUT_PER_M = 2.1
MiniMax_PRICE_OUTPUT_PER_M = 8.4


@dataclass(slots=True)
class MiniMaxM3Backend(BaseLLMBackend):
    """MiniMax M3 API backend, Anthropic-compatible protocol.

    Endpoint: https://api.minimaxi.com/anthropic
    Auth:     Token Plan subscription key (ANTHROPIC_API_KEY env or explicit param)
    Model:    MiniMax-M3 (default)

    Features:
      - Native multimodal (image + text)
      - Up to 1M token context (MSA architecture)
      - Optional extended thinking (default ON)
      - Token-usage and cost accounting per call
    """

    api_key: str
    base_url: str = "https://api.minimaxi.com/anthropic"
    model: str = "MiniMax-M3"
    max_output_tokens: int = 2048
    thinking_budget_tokens: int = 1024
    enable_thinking: bool = False  # default OFF to avoid surprise API cost; opt-in via CLI/UI
    timeout_sec: int = 120
    temperature: float = 0.1
    top_p: float = 0.9
    max_retries: int = 3
    max_concurrent: int = 8
    backend_name: str = "MiniMax"
    # Callback invoked when an error occurs and fallback is needed.
    # Signature: (error_info: dict) -> "gemma4" | "rules" | "stop"
    on_error: Callable[[dict[str, Any]], str] | None = None
    # Data-outbound policy. One of:
    #   * "api_full"    - send the full panel image + caption + OCR + GROBID
    #                     paragraphs to the MiniMax API. This is the historical
    #                     default and the only mode the upstream service
    #                     actually consumes.
    #   * "api_redacted"- strip long, identifiable text from the outbound
    #                     payload (full caption -> first 200 chars;
    #                     drop OCR text; replace panel image with a 256x256
    #                     thumbnail). Useful for sensitive preprints
    #                     (location-bearing geological context, copyright
    #                     concerns) where the model only needs the
    #                     species cue, not the whole text.
    #   * "local_only"  - never contact the MiniMax API. ``infer_panel``
    #                     and ``infer_text`` return a deterministic
    #                     no-op result that the surrounding pipeline
    #                     treats as "fallback_used=True", forcing the
    #                     rule-based path to be authoritative. This
    #                     is the correct setting for offline / air-gapped
    #                     deployments (M3 weights not yet open-sourced,
    #                     privacy-sensitive papers).
    # Audit 2026-09-03 (BLOCKER-#2): default flipped from ``api_full``
    # to ``api_redacted`` so a fresh pipeline run does NOT silently ship
    # the full PDF / panel image / full caption text / OCR text / GROBID
    # paragraphs to the MiniMax cloud. Operators must opt in explicitly
    # to ``api_full`` (full-resolution image + verbatim caption) via one of:
    #   * Environment variable:  RLPE_DATA_OUTBOUND_OPT_IN=1
    #   * CLI flag:              --i-understand-data-leaves-my-machine
    # The ``__post_init__`` below enforces the opt-in for any caller
    # that selects ``api_full``; the default policy ``api_redacted``
    # downscales images to 256x256 and caps the caption at 200 chars,
    # which was sufficient for Round 6 live OA (92.5% species rate on 5
    # papers with api_redacted alone — the historical api_full default
    # was a privacy posture, not an accuracy lever).
    data_outbound_policy: str = "api_redacted"

    def __post_init__(self) -> None:
        # Audit M11: ``max_concurrent=0`` would create a
        # ``Semaphore(0)`` which never grants — the first call to
        # ``_call_api`` would block forever (deadlock). Validate
        # ``max_concurrent >= 1`` at construction so misconfiguration
        # surfaces immediately with a clear error rather than as a
        # hang deep in the pipeline.
        if int(self.max_concurrent) < 1:
            raise ValueError(
                f"max_concurrent must be >= 1 (got {self.max_concurrent!r}); "
                f"a value of 0 would deadlock the MiniMax _sem."
            )
        # ``local_only`` does not need an API key: the backend will refuse
        # every outbound request and return a no-op result, which the
        # surrounding pipeline treats as a fallback.
        if self.data_outbound_policy not in {"api_full", "api_redacted", "local_only"}:
            raise ValueError(
                f"data_outbound_policy must be one of api_full/api_redacted/local_only, "
                f"got {self.data_outbound_policy!r}"
            )
        # Audit 2026-09-03 (BLOCKER-#2): ``api_full`` is opt-in by design.
        # The historical default silently sent full panel images + verbatim
        # captions to the MiniMax cloud, which is inappropriate for (a)
        # unpublished preprints and (b) copyright-restricted SEM plates.
        # Require an explicit env var (``RLPE_DATA_OUTBOUND_OPT_IN=1``) or
        # the matching CLI flag before allowing ``api_full`` to remain in
        # effect. ``cli.py`` translates ``--i-understand-data-leaves-my-machine``
        # into the same env var, so any caller path is covered.
        if self.data_outbound_policy == "api_full":
            opt_in = os.environ.get("RLPE_DATA_OUTBOUND_OPT_IN", "").strip().lower()
            if opt_in not in {"1", "true", "yes", "on"}:
                raise ValueError(
                    "data_outbound_policy='api_full' is opt-in only. Set "
                    "RLPE_DATA_OUTBOUND_OPT_IN=1 in the environment or pass "
                    "--i-understand-data-leaves-my-machine to the CLI before "
                    "selecting api_full. Use 'api_redacted' (default) or "
                    "'local_only' for the private posture."
                )
        # Phase 54 audit: B2 — SSRF guard. Ollama / LlamaCpp both call
        # ``_validate_llm_host`` in their ``__post_init__``; MiniMax did
        # not, so a job with ``MiniMax_endpoint="http://169.254.169.254/..."``
        # would ship the panel image + caption + the Authorization header
        # (which carries the API key) to the AWS / GCP / Azure metadata
        # endpoint. ``_validate_llm_host`` blocks link-local / unspecified /
        # multicast addresses and non-http(s) schemes; setting
        # ``RLPE_LLM_ALLOW_ANY_HOST=1`` overrides (for hardened networks).
        # Skip the check for ``local_only`` because the backend never makes
        # an outbound call in that mode.
        if self.data_outbound_policy != "local_only":
            self.base_url = _validate_llm_host(self.base_url)
        if not self.api_key and self.data_outbound_policy != "local_only":
            raise ValueError(
                "MiniMax api_key is required (set ANTHROPIC_API_KEY env or pass explicitly)."
            )
        # The ``local_only`` policy promises "no network, no SDK required":
        # the backend short-circuits every ``infer_*`` call to a deterministic
        # no-op result. In an air-gapped / offline deployment, the user
        # frequently does NOT have the ``anthropic`` SDK installed at all,
        # so the previous unconditional import broke construction. Skip the
        # SDK import (and the ``Anthropic()`` client init) entirely when the
        # policy says we will never make an outbound call.
        if self.data_outbound_policy == "local_only":
            self._anthropic = None
            self._client = None
        else:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic SDK not installed. Run: pip install 'anthropic>=0.40,<0.50' "
                    "(or set data_outbound_policy=local_only to run without the SDK)."
                ) from exc
            self._anthropic = anthropic
            # Audit M8: the anthropic SDK defaults to ``max_retries=2``
            # which silently multiplies with our outer 3-attempt loop
            # (effectively 3 * (1+2) = 9 attempts on a 500). Disable
            # the SDK's internal retries so the outer loop in
            # ``_call_api`` is the sole retry mechanism and the
            # ``total_calls`` counter remains accurate.
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_sec,
                max_retries=0,
            )
        # Per-process semaphore (cheap; limits concurrent in-flight requests).
        self._sem = threading.Semaphore(self.max_concurrent)
        # Per-thread context storage — prevents concurrent threads (e.g. m3_engine
        # workers) from overwriting each other's caption/OCR state mid-call.
        self._thread_local = threading.local()
        # Running totals (read by callers for cost dashboards).
        self._lock = threading.Lock()
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_calls: int = 0
        self.total_errors: int = 0
        # Phase 61 Plan 4 (Bug 4.8): dedicated counter for "JSON parse
        # failure but extended thinking was present". Operators need to
        # know whether a paid call returned reasoning tokens but a
        # malformed JSON body — that's a model-quality signal very
        # different from "no thinking happened" or "API timed out".
        self.failed_with_thinking: int = 0
        # Phase 61 Plan 4 (Bug 4.10): number of 4xx retries that
        # SHOULD have used the configured fallback backend. Surfaced
        # in /system/llm-status so operators can spot when their
        # PipelineConfig.extra["fallback_llm_backend"] would have
        # rescued a run.
        self.fallback_4xx_hints: int = 0
        # The configured fallback backend name (string), or None if no
        # fallback was wired. Set via ``set_fallback_backend()`` from
        # the pipeline's config-loader step.
        self._configured_fallback: str | None = None

    # ------------------------------------------------------------------ helpers

    def _build_user_content(
        self,
        panel_image,
        user_prompt: str,
        extra_image: Any = None,
    ) -> list[dict[str, Any]]:
        """Build the Anthropic multimodal user-content block.

        Parameters
        ----------
        panel_image : PIL.Image.Image | None
            The primary image (e.g. SEM plate). When ``None``, no image
            block is emitted.
        user_prompt : str
            Text prompt from the caller.
        extra_image : PIL.Image.Image | None, default ``None``
            Optional secondary image (e.g. strat column /
            paleogeographic map). When provided AND ``panel_image`` is
            not ``None``, the secondary image is appended as a SECOND
            image block so the model sees BOTH images together. Used
            by ``cross_figure_visual_inference`` for plate ↔ strat
            column cross-modal grounding (Phase C.1 / audit M-14).

        Notes
        -----
        * If ``extra_image`` is provided but ``panel_image`` is None,
          ``extra_image`` is still passed as the single image — this
          keeps the contract symmetric.
        """
        content: list[dict[str, Any]] = []
        if panel_image is not None:
            content.append(_encode_image_anthropic_block(panel_image))
        # audit M-14: support dual-image vision calls. When ``extra_image``
        # is provided we append a SECOND image block so the model can
        # reason over BOTH images (e.g. SEM plate + strat column) in a
        # single API request.
        if extra_image is not None:
            content.append(_encode_image_anthropic_block(extra_image))
        # P2-9 fix (Plan B): prepend OCR-detected panel labels and figure
        # caption to the user prompt when available. This enriches the LLM's
        # context with text that was detected independently of the user_prompt
        # (which is constructed by m3_engine with its own caption parsing).
        # Only prepend if not already embedded in user_prompt to avoid duplication.
        extra_parts: list[str] = []
        ocr_labels = getattr(self._thread_local, "ocr_labels", None) or []
        if ocr_labels:
            labels_str = ", ".join(sorted(set(ocr_labels)))
            extra_parts.append(f"[Panel labels (OCR): {labels_str}]")
        caption_text = getattr(self._thread_local, "caption_text", None) or ""
        if caption_text and caption_text not in user_prompt:
            extra_parts.append(f"[Figure caption: {caption_text}]")
        if extra_parts:
            user_prompt = "\n".join(extra_parts) + "\n" + user_prompt
        content.append({"type": "text", "text": user_prompt})
        return content

    def _build_messages(
        self,
        panel_image,
        user_prompt: str,
        extra_image: Any = None,
    ) -> list[dict[str, Any]]:
        """Build messages list for the Anthropic Messages API.

        Parameters
        ----------
        panel_image : PIL.Image.Image | None
            Primary image (e.g. SEM plate).
        user_prompt : str
            User text prompt.
        extra_image : PIL.Image.Image | None, default ``None``
            Optional secondary image (e.g. strat column). When
            provided, BOTH images are included as content blocks so
            the model can reason across them — see audit M-14.
        """
        return [
            {
                "role": "user",
                "content": self._build_user_content(panel_image, user_prompt, extra_image),
            }
        ]

    def _build_text_messages(self, user_prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": user_prompt}]

    def _build_request_kwargs(
        self, system_prompt: str, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Audit M2: per the Anthropic protocol, thinking tokens count
        # toward the ``max_tokens`` budget. The previous code only took
        # the max of (max_output_tokens, thinking_budget + 256) which
        # ignored thinking cost when ``max_output_tokens`` was already
        # large. Now we explicitly add the thinking budget when
        # thinking is enabled, so the budget covers BOTH the thinking
        # tokens AND the final output tokens the caller requested.
        max_out = self.max_output_tokens + (
            self.thinking_budget_tokens if self.enable_thinking else 0
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_out,
            "system": system_prompt,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.enable_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
        return kwargs

    @staticmethod
    def _parse_retry_after(exc: BaseException) -> float | None:
        """Extract the ``Retry-After`` header value (seconds) from an
        Anthropic SDK exception, if present.

        Per RFC 7231 §7.1.3, ``Retry-After`` may be either a number of
        seconds (decimal integer) OR an HTTP-date. This helper accepts
        the numeric form only — the date form is rare in practice for
        rate-limit responses and harder to parse safely, so we ignore it
        and let the regular backoff take over.

        Returns ``None`` when:
        * the exception has no ``response`` attribute,
        * the response has no ``headers`` attribute,
        * the header is missing or empty,
        * the value cannot be coerced to a positive float.

        Audit M-4: the previous retry loop ignored ``Retry-After`` and
        always used exponential backoff, which can prolong the
        rate-limit window and contribute to a thundering herd. Honoring
        the header is the canonical fix; we cap the wait at 60s so a
        hostile or buggy server can't pin a worker for minutes.
        """
        resp = getattr(exc, "response", None)
        if resp is None:
            return None
        headers = getattr(resp, "headers", None)
        if headers is None:
            return None
        try:
            raw = headers.get("Retry-After")  # type: ignore[union-attr]
        except Exception:
            return None
        if raw is None:
            return None
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    @staticmethod
    def _parse_retry_after_header(value: str | None) -> float:
        """Parse a ``Retry-After`` header value (string form).

        Audit 2026-08-19 Phase 4B M-24: per RFC 7231 §7.1.3 the header
        can be EITHER a non-negative integer (delta-seconds) OR an
        HTTP-date (``Wed, 21 Oct 2026 07:28:00 GMT``). The
        Phase-2c ``_parse_retry_after(exc)` ignores the HTTP-date
        form (returns ``None``) so the existing Phase-2c tests still
        pass — this new helper is the string-input companion that
        accepts BOTH forms so future retry loops can pick it up
        without a breaking change.

        Returns:
        * ``0.0`` when ``value`` is ``None``, empty, or unparseable
          (the caller falls back to exponential backoff).
        * ``min(parsed_seconds, 60.0)`` for the integer-seconds form.
        * ``max(0.0, min(delta_to_date_seconds, 60.0))`` for the
          HTTP-date form. A date in the past yields ``0.0``.

        The 60-second cap protects against hostile / buggy servers
        that pin a worker for minutes — the existing exponential
        backoff path is the safety net beyond that.
        """
        if not value:
            return 0.0
        text = str(value).strip()
        if not text:
            return 0.0
        # 1) Numeric (delta-seconds) form per RFC 7231.
        try:
            seconds = float(text)
            return max(0.0, min(seconds, 60.0))
        except (TypeError, ValueError):
            pass
        # 2) HTTP-date form per RFC 7231 (RFC 1123 / RFC 850 / asctime).
        try:
            from datetime import datetime
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(text)
            if dt is None:
                return 0.0
            # Compare against ``datetime.now(dt.tzinfo)`` so a
            # timezone-aware server time yields a correct delta.
            # A naive ``dt`` (no tzinfo) is treated as local time.
            now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
            delta = (dt - now).total_seconds()
            return max(0.0, min(delta, 60.0))
        except Exception:
            return 0.0

    def _call_api(self, system_prompt: str, messages: list[dict[str, Any]]):
        """Make API call with retry + rate-limit handling.

        If all retries are exhausted on a retriable error, raises the last
        exception. The caller (typically ``infer_panel`` / ``infer_text``) is
        responsible for invoking ``self.on_error`` — we do NOT call it here,
        to avoid double-invocation when the surrounding pipeline also
        routes errors through its own ``gemma_fallback_handler``.
        """
        kwargs = self._build_request_kwargs(system_prompt, messages)
        anthropic_mod = self._anthropic
        # Guard: in ``local_only`` mode the anthropic SDK is never imported
        # (``self._anthropic`` is None and ``self._client`` is None). The
        # public ``infer_panel`` / ``infer_text`` methods short-circuit
        # before reaching here, but a future code path that bypasses the
        # short-circuit would hit ``None.RateLimitError`` in the except
        # clauses below — an ``AttributeError`` that masks the real issue.
        if anthropic_mod is None or self._client is None:
            raise RuntimeError(
                "MiniMax _call_api invoked without an Anthropic client "
                "(data_outbound_policy=local_only or SDK not installed)."
            )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            # Audit M7: bump total_calls BEFORE the API call so a
            # failed attempt (the final one in a retry-exhausted
            # sequence) is still counted. If the call succeeds we
            # immediately bump usage below; if it fails the counter
            # still reflects the slot used. Decrementing on success
            # would require another lock + read-modify-write and is
            # more error-prone than the ``+1 at entry`` pattern.
            with self._lock:
                self.total_calls += 1
            try:
                with self._sem:
                    resp = self._client.messages.create(**kwargs)
                with self._lock:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        self.total_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                        self.total_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
                return resp
            except anthropic_mod.RateLimitError as exc:
                last_exc = exc
                # Audit M9: add jitter (random.uniform(0, 1)) to
                # exponential backoff so concurrent workers don't all
                # retry at the same instant after a 429 (thundering
                # herd against the rate-limited endpoint).
                # Audit M4: respect the ``Retry-After`` header sent by
                # the server when present — it overrides the
                # exponential backoff so we don't hammer the endpoint
                # before the rate-limit window closes.
                retry_after = self._parse_retry_after(exc)
                if retry_after is not None:
                    wait = min(retry_after + random.uniform(0, 1), 60.0)
                    logger.warning(
                        "MiniMax rate-limited (attempt %d); Retry-After=%ss, sleeping %ds: %s",
                        attempt + 1,
                        retry_after,
                        wait,
                        # M11: redact API keys that may appear in exception text.
                        _redact_api_keys(str(exc)),
                    )
                else:
                    wait = min(2**attempt, 30) + random.uniform(0, 1)
                    logger.warning(
                        "MiniMax rate-limited (attempt %d), sleeping %ds: %s",
                        attempt + 1,
                        wait,
                        # M11: redact API keys that may appear in exception text.
                        _redact_api_keys(str(exc)),
                    )
                time.sleep(wait)
            except anthropic_mod.APIConnectionError as exc:
                last_exc = exc
                wait = min(2**attempt, 30) + random.uniform(0, 1)
                logger.warning(
                    "MiniMax connection error (attempt %d), sleeping %ds: %s",
                    attempt + 1,
                    wait,
                    # M11: redact API keys that may appear in exception text.
                    _redact_api_keys(str(exc)),
                )
                time.sleep(wait)
            except anthropic_mod.APIStatusError as exc:
                last_exc = exc
                status = getattr(exc, "status_code", 500)
                # Retry policy:
                #   - 5xx and 429: always retry (transient).
                #   - 401 / 403: retry — these are often transient, e.g. an
                #     auth token that expired mid-session or a key
                #     rotation; the second attempt will surface the real
                #     failure to the user if it's permanent.
                #   - All other 4xx (400 / 404 / 422): fail fast, the
                #     request is malformed or the resource doesn't
                #     exist and retrying won't help.
                if status >= 500 or status == 429:
                    # Audit M4: prefer ``Retry-After`` from the
                    # response headers when present. 429 + Retry-After
                    # is the canonical signal "wait N seconds before
                    # retrying" — ignoring it and using exponential
                    # backoff instead leads to a thundering herd that
                    # can extend the rate-limit window.
                    retry_after = self._parse_retry_after(exc)
                    if retry_after is not None:
                        wait = min(retry_after + random.uniform(0, 1), 60.0)
                        logger.warning(
                            "MiniMax %d (attempt %d); Retry-After=%ss, sleeping %ds: %s",
                            status,
                            attempt + 1,
                            retry_after,
                            wait,
                            # M11: redact API keys that may appear in exception text.
                            _redact_api_keys(str(exc)),
                        )
                    else:
                        wait = min(2**attempt, 30) + random.uniform(0, 1)
                        logger.warning(
                            "MiniMax %d (attempt %d), sleeping %ds: %s",
                            status,
                            attempt + 1,
                            wait,
                            # M11: redact API keys that may appear in exception text.
                            _redact_api_keys(str(exc)),
                        )
                    time.sleep(wait)
                elif status in (401, 403):
                    # M10: auth errors (401/403) are not transient — retrying
                    # wastes quota and won't fix a bad/missing/expired key.
                    # Re-raise immediately so callers see the real failure on
                    # the first attempt.  Audit M4: bump ``total_errors``
                    # BEFORE the raise so the counter reflects this
                    # non-retryable failure (the retry-exhaustion block
                    # below is never reached for this branch).
                    with self._lock:
                        self.total_errors += 1
                    logger.warning(
                        "MiniMax %d (non-retryable auth error): %s",
                        status,
                        # M11: redact API keys that may appear in exception text.
                        _redact_api_keys(str(exc)),
                    )
                    raise
                else:
                    # 4xx (not retryable) -> raise with fallback recommendation.
                    # Phase 61 Plan 4 (Bug 4.10): surface a fallback
                    # recommendation so a higher-level orchestrator can
                    # retry once with a different backend (configured via
                    # PipelineConfig.extra["fallback_llm_backend"]).
                    try:
                        recommended = select_backend_after_4xx(
                            current_backend=self.backend_name,
                            configured_fallback=getattr(self, "_configured_fallback", None),
                            attempts_made=attempt + 1,
                        )
                        if recommended != self.backend_name:
                            logger.info(
                                "MiniMax 4xx: switching to fallback "
                                "backend %r (current=%r, attempts=%d)",
                                recommended,
                                self.backend_name,
                                attempt + 1,
                            )
                            with self._lock:
                                self.fallback_4xx_hints += 1
                            # Audit M4: bump ``total_errors`` BEFORE the
                            # raise so the 4xx-with-fallback-hint path
                            # also reflects in the error counter.
                            with self._lock:
                                self.total_errors += 1
                            raise FallbackRecommendedError(
                                f"MiniMax 4xx error, fallback {recommended} recommended",
                                recommended_backend=recommended,
                            ) from exc
                        # Audit M3: no real fallback configured (the
                        # helper returns the same backend when nothing
                        # better is wired up). Fail fast — don't loop
                        # 3 more times against a permanently broken
                        # request (malformed body, missing resource,
                        # etc.). Bump ``total_errors`` and re-raise the
                        # last exception so the caller's
                        # ``_make_error_result`` propagates the real
                        # reason.
                        with self._lock:
                            self.total_errors += 1
                        raise last_exc if last_exc is not None else exc
                    except FallbackRecommendedError:
                        raise
        # Retry exhaustion. ``total_calls`` already reflects every
        # attempt (bumped at entry above); just bump errors once.
        with self._lock:
            self.total_errors += 1
        if last_exc is not None:
            logger.exception(
                "MiniMax API call failed after %d retries: %s: %s",
                self.max_retries,
                type(last_exc).__name__,
                # M11: redact API keys that may appear in exception text.
                _redact_api_keys(str(last_exc)),
            )
            raise last_exc
        raise RuntimeError("MiniMax call failed without explicit exception")

    def _extract_text(self, response) -> str:
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return str(getattr(block, "text", ""))
        return ""

    def _extract_thinking(self, response) -> str:
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "thinking":
                parts.append(str(getattr(block, "thinking", "")))
        return "\n".join(parts)

    def _make_result(self, response) -> dict[str, Any]:
        text = self._extract_text(response)
        thinking = self._extract_thinking(response)
        try:
            parsed = parse_json_from_text(text)
        except Exception as exc:
            # Audit M6: route ALL parse failures (with or without
            # thinking) through a single helper so ``total_errors``
            # is bumped exactly once. Previously the
            # ``failed_with_thinking`` branch bumped both counters and
            # the no-thinking path bumped neither, leaving the
            # ``total_errors`` rate meaningless for "API OK, model
            # returned garbage" failures.
            has_thinking = bool(thinking and thinking.strip())
            try:
                self._record_parse_failure(thinking=thinking, has_thinking=has_thinking)
            except Exception:
                # Counter bookkeeping must never break the result path.
                pass
            # Audit M14: the previous code embedded ``str(exc)`` and
            # ``type(exc).__name__`` into ``reasoning`` and ``error``
            # without redacting API keys. Although the
            # ``parse_json_from_text`` exception text doesn't itself
            # contain keys, future error paths (e.g. SDK parse errors
            # when streaming) may surface a leaked key. Redact
            # defensively before writing into the result dict so the
            # key never lands in ``matches.jsonl`` or the Web UI.
            safe_exc = _redact_api_keys(str(exc))
            # Set `error` / `error_type` so downstream code (e.g.
            # apply_gemma_to_matches) can propagate it to match.metadata
            # and the FallbackHandler popup shows the real reason.
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": f"MiniMax JSON parse error: {type(exc).__name__}: {safe_exc}",
                "fallback_used": True,
                "error": f"{type(exc).__name__}: {safe_exc}",
                "error_type": "JSONParseError",
                "raw_text": text,
                "thinking": thinking,
            }
        parsed["raw_text"] = text
        parsed["thinking"] = thinking
        parsed["fallback_used"] = False
        parsed["request_id"] = getattr(response, "id", None)
        parsed["model_version"] = getattr(response, "model", self.model)
        usage = getattr(response, "usage", None)
        if usage:
            in_t = int(getattr(usage, "input_tokens", 0) or 0)
            out_t = int(getattr(usage, "output_tokens", 0) or 0)
            parsed["usage"] = {"input_tokens": in_t, "output_tokens": out_t}
            parsed["cost_cny"] = round(
                in_t / 1_000_000 * MiniMax_PRICE_INPUT_PER_M
                + out_t / 1_000_000 * MiniMax_PRICE_OUTPUT_PER_M,
                6,
            )
        return parsed

    def _make_error_result(self, exc: Exception) -> dict[str, Any]:
        # Security: ``str(exc)`` from the Anthropic SDK can include the
        # raw API key in the body of 401/403 messages (e.g. "API key not
        # valid: sk-ant-...redacted"). We strip common key prefixes before
        # the message lands in ``match.metadata["gemma_error"]`` and gets
        # written to matches.jsonl where the user can inspect it.
        raw_exc = str(exc)
        safe_exc = _redact_api_keys(raw_exc)
        return {
            "label": None,
            "species": None,
            "confidence": 0.0,
            "reasoning": f"MiniMax API error: {type(exc).__name__}: {safe_exc}",
            "fallback_used": True,
            "error": safe_exc,
            "error_type": type(exc).__name__,
        }

    # ------------------------------------------------------------------ public

    def _local_only_noop(self, reason: str) -> dict[str, Any]:
        """Deterministic no-op result used in local_only mode.

        The shape matches ``_make_error_result`` so the caller's
        fallback handler is triggered with a uniform error type.
        The caller (``apply_gemma_to_matches``) sees
        ``fallback_used=True`` and ``confidence=0.0`` and keeps the
        rule-based prediction. No network IO is attempted.
        """
        return {
            "label": None,
            "species": None,
            "confidence": 0.0,
            "reasoning": f"local_only policy: {reason}",
            "fallback_used": True,
            "error": reason,
            "error_type": "LocalOnlyPolicy",
        }

    def _redact_image(self, panel_image) -> Any:
        """Replace a panel image with a 256-px thumbnail.

        Preserves enough visual context for species cues (shape,
        symmetry) while discarding enough pixel detail to make the
        payload non-identifying for sensitive papers. Always returns
        a PIL Image (not raw bytes) so the downstream
        ``_encode_image_anthropic_block`` can reliably call
        ``Image.fromarray`` / ``image.save(format='PNG')`` without
        a type error.
        """
        if panel_image is None:
            return None
        try:
            from PIL import Image  # local import: PaddleOCR/EasyOCR users

            # may not have PIL in the path.
            if isinstance(panel_image, Image.Image):
                img = panel_image.copy()
                img.thumbnail((256, 256), Image.LANCZOS)
                return img
        except Exception:
            pass
        # Numpy / cv2 path: convert to PIL Image, resize, return PIL.
        # The previous version returned raw JPEG bytes here, which
        # broke ``_encode_image_anthropic_block`` (it only accepts
        # PIL Image or numpy ndarray -- neither of which is a
        # ``bytes`` object). Returning a PIL Image keeps the type
        # contract consistent across all call sites.
        try:
            import numpy as np
            from PIL import Image

            arr = np.asarray(panel_image)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                img = Image.fromarray(arr[..., ::-1])  # BGR -> RGB
            else:
                img = Image.fromarray(arr)
            img.thumbnail((256, 256), Image.LANCZOS)
            return img
        except Exception:
            return None

    def _redact_text(self, text: str, limit: int = 200) -> str:
        """Truncate long text payloads to a hard length cap.

        The first N characters usually contain the species
        enumeration, which is what the model needs; the rest is
        supporting prose that may include location/formation
        context the operator wants to keep private.
        """
        if not text:
            return ""
        s = str(text)
        if len(s) <= limit:
            return s
        return s[:limit] + " ...[truncated by api_redacted policy]"

    def _apply_outbound_policy(
        self, panel_image, caption_text: str, ocr_labels: list[str], user_prompt: str
    ) -> tuple[Any, str, str, list[str]]:
        """Return the redacted (panel_image, user_prompt, caption_text,
        ocr_labels) tuple based on ``self.data_outbound_policy``.

        Audit 2026-09-04 llm-2: the previous signature returned only
        ``(panel_image, user_prompt)``. caption_text and ocr_labels
        were silently ignored in the function body, then re-attached
        to the outgoing request via a thread-local side channel
        (``_build_user_content`` prepends ``[Figure caption: ...]``
        from ``self._thread_local.caption_text``). So even when the
        user opted into ``api_redacted``, the full paper caption
        still left the machine. Fix: redact caption_text /
        ocr_labels HERE and return them in the tuple so the
        thread-local stores the redacted version, not the original.
        """
        if self.data_outbound_policy == "api_full":
            return panel_image, user_prompt, caption_text, ocr_labels
        if self.data_outbound_policy == "local_only":
            # Both branches return the same shape; the caller checks
            # policy and short-circuits before this is even invoked,
            # so this is just defensive in case the policy is changed
            # at runtime.
            return (
                self._redact_image(panel_image),
                self._redact_text(user_prompt, limit=0),
                "",  # caption dropped
                [],  # ocr_labels dropped
            )
        # api_redacted
        return (
            self._redact_image(panel_image),
            self._redact_text(user_prompt, limit=200),
            "",  # caption dropped — paper-private caption text
            [],  # ocr_labels dropped — could be inferred from image
        )

    def infer_panel(
        self,
        panel_image,
        caption_text: str,
        ocr_labels: list[str],
        system_prompt: str,
        user_prompt: str,
        extra_image: Any = None,
    ) -> dict[str, Any]:
        """Run the Anthropic Messages API on a vision request.

        Parameters
        ----------
        panel_image : PIL.Image.Image | None
            Primary image (e.g. SEM plate panel). ``None`` is allowed
            (text-only mode) and matches the previous contract.
        caption_text : str
            Optional caption text associated with ``panel_image``.
        ocr_labels : list[str]
            Optional OCR-detected panel labels.
        system_prompt : str
            System prompt for the model.
        user_prompt : str
            User prompt text.
        extra_image : PIL.Image.Image | None, default ``None``
            Audit M-14: optional SECOND image (e.g. strat column /
            paleogeographic map). When provided AND ``panel_image`` is
            not ``None``, BOTH images are sent as separate image blocks
            so the model can ground one against the other in a single
            API call. Backward-compatible: existing callers that pass
            no extra image behave exactly as before.
        """
        if self.data_outbound_policy == "local_only":
            return self._local_only_noop("MiniMax disabled (data_outbound_policy=local_only)")
        # P2-9 fix (Plan B): store caption_text / ocr_labels so _build_user_content
        # can prepend them to the user prompt. This keeps all callers (including
        # m3_engine which passes empty strings) working while enabling future callers
        # to pass actual caption / OCR context through these parameters.
        #
        # Audit 2026-09-04 llm-2: must store the REDACTED caption /
        # ocr_labels (returned from _apply_outbound_policy), not the
        # originals. Otherwise the thread-local re-attachment in
        # _build_user_content leaks the paper's caption text even
        # when the user opted into api_redacted.
        try:
            img, up, redacted_caption, redacted_ocr = self._apply_outbound_policy(
                panel_image, caption_text, ocr_labels, user_prompt
            )
            self._thread_local.caption_text = redacted_caption or ""
            self._thread_local.ocr_labels = redacted_ocr or []
            # Audit M-14: redact the secondary image under the
            # ``api_redacted`` / ``local_only`` policies too, so a
            # caller asking for "extra image" never leaks the strat
            # column when the user opted out of outbound image data.
            extra = extra_image
            if self.data_outbound_policy in ("api_redacted", "local_only"):
                extra = self._redact_image(extra) if extra is not None else None
            messages = self._build_messages(img, up, extra_image=extra)
            resp = self._call_api(system_prompt, messages)
            return self._make_result(resp)
        except FallbackRecommendedError:
            # Let FallbackRecommendedError propagate to the caller (m3_engine)
            # so it can switch to the configured fallback backend.
            raise
        except Exception as exc:
            return self._make_error_result(exc)

    def infer_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.data_outbound_policy == "local_only":
            return self._local_only_noop("MiniMax disabled (data_outbound_policy=local_only)")
        try:
            _, up, _, _ = self._apply_outbound_policy(None, "", [], user_prompt)
            messages = self._build_text_messages(up)
            resp = self._call_api(system_prompt, messages)
            return self._make_result(resp)
        except FallbackRecommendedError:
            raise
        except Exception as exc:
            return self._make_error_result(exc)

    def cost_summary(self) -> dict[str, Any]:
        with self._lock:
            in_t = self.total_input_tokens
            out_t = self.total_output_tokens
            calls = self.total_calls
            errs = self.total_errors
            failed_thinking = self.failed_with_thinking
        cost = round(
            in_t / 1_000_000 * MiniMax_PRICE_INPUT_PER_M
            + out_t / 1_000_000 * MiniMax_PRICE_OUTPUT_PER_M,
            4,
        )
        return {
            "calls": calls,
            "errors": errs,
            "input_tokens": in_t,
            "output_tokens": out_t,
            "total_cost_cny": cost,
            # Phase 61 Plan 4 (Bug 4.8): surface failed-with-thinking
            # rate separately so dashboards can distinguish "API error"
            # from "API returned reasoning but malformed JSON body".
            "failed_with_thinking": failed_thinking,
        }

    def record_failed_with_thinking(self, thinking_text: str | None = None) -> None:
        """Phase 61 Plan 4 (Bug 4.8): bump the failed-with-thinking
        counter AND the total_errors counter. The two are tracked
        independently so a dashboard can show
        ``failed_with_thinking / total_calls`` as a model-quality KPI.

        ``thinking_text`` is optional — present for future use (e.g. to
        extract token counts) but not required today.

        Audit M6: delegates to ``_record_parse_failure`` so the
        ``total_errors`` bump is centralised and never double-counted
        with the caller's own bookkeeping.
        """
        self._record_parse_failure(
            thinking=thinking_text, has_thinking=bool(thinking_text and thinking_text.strip())
        )

    def _record_parse_failure(
        self, *, thinking: str | None = None, has_thinking: bool | None = None
    ) -> None:
        """Audit M6: single source of truth for JSON parse failure
        bookkeeping. Bumps ``total_errors`` exactly once, and bumps
        ``failed_with_thinking`` only when ``has_thinking`` is True.
        Both counters are guarded by ``self._lock`` so concurrent
        threads see consistent values.

        ``thinking`` is the raw text (kept for future token-count
        extraction). ``has_thinking`` defaults to ``bool(thinking and
        thinking.strip())`` if not provided explicitly.
        """
        if has_thinking is None:
            has_thinking = bool(thinking and thinking.strip())
        with self._lock:
            self.total_errors += 1
            if has_thinking:
                self.failed_with_thinking += 1
        logger.debug(
            "MiniMax: JSON parse failure (has_thinking=%s, failed_with_thinking=%d, total_errors=%d)",
            has_thinking,
            self.failed_with_thinking,
            self.total_errors,
        )

    def llm_status(self) -> dict[str, Any]:
        """Phase 61 Plan 4 (Bug 4.8): thin alias for ``cost_summary``
        used by the ``/system/llm-status`` API route. Kept as a
        separate method so the route handler does not need to know the
        internal cost-summary field names."""
        base = self.cost_summary()
        base["fallback_4xx_hints"] = self.fallback_4xx_hints
        base["configured_fallback"] = self._configured_fallback
        return base

    def set_fallback_backend(self, name: str | None) -> None:
        """Phase 61 Plan 4 (Bug 4.10): wire the configured fallback
        backend name into this backend so the 4xx retry loop can
        recommend a switch via ``select_backend_after_4xx``. ``None``
        clears the recommendation."""
        self._configured_fallback = name or None


# =============================================================================
# FallbackHandler: user-prompted fallback when MiniMax API errors occur
# =============================================================================

# Type alias for the action the handler returns.
FallbackAction = str  # one of: "gemma4", "rules", "stop", "retry"


def cli_fallback_prompt(error_info: dict[str, Any]) -> FallbackAction:
    """Default CLI fallback: print to stderr, read from stdin.

    Designed for terminal use; safe to call from background threads because
    the surrounding pipeline holds a lock around the Gemma call.

    Audit L3: if stdin is NOT a TTY (e.g. running under the API server,
    a CI job, or any background context), ``input()`` would block
    forever waiting for input that never arrives. The pre-fix code
    silently swallowed any exception from ``input()`` inside the
    FallbackHandler and fell through to the default_action, which
    meant a worker thread under the API server would just hang.
    We now raise an explicit, actionable RuntimeError so the caller
    can fall back to ``MiniMax_fallback_default`` instead.
    """
    import sys as _sys

    if not _sys.stdin.isatty():
        raise RuntimeError(
            "MiniMax_interactive=True but stdin is not a TTY; "
            "cannot prompt for fallback action in a non-interactive "
            "context. Set MiniMax_fallback_default='rules' or 'gemma4' "
            "instead, or unset MiniMax_interactive."
        )
    _sys.stderr.write(
        "\n"
        "=" * 70 + "\n"
        "[MiniMax API ERROR]\n"
        f"  type    : {error_info.get('error_type', '?')}\n"
        f"  message : {error_info.get('error', '?')}\n"
        f"  context : {error_info.get('context', '(no context)')}\n"
        "=" * 70 + "\n"
        "Choose fallback action:\n"
        "  [1] gemma4  -> switch to local Gemma4 backend (if available)\n"
        "  [2] rules   -> skip LLM, keep rule-pipeline results\n"
        "  [3] stop    -> abort the whole pipeline\n"
        "  [4] retry   -> retry the same MiniMax call once\n"
        "Enter 1/2/3/4 (default=2): "
    )
    _sys.stderr.flush()
    try:
        choice = input().strip()
    except EOFError:
        return "rules"
    return {"1": "gemma4", "2": "rules", "3": "stop", "4": "retry", "": "rules"}.get(
        choice, "rules"
    )


@dataclass(slots=True)
class FallbackHandler:
    """Resolves what to do when MiniMax M3 API encounters an error.

    Modes
    -----
    - Interactive (CLI / Web popup): register an ``on_error`` callback that
      returns one of ``"gemma4" | "rules" | "stop" | "retry"``.
    - Headless: ``default_action`` is used automatically (default: ``"rules"``).

    Usage
    -----
    >>> handler = FallbackHandler(default_action="rules")
    >>> handler.on_error = cli_fallback_prompt   # for CLI
    >>> backend = MiniMaxM3Backend(..., on_error=handler)
    """

    default_action: FallbackAction = "rules"
    on_error: Callable[[dict[str, Any]], FallbackAction] | None = None
    error_count: int = 0
    last_action: FallbackAction | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, error_info: dict[str, Any]) -> FallbackAction:
        with self._lock:
            self.error_count += 1
        if self.on_error is not None:
            try:
                action = self.on_error(error_info)
            except Exception:
                logger.exception("FallbackHandler callback raised; using default_action.")
                action = self.default_action
        else:
            action = self.default_action
        action = action if action in {"gemma4", "rules", "stop", "retry"} else self.default_action
        with self._lock:
            self.last_action = action
        return action


def resolve_minimax_api_key(extra: dict[str, Any] | None = None) -> str | None:
    """Single source of truth for MiniMax API-key resolution.

    Mirrors the key sources the pipeline actually consumes, in priority
    order:

      1. ``extra["MiniMax_api_key"]`` (GUI Settings field / CLI config);
      2. ``MiniMax_API_KEY`` / ``MINIMAX_API_KEY`` environment;
      3. ``ANTHROPIC_API_KEY`` environment — Round 18: the project's
         documented .env key (the MiniMax API speaks the Anthropic wire
         protocol, and the ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_MODEL``
         pair keeps the endpoint consistent). ``RadiolarianPipeline``
         injects the same fallback into ``extra["MiniMax_api_key"]``
         before building the backend (pipeline.py), so callers that
         resolve the key BEFORE the pipeline exists (the GUI worker's
         outbound-policy resolution) must agree or they silently
         disable the LLM (BUG-4, audit 2026-09-04).
    """
    extra = extra or {}
    key = (
        extra.get("MiniMax_api_key")
        or os.environ.get("MiniMax_API_KEY")
        or os.environ.get("MINIMAX_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    return str(key) if key else None


def build_MiniMax_backend_from_env_or_config(extra: dict[str, Any]) -> MiniMaxM3Backend:
    """Build a MiniMaxM3Backend from ``extra`` config, falling back to env vars.

    Required keys (in priority order):
      - ``MiniMax_api_key``  / ``ANTHROPIC_API_KEY``
      - ``MiniMax_endpoint`` / ``ANTHROPIC_BASE_URL``
      - ``MiniMax_model``    / ``ANTHROPIC_MODEL`` / ``MiniMax_MODEL``
    """
    # The data_outbound_policy gates whether we need an API key at all:
    #   * api_full / api_redacted -> need a key
    #   * local_only              -> key is optional; the backend will
    #                                short-circuit every outbound call
    # Audit 2026-09-04 (BLOCKER-#2 consistency fix): the dataclass
    # field default was flipped to ``api_redacted`` when ``api_full``
    # became opt-in only, but this builder still defaulted to
    # ``api_full`` — so every caller that did not set the key
    # explicitly (tests, GUI, direct PipelineConfig users) hit the
    # opt-in ValueError at construction instead of the private
    # default. Align with the dataclass.
    policy = str(extra.get("data_outbound_policy", "api_redacted"))
    # BUG-4 (audit 2026-09-04): key resolution lives in the shared
    # ``resolve_minimax_api_key`` helper so this builder, the pipeline
    # heuristic and the GUI worker's policy resolver can never drift
    # apart again. The chain includes ``ANTHROPIC_API_KEY`` (Round 18:
    # the project's documented .env key — the base_url chain below
    # honours ``ANTHROPIC_BASE_URL``, so the key/endpoint pair stays
    # self-consistent).
    api_key = resolve_minimax_api_key(extra)
    if not api_key and policy != "local_only":
        raise ValueError(
            "MiniMax api_key not set. Provide one via:\n"
            "  - PipelineConfig.extra['MiniMax_api_key']\n"
            "  - environment variable MiniMax_API_KEY or MINIMAX_API_KEY\n"
            "  - environment variable ANTHROPIC_API_KEY (Round 18 fallback:\n"
            "    the project .env documents it; MiniMax speaks the Anthropic\n"
            "    wire protocol, ANTHROPIC_BASE_URL keeps the endpoint aligned)\n"
            "  - .env file (see .env.example)\n"
            "Or set data_outbound_policy=local_only to run without the API."
        )
    base_url = (
        extra.get("MiniMax_endpoint")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.minimaxi.com/anthropic"
    )
    model = (
        extra.get("MiniMax_model")
        or os.environ.get("MiniMax_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or "MiniMax-M3"
    )
    return MiniMaxM3Backend(
        api_key=api_key or "",
        base_url=base_url,
        model=model,
        data_outbound_policy=policy,
        max_output_tokens=_coerce_int(
            extra.get("MiniMax_max_output_tokens"), default=2048, name="MiniMax_max_output_tokens"
        ),
        thinking_budget_tokens=_coerce_int(
            extra.get("MiniMax_thinking_budget_tokens"),
            default=1024,
            name="MiniMax_thinking_budget_tokens",
        ),
        # Phase 54 audit: H4 — the dataclass field default
        # (``enable_thinking: bool = False`` at line 587) and the env
        # builder previously disagreed. The dataclass says "default
        # OFF to avoid surprise API cost"; the builder said
        # ``default=True`` for the same key, so any user who
        # instantiated the backend through the config builder (the
        # common path used by CLI and Web) was running with thinking
        # ON, paying ≥1024 thinking tokens per panel call. Align the
        # builder to the dataclass default.
        enable_thinking=_coerce_bool(extra.get("MiniMax_enable_thinking"), default=False),
        timeout_sec=_coerce_int(
            extra.get("MiniMax_timeout_sec"), default=120, name="MiniMax_timeout_sec"
        ),
        temperature=_coerce_float(
            extra.get("gemma_temperature"), default=0.1, name="gemma_temperature"
        ),
        top_p=_coerce_float(extra.get("gemma_top_p"), default=0.9, name="gemma_top_p"),
        max_retries=_coerce_int(
            extra.get("MiniMax_max_retries"), default=3, name="MiniMax_max_retries"
        ),
        max_concurrent=_coerce_int(
            extra.get("MiniMax_max_concurrent"), default=8, name="MiniMax_max_concurrent"
        ),
    )


def _coerce_int(value: Any, *, default: int, name: str) -> int:
    """Coerce ``value`` to int with a safe fallback. A non-numeric string
    or None falls back to ``default`` and logs at DEBUG (not WARNING — these
    are common config typos, not operator errors).
    """
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.debug("MiniMax config: %s=%r is not an int; using default %d", name, value, default)
        return default


def _coerce_float(value: Any, *, default: float, name: str) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug("MiniMax config: %s=%r is not a float; using default %f", name, value, default)
        return default


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        # Phase 55 audit HIGH-2 fix: the previous guard only recognised
        # ('true', '1', 'yes', 'on') as True, but user config often uses
        # 'false', '0', 'no', 'off' to explicitly disable a feature.
        # Without this fix, _coerce_bool('off') → bool('off') → True,
        # silently enabling the very feature the user tried to disable.
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off", ""):
            return False
        # Unknown string — fall through to treating it as truthy.
        # NOTE: the old behaviour for unknown strings was False (the
        # membership check returned False), so a typo like 'enaable'
        # would have silently disabled the feature. The new behaviour
        # (bool(value)=True for non-empty strings) enables it, which is
        # a safer default for a feature flag. Callers that need a
        # strict unknown-string policy should pass a default explicitly.
        return bool(value)
    return bool(value)
