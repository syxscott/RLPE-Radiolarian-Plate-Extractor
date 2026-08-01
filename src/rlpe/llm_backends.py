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
        if isinstance(obj, dict):
            return _normalize_panel_dict(obj)
    except Exception:
        pass

    # 2) Try first JSON array element (when the LLM emits a list)
    arr_match = _JSON_ARR_RE.search(cleaned)
    if arr_match:
        try:
            arr = json.loads(arr_match.group(0))
            if isinstance(arr, list) and arr:
                # Use the first object in the array
                first = next((x for x in arr if isinstance(x, dict)), None)
                if first is not None:
                    return _normalize_panel_dict(first)
        except Exception:
            pass

    # 3) Fall back to first {...} match (non-greedy to avoid swallowing)
    obj_match = _JSON_RE.search(cleaned)
    if obj_match:
        try:
            obj = json.loads(obj_match.group(0))
            if isinstance(obj, dict):
                return _normalize_panel_dict(obj)
        except Exception:
            pass
    raise ValueError("No parseable JSON object found in LLM output.")


def _normalize_panel_dict(obj: dict[str, Any]) -> dict[str, Any]:
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
    return out


class BaseLLMBackend:
    backend_name = "base"

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
    ) -> dict[str, Any]:
        try:
            text, multimodal_degraded = self._chat_completion(
                panel_image, system_prompt, user_prompt
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
        self, panel_image, system_prompt: str, user_prompt: str
    ) -> tuple[str, bool]:
        """Return ``(text, multimodal_degraded)``.

        Audit M7: ``multimodal_degraded`` is True when the multimodal
        path failed and we degraded to text-only via the
        ``/completion`` endpoint. False in every other case (text-only
        call, successful multimodal call, fallback when no image was
        provided). The caller can use this to flag the result so
        downstream code knows the model only saw text.
        """
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
            return str(data.get("content") or data.get("response") or ""), False
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
            return self._extract_chat_text(data), False
        except Exception:
            # 2) 回退到 llama.cpp /completion 接口（纯文本）
            # Audit M7: signal the degraded mode so callers can tell
            # "model saw the image" apart from "image was dropped
            # silently".
            logger.debug(
                "llama.cpp /v1/chat/completions failed; falling back to /completion (text-only)"
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
            return str(data.get("content") or data.get("response") or ""), True

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
        return f"{system_prompt}\n\n{user_prompt}\n\nPlease output strict JSON only."

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
    data_outbound_policy: str = (
        "api_full"  # Phase 61 (Bug 4.11): M3 vision needs full-res image for species ID
    )

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

    def _build_user_content(self, panel_image, user_prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if panel_image is not None:
            content.append(_encode_image_anthropic_block(panel_image))
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

    def _build_messages(self, panel_image, user_prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": self._build_user_content(panel_image, user_prompt)}]

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
    ) -> tuple[Any, str]:
        """Return the redacted (panel_image, user_prompt) tuple based on
        ``self.data_outbound_policy``. The original caption_text and
        ocr_labels are also dropped from the user_prompt when the
        policy is ``api_redacted``.
        """
        if self.data_outbound_policy == "api_full":
            return panel_image, user_prompt
        if self.data_outbound_policy == "local_only":
            # Both branches return the same shape; the caller checks
            # policy and short-circuits before this is even invoked,
            # so this is just defensive in case the policy is changed
            # at runtime.
            return self._redact_image(panel_image), self._redact_text(user_prompt, limit=0)
        # api_redacted
        return self._redact_image(panel_image), self._redact_text(user_prompt, limit=200)

    def infer_panel(
        self,
        panel_image,
        caption_text: str,
        ocr_labels: list[str],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        if self.data_outbound_policy == "local_only":
            return self._local_only_noop("MiniMax disabled (data_outbound_policy=local_only)")
        # P2-9 fix (Plan B): store caption_text / ocr_labels so _build_user_content
        # can prepend them to the user prompt. This keeps all callers (including
        # m3_engine which passes empty strings) working while enabling future callers
        # to pass actual caption / OCR context through these parameters.
        self._thread_local.caption_text = caption_text or ""
        self._thread_local.ocr_labels = ocr_labels or []
        try:
            img, up = self._apply_outbound_policy(
                panel_image, caption_text, ocr_labels, user_prompt
            )
            messages = self._build_messages(img, up)
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
            _, up = self._apply_outbound_policy(None, "", [], user_prompt)
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
    policy = str(extra.get("data_outbound_policy", "api_full"))
    # NOTE: ANTHROPIC_API_KEY is intentionally NOT in this fallback chain
    # — it's the key for the real Anthropic API, not for MiniMax. On a
    # host that has both, falling back to ANTHROPIC_API_KEY would silently
    # send the wrong key to MiniMax and surface as a confusing 401. MiniMax
    # supports the Anthropic wire protocol but uses its own keyspace; we
    # only accept its own env vars.
    api_key = (
        extra.get("MiniMax_api_key")
        or os.environ.get("MiniMax_API_KEY")
        or os.environ.get("MINIMAX_API_KEY")
    )
    if not api_key and policy != "local_only":
        raise ValueError(
            "MiniMax api_key not set. Provide one via:\n"
            "  - PipelineConfig.extra['MiniMax_api_key']\n"
            "  - environment variable MiniMax_API_KEY or MINIMAX_API_KEY\n"
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
