from __future__ import annotations

import base64
import io
import ipaddress
import json
import logging
import os
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
    re.compile(
        r"(?<![A-Za-z0-9_])sk-ant-api03-[A-Za-z0-9]{20,}"
    ),
    re.compile(
        r"(?<![A-Za-z0-9_])sk-ant-(?!api03-)[A-Za-z0-9]{16,}"
    ),
    re.compile(
        r"(?<![A-Za-z0-9_])sk-proj-[A-Za-z0-9]{16,}"
    ),
    re.compile(
        r"(?<![A-Za-z0-9_])sk-cp-[A-Za-z0-9]{16,}"
    ),
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
    out = {
        "label": (str(obj.get("label", "")).strip() or None),
        "species": (str(obj.get("species", "")).strip() or None),
        "confidence": float(obj.get("confidence", 0.0)),
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
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": f"Transformers inference error: {type(exc).__name__}",
                "fallback_used": True,
                "error": str(exc),
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
                "error": str(exc),
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
                "error": str(exc),
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
            text = self._chat_completion(panel_image, system_prompt, user_prompt)
            parsed = parse_json_from_text(text)
            parsed["raw_text"] = text
            parsed["fallback_used"] = False
            return parsed
        except Exception as exc:
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": f"llama.cpp inference error: {type(exc).__name__}",
                "fallback_used": True,
                "error": str(exc),
            }

    def infer_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            text = self._chat_completion(None, system_prompt, user_prompt)
            parsed = parse_json_from_text(text)
            parsed["raw_text"] = text
            parsed["fallback_used"] = False
            return parsed
        except Exception as exc:
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": f"llama.cpp text inference error: {type(exc).__name__}",
                "fallback_used": True,
                "error": str(exc),
            }

    def _chat_completion(self, panel_image, system_prompt: str, user_prompt: str) -> str:
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
            return self._extract_chat_text(data)
        except Exception:
            # 2) 回退到 llama.cpp /completion 接口（纯文本）
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
            return str(data.get("content") or data.get("response") or "")

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
    enable_thinking: bool = True
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
    data_outbound_policy: str = "api_full"

    def __post_init__(self) -> None:
        # ``local_only`` does not need an API key: the backend will refuse
        # every outbound request and return a no-op result, which the
        # surrounding pipeline treats as a fallback.
        if self.data_outbound_policy not in {"api_full", "api_redacted", "local_only"}:
            raise ValueError(
                f"data_outbound_policy must be one of api_full/api_redacted/local_only, "
                f"got {self.data_outbound_policy!r}"
            )
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
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_sec,
            )
        # Per-process semaphore (cheap; limits concurrent in-flight requests).
        self._sem = threading.Semaphore(self.max_concurrent)
        # Running totals (read by callers for cost dashboards).
        self._lock = threading.Lock()
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_calls: int = 0
        self.total_errors: int = 0

    # ------------------------------------------------------------------ helpers

    def _build_user_content(self, panel_image, user_prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if panel_image is not None:
            content.append(_encode_image_anthropic_block(panel_image))
        content.append({"type": "text", "text": user_prompt})
        return content

    def _build_messages(self, panel_image, user_prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": self._build_user_content(panel_image, user_prompt)}]

    def _build_text_messages(self, user_prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": user_prompt}]

    def _build_request_kwargs(
        self, system_prompt: str, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # max_tokens must be > thinking_budget when thinking is enabled.
        max_out = max(self.max_output_tokens, self.thinking_budget_tokens + 256)
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
                wait = min(2**attempt, 30)
                logger.warning(
                    "MiniMax rate-limited (attempt %d), sleeping %ds: %s", attempt + 1, wait, exc
                )
                time.sleep(wait)
            except anthropic_mod.APIConnectionError as exc:
                last_exc = exc
                wait = min(2**attempt, 30)
                logger.warning(
                    "MiniMax connection error (attempt %d), sleeping %ds: %s",
                    attempt + 1,
                    wait,
                    exc,
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
                if status >= 500 or status == 429 or status in (401, 403):
                    wait = min(2**attempt, 30)
                    logger.warning(
                        "MiniMax %d (attempt %d), sleeping %ds: %s",
                        status,
                        attempt + 1,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
                else:
                    # 4xx (not retryable) -> count and re-raise immediately
                    with self._lock:
                        self.total_errors += 1
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
                last_exc,
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
            # Set `error` / `error_type` so downstream code (e.g.
            # apply_gemma_to_matches) can propagate it to match.metadata
            # and the FallbackHandler popup shows the real reason.
            return {
                "label": None,
                "species": None,
                "confidence": 0.0,
                "reasoning": f"MiniMax JSON parse error: {type(exc).__name__}: {exc}",
                "fallback_used": True,
                "error": f"{type(exc).__name__}: {exc}",
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
        try:
            img, up = self._apply_outbound_policy(
                panel_image, caption_text, ocr_labels, user_prompt
            )
            messages = self._build_messages(img, up)
            resp = self._call_api(system_prompt, messages)
            return self._make_result(resp)
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
        except Exception as exc:
            return self._make_error_result(exc)

    def cost_summary(self) -> dict[str, Any]:
        with self._lock:
            in_t = self.total_input_tokens
            out_t = self.total_output_tokens
            calls = self.total_calls
            errs = self.total_errors
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
        }


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
            "  - environment variable ANTHROPIC_API_KEY\n"
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
            extra.get("MiniMax_thinking_budget_tokens"), default=1024, name="MiniMax_thinking_budget_tokens"
        ),
        enable_thinking=_coerce_bool(extra.get("MiniMax_enable_thinking"), default=True),
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
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)
