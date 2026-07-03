"""Fake MiniMax-M3 backend for tests + smoke runs.

Replaces ``MiniMaxM3Backend`` (src/rlpe/llm_backends.py) in tests so we
can exercise the entire M3-engine pipeline (vision prompts, retry
plumbing, cost aggregation) without any outbound HTTP traffic.

Public API mirrors the methods the engine actually calls:

* ``infer_text(system_prompt, user_prompt)`` -> dict with
  ``label/species/confidence/reasoning`` keys
* ``infer_panel(panel_image, caption_text, ocr_labels, system_prompt,
  user_prompt)`` -> same shape as ``infer_text`` plus ``request_id``
  / ``model_version`` / ``usage`` / ``cost_cny`` so the M3 telemetry
  tests can assert propagation.

The fake captures every call in ``self.calls`` (list of dicts) and
exposes ``self.cost_summary()`` to keep the production cost-aggregation
code path exercised end-to-end.

This module deliberately imports zero heavy deps (no torch, no cv2, no
anthropic SDK) so it can be used from minimal-env CI runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass(slots=True)
class _CallRecord:
    """One invocation captured for test introspection."""

    method: str
    system_prompt: str
    user_prompt: str
    image: Any | None
    response: dict[str, Any]


@dataclass(slots=True)
class FakeM3Backend:
    """Deterministic MiniMax-M3 stand-in for tests + smoke runs."""

    api_key: str = "fake"
    base_url: str = "http://fake/"
    model: str = "MiniMax-M3-fake"
    enable_thinking: bool = False
    max_concurrent: int = 1
    timeout_sec: int = 30
    max_retries: int = 0
    # Canned responses. The fake picks the first entry whose
    # ``match`` callable (if given) returns True for the call's
    # system_prompt, else the last entry is returned as a default.
    canned_responses: list[dict[str, Any]] = field(default_factory=list)
    # Per-call records for test assertions.
    calls: list[_CallRecord] = field(default_factory=list)
    # Cost / token counters, mirroring MiniMaxM3Backend.
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    total_errors: int = 0
    _lock: Lock = field(default_factory=Lock)

    def _pick(self, system_prompt: str) -> dict[str, Any]:
        """Pick the canned response for this call."""
        if not self.canned_responses:
            return self._default()
        # If entries are dicts with a ``match`` key, use it; else last wins.
        for entry in self.canned_responses:
            if "match" in entry and entry["match"](system_prompt):
                return {k: v for k, v in entry.items() if k != "match"}
        return {k: v for k, v in self.canned_responses[-1].items() if k != "match"}

    def _default(self) -> dict[str, Any]:
        return {
            "label": "1",
            "species": "Genus species",
            "confidence": 0.5,
            "reasoning": "fake-default",
            "fallback_used": False,
            "request_id": "fake-req-0",
            "model_version": self.model,
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "cost_cny": 0.0007,
        }

    def _bump(self, in_t: int = 100, out_t: int = 50, *, cost: float | None = None) -> None:
        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += in_t
            self.total_output_tokens += out_t
            if cost is not None:
                pass  # cost_summary computes from token counts

    def infer_text(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        resp = self._pick(system_prompt)
        self._bump(
            in_t=resp.get("usage", {}).get("input_tokens", 100),
            out_t=resp.get("usage", {}).get("output_tokens", 50),
        )
        self.calls.append(_CallRecord("infer_text", system_prompt, user_prompt, None, resp))
        return resp

    def infer_panel(
        self,
        panel_image: Any = None,
        caption_text: str = "",
        ocr_labels: list[str] | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        resp = self._pick(system_prompt)
        self._bump(
            in_t=resp.get("usage", {}).get("input_tokens", 100),
            out_t=resp.get("usage", {}).get("output_tokens", 50),
        )
        self.calls.append(
            _CallRecord(
                "infer_panel",
                system_prompt,
                user_prompt,
                panel_image,
                resp,
            )
        )
        return resp

    def cost_summary(self) -> dict[str, Any]:
        with self._lock:
            in_t = self.total_input_tokens
            out_t = self.total_output_tokens
            calls = self.total_calls
            errs = self.total_errors
        cost = round(in_t / 1_000_000 * 2.1 + out_t / 1_000_000 * 8.4, 4)
        return {
            "calls": calls,
            "errors": errs,
            "input_tokens": in_t,
            "output_tokens": out_t,
            "total_cost_cny": cost,
        }
