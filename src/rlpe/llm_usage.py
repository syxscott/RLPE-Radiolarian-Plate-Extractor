"""Run-level LLM usage collector.

This module is intentionally free of cv2 / paddleocr / torch imports so
that test environments without those heavy backends can still exercise
the sidecar-write path.
"""

from __future__ import annotations

from typing import Any


def collect_llm_usage(runtime: Any) -> dict[str, Any] | None:
    """Collect run-level LLM usage from a Gemma runtime.

    Returns a JSON-serialisable dict suitable for the
    ``output/manifests/llm_usage.json`` sidecar. Returns None when no
    LLM backend was used (e.g. rules-only pipelines).

    Schema is purposefully loose — this is a sidecar, not the published
    RunOutput schema. The web UI / API /system/llm-status aggregator
    can build its own view from these fields.

    The function prefers ``backend.cost_summary()`` (a dict) and falls
    back to ``total_calls`` / ``total_errors`` / ``total_input_tokens``
    / ``total_output_tokens`` attributes when the summary is unavailable.
    """
    if runtime is None:
        return None
    backend = getattr(runtime, "backend", None)
    if backend is None:
        return None
    summary: dict[str, Any] = {}
    backend_name = getattr(backend, "backend_name", None)
    if backend_name:
        summary["backend"] = str(backend_name)
    model = getattr(backend, "model", None)
    if model:
        summary["model"] = str(model)
    # Preferred path: backend.cost_summary() returns dict.
    cost_fn = getattr(backend, "cost_summary", None)
    if callable(cost_fn):
        try:
            cs = cost_fn()
        except Exception:
            cs = None
        if isinstance(cs, dict):
            for k, v in cs.items():
                summary.setdefault(k, v)
    # Fallback: read accumulated counters directly. Use setdefault so
    # cost_summary keys take precedence over raw counter attributes
    # when both are present (different naming conventions).
    counter_map = {
        "total_calls": ("calls", "total_calls"),
        "total_errors": ("errors", "total_errors"),
        "total_input_tokens": ("input_tokens", "total_input_tokens"),
        "total_output_tokens": ("output_tokens", "total_output_tokens"),
    }
    for attr, (preferred, alt) in counter_map.items():
        if preferred in summary or alt in summary:
            continue
        val = getattr(backend, attr, None)
        if isinstance(val, (int, float)):
            summary[alt] = int(val)
    # Surface the sidecar only when at least one real usage signal is
    # present, so rules-only / local-only runs do not pollute the
    # output dir with an empty bundle.
    has_signal = any(
        k in summary
        for k in (
            "calls",
            "total_calls",
            "input_tokens",
            "total_input_tokens",
            "total_cost_cny",
        )
    )
    if not has_signal:
        return None
    return summary
