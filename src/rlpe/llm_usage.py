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
        # Require strictly positive counters: a fresh backend that has
        # not been called yet still has ``total_calls=0`` etc. (init
        # values), and writing those into the sidecar pollutes
        # ``output/manifests/llm_usage.json`` with a misleading zero
        # record. Bug fix: only surface counters when at least one call
        # actually ran.
        if isinstance(val, (int, float)) and val > 0:
            summary[alt] = int(val)
    # Surface the sidecar only when at least one real usage signal is
    # present, so rules-only / local-only runs do not pollute the
    # output dir with an empty bundle. A key whose value is 0 does NOT
    # count as signal: ``cost_summary()`` always emits zero-valued
    # ``calls``/``errors`` keys for a fresh backend, and the previous
    # key-presence check turned that into a misleading all-zeros
    # ``llm_usage.json`` for runs whose LLM calls happened in worker
    # subprocesses (F19 batch isolation).
    has_signal = any(
        isinstance(summary.get(k), (int, float)) and summary[k] > 0
        for k in (
            "calls",
            "total_calls",
            "input_tokens",
            "total_input_tokens",
            "output_tokens",
            "total_output_tokens",
        )
    )
    if not has_signal:
        return None
    return summary


_COUNTER_KEYS = ("calls", "errors", "input_tokens", "output_tokens")


def merge_llm_usage(summaries: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Merge several per-worker usage summaries into one run summary.

    With ``batch_isolation="subprocess"`` the LLM calls happen inside
    ``python -m rlpe.worker`` children whose counters die with the
    process; the parent aggregates the per-paper sidecars through this
    helper so ``llm_usage.json`` reflects the true run totals.

    Numeric counter keys are summed; ``backend`` / ``model`` strings are
    taken from the first summary that carries them (workers share the
    same preset, so they agree). Returns None when every input is None
    or carries no positive counter.
    """
    merged: dict[str, Any] = {}
    for s in summaries:
        if not s:
            continue
        for key in _COUNTER_KEYS:
            val = s.get(key)
            if isinstance(val, (int, float)):
                merged[key] = merged.get(key, 0) + val
        for key in ("backend", "model"):
            if key not in merged and s.get(key):
                merged[key] = s[key]
    if not any(isinstance(merged.get(k), (int, float)) and merged[k] > 0 for k in _COUNTER_KEYS):
        return None
    return merged
