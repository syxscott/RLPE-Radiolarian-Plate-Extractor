from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional local Gemma dependency
    torch = None  # type: ignore[assignment]
from PIL import Image
from tqdm import tqdm

from .llm_backends import (
    BaseLLMBackend,
    LlamaCppGemmaBackend,
    OllamaGemmaBackend,
    TransformersGemmaBackend,
)
from .types import MatchResult

# Audit 2026-08-19 Phase 4C (Bug M-10): historically Gemma hard-coded its
# own copy of the per-panel system prompt here. When M3's prompt was
# updated upstream, this copy drifted — silently invalidating the Gemma
# fallback path. The canonical prompts now live in ``m3_engine`` and
# Gemma pulls them via ``get_prompt_registry()``.
try:
    from .m3_engine import get_prompt_registry
except Exception:  # pragma: no cover - tolerate missing m3_engine in envs
    def get_prompt_registry() -> tuple[dict[str, str], str]:  # type: ignore[no-redef]
        """Stub fallback so this module imports without m3_engine.

        In normal install paths, ``rlpe.m3_engine`` is always available;
        the stub exists only for sparse test/dev environments that
        exclude the M3 module. Phase 4E: mirrors the real signature
        ``(dict, version_str)`` so callers that unpack the tuple
        don't break in offline test environments.
        """
        return {}, "v0.0.0-stub"


# Cache the M3 prompts at import time. Stale once per process is fine
# because the prompts are constants — re-reading on every call would
# just re-import the dict.
_PROMPTS_CACHE: dict[str, str] | None = None
_PROMPTS_VERSION: str | None = None


def _get_m3_prompts() -> dict[str, str]:
    """Lazily load (and cache) the M3 prompt registry.

    Returns the dict from ``m3_engine.get_prompt_registry()``. If M3
    is unavailable (stub used above), returns an empty dict; the
    helpers below detect that and fall back to legacy inline prompts.

    Phase 4E (audit 2026-08-19): ``get_prompt_registry()`` now returns
    a 2-tuple ``(dict, version)``; this helper unpacks and caches only
    the dict side. ``_get_m3_prompt_version()`` exposes the version.
    """
    global _PROMPTS_CACHE, _PROMPTS_VERSION
    if _PROMPTS_CACHE is None:
        try:
            result = get_prompt_registry()
        except Exception:
            _PROMPTS_CACHE = {}
            _PROMPTS_VERSION = "v0.0.0-stub"
        else:
            # Phase 4E: handle both old (dict) and new (tuple) shapes
            # for forward / backward compatibility during the migration.
            if isinstance(result, tuple) and len(result) == 2:
                _PROMPTS_CACHE, _PROMPTS_VERSION = result
            elif isinstance(result, dict):
                _PROMPTS_CACHE = result
                _PROMPTS_VERSION = "v0.0.0-unknown"
            else:
                _PROMPTS_CACHE = {}
                _PROMPTS_VERSION = "v0.0.0-unknown"
    return _PROMPTS_CACHE


def _get_m3_prompt_version() -> str:
    """Return the cached prompt-registry version string.

    Audit 2026-08-19 Phase 4E: complements ``_get_m3_prompts()`` so
    callers can stamp a result with the prompt revision used to
    produce it. Returns ``"v0.0.0-unknown"`` if the registry was
    never loaded (e.g. m3_engine unavailable).
    """
    if _PROMPTS_VERSION is None:
        # Trigger the lazy load (may set _PROMPTS_VERSION or leave it
        # as a stub string).
        _get_m3_prompts()
    return _PROMPTS_VERSION or "v0.0.0-unknown"


# Convenience accessors. Each ``_get_system_prompt(stage)`` returns the
# canonical M3 prompt for ``stage`` so the API surface matches what
# the tests expect (``gemma._get_system_prompt(stage)``).
_STAGE_ALIASES = {
    "match_panel": "match_panel",
    "match_panel_visual_only": "match_panel_visual_only",
    "match": "match_panel",
    "zh": "match_panel",
    "en": "match_panel_visual_only",
}


def _get_system_prompt(stage: str = "match_panel") -> str | None:
    """Return the M3 system prompt for ``stage``, or None if missing.

    ``stage`` is one of the keys in ``m3_engine.get_prompt_registry()``
    (``match_panel``, ``match_panel_visual_only``,
    ``classify_plate``, etc.). Aliases ``zh`` / ``en`` / ``match``
    resolve to ``match_panel`` / ``match_panel_visual_only`` /
    ``match_panel`` respectively for legacy call sites.
    """
    prompts = _get_m3_prompts()
    if not prompts:
        return None
    resolved = _STAGE_ALIASES.get(stage, stage)
    return prompts.get(resolved)


# Audit 2026-08-19 Phase 4C (Bug M-12): M3 emits different field-name
# variants across stages and migration cycles. ``confidence`` was
# renamed to ``conf_score`` in some prompts and to ``c_score`` in
# downstream layout/YOLO paths; ``verbatim_name`` is a recent
# schema (2026-08-19) that replaced ``raw_name`` / ``name`` / ``taxon``
# in earlier prompts. Gemma post-processing used to assume the M3
# names directly, leading to silent zero-confidence fallbacks when
# the names drifted.
_CONFIDENCE_FIELD_FALLBACK = (
    "confidence",
    "conf_score",
    "c_score",
    "score",
)
_NAME_FIELD_FALLBACK = (
    "verbatim_name",
    "raw_name",
    "name",
    "taxon",
)


def _pick_field(payload: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    """Return the first present key from ``candidates`` in ``payload``.

    Used for forward-compatibility when M3 renames a field across
    prompt updates: ``_pick_field(out, _CONFIDENCE_FIELD_FALLBACK)``
    returns ``payload.get("conf_score")`` if the M3 prompt emits
    ``conf_score`` and only that name.
    """
    if not isinstance(payload, dict):
        return None
    for key in candidates:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


@dataclass(slots=True)
class GemmaRuntime:
    backend: BaseLLMBackend
    backend_name: str


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_gemma4_model(
    model_path: str,
    use_4bit: bool = True,
    bfloat16: bool = True,
    device_map: str = "auto",
) -> GemmaRuntime:
    """Load local Gemma4 model, preferring multimodal variant."""
    set_global_seed(42)
    try:
        import transformers
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(f"Transformers import failed: {exc}")

    if torch is None:
        raise RuntimeError("PyTorch is required for the transformers Gemma backend")
    dtype = torch.bfloat16 if bfloat16 else torch.float16
    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    # Try multimodal path first.
    try:
        processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        mm_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if mm_cls is None:
            raise RuntimeError("AutoModelForImageTextToText not available in current Transformers.")
        model = mm_cls.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )
        model.eval()
        backend = TransformersGemmaBackend(
            model=model, processor=processor, tokenizer=None, is_multimodal=True
        )
        return GemmaRuntime(backend=backend, backend_name="transformers")
    except Exception as exc:
        mm_error = exc

    # Fallback: text-only model
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )
        model.eval()
        backend = TransformersGemmaBackend(
            model=model, processor=None, tokenizer=tokenizer, is_multimodal=False
        )
        return GemmaRuntime(backend=backend, backend_name="transformers")
    except Exception as lm_exc:
        raise RuntimeError(f"Gemma load failed. multimodal_error={mm_error}; text_error={lm_exc}")


def load_gemma4_ollama(
    model_name: str,
    host: str = "http://127.0.0.1:11434",
    timeout_sec: int = 120,
    temperature: float = 0.10,
    top_p: float = 0.90,
) -> GemmaRuntime:
    backend = OllamaGemmaBackend(
        model=model_name,
        host=host,
        timeout_sec=timeout_sec,
        temperature=temperature,
        top_p=top_p,
    )
    return GemmaRuntime(backend=backend, backend_name="ollama")


def load_gemma4_llamacpp(
    host: str = "http://127.0.0.1:8080",
    model_name: str | None = None,
    timeout_sec: int = 120,
    temperature: float = 0.10,
    top_p: float = 0.90,
) -> GemmaRuntime:
    backend = LlamaCppGemmaBackend(
        host=host,
        model=model_name,
        timeout_sec=timeout_sec,
        temperature=temperature,
        top_p=top_p,
    )
    return GemmaRuntime(backend=backend, backend_name="llamacpp")


def build_gemma_backend_from_config(extra: dict[str, Any]) -> GemmaRuntime:
    backend = str(extra.get("llm_backend", "transformers")).lower()
    if backend in {"minimax", "minimax-m3", "minimax_api"}:
        from .llm_backends import build_MiniMax_backend_from_env_or_config

        runtime_backend = build_MiniMax_backend_from_env_or_config(extra)
        return GemmaRuntime(backend=runtime_backend, backend_name="MiniMax")
    if backend in {"llama.cpp", "llamacpp", "llama_cpp"}:
        host = str(extra.get("llama_host", "http://127.0.0.1:8080"))
        model_name = (
            extra.get("llama_model") or extra.get("ollama_model") or extra.get("gemma_model_path")
        )
        timeout_sec = int(extra.get("llama_timeout_sec", 120))
        return load_gemma4_llamacpp(
            host=host,
            model_name=str(model_name) if model_name else None,
            timeout_sec=timeout_sec,
        )
    if backend == "ollama":
        model_name = str(extra.get("ollama_model") or extra.get("gemma_model_path") or "gemma4")
        host = str(extra.get("ollama_host", "http://127.0.0.1:11434"))
        timeout_sec = int(extra.get("gemma_timeout_sec", 120))
        return load_gemma4_ollama(model_name=model_name, host=host, timeout_sec=timeout_sec)
    return load_gemma4_model(
        model_path=str(extra.get("gemma_model_path")),
        use_4bit=bool(extra.get("gemma_use_4bit", True)),
        bfloat16=bool(extra.get("gemma_bfloat16", True)),
        device_map=str(extra.get("gemma_device_map", "auto")),
    )


def gemma_match_panel(
    runtime: GemmaRuntime,
    panel_image: Image.Image,
    caption_text: str,
    ocr_labels: list[str],
    system_prompt: str | None = None,
    max_new_tokens: int = 220,
    temperature: float = 0.10,
    top_p: float = 0.90,
) -> dict[str, Any]:
    # Audit 2026-08-19 Phase 4C (Bug M-10): prefer the M3 ``match_panel``
    # prompt over the legacy ``GEMMA_SYSTEM_PROMPT_ZH`` so the Gemma
    # fallback after an M3 failure uses the SAME JSON contract M3 was
    # emitting. We still honour an explicit ``system_prompt`` arg so
    # callers that need a custom instruction (e.g. tests, ablations)
    # retain the override. Audit guard: when M3 is unavailable we fall
    # back to the legacy hard-coded ZH prompt so existing single-env
    # installs without ``m3_engine`` keep working.
    if system_prompt is None:
        m3_prompt = _get_system_prompt("match_panel")
        prompt = m3_prompt if m3_prompt else GEMMA_SYSTEM_PROMPT_ZH
    else:
        prompt = system_prompt
    user_prompt = (
        "[Caption]\n"
        f"{caption_text}\n\n"
        "[OCR labels]\n"
        f"{ocr_labels}\n\n"
        "请判断该panel最可能对应的label与拉丁学名。严格输出JSON，不要输出其他文本。"
    )

    return runtime.backend.infer_panel(
        panel_image=panel_image,
        caption_text=caption_text,
        ocr_labels=ocr_labels,
        system_prompt=prompt,
        user_prompt=user_prompt,
    )


def gemma_extract_text_json(
    runtime: GemmaRuntime, system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    return runtime.backend.infer_text(system_prompt=system_prompt, user_prompt=user_prompt)


def apply_gemma_to_matches(
    runtime: GemmaRuntime,
    matches: list[MatchResult],
    caption_text: str,
    ocr_labels: list[str],
    conf_threshold: float = 0.70,
    prompt_lang: str = "zh",
) -> list[MatchResult]:
    # Audit 2026-08-19 Phase 4C (Bug M-10): prefer M3's ``match_panel``
    # prompt so the fallback path uses the SAME JSON contract M3 was
    # emitting (single source of truth for prompts). The two legacy
    # ZH / EN inline prompts remain as a last-resort fallback only
    # when M3 is unavailable.
    is_zh = prompt_lang.lower().startswith("zh")
    if is_zh:
        m3_prompt = _get_system_prompt("match_panel")
        prompt: str = m3_prompt if m3_prompt else GEMMA_SYSTEM_PROMPT_ZH
    else:
        # English pipeline: M3 doesn't ship a dedicated EN match_panel
        # prompt, so fall back to the legacy inline English prompt.
        m3_prompt = _get_system_prompt("match_panel_visual_only")
        prompt = m3_prompt if m3_prompt else GEMMA_SYSTEM_PROMPT_EN
    for match in matches:
        if not match.panel_path:
            continue
        panel_path = Path(match.panel_path)
        if not panel_path.exists():
            match.metadata["gemma_used"] = False
            match.metadata["gemma_error"] = "panel_not_found"
            continue

        try:
            with Image.open(panel_path) as im:
                panel_image = im.convert("RGB")
                out = gemma_match_panel(
                    runtime=runtime,
                    panel_image=panel_image,
                    caption_text=caption_text,
                    ocr_labels=ocr_labels,
                    system_prompt=prompt,
                )
        except Exception as exc:
            match.metadata["gemma_used"] = False
            match.metadata["gemma_error"] = str(exc)
            continue

        # Audit 2026-08-19 Phase 4C (Bug M-11): field-name fallback for
        # confidence / species. M3 emits ``confidence`` today but
        # earlier prompts shipped ``conf_score`` / ``c_score``; the
        # verbatim/raw-name field was renamed ``verbatim_name`` (2026-
        # 08-19 schema) but older payloads carried ``raw_name`` /
        # ``name`` / ``taxon``. Without the fallback a successful M3
        # call that emitted ``conf_score`` would have been silently
        # mapped to ``gemma_conf = 0.0`` and the row marked fallback.
        conf_raw = _pick_field(out, _CONFIDENCE_FIELD_FALLBACK)
        try:
            gemma_conf = float(conf_raw) if conf_raw is not None else 0.0
        except (TypeError, ValueError):
            gemma_conf = 0.0
        species_raw = _pick_field(out, _NAME_FIELD_FALLBACK)
        match.metadata["gemma_confidence"] = gemma_conf
        # Always provide a reasoning string. Empty reasoning leaves the
        # frontend's "why was this overridden?" tooltip blank, which
        # operators have reported as confusing — they couldn't tell
        # whether the LLM was silent or the metadata field was lost.
        match.metadata["gemma_reasoning"] = (
            out.get("reasoning") or "No reasoning provided by LLM backend"
        )
        # Bug #3 fix: only propagate error info when the call genuinely
        # failed. A successful call may still carry an "error" key from a
        # previous raw_text echo, and stamping it would cause
        # ``RadiolarianPipeline._matches_have_fallback_error`` to misclassify
        # a successful call as failed, triggering unnecessary fallback.
        actually_failed = bool(out.get("fallback_used")) or gemma_conf < conf_threshold
        # Telemetry (cost / request id / model version / token usage) MUST be
        # propagated on every call that returns them, success or failure.
        # The previous version only stamped them inside the failure branch,
        # which silently hid MiniMax usage in successful runs and made
        # /system/llm-status report zero cost on the default MiniMax path.
        # error / error_type remain gated by ``actually_failed`` (Bug #3
        # regression guard).
        if out.get("request_id"):
            match.metadata["MiniMax_request_id"] = str(out.get("request_id"))
        if out.get("cost_cny") is not None:
            match.metadata["MiniMax_cost_cny"] = float(out.get("cost_cny"))
        if out.get("model_version"):
            match.metadata["MiniMax_model_version"] = str(out.get("model_version"))
        if isinstance(out.get("usage"), dict):
            match.metadata["MiniMax_usage"] = dict(out.get("usage"))
        if actually_failed:
            if out.get("error"):
                match.metadata["gemma_error"] = str(out.get("error"))
            if out.get("error_type"):
                match.metadata["gemma_error_type"] = str(out.get("error_type"))

        if gemma_conf >= conf_threshold and not out.get("fallback_used"):
            match.panel_id = out.get("label") or match.panel_id
            # Prefer the resolved species (confidence-aware fallback);
            # only fall back to ``label`` if the new code path somehow
            # produced nothing.
            match.species = species_raw or out.get("species") or match.species
            match.label_text = out.get("label") or match.label_text
            match.confidence = max(match.confidence, gemma_conf)
            match.metadata["gemma_used"] = True
        else:
            match.metadata["gemma_used"] = False
            # Distinguish "M3 said this is not a radiolarian specimen" from a
            # real low-confidence verdict.  A "not a specimen" answer is a
            # normal pipeline outcome (the panel was just a page header /
            # placeholder), not a fallback error to surface to the user.
            if out.get("is_radiolarian") is False:
                match.metadata["m3_rejected_non_radiolarian"] = True
                match.metadata["gemma_reasoning"] = (
                    out.get("reasoning") or "M3: not a radiolarian specimen"
                )
            else:
                match.metadata["gemma_fallback"] = True
    return matches


def batch_gemma_postprocess_rows(
    runtime: GemmaRuntime,
    rows: list[dict[str, Any]],
    conf_threshold: float = 0.70,
    prompt_lang: str = "zh",
) -> list[dict[str, Any]]:
    """Apply Gemma to exported rows (dict mode), with progress bar.

    Returns a new list of dicts; the input *rows* are not modified.
    """
    # Audit 2026-08-19 Phase 4C (Bug M-10): prefer M3 ``match_panel``
    # prompt so the batch fallback uses the same JSON contract M3 was
    # emitting.
    is_zh = prompt_lang.lower().startswith("zh")
    if is_zh:
        m3_prompt = _get_system_prompt("match_panel")
        prompt: str = m3_prompt if m3_prompt else GEMMA_SYSTEM_PROMPT_ZH
    else:
        m3_prompt = _get_system_prompt("match_panel_visual_only")
        prompt = m3_prompt if m3_prompt else GEMMA_SYSTEM_PROMPT_EN
    out_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="Gemma postprocess"):
        new_row = dict(row)
        panel_path = new_row.get("panel_path")
        if not panel_path or not Path(panel_path).exists():
            new_row["gemma_used"] = False
            new_row["gemma_confidence"] = 0.0
            out_rows.append(new_row)
            continue

        try:
            with Image.open(panel_path) as im:
                result = gemma_match_panel(
                    runtime=runtime,
                    panel_image=im.convert("RGB"),
                    caption_text=(
                        new_row.get("caption_text") or new_row.get("caption_snippet") or ""
                    ),
                    ocr_labels=new_row.get("ocr_labels", []),
                    system_prompt=prompt,
                )
        except Exception as exc:
            new_row["gemma_used"] = False
            new_row["gemma_error"] = str(exc)
            out_rows.append(new_row)
            continue

        # Audit 2026-08-19 Phase 4C (Bug M-11): field-name fallback for
        # confidence (same rationale as ``apply_gemma_to_matches``).
        conf_raw = _pick_field(result, _CONFIDENCE_FIELD_FALLBACK)
        try:
            new_row["gemma_confidence"] = float(conf_raw) if conf_raw is not None else 0.0
        except (TypeError, ValueError):
            new_row["gemma_confidence"] = 0.0
        species_raw = _pick_field(result, _NAME_FIELD_FALLBACK)
        new_row["gemma_reasoning"] = result.get("reasoning", "")
        # Propagate error info from MiniMax / Ollama / Transformers backends
        # so downstream tools (e.g. FallbackHandler) can see the real reason.
        if result.get("error"):
            new_row["gemma_error"] = str(result.get("error"))
        if result.get("error_type"):
            new_row["gemma_error_type"] = str(result.get("error_type"))
        if result.get("request_id"):
            new_row["MiniMax_request_id"] = str(result.get("request_id"))
        if result.get("cost_cny") is not None:
            new_row["MiniMax_cost_cny"] = float(result.get("cost_cny"))
        if result.get("model_version"):
            new_row["MiniMax_model_version"] = str(result.get("model_version"))
        # M22: propagate per-call ``usage`` token accounting into the
        # row. The non-batch path already pipes it through via
        # ``_telemetry_subset``; the batch path below was silently
        # dropping it, so /system/llm-status under-counted tokens
        # for any row processed by ``gemma_batch_enrich``.
        if isinstance(result.get("usage"), dict):
            new_row["MiniMax_usage"] = dict(result["usage"])
        if new_row["gemma_confidence"] >= conf_threshold:
            new_row["panel_id"] = result.get("label") or new_row.get("panel_id")
            # Audit 2026-08-19 Phase 4C (Bug M-11): prefer the
            # fallback-aware species resolution (verbatim_name /
            # raw_name / name / taxon) before the bare ``species`` key.
            new_row["species"] = species_raw or result.get("species") or new_row.get("species")
            new_row["label_text"] = result.get("label") or new_row.get("label_text")
            new_row["gemma_used"] = True
        else:
            new_row["gemma_used"] = False
            new_row["gemma_fallback"] = True
        out_rows.append(new_row)
    return out_rows
