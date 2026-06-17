from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .llm_backends import (
    BaseLLMBackend,
    LlamaCppGemmaBackend,
    OllamaGemmaBackend,
    TransformersGemmaBackend,
)
from .types import MatchResult

GEMMA_SYSTEM_PROMPT_ZH = """
你是古生物分类学与放射虫图版解析专家，服务于RLPE项目。
任务：给定单个panel图像、图版说明（caption）、OCR标签候选，判断该panel对应的标签与拉丁学名（属/种）。
场景特点：老文献、扫描噪声、跨页说明、多个 specimen、箭头指向、视角混合（apical/lateral等）。
请遵循：
1) 先内部分析证据（不要输出冗长思维过程），只输出结构化JSON。
2) 优先依据：panel可见标签 > caption中label-物种对子句 > 形态/语义一致性。
3) 若信息不足，给出最可能候选并降低confidence。
4) 必须输出字段：label, species, confidence, reasoning。
5) confidence范围[0,1]，保留两位小数。
输出格式（严格JSON）：
{"label":"A","species":"Actinomma leptodermum","confidence":0.87,"reasoning":"依据caption中(A)...与图中标签A一致"}
""".strip()


GEMMA_SYSTEM_PROMPT_EN = """
You are an expert in radiolarian paleontology and taxonomic plate interpretation for RLPE.
Task: Given one panel image, caption context, and OCR label candidates, infer the best label-to-Latin-taxon match.
Challenges: noisy scans, cross-page captions, multi-specimen panels, arrow annotations, mixed views.
Rules:
1) Think internally but DO NOT reveal long chain-of-thought; return concise evidence in JSON only.
2) Prioritize: visible panel label > caption label-taxon clause > morphology/semantic consistency.
3) If uncertain, provide best candidate with lower confidence.
4) Required keys: label, species, confidence, reasoning.
5) confidence in [0,1], rounded to 2 decimals.
Strict output JSON:
{"label":"A","species":"Actinomma leptodermum","confidence":0.87,"reasoning":"caption clause (A) agrees with visible label A"}
""".strip()


@dataclass(slots=True)
class GemmaRuntime:
    backend: BaseLLMBackend
    backend_name: str


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
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
    prompt = system_prompt or GEMMA_SYSTEM_PROMPT_ZH
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
    prompt = (
        GEMMA_SYSTEM_PROMPT_ZH if prompt_lang.lower().startswith("zh") else GEMMA_SYSTEM_PROMPT_EN
    )
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

        gemma_conf = float(out.get("confidence", 0.0))
        match.metadata["gemma_confidence"] = gemma_conf
        # Always provide a reasoning string. Empty reasoning leaves the
        # frontend's "why was this overridden?" tooltip blank, which
        # operators have reported as confusing — they couldn't tell
        # whether the LLM was silent or the metadata field was lost.
        match.metadata["gemma_reasoning"] = (
            out.get("reasoning") or "No reasoning provided by LLM backend"
        )
        # Propagate MiniMax / Ollama / Transformers error info so the
        # FallbackHandler popup can show the real reason (not "no detailed error").
        if out.get("error"):
            match.metadata["gemma_error"] = str(out.get("error"))
        if out.get("error_type"):
            match.metadata["gemma_error_type"] = str(out.get("error_type"))
        if out.get("request_id"):
            match.metadata["MiniMax_request_id"] = str(out.get("request_id"))
        if out.get("cost_cny") is not None:
            match.metadata["MiniMax_cost_cny"] = float(out.get("cost_cny"))
        if out.get("model_version"):
            match.metadata["MiniMax_model_version"] = str(out.get("model_version"))

        if gemma_conf >= conf_threshold:
            match.panel_id = out.get("label") or match.panel_id
            match.species = out.get("species") or match.species
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
    prompt = (
        GEMMA_SYSTEM_PROMPT_ZH if prompt_lang.lower().startswith("zh") else GEMMA_SYSTEM_PROMPT_EN
    )
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

        new_row["gemma_confidence"] = float(result.get("confidence", 0.0))
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
        if new_row["gemma_confidence"] >= conf_threshold:
            new_row["panel_id"] = result.get("label") or new_row.get("panel_id")
            new_row["species"] = result.get("species") or new_row.get("species")
            new_row["label_text"] = result.get("label") or new_row.get("label_text")
            new_row["gemma_used"] = True
        else:
            new_row["gemma_used"] = False
            new_row["gemma_fallback"] = True
        out_rows.append(new_row)
    return out_rows
