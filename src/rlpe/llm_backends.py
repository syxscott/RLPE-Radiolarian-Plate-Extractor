from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from typing import Any

import requests


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_from_text(text: str) -> dict[str, Any]:
    match = _JSON_RE.search((text or "").strip())
    if not match:
        raise ValueError("No JSON object found in LLM output.")
    obj = json.loads(match.group(0))
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

    def infer_panel(self, panel_image, caption_text: str, ocr_labels: list[str], system_prompt: str, user_prompt: str) -> dict[str, Any]:
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

    def infer_panel(self, panel_image, caption_text: str, ocr_labels: list[str], system_prompt: str, user_prompt: str) -> dict[str, Any]:
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

    def infer_panel(self, panel_image, caption_text: str, ocr_labels: list[str], system_prompt: str, user_prompt: str) -> dict[str, Any]:
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
            resp = requests.post(f"{self.host.rstrip('/')}/api/generate", json=payload, timeout=self.timeout_sec)
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
            resp = requests.post(f"{self.host.rstrip('/')}/api/generate", json=payload, timeout=self.timeout_sec)
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

    def infer_panel(self, panel_image, caption_text: str, ocr_labels: list[str], system_prompt: str, user_prompt: str) -> dict[str, Any]:
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
            "messages": [self._system_message(system_prompt), self._user_message(user_prompt, panel_image)],
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
            prompt = self._build_text_prompt(system_prompt, user_prompt)
            completion_payload = {
                "prompt": prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": False,
            }
            if self.model:
                completion_payload["model"] = self.model
            resp = requests.post(self.host.rstrip("/") + "/completion", json=completion_payload, timeout=self.timeout_sec)
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
