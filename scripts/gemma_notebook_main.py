from __future__ import annotations

"""
Notebook-style main script for quick Gemma4 integration test.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image

from rlpe.gemma_postprocess import (
    GEMMA_SYSTEM_PROMPT_EN,
    GEMMA_SYSTEM_PROMPT_ZH,
    gemma_match_panel,
    load_gemma4_model,
)


CONFIG = {
    "MODEL_PATH": "/home/user/models/gemma-4-E4B",  # 用户运行时替换
    "use_gemma4": True,
    "gemma_conf_threshold": 0.70,
    "gemma_system_prompt_lang": "zh",
    "single_test_only": True,
    "single_panel_path": "/data/rlpe/panels/panel_01.png",
    "single_caption_text": "Fig. 3. (A) Actinomma leptodermum ... (B) Spongodiscus sp. ...",
    "single_ocr_labels": ["A", "B", "3"],
}


def build_runtime_from_config(cfg: dict):
    return load_gemma4_model(
        model_path=cfg["MODEL_PATH"],
        use_4bit=True,
        bfloat16=True,
        device_map="auto",
    )


def run_single_test(cfg: dict, runtime):
    prompt = GEMMA_SYSTEM_PROMPT_ZH if cfg.get("gemma_system_prompt_lang", "zh") == "zh" else GEMMA_SYSTEM_PROMPT_EN
    with Image.open(cfg["single_panel_path"]) as im:
        result = gemma_match_panel(
            runtime=runtime,
            panel_image=im.convert("RGB"),
            caption_text=cfg["single_caption_text"],
            ocr_labels=cfg["single_ocr_labels"],
            system_prompt=prompt,
        )
    print("single result:", result)
    return result


def main() -> int:
    if not CONFIG.get("use_gemma4", False):
        print("Gemma disabled in CONFIG.")
        return 0
    runtime = build_runtime_from_config(CONFIG)
    if CONFIG.get("single_test_only", True):
        run_single_test(CONFIG, runtime)
    else:
        print("Set single_test_only=True or integrate this runtime into rlpe.pipeline.RadiolarianPipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
