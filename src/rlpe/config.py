from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Recognised extra-config keys; any key outside this set triggers a warning.
_KNOWN_EXTRA_KEYS = {
    "sam2_checkpoint", "sam2_model_cfg", "sam2_grid_size",
    "sam2_max_point_prompts", "sam2_max_box_prompts",
    "use_neural_matcher", "matcher_checkpoint_path",
    "taxon_hf_model_path", "taxon_lexicon_path",
    "use_gemma4", "llm_backend", "gemma_model_path",
    "llama_model", "llama_host", "llama_timeout_sec",
    "ollama_model", "ollama_host", "gemma_timeout_sec",
    "gemma_conf_threshold", "gemma_prompt_lang",
    "gemma_use_4bit", "gemma_bfloat16", "gemma_device_map",
    "use_geology_llm", "gemma_init_error",
    # OpenDataLoader integration
    "use_opendataloader", "od_use_ocr", "od_ocr_lang",
    "od_merge_gap_pt",
}


@dataclass(slots=True)
class PipelineConfig:
    """Pipeline configuration."""

    pdf_dir: Path
    work_dir: Path
    grobid_url: str = "http://localhost:8070"
    output_dir: Path | None = None
    use_gpu: bool = True
    ocr_backend: str = "paddleocr"
    taxon_model: str = "en_eco"
    min_panel_score: float = 0.80
    caption_window: int = 2
    num_workers: int = 4
    render_dpi: int = 200
    save_intermediate: bool = True
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.extra.keys()) - _KNOWN_EXTRA_KEYS
        if unknown:
            logger.warning("Unknown extra config keys (typo?): %s", sorted(unknown))

    def resolved_output_dir(self) -> Path:
        return self.output_dir or (self.work_dir / "output")

    def figures_dir(self) -> Path:
        return self.resolved_output_dir() / "figures"

    def tei_dir(self) -> Path:
        return self.resolved_output_dir() / "tei"

    def panels_dir(self) -> Path:
        return self.resolved_output_dir() / "panels"

    def manifests_dir(self) -> Path:
        return self.resolved_output_dir() / "manifests"
