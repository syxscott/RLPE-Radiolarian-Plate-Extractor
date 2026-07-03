from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Recognised extra-config keys; any key outside this set triggers a warning.
_KNOWN_EXTRA_KEYS = {
    "sam2_checkpoint",
    "sam2_model_cfg",
    "sam2_grid_size",
    "sam2_max_point_prompts",
    "sam2_max_box_prompts",
    "use_neural_matcher",
    "matcher_checkpoint_path",
    "taxon_hf_model_path",
    "taxon_lexicon_path",
    "use_gemma4",
    "llm_backend",
    "gemma_model_path",
    "llama_model",
    "llama_host",
    "llama_timeout_sec",
    "ollama_model",
    "ollama_host",
    "gemma_timeout_sec",
    "gemma_conf_threshold",
    "gemma_prompt_lang",
    "gemma_use_4bit",
    "gemma_bfloat16",
    "gemma_device_map",
    "use_geology_llm",
    "gemma_init_error",
    # MiniMax M3 API (Anthropic-compatible)
    "MiniMax_api_key",
    "MiniMax_endpoint",
    "MiniMax_model",
    "MiniMax_max_concurrent",
    "MiniMax_timeout_sec",
    "MiniMax_max_retries",
    "MiniMax_enable_thinking",
    "MiniMax_thinking_budget_tokens",
    "MiniMax_max_output_tokens",
    "MiniMax_fallback_default",
    "MiniMax_interactive",
    "data_outbound_policy",
    "_MiniMax_external_handler",  # injected by web/API layer
    # OpenDataLoader integration
    "use_opendataloader",
    "od_use_ocr",
    "od_ocr_lang",
    "od_merge_gap_pt",
    # M3 5-stage semantic engine
    "m3_enhanced_mode",
    "m3_stage_1",
    "m3_stage_2",
    "m3_stage_3",
    "m3_stage_4",
    "m3_stage_5",
    "m3_match_samples",
    "m3_diagnostic_dir",
    "m3_skip_match_on_empty_caption",
    "m3_retry_without_thinking",
    "m3_temperature",
    "m3_thinking_budget",
    # LLM-first extraction (opt-in; default True when Gemma runtime is set)
    "use_llm_first",
    # Multi-modal geology vision (Commit 2 / Round 3)
    "use_geo_vision",
    "geo_vision_figure_types",
    # Paleobiology Database (opt-in)
    "use_paleodb",
    "paleodb_max_occurrences",
    "paleodb_endpoint",
    "paleodb_cache_dir",
    "paleodb_offline",
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
