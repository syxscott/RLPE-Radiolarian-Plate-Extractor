from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
