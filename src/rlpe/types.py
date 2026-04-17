from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CaptionEntity:
    text: str
    start: int | None = None
    end: int | None = None
    label: str | None = None
    score: float | None = None


@dataclass(slots=True)
class CaptionRecord:
    paper_id: str
    figure_id: str
    caption: str
    entities: list[CaptionEntity] = field(default_factory=list)
    figure_number: str | None = None
    page_index: int | None = None
    panel_labels: list[str] = field(default_factory=list)
    source_xml: str | None = None


@dataclass(slots=True)
class PanelCandidate:
    panel_id: str | None
    bbox: tuple[int, int, int, int]
    score: float
    region_id: str | None = None
    source_page: int | None = None
    panel_index: int | None = None
    mask_path: str | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MatchResult:
    paper_id: str
    figure_id: str
    panel_id: str | None
    species: str | None
    panel_path: str | None
    bbox: list[int] | None
    confidence: float
    label_text: str | None = None
    caption_snippet: str | None = None
    ocr_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PageRecord:
    page_index: int
    image_path: str
    text: str = ""
    width: int = 0
    height: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FigureRegion:
    page_index: int
    bbox: tuple[int, int, int, int]
    crop_path: str | None = None
    score: float = 0.0
    region_id: str | None = None
    kind: str = "figure"
    metadata: dict[str, Any] = field(default_factory=dict)
