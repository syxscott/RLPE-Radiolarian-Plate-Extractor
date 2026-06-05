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
    paper_metadata: PaperMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.paper_metadata is not None:
            d["paper_metadata"] = self.paper_metadata.to_dict()
        return d


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


@dataclass(slots=True)
class PaperMetadata:
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    abstract: str | None = None
    keywords: list[str] = field(default_factory=list)
    publisher: str | None = None
    page_count: int | None = None
    source: str = ""           # "grobid" | "opendataloader" | "none"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaxonomyMatch:
    """A taxonomic record returned by the Paleobiology Database."""
    name: str
    rank: str | None = None
    status: str | None = None
    common_name: str | None = None
    kingdom: str | None = None
    phylum: str | None = None
    class_: str | None = None
    order: str | None = None
    family: str | None = None
    genus: str | None = None
    match_score: float = 0.0
    source: str = "paleodb"   # "paleodb" | "cache" | "offline"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Map the class_ key back to "class" for JSON readability
        if "class_" in d:
            d["class"] = d.pop("class_")
        return d


@dataclass(slots=True)
class OccurrenceSummary:
    """A fossil-occurrence record from the Paleobiology Database."""
    species_name: str
    occurrence_id: str | None = None
    collection_id: str | None = None
    early_interval: str | None = None
    late_interval: str | None = None
    max_ma: float | None = None
    min_ma: float | None = None
    locality: str | None = None
    country: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    formation: str | None = None
    source: str = "paleodb"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
