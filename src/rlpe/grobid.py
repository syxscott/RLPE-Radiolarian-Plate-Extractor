from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from .types import CaptionEntity, CaptionRecord
from .layout import extract_figure_number
from .utils import ensure_dir, stable_id


@dataclass(slots=True)
class GrobidResult:
    paper_id: str
    pdf_path: Path
    tei_path: Path | None
    tei_xml: str | None
    captions: list[CaptionRecord]
    fulltext_sections: list[dict[str, str]]
    success: bool
    error: str | None = None


class GrobidClient:
    def __init__(self, server_url: str = "http://localhost:8070", timeout: int = 300) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def process_pdf(self, pdf_path: Path, output_dir: Path) -> GrobidResult:
        paper_id = stable_id(pdf_path)
        tei_dir = ensure_dir(output_dir / "tei")
        tei_path = tei_dir / f"{paper_id}.tei.xml"
        try:
            with pdf_path.open("rb") as f:
                resp = requests.post(
                    f"{self.server_url}/api/processFulltextDocument",
                    files={"input": (pdf_path.name, f, "application/pdf")},
                    data={"consolidateHeader": "1", "consolidateCitations": "1"},
                    timeout=self.timeout,
                )
            resp.raise_for_status()
            tei_path.write_text(resp.text, encoding="utf-8")
            captions = parse_captions_from_tei(resp.text, paper_id=paper_id, source_xml=str(tei_path))
            sections = parse_fulltext_sections_from_tei(resp.text)
            return GrobidResult(
                paper_id=paper_id,
                pdf_path=pdf_path,
                tei_path=tei_path,
                tei_xml=resp.text,
                captions=captions,
                fulltext_sections=sections,
                success=True,
            )
        except Exception as exc:
            return GrobidResult(
                paper_id=paper_id,
                pdf_path=pdf_path,
                tei_path=None,
                tei_xml=None,
                captions=[],
                fulltext_sections=[],
                success=False,
                error=str(exc),
            )


def parse_captions_from_tei(tei_xml: str, paper_id: str, source_xml: str | None = None) -> list[CaptionRecord]:
    if not tei_xml.strip():
        return []
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return []

    ns = {"tei": root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""}
    captions: list[CaptionRecord] = []
    for idx, fig in enumerate(root.findall(".//tei:figure", ns) if ns["tei"] else root.findall(".//figure"), start=1):
        fig_id = fig.attrib.get("xml:id") or fig.attrib.get("id") or f"fig{idx}"
        caption = extract_figure_caption(fig, ns)
        figure_number = fig.attrib.get("n") or extract_figure_number(caption) or _figure_number_from_id(fig_id)
        panel_labels = extract_panel_labels_from_caption(caption)
        entities = extract_taxon_candidates(caption)
        captions.append(
            CaptionRecord(
                paper_id=paper_id,
                figure_id=fig_id,
                caption=caption,
                entities=entities,
                figure_number=figure_number,
                panel_labels=panel_labels,
                source_xml=source_xml,
            )
        )
    return captions


def extract_figure_caption(fig: ET.Element, ns: dict[str, str]) -> str:
    parts: list[str] = []
    for tag in ("head", "figDesc", "note", "label"):
        node = fig.find(f"tei:{tag}", ns) if ns.get("tei") else fig.find(tag)
        if node is not None:
            text = " ".join(t.strip() for t in node.itertext() if t and t.strip())
            if text:
                parts.append(text)
    if not parts:
        text = " ".join(t.strip() for t in fig.itertext() if t and t.strip())
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def extract_taxon_candidates(text: str) -> list[CaptionEntity]:
    if not text:
        return []
    pattern = re.compile(r"\b([A-Z][a-zA-Z-]+\s+(?:sp\.|spp\.|cf\.|aff\.|[a-z][a-zA-Z-]+))\b")
    out: list[CaptionEntity] = []
    for m in pattern.finditer(text):
        out.append(CaptionEntity(text=m.group(1), start=m.start(1), end=m.end(1), label="taxon", score=0.65))
    return out


def extract_panel_labels_from_caption(text: str) -> list[str]:
    if not text:
        return []
    labels: list[str] = []
    pattern = re.compile(r"(?:\(|\[)?([A-Z]|[0-9]{1,2})(?:\)|\])?(?=\s*[:\.\-\)]|\s|,)")
    for m in pattern.finditer(text):
        label = m.group(1)
        if label not in labels:
            labels.append(label)
    return labels


def _figure_number_from_id(fig_id: str) -> str | None:
    m = re.search(r"(\d+[A-Za-z]?)", fig_id)
    return m.group(1) if m else None


def parse_fulltext_sections_from_tei(tei_xml: str) -> list[dict[str, str]]:
    """Parse structured full text sections from TEI for geology/systematics extraction."""
    if not tei_xml.strip():
        return []
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return []

    ns = {"tei": root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""}
    sections: list[dict[str, str]] = []
    divs = root.findall(".//tei:text//tei:body//tei:div", ns) if ns["tei"] else root.findall(".//text//body//div")
    for idx, div in enumerate(divs, start=1):
        head = div.find("tei:head", ns) if ns["tei"] else div.find("head")
        title = " ".join(t.strip() for t in head.itertext() if t and t.strip()) if head is not None else f"section_{idx}"

        paragraphs = div.findall("tei:p", ns) if ns["tei"] else div.findall("p")
        text = "\n".join(" ".join(t.strip() for t in p.itertext() if t and t.strip()) for p in paragraphs)
        text = text.strip()
        if not text:
            continue
        section_type = infer_section_type(title)
        sections.append({"section_id": f"sec_{idx}", "title": title, "section_type": section_type, "text": text})
    return sections


def infer_section_type(title: str) -> str:
    t = (title or "").lower()
    if "systematic" in t or "paleontology" in t:
        return "systematic_paleontology"
    if "geological" in t or "setting" in t or "stratigraph" in t:
        return "geological_setting"
    if "material" in t or "method" in t:
        return "materials_methods"
    return "other"


def process_pdf_dir(pdf_dir: Path, output_dir: Path, server_url: str = "http://localhost:8070") -> list[GrobidResult]:
    client = GrobidClient(server_url=server_url)
    results = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        results.append(client.process_pdf(pdf_path, output_dir))
    return results
