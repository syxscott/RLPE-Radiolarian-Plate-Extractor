from __future__ import annotations

import re
from dataclasses import dataclass

from .ocr import OCRToken
from .taxon import TaxonEntity
from .types import CaptionRecord, MatchResult, PanelCandidate


SUBPANEL_LABEL_PATTERN = re.compile(r"(?:\(|\[)?([A-Z]|[0-9]{1,2})(?:\)|\])?(?=\s*[:\.\-\)]|\s|,)"
)
TAXON_LIKE_PATTERN = re.compile(r"\b([A-Z][a-zA-Z-]+\s+[a-z][a-zA-Z-]+(?:\s+(?:sp\.|spp\.|cf\.|aff\.))?)\b")


@dataclass(slots=True)
class MatchBundle:
    panel_id: str | None
    species: str | None
    label_text: str | None
    confidence: float


def extract_panel_labels(caption_text: str) -> list[str]:
    if not caption_text:
        return []
    labels: list[str] = []
    # Prefer explicit subpanel markers like (A), A., B-, 1), etc.
    for m in SUBPANEL_LABEL_PATTERN.finditer(caption_text):
        label = m.group(1).strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def extract_taxa_from_caption(caption_text: str) -> list[str]:
    if not caption_text:
        return []
    taxa: list[str] = []
    for m in TAXON_LIKE_PATTERN.finditer(caption_text):
        tax = m.group(1).strip()
        if tax and tax not in taxa:
            taxa.append(tax)
    return taxa


def label_tokens_from_ocr(tokens: list[OCRToken]) -> list[OCRToken]:
    out: list[OCRToken] = []
    for tok in tokens:
        text = tok.text.strip()
        if re.fullmatch(r"[A-Z]", text) or re.fullmatch(r"\d{1,2}", text):
            out.append(tok)
    return out


def assign_panels_to_labels(panels: list[PanelCandidate], labels: list[str], ocr_tokens: list[OCRToken]) -> list[str | None]:
    if not panels:
        return []
    if not labels:
        # Fallback to OCR labels if available.
        ocr_labels = [tok.text.strip() for tok in label_tokens_from_ocr(ocr_tokens)]
        labels = ocr_labels
    out: list[str | None] = []
    for i, _ in enumerate(panels):
        out.append(labels[i] if i < len(labels) else None)
    return out


def match_panels(
    paper_id: str,
    figure_id: str,
    caption: CaptionRecord,
    panels: list[PanelCandidate],
    ocr_tokens: list[OCRToken],
    taxon_entities: list[TaxonEntity],
) -> list[MatchResult]:
    labels = caption.panel_labels or extract_panel_labels(caption.caption)
    taxa = [t.text for t in taxon_entities] or extract_taxa_from_caption(caption.caption)
    assigned_labels = assign_panels_to_labels(panels, labels, ocr_tokens)
    panel_label_tokens = {tok.text.strip(): tok for tok in label_tokens_from_ocr(ocr_tokens)}

    matches: list[MatchResult] = []
    for idx, panel in enumerate(panels):
        panel_id = assigned_labels[idx] if idx < len(assigned_labels) else panel.panel_id
        best_species = taxa[idx] if idx < len(taxa) else (taxa[0] if taxa else None)
        ocr_text = " ".join(tok.text for tok in ocr_tokens if _token_in_panel(tok, panel))
        label_text = None
        if panel_id and panel_id in panel_label_tokens:
            label_text = panel_label_tokens[panel_id].text
        confidence = float(panel.score)
        if panel_id:
            confidence += 0.08
        if best_species:
            confidence += 0.12
        if ocr_text:
            confidence += 0.03
        confidence = min(0.99, confidence)
        matches.append(
            MatchResult(
                paper_id=paper_id,
                figure_id=figure_id,
                panel_id=panel_id,
                species=best_species,
                label_text=label_text or panel_id,
                panel_path=panel.image_path,
                bbox=list(panel.bbox),
                confidence=confidence,
                caption_snippet=caption.caption[:240] if caption.caption else None,
                ocr_text=ocr_text or None,
                metadata={
                    "panel_score": panel.score,
                    "ocr_count": len(ocr_tokens),
                    "taxon_count": len(taxon_entities),
                    "figure_number": caption.figure_number,
                    "page_index": caption.page_index,
                },
            )
        )

    if not matches and (taxa or labels):
        matches.append(
            MatchResult(
                paper_id=paper_id,
                figure_id=figure_id,
                panel_id=labels[0] if labels else None,
                species=taxa[0] if taxa else None,
                label_text=labels[0] if labels else None,
                panel_path=None,
                bbox=None,
                confidence=0.35,
                caption_snippet=caption.caption[:240] if caption.caption else None,
            )
        )
    return matches


def _token_in_panel(token: OCRToken, panel: PanelCandidate) -> bool:
    x, y, w, h = panel.bbox
    tx, ty, tw, th = token.bbox
    center_x = tx + tw / 2
    center_y = ty + th / 2
    return x <= center_x <= x + w and y <= center_y <= y + h
