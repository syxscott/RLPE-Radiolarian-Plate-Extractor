from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

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


class PanelLabelSpeciesMatcher(nn.Module if nn is not None else object):
    """轻量匹配网络：
    - 输入：panel/label/species 节点特征
    - 输出：panel-label 与 panel-species 的关联logits
    """

    def __init__(self, feature_dim: int = 12, hidden_dim: int = 64):
        if nn is None:
            raise RuntimeError("PyTorch is required for PanelLabelSpeciesMatcher.")
        super().__init__()
        self.panel_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.label_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.species_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, panel_feats: torch.Tensor, label_feats: torch.Tensor, species_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.panel_encoder(panel_feats)
        l = self.label_encoder(label_feats)
        s = self.species_encoder(species_feats)
        # 双塔点积打分（可训练）
        panel_label_logits = p @ l.t()
        panel_species_logits = p @ s.t()
        return panel_label_logits, panel_species_logits


class NeuralGraphMatcher:
    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        if torch is None:
            raise RuntimeError("PyTorch is not available for NeuralGraphMatcher.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PanelLabelSpeciesMatcher().to(self.device)
        self.is_trained = False
        if checkpoint_path:
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                state = ckpt.get("state_dict", ckpt)
                self.model.load_state_dict(state, strict=False)
                self.is_trained = True
            except Exception:
                self.is_trained = False
        self.model.eval()

    def match(
        self,
        panels: list[PanelCandidate],
        ocr_label_tokens: list[OCRToken],
        taxa: list[str],
        image_shape: tuple[int, int] | None,
    ) -> tuple[list[str | None], list[str | None], list[float]]:
        if not panels:
            return [], [], []
        if not ocr_label_tokens and not taxa:
            n = len(panels)
            return [None] * n, [None] * n, [0.0] * n

        h, w = image_shape if image_shape else (1000, 1000)

        panel_feats = torch.tensor([_panel_features(p, w, h, idx=i) for i, p in enumerate(panels)], dtype=torch.float32, device=self.device)
        label_feats = torch.tensor(
            [_label_features(t, w, h, idx=i) for i, t in enumerate(ocr_label_tokens)] or [[0.0] * 12],
            dtype=torch.float32,
            device=self.device,
        )
        species_feats = torch.tensor(
            [_species_features(name, idx=i) for i, name in enumerate(taxa)] or [[0.0] * 12],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            logits_pl, logits_ps = self.model(panel_feats, label_feats, species_feats)
            probs_pl = logits_pl.softmax(dim=-1).detach().cpu().numpy()
            probs_ps = logits_ps.softmax(dim=-1).detach().cpu().numpy()

        label_assign = _bipartite_assign(probs_pl, [tok.text.strip() for tok in ocr_label_tokens])
        species_assign = _bipartite_assign(probs_ps, taxa)

        confs: list[float] = []
        for i in range(len(panels)):
            p1 = float(np.max(probs_pl[i])) if probs_pl.shape[1] > 0 else 0.0
            p2 = float(np.max(probs_ps[i])) if probs_ps.shape[1] > 0 else 0.0
            confs.append((p1 + p2) * 0.5)
        return label_assign, species_assign, confs


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
    use_neural_matcher: bool = False,
    matcher_checkpoint_path: str | None = None,
    image_shape: tuple[int, int] | None = None,
) -> list[MatchResult]:
    labels = caption.panel_labels or extract_panel_labels(caption.caption)
    taxa = [t.text for t in taxon_entities] or extract_taxa_from_caption(caption.caption)
    ocr_label_tokens = label_tokens_from_ocr(ocr_tokens)

    # 1) 默认规则分配（可回退）。
    assigned_labels = assign_panels_to_labels(panels, labels, ocr_tokens)
    assigned_species = [taxa[i] if i < len(taxa) else (taxa[0] if taxa else None) for i in range(len(panels))]
    neural_conf = [0.0] * len(panels)

    # 2) 可选神经图匹配。
    matcher_used = False
    if use_neural_matcher:
        try:
            matcher = NeuralGraphMatcher(checkpoint_path=matcher_checkpoint_path)
            merged_label_tokens = ocr_label_tokens or [OCRToken(text=l, confidence=0.5, bbox=(0, 0, 1, 1)) for l in labels]
            n_labels, n_species, n_conf = matcher.match(
                panels=panels,
                ocr_label_tokens=merged_label_tokens,
                taxa=taxa,
                image_shape=image_shape,
            )
            if any(v is not None for v in n_labels) or any(v is not None for v in n_species):
                assigned_labels = n_labels
                assigned_species = n_species
                neural_conf = n_conf
                matcher_used = True
        except Exception:
            matcher_used = False

    panel_label_tokens = {tok.text.strip(): tok for tok in ocr_label_tokens}

    matches: list[MatchResult] = []
    for idx, panel in enumerate(panels):
        panel_id = assigned_labels[idx] if idx < len(assigned_labels) else panel.panel_id
        best_species = assigned_species[idx] if idx < len(assigned_species) else (taxa[0] if taxa else None)
        ocr_text = " ".join(tok.text for tok in ocr_tokens if _token_in_panel(tok, panel))
        label_text = None
        if panel_id and panel_id in panel_label_tokens:
            label_text = panel_label_tokens[panel_id].text
        confidence = float(panel.score)
        if matcher_used:
            confidence = max(confidence, float(neural_conf[idx]) if idx < len(neural_conf) else confidence)
        else:
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
                    "matcher_used": matcher_used,
                    "matcher_type": "neural-graph" if matcher_used else "heuristic",
                    "matcher_conf": neural_conf[idx] if idx < len(neural_conf) else 0.0,
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


def _panel_features(panel: PanelCandidate, img_w: int, img_h: int, idx: int) -> list[float]:
    x, y, w, h = panel.bbox
    cx, cy = x + w * 0.5, y + h * 0.5
    area = max(1.0, w * h)
    return [
        x / max(1, img_w),
        y / max(1, img_h),
        w / max(1, img_w),
        h / max(1, img_h),
        cx / max(1, img_w),
        cy / max(1, img_h),
        area / max(1.0, img_w * img_h),
        float(idx) / 100.0,
        float(panel.score),
        1.0,
        0.0,
        0.0,
    ]


def _label_features(token: OCRToken, img_w: int, img_h: int, idx: int) -> list[float]:
    x, y, w, h = token.bbox
    cx, cy = x + w * 0.5, y + h * 0.5
    area = max(1.0, w * h)
    val = _label_to_scalar(token.text)
    return [
        x / max(1, img_w),
        y / max(1, img_h),
        w / max(1, img_w),
        h / max(1, img_h),
        cx / max(1, img_w),
        cy / max(1, img_h),
        area / max(1.0, img_w * img_h),
        float(idx) / 100.0,
        float(token.confidence),
        val,
        1.0,
        0.0,
    ]


def _species_features(name: str, idx: int) -> list[float]:
    genus_len = len(name.split(" ")[0]) if name else 0
    words = len(name.split()) if name else 0
    has_qual = 1.0 if re.search(r"\b(sp\.|spp\.|cf\.|aff\.)\b", name or "", re.IGNORECASE) else 0.0
    return [
        float(genus_len) / 30.0,
        float(words) / 6.0,
        has_qual,
        float(idx) / 100.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _label_to_scalar(text: str) -> float:
    t = (text or "").strip()
    if re.fullmatch(r"[A-Z]", t):
        return (ord(t) - ord("A") + 1) / 26.0
    if re.fullmatch(r"\d{1,2}", t):
        return min(1.0, int(t) / 20.0)
    return 0.0


def _bipartite_assign(prob_matrix: np.ndarray, items: list[str]) -> list[str | None]:
    if prob_matrix.size == 0 or not items:
        return [None] * (prob_matrix.shape[0] if prob_matrix.ndim > 0 else 0)
    n_panels, n_items = prob_matrix.shape
    assigned: list[str | None] = [None] * n_panels

    # 优先使用Hungarian最优匹配；缺失scipy时回退贪心。
    try:
        from scipy.optimize import linear_sum_assignment

        cost = -prob_matrix
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if r < n_panels and c < n_items:
                assigned[r] = items[c]
        return assigned
    except Exception:
        used = set()
        for r in range(n_panels):
            order = np.argsort(-prob_matrix[r])
            for c in order:
                if int(c) not in used:
                    assigned[r] = items[int(c)]
                    used.add(int(c))
                    break
        return assigned
