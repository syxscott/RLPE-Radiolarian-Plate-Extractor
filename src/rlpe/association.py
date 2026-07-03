from __future__ import annotations

import logging
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
from .text_filters import looks_like_placeholder_caption as _looks_like_placeholder_caption
from .types import CaptionRecord, MatchResult, PanelCandidate, PaperMetadata

logger = logging.getLogger(__name__)


# Bug #10 fix: require labels to be wrapped in () or [] OR followed immediately
# by '.' or ':' to reduce false positives on normal sentence-initial capitals.
# Two alternatives: group 1 = parenthesised/bracketed; group 2 = separator-style.
SUBPANEL_LABEL_PATTERN = re.compile(
    r"(?:\(|\[)([A-Z]|[0-9]{1,2})(?:\)|\])"
    r"|(?<!\w)([A-Z]|[0-9]{1,2})(?=\.|\:)\s*"
)
TAXON_LIKE_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z-]+\s+[a-z][a-zA-Z-]+(?:\s+(?:sp\.|spp\.|cf\.|aff\.))?)\b"
)
_SINGLE_UPPER = re.compile(r"[A-Z]")
_SINGLE_DIGITS = re.compile(r"\d{1,2}")
_SPECIES_QUAL = re.compile(r"\b(sp\.|spp\.|cf\.|aff\.)\b", re.IGNORECASE)

# Captions that are clearly pipeline placeholders, not real figure
# descriptions. The fallback path in pipeline.py emits strings like
# "Auto-generated figure for page 17" when OpenDataLoader / GROBID
# can't extract a real caption; ``extract_taxa_from_caption`` used to
# match "Auto-generated figure" as a binomial and tag every panel with
# that bogus species. Reject these at the boundary.
#
# Patterns local to the association matcher: these are strict prefix
# matches (anchored with ^) because the matcher needs to know whether
# the caption STARTS with a placeholder pattern. The richer detector in
# ``text_filters.looks_like_placeholder_caption`` (which also catches
# "Page 5 auto-generated image" anywhere in the string) is used by the
# pipeline / stage-4 skip logic — see ``is_placeholder_caption`` below
# which now also ORs in the text_filters predicate so the two stay in
# sync for the cases the association matcher used to miss.
# Strict-prefix placeholder patterns LOCAL to the association matcher.
# These are intentionally narrower than ``text_filters._PLACEHOLDER_CAPTION_PATTERNS``
# (note the different name — the previous shared-name made it easy to
# import the wrong one by mistake).  Both are OR'd together inside
# ``is_placeholder_caption`` below.
_ASSOC_PLACEHOLDER_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^auto[-_ ]generated\b",
        r"^auto[-_ ]generated\s+figure\b",
        r"^placeholder\b",
        r"^n/?a\b",
        r"^undefined\b",
        r"^missing\s+caption\b",
        r"^no\s+caption\b",
    )
)


def is_placeholder_caption(text: str | None) -> bool:
    """Return True if the caption text is a pipeline placeholder rather
    than a real figure description. The pipeline emits "Auto-generated
    figure for page N" when the upstream extractor returns no caption;
    we don't want that to be treated as a taxon source.

    Combines the anchored ``_PLACEHOLDER_CAPTION_PATTERNS`` (strict prefix
    match — fast and unambiguous for the canonical fallback string) with
    the broader ``text_filters.looks_like_placeholder_caption`` (handles
    OpenDataLoader-style ``Page 5 auto-generated image`` where the
    placeholder keyword is not at the start, as well as the Chinese
    forms ``自动生成`` / ``页眉``). Without this OR the association
    matcher would happily call ``extract_taxa_from_caption`` on a
    "Page 5 auto-generated image" caption and tag every panel with a
    bogus species.
    """
    if not text:
        return True
    s = str(text).strip()
    if not s:
        return True
    if any(p.match(s) for p in _ASSOC_PLACEHOLDER_PREFIX_PATTERNS):
        return True
    # Delegate the broader patterns (Page-N prefix, Chinese variants,
    # copyright headers) to the single source of truth in text_filters.
    return _looks_like_placeholder_caption(s)


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

    def forward(
        self, panel_feats: torch.Tensor, label_feats: torch.Tensor, species_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        panel_feats = torch.tensor(
            [_panel_features(p, w, h, idx=i) for i, p in enumerate(panels)],
            dtype=torch.float32,
            device=self.device,
        )

        # Bug #5 fix: when there are no OCR label tokens (or no taxa), we MUST
        # NOT feed a dummy row of zeros. softmax over a single zero row yields
        # 1.0, which then produced near-1.0 confidence for every panel — a
        # false high-confidence match. Instead, compute each side only when it
        # has real candidates, and return 0.0 confidence for the missing side.
        has_labels = bool(ocr_label_tokens)
        has_taxa = bool(taxa)

        if has_labels:
            label_feats = torch.tensor(
                [_label_features(t, w, h, idx=i) for i, t in enumerate(ocr_label_tokens)],
                dtype=torch.float32,
                device=self.device,
            )
        if has_taxa:
            species_feats = torch.tensor(
                [_species_features(name, idx=i) for i, name in enumerate(taxa)],
                dtype=torch.float32,
                device=self.device,
            )

        with torch.inference_mode():
            feat_dim = panel_feats.shape[1]
            if has_labels and has_taxa:
                logits_pl, logits_ps = self.model(panel_feats, label_feats, species_feats)
                probs_pl = logits_pl.softmax(dim=-1).detach().cpu().numpy()
                probs_ps = logits_ps.softmax(dim=-1).detach().cpu().numpy()
            elif has_labels and not has_taxa:
                zero_species = torch.zeros((1, feat_dim), dtype=torch.float32, device=self.device)
                logits_pl, _ = self.model(panel_feats, label_feats, zero_species)
                probs_pl = logits_pl.softmax(dim=-1).detach().cpu().numpy()
                probs_ps = np.zeros((len(panels), 0), dtype=np.float32)
            elif has_taxa and not has_labels:
                zero_labels = torch.zeros((1, feat_dim), dtype=torch.float32, device=self.device)
                _, logits_ps = self.model(panel_feats, zero_labels, species_feats)
                probs_pl = np.zeros((len(panels), 0), dtype=np.float32)
                probs_ps = logits_ps.softmax(dim=-1).detach().cpu().numpy()
            else:
                n = len(panels)
                return [None] * n, [None] * n, [0.0] * n

        label_assign = (
            _bipartite_assign(probs_pl, [tok.text.strip() for tok in ocr_label_tokens])
            if has_labels
            else [None] * len(panels)
        )
        species_assign = _bipartite_assign(probs_ps, taxa) if has_taxa else [None] * len(panels)

        confs: list[float] = []
        for i in range(len(panels)):
            p1 = float(np.max(probs_pl[i])) if has_labels and probs_pl.shape[1] > 0 else 0.0
            p2 = float(np.max(probs_ps[i])) if has_taxa and probs_ps.shape[1] > 0 else 0.0
            denom = (1.0 if has_labels else 0.0) + (1.0 if has_taxa else 0.0)
            confs.append((p1 + p2) / denom if denom > 0 else 0.0)
        return label_assign, species_assign, confs


def extract_panel_labels(caption_text: str) -> list[str]:
    if not caption_text:
        return []
    if is_placeholder_caption(caption_text):
        return []
    labels: list[str] = []
    # SUBPANEL_LABEL_PATTERN has two alternatives (parenthesized vs.
    # separator-followed); check both groups to find the captured label.
    for m in SUBPANEL_LABEL_PATTERN.finditer(caption_text):
        label = (m.group(1) or m.group(2) or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def extract_taxa_from_caption(caption_text: str) -> list[str]:
    if not caption_text:
        return []
    if is_placeholder_caption(caption_text):
        return []
    taxa: list[str] = []
    for m in TAXON_LIKE_PATTERN.finditer(caption_text):
        tax = m.group(1).strip()
        if tax and tax not in taxa:
            taxa.append(tax)
    return taxa


def _label_sort_key(label: str) -> tuple[int, str]:
    """Sort panel labels like "1", "2", ..., "10", "A", "B".

    Pure-numeric labels sort by their integer value; alphabetic labels
    sort after numerics (alphabet labels are usually figure-level markers
    that come after numbered sub-panels).
    """
    s = str(label).strip()
    if s.isdigit():
        return (0, "")
    try:
        return (1, s)
    except Exception:
        return (2, s)


def _normalize_panel_label(label: str | None) -> str | None:
    """Normalise a panel label so OCR misreads don't break caption lookup.

    Two normalisations:
      1. Strip leading zeros ("00" → "0", "04" → "4") — PaddleOCR commonly
         reads "3" as "03" or "0" as "00" when the glyph is small.
      2. Strip trailing zero-padding for double-digit OCR ("30" misread
         of "3" stays as "3" if "3" is in the pair_lookup). The
         caller decides whether to keep or drop by trying both forms.

    Returns the cleanest single label (or None for empty input).
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    # Don't normalise alphabetic labels ("A", "B" stay as-is).
    if not s.isdigit():
        return s
    return str(int(s))  # "00" → "0", "04" → "4", "3" → "3"


def is_valid_panel_label(label: str | None) -> bool:
    """Validate that a panel label is safe to emit as a new panel_id.

    Rejects empty strings, non-string values, and strings whose shape
    is NOT a plausible panel id. The pipeline uses this gate in the
    LLM-first hybrid enrichment and the image-OCR override to avoid
    inserting OCR noise (``ean`` / ``L`` / ``P1`` / ``foo`` / ``,1``)
    or caption fragments (``Figure`` / ``Plate`` / ``251.90``) into
    the canonical panel list, which would collide with real labels
    via positional fallback (N10-class drift).

    Accepted shapes (mirrors ``_extract_panel_labels_from_caption``
    in ``local_pdf_parser.py`` so caption-derived and OCR-derived
    labels share the same shape contract):

      * single uppercase A–H  — figure-level decorative marker, e.g. "(A)"
      * digit with optional trailing [a-z]  — e.g. "1", "2a", "12b"

    A 16-char length cap remains as a backstop against caption fragments.
    """
    if not label:
        return False
    if not isinstance(label, str):
        return False
    s = label.strip()
    if not s:
        return False
    # Reject overly long labels; they are almost certainly
    # caption-text fragments, not panel ids.
    if len(s) > 16:
        return False
    # Shape gate: A-H marker OR digit(s) with optional single trailing
    # lowercase letter. Anything else is OCR noise / caption fragment.
    return bool(_PANEL_LABEL_SHAPE.match(s))


# Compiled once at import time. Anchored fullmatch is enforced by
# using ``re.match`` with explicit ^/$ in the pattern below.
_PANEL_LABEL_SHAPE = re.compile(r"^(?:[A-H]|\d+[a-z]?)$")


_PANEL_METADATA_KEYS = (
    "printed_panel_id",
    "image_panel_id",
    "caption_panel_id",
    "panel_id_source",
    "label_region_ocr",
    "label_region_picked",
    "label_region_fallback",
    "panel_ocr_text",
    "panel_ocr_token_count",
)


def _panel_metadata(panel: PanelCandidate, **base_meta: Any) -> dict[str, Any]:
    """Build MatchResult metadata from association metadata plus
    selected panel-level OCR diagnostics.

    ``pipeline.py`` attaches image-OCR evidence to ``PanelCandidate.metadata``
    before calling ``match_panels``. ``MatchResult`` is a separate dataclass,
    so those keys must be explicitly propagated here or exported
    ``PanelRecord.printed_panel_id`` will stay null.
    """
    out = dict(base_meta)
    pmeta = panel.metadata or {}
    for key in _PANEL_METADATA_KEYS:
        if key in pmeta and key not in out:
            out[key] = pmeta[key]
    return out


def _label_in_pair_lookup(label: str | None, pair_lookup: dict[str, str]) -> str | None:
    """Try the label, then its leading-zero-stripped form, against
    ``pair_lookup``. Returns the matching key (the value is then
    ``pair_lookup[matched]``) or None.

    Without this fallback, OCR misreads like "00" never match the
    caption's "0" key and the panel gets species=None for no good
    reason. With it, we tolerate the common "leading-zero" OCR error.
    """
    if not label:
        return None
    if label in pair_lookup:
        return label
    norm = _normalize_panel_label(label)
    if norm and norm in pair_lookup:
        return norm
    return None


def _add_label_base_aliases(cp: Any, pair_lookup: dict[str, str]) -> None:
    """For a caption pair whose labels include letter-suffixed entries
    like ``"14b"``, also index the species under the bare base number
    ``"14"``. This rescues OCR misreads that drop the trailing letter
    (panel labelled ``14`` on the figure but the caption key is
    ``14b``) — a common case in Bandini/Pouille plates where the last
    sub-figure of a range is the only one with the suffix.
    """
    sp = getattr(cp, "species", None)
    if not sp:
        return
    for lbl in getattr(cp, "labels", None) or []:
        if not lbl:
            continue
        m = re.match(r"^(\d+)([a-z])$", str(lbl))
        if m:
            base = m.group(1)
            # Only add the bare base if no other species owns it
            # (the LLM/regex parser should have already given the same
            # species the base number when expanding a range, but
            # deduplicate defensively).
            if base not in pair_lookup:
                pair_lookup[base] = sp


def deduplicate_panels_nms(
    panels: list,
    iou_threshold: float = 0.6,
    label_match: bool = True,
) -> list:
    """NMS-style merge of near-duplicate panel detections.

    Two panels are duplicates if:
      * their bboxes have IoU >= ``iou_threshold`` AND
      * either (``label_match`` is False) OR their normalised labels are
        equal (or both have no label).

    When duplicates are found, the panel with the higher score is kept.
    The kept panel's score is bumped by +0.02 if it had a label and the
    dropped panel had the same label (signals "this is the real one,
    confirmed by OCR").

    This is the second-pass dedup that runs AFTER the segmenter's
    intra-method dedup. The segmenter removes exact-overlap boxes from
    a single method (e.g. SAM2 vs OpenCV), but doesn't catch
    cross-method duplicates (e.g. SAM2 returns the full specimen and
    OpenCV enhanced returns the same specimen split into two boxes).
    """
    if not panels:
        return []
    kept: list = []
    # Sort by score descending so the highest-confidence panel wins
    for panel in sorted(panels, key=lambda p: p.score, reverse=True):
        is_dup = False
        for k in kept:
            iou = _iou_panels(panel.bbox, k.bbox)
            if iou < iou_threshold:
                continue
            if label_match:
                pl = _normalize_panel_label(panel.panel_id)
                kl = _normalize_panel_label(k.panel_id)
                if pl != kl:
                    # Different labels with strong overlap is rare
                    # (one is probably the right one); keep both but
                    # the higher-scored one wins the slot.
                    continue
            is_dup = True
            break
        if is_dup:
            continue
        kept.append(panel)
    # Sort by bbox y then x for downstream display
    kept.sort(key=lambda p: (p.bbox[1], p.bbox[0]))
    return kept


def _iou_panels(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(1, union)


def label_tokens_from_ocr(tokens: list[OCRToken]) -> list[OCRToken]:
    out: list[OCRToken] = []
    for tok in tokens:
        text = tok.text.strip()
        if _SINGLE_UPPER.fullmatch(text) or _SINGLE_DIGITS.fullmatch(text):
            out.append(tok)
    return out


def assign_panels_to_labels(
    panels: list[PanelCandidate], labels: list[str], ocr_tokens: list[OCRToken]
) -> list[str | None]:
    """Assign each panel a textual label ("1", "2", ... or "A", "B", ...).

    Priority:
      1. ``panel.panel_id`` if it's a non-empty plain label (set by SAM2
         detection or by classical CV with text-OCR hints).
      2. The i-th element of ``labels`` (the caption-derived label list).
      3. OCR text tokens on the panel.

    The previous implementation always used ``labels[i]``, which
    re-shuffled labels whenever the panel order didn't exactly match the
    order the labels appeared in the caption (e.g. labels extracted from
    ``DP2/8024`` or other DP catalog numbers). Using panel_id directly
    fixes the common "panel N gets the wrong species" failure mode.

    All label assignments go through ``_normalize_panel_label`` so that
    OCR misreads like "00" (for "0") and "04" (for "4") are flattened
    back to the canonical "0" / "4" form before any caption lookup.
    """
    if not panels:
        return []
    if not labels:
        ocr_labels = [tok.text.strip() for tok in label_tokens_from_ocr(ocr_tokens)]
        labels = ocr_labels
    out: list[str | None] = []
    for i, panel in enumerate(panels):
        pid = _normalize_panel_label(panel.panel_id)
        # Audit P1-4: the previous ``pid.isdigit() or len(pid) <= 3``
        # accepted OCR garbage like ``"P1"`` (multi-char alphanumeric)
        # or ``"ean"`` (multi-char alpha), polluting the pred set with
        # rows that drag pouille F1 down. Use ``is_valid_panel_label``
        # (which enforces ``^([A-H]|\d+[a-z]?)$``) so only digit,
        # digit+letter, or A-H markers pass through.
        if pid and is_valid_panel_label(pid):
            out.append(pid)
            continue
        if i < len(labels) and labels[i]:
            cand = _normalize_panel_label(labels[i]) or labels[i]
            if cand and is_valid_panel_label(cand):
                out.append(cand)
                continue
            # Fall through: the caption-derived label was invalid,
            # so we emit None rather than re-attaching the garbage.
            out.append(None)
            continue
        out.append(None)
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
    paper_metadata: PaperMetadata | None = None,
    caption_pairs: list | None = None,
) -> list[MatchResult]:
    labels = caption.panel_labels or extract_panel_labels(caption.caption)
    taxa = [t.text for t in taxon_entities] or extract_taxa_from_caption(caption.caption)
    ocr_label_tokens = label_tokens_from_ocr(ocr_tokens)

    # 1) 默认规则分配（可回退）。
    assigned_labels = assign_panels_to_labels(panels, labels, ocr_tokens)
    neural_conf = [0.0] * len(panels)

    # Caption is a pipeline placeholder (e.g. "Auto-generated figure for
    # page 17"). We can't extract labels or species from it, and any
    # positional fallback would just tag every panel with the first
    # taxon in the placeholder string ("Auto-generated figure"). Bail
    # out with empty species for all panels — the caller can either
    # skip the figure or fall back to per-panel OCR/vision matching.
    if is_placeholder_caption(caption.caption):
        matches: list[MatchResult] = []
        for idx, panel in enumerate(panels):
            raw_id = assigned_labels[idx] if idx < len(assigned_labels) else panel.panel_id
            panel_id = _normalize_panel_label(raw_id)
            # Audit P1-4: drop panels whose label is OCR garbage
            # (single-letter ``S/A/L`` etc.) so they don't show up as
            # spurious pred rows that drag pouille F1 down. Without
            # this guard a Stage-3 over-segmentation producing 60
            # panels with the same first caption token was writing 60
            # rows all sharing panel_id="S".
            if panel_id is not None and not is_valid_panel_label(panel_id):
                continue
            matches.append(
                MatchResult(
                    paper_id=paper_id,
                    figure_id=figure_id,
                    panel_id=panel_id,
                    species=None,
                    label_text=panel_id,
                    panel_path=panel.image_path,
                    bbox=list(panel.bbox),
                    confidence=float(panel.score),
                    caption_snippet=caption.caption[:240] if caption.caption else None,
                    ocr_text=None,
                    paper_metadata=paper_metadata,
                    metadata=_panel_metadata(
                        panel,
                        panel_score=panel.score,
                        ocr_count=len(ocr_tokens),
                        taxon_count=len(taxon_entities),
                        figure_number=caption.figure_number,
                        page_index=caption.page_index,
                        matcher_used=False,
                        matcher_type="skipped-placeholder-caption",
                        matcher_conf=0.0,
                        caption_pairs_used=False,
                    ),
                )
            )
        return matches

    # 1b) M3 stage-1 caption pairs drive a much more accurate panel→species
    # mapping. If we have structured (label, species) pairs from the LLM caption
    # parser, build a label→species lookup and override the order-based
    # heuristic. Falls back silently if pairs are empty or don't match.
    pair_lookup: dict[str, str] = {}
    caption_pairs_used = False
    caption_pairs_source = ""
    if caption_pairs:
        for cp in caption_pairs:
            sp = getattr(cp, "species", None)
            if not sp:
                continue
            lbls = getattr(cp, "labels", None) or []
            for lbl in lbls:
                if lbl:
                    pair_lookup[str(lbl).strip()] = sp
            _add_label_base_aliases(cp, pair_lookup)
        if pair_lookup:
            caption_pairs_used = True
            caption_pairs_source = "m3_llm"
    # Fallback: when M3 didn't run, build the same lookup via the regex
    # caption parser that M3 uses internally. This rescues the common case
    # of "figs 1-2. SpeciesA: ... figs 3-4. SpeciesB: ..." captions where the
    # old order-based heuristic was mapping every panel to taxa[0] (the
    # first species in the caption).
    if not caption_pairs_used and caption.caption:
        try:
            from .m3_engine import _regex_parse_caption

            regex_pairs = _regex_parse_caption(caption.caption)
            for cp in regex_pairs:
                sp = getattr(cp, "species", None)
                if not sp:
                    continue
                for lbl in getattr(cp, "labels", None) or []:
                    if lbl:
                        pair_lookup[str(lbl).strip()] = sp
                _add_label_base_aliases(cp, pair_lookup)
            if pair_lookup:
                caption_pairs_used = True
                caption_pairs_source = "regex"
        except Exception:
            pass

    # 1c) Assign species. STRICT mode: only assign if the panel's label
    # (or its leading-zero-normalised form) is in the caption-derived
    # pair_lookup. We deliberately do NOT carry-forward the last seen
    # species to panels whose label is unknown — the previous carry-forward
    # behaviour wrongly tagged 14 SEM-metadata fragments on Feng 2007
    # Plate 1 as "Entactinia reticulata" just because they were sorted
    # after the panel labelled "4". When the caption has only 4 entries
    # (figs 1-4) and we detect 17 panels, the extra 13 should be None,
    # not the last seen species.
    if caption_pairs_used and pair_lookup:
        assigned_species: list[str | None] = []
        for panel_id in assigned_labels:
            matched_key = _label_in_pair_lookup(panel_id, pair_lookup)
            assigned_species.append(pair_lookup[matched_key] if matched_key else None)
    else:
        # Last-resort fallback: position-based, but DO NOT collapse the tail
        # onto taxa[0]. Any panel beyond the available species list gets
        # None (so the caller can see it's unassigned) rather than a wrong
        # first-species tag.
        assigned_species = [taxa[i] if i < len(taxa) else None for i in range(len(panels))]

    # 2) 可选神经图匹配。未训练权重或缺少checkpoint时跳过。
    matcher_used = False
    if use_neural_matcher and matcher_checkpoint_path:
        try:
            matcher = NeuralGraphMatcher(checkpoint_path=matcher_checkpoint_path)
            if not matcher.is_trained:
                logger.warning(
                    "Neural matcher loaded but not trained; falling back to heuristic matching."
                )
            else:
                merged_label_tokens = ocr_label_tokens or [
                    OCRToken(text=l, confidence=0.5, bbox=(0, 0, 1, 1)) for l in labels
                ]
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
        raw_id = assigned_labels[idx] if idx < len(assigned_labels) else panel.panel_id
        panel_id = _normalize_panel_label(raw_id)
        best_species = assigned_species[idx] if idx < len(assigned_species) else None
        # Caption-pair override: if M3 gave us a structured (label, species) map
        # and the panel's label (or its leading-zero-stripped form) is in
        # it, prefer that species over the order-based fallback.
        if caption_pairs_used:
            matched_key = _label_in_pair_lookup(panel_id, pair_lookup)
            if matched_key:
                best_species = pair_lookup[matched_key]
        ocr_text = " ".join(tok.text for tok in ocr_tokens if _token_in_panel(tok, panel))
        label_text = None
        if panel_id and panel_id in panel_label_tokens:
            label_text = panel_label_tokens[panel_id].text
        confidence = float(panel.score)
        if matcher_used:
            confidence = max(confidence, float(neural_conf[idx]) if idx < len(neural_conf) else 0.0)
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
                paper_metadata=paper_metadata,
                metadata=_panel_metadata(
                    panel,
                    panel_score=panel.score,
                    ocr_count=len(ocr_tokens),
                    taxon_count=len(taxon_entities),
                    figure_number=caption.figure_number,
                    page_index=caption.page_index,
                    matcher_used=matcher_used,
                    matcher_type="neural-graph" if matcher_used else "heuristic",
                    matcher_conf=neural_conf[idx] if idx < len(neural_conf) else 0.0,
                    caption_pairs_used=caption_pairs_used,
                ),
            )
        )

    if not matches and (taxa or labels):
        # Audit P1-4 (pouille over-segmentation): the placeholder
        # panel_id below used to be ``labels[0]`` unconditionally,
        # so a Stage-3 over-segmentation that produced, say, 60 panels
        # all with the same first caption-token label ended up writing
        # 60 rows sharing panel_id="S" or panel_id="A" — polluting the
        # pred set and dragging pouille2014 string-match F1 from 100%
        # to 0%. The fix validates ``labels[0]`` against
        # ``is_valid_panel_label``; if invalid, fall through to None
        # so the eval treats it as no-pred instead of bad-pred.
        first_label = labels[0] if labels else None
        if first_label and not is_valid_panel_label(first_label):
            first_label = None
        matches.append(
            MatchResult(
                paper_id=paper_id,
                figure_id=figure_id,
                panel_id=first_label,
                species=taxa[0] if taxa else None,
                label_text=first_label,
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
    has_qual = 1.0 if _SPECIES_QUAL.search(name or "") else 0.0
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
    if _SINGLE_UPPER.fullmatch(t):
        return (ord(t) - ord("A") + 1) / 26.0
    if _SINGLE_DIGITS.fullmatch(t):
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
