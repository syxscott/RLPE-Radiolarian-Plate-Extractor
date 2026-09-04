from __future__ import annotations

import functools
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
# Phase 60 Plan 3 (Bug 3.1): the previous pattern only allowed the
# ``cf./aff.`` qualifier AFTER the species epithet (``Genus sp. cf.``),
# but real radiolarian captions (Bandini 2011 pl08 / pl09) use the
# inverted shape where the qualifier + compared species come AFTER the
# epithet: ``Genus species cf. S. excelsa``. The canonical binomial is
# matched by group 1; a SEPARATE pattern ``TAXON_CF_COMPARE_PATTERN``
# below picks up the trailing comparison reference so the compared
# species epithet is not silently dropped.
# audit 2026-07-31: both tokens now require ≥3 letters so English
# phrase fragments like "An attempt of" / "Explanation of" can never
# match (2-letter words are never epithets; ICZN epithets are ≥3
# letters). The word filter in ``extract_taxa_from_caption`` applies
# the stopword / phrase-word lists on top.
TAXON_LIKE_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z-]{2,}\s+[a-z][a-zA-Z-]{2,}(?:\s+(?:sp\.|spp\.|cf\.|aff\.))?)\b"
)
# audit 2026-07-31: English function / phrase words that must never be
# accepted as genus or epithet tokens. "attempt", "explanation",
# "plateau", "figure" etc. are the real false positives seen in
# production output ("An attempt of" was shipped as a species).
_TAXON_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "are",
        "was",
        "were",
        "for",
        "with",
        "from",
        "this",
        "that",
        "their",
        "which",
        "shows",
        "show",
        "showing",
        "attempt",
        "attempts",
        "scale",
        "figure",
        "figures",
        "fig",
        "plate",
        "plates",
        "explanation",
        "explanationof",
        "specimen",
        "specimens",
        "upper",
        "lower",
        "middle",
        "early",
        "late",
        "part",
        "parts",
        "view",
        "views",
        "section",
        "sections",
        "sample",
        "samples",
        "locality",
        "localities",
        "age",
        "ages",
        "stage",
        "stages",
        "formation",
        "units",
        "unit",
        "sequence",
        "area",
        "areas",
        "time",
        "times",
        "reconstruction",
        "domain",
        "region",
        "regions",
        "basin",
        "basins",
        "succession",
        "interval",
        "intervals",
        "belt",
        "zone",
        "zones",
    }
)
# Phase 60 Plan 3 (Bug 3.1): ``cf. <Author>. <epithet>`` / ``aff. <Author>. <epithet>``
# — separate scan so the trailing comparison reference is preserved
# as a taxon epithet. Two shapes are common in radiolarian captions:
#
#   * ``cf. S. excelsa``     — single-letter author initial + epithet
#   * ``cf. Stichocapsa excelsa`` — full compared genus + epithet
#
# We capture the epithet only (group 2) so the ``S.`` author initial
# doesn't pollute the taxon list downstream — the epithet alone is
# the canonical ICZN signature of the compared species.
TAXON_CF_COMPARE_PATTERN = re.compile(
    r"\b(?:cf\.|aff\.)\s+"
    # either an author initial token (``S.``) OR a real genus name
    r"(?:[A-Z]\.|([A-Z][a-zA-Z-]+))\s+"
    r"([a-z][a-zA-Z-]+)"
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
                # Round 15 audit: weights_only=True blocks arbitrary code
                # execution via pickle deserialization (PyTorch 2.6+).
                # Fallback to legacy behaviour for old checkpoints that
                # contain pickled objects (e.g. optimiser state).
                try:
                    ckpt = torch.load(
                        checkpoint_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                except TypeError:
                    # Older PyTorch (<1.13) doesn't accept weights_only.
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
    # Audit 2026-09-04 taxon-2: normalize OCR character noise BEFORE
    # pattern matching, not on the extracted output —
    # ``TAXON_LIKE_PATTERN``'s character class is ASCII-only, so a
    # genus token carrying a stray digit ("Sponguru1") or a macron
    # ("Archaeodictyomitrā") never matched in the first place and
    # post-hoc normalization could not recover it. The digit-one rule
    # only fires after a letter ("Fig 1" / "sp. 1" keep their digit).
    from .ocr_corrections import _normalize_ocr_chars

    caption_text = _normalize_ocr_chars(caption_text)
    # audit 2026-07-31: strip the "Explanation of Plate N." header the
    # same way taxon.py does — "Explanation of" is not a binomial.
    from .taxon import TaxonRecognizer

    text = TaxonRecognizer._clean_caption_for_taxon(caption_text)
    taxa: list[str] = []
    for m in TAXON_LIKE_PATTERN.finditer(text):
        tax = m.group(1).strip()
        words = tax.split()
        if not words:
            continue
        # Word-boundary filter: neither the genus token nor the
        # epithet may be an English function/phrase word ("An
        # attempt of", "Explanation of", "The specimen").
        if words[0].lower().rstrip(".,;:?!") in _TAXON_STOP_WORDS:
            continue
        if len(words) > 1 and words[1].lower().rstrip(".,;:?!") in _TAXON_STOP_WORDS:
            continue
        if tax and tax not in taxa:
            taxa.append(tax)
    # Phase 60 Plan 3 (Bug 3.1): also surface trailing cf./aff.
    # comparison references so the compared species epithet isn't
    # silently dropped (e.g. Bandini 2011 pl08 / pl09 ``cf. S. excelsa``
    # used to disappear from the panel→species mapping).
    #
    # Audit 2026-09-01 BL-23: the previous code scanned the *original*
    # ``caption_text`` for cf./aff. comparison references even though
    # the canonical pattern matching loop above uses ``text`` (the
    # ``_clean_caption_for_taxon`` output) — so the regex could miss
    # the comparison epithet when the cleaner had stripped it. Use the
    # same ``text`` for both passes for audit consistency.
    for m in TAXON_CF_COMPARE_PATTERN.finditer(text):
        compared = m.group(2).strip() if m.group(2) else None
        if compared and compared not in taxa:
            taxa.append(compared)
    return taxa


def _label_sort_key(label: str) -> tuple[int, int, str]:
    """Sort panel labels like "1", "2", ..., "10", "A", "B".

    Pure-numeric labels sort by their integer value; alphabetic labels
    sort after numerics (alphabet labels are usually figure-level markers
    that come after numbered sub-panels).

    Audit 2026-09-01 BL-24: the previous implementation returned
    ``(0, "")`` for *every* numeric label, which under stable sort meant
    ``["10", "1", "2", "11"]`` sorted in *insertion* order rather than
    numeric order — panels 10/11 came *before* panels 1/2 in the
    caption-pair scan, mismatching the in-plate visual order and
    inflating image-verified F1 by counting off-by-one matches as
    positives. Encode the integer value into the sort tuple.

    Audit 2026-09-04 (CI regression fix): return a 3-tuple
    ``(rank, int_val, raw)`` so all pure-digit labels share rank 0
    but still sort by their integer value, while all alpha labels
    share rank 1 and sort lexicographically. Stable sort + the
    integer discriminator fixes the ``[9, 10, 1, 2]``-vs-correct-
    ``[1, 2, 9, 10]`` ordering bug without breaking the existing
    numeric-vs-alpha separation.
    """
    s = str(label).strip()
    if s.isdigit():
        try:
            return (0, int(s), "")
        except ValueError:
            return (0, 0, "")
    try:
        return (1, 0, s)
    except Exception:
        return (2, 0, s)


@functools.lru_cache(maxsize=8192)
def _normalize_panel_label(label: str | None) -> str | None:
    """Normalise a panel label so OCR misreads don't break caption lookup.

    Audit 2026-09-01 (PERF-9): wrapped in ``lru_cache`` because the
    same panel labels recur across every panel / figure / paper
    (typical paper: 200 panels × 5 candidate labels = 1000 calls).
    Cache size 8192 covers a 4000-panel paper with the typical
    dup-ratio. The function is pure (input → output) so caching is
    safe.

    Three normalisations:
      1. Strip leading zeros ("00" → "0", "04" → "4") — PaddleOCR commonly
         reads "3" as "03" or "0" as "00" when the glyph is small.
      2. Strip trailing zero-padding for double-digit OCR ("30" misread
         of "3" stays as "3" if "3" is in the pair_lookup). The
         caller decides whether to keep or drop by trying both forms.
      3. Phase 62 Plan 5 (Bug 5.10): recover from digit+letter+single-
         trailing-digit OCR misreads ("3a0" → "3a", "12b5" → "12b").
         A trailing single digit on an otherwise-valid digit+letter
         label is almost always OCR noise (the next character's
         glyph has bled into the OCR window); strip it. Multi-digit
         trailing ("3a00") is left as-is — too ambiguous to recover.

    Returns the cleanest single label (or None for empty input).
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    # Phase 62 Plan 5 (Bug 5.10): digit + letter + single trailing
    # digit OCR misread → strip the trailing digit. We require the
    # base (without trailing digit) to look like a valid panel
    # label shape so we never silently produce garbage.
    #
    # Audit 2026-09-01 BL-25: the existing fallback only handled
    # ``<digit><letter><digit>`` (3 chars, e.g. "A04"). Real OCR
    # produces 4-character shapes like "A04", "B07", "10A" (where
    # the trailing digit can be paired with multiple letters/digits).
    # Generalise to "strip trailing digit iff the stripped base is a
    # valid panel label".
    while len(s) >= 3 and s[-1].isdigit():
        candidate = s[:-1]
        # Only strip if the candidate shape is itself a panel label
        # we recognise (pure digits OR digits+letters OR letter+digit
        # OR letter+letter). Otherwise we'd mangle e.g. "A04b".
        if candidate.isdigit() or candidate.isalpha() or _LOOKS_LIKE_LABEL(candidate):
            s = candidate
        else:
            break
    # Don't normalise alphabetic labels ("A", "B" stay as-is).
    if not s.isdigit():
        return s
    return str(int(s))  # "00" → "0", "04" → "4", "3" → "3"


# Helper for BL-25 — ``_normalize_panel_label`` stripping loop. A
# string "looks like" a panel label if it matches the same shape regex
# used by ``is_valid_panel_label``: pure letter / pure digit /
# digit+letter / letter+digit. We deliberately keep this minimal so
# the loop terminates quickly on garbage input.
_LABEL_SHAPE_FOR_STRIP = re.compile(r"^[A-Za-z]?[A-Za-z0-9]{0,3}$")


def _LOOKS_LIKE_LABEL(s: str) -> bool:
    return bool(_LABEL_SHAPE_FOR_STRIP.match(s))


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
    # Shape gate: A-H marker OR 1-3 digits with optional single trailing
    # lowercase letter. Anything else is OCR noise / caption fragment.
    # Capped at 3 digits because real plates don't carry > 50 panels
    # and OCR often mistakes scale-bar numbers like "100 µm", "200 µm"
    # for panel ids if the " µm" suffix is dropped by the OCR engine
    # (Feng_2006 audit: scale-bar text "100" → 5-digit panel_id "86500"
    # leaked through when the digit pattern had no length cap).
    return bool(_PANEL_LABEL_SHAPE.match(s))


# Compiled once at import time. Anchored fullmatch is enforced by
# using ``re.match`` with explicit ^/$ in the pattern below.
# Audit Bug 2: reject leading zeros (e.g. "007", "04") so the
# function's contract is self-consistent even when called on a
# label that has NOT been pre-normalized. ``[1-9]\d*`` matches
# "1", "12", "123" but not "0", "00", "007". The single "0"
# case is handled by the digit+letter optional: panel_id="0"
# is normalized to "0" by _normalize_panel_label and is a
# legitimate panel index in some papers, so we accept it here.
#
# Audit 2026-09-01 BL-26: the previous regex ``[1-9]\d{0,2}[a-z]?``
# accepted "10a" but rejected "0a" — even though some Triassic
# papers use "0a" / "0b" for transitional panels at the top of a
# plate. Add an explicit ``0[a-z]?`` branch so the validation shape
# is consistent for panel 0 across papers.
_PANEL_LABEL_SHAPE = re.compile(r"^(?:[A-H]|[1-9]\d{0,2}[a-z]?|0[a-z]?|0)$")


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
            iou = _iou(panel.bbox, k.bbox)
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


def _iou(a, b) -> float:
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
        # Audit 2026-09-01 BL-27: the previous code only consulted
        # ``labels`` (caption-derived) when ``panel.panel_id`` was
        # invalid; if BOTH the panel_id and the caption label were
        # invalid, the loop emitted None — silently dropping the
        # real panel from the pred row set even though ``ocr_tokens``
        # may have a perfectly valid label token for the panel's
        # bbox. Add an OCR-token fallback: if any ``ocr_tokens`` entry
        # with a valid label shape lies inside the panel bbox, use
        # that. This recovers panels that the OCR pipeline correctly
        # identified but the caption parser mis-shaped.
        _panel_bbox = getattr(panel, "bbox", None)
        if _panel_bbox is not None and ocr_tokens:
            matched_cand: str | None = None
            for tok in ocr_tokens:
                ttext = (tok.text or "").strip()
                if not ttext:
                    continue
                cand = _normalize_panel_label(ttext) or ttext
                if cand and is_valid_panel_label(cand) and _token_in_panel(tok, panel):
                    matched_cand = cand
                    break
            if matched_cand is not None:
                out.append(matched_cand)
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
    # Audit 2026-09-01 (architectural P0 #25): the previous short-circuit
    # ``taxa = [...] or ...`` only used the caption-extraction branch when
    # entity extraction happened to return empty — the two sources never
    # merged. This made ``taxa`` non-deterministic across runs (entity
    # detection slightly varies run-to-run, but caption parsing is
    # stable), breaking cross-paper evaluation reproducibility. Always run
    # both branches and union with deterministic precedence (entity-first,
    # caption-supplement, dedup by lower-cased full text).
    _ent_taxa = [t.text for t in taxon_entities]
    _cap_taxa = extract_taxa_from_caption(caption.caption)
    _seen: set[str] = set()
    taxa: list[str] = []
    for _src in (_ent_taxa, _cap_taxa):
        for _t in _src:
            _key = _t.strip().lower()
            if not _key or _key in _seen:
                continue
            _seen.add(_key)
            taxa.append(_t)
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
            # Audit P1-4: when panel_id is OCR garbage (e.g. single-letter
            # "S/A/L"), still emit a MatchResult so no panel is silently
            # dropped.  Use panel_id=None so the caller can distinguish
            # "no label detected" from "valid label".  The main loop
            # (line 831) emits MatchResult for ALL panels regardless of
            # label validity — the placeholder branch must match that
            # behaviour to avoid losing panel data.
            if panel_id is not None and not is_valid_panel_label(panel_id):
                panel_id = None
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
                    # Phase 55 audit fix: propagate panel.panel_index so the
                    # published pipeline_panel_index is no longer permanently
                    # None. getattr guards against PanelCandidates that never
                    # had panel_index set (e.g. callers other than the
                    # classical pipeline path).
                    panel_index=getattr(panel, "panel_index", None),
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
        except Exception as exc:
            logger.debug("caption regex fallback failed: %s", exc)

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
        # Audit 2026-09-01 (architectural P0 #7): previously the condition
        # was only ``caption_pairs_used`` — meaning whenever M3 returned
        # ANY caption-pair (even one with low confidence), it would
        # overwrite the neural-matcher's result. This capped the trained
        # matcher at the regex/M3 pair_lookup ceiling, turning the neural
        # head into dead code. Now require BOTH ``caption_pairs_used`` AND
        # ``not matcher_used`` so the neural matcher is honoured when it
        # ran; the pair_lookup only fills gaps where the matcher did not.
        if caption_pairs_used and not matcher_used:
            matched_key = _label_in_pair_lookup(panel_id, pair_lookup)
            if matched_key:
                best_species = pair_lookup[matched_key]
        ocr_text = " ".join(tok.text for tok in ocr_tokens if _token_in_panel(tok, panel))
        label_text = None
        if panel_id and panel_id in panel_label_tokens:
            label_text = panel_label_tokens[panel_id].text
        confidence = float(panel.score)
        if matcher_used:
            # Phase 54 audit m12: neural-matcher branch was missing the
            # min(0.99, ...) clamp that the rule-based branch applies,
            # so a perfect (1.0) neural score leaked through and
            # polluted downstream filtering. Mirror the rule branch.
            confidence = min(
                0.99,
                max(confidence, float(neural_conf[idx]) if idx < len(neural_conf) else 0.0),
            )
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
                # Phase 55 audit fix: propagate panel.panel_index so
                # pipeline_panel_index is no longer permanently None.
                panel_index=getattr(panel, "panel_index", None),
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
                # Phase 55 audit fix: no PanelCandidate exists in this path
                # (it is a pure fallback when match_panels produced no
                # results), so panel_index stays None — but the field must
                # still be present so MatchResult.to_dict() is consistent.
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
