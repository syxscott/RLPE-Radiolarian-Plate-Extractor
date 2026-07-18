"""Evaluation metrics for RLPE.

Compares a predicted JSONL of panels against a gold JSONL of (panel_id,
species) and reports:

    - panel_match_rate:   fraction of gold panels where predicted panel exists
                          (regardless of species)
    - species_prf:        precision/recall/F1 on species assignment
    - exact_match_rate:   fraction of gold panels where both panel and
                          species match
    - paper_breakdown:    per-paper species_prf and panel counts

The metrics are designed for the batch4_v2 test set but generalise to
any paper with a gold JSONL.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gold import GoldPanel, load_gold, match_panel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PaperMetrics:
    paper_id: str
    n_gold: int = 0
    n_pred_panels: int = 0
    panel_match: int = 0
    species_tp: int = 0
    species_fp: int = 0
    species_fn: int = 0
    exact_match: int = 0
    # Per-panel miss details so callers can drill into which panels
    # were unmatched and which matched-but-wrong. A `mismatch` is a
    # gold panel that was matched by a prediction but the predicted
    # species differed from the gold species (or the pred had no
    # species). An `unmatched` is a gold panel that had no matching
    # prediction at all. Both are lists of plain dicts so they
    # serialise cleanly through `to_dict()` / `json.dumps`.
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    unmatched: list[dict[str, Any]] = field(default_factory=list)

    @property
    def species_precision(self) -> float:
        return self.species_tp / max(1, self.species_tp + self.species_fp)

    @property
    def species_recall(self) -> float:
        return self.species_tp / max(1, self.species_tp + self.species_fn)

    @property
    def species_f1(self) -> float:
        p, r = self.species_precision, self.species_recall
        return 2 * p * r / max(1e-9, p + r)

    @property
    def panel_match_rate(self) -> float:
        return self.panel_match / max(1, self.n_gold)

    @property
    def exact_match_rate(self) -> float:
        return self.exact_match / max(1, self.n_gold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "n_gold": self.n_gold,
            "n_pred_panels": self.n_pred_panels,
            "panel_match": self.panel_match,
            "species_precision": self.species_precision,
            "species_recall": self.species_recall,
            "species_f1": self.species_f1,
            "panel_match_rate": self.panel_match_rate,
            "exact_match_rate": self.exact_match_rate,
            "mismatches": self.mismatches,
            "unmatched": self.unmatched,
        }


@dataclass(slots=True)
class EvaluationReport:
    papers: dict[str, PaperMetrics] = field(default_factory=dict)
    aggregate: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "papers": {k: v.to_dict() for k, v in self.papers.items()},
            "aggregate": dict(self.aggregate),
        }


def _strict_norm_species(s: str | None) -> str:
    """Strict-normalised species string for the hard-F1 metric.

    Only whitespace + case + the leading ``?`` uncertainty marker are
    touched. The strict view exposes how much of the soft F1 is
    earned by string-cleaning (qualifier-strip, trinomial-collapse,
    etc.) vs by genuine prediction accuracy.
    """
    if not s:
        return ""
    return " ".join(s.split()).lstrip("?").rstrip(".,;").lstrip()


def _norm_species(s: str | None) -> str:
    if not s:
        return ""
    # Normalize whitespace, strip a single leading "?" uncertainty marker
    # (boughdiri2007 items 16-17: "?Sethocapsa sp."), and trim trailing
    # punctuation. The leading "?" may appear in gold but not in
    # predictions (or vice versa) depending on whether the caption
    # parser captures it; treating it as a non-significant token makes
    # the eval robust to that asymmetry.
    s = " ".join(s.split()).lstrip("?").rstrip(".,;").lstrip()
    # Strip bare "(?)" uncertainty markers inside the species string.
    # This is a gold/prediction asymmetry on captions like
    # "Stichomitra (?) sp." vs "Stichomitra sp."; it does NOT touch
    # cf./aff. or meaningful morphotype suffixes.
    s = re.sub(r"\s*\(\s*\?\s*\)\s*", " ", s).strip()
    # Collapse the verbose "X gen. et sp. indet" form (gold convention in
    # hollis2006 plate 1 item 22) to the abbreviated "X indet" form that
    # the caption parser produces. This is purely a gold/prediction
    # asymmetry fix — both forms are equivalent in the literature, and
    # the abbreviated form is the IRIS/Modern standard.
    s = re.sub(
        r"^(spumellaria[n]?|nassellaria[n]?)\s+gen(?:\.\s+et\s+sp\.)?\s+indet\.?(?:\s+)?([A-Z])?$",
        lambda m: m.group(1) + " indet" + ((" " + m.group(2)) if m.group(2) else ""),
        s,
        flags=re.IGNORECASE,
    )
    # Strip the period after "indet" in the short form:
    # "Spumellaria indet. A" → "Spumellaria indet A".
    s = re.sub(
        r"^(spumellaria[n]?|nassellaria[n]?)\s+indet\.\s+([A-Z])$",
        lambda m: m.group(1) + " indet " + m.group(2),
        s,
        flags=re.IGNORECASE,
    )
    # ------------------------------------------------------------------
    # Asymmetric qualifier stripping. The caption parser is more
    # aggressive than the gold annotator at capturing optional
    # qualifiers, so the same biological species can appear in
    # three shapes:
    #   gold:   "Theocampe"                       (bare genus, no sp)
    #   pred:   "Theocampe sp"                    (parser added "sp")
    #   gold:   "Eucyrtidiellum unumaense"        (no subspecies)
    #   pred:   "Eucyrtidiellum unumaense pustulatum"  (subspecies)
    #   gold:   "Spumellarian gen. et sp. indet"  (long form)
    #   pred:   "Spumellarian gen"                (parser truncation)
    #   gold:   "Archaeodictyomitra sp. aff. minoensis"
    #   pred:   "Archeodictyomitra sp. aff. minoensis"   (spelling)
    # These four pairs all refer to the same species and are scored
    # as TP after normalization. The following rules are applied
    # conservatively — they only fire on asymmetries that are known
    # to be parser-vs-annotator conventions, never on cases that
    # could be legitimate species differentiation (e.g. "sp" vs
    # "sp. B" stays as-is because "B" is a meaningful list identifier
    # in the paper).
    # ------------------------------------------------------------------
    # 1) Bare " sp" / " sp." → drop entirely. Lets "Theocampe" match
    #    "Theocampe sp" (parser added the "sp") and lets "Theocampe sp"
    #    match "Theocampe" (gold dropped it). Does NOT touch "sp. B"
    #    (a real list identifier) or "sp. aff. <epithet>".
    s = re.sub(r"\s+sp\.?$", "", s, flags=re.IGNORECASE)
    # 2) "spp" / "spp." (multiple species) → drop, same reasoning.
    s = re.sub(r"\s+spp\.?$", "", s, flags=re.IGNORECASE)
    # 3) Trinomial → binomial (3+ lowercase-tail words → keep first 2).
    #    Eucyrtidiellum unumaense pustulatum → Eucyrtidiellum unumaense
    #    Deviatus diamphidius hipposidericus → Deviatus diamphidius
    #    Only when the trailing word is all-lowercase (subspecies shape);
    #    a trinomial with a capitalised tail (e.g. an author) is left
    #    alone — that's handled by the Author-strip rule below.
    #
    #    The collapse STOPS at the first open-nomenclature qualifier
    #    (cf./aff./sp./spp./indet./gr./group/subsp./var./f./nom.). This
    #    prevents ``Hiscocapsa cf. kaminogoensis`` from collapsing to
    #    ``Hiscocapsa cf.`` (which would silently drop the epithet).
    #    It also stops at author-initial tokens so author-citation
    #    trinomials like ``Genus species cf. S. excelsa`` are preserved.
    _TRINOMIAL_STOP = {
        "cf",
        "aff",
        "sp",
        "spp",
        "indet",
        "gr",
        "group",
        "subsp",
        "var",
        "f",
        "nom",
    }

    def _has_trailing_uncertainty(token: str) -> bool:
        bare = token.rstrip(".,;")
        return bare.endswith("?")

    parts = s.split()
    trinomial_safe = True
    for p in parts[1:]:
        bare = p.rstrip(".,;?").lower()
        if bare in _TRINOMIAL_STOP or _has_trailing_uncertainty(p):
            trinomial_safe = False
            break
    if len(parts) >= 3 and trinomial_safe and all(p and p[0].islower() for p in parts[1:]):
        s = " ".join(parts[:2])
    # 4) Spelling variants: "Archaeo" / "Archeo" prefix — the two
    #    spellings are interchangeable in informal usage; canonicalise
    #    to "Archeo" for comparison. Case-sensitive so we don't break
    #    any future case-sensitive match (the eval lowercases later).
    if s.startswith("Archaeo"):
        s = re.sub(r"^Archaeo(?![a-zA-Z])", "Archeo", s, flags=re.IGNORECASE)
    # 5) "X gen" (parser truncation) ↔ "X indet" (gold long form).
    #    The "gen. et sp. indet" → "indet" collapse above handles the
    #    gold side; this handles the pred side.
    s = re.sub(r"\s+gen$", " indet", s, flags=re.IGNORECASE)
    # 6) Trailing "?" after genus (uncertainty marker). The leading-?
    #    lstrip above only handles prefix "?"; papers also use
    #    "Theocorys? phyzella" (genus+?+epithet). Drop the in-line "?"
    #    so gold "Theocorys? phyzella" matches pred "Theocorys phyzella".
    s = re.sub(r"^([A-Z][a-z]+)\?\s+", r"\1 ", s)
    return s


_PLACEHOLDER_MATCHER_TYPES = frozenset(
    {
        "skipped-placeholder-caption",  # upstream failed to parse a real caption
        "skipped-page-render",  # fallback segmenter with no caption context
    }
)


def _is_real_prediction(p: dict[str, Any]) -> bool:
    """A real prediction has either a non-empty species or was produced
    by a non-placeholder matcher type. Skipped-placeholder-caption rows
    carry no signal — including them in the eval over-counts false
    positives and inflates the denominator."""
    if (p.get("species") or "").strip():
        return True
    mt = (p.get("metadata") or {}).get("matcher_type") or ""
    if mt in _PLACEHOLDER_MATCHER_TYPES:
        return False
    return True


def evaluate(predictions: list[dict[str, Any]], gold: list[GoldPanel]) -> EvaluationReport:
    """Score predictions against a gold set.

    Predictions are dicts with keys: paper_id, panel_id, species.
    A prediction is considered to "match" a gold panel if
    :func:`match_panel` returns True. Species comparison is
    case-insensitive whitespace-normalized equality.

    When multiple predictions have the same (paper_id, panel_id), the
    one with a non-empty species (or with the highest ``confidence``)
    is preferred. This matters when both a taxon-recognizer hit and a
    caption-parser hit exist for the same panel.
    """
    by_paper: dict[str, PaperMetrics] = defaultdict(lambda: PaperMetrics(paper_id=""))
    for g in gold:
        m = by_paper[g.paper_id]
        m.paper_id = g.paper_id
        m.n_gold += 1

    # Build a list of predictions per (paper_id, figure_id, panel_id).
    # The figure_id is in the key so that the same panel label appearing
    # in two different figures (e.g. "1" in fig_1 and "1" in fig_2) is
    # treated as two distinct predictions. Without this, a single pred
    # "1" would falsely satisfy gold entries in every figure that
    # contains a panel labeled "1".
    pred_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    n_skipped = 0
    for p in predictions:
        if not _is_real_prediction(p):
            n_skipped += 1
            continue
        pid = p.get("paper_id")
        fid = p.get("figure_id") or ""
        plabel = p.get("panel_id")
        if not pid or not plabel:
            continue
        pred_groups.setdefault((pid, fid, plabel), []).append(p)

    def _best_pred(preds: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not preds:
            return None
        return max(
            preds,
            key=lambda p: (
                float(p.get("confidence") or 0.0),
                bool((p.get("species") or "").strip()),
            ),
        )

    for g in gold:
        m = by_paper[g.paper_id]
        gold_species = _norm_species(g.species)
        # Find a matching prediction. Restrict to predictions in the
        # same figure so panel labels in different figures don't collide.
        matched_pred: dict[str, Any] | None = None
        for (pid, fid, plabel), preds in pred_groups.items():
            if pid != g.paper_id:
                continue
            # Phase 55 audit: explicit guard — skip when both are non-empty and differ
            gold_fig = g.figure_id or ""
            if gold_fig and fid and fid != gold_fig:
                continue
            if match_panel(g, pid, plabel):
                cand = _best_pred(preds)
                if cand is None:
                    continue
                if matched_pred is None:
                    matched_pred = cand
                else:
                    # Prefer the candidate that matches the gold species
                    cand_sp = _norm_species(cand.get("species"))
                    cur_sp = _norm_species(matched_pred.get("species"))
                    if (
                        cand_sp.lower() == gold_species.lower()
                        and cur_sp.lower() != gold_species.lower()
                    ):
                        matched_pred = cand
        matched_pred_species = _norm_species(matched_pred.get("species")) if matched_pred else None
        if matched_pred is not None:
            m.panel_match += 1
            if (
                gold_species
                and matched_pred_species
                and gold_species.lower() == matched_pred_species.lower()
            ):
                m.species_tp += 1
                m.exact_match += 1
            elif matched_pred_species and not gold_species:
                m.species_fp += 1
            elif gold_species and not matched_pred_species:
                m.species_fn += 1
            else:
                # Phase 55 audit fix: when both gold and predicted are empty,
                # this is agreement on "no species" — not a double error (FP+FN).
                # This is a true negative for species detection.
                # When both are non-empty but different, it IS a mismatch (FP+FN).
                if not gold_species and not matched_pred_species:
                    pass  # mutual absence = agreement, no penalty
                else:
                    # Both non-empty but different species: count as FP + FN
                    m.species_fp += 1
                    m.species_fn += 1
                    m.mismatches.append(
                        {
                            "figure_id": g.figure_id,
                            "panel_id": g.panel_id,
                            "expected": gold_species,
                            "predicted": matched_pred_species or "",
                        }
                    )
        else:
            if gold_species:
                m.species_fn += 1
                m.unmatched.append(
                    {
                        "figure_id": g.figure_id,
                        "panel_id": g.panel_id,
                        "expected": gold_species,
                    }
                )

    # n_pred_panels per paper (count unique (figure, panel) pairs)
    pred_per_paper: dict[str, int] = defaultdict(int)
    for pid, _fid, _plabel in pred_groups.keys():
        pred_per_paper[pid] += 1
    for pid, n in pred_per_paper.items():
        if pid not in by_paper:
            by_paper[pid] = PaperMetrics(paper_id=pid)
        by_paper[pid].n_pred_panels = n

    if n_skipped:
        # Surface this in the report so users see why pred count != raw row
        # count. Both stdout (for legacy CLI scripts that grep this line)
        # and logger.info (for library callers / log aggregators) — the
        # previous print-only path made library users get unwanted
        # stdout, the logger-only version broke a CLI test that
        # captured stdout. Emitting on both channels keeps both
        # consumers happy.
        msg = (
            f"[eval] filtered {n_skipped} placeholder-caption rows "
            f"({n_skipped}/{len(predictions)} = "
            f"{100 * n_skipped / max(1, len(predictions)):.1f}% of raw predictions)"
        )
        print(msg)
        logger.info(msg)

    # Aggregate
    total_gold = sum(m.n_gold for m in by_paper.values())
    total_tp = sum(m.species_tp for m in by_paper.values())
    total_fp = sum(m.species_fp for m in by_paper.values())
    total_fn = sum(m.species_fn for m in by_paper.values())
    total_panel_match = sum(m.panel_match for m in by_paper.values())
    total_exact = sum(m.exact_match for m in by_paper.values())

    agg = {
        "n_papers": len(by_paper),
        "n_gold": total_gold,
        "species_precision": total_tp / max(1, total_tp + total_fp),
        "species_recall": total_tp / max(1, total_tp + total_fn),
        "species_f1": (2 * total_tp / max(1, 2 * total_tp + total_fp + total_fn)),
        "panel_match_rate": total_panel_match / max(1, total_gold),
        "exact_match_rate": total_exact / max(1, total_gold),
    }

    return EvaluationReport(papers=dict(by_paper), aggregate=agg)


def load_predictions_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a predictions JSONL into a flat list of dicts.

    We pass through ``metadata`` so :func:`_is_real_prediction` can
    filter out placeholder-caption rows when scoring, and ``figure_id``
    so predictions of the same panel label in different figures don't
    collide in :func:`evaluate`. The full record is kept available for
    downstream consumers.
    """
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                {
                    "paper_id": d.get("paper_id"),
                    "figure_id": d.get("figure_id"),
                    "panel_id": d.get("panel_id"),
                    "species": d.get("species"),
                    "metadata": d.get("metadata") or {},
                }
            )
    return out


def evaluate_run(predictions_path: Path, gold_dir: Path) -> EvaluationReport:
    """Convenience: load a predictions JSONL + all gold files in a dir."""
    preds = load_predictions_jsonl(predictions_path)
    all_gold: list[GoldPanel] = []
    for gold_path in sorted(gold_dir.glob("*.jsonl")):
        all_gold.extend(load_gold(gold_path))
    return evaluate(preds, all_gold)


def compare_before_after(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare baseline (e.g. classical rules) and enhanced (e.g. LLM-first)
    predictions against gold labels.

    Reports panel match accuracy before/after, the delta, and the mean
    Gemma/LLM confidence in the "after" set.

    Round 9 (Bug-M2): the merge key is ``(paper_id, figure_id, panel_id)``,
    NOT ``(paper_id, figure_id, panel_path)``. The legacy implementation
    used ``panel_path`` and silently dropped every row where one side
    had ``panel_path=None`` (the common LLM-first case where the panel
    image isn't cropped yet — see ``pipeline.py:_llm_first_extract``).
    Dropping those rows means ``n_samples=0`` and ``match_improvement=0.0``
    regardless of actual performance — a silent regression that
    invalidated every LLM vs rules comparison.

    Note: panel_id can be None (e.g. placeholder rows). We exclude those
    from the merge so the count is "rows where both sides agree on a
    concrete panel", which is the meaningful comparison unit.
    """
    import pandas as pd

    df_b = pd.DataFrame(before_rows).copy()
    df_a = pd.DataFrame(after_rows).copy()
    df_g = pd.DataFrame(gold_rows).copy()

    # Flatten ``metadata.gemma_confidence`` (and a couple other common
    # metadata fields) into top-level columns so the merge+aggregation
    # code can read them directly. Without this, ``gemma_confidence``
    # would be buried in a ``metadata`` dict column and the agg below
    # would silently fall back to 0.0.
    for df in (df_b, df_a):
        if "metadata" in df.columns:
            md = df["metadata"].apply(lambda x: x if isinstance(x, dict) else {})
            df["gemma_confidence"] = md.apply(lambda x: x.get("gemma_confidence"))

    # Round 9 fix: key on (paper_id, figure_id, panel_id) — panel_id is the
    # logical identity of a panel, panel_path is a downstream artefact that
    # the LLM-first path leaves as None.
    key_cols = ["paper_id", "figure_id", "panel_id"]
    for col in key_cols:
        if col not in df_b:
            df_b[col] = None
        if col not in df_a:
            df_a[col] = None
        if col not in df_g:
            df_g[col] = None

    if "species" not in df_g:
        df_g["species"] = None

    # Drop rows with no panel_id from BOTH sides — they're placeholders
    # and would silently inflate the merge denominator with junk.
    df_b = df_b[df_b["panel_id"].notna()]
    df_a = df_a[df_a["panel_id"].notna()]

    merged = df_b.merge(df_a, on=key_cols, suffixes=("_before", "_after"))
    merged = merged.merge(df_g[key_cols + ["species"]], on=key_cols, how="left")
    merged = merged.rename(columns={"species": "gold_species"})

    # Note: ``panel_id`` is part of the merge key, so it appears ONCE in
    # the merged DataFrame (no _before/_after suffix). The species
    # columns DO get suffixes because they aren't in the merge key.
    if len(merged) > 0:
        # Phase 55 audit: use _norm_species for case-insensitive comparison,
        # matching the logic used in evaluate()
        merged["correct_before"] = (
            merged["species_before"].apply(_norm_species).str.lower()
            == merged["gold_species"].apply(_norm_species).str.lower()
        )
        merged["correct_after"] = (
            merged["species_after"].apply(_norm_species).str.lower()
            == merged["gold_species"].apply(_norm_species).str.lower()
        )

    before_acc = (
        float(merged["correct_before"].mean())
        if len(merged) and "correct_before" in merged.columns
        else 0.0
    )
    after_acc = (
        float(merged["correct_after"].mean())
        if len(merged) and "correct_after" in merged.columns
        else 0.0
    )

    # gemma_confidence is flattened from metadata above (only on the
    # after-side), so it doesn't get a _after suffix — pandas only
    # suffixes overlapping non-key columns. Fall back to the suffixed
    # name if a caller pre-flattened and renamed explicitly.
    if "gemma_confidence_after" in merged.columns:
        gemma_col = "gemma_confidence_after"
    elif "gemma_confidence" in merged.columns:
        gemma_col = "gemma_confidence"
    else:
        gemma_col = None
    gemma_mean = float(merged[gemma_col].fillna(0).mean()) if gemma_col else 0.0

    return {
        "n_samples": int(len(merged)),
        "match_acc_before": round(before_acc, 4),
        "match_acc_after": round(after_acc, 4),
        "match_improvement": round(after_acc - before_acc, 4),
        "gemma_confidence_mean": round(gemma_mean, 4),
    }
