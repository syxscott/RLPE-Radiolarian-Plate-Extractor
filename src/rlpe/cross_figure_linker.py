"""Cross-figure linker: link plate species to paper's strat column / map.

Phase 65 Plan A.2: implements a 3-strategy linker that produces one
``LinkResult`` per panel, encoding which paper-level geology fact
(strat column / litholog / paleogeographic map) the panel's species
most likely came from. The strategies, in order of confidence:

1. **Sample ID direct match** (``confidence=1.0``,
   ``source="sample_match"``) — regex-extract sample / loc / ID tokens
   from the panel's caption and look them up in the paper's
   strat-column / litholog / paleogeographic-map figures. A hit
   directly links the panel to that figure's geology context.
2. **Locality string share** (``confidence=0.7``,
   ``source="locality_match"``) — same paper, same locality string
   (``Tunisia``, ``Greece``, ``Sicily``, ``NW Turkey``, …) shared
   between the plate caption and any paper-level geo figure.
3. **M3 cross-figure inference** (``confidence=0.3-0.6``,
   ``source="m3_inference"``) — for unlinked plates, send paper
   figure summary + plate caption to M3 and let it infer the most
   likely formation / age. Implemented as an optional callback so
   tests can use a deterministic FakeM3Backend.

If all 3 strategies fail the panel gets a single ``LinkResult`` with
``source="unlinked"`` and ``confidence=0.0`` (so the export pipeline
still writes the row with a source tag, just an honest "we have no
idea" marker).

Public API
----------
* :func:`link_species_to_geology(panels, paper_figures, m3_engine=None)`
  — list of panels + paper-level figure summaries → list of
  ``LinkResult``.
* :class:`LinkResult` — dataclass result.

The module is pure: no I/O, no logging, no side effects. The caller is
responsible for writing the ``LinkResult`` into the panel's
``metadata.geology_links`` (with the ``source`` field propagated into
``coord_source`` or a new dedicated ``link_source`` field — see the
schema below).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from .cross_refs import CrossRef, parse_cross_refs
from .sample_id_extractor import (
    SampleID,
    _LOCALITY_BLOCKLIST,
    extract_age_terms,
    extract_locality,
    extract_sample_ids,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

LINK_SOURCE_SAMPLE = "sample_match"
LINK_SOURCE_LOCALITY = "locality_match"
LINK_SOURCE_M3 = "m3_inference"
LINK_SOURCE_CROSS_REF = "cross_ref"
LINK_SOURCE_UNLINKED = "unlinked"


@dataclass(frozen=True, slots=True)
class LinkResult:
    """One cross-figure linkage between a panel and a paper-level figure.

    Attributes
    ----------
    panel_id : str | None
        The panel id (``MatchResult.panel_id``). ``None`` for figure-
        level fallback links.
    species : str | None
        The panel's species name (or ``None`` for unlinked).
    figure_id : str | None
        The linked paper-level figure id (strat column / litholog /
        map / range chart). ``None`` when ``source == "unlinked"``.
    formation : str | None
        Formation name inferred from the linked figure (verbatim).
    age : str | None
        Age / chronostratigraphy string (verbatim).
    locality : str | None
        Locality string (verbatim).
    confidence : float
        Link confidence in ``[0.0, 1.0]``. ``1.0`` for sample ID hits,
        ``0.7`` for locality hits, ``0.3-0.6`` for M3 inferences,
        ``0.0`` for the unlinked fallback.
    source : str
        One of ``"sample_match"``, ``"locality_match"``,
        ``"m3_inference"``, ``"unlinked"``.
    evidence : str
        Human-readable evidence string (the matched sample id, locality,
        or M3 prompt summary) for audit / debugging.
    """

    panel_id: str | None
    species: str | None
    figure_id: str | None
    formation: str | None
    age: str | None
    locality: str | None
    confidence: float
    source: str
    evidence: str = ""


# ---------------------------------------------------------------------------
# M3 callback protocol
# ---------------------------------------------------------------------------

class M3InferenceCallable(Protocol):
    """Protocol for the M3 cross-figure inference callable.

    Tests use a ``FakeM3Backend``-backed stub; production code passes
    ``m3_engine.infer_species_age_formation``.
    """

    def __call__(
        self,
        panel_caption: str,
        paper_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Paper figure shape (loose dict so callers can pass Phase B FigureRecord or raw dict)
# ---------------------------------------------------------------------------

PaperFigureLike = Mapping[str, Any]


def _figure_caption(figure: PaperFigureLike) -> str:
    """Best-effort caption string from a paper figure dict."""
    for key in ("caption", "section_title", "evidence_text"):
        val = figure.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _figure_formation(figure: PaperFigureLike) -> str | None:
    """Best-effort formation string."""
    for key in ("formation", "group", "member"):
        val = figure.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _figure_age(figure: PaperFigureLike) -> str | None:
    """Best-effort age / chronostratigraphy string."""
    for key in ("age", "chronostratigraphy", "biozone"):
        val = figure.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _figure_locality(figure: PaperFigureLike) -> str | None:
    """Best-effort locality string."""
    for key in ("locality", "country", "section_title"):
        val = figure.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _figure_type(figure: PaperFigureLike) -> str:
    """Return the figure_type field, lower-cased."""
    val = figure.get("figure_type") or figure.get("type") or ""
    return str(val).lower()


# Display-number suffix of a figure, e.g. "3" for "Fig. 3" / "Pl. 3".
# Strategy 4 (cross_refs) needs this to map a caption mention like
# "see Fig. 3" back to the underlying figure dict. We prefer the
# pipeline-stamped ``figure_number`` / ``figure_num`` field, then fall
# back to scraping the figure_id string ("fig_3" / "od_plate_..._pl03"
# → "3"), then to scanning the caption.
_FIG_NUM_FROM_ID = re.compile(r"(?:\b|_)(?:fig|pl|plate)?_?(\d+)(?:\b|$)")


def _extract_figure_number(
    figure: PaperFigureLike, *, fid: str = "", ftype: str = ""
) -> str | None:
    """Return the display number of a figure, e.g. "3" for "Fig. 3".

    Looks at three sources in order of preference:

    1. ``figure.figure_number`` / ``figure.figure_num`` (pipeline-stamped)
    2. Regex extraction from the figure_id string
    3. Regex extraction from the caption (delegated to layout)
    """
    # 1. Explicit field
    for key in ("figure_number", "figure_num"):
        val = figure.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # 2. From figure_id
    if fid:
        m = _FIG_NUM_FROM_ID.search(fid)
        if m:
            return m.group(1)
    # 3. From caption (avoid heavy layout import; use inline regex)
    cap = _figure_caption(figure)
    if cap:
        m = re.search(
            r"\b(?:Fig(?:ure)?|Pl(?:ate)?)\s*\.?\s*(\d+[A-Za-z]?)",
            cap,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Index paper figures for fast lookup
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _FigureIndex:
    """Indexed view over the paper's strat/map/litholog/range figures."""

    sample_id_to_figure: dict[str, PaperFigureLike] = field(default_factory=dict)
    locality_to_figure: dict[str, PaperFigureLike] = field(default_factory=dict)
    summary: list[dict[str, Any]] = field(default_factory=list)

    def all_figures(self) -> list[PaperFigureLike]:
        return [s["figure"] for s in self.summary]


def _build_figure_index(
    paper_figures: Iterable[PaperFigureLike],
) -> _FigureIndex:
    """Build the lookup index from a list of paper-level figures."""
    idx = _FigureIndex()
    seen_figure_ids: set[str] = set()

    for fig in paper_figures:
        fig_id = fig.get("figure_id") or ""
        if fig_id and fig_id in seen_figure_ids:
            continue
        if fig_id:
            seen_figure_ids.add(fig_id)

        ftype = _figure_type(fig)
        # We only index the non-plate figures (strat column / litholog /
        # paleogeographic map / range chart). Plates are what we link FROM.
        if ftype not in (
            "strat_column", "stratigraphic_column",
            "litholog_column", "litholog",
            "paleogeographic_map", "map",
            "range_chart",
        ):
            continue

        caption = _figure_caption(fig)
        formation = _figure_formation(fig)
        age = _figure_age(fig)
        locality = _figure_locality(fig)

        # Sample IDs from figure caption
        sample_ids = extract_sample_ids(caption)
        for sid in sample_ids:
            key = sid.value.casefold()
            if key not in idx.sample_id_to_figure:
                idx.sample_id_to_figure[key] = fig

        # Localities from figure caption. The panel-side extractor
        # requires a ``from/at/in`` prefix to avoid spurious matches in
        # long captions; on the figure side we want any capitalized
        # locality phrase, so we run a broader scan here.
        localities = _extract_locality_phrases(caption)
        for loc in localities:
            key = loc.casefold()
            if key not in idx.locality_to_figure:
                idx.locality_to_figure[key] = fig
            # Also add to sample_id_to_figure so a "Loc. Tunisia" on
            # the plate can match a bare "Tunisia" on the strat column
            # via Strategy 1.
            if key not in idx.sample_id_to_figure:
                idx.sample_id_to_figure[key] = fig
        # Also pick up direct locality fields on the figure itself.
        if locality:
            key = locality.casefold()
            if key not in idx.locality_to_figure:
                idx.locality_to_figure[key] = fig
            if key not in idx.sample_id_to_figure:
                idx.sample_id_to_figure[key] = fig

        idx.summary.append({
            "figure_id": fig_id,
            "figure_type": ftype,
            "caption": caption,
            "formation": formation,
            "age": age,
            "locality": locality,
            # ``figure_num`` is the display-number extracted from the
            # caption (e.g. "3" from "Fig. 3"). Strategy 4
            # (cross_refs) uses this to map ``CrossRef.target_figure_num``
            # back to the underlying figure.
            "figure_num": _extract_figure_number(fig, fid=fig_id, ftype=ftype),
            "figure": fig,
        })

    return idx


# Capitalized-word run finder used during indexing: when the *figure*
# caption is short (just "Tunisia outcrop, Scaglia Fm"), we want any
# Capitalized phrase to count as a locality candidate. We then filter
# against the blocklist.
_BARE_CAPITALIZED_RE = re.compile(
    r"\b([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3})\b"
)

# _LOCALITY_BLOCKLIST is imported from sample_id_extractor above and used directly.
def _extract_locality_phrases(caption: str) -> list[str]:
    """Figure-side locality extractor: any Capitalized phrase, deduped.

    We deliberately do NOT require a ``from/at/in`` prefix here because
    strat-column / map captions are usually short noun phrases
    (``"Tunisia outcrop, Scaglia Fm"``) without prepositions. The
    blocklist filters out chronostratigraphy terms that share the
    Capitalized-word shape.
    """
    if not caption:
        return []
    out: list[str] = []
    for m in _BARE_CAPITALIZED_RE.finditer(caption):
        phrase = m.group(1).strip()
        if not phrase or len(phrase) < 3:
            continue
        if phrase.casefold() in _LOCALITY_BLOCKLIST:
            continue
        out.append(phrase)
    return list(dict.fromkeys(out))  # dedupe, preserve order


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _panel_caption(panel: Any) -> str:
    """Best-effort caption snippet for a panel.

    Accepts a MatchResult dataclass or a dict. We try
    ``caption_snippet`` first (the per-panel slice), then fall back to
    the panel's ``metadata.caption`` if present, then to
    ``metadata.figure_caption`` (figure-level).
    """
    if hasattr(panel, "caption_snippet"):
        val = getattr(panel, "caption_snippet")
        if val:
            return str(val)
    if isinstance(panel, dict):
        val = panel.get("caption_snippet") or panel.get("caption")
        if val:
            return str(val)
        meta = panel.get("metadata") or {}
        val = meta.get("caption") or meta.get("figure_caption")
        if val:
            return str(val)
    else:
        meta = getattr(panel, "metadata", None) or {}
        val = meta.get("caption") or meta.get("figure_caption")
        if val:
            return str(val)
    return ""


def _panel_species(panel: Any) -> str | None:
    if hasattr(panel, "species"):
        val = getattr(panel, "species")
        if val:
            return str(val)
    if isinstance(panel, dict):
        val = panel.get("species")
        if val:
            return str(val)
    return None


def _panel_paper_id(panel: Any) -> str:
    if hasattr(panel, "paper_id"):
        return str(getattr(panel, "paper_id") or "")
    if isinstance(panel, dict):
        return str(panel.get("paper_id") or "")
    return ""


def _panel_panel_id(panel: Any) -> str | None:
    if hasattr(panel, "panel_id"):
        return getattr(panel, "panel_id")
    if isinstance(panel, dict):
        return panel.get("panel_id")
    return None


def _panel_figure_id(panel: Any) -> str | None:
    """Return the figure_id that the panel belongs to.

    Used by Strategy 4 (cross_refs) to filter out self-references in
    the panel's caption ("Fig. 2" mentioned inside Fig. 2's own
    caption). Looks at ``panel.figure_id`` directly, then
    ``panel.metadata.figure_id``.
    """
    if isinstance(panel, dict):
        val = panel.get("figure_id")
        if val:
            return str(val)
        meta = panel.get("metadata") or {}
        val = meta.get("figure_id")
        if val:
            return str(val)
        return None
    val = getattr(panel, "figure_id", None)
    if val:
        return str(val)
    meta = getattr(panel, "metadata", None) or {}
    val = meta.get("figure_id") if isinstance(meta, dict) else None
    return str(val) if val else None


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _strategy1_sample_match(
    panel: Any,
    fig_index: _FigureIndex,
) -> LinkResult | None:
    """Sample ID direct match.

    Confidence 1.0 if exactly one paper figure contains the same
    sample / loc / ID token; 0.9 if multiple figures share the token
    (ambiguous — we still link, but the caller may want to flag for
    review).

    We consider two kinds of matches:

    * Sample-ID tokens (``Sample S1``, ``ID-42``, etc.) — looked up in
      ``fig_index.sample_id_to_figure``.
    * Bare locality tokens that appear on the panel via ``Loc.``
      keyword — also matched against ``fig_index.sample_id_to_figure``
      AND ``fig_index.locality_to_figure`` so the same Tunisia string
      gets linked whether it was on the plate or on the strat column.
    """
    caption = _panel_caption(panel)
    if not caption:
        return None
    sample_ids = extract_sample_ids(caption)
    if not sample_ids:
        return None

    # Count hits per figure; pick the figure with the most matches.
    hit_counts: dict[str, int] = {}
    hit_figures: dict[str, PaperFigureLike] = {}
    evidence_tokens: list[str] = []

    for sid in sample_ids:
        key = sid.value.casefold()
        # Try sample-id index first, then fall back to locality index
        # so a "Loc. Tunisia" on the plate matches a bare "Tunisia"
        # on the strat column.
        fig = (
            fig_index.sample_id_to_figure.get(key)
            or fig_index.locality_to_figure.get(key)
        )
        if fig is None:
            continue
        fig_id = str(fig.get("figure_id") or "")
        if not fig_id:
            continue
        hit_counts[fig_id] = hit_counts.get(fig_id, 0) + 1
        hit_figures[fig_id] = fig
        evidence_tokens.append(sid.value)

    if not hit_figures:
        return None

    # Pick the figure with most hits; tie-break by figure_id for stability.
    best_fig_id = max(sorted(hit_counts), key=lambda k: hit_counts[k])
    fig = hit_figures[best_fig_id]
    confidence = 1.0 if len(hit_figures) == 1 else 0.9
    return LinkResult(
        panel_id=_panel_panel_id(panel),
        species=_panel_species(panel),
        figure_id=best_fig_id,
        formation=_figure_formation(fig),
        age=_figure_age(fig),
        locality=_figure_locality(fig),
        confidence=confidence,
        source=LINK_SOURCE_SAMPLE,
        evidence="sample_id_match: " + ", ".join(evidence_tokens),
    )


def _strategy2_locality_match(
    panel: Any,
    fig_index: _FigureIndex,
) -> LinkResult | None:
    """Locality string share.

    Confidence 0.7. If the panel caption contains a locality phrase
    that also appears in any paper-level figure's caption, link.
    """
    caption = _panel_caption(panel)
    if not caption:
        return None
    localities = extract_locality(caption)
    if not localities:
        return None

    for loc in localities:
        key = loc.casefold()
        fig = fig_index.locality_to_figure.get(key)
        if fig is None:
            continue
        return LinkResult(
            panel_id=_panel_panel_id(panel),
            species=_panel_species(panel),
            figure_id=str(fig.get("figure_id") or ""),
            formation=_figure_formation(fig),
            age=_figure_age(fig),
            locality=loc,
            confidence=0.7,
            source=LINK_SOURCE_LOCALITY,
            evidence=f"locality_match: {loc}",
        )
    return None


def _strategy3_m3_inference(
    panel: Any,
    fig_index: _FigureIndex,
    m3_inference: M3InferenceCallable | None,
) -> LinkResult | None:
    """M3 cross-figure inference.

    Confidence 0.3-0.6 depending on what M3 returns. If ``m3_inference``
    is None or returns a low-quality answer, returns None so the
    caller falls back to the unlinked row.
    """
    if m3_inference is None:
        return None

    paper_context = {
        "figures": fig_index.summary,
    }
    panel_caption = _panel_caption(panel)
    try:
        result = m3_inference(panel_caption, paper_context)
    except Exception:
        return None

    if not isinstance(result, dict):
        return None

    formation = result.get("formation") or None
    age = result.get("age") or None
    locality = result.get("locality") or None
    figure_id = result.get("figure_id") or None
    confidence = float(result.get("confidence", 0.3) or 0.3)

    # Clamp to spec range.
    confidence = max(0.3, min(0.6, confidence))

    return LinkResult(
        panel_id=_panel_panel_id(panel),
        species=_panel_species(panel),
        figure_id=str(figure_id) if figure_id else None,
        formation=str(formation) if formation else None,
        age=str(age) if age else None,
        locality=str(locality) if locality else None,
        confidence=confidence,
        source=LINK_SOURCE_M3,
        evidence=f"m3_inference: conf={confidence}",
    )


def _strategy4_cross_refs_match(
    panel: Any,
    fig_index: _FigureIndex,
) -> LinkResult | None:
    """Cross-reference match (Strategy 4).

    When a panel's caption mentions another figure by display name
    ("see Fig. 3", "as in Pl. 2", "compared with Figure 5"), use
    ``rlpe.cross_refs.parse_cross_refs`` to find those mentions and
    map each one back to a paper-level figure (strat / litholog /
    map / range chart) via ``figure_num``.

    Confidence:
    - 0.85 if exactly one cross-ref resolves to a single figure
    - 0.75 if multiple cross-refs all point to the same figure (still
      unambiguous, just less confident than a single clear mention)
    - Not fired if the only mentions are self-references
      (handled by ``current_fig_id`` inside ``parse_cross_refs``)
    - Not fired if no cross-ref resolves to a paper-level figure in
      the index

    Audit 2026-08-16 (fill-gaps): this strategy was previously dead
    code (``rlpe.cross_refs`` had tests but no caller). Wiring it in
    here gives the linker an explicit-textual-match tier between the
    regex locality match (0.7) and the M3 inference fallback
    (0.3-0.6).
    """
    caption = _panel_caption(panel)
    if not caption:
        return None
    panel_fig_id = _panel_figure_id(panel) or ""
    refs = parse_cross_refs(caption, current_fig_id=panel_fig_id)
    if not refs:
        return None

    # Map each CrossRef to a paper figure via ``figure_num``.
    # First match wins per CrossRef; we keep the unique set of figures.
    matched_figs: dict[str, PaperFigureLike] = {}
    evidence_tokens: list[str] = []
    for ref in refs:
        for s in fig_index.summary:
            if s.get("figure_num") == ref.target_figure_num:
                fid = str(s.get("figure_id") or "")
                if fid and fid not in matched_figs:
                    matched_figs[fid] = s["figure"]
                    evidence_tokens.append(ref.target_figure)
                break

    if not matched_figs:
        return None

    # Pick the first match (insertion order = caption order, stable).
    best_fid, best_fig = next(iter(matched_figs.items()))
    confidence = 0.85 if len(matched_figs) == 1 else 0.75
    return LinkResult(
        panel_id=_panel_panel_id(panel),
        species=_panel_species(panel),
        figure_id=best_fid,
        formation=_figure_formation(best_fig),
        age=_figure_age(best_fig),
        locality=_figure_locality(best_fig),
        confidence=confidence,
        source=LINK_SOURCE_CROSS_REF,
        evidence="cross_ref_match: " + ", ".join(evidence_tokens),
    )


def _unlinked_fallback(panel: Any) -> LinkResult:
    """Returned when no strategy matched.

    Always non-None so the exported row carries an explicit "unlinked"
    tag rather than being silently absent.
    """
    return LinkResult(
        panel_id=_panel_panel_id(panel),
        species=_panel_species(panel),
        figure_id=None,
        formation=None,
        age=None,
        locality=None,
        confidence=0.0,
        source=LINK_SOURCE_UNLINKED,
        evidence="no strategy matched",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def link_species_to_geology(
    panels: Iterable[Any],
    paper_figures: Iterable[PaperFigureLike],
    m3_engine: Any | None = None,
    m3_inference_callable: M3InferenceCallable | None = None,
) -> list[LinkResult]:
    """Run the 3-strategy linker on a paper's panels.

    Parameters
    ----------
    panels : iterable
        Panel-level objects (MatchResult dataclass or dict). Must
        expose at least ``caption_snippet`` (or
        ``metadata.caption``), ``paper_id``, ``panel_id``, ``species``.
    paper_figures : iterable
        Paper-level figure summaries (Phase B FigureRecord or a
        raw dict). Only ``strat_column`` / ``litholog_column`` /
        ``paleogeographic_map`` / ``range_chart`` figure_types are
        indexed for matching.
    m3_engine : optional
        A ``M3Engine`` instance with an
        ``infer_species_age_formation(panel_caption, paper_context)``
        method. If provided, we adapt it to a plain callable. If
        ``None``, Strategy 3 is skipped.
    m3_inference_callable : optional
        Direct override for the M3 inference callback. Takes
        precedence over ``m3_engine``.

    Returns
    -------
    list[LinkResult]
        One entry per panel. The list preserves input order.
    """
    fig_index = _build_figure_index(paper_figures)

    callback: M3InferenceCallable | None = m3_inference_callable
    if callback is None and m3_engine is not None:
        method = getattr(m3_engine, "infer_species_age_formation", None)
        if callable(method):
            callback = method

    out: list[LinkResult] = []
    for panel in panels:
        # Cross-paper isolation: only link against the panel's own paper's
        # figures. The index is per-call so this is automatic, but we
        # still skip panels whose ``paper_id`` doesn't match any of the
        # paper_figures we received (defensive).
        panel_paper = _panel_paper_id(panel)
        paper_in_input = any(
            str(f.get("paper_id") or "") == panel_paper
            for f in paper_figures
        )
        if not paper_in_input and panel_paper:
            # Caller passed figures for a different paper; emit unlinked
            # to make the boundary explicit.
            out.append(_unlinked_fallback(panel))
            continue

        result = (
            _strategy1_sample_match(panel, fig_index)
            or _strategy2_locality_match(panel, fig_index)
            or _strategy4_cross_refs_match(panel, fig_index)
            or _strategy3_m3_inference(panel, fig_index, callback)
            or _unlinked_fallback(panel)
        )
        out.append(result)

    return out


# Source string stamped onto every Phase C visual link so downstream
# audit / GUI / export can distinguish them from Phase A text-only
# links.
VISUAL_LINK_SOURCE = "m3_visual"

# Phase 66 Plan C.3: figures considered "anchor" figures for the
# visual-coordinate trigger. If the paper has any of these alongside
# a plate, Phase C may fire. Mirrors the list in
# ``_build_figure_index`` above.
_ANCHOR_FIGURE_TYPES = frozenset({
    "strat_column", "stratigraphic_column",
    "litholog_column", "litholog",
    "paleogeographic_map", "map",
})

# Phase 66 Plan C.3: figure_type values that count as "plate" for the
# trigger condition. Papers without a plate don't need Phase C
# (there's nothing to link).
_PLATE_FIGURE_TYPES = frozenset({
    "plate", "plate_image",
})


# ---------------------------------------------------------------------------
# Phase 66 Plan C.3 — visual coordinate trigger
# ---------------------------------------------------------------------------

def _panel_link_source(panel: Any) -> str | None:
    """Best-effort link_source from a panel dict/object."""
    if isinstance(panel, dict):
        meta = panel.get("metadata") or {}
        return meta.get("link_source")
    meta = getattr(panel, "metadata", None) or {}
    if isinstance(meta, dict):
        return meta.get("link_source")
    return getattr(meta, "link_source", None)


def _has_plate_and_anchor(figures: Iterable[PaperFigureLike]) -> tuple[bool, PaperFigureLike | None, PaperFigureLike | None]:
    """Return ``(has_both, plate_fig, anchor_fig)``.

    Returns the FIRST plate figure and the FIRST anchor figure (strat
    column / litholog / paleogeographic map). Phase C only needs one
    of each — multi-plate / multi-strat combos use the first match.
    """
    plate = None
    anchor = None
    for fig in figures:
        ftype = _figure_type(fig)
        if ftype in _PLATE_FIGURE_TYPES and plate is None:
            plate = fig
        elif ftype in _ANCHOR_FIGURE_TYPES and anchor is None:
            anchor = fig
        if plate is not None and anchor is not None:
            break
    return (plate is not None and anchor is not None, plate, anchor)


def link_visual_coordinates(
    panels: Iterable[Any],
    paper_figures: Iterable[PaperFigureLike],
    m3_engine: Any | None = None,
) -> list[list[dict[str, Any]]]:
    """Phase 66 Plan C.3 — vision-based cross-figure linkage.

    Runs the ``cross_figure_visual_inference`` M3 method on each panel
    whose Phase A Strategy-1 (sample_match) didn't reach confidence
    1.0 AND whose paper has BOTH a plate figure AND a strat column /
    litholog / paleogeographic map. The returned visual links are
    stored as Phase C precision refinements — they don't replace the
    Phase A linkage.

    Parameters
    ----------
    panels : iterable
        Panel-level objects (MatchResult / dict). Each must expose
        ``metadata.link_source`` (set by Phase A) so we can detect
        the "Strategy 1 already nailed it" case.
    paper_figures : iterable
        Paper-level figure summaries (Phase B FigureRecord or dict).
    m3_engine : optional
        A ``M3Engine`` instance with a
        ``cross_figure_visual_inference(plate_image, strat_image,
        plate_caption, strat_caption)`` method. If ``None`` or the
        method is missing, Phase C is silently skipped.

    Returns
    -------
    list[list[dict]]
        Outer list indexed by panel (preserves input order). Inner
        list is the visual links for that panel — empty when the
        trigger condition is not met OR M3 returned nothing usable.
        Each entry has keys::

          {
            "target_figure_id": str,
            "target_layer": int | None,
            "target_age": str | None,
            "target_formation": str | None,
            "confidence": float (0.0-1.0),
            "source": "m3_visual",
          }

    Notes
    -----
    * The Phase A ``link_source`` field is the gate — panels marked
      ``"sample_match"`` (confidence 1.0 or 0.9) are skipped because
      Strategy 1 is intrinsically the strongest Phase A signal.
    * The plate + anchor requirement is structural: without a strat
      column or map, there's nothing to visually link to. A paper
      with only plates is Phase A's territory.
    * The caller is responsible for writing the inner lists into
      ``panel.metadata.cross_figure_visual_links`` (Task C.4 does
      this from the pipeline).
    """
    panels_list = list(panels)
    figures_list = list(paper_figures)

    has_both, plate_fig, anchor_fig = _has_plate_and_anchor(figures_list)
    if not has_both:
        return [[] for _ in panels_list]

    # No M3 engine (or no visual method) → silent skip, same as
    # fallback_used upstream.
    if m3_engine is None:
        return [[] for _ in panels_list]
    visual_method = getattr(m3_engine, "cross_figure_visual_inference", None)
    if not callable(visual_method):
        return [[] for _ in panels_list]

    # We don't actually have real images at this layer (the pipeline
    # passes them in separately in Task C.4). For the trigger-logic
    # function we pass None — the visual method handles missing
    # images via its ``image.width < 32`` early-return path.
    plate_caption = _figure_caption(plate_fig) if plate_fig is not None else ""
    anchor_caption = _figure_caption(anchor_fig) if anchor_fig is not None else ""
    anchor_id = (
        str(anchor_fig.get("figure_id") or "") if anchor_fig is not None else ""
    )

    out: list[list[dict[str, Any]]] = []
    for panel in panels_list:
        link_source = _panel_link_source(panel)
        # The trigger condition: skip panels whose Phase A Strategy 1
        # already nailed them. Everything else (locality_match,
        # m3_inference, unlinked) gets the visual treatment.
        if link_source == LINK_SOURCE_SAMPLE:
            out.append([])
            continue

        try:
            result = visual_method(
                None, None, plate_caption, anchor_caption,
            )
        except Exception:
            # Defensive: a backend exception must never propagate up
            # the pipeline. Phase C silently degrades to empty.
            out.append([])
            continue
        panels_data = result.get("plate_panels") if isinstance(result, dict) else None
        if not isinstance(panels_data, list) or not panels_data:
            out.append([])
            continue

        # Build the per-panel link list. We emit one link per panel
        # entry M3 returned; the panel itself doesn't filter by
        # cell_label (the schema stores them all and the GUI picks
        # the right one per printed_panel_id).
        links: list[dict[str, Any]] = []
        for entry in panels_data:
            if not isinstance(entry, dict):
                continue
            try:
                conf = float(entry.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            links.append({
                "target_figure_id": anchor_id,
                "target_layer": entry.get("links_to_strat_layer"),
                "target_age": entry.get("links_to_age"),
                "target_formation": entry.get("links_to_formation"),
                "confidence": conf,
                "source": VISUAL_LINK_SOURCE,
            })
        out.append(links)

    return out


__all__ = [
    "LinkResult",
    "M3InferenceCallable",
    "link_species_to_geology",
    "link_visual_coordinates",
    "LINK_SOURCE_SAMPLE",
    "LINK_SOURCE_LOCALITY",
    "LINK_SOURCE_M3",
    "LINK_SOURCE_UNLINKED",
    "VISUAL_LINK_SOURCE",
]
