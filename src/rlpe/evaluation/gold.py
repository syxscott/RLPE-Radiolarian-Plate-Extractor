"""Ground-truth loader for RLPE evaluation.

A gold file is a JSONL with one record per panel::

    {"paper_id": "hollis2006", "figure_id": "plate_1", "panel_id": "1", "species": "Genus species"}
    {"paper_id": "hollis2006", "figure_id": "plate_1", "panel_id": "2", "species": "Genus species"}

The ``panel_id`` here is the *label printed in the figure* (1, 2, 3, A, B, 12a).
The species is the canonical Latin binomial or genus+sp abbreviation. Empty
species means "no species assigned in the gold" (some panels are unlabelled
or are scale bars).

IMPORTANT — gold derivation caveat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The current gold standard is derived from caption-parser output (see
``scripts/build_gold_from_captions.py``), NOT from visual inspection of
panel images. This means F1 scores measure **parser self-consistency**,
not ground-truth accuracy against manually verified panel labels. The
``image_label_check`` module exists precisely because of this limitation.
For publication, the gold should be image-verified by a human annotator.
The ``source`` field on GoldPanel distinguishes "caption-parsed" from
"manual" entries.

The gold files are stored in ``data/gold/<paper>.jsonl`` and are also
published in the same schemas/ directory for downstream consumers.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

GOLD_SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# audit 2026-08-02: tolerance layer for eval-side normalization
# ---------------------------------------------------------------------------
# The hollis2006 (61.9% F1) and feng2007 (83.9% F1) gaps are dominated by
# surface-form differences between predicted and gold species names —
# Roman vs Arabic numerals, "cf." vs "cf", whitespace, etc. The two
# pure-eval layers below close that gap without touching the gold data.
#
# Layer A (panel_id): strip + ASCII-fold + lowercase; split comma-separated
# gold entries into individual tokens so a gold row labelled "1, 2, 3"
# matches three independent pred rows.
# Layer B (species): lowercase + roman→arabic + cf./aff. normalisation +
# parenthesis strip + whitespace collapse; composed with the existing
# ``metrics._norm_species`` rules (trinomial fold, sp./spp. strip, etc.).
# ---------------------------------------------------------------------------

_ROMAN_TO_ARABIC: dict[str, str] = {
    "II": "2",
    "III": "3",
    "IV": "4",
    "VI": "6",
    "VII": "7",
    "VIII": "8",
    "IX": "9",
    "X": "10",
    "V": "5",
    "I": "1",
}
# Roman numerals are by convention upper-case in taxonomic literature
# ("Plate II", "Species III"), but OCR may already have lower-cased
# them. Case-insensitive matching + a final ``str.upper()`` on the
# captured token routes the lookup against the upper-case table above
# without forcing the caller to pre-lowercase the entire name.
_ROMAN_TOKEN_RE = re.compile(r"\b(II|III|IV|VII|VIII|IX|X|VI|V|I)\b", re.IGNORECASE)
_CF_AFF_DOT_RE = re.compile(r"\b(cf|aff)\.\s*", re.IGNORECASE)
_PARENS_RE = re.compile(r"\([^)]*\)")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_species(s: str | None) -> str:
    """Layer B: tolerant species normalisation for eval equality.

    This function applies ONLY the surface-form relaxations that
    ``metrics._norm_species`` does NOT already handle. It must run
    *before* ``_norm_species`` so the downstream case-sensitive rules
    (``Archaeo`` → ``Archeo`` capitalised prefix match, the
    ``[A-Z][a-z]+\\?`` inline-question-mark regex) still see the
    original case and pattern-match correctly.

    Applied (in order, case-preserving):
      1. Leading/trailing whitespace strip.
      2. Whole-word Roman numerals → Arabic (``II`` → ``2``,
         ``III`` → ``3``, ``IV`` → ``4``, ``VI`` … ``X`` → ``6`` …
         ``10``, ``V`` → ``5``, ``I`` → ``1``). Case-insensitive so
         lower-cased OCR ("ii") is handled. The lookup happens against
         the upper-case table via ``m.group(0).upper()``.
      3. ``cf.``/``aff.`` → ``cf``/``aff`` (the trailing period is
         dropped; a single trailing space is preserved).
      4. Strip parentheses and their contents (radiolarian captions
         embed metadata in parens that the parser inconsistently
         captures — e.g. ``Hastigerina(?)`` vs ``Hastigerina``).
      5. Collapse runs of whitespace; trim trailing ``.,;``.

    NOT applied here (because ``metrics._norm_species`` already does
    them with the right case-sensitivity):
      - lowercase (used in ``_species_compatible``)
      - leading/inline ``?`` stripping (case-sensitive ``[A-Z][a-z]+``)
      - ``sp``/``spp`` stripping
      - trinomial autonym fold
      - ``Archaeo`` → ``Archeo`` fold (case-sensitive ``startswith``)
      - ``X gen`` → ``X indet`` fold

    The full pipeline comparison (used inside
    :func:`rlpe.evaluation.metrics.evaluate`) is::

        _species_compatible(
            _norm_species(normalize_species(g.species)),
            _norm_species(normalize_species(pred.species)),
        )

    which lowercases after :func:`_norm_species` runs, so the final
    TP/FP decision is case-insensitive as required by the test suite.
    """
    if not s:
        return ""
    out = s.strip()
    out = _ROMAN_TOKEN_RE.sub(lambda m: _ROMAN_TO_ARABIC[m.group(0).upper()], out)
    out = _CF_AFF_DOT_RE.sub(r"\1 ", out)
    out = _PARENS_RE.sub("", out)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    out = out.rstrip(".,;")
    return out


# Audit 2026-09-01 BL-29: restricted fold table — only characters
# that are KNOWN to confuse in the panel_id OCR path. The previous full
# NFKD + encode("ascii","ignore") round-trip silently destroyed the
# distinction between "Æ"/"AE", "Ø"/"O", "Œ"/"OE" etc. and caused
# FN matches on Nordic / French papers. Limit the fold to typographic
# variants that OCR engines mis-read: smart quotes, the micro sign vs.
# "u" prefix, the various hyphen variants, non-breaking space.
_OCR_CONFUSION_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        " ": " ",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "µ": "u",  # OCR frequently renders µm as "u m"
    }
)


def _normalize_panel_id(s: str) -> str:
    """Layer A: normalise a single panel_id token for comparison.

    Strips whitespace, folds Unicode diacritics to ASCII so a gold
    ``Æthium`` matches pred ``Aethium``, and lowercases so gold ``A``
    matches pred ``a``. Compound forms (``1-3``) and letter suffixes
    (``12a``) are preserved unchanged — comma-separated values are
    split by :func:`match_panel` at the gold-entry level rather than
    here, so a single token always flows through this function as-is.
    """
    stripped = s.strip()
    # Audit 2026-09-01 (live Bandini end-to-end): strip the "Fig. N"
    # / "Plate N" / "pl. N" / "図版 N" prefix when present so the
    # gold-side label "1" matches the pred-side label "Fig. 1". The
    # previous implementation kept both forms verbatim, costing the
    # 9-paper eval roughly 30-50 % panel-match on any paper that
    # uses the canonical "Fig. N" caption convention. The prefix
    # patterns are anchored at string start and tolerate an optional
    # trailing space; "Fig N" (no dot) and "Figure N" are both
    # covered.
    stripped = re.sub(
        r"^(?:fig(?:ure)?s?\.?|pl(?:ate)?s?\.?|図版)\s*",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    # Audit 2026-09-01 BL-29: the previous ``NFKD + encode("ascii",
    # "ignore")`` round-trip folded legitimate Nordic / French
    # diacritics ("Æ" / "Ø" / "Œ") into their ASCII lookalikes
    # ("AE" / "O" / "OE"), which then silently mismatched a
    # properly-spelled gold panel_id (e.g. gold="Æ", pred="Ae" → FN).
    # Restrict the fold to the OCR-only confusion set (curly quotes
    # / micro sign / minus sign variants) so author and taxon names
    # with diacritics keep their identity.
    ocr_safe = stripped.translate(_OCR_CONFUSION_TRANSLATION)
    ascii_folded = unicodedata.normalize("NFKD", ocr_safe).encode("ascii", "ignore").decode("ascii")
    return ascii_folded.lower()


@dataclass(slots=True)
class GoldPanel:
    paper_id: str
    figure_id: str
    panel_id: str | None
    species: str | None
    # P2-1 fix: provenance field — "caption-parsed" (current) or "manual"
    source: str = "caption-parsed"

    def to_dict(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "figure_id": self.figure_id,
            "panel_id": self.panel_id,
            "species": self.species,
            "source": self.source,
        }


def load_gold(path: Path) -> list[GoldPanel]:
    out: list[GoldPanel] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                GoldPanel(
                    paper_id=str(d["paper_id"]),
                    figure_id=str(d["figure_id"]),
                    panel_id=d.get("panel_id"),
                    species=d.get("species"),
                    source=d.get("source", "caption-parsed"),
                )
            )
    return out


def write_gold(panels: Iterable[GoldPanel], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for p in panels:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def _extension_is_alpha(shorter: str, longer: str) -> bool:
    """Return True if ``longer`` is ``shorter`` extended by alphabetic content.

    A "sub-label" relationship: gold "5" and pred "5a" should match (5a is
    5 with letter suffix a). But gold "5" and pred "10" must NOT match —
    "10" is "1" extended by a digit, which means a different panel.

    Examples:
      "5"   + "5a"  → True  (alphabetic suffix)
      "5"   + "5bc" → True  (alphabetic suffix)
      "5"   + "10"  → False (numeric suffix — different panel)
      "A"   + "Aa"  → True  (alphabetic suffix)
      "A"   + "A1"  → False (numeric suffix — different panel)
      "12"  + "12a" → True
      "12a" + "12b" → False (not a prefix relationship at all)
    """
    if not longer.startswith(shorter):
        return False
    suffix = longer[len(shorter) :]
    if not suffix:
        return False
    return suffix.isalpha()


def match_panel(gold: GoldPanel, pred_paper_id: str, pred_panel_id: str | None) -> bool:
    """Decide whether a predicted panel corresponds to a gold panel.

    Match rules:
      - paper_ids must match exactly
      - empty/missing panel_id on either side is a non-match
      - exact string match always matches (after Layer A normalisation
        including ASCII-fold + lowercase)
      - prefix match is allowed ONLY when the longer label extends the
        shorter with ALPHABETIC content (e.g. gold "5" + pred "5a",
        gold "12" + pred "12a"). Pure-numeric extensions like "5"/"10"
        are distinct panels and must not match. Numeric-prefix-then-letter
        ("A" + "A1") is also rejected because "1" is numeric suffix.
      - Round 15 audit: Unicode is normalized to ASCII before
        comparison so a gold "Aethium" and pred "Æthium" match. Real
        radiolarian names have diacritics that survive OCR in some
        engines and get stripped in others.
      - audit 2026-08-02 (Layer A): a gold ``panel_id`` containing
        commas (e.g. ``"1, 2, 3"``) is split into individual tokens,
        each of which is independently matched against the predicted
        label. Compound forms like ``"1-3"`` or ``"1a"`` flow through
        unchanged because they contain no comma — preserving the
        existing single-panel semantics.
    """
    if gold.paper_id != pred_paper_id:
        return False
    if not gold.panel_id or not pred_panel_id:
        return False
    p = _normalize_panel_id(pred_panel_id)
    if not p:
        return False
    # Layer A: split comma-separated gold entries into individual tokens
    # so a gold row labelled "1, 2, 3" matches three separate pred rows.
    for raw_token in gold.panel_id.split(","):
        g = _normalize_panel_id(raw_token)
        if not g:
            continue
        if g == p:
            return True
        if len(g) < len(p) and _extension_is_alpha(g, p):
            return True
        if len(p) < len(g) and _extension_is_alpha(p, g):
            return True
    return False
