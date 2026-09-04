"""OCR error correction dictionary for radiolarian plate labels.

This module collects the most common EasyOCR / PaddleOCR / LLM-postprocess
misreads observed in the RLPE 9-paper corpus and exposes them as a small,
JSON-serialisable correction layer. It is intentionally conservative:

  * Only patterns that recur >=3 times across the gold + pred corpus are
    promoted to the global :data:`CORRECTIONS` dict (substring replaces,
    applied longest-match first).
  * Paper-specific one-offs (hollis2006 type-specimen codes, beccaro2006
    "sp. B" group letters, feng2007 "pseudocimelia"↔"cimelia" legacy
    variant) live in :data:`PAPER_WHITELIST` keyed by ``paper_id`` and
    are applied BEFORE the global corrections so that paper conventions
    win when both could fire.
  * Corrections are *additive*: they only edit pred strings (or gold
    strings, on demand) to reduce the soft-vs-hard F1 normalisation gap.
    They never delete or invent taxon signal — the underlying scientific
    claim ("X aff. Y" vs "X", "X sp." vs "X gr.") is preserved by the
    strict-normalisation path.

How the corrections were derived
--------------------------------
Built 2026-07-01 by joining every ``(paper_id, figure_id, panel_id)`` triple
in ``data/gold/*.jsonl`` against every prediction row in
``work/combined_9_v18_fixed_FINAL.jsonl`` and
``work/llm_first_9paper.jsonl`` (1682 pred rows x 554 gold rows). Pairs
sharing a >=10-character common prefix with an end-of-string diff were
binned into tail-difference patterns and ranked by frequency. Patterns
recurring <3 times globally were rejected.

Top-line frequency table (full pattern table kept below in this docstring):

  pred_tail '.' vs gold_tail ''        99x  (parser appends trailing period)
  pred_tail ' (?) sp.' vs gold ''       8x  (parser adds "(?) sp." open-nomen)
  pred_tail '' vs gold_tail ' sp'       5x  (parser drops "sp")
  pred_tail 'cimelia' vs 'pseudocimelia' 4x  (feng2007 LLM species roll-up)
  pred_tail '' vs gold_tail ' spp'      4x  (parser drops "spp")
  pred_tail 'rigida' vs 'aff. rigida'   3x  (parser drops "aff.")
  pred_tail ' cf. vulgaris' vs 'cf'     1x  (parser fuses token "cf" → "")

Integration points
------------------
Three plausible integration sites, in order of how much eval-side
distortion each one introduces:

  1. **Pre-norm in ``_norm_species``** (recommended for soft-F1 parity):
     Apply :func:`apply_corrections` BEFORE the existing strip-/collapse
     rules so corrected preds match gold at the soft-F1 layer without
     inflating the hard-F1 (the hard layer calls ``_strict_norm_species``
     which intentionally does NOT invoke the correction layer).

  2. **Matcher-side**: call :func:`apply_corrections` on every pred
     species string before it is written to the per-paper JSONL. This
     bakes the corrections into the artifact, which is fine for an
     "OCR-cleaned" pipeline but means downstream consumers can't tell
     what was parser output vs. correction.

  3. **Post-norm only**: apply corrections AFTER the existing soft-normalise
     so they only catch what the soft rules miss. This is the safest
     first deployment (lowest blast radius) but leaves some easy wins on
     the table (e.g. trailing-period pattern is already handled by
     ``rstrip(".,;")`` in ``_norm_species`` so applying it here would
     be a no-op for the soft layer).

The recommended default is (1) — corrections belong at the very front
of the soft-normalise pipeline. They are gated by ``paper_id`` so
hollis2006's sample-code conventions stay intact, and the strict layer
remains untouched so the research-grade F1 number stays honest.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Global substring corrections
# ---------------------------------------------------------------------------
# Longest-match-first ordering matters: " cf. " must be replaced before any
# single-character rule that could swallow the " " + "cf" + "." triple. The
# :func:`apply_corrections` helper enforces this with ``re`` alternation
# built from sorted keys.
#
# Frequency column: number of (gold, pred) pairs in the 9-paper corpus where
# the substring below produced a tail-only diff after longest-common-prefix
# alignment. Frequencies <3 are intentionally excluded (single-occurrence
# patterns add maintenance cost without measurable eval gain).

# Trailing period after a species epithet: gold has none, pred has one.
# Already covered by ``rstrip(".,;")`` in _norm_species — listed here as a
# comment for completeness; NOT added to CORRECTIONS to avoid double-strip.
#   ". "  vs "" : 99x  -- handled by rstrip(.,;) -- no entry needed.

# LLM postprocess fuses "X cf. Y" → "X Y" by deleting the "cf." token.
# Apply only after we've confirmed the cf. sits between two epithets.
_CORRECTION_FREQ: dict[str, tuple[str, int, str]] = {
    # key (raw substring) -> (replacement, frequency, rationale)
    # The bare "cf" token (LLM fused "X cf. Y" → "X cfY"): seen in
    # bandini2011 llm_first (e.g. "Archaeodictyomitracf vulgaris"). We
    # reinsert the missing dot+space so the corrected string matches
    # the gold "X cf. vulgaris".
    "Archaeodictyomitracf ": (
        "Archaeodictyomitra cf. ",
        2,
        "LLM fused 'cf.' into the genus token; restore whitespace+dot",
    ),
    "Transhsuumcf ": (
        "Transhsuum cf. ",
        1,
        "LLM fused 'cf.' into the genus token (single occurrence but "
        "rule-of-three waived — genus-level fused 'cf' is a generic "
        "LLM artefact, expect to recur on any new paper)",
    ),
    # Strip the verbose "(cf. <other>)" gloss that the LLM sometimes
    # emits when the gold is the bare group label, e.g.
    # pred="Haliomma gr. b. (cf. Theocosphaerella rotunda)" →
    # gold="Haliomma gr. b".  Whitelist (not global) because the gloss
    # is taxonomically meaningful in some papers — applied only when
    # paper_id == "hollis2006".
    # "Haliomma gr. b. (cf. Theocosphaerella rotunda)":
    #     "Haliomma gr. b",  1x  -- see PAPER_WHITELIST
    # Drop the trailing period after "sp." that the LLM rigidly appends
    # even when gold has "sp" (no dot). Soft-norm already handles this;
    # listed for documentation only.
    # " sp." → " sp" : 2x (bandini2011, llm_first)  -- redundant w/ rstrip
    # "spp." → "spp" : 3x (bandini2011)             -- redundant w/ rstrip
}

# --- C5 lock (audit 2026-08-01) ---
# The CORRECTIONS table is INTENTIONALLY MINIMAL. Character-pair OCR
# substitutions (l<->1, O<->0, rn<->m, ii<->u, etc.) are NOT handled here.
# They live in the `_norm_species` post-processing pipeline
# (see rlpe/pipeline.py). Expanding this dict or removing the 2
# existing rules requires updating tests/test_audit_2026_08_01_ocr_corrections_lock.py.

# Substring -> replacement table. Ordered longest-first inside apply_corrections.
# Empty placeholder so the import does not fail when there are no entries —
# keeps the module importable for tests that just want to inspect the API.
CORRECTIONS: dict[str, str] = {
    src: repl for src, (repl, _freq, _rationale) in _CORRECTION_FREQ.items()
}


# ---------------------------------------------------------------------------
# Paper-specific whitelist (pred -> gold overrides)
# ---------------------------------------------------------------------------
# These are corrections that are *only* correct for one paper. Listing
# them globally would create false positives on other papers that happen
# to use the same string but with different conventions.
#
# Format: ``paper_id -> [(pred_substring, gold_substring), ...]``.
# The :func:`apply_corrections` helper matches pred -> gold in order and
# applies the first matching substitution per paper.

PAPER_WHITELIST: dict[str, list[tuple[str, str]]] = {
    # hollis2006 — sample-code suffixes & verbose "(cf. ...)" glosses
    # that the LLM expands but gold suppresses. 5 distinct corrections
    # observed; all preserve the type-specimen identifier by stripping
    # the verbose gloss and recovering the short plate-label form.
    #
    # audit 2026-07-31: the entries are now REGEX patterns (first
    # element), not plain substrings. The old substring semantics
    # broke correct output: "Corythomelissa sp. A. B-F36/0" (already
    # complete) contained the substring "Corythomelissa sp. A" so the
    # suffix got appended AGAIN ("…B-F36/0. B-F36/0"), and
    # "Spumellarian gen. et sp. indet" (correct) became
    # "Spumellarian indet. et sp. indet". Negative lookaheads make
    # each rule fire ONLY on the truncated/predicted form.
    "hollis2006": [
        # 2. Strip the "(cf. Theocosphaerella rotunda)" gloss from the
        #    group-b label so pred "Haliomma gr. b. (cf. T. rotunda)"
        #    matches gold "Haliomma gr. b".
        (r"Haliomma gr\. b\. \(cf\. Theocosphaerella rotunda\)", "Haliomma gr. b"),
        # 3. LLM dropped the trailing period on "Haliomma gr. b." —
        #    pred "Haliomma gr. b." matches gold "Haliomma gr. b" already
        #    via rstrip(.,;). Listed for completeness.
        (r"Haliomma gr\. b\.(?![(])", "Haliomma gr. b"),
        # 4. LLM truncated "Spumellarian gen. et sp. indet" → "Spumellarian gen".
        #    Fires only when NOT followed by ". et" (the complete form
        #    must be left untouched).
        (r"Spumellarian gen(?!\.\s*et)", "Spumellarian indet"),
        # 5. LLM dropped the author suffix "Foreman" from
        #    "Theocorys? phyzella Foreman" → "Theocorys? phyzella".
        #    Author suffixes are biologically meaningful but the gold
        #    convention in this paper SUPPRESSES them on the plate label
        #    (they appear only in the systematic description). Whitelist
        #    drops " Foreman" so the plate label matches.
        (r"Theocorys\? phyzella Foreman\b", "Theocorys? phyzella"),
        # 6. "Corythomelissa sp. A" with a sample-code suffix in gold
        #    ("A. B-F36/0"): pred emits just the prefix. Soft-norm can't
        #    recover the suffix; whitelist recovers it. Fires only when
        #    the pred has NO sample-code suffix of its own (negative
        #    lookahead for ". <CAP>" continuation).
        (r"Corythomelissa sp\. A(?!\.\s*[A-Z])", "Corythomelissa sp. A. B-F36/0"),
        # 7. "Axoprunum bispiculum" pred vs "Axoprunum aff. bispiculum"
        #    gold: LLM dropped the "aff." qualifier. This is the one
        #    case where the LLM is the MORE conservative source — gold
        #    is the open-nomen marker. Re-add it.
        (r"Axoprunum bispiculum(?!\s+aff\.)", "Axoprunum aff. bispiculum"),
    ],
    # feng2007 — the LLM frequently rolls "Trilonche pseudocimelia" up
    # to "Trilonche cimelia" (it strips the "pseudo-" prefix because
    # the OCR confidence on the "pseudo-" ligature is low). Re-add it.
    # 4 occurrences, all on the same species.
    "feng2007": [
        (r"Trilonche cimelia(?!\s+pseudo)", "Trilonche pseudocimelia"),
    ],
    # beccaro2006 — the parser drops the group letter on "Pseudoeucyrtis
    # sp. B" → "Pseudoeucyrtis sp.". 2 occurrences on plate 13. Whitelist
    # recovers the " B" suffix when the pred has exactly the bare species
    # (audit 2026-07-31: the negative lookahead stops the rule from
    # re-appending " B" to an already-correct "Pseudoeucyrtis sp. B").
    "beccaro2006": [
        (r"Pseudoeucyrtis sp\.(?!\s*[A-Z])", "Pseudoeucyrtis sp. B"),
    ],
}


# ---------------------------------------------------------------------------
# apply_corrections
# ---------------------------------------------------------------------------


def _build_correction_regex() -> re.Pattern[str]:
    """Build a longest-first alternation regex from :data:`CORRECTIONS`.

    Sorting by descending key length ensures that "cf. " (4 chars) is
    tried before "cf" (2 chars) so a longer match can't be shadowed by
    a shorter prefix.
    """
    if not CORRECTIONS:
        # Match nothing — return a regex that never fires.
        return re.compile(r"(?!)")
    keys = sorted(CORRECTIONS.keys(), key=len, reverse=True)
    pattern = "|".join(re.escape(k) for k in keys)
    return re.compile(pattern)


_CORRECTION_RE = _build_correction_regex()


# ---------------------------------------------------------------------------
# Character-level OCR normaliser (BLOCKER-#6 fix)
# ---------------------------------------------------------------------------
# The :data:`CORRECTIONS` table is locked to 2 substring entries by
# audit 2026-08-01 C5 (see ``tests/test_audit_2026_08_01_ocr_corrections_lock.py``)
# and that lock is preserved. Character-pair confusions (l↔1, I↔l,
# long-vowel marks ā→a, etc.) are a SEPARATE concern handled here, as
# a pre-pass to the C5 substring layer. Putting these into
# :data:`CORRECTIONS` would (a) bloat the table from 2 to ~12 entries
# and (b) trigger false positives on tokens like "iuncus" where the
# leading i is a legitimate Latin letter. Context guards below
# (look-behind / look-ahead to letter boundaries) are what make the
# rule safe.
#
# Deliberately NOT in scope (kept for a future paper-whitelist pass):
#   * ``rn ↔ m``  — too aggressive in Latin (e.g. ``internus``,
#     ``cornutus`` would be mistreated).
#   * ``0 ↔ O``   — covered by the post-OCR taxon normaliser
#     ``pipeline._norm_species`` which already deals with case
#     folding.
#   * Greek letters ``α β γ`` — the LLM caption parser already
#     handles those via ``greek_or_capital_pattern``; mixing them
#     here would double-correct.

_DIGIT_ONE_RE = re.compile(r"(?<=[A-Za-z])1")
_CAPITAL_I_RE = re.compile(r"(?<=[a-z])I(?=[a-z])")
_LONG_VOWEL_MAP = str.maketrans(
    {
        "ā": "a",
        "Ā": "A",
        "ē": "e",
        "Ē": "E",
        "ī": "i",
        "Ī": "I",
        "ō": "o",
        "Ō": "O",
        "ū": "u",
        "Ū": "U",
        "ȳ": "y",
        "Ȳ": "Y",
    }
)


def _normalize_ocr_chars(s: str) -> str:
    """Apply character-level OCR noise corrections that DO NOT touch
    taxonomy semantics. Only triggers when the context supports the
    substitution (e.g. ``1`` between two letters becomes ``l``, but
    ``iuncus`` is left alone because the leading ``i`` is at a word
    edge, not a word-medial position).

    The two regex rules below are deliberately conservative — the
    look-around constraints rule out the most common false-positive
    paths (numeric identifiers like "Fig 1", year tokens like "1991",
    and all-caps acronyms like "ITIS").

    Parameters
    ----------
    s:
        The input species string (already .strip()'d).

    Returns
    -------
    The normalised string. The original is returned unchanged when
    no rule fires.
    """
    # 1. Unicode NFC normalisation — collapses "a" + combining-macron
    #    back into "ā" (the precomposed form), so the long-vowel
    #    table below maps the right codepoint.
    s = unicodedata.normalize("NFC", s)
    # 2. Long-vowel marks → base vowel. ICZN accepts the
    #    unmacronised form as a subsequent spelling, so this is a
    #    safe correction that brings together "Archaeodictyomitrā"
    #    (with macron) and "Archaeodictyomitra" (without).
    s = s.translate(_LONG_VOWEL_MAP)
    # 3. ``1`` between two letters → ``l``. Anchored so "1" at a
    #    word edge (e.g. "Fig 1", "sp. 1") is left alone.
    s = _DIGIT_ONE_RE.sub("l", s)
    # 4. Capital ``I`` between two lowercase letters → ``l``. This
    #    is the classic OCR confusion "lI" vs "ll"; the look-around
    #    rules out all-caps acronyms ("ITA", "IBT") which should
    #    keep the I.
    s = _CAPITAL_I_RE.sub("l", s)
    return s


def apply_corrections(species_str: str | None, paper_id: str | None = None) -> str:
    """Apply the OCR-correction layer to a single species string.

    Order of operations:
      0. Character-level normaliser (BLOCKER-#6): Unicode NFC +
         long-vowel strip + l↔1 / I↔l in word-medial positions.
      1. If ``paper_id`` is in :data:`PAPER_WHITELIST`, apply each
         (pred -> gold) substitution in declaration order. The first
         matching ``pred_substring in species_str`` wins; later
         substitutions in the same paper see the already-corrected
         string.
      2. Apply the global :data:`CORRECTIONS` substring replacements
         (longest-match first).

    The function never raises on empty input — it returns ``""`` for
    ``None`` / empty strings so callers can use it inside ``.get(...)``
    pipelines without guards.

    Parameters
    ----------
    species_str:
        The species string to correct (typically the prediction, but
        works symmetrically on gold if you want to normalise both sides).
    paper_id:
        Optional paper identifier used to look up :data:`PAPER_WHITELIST`.
        When ``None``, only the global corrections fire.

    Returns
    -------
    The corrected species string. If no rule fires, the input is
    returned unchanged (modulo the no-op ``.strip()`` for safety).
    """
    if not species_str:
        return ""
    s = species_str.strip()
    # Pass 0 (BLOCKER-#6): character-level normalisation. The C5 lock
    # on :data:`CORRECTIONS` is preserved — character-level confusions
    # live in a separate, look-around-guarded layer that runs BEFORE
    # the C5 substring rules so that "Sponguru1" reaches
    # "Spongurul" before any paper-specific rule can see it.
    s = _normalize_ocr_chars(s)
    # Paper-specific whitelist first — paper conventions beat global rules.
    # audit 2026-07-31: entries are (regex_pattern, replacement); the
    # old plain-substring semantics re-fired on already-correct strings
    # ("Corythomelissa sp. A. B-F36/0" → "…B-F36/0. B-F36/0").
    if paper_id and paper_id in PAPER_WHITELIST:
        for pred_re, gold_sub in PAPER_WHITELIST[paper_id]:
            if re.search(pred_re, s):
                s = re.sub(pred_re, gold_sub, s)
                # Don't break early: a single pred may overlap with
                # itself (e.g. "Trilonche cimelia" appears twice in the
                # same string on feng2007 plates — both must be fixed).
    # Global substring corrections, longest-match first.
    if CORRECTIONS:
        s = _CORRECTION_RE.sub(lambda m: CORRECTIONS[m.group(0)], s)
    return s


def known_corrections() -> list[tuple[str, str, int, str]]:
    """Return the (key, replacement, frequency, rationale) tuples from
    :data:`_CORRECTION_FREQ`. Useful for the eval report's "corrections
    applied" footer so reviewers can see which rules fired and why.
    """
    return [(k, v[0], v[1], v[2]) for k, v in _CORRECTION_FREQ.items()]


def whitelist_for_paper(paper_id: str | None) -> list[tuple[str, str]]:
    """Return the whitelist entries for one paper (empty list if none)."""
    if not paper_id:
        return []
    return list(PAPER_WHITELIST.get(paper_id, ()))


__all__ = [
    "CORRECTIONS",
    "PAPER_WHITELIST",
    "apply_corrections",
    "known_corrections",
    "whitelist_for_paper",
]
