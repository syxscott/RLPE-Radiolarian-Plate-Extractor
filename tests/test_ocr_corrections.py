"""Tests for :mod:`rlpe.ocr_corrections`.

Locks in the correction layer's behaviour so future changes to the global
``CORRECTIONS`` dict or any paper's ``PAPER_WHITELIST`` entry are caught
by CI. Each test case corresponds to a (pred, gold) pair actually observed
in the 9-paper corpus — see :mod:`rlpe.ocr_corrections` for the source
derivation.
"""
from __future__ import annotations

import pytest

from rlpe.ocr_corrections import (
    CORRECTIONS,
    PAPER_WHITELIST,
    apply_corrections,
    known_corrections,
    whitelist_for_paper,
)


# ---------------------------------------------------------------------------
# apply_corrections: empty / None input
# ---------------------------------------------------------------------------

def test_apply_corrections_none_returns_empty():
    assert apply_corrections(None) == ""


def test_apply_corrections_empty_returns_empty():
    assert apply_corrections("") == ""


def test_apply_corrections_whitespace_only_returns_empty():
    assert apply_corrections("   \t\n  ") == ""


# ---------------------------------------------------------------------------
# apply_corrections: paper-specific whitelist (hollis2006)
# ---------------------------------------------------------------------------

def test_hollis_gloss_stripped():
    """LLM emits the verbose "(cf. Theocosphaerella rotunda)" gloss on
    the group-b label; gold uses the bare group label."""
    pred = "Haliomma gr. b. (cf. Theocosphaerella rotunda)"
    assert apply_corrections(pred, "hollis2006") == "Haliomma gr. b"


def test_hollis_gr_b_trailing_period():
    """The period after "gr. b." is the only diff; rstrip leaves it
    alone in strict mode but the correction layer maps it."""
    pred = "Haliomma gr. b."
    assert apply_corrections(pred, "hollis2006") == "Haliomma gr. b"


def test_hollis_spumellarian_gen_to_indet():
    """LLM truncates the long-form "gen. et sp. indet" to bare "gen"."""
    pred = "Spumellarian gen"
    assert apply_corrections(pred, "hollis2006") == "Spumellarian indet"


def test_hollis_axoprunum_aff_restored():
    """LLM drops the open-nomen "aff." qualifier on Axoprunum."""
    pred = "Axoprunum bispiculum"
    assert apply_corrections(pred, "hollis2006") == "Axoprunum aff. bispiculum"


def test_hollis_foreman_suffix_dropped():
    """LLM emits the author suffix "Foreman"; gold drops it for the
    plate-label convention used in this paper."""
    pred = "Theocorys? phyzella Foreman"
    assert apply_corrections(pred, "hollis2006") == "Theocorys? phyzella"


# ---------------------------------------------------------------------------
# apply_corrections: paper-specific whitelist (feng2007)
# ---------------------------------------------------------------------------

def test_feng_trilonche_pseudo_restored():
    """LLM drops the "pseudo-" prefix on Trilonche pseudocimelia."""
    pred = "Trilonche cimelia"
    assert apply_corrections(pred, "feng2007") == "Trilonche pseudocimelia"


# ---------------------------------------------------------------------------
# apply_corrections: paper-specific whitelist (beccaro2006)
# ---------------------------------------------------------------------------

def test_beccaro_pseudoeucyrtis_group_b_restored():
    """Parser drops the group letter " B" on Pseudoeucyrtis."""
    pred = "Pseudoeucyrtis sp."
    assert apply_corrections(pred, "beccaro2006") == "Pseudoeucyrtis sp. B"


# ---------------------------------------------------------------------------
# apply_corrections: global substring corrections
# ---------------------------------------------------------------------------

def test_global_archaeodictyomitracf_split():
    """LLM fuses 'cf.' into the genus token."""
    pred = "Archaeodictyomitracf vulgaris"
    out = apply_corrections(pred, "bandini2011")
    assert out == "Archaeodictyomitra cf. vulgaris"


def test_global_transhsuumcf_split():
    """Same fused-genus pattern, different genus."""
    pred = "Transhsuumcf maxwelli"
    out = apply_corrections(pred, "bandini2011")
    assert out == "Transhsuum cf. maxwelli"


# ---------------------------------------------------------------------------
# apply_corrections: passthrough / no-op cases
# ---------------------------------------------------------------------------

def test_passthrough_when_no_rule_fires():
    """A normal pred string with no known OCR error returns unchanged."""
    pred = "Pseudodictyomitra sp."
    assert apply_corrections(pred, "bandini2011") == pred


def test_unknown_paper_id_skips_whitelist():
    """An unknown paper_id must still apply global corrections but skip
    the per-paper whitelist."""
    pred = "Haliomma gr. b. (cf. Theocosphaerella rotunda)"
    # Without a paper_id, the hollis whitelist does NOT fire.
    assert apply_corrections(pred) == pred


# ---------------------------------------------------------------------------
# API surface / introspection
# ---------------------------------------------------------------------------

def test_corrections_dict_is_json_serialisable():
    """The CORRECTIONS dict must round-trip through json.dumps."""
    import json
    encoded = json.dumps(CORRECTIONS)
    decoded = json.loads(encoded)
    assert decoded == CORRECTIONS


def test_paper_whitelist_is_json_serialisable():
    """The PAPER_WHITELIST values are lists of (str, str) tuples — verify
    the JSON-friendly form (lists of [src, dst] pairs) round-trips."""
    import json
    payload = {
        pid: [[src, dst] for src, dst in rules]
        for pid, rules in PAPER_WHITELIST.items()
    }
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded == payload


def test_known_corrections_returns_quadruples():
    """known_corrections() should yield (src, dst, freq, rationale) tuples."""
    rows = known_corrections()
    assert rows, "expected at least one entry"
    for row in rows:
        assert len(row) == 4
        src, dst, freq, rationale = row
        assert isinstance(src, str)
        assert isinstance(dst, str)
        assert isinstance(freq, int) and freq >= 1
        assert isinstance(rationale, str) and rationale


def test_whitelist_for_paper_unknown_returns_empty():
    assert whitelist_for_paper("does-not-exist") == []


def test_whitelist_for_paper_none_returns_empty():
    assert whitelist_for_paper(None) == []


def test_whitelist_for_paper_known_returns_list():
    rows = whitelist_for_paper("hollis2006")
    assert rows, "hollis2006 should have at least one whitelist entry"
    for src, dst in rows:
        assert isinstance(src, str)
        assert isinstance(dst, str)
