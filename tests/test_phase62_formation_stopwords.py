"""Phase 62 Plan 5 (Bug 5.5): formation regex must reject stopword prefixes.

The ``_FORMATION_RE`` regex (and its sibling ``_GROUP_RE`` /
``_MEMBER_RE``) accept any ``[A-Z][A-Za-z\\-]{0,30}?`` prefix before
the "Formation"/"Fm." keyword. The existing ``_strip_leading_article``
helper strips "The"/"A"/"An" only, so stopword prefixes like "In",
"Of", "And", "From", "Near", "By" survive and produce garbage
records like ``formation="In Group"`` / ``formation="Of Formation"``
/ ``group="By Group"``.

Examples of garbage the previous pipeline produced (verified
empirically against the live pipeline):

  - "In Group we found abundant fossils" → group="In Group"
  - "Of Group we infer a different age" → group="Of Group"
  - "By Group we mean the type section" → group="By Group"
  - "From Member we infer the contact"  → member="From Member"

The fix: extend ``_strip_leading_article`` (or add a sibling stopword
filter) to strip the additional stopword prefixes. We choose the
sibling-filter approach so the article-strip semantics stay
backward-compatible (the new filter only ADDS rejection of more
prefixes; it does not change the existing article-strip behaviour).
"""

from __future__ import annotations

from rlpe.geology_extraction import extract_geology_from_sections

_FORMATION_STOPWORD_PREFIXES = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "and",
        "from",
        "near",
        "by",
    }
)


def _first_word_of_prefix(name: str, keyword: str) -> str:
    for kw in (keyword, keyword.rstrip(".")):
        if name.endswith(kw):
            stripped = name[: -len(kw)].rstrip()
            return stripped.split(" ", 1)[0].lower() if stripped else ""
    return name.split(" ", 1)[0].lower()


def _assert_no_stopword_prefix(record_value, keyword):
    if not record_value:
        return
    first_word = _first_word_of_prefix(record_value, keyword)
    assert first_word not in _FORMATION_STOPWORD_PREFIXES, (
        f"{keyword} {record_value!r} starts with stopword {first_word!r}"
    )


def test_group_rejects_in_prefix():
    """'In Group we found fossils' → group='In Group' must be dropped."""
    text = "In Group we found abundant fossils"
    records = extract_geology_from_sections(
        [{"text": text, "title": "t", "section_type": "geological_setting"}]
    )
    groups = [r.group for r in records if r.group]
    for gname in groups:
        _assert_no_stopword_prefix(gname, "Group")


def test_group_rejects_of_prefix():
    """'Of Group we infer age' → group='Of Group' must be dropped."""
    text = "Of Group we infer a different age"
    records = extract_geology_from_sections(
        [{"text": text, "title": "t", "section_type": "geological_setting"}]
    )
    groups = [r.group for r in records if r.group]
    for gname in groups:
        _assert_no_stopword_prefix(gname, "Group")


def test_group_rejects_by_prefix():
    """'By Group we mean the type section' → group='By Group' must be dropped."""
    text = "By Group we mean the type section"
    records = extract_geology_from_sections(
        [{"text": text, "title": "t", "section_type": "geological_setting"}]
    )
    groups = [r.group for r in records if r.group]
    for gname in groups:
        _assert_no_stopword_prefix(gname, "Group")


def test_member_rejects_from_prefix():
    """'From Member we infer the contact' → member='From Member' must be dropped."""
    text = "From Member we infer the contact"
    records = extract_geology_from_sections(
        [{"text": text, "title": "t", "section_type": "geological_setting"}]
    )
    members = [r.member for r in records if r.member]
    for mname in members:
        _assert_no_stopword_prefix(mname, "Member")


def test_formation_rejects_of_prefix():
    """'Of Formation we will speak later' → formation='Of Formation' must be dropped."""
    text = "Of Formation we will speak later in this section"
    records = extract_geology_from_sections(
        [{"text": text, "title": "t", "section_type": "geological_setting"}]
    )
    formations = [r.formation for r in records if r.formation]
    for fname in formations:
        _assert_no_stopword_prefix(fname, "Formation")


def test_real_formation_name_passes():
    """Regression: 'The Fonzaso Formation' (article + real name) must
    still produce 'Fonzaso Formation' on the record."""
    text = "The Fonzaso Formation is siliceous limestone"
    records = extract_geology_from_sections(
        [{"text": text, "title": "t", "section_type": "geological_setting"}]
    )
    formations = [r.formation for r in records if r.formation]
    assert "Fonzaso Formation" in formations, (
        f"expected 'Fonzaso Formation' in records, got: {formations}"
    )


def test_real_group_name_passes():
    """Regression: 'Sicanian Group' must still pass."""
    text = "The Sicanian Group contains the Lower Member"
    records = extract_geology_from_sections(
        [{"text": text, "title": "t", "section_type": "geological_setting"}]
    )
    groups = [r.group for r in records if r.group]
    assert "Sicanian Group" in groups, f"expected 'Sicanian Group' in records, got: {groups}"
