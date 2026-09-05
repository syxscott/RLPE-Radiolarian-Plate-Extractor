"""Regression guard: audit 2026-09-04 geo-3 — bare epoch / era names
(``"Cretaceous"``, ``"Miocene"``) in :data:`rlpe.geology_extraction._PALEO_KEYWORDS`
were triggering a ``"paleo"`` label on any section whose 400-char
window happened to contain one — even when the section was clearly
modern context ("Cretaceous basalt exposed in present-day France",
"Miocene volcanics in modern-day California"). The result: modern
coordinates were routed into the Rodrigues rotation as if they were
a Cretaceous site, producing wrong paleo_latitude values.

The fix (predates this test): keep only the QUALIFIED forms in the
keyword list — "in the eocene", "in cretaceous", etc. — so a bare
"Miocene" / "Cretaceous" alone is not enough to trigger paleo
classification. The qualified forms require a temporal preposition
("in the", "during the", "ago", "Ma") which appears in paleo
contexts but not in paleogeographic-descriptor contexts.

This file is a SOURCE GUARD. It does NOT exercise the runtime
classifier (already correct) — it pins the keyword list shape so
future contributors cannot re-introduce bare epoch names. A bare
"cretaceous" / "miocene" / etc. entry would silently re-introduce
the audit 2026-09-04 geo-3 regression.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.geology_extraction import _PALEO_KEYWORDS  # noqa: E402

# Epoch / era names that, if present in _PALEO_KEYWORDS as bare
# entries (without "in", "during", "the"), trigger the regression.
# Keep this list strict — only true ICZN / ICS rank names that are
# ALSO common paleogeographic descriptors.
_BARE_NAMES = (
    "cretaceous",
    "jurassic",
    "triassic",
    "permian",
    "devonian",
    "ordovician",
    "silurian",
    "cambrian",
    "carboniferous",
    "mesozoic",
    "cenozoic",
    "paleozoic",
    "paleogene",
    "neogene",
    "eocene",
    "oligocene",
    "miocene",
    "pliocene",
    "pleistocene",
)


class TestNoBareEpochInPaleoKeywords:
    def test_no_bare_epoch_entries(self):
        offenders = [kw for kw in _PALEO_KEYWORDS if kw.lower().strip() in _BARE_NAMES]
        assert not offenders, (
            "audit 2026-09-04 geo-3 regression: bare epoch/era names in "
            "_PALEO_KEYWORDS trigger paleo classification on paleogeographic "
            f"descriptors. Offending entries: {offenders!r}"
        )

    def test_only_qualified_forms_present(self):
        # Every entry must contain at least one temporal preposition
        # ("in", "during", "the") OR a numeric / qualifier cue ("Ma",
        # "ago", "reconstructed"). A bare noun alone is forbidden.
        qualified_markers = (
            " in ",
            "during",
            "the ",
            "ago",
            "ma ",
            "mya",
            "reconstructed",
            "was located",
            "was situated",
            "lay at",
            "deposition",
            "paleogeographic",
            "paleolatitude",
            "paleolongitude",
            "at the time",
            "at that time",
        )
        # A leading-space check handles the "in X" forms (e.g. "in triassic"
        # is preceded by a word boundary).
        violations = []
        for kw in _PALEO_KEYWORDS:
            kl = " " + kw.lower() + " "
            if any(m in kl for m in qualified_markers):
                continue
            # Bare names that survived the previous check are the
            # only things that should fail here.
            if kw.lower().strip() in _BARE_NAMES:
                violations.append(kw)
        assert not violations, (
            f"unqualified entries: {violations!r} — bare epoch/era names "
            "must be wrapped in 'in X' / 'during X' / 'the X' form"
        )

    def test_paleo_classifier_runtime_correct(self):
        # Belt-and-braces: the runtime classifier on a known-bad
        # input must not return "paleo".
        from rlpe.geology_extraction import _classify_coordinate_age

        text = (
            "Locality: present-day Sicily, Italy. "
            "Cretaceous basalt is exposed along the northern coast."
        )
        ms = text.index("Sicily")
        me = text.index("Italy") + len("Italy")
        label = _classify_coordinate_age(text, ms, me)
        assert label != "paleo", f"runtime mis-classified as paleo: {label!r}"

    def test_qualified_paleo_phrase_still_detected(self):
        # Sanity: the qualified form still triggers paleo.
        from rlpe.geology_extraction import _classify_coordinate_age

        text = "At 38.1N, 14.3E during the Cretaceous the region was at 12S paleolatitude."
        ms = text.index("38.1")
        me = text.index("14.3") + len("14.3")
        label = _classify_coordinate_age(text, ms, me)
        assert label == "paleo", label

    def test_keyword_list_is_tuple_of_strings(self):
        # Pin the data shape.
        assert isinstance(_PALEO_KEYWORDS, tuple)
        for kw in _PALEO_KEYWORDS:
            assert isinstance(kw, str)
            assert kw, "empty keyword would match anywhere"
