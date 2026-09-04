"""Regression: audit 2026-09-04 geo-6 —
:func:`rlpe.paleo_reconstruction.infer_plate_id` used ``k in
loc_lower`` for locality-substring matching. The country keyword
``"oman"`` is a substring of ``"Romania"`` (and ``"Romanian"``,
``"romani"`` etc.) — so any locality mentioning Romania / Bucovina
(site of real Cretaceous radiolarian studies) was misclassified to
the ``"Arabia"`` plate, producing an Africa/Arabia paleo
reconstruction instead of the correct Eurasian one.

The substring scan also triggered other false positives:
    * ``"iran"`` in ``"trans-iranian"``  → would falsely match Iran
    * ``"china"`` in ``"machina"``       → would falsely match China
    * ``"korea"`` in ``"koreana"``       → would falsely match Korea

Fix contract: country-keyword matches in locality text require a
word boundary on BOTH sides of the keyword — so "Sultanate of
Oman" still matches (the keyword is a standalone token) but
"Romania" does NOT match "oman".
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlpe.paleo_reconstruction import infer_plate_id  # noqa: E402


class TestSubstringFalsePositivesRejected:
    def test_romania_not_matched_as_oman(self):
        # Audit 2026-09-04 geo-6: "oman" in "Romania" was matched
        # because the locality-substring scan used ``k in loc_lower``
        # (substring, not word-boundary). The Romanian Cretaceous
        # Carpathian site (Bucovina) was mis-routed to the "Arabia"
        # plate, producing wrong paleo coordinates.
        plate = infer_plate_id(locality="Bucovina, Romania")
        assert plate != "Arabia", (
            f"Romania must not match 'oman' substring; got {plate!r}"
        )
        assert plate == "Eurasia", plate

    def test_romania_explicit_country(self):
        plate = infer_plate_id(country="Romania")
        assert plate == "Eurasia", plate

    def test_macedonia_not_matched_as_donia(self):
        # "Macedonia" contains "don" (a potential keyword for "London"?).
        # This is a smoke test that word-boundary is enforced
        # regardless of the keyword being a "real" word.
        plate = infer_plate_id(locality="Republic of Macedonia")
        # Macedonia is not in our country table; we just verify it
        # does NOT match some unrelated substring (e.g. "don" → Don
        # basin).
        assert plate != "Don", plate


class TestSubstringTruePositivesPreserved:
    def test_oman_still_matches(self):
        # Sanity: standalone "Oman" still maps to Arabia.
        plate = infer_plate_id(country="Oman")
        assert plate == "Arabia", plate

    def test_oman_in_multiword_locality(self):
        # "Sultanate of Oman" — the keyword is a standalone token.
        plate = infer_plate_id(locality="Sultanate of Oman")
        assert plate == "Arabia", plate

    def test_italy_in_locality(self):
        plate = infer_plate_id(locality="Northern Italy")
        assert plate == "Adria", plate


class TestNoOtherSubstringCollisions:
    def test_iran_not_matched_in_irani(self):
        # "Iran" should not match a random "irani" token.
        plate = infer_plate_id(locality="Pezhanirani section")
        assert plate != "Iran", plate

    def test_korea_not_matched_in_koreana(self):
        plate = infer_plate_id(locality="flora koreana")
        assert plate != "Amuria", plate
