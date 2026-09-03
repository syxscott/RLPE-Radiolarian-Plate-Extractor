"""Regression tests for audit 2026-08-01 batch W1 — C5 ocr_corrections.py design contract lock."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_MODULE_PATH = _SRC / "rlpe" / "ocr_corrections.py"

# The two rules the CORRECTIONS table is locked to. Adding or removing an
# entry in `_CORRECTION_FREQ` MUST come with an update to this list.
_LOCKED_KEYS = ("Archaeodictyomitracf ", "Transhsuumcf ")


class TestOCRCorrectionsContract:
    """C5: CORRECTIONS is intentionally minimal — lock the contract.

    The :data:`CORRECTIONS` substring table is locked to 2 entries.
    Character-level OCR confusions (l↔1, I↔l, long-vowel marks) are
    handled by a SEPARATE pre-pass function
    ``_normalize_ocr_chars`` (added 2026-09-03 audit BLOCKER-#6) with
    look-around guards. The character confusions that were
    historically NOT in this layer (``0↔O`` and ``rn↔m``) remain
    downstream in ``pipeline._norm_species``; adding them to
    :data:`CORRECTIONS` would bloat the table from 2 to ~12 entries
    and trigger false positives on tokens like "iuncus" or
    "P0lacsekia".

    These tests fail loudly if a future contributor grows or shrinks
    the dict silently.
    """

    def test_two_existing_rules(self):
        from rlpe.ocr_corrections import apply_corrections

        out = apply_corrections("Archaeodictyomitracf X", "any_paper")
        assert "Archaeodictyomitra cf." in out, out

        out = apply_corrections("Transhsuumcf Y", "any_paper")
        assert "Transhsuum cf." in out, out

    def test_no_character_pair_corrections(self):
        from rlpe.ocr_corrections import apply_corrections

        # 1 <-> l confusion: NOT corrected by this layer.
        assert apply_corrections("Cenospaera", "any") == "Cenospaera"
        # 0 <-> O confusion: NOT corrected by this layer.
        assert apply_corrections("P0lacsekia", "any") == "P0lacsekia"

    def test_dict_size_locked(self):
        from rlpe.ocr_corrections import CORRECTIONS

        assert len(CORRECTIONS) == 2, (
            "CORRECTIONS size changed; update "
            "tests/test_audit_2026_08_01_ocr_corrections_lock.py deliberately. "
            f"Got: {sorted(CORRECTIONS)}"
        )
        assert sorted(CORRECTIONS) == sorted(_LOCKED_KEYS)

    def source_guard_test_no_added_corrections(self):
        """Source-level guard: count live `_CORRECTION_FREQ` entries."""
        src = _MODULE_PATH.read_text(encoding="utf-8")

        # Isolate the dict literal body (commented-out candidate rules live
        # inside it and must NOT be counted).
        m = re.search(
            r"^_CORRECTION_FREQ[^=]*=\s*\{(?P<body>.*?)^\}",
            src,
            re.M | re.S,
        )
        assert m is not None, "_CORRECTION_FREQ dict literal not found"
        body = m.group("body")

        # Live entries look like:  `    "<key>": (`  — commented ones start `#`.
        keys = re.findall(r'^\s+"([^"]+)":\s*\(', body, re.M)
        assert keys == list(_LOCKED_KEYS), (
            "_CORRECTION_FREQ entries changed; this is a DESIGN CHOICE lock. "
            "Update tests/test_audit_2026_08_01_ocr_corrections_lock.py "
            f"if the change is intentional. Found: {keys}"
        )

        # The explicit C5 contract comment must stay in place.
        assert "C5 lock (audit 2026-08-01)" in src
        assert "INTENTIONALLY MINIMAL" in src

    def test_source_guard_no_added_corrections(self):
        self.source_guard_test_no_added_corrections()
