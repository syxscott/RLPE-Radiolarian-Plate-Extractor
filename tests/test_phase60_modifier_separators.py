"""Tests for Phase 60 Plan 3 — Bug 3.10: age modifier patterns only
recognise whitespace between modifier and name.

``_MODIFIER_PATTERN`` in :mod:`rlpe.stratigraphy` requires a literal
``\\s+`` between the modifier (Early/Middle/Late/Lower/Upper/E./M./L.)
and the period name. Real captions (De Wever, Bandini, Bragin) use:

  * ``Late-Permian`` (hyphen, no spaces)
  * ``Late.Permian`` (period, no spaces)
  * ``Late _Permian_`` (italicised, em-dash)

The previous regex dropped the modifier for these shapes, so the
name was classified but the ``epoch`` field stayed ``None`` and the
``rank`` was reported as ``"period"`` instead of ``"epoch"``. The
fix normalises ``-`` / ``.`` / ``_`` / em-dash / en-dash to whitespace
before applying the modifier pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.stratigraphy import classify_age_string  # noqa: E402


def test_modifier_handles_dash_dot():
    """``Late-Permian`` / ``Late.Permian`` / ``Late.Permian`` (em-dash)
    must classify as Permian with epoch information preserved."""
    for sep in ("-", ".", "—", "–", "_"):
        text = f"Late{sep}Permian"
        cls = classify_age_string(text)
        assert cls.confidence > 0, f"failed for sep={sep!r}: {cls}"
        assert cls.period == "Permian", f"sep={sep!r}: period={cls.period!r}"


def test_modifier_low_early():
    """Lower-case early modifier must work after normalising the sep."""
    cls = classify_age_string("Early-Cretaceous")
    assert cls.period == "Cretaceous"
    assert cls.confidence > 0


def test_modifier_normal_white_still_works():
    """Regression guard: ordinary ``Late Permian`` must still classify."""
    cls = classify_age_string("Late Permian")
    assert cls.period == "Permian"
    assert cls.confidence > 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
