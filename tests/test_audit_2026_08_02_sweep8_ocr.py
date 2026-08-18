"""Sweep 8 (2026-08-02 audit follow-up): OCR quality cleanup.

Three small fixes landed in this sweep:

1. **N6** — paddleocr 3.x can return ``rec_texts`` longer than
   ``dt_polys`` (text recognised but no detector box). Previously the
   dict-branch loop ``break``'d on the first orphan and silently dropped
   every subsequent text from the output. Now ``continue`` + ``logger.
   warning(...)`` so operators can spot the version mismatch.

2. **O2** — ``OCRBackend.recognize`` had a bare
   ``except Exception: return []`` that silently swallowed backend
   errors. Now ``logger.warning(..., exc_info=True)`` so the failure
   is actionable in the run log.

3. **C3** — ``_normalize_paddle_result`` recursed into itself for each
   element of a list-of-dicts. Extracted the dict branch into a new
   ``_normalize_paddle_dict`` helper so the list-of-dicts branch is
   iterative. The recursion was always 1-deep but bounded-stack code
   is fragile.

These tests pin the design so a future refactor doesn't silently
regress.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.ocr import OCRBackend


# ----- N6: silent box drops -----------------------------------------------


class TestSweep8N6SilentBoxDrops:
    """N6 — orphan rec_texts (no dt_polys) must log a warning, NOT break
    silently."""

    def test_orphan_text_logs_warning(self, caplog):
        """paddleocr 3.x result with more texts than boxes must log a
        warning AND emit the texts that DO have boxes (the orphans are
        skipped, the matched ones pass through)."""
        result = {
            "rec_texts": ["alpha", "beta", "gamma"],
            "rec_scores": [0.99, 0.95, 0.90],
            "dt_polys": [
                [[0, 0], [10, 0], [10, 10], [0, 10]],  # alpha has a box
            ],
        }
        with caplog.at_level(logging.WARNING, logger="rlpe.ocr"):
            out = OCRBackend._normalize_paddle_result(result)
        # The two texts WITHOUT boxes are skipped (not in `out`).
        assert len(out) == 1
        assert out[0][1] == "alpha"
        # The skip is logged with actionable detail.
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "paddleocr" in r.message.lower()
        ]
        assert len(warnings) >= 1, (
            "Expected at least one paddleocr-version-mismatch warning; "
            "got none. The N6 fix may have reverted to silent ``break``."
        )
        # The warning must mention both counts so the operator can
        # diagnose at a glance.
        msg = warnings[0].message
        assert "3" in msg and "1" in msg, (
            f"warning should mention rec_texts=3 vs dt_polys=1; got: {msg}"
        )

    def test_no_orphan_no_warning(self, caplog):
        """When all texts have boxes, no warning fires."""
        result = {
            "rec_texts": ["alpha", "beta"],
            "rec_scores": [0.99, 0.95],
            "dt_polys": [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[0, 20], [10, 20], [10, 30], [0, 30]],
            ],
        }
        with caplog.at_level(logging.WARNING, logger="rlpe.ocr"):
            out = OCRBackend._normalize_paddle_result(result)
        assert len(out) == 2
        # No paddleocr warnings expected.
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "paddleocr" in r.message.lower()
        ]
        assert not warnings, (
            f"Unexpected paddleocr warning on matched-shape result: "
            f"{[r.message for r in warnings]}"
        )

    def test_break_replaced_with_continue_source_guard(self):
        """Pin the structural fix: the dict branch must NOT contain
        ``break`` on the orphan-text condition (which would silently
        drop every subsequent text)."""
        from rlpe import ocr as ocr_mod

        src = Path(ocr_mod.__file__).read_text(encoding="utf-8")
        # The N6 fix changed ``break`` to ``continue`` inside the
        # rec_texts loop. The ``continue`` follows a multi-line
        # ``logger.warning(...)`` call so a 200-char window is too
        # tight; widen to 1000 chars.
        assert "if i >= len(polys):" in src
        idx = src.find("if i >= len(polys):")
        snippet = src[idx:idx + 1000]
        assert "continue" in snippet, (
            "Orphan-text handling no longer `continue`s; the N6 fix "
            "may have regressed to silent `break`"
        )
        # The forbidden ``break`` pattern must not be present in the
        # orphan-handling block.
        assert "    break\n" not in snippet, (
            "Orphan-text handling contains `break`; N6 silent-drop "
            "regression — every text after the first orphan is dropped"
        )


# ----- O2: silent OCR exception swallowing --------------------------------


class TestSweep8O2OCRExceptionLogging:
    """O2 — OCR backend failures must be logged, not silently swallowed."""

    def test_recognize_logs_on_failure(self, caplog, monkeypatch):
        """When the OCR backend raises, ``recognize`` must log a warning
        with the exception type + traceback before returning ``[]``."""
        # Build a fake engine that always raises. Use the easyocr
        # backend branch (``readtext``) so we don't need to mock
        # paddleocr's ``.ocr(image, cls=True)`` call signature.
        class _RaisingEngine:
            def readtext(self, image):
                raise RuntimeError("PaddleOCR CUDA OOM")

        # Construct via __init__ so all attrs (backend, lang, _lock, _engine)
        # are set, then swap _engine.
        backend = OCRBackend(backend="easyocr", use_gpu=False, lang="en")
        backend._engine = _RaisingEngine()
        # Build a tiny dummy image.
        import numpy as np

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with caplog.at_level(logging.WARNING, logger="rlpe.ocr"):
            tokens = backend.recognize(img)
        assert tokens == []
        # The exception must be logged with both type and message.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("RuntimeError" in r.message for r in warnings), (
            "OCR exception type not in warning message; O2 fix regressed"
        )
        assert any("PaddleOCR CUDA OOM" in r.message for r in warnings), (
            "OCR exception message not in warning; O2 fix regressed"
        )
        # exc_info=True must be set so the traceback is captured.
        assert any(r.exc_info is not None for r in warnings), (
            "OCR warning logged without exc_info — operator can't see traceback"
        )


# ----- C3: recursive _normalize_paddle_result → iterative ----------------


class TestSweep8C3IterativeNormalize:
    """C3 — list-of-dicts must be normalised iteratively."""

    def test_helper_extract_for_dict_branch(self):
        """``_normalize_paddle_dict`` must exist as a separate helper so
        the list-of-dicts branch can call it iteratively."""
        assert hasattr(OCRBackend, "_normalize_paddle_dict"), (
            "_normalize_paddle_dict helper missing — C3 fix regressed; "
            "list-of-dicts branch may have re-recursed"
        )

    def test_list_of_dicts_returns_flat_output(self):
        """A list-of-dicts result must produce a flat list of tuples
        (one per dict entry, not nested)."""
        result = [
            {
                "rec_texts": ["alpha"],
                "rec_scores": [0.99],
                "dt_polys": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
            },
            {
                "rec_texts": ["beta"],
                "rec_scores": [0.95],
                "dt_polys": [[[0, 20], [10, 20], [10, 30], [0, 30]]],
            },
        ]
        out = OCRBackend._normalize_paddle_result(result)
        assert len(out) == 2
        texts = {t for _, t, _ in out}
        assert texts == {"alpha", "beta"}

    def test_no_recursive_call_to_normalize_paddle_result(self):
        """Source guard: the list-of-dicts branch must NOT call
        ``_normalize_paddle_result`` recursively (the C3 fix)."""
        from rlpe import ocr as ocr_mod

        src = Path(ocr_mod.__file__).read_text(encoding="utf-8")
        # Find the list-of-dicts branch and assert it does NOT call
        # ``_normalize_paddle_result(d)`` for any element.
        list_branch = src.find("if isinstance(result, list):")
        assert list_branch > 0, "list-of-dicts branch missing"
        snippet = src[list_branch:list_branch + 800]
        assert "_normalize_paddle_result(" not in snippet.replace(
            "_normalize_paddle_result(\n", ""
        ).replace(
            # The top-level def line is allowed.
            "def _normalize_paddle_result(", "def _normalize_paddle_dict("
        ), (
            "list-of-dicts branch still recurses into "
            "_normalize_paddle_result(d) — C3 fix regressed"
        )
        # And it MUST call the iterative helper.
        assert "_normalize_paddle_dict(" in snippet, (
            "list-of-dicts branch no longer calls the iterative "
            "_normalize_paddle_dict helper — refactor incomplete"
        )


# ----- End-to-end: paddleocr 2.x tuple/list shape still works -----------


class TestSweep8RegressionCompat:
    """Make sure the refactor didn't break the 2.x / 3.x compatibility
    for matched-shape results."""

    def test_paddleocr_2x_tuple(self):
        """Paddleocr 2.x tuple shape: ``(list_of_lines, None)`` where
        each line is ``[box, (text, conf)]``."""
        result = (
            [
                [  # one line = [box, payload]
                    [[0, 0], [10, 0], [10, 10], [0, 10]],  # box
                    ("alpha", 0.99),                        # (text, conf)
                ],
            ],
            None,
        )
        out = OCRBackend._normalize_paddle_result(result)
        assert len(out) == 1
        assert out[0][1] == "alpha"

    def test_paddleocr_3x_dict_matched(self):
        result = {
            "rec_texts": ["alpha", "beta"],
            "rec_scores": [0.99, 0.95],
            "dt_polys": [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[0, 20], [10, 20], [10, 30], [0, 30]],
            ],
        }
        out = OCRBackend._normalize_paddle_result(result)
        assert len(out) == 2
        assert {t for _, t, _ in out} == {"alpha", "beta"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])