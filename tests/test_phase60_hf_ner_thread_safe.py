"""Tests for Phase 60 Plan 3 — Bug 3.8: TaxonRecognizer._hf_ner init
was OUTSIDE the lazy-init lock — two concurrent threads could each
create their own HuggingFace pipeline (real memory leak + thread
race on the global ``self._hf_ner`` assignment).

The fix moves the ``pipeline(...)`` call INSIDE ``with self._lock:``
so only one thread initialises the pipeline even under concurrent
``_lazy_init()`` calls.

The test inspects the source code (rather than spawning real HF
pipelines, which would require GPU + the transformers library) to
confirm the init line sits inside the ``with self._lock:`` block.
This is a deterministic source-guard: any future code move that
breaks the contract will fail the test before it ships.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_hf_ner_init_under_lock():
    """The ``pipeline(...)`` call that initialises ``self._hf_ner``
    must sit INSIDE the ``with self._lock:`` block in
    ``TaxonRecognizer._lazy_init``."""
    import inspect

    from rlpe.taxon import TaxonRecognizer

    src = inspect.getsource(TaxonRecognizer._lazy_init)
    lines = src.splitlines()
    # Find the lock entry line.
    lock_line_idx = None
    for i, line in enumerate(lines):
        if "with self._lock" in line and ":" in line:
            lock_line_idx = i
            break
    assert lock_line_idx is not None, (
        "Could not find `with self._lock:` block in _lazy_init:\n" + src
    )
    # Find the matching `pipeline(` call (HF init). The pattern is
    # usually split across lines (``self._hf_ner = pipeline(\n
    #     task="token-classification",\n    ...)``), so we accept
    # either the ``pipeline(`` line or a line containing the
    # ``self._hf_ner = pipeline`` assignment.
    pipeline_line_idx = None
    for i, line in enumerate(lines):
        if "self._hf_ner = pipeline" in line or "self._hf_ner = pipeline(" in line:
            pipeline_line_idx = i
            break
    # Fallback: any line starting ``pipeline(`` after the import.
    if pipeline_line_idx is None:
        for i, line in enumerate(lines):
            if line.strip().startswith("pipeline("):
                pipeline_line_idx = i
                break
    assert pipeline_line_idx is not None, (
        "Could not find `pipeline(` call in _lazy_init:\n" + src
    )
    assert pipeline_line_idx > lock_line_idx, (
        "pipeline(...) call at line %d sits BEFORE the lock entry "
        "at line %d — the HF init is not protected by self._lock.\n%s"
        % (pipeline_line_idx, lock_line_idx, src)
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])