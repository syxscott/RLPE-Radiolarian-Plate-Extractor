"""Regression test: celery skeleton has been removed.

Phase 2026-08-16 cleanup: the celery worker had no production
callers (README admitted it was a skeleton). Jobs run via FastAPI
BackgroundTasks + on-disk persistence (Phase 49+).
"""

from __future__ import annotations

import pytest


def test_worker_module_deleted():
    """src/rlpe/worker/ should not exist after cleanup."""
    with pytest.raises(ImportError):
        from rlpe import worker  # noqa: F401
        from rlpe.worker import tasks  # noqa: F401


def test_no_celery_in_source_tree():
    """No src/ tree file should import celery anymore."""
    import os
    import pathlib

    src_root = pathlib.Path(__file__).parent.parent / "src"
    hits = []
    for py in src_root.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        if "from celery" in text or "import celery" in text:
            hits.append(str(py.relative_to(src_root)))
    assert hits == [], f"unexpected celery references in src/: {hits}"
