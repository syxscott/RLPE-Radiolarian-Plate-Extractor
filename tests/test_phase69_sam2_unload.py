"""Phase 69 — SAM2 predictor unload between web jobs.

Audit 2026-08-02 (N5): on the web server each job constructs a fresh
``RadiolarianPipeline`` → fresh ``PanelSegmenter`` → fresh SAM2
predictor (~900 MB on the GPU). When the job finishes the pipeline
object goes out of scope and Python's refcount drops the predictor
— but PyTorch's CUDA caching allocator does NOT release the cached
blocks back to the driver until ``torch.cuda.empty_cache()`` is called.
With sequential uploads the GPU footprint grew linearly until OOM on
the 5th-6th job.

Phase 69 fix: ``PanelSegmenter.unload_sam2()`` drops the predictor
reference AND calls ``torch.cuda.empty_cache()``. The web API's job
runner calls this in the ``finally`` block (after both success and
failure paths) so memory is released regardless of outcome.

These tests verify:
- The method is a no-op when SAM2 was never initialised.
- The method drops ``_predictor`` after initialisation (mocked).
- The method handles missing torch gracefully.
- The method handles missing CUDA gracefully.
- The method calls ``torch.cuda.empty_cache()`` exactly once when CUDA
  is available.
- The web API's ``_run_job`` calls ``unload_sam2()`` on every exit
  path (success, cancelled, exception) via the ``finally`` block.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.segmentation import PanelSegmenter, SegmentationConfig


class TestUnloadSam2:
    def test_noop_when_predictor_uninitialised(self):
        """If SAM2 was never loaded, unload_sam2 is a cheap no-op."""
        seg = PanelSegmenter()
        assert seg._predictor is None
        # Must not raise (no torch, no CUDA, no predictor).
        seg.unload_sam2()
        assert seg._predictor is None

    def test_drops_predictor_after_initialisation(self):
        """After unload, _predictor is None so the next segment call
        rebuilds from scratch instead of adding to GPU memory."""
        seg = PanelSegmenter()
        # Simulate a previously-loaded predictor (we don't actually
        # build SAM2 because the sam2 package isn't always installed).
        fake_predictor = MagicMock(name="SAM2ImagePredictor")
        seg._predictor = fake_predictor
        # Patch torch.empty_cache so we don't depend on CUDA.
        with patch("torch.cuda.empty_cache") as mock_empty_cache, \
             patch("torch.cuda.is_available", return_value=True):
            seg.unload_sam2()
        # Predictor reference dropped.
        assert seg._predictor is None
        # CUDA cache released.
        mock_empty_cache.assert_called_once()

    def test_handles_missing_torch_gracefully(self):
        """If torch isn't importable (CPU-only env), unload still
        succeeds — it just can't call empty_cache."""
        seg = PanelSegmenter()
        seg._predictor = MagicMock(name="SAM2ImagePredictor")
        # Block the torch import inside unload_sam2.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch not installed in this env")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            seg.unload_sam2()
        # Predictor still dropped even though empty_cache couldn't run.
        assert seg._predictor is None

    def test_handles_missing_cuda_gracefully(self):
        """If torch is importable but CUDA isn't (CPU-only runtime),
        unload still drops the predictor."""
        seg = PanelSegmenter()
        seg._predictor = MagicMock(name="SAM2ImagePredictor")
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.cuda.empty_cache") as mock_empty_cache:
            seg.unload_sam2()
        assert seg._predictor is None
        # empty_cache must NOT be called when CUDA is unavailable.
        mock_empty_cache.assert_not_called()

    def test_safe_to_call_twice(self):
        """Idempotent: calling unload_sam2 twice in a row is fine
        (matches the design — the runner may call it on cancel +
        again in finally)."""
        seg = PanelSegmenter()
        seg._predictor = MagicMock(name="SAM2ImagePredictor")
        with patch("torch.cuda.is_available", return_value=False):
            seg.unload_sam2()
            seg.unload_sam2()  # second call is a no-op
        assert seg._predictor is None

    def test_empty_cache_exception_is_swallowed(self):
        """If ``empty_cache`` raises (e.g. CUDA driver in a weird state),
        unload still completes — never propagate from cleanup."""
        seg = PanelSegmenter()
        seg._predictor = MagicMock(name="SAM2ImagePredictor")
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache",
                   side_effect=RuntimeError("CUDA driver hung")):
            # Must not raise.
            seg.unload_sam2()
        # Predictor was still dropped even though empty_cache failed.
        assert seg._predictor is None


class TestWebApiUnloadsSam2:
    """Pin the design: the web API MUST call ``unload_sam2()`` in the
    job runner's finally block. This is the structural fix for the
    5th-6th-job OOM — without this guard, future refactors of the
    runner would silently regress to pinning N × ~900MB."""

    def test_run_job_finally_calls_unload_sam2(self):
        """Source guard: ``_run_job`` in api/app.py references
        ``unload_sam2()`` inside its finally block (the block at line
        ~2587 that owns ``stop_hb``/``hb_thread.join`` — NOT the small
        ``finally`` inside ``_run_job_with_concurrency``)."""
        src = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "api" / "app.py").read_text(
            encoding="utf-8"
        )
        # The structural fix must be present somewhere in the file.
        assert "unload_sam2()" in src, (
            "unload_sam2() call missing from api/app.py — sequential web "
            "uploads will pin N × ~900MB on the GPU until OOM"
        )
        # There are 3 ``stop_hb.set()`` sites in the file — two are in
        # pre-flight cancel checks (lines ~2239/2245) and the LAST one
        # is the cleanup inside ``finally``. Anchor on the last one to
        # pin the structural relationship to the finally block.
        anchor = "stop_hb.set()"
        last_idx = src.rfind(anchor)
        assert last_idx > 0, "stop_hb.set() not found — refactored job runner?"
        # The runner finally block starts BEFORE the ``stop_hb.set()``
        # call (the cleanup preamble). Sweep 7 (N2) added ~15 lines of
        # MiniMax_fallback_handler cleanup between ``finally:`` and
        # ``stop_hb.set()``; widen the window to 3000 chars to keep
        # this source guard robust to future cleanups.
        window_start = max(0, last_idx - 3000)
        window = src[window_start:last_idx + 200]
        assert "finally:" in window, (
            "finally: not found near stop_hb.set() — refactor may have "
            "moved cleanup out of finally"
        )
        assert "unload_sam2()" in window, (
            "unload_sam2() must be inside the _run_job finally block "
            "(the same one that calls stop_hb.set()). Otherwise "
            "cancellation/failure paths leak SAM2 memory."
        )

    def test_run_job_binds_pipeline_for_finally(self):
        """The pipeline variable must be bound BEFORE ``.run()`` so the
        ``finally`` block can reference it on the exception path.

        Without this binding, ``unload_sam2()`` only runs on the
        success path (where ``_pipeline`` is defined) and silently
        no-ops on cancellation / failure (where ``_pipeline`` was
        never assigned because ``.run()`` raised before assignment).
        The audit caught this exact class of leak in M-YO-1.
        """
        src = (Path(__file__).resolve().parents[1] / "src" / "rlpe" / "api" / "app.py").read_text(
            encoding="utf-8"
        )
        # Look for the binding pattern: ``_pipeline = RadiolarianPipeline(...)``.
        assert "_pipeline = RadiolarianPipeline(" in src, (
            "_pipeline binding missing — finally block can't reference the "
            "pipeline on exception paths"
        )
        # And the call must use ``_pipeline_for_cleanup`` (the locals()
        # lookup that handles the unbound-on-early-failure case).
        assert "_pipeline_for_cleanup" in src, (
            "finally block must look up _pipeline via locals().get() to "
            "handle the case where cfg validation failed before _pipeline "
            "was assigned"
        )


class TestSegmentationConfig:
    """Lock the segmentation config defaults that interact with
    unload_sam2 behaviour."""

    def test_use_sam2_default_is_true(self):
        """Default config has use_sam2=True so the audit-fix path
        (CUDA cache release) is actually exercised on the default
        pipeline."""
        cfg = SegmentationConfig()
        assert cfg.use_sam2 is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
