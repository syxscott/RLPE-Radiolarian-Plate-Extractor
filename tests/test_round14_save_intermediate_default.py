"""Round 14 source-guard: save_intermediate default is OFF.

A 5-paper live smoke run (work/oa_smoke_v3, July 2026) produced 8993
``auto_fig_p*.json`` files = 116.9 GB of intermediate state that
**no eval script reads**. The ``save_intermediate`` flag defaulted to
``True``, so every pipeline invocation dumped per-region JSONs that
nobody consumed.

The fix flips the default to ``False``. The flag is still settable
per-config (e.g. for debugging the per-region chain) but the safe
default for production runs is OFF.

This test locks in two invariants:

  1. The default in :class:`PipelineConfig` is ``False``.
  2. The pipeline code path that writes the heavy per-region JSON is
     gated on ``self.config.save_intermediate`` (so the user's --save-
     intermediate CLI flag actually works and isn't a no-op).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_save_intermediate_default_is_false():
    """``PipelineConfig.save_intermediate`` must default to False.

    The previous default (True) caused every pipeline run to dump
    ~9000 per-region JSONs (= 117 GB on a 5-paper smoke run). No
    eval script reads those files; the canonical output is the
    per-row ``manifests/matches.jsonl`` + ``run_output.json``.
    """
    import inspect

    from rlpe.config import PipelineConfig

    # Inspect the dataclass field's default directly. The previous
    # test constructed ``PipelineConfig()`` but the constructor now
    # requires pdf_dir and work_dir (round 7+). Reading the
    # ``__dataclass_fields__`` entry gets the true default without
    # having to instantiate.
    field = PipelineConfig.__dataclass_fields__["save_intermediate"]
    assert field.default is False, (
        f"save_intermediate default flipped back to {field.default!r} — "
        f"every pipeline run will dump ~117 GB of unused intermediate "
        f"JSON. Set to False."
    )


def test_pipeline_intermediate_writes_are_gated():
    """The pipeline's intermediate-JSON write sites must be inside
    ``if self.config.save_intermediate:`` blocks — otherwise the
    user's CLI flag is a no-op and the 117 GB problem comes back."""
    pipeline_src = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "pipeline.py"
    src = pipeline_src.read_text(encoding="utf-8")

    # Find the two _process_region / _process_one_pdf_grobid write sites
    # that produce per-region or per-figure JSON. The structure is:
    #
    #     if self.config.save_intermediate:
    #         write_json(...)
    #
    # If this guard is missing the file gets written unconditionally.
    n_gated = src.count("if self.config.save_intermediate:")
    assert n_gated >= 2, (
        f"Expected at least 2 'if self.config.save_intermediate:' guards "
        f"in pipeline.py, found {n_gated}. The flag would be a no-op."
    )


def test_intermediate_path_size_documented():
    """The config docstring must warn about the disk cost so future
    contributors don't silently flip the default back to True.

    The audit found that flipping the default would re-introduce a
    ~117 GB smoke run, so the warning needs to stay near the field
    declaration.
    """
    config_src = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "config.py"
    src = config_src.read_text(encoding="utf-8")
    # Locate the save_intermediate field declaration and ensure the
    # following 12 lines mention disk cost. We allow a generous window
    # because the comment is multi-line.
    idx = src.find("save_intermediate: bool =")
    assert idx > 0, "Could not find save_intermediate field in config.py"
    window = src[idx : idx + 1500]
    assert "GB" in window or "disk" in window.lower(), (
        "save_intermediate field is missing a disk-cost warning. A future "
        "contributor might flip it back to True and re-introduce the 117 GB "
        "smoke-output bug. The docstring should mention the cost."
    )
