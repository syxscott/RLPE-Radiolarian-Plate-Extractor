"""Phase 61 Plan 4 (Bug 4.3): --deterministic flag for reproducible runs.

All LLM calls previously used temperature=0.1 / do_sample=True; same paper
+ same code produced different outputs every run, jittering F1 by
2-5 points.

The fix exposes ``--deterministic`` and PipelineConfig.extra["deterministic"];
when True, every backend sets temperature=0.0 and do_sample=False and
seeds Python / Torch / numpy RNGs.
"""

from __future__ import annotations

import pytest

from rlpe.config import _KNOWN_EXTRA_KEYS, PipelineConfig


def test_deterministic_in_known_keys():
    """PipelineConfig must accept ``deterministic`` without warning."""
    cfg = PipelineConfig(pdf_dir="/tmp/a", work_dir="/tmp/b")
    assert "deterministic" in _KNOWN_EXTRA_KEYS
    # Setting it should not raise the unknown-key warning path.
    cfg.extra["deterministic"] = True
    assert cfg.extra.get("deterministic") is True


def test_deterministic_sets_temperature_zero():
    """resolve_deterministic_kwargs returns temperature=0 / do_sample=False
    when the flag is True (regardless of default)."""
    from rlpe.llm_backends import resolve_deterministic_kwargs

    base = {"temperature": 0.7, "do_sample": True, "seed": None}
    out = resolve_deterministic_kwargs(base, deterministic=True)
    assert out["temperature"] == 0.0
    assert out["do_sample"] is False
    assert out["seed"] == 42


def test_deterministic_off_keeps_base():
    """When the flag is False, base kwargs pass through unchanged."""
    from rlpe.llm_backends import resolve_deterministic_kwargs

    base = {"temperature": 0.7, "do_sample": True, "seed": 7}
    out = resolve_deterministic_kwargs(base, deterministic=False)
    assert out == base


def test_cli_flag_exists():
    """CLI must accept ``--deterministic``."""
    import argparse

    # Import the CLI parser builder without running main().
    from rlpe.cli import build_parser

    parser = build_parser()
    # Parse minimal args.
    ns = parser.parse_args(["--pdf-dir", "/tmp/a", "--work-dir", "/tmp/b", "--deterministic"])
    assert ns.deterministic is True
    ns2 = parser.parse_args(["--pdf-dir", "/tmp/a", "--work-dir", "/tmp/b"])
    assert ns2.deterministic is False
