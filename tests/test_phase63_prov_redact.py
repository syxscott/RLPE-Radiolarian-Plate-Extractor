"""Tests for Phase 63 Plan 6 — Bug 6.9: ``provenance.config_snapshot``
must NOT contain ``MiniMax_api_key`` or any other API-key string.

Before: ``provenance.config_snapshot`` was the full
``PipelineConfig.model_dump()`` without redaction. If the user
started a run with ``MiniMax_api_key=sk-ant-api03-abcdef...``,
the API key leaked into ``run_output.json`` / ``run_output.provenance.json`` /
matches.jsonl, then to disk on the export. A reviewer sharing
the export for reproducibility would leak their key.

After: ``_config_snapshot`` walks the config dict, removes known
API-key fields by name (``MiniMax_api_key``,
``MiniMax_extra_headers``, ``_MiniMax_external_handler``, ...),
and applies ``rlpe.llm_backends._redact_api_keys`` to every
remaining string value so a stray ``sk-...`` token embedded in a
prompt / endpoint / header never reaches the export.

Plan 3 (round 13) already added ``_redact_api_keys`` for error
strings; this fix extends the same protection to the provenance
sidecar which persists across runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.provenance.stamp import _config_snapshot, build_provenance  # noqa: E402


class _FakeConfig:
    """Stand-in for PipelineConfig.

    Contains every known API-key-bearing attribute. The class exposes
    ``to_dict`` (mimicking PipelineConfig) but ALSO has a manual
    fallback to ``vars()`` for keys ``to_dict`` missed.
    """

    def __init__(
        self,
        minimax_api_key: str = "sk-ant-api03-" + "a" * 48,
        minimax_endpoint: str = "https://api.example.com",
        grobid_url: str = "http://localhost:8070",
        # Stray token embedded inside a prompt (should still be redacted)
        prompt_with_key: str = "Sonnet 4.7 sk-cp-" + "X" * 32 + " example",
        _MiniMax_external_handler: str = "sk-proj-" + "a" * 32,
        gemma_use_4bit: bool = True,
    ) -> None:
        self.MiniMax_api_key = minimax_api_key
        self.MiniMax_endpoint = minimax_endpoint
        self.grobid_url = grobid_url
        self.prompt_with_key = prompt_with_key
        self._MiniMax_external_handler = _MiniMax_external_handler
        self.gemma_use_4bit = gemma_use_4bit

    def to_dict(self) -> dict:
        return self.__dict__


def test_redact_api_keys_in_provenance_strips_known_fields():
    """The known API-key fields are removed entirely from config_snapshot."""
    cfg = _FakeConfig()
    snap = _config_snapshot(cfg)
    assert "MiniMax_api_key" not in snap, (
        f"MiniMax_api_key leaked into config_snapshot: {snap.get('MiniMax_api_key')!r}"
    )
    # _MiniMax_external_handler is private-prefixed; the fallback walker
    # does NOT include it (we filter out keys starting with ``_``).
    # Bottom line: the API key string is not present in the snapshot
    # under any name.
    flat = str(snap)
    assert "sk-" not in flat, (
        f"config_snapshot still contains an 'sk-' token: {flat!r}"
    )


def test_redact_api_keys_in_provenance_redacts_stray_tokens():
    """``sk-...`` tokens embedded in OTHER fields are also stripped."""
    cfg = _FakeConfig()
    snap = _config_snapshot(cfg)
    # The stray ``sk-cp-...`` inside ``prompt_with_key`` should be
    # replaced with [REDACTED].
    if "prompt_with_key" in snap:
        assert "sk-cp-" not in snap["prompt_with_key"], (
            f"stray sk-cp- token survived in prompt_with_key: {snap['prompt_with_key']!r}"
        )


def test_redact_api_keys_pseudo():
    """Walk a hand-built config dict; known API keys are stripped."""
    cfg_dict = {
        "MiniMax_api_key": "sk-ant-api03-" + "a" * 48,
        "MiniMax_endpoint": "https://api.example.com",
        "_MiniMax_external_handler": "sk-proj-" + "a" * 32,
        "gemma_use_4bit": True,
        "nested": {
            "minimax_api_key_LOWER": "sk-cp-" + "X" * 32,  # case-insensitive name match
            "endpoint": "https://api.example.com",
        },
    }
    snap = _config_snapshot(cfg_dict)
    flat = str(snap)
    assert "sk-" not in flat, (
        f"config_snapshot still contains 'sk-' token: {flat!r}. "
        "_redact_api_keys not applied recursively."
    )


def test_redact_api_keys_pipeline_config_class():
    """build_provenance() with a PipelineConfig-style dict produces a
    redacted config_snapshot."""
    cfg = {
        "MiniMax_api_key": "sk-ant-api03-" + "a" * 48,
        "MiniMax_endpoint": "https://api.example.com",
        "grobid_url": "http://localhost:8070",
    }
    prov = build_provenance(config=cfg)
    snap_str = str(prov.config_snapshot)
    assert "sk-ant-api03-" not in snap_str, (
        f"build_provenance leaked API key: {snap_str!r}"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
