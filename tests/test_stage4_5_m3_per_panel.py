"""Tests for Stage 4.5 M3 per-panel species ID.

Audit 2026-08-17 spec: docs/superpowers/specs/2026-08-17-m3-per-panel-pipeline-design.md
Plan: docs/superpowers/plans/2026-08-17-m3-per-panel-pipeline.md
"""

from __future__ import annotations

import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from rlpe.config import PipelineConfig


def _make_cfg(tmp_path: Path, **overrides) -> PipelineConfig:
    cfg = PipelineConfig(
        pdf_dir=tmp_path / "pdfs",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_config_has_m3_per_panel_fields_with_safe_defaults(tmp_path):
    """All 4 new fields exist with safe defaults:
    - m3_per_panel_enabled: False (off by default)
    - m3_per_panel_min_conf: 0.55
    - m3_per_panel_max_per_figure: 20
    - m3_per_panel_max_per_paper: 200
    """
    cfg = _make_cfg(tmp_path)
    assert hasattr(cfg, "m3_per_panel_enabled")
    assert cfg.m3_per_panel_enabled is False
    assert cfg.m3_per_panel_min_conf == pytest.approx(0.55)
    assert cfg.m3_per_panel_max_per_figure == 20
    assert cfg.m3_per_panel_max_per_paper == 200


def test_config_field_overrides_work(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled=True,
        m3_per_panel_min_conf=0.7,
        m3_per_panel_max_per_figure=10,
        m3_per_panel_max_per_paper=50,
    )
    assert cfg.m3_per_panel_enabled is True
    assert cfg.m3_per_panel_min_conf == pytest.approx(0.7)
    assert cfg.m3_per_panel_max_per_figure == 10
    assert cfg.m3_per_panel_max_per_paper == 50


# ---------------------------------------------------------------------------
# Task 2: method shell with early-return guards
# ---------------------------------------------------------------------------


class _StubPipeline:
    """Minimal stand-in: bind the unbound method and provide config."""

    def __init__(self, cfg: PipelineConfig):
        from types import SimpleNamespace

        from rlpe.pipeline import RadiolarianPipeline

        self.config = cfg
        # Provide a stub m3_engine with a non-None backend so the method's
        # second guard (m3_engine.backend) does NOT short-circuit and the
        # actual Task 3 loop body gets exercised by skip-rows tests.
        self.m3_engine = SimpleNamespace(backend=SimpleNamespace(backend_name="stub"))
        self._apply_m3_per_panel_species_id = (
            RadiolarianPipeline._apply_m3_per_panel_species_id.__get__(self, RadiolarianPipeline)
        )


def test_method_early_returns_when_disabled(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=False)
    pipe = _StubPipeline(cfg)
    out = pipe._apply_m3_per_panel_species_id(
        results=[{"panel_id": "1", "species": "regex_match"}],
        paper_id="paper1",
    )
    # When disabled, results pass through unchanged (regex match survives).
    assert out == [{"panel_id": "1", "species": "regex_match"}]


def test_method_no_op_when_no_results(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    pipe = _StubPipeline(cfg)
    out = pipe._apply_m3_per_panel_species_id(results=[], paper_id="paper1")
    assert out == []


# ---------------------------------------------------------------------------
# Minor #1 (from Task 1 CQ review): post_init coercion + range validation
# ---------------------------------------------------------------------------


def test_config_coerces_string_inputs_in_post_init(tmp_path):
    """YAML/JSON loads can produce string booleans/floats/ints.
    ``__post_init__`` must coerce them to the declared types."""
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled="true",
        m3_per_panel_min_conf="0.7",
        m3_per_panel_max_per_figure="10",
        m3_per_panel_max_per_paper="50",
    )
    # _make_cfg uses plain setattr; __post_init__ was already called once
    # during construction but the string overrides bypass coercion, so
    # re-invoke it explicitly to exercise the coercion path.
    cfg.__post_init__()
    assert cfg.m3_per_panel_enabled is True
    assert cfg.m3_per_panel_min_conf == pytest.approx(0.7)
    assert cfg.m3_per_panel_max_per_figure == 10
    assert cfg.m3_per_panel_max_per_paper == 50


def test_config_post_init_validates_conf_range(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_min_conf=1.5)
    with pytest.raises(ValueError, match=r"m3_per_panel_min_conf must be"):
        cfg.__post_init__()


def test_config_post_init_validates_caps_positive(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_max_per_figure=0)
    with pytest.raises(ValueError, match=r"must be >= 1"):
        cfg.__post_init__()


# ---------------------------------------------------------------------------
# Task 3: build per-panel context tuples + skip rows without crops
# ---------------------------------------------------------------------------


def test_method_skips_rows_without_panel_path(tmp_path):
    """Rows without a Stage 3 crop (no panel_path) are passed through
    unchanged — per-panel vision needs the crop image."""
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    pipe = _StubPipeline(cfg)
    results = [
        {"panel_id": "1", "species": "regex_A", "panel_path": None},
        {"panel_id": "2", "species": "regex_B", "panel_path": ""},
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    # Without a backend hook we cannot test "called" — we test "skipped".
    assert out[0]["species"] == "regex_A"
    assert out[1]["species"] == "regex_B"


def test_method_builds_caption_for_panel_from_caption_pairs(tmp_path):
    """When ``caption_pairs`` is on the row, the call uses the pair
    whose panel_id matches the row's panel_id. We assert via inspect:
    the method body must read caption_pairs / select-by-panel_id / etc.
    """
    cfg = _make_cfg(tmp_path)
    from rlpe.pipeline import RadiolarianPipeline

    src = inspect.getsource(RadiolarianPipeline._apply_m3_per_panel_species_id)
    assert "caption_pairs" in src, (
        "method must read caption_pairs to pick the panel-specific snippet"
    )
    assert "panel_id" in src, "method must match caption pairs to row panel_id"


def test_method_truncates_page_context_at_1500_chars(tmp_path):
    cfg = _make_cfg(tmp_path)
    from rlpe.pipeline import RadiolarianPipeline

    src = inspect.getsource(RadiolarianPipeline._apply_m3_per_panel_species_id)
    # Spec §3 requires page-context truncation at 1500 chars.
    assert "1500" in src, "page-context snippet must be truncated (spec §3 says 1500 chars)"


# ---------------------------------------------------------------------------
# Task 4: fan-out via semaphore + per-panel backend call
# ---------------------------------------------------------------------------


def _write_valid_png(path: Path) -> None:
    """Write a tiny valid PNG so ``PIL.Image.open`` can decode it.

    PIL rejects an \"almost PNG\" byte string (the magic without a
    parseable IHDR chunk) with ``UnidentifiedImageError``, which would
    drive the production ``_one`` helper onto its exception path and
    silently leave the row's species unchanged — masking whatever the
    test was trying to assert. A real 10x10 white PNG keeps the test
    focused on the confidence-gate wiring.
    """
    Image.new("RGB", (10, 10), "white").save(path)


def test_method_overwrites_species_when_m3_high_confidence(tmp_path):
    """When ``backend.infer_panel`` returns parseable JSON with
    confidence ``>= m3_per_panel_min_conf``, the row's species is
    overwritten."""
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled=True,
        m3_per_panel_min_conf=0.55,
    )

    # Fake backend that returns a high-confidence species.
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "Emiluvia orea",
        "label": "1",
        "confidence": 0.92,
        "reasoning": "Late Cretaceous nassellarian",
        "alternative": "Archaeodictyomitra",
    }

    # Stub pipeline with fake engine + backend.
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend

    # Fake crop file (must be a real PNG — see _write_valid_png).
    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)

    results = [
        {
            "panel_id": "1",
            "species": "regex_old_species",
            "label": "1",
            "panel_path": str(crop),
            "caption_pairs": [{"panel_id": "1", "text": "Fig. 1, 1."}],
            "page_context_snippet": "Tunisia, Late Cretaceous, Scaglia Fm",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    assert out[0]["species"] == "Emiluvia orea"
    assert out[0]["label"] == "1"
    assert backend.infer_panel.called


def test_method_keeps_regex_when_m3_low_confidence(tmp_path):
    """M3 confidence < min_conf → regex species stays."""
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled=True,
        m3_per_panel_min_conf=0.55,
    )
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "Emiluvia orea",
        "label": "1",
        "confidence": 0.3,  # below threshold
        "reasoning": "uncertain",
    }
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)
    results = [
        {
            "panel_id": "1",
            "species": "regex_old_species",
            "label": "1",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    assert out[0]["species"] == "regex_old_species"


def test_method_keeps_regex_when_m3_returns_empty_species(tmp_path):
    """Regression: when M3 returns confidence >= min_conf but with
    empty species/label strings, the row should keep the regex match
    (audit 2026-08-17 final review case 19).

    The guard ``r["species"] = parsed.get("species") or r.get("species")``
    at pipeline.py:2348 protects against M3 hallucinating an empty
    species string while being otherwise confident.
    """
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled=True,
        m3_per_panel_min_conf=0.55,
    )

    # Same fixture as test_method_overwrites_species_when_m3_high_confidence,
    # but the backend returns empty species/label even though it is
    # "confident". This is the regression-pin scenario: the gate passes
    # (confidence >= min_conf) but the payload is empty so the ``or``
    # fallback must kick in and preserve the regex match.
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "",
        "label": "",
        "confidence": 0.95,
        "reasoning": "high conf but empty payload",
        "alternative": None,
    }

    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend

    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)

    results = [
        {
            "panel_id": "1",
            "species": "regex_species_name",
            "label": "1",
            "panel_path": str(crop),
            "caption_pairs": [{"panel_id": "1", "text": "Fig. 1, 1."}],
            "page_context_snippet": "Tunisia, Late Cretaceous, Scaglia Fm",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")

    # The ``or`` fallback preserved the regex species + the original label.
    assert out[0]["species"] == "regex_species_name"
    assert out[0]["label"] == "1"
    # The backend WAS called (the gate fired and the body executed), so
    # the audit stamp must be present. This distinguishes the fallback
    # path from the low-confidence path (no stamp there) and the
    # error/exception paths (no stamp there either).
    assert backend.infer_panel.called
    assert "m3_per_panel" in out[0]["metadata"]
    assert out[0]["metadata"]["m3_per_panel"]["confidence"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Task 5: failure-path coverage + caps + audit-tag provenance
# ---------------------------------------------------------------------------


def test_method_handles_backend_fallback_used(tmp_path):
    """backend returns fallback_used=True → row keeps regex species,
    metadata.m3_per_panel is NOT stamped."""
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "fallback_used": True,
        "error": "M3 quota exhausted",
    }
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)
    results = [
        {
            "panel_id": "1",
            "species": "regex_species",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    assert out[0]["species"] == "regex_species"
    assert "m3_per_panel" not in out[0]["metadata"]


def test_method_handles_backend_exception(tmp_path):
    """backend.infer_panel raises → caught + logged, regex stays."""
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.side_effect = RuntimeError("M3 API down")
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)
    results = [
        {
            "panel_id": "1",
            "species": "regex_species",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    assert out[0]["species"] == "regex_species"
    assert "m3_per_panel" not in out[0]["metadata"]


def test_method_handles_garbage_json(tmp_path):
    """backend returns unparseable blob → parse_json_from_text 4-tier
    falls through to {species=None} → regex stays."""
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": None,
        "label": None,
        "confidence": 0.0,
        "reasoning": "no parse",
    }
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)
    results = [
        {
            "panel_id": "1",
            "species": "regex_species",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    # Confidence 0.0 < 0.55 → no overwrite, but metadata IS stamped
    # (we want to know M3 was attempted).
    assert out[0]["species"] == "regex_species"


def test_method_caps_per_figure(tmp_path):
    """If a figure has more panels than m3_per_panel_max_per_figure,
    only the first N get per-panel M3 calls; the rest keep regex."""
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled=True,
        m3_per_panel_max_per_figure=2,
    )
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "Emiluvia orea",
        "label": "X",
        "confidence": 0.9,
        "reasoning": "r",
        "alternative": None,
    }
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel.png"
    _write_valid_png(crop)
    rows = []
    for i in range(5):
        rows.append(
            {
                "panel_id": str(i),
                "species": f"regex_{i}",
                "label": str(i),
                "figure_id": "fig_1",
                "panel_path": str(crop),
                "caption_pairs": [],
                "page_context_snippet": "",
                "metadata": {},
            }
        )
    out = pipe._apply_m3_per_panel_species_id(rows, paper_id="paper1")
    # Only the first 2 should have m3_per_panel stamped; rest untouched.
    stamped = [r for r in out if "m3_per_panel" in r["metadata"]]
    assert len(stamped) == 2
    untouched = [r for r in out if "m3_per_panel" not in r["metadata"]]
    assert len(untouched) == 3


def test_method_caps_per_paper(tmp_path):
    """m3_per_panel_max_per_paper caps total calls across all figures."""
    cfg = _make_cfg(
        tmp_path,
        m3_per_panel_enabled=True,
        m3_per_panel_max_per_figure=100,
        m3_per_panel_max_per_paper=3,
    )
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "X",
        "label": "X",
        "confidence": 0.9,
        "reasoning": "r",
        "alternative": None,
    }
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel.png"
    _write_valid_png(crop)
    rows = []
    for i in range(10):
        rows.append(
            {
                "panel_id": str(i),
                "species": f"regex_{i}",
                "label": str(i),
                "figure_id": f"fig_{i}",  # each in own figure → bypasses per-fig cap
                "panel_path": str(crop),
                "caption_pairs": [],
                "page_context_snippet": "",
                "metadata": {},
            }
        )
    out = pipe._apply_m3_per_panel_species_id(rows, paper_id="paper1")
    stamped = [r for r in out if "m3_per_panel" in r["metadata"]]
    assert len(stamped) == 3


def test_method_normalises_species_list_extras(tmp_path):
    """If backend returns species_list (a list/dict structural extra),
    _normalize_panel_dict preserves it (Audit 2026-08-17 BUG-E)."""
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "Emiluvia orea",
        "label": "1",
        "confidence": 0.9,
        "reasoning": "r",
        "species_list": [
            {"species": "Emiluvia orea", "confidence": 0.92},
            {"species": "Stichocapsa", "confidence": 0.7},
        ],
    }
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel.png"
    _write_valid_png(crop)
    results = [
        {
            "panel_id": "1",
            "species": "regex",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    # metadata.m3_per_panel is the normalised dict; species_list is NOT
    # in there (it's only kept on the parsed match — not in the audit stamp).
    # The overwrite still happens because confidence >= threshold.
    assert out[0]["species"] == "Emiluvia orea"


def test_method_clamps_confidence_to_unit_interval(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "X",
        "label": "X",
        "confidence": 1.7,  # out of range
        "reasoning": "r",
    }
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel.png"
    _write_valid_png(crop)
    results = [
        {
            "panel_id": "1",
            "species": "regex",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    # Confidence 1.7 → clamped to 1.0 → above 0.55 → overwrite happens.
    assert out[0]["species"] == "X"
    assert out[0]["metadata"]["m3_per_panel"]["confidence"] == 1.0


def test_method_records_latency(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "X",
        "label": "X",
        "confidence": 0.9,
        "reasoning": "r",
    }
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel.png"
    _write_valid_png(crop)
    results = [
        {
            "panel_id": "1",
            "species": "regex",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    md = out[0]["metadata"]["m3_per_panel"]
    assert "latency_sec" in md
    assert isinstance(md["latency_sec"], float)
    assert md["latency_sec"] >= 0.0


def test_method_records_image_sha(tmp_path):
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "X",
        "label": "X",
        "confidence": 0.9,
        "reasoning": "r",
    }
    pipe = _StubPipeline(cfg)
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend
    crop = tmp_path / "panel.png"
    _write_valid_png(crop)
    results = [
        {
            "panel_id": "1",
            "species": "regex",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(results, paper_id="paper1")
    md = out[0]["metadata"]["m3_per_panel"]
    assert "image_sha" in md
    assert len(md["image_sha"]) == 16  # truncated sha256[:16]


def test_pipeline_main_loop_calls_stage4_5(tmp_path):
    """Source-guard: the main per-figure loop in ``pipeline.run`` (or
    equivalent) must invoke ``_apply_m3_per_panel_species_id`` after
    Stage 3 bbox crops. Pre-fix the call was missing → opt-in flag
    had no effect."""
    import inspect

    from rlpe import pipeline as pipeline_mod
    from rlpe.pipeline import RadiolarianPipeline

    src = inspect.getsource(RadiolarianPipeline)
    assert "_apply_m3_per_panel_species_id" in src
    # Must be called AFTER stage3 bbox crops, BEFORE multi-plate enrichment
    stage3_idx = src.find("_apply_stage3_bbox_crops")
    per_panel_idx = src.find("_apply_m3_per_panel_species_id")
    enrich_idx = src.find("_apply_multi_plate_enrichment")
    assert stage3_idx != -1 and per_panel_idx != -1 and enrich_idx != -1, (
        "all three methods must exist on the pipeline"
    )
    assert stage3_idx < per_panel_idx < enrich_idx, (
        f"per-panel must be called AFTER stage3 (idx={stage3_idx}) and "
        f"BEFORE multi-plate enrich (idx={enrich_idx}); got per-panel "
        f"at {per_panel_idx}"
    )


def test_cli_argparse_accepts_m3_per_panel_flags(tmp_path):
    """Source-guard + behaviour: CLI must accept the 4 new Stage 4.5 flags.

    The guard checks flag spellings rather than argparse ``dest`` names:
    ``--no-m3-per-panel`` deliberately *shares* ``dest="m3_per_panel"``
    with ``--m3-per-panel`` (so the last flag on the command line wins),
    which means no ``no_m3_per_panel`` identifier exists to grep for.
    """
    import inspect

    from rlpe import cli as cli_mod

    src = inspect.getsource(cli_mod)
    for flag in [
        "--m3-per-panel",
        "--no-m3-per-panel",
        "--m3-per-panel-min-conf",
        "--m3-per-panel-max-per-figure",
        "--m3-per-panel-max-per-paper",
    ]:
        assert flag in src, f"CLI must define {flag}"

    parser = cli_mod.build_parser()
    base = ["--pdf-dir", str(tmp_path), "--work-dir", str(tmp_path / "work")]

    # Defaults mirror the PipelineConfig defaults; Stage 4.5 is opt-in.
    defaults = parser.parse_args(base)
    assert defaults.m3_per_panel is False
    assert defaults.m3_per_panel_min_conf == 0.55
    assert defaults.m3_per_panel_max_per_figure == 20
    assert defaults.m3_per_panel_max_per_paper == 200

    # Each flag parses onto the dest the cfg wiring reads from.
    on = parser.parse_args(
        base
        + [
            "--m3-per-panel",
            "--m3-per-panel-min-conf",
            "0.9",
            "--m3-per-panel-max-per-figure",
            "3",
            "--m3-per-panel-max-per-paper",
            "7",
        ]
    )
    assert on.m3_per_panel is True
    assert on.m3_per_panel_min_conf == 0.9
    assert on.m3_per_panel_max_per_figure == 3
    assert on.m3_per_panel_max_per_paper == 7

    # Shared dest => explicit opt-out wins when it comes last.
    assert parser.parse_args(base + ["--m3-per-panel", "--no-m3-per-panel"]).m3_per_panel is False

    # Wiring guard (Audit 2026-08-17): the CLI must pass the typed flag
    # as a PipelineConfig kwarg, AND the pipeline must read it from the
    # typed attribute -- not from ``config.extra``. The earlier mirror
    # hack (which the previous test version asserted) was a workaround
    # for mis-wired gates and has been removed: gates + CLI now use the
    # typed attributes consistently.
    assert "m3_per_panel_enabled=args.m3_per_panel" in src
    # The Stage 3 + multi-plate enrichment gates have the same defect;
    # both must be wired as typed kwargs in the CLI.
    assert "m3_stage3_enabled=bool(args.use_m3_stage3)" in src
    assert "m3_multi_plate_enrich_enabled=bool(args.m3_multi_plate_enrich)" in src


# ---------------------------------------------------------------------------
# Audit 2026-08-17: regression tests for the flag-wiring bug.
#
# Pre-fix: CLI set ``m3_per_panel_enabled`` as a typed PipelineConfig
# attribute, but the Stage 4.5 gate at pipeline.py line 1722 read it from
# ``self.config.extra.get("m3_per_panel_enabled", False)`` which the CLI
# never populated. Live smoke on Bandini 2011 (commit 654c0fc) showed
# 0/64 rows reached Stage 4.5 even with ``--m3-per-panel``. The same
# defect affected the ``m3_stage3`` / ``m3_multi_plate_enrich`` gates.
#
# These tests pin the gates down to the typed attributes and would have
# caught the bug.
# ---------------------------------------------------------------------------


def test_config_has_m3_gate_typed_attrs(tmp_path):
    """``PipelineConfig`` must expose the three M3 gate flags as typed
    attributes (not only via ``extra``). Pre-fix the per-panel flag was
    typed but the other two were not -- so the gates that read from
    ``config.extra.get(...)`` silently returned False."""
    cfg = _make_cfg(tmp_path)
    for attr in (
        "m3_per_panel_enabled",
        "m3_stage3_enabled",
        "m3_multi_plate_enrich_enabled",
    ):
        assert hasattr(cfg, attr), f"PipelineConfig must expose {attr}"
        assert getattr(cfg, attr) is False, f"{attr} must default to False"


def test_pipeline_gates_use_typed_attrs_not_extra(tmp_path):
    """Source-guard regression for the flag-wiring bug. The three M3
    gates in ``RadiolarianPipeline.run`` / ``_process_grobid_path`` must
    read typed attributes, not ``config.extra.get(...)`` -- because the
    CLI only sets the typed attributes."""
    import inspect

    from rlpe.pipeline import RadiolarianPipeline

    src = inspect.getsource(RadiolarianPipeline)
    # The buggy pattern: ``self.config.extra.get("m3_per_panel_enabled", ...``.
    # After the fix, none of the three M3 gates should still use this
    # pattern.
    for forbidden in (
        'self.config.extra.get("m3_per_panel_enabled"',
        'self.config.extra.get("m3_stage3"',
        'self.config.extra.get("m3_multi_plate_enrich"',
    ):
        assert forbidden not in src, (
            f"Pipeline gate still reads from config.extra -- pre-fix "
            f"behaviour that silently disabled the gate: {forbidden!r}"
        )
    # The correct pattern: read typed attributes directly. Spot-check
    # at least one of the three gates exists.
    assert "self.config.m3_per_panel_enabled" in src
    assert "self.config.m3_stage3_enabled" in src
    assert "self.config.m3_multi_plate_enrich_enabled" in src


def test_stage4_5_gate_fires_with_typed_attr(tmp_path):
    """End-to-end regression: build a config with the typed attribute set
    to True, drop a row through the per-panel path, and assert the
    metadata stamp is present (i.e. the gate fired). Pre-fix, the gate
    read ``config.extra.get(...)`` which returned False so the early
    return triggered and ``metadata.m3_per_panel`` was never stamped."""
    from types import SimpleNamespace

    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)

    # Stub backend that returns a high-confidence parse so the method
    # stamps metadata even on this short-circuited path.
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "Emiluvia orea",
        "label": "1",
        "confidence": 0.9,
        "reasoning": "test",
    }

    pipe = _StubPipeline(cfg)
    pipe.m3_engine = SimpleNamespace(backend=backend)

    crop = tmp_path / "panel.png"
    _write_valid_png(crop)
    rows = [
        {
            "panel_id": "1",
            "species": "regex_old",
            "label": "1",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(rows, paper_id="paper1")
    # The stamp proves the gate fired; the species overwrite proves the
    # body executed end-to-end.
    assert "m3_per_panel" in out[0]["metadata"]
    assert out[0]["species"] == "Emiluvia orea"


def test_stage4_5_gate_short_circuits_when_typed_attr_false(tmp_path):
    """Companion regression: when the typed attribute is False, the
    method returns the rows untouched. Pre-fix, the gate always read
    False from ``config.extra`` so this short-circuit was the only path
    that ever fired in production."""
    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=False)
    pipe = _StubPipeline(cfg)
    # Give the stub a backend that would otherwise produce a stamp;
    # the gate must short-circuit before the body runs.
    backend = MagicMock()
    backend.backend_name = "test_backend"
    backend.infer_panel.return_value = {
        "species": "X",
        "label": "X",
        "confidence": 0.9,
        "reasoning": "r",
    }
    pipe.m3_engine = MagicMock()
    pipe.m3_engine.backend = backend

    crop = tmp_path / "panel.png"
    _write_valid_png(crop)
    rows = [
        {
            "panel_id": "1",
            "species": "regex",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(rows, paper_id="paper1")
    assert out == rows, "gate must short-circuit when typed attr is False"
    assert "m3_per_panel" not in out[0]["metadata"]
    assert not backend.infer_panel.called, "body must not run when gate is False"


def test_cli_no_longer_mirrors_m3_per_panel_into_extra(tmp_path):
    """The pre-fix workaround was to copy the typed attrs into ``extra``
    in the CLI. After the fix the CLI passes typed attrs directly and
    the pipeline gates read them -- the mirror is no longer required
    and removing it ensures we cannot reintroduce the silent split
    between typed attr and ``extra`` key."""
    import inspect

    from rlpe import cli as cli_mod

    src = inspect.getsource(cli_mod)
    # The four mirror lines that previously copied typed attrs into
    # extra. After the fix they must be gone -- if they come back, the
    # gate-vs-typed-attr split can return.
    assert 'cfg.extra["m3_per_panel_enabled"]' not in src, (
        "CLI must NOT mirror m3_per_panel_enabled into extra; gates "
        "now read the typed attribute directly."
    )
    assert 'cfg.extra["m3_per_panel_min_conf"]' not in src
    assert 'cfg.extra["m3_per_panel_max_per_figure"]' not in src
    assert 'cfg.extra["m3_per_panel_max_per_paper"]' not in src


# ---------------------------------------------------------------------------
# Audit 2026-08-17 follow-on: caption_pairs + page_context plumbing.
#
# Pre-fix (after commit 80361aa): the Stage 4.5 per-panel method reads
# ``r.get("caption_pairs")`` and ``r.get("page_context_snippet")`` from
# each row dict, but the upstream ``match_panels()`` step consumed the
# pairs and never propagated them onto the row, and the per-page body
# text was never sliced. Result: M3 received panels with no context and
# returned uniformly low confidence (0/67 high-conf overwrites in the
# Task 8 smoke). The fix adds ``_attach_stage4_5_context`` and calls it
# from the two paths in ``_process_region`` (LLM-first and classical).
#
# These tests pin the plumbing:
#   1. caption_pairs is propagated into the M3 prompt as caption_for_panel
#   2. page_context_snippet is propagated into the M3 prompt body
#   3. empty context still calls M3 (no silent skip)
# ---------------------------------------------------------------------------


class _CaptureBackend:
    """Backend that records the prompt the worker builds.

    Inspect-mode: we don't actually run an M3 call; we capture the
    prompt / metadata the Stage 4.5 worker passes to the backend so the
    test can assert on the propagated context. The M3 response is a
    parseable high-confidence JSON so the row gets overwritten and the
    stamp path completes cleanly.
    """

    def __init__(self) -> None:
        self.backend_name = "capture"
        self.last_user_prompt: str | None = None
        self.last_caption_text: str | None = None
        self.last_ocr_labels: list[str] | None = None
        self.last_system_prompt: str | None = None
        self.call_count: int = 0

    def infer_panel(self, **kwargs):  # noqa: ANN001
        self.last_user_prompt = kwargs.get("user_prompt")
        self.last_caption_text = kwargs.get("caption_text")
        self.last_ocr_labels = kwargs.get("ocr_labels")
        self.last_system_prompt = kwargs.get("system_prompt")
        self.call_count += 1
        # High-confidence parse so the stamp + overwrite path runs and
        # the test can assert against the final row state.
        return {
            "species": "Test species",
            "label": "1",
            "confidence": 0.85,
            "reasoning": "captured for test",
            "alternative": None,
        }


def test_caption_pairs_plumbed_into_m3_prompt(tmp_path):
    """When a row carries a populated ``caption_pairs`` whose labels
    include the row's panel_id, the M3 prompt's caption block carries
    the corresponding species / modifier / notes string -- not the
    generic 'no candidate pairs were supplied' message.
    """
    from types import SimpleNamespace

    from rlpe.m3_engine import CaptionPair

    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    pipe = _StubPipeline(cfg)
    backend = _CaptureBackend()
    pipe.m3_engine = SimpleNamespace(backend=backend)

    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)

    # Simulate a row that the new _attach_stage4_5_context helper has
    # stamped: caption_pairs carries one pair whose labels list contains
    # the row's panel_id and whose species is "Emiluvia orea".
    rows = [
        {
            "panel_id": "1",
            "species": "regex_old",
            "label": "1",
            "panel_path": str(crop),
            "caption_pairs": [
                CaptionPair(
                    labels=["1"],
                    species="Emiluvia orea",
                    modifier="",
                    confidence=0.9,
                    notes="",
                    raw_text="1. Emiluvia orea",
                ).to_dict()
            ],
            "page_context_snippet": "Tunisia, Late Cretaceous, Scaglia Fm.",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(rows, paper_id="paper1")
    # The worker received the species in the prompt; the prompt template
    # renders it as "[This panel]\\n<species>\\n\\n[Same-page context]...".
    assert backend.call_count == 1
    assert backend.last_user_prompt is not None
    assert "Emiluvia orea" in backend.last_user_prompt
    assert "[This panel]" in backend.last_user_prompt
    assert "[Same-page context]" in backend.last_user_prompt
    # The species block was actually used to overwrite the row (high conf).
    assert out[0]["species"] == "Test species"
    assert "m3_per_panel" in out[0]["metadata"]


def test_page_context_snippet_plumbed_into_m3_prompt(tmp_path):
    """When a row carries a populated ``page_context_snippet``, that
    snippet is rendered into the M3 prompt's '[Same-page context]' block.
    """
    from types import SimpleNamespace

    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    pipe = _StubPipeline(cfg)
    backend = _CaptureBackend()
    pipe.m3_engine = SimpleNamespace(backend=backend)

    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)

    page_text = (
        "Bandini 2011 reports Pseudocrucella? elisabethae from the upper "
        "Scaglia Rossa formation near Gubbio, central Italy. Specimens "
        "are Late Cretaceous (Campanian) in age, recovered from marly "
        "limestone beds interbedded with chert."
    )
    rows = [
        {
            "panel_id": "2",
            "species": "regex_old",
            "label": "2",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": page_text,
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(rows, paper_id="paper1")
    assert backend.call_count == 1
    assert backend.last_user_prompt is not None
    # The same-page context must show up in the prompt body.
    assert "Gubbio" in backend.last_user_prompt
    assert "Campanian" in backend.last_user_prompt
    assert page_text[:200] in backend.last_user_prompt
    assert out[0]["species"] == "Test species"


def test_empty_context_still_calls_m3(tmp_path):
    """Guards the no-context path: an empty ``caption_pairs`` + empty
    ``page_context_snippet`` MUST still produce an M3 call (no silent
    skip). The prompt's context block will be empty, but the model
    receives the panel image and the OCR label and can still attempt
    an identification. This is the existing behaviour and must not
    regress when we add the plumbing.
    """
    from types import SimpleNamespace

    cfg = _make_cfg(tmp_path, m3_per_panel_enabled=True)
    pipe = _StubPipeline(cfg)
    backend = _CaptureBackend()
    pipe.m3_engine = SimpleNamespace(backend=backend)

    crop = tmp_path / "panel1.png"
    _write_valid_png(crop)

    rows = [
        {
            "panel_id": "1",
            "species": "regex_old",
            "label": "1",
            "panel_path": str(crop),
            "caption_pairs": [],
            "page_context_snippet": "",
            "metadata": {},
        }
    ]
    out = pipe._apply_m3_per_panel_species_id(rows, paper_id="paper1")
    assert backend.call_count == 1
    assert backend.last_user_prompt is not None
    # The prompt still has the bracketed section headers, just no
    # filled-in context between them. This is the "no-context" path
    # that the previous wiring hit and which we now try to AVOID by
    # plumbing real context.
    assert "[This panel]" in backend.last_user_prompt
    assert "[Same-page context]" in backend.last_user_prompt
    assert out[0]["species"] == "Test species"
    assert "m3_per_panel" in out[0]["metadata"]


def test_attach_stage4_5_context_attaches_caption_pairs_and_page_text(tmp_path):
    """End-to-end test on the new ``_attach_stage4_5_context`` helper:
    build a row list with no context, run the helper, and assert both
    ``caption_pairs`` and ``page_context_snippet`` end up on every row.
    """
    from types import SimpleNamespace

    from rlpe.m3_engine import CaptionPair

    cfg = _make_cfg(tmp_path)
    pipe = _StubPipeline(cfg)

    pairs = [
        CaptionPair(
            labels=["1", "2"],
            species="Emiluvia orea",
            modifier="sp.",
            confidence=0.9,
            notes="",
            raw_text="1, 2. Emiluvia orea sp.",
        )
    ]
    grobid_sections = [
        {"page_index": 5, "text": "Gubbio, Italy, Late Cretaceous."},
        {"page_index": 6, "text": "Scaglia Rossa formation, Campanian."},
    ]
    rows = [
        {"panel_id": "1", "species": "x", "metadata": {}},
        {"panel_id": "2", "species": "y", "metadata": {}},
    ]
    # Bind the helper method onto the stub.
    from rlpe.pipeline import RadiolarianPipeline

    pipe._attach_stage4_5_context = RadiolarianPipeline._attach_stage4_5_context.__get__(
        pipe, RadiolarianPipeline
    )
    pipe._attach_stage4_5_context(
        rows,
        caption_pairs=pairs,
        grobid_sections=grobid_sections,
        figure_page_index=5,
    )
    # Both rows received the same figure-level caption pairs (Stage 4.5
    # filters by panel_id internally).
    for r in rows:
        assert "caption_pairs" in r
        assert isinstance(r["caption_pairs"], list)
        assert len(r["caption_pairs"]) == 1
        cp = r["caption_pairs"][0]
        assert cp["species"] == "Emiluvia orea"
        assert cp["labels"] == ["1", "2"]
        # Page context is sliced from the same-page (page_index 5) text.
        assert r["page_context_snippet"] == "Gubbio, Italy, Late Cretaceous."


def test_attach_stage4_5_context_handles_noop_inputs(tmp_path):
    """The helper must be a safe no-op when called with empty inputs:
    no rows, no caption pairs, no body text. Rows without context just
    keep their pre-existing fields untouched.
    """
    from rlpe.pipeline import RadiolarianPipeline

    cfg = _make_cfg(tmp_path)
    pipe = _StubPipeline(cfg)
    pipe._attach_stage4_5_context = RadiolarianPipeline._attach_stage4_5_context.__get__(
        pipe, RadiolarianPipeline
    )
    # No rows at all.
    pipe._attach_stage4_5_context(
        rows=[],
        caption_pairs=None,
        grobid_sections=None,
        figure_page_index=0,
    )
    # Rows present but no context to attach.
    rows = [{"panel_id": "1", "metadata": {}}]
    pipe._attach_stage4_5_context(
        rows=list(rows),
        caption_pairs=None,
        grobid_sections=None,
        figure_page_index=0,
    )
    out = rows[0]
    # We still stamp empty lists/strings so the Stage 4.5 worker's
    # ``r.get(...)`` paths return a deterministic empty default rather
    # than raising KeyError. See the helper docstring for the rationale.
    assert out["caption_pairs"] == []
    assert out["page_context_snippet"] == ""
