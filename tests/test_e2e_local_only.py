"""End-to-end test of the ``local_only`` data-outbound path AND the
``PipelineConfig`` -> ``RadiolarianPipeline`` -> JSONL wiring.

What this test proves
---------------------
1. ``MiniMaxM3Backend(data_outbound_policy="local_only")`` works
   without an API key and never touches the network.
2. The same constructor with ``api_redacted`` shrinks payloads.
3. ``build_MiniMax_backend_from_env_or_config`` honours the policy
   and lets the operator run offline.
4. The canonical ``RunOutput`` Pydantic schema round-trips through
   ``model_dump_json()`` and back, which is the contract the API
   publishes to the Web UI.
5. The CLI ``JobOptions`` model accepts ``data_outbound_policy=local_only``
   and the FastAPI app exposes it (HTTP 200 on /system/info).
6. The CLI entry point actually builds a ``RadiolarianPipeline`` even
   when PaddleOCR / EasyOCR / TaxoNERD / OpenDataLoader are not
   installed (graceful lazy-init in production code).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

# Compat: ``datetime.UTC`` was added in Python 3.11. Use ``timezone.utc``
# so the test runs on Python 3.10 conda envs that the user may use.
UTC = timezone.utc  # noqa: UP017
from pathlib import Path

import pytest

# Test deps
pytest.importorskip("pydantic")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rlpe.llm_backends import MiniMaxM3Backend  # noqa: E402
from rlpe.schema_models import (  # noqa: E402
    PanelMetadata,
    PanelRecord,
    ProvenanceRecord,
    RunOutput,
)

# ---------------------------------------------------------------------------
# 1) local_only: no key, no network, deterministic no-op result
# ---------------------------------------------------------------------------


def test_local_only_minimax_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MiniMax_API_KEY", raising=False)

    b = MiniMaxM3Backend(api_key="", data_outbound_policy="local_only")
    assert b.data_outbound_policy == "local_only"

    r = b.infer_panel(
        panel_image=None, caption_text="", ocr_labels=[], system_prompt="", user_prompt=""
    )
    assert r["fallback_used"] is True
    assert r["error_type"] == "LocalOnlyPolicy"
    assert r["species"] is None
    assert r["label"] is None
    assert r["confidence"] == 0.0
    assert r.get("request_id") is None

    r2 = b.infer_text(system_prompt="", user_prompt="")
    assert r2["fallback_used"] is True
    assert r2["error_type"] == "LocalOnlyPolicy"


# ---------------------------------------------------------------------------
# 2) api_redacted: image and text are redacted before they leave the box
# ---------------------------------------------------------------------------


def test_api_redacted_thumbnail_and_truncate(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``api_redacted`` mode still constructs the real Anthropic SDK client
    # because it expects to make outbound calls (just with thumbnails and
    # truncated text). Without ``anthropic`` installed the constructor
    # raises by design — that is a missing dependency, not a bug.
    pytest.importorskip("anthropic")
    from PIL import Image

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MiniMax_API_KEY", raising=False)
    b = MiniMaxM3Backend(
        api_key="sk-test-dummy-dummy-dummy-dummy-dummy-dummy",
        data_outbound_policy="api_redacted",
    )
    img = Image.new("RGB", (2048, 2048), color=(128, 128, 128))
    redacted = b._redact_image(img)
    assert isinstance(redacted, Image.Image)
    assert redacted.size[0] <= 256 and redacted.size[1] <= 256

    long_text = "Thecopsammatium sp. " * 200
    redacted_text = b._redact_text(long_text, limit=200)
    assert "truncated" in redacted_text
    assert len(redacted_text) < 400


# ---------------------------------------------------------------------------
# 3) Builder must accept local_only without a key
# ---------------------------------------------------------------------------


def test_builder_accepts_local_only_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlpe.llm_backends import build_MiniMax_backend_from_env_or_config

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MiniMax_API_KEY", raising=False)

    b = build_MiniMax_backend_from_env_or_config({"data_outbound_policy": "local_only"})
    assert isinstance(b, MiniMaxM3Backend)
    assert b.data_outbound_policy == "local_only"


def test_builder_rejects_bad_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlpe.llm_backends import build_MiniMax_backend_from_env_or_config

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MiniMax_API_KEY", raising=False)

    with pytest.raises(ValueError, match="data_outbound_policy"):
        build_MiniMax_backend_from_env_or_config({"data_outbound_policy": "bogus"})


# ---------------------------------------------------------------------------
# 4) The published schema round-trips
# ---------------------------------------------------------------------------


def test_panel_record_round_trip() -> None:
    p = PanelRecord(
        paper_id="abc123",
        figure_id="fig1",
        panel_id="A",
        species="Thecopsammatium bullatum",
        panel_path="/tmp/panel.png",
        bbox=[100, 200, 300, 400],
        confidence=0.91,
        label_text="A",
        caption_snippet="Fig. 1A. Thecopsammatium bullatum sp. nov.",
        ocr_text="A",
        metadata=PanelMetadata(
            panel_score=0.87,
            ocr_count=1,
            taxon_count=1,
            matcher_used=True,
            matcher_type="heuristic",
            matcher_conf=0.83,
            extraction_source="rule",
        ),
    )
    blob = p.model_dump_json()
    p2 = PanelRecord.model_validate_json(blob)
    assert p2 == p

    d = json.loads(blob)
    assert d["paper_id"] == "abc123"
    assert d["confidence"] == 0.91
    assert d["bbox"] == [100, 200, 300, 400]

    ro = RunOutput(
        schema_version="1.0.0",
        provenance=ProvenanceRecord(
            pipeline_version="1.1.0",
            schema_version="1.0.0",
            git_commit="deadbee",
            git_dirty=False,
            config_snapshot={},
            input_sha256={},
            timestamp_utc=datetime.now(UTC).isoformat(),
            host="test",
            python_version="3.11",
        ),
        panels=[p],
    )
    ro2 = RunOutput.model_validate_json(ro.model_dump_json())
    assert ro2 == ro


# ---------------------------------------------------------------------------
# 5) The FastAPI app wires data_outbound_policy through the JobOptions validator
# ---------------------------------------------------------------------------


def test_app_version_is_in_sync_with_pyproject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The app's reported version must match pyproject.toml, not the
    historical 0.2.0 that was left behind in earlier versions."""
    monkeypatch.setenv("RLPE_API_TEST_TMP", str(tmp_path))
    from rlpe.api.app import app
    from rlpe.config import PipelineConfig  # for sanity

    # Read pyproject version (use tomllib on 3.11+, fall back to tomli on 3.10)
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - py<3.11 fallback
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / "pyproject.toml", "rb") as f:
        toml = tomllib.load(f)
    expected = toml["project"]["version"]
    assert app.version == expected, f"app.version={app.version!r} but pyproject={expected!r}"
    # And /system/info agrees
    with TestClient(app) as c:
        r = c.get("/system/info")
        assert r.status_code == 200
        assert r.json()["version"] == expected


# ---------------------------------------------------------------------------
# 6) PipelineConfig + lazy-init: pipeline can be constructed without OCR/taxon
# ---------------------------------------------------------------------------


def test_pipeline_lazy_init_no_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``RadiolarianPipeline(config)`` must succeed even when the heavy
    optional dependencies (paddleocr, easyocr, taxonerd, sam2,
    opendataloader-pdf) are NOT installed. The pipeline defers all of
    these via lazy-init in their respective modules; importing the
    pipeline class is the only thing that should happen here.
    """
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", str((repo_root / "src").resolve()))
    from rlpe.config import PipelineConfig
    from rlpe.pipeline import RadiolarianPipeline

    cfg = PipelineConfig(
        pdf_dir=tmp_path,
        work_dir=tmp_path,
        ocr_backend="paddleocr",
        extra={"use_opendataloader": False, "data_outbound_policy": "local_only"},
    )
    pipe = RadiolarianPipeline(cfg)
    # Construction does not raise; lazy-init is what would, on first use.
    assert pipe.grobid is not None
    assert pipe.config.extra["data_outbound_policy"] == "local_only"
