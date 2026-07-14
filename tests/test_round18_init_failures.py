"""Round 18 source-guard tests: silent init-failure fixes.

Locks in fixes for 3 silent fallbacks found when the user ran a
Suzuki 2011 paper and got only 1 garbage row ("Annual number of"
matched as a species). Server log showed:

  Gemma4 backend init failed: MiniMax api_key not set.
  PaddleOCR init failed; falling back to EasyOCR
  TaxoNERD init failed (model='en_eco'): __init__() got an
    unexpected keyword argument 'model'

Each of these silently fell back to a less-capable path with
only a generic warning. The fixes here surface the actual reason
and use the right kwarg names for the installed library versions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(path: str) -> str:
    return Path(__file__).resolve().parents[1].joinpath(path).read_text(encoding="utf-8")


# --- 1) PaddleOCR init must use the device= kwarg, not use_gpu ---------


def test_paddleocr_uses_device_kwarg():
    """PaddleOCR 3.x replaced ``use_gpu`` with ``device``. The init
    code must try ``device`` first (3.x) and fall back to ``use_gpu``
    on TypeError for 2.x users."""
    src = _read("src/rlpe/ocr.py")
    assert "device=device_kw" in src or 'device="cpu"' in src or 'device="gpu"' in src, (
        "ocr.py is not using PaddleOCR 3.x's device= kwarg. The "
        "previous use_gpu=True kwarg raises ValueError on 3.x and "
        "silently fell back to EasyOCR."
    )


def test_paddleocr_failure_includes_exception_details():
    """The PaddleOCR failure warning must include the exception type
    + message so the operator can see WHY it failed (missing
    paddlepaddle? Wrong API?). The previous 'PaddleOCR init failed'
    one-liner masked the actual error."""
    src = _read("src/rlpe/ocr.py")
    # Find the paddleocr exception handler and verify it includes
    # type+message in the log call.
    idx = src.find("PaddleOCR init failed")
    assert idx > 0, "PaddleOCR failure warning is missing entirely"
    window = src[idx : idx + 400]
    assert "type(exc).__name__" in window and "exc" in window, (
        "PaddleOCR failure log must include type(exc).__name__ and "
        "the exception message — otherwise operators can't tell why "
        "PaddleOCR isn't working."
    )


# --- 2) TaxoNERD init must not pass model= kwarg ---------------------


def test_taxonerd_does_not_pass_model_kwarg():
    """TaxoNERD 1.5.x's __init__ signature is
    (self, prefer_gpu=False, verbose=False, logger=None). The
    previous ``TaxoNERD(model=self.model)`` raised TypeError on
    every init and silently fell back to regex-based species
    extraction."""
    src = _read("src/rlpe/taxon.py")
    # Walk the source line-by-line, tracking docstring state, and
    # only count TaxoNERD() calls on lines that are live code.
    import re

    has_live_call_with_prefer_gpu = False
    has_live_call_with_model = False
    in_docstring = False
    quote_char = None
    for line in src.splitlines():
        # Toggle docstring state on triple-quote boundaries
        # (handle both """ and ''' via the quote_char variable).
        if not in_docstring:
            stripped = line.lstrip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Check if it's a single-line docstring
                rest = stripped[3:]
                if '"""' in rest or "'''" in rest:
                    pass  # single-line, no state change
                else:
                    in_docstring = True
                    quote_char = stripped[:3]
                    continue
        else:
            if quote_char in line:
                in_docstring = False
                quote_char = None
            continue  # skip until docstring closes
        # Strip # comments — they may legitimately reference
        # the historical TaxoNERD(model=...) form.
        code = line.split("#", 1)[0]
        for m in re.finditer(r"TaxoNERD\([^)]*\)", code):
            snippet = m.group(0)
            if "model=" in snippet:
                has_live_call_with_model = True
            if "prefer_gpu" in snippet:
                has_live_call_with_prefer_gpu = True
    assert not has_live_call_with_model, (
        "taxon.py has a live TaxoNERD(model=...) call. TaxoNERD "
        "1.5.x's __init__ doesn't accept 'model' — the call "
        "raises TypeError and silently falls back to regex."
    )
    assert has_live_call_with_prefer_gpu, (
        "taxon.py must have a live TaxoNERD(prefer_gpu=...) call "
        "(or a bare TaxoNERD() fallback for 1.4.x)."
    )


# --- 3) ANTHROPIC_API_KEY must be a valid fallback source ----------


def test_pipeline_accepts_anthropic_api_key_as_fallback():
    """The project's .env uses ANTHROPIC_API_KEY as the documented
    user-facing key (Claude-Code-compatible name). The pipeline
    must inject this into MiniMax_api_key when no vendor-specific
    key is set, so web-UI jobs don't silently lose their key."""
    src = _read("src/rlpe/pipeline.py")
    # The injection happens in __init__ before _try_init_gemma.
    # Look for the pattern: 'if not self.config.extra.get(...) and
    # ANTHROPIC_API_KEY: ... self.config.extra["MiniMax_api_key"] ='
    assert "ANTHROPIC_API_KEY" in src, "Pipeline doesn't reference ANTHROPIC_API_KEY"
    assert (
        'extra["MiniMax_api_key"]' in src
        and "ANTHROPIC_API_KEY" in src.split('extra["MiniMax_api_key"]', 1)[0]
    ), (
        "pipeline.py doesn't inject ANTHROPIC_API_KEY into "
        "self.config.extra['MiniMax_api_key']. The MiniMax API is "
        "Anthropic-protocol — the same key works — but the backend "
        "builder only reads the explicit vendor key."
    )
