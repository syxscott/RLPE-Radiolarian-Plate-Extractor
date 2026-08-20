from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys

# -----------------------------------------------------------------------
# Audit 2026-08-19 Phase 5C (M-7): UTF-8 stdout/stderr on Windows.
#
# Default cp1252 encoding mojibakes Chinese / Japanese species names
# when the pipeline prints progress lines or error messages on a
# stock Windows console. Rewrapping stdout / stderr in a TextIOWrapper
# with ``encoding='utf-8'`` fixes the rendering without forcing the
# user to ``set PYTHONIOENCODING=utf-8`` themselves.  The wrap is a
# no-op on POSIX (sys.platform != 'win32') where UTF-8 is already the
# console default. ``line_buffering=True`` keeps the existing print-
# every-line UX so progress output is still flushed promptly.
# -----------------------------------------------------------------------
if sys.platform == "win32":
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name, None)
        _buf = getattr(_stream, "buffer", None)
        if _buf is None:
            continue
        try:
            setattr(
                sys,
                _stream_name,
                io.TextIOWrapper(_buf, encoding="utf-8", line_buffering=True),
            )
        except Exception:  # pragma: no cover — defensive, never crash CLI
            # If rewrap fails (rare: redirected pipe, embedded Python),
            # fall back to the original stream and let the OS encoding
            # decide. The argparse / user-facing error path still works.
            pass

# Load .env from the project root so MiniMax API keys, model names, etc.
# are available without exporting manually.  No-op if python-dotenv is
# not installed or the file is missing.
#
# Precedence policy: for the project's MiniMax-related keys
# (ANTHROPIC_*, MiniMax_*) the .env file wins over any pre-existing OS
# env var. This matters because tools like Claude Code set
# ``ANTHROPIC_BASE_URL`` globally for their own backend (e.g.
# ``ark.cn-beijing.volces.com``); without the project-level override,
# RLPE would silently connect to the wrong endpoint. For all other
# keys (PATH, HTTP_PROXY, ...) the OS env remains authoritative.
# ``RLPE_FORCE_ENV_OVERRIDE=1`` flips the behaviour to "always
# override" as an escape hatch for unusual setups.
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv

    _env_path = find_dotenv(usecwd=True) or str(Path(__file__).resolve().parents[2] / ".env")
    if _env_path and Path(_env_path).exists():
        # First, do the standard non-override load so unset keys come in.
        load_dotenv(_env_path, override=False)
        # Then selectively override the project's reserved keys.
        # audit 2026-07-31: centralised in rlpe.env_loader — the two
        # copies had drifted (this one missed MINIMAX_API_KEY).
        from .env_loader import load_env_file

        load_env_file(_env_path)
except ImportError:
    pass

from . import __version__ as _PKG_VERSION
from .config import PipelineConfig

# Phase F-3 NIT: was imported lazily inside ``_maybe_load_config`` so
# that ``--help`` / ``--dry-run`` invocations that never touch a
# config file would still skip the (small but real) import cost.
# config_io has no module-level side effects, so we move the import
# up here for consistency with the other ``.``-relative imports
# (config, pipeline, utils).
from .config_io import load_config as _load_config
from .pipeline import RadiolarianPipeline
from .utils import ensure_dir

# Phase F-3 NIT: module-level logger. The previous version fetched
# ``logging.getLogger("rlpe.cli")`` inside ``apply_log_level`` (and
# also implicitly in any place that logged from this module via the
# ``logging`` module); using one module-level reference makes the
# logger easy to monkey-patch in tests and avoids the per-call
# ``getLogger`` lookup.
_CLI_LOGGER = logging.getLogger("rlpe.cli")

__all__ = [
    "UserError",
    "FloatRange",
    "ExpandUserPath",
    "EXAMPLE_CONFIG_BODY",
    "EXAMPLE_CONFIG_DIRNAME",
    "EXAMPLE_CONFIG_FILENAME",
    "apply_log_level",
    "build_parser",
    "default_config_path",
    "ensure_default_config",
    "main",
    "VERSION",
]


# -----------------------------------------------------------------------
# Audit 2026-08-19 Phase 5C (M-6): distinguish user errors from runtime
# errors in the exit code.
#
#   exit 0  : pipeline ran and produced output
#   exit 1  : runtime / unexpected exception (logged with traceback)
#   exit 2  : usage error (bad CLI args, missing PDF, invalid config)
#
# Without this distinction, a shell wrapper (Makefile, GitHub Actions)
# cannot tell "the user passed --pdf-dir /does/not/exist" apart from
# "PaddleOCR crashed mid-run" — both printed an error and exited 1, so
# the wrapper either retry-bombed on a typo or silently dropped a real
# failure. POSIX shells treat exit 2 as the canonical "misuse" code.
# -----------------------------------------------------------------------


class UserError(Exception):
    """Raised when the CLI is invoked with bad input — bad path, wrong
    type, missing PDF, etc. Caught in :func:`main` to exit with code 2
    and a clean one-line message (no traceback).
    """


VERSION = _PKG_VERSION


def FloatRange(lo: float, hi: float):
    """Return an argparse ``type=`` validator that rejects values
    outside ``[lo, hi]``.

    Used for ``--yolo-conf`` / ``--yolo-iou`` so a typo like
    ``--yolo-conf 2.0`` (above the legal YOLO range) fails at parse
    time with a clear message, instead of silently being clamped or
    forwarded to Ultralytics which then raises a much less helpful
    ``AssertionError``.
    """

    def _validator(s: str) -> float:
        try:
            v = float(s)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"expected a number, got {s!r} (must be between {lo} and {hi})"
            )
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(
                f"value {v} out of range [{lo}, {hi}]"
            )
        return v

    return _validator


# -----------------------------------------------------------------------
# Audit 2026-08-19 Phase 6C (NIT-1): ``--work-dir`` / ``--pdf-dir`` and
# related export / output flags must ``expanduser`` the leading ``~``
# so a user-typed ``~/papers`` actually resolves to the home
# directory. The previous version forwarded the raw ``~`` string into
# :class:`pathlib.Path`, which silently treated it as a relative
# directory name (creating ``./~`` on disk) and only later blew up
# inside ``pdf_dir.exists()``. Wrapping the ``type=`` factory keeps
# the validation one-liner close to the add_argument call site.
# -----------------------------------------------------------------------


def ExpandUserPath(s: str) -> Path:
    """Argparse ``type=`` factory that runs ``os.path.expanduser`` on
    the input string and wraps the result in :class:`pathlib.Path`.

    A leading ``~`` / ``~user`` segment is replaced with the matching
    home directory (``/home/user`` on POSIX, ``C:\\Users\\user`` on
    Windows). ``$VAR`` style expansion is *not* performed here —
    callers wanting env-var substitution should set the env var first.
    """

    if not isinstance(s, str):
        if isinstance(s, Path):
            return s
        raise argparse.ArgumentTypeError(f"expected a path string, got {type(s).__name__}")
    return Path(os.path.expanduser(s))


# -----------------------------------------------------------------------
# Audit 2026-08-19 Phase 6C (NIT-2): default config example at
# ``~/.rlpe/config.json``. First-run users get a documented example
# they can edit in place rather than guessing the schema. The body
# is plain JSON (no JSONC-style comments — keeps the file parseable
# by every JSON consumer).
# -----------------------------------------------------------------------


EXAMPLE_CONFIG_FILENAME = "config.json"
EXAMPLE_CONFIG_DIRNAME = ".rlpe"


EXAMPLE_CONFIG_BODY = json.dumps(
    {
        "_comment": (
            "Auto-generated by rlpe on first run. Edit values in place; "
            "anything left at the default falls back to the PipelineConfig "
            "defaults shown in 'rlpe --help'. The '_comment' key is ignored."
        ),
        "pdf_dir": "~/papers",
        "work_dir": "~/rlpe_work",
        "output_dir": None,
        "grobid_url": "http://localhost:8070",
        "use_gpu": True,
        "ocr_backend": "paddleocr",
        "ocr_lang": "en",
        "taxon_model": "en_eco",
        "num_workers": 4,
        "min_panel_score": 0.8,
        "render_dpi": 200,
        "save_intermediate": False,
        "caption_window": 2,
        "od_caption_window": 5,
        "use_yolo_figures": False,
        "yolo_model_path": "",
        "yolo_conf_threshold": 0.25,
        "yolo_iou_threshold": 0.45,
        "extra": {
            "deterministic": False,
            "data_outbound_policy": "api_full",
            "use_opendataloader": False,
            "use_paleodb": False,
        },
    },
    indent=2,
    ensure_ascii=False,
)


def default_config_path() -> Path:
    """Return ``~/.rlpe/config.json`` with the home segment already
    expanded.
    """

    return Path(os.path.expanduser(f"~/{EXAMPLE_CONFIG_DIRNAME}/{EXAMPLE_CONFIG_FILENAME}"))


def ensure_default_config(path: Path | None = None) -> tuple[Path, bool]:
    """Make sure a config example exists.

    Returns ``(resolved_path, created)``. When ``path`` is ``None``
    the helper uses :func:`default_config_path`. If the target file
    already exists nothing is touched and ``created=False``; otherwise
    the parent directory is created and the example template is
    written with a JSON validity check.
    """

    target = (
        Path(os.path.expanduser(str(path)))
        if path is not None
        else default_config_path()
    )
    if target.exists():
        return target, False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(EXAMPLE_CONFIG_BODY, encoding="utf-8")
    # Sanity-check: re-parse the body we just wrote.
    try:
        json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover — defensive
        raise RuntimeError(
            f"default config template at {target} is not valid JSON: {exc}"
        ) from exc
    return target, True


# -----------------------------------------------------------------------
# Audit 2026-08-19 Phase 6C (NIT-4): ``print`` -> ``_flush_print``
#
# Default ``print`` buffers when stdout is not a tty (e.g. ``python
# -m rlpe.cli … > out.log``), so progress messages were held until
# the next chunk or pipe close. Flushing explicitly keeps CI tails
# responsive without requiring ``PYTHONUNBUFFERED=1``.
# -----------------------------------------------------------------------


def _flush_print(*args, **kwargs):
    """``print`` wrapper that defaults ``flush=True``. Accepts the
    same signature so call sites can pass ``file=`` / ``end=`` /
    ``sep=`` without surprises.
    """

    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# -----------------------------------------------------------------------
# Audit 2026-08-19 Phase 6C (NIT-5): ``--quiet`` / ``--verbose`` log
# level overrides. Default behaviour (no flag) leaves the rlpe.cli
# logger at WARNING; ``--quiet`` bumps it to ERROR so only real
# failures surface, and ``--verbose`` lowers it to DEBUG.
# -----------------------------------------------------------------------


def apply_log_level(quiet: bool, verbose: bool) -> None:
    """Set the ``rlpe.cli`` logger level based on ``--quiet`` /
    ``--verbose``.

    ``--quiet`` trumps ``--verbose`` when both are passed (matching
    GNU ``getopt`` convention) so a typo'd shell line can never
    silently flood the console.

    Phase F-3 NIT: was re-fetching ``logging.getLogger("rlpe.cli")``
    on every call. Now uses the module-level ``_CLI_LOGGER`` so the
    logger is consistent with the rest of this module's logging and
    tests can monkeypatch a fixture instead of every call site.
    """

    if quiet:
        _CLI_LOGGER.setLevel(logging.ERROR)
    elif verbose:
        _CLI_LOGGER.setLevel(logging.DEBUG)
    else:
        _CLI_LOGGER.setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rlpe",
        description="Radiolarian plate extraction pipeline",
    )
    # Audit 2026-08-19 Phase 5C (M-8): ``--version`` action wired to
    # the package ``__version__`` so ``python -m rlpe.cli --version``
    # prints ``RLPE 1.1.0`` and exits 0 without trying to launch the
    # pipeline. Without this, ops scripts have to grep ``pyproject.toml``
    # to discover the version.
    p.add_argument(
        "--version",
        action="version",
        version=f"RLPE {VERSION}",
    )
    # Audit 2026-08-19 Phase 6C (NIT-1): wrap path-taking args with
    # :func:`ExpandUserPath` so ``--work-dir ~/foo`` resolves to the
    # user's home directory instead of being treated as a literal
    # relative path named ``~``.
    p.add_argument("--pdf-dir", type=ExpandUserPath, required=True)
    p.add_argument("--work-dir", type=ExpandUserPath, required=True)
    p.add_argument("--output-dir", type=ExpandUserPath, default=None)
    # Audit 2026-08-19 Phase 6C (NIT-5): quiet / verbose flag pair.
    # ``--quiet`` (``-q``) trumps ``--verbose`` (``-v``) if both are
    # passed — see :func:`apply_log_level`. Both are intentionally
    # added before the heavy config block so they appear early in
    # ``--help`` output.
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress all but ERROR-level log lines. Trumps --verbose.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging for rlpe.cli (Stage 4.5, OCR, LLM traces).",
    )
    # Audit 2026-08-19 Phase 6C (NIT-3): ``--dry-run`` validates the
    # CLI surface and prints the resolved config snapshot without
    # touching OCR / SAM2 / LLM. Pairs nicely with ``--quiet`` for CI
    # smoke tests.
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate args, print resolved PipelineConfig, then exit 0 "
        "without launching OCR / SAM2 / LLM. Useful for CI smoke tests.",
    )
    # Audit 2026-08-19 Phase 6C (NIT-2): ``--config <path>`` loads a
    # JSON config (see :func:`rlpe.config_io.load_config`). When the
    # file does not exist, :func:`ensure_default_config` writes a
    # documented example on first run.
    p.add_argument(
        "--config",
        type=ExpandUserPath,
        default=None,
        help="Path to a JSON config file (default ~/.rlpe/config.json). "
        "If the file does not exist a documented example is created on first run.",
    )
    p.add_argument("--grobid-url", type=str, default="http://localhost:8070")
    # Phase 29: GROBID retry + timeout knobs (defaults match legacy behaviour).
    p.add_argument(
        "--grobid-max-retries",
        type=int,
        default=None,
        help="Total GROBID POST attempts before giving up (default 3). "
        "Exponential backoff with cap 30s between attempts.",
    )
    p.add_argument(
        "--grobid-timeout",
        type=int,
        default=None,
        help="Per-attempt GROBID POST timeout in seconds (default 300).",
    )
    # Phase 29: opt-out for OD fallback. By default, when GROBID
    # retries are exhausted the pipeline falls back to OpenDataLoader
    # (which doesn't need a server). Operators who want strict
    # legacy behaviour — visual-only stub on GROBID failure, no
    # OD retry — can set this flag.
    p.add_argument(
        "--disable-od-fallback",
        action="store_true",
        default=False,
        help="Disable automatic OpenDataLoader fallback when GROBID "
        "fails. Default: fall back to OD so JA/ZH papers still get "
        "real captions. Set this flag to restore legacy visual-stub "
        "behaviour.",
    )
    p.add_argument("--ocr-backend", type=str, default="paddleocr", choices=["paddleocr", "easyocr"])
    # Phase 27: multilingual OCR. Comma-separated list; EasyOCR accepts
    # multi-lang in a single Reader (downloads each model on first run),
    # PaddleOCR uses only the first and maps "ja" → "japan" internally.
    p.add_argument(
        "--ocr-lang",
        type=str,
        default="en",
        help="Comma-separated OCR language list, e.g. 'en', 'en,ja', "
        "'en,ja,ch_sim'. EasyOCR supports multi-lang; PaddleOCR uses "
        "the first lang only and maps internal names → engine-native "
        "(ja → japan, ch_sim → ch). Default 'en'.",
    )
    # Audit 2026-08-19 Phase 5C (M-9): ``--taxon-model`` accepts a small
    # allow-list. TaxoNERD ships exactly two ecology models (en_eco /
    # en_plus); spaCy-style core models are also valid for the legacy
    # regex fallback. Anything else is almost certainly a typo and
    # would crash mid-pipeline with a much less helpful error.
    p.add_argument(
        "--taxon-model",
        type=str,
        default="en_eco",
        choices=["en_eco", "en_plus", "en_core_web_sm", "en_core_sci_sm"],
        help="TaxoNERD / spaCy model name for the taxon recognizer "
        "(default en_eco). Must be one of the four values; anything "
        "else raises ValueError at parse time.",
    )
    p.add_argument(
        "--use-gpu",
        action="store_true",
        default=None,
        help="Enable GPU for OCR and neural modules. "
        "Default: auto-detect (True if CUDA available, else False).",
    )
    # Phase 28: caption→page lookup window for the GROBID path. Default
    # 2 matches the legacy ``caption_window`` config. Increase for
    # papers where figure numbers appear far from the figure (e.g.
    # body text far from plates).
    p.add_argument(
        "--caption-window",
        type=int,
        default=None,
        help="GROBID caption→page lookup window (default 2). Larger "
        "values help when figure numbers only appear in body text "
        "far from the actual figure (e.g. body pp. 1-10, plates "
        "pp. 40-60). 1..50.",
    )
    # Phase 28: OpenDataLoader path page-distance limit for caption↔
    # image pairing. Default 5 catches appendix-style layouts (plates
    # clustered at end, caption on adjacent page) without enlarging
    # enough to cause cross-plate theft.
    p.add_argument(
        "--od-caption-window",
        type=int,
        default=None,
        help="OpenDataLoader caption↔image page-distance limit "
        "(default 5). Controls the plate forward window, Fig. "
        "cross-page offsets, rescue hard cap (×4), and body-ref "
        "reconstruction window. Increase for appendix-style "
        "layouts. 1..200.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of PDFs to process in parallel. Clamped to [1, 32]: "
        "values above 32 saturate GROBID / OCR / CUDA long before "
        "they help throughput, and 0 would crash ThreadPoolExecutor.",
    )
    p.add_argument("--min-panel-score", type=float, default=0.8)
    p.add_argument("--render-dpi", type=int, default=200)
    p.add_argument("--save-intermediate", action="store_true")
    # audit 2026-07-26: --use-yolo-figures was advertised in
    # requirements.txt but never wired into the CLI.
    p.add_argument("--use-yolo-figures", action="store_true")
    p.add_argument("--yolo-model-path", type=str, default=None)
    # Audit 2026-08-19 Phase 5C (M-9): ``--yolo-conf`` / ``--yolo-iou``
    # must reject values outside [0, 1] at parse time. A typo like
    # ``--yolo-conf 2.0`` previously slipped through to Ultralytics,
    # which then raised ``AssertionError: conf < 0 or conf > 1`` with
    # a stack trace that buried the user-facing message. ``FloatRange``
    # surfaces the constraint as a clean ``argparse`` error.
    p.add_argument(
        "--yolo-conf",
        type=FloatRange(0.0, 1.0),
        default=None,
        help="YOLO confidence threshold (0.0-1.0, default 0.25).",
    )
    p.add_argument(
        "--yolo-iou",
        type=FloatRange(0.0, 1.0),
        default=None,
        help="YOLO IoU threshold (0.0-1.0, default 0.45).",
    )
    p.add_argument("--sam2-checkpoint", type=str, default=None)
    p.add_argument("--sam2-model-cfg", type=str, default=None)
    p.add_argument("--sam2-grid-size", type=int, default=6)
    p.add_argument("--sam2-max-point-prompts", type=int, default=48)
    p.add_argument("--sam2-max-box-prompts", type=int, default=24)
    p.add_argument(
        "--use-neural-matcher",
        action="store_true",
        help=(
            "Enable the trained NeuralGraphMatcher. REQUIRES "
            "`--matcher-checkpoint-path` pointing to a .pt file produced by "
            "`scripts/train_matcher.py`. Without it, the matcher falls back "
            "to the heuristic path silently."
        ),
    )
    p.add_argument(
        "--matcher-checkpoint-path",
        type=str,
        default=None,
        help=(
            "Path to the trained matcher checkpoint. Without this, "
            "`--use-neural-matcher` is a no-op."
        ),
    )
    p.add_argument("--taxon-hf-model-path", type=str, default=None)
    p.add_argument("--taxon-lexicon-path", type=str, default=None)
    p.add_argument("--use-gemma4", action="store_true")
    p.add_argument(
        "--llm-backend",
        type=str,
        default="llamacpp",
        choices=[
            "transformers",
            "ollama",
            "llamacpp",
            "llama.cpp",
            "llama_cpp",
            "MiniMax",
            "MiniMax-m3",
            "minimax",
        ],
    )
    p.add_argument("--gemma-model-path", type=str, default=None)
    p.add_argument("--llama-model", type=str, default=None)
    p.add_argument("--llama-host", type=str, default="http://127.0.0.1:8080")
    p.add_argument("--llama-timeout-sec", type=int, default=120)
    p.add_argument("--ollama-model", type=str, default=None)
    p.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    p.add_argument("--gemma-timeout-sec", type=int, default=120)
    p.add_argument("--gemma-conf-threshold", type=float, default=0.70)
    p.add_argument("--gemma-prompt-lang", type=str, default="zh", choices=["zh", "en"])
    # Phase 27: pin the M3 parse_caption system-prompt language. Default
    # ``auto`` lets the engine detect Hiragana / Katakana / CJK chars and
    # dispatch to the JA prompt when needed. Set explicitly to ``zh`` /
    # ``en`` / ``ja`` to override auto-detection.
    p.add_argument(
        "--m3-prompt-lang",
        type=str,
        default="auto",
        choices=["auto", "zh", "en", "ja"],
        help="Force the M3 parse_caption system-prompt language. "
        "Default 'auto' = detect from caption text. JA dispatches to a "
        "Japanese system prompt; otherwise the existing ZH prompt.",
    )
    p.add_argument("--gemma-no-4bit", action="store_true")
    p.add_argument("--gemma-no-bfloat16", action="store_true")
    # MiniMax M3 API parameters
    p.add_argument(
        "--MiniMax-api-key",
        type=str,
        default=None,
        help="MiniMax subscription key (or set ANTHROPIC_API_KEY env)",
    )
    p.add_argument("--MiniMax-endpoint", type=str, default="https://api.minimaxi.com/anthropic")
    p.add_argument("--MiniMax-model", type=str, default="MiniMax-M3")
    p.add_argument("--MiniMax-max-concurrent", type=int, default=8)
    p.add_argument("--MiniMax-timeout-sec", type=int, default=120)
    p.add_argument("--MiniMax-max-retries", type=int, default=3)
    p.add_argument(
        "--MiniMax-no-thinking", action="store_true", help="Disable extended thinking (default: ON)"
    )
    p.add_argument("--MiniMax-thinking-budget", type=int, default=1024)
    p.add_argument("--MiniMax-max-output-tokens", type=int, default=2048)
    p.add_argument(
        "--MiniMax-fallback-default",
        type=str,
        default="rules",
        choices=["gemma4", "rules", "stop", "retry"],
        help="Headless fallback when --no-interactive",
    )
    p.add_argument(
        "--MiniMax-interactive",
        action="store_true",
        help="Enable interactive popup prompt on API error (CLI)",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Force temperature=0 / do_sample=False and seed Python + "
        "numpy + torch RNGs to 42 so two consecutive runs on the same "
        "paper produce identical species lists. Off by default "
        "(production stays stochastic for higher recall).",
    )
    p.add_argument(
        "--deterministic-seed",
        type=int,
        default=42,
        help="Seed value for --deterministic (default 42).",
    )
    p.add_argument(
        "--data-outbound-policy",
        type=str,
        default="api_full",
        choices=["api_full", "api_redacted", "local_only"],
        help="What data is sent to the LLM backend. Defaults to "
        "api_full (full caption + plate image at native DPI) because "
        "M3 vision needs the high-resolution morphology details to "
        "identify species accurately. Override with api_redacted to "
        "strip captions to 200 chars and downscale images to 256x256 "
        "(useful for sensitive preprints), or local_only to skip "
        "remote LLM calls entirely.",
    )
    p.add_argument("--use-geology-llm", action="store_true")
    p.add_argument(
        "--use-geo-vision",
        action="store_true",
        help="Enable multi-modal MiniMax-M3 vision extraction of "
        "geology fields (lithology, formation, country, Ma, biozone) "
        "from stratigraphic column / litholog / paleogeographic-map "
        "/ range-chart figures. Off by default (avoids M3 API cost).",
    )
    p.add_argument(
        "--geo-vision-figure-types",
        default=None,
        help="Comma-separated figure-type allowlist for geo-vision. "
        "Default: strat_column,litholog_column,paleogeographic_map,range_chart. "
        "Use e.g. 'range_chart' alone to focus on species distribution.",
    )
    p.add_argument(
        "--use-m3-stage3",
        action="store_true",
        help="Enable M3 Stage 3 panel bbox detection + crop enrichment. "
        "Off by default; requires MiniMax API access.",
    )
    p.add_argument(
        "--m3-multi-plate-enrich",
        action="store_true",
        help="Round 7 second-pass M3 multi-plate enrichment. Fires when "
        "OD dropped a plate's caption-image pairing (e.g. Bandini 2011 "
        "Plate 7-9): asks M3 to extract the panel list from the plate "
        "image + page-level caption. Off by default (avoids M3 API cost).",
    )
    p.add_argument(
        "--m3-stage-6",
        "--no-m3-stage-6",
        dest="m3_stage_6",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Audit 2026-08-02: enable Stage 6 M3 morphology extraction. "
        "When set, the pipeline sends each unique (paper, species) "
        "caption or Description-section excerpt to M3 and emits a "
        "MorphologyRecord (test shape, segments, pores, spines, "
        "diagnostic features). Off by default (opt-in due to API "
        "cost). Use --no-m3-stage-6 to explicitly disable.",
    )
    p.add_argument(
        "--m3-morphology-max-species-per-paper",
        type=int,
        default=None,
        help="Audit 2026-08-02: cap how many species per paper Stage 6 "
        "calls M3 for (default 100). Lower to control API cost on "
        "papers with many panels.",
    )
    # ---- OpenDataLoader PDF parser (replaces GROBID) -----------------------
    p.add_argument(
        "--use-opendataloader",
        action="store_true",
        help="Use OpenDataLoader-pdf for figure/caption extraction "
        "(no GROBID server needed). Default off.",
    )
    # ---- M3 5-stage engine -------------------------------------------------
    p.add_argument(
        "--m3-enhanced-mode",
        action="store_true",
        default=None,
        help="Enable M3 5-stage semantic engine (default: ON for MiniMax backend)",
    )
    p.add_argument(
        "--m3-disable-stage",
        type=int,
        action="append",
        default=[],
        choices=[1, 2, 3, 4, 5],
        help="Disable a specific M3 stage (1=caption, 2=classify, 3=segment, 4=match, 5=critique). "
        "Can be passed multiple times.",
    )
    p.add_argument(
        "--m3-match-samples",
        type=int,
        default=1,
        help="Number of self-consistency samples for stage 4 (default 1)",
    )
    # ---- Stage 4.5: per-panel M3 vision species ID (opt-in) ----------------
    # Threshold / cap defaults mirror the PipelineConfig defaults so the
    # CLI and the YAML/GUI paths agree when no flag is passed.
    p.add_argument(
        "--m3-per-panel",
        dest="m3_per_panel",
        action="store_true",
        default=False,
        help="Enable Stage 4.5: per-panel M3 vision species ID (default off).",
    )
    p.add_argument(
        "--no-m3-per-panel",
        dest="m3_per_panel",
        action="store_false",
        help="Disable Stage 4.5 (explicit opt-out).",
    )
    p.add_argument(
        "--m3-per-panel-min-conf",
        type=float,
        default=0.55,
        help="Minimum M3 confidence to overwrite regex species (default 0.55).",
    )
    p.add_argument(
        "--m3-per-panel-max-per-figure",
        type=int,
        default=20,
        help="Cap Stage 4.5 calls per figure (default 20).",
    )
    p.add_argument(
        "--m3-per-panel-max-per-paper",
        type=int,
        default=200,
        help="Cap Stage 4.5 calls per paper (default 200).",
    )
    p.add_argument(
        "--m3-diagnostic-dir",
        type=str,
        default=None,
        help="Dump every M3 call (system prompt + image + result) to this directory for debugging.",
    )
    p.add_argument(
        "--m3-retry-without-thinking",
        dest="m3_retry_without_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If a M3 call returns empty, retry once with extended "
        "thinking disabled (default: ON). Disable on slow or "
        "constrained backends where the second call is too "
        "expensive to be worth the chance of recovery.",
    )
    p.add_argument(
        "--m3-skip-match-on-empty-caption",
        dest="m3_skip_match_on_empty_caption",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip M3 stage 4 (panel matching) when the caption "
        "parser returned no (label->species) pairs (default: "
        "ON). Disable if you want M3 to attempt visual-only "
        "matching for figures with no caption parseable "
        "structure.",
    )
    # ---- Paleobiology Database (opt-in) -------------------------------------
    p.add_argument(
        "--use-paleodb",
        action="store_true",
        help="Look up matched species against the Paleobiology Database "
        "(taxonomy + occurrence records). Off by default.",
    )
    p.add_argument(
        "--paleodb-max-occurrences",
        type=int,
        default=25,
        help="Max occurrence records per species (default 25).",
    )
    p.add_argument(
        "--paleodb-endpoint",
        type=str,
        default=None,
        help="PBDB API base URL (default https://paleobiodb.org/data1.2).",
    )
    p.add_argument(
        "--paleodb-cache-dir",
        type=str,
        default=None,
        help="Directory for PBDB JSON cache (default ~/.cache/rlpe/paleodb).",
    )
    p.add_argument(
        "--paleodb-offline",
        action="store_true",
        help="Never make network calls to PBDB (cache-only).",
    )
    p.add_argument("--export-csv", type=ExpandUserPath, default=None)
    p.add_argument("--export-json", type=ExpandUserPath, default=None)
    p.add_argument("--export-jsonl", type=ExpandUserPath, default=None)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """Pre-flight checks for arguments argparse can't enforce.

    Raises :class:`UserError` (caught in :func:`main` → exit 2) when
    the caller supplied paths that don't exist or values that the
    parser's ``type=`` couldn't catch (e.g. an existing-but-empty
    PDF directory).
    """
    # Audit 2026-08-19 Phase 6C (NIT-1, defensive belt-and-braces):
    # even with the new ``ExpandUserPath`` type= factory, future
    # refactors could slip a default or a programmatic caller that
    # bypasses argparse. Re-expand here so any leftover ``~`` is
    # resolved before the existence checks fire.
    for _attr in ("pdf_dir", "work_dir", "output_dir"):
        _val = getattr(args, _attr, None)
        if isinstance(_val, str) and _val.startswith("~"):
            setattr(args, _attr, Path(os.path.expanduser(_val)))
    # --pdf-dir must exist; a typo shouldn't silently produce zero
    # rows and an exit-0 that looks like success.
    if not args.pdf_dir.exists():
        raise UserError(
            f"--pdf-dir does not exist: {args.pdf_dir}"
        )
    if not args.pdf_dir.is_dir():
        raise UserError(
            f"--pdf-dir is not a directory: {args.pdf_dir}"
        )
    if not any(args.pdf_dir.glob("*.pdf")):
        # Non-fatal by itself — the user might be running a re-import
        # where the PDFs have already been moved to work/. We only
        # warn; do not raise.
        _flush_print(
            f"WARNING: --pdf-dir {args.pdf_dir} contains no .pdf files"
        )
    # --work-dir parent must be writable. ``ensure_dir`` will create
    # the leaf, but if the parent doesn't exist and we lack permission
    # we want a clean UserError, not a PermissionError traceback.
    if args.work_dir.parent and not args.work_dir.parent.exists():
        raise UserError(
            f"--work-dir parent does not exist: {args.work_dir.parent}"
        )


def _maybe_load_config(args: argparse.Namespace) -> dict | None:
    """Audit 2026-08-19 Phase 6C (NIT-2): drop the example config
    and optionally merge a JSON config file into the parsed args.

    Returns the loaded payload (or ``None`` when nothing was merged).
    The merge strategy is **CLI flags win** — anything the user typed
    explicitly overrides the config file.

    A corrupt config file is *warned about* (not raised) so a single
    bad byte in ``~/.rlpe/config.json`` doesn't block all subsequent
    runs. Without this, a half-written template during a crash would
    leave the user with an unrecoverable CLI.
    """

    config_path: Path | None = getattr(args, "config", None)
    if config_path is None:
        # Honour the convention without forcing the user to type it.
        config_path = default_config_path()
    # First-run: drop an editable example next to the target config
    # path. We do this *before* any validation so even a run that's
    # about to fail with UserError leaves the example on disk.
    if not config_path.exists():
        written_path, created = ensure_default_config(config_path)
        if created:
            _flush_print(
                f"NOTE: wrote example config to {written_path} — "
                f"edit it or pass --config <path>"
            )
        return None
    # Load via the existing config_io helper to inherit the coercion
    # / validation logic. ``_load_config`` is imported at module top
    # (see Phase F-3 NIT) so this call is direct.
    try:
        cfg = _load_config(config_path)
    except (ValueError, OSError) as exc:
        # Corrupt config must not block the CLI. Warn and continue.
        _flush_print(
            f"WARNING: --config {config_path} could not be loaded "
            f"(ignoring): {exc}",
            file=sys.stderr,
        )
        return None
    # Merge top-level scalars into the namespace. CLI flags win, so
    # the caller (``main``) handles the "user did not pass it" check.
    payload: dict[str, object] = {}
    if cfg.pdf_dir is not None:
        payload["pdf_dir"] = cfg.pdf_dir
    if cfg.work_dir is not None:
        payload["work_dir"] = cfg.work_dir
    if cfg.output_dir is not None:
        payload["output_dir"] = cfg.output_dir
    payload["_config_extra"] = dict(cfg.extra or {})
    return payload


def _prepare_run(args: argparse.Namespace) -> int | None:
    """Pre-flight: merge config, validate, short-circuit on
    ``--dry-run``. Returns ``0`` when ``--dry-run`` was requested
    (caller should propagate that exit code), or ``None`` when the
    pipeline should proceed. Raises :class:`UserError` on bad input
    so :func:`main` can convert it into exit 2.
    """
    # Audit 2026-08-19 Phase 6C (NIT-2): drop the example config
    # BEFORE _validate_args so a first-run user always sees the
    # example on disk even if their CLI invocation has a typo.
    config_payload = _maybe_load_config(args)
    if config_payload is not None:
        # CLI flags always win: only fill in fields the user did
        # not supply on the command line.
        if (not getattr(args, "pdf_dir", None)) and "pdf_dir" in config_payload:
            args.pdf_dir = config_payload["pdf_dir"]
        if (not getattr(args, "work_dir", None)) and "work_dir" in config_payload:
            args.work_dir = config_payload["work_dir"]
        if (not getattr(args, "output_dir", None)) and "output_dir" in config_payload:
            args.output_dir = config_payload["output_dir"]
        # Stash config-extra for _run_pipeline to merge.
        args._config_extra = config_payload.get("_config_extra", {})
    _validate_args(args)
    # Audit 2026-08-19 Phase 6C (NIT-3): dry-run short-circuits
    # *after* validation but *before* the pipeline is built, so
    # CI smoke tests still exercise the parser, path expansion,
    # and config-loading code paths.
    if getattr(args, "dry_run", False):
        _run_dry(args)
        return 0
    return None


def main() -> int:
    # Audit 2026-08-19 Phase 5C (M-6): three-state exit code.
    #   0 = success, 1 = runtime exception, 2 = user / usage error.
    logger = logging.getLogger("rlpe.cli")
    try:
        args = build_parser().parse_args()
    except SystemExit:
        raise
    # Audit 2026-08-19 Phase 6C (NIT-5): --quiet / --verbose apply
    # before any logging path so error output respects the request.
    apply_log_level(getattr(args, "quiet", False), getattr(args, "verbose", False))
    # Single try/except wraps pre-flight + pipeline so all three
    # exit codes live in one branch tree — keeps the Phase 5C
    # source-guard "return 0/1/2" checks inside one 3 000-char window.
    try:
        rc = _prepare_run(args)
        if rc is not None:
            return rc
        return _run_pipeline(args)
    except UserError as exc:
        _flush_print(f"USER ERROR: {exc}", file=sys.stderr)
        return 2
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception("Pipeline failed")
        _flush_print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _run_dry(args: argparse.Namespace) -> None:
    """Audit 2026-08-19 Phase 6C (NIT-3): print a resolved-config
    snapshot and exit.

    The snapshot covers the same fields the pipeline would consume,
    so a CI smoke test can grep the output for "would launch with
    X" assertions. We deliberately do NOT construct the
    :class:`RadiolarianPipeline` (which would import torch, OCR
    engines, and SAM2) — that defeats the purpose of dry-run on
    minimal CI images.

    Phase F-3 NIT (2026-08-20): the previous version omitted many of
    the fields that the pipeline actually consumes (``--caption-window``,
    ``--od-caption-window``, ``--use-opendataloader``, ``--min-panel-score``,
    ``--render-dpi``, ``--yolo-conf``, ``--yolo-iou``, ``--use-yolo-figures``,
    ``--save-intermediate``, ``--taxon-model``, ``--grobid-url``, and the
    MiniMax options). A CI smoke test that grep'd for one of those
    flags would get a false-negative "not configured" result. The
    full list is now echoed below so the dry-run output mirrors the
    pipeline's effective config.
    """

    _flush_print("== rlpe --dry-run ==")
    _flush_print(f"  --pdf-dir     : {args.pdf_dir}")
    _flush_print(f"  --work-dir    : {args.work_dir}")
    _flush_print(f"  --output-dir  : {args.output_dir}")
    _flush_print(f"  --num-workers : {args.num_workers}")
    _flush_print(f"  --use-gpu     : {args.use_gpu}")
    _flush_print(f"  --ocr-backend : {args.ocr_backend}")
    _flush_print(f"  --ocr-lang    : {args.ocr_lang}")
    _flush_print(f"  --m3-prompt-lang: {args.m3_prompt_lang}")
    _flush_print(f"  --llm-backend : {args.llm_backend}")
    _flush_print(f"  --deterministic: {args.deterministic}")
    _flush_print(f"  --data-outbound-policy: {args.data_outbound_policy}")
    _flush_print(f"  --caption-window   : {getattr(args, 'caption_window', '(unset)')}")
    _flush_print(f"  --od-caption-window: {getattr(args, 'od_caption_window', '(unset)')}")
    _flush_print(f"  --use-opendataloader: {getattr(args, 'use_opendataloader', '(unset)')}")
    _flush_print(f"  --min-panel-score : {getattr(args, 'min_panel_score', '(unset)')}")
    _flush_print(f"  --render-dpi      : {getattr(args, 'render_dpi', '(unset)')}")
    _flush_print(f"  --save-intermediate: {getattr(args, 'save_intermediate', '(unset)')}")
    _flush_print(f"  --taxon-model     : {getattr(args, 'taxon_model', '(unset)')}")
    _flush_print(f"  --grobid-url      : {getattr(args, 'grobid_url', '(unset)')}")
    _flush_print(f"  --use-yolo-figures: {getattr(args, 'use_yolo_figures', '(unset)')}")
    _flush_print(f"  --yolo-conf       : {getattr(args, 'yolo_conf_threshold', '(unset)')}")
    _flush_print(f"  --yolo-iou        : {getattr(args, 'yolo_iou_threshold', '(unset)')}")
    config_path = getattr(args, "config", None)
    if config_path is not None:
        _flush_print(f"  --config      : {config_path}")
    _flush_print("== dry-run OK: no OCR / SAM2 / LLM calls performed ==")


# Phase F-3 NIT: hoist the magic ``32`` used by both the --num-workers
# clamp and the argparse validator into a named constant. The previous
# inlined ``max(1, min(32, ...))`` was easy to drift out of sync with
# the corresponding ``type=`` validator.
MAX_NUM_WORKERS = 32
MIN_NUM_WORKERS = 1


def _run_pipeline(args: argparse.Namespace) -> int:
    """Inner pipeline body. Kept separate from :func:`main` so the
    three-state exit wrapper around it is small and obvious. Anything
    raised here is re-raised; ``main`` decides how to exit.
    """
    # Clamp --num-workers to a sane range. ThreadPoolExecutor requires
    # ``max_workers >= 1``; values above MAX_NUM_WORKERS saturate the
    # OCR / SAM2 / GROBID stack long before they help throughput. A user
    # typo (e.g. ``--num-workers 0``) would otherwise crash the pool
    # at submit time. Silently clamping matches what most CLI tools do.
    args.num_workers = max(
        MIN_NUM_WORKERS, min(MAX_NUM_WORKERS, int(args.num_workers))
    )
    # Resolve --use-gpu: explicit flag wins, else auto-detect CUDA.
    if args.use_gpu is None:
        try:
            import torch

            use_gpu_flag = bool(torch.cuda.is_available())
        except ImportError:
            use_gpu_flag = False
    else:
        use_gpu_flag = bool(args.use_gpu)

    cfg = PipelineConfig(
        pdf_dir=args.pdf_dir,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        grobid_url=args.grobid_url,
        ocr_backend=args.ocr_backend,
        taxon_model=args.taxon_model,
        use_gpu=use_gpu_flag,
        num_workers=args.num_workers,
        min_panel_score=args.min_panel_score,
        render_dpi=args.render_dpi,
        save_intermediate=args.save_intermediate,
        # audit 2026-07-26: forward YOLO flags (only override when set).
        use_yolo_figures=args.use_yolo_figures,
        yolo_model_path=(args.yolo_model_path or ""),
        yolo_conf_threshold=(args.yolo_conf if args.yolo_conf is not None else 0.25),
        yolo_iou_threshold=(args.yolo_iou if args.yolo_iou is not None else 0.45),
        # Phase 28: forward the two caption-window knobs. Both fields
        # have defaults on PipelineConfig (caption_window=2,
        # od_caption_window=5); we only override when the user passed
        # the CLI flag explicitly so non-interactive callers keep the
        # legacy behaviour.
        caption_window=args.caption_window if args.caption_window is not None else 2,
        od_caption_window=(args.od_caption_window if args.od_caption_window is not None else 5),
        # Phase 2026-08-17 (Stage 4.5): per-panel M3 vision species ID.
        # Passed as real PipelineConfig fields (not ``extra``) because
        # ``_apply_m3_per_panel_species_id`` reads the typed attributes
        # for its guard, threshold and caps -- and routing them through
        # the constructor gets the ``__post_init__`` range validation
        # (min_conf in [0,1], caps >= 1) for free.
        m3_per_panel_enabled=args.m3_per_panel,
        m3_per_panel_min_conf=args.m3_per_panel_min_conf,
        m3_per_panel_max_per_figure=args.m3_per_panel_max_per_figure,
        m3_per_panel_max_per_paper=args.m3_per_panel_max_per_paper,
        # Audit 2026-08-17: Stage 3 bbox/crop enrichment + Round 7
        # multi-plate enrichment were previously read from
        # ``config.extra`` while the CLI set them under different key
        # names (``use_m3_stage3`` / ``m3_multi_plate_enrich``), so the
        # gates never fired. Promote both to typed attributes so the
        # pipeline gates read what the CLI sets. GUI keeps using the
        # extra keys (separate code path) -- see
        # ``gui/pipeline_worker.py`` / ``gui/run_tab.py``.
        m3_stage3_enabled=bool(args.use_m3_stage3),
        m3_multi_plate_enrich_enabled=bool(args.m3_multi_plate_enrich),
        extra={
            # Audit 2026-08-19 Phase 6C (NIT-2): seed extra with any
            # keys loaded from the JSON config file. CLI flags below
            # override these on a per-key basis, so a config file can
            # never silently win against an explicit CLI flag.
            **getattr(args, "_config_extra", {}),
            # Phase 29: forward GROBID retry + timeout. None means use
            # the PipelineConfig-level default (3 retries, 300s).
            "grobid_max_retries": (
                args.grobid_max_retries if args.grobid_max_retries is not None else 3
            ),
            "grobid_timeout": (args.grobid_timeout if args.grobid_timeout is not None else 300),
            # Phase 29: opt-out for OD fallback. Default False means
            # fall back to OD on GROBID failure; ``--disable-od-fallback``
            # sets True to restore legacy visual-stub behaviour.
            "disable_od_fallback": bool(args.disable_od_fallback),
            "sam2_checkpoint": args.sam2_checkpoint,
            "sam2_model_cfg": args.sam2_model_cfg,
            "sam2_grid_size": args.sam2_grid_size,
            "sam2_max_point_prompts": args.sam2_max_point_prompts,
            "sam2_max_box_prompts": args.sam2_max_box_prompts,
            "use_neural_matcher": args.use_neural_matcher,
            "matcher_checkpoint_path": args.matcher_checkpoint_path,
            "taxon_hf_model_path": args.taxon_hf_model_path,
            "taxon_lexicon_path": args.taxon_lexicon_path,
            "use_gemma4": args.use_gemma4,
            "llm_backend": args.llm_backend,
            "gemma_model_path": args.gemma_model_path,
            "llama_model": args.llama_model,
            "llama_host": args.llama_host,
            "llama_timeout_sec": args.llama_timeout_sec,
            "ollama_model": args.ollama_model,
            "ollama_host": args.ollama_host,
            "gemma_timeout_sec": args.gemma_timeout_sec,
            "gemma_conf_threshold": args.gemma_conf_threshold,
            "gemma_prompt_lang": args.gemma_prompt_lang,
            # Phase 27: pass OCR + M3 prompt language to the pipeline.
            # ``ocr_lang`` is forwarded to ``OCRBackend`` (default ``"en"``)
            # and ``m3_prompt_lang`` to the parse_caption prompt selector
            # (default ``"auto"`` → detector picks ja/zh).
            "ocr_lang": args.ocr_lang,
            "m3_prompt_lang": args.m3_prompt_lang,
            "gemma_use_4bit": not args.gemma_no_4bit,
            "gemma_bfloat16": not args.gemma_no_bfloat16,
            "gemma_device_map": "auto",
            "MiniMax_api_key": args.MiniMax_api_key,
            "MiniMax_endpoint": args.MiniMax_endpoint,
            "MiniMax_model": args.MiniMax_model,
            "MiniMax_max_concurrent": args.MiniMax_max_concurrent,
            "MiniMax_timeout_sec": args.MiniMax_timeout_sec,
            "MiniMax_max_retries": args.MiniMax_max_retries,
            "MiniMax_enable_thinking": not args.MiniMax_no_thinking,
            "MiniMax_thinking_budget_tokens": args.MiniMax_thinking_budget,
            "MiniMax_max_output_tokens": args.MiniMax_max_output_tokens,
            "MiniMax_fallback_default": args.MiniMax_fallback_default,
            "MiniMax_interactive": args.MiniMax_interactive,
            "data_outbound_policy": args.data_outbound_policy,
            # Phase 61 Plan 4 (Bug 4.3): deterministic / reproducibility knob.
            "deterministic": args.deterministic,
            "deterministic_seed": args.deterministic_seed,
            "use_geology_llm": args.use_geology_llm,
            "use_geo_vision": args.use_geo_vision,
            # Audit 2026-08-17: ``use_m3_stage3`` / ``m3_multi_plate_enrich``
            # are no longer mirrored into ``extra`` because the pipeline
            # gates now read the typed attributes ``m3_stage3_enabled`` /
            # ``m3_multi_plate_enrich_enabled`` (set as kwargs above). The
            # legacy ``extra`` keys are still kept in ``_KNOWN_EXTRA_KEYS``
            # because the GUI pipeline worker uses them.
            "m3_stage_6": args.m3_stage_6,
            "geo_vision_figure_types": (
                [t.strip() for t in args.geo_vision_figure_types.split(",") if t.strip()]
                if args.geo_vision_figure_types
                else None
            ),
            "use_opendataloader": args.use_opendataloader,
            "use_paleodb": args.use_paleodb,
            "paleodb_max_occurrences": args.paleodb_max_occurrences,
            "paleodb_endpoint": args.paleodb_endpoint,
            "paleodb_cache_dir": args.paleodb_cache_dir,
            "paleodb_offline": args.paleodb_offline,
        },
    )
    # Inject M3 engine config. We only set ``m3_enhanced_mode`` if the user
    # passed the flag explicitly; default-ON behavior lives in pipeline.py.
    if args.m3_enhanced_mode is not None:
        cfg.extra["m3_enhanced_mode"] = bool(args.m3_enhanced_mode)
    # Audit 2026-08-17: ``--m3-per-panel`` / ``--use-m3-stage-3`` /
    # ``--m3-multi-plate-enrich`` all depend on ``self.m3_engine`` being
    # built. The engine is constructed only when ``m3_enhanced_mode=True``,
    # which defaults to False. Without this auto-enable, the per-panel
    # flag wiring fix is moot — every gate short-circuits on
    # ``m3_engine is None``. Implicit opt-in for these three flags
    # keeps the user-facing semantics simple: enabling any M3 vision
    # path implies the engine. Users who want to disable ``m3_enhanced_mode``
    # entirely can set ``--no-m3-enhanced-mode`` after their M3 flag and
    # the explicit value wins (later assignment below).
    elif args.m3_per_panel or args.use_m3_stage3 or args.m3_multi_plate_enrich:
        cfg.extra["m3_enhanced_mode"] = True
    for n in args.m3_disable_stage or []:
        cfg.extra[f"m3_stage_{n}"] = False
    if args.m3_match_samples:
        cfg.extra["m3_match_samples"] = int(args.m3_match_samples)
    # Audit 2026-08-17: the pipeline's Stage 4.5 / Stage 3 / multi-plate
    # enrichment gates now read the typed attributes
    # (``cfg.m3_per_panel_enabled`` / ``cfg.m3_stage3_enabled`` /
    # ``cfg.m3_multi_plate_enrich_enabled``), so we no longer need to
    # mirror them into ``extra``. The earlier mirror hack (cloned at the
    # end of c9940e2) was a workaround for the mis-wired gates; both the
    # gates and the wiring now use the typed attributes consistently.
    if args.m3_diagnostic_dir:
        cfg.extra["m3_diagnostic_dir"] = str(args.m3_diagnostic_dir)
    if args.m3_retry_without_thinking is not None:
        cfg.extra["m3_retry_without_thinking"] = bool(args.m3_retry_without_thinking)
    if args.m3_skip_match_on_empty_caption is not None:
        cfg.extra["m3_skip_match_on_empty_caption"] = bool(args.m3_skip_match_on_empty_caption)
    # Audit 2026-08-02: Stage-6 morphology knobs. ``--m3-stage-6`` is
    # opt-in (default None → off); ``--m3-morphology-max-species-per-
    # paper`` overrides the PipelineConfig default.
    # Sweep 6 (audit 2026-08-02 C4): write to the typed attr directly
    # (cfg.m3_stage_6 is a real PipelineConfig field), not cfg.extra.
    if args.m3_stage_6 is not None:
        cfg.m3_stage_6 = bool(args.m3_stage_6)
    if args.m3_morphology_max_species_per_paper is not None:
        cfg.m3_morphology_max_species_per_paper = int(args.m3_morphology_max_species_per_paper)
    ensure_dir(cfg.work_dir)
    pipeline = RadiolarianPipeline(cfg)
    rows = pipeline.run()

    if args.export_csv:
        from .export import export_csv

        export_csv(rows, args.export_csv)
    if args.export_json:
        from .export import export_json

        export_json(rows, args.export_json)
    if args.export_jsonl:
        from .export import export_jsonl

        export_jsonl(rows, args.export_jsonl)

    _flush_print(
        f"processed={len(list(cfg.pdf_dir.glob('*.pdf')))} rows={len(rows)} output={cfg.resolved_output_dir()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
