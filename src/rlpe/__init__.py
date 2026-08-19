"""RLPE: Radiolarian Literature Plate Extractor."""

# Single source of truth for the package version. Mirrored in
# ``pyproject.toml`` (``version = "1.1.0"``) and exposed here so the
# CLI's ``--version`` flag, GUI ``APP_VERSION`` and FastAPI
# ``info.version`` all read the same string without circular imports.
__version__: str = "1.1.0"

from .config import PipelineConfig


# Defer the heavy ``pipeline`` import until the caller actually asks
# for ``RadiolarianPipeline``. The full pipeline pulls in torch /
# gemma / paddleocr, none of which are needed by the lightweight
# helpers (config, evaluation, opendataloader_extractor, segmentation)
# that ``scripts/evaluate.py`` and the eval-only entry points use.
def __getattr__(name):
    if name == "RadiolarianPipeline":
        from .pipeline import RadiolarianPipeline

        return RadiolarianPipeline
    raise AttributeError(f"module 'rlpe' has no attribute {name!r}")


__all__ = ["PipelineConfig", "RadiolarianPipeline", "__version__"]
