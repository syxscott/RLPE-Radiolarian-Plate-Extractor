"""RLPE GUI — Qt6 desktop application for radiolarian plate extraction.

Phase 32 + 33 deliver a complete native desktop GUI built on
PySide6 / Qt6. This package does not duplicate the FastAPI web UI;
it is a parallel interface that uses the same underlying
``RadiolarianPipeline`` class. Both surfaces can coexist.

Entry point: ``python main.py`` (or ``rlpe-gui`` script).
"""

from .constants import APP_VERSION as __version__

__all__ = ["main", "run_app", "__version__"]
