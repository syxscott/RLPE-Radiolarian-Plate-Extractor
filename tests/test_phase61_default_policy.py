"""Phase 61 Plan 4 (Bug 4.11): default data_outbound_policy must be api_full.

The default of ``api_redacted`` redacts images to a 256x256 thumbnail
and captions to 200 chars. MiniMax M3 needs the full-resolution image
to identify species accurately — thumbnail redaction silently drops
diagnostic morphology details (spine counts, lattice geometry) and
kills scientific F1.

The fix changes the CLI default and the backend's dataclass default to
``api_full``. ``api_redacted`` is still available via the flag for
operators working with sensitive preprints.
"""

from __future__ import annotations

import pytest


def test_default_policy_is_api_redacted():
    """Audit 2026-09-03 (BLOCKER-#2 fix): CLI default for
    --data-outbound-policy flipped from ``api_full`` to
    ``api_redacted`` so a fresh CLI run does NOT silently ship
    full-resolution images + verbatim captions + OCR + GROBID
    paragraphs to MiniMax. Operators who need the historical
    full-resolution behaviour must opt in explicitly via
    ``--i-understand-data-leaves-my-machine`` (which sets the
    ``RLPE_DATA_OUTBOUND_OPT_IN=1`` env var checked in
    ``MiniMaxM3Backend.__post_init__``)."""
    from rlpe.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(["--pdf-dir", "/tmp/a", "--work-dir", "/tmp/b"])
    assert ns.data_outbound_policy == "api_redacted"


def test_backend_default_is_api_redacted():
    """Audit 2026-09-03 (BLOCKER-#2 fix): MiniMaxM3Backend dataclass
    default for data_outbound_policy is now ``api_redacted`` so
    a fresh pipeline run does NOT silently ship full-resolution
    images to MiniMax. Operators opt in to ``api_full`` via the
    new CLI flag ``--i-understand-data-leaves-my-machine`` which
    sets the ``RLPE_DATA_OUTBOUND_OPT_IN=1`` env var checked in
    ``MiniMaxM3Backend.__post_init__``."""
    from dataclasses import fields

    from rlpe.llm_backends import MiniMaxM3Backend

    field_obj = next(f for f in fields(MiniMaxM3Backend) if f.name == "data_outbound_policy")
    assert field_obj.default == "api_redacted"
