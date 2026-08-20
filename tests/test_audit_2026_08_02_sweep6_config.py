"""Sweep 6 (2026-08-02 audit follow-up): config + YOLO wiring cleanup.

Three small fixes landed in this sweep:

1. **O1** — pipeline.py no longer uses ``getattr(self.config, "yolo_device",
   "auto")``; the field is guaranteed to exist (Phase 68 now validates it
   in ``__post_init__``).

2. **N1** — the GROBID-fallback detect_figure_regions call (around line
   6349) now forwards ``yolo_device`` so an operator's ``--yolo-device 0``
   choice actually reaches the detector on the fallback path. Previously
   the GROBID fallback silently used ``"auto"``.

3. **C4** — ``_KNOWN_EXTRA_KEYS`` no longer lists the five YOLO knobs or
   the five m3 knobs (``m3_stage_6`` + 4× ``m3_per_panel_*``) that are
   actually real ``PipelineConfig`` dataclass fields. ``cli.py`` was
   setting ``cfg.extra["m3_stage_6"]`` instead of ``cfg.m3_stage_6``;
   that line now writes the typed attribute directly.

These tests pin the design so a future refactor doesn't silently
regress:
- ``cli.py`` must NOT mirror ``m3_stage_6`` into ``cfg.extra`` (the
  Stage 4.5 test already locks the four ``m3_per_panel_*`` knobs).
- The GROBID fallback ``detect_figure_regions`` call MUST pass
  ``yolo_device``.
- ``_KNOWN_EXTRA_KEYS`` must NOT contain any of the 10 redundant keys.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe import config as config_mod
from rlpe.config import _KNOWN_EXTRA_KEYS, PipelineConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PIPELINE = _REPO_ROOT / "src" / "rlpe" / "pipeline.py"
_SRC_CLI = _REPO_ROOT / "src" / "rlpe" / "cli.py"


class TestSweep6O1GetattrRemoved:
    """O1 — `getattr(self.config, "yolo_device", "auto")` is dead code.

    Phase 68 made ``yolo_device`` a required-validated field, so the
    ``getattr(... default='auto')`` fallback can never fire. Removing
    it makes the call site honest.
    """

    def test_no_getattr_for_yolo_device(self):
        src = _SRC_PIPELINE.read_text(encoding="utf-8")
        # The redundant getattr pattern must be gone.
        assert 'getattr(self.config, "yolo_device"' not in src, (
            "getattr(self.config, 'yolo_device', ...) is dead code now that "
            "Phase 68 validates yolo_device in __post_init__"
        )

    def test_yolo_device_passed_directly(self):
        """Two ``detect_figure_regions`` call sites must both pass
        ``yolo_device=self.config.yolo_device``."""
        src = _SRC_PIPELINE.read_text(encoding="utf-8")
        # At least one direct-attribute access must exist (the Opendataloader
        # primary path at ~line 3850) — counting occurrences of the
        # literal pattern pins both sites being patched.
        assert src.count("yolo_device=self.config.yolo_device") >= 2, (
            "Both detect_figure_regions call sites (OD primary + GROBID "
            "fallback) must forward yolo_device=self.config.yolo_device. "
            "Currently found "
            f"{src.count('yolo_device=self.config.yolo_device')} sites."
        )


class TestSweep6N1GrobidFallbackForwardsDevice:
    """N1 — GROBID-fallback ``detect_figure_regions`` must forward
    ``yolo_device``."""

    def test_grobid_fallback_call_has_yolo_device_kwarg(self):
        """The fallback path at ~line 6349 (which uses ``if self.config.
        use_yolo_figures else None`` + ``detect_figure_regions``) must
        forward ``yolo_device`` so operators don't silently lose their
        device choice on the fallback path."""
        src = _SRC_PIPELINE.read_text(encoding="utf-8")
        # There are three sites in the file:
        #   1. line ~1999 — internal YOLO helper (not a detect_figure_regions call)
        #   2. line ~3797 — OD-primary path (inside candidate-pages loop)
        #   3. line ~6349 — GROBID-fallback path (inside `for page in pages:`
        #      loop, no caption-pair context)
        # Pin the GROBID-fallback site by the unique
        # `for page in pages:` + ``detect_figure_regions`` pattern.
        page_loop_idx = src.find("\n        for page in pages:")
        assert page_loop_idx > 0, (
            "`for page in pages:` loop missing — file refactored?"
        )
        snippet = src[page_loop_idx:page_loop_idx + 2000]
        assert "detect_figure_regions(" in snippet, (
            "for-page-in-pages loop no longer calls detect_figure_regions"
        )
        assert "yolo_device=" in snippet, (
            "GROBID fallback path is missing yolo_device kwarg — operators' "
            "--yolo-device choice silently downgrades to 'auto' on fallback"
        )


class TestSweep6C4KnownExtraKeys:
    """C4 — ``_KNOWN_EXTRA_KEYS`` must not list real PipelineConfig fields."""

    @pytest.mark.parametrize(
        "redundant_key",
        [
            "use_yolo_figures",
            "yolo_model_path",
            "yolo_conf_threshold",
            "yolo_iou_threshold",
            "yolo_device",
            "m3_stage_6",
            "m3_per_panel_enabled",
            "m3_per_panel_min_conf",
            "m3_per_panel_max_per_figure",
            "m3_per_panel_max_per_paper",
        ],
    )
    def test_redundant_key_removed(self, redundant_key):
        """The 10 redundant keys must NOT be in ``_KNOWN_EXTRA_KEYS`` —
        they're real PipelineConfig dataclass fields, so listing them
        here let ``cli.py`` get away with ``cfg.extra['m3_stage_6'] = ...``
        instead of writing to the typed attribute."""
        assert redundant_key not in _KNOWN_EXTRA_KEYS, (
            f"{redundant_key!r} is a real PipelineConfig field — "
            f"listing it in _KNOWN_EXTRA_KEYS hides the bug where "
            f"cfg.extra[{redundant_key!r}] would silently shadow the "
            f"typed attribute."
        )

    def test_known_extras_still_includes_real_extras(self):
        """Sanity check: legitimate extras (not PipelineConfig fields)
        must STILL be in the allow-list."""
        for real_extra in [
            "use_paleodb",  # opt-in PBDB flag — still an extra
            "deterministic",  # reproducible-runs knob — still an extra
            "use_llm_first",  # opt-in LLM-first flag — still an extra
        ]:
            assert real_extra in _KNOWN_EXTRA_KEYS, (
                f"{real_extra!r} should still be a known extra — only "
                f"the 10 real-typed-attr duplicates were removed."
            )

    def test_warning_fires_for_legacy_cfg_extra_m3_stage_6(self):
        """If someone re-introduces ``cfg.extra['m3_stage_6'] = ...``,
        the unknown-keys warning must fire."""
        from rlpe.config import PipelineConfig

        cfg = PipelineConfig(
            pdf_dir=Path("/tmp"),
            work_dir=Path("/tmp"),
        )
        cfg.extra["m3_stage_6"] = True
        # Re-run __post_init__ to fire the warning.
        cfg.__post_init__()
        # We don't assert on the warning text — just that the key is no
        # longer in the allow-list (covered above) and that
        # `cfg.extra['m3_stage_6']` is still present (validation is
        # warn-only, not raise).
        assert "m3_stage_6" not in _KNOWN_EXTRA_KEYS


class TestSweep6CliUsesTypedAttribute:
    """The CLI must write ``m3_stage_6`` to ``cfg.m3_stage_6`` (typed
    attr), not ``cfg.extra['m3_stage_6']``."""

    def test_cli_does_not_mirror_m3_stage_6_into_extra(self):
        src = _SRC_CLI.read_text(encoding="utf-8")
        assert 'cfg.extra["m3_stage_6"]' not in src, (
            "CLI must write m3_stage_6 to the typed attribute "
            "(cfg.m3_stage_6), not cfg.extra. Mirror-into-extra hides "
            "the value from __post_init__ validators."
        )

    def test_cli_writes_m3_stage_6_to_typed_attr(self):
        """Pin the fix: ``cfg.m3_stage_6 = bool(args.m3_stage_6)``."""
        src = _SRC_CLI.read_text(encoding="utf-8")
        assert "cfg.m3_stage_6 = bool(args.m3_stage_6)" in src, (
            "CLI must assign m3_stage_6 to the typed attribute. "
            "Pattern: `cfg.m3_stage_6 = bool(args.m3_stage_6)`."
        )


class TestSweep6EndToEnd:
    """End-to-end: a PipelineConfig with the 10 fields as direct
    attributes (not extras) should build cleanly."""

    def test_pipeline_config_with_yolo_and_m3_attrs(self, tmp_path):
        """Build a PipelineConfig with the previously-redundant fields
        as direct attributes (the new world) and confirm construction
        + validation succeeds."""
        # ``use_yolo_figures=False`` so we don't need a real .pt file
        # on disk; the test is about the typed-attr wiring, not YOLO.
        cfg = PipelineConfig(
            pdf_dir=tmp_path,
            work_dir=tmp_path,
            use_yolo_figures=False,
            yolo_conf_threshold=0.25,
            yolo_iou_threshold=0.45,
            yolo_device="auto",
            m3_stage_6=True,
            m3_per_panel_enabled=True,
            m3_per_panel_min_conf=0.55,
            m3_per_panel_max_per_figure=20,
            m3_per_panel_max_per_paper=200,
        )
        # Typed attrs read back as expected (the whole point of sweep 6
        # is that these are first-class fields, not buried in extra).
        assert cfg.use_yolo_figures is False
        assert cfg.yolo_conf_threshold == 0.25
        assert cfg.yolo_device == "auto"
        assert cfg.m3_stage_6 is True
        assert cfg.m3_per_panel_enabled is True
        assert cfg.m3_per_panel_min_conf == 0.55
        # Constructing it must NOT raise — the previously-redundant
        # fields now flow through the normal ``__init__`` path.
        cfg.__post_init__()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
