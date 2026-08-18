"""Phase 68 — yolo_device early validation.

The 2026-08-02 audit found that ``PipelineConfig.yolo_device`` accepts
any string at config-parse time. A typo (e.g. ``yolo_device="garbage"``)
silently sailed through YAML/JSON parsing and only crashed inside the
first ``YOLO(..., device="garbage")`` call with a cryptic
``AssertionError`` from ultralytics.

Phase 68 fix: an explicit allow-list check in
``PipelineConfig.__post_init__`` raises ``ValueError`` at config-load
time, with a clear message listing the valid values.

Valid values per the design comment at ``src/rlpe/config.py:182`` +
ultralytics conventions:
    - "auto" / ""  : let ultralytics auto-select (default)
    - "cpu"        : force CPU
    - "cuda"       : any CUDA-capable GPU
    - "mps"        : Apple Silicon GPU
    - "0".."7"     : specific GPU index
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.config import PipelineConfig


def _make_config(tmp_path: Path, **overrides) -> PipelineConfig:
    """Build a minimal PipelineConfig with the given yolo_device override."""
    kwargs = dict(
        pdf_dir=tmp_path,
        work_dir=tmp_path,
        save_intermediate=False,
        min_panel_score=0.0,
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


class TestYoloDeviceValidation:
    def test_default_auto_is_accepted(self, tmp_path):
        """Default yolo_device='auto' passes validation."""
        cfg = _make_config(tmp_path)
        assert cfg.yolo_device == "auto"

    @pytest.mark.parametrize(
        "valid",
        ["auto", "cpu", "cuda", "mps", "0", "1", "2", "3", "4", "5", "6", "7"],
    )
    def test_valid_values_accepted(self, tmp_path, valid):
        """All values in the design allow-list pass."""
        cfg = _make_config(tmp_path, yolo_device=valid)
        assert cfg.yolo_device == valid

    @pytest.mark.parametrize(
        "invalid",
        [
            "garbage",
            "GPU",  # wrong case
            "cuda:0",  # ultralytics accepts this but we don't (single GPU only)
            "-1",  # negative
            "8",  # out of 0..7 range
            "true",  # bool-as-string confusion
            "0,1",  # multi-GPU not supported by the pipeline
            "device0",  # prefixed garbage
        ],
    )
    def test_invalid_values_rejected(self, tmp_path, invalid):
        """Out-of-allow-list values raise ValueError at construction time."""
        with pytest.raises(ValueError, match="yolo_device must be one of"):
            _make_config(tmp_path, yolo_device=invalid)

    def test_invalid_error_lists_valid_values(self, tmp_path):
        """The error message names the valid set, so the user can fix it
        without grepping the source."""
        with pytest.raises(ValueError) as excinfo:
            _make_config(tmp_path, yolo_device="garbage")
        msg = str(excinfo.value)
        # Allow-list keywords present in the error message.
        for keyword in ["auto", "cpu", "cuda", "mps"]:
            assert keyword in msg, f"error msg missing {keyword!r}: {msg}"
        # User's bad value is echoed back so they know what was rejected.
        assert "garbage" in msg

    def test_invalid_fails_at_construction_not_at_inference(self, tmp_path):
        """A bad yolo_device must raise ValueError at PipelineConfig()
        construction time, NOT silently propagate to the first YOLO(...).

        This is the exact regression the audit caught: the user passed
        ``yolo_device="garbage"`` and only discovered the typo when
        YOLO() crashed minutes later with a cryptic AssertionError.
        """
        try:
            _make_config(tmp_path, yolo_device="garbage")
        except ValueError as exc:
            # Must mention "yolo_device" so the user can identify which
            # config key is wrong.
            assert "yolo_device" in str(exc)
        else:
            pytest.fail(
                "PipelineConfig(yolo_device='garbage') must raise ValueError; "
                "silent acceptance is the Phase 68 regression"
            )


# --- Source guard ---------------------------------------------------------
#
# The Phase 68 fix relies on `_VALID_YOLO_DEVICES` being a frozenset of
# the canonical allow-list AND `__post_init__` checking it. A future
# refactor that drops either piece would silently re-introduce the
# cryptic AssertionError at first inference.


class TestPhase68SourceGuard:
    def test_valid_yolo_devices_constant_exists(self):
        """``_VALID_YOLO_DEVICES`` must be defined in config.py."""
        from rlpe import config as cfg_mod

        assert hasattr(cfg_mod, "_VALID_YOLO_DEVICES"), (
            "_VALID_YOLO_DEVICES removed from config.py — Phase 68 validation "
            "will fall back to accepting any string"
        )
        # Must be a frozenset (immutable) of strings.
        valid = cfg_mod._VALID_YOLO_DEVICES
        assert isinstance(valid, frozenset)
        assert all(isinstance(v, str) for v in valid)

    def test_valid_yolo_devices_includes_core_set(self):
        """The allow-list must include the values that actually work."""
        from rlpe import config as cfg_mod

        valid = cfg_mod._VALID_YOLO_DEVICES
        # Documented values.
        for must_have in ("auto", "cpu", "cuda", "mps"):
            assert must_have in valid, f"missing {must_have!r} from _VALID_YOLO_DEVICES"
        # All 8 GPU indices.
        for i in range(8):
            assert str(i) in valid, f"missing GPU index {i!r} from _VALID_YOLO_DEVICES"

    def test_post_init_validates_yolo_device(self):
        """``__post_init__`` must check ``yolo_device`` against the allow-list."""
        from rlpe import config as cfg_mod

        src = Path(cfg_mod.__file__).read_text(encoding="utf-8")
        # Both pieces of the design must be present.
        assert "_VALID_YOLO_DEVICES" in src, "_VALID_YOLO_DEVICES reference missing"
        assert (
            "self.yolo_device not in _VALID_YOLO_DEVICES" in src
            or 'self.yolo_device not in _VALID_YOLO_DEVICES' in src
        ), (
            "__post_init__ no longer validates yolo_device against _VALID_YOLO_DEVICES; "
            "typos in yolo_device will silently reach YOLO(...) and crash with "
            "a cryptic ultralytics AssertionError"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
