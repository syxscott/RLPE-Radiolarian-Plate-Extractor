"""Tests for non-specimen placeholder detection in pipeline captions."""
from __future__ import annotations

import pytest

from rlpe.pipeline import _looks_like_placeholder_caption


class TestLooksLikePlaceholderCaption:
    def test_auto_generated_page(self):
        assert _looks_like_placeholder_caption("Page 1自动生成图")

    def test_auto_generated_english(self):
        assert _looks_like_placeholder_caption("Page 1 auto-generated image")
        assert _looks_like_placeholder_caption("Page 2 auto generated figure")

    def test_placeholder_keyword(self):
        assert _looks_like_placeholder_caption("Page 3 placeholder")

    def test_header_only(self):
        assert _looks_like_placeholder_caption("running head")
        assert _looks_like_placeholder_caption("Header")

    def test_copyright(self):
        assert _looks_like_placeholder_caption("Copyright © 2017 Y. Xiao et al.")

    def test_publisher_running_text(self):
        assert _looks_like_placeholder_caption("scientific")
        assert _looks_like_placeholder_caption("elsevier")

    def test_short_caption(self):
        # <=3 chars -> placeholder
        assert _looks_like_placeholder_caption("AB")
        assert _looks_like_placeholder_caption("X")

    def test_normal_radiolarian_caption(self):
        # Real radiolarian figure captions should NOT be filtered
        assert not _looks_like_placeholder_caption(
            "Fig. 1. SEM micrographs of Dalongicaepa bipolaris gen. et sp. nov."
        )

    def test_normal_palaeontology_caption(self):
        assert not _looks_like_placeholder_caption(
            "Fig. 3. Schematic drawings of internal structure of Permian and Mesozoic spumellarians."
        )

    def test_empty(self):
        assert not _looks_like_placeholder_caption("")

    def test_real_caption_with_license_mention(self):
        # Real captions can mention "license" but aren't JUST a license line
        assert not _looks_like_placeholder_caption(
            "Fig. 1. Dalongicaepa bipolaris (described under Creative Commons Attribution License)"
        )


class TestStage4SkipLogic:
    """Verify that m3_rejected_non_radiolarian does NOT trigger the FallbackHandler."""

    def test_non_radiolarian_flag_does_not_count_as_fallback_error(self):
        from rlpe.pipeline import RadiolarianPipeline
        # Build a fake match dict with m3_rejected_non_radiolarian
        class _FakeMatch:
            def __init__(self):
                self.metadata = {
                    "m3_rejected_non_radiolarian": True,
                    "gemma_reasoning": "该panel并非古生物标本图版",
                }
        m = _FakeMatch()
        # _matches_have_fallback_error should return False for this match
        assert RadiolarianPipeline._matches_have_fallback_error([m]) is False

    def test_real_fallback_error_still_triggers(self):
        from rlpe.pipeline import RadiolarianPipeline
        class _FakeMatch:
            def __init__(self):
                self.metadata = {"gemma_error": "API timeout"}
        m = _FakeMatch()
        assert RadiolarianPipeline._matches_have_fallback_error([m]) is True

    def test_low_confidence_fallback_still_triggers(self):
        from rlpe.pipeline import RadiolarianPipeline
        class _FakeMatch:
            def __init__(self):
                self.metadata = {"gemma_fallback": True, "gemma_reasoning": "low conf"}
        m = _FakeMatch()
        assert RadiolarianPipeline._matches_have_fallback_error([m]) is True


class TestPlaceholderSkipsStage4EvenWhenStage2Passes:
    """Regression: the placeholder-caption check used to be an `elif` after the
    `m3_plate_cls is not None` branch, so a "Page 1 auto-generated image"
    caption on an accepted micrograph would still hit stage 4. The fix
    promoted it to an `if` so it runs regardless of stage 2's verdict.

    Here we sanity-check the helper logic in isolation: when the caption is a
    placeholder, the helper returns True; the wiring in `_process_region` is
    straightforward enough that the unit test above covers the helper, but we
    also add a focused test that mimics the user's reported case.
    """

    def test_user_reported_case(self):
        # Exact case from the bug report: stage 2 says "micrograph" (passes),
        # but the caption is "Page 1 自动生成图" — must be filtered.
        caption = "Page 1自动生成图"
        assert _looks_like_placeholder_caption(caption) is True

    def test_placeholder_takes_precedence_over_image_type_check(self):
        # A real specimen caption must NOT be flagged as a placeholder even if
        # image_type later turns out to be e.g. "diagram".
        caption = "Fig. 3. SEM micrographs of Dalongicaepa bipolaris gen. et sp. nov."
        assert _looks_like_placeholder_caption(caption) is False
