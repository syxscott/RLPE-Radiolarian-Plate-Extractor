"""Regression: audit 2026-09-04 llm-2 — the
``api_redacted`` data-outbound policy is supposed to strip the
paper's figure caption from the request before sending it to the
Anthropic API. The current code accepts ``caption_text`` and
``ocr_labels`` into ``_apply_outbound_policy`` but ignores them in
the function body — only ``panel_image`` and ``user_prompt`` get
redacted. The original caption is then re-attached later via a
thread-local side channel:

    src/rlpe/llm_backends.py:2370
        self._thread_local.caption_text = caption_text or ""
    src/rlpe/llm_backends.py:1777-1783
        # in _build_user_content:
        caption_text = getattr(self._thread_local, "caption_text", None) or ""
        if caption_text and caption_text not in user_prompt:
            extra_parts.append(f"[Figure caption: {caption_text}]")

So the ``api_redacted`` redaction is silently undone right after it
runs — the full caption text (containing paper-private figure
caption text the user did NOT opt to share) is sent to the API
anyway.

Real failure mode: a user who explicitly sets
``data_outbound_policy = "api_redacted"`` (the safe default
introduced in Phase A of audit 2026-09-03 BLOCKER-#2) still has
the paper's caption text leave their machine. They opted out;
they got opted-in.

Fix contract:
  * ``_apply_outbound_policy`` must actually redact ``caption_text``
    and ``ocr_labels`` when the policy is ``api_redacted`` (return
    them as part of the redacted tuple).
  * The thread-local must store the REDACTED caption / ocr_labels,
    not the originals — so the re-attachment in
    ``_build_user_content`` is harmless under ``api_redacted``.
  * Under ``api_full``, caption and ocr_labels must pass through
    unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestApplyOutboundPolicyRedactsCaption:
    def test_api_redacted_strips_caption_text(self):
        """Under api_redacted, the returned caption_text must be
        redacted (empty or a stub) — NOT the original."""
        from rlpe.llm_backends import MiniMaxM3Backend

        # Bypass full backend init — only the redaction policy matters.
        bk = MiniMaxM3Backend.__new__(MiniMaxM3Backend)
        bk.data_outbound_policy = "api_redacted"

        sample_caption = (
            "Plate 5. Spumellarians from the Lower Cretaceous of the "
            "Western Tethys. Confidential species list: Genus alpha, "
            "Genus beta. Specimen from private collection."
        )
        result = bk._apply_outbound_policy(
            panel_image=None,
            caption_text=sample_caption,
            ocr_labels=["1", "2", "3"],
            user_prompt="prompt",
        )
        # The new contract: result is (image, prompt, caption, ocr_labels).
        assert len(result) == 4, (
            f"audit 2026-09-04 llm-2: _apply_outbound_policy must "
            f"return 4-tuple (image, prompt, caption, ocr_labels), "
            f"got len={len(result)}"
        )
        redacted_caption = result[2]
        assert redacted_caption != sample_caption, (
            f"audit 2026-09-04 llm-2: api_redacted must redact "
            f"caption_text but returned it unchanged. "
            f"This is the bug: the policy signature accepts caption_text "
            f"but the body ignores it, so the full paper caption "
            f"leaves the machine despite the user opting out."
        )
        assert redacted_caption == "" or "<redacted>" in redacted_caption.lower(), (
            f"audit 2026-09-04 llm-2: redacted caption must be empty "
            f"or a stub, got {redacted_caption!r}"
        )

    def test_api_redacted_strips_ocr_labels(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        bk = MiniMaxM3Backend.__new__(MiniMaxM3Backend)
        bk.data_outbound_policy = "api_redacted"
        result = bk._apply_outbound_policy(
            panel_image=None,
            caption_text="anything",
            ocr_labels=["1", "2", "3"],
            user_prompt="prompt",
        )
        redacted_ocr = result[3]
        # ocr_labels under api_redacted: either empty or heavily
        # reduced (the printed panel IDs alone aren't sensitive, but
        # they could be — the safe default is to drop them too).
        assert redacted_ocr == [] or redacted_ocr is None, (
            f"audit 2026-09-04 llm-2: api_redacted must strip "
            f"ocr_labels, got {redacted_ocr}"
        )

    def test_api_full_passes_caption_through(self):
        """Under api_full (opt-in), caption and ocr_labels pass
        through unchanged — that's the whole point of opting in."""
        from rlpe.llm_backends import MiniMaxM3Backend

        bk = MiniMaxM3Backend.__new__(MiniMaxM3Backend)
        bk.data_outbound_policy = "api_full"
        sample_caption = "Plate 5. Spumellarians from the Lower Cretaceous"
        result = bk._apply_outbound_policy(
            panel_image=None,
            caption_text=sample_caption,
            ocr_labels=["1", "2"],
            user_prompt="prompt",
        )
        assert len(result) == 4
        assert result[2] == sample_caption
        assert result[3] == ["1", "2"]


class TestInferPanelDoesNotLeakCaptionUnderRedacted:
    """End-to-end: when policy is api_redacted, the messages list
    sent to the API must NOT contain the original caption."""

    def _make_backend(self, policy: str):
        from rlpe.llm_backends import MiniMaxM3Backend

        bk = MiniMaxM3Backend.__new__(MiniMaxM3Backend)
        bk.data_outbound_policy = policy
        bk._thread_local = type("TL", (), {})()  # fresh thread-local
        return bk

    def _stub_capture(self, bk):
        """Stub _call_api to capture the messages list. Don't stub
        _build_messages — we want the real _build_user_content to
        run, because that's where the caption re-attachment happens
        and that's the bug we're testing."""
        captured: dict = {}

        def fake_call_api(system_prompt, messages):
            captured["messages"] = messages
            return {"content": [{"type": "text", "text": "{}"}]}

        bk._call_api = fake_call_api
        return captured

    def test_redacted_caption_not_in_user_prompt(self, monkeypatch):
        bk = self._make_backend("api_redacted")
        captured = self._stub_capture(bk)
        sample_caption = (
            "SECRET-CAPTION-DO-NOT-LEAK Spumellarians from the "
            "Lower Cretaceous with private species list Genus alpha."
        )
        bk.infer_panel(
            panel_image=None,
            caption_text=sample_caption,
            ocr_labels=["1", "2"],
            system_prompt="system",
            user_prompt="basic prompt",
        )
        messages_text = str(captured["messages"])
        assert "SECRET-CAPTION-DO-NOT-LEAK" not in messages_text, (
            f"audit 2026-09-04 llm-2: api_redacted still leaks the "
            f"original caption text via the thread-local re-attachment "
            f"in _build_user_content. messages={messages_text[:500]}"
        )
        assert "Genus alpha" not in messages_text, (
            f"audit 2026-09-04 llm-2: api_redacted leaks caption body. "
            f"messages={messages_text[:500]}"
        )

    def test_full_policy_keeps_caption(self, monkeypatch):
        """Under api_full, the caption SHOULD appear — that's the
        opt-in."""
        bk = self._make_backend("api_full")
        captured = self._stub_capture(bk)
        sample_caption = "EXPLICIT-OPT-IN Spumellarians from Lower Cretaceous"
        bk.infer_panel(
            panel_image=None,
            caption_text=sample_caption,
            ocr_labels=["1"],
            system_prompt="system",
            user_prompt="basic prompt",
        )
        messages_text = str(captured["messages"])
        assert "EXPLICIT-OPT-IN" in messages_text, (
            f"audit 2026-09-04 llm-2: api_full should pass caption "
            f"through, but it's missing from messages. "
            f"messages={messages_text[:500]}"
        )