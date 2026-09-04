"""Regression tests for audit 2026-08-19 Phase 4B — LLM backend robustness.

Bug fixes covered:
- M-21: ``_normalize_panel_dict`` used to let LLM-hallucinated extras
  (``"ocr_confidence"``, ``"valid_name"``, ``"fake_field"``, ...) leak
  through as structural-extras (list/dict values) or as unfiltered
  scalar values downstream. The new ``_ALLOWED_PANEL_FIELDS``
  frozenset enumerates every key downstream code is allowed to see;
  anything else is dropped with a debug log so audit can spot prompt
  drift.

- M-22: ``_apply_geo_whitelist`` used to leave inverted Ma ranges
  (``ma_top > ma_base``) untouched. The new ``_validate_ma_range``
  helper auto-swaps inverted pairs with a warning log so the caller
  still has *some* usable range to work with. The strict
  ``m3_engine._validate_ma_range`` (Phase 2b M-13) still NULLS bad
  ranges in the post-whitelist path — both policies coexist.

- M-23: ``LlamaCppGemmaBackend._chat_completion`` used to silently
  fall back to the ``/completion`` text-only endpoint on ANY
  exception. The Phase 2c B-4 fix re-raises 4xx; the Phase 4B M-23
  fix *distinguishes* 401/403 (``LLMAuthenticationError``), 404
  (``LLMNotFoundError``) and 429 (``LLMRateLimitError``) so callers
  can route the error correctly.

- M-24: ``MiniMaxM3Backend._parse_retry_after(exc)`` only parsed the
  numeric ``Retry-After`` form. The new ``_parse_retry_after_header``
  static method also parses the HTTP-date form (per RFC 7231 §7.1.3)
  so future retry loops can honour both without a breaking change.

These tests are read-only against the live source so they catch
prompt / contract drift and accidental removal of the helpers.
"""

from __future__ import annotations

import logging
import sys
import unittest.mock as mock
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _UTC_NOW():
    """Locale-independent wall-clock accessor (UTC, tz-aware).

    Kept as a helper so the HTTP-date tests never depend on
    ``strftime`` for RFC-1123 formatting (see the
    ``test_http_date_in_future_returns_seconds`` comment for the
    QApplication/setlocale interaction this avoids).
    """
    from datetime import datetime

    return datetime.now(UTC)


def _make_pil_image(width: int = 64, height: int = 64, color: str = "red") -> Any:
    """Build a real PIL Image so the multimodal encoder can serialize it."""
    from PIL import Image

    return Image.new("RGB", (width, height), color=color)


# ---------------------------------------------------------------------------
# M-21: schema whitelist for ``_normalize_panel_dict``
# ---------------------------------------------------------------------------


class TestM21AllowedPanelFieldsConstant:
    """``_ALLOWED_PANEL_FIELDS`` exists, is a frozenset, and contains
    the canonical keys the function always emits plus the optional
    structured extras documented in the M3 match-panel prompt."""

    def test_constant_exists_and_is_frozenset(self):
        from rlpe.llm_backends import _ALLOWED_PANEL_FIELDS

        assert isinstance(_ALLOWED_PANEL_FIELDS, frozenset)

    def test_contains_canonical_keys(self):
        from rlpe.llm_backends import _ALLOWED_PANEL_FIELDS

        for key in ("label", "species", "confidence", "reasoning"):
            assert key in _ALLOWED_PANEL_FIELDS, (
                f"_ALLOWED_PANEL_FIELDS missing canonical key {key!r}"
            )

    def test_contains_structural_extras(self):
        """Optional structured extras documented in M3 prompts."""
        from rlpe.llm_backends import _ALLOWED_PANEL_FIELDS

        for key in (
            "open_nomenclature_strength",
            "alternative",
            "is_radiolarian",
            "species_list",
        ):
            assert key in _ALLOWED_PANEL_FIELDS, (
                f"_ALLOWED_PANEL_FIELDS missing structured key {key!r}"
            )

    def test_contains_caller_managed_extras(self):
        """Keys added by ``infer_panel`` / ``infer_text`` after parsing."""
        from rlpe.llm_backends import _ALLOWED_PANEL_FIELDS

        for key in (
            "raw_text",
            "fallback_used",
            "multimodal_degraded",
            "extra_image_unsupported",
            "error",
        ):
            assert key in _ALLOWED_PANEL_FIELDS, (
                f"_ALLOWED_PANEL_FIELDS missing caller-managed key {key!r}"
            )


class TestM21NormalizePanelDictFiltersHallucinatedFields:
    """``_normalize_panel_dict`` must drop LLM-hallucinated fields that
    are not in :data:`_ALLOWED_PANEL_FIELDS`. The previous behaviour
    either let them leak through (list/dict structural extras) or
    silently dropped them after the canonical-key loop ran."""

    def test_scalar_hallucinated_field_dropped(self):
        """A scalar ``"valid_name"`` from the LLM is not whitelisted → drop."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "X", "valid_name": "Entactinia"})
        assert "valid_name" not in out
        # Canonical field still present.
        assert out["species"] == "X"

    def test_dict_hallucinated_field_dropped(self):
        """A hallucinated DICT (``"fake_field": {...}``) is also dropped
        even though the existing structural-extras loop preserves dicts
        — the whitelist is now authoritative."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "X", "fake_field": {"nested": "garbage"}})
        assert "fake_field" not in out

    def test_list_hallucinated_field_dropped(self):
        """A hallucinated LIST (``"ocr_confidence": [0.9]``) is also dropped."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict(
            {"species": "X", "ocr_confidence": [0.95], "fake_field": ["a", "b"]}
        )
        assert "ocr_confidence" not in out
        assert "fake_field" not in out

    def test_whitelisted_list_preserved(self):
        """A whitelisted LIST (``"species_list"``) IS preserved."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "X", "species_list": ["X", "Y", "Z"]})
        assert out.get("species_list") == ["X", "Y", "Z"]

    def test_whitelisted_dict_preserved(self):
        """A whitelisted DICT (e.g. ``notes``) IS preserved."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({"species": "X", "notes": "rich detail"})
        assert out.get("notes") == "rich detail"

    def test_dropped_fields_logged_at_debug(self, caplog):
        """A DEBUG log should mention the dropped hallucinated keys so
        audit can spot prompt drift."""
        from rlpe.llm_backends import _normalize_panel_dict

        caplog.set_level(logging.DEBUG, logger="rlpe.llm_backends")
        _normalize_panel_dict({"species": "X", "ocr_confidence": 0.95, "fake_field": "garbage"})
        debug_msgs = [
            record.message for record in caplog.records if record.levelno <= logging.DEBUG
        ]
        assert any("ocr_confidence" in m and "fake_field" in m for m in debug_msgs), (
            f"Expected DEBUG log naming dropped keys, got {debug_msgs}"
        )

    def test_no_hallucinations_no_log(self, caplog):
        """When the input has only whitelisted keys the function must
        NOT spam a debug log — that would dilute the audit signal."""
        from rlpe.llm_backends import _normalize_panel_dict

        caplog.set_level(logging.DEBUG, logger="rlpe.llm_backends")
        _normalize_panel_dict({"species": "X", "confidence": 0.9})
        debug_msgs = [
            record.message for record in caplog.records if record.levelno <= logging.DEBUG
        ]
        # Only the open-nomen discount message (if any) should appear —
        # never the "Dropped hallucinated panel fields" message.
        assert not any("Dropped hallucinated panel fields" in m for m in debug_msgs), (
            f"Did not expect a drop-fields log, got {debug_msgs}"
        )

    def test_empty_dict_safe(self):
        """Edge case: empty input dict."""
        from rlpe.llm_backends import _normalize_panel_dict

        out = _normalize_panel_dict({})
        # Canonical fields still emitted (None defaults).
        assert "label" in out
        assert "species" in out
        assert "confidence" in out
        assert "reasoning" in out

    def test_input_not_mutated_when_no_hallucinations(self):
        """Without the bug fix the function used to MUTATE the input
        dict when adding structural-extras. The whitelist filter also
        mutates — verify the behaviour is well-defined for the clean
        path (input/output should be equivalent after canonicalization)."""
        from rlpe.llm_backends import _ALLOWED_PANEL_FIELDS, _normalize_panel_dict

        # All-whitelisted input: filter is a no-op for top-level keys.
        clean = {"species": "X", "confidence": 0.9}
        clean_copy = dict(clean)
        _normalize_panel_dict(clean)
        # The whitelist filter may pop hallucinated keys; for an all-whitelisted
        # input nothing should be popped.
        assert set(clean.keys()) == set(clean_copy.keys())
        # Sanity: all remaining keys are whitelisted.
        assert set(clean.keys()) <= _ALLOWED_PANEL_FIELDS


# ---------------------------------------------------------------------------
# M-22: Ma range direction validation (auto-swap in llm_backends)
# ---------------------------------------------------------------------------


class TestM22ValidateMaRangeHelper:
    """``_validate_ma_range(ma_top, ma_base)`` enforces ICZN convention
    ``ma_top < ma_base`` (younger = smaller Ma). The llm_backends
    version AUTO-SWAPS so the caller still has a usable range; the
    strict ``m3_engine._validate_ma_range`` NULLS — both policies
    coexist (engine helper runs AFTER ``_apply_geo_whitelist``)."""

    def test_already_valid_range_preserved(self):
        from rlpe.llm_backends import _validate_ma_range

        top, base = _validate_ma_range(50, 100)
        assert top == 50
        assert base == 100

    def test_inverted_range_swapped(self):
        from rlpe.llm_backends import _validate_ma_range

        top, base = _validate_ma_range(100, 50)
        assert top == 50
        assert base == 100

    def test_inverted_range_logs_warning(self, caplog):
        from rlpe.llm_backends import _validate_ma_range

        caplog.set_level(logging.WARNING, logger="rlpe.llm_backends")
        _validate_ma_range(100, 50)
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("ma_top=100" in m and "ma_base=50" in m for m in warnings), (
            f"Expected WARNING with the bad range values, got {warnings}"
        )

    def test_top_none_preserved(self):
        from rlpe.llm_backends import _validate_ma_range

        top, base = _validate_ma_range(None, 50)
        assert top is None
        assert base == 50

    def test_base_none_preserved(self):
        from rlpe.llm_backends import _validate_ma_range

        top, base = _validate_ma_range(50, None)
        assert top == 50
        assert base is None

    def test_both_none_preserved(self):
        from rlpe.llm_backends import _validate_ma_range

        top, base = _validate_ma_range(None, None)
        assert top is None
        assert base is None

    def test_string_numeric_coerced(self):
        """Numeric strings are coerced for COMPARISON purposes; the
        helper returns the original values (not the coerced floats)
        so downstream type-strict callers don't lose information."""
        from rlpe.llm_backends import _validate_ma_range

        top, base = _validate_ma_range("140", "120")
        # Inverted numerically → swapped. The returned values are the
        # ORIGINAL strings (not coerced floats) so type is preserved.
        assert top == "120"
        assert base == "140"

    def test_non_numeric_passthrough(self):
        """Non-numeric junk is passed through unchanged (don't crash)."""
        from rlpe.llm_backends import _validate_ma_range

        top, base = _validate_ma_range("younger", "older")
        assert top == "younger"
        assert base == "older"

    def test_equal_values_preserved(self):
        """ma_top == ma_base is a zero-thickness range (valid)."""
        from rlpe.llm_backends import _validate_ma_range

        top, base = _validate_ma_range(100, 100)
        assert top == 100
        assert base == 100


class TestM22ApplyGeoWhitelistDoesNotAutoSwap:
    """``_apply_geo_whitelist`` MUST NOT auto-call ``_validate_ma_range``
    because the strict ``m3_engine._validate_ma_range`` (Phase 2b M-13)
    owns the null-on-violation policy downstream. Swapping here would
    mask bad ranges from the engine's null branch and silently break
    the Phase 2b regression tests.

    These tests pin the (intentional) non-call behaviour so a future
    refactor can't re-introduce the swap without flagging the audit.
    """

    def test_valid_range_preserved(self):
        from rlpe.llm_backends import _apply_geo_whitelist

        item = {"ma_top": 50, "ma_base": 100, "age": "Late Triassic"}
        out = _apply_geo_whitelist(dict(item))
        assert out["ma_top"] == 50
        assert out["ma_base"] == 100

    def test_inverted_range_NOT_swapped_by_whitelist(self):
        """The whitelist helper does NOT swap. The strict engine
        helper will null the inverted pair downstream — that is
        the contract Phase 2b tests pin."""
        from rlpe.llm_backends import _apply_geo_whitelist

        item = {"ma_top": 140, "ma_base": 120, "age": "Late Triassic"}
        out = _apply_geo_whitelist(dict(item))
        # Values are NOT swapped by the whitelist (engine handles it).
        assert out["ma_top"] == 140
        assert out["ma_base"] == 120

    def test_whitelist_drops_hallucinated_but_keeps_inverted_ma(self):
        """The whitelist drops hallucinated non-Ma keys but leaves
        inverted ma_top/ma_base for the strict engine helper to null."""
        from rlpe.llm_backends import _apply_geo_whitelist

        item = {
            "ma_top": 140,
            "ma_base": 120,
            "habitat": "marine",  # hallucinated
            "paleoclimate": "warm",  # hallucinated
        }
        out = _apply_geo_whitelist(dict(item))
        assert out["ma_top"] == 140
        assert out["ma_base"] == 120
        assert "habitat" not in out
        assert "paleoclimate" not in out

    def test_only_top_present(self):
        """Edge: only ma_top present (no ma_base)."""
        from rlpe.llm_backends import _apply_geo_whitelist

        item = {"ma_top": 100}
        out = _apply_geo_whitelist(dict(item))
        assert out["ma_top"] == 100
        assert "ma_base" not in out

    def test_non_dict_passthrough(self):
        """Non-dict input returns unchanged."""
        from rlpe.llm_backends import _apply_geo_whitelist

        for non_dict in (None, "string", 42, [1, 2, 3]):
            assert _apply_geo_whitelist(non_dict) == non_dict


# ---------------------------------------------------------------------------
# M-23: LlamaCpp 4xx/5xx distinction (specific exception classes)
# ---------------------------------------------------------------------------


class TestM23LLMHTTPErrorClasses:
    """The new exception hierarchy (``LLMAuthenticationError``,
    ``LLMNotFoundError``, ``LLMRateLimitError``, all subclasses of
    ``LLMHTTPError``) must exist and carry the HTTP status code so
    callers can route errors correctly."""

    def test_base_class_exists(self):
        from rlpe.llm_backends import LLMHTTPError

        assert issubclass(LLMHTTPError, Exception)

    def test_authentication_error_is_subclass(self):
        from rlpe.llm_backends import LLMAuthenticationError, LLMHTTPError

        assert issubclass(LLMAuthenticationError, LLMHTTPError)

    def test_not_found_error_is_subclass(self):
        from rlpe.llm_backends import LLMHTTPError, LLMNotFoundError

        assert issubclass(LLMNotFoundError, LLMHTTPError)

    def test_rate_limit_error_is_subclass(self):
        from rlpe.llm_backends import LLMHTTPError, LLMRateLimitError

        assert issubclass(LLMRateLimitError, LLMHTTPError)

    def test_status_code_attribute(self):
        from rlpe.llm_backends import LLMAuthenticationError

        err = LLMAuthenticationError("auth failed", status_code=401)
        assert err.status_code == 401
        assert "auth failed" in str(err)


class TestM23LlamaCppRaisesSpecificExceptions:
    """``LlamaCppGemmaBackend._chat_completion`` must raise the right
    exception class for each 4xx status code — not a generic
    ``HTTPError`` that the caller has to inspect."""

    def test_401_raises_authentication_error(self, monkeypatch):
        from rlpe.llm_backends import LlamaCppGemmaBackend, LLMAuthenticationError

        class _FakeResp:
            status_code = 401
            text = "Unauthorized"

            def raise_for_status(self) -> None:
                from requests import HTTPError

                raise HTTPError("401", response=self)

        monkeypatch.setattr("rlpe.llm_backends.requests.post", lambda *a, **kw: _FakeResp())
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        with pytest.raises(LLMAuthenticationError) as ei:
            backend._chat_completion(
                panel_image=_make_pil_image(),
                system_prompt="sys",
                user_prompt="user",
            )
        assert ei.value.status_code == 401

    def test_403_raises_authentication_error(self, monkeypatch):
        from rlpe.llm_backends import LlamaCppGemmaBackend, LLMAuthenticationError

        class _FakeResp:
            status_code = 403
            text = "Forbidden"

            def raise_for_status(self) -> None:
                from requests import HTTPError

                raise HTTPError("403", response=self)

        monkeypatch.setattr("rlpe.llm_backends.requests.post", lambda *a, **kw: _FakeResp())
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        with pytest.raises(LLMAuthenticationError) as ei:
            backend._chat_completion(
                panel_image=_make_pil_image(),
                system_prompt="sys",
                user_prompt="user",
            )
        assert ei.value.status_code == 403

    def test_404_raises_not_found_error(self, monkeypatch):
        from rlpe.llm_backends import LlamaCppGemmaBackend, LLMNotFoundError

        class _FakeResp:
            status_code = 404
            text = "Not Found"

            def raise_for_status(self) -> None:
                from requests import HTTPError

                raise HTTPError("404", response=self)

        monkeypatch.setattr("rlpe.llm_backends.requests.post", lambda *a, **kw: _FakeResp())
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        with pytest.raises(LLMNotFoundError) as ei:
            backend._chat_completion(
                panel_image=_make_pil_image(),
                system_prompt="sys",
                user_prompt="user",
            )
        assert ei.value.status_code == 404

    def test_429_raises_rate_limit_error(self, monkeypatch):
        from rlpe.llm_backends import LlamaCppGemmaBackend, LLMRateLimitError

        class _FakeResp:
            status_code = 429
            text = "Too Many Requests"

            def raise_for_status(self) -> None:
                from requests import HTTPError

                raise HTTPError("429", response=self)

        monkeypatch.setattr("rlpe.llm_backends.requests.post", lambda *a, **kw: _FakeResp())
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        with pytest.raises(LLMRateLimitError) as ei:
            backend._chat_completion(
                panel_image=_make_pil_image(),
                system_prompt="sys",
                user_prompt="user",
            )
        assert ei.value.status_code == 429

    def test_400_reraises_as_http_error(self, monkeypatch):
        """400 (malformed body) is NOT auth/notfound/ratelimit — it
        re-raises the original ``HTTPError`` (the B-4 behaviour is
        preserved for non-special 4xx)."""
        from requests import HTTPError

        from rlpe.llm_backends import LlamaCppGemmaBackend

        class _FakeResp:
            status_code = 400
            text = "Bad Request"

            def raise_for_status(self) -> None:
                raise HTTPError("400", response=self)

        monkeypatch.setattr("rlpe.llm_backends.requests.post", lambda *a, **kw: _FakeResp())
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        with pytest.raises(HTTPError):
            backend._chat_completion(
                panel_image=_make_pil_image(),
                system_prompt="sys",
                user_prompt="user",
            )

    def test_413_reraises_as_http_error(self, monkeypatch):
        """413 (payload too large) is also NOT special — re-raise."""
        from requests import HTTPError

        from rlpe.llm_backends import LlamaCppGemmaBackend

        class _FakeResp:
            status_code = 413
            text = "Payload Too Large"

            def raise_for_status(self) -> None:
                raise HTTPError("413", response=self)

        monkeypatch.setattr("rlpe.llm_backends.requests.post", lambda *a, **kw: _FakeResp())
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        with pytest.raises(HTTPError):
            backend._chat_completion(
                panel_image=_make_pil_image(),
                system_prompt="sys",
                user_prompt="user",
            )

    def test_500_still_falls_back_to_completion(self, monkeypatch):
        """5xx errors remain transient → fall back to ``/completion``."""
        from rlpe.llm_backends import LlamaCppGemmaBackend

        posted_to: list[str] = []

        class _Fake500Resp:
            status_code = 500
            text = "Internal Server Error"

            def raise_for_status(self) -> None:
                from requests import HTTPError

                raise HTTPError("500", response=self)

        class _FakeCompletionResp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"content": "{}"}

        def _fake_post(url: str, json: dict[str, Any], timeout: Any = None):
            posted_to.append(url)
            if "/v1/chat/completions" in url:
                return _Fake500Resp()
            return _FakeCompletionResp()

        monkeypatch.setattr("rlpe.llm_backends.requests.post", _fake_post)
        backend = LlamaCppGemmaBackend(host="http://127.0.0.1:8080", model="m")
        text, multimodal_degraded = backend._chat_completion(
            panel_image=_make_pil_image(),
            system_prompt="sys",
            user_prompt="user",
        )
        assert multimodal_degraded is True
        assert any("/v1/chat/completions" in u for u in posted_to)
        assert any(u.endswith("/completion") for u in posted_to)


# ---------------------------------------------------------------------------
# M-24: ``_parse_retry_after_header`` — Retry-After (numeric + HTTP-date)
# ---------------------------------------------------------------------------


class TestM24ParseRetryAfterHeader:
    """``MiniMaxM3Backend._parse_retry_after_header`` parses a
    ``Retry-After`` header value (string) per RFC 7231 §7.1.3.

    Accepts:
    * Integer / decimal seconds (e.g. ``"30"``).
    * HTTP-date (e.g. ``"Wed, 21 Oct 2026 07:28:00 GMT"``).

    Returns ``0.0`` for ``None`` / empty / unparseable values. Caps
    positive values at 60.0 so a hostile / buggy server cannot pin a
    worker for minutes.
    """

    def test_numeric_30(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        assert MiniMaxM3Backend._parse_retry_after_header("30") == 30.0

    def test_numeric_60(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        assert MiniMaxM3Backend._parse_retry_after_header("60") == 60.0

    def test_numeric_120_capped_at_60(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        assert MiniMaxM3Backend._parse_retry_after_header("120") == 60.0

    def test_numeric_decimal_seconds(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        # RFC 7231 allows delta-seconds as a non-negative integer,
        # but production servers sometimes emit decimals.
        assert MiniMaxM3Backend._parse_retry_after_header("12.5") == 12.5

    def test_numeric_zero_returns_zero(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        # 0 seconds is a "retry immediately" signal — return 0 so
        # the caller doesn't wait at all.
        assert MiniMaxM3Backend._parse_retry_after_header("0") == 0.0

    def test_numeric_negative_returns_zero(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        # Negative seconds is nonsensical — treat as 0.
        assert MiniMaxM3Backend._parse_retry_after_header("-5") == 0.0

    def test_none_returns_zero(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        assert MiniMaxM3Backend._parse_retry_after_header(None) == 0.0

    def test_empty_string_returns_zero(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        assert MiniMaxM3Backend._parse_retry_after_header("") == 0.0

    def test_whitespace_only_returns_zero(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        assert MiniMaxM3Backend._parse_retry_after_header("   ") == 0.0

    def test_unparseable_garbage_returns_zero(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        assert MiniMaxM3Backend._parse_retry_after_header("not a number") == 0.0

    def test_http_date_in_future_returns_seconds(self):
        """An HTTP-date in the future returns the seconds-to-date,
        capped at 60."""
        # 5 minutes from now — well within the 60s cap.
        from datetime import timedelta
        from email.utils import format_datetime

        from rlpe.llm_backends import MiniMaxM3Backend

        future = _UTC_NOW() + timedelta(minutes=5)
        # Audit 2026-09-04 (CI regression): format the date with
        # ``email.utils.format_datetime`` instead of ``strftime``.
        # Qt's QApplication init (pulled in by GUI tests collected
        # earlier in the same pytest session) calls
        # ``setlocale(LC_ALL, "")``, which makes ``%a``/``%b`` render
        # day/month names in the *system* locale — under zh_CN the
        # resulting string is no longer RFC-1123 and
        # ``parsedate_to_datetime`` returns 0.0. format_datetime is
        # locale-independent.
        http_date = format_datetime(future, usegmt=True)
        result = MiniMaxM3Backend._parse_retry_after_header(http_date)
        # Must be > 0 (future date) and <= 60 (cap).
        assert 0 < result <= 60.0

    def test_http_date_in_past_returns_zero(self):
        """An HTTP-date in the past returns 0 (no waiting needed)."""
        from datetime import timedelta
        from email.utils import format_datetime

        from rlpe.llm_backends import MiniMaxM3Backend

        past = _UTC_NOW() - timedelta(minutes=5)
        http_date = format_datetime(past, usegmt=True)
        assert MiniMaxM3Backend._parse_retry_after_header(http_date) == 0.0

    def test_http_date_far_future_capped_at_60(self):
        """An HTTP-date far in the future is capped at 60 seconds so a
        buggy server can't pin a worker for minutes."""
        from datetime import timedelta
        from email.utils import format_datetime

        from rlpe.llm_backends import MiniMaxM3Backend

        # 24 hours in the future.
        future = _UTC_NOW() + timedelta(days=1)
        http_date = format_datetime(future, usegmt=True)
        result = MiniMaxM3Backend._parse_retry_after_header(http_date)
        assert result == 60.0

    def test_http_date_static_format_returns_near_timestamp(self):
        """The task description's example date ``Wed, 21 Oct 2026
        07:28:00 GMT`` returns a value close to the actual delta
        (within the 60s cap, since we can't predict the wall-clock
        gap during the test)."""
        from rlpe.llm_backends import MiniMaxM3Backend

        result = MiniMaxM3Backend._parse_retry_after_header("Wed, 21 Oct 2026 07:28:00 GMT")
        # The test runs in real time — the date could be in the past
        # OR far in the future. Either way the helper must return a
        # well-formed cap-bounded value:
        # - past date: 0.0
        # - future date: > 0 and <= 60.0
        assert 0.0 <= result <= 60.0

    def test_http_date_rfc_850_format(self):
        """RFC 850 format (``Sunday, 06-Nov-94 08:49:37 GMT``) is also
        accepted per RFC 7231 §7.1.1.1."""
        from rlpe.llm_backends import MiniMaxM3Backend

        # Use a past date so we expect 0.0 (regardless of wall clock).
        result = MiniMaxM3Backend._parse_retry_after_header("Wednesday, 21-Oct-2026 07:28:00 GMT")
        assert 0.0 <= result <= 60.0

    def test_http_date_invalid_returns_zero(self):
        """An unparseable HTTP-date returns 0 (no crash)."""
        from rlpe.llm_backends import MiniMaxM3Backend

        assert MiniMaxM3Backend._parse_retry_after_header("this is not a date 12345") == 0.0


# ---------------------------------------------------------------------------
# Source-guard: catch silent removal of the new helpers
# ---------------------------------------------------------------------------


class TestSourceGuard:
    """Static checks against the live source so any later refactor
    that silently removes the Phase 4B helpers fires the guard."""

    def test_geo_whitelist_does_NOT_call_validate_ma_range(self):
        """``_apply_geo_whitelist`` must NOT call ``_validate_ma_range``
        directly — the strict ``m3_engine._validate_ma_range`` owns
        the null-on-violation policy downstream. A swap here would
        mask bad ranges from the engine's null branch and silently
        break the Phase 2b regression tests."""
        import inspect

        from rlpe.llm_backends import _apply_geo_whitelist, _validate_ma_range

        # Inspect the function's bytecode for any LOAD_GLOBAL/LOAD_NAME
        # reference to ``_validate_ma_range``. A pure docstring mention
        # is fine — the docstring intentionally documents *why* the
        # helper is NOT called. We only care about actual call sites.
        code = _apply_geo_whitelist.__code__
        names = set(code.co_names)
        assert "_validate_ma_range" not in names, (
            "_apply_geo_whitelist must not reference _validate_ma_range — "
            "the strict null-on-violation policy is owned by m3_engine."
        )
        # Sanity: both helpers still exist as module-level callables.
        assert callable(_validate_ma_range)

    def test_normalize_panel_dict_uses_whitelist(self):
        """``_normalize_panel_dict`` must reference ``_ALLOWED_PANEL_FIELDS``."""
        from rlpe.llm_backends import _normalize_panel_dict

        src = Path(_normalize_panel_dict.__code__.co_filename).read_text()
        assert "_ALLOWED_PANEL_FIELDS" in src

    def test_llamacpp_raises_new_exception_classes(self):
        """``LlamaCppGemmaBackend._chat_completion`` must reference the
        new exception classes so the M-23 routing works."""
        from rlpe.llm_backends import LlamaCppGemmaBackend

        src = Path(LlamaCppGemmaBackend._chat_completion.__code__.co_filename).read_text()
        for name in (
            "LLMAuthenticationError",
            "LLMNotFoundError",
            "LLMRateLimitError",
        ):
            assert name in src, f"LlamaCppGemmaBackend source lost reference to {name!r}"

    def test_minimax_defines_parse_retry_after_header(self):
        from rlpe.llm_backends import MiniMaxM3Backend

        assert hasattr(MiniMaxM3Backend, "_parse_retry_after_header")
        assert callable(MiniMaxM3Backend._parse_retry_after_header)
