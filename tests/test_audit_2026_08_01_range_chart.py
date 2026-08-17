"""Regression tests for audit 2026-08-01 batch W2 — range_chart_extractor M3/M21."""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chart(species_ranges: list[tuple[str, float | None]]):
    """Build a ``RangeChartResult`` from a list of (species, per-row conf).

    Each tuple becomes one ``SpeciesRange`` in a single section so tests
    can drive ``build_geology_links_for_panels`` without a real API call.
    """
    from rlpe.range_chart_extractor import (
        RangeChartResult,
        SpeciesRange,
    )

    chart = RangeChartResult(
        figure_id="fig1",
        paper_id="bandini2006",
        caption="Range chart of Archaeodictyomitra species across sections",
    )
    for name, conf in species_ranges:
        sr = SpeciesRange(
            species=name,
            section="Section A",
            range_top="Bed 9",
            range_base="Bed 1",
            biozone="",
            confidence=conf if conf is not None else 0.9,
        )
        chart.species_ranges.append(sr)
    return chart


# ---------------------------------------------------------------------------
# Bug M3 — genus disambiguation
# ---------------------------------------------------------------------------


class TestSpeciesDisambiguation:
    """Bug M3: a bare-genus panel must NOT silently link to the first
    chart species of that genus when multiple species exist.

    Pre-fix: ``build_geology_links_for_panels`` took the first ``sp_key``
    in iteration order that started with the panel's normalized genus
    string. That linked Bandini 2006's bare ``Archaeodictyomitra`` to
    ``A. rigida`` even though the chart carried 5 species.
    Post-fix: we collect ALL genus matches, count distinct species, and
    skip the link entirely when count >= 2 (disambiguation requires
    species-level matching).
    """

    def test_genus_with_multiple_species_no_match(self):
        from rlpe.range_chart_extractor import build_geology_links_for_panels

        chart = _make_chart(
            [
                ("Archaeodictyomitra rigida", 0.95),
                ("Archaeodictyomitra vulgaris", 0.92),
                ("Archaeodictyomitra chalara", 0.88),
            ]
        )
        panel_records = [{"species": "Archaeodictyomitra", "panel_id": "pl01-001"}]
        links = build_geology_links_for_panels(chart, panel_records)
        assert links == [], (
            f"bare-genus panel must not link to any of 3 chart species, "
            f"got {len(links)} link(s): {[l.get('species') for l in links]}"
        )

    def test_species_match_with_unique_genus_species(self):
        from rlpe.range_chart_extractor import build_geology_links_for_panels

        # Only ONE chart species of "GenusX" — link is unambiguous.
        chart = _make_chart([("GenusX unica", 0.85)])
        panel_records = [{"species": "GenusX", "panel_id": "pl07-002"}]
        links = build_geology_links_for_panels(chart, panel_records)
        assert len(links) == 1
        link = links[0]
        assert link["species"] == "GenusX unica"
        # Confidence must come from the per-species row, not from a default.
        assert link["confidence"] == 0.85

    def test_exact_species_match_still_works(self):
        from rlpe.range_chart_extractor import build_geology_links_for_panels

        # Panel has a full binomial — must match its exact species even
        # when the chart carries another species under the same genus.
        chart = _make_chart(
            [
                ("A. rigida", 0.95),
                ("A. vulgaris", 0.92),
            ]
        )
        panel_records = [{"species": "A. rigida", "panel_id": "pl09-001"}]
        links = build_geology_links_for_panels(chart, panel_records)
        # We should match exactly the rigida record (not vulgaris).
        assert len(links) >= 1
        matched = [l for l in links if l.get("species") == "A. rigida"]
        assert matched, (
            f"exact match for A. rigida missing; got species: {[l.get('species') for l in links]}"
        )
        assert matched[0]["confidence"] == 0.95


# ---------------------------------------------------------------------------
# Bug M21 — status="error" + error_message on failure paths
# ---------------------------------------------------------------------------


class TestStatusField:
    """Bug M21: ``_safe_json_loads`` failures and image-open failures used
    to return ``status="ok"`` empty results — indistinguishable from
    "API said no data here". The fix sets ``status="error"`` and populates
    ``error_message`` with the exception class + message so callers can
    tell transport / parse failures apart from genuine empty extractions.
    """

    def test_range_chart_result_has_error_message_field(self):
        """The dataclass must carry ``error_message`` defaulting to None."""
        from rlpe.range_chart_extractor import RangeChartResult

        field_names = {f.name for f in fields(RangeChartResult)}
        assert "error_message" in field_names, (
            f"RangeChartResult missing 'error_message' field; have {sorted(field_names)}"
        )
        instance = RangeChartResult()
        assert instance.error_message is None
        assert instance.status == "ok"  # default still ok
        d = instance.to_dict()
        assert "error_message" in d
        assert d["error_message"] is None
        assert d["status"] == "ok"

    def test_safe_json_loads_returns_error_status(self, tmp_path):
        """When the M3 backend returns non-JSON prose, extract_range_chart
        must surface ``status='error'`` with a populated ``error_message``."""
        from rlpe.range_chart_extractor import extract_range_chart

        # Set up a real (tiny) image so the image-open step succeeds.
        img_path = tmp_path / "chart.png"
        # 1x1 transparent PNG; minimum valid image bytes for the open() call.
        img_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4"
            b"\x89\x00\x00\x00\rIDATx\x9cc\xfc\xcf\xc0P\x0f\x00\x05\x01\x01\x02"
            b"\xcf\xa0.\xcd\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        # Build a 200 response whose raw text is plain prose (no JSON).
        # ``requests.post`` is used as ``with requests.post(...) as _resp:``,
        # so __enter__ must return the same mock instance — MagicMock's
        # default __enter__ returns a *new* MagicMock that drops our
        # attributes.
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False
        fake_resp.json.return_value = {"content": [{"type": "text", "text": "Sorry, not JSON"}]}

        from rlpe import range_chart_extractor as rce

        with (
            patch.object(
                rce,
                "_safe_json_loads",
                side_effect=ValueError("no JSON object found in 'Sorry, not JSON'"),
            ),
            patch("rlpe.range_chart_extractor.requests.post", return_value=fake_resp),
        ):
            result = extract_range_chart(
                paper_id="p",
                figure_id="f",
                caption="range chart",
                image_path=str(img_path),
                api_key="k",
                base_url="https://example.invalid",
                model="m",
                timeout_sec=10,
            )

        assert result.status == "error", (
            f"expected status='error' on JSON-parse failure, got {result.status!r}"
        )
        assert result.error_message is not None, (
            "error_message must be populated on JSON-parse failure"
        )
        assert "ValueError" in result.error_message, (
            f"error_message should include exception class; got {result.error_message!r}"
        )
        assert "no JSON object found" in result.error_message

    def test_image_open_failure_returns_error_status(self, tmp_path):
        """Passing a non-existent image path must produce status='error' +
        populated error_message, NOT the previous silent empty result."""
        from rlpe.range_chart_extractor import extract_range_chart

        missing = tmp_path / "does_not_exist.png"
        # Don't create it; the open() call inside extract_range_chart
        # will raise FileNotFoundError (an OSError subclass).
        result = extract_range_chart(
            paper_id="p",
            figure_id="f",
            caption="range chart",
            image_path=str(missing),
            api_key="k",
            base_url="https://example.invalid",
            model="m",
            timeout_sec=10,
        )

        assert result.status == "error", (
            f"expected status='error' when image is missing, got {result.status!r}"
        )
        assert result.error_message is not None
        assert "OSError" in result.error_message, (
            f"error_message should include OSError class; got {result.error_message!r}"
        )
        assert result.confidence == 0.0
        assert result.species_ranges == []

    def test_happy_path_returns_ok_status(self, tmp_path):
        """Valid image + valid JSON response must produce status='ok'
        and a populated RangeChartResult. Regression guard against
        flipping the default the wrong way."""
        from rlpe.range_chart_extractor import extract_range_chart

        img_path = tmp_path / "chart.png"
        img_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4"
            b"\x89\x00\x00\x00\rIDATx\x9cc\xfc\xcf\xc0P\x0f\x00\x05\x01\x01\x02"
            b"\xcf\xa0.\xcd\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        happy_payload = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "sections": [
                                {
                                    "name": "Pingdingshan",
                                    "age_range": "Late Permian",
                                    "formations": ["Talung Fm"],
                                }
                            ],
                            "species_ranges": [
                                {
                                    "species": "Neoalbaillella optima",
                                    "section": "Pingdingshan",
                                    "range_top": "Bed 9",
                                    "range_base": "Bed 7",
                                    "biozone": "N. optima Zone",
                                    "confidence": 0.92,
                                }
                            ],
                            "biozones": [],
                            "other_fossils": [],
                            "confidence": 0.85,
                        }
                    ),
                }
            ]
        }
        # __enter__ must return the same mock so ``with requests.post(...) as resp:``
        # exposes our ``status_code`` / ``json`` attributes.
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False
        fake_resp.json.return_value = happy_payload

        with patch("rlpe.range_chart_extractor.requests.post", return_value=fake_resp):
            result = extract_range_chart(
                paper_id="beccaro2006",
                figure_id="fig_range",
                caption="Range chart of Permian radiolarians.",
                image_path=str(img_path),
                api_key="k",
                base_url="https://example.invalid",
                model="m",
                timeout_sec=10,
            )

        assert result.status == "ok", (
            f"happy path must be status='ok', got {result.status!r} "
            f"with error_message={result.error_message!r}"
        )
        assert result.error_message is None
        assert result.confidence == 0.85
        assert len(result.species_ranges) == 1
        assert result.species_ranges[0].species == "Neoalbaillella optima"
        # And to_dict() round-trips both fields.
        d = result.to_dict()
        assert d["status"] == "ok"
        assert d["error_message"] is None
