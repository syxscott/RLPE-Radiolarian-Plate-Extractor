"""Regression: audit 2026-09-04 eval-7 — the image-verified F1
counter in :mod:`scripts.evaluate_image_verified` looked up panel
crops using the **GOLD** panel_id:

    crop = find_panel_crop(panels_root, pid, g.figure_id, str(g.panel_id))

The pipeline, however, writes crops using the **pred** row's
position in the result list:

    crop_path = crop_dir / f"panel_{panel_idx:02d}.png"

where ``panel_idx = orig_i + 1`` is the result-row index — NOT the
predicted panel_id string. So when pred panel_id != gold panel_id
(common: OCR misreads "3" as "8", LLM hallucinates a wrong label),
the gold-driven lookup returns ``None`` and the panel is silently
skipped from the image-verified counter. When they DO happen to
match, the OCR then re-reads the crop and trivially confirms the
pred's already-correct string — i.e. the verification is
**tautological**: it never tests whether the printed label in the
actual image matches gold.

Two failure modes from a single root cause:

  1. **Under-count**: gold says panel_id="7" but pipeline wrote
     ``panel_03.png`` for the pred that matched gold #3 in spatial
     order. The gold-driven lookup misses panel 7 entirely.
  2. **Tautological verification**: when pred guessed "1" and gold
     also says "1", the script OCRs the crop, sees "1", compares to
     gold "1", counts as verified — even if the crop was generated
     by the same pred-string that wrote itself into the filename.

Fix contract:
  * Iterate PREDS (with valid ``panel_path``) for crop lookup, not
    gold panels.
  * Match pred → gold by (figure_id, position index within the
    figure). Spatial order is preserved by the pipeline.
  * OCR'd label from the crop is compared against the matched gold
    panel's panel_id (real ground truth), not against the pred
    panel_id string.

This test pins the contract via behavioural runs against synthetic
fixtures. We monkeypatch :func:`easyocr_panel_label` so we don't
need real EasyOCR / cv2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluate_image_verified import evaluate_image_verified  # noqa: E402


def _write_crop(panels_root: Path, paper_id: str, figure_id: str, idx: int) -> Path:
    """Create a 1x1 PNG crop at the pipeline's position-keyed filename.

    Filename: ``panel_{idx:02d}.png`` (matches pipeline.py:3163,
    5950 conventions).
    """
    from PIL import Image

    fig_dir = panels_root / paper_id / figure_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / f"panel_{idx:02d}.png"
    Image.new("RGB", (1, 1), color="white").save(p)
    return p


def _make_easyocr_stub(monkeypatch: pytest.MonkeyPatch, label_map: dict[str, str]):
    """Replace ``easyocr_panel_label`` so the eval sees an OCR'd
    label of our choosing for each crop path.

    The stub: ``label_map[path]`` returns the OCR'd string for that
    crop; missing keys return ``None`` (OCR failure).
    """
    import scripts.evaluate_image_verified as evmod

    def stub(image_path, reader=None):  # noqa: ARG001 — signature parity
        return label_map.get(str(image_path))

    monkeypatch.setattr(evmod, "easyocr_panel_label", stub)
    # Audit 2026-09-05: inject a FAKE ``easyocr`` module instead of
    # importing the real one — the CI test extras don't include
    # easyocr, and this stub never needs real OCR. Injecting into
    # ``sys.modules`` also means the eval's own lazy
    # ``import easyocr`` resolves to the fake, so the eval sees a
    # usable reader and runs the full OCR loop on every environment.
    import types

    _fake_easyocr = types.ModuleType("easyocr")
    monkeypatch.setitem(sys.modules, "easyocr", _fake_easyocr)
    import easyocr as _easyocr  # noqa: E402 — resolves to the injected fake

    class _DummyReader:
        def readtext(self, img, detail=1):  # noqa: ARG002
            return []

    monkeypatch.setattr(_easyocr, "Reader", lambda *a, **kw: _DummyReader(), raising=False)


class TestImageVerifiedUsesPredCropPath:
    def test_non_numeric_gold_panel_id_does_not_break_verification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Real bug: gold uses non-numeric panel_ids like "7a"
        while pipeline writes position-keyed crops (panel_01.png).

        OLD code: ``find_panel_crop(..., "7a")`` → tries
        ``panel_7a.png``, ``panel_07a.png``, ``panel_7a.jpg`` —
        ``int("7a")`` raises ValueError, caught, fallback stays
        "7a" — none of those filenames exist → ``None`` returned
        → panel silently skipped. ``n_checked == 0`` → paper
        reported as blocked.

        NEW code: iterates preds, uses pred's ``panel_path`` which
        points to the real crop. OCR reads the printed label "7a",
        compares to gold.panel_id "7a" → verified.

        This test proves the fix works for the common case of
        non-numeric gold panel_ids.
        """
        panels_root = tmp_path / "panels"
        gold_dir = tmp_path / "gold"
        pred_jsonl = tmp_path / "preds.jsonl"

        # Pipeline writes position-keyed crops (panel_01.png etc.)
        crops = [
            _write_crop(panels_root, "p1", "fig1", 1),
            _write_crop(panels_root, "p1", "fig1", 2),
            _write_crop(panels_root, "p1", "fig1", 3),
        ]
        # OCR reads "7a", "7b", "7c" from the crops (matching gold)
        label_map = {
            str(crops[0]): "7a",
            str(crops[1]): "7b",
            str(crops[2]): "7c",
        }
        _make_easyocr_stub(monkeypatch, label_map)

        # Gold: non-numeric panel_ids like the real corpus
        gold_dir.mkdir(parents=True, exist_ok=True)
        (gold_dir / "p1.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "paper_id": "p1",
                        "figure_id": "fig1",
                        "panel_id": ["7a", "7b", "7c"][i],
                        "species": f"Genus {i + 1}",
                    }
                )
                for i in range(3)
            )
        )

        # Preds: panel_paths point to the real crops
        pred_rows = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": ["7a", "7b", "7c"][i],
                "species": f"Genus {i + 1}",
                "panel_path": str(crops[i]),
            }
            for i in range(3)
        ]
        pred_jsonl.write_text("\n".join(json.dumps(r) for r in pred_rows))

        report = evaluate_image_verified(
            pred_jsonl=pred_jsonl, gold_dir=gold_dir, panels_root=panels_root
        )
        paper = report["papers"]["p1"]
        # Pred-driven: n_checked == 3 (every pred has panel_path → crop)
        assert paper["n_checked"] == 3, (
            f"audit 2026-09-04 eval-7: pred-panel_path-driven lookup "
            f"should check all 3 preds via pred's panel_path, but "
            f"gold-driven lookup found only "
            f"n_checked={paper['n_checked']}. Non-numeric gold panel_ids "
            f"like '7a' must not break verification."
        )
        # OCR'd labels (7a,7b,7c) match gold (7a,7b,7c) → all verified
        assert paper["n_image_verified"] == 3, (
            f"audit 2026-09-04 eval-7: OCR'd labels should match "
            f"gold panel_ids for all 3 preds, got "
            f"n_image_verified={paper['n_image_verified']}"
        )

    def test_pred_panel_path_drives_crop_lookup_when_gold_unfindable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Gold panel_ids are formatted differently from crop
        filenames (gold uses letter+number, pipeline writes
        position-keyed). OLD code's find_panel_crop can't locate
        the crop → n_checked = 0. NEW code uses pred.panel_path
        and finds the crop directly."""
        panels_root = tmp_path / "panels"
        gold_dir = tmp_path / "gold"
        pred_jsonl = tmp_path / "preds.jsonl"

        crops = [
            _write_crop(panels_root, "p1", "fig1", 1),
            _write_crop(panels_root, "p1", "fig1", 2),
        ]
        # OCR reads the printed labels "1" and "2"
        label_map = {str(crops[0]): "1", str(crops[1]): "2"}
        _make_easyocr_stub(monkeypatch, label_map)

        gold_dir.mkdir(parents=True, exist_ok=True)
        (gold_dir / "p1.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "paper_id": "p1",
                        "figure_id": "fig1",
                        "panel_id": "Fig.1.A" if i == 0 else "Fig.1.B",
                        "species": f"Genus {i + 1}",
                    }
                )
                for i in range(2)
            )
        )

        # Pred panel_ids differ from gold — but pred knows its crop
        pred_rows = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "wrong-A" if i == 0 else "wrong-B",
                "species": f"Genus {i + 1}",
                "panel_path": str(crops[i]),
            }
            for i in range(2)
        ]
        pred_jsonl.write_text("\n".join(json.dumps(r) for r in pred_rows))

        report = evaluate_image_verified(
            pred_jsonl=pred_jsonl, gold_dir=gold_dir, panels_root=panels_root
        )
        paper = report["papers"]["p1"]
        # Pred-driven: n_checked == 2 (both preds have valid panel_paths)
        assert paper["n_checked"] == 2, (
            f"audit 2026-09-04 eval-7: pred-driven lookup should "
            f"find 2 crops via pred.panel_path. Got n_checked="
            f"{paper['n_checked']}. (OLD code: gold-driven lookup "
            f"could not find 'panel_Fig.1.A.png' and missed the crop.)"
        )
        # OCR'd labels (1,2) DO NOT match gold ("Fig.1.A", "Fig.1.B") —
        # this is correct behaviour: the image-verified check rejects
        # the verification because OCR'd text doesn't equal gold.
        # The proof here is that n_checked > 0: we DID check the crop.
        # OCR vs gold mismatch is honest reporting.
        assert paper["n_image_verified"] == 0  # OCR != gold

    def test_no_tautology_when_ocr_disagrees_with_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Prove non-tautology: pred and gold both say "1", but
        OCR reads the actual image label as "9" — the verification
        must FAIL. If the script trusted pred's panel_id or
        tautologically confirmed gold's, this would incorrectly
        pass."""
        panels_root = tmp_path / "panels"
        gold_dir = tmp_path / "gold"
        pred_jsonl = tmp_path / "preds.jsonl"

        c1 = _write_crop(panels_root, "p1", "fig1", 1)
        # OCR reads "9" (the actual printed label in the image)
        label_map = {str(c1): "9"}
        _make_easyocr_stub(monkeypatch, label_map)

        gold_dir.mkdir(parents=True, exist_ok=True)
        (gold_dir / "p1.jsonl").write_text(
            json.dumps(
                {
                    "paper_id": "p1",
                    "figure_id": "fig1",
                    "panel_id": "1",
                    "species": "Genus 1",
                }
            )
        )

        pred_rows = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "1",  # pred also says "1"
                "species": "Genus 1",
                "panel_path": str(c1),
            }
        ]
        pred_jsonl.write_text(json.dumps(pred_rows[0]))

        report = evaluate_image_verified(
            pred_jsonl=pred_jsonl, gold_dir=gold_dir, panels_root=panels_root
        )
        paper = report["papers"]["p1"]
        # Crop was found (via pred's panel_path) and OCR'd → n_checked = 1
        assert paper["n_checked"] == 1
        # But OCR'd label "9" != gold "1" → NOT verified.
        # This is the proof that the verification is NOT tautological:
        # the OCR result is the deciding signal, not pred's string.
        assert paper["n_image_verified"] == 0, (
            f"audit 2026-09-04 eval-7: verification is tautological — "
            f"OCR'd '9' should NOT match gold '1' but reported "
            f"n_image_verified={paper['n_image_verified']}. The fix "
            f"must compare OCR result to gold, not pred."
        )

    def test_missing_pred_panel_path_skips_pred(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A pred WITHOUT a panel_path cannot be image-verified —
        the pipeline never cropped it. The fix must skip such preds
        (do not fabricate a path from gold.panel_id)."""
        panels_root = tmp_path / "panels"
        gold_dir = tmp_path / "gold"
        pred_jsonl = tmp_path / "preds.jsonl"

        c1 = _write_crop(panels_root, "p1", "fig1", 1)
        label_map = {str(c1): "1"}
        _make_easyocr_stub(monkeypatch, label_map)

        gold_dir.mkdir(parents=True, exist_ok=True)
        (gold_dir / "p1.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "paper_id": "p1",
                        "figure_id": "fig1",
                        "panel_id": str(i + 1),
                        "species": f"Genus {i + 1}",
                    }
                )
                for i in range(2)
            )
        )

        pred_rows = [
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "1",
                "species": "Genus 1",
                "panel_path": str(c1),
            },
            {
                "paper_id": "p1",
                "figure_id": "fig1",
                "panel_id": "2",
                "species": "Genus 2",
                # No panel_path → pipeline never cropped it
            },
        ]
        pred_jsonl.write_text("\n".join(json.dumps(r) for r in pred_rows))

        report = evaluate_image_verified(
            pred_jsonl=pred_jsonl, gold_dir=gold_dir, panels_root=panels_root
        )
        paper = report["papers"]["p1"]
        # Only pred 1 has a panel_path → n_checked = 1, pred 2 skipped.
        assert paper["n_checked"] == 1
        # OCR'd "1" matches gold "1" → 1 verified.
        assert paper["n_image_verified"] == 1
