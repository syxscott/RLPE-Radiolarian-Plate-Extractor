"""Audit 2026-09-05 — orphan-plate-page OCR rescue + zero-row visibility.

Regression tests for the completeness test findings (random 3-paper
sample vs manual reading):

* Soeka_2019 lost BOTH SEM plates (24 panels) silently: the caption
  pages' text layer uses a shifted encoding ("Plate" → "3ODWH"), so
  ``_find_plate_captions`` matched nothing and the plate pages never
  became figure entries. ``_rescue_orphan_plate_pages`` now renders the
  caption band of uncovered image pages, OCRs it, and promotes the page
  to a figure when the text carries a caption marker (Plate/Figs/
  clause list).
* A paper that yields ZERO rows after all fallbacks used to vanish
  (console-only warning, wrong path hint, checkpoint written anyway).
  ``_process_one_pdf`` now emits an ``_ingestion_zero_rows`` stub row
  that survives ``_finalize_rows`` (``ingestion_warning`` flag) and
  surfaces in ``run_output.warnings``.

Test notes:
* The synthetic-PDF tests inject a FAKE easyocr module (same pattern
  as test_eval7) so they run on CI without EasyOCR installed.
* The real-Soeka test needs actual EasyOCR (CPU) — it is skipped when
  easyocr or the corpus PDF is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from rlpe.opendataloader_extractor import (  # noqa: E402
    OpenDataLoaderExtractor,
    _rescue_orphan_plate_pages_marker_check,
)

_SOEKA = (
    _ROOT
    / "放射虫论文_OA_download"
    / "Soeka_2019 - Scientific Contributions Oil and Gas - NEW SPECIES OF "
    "RADIOLARIA FROM THE ISLAND OF BUTON, SOUTH EAST SULAWESI.pdf"
)


def _make_plate_pdf(tmp_path: Path) -> Path:
    """Build a 1-page PDF whose only content is a large image — no text
    layer (what OD sees in scanned/obfuscated-caption papers)."""
    import fitz
    from PIL import Image

    img_path = tmp_path / "plate.png"
    Image.new("RGB", (400, 500), color=(240, 240, 240)).save(img_path)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(100, 200, 500, 700), filename=str(img_path))
    out = tmp_path / "plate_only.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _inject_fake_easyocr(monkeypatch: pytest.MonkeyPatch, ocr_text: str) -> None:
    """Inject a fake ``easyocr`` module whose Reader.readtext returns one
    detection covering ``ocr_text`` (mirrors test_eval7's pattern)."""

    class _FakeReader:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002/ANN003
            pass

        def readtext(self, img, **kwargs):  # noqa: ANN001/ANN003
            # EasyOCR result shape: [bbox, text, confidence].
            return [
                ([[0, 0], [10, 0], [10, 10], [0, 10]], ocr_text, 0.9),
            ]

    fake = pytest.importorskip("types").ModuleType("easyocr")
    fake.Reader = _FakeReader
    monkeypatch.setitem(sys.modules, "easyocr", fake)


class TestOrphanPlatePageRescue:
    @staticmethod
    def _make_data_with_source(tmp_path: Path) -> tuple[dict, Path]:
        """Build the OD-tree image element AND the on-disk image file it
        references (``_resolve_image_paths`` requires a real file)."""
        from PIL import Image

        img_dir = tmp_path / "od_output" / "p1" / "plate_only_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_file = img_dir / "imageFile1.png"
        Image.new("RGB", (400, 500), color=(240, 240, 240)).save(img_file)
        data = {
            "kids": [
                {
                    "type": "image",
                    "id": 1,
                    "page number": 1,
                    "bounding box": [100.0, 142.0, 500.0, 642.0],
                    "source": "plate_only_images/imageFile1.png",
                }
            ]
        }
        # od_dir mirrors what extract() passes: output_dir/od_output/<paper_id>
        return data, img_dir.parent

    def test_synthetic_plate_page_promoted(self, tmp_path, monkeypatch):
        """A 1-page PDF with a big image and no text layer must be
        promoted to a figure when the band OCR reads a caption."""
        _inject_fake_easyocr(monkeypatch, "Plate 1. Figs 1-2. Testus species Example")
        pdf = _make_plate_pdf(tmp_path)
        # OD finds no figures (no text layer at all → no captions).
        figures: list = []
        data, od_dir = self._make_data_with_source(tmp_path)
        ex = OpenDataLoaderExtractor(use_ocr=True)
        out = ex._rescue_orphan_plate_pages(pdf, figures, data, od_dir, "p1")
        assert len(out) == 1, "the orphan plate page should be promoted"
        fig = out[0]
        assert "p001" in fig.figure_id
        assert fig.metadata["caption_recovered_via"] == "ocr_page_rescue"
        assert "Plate 1" in (fig.caption_text or "")
        assert fig.image_paths and Path(fig.image_paths[0]).exists()

    def test_page_without_caption_marker_not_promoted(self, tmp_path, monkeypatch):
        """A band OCR that reads non-caption text (no marker, <3 clauses)
        must NOT be promoted — never fabricate captions."""
        _inject_fake_easyocr(monkeypatch, "Sample location map of the region")
        pdf = _make_plate_pdf(tmp_path)
        data, od_dir = self._make_data_with_source(tmp_path)
        ex = OpenDataLoaderExtractor(use_ocr=True)
        out = ex._rescue_orphan_plate_pages(pdf, [], data, od_dir, "p1")
        assert out == []

    def test_duplicate_caption_suppressed(self, tmp_path, monkeypatch):
        """A stray neighbouring image whose band OCR reads the SAME
        caption as an existing pair must be skipped (Soeka p3 case)."""
        _inject_fake_easyocr(
            monkeypatch,
            "Figure 2 Schematic tectonic configuration of Buton and adjacent areas",
        )
        pdf = _make_plate_pdf(tmp_path)
        from rlpe.opendataloader_extractor import FigureCaptionPair

        existing = [
            FigureCaptionPair(
                figure_id="od_fig_x_p004_01",
                page_number=4,
                image_paths=["a.png"],
                caption_text=(
                    "Figure 2 Schematic tectonic configuration of Buton "
                    "and adjacent areas (modified from Van Marle, 1989)"
                ),
                merged_bbox=(1, 1, 2, 2),
            )
        ]
        data, od_dir = self._make_data_with_source(tmp_path)
        ex = OpenDataLoaderExtractor(use_ocr=True)
        out = ex._rescue_orphan_plate_pages(pdf, existing, data, od_dir, "p1")
        # The method returns ``figures + rescued``; a suppressed rescue
        # means NO new entries beyond the input list.
        assert len(out) == len(existing), "near-duplicate caption must be suppressed"

    def test_small_images_skipped(self, tmp_path, monkeypatch):
        """Images covering <15% of the page (logos, decorations) must not
        trigger the OCR rescue."""
        _inject_fake_easyocr(monkeypatch, "Plate 1. Testus species Example")
        pdf = _make_plate_pdf(tmp_path)
        data = {
            "kids": [
                {
                    "type": "image",
                    "id": 1,
                    "page number": 1,
                    "bounding box": [10.0, 10.0, 60.0, 50.0],  # tiny logo
                }
            ]
        }
        ex = OpenDataLoaderExtractor(use_ocr=True)
        out = ex._rescue_orphan_plate_pages(pdf, [], data, tmp_path, "p1")
        assert out == []

    @pytest.mark.skipif(
        not _SOEKA.exists(),
        reason="Soeka_2019 corpus PDF not present",
    )
    @pytest.mark.skipif(
        pytest.importorskip("importlib.util").find_spec("easyocr") is None,
        reason="EasyOCR not installed",
    )
    def test_real_soeka_plates_rescued(self, tmp_path):
        """The direct incident sentinel: Soeka's two obfuscated-caption
        plate pages (p9, p10) must be recovered by the rescue pass."""
        ex = OpenDataLoaderExtractor(use_ocr=True)
        res = ex.extract(_SOEKA, tmp_path / "od")
        assert res.success
        rescued = [
            f for f in res.figures if f.metadata.get("caption_recovered_via") == "ocr_page_rescue"
        ]
        pages = {f.page_number for f in rescued}
        assert {9, 10} <= pages, f"expected the p9/p10 plate pages to be rescued, got pages {pages}"
        # The recovered captions carry species clauses.
        joined = " ".join(f.caption_text or "" for f in rescued)
        assert "Butonastrum" in joined or "Lithocyclia" in joined


class TestZeroRowVisibility:
    def test_zero_row_stub_emitted_and_survives_finalize(self, tmp_path, monkeypatch):
        """A paper whose extraction yields 0 rows must produce an
        ``_ingestion_zero_rows`` stub that survives _finalize_rows."""
        from rlpe.config import PipelineConfig
        from rlpe.pipeline import RadiolarianPipeline

        cfg = PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w")
        cfg.extra["use_opendataloader"] = False
        pipe = RadiolarianPipeline(cfg)
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 stub")
        # Force both inner paths to return zero rows.
        monkeypatch.setattr(pipe, "_process_one_pdf_grobid", lambda *a, **k: [])
        rows = pipe._process_one_pdf(pdf)
        assert len(rows) == 1
        assert rows[0]["figure_id"] == "_ingestion_zero_rows"
        assert rows[0]["metadata"]["ingestion_warning"] is True
        # And it survives the finalize filter (called with the stub).
        kept = pipe._finalize_rows(rows)
        assert any(r["metadata"].get("ingestion_warning") for r in kept), (
            "_finalize_rows dropped the zero-row warning stub"
        )

    def test_marker_check_helper(self):
        """The lenient marker regex accepts plural 'Figs.' + OCR noise
        ('Figs. [-2' — a misread digit), and the clause-list heuristic
        accepts header-less species lists."""
        assert _rescue_orphan_plate_pages_marker_check("Figs. [-2. Heliodiscus cf H.")[0]
        assert _rescue_orphan_plate_pages_marker_check("Plate 1. Testus species")[0]
        # Header-less clause list (Soeka p10 shape).
        text = (
            "Spongatractus pachystylus (Ehrenberg) 8. Amphicraspedum prolixum "
            "Sanfilippo 9. Actinoma panujui Soeka 10. Spongotrochus buskamali"
        )
        ok, _ = _rescue_orphan_plate_pages_marker_check(text)
        assert ok, "clause-list heuristic should accept a header-less plate caption"
        # Non-caption text must fail both.
        ok, _ = _rescue_orphan_plate_pages_marker_check("Sample location map of the region")
        assert not ok


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
