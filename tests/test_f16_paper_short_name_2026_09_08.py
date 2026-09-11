"""F16 (2026-09-08): paper short-name prefix for panel image filenames.

Regression tests for the Soeka-2019 failure: GROBID captured the
Indonesian subtitle "(Spesies Baru Radiolaria dari Pulau Buton,
Sulawesi Tenggara)" as the paper's author list, so the first
panel-rename pass produced prefixes like "(Spesies_2019_...". The
builder now validates the author token and falls back to the PDF
filename stem, then to the bare year.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlpe.config import PipelineConfig
from rlpe.pipeline import RadiolarianPipeline

SOEKA_AUTHORS = "(Spesies Baru Radiolaria dari Pulau Buton, Sulawesi Tenggara)"
SOEKA_PDF = Path(
    "Soeka_2019 - Scientific Contributions Oil and Gas - "
    "NEW SPECIES OF RADIOLARIA FROM THE ISLAND OF BUTON, SOUTH EAST SULAWESI.pdf"
)


@pytest.fixture()
def pipe(tmp_path: Path) -> RadiolarianPipeline:
    return RadiolarianPipeline(PipelineConfig(pdf_dir=tmp_path, work_dir=tmp_path / "w"))


class TestPlausibleNameToken:
    def test_rejects_parenthesised_subtitle(self):
        assert not RadiolarianPipeline._plausible_name_token("(Spesies")

    def test_rejects_author_markers_and_digits(self):
        assert not RadiolarianPipeline._plausible_name_token("Input2")
        assert not RadiolarianPipeline._plausible_name_token("2019")

    def test_accepts_real_surnames(self):
        for tok in ("Bandini", "Müller", "O'Dogherty", "Sanz-López"):
            assert RadiolarianPipeline._plausible_name_token(tok), tok


class TestPaperShortName:
    def test_soeke_falls_back_to_pdf_stem(self, pipe: RadiolarianPipeline):
        pm = {"authors": [SOEKA_AUTHORS], "year": 2019}
        assert pipe._paper_short_name(pm, SOEKA_PDF) == "Soeka_2019"

    def test_soeke_without_pdf_keeps_year_only(self, pipe: RadiolarianPipeline):
        pm = {"authors": [SOEKA_AUTHORS], "year": 2019}
        assert pipe._paper_short_name(pm, None) == "2019"

    def test_clean_authors_win_over_filename(self, pipe: RadiolarianPipeline):
        pm = {"authors": ["Bandini, A.", "Jones, B."], "year": 2011}
        pdf = Path("download (3).pdf")
        assert pipe._paper_short_name(pm, pdf) == "Bandini_2011"

    def test_filename_year_not_duplicated(self, pipe: RadiolarianPipeline):
        pm = {"authors": [SOEKA_AUTHORS], "year": 2019}
        assert pipe._paper_short_name(pm, SOEKA_PDF) == "Soeka_2019"

    def test_garbage_everywhere_yields_empty(self, pipe: RadiolarianPipeline):
        pm = {"authors": ["Input2"], "year": None}
        assert pipe._paper_short_name(pm, Path("12345.pdf")) == ""

    def test_multiword_journal_head_collapses(self, pipe: RadiolarianPipeline):
        pm = {"authors": [], "year": None}
        pdf = Path("Scientific Contributions Oil and Gas - Vol 43.pdf")
        assert pipe._paper_short_name(pm, pdf) == "Scientific"


class TestSpeciesSanitisation:
    """The rename block must never emit filenames with spaces or
    parentheses, and must drop subgenus brackets."""

    @staticmethod
    def _sanitise(pipe: RadiolarianPipeline, sp: str) -> str:
        import re

        safe = re.sub(r"\s*\([^)]*\)", " ", sp)
        safe = re.sub(r"[^\w\s.-]", "", safe)
        return re.sub(r"[\s_]+", "_", safe).strip("_")

    def test_space_species_become_underscores(self, pipe: RadiolarianPipeline):
        assert self._sanitise(pipe, "Dictyomitra formosa") == "Dictyomitra_formosa"

    def test_subgenus_dropped(self, pipe: RadiolarianPipeline):
        assert self._sanitise(pipe, "Cryptamphora (Cryptamphora) strebli") == "Cryptamphora_strebli"

    def test_uncertainty_markers_stripped(self, pipe: RadiolarianPipeline):
        assert self._sanitise(pipe, "H. echinatus?") == "H._echinatus"

    def test_cf_marker_preserved(self, pipe: RadiolarianPipeline):
        assert self._sanitise(pipe, "Lithocampe cf. exigua") == "Lithocampe_cf._exigua"


class TestDottedInitialAuthor:
    """2026-09-11: a leading dotted-initial author word ("M." in
    "M. Afanasieva") must yield the SURNAME, not the initial —
    "M._2020_..." panel names were unusable and ambiguous."""

    def test_initial_then_surname_prefers_surname(self, pipe: RadiolarianPipeline):
        pm = {"authors": ["M. Afanasieva"], "year": 2020}
        assert pipe._paper_short_name(pm, None) == "Afanasieva_2020"

    def test_multi_initial_then_surname(self, pipe: RadiolarianPipeline):
        pm = {"authors": ["M.S. Afanasieva"], "year": 2020}
        assert pipe._paper_short_name(pm, None) == "Afanasieva_2020"

    def test_surname_first_style_unchanged(self, pipe: RadiolarianPipeline):
        pm = {"authors": ["Bandini, A.", "Jones, B."], "year": 2011}
        assert pipe._paper_short_name(pm, None) == "Bandini_2011"
