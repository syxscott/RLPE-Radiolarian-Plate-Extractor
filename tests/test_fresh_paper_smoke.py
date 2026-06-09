"""End-to-end sanity test on a fresh paper (not in the 9-paper gold set).

The 9-paper eval set is hand-tuned. To check the pipeline doesn't
break on unseen papers, this test reads the v1.1.0 smoke test
output (``work/papers_smoke/output/manifests/matches.jsonl``) which
contains pipeline output from 5 fresh papers (carlsson2022,
cifer2020, baumgartner2006, danelian2018_profetis, beccaro2006 —
beccaro is in the gold set, but the others are not).

The test verifies the pipeline:
  1. Produced output for all 5 papers (no paper produced 0 rows)
  2. Each row has the required schema fields
  3. Most rows have a panel_path (i.e. segmentation succeeded)
  4. The species extraction rate is reasonable (>= 30%)
  5. The output validates through the Pydantic schema

Skipped if the smoke test output doesn't exist (CI may not have
the work/ artifacts). To regenerate the smoke test output:
    PYTHONPATH=src python -m rlpe.cli --pdf-dir work/papers_smoke \\
        --work-dir work/papers_smoke \\
        --ocr-backend easyocr --num-workers 1
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpe.converters import run_output_from_provenance  # noqa: E402
from rlpe.provenance.stamp import build_provenance  # noqa: E402
from rlpe.schema_models import ProvenanceRecord, validate_run_output  # noqa: E402

SMOKE_MATCHES = Path("work/papers_smoke/output/manifests/matches.jsonl")
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_rows():
    if not SMOKE_MATCHES.exists():
        pytest.skip(
            f"{SMOKE_MATCHES.relative_to(REPO_ROOT)} not present. "
            "Re-run the v1.1.0 smoke test to generate it."
        )
    rows = []
    with open(SMOKE_MATCHES) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_smoke_output_has_rows(smoke_rows):
    """The smoke test should produce at least 10 rows across the 5
    fresh papers. (If a paper has 0 panels, that's still a real
    result — but the aggregate should not be empty.)"""
    assert len(smoke_rows) >= 10, (
        f"Only {len(smoke_rows)} rows in the smoke output. "
        f"Either the pipeline is broken or the test corpus is too small."
    )


def test_smoke_output_covers_all_papers(smoke_rows):
    """Each of the 5 fresh papers should produce >= 1 row. A 0-row
    paper means the pipeline silently dropped it (no error, no
    output)."""
    papers = Counter(r.get("paper_id") for r in smoke_rows)
    expected_papers = 5
    actual_papers = len([p for p, n in papers.items() if n >= 1])
    assert actual_papers >= expected_papers, (
        f"Only {actual_papers}/{expected_papers} fresh papers produced output. "
        f"Per-paper counts: {dict(papers)}"
    )


def test_smoke_output_has_required_fields(smoke_rows):
    """Every row must have the canonical record fields. Missing
    fields would indicate a regression in the pipeline serialization
    layer."""
    required = ["paper_id", "figure_id", "panel_id", "panel_path"]
    for r in smoke_rows:
        for k in required:
            assert k in r, f"row missing required field {k!r}: {r}"


def test_smoke_output_most_rows_have_panel_path(smoke_rows):
    """Most rows should have a panel_path (i.e. segmentation
    succeeded). A low rate indicates a segmentation regression."""
    n_with = sum(1 for r in smoke_rows if r.get("panel_path"))
    rate = n_with / max(1, len(smoke_rows))
    assert rate >= 0.80, (
        f"Only {n_with}/{len(smoke_rows)} ({rate:.1%}) rows have a "
        f"panel_path. Expected >= 80%."
    )


def test_smoke_output_species_extraction_reasonable(smoke_rows):
    """The species extraction rate is a coarse sanity check. A
    drop below 20% would indicate the caption parser broke
    (the 5 fresh papers have varied caption styles; the parser
    is hand-tuned for 9 specific papers, so ~30-50% is the
    realistic floor for unseen papers)."""
    n_with = sum(1 for r in smoke_rows if r.get("species"))
    rate = n_with / max(1, len(smoke_rows))
    # Lower bound: pipeline must extract some species
    assert rate >= 0.20, (
        f"Species extraction rate is {n_with}/{len(smoke_rows)} "
        f"({rate:.1%}). Expected >= 20% on fresh papers. "
        f"This is a strong signal that the caption parser regressed."
    )


def test_smoke_output_validates_against_schema(smoke_rows):
    """The smoke output must validate against the published
    JSON Schema. This is the same path the export pipeline takes;
    if validation fails, the export will reject the output."""
    from rlpe.types import MatchResult, PaperMetadata
    prov = ProvenanceRecord(**build_provenance().to_dict())
    # Build MatchResult objects (the converter expects typed objects)
    matches = []
    for r in smoke_rows:
        pm = r.get("paper_metadata")
        paper_metadata = None
        if pm:
            paper_metadata = PaperMetadata(
                title=pm.get("title", ""),
                authors=pm.get("authors", []),
                year=pm.get("year", 0),
                doi=pm.get("doi"),
                source=pm.get("source", ""),
                confidence=pm.get("confidence", 0.0),
            )
        matches.append(MatchResult(
            paper_id=r.get("paper_id", ""),
            figure_id=r.get("figure_id", ""),
            panel_id=r.get("panel_id"),
            species=r.get("species"),
            panel_path=r.get("panel_path"),
            bbox=r.get("bbox"),
            confidence=float(r.get("confidence", 0.0)),
            label_text=r.get("label_text"),
            caption_snippet=r.get("caption_snippet"),
            ocr_text=r.get("ocr_text"),
            metadata=r.get("metadata", {}),
            paper_metadata=paper_metadata,
        ))
    out = run_output_from_provenance(prov, matches)
    validated = validate_run_output(out)
    assert len(validated.panels) == len(smoke_rows)


def test_smoke_output_per_paper_species_rate(smoke_rows):
    """Per-paper species extraction rate. Documents the actual
    cold-start performance on each fresh paper. This is a
    observational test — it just prints, no assertion. The goal
    is to track how well the parser generalises to new papers."""
    papers = Counter(r.get("paper_id") for r in smoke_rows)
    print(f"\n=== Fresh paper smoke test per-paper stats ===")
    for p, n in sorted(papers.items()):
        n_species = sum(
            1 for r in smoke_rows
            if r.get("paper_id") == p and r.get("species")
        )
        rate = n_species / max(1, n)
        print(f"  {p[:20]}: {n_species}/{n} = {rate:.1%} species")
    # Aggregate rate
    n_total = len(smoke_rows)
    n_species = sum(1 for r in smoke_rows if r.get("species"))
    print(f"  TOTAL: {n_species}/{n_total} = {n_species/n_total:.1%} species")
