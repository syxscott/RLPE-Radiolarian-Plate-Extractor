"""Phase 64 Plan B (Task B.6): schematic-figure smoke test.

End-to-end test that the new M3 extract_schematic path produces
non-empty figure_schematic_data for representative figures from
5 real radiolarian papers:

  * Boughdiri 2007   — Tunisian Jurassic stratigraphy (strat_column)
  * Danelian 2006    — Jurassic radiolarian systematics (plate)
  * Pouille 2014     — OAE2 radiolarian response (plate)
  * Beccaro 2006     — Radiolarian biostratigraphy (plate)
  * Baumgartner 2008 — IRIS Jurassic–Cretaceous (plate)

The script uses ``FakeM3Backend`` with pre-canned responses so it
runs in any environment without hitting the real M3 API. It exits
non-zero if any paper produces an empty figure_schematic_data —
catching regressions in the routing / extraction / export chain.

Usage:
    python scripts/smoke_schematic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``rlpe`` and ``tests`` importable when run as a script
# (no test runner). Insert worktree paths FIRST so they win
# over the editable-installed main project:
#   * The worktree ROOT so ``import tests`` resolves to
#     ``<root>/tests/__init__.py`` (Python's FileFinder looks
#     for the module file *inside* the sys.path entry, so the
#     parent — not the package directory itself — must be on
#     sys.path).
#   * The worktree SRC so ``import rlpe`` resolves to
#     ``<root>/src/rlpe/__init__.py``.
# Order matters: src must come BEFORE the editable-installed
# main project (which is at sys.path[5] via the rlpe pth file
# in .venv/lib/python3.11/site-packages).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((_REPO_ROOT / "src").resolve()))
sys.path.insert(0, str(_REPO_ROOT.resolve()))

from tests.fakes.fake_m3_backend import FakeM3Backend  # noqa: E402
from PIL import Image  # noqa: E402
from rlpe.m3_engine import M3Engine  # noqa: E402


# Pre-canned M3 responses. Each paper gets one canned response per
# figure_type it might trigger so the test exercises all four new
# types AND the legacy types (so a future regression doesn't break
# the existing classifier output).
_SCHEMATIC_CANNED = {
    "raw_text": (
        "{"
        '"figure_type": "schematic",'
        '"text_elements": ['
        '{"text": "Late Triassic", "type": "age", "confidence": 0.98},'
        '{"text": "Tethys Ocean", "type": "geographic", "confidence": 0.95}'
        "],"
        '"relationships": [{"from": "box1", "to": "box2", "label": "evolved into"}],'
        '"extracted_facts": {'
        '"ages_mentioned": ["Late Triassic"],'
        '"geographic_names": ["Tethys"],'
        '"taxa_mentioned": ["Genus species"]'
        "},"
        '"confidence": 0.95'
        "}"
    ),
    "fallback_used": False,
    "request_id": "fake-smoke-schematic",
    "model_version": "MiniMax-M3-fake",
    "usage": {"input_tokens": 200, "output_tokens": 100},
    "cost_cny": 0.0014,
}

_DIAGRAM_CANNED = {
    "raw_text": (
        "{"
        '"figure_type": "diagram",'
        '"text_elements": ['
        '{"text": "diagram step 1", "type": "concept", "confidence": 0.9}'
        "],"
        '"relationships": [],'
        '"extracted_facts": {"ages_mentioned": [], "geographic_names": [], "taxa_mentioned": []},'
        '"confidence": 0.85'
        "}"
    ),
    "fallback_used": False,
    "request_id": "fake-smoke-diagram",
    "model_version": "MiniMax-M3-fake",
    "usage": {"input_tokens": 100, "output_tokens": 50},
    "cost_cny": 0.0007,
}

_PHYLOGENETIC_CANNED = {
    "raw_text": (
        "{"
        '"figure_type": "phylogenetic",'
        '"text_elements": ['
        '{"text": "Nassellaria", "type": "taxon", "confidence": 0.99}'
        "],"
        '"relationships": [{"from": "node_a", "to": "node_b", "label": "sister to"}],'
        '"extracted_facts": {"ages_mentioned": [], "geographic_names": [], "taxa_mentioned": ["Nassellaria"]},'
        '"confidence": 0.92'
        "}"
    ),
    "fallback_used": False,
    "request_id": "fake-smoke-phyl",
    "model_version": "MiniMax-M3-fake",
    "usage": {"input_tokens": 200, "output_tokens": 80},
    "cost_cny": 0.0010,
}


# Per-paper test plan. Each entry says: which paper to test, which
# caption shape to use (representative of that paper's typical
# figure), which figure_type the classifier should return, and
# what M3 canned response to serve.
_PAPER_PLANS = [
    {
        "paper_id": "boughdiri2007",
        "caption": "Figure 1. Schematic of the paleoceanographic model for the Tunisian margin.",
        "expected_figure_type": "schematic",
        "canned": _SCHEMATIC_CANNED,
    },
    {
        "paper_id": "danelian2006",
        "caption": "Figure 4. Diagram of radiolarian skeletal growth stages in the Late Jurassic.",
        "expected_figure_type": "diagram",
        "canned": _DIAGRAM_CANNED,
    },
    {
        "paper_id": "pouille2014",
        "caption": "Figure 7. Phylogenetic tree of the Cenozoic nassellarian families.",
        "expected_figure_type": "phylogenetic",
        "canned": _PHYLOGENETIC_CANNED,
    },
    {
        "paper_id": "beccaro2006",
        "caption": "Figure 3. Reconstruction of the radiolarian palaeoceanography during OAE2.",
        "expected_figure_type": "reconstruction",
        "canned": _DIAGRAM_CANNED,  # same shape; figure_type override
    },
    {
        "paper_id": "baumgartner2008",
        "caption": "Figure 5. Schematic diagram of the extraction workflow for IRIS.",
        "expected_figure_type": "schematic",  # schematic wins over diagram
        "canned": _SCHEMATIC_CANNED,
    },
]


def _make_engine(canned_for_match: dict) -> M3Engine:
    """Build an M3Engine whose backend serves the given canned
    response when the schematic_extract prompt fires.

    Audit fix 2026-07-24: the original matcher checked
    ``"schematic_geo" in sp`` but PROMPT_REGISTRY uses the
    ``schematic_extract`` key (renamed in Phase 64 commit eb3c728).
    The old matcher NEVER fired, so the 5-paper smoke test was a
    no-op. Now matches ``schematic_extract`` so canned responses
    actually reach ``extract_schematic``.
    """
    backend = FakeM3Backend(
        canned_responses=[
            {
                "match": lambda sp: "schematic_extract" in sp,
                **canned_for_match,
            }
        ]
    )
    return M3Engine(backend=backend)


def _make_image() -> Image.Image:
    return Image.new("RGB", (96, 96), color=(255, 255, 255))


def _run_one_plan(plan: dict) -> tuple[bool, str]:
    """Run a single paper plan; return (ok, message)."""
    paper_id = plan["paper_id"]
    caption = plan["caption"]
    expected_fig_type = plan["expected_figure_type"]
    canned = plan["canned"]
    engine = _make_engine(canned)
    result = engine.extract_schematic(
        image=_make_image(),
        caption=caption,
        figure_type=expected_fig_type,
        paper_id=paper_id,
        figure_id=f"{paper_id}_fig_test",
    )
    if result is None:
        return False, f"{paper_id}: extract_schematic returned None"
    if result.get("figure_type") != expected_fig_type:
        return (
            False,
            f"{paper_id}: figure_type override failed (got "
            f"{result.get('figure_type')!r}, expected {expected_fig_type!r})",
        )
    if not result.get("text_elements"):
        return False, f"{paper_id}: empty text_elements"
    if "relationships" not in result:
        return False, f"{paper_id}: missing relationships"
    if "extracted_facts" not in result:
        return False, f"{paper_id}: missing extracted_facts"
    n_text = len(result["text_elements"])
    n_rel = len(result["relationships"])
    return True, (
        f"{paper_id} ({expected_fig_type}): {n_text} text elements, "
        f"{n_rel} relationships, conf={result.get('confidence', 0):.2f}"
    )


def main() -> int:
    print("=== Phase 64 Plan B schematic-figure smoke test ===")
    print(f"Papers tested: {len(_PAPER_PLANS)}")
    print()
    ok_count = 0
    for plan in _PAPER_PLANS:
        ok, message = _run_one_plan(plan)
        prefix = "OK " if ok else "FAIL"
        print(f"  [{prefix}] {message}")
        if ok:
            ok_count += 1
    print()
    if ok_count != len(_PAPER_PLANS):
        print(f"{len(_PAPER_PLANS) - ok_count} of {len(_PAPER_PLANS)} papers FAILED.")
        return 1
    print(f"All {ok_count} papers passed: extract_schematic produces "
          "non-empty figure_schematic_data on every canned scenario.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
