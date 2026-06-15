"""Build the 4-paper ground-truth gold set from full plate captions.

This script reads the full plate captions (extracted from the
OpenDataLoader or manifest output of the batch4_v2 run) and applies
RLPE's three caption parsers (standard, Danelian, Pouille) to derive
the canonical (panel_label, species) mapping.

For papers where the parsers handle all clauses (danelian, hollis,
bandini), the gold is the parser output. For pouille — where the
caption is "figs N. Species; figs M. Species" rather than the Pouille
"Species (Pl. N, fig M)" form that the current parser expects — the
gold is hand-curated from the published caption text.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlpe.evaluation import GoldPanel, write_gold  # noqa: E402
from rlpe.m3_engine import _regex_parse_caption  # noqa: E402


# Map paper_id (hash) → manifest directory
PAPER_MANIFESTS = {
    "900c13a3f2473740": REPO_ROOT / "work" / "batch4_v2" / "output" / "manifests" / "900c13a3f2473740",
    "3d554d642954c720": REPO_ROOT / "work" / "batch4_v2" / "output" / "manifests" / "3d554d642954c720",
    "0af2fd3865413764": REPO_ROOT / "work" / "batch4_v2" / "output" / "manifests" / "0af2fd3865413764",
    "cb2011ef7be94959": REPO_ROOT / "work" / "batch4_v2" / "output" / "manifests" / "cb2011ef7be94959",
}

# Map paper_id (hash) → paper short name
PAPER_SHORT_NAMES = {
    "900c13a3f2473740": "danelian2006",
    "3d554d642954c720": "hollis2006",
    "0af2fd3865413764": "bandini2011",
    "cb2011ef7be94959": "pouille2014",
}


def _figure_id_from_manifest(manifest_path: Path) -> str:
    import json
    with open(manifest_path) as f:
        d = json.load(f)
    return d["figure_id"]


def _load_captions(paper_id: str) -> list[tuple[str, str]]:
    """Return [(figure_id, full_caption), ...] for a paper."""
    out: list[tuple[str, str]] = []
    manifest_dir = PAPER_MANIFESTS[paper_id]
    for manifest_path in sorted(manifest_dir.glob("*.json")):
        import json
        with open(manifest_path) as f:
            d = json.load(f)
        cap = (d.get("caption") or "").strip()
        if cap and "Plate" in cap or "plate" in cap or "Fig" in cap or "fig" in cap:
            out.append((d["figure_id"], cap))
    return out


def _build_gold_danelian(paper_id: str) -> list[GoldPanel]:
    """Danelian 2006: caption format "1) Species; 2-3) Species". Parser handles it."""
    gold: list[GoldPanel] = []
    for fig_id, caption in _load_captions(paper_id):
        pairs = _regex_parse_caption(caption)
        for pair in pairs:
            for label in pair.labels:
                gold.append(GoldPanel(
                    paper_id=paper_id, figure_id=fig_id,
                    panel_id=label, species=pair.species,
                ))
    return gold


def _build_gold_hollis(paper_id: str) -> list[GoldPanel]:
    """Hollis 2006: caption format "1. Species\n2. Species" with a leading preamble.

    Hollis species shape: Genus + optional qualifiers (aff./cf./gr./?/sp.)
    + epithet. The line ends with the author in ALL CAPS (e.g. "HAECKEL"),
    in parentheses, or in sample/location info like "CH12076, P30/f534".
    We capture the species and stop before the first ALL-CAPS or parenthesized
    token.
    """
    # Tokenize after the label: take tokens until we hit a stopper.
    # Stoppers: ALL-CAPS token, "(...)", number, ","
    line_re = re.compile(r"^\s*(\d{1,3})\.\s+(.+)$")
    stop_re = re.compile(r"[\(,]|\b[A-Z]{2,}\b|\bCH\d|\bP\d+|\bRP\d|\d+\s*m\b")
    gold: list[GoldPanel] = []
    for fig_id, caption in _load_captions(paper_id):
        for line in caption.splitlines():
            m = line_re.match(line)
            if not m:
                continue
            label = m.group(1)
            rest = m.group(2)
            # Find the first stopper position
            sm = stop_re.search(rest)
            species = rest[:sm.start()].strip() if sm else rest.strip()
            # Strip trailing period
            species = species.rstrip(".").rstrip(",")
            if not species:
                continue
            gold.append(GoldPanel(
                paper_id=paper_id, figure_id=fig_id,
                panel_id=label, species=species,
            ))
    return gold


def _build_gold_bandini(paper_id: str) -> list[GoldPanel]:
    """Bandini 2011: caption format "Fig. 1 Caneta (?) sp. Fig. 2 Archaeodictyomitra...".

    Uses standard "Fig. N Species" pattern; the existing parser handles it.
    """
    gold: list[GoldPanel] = []
    for fig_id, caption in _load_captions(paper_id):
        pairs = _regex_parse_caption(caption)
        for pair in pairs:
            for label in pair.labels:
                gold.append(GoldPanel(
                    paper_id=paper_id, figure_id=fig_id,
                    panel_id=label, species=pair.species,
                ))
    return gold


# Hand-curated pouille gold. The caption is "figs 1–4. Species; figs 5–7. Species"
# — neither standard Fig-prefix regex (caps go before the species) nor
# Pouille (Pl. N, fig M) matches it cleanly. So we hand-code it.
_POUILLE_GOLD_RAW = """
figs 1–4. Syntagentactinia? sp. cf. S. excelsa
figs 5–7. Syntagentactinia biocculosa
figs 8–11. Syntagentactinia? sp. aff. S. excelsa
figs 12–14b. Syntagentactinia? angulata
figs 15–18. Syntagentactinia? sp. cf. S. biocculosa
fig 19. Syntagentactinia? sp. A
""".strip().splitlines()


def _parse_pouille_clause(line: str) -> list[tuple[str, str]]:
    """Parse a pouille-style 'figs 1-4. Species' clause into [(label, species)]."""
    m = re.match(r"^figs?\s+(\d+)(?:[a-z\-–—]+)?(?:[,\s\-–—]+\d+[a-z\-–—]*)*\.\s+(.*)$", line)
    if not m:
        return []
    species = m.group(2).strip()
    # Labels: extract just the leading number (the rest of the label list is
    # in the regex; for gold we only need the canonical "first" number).
    return [(m.group(1), species)]


def _build_gold_pouille(paper_id: str) -> list[GoldPanel]:
    # Read the actual paper manifests to find the pouille plate figure_id
    figure_ids: list[str] = []
    for manifest_path in sorted(PAPER_MANIFESTS[paper_id].glob("*.json")):
        import json
        with open(manifest_path) as f:
            d = json.load(f)
        # The pouille plate 1 manifest is the one with reconstructed caption
        # containing "biocculosa". We accept any figure on the first
        # plate-bearing page; for batch4_v2 pouille has 3 figure manifests
        # in the test set, all sharing the same caption.
        figure_ids.append(d["figure_id"])
    if not figure_ids:
        return []
    fig_id = figure_ids[0]
    gold: list[GoldPanel] = []
    for line in _POUILLE_GOLD_RAW:
        parsed = _parse_pouille_clause(line)
        for label, species in parsed:
            gold.append(GoldPanel(
                paper_id=paper_id, figure_id=fig_id,
                panel_id=label, species=species,
            ))
    return gold


BUILDERS = {
    "900c13a3f2473740": _build_gold_danelian,
    "3d554d642954c720": _build_gold_hollis,
    "0af2fd3865413764": _build_gold_bandini,
    "cb2011ef7be94959": _build_gold_pouille,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build gold sets from full captions")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "data" / "gold",
                        help="Output directory for *.jsonl gold files")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for paper_id, builder in BUILDERS.items():
        short = PAPER_SHORT_NAMES[paper_id]
        out_path = args.out_dir / f"{short}.jsonl"
        gold = builder(paper_id)
        n = write_gold(gold, out_path)
        print(f"{short}: {n} gold panels → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
