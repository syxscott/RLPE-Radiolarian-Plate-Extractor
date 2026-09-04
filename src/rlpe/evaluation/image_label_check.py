"""Image-label sanity check: re-OCR each prediction's panel image and
compare to the predicted panel_id.

Background
----------
The standard string-match eval (``scripts/evaluate.py``) compares
predictions to gold on ``(paper_id, figure_id, panel_id, species)`` and
*cannot* tell whether the predicted panel_id matches what is actually
visible in the panel image. This was the source of the N10 bug:
predictions had a high string-match F1 (98.19%) but visually the
panel_id was wrong for ~87% of panels (positional fallback, not image
OCR).

This module reads each prediction row, locates its panel image, OCRs
it, and compares the OCR'd label to the predicted panel_id. It
reports:

  - n_checked           : predictions with a resolvable panel image
  - n_ocr_has_label     : OCR returned ≥ 1 numeric token
  - n_image_label_match : pred panel_id == first numeric OCR token
  - image_label_match_rate : n_image_label_match / n_checked

The function is opt-in via the ``--image-label-check`` flag on
``scripts/evaluate.py``; it adds ~5-15 min on a 9-paper corpus because
EasyOCR runs on every panel image. Pass ``--image-label-cache`` to
reuse OCR results across runs — the cache key is the panel image's
``(size, mtime)`` tuple, so it transparently invalidates when a panel
is regenerated.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# audit 2026-07-26 M13: accept an optional trailing letter so labels
# like "7a", "12b", "A1" are not silently dropped from the OCR-token
# filter (some plates use alphanumeric panel IDs).
_NUM_RE = re.compile(r"^\d{1,3}[A-Za-z]?$")


@dataclass(slots=True)
class ImageLabelStats:
    paper_id: str
    n_checked: int = 0
    n_ocr_has_label: int = 0
    n_image_label_match: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def image_label_match_rate(self) -> float:
        return self.n_image_label_match / max(1, self.n_checked)

    @property
    def ocr_coverage(self) -> float:
        return self.n_ocr_has_label / max(1, self.n_checked)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "n_checked": self.n_checked,
            "n_ocr_has_label": self.n_ocr_has_label,
            "n_image_label_match": self.n_image_label_match,
            "ocr_coverage": self.ocr_coverage,
            "image_label_match_rate": self.image_label_match_rate,
            "mismatches": self.mismatches[:20],
        }


def _resolve_panel_path(panel_path: str, root: Path) -> Path | None:
    """Locate the panel image. Mirrors the resolver in
    ``scripts/reassign_panel_id_v18.py``: if the relative path in the
    pred file no longer exists, search for the file by tail
    (``<paper_id>/<fig>/panel_NN.png``) under any ``work/*/panels/``
    or ``work/*/output/panels/`` directory.

    The latter pattern (with ``/output/``) is the layout used by
    refresh runs like ``work/beccaro_only_out/output/panels/...``;
    the former is the layout used by the v18 panel-id reassignment
    run. Pred files written by one run may be OCR'd against a
    different run's panels, so both layouts are checked.
    """
    if not panel_path:
        return None
    p = Path(panel_path)
    if p.exists() and p.is_file():
        return p
    parts = p.parts
    if len(parts) < 3 or "panel_" not in parts[-1]:
        return None
    tail = Path(*parts[-3:])
    # Audit 2026-09-01 CR-33: the previous implementation built the
    # glob pattern by string-formatting the unescaped paper_id /
    # figure_id components. A paper_id containing a ``*`` or ``[``
    # (extremely unusual, but possible for a synthetic / adversarial
    # PDF) would match unintended paths under ``work/`` — a glob
    # injection that could surface as an info-disclosure vector in
    # the web UI's ``/jobs/{id}/results`` endpoint. Escape the
    # user-controlled components with ``glob.escape`` so the only
    # ``*`` in the pattern is the literal ``work/*`` (a controlled
    # directory).
    import glob as _glob

    safe_paper = _glob.escape(tail.parent.parent.name)
    safe_fig = _glob.escape(tail.parent.name)
    safe_panel = _glob.escape(tail.name)
    candidates = list(root.glob(f"work/*/panels/{safe_paper}/{safe_fig}/{safe_panel}"))
    if not candidates:
        # Some runs (e.g. work/beccaro_only_out) put panels under an
        # extra ``output/`` segment. Try that layout too.
        candidates = list(root.glob(f"work/*/output/panels/{safe_paper}/{safe_fig}/{safe_panel}"))
    if not candidates:
        candidates = list(root.glob(f"work/*/panels/{safe_paper}/**/{safe_panel}"))
    if not candidates:
        candidates = list(root.glob(f"work/*/output/panels/{safe_paper}/**/{safe_panel}"))
    if not candidates:
        return None
    # Audit 2026-09-01 BL-32: ``root.glob`` returns matches in
    # filesystem-order, which on some filesystems is creation order
    # and on others is hash-of-name order — neither is stable across
    # runs. The image-label-check rate then jitters by 1-3 pp run to
    # run because different ``candidates[0]`` carries different mtime
    # / size / OCR results. Sort by mtime (most-recently-modified
    # first) so the selected candidate is deterministic and matches
    # the most recent run for that paper.
    try:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        # If stat() fails (broken symlink, race) fall back to path
        # order — better than crashing the whole eval.
        candidates.sort()
    return candidates[0]


def run_image_label_check(
    predictions: list[dict[str, Any]],
    root: Path,
    max_mismatches_per_paper: int = 20,
    reader=None,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Run the OCR-vs-prediction sanity check.

    ``reader`` is an optional pre-initialised EasyOCR reader; if None,
    a CPU EasyOCR reader is lazily created. Pass a reader from outside
    if you want to amortise init across multiple calls (e.g. test
    suite).

    ``cache_path``: optional on-disk cache of OCR results. When set,
    results are loaded on entry and saved on exit. The cache key is
    the panel image's ``(size, mtime_ns)`` tuple, so it transparently
    invalidates when a panel is regenerated (e.g. after a pipeline
    run with different settings). A second run with unchanged panels
    is essentially free — the OCR step is skipped entirely for every
    cached panel.
    """
    if reader is None:
        import easyocr

        # P4-1 fix: add Latin (taxonomic terms), German, French for European papers.
        reader = easyocr.Reader(["en", "la", "de", "fr"], gpu=False)
    import numpy as np
    from PIL import Image

    # Load the OCR cache. Schema: {panel_path_str: [size, mtime_ns, [numeric_tokens]]}.
    # The size+mtime pair is the cache key (cheap to compute, ~zero
    # false negatives for unchanged files); the numeric_tokens list is
    # the OCR output we want to skip on the next run.
    cache: dict[str, list[Any]] = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}

    by_paper: dict[str, ImageLabelStats] = defaultdict(lambda: ImageLabelStats(paper_id=""))
    n_cache_hits = 0
    for p in predictions:
        pid = p.get("paper_id") or "unknown"
        st = by_paper[pid]
        st.paper_id = pid
        panel_path = p.get("panel_path", "")
        resolved = _resolve_panel_path(panel_path, root)
        if resolved is None:
            continue
        st.n_checked += 1
        # Cache lookup. Keyed on the original (un-resolved) panel_path
        # because that is what callers would regenerate with; the
        # size+mtime on the resolved file is what actually detects
        # staleness (resolves can be re-glob'd across runs).
        cache_key = panel_path or str(resolved)
        cached_entry = cache.get(cache_key)
        nums: list[str] | None = None
        try:
            stat = resolved.stat()
            current_meta = [stat.st_size, stat.st_mtime_ns]
        except OSError:
            current_meta = None
        if (
            cached_entry is not None
            and current_meta is not None
            and cached_entry[0] == current_meta[0]
            and cached_entry[1] == current_meta[1]
        ):
            nums = cached_entry[2]
            n_cache_hits += 1
        if nums is None:
            try:
                arr = np.array(Image.open(str(resolved)).convert("RGB"))
            except Exception:
                continue
            try:
                _ocr_tokens = reader.readtext(arr)
            except Exception:
                continue
            # Audit 2026-09-01 BL-33: keep the full token list (incl.
            # tokens without a leading digit) so we can identify
            # scale-bar candidates later in the loop. The previous
            # code dropped them and then had no way to tell
            # "100µm" apart from "100".
            tokens = [t[1].strip() for t in _ocr_tokens]
            nums = [t for t in tokens if _NUM_RE.match(t)]
            if current_meta is not None:
                cache[cache_key] = current_meta + [nums]
        if not nums:
            continue
        st.n_ocr_has_label += 1
        # Audit 2026-09-01 BL-33: the previous code compared the
        # OCR's *first* numeric token against the pred panel_id, but
        # the first number in a panel's image is almost always the
        # scale-bar magnitude ("100", "50 µm", "10 mm") rather than
        # the panel label. On wever2006 the rate therefore hovered
        # around 8-12 % — far below reality — and the sweep reports
        # said "image_label_match_rate is unreliable" without
        # identifying the root cause.
        #
        # Strategy: pick the *last* numeric token (panel labels are
        # conventionally the right-most number in OCR left-to-right
        # layouts). If multiple numbers are present AND the token list
        # has a scale-bar unit ("µm"/"mm"/"cm"), drop the unit and use
        # the next-to-last number. This handles both single-number
        # panels ("3") and dual-number panels ("100µm 3").
        if (
            "tokens" in locals()
            and len(tokens) >= 2
            and any(u in " ".join(tokens).lower() for u in ("µm", "mm", "cm", "μm"))
        ):
            # Walk back from the right; the first numeric token that
            # is NOT followed by a unit glyph is the panel label.
            unit_set = ("µm", "mm", "cm", "μm")
            last_unit_idx = max(
                (i for i, tok in enumerate(tokens) if any(u in tok.lower() for u in unit_set)),
                default=-1,
            )
            # Find the first pure-numeric token to the LEFT of the
            # scale-bar position.
            candidate_nums = [
                int(tok.rstrip(".,:;")) for tok in tokens[:last_unit_idx] if _NUM_RE.match(tok)
            ]
            if candidate_nums:
                first = candidate_nums[-1]
            else:
                # Scale-bar only, no panel label found — skip rather
                # than count a false positive.
                continue
        elif len(nums) >= 2:
            # No scale-bar glyphs but ≥2 numbers: pick the LAST one.
            first = int(nums[-1].rstrip(".,:;"))
        else:
            first = int(nums[0].rstrip(".,:;"))
        pred_label = (p.get("panel_id") or "").strip()
        if first == pred_label:
            st.n_image_label_match += 1
        elif len(st.mismatches) < max_mismatches_per_paper:
            st.mismatches.append(
                {
                    "figure_id": p.get("figure_id"),
                    "panel_path": str(resolved.relative_to(root))
                    if resolved.is_relative_to(root)
                    else str(resolved),
                    "pred_panel_id": pred_label,
                    "ocr_label": first,
                    "all_ocr_numbers": nums,
                }
            )

    total_checked = sum(s.n_checked for s in by_paper.values())
    total_has_label = sum(s.n_ocr_has_label for s in by_paper.values())
    total_match = sum(s.n_image_label_match for s in by_paper.values())
    # Persist the cache for the next run. Skip the write if nothing was
    # actually computed (an all-hit run still wants a write — the cache
    # was loaded and is the same dict reference, so this is cheap).
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            # Don't fail the eval just because we couldn't persist the cache.
            import logging

            logging.getLogger(__name__).warning(
                "Failed to write image-label cache to %s: %s",
                cache_path,
                exc,
            )
    return {
        "papers": {k: v.to_dict() for k, v in by_paper.items()},
        "aggregate": {
            "n_checked": total_checked,
            "n_ocr_has_label": total_has_label,
            "n_image_label_match": total_match,
            "ocr_coverage": total_has_label / max(1, total_checked),
            "image_label_match_rate": total_match / max(1, total_checked),
            "n_cache_hits": n_cache_hits,
        },
    }
