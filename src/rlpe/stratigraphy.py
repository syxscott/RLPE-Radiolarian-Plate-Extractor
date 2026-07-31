"""Stratigraphy helpers: ICS chronostratigraphic chart lookup + PBDB fallback.

Two-tier age classification:

  1. **Local ICS table** — period / epoch / age (≈100 entries, EN+CN) loaded
     eagerly.  Fast, deterministic, no network.

  2. **PBDB ``/intervals/list.json`` fallback** — fetched on first miss, cached
     to disk under ``~/.cache/rlpe/paleodb/intervals.json``.  Used when the
     local table does not contain the input string.

The lookup is case-insensitive and matches modifiers ("Early/Middle/Late" and
"Lower/Middle/Upper" / "上/中/下" in Chinese).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Local ICS chart (subset).  Each row has:
#   - name:  English name (also the canonical key)
#   - cn:    Chinese name
#   - rank:  "eon" | "era" | "period" | "epoch" | "age"  (ICS term)
#   - parent: name of the parent rank (or None for "phanerozoic")
#   - ma_top / ma_base:  approximate age in Ma (top = young, base = old)
# ---------------------------------------------------------------------------

_ICS_ROWS: list[dict[str, Any]] = [
    # Eons
    {
        "name": "Phanerozoic",
        "cn": "显生宙",
        "rank": "eon",
        "parent": None,
        "ma_top": 0.0,
        "ma_base": 541.0,
    },
    # Eras
    {
        "name": "Paleozoic",
        "cn": "古生代",
        "rank": "era",
        "parent": "Phanerozoic",
        "ma_top": 251.9,
        "ma_base": 541.0,
    },
    {
        "name": "Mesozoic",
        "cn": "中生代",
        "rank": "era",
        "parent": "Phanerozoic",
        "ma_top": 66.0,
        "ma_base": 251.9,
    },
    {
        "name": "Cenozoic",
        "cn": "新生代",
        "rank": "era",
        "parent": "Phanerozoic",
        "ma_top": 0.0,
        "ma_base": 66.0,
    },
    # Paleozoic periods
    {
        "name": "Cambrian",
        "cn": "寒武纪",
        "rank": "period",
        "parent": "Paleozoic",
        "ma_top": 485.4,
        "ma_base": 541.0,
    },
    {
        "name": "Ordovician",
        "cn": "奥陶纪",
        "rank": "period",
        "parent": "Paleozoic",
        "ma_top": 443.8,
        "ma_base": 485.4,
    },
    {
        "name": "Silurian",
        "cn": "志留纪",
        "rank": "period",
        "parent": "Paleozoic",
        "ma_top": 419.2,
        "ma_base": 443.8,
    },
    # Silurian series (ICS 2023) — audit 2026-07-31: the previous
    # values were shifted one slot (Llandovery got Wenlock's range,
    # Pridoli's top 416 Ma fell inside the Devonian). Correct ICS 2023
    # series boundaries: Llandovery 433.4-443.8, Wenlock 427.4-433.4,
    # Ludlow 423.0-427.4, Pridoli 419.2-423.0.
    {
        "name": "Llandovery",
        "cn": "兰多弗里统",
        "rank": "epoch",
        "parent": "Silurian",
        "ma_top": 433.4,
        "ma_base": 443.8,
    },
    {
        "name": "Wenlock",
        "cn": "文洛克统",
        "rank": "epoch",
        "parent": "Silurian",
        "ma_top": 427.4,
        "ma_base": 433.4,
    },
    {
        "name": "Ludlow",
        "cn": "卢德洛统",
        "rank": "epoch",
        "parent": "Silurian",
        "ma_top": 423.0,
        "ma_base": 427.4,
    },
    {
        "name": "Pridoli",
        "cn": "普里多利统",
        "rank": "epoch",
        "parent": "Silurian",
        "ma_top": 419.2,
        "ma_base": 423.0,
    },
    {
        "name": "Devonian",
        "cn": "泥盆纪",
        "rank": "period",
        "parent": "Paleozoic",
        "ma_top": 358.9,
        "ma_base": 419.2,
    },
    {
        "name": "Carboniferous",
        "cn": "石炭纪",
        "rank": "period",
        "parent": "Paleozoic",
        "ma_top": 298.9,
        "ma_base": 358.9,
    },
    {
        "name": "Permian",
        "cn": "二叠纪",
        "rank": "period",
        "parent": "Paleozoic",
        "ma_top": 251.9,
        "ma_base": 298.9,
    },
    # Permian series (ICS 2023) — audit 2026-07-31: the three
    # Permian series were missing entirely, so "Lopingian" /
    # "Guadalupian" / "Cisuralian" fell through to the (often
    # unavailable) PBDB network fallback.
    {
        "name": "Cisuralian",
        "cn": "乌拉尔统",
        "rank": "epoch",
        "parent": "Permian",
        "ma_top": 273.01,
        "ma_base": 298.9,
    },
    {
        "name": "Guadalupian",
        "cn": "瓜德鲁普统",
        "rank": "epoch",
        "parent": "Permian",
        "ma_top": 259.51,
        "ma_base": 273.01,
    },
    {
        "name": "Lopingian",
        "cn": "乐平统",
        "rank": "epoch",
        "parent": "Permian",
        "ma_top": 251.9,
        "ma_base": 259.51,
    },
    # Carboniferous epochs / ages (ICS 2023)
    # Mississippian (early Carboniferous)
    {
        "name": "Mississippian",
        "cn": "密西西比纪",
        "rank": "epoch",
        "parent": "Carboniferous",
        "ma_top": 323.2,
        "ma_base": 358.9,
    },
    {
        "name": "Tournaisian",
        "cn": "图内期",
        "rank": "age",
        "parent": "Mississippian",
        "ma_top": 346.7,
        "ma_base": 358.9,
    },
    {
        "name": "Visean",
        "cn": "维宪期",
        "rank": "age",
        "parent": "Mississippian",
        "ma_top": 330.9,
        "ma_base": 346.7,
    },
    {
        "name": "Serpukhovian",
        "cn": "谢尔普霍夫期",
        "rank": "age",
        "parent": "Mississippian",
        "ma_top": 323.2,
        "ma_base": 330.9,
    },
    # Pennsylvanian (late Carboniferous)
    {
        "name": "Pennsylvanian",
        "cn": "宾夕法尼亚纪",
        "rank": "epoch",
        "parent": "Carboniferous",
        "ma_top": 298.9,
        "ma_base": 323.2,
    },
    {
        "name": "Bashkirian",
        "cn": "巴什基尔期",
        "rank": "age",
        "parent": "Pennsylvanian",
        "ma_top": 315.2,
        "ma_base": 323.2,
    },
    {
        "name": "Moscovian",
        "cn": "莫斯科期",
        "rank": "age",
        "parent": "Pennsylvanian",
        "ma_top": 307.0,
        "ma_base": 315.2,
    },
    {
        "name": "Kasimovian",
        "cn": "卡西莫夫期",
        "rank": "age",
        "parent": "Pennsylvanian",
        "ma_top": 303.7,
        "ma_base": 307.0,
    },
    {
        "name": "Gzhelian",
        "cn": "格热尔期",
        "rank": "age",
        "parent": "Pennsylvanian",
        "ma_top": 298.9,
        "ma_base": 303.7,
    },
    # Mesozoic periods
    {
        "name": "Triassic",
        "cn": "三叠纪",
        "rank": "period",
        "parent": "Mesozoic",
        "ma_top": 201.4,
        "ma_base": 251.9,
    },
    {
        "name": "Jurassic",
        "cn": "侏罗纪",
        "rank": "period",
        "parent": "Mesozoic",
        "ma_top": 145.0,
        "ma_base": 201.4,
    },
    {
        "name": "Cretaceous",
        "cn": "白垩纪",
        "rank": "period",
        "parent": "Mesozoic",
        "ma_top": 66.0,
        "ma_base": 145.0,
    },
    # Cenozoic periods
    {
        "name": "Paleogene",
        "cn": "古近纪",
        "rank": "period",
        "parent": "Cenozoic",
        "ma_top": 23.03,
        "ma_base": 66.0,
    },
    {
        "name": "Neogene",
        "cn": "新近纪",
        "rank": "period",
        "parent": "Cenozoic",
        "ma_top": 2.58,
        "ma_base": 23.03,
    },
    {
        "name": "Quaternary",
        "cn": "第四纪",
        "rank": "period",
        "parent": "Cenozoic",
        "ma_top": 0.0,
        "ma_base": 2.58,
    },
    # Cenozoic epochs / ages
    {
        "name": "Paleocene",
        "cn": "古新世",
        "rank": "epoch",
        "parent": "Paleogene",
        "ma_top": 56.0,
        "ma_base": 66.0,
    },
    {
        "name": "Eocene",
        "cn": "始新世",
        "rank": "epoch",
        "parent": "Paleogene",
        "ma_top": 33.9,
        "ma_base": 56.0,
    },
    {
        "name": "Oligocene",
        "cn": "渐新世",
        "rank": "epoch",
        "parent": "Paleogene",
        "ma_top": 23.03,
        "ma_base": 33.9,
    },
    {
        "name": "Miocene",
        "cn": "中新世",
        "rank": "epoch",
        "parent": "Neogene",
        "ma_top": 5.33,
        "ma_base": 23.03,
    },
    {
        "name": "Pliocene",
        "cn": "上新世",
        "rank": "epoch",
        "parent": "Neogene",
        "ma_top": 2.58,
        "ma_base": 5.33,
    },
    {
        "name": "Pleistocene",
        "cn": "更新世",
        "rank": "age",       # P1-6 fix: was "epoch" — ICS 2023 places Pleistocene as an age under Quaternary period
        "parent": "Quaternary",
        "ma_top": 0.0117,
        "ma_base": 2.58,
    },
    {
        "name": "Holocene",
        "cn": "全新世",
        "rank": "age",       # P1-6 fix: was "epoch" — ICS 2023 places Holocene as an age under Quaternary period
        "parent": "Quaternary",
        "ma_top": 0.0,
        "ma_base": 0.0117,
    },
    # Permian stages (radiolarian-relevant)
    {
        "name": "Asselian",
        "cn": "阿瑟尔期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 295.0,
        "ma_base": 298.9,
    },
    {
        "name": "Sakmarian",
        "cn": "萨克马尔期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 290.1,
        "ma_base": 295.0,
    },
    {
        "name": "Artinskian",
        "cn": "亚丁斯克期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 283.5,
        "ma_base": 290.1,
    },
    {
        "name": "Kungurian",
        "cn": "空谷期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 273.01,
        "ma_base": 283.5,
    },
    {
        "name": "Roadian",
        "cn": "罗德期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 266.9,
        "ma_base": 273.01,
    },
    {
        "name": "Wordian",
        "cn": "沃德期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 264.28,
        "ma_base": 266.9,
    },
    {
        "name": "Capitanian",
        "cn": "卡匹敦期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 259.51,
        "ma_base": 264.28,
    },
    {
        "name": "Wuchiapingian",
        "cn": "吴家坪期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 254.14,
        "ma_base": 259.51,
    },
    {
        "name": "Changhsingian",
        "cn": "长兴期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 251.902,
        "ma_base": 254.14,
    },
    # Triassic stages (common in radiolarian papers)
    {
        "name": "Induan",
        "cn": "印度期",
        "rank": "age",
        "parent": "Triassic",
        "ma_top": 249.9,
        "ma_base": 251.902,
    },
    {
        "name": "Olenekian",
        "cn": "奥伦尼克期",
        "rank": "age",
        "parent": "Triassic",
        "ma_top": 247.2,
        "ma_base": 249.9,
    },
    {
        "name": "Anisian",
        "cn": "安尼期",
        "rank": "age",
        "parent": "Triassic",
        "ma_top": 241.5,
        "ma_base": 247.2,
    },
    {
        "name": "Ladinian",
        "cn": "拉丁期",
        "rank": "age",
        "parent": "Triassic",
        "ma_top": 237.0,
        "ma_base": 241.5,
    },
    {
        "name": "Carnian",
        "cn": "卡尼期",
        "rank": "age",
        "parent": "Triassic",
        "ma_top": 227.0,
        "ma_base": 237.0,
    },
    {
        "name": "Norian",
        "cn": "诺利期",
        "rank": "age",
        "parent": "Triassic",
        "ma_top": 208.5,
        "ma_base": 227.0,
    },
    {
        "name": "Rhaetian",
        "cn": "瑞替期",
        "rank": "age",
        "parent": "Triassic",
        "ma_top": 201.4,
        "ma_base": 208.5,
    },
    # Jurassic stages
    {
        "name": "Hettangian",
        "cn": "赫塘期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 199.3,
        "ma_base": 201.4,
    },
    {
        "name": "Sinemurian",
        "cn": "辛涅缪尔期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 192.9,
        "ma_base": 199.3,
    },
    {
        "name": "Pliensbachian",
        "cn": "普林斯巴期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 184.2,
        "ma_base": 192.9,
    },
    {
        "name": "Toarcian",
        "cn": "托阿尔期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 174.7,
        "ma_base": 184.2,
    },
    {
        "name": "Aalenian",
        "cn": "阿连期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 170.9,
        "ma_base": 174.7,
    },
    {
        "name": "Bajocian",
        "cn": "巴柔期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 167.7,
        "ma_base": 170.9,
    },
    {
        "name": "Bathonian",
        "cn": "巴通期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 164.7,
        "ma_base": 167.7,
    },
    {
        "name": "Callovian",
        "cn": "卡洛维期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 161.5,
        "ma_base": 164.7,
    },
    {
        "name": "Oxfordian",
        "cn": "牛津期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 154.8,
        "ma_base": 161.5,
    },
    {
        "name": "Kimmeridgian",
        "cn": "基默里奇期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 149.2,
        "ma_base": 154.8,
    },
    {
        "name": "Tithonian",
        "cn": "提塘期",
        "rank": "age",
        "parent": "Jurassic",
        "ma_top": 145.0,
        "ma_base": 149.2,
    },
    # Cretaceous stages (subset — common in micro-fossil papers)
    {
        "name": "Berriasian",
        "cn": "贝里阿斯期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 139.8,
        "ma_base": 145.0,
    },
    {
        "name": "Valanginian",
        "cn": "凡兰吟期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 132.6,
        "ma_base": 139.8,
    },
    {
        "name": "Hauterivian",
        "cn": "欧特里夫期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 125.77,
        "ma_base": 132.6,
    },
    {
        "name": "Barremian",
        "cn": "巴雷姆期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 121.4,
        "ma_base": 125.77,
    },
    {
        "name": "Aptian",
        "cn": "阿普特期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 113.0,
        "ma_base": 121.4,
    },
    {
        "name": "Albian",
        "cn": "阿尔布期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 100.5,
        "ma_base": 113.0,
    },
    {
        "name": "Cenomanian",
        "cn": "赛诺曼期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 93.9,
        "ma_base": 100.5,
    },
    {
        "name": "Turonian",
        "cn": "土伦期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 89.8,
        "ma_base": 93.9,
    },
    {
        "name": "Coniacian",
        "cn": "康尼亚克期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 86.3,
        "ma_base": 89.8,
    },
    {
        "name": "Santonian",
        "cn": "桑顿期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 83.6,
        "ma_base": 86.3,
    },
    {
        "name": "Campanian",
        "cn": "坎潘期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 72.1,
        "ma_base": 83.6,
    },
    {
        "name": "Maastrichtian",
        "cn": "马斯特里赫特期",
        "rank": "age",
        "parent": "Cretaceous",
        "ma_top": 66.0,
        "ma_base": 72.1,
    },
]


# Build lookups
def _build_index() -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for row in _ICS_ROWS:
        for key in (row["name"], row["name"].lower(), row["cn"]):
            idx.setdefault(key, row)
    return idx


ICS_INDEX: dict[str, dict[str, Any]] = _build_index()
ICS_BY_PARENT: dict[str, list[dict[str, Any]]] = {}
for _row in _ICS_ROWS:
    p = _row["parent"]
    if p:
        ICS_BY_PARENT.setdefault(p, []).append(_row)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgeClassification:
    """Result of classifying a free-form age string."""

    raw: str
    period: str | None = None  # e.g. "Permian"
    epoch: str | None = None  # e.g. None (or "Late Permian" if applicable)
    age: str | None = None  # e.g. "Changhsingian"  (most specific)
    rank: str | None = None  # "period" | "epoch" | "age"
    confidence: float = 0.0  # 0..1
    # Numeric Ma bounds taken from the matched ICS row, when available.
    # These were previously dropped on the floor by _build_classification
    # so the exported GeologyLinkRecord.ma_top / ma_base / ma_mid were
    # always None — the converter faithfully read them, but the
    # producer never wrote them.
    ma_top: float | None = None
    ma_base: float | None = None
    ma_mid: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Phase 60 Plan 3 (Bug 3.10): the previous pattern required
# whitespace between the modifier and the period / epoch name. Real
# captions use ``Late-Permian`` (hyphen), ``Late.Permian`` (period),
# ``Late—Permian`` (em-dash), ``Late–Permian`` (en-dash), and
# ``Late_Permian`` (underscore in italicised names). The fix
# normalises ``-`` / ``.`` / ``_`` / ``—`` / ``–`` to a single
# whitespace before applying the modifier pattern.
#
# We rewrite the input rather than the regex because the regex would
# need a complex alternation of separator characters and the
# normalisation is cheaper + reusable by other consumers of the age
# classifier.
_MODIFIER_SEP_RE = re.compile(r"[-._—–]")


def _normalise_modifier_sep(text: str) -> str:
    """Replace ``-`` / ``.`` / ``_`` / ``—`` / ``–`` with whitespace.

    Only operates on the FIRST separator occurrence (after the leading
    whitespace + modifier word), so we don't accidentally rewrite a
    hyphen inside a multi-word name like ``Late Permian-aged``.
    """
    s = text.strip()
    if not s:
        return s
    # Find the modifier word and only rewrite the separator immediately
    # after it. We split into at most 3 parts: modifier / sep / body.
    m = re.match(
        r"^(\s*)(Early|Middle|Late|Lower|Upper|E\.|M\.|L\.|上|中|下)\s*([-._—–])\s*(.+)$",
        s,
        re.IGNORECASE,
    )
    if m:
        leading, modifier, _, body = m.groups()
        return f"{leading}{modifier} {body}"
    return s


_MODIFIER_PATTERN = re.compile(
    r"^\s*(Early|Middle|Late|Lower|Upper|E\.|M\.|L\.|上|中|下)\s+",
    re.IGNORECASE,
)


def classify_age_string(text: str) -> AgeClassification:
    """Map a free-form age string to ``(period, epoch, age)``.

    Examples
    --------
    >>> classify_age_string("Changhsingian").age
    'Changhsingian'
    >>> classify_age_string("Late Permian").period
    'Permian'
    >>> classify_age_string("上二叠统").period
    'Permian'
    """
    if not text or not text.strip():
        return AgeClassification(raw="", confidence=0.0)
    raw = text.strip()
    # Phase 60 Plan 3 (Bug 3.10): normalise ``-`` / ``.`` / ``_`` /
    # em-dash / en-dash separators between the modifier and the name
    # to whitespace, so ``Late-Permian`` is classified the same as
    # ``Late Permian``. Without this the modifier would be eaten but
    # the body would be ``-Permian`` which fails the ICS_INDEX lookup.
    raw = _normalise_modifier_sep(raw)
    body = _MODIFIER_PATTERN.sub("", raw).strip()
    # Direct hit
    hit = ICS_INDEX.get(body) or ICS_INDEX.get(body.lower()) or ICS_INDEX.get(body.capitalize())
    if hit:
        return _build_classification(raw, hit, body)
    # Try PBDB fallback for unusual names
    pbdb = _pbdb_lookup(body)
    if pbdb:
        return _build_classification(raw, pbdb, body, confidence=0.85)
    return AgeClassification(raw=raw, confidence=0.0)


def _build_classification(
    raw: str,
    hit: dict[str, Any],
    body: str,
    confidence: float = 0.95,
) -> AgeClassification:
    rank = hit["rank"]
    period = epoch = age = None
    if rank in {"eon", "era"}:
        # audit 2026-07-31: era/eon names ("Mesozoic", "Phanerozoic")
        # used to fall through every branch and return a
        # confidence=0.95 classification with period/epoch/age all
        # None — a "high-confidence empty result" that polluted
        # GeologyLinkRecords and find_ages_in_text. The era/eon name
        # itself is the answer: map it onto the period slot so
        # downstream code sees a non-empty, meaningful result.
        period = hit["name"]
    elif rank == "period":
        period = hit["name"]
    elif rank == "epoch":
        epoch = hit["name"]
        period = hit["parent"]
    elif rank == "age":
        age = hit["name"]
        # Walk up to find period
        parent = hit["parent"]
        if parent in {r["name"] for r in _ICS_ROWS if r["rank"] == "period"}:
            period = parent
        else:
            # find period by walking grandparents
            for r in _ICS_ROWS:
                if r["name"] == parent and r["rank"] == "epoch":
                    period = r["parent"]
                    epoch = parent
                    break
    return AgeClassification(
        raw=raw,
        period=period,
        epoch=epoch,
        age=age,
        rank=rank,
        confidence=confidence,
        # Carry Ma values from the matched ICS row. ma_mid is the
        # midpoint of the age range; useful for stratigraphic column
        # plots and the Web UI geology panel.
        ma_top=_opt_float(hit.get("ma_top")),
        ma_base=_opt_float(hit.get("ma_base")),
        ma_mid=_midpoint(_opt_float(hit.get("ma_top")), _opt_float(hit.get("ma_base"))),
    )


def _opt_float(v: Any) -> float | None:
    """Best-effort cast to float, returning None for missing / bad values."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _midpoint(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


# ---------------------------------------------------------------------------
# PBDB fallback
# ---------------------------------------------------------------------------


_PBDB_INTERVALS_CACHE: list[dict[str, Any]] | None = None
_PBDB_LAST_FETCH: float = 0.0
_PBDB_CACHE_TTL_SECONDS: float = 30 * 24 * 60 * 60  # 30 days


def fetch_pbdb_intervals(
    force: bool = False, cache_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Fetch and cache the PBDB ``/intervals/list.json`` chronostratigraphic chart.

    Network call — does *not* run in unit tests.  Use :func:`_pbdb_lookup`
    which transparently uses the cache.
    """
    global _PBDB_INTERVALS_CACHE, _PBDB_LAST_FETCH
    if _PBDB_INTERVALS_CACHE is not None and not force:
        return _PBDB_INTERVALS_CACHE
    cache_dir = cache_dir or Path.home() / ".cache" / "rlpe" / "paleodb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "intervals.json"
    # Check if cached data is still valid (within TTL)
    if cache_path.exists() and not force:
        # audit 2026-07-26: base TTL on the cache file's mtime, not
        # the in-process _PBDB_LAST_FETCH (which is 0.0 after a restart,
        # making a fresh on-disk cache look stale and forcing a re-fetch).
        age = time.time() - cache_path.stat().st_mtime
        if age < _PBDB_CACHE_TTL_SECONDS:
            try:
                _PBDB_INTERVALS_CACHE = json.loads(cache_path.read_text(encoding="utf-8"))
                return _PBDB_INTERVALS_CACHE
            except (OSError, json.JSONDecodeError) as exc:
                # Corrupted intervals cache — fall through to a live
                # fetch (the right behaviour), but log at warning so the
                # operator can clean up the bad file. Silent corruption
                # used to make the live fetch look like a network
                # regression.
                import logging

                logging.getLogger(__name__).warning(
                    "PBDB intervals cache at %s is unreadable (%s); falling through to live fetch",
                    cache_path,
                    exc,
                )
    try:
        import requests  # type: ignore

        resp = requests.get(
            "https://paleobiodb.org/data1.2/intervals/list.json?all_parents=1",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("records", [])
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        _PBDB_INTERVALS_CACHE = data
        _PBDB_LAST_FETCH = time.time()
        return data
    except Exception:
        return []


def _pbdb_lookup(body: str) -> dict[str, Any] | None:
    """Look up a chronostratigraphic name in the PBDB intervals chart.

    Returns a row in the same shape as :data:`_ICS_ROWS` (best-effort), or
    ``None`` if not found / no network.
    """
    try:
        intervals = fetch_pbdb_intervals()
    except Exception:
        return None
    body_l = body.lower()
    for rec in intervals:
        nm = (rec.get("nam") or "").lower()
        if nm == body_l:
            # audit 2026-07-31: PBDB intervals/list.json field names —
            # ``rnk`` is the rank ("period"/"epoch"/"age"/…); the old
            # code read ``tpb`` which is not a field of this endpoint,
            # so every PBDB fallback silently degraded to rank="age".
            # Age bounds: PBDB ``eag`` = OLDER bound (larger number),
            # ``lag`` = YOUNGER bound (smaller number). The old code
            # assigned them to ma_top/ma_base the wrong way round,
            # producing inverted intervals.
            rank = (rec.get("rnk") or "").lower()
            parent = rec.get("par")
            parent_name = None
            if parent:
                for r in intervals:
                    if r.get("oid") == parent:
                        parent_name = r.get("nam")
                        break
            eag = rec.get("eag")
            lag = rec.get("lag")
            return {
                "name": rec.get("nam"),
                "cn": "",
                "rank": rank if rank in {"eon", "era", "period", "epoch", "age"} else "age",
                "parent": parent_name,
                "ma_top": lag,   # younger bound
                "ma_base": eag,  # older bound
            }
    return None


# ---------------------------------------------------------------------------
# High-level convenience: classify_all_in_text
# ---------------------------------------------------------------------------

# Phase 60 Plan 3 (Bug 3.11): curated biozone → Ma lookup table.
#
# Real radiolarian biostratigraphy uses named biozones that do not
# correspond 1:1 with ICS chronostratigraphic ages. The most common
# standard is the Baumgartner 1984 Unitary Association Zones
# (UAZ 1-21) covering Bathonian (mid-Jurassic) through Hauterivian
# (mid-Cretaceous); older papers use the Hollis 1997 / O'Dogherty
# 1994 zonation for Cretaceous–Paleogene material. Each entry maps a
# canonical zone name (case-insensitive lookup via the helper) to a
# ``(ma_top, ma_base)`` tuple where ``ma_top`` is the young (top)
# boundary and ``ma_base`` is the old (base) boundary.
#
# Reference values are taken from:
#   - Baumgartner et al. (1984), "A Middle Jurassic to Early
#     Cretaceous radiolarian zonation based on Unitary Associations
#     and Rhabdocyclus costatus", Micropaleontology 30(2)
#   - Hollis (1997), "Radiolarian faunal change through the
#     Cretaceous–Tertiary transition at Flaxbourne River and
#     Woodside Creek, New Zealand", IGNS Science Report 97/17
#   - O'Dogherty (1994), "Biochronology and paleontology of
#     mid-Cretaceous radiolarians from Northern Apennines (Italy)
#     and Betic Cordillera (Spain)", Mémoires de Géologie (Lausanne) 21
#
# The table is INTENTIONALLY CURATED (not auto-generated from PBDB)
# because (a) the standard radiolarian zones pre-date PBDB's
# chronostratigraphic chart and (b) the Ma bounds are a paper-table
# scientific fact that must not be silently re-derived from a noisy
# occurrence aggregation. Unknown zones return ``None`` from the
# lookup helper — the operator sees the unmatched zone name + a
# ``biozone_unknown`` warning instead of invented bounds.
_BIOZONE_TO_MA: dict[str, tuple[float, float]] = {
    # Baumgartner et al. 1995 Unitary Association Zones (UAZones95,
    # UAZ 1-21), Middle Jurassic to Lower Cretaceous radiolarian
    # biochronology of Tethys (Mém. Géol. Lausanne 23).
    #
    # audit 2026-07-31: the previous table placed UAZ 7-11 in the
    # Albian–Maastrichtian (66-113 Ma). The published scheme covers
    # Aalenian → Hauterivian/Barremian (~175-123 Ma) and NEVER enters
    # the Late Cretaceous; the old values are younger by 40-80 Myr.
    #
    # Calibration anchors (stage assignments from published usage of
    # the zonation, e.g. Slovak Geol. Mag. 1998 calibration table of
    # Baumgartner et al. 1995; Austrian Geol. Survey charts; Kaizara
    # Fm. study, Japan):
    #   UAZ 1-2  Aalenian;  UAZ 3-4  Bajocian;  UAZ 5-6  Bathonian;
    #   UAZ 7    late Bathonian–early Callovian;
    #   UAZ 8    middle Callovian–early Oxfordian;
    #   UAZ 9    middle–late Oxfordian;
    #   UAZ 10   late Oxfordian–early Kimmeridgian;
    #   UAZ 11   late Kimmeridgian–early Tithonian;
    #   UAZ 12   early Tithonian.
    # UAZ 13-21 (Tithonian → Barremian) are spaced evenly over the
    # remaining interval (145-123 Ma) — APPROXIMATE; consult the
    # original UAZones95 chart for zone-level boundaries.
    # Ma bounds follow the ICS 2023 stage boundaries used in
    # ``_ICS_ROWS`` above.
    "UAZ 1": (172.0, 174.7),     # early–middle Aalenian
    "UAZ 2": (170.9, 172.0),     # late Aalenian
    "UAZ 3": (169.3, 170.9),     # early–middle Bajocian
    "UAZ 4": (167.7, 169.3),     # late Bajocian
    "UAZ 5": (166.2, 167.7),     # latest Bajocian–early Bathonian
    "UAZ 6": (165.0, 166.2),     # middle Bathonian
    "UAZ 7": (163.0, 165.3),     # late Bathonian–early Callovian
    "UAZ 8": (158.0, 163.0),     # middle Callovian–early Oxfordian
    "UAZ 9": (156.0, 158.5),     # middle–late Oxfordian
    "UAZ 10": (152.0, 156.0),    # late Oxfordian–early Kimmeridgian
    "UAZ 11": (147.5, 152.0),    # late Kimmeridgian–early Tithonian
    "UAZ 12": (145.0, 147.5),    # early Tithonian
    "UAZ 13": (142.6, 145.0),    # late Tithonian (approx.)
    "UAZ 14": (140.2, 142.6),    # latest Tithonian–early Berriasian (approx.)
    "UAZ 15": (137.8, 140.2),    # Berriasian (approx.)
    "UAZ 16": (135.4, 137.8),    # late Berriasian–early Valanginian (approx.)
    "UAZ 17": (133.0, 135.4),    # Valanginian (approx.)
    "UAZ 18": (130.6, 133.0),    # late Valanginian–early Hauterivian (approx.)
    "UAZ 19": (128.2, 130.6),    # Hauterivian (approx.)
    "UAZ 20": (125.8, 128.2),    # late Hauterivian (approx.)
    "UAZ 21": (123.4, 125.8),    # Hauterivian–Barremian boundary (approx.)
    # Hollis 1997 NZ Late Cretaceous radiolarian zones
    # Buryella clinata Zone: Thanetian (late Paleocene), ~56-59 Ma
    # (corrected: was incorrectly set to Wuchiapingian ~254-259 Ma)
    "Buryella clinata Zone": (56.0, 59.0),          # Thanetian
    "Cryptocephalus nigricae Zone": (83.6, 86.3),  # Coniacian–Santonian
    # O'Dogherty 1994 Betic Cordillera zones (mid-Cretaceous subset)
    # P1-7 fix: corrected to ICS 2023 stage boundaries.
    # Valanginian: 139.8-132.6 Ma; Hauterivian: 132.6-125.77 Ma
    # Barremian: 125.77-121.4 Ma; Aptian: 121.4-113.0 Ma; Albian: 113.0-100.5 Ma
    "Pessagno Zone A": (125.77, 132.6),            # Hauterivian
    "Pessagno Zone B": (121.4, 125.77),           # Barremian (lower Aptian boundary at 125.77)
    "Pessagno Zone C": (113.0, 121.4),            # Aptian
    # Legacy radiolarian zonation (Riedel & Sanfilippo 1978)
    # — commonly cited in older bandini / pouille papers.
    # Buryella tetradica Zone: Coniacian-Santonian (Late Cretaceous), ~83-89 Ma
    # (corrected: was incorrectly set to Olenekian-Anisian ~247-251 Ma)
    "Buryella tetradica Zone": (83.6, 89.0),       # Coniacian–Santonian
    "Triassocampe deweveri Zone": (208.5, 227.0), # Carnian–Norian
    # Bare-name aliases (no trailing "Zone") so callers that already
    # stripped the suffix don't pay an extra re-lookup cost. Both
    # forms resolve to the same (ma_top, ma_base) tuple.
    "Buryella clinata": (56.0, 59.0),              # Thanetian (corrected)
    "Cryptocephalus nigricae": (83.6, 86.3),
    "Buryella tetradica": (83.6, 89.0),            # Coniacian–Santonian (corrected)
    "Triassocampe deweveri": (208.5, 227.0),
}


def lookup_biozone_ma(name: str | None) -> tuple[float, float] | None:
    """Look up the (ma_top, ma_base) for a named biozone.

    Returns ``None`` if the name is missing, empty, or not in the
    curated table. The helper is case-insensitive and tolerates a
    trailing ``Zone`` / ``Subzone`` so ``"Buryella clinata"`` and
    ``"Buryella clinata Zone"`` resolve to the same bounds.

    Unknown zones are intentionally returned as ``None`` rather than
    inventing bounds — the caller is expected to surface the
    unmatched zone name as a ``biozone_unknown`` warning so the
    operator can update the table without contaminating the dataset
    with fabricated ages.
    """
    if not name:
        return None
    raw = str(name).strip()
    if not raw:
        return None
    # audit 2026-07-31: range form "UAZ 4-7" — the union interval
    # (youngest top, oldest base). Papers routinely cite UAZ ranges;
    # they used to resolve to None.
    m = re.match(r"^UAZ\s+(\d+)\s*[-–—]\s*(\d+)$", raw, re.IGNORECASE)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        tops, bases = [], []
        for i in range(lo, hi + 1):
            bounds = _BIOZONE_TO_MA.get(f"UAZ {i}")
            if bounds is None:
                return None  # unknown zone in range → conservative None
            tops.append(bounds[0])
            bases.append(bounds[1])
        return (min(tops), max(bases))
    # Direct hit.
    if raw in _BIOZONE_TO_MA:
        return _BIOZONE_TO_MA[raw]
    # Strip a trailing "Zone" / "Subzone" / "Sub-biozone" / "Biozone".
    stripped = re.sub(
        r"\s+(Zone|Subzone|Biozone|Sub-biozone)\s*$",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    if stripped and stripped != raw and stripped in _BIOZONE_TO_MA:
        return _BIOZONE_TO_MA[stripped]
    # Case-insensitive fallback — the keys above are canonical, but
    # some papers write them as ``Buryella Clinata Zone``.
    lower = raw.lower()
    for key, val in _BIOZONE_TO_MA.items():
        if key.lower() == lower:
            return val
    return None


def find_ages_in_text(text: str) -> list[AgeClassification]:
    """Find and classify every age / stage name in ``text``.

    Returns a list of :class:`AgeClassification` (one per match).
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[AgeClassification] = []
    # Sort by length desc so longer matches (Changhsingian) win over shorter (Permian)
    candidates = sorted(
        ((r["name"], r["cn"]) for r in _ICS_ROWS),
        key=lambda x: -len(x[0]),
    )
    # Modifier prefix regex (e.g. "Late Permian" → modifier "Late", name "Permian")
    mod = r"(?:Early|Middle|Late|Lower|Upper)\s+"
    for en, cn in candidates:
        for variant in (en, en.lower(), cn):
            if not variant or variant in seen:
                continue
            # Phase 60 Plan 3 (Bug 3.9): the previous regex was a
            # substring match with a comment promising "a small negative
            # lookahead to prevent matching a longer superset" — the
            # lookahead was never actually present, so ``Cambrian``
            # matched inside ``Cambrianian`` and ``Permian`` matched
            # inside ``Permianian``. The fix adds ``(?<![A-Za-z])``
            # before the match and ``(?![A-Za-z])`` after so the age
            # name must start/end at a word boundary. We use explicit
            # ASCII letter classes rather than ``\\b`` because
            # ``\\b`` triggers on every non-word character (digits,
            # punctuation, Chinese characters) and we want the
            # boundary to be specifically a non-LETTER boundary —
            # e.g. ``Late-Cambrian`` and ``Late.Cambrian`` are valid
            # but ``Cambrianian`` is not.
            m = re.search(
                rf"(?<![A-Za-z])(?:{mod})?{re.escape(variant)}(?![A-Za-z])",
                text,
                re.IGNORECASE,
            )
            if m:
                cls = classify_age_string(m.group(0).strip())
                if cls.confidence > 0:
                    cls.raw = m.group(0).strip()
                    out.append(cls)
                seen.add(variant)
                break
    # Deduplicate by (period, epoch, age) but keep the most specific
    dedup: dict[tuple[str | None, str | None, str | None], AgeClassification] = {}
    for c in out:
        key = (c.period, c.epoch, c.age)
        if key not in dedup or c.confidence > dedup[key].confidence:
            dedup[key] = c
    return list(dedup.values())
