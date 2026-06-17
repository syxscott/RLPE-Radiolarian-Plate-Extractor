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
        "rank": "epoch",
        "parent": "Quaternary",
        "ma_top": 0.0117,
        "ma_base": 2.58,
    },
    {
        "name": "Holocene",
        "cn": "全新世",
        "rank": "epoch",
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MODIFIER_PATTERN = re.compile(
    r"^\s*(Early|Middle|Late|Lower|Middle|Upper|E\.|M\.|L\.|上|中|下)\s+",
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
    if rank == "period":
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
    )


# ---------------------------------------------------------------------------
# PBDB fallback
# ---------------------------------------------------------------------------


_PBDB_INTERVALS_CACHE: list[dict[str, Any]] | None = None
_PBDB_LAST_FETCH: float = 0.0


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
    if cache_path.exists() and not force:
        try:
            _PBDB_INTERVALS_CACHE = json.loads(cache_path.read_text(encoding="utf-8"))
            _PBDB_LAST_FETCH = cache_path.stat().st_mtime
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
            rank = (rec.get("tpb") or "").lower()  # eon/era/period/epoch/age
            parent = rec.get("par")
            parent_name = None
            if parent:
                for r in intervals:
                    if r.get("oid") == parent:
                        parent_name = r.get("nam")
                        break
            return {
                "name": rec.get("nam"),
                "cn": "",
                "rank": rank if rank in {"eon", "era", "period", "epoch", "age"} else "age",
                "parent": parent_name,
                "ma_top": rec.get("eag"),
                "ma_base": rec.get("lag"),
            }
    return None


# ---------------------------------------------------------------------------
# High-level convenience: classify_all_in_text
# ---------------------------------------------------------------------------


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
            # Use a non-boundary search — age names are unique enough that
            # substring collisions are rare. We add a small negative lookahead
            # to prevent matching a longer superset of a known name (e.g.
            # don't match "Permian" inside "Permianian").
            m = re.search(rf"(?:{mod})?{re.escape(variant)}", text, re.IGNORECASE)
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
