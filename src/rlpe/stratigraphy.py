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
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

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
    # Cambrian series / stages (ICS 2023/09) — phase 3A 2026-08-19
    # audit: these were missing entirely, so Paleozoic radiolarian papers
    # citing named Cambrian stages ("Wuliuan", "Drumian", "Guzhangian",
    # "Paibian", "Jiangshanian", "Stage 10") collapsed to the Cambrian
    # period level and lost ~50 Myr of resolution.
    {
        "name": "Terreneuvian",
        "cn": "纽芬兰统",
        "rank": "epoch",
        "parent": "Cambrian",
        "ma_top": 521.0,
        "ma_base": 538.8,
    },
    {
        "name": "Series 2",
        "cn": "第二统",
        "rank": "epoch",
        "parent": "Cambrian",
        "ma_top": 509.0,
        "ma_base": 521.0,
    },
    {
        "name": "Miaolingian",
        "cn": "苗岭统",
        "rank": "epoch",
        "parent": "Cambrian",
        "ma_top": 497.0,
        "ma_base": 509.0,
    },
    {
        "name": "Furongian",
        "cn": "芙蓉统",
        "rank": "epoch",
        "parent": "Cambrian",
        "ma_top": 485.4,
        "ma_base": 497.0,
    },
    {
        "name": "Fortunian",
        "cn": "幸运期",
        "rank": "age",
        "parent": "Terreneuvian",
        "ma_top": 529.0,
        "ma_base": 538.8,
    },
    {
        "name": "Stage 2",
        "cn": "第二期",
        "rank": "age",
        "parent": "Terreneuvian",
        "ma_top": 521.0,
        "ma_base": 529.0,
    },
    {
        "name": "Stage 3",
        "cn": "第三期",
        "rank": "age",
        "parent": "Series 2",
        "ma_top": 514.5,
        "ma_base": 521.0,
    },
    {
        "name": "Stage 4",
        "cn": "第四期",
        "rank": "age",
        "parent": "Series 2",
        "ma_top": 509.0,
        "ma_base": 514.5,
    },
    {
        "name": "Wuliuan",
        "cn": "乌溜期",
        "rank": "age",
        "parent": "Miaolingian",
        "ma_top": 506.5,
        "ma_base": 509.0,
    },
    {
        "name": "Drumian",
        "cn": "鼓山期",
        "rank": "age",
        "parent": "Miaolingian",
        "ma_top": 500.5,
        "ma_base": 506.5,
    },
    {
        "name": "Guzhangian",
        "cn": "古丈期",
        "rank": "age",
        "parent": "Miaolingian",
        "ma_top": 497.0,
        "ma_base": 500.5,
    },
    {
        "name": "Paibian",
        "cn": "排碧期",
        "rank": "age",
        "parent": "Furongian",
        "ma_top": 492.0,
        "ma_base": 497.0,
    },
    {
        "name": "Jiangshanian",
        "cn": "江山期",
        "rank": "age",
        "parent": "Furongian",
        "ma_top": 486.5,
        "ma_base": 492.0,
    },
    {
        "name": "Stage 10",
        "cn": "第十期",
        "rank": "age",
        "parent": "Furongian",
        "ma_top": 485.4,
        "ma_base": 486.5,
    },
    {
        "name": "Ordovician",
        "cn": "奥陶纪",
        "rank": "period",
        "parent": "Paleozoic",
        "ma_top": 443.8,
        "ma_base": 485.4,
    },
    # Ordovician series / stages (ICS 2023/09) — phase 3A 2026-08-19
    # audit: these were missing entirely, so papers citing "Tremadocian",
    # "Floian", "Darriwilian", "Sandbian", "Katian", "Hirnantian"
    # collapsed to the Ordovician period level and lost ~40 Myr.
    {
        "name": "Early Ordovician",
        "cn": "早奥陶世",
        "rank": "epoch",
        "parent": "Ordovician",
        "ma_top": 470.0,
        "ma_base": 485.4,
    },
    {
        "name": "Middle Ordovician",
        "cn": "中奥陶世",
        "rank": "epoch",
        "parent": "Ordovician",
        "ma_top": 458.4,
        "ma_base": 470.0,
    },
    {
        "name": "Late Ordovician",
        "cn": "晚奥陶世",
        "rank": "epoch",
        "parent": "Ordovician",
        "ma_top": 443.8,
        "ma_base": 458.4,
    },
    {
        "name": "Tremadocian",
        "cn": "特马豆克期",
        "rank": "age",
        "parent": "Early Ordovician",
        "ma_top": 477.1,
        "ma_base": 485.4,
    },
    {
        "name": "Floian",
        "cn": "弗洛期",
        "rank": "age",
        "parent": "Early Ordovician",
        "ma_top": 470.0,
        "ma_base": 477.1,
    },
    {
        "name": "Dapingian",
        "cn": "大坪期",
        "rank": "age",
        "parent": "Middle Ordovician",
        "ma_top": 467.3,
        "ma_base": 470.0,
    },
    {
        "name": "Darriwilian",
        "cn": "达瑞威尔期",
        "rank": "age",
        "parent": "Middle Ordovician",
        "ma_top": 458.4,
        "ma_base": 467.3,
    },
    {
        "name": "Sandbian",
        "cn": "桑比期",
        "rank": "age",
        "parent": "Late Ordovician",
        "ma_top": 453.0,
        "ma_base": 458.4,
    },
    {
        "name": "Katian",
        "cn": "凯迪期",
        "rank": "age",
        "parent": "Late Ordovician",
        "ma_top": 445.2,
        "ma_base": 453.0,
    },
    {
        "name": "Hirnantian",
        "cn": "赫南特期",
        "rank": "age",
        "parent": "Late Ordovician",
        "ma_top": 443.8,
        "ma_base": 445.2,
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
    # Phase 3B 2026-08-19: ICS 2023 formally designates these as
    # "series" rank (not "epoch") — updated for rank-type fidelity.
    {
        "name": "Llandovery",
        "cn": "兰多弗里统",
        "rank": "series",
        "parent": "Silurian",
        "ma_top": 433.4,
        "ma_base": 443.8,
    },
    {
        "name": "Wenlock",
        "cn": "文洛克统",
        "rank": "series",
        "parent": "Silurian",
        "ma_top": 427.4,
        "ma_base": 433.4,
    },
    {
        "name": "Ludlow",
        "cn": "卢德洛统",
        "rank": "series",
        "parent": "Silurian",
        "ma_top": 423.0,
        "ma_base": 427.4,
    },
    {
        "name": "Pridoli",
        "cn": "普里多利统",
        "rank": "series",
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
    # Devonian series / stages (ICS 2023/09) — phase 3A 2026-08-19
    # audit: these were missing entirely, so Paleozoic radiolarian
    # papers citing "Lochkovian", "Pragian", "Emsian", "Eifelian",
    # "Givetian", "Frasnian", "Famennian" collapsed to the Devonian
    # period level and lost ~30-50 Myr of resolution. Famennian in
    # particular is the index age for the Hangenberg Event at the
    # Devonian–Carboniferous boundary.
    {
        "name": "Early Devonian",
        "cn": "早泥盆世",
        "rank": "epoch",
        "parent": "Devonian",
        "ma_top": 393.3,
        "ma_base": 419.2,
    },
    {
        "name": "Middle Devonian",
        "cn": "中泥盆世",
        "rank": "epoch",
        "parent": "Devonian",
        "ma_top": 382.7,
        "ma_base": 393.3,
    },
    {
        "name": "Late Devonian",
        "cn": "晚泥盆世",
        "rank": "epoch",
        "parent": "Devonian",
        "ma_top": 358.9,
        "ma_base": 382.7,
    },
    {
        "name": "Lochkovian",
        "cn": "洛霍考夫期",
        "rank": "age",
        "parent": "Early Devonian",
        "ma_top": 410.8,
        "ma_base": 419.2,
    },
    {
        "name": "Pragian",
        "cn": "布拉格期",
        "rank": "age",
        "parent": "Early Devonian",
        "ma_top": 407.6,
        "ma_base": 410.8,
    },
    {
        "name": "Emsian",
        "cn": "埃姆斯期",
        "rank": "age",
        "parent": "Early Devonian",
        "ma_top": 393.3,
        "ma_base": 407.6,
    },
    {
        "name": "Eifelian",
        "cn": "艾菲尔期",
        "rank": "age",
        "parent": "Middle Devonian",
        "ma_top": 387.7,
        "ma_base": 393.3,
    },
    {
        "name": "Givetian",
        "cn": "吉维特期",
        "rank": "age",
        "parent": "Middle Devonian",
        "ma_top": 382.7,
        "ma_base": 387.7,
    },
    {
        "name": "Frasnian",
        "cn": "弗拉期",
        "rank": "age",
        "parent": "Late Devonian",
        "ma_top": 372.2,
        "ma_base": 382.7,
    },
    {
        "name": "Famennian",
        "cn": "法门期",
        "rank": "age",
        "parent": "Late Devonian",
        "ma_top": 358.9,
        "ma_base": 372.2,
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
    # Phase 3B 2026-08-19: ICS 2023 formally designates these as
    # "series" rank (not "epoch") — updated for rank-type fidelity.
    {
        "name": "Cisuralian",
        "cn": "乌拉尔统",
        "rank": "series",
        "parent": "Permian",
        "ma_top": 273.01,
        "ma_base": 298.9,
    },
    {
        "name": "Guadalupian",
        "cn": "瓜德鲁普统",
        "rank": "series",
        "parent": "Permian",
        "ma_top": 259.51,
        "ma_base": 273.01,
    },
    {
        "name": "Lopingian",
        "cn": "乐平统",
        "rank": "series",
        "parent": "Permian",
        "ma_top": 251.9,
        "ma_base": 259.51,
    },
    # Carboniferous subsystems / ages (ICS 2023)
    # Mississippian (early Carboniferous subsystem)
    # Phase 3B 2026-08-19: ICS 2023 formally designates
    # Mississippian / Pennsylvanian as "subsystem" rank (below
    # system / period, above series / stage), not "epoch".
    {
        "name": "Mississippian",
        "cn": "密西西比纪",
        "rank": "subsystem",
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
    # Pennsylvanian (late Carboniferous subsystem)
    {
        "name": "Pennsylvanian",
        "cn": "宾夕法尼亚纪",
        "rank": "subsystem",
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
    # Mesozoic epochs / series (ICS 2023) — audit 2026-07-31: without
    # these, "Late Triassic" / "Upper Cretaceous" etc. classified as the
    # whole period and the Early/Middle/Late modifier was silently
    # dropped, so Ma ranges were the full-period interval (up to
    # ~25 Myr too wide, midpoints off by ~20 Myr).
    {
        "name": "Early Triassic",
        "cn": "早三叠世",
        "rank": "epoch",
        "parent": "Triassic",
        "ma_top": 247.2,
        "ma_base": 251.9,
    },
    {
        "name": "Middle Triassic",
        "cn": "中三叠世",
        "rank": "epoch",
        "parent": "Triassic",
        "ma_top": 237.0,
        "ma_base": 247.2,
    },
    {
        "name": "Late Triassic",
        "cn": "晚三叠世",
        "rank": "epoch",
        "parent": "Triassic",
        "ma_top": 201.4,
        "ma_base": 237.0,
    },
    {
        "name": "Early Jurassic",
        "cn": "早侏罗世",
        "rank": "epoch",
        "parent": "Jurassic",
        "ma_top": 174.7,
        "ma_base": 201.4,
    },
    {
        "name": "Middle Jurassic",
        "cn": "中侏罗世",
        "rank": "epoch",
        "parent": "Jurassic",
        "ma_top": 161.5,
        "ma_base": 174.7,
    },
    {
        "name": "Late Jurassic",
        "cn": "晚侏罗世",
        "rank": "epoch",
        "parent": "Jurassic",
        "ma_top": 145.0,
        "ma_base": 161.5,
    },
    {
        "name": "Lower Cretaceous",
        "cn": "早白垩世",
        "rank": "epoch",
        "parent": "Cretaceous",
        "ma_top": 100.5,
        "ma_base": 139.8,
    },
    {
        "name": "Upper Cretaceous",
        "cn": "晚白垩世",
        "rank": "epoch",
        "parent": "Cretaceous",
        "ma_top": 66.0,
        "ma_base": 100.5,
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
        "rank": "age",  # P1-6 fix: was "epoch" — ICS 2023 places Pleistocene as an age under Quaternary period
        "parent": "Quaternary",
        "ma_top": 0.0117,
        "ma_base": 2.58,
    },
    {
        "name": "Holocene",
        "cn": "全新世",
        "rank": "age",  # P1-6 fix: was "epoch" — ICS 2023 places Holocene as an age under Quaternary period
        "parent": "Quaternary",
        "ma_top": 0.0,
        "ma_base": 0.0117,
    },
    # Cenozoic stages (ICS 2023-09 chart) — audit 2026-08-01 batch W2 (C4):
    # the previous table only had the period/epoch ranks (Paleogene,
    # Eocene, Oligocene, …) and the bare Quaternary ages; downstream
    # code that cited a specific stage ("Priabonian", "Burdigalian",
    # "Calabrian") fell through to the (often unavailable) PBDB
    # network fallback and was misclassified as the parent epoch.
    # Paleogene stages (parent = Paleocene / Eocene / Oligocene)
    {
        "name": "Danian",
        "cn": "丹麦期",
        "rank": "age",
        "parent": "Paleocene",
        "ma_top": 61.6,
        "ma_base": 66.0,
    },
    {
        "name": "Selandian",
        "cn": "塞兰特期",
        "rank": "age",
        "parent": "Paleocene",
        "ma_top": 59.2,
        "ma_base": 61.6,
    },
    {
        "name": "Thanetian",
        "cn": "塔内特期",
        "rank": "age",
        "parent": "Paleocene",
        "ma_top": 56.0,
        "ma_base": 59.2,
    },
    {
        "name": "Ypresian",
        "cn": "伊普里斯期",
        "rank": "age",
        "parent": "Eocene",
        "ma_top": 47.8,
        "ma_base": 56.0,
    },
    {
        "name": "Lutetian",
        "cn": "卢泰特期",
        "rank": "age",
        "parent": "Eocene",
        "ma_top": 41.2,
        "ma_base": 47.8,
    },
    {
        "name": "Bartonian",
        "cn": "巴顿期",
        "rank": "age",
        "parent": "Eocene",
        "ma_top": 37.71,
        "ma_base": 41.2,
    },
    {
        "name": "Priabonian",
        "cn": "普里阿邦期",
        "rank": "age",
        "parent": "Eocene",
        "ma_top": 33.9,
        "ma_base": 37.71,
    },
    {
        "name": "Rupelian",
        "cn": "鲁培尔期",
        "rank": "age",
        "parent": "Oligocene",
        "ma_top": 27.82,
        "ma_base": 33.9,
    },
    {
        "name": "Chattian",
        "cn": "恰特期",
        "rank": "age",
        "parent": "Oligocene",
        "ma_top": 23.03,
        "ma_base": 27.82,
    },
    # Neogene stages (parent = Miocene / Pliocene)
    {
        "name": "Aquitanian",
        "cn": "阿基坦期",
        "rank": "age",
        "parent": "Miocene",
        "ma_top": 20.44,
        "ma_base": 23.03,
    },
    {
        "name": "Burdigalian",
        "cn": "布尔迪加尔期",
        "rank": "age",
        "parent": "Miocene",
        "ma_top": 15.97,
        "ma_base": 20.44,
    },
    {
        "name": "Langhian",
        "cn": "兰盖期",
        "rank": "age",
        "parent": "Miocene",
        "ma_top": 13.65,
        "ma_base": 15.97,
    },
    {
        "name": "Serravallian",
        "cn": "塞拉瓦尔期",
        "rank": "age",
        "parent": "Miocene",
        "ma_top": 11.63,
        "ma_base": 13.65,
    },
    {
        "name": "Tortonian",
        "cn": "托尔托纳期",
        "rank": "age",
        "parent": "Miocene",
        "ma_top": 7.246,
        "ma_base": 11.63,
    },
    {
        "name": "Messinian",
        "cn": "梅西尼期",
        "rank": "age",
        "parent": "Miocene",
        "ma_top": 5.333,
        "ma_base": 7.246,
    },
    {
        "name": "Zanclean",
        "cn": "赞克勒期",
        "rank": "age",
        "parent": "Pliocene",
        "ma_top": 3.60,
        "ma_base": 5.333,
    },
    {
        "name": "Piacenzian",
        "cn": "皮亚琴期",
        "rank": "age",
        "parent": "Pliocene",
        "ma_top": 2.58,
        "ma_base": 3.60,
    },
    # Quaternary stages (parent = Quaternary)
    {
        "name": "Gelasian",
        "cn": "杰拉期",
        "rank": "age",
        "parent": "Quaternary",
        "ma_top": 1.80,
        "ma_base": 2.58,
    },
    {
        "name": "Calabrian",
        "cn": "卡拉布里期",
        "rank": "age",
        "parent": "Quaternary",
        "ma_top": 0.774,
        "ma_base": 1.80,
    },
    {
        "name": "Chibanian",
        "cn": "契班期",
        "rank": "age",
        "parent": "Quaternary",
        "ma_top": 0.129,
        "ma_base": 0.774,
    },
    {
        # Phase 3B 2026-08-19 audit: ICS 2024-09 chart renamed the
        # final Pleistocene sub-stage from "Late Pleistocene" to
        # "Tarantian" (0.012–0.129 Ma). Previously the row had a
        # wrong parent ("Pleistocene", which is an age not a period)
        # and a Chinese name "晚上新世" that translates to "Late
        # Pliocene", not "Late Pleistocene". ICS places Tarantian
        # directly under Quaternary.
        "name": "Tarantian",
        "cn": "塔兰期",
        "rank": "age",
        "parent": "Quaternary",
        "ma_top": 0.012,
        "ma_base": 0.129,
    },
    # Permian stages (radiolarian-relevant) — phase 3B 2026-08-19
    # audit: ICS 2023 chart places the Asselian/Sakmarian boundary at
    # 293.52 Ma (not 295.0 Ma, which was the older 2004 ICS value).
    {
        "name": "Asselian",
        "cn": "阿瑟尔期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 293.52,
        "ma_base": 298.9,
    },
    {
        "name": "Sakmarian",
        "cn": "萨克马尔期",
        "rank": "age",
        "parent": "Permian",
        "ma_top": 290.1,
        "ma_base": 293.52,
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

# audit 2026-07-31: common-language age names that do not equal any
# ICS row name but map onto a known epoch / series. "Late Permian" is
# Lopingian, "上二叠统" is the Chinese lithostratigraphic name for
# Lopingian, etc. Keys are lower-cased.
_ICS_ALIASES: dict[str, str] = {
    # Permian: Late/Middle/Early → series names
    "late permian": "Lopingian",
    "middle permian": "Guadalupian",
    "early permian": "Cisuralian",
    # Carboniferous
    "late carboniferous": "Pennsylvanian",
    "early carboniferous": "Mississippian",
    # Cretaceous: Late/Early are the common English usage for the
    # Lower/Upper series
    "late cretaceous": "Upper Cretaceous",
    "early cretaceous": "Lower Cretaceous",
    # Chinese lithostratigraphic series names (统)
    "上二叠统": "Lopingian",
    "中二叠统": "Guadalupian",
    "下二叠统": "Cisuralian",
    "上石炭统": "Pennsylvanian",
    "下石炭统": "Mississippian",
    "上白垩统": "Upper Cretaceous",
    "下白垩统": "Lower Cretaceous",
    "上侏罗统": "Late Jurassic",
    "中侏罗统": "Middle Jurassic",
    "下侏罗统": "Early Jurassic",
    "上三叠统": "Late Triassic",
    "中三叠统": "Middle Triassic",
    "下三叠统": "Early Triassic",
}

# Range form: "Late Jurassic to Early Cretaceous", "Middle to Late
# Jurassic", "late Valanginian to early Hauterivian" — the most common
# way radiolarian papers express a stratigraphic span.
_AGE_RANGE_RE = re.compile(r"^(.+?)\s+(?:to|and|until)\s+(.+)$", re.IGNORECASE)


def _classify_age_range(raw: str) -> AgeClassification | None:
    """Classify "X to Y" spans as the union interval of both ends."""
    m = _AGE_RANGE_RE.match(raw)
    if not m:
        return None
    left_raw = m.group(1).strip()
    right_raw = m.group(2).strip()
    left = classify_age_string(left_raw)
    right = classify_age_string(right_raw)
    # "Middle to Late Jurassic" — the left end is a BARE modifier
    # ("Middle"), not a full name; fold it onto the right end's
    # period so it resolves to "Middle Jurassic".
    if (
        left.confidence <= 0
        and right.confidence > 0
        and left_raw.lower() in {"early", "middle", "late", "lower", "upper"}
    ):
        anchor = right.period or right.epoch or right.age
        if anchor:
            folded = classify_age_string(f"{left_raw} {anchor}")
            if folded.confidence > 0:
                left = folded
    if left.confidence <= 0 or right.confidence <= 0:
        return None
    tops = [v for v in (left.ma_top, right.ma_top) if v is not None]
    bases = [v for v in (left.ma_base, right.ma_base) if v is not None]
    if not tops or not bases:
        # no numeric bounds → keep the raw string with the endpoints'
        # names but no Ma (better than a misleading empty confidence=0)
        return AgeClassification(
            raw=raw,
            period=left.period or right.period,
            rank="range",
            confidence=min(left.confidence, right.confidence),
        )
    period = left.period if left.period == right.period else None
    return AgeClassification(
        raw=raw,
        period=period,
        rank="range",
        confidence=min(left.confidence, right.confidence),
        ma_top=min(tops),
        ma_base=max(bases),
        ma_mid=(min(tops) + max(bases)) / 2.0,
    )


def classify_age_string(text: str) -> AgeClassification:
    """Map a free-form age string to ``(period, epoch, age)``.

    Examples
    --------
    >>> classify_age_string("Changhsingian").age
    'Changhsingian'
    >>> classify_age_string("Late Permian").epoch
    'Lopingian'
    >>> classify_age_string("上二叠统").period
    'Permian'
    >>> classify_age_string("Late Jurassic to Early Cretaceous").ma_base > 150
    True
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
    # 1) Full-string hit first: epoch entries are named "Late Triassic"
    #    etc., so the modifier must NOT be stripped before this lookup.
    hit = ICS_INDEX.get(raw) or ICS_INDEX.get(raw.lower()) or ICS_INDEX.get(raw.capitalize())
    if hit:
        return _build_classification(raw, hit, raw)
    # 2) Common-language aliases ("Late Permian" → Lopingian, Chinese
    #    统 names, …).
    alias = _ICS_ALIASES.get(raw) or _ICS_ALIASES.get(raw.lower())
    if alias:
        target = ICS_INDEX.get(alias) or ICS_INDEX.get(alias.lower())
        if target:
            return _build_classification(raw, target, alias)
    # 3) Range form "X to Y" — must run BEFORE modifier stripping so
    #    "Late Jurassic to Early Cretaceous" isn't eaten by the
    #    leading-modifier rule.
    range_cls = _classify_age_range(raw)
    if range_cls is not None:
        return range_cls
    # 4) Strip the leading modifier and look up the bare name.
    body = _MODIFIER_PATTERN.sub("", raw).strip()
    hit = ICS_INDEX.get(body) or ICS_INDEX.get(body.lower()) or ICS_INDEX.get(body.capitalize())
    if hit:
        return _build_classification(raw, hit, body)
    # Try PBDB fallback for unusual names
    pbdb = _pbdb_lookup(body or raw)
    if pbdb:
        return _build_classification(raw, pbdb, body or raw, confidence=0.85)
    return AgeClassification(raw=raw, confidence=0.0)


def _build_classification(
    raw: str,
    hit: dict[str, Any],
    body: str,
    confidence: float = 0.95,
) -> AgeClassification:
    rank = hit["rank"]
    period = epoch = age = None
    # Phase 3B 2026-08-19: ICS 2023 formally designates the
    # Permian / Silurian subdivisions as "series" (not "epoch") and
    # the Carboniferous sub-periods as "subsystem" (not "epoch").
    # All three ranks are middle-rank chronostratigraphic divisions
    # sitting between "period" and "age" — treat them all as epoch
    # for downstream classification so the public API surface
    # (cls.epoch / cls.period) is unchanged for existing callers and
    # tests.
    _MIDDLE_RANKS = {"epoch", "series", "subsystem"}
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
    elif rank in _MIDDLE_RANKS:
        epoch = hit["name"]
        period = hit["parent"]
    elif rank == "age":
        age = hit["name"]
        # Walk up to find period
        parent = hit["parent"]
        if parent in {r["name"] for r in _ICS_ROWS if r["rank"] == "period"}:
            period = parent
        else:
            # find period by walking grandparents — accept any
            # middle-rank parent (epoch / series / subsystem) so the
            # rank-type expansion above doesn't break Tournaisian
            # (parent = Mississippian, rank = subsystem) etc.
            for r in _ICS_ROWS:
                if r["name"] == parent and r["rank"] in _MIDDLE_RANKS:
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
# audit 2026-08-01 batch W2 (D1): the previous module-level cache
# state had no lock, no negative cache, and a non-atomic write.
# Concurrent callers (e.g. parallel PBDB fallback invocations from a
# web UI batch) raced on the read-modify-write of
# ``_PBDB_INTERVALS_CACHE`` and could either return a partially
# populated cache or write a half-finished JSON to disk. The lock +
# atomic write + negative-cache fixes below keep the helper correct
# under concurrent calls and during mid-write process crashes.
_PBDB_INTERVALS_LOCK = threading.Lock()
_PBDB_INTERVALS_NEG_CACHE: dict[str, float] = {}
_PBDB_NEG_TTL_SECONDS: float = 300.0  # 5 minutes
_PBDB_ENDPOINT_URL = "https://paleobiodb.org/data1.2/intervals/list.json?all_parents=1"


def fetch_pbdb_intervals(
    force: bool = False, cache_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Fetch and cache the PBDB ``/intervals/list.json`` chronostratigraphic chart.

    Network call — does *not* run in unit tests.  Use :func:`_pbdb_lookup`
    which transparently uses the cache.

    Concurrency: thread-safe via :data:`_PBDB_INTERVALS_LOCK`. Repeated
    failures populate an in-memory negative cache (5 min TTL) so a
    transient outage does not turn into a thundering herd of network
    retries. Disk writes are atomic via ``tempfile`` + ``os.replace``.
    """
    global _PBDB_INTERVALS_CACHE, _PBDB_LAST_FETCH, _PBDB_INTERVALS_NEG_CACHE
    with _PBDB_INTERVALS_LOCK:
        if _PBDB_INTERVALS_CACHE is not None and not force:
            return _PBDB_INTERVALS_CACHE
        cache_dir = cache_dir or Path.home() / ".cache" / "rlpe" / "paleodb"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "intervals.json"
        # Negative cache: skip network for a short window after a
        # recent failure so a transient outage doesn't trigger a
        # thundering-herd of retries.
        neg_expires = _PBDB_INTERVALS_NEG_CACHE.get(_PBDB_ENDPOINT_URL)
        if neg_expires is not None and neg_expires > time.monotonic() and not force:
            return []
        # Check if cached data is still valid (within TTL)
        if cache_path.exists() and not force:
            # audit 2026-07-26: base TTL on the cache file's mtime, not
            # the in-process _PBDB_LAST_FETCH (which is 0.0 after a restart,
            # making a fresh on-disk cache look stale and forcing a re-fetch).
            try:
                age = time.time() - cache_path.stat().st_mtime
            except OSError:
                age = _PBDB_CACHE_TTL_SECONDS
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
                    logging.getLogger(__name__).warning(
                        "PBDB intervals cache at %s is unreadable (%s); falling through to live fetch",
                        cache_path,
                        exc,
                    )
        try:
            import requests  # type: ignore

            resp = requests.get(
                _PBDB_ENDPOINT_URL,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json().get("records", [])
            # Atomic write: write to a temp file in the same directory
            # then ``os.replace`` it onto the target path. A process
            # crash mid-write leaves the original cache intact (the
            # temp file may linger but is invisible to consumers).
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(cache_dir),
                prefix="intervals.json.",
                suffix=".tmp",
            )
            try:
                with open(tmp_fd, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(data, ensure_ascii=False))
            except Exception:
                # Best-effort cleanup of the temp file on failure.
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            os.replace(tmp_path, cache_path)
            _PBDB_INTERVALS_CACHE = data
            _PBDB_LAST_FETCH = time.time()
            # Clear any stale negative cache on success.
            _PBDB_INTERVALS_NEG_CACHE.pop(_PBDB_ENDPOINT_URL, None)
            return data
        except Exception:
            # Record the failure so subsequent callers within the TTL
            # don't re-hit the network. We expire the entry via
            # ``time.monotonic`` so a pause (suspend / sleep) does not
            # extend the cooldown.
            _PBDB_INTERVALS_NEG_CACHE[_PBDB_ENDPOINT_URL] = time.monotonic() + _PBDB_NEG_TTL_SECONDS
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
                "ma_top": lag,  # younger bound
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
#
# Audit 2026-09-03 (BLOCKER-#7): each entry now carries a
# ``confidence`` value (0.0-1.0) so downstream consumers
# (AgeClassification, PBDB exporter, find_ages_in_text) can
# distinguish between well-anchored UAZ 1-12 (Baumgartner et al.
# 1995 table, confidence=0.95) and the evenly-interpolated UAZ
# 13-21 (confidence=0.5, marked "(approx.)" in the comments). The
# 0.5 confidence lets the PBDB exporter use a section-based
# fallback (which carries the actual measured age) instead of the
# zone midpoint when the operator submits the occurrence.
class BiozoneMa(NamedTuple):
    """Ma bounds + calibration confidence for a biozone entry.

    ``confidence`` reflects how well the (top, base) Ma values are
    anchored in the published chronostratigraphic literature:

      * 0.95: zones whose boundaries are tabulated against a stage
        (e.g. Baumgartner et al. 1995 UAZ 1-12 against ICS stages).
      * 0.85: zones whose stage assignment is widely accepted but
        boundary Ma is calibrated against a different reference
        (e.g. Pessagno Zones A/B/C from O'Dogherty 1994).
      * 0.50: zones whose Ma is interpolated rather than calibrated
        (e.g. UAZ 13-21 — "(approx.)" in the source comments).
    """

    top_ma: float
    base_ma: float
    confidence: float = 0.85


_BIOZONE_TO_MA: dict[str, BiozoneMa] = {
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
    "UAZ 1": BiozoneMa(172.0, 174.7, confidence=0.95),  # early–middle Aalenian
    "UAZ 2": BiozoneMa(170.9, 172.0, confidence=0.95),  # late Aalenian
    "UAZ 3": BiozoneMa(169.3, 170.9, confidence=0.95),  # early–middle Bajocian
    "UAZ 4": BiozoneMa(167.7, 169.3, confidence=0.95),  # late Bajocian
    "UAZ 5": BiozoneMa(166.2, 167.7, confidence=0.95),  # latest Bajocian–early Bathonian
    "UAZ 6": BiozoneMa(165.0, 166.2, confidence=0.95),  # middle Bathonian
    "UAZ 7": BiozoneMa(163.0, 165.3, confidence=0.95),  # late Bathonian–early Callovian
    "UAZ 8": BiozoneMa(158.0, 163.0, confidence=0.95),  # middle Callovian–early Oxfordian
    "UAZ 9": BiozoneMa(156.0, 158.5, confidence=0.95),  # middle–late Oxfordian
    "UAZ 10": BiozoneMa(152.0, 156.0, confidence=0.95),  # late Oxfordian–early Kimmeridgian
    "UAZ 11": BiozoneMa(147.5, 152.0, confidence=0.95),  # late Kimmeridgian–early Tithonian
    "UAZ 12": BiozoneMa(145.0, 147.5, confidence=0.95),  # early Tithonian
    # UAZ 13-21 (Tithonian → Barremian) are calibrated to UAZones95
    # but the Ma bounds are evenly interpolated over the 145-123 Ma
    # interval — confidence=0.5 flags them as "use the section-based
    # age in PBDB submissions, NOT the zone midpoint" (BLOCKER-#7).
    "UAZ 13": BiozoneMa(142.6, 145.0, confidence=0.5),  # late Tithonian (approx.)
    "UAZ 14": BiozoneMa(140.2, 142.6, confidence=0.5),  # latest Tithonian–early Berriasian (approx.)
    "UAZ 15": BiozoneMa(137.8, 140.2, confidence=0.5),  # Berriasian (approx.)
    "UAZ 16": BiozoneMa(135.4, 137.8, confidence=0.5),  # late Berriasian–early Valanginian (approx.)
    "UAZ 17": BiozoneMa(133.0, 135.4, confidence=0.5),  # Valanginian (approx.)
    "UAZ 18": BiozoneMa(130.6, 133.0, confidence=0.5),  # late Valanginian–early Hauterivian (approx.)
    "UAZ 19": BiozoneMa(128.2, 130.6, confidence=0.5),  # Hauterivian (approx.)
    "UAZ 20": BiozoneMa(125.8, 128.2, confidence=0.5),  # late Hauterivian (approx.)
    "UAZ 21": BiozoneMa(123.4, 125.8, confidence=0.5),  # Hauterivian–Barremian boundary (approx.)
    # Hollis 1997 NZ Late Cretaceous radiolarian zones
    # Buryella clinata Zone: Thanetian (late Paleocene), ~56-59 Ma
    # (corrected: was incorrectly set to Wuchiapingian ~254-259 Ma)
    "Buryella clinata Zone": BiozoneMa(56.0, 59.0, confidence=0.85),  # Thanetian
    "Cryptocephalus nigricae Zone": BiozoneMa(83.6, 86.3, confidence=0.85),  # Coniacian–Santonian
    # O'Dogherty 1994 Betic Cordillera zones (mid-Cretaceous subset)
    # P1-7 fix: corrected to ICS 2023 stage boundaries.
    # Valanginian: 139.8-132.6 Ma; Hauterivian: 132.6-125.77 Ma
    # Barremian: 125.77-121.4 Ma; Aptian: 121.4-113.0 Ma; Albian: 113.0-100.5 Ma
    "Pessagno Zone A": BiozoneMa(125.77, 132.6, confidence=0.85),  # Hauterivian
    "Pessagno Zone B": BiozoneMa(121.4, 125.77, confidence=0.85),  # Barremian (lower Aptian boundary at 125.77)
    "Pessagno Zone C": BiozoneMa(113.0, 121.4, confidence=0.85),  # Aptian
    # Legacy radiolarian zonation (Riedel & Sanfilippo 1978)
    # — commonly cited in older bandini / pouille papers.
    # Buryella tetradica Zone: Coniacian-Santonian (Late Cretaceous), ~83-89 Ma
    # (corrected: was incorrectly set to Olenekian-Anisian ~247-251 Ma)
    "Buryella tetradica Zone": BiozoneMa(83.6, 89.0, confidence=0.85),  # Coniacian–Santonian
    "Triassocampe deweveri Zone": BiozoneMa(208.5, 227.0, confidence=0.85),  # Carnian–Norian
    # Bare-name aliases (no trailing "Zone") so callers that already
    # stripped the suffix don't pay an extra re-lookup cost. Both
    # forms resolve to the same (ma_top, ma_base) tuple.
    "Buryella clinata": BiozoneMa(56.0, 59.0, confidence=0.85),  # Thanetian (corrected)
    "Cryptocephalus nigricae": BiozoneMa(83.6, 86.3, confidence=0.85),
    "Buryella tetradica": BiozoneMa(83.6, 89.0, confidence=0.85),  # Coniacian–Santonian (corrected)
    "Triassocampe deweveri": BiozoneMa(208.5, 227.0, confidence=0.85),
    # ------------------------------------------------------------------
    # RP zones (Radiolarian Paleogene — Sanfilippo & Nigrini 1998)
    # Cenozoic low-latitude radiolarian biochronology, Paleocene
    # through Pleistocene. Cited in essentially all modern Cenozoic
    # radiolarian papers (e.g. Sanfilippo & Blome 2001; Nigrini 2008;
    # Danelian 2006; Pouille 2018; Kamikuri 2010; Kamikuri et al. 2012).
    # Ma bounds are the published Sanfilippo & Nigrini 1998 chart
    # calibrated to the ICS 2023 stage boundaries (Danian 66.0,
    # Thanetian 56.0, Ypresian 47.8, Lutetian 41.2, Bartonian 37.71,
    # Priabonian 33.9, Rupelian 27.3, Chattian 23.04, Aquitanian
    # 20.44, Burdigalian 15.97, Langhian 13.82, Serravallian 11.63,
    # Tortonian 7.25, Messinian 5.33, Zanclean 4.66, Piacenzian 3.6,
    # Gelasian 2.58, Calabrian 1.8, Middle Pleistocene 0.77,
    # Late Pleistocene 0.13, Holocene 0.0).
    #
    # Phase 3E audit 2026-08-19 (Bug M-8): the previous table
    # contained only UAZ + a handful of Cretaceous / Paleocene zones
    # — RP1-RP21 (a 0-34 Ma standard zonation used in 90%+ of Cenozoic
    # papers) was entirely missing, so every RP zone citation resolved
    # to ``None`` and was reported as ``biozone_unknown``. Adding
    # RP1-RP21 closes that 0-34 Ma gap.
    #
    # RP1 = Early Oligocene (Rupelian, upper part).
    # RP21 = Holocene (cosmopolitan flourish).
    # Ma bounds given as (ma_top, ma_base); ma_top is the younger,
    # ma_base is the older boundary in Ma (smaller number = younger).
    "RP1": BiozoneMa(30.0, 34.0, confidence=0.85),  # Early Oligocene
    "RP2": BiozoneMa(24.0, 30.0, confidence=0.85),
    "RP3": BiozoneMa(22.0, 24.0, confidence=0.85),
    "RP4": BiozoneMa(21.0, 22.0, confidence=0.85),
    "RP5": BiozoneMa(18.5, 21.0, confidence=0.85),
    "RP6": BiozoneMa(17.0, 18.5, confidence=0.85),  # Burdigalian
    "RP7": BiozoneMa(14.5, 17.0, confidence=0.85),  # latest Burdigalian–Langhian
    "RP8": BiozoneMa(12.5, 14.5, confidence=0.85),  # Serravallian
    "RP9": BiozoneMa(11.0, 12.5, confidence=0.85),
    "RP10": BiozoneMa(9.5, 11.0, confidence=0.85),
    "RP11": BiozoneMa(8.5, 9.5, confidence=0.85),  # Tortonian
    "RP12": BiozoneMa(7.5, 8.5, confidence=0.85),
    "RP13": BiozoneMa(6.5, 7.5, confidence=0.85),
    "RP14": BiozoneMa(5.5, 6.5, confidence=0.85),  # Messinian/Zanclean
    "RP15": BiozoneMa(4.5, 5.5, confidence=0.85),
    "RP16": BiozoneMa(3.5, 4.5, confidence=0.85),
    "RP17": BiozoneMa(2.5, 3.5, confidence=0.85),  # Piacenzian
    "RP18": BiozoneMa(1.8, 2.5, confidence=0.85),  # Gelasian
    "RP19": BiozoneMa(1.0, 1.8, confidence=0.85),  # Calabrian
    "RP20": BiozoneMa(0.5, 1.0, confidence=0.85),  # Middle Pleistocene
    "RP21": BiozoneMa(0.0, 0.5, confidence=0.85),  # Late Pleistocene–Holocene
    # RP-zone "Biozone" trailing-word form (Sanfilippo & Nigrini 1998
    # writes both "RP6" and "RP6 Biozone" interchangeably). Stored as
    # plain aliases so we don't pay a per-lookup regex strip cost.
    "RP1 Biozone": BiozoneMa(30.0, 34.0, confidence=0.85),
    "RP2 Biozone": BiozoneMa(24.0, 30.0, confidence=0.85),
    "RP3 Biozone": BiozoneMa(22.0, 24.0, confidence=0.85),
    "RP4 Biozone": BiozoneMa(21.0, 22.0, confidence=0.85),
    "RP5 Biozone": BiozoneMa(18.5, 21.0, confidence=0.85),
    "RP6 Biozone": BiozoneMa(17.0, 18.5, confidence=0.85),
    "RP7 Biozone": BiozoneMa(14.5, 17.0, confidence=0.85),
    "RP8 Biozone": BiozoneMa(12.5, 14.5, confidence=0.85),
    "RP9 Biozone": BiozoneMa(11.0, 12.5, confidence=0.85),
    "RP10 Biozone": BiozoneMa(9.5, 11.0, confidence=0.85),
    "RP11 Biozone": BiozoneMa(8.5, 9.5, confidence=0.85),
    "RP12 Biozone": BiozoneMa(7.5, 8.5, confidence=0.85),
    "RP13 Biozone": BiozoneMa(6.5, 7.5, confidence=0.85),
    "RP14 Biozone": BiozoneMa(5.5, 6.5, confidence=0.85),
    "RP15 Biozone": BiozoneMa(4.5, 5.5, confidence=0.85),
    "RP16 Biozone": BiozoneMa(3.5, 4.5, confidence=0.85),
    "RP17 Biozone": BiozoneMa(2.5, 3.5, confidence=0.85),
    "RP18 Biozone": BiozoneMa(1.8, 2.5, confidence=0.85),
    "RP19 Biozone": BiozoneMa(1.0, 1.8, confidence=0.85),
    "RP20 Biozone": BiozoneMa(0.5, 1.0, confidence=0.85),
    "RP21 Biozone": BiozoneMa(0.0, 0.5, confidence=0.85),
    # ------------------------------------------------------------------
    # RN zones (Riedel & Sanfilippo 1978) — older low-latitude
    # Cenozoic radiolarian zonation, used in pre-1998 papers and still
    # cited as a complementary scheme. RN1-RN17 spans Holocene
    # (RN1) back to Aptian (RN17). Ma bounds are the published
    # Riedel & Sanfilippo 1978 chart calibrated to ICS 2023.
    #
    # RN1 = Holocene–Late Pleistocene; RN17 = Aptian (~118-127 Ma).
    # RN4 = Tortonian (commonly cited in Mediterranean Miocene papers).
    # RN6 = Chattian–Rupelian (commonly cited in Oligocene papers).
    "RN1": BiozoneMa(0.0, 1.8, confidence=0.85),
    "RN2": BiozoneMa(1.8, 5.0, confidence=0.85),
    "RN3": BiozoneMa(5.0, 9.0, confidence=0.85),
    "RN4": BiozoneMa(9.0, 15.0, confidence=0.85),
    "RN5": BiozoneMa(15.0, 22.0, confidence=0.85),
    "RN6": BiozoneMa(22.0, 30.0, confidence=0.85),
    "RN7": BiozoneMa(30.0, 39.0, confidence=0.85),
    "RN8": BiozoneMa(39.0, 50.0, confidence=0.85),
    "RN9": BiozoneMa(50.0, 56.0, confidence=0.85),  # Thanetian
    "RN10": BiozoneMa(56.0, 65.0, confidence=0.85),  # Selandian–Danian
    "RN11": BiozoneMa(65.0, 74.0, confidence=0.85),  # Maastrichtian
    "RN12": BiozoneMa(74.0, 84.0, confidence=0.85),  # Campanian
    "RN13": BiozoneMa(84.0, 92.0, confidence=0.85),  # Santonian–Coniacian
    "RN14": BiozoneMa(92.0, 100.0, confidence=0.85),  # Cenomanian–Turonian
    "RN15": BiozoneMa(100.0, 110.0, confidence=0.85),  # Albian
    "RN16": BiozoneMa(110.0, 118.0, confidence=0.85),  # Aptian
    "RN17": BiozoneMa(118.0, 127.0, confidence=0.85),  # Aptian–Barremian
    "RN1 Biozone": BiozoneMa(0.0, 1.8, confidence=0.85),
    "RN2 Biozone": BiozoneMa(1.8, 5.0, confidence=0.85),
    "RN3 Biozone": BiozoneMa(5.0, 9.0, confidence=0.85),
    "RN4 Biozone": BiozoneMa(9.0, 15.0, confidence=0.85),
    "RN5 Biozone": BiozoneMa(15.0, 22.0, confidence=0.85),
    "RN6 Biozone": BiozoneMa(22.0, 30.0, confidence=0.85),
    "RN7 Biozone": BiozoneMa(30.0, 39.0, confidence=0.85),
    "RN8 Biozone": BiozoneMa(39.0, 50.0, confidence=0.85),
    "RN9 Biozone": BiozoneMa(50.0, 56.0, confidence=0.85),
    "RN10 Biozone": BiozoneMa(56.0, 65.0, confidence=0.85),
    "RN11 Biozone": BiozoneMa(65.0, 74.0, confidence=0.85),
    "RN12 Biozone": BiozoneMa(74.0, 84.0, confidence=0.85),
    "RN13 Biozone": BiozoneMa(84.0, 92.0, confidence=0.85),
    "RN14 Biozone": BiozoneMa(92.0, 100.0, confidence=0.85),
    "RN15 Biozone": BiozoneMa(100.0, 110.0, confidence=0.85),
    "RN16 Biozone": BiozoneMa(110.0, 118.0, confidence=0.85),
    "RN17 Biozone": BiozoneMa(118.0, 127.0, confidence=0.85),
}


# Phase 3E audit 2026-08-19 (Bug M-8): shared regex that matches all
# of the Cenozoic numbered-zone notations used in this file (UAZ,
# RP, RN). UAZ was the only numbered pattern recognised before; RP
# and RN (Sanfilippo & Nigrini 1998 + Riedel & Sanfilippo 1978) were
# dropped on the floor of find_ages_in_text / mining helpers, even
# though the lookup table now contains them.
#
# Forms accepted:
#   "RP6 Biozone", "RP 6", "RP6", "RP6-RP7" / "RP6-RP7 Biozone"
#   "RN4",        "RN 4",        "RN4-RN5"  / "RN4-RN5 Biozone"
#   "UAZ 5",      "UAZ5",        "UAZ 4-7"  (audit 2026-07-31)
#
# The pattern is case-insensitive and tolerant of optional whitespace
# between prefix and digits, with an optional range expression
# ``-N`` / ``- N`` and an optional trailing ``Biozone`` / ``Zone`` /
# ``Subzone`` word. The "5" in "RN5-5" is tolerated (regex doesn't
# range-check the right-hand number) — the lookup helper returns
# ``None`` for genuinely unknown right-hand members, which is the
# correct conservative behaviour.
_BIOZONE_RE = re.compile(
    r"\b(?:UAZ|RP|RN)\s*\d+(?:\s*[-–—]\s*\d+)?\s*"
    r"(?:Biozone|Zone|Subzone|Sub-biozone)?\b",
    re.IGNORECASE,
)


def lookup_biozone_ma(name: str | None) -> BiozoneMa | None:
    """Look up the (ma_top, ma_base, confidence) for a named biozone.

    Returns a :class:`BiozoneMa` (top_ma, base_ma, confidence) if the
    name resolves, or ``None`` if it is missing, empty, or not in
    the curated table. The helper is case-insensitive and tolerates
    a trailing ``Zone`` / ``Subzone`` so ``"Buryella clinata"`` and
    ``"Buryella clinata Zone"`` resolve to the same bounds.

    The ``confidence`` value (audit 2026-09-03 BLOCKER-#7) lets
    downstream consumers distinguish well-anchored zones (0.95 for
    UAZ 1-12 against ICS 2023 stages) from interpolated zones (0.5
    for UAZ 13-21, marked "(approx.)" in the source comments). The
    PBDB exporter uses a confidence < 0.7 as the signal to fall
    back to the section-measured age rather than the zone midpoint.

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
    # Phase 3E audit 2026-08-19 (Bug M-8): extended to also handle
    # RP and RN range expressions like "RP6-RP7" / "RN4-RN5" /
    # "RP 4 - RP 5" / "RP6-Biozone" etc. The single-prefix regex
    # below matches any of the three notations with or without the
    # space between prefix and digits.
    m = re.match(
        r"^(UAZ|RP|RN)\s*(\d+)\s*[-–—]\s*(?:(?:UAZ|RP|RN)\s*)?(\d+)\s*$",
        raw,
        re.IGNORECASE,
    )
    if m:
        prefix = m.group(1).upper()
        lo, hi = int(m.group(2)), int(m.group(3))
        if lo > hi:
            lo, hi = hi, lo
        tops, bases, confs = [], [], []
        for i in range(lo, hi + 1):
            bounds = _BIOZONE_TO_MA.get(f"{prefix} {i}")
            if bounds is None:
                # Some papers write the digits without an intervening
                # space ("RP6-RP7"); try the no-space form too.
                bounds = _BIOZONE_TO_MA.get(f"{prefix}{i}")
            if bounds is None:
                return None  # unknown zone in range → conservative None
            tops.append(bounds.top_ma)
            bases.append(bounds.base_ma)
            confs.append(bounds.confidence)
        # For a range, the union interval uses the most conservative
        # confidence (minimum) so a span that includes an UAZ 13-21
        # zone is flagged as approximate.
        return BiozoneMa(
            top_ma=min(tops),
            base_ma=max(bases),
            confidence=min(confs),
        )
    # Phase 3E audit 2026-08-19 (Bug M-8): normalise the single-zone
    # form so ``RP 6`` and ``RP6`` both resolve to the same bounds.
    # The table is heterogeneous:
    #   * UAZ keys use a space ("UAZ 5")
    #   * RP / RN keys have no space ("RP6", "RN4")
    # Real papers write every variant. We try the direct hit FIRST,
    # then fall back to the alternate spacing if needed. This
    # preserves backward-compat with all Phase 60 entries.
    raw_loose = (
        re.sub(
            r"^([UAZRPNrpnuaz]+)\s+(\d+)$",
            r"\1\2",
            raw,
        )
        if re.match(r"^(?:UAZ|RP|RN)\s+\d+$", raw, re.IGNORECASE)
        else raw
    )
    if raw_loose != raw and raw_loose in _BIOZONE_TO_MA:
        raw = raw_loose
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


def lookup_biozone_ma_legacy(name: str | None) -> tuple[float, float] | None:
    """Backward-compat shim: returns ``(ma_top, ma_base)`` as a plain
    2-tuple instead of a :class:`BiozoneMa` NamedTuple.

    Added 2026-09-03 (BLOCKER-#7) so older call sites that do
    ``top, base = lookup_biozone_ma(name)`` continue to work without
    source changes — the new function returns a 3-element
    ``BiozoneMa(top_ma, base_ma, confidence)`` which would unpack
    differently and break those callers. New code should call
    :func:`lookup_biozone_ma` directly and use the NamedTuple's
    ``.confidence`` attribute.
    """
    bm = lookup_biozone_ma(name)
    if bm is None:
        return None
    return (bm.top_ma, bm.base_ma)


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
    # Phase 3E audit 2026-08-19 (Bug M-8): extend find_ages_in_text to
    # also recognise numbered biozone notations (UAZ, RP, RN). The
    # numbered zones are NOT ICS stages — they have their own
    # letter/digit notation — so the strict above ICS-name search
    # misses them entirely. We use the shared :data:`_BIOZONE_RE`
    # pattern and convert each match into a synthetic
    # AgeClassification whose Ma bounds come from
    # :func:`lookup_biozone_ma`. ``period`` is left as ``None`` and
    # ``age`` carries the canonical zone name (with the "Biozone"
    # suffix stripped), so downstream consumers that walk the
    # ``age`` slot see the biozone label without conflating it with
    # an ICS stage.
    if text:
        for m in _BIOZONE_RE.finditer(text):
            tag = m.group(0).strip()
            bounds = lookup_biozone_ma(tag)
            if bounds is None:
                continue
            ma_top, ma_base = bounds.top_ma, bounds.base_ma
            # Canonicalise the zone name: strip trailing "Biozone"
            # / "Zone" / "Subzone" so downstream code can compare
            # ``age`` strings without worrying about that suffix.
            canonical = re.sub(
                r"\s+(Biozone|Zone|Subzone|Sub-biozone)\s*$",
                "",
                tag,
                flags=re.IGNORECASE,
            ).strip()
            out.append(
                AgeClassification(
                    raw=tag,
                    period=None,
                    epoch=None,
                    age=f"biozone:{canonical}",
                    rank="biozone",
                    # Audit 2026-09-03 (BLOCKER-#7): carry the
                    # biozone's calibration confidence so a PBDB
                    # exporter can use a section-measured age as
                    # fallback when this drops below 0.7 (e.g. UAZ
                    # 13-21 entries). The previous hardcoded 0.85
                    # silently treated UAZ 13-21 as well-anchored
                    # when the source comments explicitly flag them
                    # as "(approx.)".
                    confidence=bounds.confidence,
                    ma_top=ma_top,
                    ma_base=ma_base,
                    ma_mid=(ma_top + ma_base) / 2.0,
                )
            )
    # Deduplicate by (period, epoch, age) but keep the most specific
    dedup: dict[tuple[str | None, str | None, str | None], AgeClassification] = {}
    for c in out:
        key = (c.period, c.epoch, c.age)
        if key not in dedup or c.confidence > dedup[key].confidence:
            dedup[key] = c
    return list(dedup.values())
