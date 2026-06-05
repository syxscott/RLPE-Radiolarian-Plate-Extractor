"""M3-Centric Semantic Figure Understanding Engine.

This module turns MiniMax M3 (a multimodal LLM with extended thinking) from
"末端匹配器" into the *semantic engine* of the radiolarian plate pipeline.
The pipeline is decomposed into 5 cascading M3 stages, each producing
structured JSON consumed by the next stage.  Any stage can fail gracefully
and the next-best classical method takes over.

Stages
------
1. ``parse_caption``         text ->  ``CaptionPair[]``  (label -> species mapping)
2. ``classify_plate``        vision -> ``PlateClassification``  (filter non-radiolarian)
3. ``segment_panels``        vision -> ``PanelBox[]``  (panel bboxes + visible labels)
4. ``match_panel``           vision+text -> ``PanelMatch``  (per-panel species assignment)
5. ``critique_matches``      vision+text -> ``Critique[]``  (cross-panel consistency)

The novelty: M3's multimodal extended-thinking is used as a *joint* document
understanding engine, not a per-panel classifier.  Each stage is small and
focused, the JSON contract is strict, and stages can be turned on/off via
``PipelineConfig.extra['m3_stage_<n>'] = True/False`` for ablation.

Cost (rough): ~¥0.10/figure with 10 panels; 4 figures ≈ ¥0.40.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON parsing helpers (more lenient than the backend's default)
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _safe_json_loads(text: str) -> Any:
    """Parse JSON from text, tolerating ```json fences and preamble.

    Tries the whole text first, then a JSON array, then a JSON object. If
    parsing fails on a top-level array (e.g. M3 produced a syntax error
    inside), falls back to extracting *individual* balanced objects from
    the text so we can recover as much structure as possible.
    """
    if not text:
        raise ValueError("empty text")
    text = text.strip()
    # Strip common code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # 1) Whole text
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) First array match
    arr_match = _JSON_ARRAY_RE.search(text)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except Exception:
            pass
    # 3) First object match
    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass
    # 4) Best-effort: find every balanced {...} block in the text and
    #    parse them individually. Useful when the LLM emits a malformed
    #    array (missing comma, extra brace) but each object is valid.
    items = _extract_balanced_objects(text)
    if items:
        return items
    raise ValueError(f"No JSON object/array found in text: {text[:200]!r}")


def _extract_balanced_objects(text: str) -> list[Any]:
    """Find every balanced ``{...}`` substring in ``text`` and parse each."""
    out: list[Any] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c != "{":
            i += 1
            continue
        # Try to find the matching closing brace, tracking nesting.
        depth = 0
        j = i
        in_str = False
        escape = False
        while j < n:
            ch = text[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth != 0 or j >= n:
            break
        snippet = text[i : j + 1]
        try:
            out.append(json.loads(snippet))
        except Exception:
            # Try the "balanced to the closest closing brace" approach.
            # If this also fails, skip and continue.
            pass
        i = j + 1
    return out


# Regex-based caption parser. Handles the convention used in most OA
# radiolarian papers, where each clause is "figs X-Y. Species Name" or
# "fig N. Species Name", separated by periods, semicolons, or newlines.
# Same pattern family as the LLM prompt example, but deterministic.
# Range separator: hyphen-minus (-), en-dash (–), or em-dash (—).
# Handles "Fig. 1 Svinitzium cf. kamoense" (cf./aff. mid-binomial),
# "Fig. 4 Hiscocapsa lugeoni" (plain binomial), and
# "Fig. 6 Praewilliriedellum sp." (genus + sp.).
# Common Unicode ligatures that appear in OCR'd caption text from
# OpenDataLoader/GROBID. Normalised to ASCII before regex matching so the
# patterns below stay ASCII-only. Notable: "ﬁgs" (U+FB01 + gs) is the most
# common form seen on radiolarian plate captions — a single-character
# ligature is the difference between a parse success and a silent failure.
_LIGATURE_MAP: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "Ĳ": "IJ",
    "ĳ": "ij",
    # Curly quotes / dashes
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    " ": " ",
}


def _normalize_caption_text(text: str) -> str:
    """Normalise ligatures and curly punctuation in a caption to their ASCII
    equivalents before regex matching. Without this, the U+FB01 ligature in
    "ﬁgs" makes ``_CAPTION_CLAUSE_RE`` miss every clause and the parser
    returns zero pairs."""
    if not text:
        return text
    return "".join(_LIGATURE_MAP.get(ch, ch) for ch in text)


# Regex-based caption parser. Handles the convention used in most OA
# radiolarian papers, where each clause is "figs X-Y. Species Name" or
# "fig N. Species Name", separated by periods, semicolons, or newlines.
# Same pattern family as the LLM prompt example, but deterministic.
# Range separator: hyphen-minus (-), en-dash (–), or em-dash (—).
# Handles "Fig. 1 Svinitzium cf. kamoense" (cf./aff. mid-binomial),
# "Fig. 4 Hiscocapsa lugeoni" (plain binomial), and
# "Fig. 6 Praewilliriedellum sp." (genus + sp.).
_CAPTION_CLAUSE_RE = re.compile(
    r"(?:[Ff]igs?\.?)\s*"
    r"((?:\d+(?:\s*[,\-–—]\s*\d+)*(?:\s*,\s*\d+(?:\s*[,\-–—]\s*\d+)*)*))"  # label list "1-3, 5, 7-9"
    r"\s*[\.:]?\s*"
    r"([A-Z][a-zA-Z-]+"  # Genus (capitalized)
    r"(?:"  # optionally followed by epithet, possibly with cf./aff. between
    r"(?:\s+(?:cf\.|aff\.)\s+[a-z][a-zA-Z-]+)"  # cf./aff. + species
    r"|"
    r"(?:\s+[a-z][a-zA-Z-]+)"  # plain species epithet
    r")?"
    r")"
    r"(\s+(?:n\.\s*sp\.|sp\.\s*nov\.|sp\.|spp\.|cf\.|aff\.|n\.\s*gen\.\s*&\s*sp\.|nov\.))?",
)

# Pouille-style "Species (Pl. N, figs M)" clause. Used as a fallback
# when the OpenDataLoader pass synthesised a caption shaped like
#   "Syntagentactinia biocculosa (Pl. 1, figs 5)
#    Syntagentactinia? angulata n. sp. (Pl. 1, figs 12–14b) ..."
# i.e. species name FIRST, then the parenthetical plate/fig reference.
# _CAPTION_CLAUSE_RE only matches the inverse "fig N. Species" form.
# Group 1: species (with optional "?" uncertainty marker and a third
#          epithet for trinomial names). Group 2: optional modifier
#          (n. sp., sp., cf., aff., spp., sp. nov.). Group 3: figure
#          label list ("5" or "12-14b" or "8-11, 14").
_POUILE_CLAUSE_RE = re.compile(
    r"^([A-Z][a-z]+"                       # Genus (capitalized)
    r"(?:"                                  # optional uncertainty or epithet
    r"\s*\?\s+sp\."                        #   "? sp."
    r"|"
    r"\s*\?\s+[a-z][a-z\-]+"               #   "? epithet"
    r"|"
    r"\s+[a-z][a-z\-]+"                    #   plain epithet
    r")*"
    r")"
    r"(\s+(?:n\.\s*sp\.|sp\.\s*nov\.|sp\.|spp\.|cf\.|aff\.))?"  # optional modifier
    r"\s*"
    r"\([^)]*?"                             # opening paren
    r"[Pp](?:l|late)\.?\s*\d+"             # Pl. N
    r"\s*,\s*"
    r"[Ff]igs?\.?\s*"
    r"(\d+[a-z]?"                          # first fig num
    r"(?:\s*[\-–—]\s*\d+[a-z]?)?"
    r"(?:\s*,\s*\d+[a-z]?"
    r"(?:\s*[\-–—]\s*\d+[a-z]?)?"
    r")*"
    r")\s*\)"
)


def _regex_expand_label_list(s: str) -> list[str]:
    """Expand "1-3" → ["1","2","3"]; "1, 3" → ["1","3"]; "13, 16" → ["13","16"];
    "12-14b" → ["12","13","14b"] (trailing letter applies only to the last
    label of the range — bandini/pouille papers use "figs 12-14b" for a
    single sub-figure that spans the plate, not "figs 12b, 13b, 14b").

    The pre-existing ``_expand_label_range`` (defined later in this file) only
    handles ``A-D`` and ``1-5`` ranges, not comma-separated lists. We need a
    version that handles both for caption parsing.
    """
    out: list[str] = []
    for chunk in re.split(r"[,;]", s):
        chunk = chunk.strip().strip(".").strip()
        if not chunk:
            continue
        # Range with optional letter suffix on the upper bound:
        # "1-3" or "12-14b" — the suffix applies only to the last label.
        m = re.match(r"(\d+)\s*[–\-—]\s*(\d+)([a-z]?)$", chunk)
        if m:
            try:
                lo, hi = int(m.group(1)), int(m.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                expanded = [str(x) for x in range(lo, hi)]
                last = str(hi) + m.group(3)
                expanded.append(last)
                out.extend(expanded)
            except Exception:
                out.append(chunk)
        else:
            # Single number, possibly with a letter suffix ("5a", "14b").
            m2 = re.match(r"(\d+)([a-z]?)$", chunk)
            if m2:
                out.append(m2.group(1) + m2.group(2))
            else:
                out.append(chunk)
    seen: set[str] = set()
    result: list[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            result.append(x)
    return result


def _regex_parse_caption(caption_text: str) -> list[CaptionPair]:
    """Regex-only caption parser used as a fallback when the LLM is unavailable.

    Returns ``CaptionPair`` objects with ``confidence=0.7`` (lower than the
    LLM's typical 0.9) so the caller can prefer LLM results when both are
    present.  Handles:
        "figs 1-3. Entactinia itsukichiensis: ..."
        "fig 1, 4. Trilonche crassispinosa (Sashida & Tonishi): ..."
        "figs 5-8. Provisocyntra densa Feng n. sp.: ..."
    """
    if not caption_text:
        return []
    # Normalise ligatures ("ﬁgs" → "figs") and curly punctuation first;
    # otherwise the U+FB01 ligature in OpenDataLoader output makes
    # _CAPTION_CLAUSE_RE miss every clause and return zero pairs.
    text = _normalize_caption_text(caption_text)
    # Strip leading "Explanation of Plate N." so the regex doesn't anchor on
    # the word "Explanation" (which is not a real label).
    text = re.sub(
        r"^\s*Explanation\s+of\s+Plate\s+\d+\s*[\.:,]?\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    pairs: list[CaptionPair] = []
    seen_labels: set[str] = set()
    for m in _CAPTION_CLAUSE_RE.finditer(text):
        labels_raw = m.group(1)
        species = (m.group(2) or "").strip()
        modifier = (m.group(3) or "").strip()
        if not species:
            continue
        # Drop "Genus & species indet." type names (not a binomial).
        if "indet" in species.lower() or "& species" in species.lower():
            continue
        labels = _regex_expand_label_list(labels_raw)
        if not labels:
            continue
        # Skip duplicates of labels (same label assigned to two species)
        if any(lbl in seen_labels for lbl in labels):
            continue
        for lbl in labels:
            seen_labels.add(lbl)
        pairs.append(CaptionPair(
            labels=labels,
            species=species,
            modifier=modifier,
            confidence=0.7,
            notes="regex_fallback",
            raw_text=m.group(0)[:120],
        ))

    # Pouille-style caption fallback: when the OpenDataLoader pass
    # couldn't find a real "Plate N" header (Pouille 2014 has none) it
    # reconstructs a synthetic caption shaped like
    #   "Plate 1. (Reconstructed from systematic descriptions)
    #    Syntagentactinia biocculosa (Pl. 1, figs 5)
    #    Syntagentactinia? angulata n. sp. (Pl. 1, figs 12–14b) ..."
    # where the species comes BEFORE the "(Pl. N, figs M)" reference.
    # _CAPTION_CLAUSE_RE only matches the inverse "Fig. N. Species" form,
    # so without this pass the regex parser returns zero pairs and the
    # order-based fallback tags every panel with taxa[0].
    seen_lines: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line in seen_lines:
            continue
        seen_lines.add(line)
        m = _POUILE_CLAUSE_RE.match(line)
        if not m:
            continue
        species = m.group(1).strip()
        modifier = (m.group(2) or "").strip()
        labels_raw = m.group(3)
        # Filter: skip the synthetic header line "Plate N. (Reconstructed ...)"
        # (matched by the regex when the only "fig" mention is in the
        # reconstructed-from marker) and obvious non-species rows.
        if "Reconstructed" in species or "Reconstructed" in modifier:
            continue
        if "indet" in species.lower() or "& species" in species.lower():
            continue
        labels = _regex_expand_label_list(labels_raw)
        if not labels:
            continue
        # Dedup: skip if any of the expanded labels is already assigned
        # to an earlier species. We dedup the base number (e.g. "14"
        # from "14b") so a later clause that mentions "fig 14" can be
        # safely skipped.
        base_labels: set[str] = set()
        for lbl in labels:
            base = re.match(r"(\d+)", lbl)
            if base:
                base_labels.add(base.group(1))
        if base_labels & seen_labels:
            continue
        # Also: if any expanded label was already used as the base of
        # another species, skip. This handles "fig 14" coming after
        # "figs 12-14b" — "14" is the base of "14b" which was already
        # taken.
        if any(lbl in seen_labels for lbl in base_labels):
            continue
        for lbl in labels:
            seen_labels.add(lbl)
        # And remember the base numbers so subsequent "fig 14" mentions
        # don't get assigned too.
        for lbl in base_labels:
            seen_labels.add(lbl)
        pairs.append(CaptionPair(
            labels=labels,
            species=species,
            modifier=modifier,
            confidence=0.65,
            notes="regex_fallback_pouille",
            raw_text=line[:120],
        ))
    return pairs


# ---------------------------------------------------------------------------
# Data classes for stage I/O
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CaptionPair:
    """A parsed (label-set -> species) clause from a caption."""

    labels: list[str]            # e.g. ["A", "B"] or ["3", "4"]
    species: str                 # canonical Latin name
    modifier: str = ""           # "sp.", "cf.", "aff.", "?", "n. sp."
    confidence: float = 0.9      # M3's self-assessed parse confidence
    notes: str = ""              # optional parsing notes
    raw_text: str = ""           # original caption span that produced this pair

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlateClassification:
    """M3's view of the entire plate."""

    is_radiolarian_plate: bool = True
    image_type: str = "micrograph"   # "micrograph" | "SEM" | "photomicrograph" | "diagram" | "photo" | "other"
    panel_count_estimate: int | None = None
    specimen_count_estimate: int | None = None
    quality: str = "ok"              # "good" | "ok" | "poor"
    dominant_taxa: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PanelBox:
    """M3's view of an individual panel within the plate."""

    panel_id: str
    bbox: tuple[int, int, int, int]   # (x, y, w, h) in plate pixel coordinates
    visible_label: str | None = None  # e.g. "A" if M3 sees the letter on the panel
    morphology: str = ""              # one-line morphology hint
    confidence: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


@dataclass(slots=True)
class PanelMatch:
    """M3's per-panel species assignment."""

    panel_id: str
    label: str | None
    species: str | None
    confidence: float
    reasoning: str
    alternative: str | None = None
    is_radiolarian: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Critique:
    """M3 self-critique of an existing per-panel match."""

    panel_id: str
    verdict: str            # "agree" | "disagree" | "uncertain"
    suggested_species: str | None = None
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage prompts (Chinese; designed for extended-thinking M3)
# ---------------------------------------------------------------------------

_PARSE_CAPTION_SYSTEM = """你是放射虫古生物学专家，专长是从图版说明（caption）中抽取"图版label-拉丁学名"映射。

任务：给定一段非结构化的图版说明文字，输出所有 (label集合 → 物种) 配对。

输出规则：
1. 严格 JSON 数组，每项一个配对。
2. labels: 字符串数组，按图版印刷顺序排。处理 A,B / A-D / 3,4 / 3-5 / A-C, 4 等形式。
3. species: 拉丁学名（属 + 种加词，可含 sp. / cf. / aff. / n. sp. 等修饰）。
4. modifier: 单独的修饰标记（"sp."/"cf."/"aff."/"n. sp."/"?"/""）。
5. confidence: 0-1，反映你对自己解析的把握。
6. notes: 简短的解析说明（"括注=scale 50μm, 忽略"等），没有则空串。
7. raw_text: 产生该配对的原文片段；找不到则空串。

示例输入：
"图3 扫描电镜照片。A-D: Tetraspongodiscus stauracanthus n. sp.; E, F: Falcispongus scalaris sp. nov. Scale bars = 50 μm in A, C; 30 μm in B, D-F."

示例输出：
[{"labels":["A","B","C","D"],"species":"Tetraspongodiscus stauracanthus","modifier":"n. sp.","confidence":0.97,"notes":"","raw_text":"A-D: Tetraspongodiscus stauracanthus n. sp."},{"labels":["E","F"],"species":"Falcispongus scalaris","modifier":"sp. nov.","confidence":0.95,"notes":"","raw_text":"E, F: Falcispongus scalaris sp. nov."}]

只输出 JSON 数组，不要任何解释文本。"""


_CLASSIFY_PLATE_SYSTEM = """你是放射虫图版审查员。请判断一张给定的图像是否是一张"含放射虫标本的图版（plate）"。

任务：观察图像并返回严格 JSON。

判定要点：
1. is_radiolarian_plate: true/false。
   - true  = 主要内容是放射虫扫描电镜/光镜照片，含 1 个或多个标本 panel。
   - false = 是普通正文页、参考文献、目录、章节标题、方程、地图、柱状图、相图、表格、流程图、纯文字页等。
2. image_type: "SEM" / "micrograph" / "photomicrograph" / "diagram" / "photo" / "other"。
3. panel_count_estimate: 你能看到的独立 panel（带 A/B/C 标签的小图）数量，估不出则 null。
4. specimen_count_estimate: 估计的标本总数；估不出则 null。
5. quality: "good" / "ok" / "poor" — 图像清晰度/对比度/裁剪质量。
6. dominant_taxa: 若可识别，列出 1-3 个最可能的属名（识别不出则空数组）。
7. reasoning: 1-2 句解释你为什么这么判断。

输出示例（是放射虫图版）：
{"is_radiolarian_plate":true,"image_type":"SEM","panel_count_estimate":6,"specimen_count_estimate":8,"quality":"good","dominant_taxa":["Tetraspongodiscus"],"reasoning":"看到 6 个带 A-F 标签的 SEM 标本 panel，对比度高。"}

输出示例（非图版）：
{"is_radiolarian_plate":false,"image_type":"other","panel_count_estimate":null,"specimen_count_estimate":null,"quality":"ok","dominant_taxa":[],"reasoning":"这是一段正文的标题页，'Applications' 字样，无标本图像。"}

只输出严格 JSON，不要解释。"""


_SEGMENT_PANELS_SYSTEM = """你是放射虫图版版面分析专家。任务：从一张放射虫图版中识别出每个独立 panel（每个 panel 通常带一个 A/B/C/D...字母标签）的边界框。

输出规则（严格 JSON 数组）：
1. panel_id: 字符串，从可见标签读取，例如 "A"、"B"；若没有标签，用 "P1"/"P2"...
2. bbox: [x, y, w, h]，像素坐标，原点在左上。
3. visible_label: 看到的标签字符串；看不到则 null。
4. morphology: 一句话描述这个 panel 里标本的形态特征（"球形壳，三主轴，格状壁孔"等）。
5. confidence: 0-1，反映你对自己框选和形态描述的把握。

重要：
- 框必须紧贴标本内容（包括 scale bar 旁边的内容，但裁掉纯白边）。
- 多个 panel 之间不要重叠。
- 如果整图就一个 panel，输出单元素数组覆盖整图（bbox=[0,0,W,H]）。
- 若图版里有箭头、字母标签、scale bar，把它们一起包进对应 panel 框内。
- 不要包含图说文字（caption 文字在图版外面）。

只输出 JSON 数组，不要解释。"""


_MATCH_PANEL_SYSTEM_VISUAL_ONLY = """你是放射虫古生物学专家，负责为单个 panel 标本做形态学鉴定（visual-only 模式）。

场景：图版说明（caption）未能被提取，**没有候选物种清单**。请完全依靠 panel 图像本身的形态特征做最可能的分类。

任务：观察一个 panel 图像 + 整张图版（用于上下文） + 可选可见字母标签，输出最可能的属/种鉴定。

输出（严格 JSON）：
1. label: 该 panel 的字母标签（如有 "A"），否则用 "?"。
2. species: 拉丁学名（属 + 种加词）；若是新种、未知种或仅可鉴定到属，给出属名 + " sp." 或 "sp. indet."；若完全无法鉴定，设为 null。
3. confidence: 0-1，反映判定的把握。**visual-only 模式请保守**：conf >= 0.6 仅在你对形态非常确定时给出。
4. reasoning: 1-2 句说明形态依据（"球形壳 + 6 根对称主刺 + 格状壳壁 → Archaeosemantis 形态类型" 等）。
5. alternative: 第二可能的物种；无则 null。
6. is_radiolarian: true/false — 这真的是放射虫吗？若 false 则 species 设为 null。

注意：
- 不要虚构从未在公开文献中记录的物种。
- 形态不清晰时倾向给属名 + sp.，并降低 confidence。
- 若图中可见 A/B/C 字母标签，原样输出。

只输出严格 JSON。"""


_MATCH_PANEL_SYSTEM = """你是放射虫古生物学专家，负责为单个 panel 标本指定其最可能的分类（label + 拉丁学名）。

任务：观察一个 panel 图像 + 候选标签-物种配对列表 + 整张图版说明，输出最可能的配对及推理。

输入：
- 图像：一个 panel 标本的裁剪图。
- 候选配对：从图版说明中解析出的 (label集合 → 物种) 列表（已去重、按字母顺序排好）。
- 提示标签：M3 之前在图版上看到的可见字母（如 "A"），可能为 null。
- 完整图说：作为辅助上下文。

输出（严格 JSON）：
1. label: 你判定的 panel 对应的字母（A/B/C...）；若该 panel 不属于任何候选配对，给出最可能的字母或 null。
2. species: 拉丁学名。
3. confidence: 0-1，反映判定的把握。
4. reasoning: 1-2 句解释（"caption 中 A-B 配 X；图上标签为 A；形态与 X 一致" 等）。
5. alternative: 第二可能的物种；无则 null。
6. is_radiolarian: true/false — 这真的是放射虫吗？若 false 则 species 设为 null。

判定优先级：
1) 图上可见字母标签（最高）。
2) caption 中明确写出的 label-物种 子句。
3) 形态/语义一致性（最低；只能用于在多个候选间 tie-break）。

注意：
- 引用候选配对中的物种名时，**保持原始拼写**（包括大小写、空格）。
- 候选配对为空时，species 设为 null 并降低 confidence。
- 不要凭空编造从未在 caption 出现过的物种名。

只输出严格 JSON。"""


_CRITIQUE_SYSTEM = """你是放射虫分类学审查员。任务：交叉验证一组已经做出的 panel→物种配对，纠正明显错误。

输入：
- 完整图版图像。
- 已配对结果：每个 panel 的 (panel_id, label, species, confidence, reasoning)。
- 图版说明：caption 原文。

请对**每个 panel** 给出评判（严格 JSON 数组，每项一个）：
1. panel_id: 字符串（与输入一致）。
2. verdict: "agree" / "disagree" / "uncertain"。
3. suggested_species: 若 disagree/uncertain 且有更合理候选，给出拉丁名；否则 null。
4. confidence: 0-1，反映你的判断把握。
5. reasoning: 1 句解释。

判定规则：
- 如果原配对 species 与 caption 中同一 label 的候选一致 → agree。
- 如果 caption 中该 label 应是 X，但你看到图上形态明显属于 Y（如球形 vs 钟形）→ disagree 并给 Y。
- 如果信息不足判断 → uncertain，suggested_species 可给一个第二可能。
- 若 panel 不是放射虫（is_radiolarian=false）→ 直接 agree 不要改。

只输出严格 JSON 数组。"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class M3Engine:
    """M3-Centric 5-stage engine.

    Usage
    -----
    >>> engine = M3Engine(backend, config={"m3_stage_1": True, ...})
    >>> pairs = engine.parse_caption(caption_text)
    >>> cls = engine.classify_plate(plate_image)
    >>> if not cls.is_radiolarian_plate:
    ...     return []
    >>> panels = engine.segment_panels(plate_image)
    >>> matches = [engine.match_panel(p.image, pairs, p.visible_label) for p in panels]
    >>> critiques = engine.critique_matches(plate_image, matches, caption_text)
    >>> final = engine.apply_critiques(matches, critiques)
    """

    def __init__(self, backend: Any, config: dict[str, Any] | None = None) -> None:
        self.backend = backend
        self.config = dict(config or {})
        # Stage toggles. Default: all on.
        for i in range(1, 6):
            self.config.setdefault(f"m3_stage_{i}", True)
        # Temperature overrides for stages that need more creative reasoning.
        self.config.setdefault("m3_temperature", 0.1)
        # Self-consistency: re-sample stage 4 N times and majority-vote.
        self.config.setdefault("m3_match_samples", 1)
        # Thinking budget for vision stages (more thinking for harder visual reasoning).
        self.config.setdefault("m3_thinking_budget", 1024)
        # Skip stage-4 per-panel matching if caption parser found zero pairs.
        # If False, stage 4 still runs in a "visual-only" mode where M3
        # identifies the specimen from morphology alone (lower confidence).
        self.config.setdefault("m3_skip_match_on_empty_caption", False)
        # Diagnostic dump: also save M3 raw output to this directory (None = off).
        self.config.setdefault("m3_diagnostic_dir", None)
        self._diagnostic_counter = 0

    # ------------------------------------------------------------------ stage 1
    def parse_caption(self, caption_text: str) -> list[CaptionPair]:
        """Stage 1: caption text -> structured (label, species) pairs.

        Tries the LLM first; if the LLM returns nothing (rate-limited, low
        quality, model errors), falls back to a regex-based parser that
        handles the most common caption formats:
            "fig 1. Species A" / "figs 1-3. Species B" / "fig 1, 4. Species C"
        """
        if not self._stage_enabled(1) or not caption_text or not caption_text.strip():
            return []
        # Configurable: skip the LLM and go straight to the regex parser.
        # Useful for tests and for cost-sensitive runs where the regex is
        # accurate enough for the caption convention at hand.
        if self.config.get("m3_caption_regex_only", False):
            fallback = _regex_parse_caption(caption_text)
            if fallback:
                logger.info("Stage 1 parse_caption -> %d pairs (regex only, m3_caption_regex_only=True)", len(fallback))
            return fallback
        prompt = (
            "请解析下列图版说明，输出严格的 JSON 数组（label->物种 配对列表）。"
            "\n\n[Caption]\n" + caption_text.strip() + "\n\n[输出 JSON]"
        )
        raw = self._infer_text(_PARSE_CAPTION_SYSTEM, prompt)
        if not raw or raw.get("fallback_used"):
            return []
        try:
            data = _safe_json_loads(raw.get("raw_text", ""))
        except Exception as exc:
            logger.warning("Stage 1 JSON parse failed: %s", exc)
            return []
        pairs: list[CaptionPair] = []
        if not isinstance(data, list):
            return []
        for item in data:
            if not isinstance(item, dict):
                continue
            labels = item.get("labels") or []
            species = str(item.get("species") or "").strip()
            if not labels or not species:
                continue
            if isinstance(labels, str):
                labels = _expand_label_range(labels)
            elif not isinstance(labels, list):
                continue
            labels = [str(x).strip() for x in labels if str(x).strip()]
            if not labels:
                continue
            try:
                conf = float(item.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            pairs.append(
                CaptionPair(
                    labels=labels,
                    species=species,
                    modifier=str(item.get("modifier") or "").strip(),
                    confidence=max(0.0, min(1.0, conf)),
                    notes=str(item.get("notes") or "").strip(),
                    raw_text=str(item.get("raw_text") or "").strip(),
                )
            )
        logger.info("Stage 1 parse_caption -> %d pairs", len(pairs))
        if pairs:
            return pairs
        # Fallback: regex-based caption parser. Avoids the cost (and flakiness)
        # of an LLM call when the caption is well-formed enough to parse
        # deterministically. Handles the convention used in most OA radiolarian
        # papers: ``figs 1-3. Species A. figs 4-5. Species B.``.
        fallback = _regex_parse_caption(caption_text)
        if fallback:
            logger.info("Stage 1 parse_caption -> %d pairs (regex fallback)", len(fallback))
        return fallback

    # ------------------------------------------------------------------ stage 2
    def classify_plate(self, plate_image: Image.Image) -> PlateClassification:
        """Stage 2: plate image -> semantic classification (filter non-plate)."""
        if not self._stage_enabled(2) or plate_image is None:
            return PlateClassification()
        prompt = (
            "观察这张图像，判断它是否是放射虫图版。"
            "严格输出 JSON（is_radiolarian_plate / image_type / panel_count_estimate / "
            "specimen_count_estimate / quality / dominant_taxa / reasoning）。"
        )
        raw = self._infer_vision(_CLASSIFY_PLATE_SYSTEM, prompt, plate_image)
        if not raw or raw.get("fallback_used"):
            return PlateClassification()
        try:
            data = _safe_json_loads(raw.get("raw_text", ""))
        except Exception as exc:
            logger.warning("Stage 2 JSON parse failed: %s", exc)
            return PlateClassification()
        if not isinstance(data, dict):
            return PlateClassification()
        cls = PlateClassification(
            is_radiolarian_plate=bool(data.get("is_radiolarian_plate", True)),
            image_type=str(data.get("image_type") or "micrograph"),
            panel_count_estimate=_safe_int(data.get("panel_count_estimate")),
            specimen_count_estimate=_safe_int(data.get("specimen_count_estimate")),
            quality=str(data.get("quality") or "ok"),
            dominant_taxa=[str(x) for x in (data.get("dominant_taxa") or []) if str(x).strip()],
            reasoning=str(data.get("reasoning") or "").strip(),
        )
        logger.info(
            "Stage 2 classify_plate -> is_radiolarian=%s type=%s panels≈%s taxa=%s",
            cls.is_radiolarian_plate, cls.image_type, cls.panel_count_estimate, cls.dominant_taxa,
        )
        return cls

    # ------------------------------------------------------------------ stage 3
    def segment_panels(self, plate_image: Image.Image, hint_count: int | None = None) -> list[PanelBox]:
        """Stage 3: plate image -> panel bboxes + visible labels."""
        if not self._stage_enabled(3) or plate_image is None:
            return []
        w, h = plate_image.size
        hint = f"\n\n提示：预期 panel 数量约 {hint_count}。\n" if hint_count else ""
        prompt = (
            f"观察这张放射虫图版（{w}x{h} 像素），识别每个独立 panel 的边界框 + 可见字母标签。"
            f"{hint}"
            "严格输出 JSON 数组（panel_id / bbox / visible_label / morphology / confidence）。"
        )
        raw = self._infer_vision(_SEGMENT_PANELS_SYSTEM, prompt, plate_image)
        if not raw or raw.get("fallback_used"):
            return []
        try:
            data = _safe_json_loads(raw.get("raw_text", ""))
        except Exception as exc:
            logger.warning("Stage 3 JSON parse failed: %s", exc)
            return []
        if not isinstance(data, list):
            return []
        panels: list[PanelBox] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            bbox = _coerce_bbox(item.get("bbox"), w, h)
            if bbox is None:
                continue
            try:
                conf = float(item.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            panels.append(
                PanelBox(
                    panel_id=str(item.get("panel_id") or f"P{len(panels)+1}"),
                    bbox=bbox,
                    visible_label=(str(item.get("visible_label")).strip()
                                   if item.get("visible_label") is not None else None),
                    morphology=str(item.get("morphology") or "").strip(),
                    confidence=max(0.0, min(1.0, conf)),
                )
            )
        # Sort top-to-bottom, left-to-right and re-assign sequential ids if none.
        panels.sort(key=lambda p: (p.bbox[1], p.bbox[0]))
        for i, p in enumerate(panels, start=1):
            if not p.panel_id or p.panel_id == f"P{i-1}":
                p.panel_id = f"P{i}"
        logger.info("Stage 3 segment_panels -> %d panels", len(panels))
        return panels

    # ------------------------------------------------------------------ stage 4
    def match_panel(
        self,
        panel_image: Image.Image,
        caption_pairs: list[CaptionPair],
        caption_text: str = "",
        suggested_label: str | None = None,
    ) -> PanelMatch:
        """Stage 4: panel image + caption pairs -> (label, species, confidence).

        Two modes:
          - With caption pairs: standard M3 matcher with constrained candidates.
          - Visual-only (no caption): M3 does morphology-based identification
            with a separate prompt and conservative confidence.
        """
        if not self._stage_enabled(4) or panel_image is None:
            return PanelMatch(
                panel_id="?", label=None, species=None, confidence=0.0,
                reasoning="stage 4 disabled",
            )
        visual_only = not caption_pairs
        if visual_only:
            system_prompt = _MATCH_PANEL_SYSTEM_VISUAL_ONLY
            hint = f"\n提示标签（来自 M3 阶段 3）：{suggested_label}\n" if suggested_label else ""
            caption_block = f"\n[完整图说（仅供参考，可能为空）]\n{caption_text.strip()}\n" if caption_text else ""
            prompt = (
                f"{caption_block}"
                f"{hint}\n"
                "模式：visual-only。无候选物种清单，请完全依靠形态特征鉴定。\n"
                "请为该 panel 选出最可能的属/种，严格输出 JSON。"
            )
        else:
            system_prompt = _MATCH_PANEL_SYSTEM
            pairs_json = json.dumps([p.to_dict() for p in caption_pairs], ensure_ascii=False, indent=2)
            hint = f"\n提示标签（来自 M3 阶段 3）：{suggested_label}\n" if suggested_label else ""
            caption_block = f"\n[完整图说]\n{caption_text.strip()}\n" if caption_text else ""
            prompt = (
                "[候选配对（caption 解析）]\n"
                f"{pairs_json}\n"
                f"{caption_block}"
                f"{hint}\n"
                "请为该 panel 选出最可能的 label + 物种，严格输出 JSON。"
            )
        # Optional self-consistency: sample N times at higher temperature and vote.
        n_samples = max(1, int(self.config.get("m3_match_samples", 1)))
        results: list[dict[str, Any]] = []
        last_error: str | None = None
        for _ in range(n_samples):
            raw = self._infer_vision(system_prompt, prompt, panel_image)
            if not raw or raw.get("fallback_used"):
                # Capture the backend error so we can distinguish a real API
                # failure from "M3 said this isn't a radiolarian". The pipeline
                # uses raw["error"] to route through the FallbackHandler.
                if raw and raw.get("error"):
                    last_error = str(raw.get("error"))
                continue
            try:
                data = _safe_json_loads(raw.get("raw_text", ""))
            except Exception:
                continue
            if isinstance(data, dict):
                results.append(data)
        if not results:
            return PanelMatch(
                panel_id="?", label=None, species=None, confidence=0.0,
                reasoning=(f"M3 error: {last_error}" if last_error
                           else "M3 returned no parseable output"),
                is_radiolarian=False,
                raw={"error": last_error} if last_error else {},
            )
        # Majority-vote on (label, species); keep best confidence.
        votes: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
        for r in results:
            key = (r.get("label"), r.get("species"))
            votes.setdefault(key, []).append(r)
        best_key, best_group = max(votes.items(), key=lambda kv: len(kv[1]))
        best = max(best_group, key=lambda r: float(r.get("confidence") or 0))
        try:
            conf = float(best.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        # If there's a runner-up, surface as alternative.
        runner_up: str | None = None
        if len(votes) > 1:
            sorted_groups = sorted(votes.values(), key=lambda g: -len(g))
            if len(sorted_groups) > 1:
                ru = max(sorted_groups[1], key=lambda r: float(r.get("confidence") or 0))
                ru_sp = str(ru.get("species") or "").strip() or None
                if ru_sp and ru_sp != best.get("species"):
                    runner_up = ru_sp
        return PanelMatch(
            panel_id="?",
            label=(str(best.get("label") or "").strip() or None),
            species=(str(best.get("species") or "").strip() or None),
            confidence=max(0.0, min(1.0, conf)),
            reasoning=str(best.get("reasoning") or "").strip(),
            alternative=runner_up,
            is_radiolarian=bool(best.get("is_radiolarian", True)),
            raw={"votes": len(results), "agreement": len(best_group) / max(1, len(results))},
        )

    # ------------------------------------------------------------------ stage 5
    def critique_matches(
        self,
        plate_image: Image.Image,
        matches: list[PanelMatch],
        caption_text: str = "",
        caption_pairs: list[CaptionPair] | None = None,
    ) -> list[Critique]:
        """Stage 5: full plate + per-panel matches -> critique / override list."""
        if not self._stage_enabled(5) or plate_image is None or not matches:
            return []
        matches_json = json.dumps(
            [
                {
                    "panel_id": m.panel_id,
                    "label": m.label,
                    "species": m.species,
                    "confidence": m.confidence,
                    "reasoning": m.reasoning,
                }
                for m in matches
            ],
            ensure_ascii=False, indent=2,
        )
        cap_block = f"\n[图说]\n{caption_text.strip()}\n" if caption_text else ""
        pairs_block = ""
        if caption_pairs:
            pairs_block = (
                "\n[图说解析出的候选配对]\n"
                + json.dumps([p.to_dict() for p in caption_pairs], ensure_ascii=False, indent=2)
                + "\n"
            )
        prompt = (
            "[当前配对结果]\n"
            f"{matches_json}\n"
            f"{cap_block}"
            f"{pairs_block}\n"
            "请逐个 panel 评判（agree / disagree / uncertain），必要时给出 suggested_species。严格 JSON 数组。"
        )
        raw = self._infer_vision(_CRITIQUE_SYSTEM, prompt, plate_image)
        if not raw or raw.get("fallback_used"):
            return []
        try:
            data = _safe_json_loads(raw.get("raw_text", ""))
        except Exception as exc:
            logger.warning("Stage 5 JSON parse failed: %s", exc)
            return []
        if not isinstance(data, list):
            return []
        out: list[Critique] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict") or "").strip().lower()
            if verdict not in {"agree", "disagree", "uncertain"}:
                verdict = "uncertain"
            try:
                conf = float(item.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            out.append(
                Critique(
                    panel_id=str(item.get("panel_id") or ""),
                    verdict=verdict,
                    suggested_species=(str(item.get("suggested_species") or "").strip() or None),
                    confidence=max(0.0, min(1.0, conf)),
                    reasoning=str(item.get("reasoning") or "").strip(),
                )
            )
        logger.info("Stage 5 critique_matches -> %d critiques", len(out))
        return out

    # ------------------------------------------------------------------ apply
    @staticmethod
    def apply_critiques(
        matches: list[PanelMatch],
        critiques: list[Critique],
        override_threshold: float = 0.6,
    ) -> list[PanelMatch]:
        """Apply critiques to matches in place. Returns the same list (mutated)."""
        by_id: dict[str, Critique] = {c.panel_id: c for c in critiques}
        for m in matches:
            c = by_id.get(m.panel_id)
            if not c:
                continue
            if c.verdict == "agree":
                m.raw.setdefault("critique", {"verdict": "agree", "confidence": c.confidence})
                continue
            if c.suggested_species and c.confidence >= override_threshold:
                m.raw["critique"] = {
                    "verdict": c.verdict,
                    "from": m.species,
                    "to": c.suggested_species,
                    "confidence": c.confidence,
                    "reasoning": c.reasoning,
                }
                m.species = c.suggested_species
                # Lower confidence since we overrode
                m.confidence = min(m.confidence, max(0.3, c.confidence))
        return matches

    # ------------------------------------------------------------------ helpers
    def _stage_enabled(self, n: int) -> bool:
        return bool(self.config.get(f"m3_stage_{n}", True))

    def _infer_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.backend is None:
            return {"fallback_used": True, "error": "no backend"}
        try:
            res = self.backend.infer_text(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            logger.exception("M3 text inference failed")
            return {"fallback_used": True, "error": str(exc)}
        # Retry without thinking if the response is empty.
        if (
            self.config.get("m3_retry_without_thinking", True)
            and (res.get("fallback_used") or not (res.get("raw_text") or "").strip())
            and getattr(self.backend, "enable_thinking", False)
        ):
            logger.info("M3 text returned empty; retrying with thinking disabled")
            saved = self.backend.enable_thinking
            try:
                self.backend.enable_thinking = False
                res2 = self.backend.infer_text(
                    system_prompt=system_prompt, user_prompt=user_prompt,
                )
            except Exception as exc:
                logger.warning("M3 text retry failed: %s", exc)
                res2 = res
            finally:
                self.backend.enable_thinking = saved
            if (res2.get("raw_text") or "").strip():
                res = res2
        return res

    def _infer_vision(
        self, system_prompt: str, user_prompt: str, image: Image.Image
    ) -> dict[str, Any]:
        if self.backend is None:
            return {"fallback_used": True, "error": "no backend"}
        # First attempt — with thinking enabled (the default).
        try:
            res = self.backend.infer_panel(
                panel_image=image,
                caption_text="",  # we put context in user_prompt
                ocr_labels=[],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.exception("M3 vision inference failed")
            return {"fallback_used": True, "error": str(exc)}
        self._maybe_dump_diagnostic(image, system_prompt, user_prompt, res)
        # Retry without thinking if the first attempt produced no text
        # (known M3 issue when thinking exhausts the output budget).
        if (
            self.config.get("m3_retry_without_thinking", True)
            and (res.get("fallback_used") or not (res.get("raw_text") or "").strip())
            and getattr(self.backend, "enable_thinking", False)
        ):
            logger.info("M3 returned empty text; retrying with thinking disabled")
            saved = self.backend.enable_thinking
            try:
                self.backend.enable_thinking = False
                res2 = self.backend.infer_panel(
                    panel_image=image,
                    caption_text="",
                    ocr_labels=[],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as exc:
                logger.warning("M3 retry without thinking failed: %s", exc)
                res2 = res
            finally:
                self.backend.enable_thinking = saved
            if (res2.get("raw_text") or "").strip():
                res = res2
        return res

    def _maybe_dump_diagnostic(
        self, image: Image.Image, system_prompt: str, user_prompt: str, result: dict[str, Any]
    ) -> None:
        out_dir = self.config.get("m3_diagnostic_dir")
        if not out_dir:
            return
        try:
            from pathlib import Path
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            self._diagnostic_counter += 1
            idx = self._diagnostic_counter
            image.save(p / f"img_{idx:04d}.png")
            with (p / f"call_{idx:04d}.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "result": result,
                    },
                    f, ensure_ascii=False, indent=2, default=str,
                )
        except Exception as exc:
            logger.debug("Diagnostic dump failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LABEL_RANGE_RE = re.compile(r"^\s*([A-Za-z0-9]+)\s*[-–~]\s*([A-Za-z0-9]+)\s*$")


def _expand_label_range(s: str) -> list[str]:
    """Expand 'A-D' / '3-5' to ['A','B','C','D']; pass through if not a range."""
    m = _LABEL_RANGE_RE.match(s)
    if not m:
        return [s.strip()] if s.strip() else []
    a, b = m.group(1), m.group(2)
    # Letters
    if len(a) == 1 and len(b) == 1 and a.isalpha() and b.isalpha():
        if a.isupper() and b.isupper():
            return [chr(c) for c in range(ord(a), ord(b) + 1)]
        if a.islower() and b.islower():
            return [chr(c) for c in range(ord(a), ord(b) + 1)]
    # Digits
    if a.isdigit() and b.isdigit():
        return [str(i) for i in range(int(a), int(b) + 1)]
    return [s]


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _coerce_bbox(v: Any, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """Coerce M3 bbox output (often normalized 0-1) to absolute pixel coords."""
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        nums = [float(x) for x in v]
    except Exception:
        return None
    # Detect normalization: all in [0, 1] and image larger.
    if max(nums) <= 1.01 and (img_w > 100 or img_h > 100):
        x, y, w, h = nums
        return (
            max(0, int(x * img_w)),
            max(0, int(y * img_h)),
            min(img_w - max(0, int(x * img_w)), max(1, int(w * img_w))),
            min(img_h - max(0, int(y * img_h)), max(1, int(h * img_h))),
        )
    return (
        max(0, int(nums[0])),
        max(0, int(nums[1])),
        max(1, int(nums[2])),
        max(1, int(nums[3])),
    )
