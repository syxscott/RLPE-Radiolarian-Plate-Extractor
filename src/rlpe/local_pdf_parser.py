"""Offline PDF parser that fills the role of GROBID + OCR + TaxoNERD
using only ``pdfplumber`` / ``pypdf`` (already in the environment).

This module parses a radiolarian / palaeontology PDF into the same
data shape GROBID would produce, so the rest of the pipeline can be
source-agnostic. It is the **default** extractor on environments
without GROBID / OCR / TaxoNERD installed (i.e. our dev box).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import CaptionEntity, CaptionRecord, PaperMetadata

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LocalParserResult:
    """Source-agnostic equivalent of :class:`rlpe.grobid.GrobidResult`."""

    paper_id: str
    pdf_path: Path
    captions: list[CaptionRecord] = field(default_factory=list)
    fulltext_sections: list[dict[str, str]] = field(default_factory=list)
    paper_metadata: PaperMetadata | None = None
    success: bool = True
    error: str | None = None


# ---- heuristic patterns ---------------------------------------------------

_FIG_CAPTION_RE = re.compile(
    r"(?P<fig>Fig(?:ure|\.)?\s*(?P<num>\d+[A-Za-z]?)\s*[\.:]\s*)"
    r"(?P<cap>(?:[^\n]|\n(?!\s*(?:Fig(?:ure|\.)?\s*\d|Diagnosis|"
    r"Description|Remarks|Etymology|Systematic\s+Palaeontology|"
    r"Systematic\s+[Pp]aleontology|Material\s+examined|Type\s+material|"
    r"Holotype|Paratypes|Type\s+specimen))[^\n])*)",
    re.MULTILINE,
)

_SECTION_HEAD_RE = re.compile(
    r"^\s*(?P<title>(?:Systematic\s+(?:[Pp]aleontology|[Pp]alaeontology))"
    r"|(?:Systematics?)|(?:Geological\s+[Ss]etting)|(?:Geological\s+[Ss]tratigraphy)"
    r"|(?:Materials?\s+and\s+[Mm]ethods?)|(?:Methods?)|(?:Introduction)"
    r"|(?:Material)|(?:Results?)|(?:Discussion)|(?:Conclusions?)|(?:References?)|"
    r"(?:Acknowledg\w*)|(?:Stratigraphy)|(?:Locality)|(?:Type\s+material))\s*\.?\s*$",
    re.MULTILINE,
)

# Strict species regex: ``Genus species`` with both words >= 4 letters.
# Optional ``gen. et sp. nov.`` / ``sp. nov.`` / ``cf.`` / ``aff.`` suffix.
_SPECIES_RE = re.compile(
    r"\b([A-Z][a-z]{3,}\s+[a-z][a-zA-Z\-]{3,})"
    r"(?:\s+(?:sp\.\s*nov\.|sp\.|spp\.|cf\.|aff\.|gen\.\s*et\s*sp\.\s*nov\.))?"
)

# Common false positives to filter out after species extraction.
_SPECIES_DENYLIST: set[str] = {
    "Spongy cortical", "Spongy tissue", "Central part",
    "Tetraspongodiscus stauracan", "Terminology fol",
    "Although micro", "Dalongicaepa high",
    "Rencunping section", "South China", "Dalong Formation", "West Texas",
    "Acta Palaeontologica", "Type material", "Type species", "Type genus",
    "Acta Palaeontologica Polonica", "Journal Paleontology",
}

# Drop standalone numerals (often panel subscripts). Applied to flatten.
_NOISE_NUMERALS_RE = re.compile(r"\b\d{1,2}\b(?:\s+\b\d{1,2}\b)*")

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _infer_section_type(title: str) -> str:
    t = (title or "").lower()
    if "systematic" in t or "palaeontol" in t or "paleontol" in t:
        return "systematic_paleontology"
    if "geological" in t or "stratigr" in t or "setting" in t:
        return "geological_setting"
    if "material" in t or "method" in t:
        return "materials_methods"
    return "other"


def _stable_id(pdf_path: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update(str(pdf_path.resolve()).encode("utf-8"))
    try:
        h.update(str(pdf_path.stat().st_size).encode())
    except OSError:
        pass
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Paper-level metadata
# ---------------------------------------------------------------------------


def _parse_title_and_authors(page1_lines: list[str]) -> tuple[str | None, list[str], int]:
    """Heuristic: in most journals the first ~4 lines of page 1 are
    the title (2-3 lines) followed by an ALL-CAPS author line.

    Returns ``(title, authors, title_line_count)``.
    """
    if not page1_lines:
        return None, [], 0
    title_lines: list[str] = []
    authors: list[str] = []
    title_line_count = 0
    i = 0
    # Skip leading empty lines
    while i < len(page1_lines) and not page1_lines[i].strip():
        i += 1
    # Collect title lines
    while i < len(page1_lines):
        ln = page1_lines[i].strip()
        if not ln:
            i += 1
            continue
        if re.match(r"^[A-Z]{2,}[\w\s,\.\-]+(?:,\s*[A-Z]\.){1,}", ln):
            authors = [ln]
            i += 1
            break
        if re.match(r"^[A-Z][a-z]+,\s*[A-Z]\.[\w\.,\s]*", ln) and " and " in ln:
            return None, [], 0
        if len(title_lines) >= 4:
            break
        if len(ln) > 250:
            break
        title_lines.append(ln)
        title_line_count += 1
        i += 1
        if i < len(page1_lines):
            nxt = page1_lines[i].strip()
            if re.match(r"^[A-Z]{2,}", nxt) and "," in nxt:
                authors = [nxt]
                title_line_count += 1
                break
    title = " ".join(title_lines).strip() if title_lines else None
    return title, authors, title_line_count


def _parse_abstract(page1_lines: list[str], start_at: int) -> str:
    """Return the abstract block on page 1, starting from
    ``start_at`` (which is the line right after title + author +
    citation).
    """
    body: list[str] = []
    i = start_at
    while i < len(page1_lines):
        ln = page1_lines[i].strip()
        if not ln:
            if body:
                break
            i += 1
            continue
        # Stop conditions
        low = ln.lower()
        if low.startswith("key words") or low.startswith("keywords"):
            break
        if "@" in ln and _EMAIL_RE.search(ln):
            break
        if re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+\s*[\[\(]", ln) and "," not in ln[:30]:
            break
        if "received " in low and re.search(r"\b(19|20)\d{2}\b", ln):
            break
        if re.match(r"^Copyright\s*\W", ln, re.IGNORECASE):
            break
        body.append(ln)
        i += 1
    text = " ".join(body)
    text = re.sub(r"\bKey words:.*$", "", text, flags=re.DOTALL).strip()
    return text


def _parse_paper_metadata(pages: list[dict[str, Any]], paper_id: str) -> PaperMetadata:
    meta = PaperMetadata(source="local_pdf_parser", confidence=0.55)
    if not pages:
        return meta
    page1_text = pages[0]["text"]
    page1_lines = page1_text.splitlines()
    title, authors, title_count = _parse_title_and_authors(page1_lines)
    meta.title = title
    meta.authors = authors
    # DOI / year
    combined = "\n".join(p["text"] for p in pages[:2])
    m = _DOI_RE.search(combined)
    if m:
        meta.doi = m.group(0).rstrip(".,;)")
    m = _YEAR_RE.search(combined)
    if m:
        try:
            meta.year = int(m.group(1))
        except ValueError:
            pass
    # Journal
    journal_keys = (
        "Acta Palaeontologica Polonica",
        "Journal of Paleontology",
        "Palaeontology",
        "Micropalaeontology",
        "Marine Micropaleontology",
        "Stratigraphy",
        "Bulletin",
    )
    for ln in page1_text.splitlines():
        if any(k in ln for k in journal_keys) and _YEAR_RE.search(ln):
            meta.journal = ln.strip()
            break
    if not meta.journal:
        # Fallback: first line that mentions a known journal name only
        for ln in page1_text.splitlines()[:8]:
            if any(k in ln for k in journal_keys):
                meta.journal = ln.strip()
                break
    # Abstract: skip title lines + author line + 1 cite line
    start_at = title_count + len(authors) + 1
    meta.abstract = _parse_abstract(page1_lines, start_at=start_at)
    # Confidence
    filled = sum(1 for k in ("title", "doi", "abstract", "year", "journal") if getattr(meta, k))
    if filled:
        meta.confidence = min(0.95, 0.4 + 0.12 * filled)
    return meta


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------


def _normalise_caption(text: str) -> str:
    # Merge hyphenated word-breaks: pdfplumber splits a hyphenated
    # word at the line break as "stauracan-" + "thus", which then turns
    # into "stauracanthus" if we just join. We DO want to keep the
    # hyphen if the second part is upper-case (real compound word), but
    # in 99% of cases the second part is lower-case continuation.
    text = re.sub(r"-\s+([a-z])", r"\1", text)
    # Drop short runs of 1-2 digit noise tokens (panel subscript numbers)
    text = re.sub(r"(?:\s*\b\d{1,2}\b){1,4}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_panel_labels_from_caption(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\(([A-H])(?:\s*[\u2013\-]\s*([A-H]))?\)", text):
        a, b = m.group(1), m.group(2)
        if b and a < b:
            for c in range(ord(a), ord(b) + 1):
                label = chr(c)
                if label not in seen:
                    seen.add(label)
                    out.append(label)
        else:
            if a not in seen:
                seen.add(a)
                out.append(a)
    for m in re.finditer(r"\((\d{1,2}[a-z]?)\)", text):
        lab = m.group(1)
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out


def _parse_captions(pages: list[dict[str, Any]], paper_id: str) -> list[CaptionRecord]:
    """Per-page line-based caption extraction.

    We do NOT flatten the page to a single line because radiolarian
    papers are often 2-column: the caption is followed (or preceded) by
    the diagnosis / description text on the same page. A naive flatten
    would let Fig. 3's caption swallow Fig. 4's caption.

    Strategy: for each page, walk the lines until we find one starting
    with ``Fig. N.`` / ``Figure N.``; capture the rest of that line
    plus subsequent non-empty lines until the next blank line, the
    next ``Fig. N``-prefixed line, or a known subheading (Diagnosis.,
    Description., etc.).
    """
    captions: list[CaptionRecord] = []
    seen_fig_nums: set[str] = set()
    subhead_re = re.compile(
        r"^(?:Diagnosis|Description|Remarks|Etymology|"
        r"Systematic\s+Palaeontology|Systematic\s+[Pp]aleontology|"
        r"Material\s+examined|Type\s+material|Holotype|Paratypes|"
        r"Type\s+specimen)\b",
        re.IGNORECASE,
    )
    next_fig_re = re.compile(r"^\s*Fig(?:ure|\.)?\s*\d+", re.IGNORECASE)
    for page in pages:
        lines = page["text"].splitlines()
        i = 0
        while i < len(lines):
            ln = lines[i]
            m = re.match(
                r"^\s*Fig(?:ure|\.)?\s*(?P<num>\d+[A-Za-z]?)\s*[\.:]\s*(?P<rest>.*)$",
                ln,
            )
            if not m:
                i += 1
                continue
            fig_n = m.group("num")
            if fig_n in seen_fig_nums:
                i += 1
                continue
            seen_fig_nums.add(fig_n)
            cap_parts: list[str] = [m.group("rest")] if m.group("rest").strip() else []
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt.strip():
                    break
                if next_fig_re.match(nxt):
                    break
                if subhead_re.match(nxt.strip()):
                    break
                cap_parts.append(nxt)
                j += 1
            cap_text_raw = " ".join(cap_parts)
            cap_text = _normalise_caption(cap_text_raw)
            if not cap_text or len(cap_text) < 20:
                i = j
                continue
            if len(cap_text) > 800:
                cap_text = cap_text[:800].rsplit("\n\n", 1)[0].strip()
            figure_id = f"local_p{page['index']:03d}_fig{fig_n}"
            entities = []
            for sm in _SPECIES_RE.finditer(cap_text):
                tok = sm.group(1).strip()
                if tok in _SPECIES_DENYLIST:
                    continue
                parts = tok.split()
                if len(parts) < 2 or len(parts[1]) < 4:
                    continue
                entities.append(
                    CaptionEntity(
                        text=tok,
                        start=sm.start(1),
                        end=sm.end(1),
                        label="taxon",
                        score=0.7,
                    )
                )
            panel_labels = _extract_panel_labels_from_caption(cap_text)
            captions.append(
                CaptionRecord(
                    paper_id=paper_id,
                    figure_id=figure_id,
                    caption=cap_text,
                    entities=entities,
                    figure_number=fig_n,
                    page_index=page["index"],
                    panel_labels=panel_labels,
                    source_xml=None,
                )
            )
            i = j
    return captions

# Fulltext sections
# ---------------------------------------------------------------------------


def _parse_fulltext_sections(pages: list[dict[str, Any]]) -> list[dict[str, str]]:
    full_text = "\n".join(p["text"] for p in pages)
    sections: list[dict[str, str]] = []
    headers: list[tuple[int, str]] = []
    for m in _SECTION_HEAD_RE.finditer(full_text):
        headers.append((m.start(), m.group("title")))
    if not headers:
        return [{"title": "fulltext", "section_type": "other", "text": full_text.strip()}]
    intro = full_text[: headers[0][0]].strip()
    if len(intro) > 200:
        sections.append({"title": "Introduction", "section_type": "other", "text": intro})
    for i, (pos, title) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(full_text)
        body = full_text[pos + len(title):end].strip()
        body = re.sub(r"\b[A-Z]{2,}[A-Z ,\.\d]{6,}\b\s*\d{2,4}\b", "", body)
        if not body or len(body) < 50:
            continue
        sections.append(
            {
                "title": title.strip(),
                "section_type": _infer_section_type(title),
                "text": body,
            }
        )
    return sections


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def parse_pdf(pdf_path: Path) -> LocalParserResult:
    """Parse a PDF with pdfplumber, returning a GROBID-shaped result.

    Falls back to ``pypdf`` if pdfplumber is missing or fails.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return _parse_pdf_pypdf(pdf_path)

    paper_id = _stable_id(pdf_path)
    pages_text: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                try:
                    txt = page.extract_text() or ""
                except Exception:  # noqa: BLE001
                    txt = ""
                pages_text.append({"index": idx, "text": txt})
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber failed on %s: %s; falling back to pypdf", pdf_path, exc)
        return _parse_pdf_pypdf(pdf_path)

    captions = _parse_captions(pages_text, paper_id)
    sections = _parse_fulltext_sections(pages_text)
    metadata = _parse_paper_metadata(pages_text, paper_id)
    return LocalParserResult(
        paper_id=paper_id,
        pdf_path=pdf_path,
        captions=captions,
        fulltext_sections=sections,
        paper_metadata=metadata,
        success=True,
    )


def _parse_pdf_pypdf(pdf_path: Path) -> LocalParserResult:
    try:
        import pypdf  # type: ignore
    except ImportError:
        return LocalParserResult(
            paper_id=_stable_id(pdf_path),
            pdf_path=pdf_path,
            success=False,
            error="Neither pdfplumber nor pypdf available",
        )
    paper_id = _stable_id(pdf_path)
    pages_text: list[dict[str, Any]] = []
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for idx, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                txt = ""
            pages_text.append({"index": idx, "text": txt})
    except Exception as exc:  # noqa: BLE001
        return LocalParserResult(
            paper_id=paper_id,
            pdf_path=pdf_path,
            success=False,
            error=f"pypdf failed: {exc}",
        )
    captions = _parse_captions(pages_text, paper_id)
    sections = _parse_fulltext_sections(pages_text)
    metadata = _parse_paper_metadata(pages_text, paper_id)
    return LocalParserResult(
        paper_id=paper_id,
        pdf_path=pdf_path,
        captions=captions,
        fulltext_sections=sections,
        paper_metadata=metadata,
        success=True,
    )


def detect_pdf_extraction_source(pdf_path: Path) -> str:
    """Decide which extractor to use for ``pdf_path``.

    Returns one of:
      * ``"local_pdf_parser"`` if pdfplumber is installed and the
        PDF was produced by a typical word processor (i.e. text
        extraction succeeds for > 50% of pages)
      * ``"opendataloader"`` if opendataloader-pdf is installed
      * ``"grobid"`` as a final fallback
    """
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(str(pdf_path)) as pdf:
            n = len(pdf.pages)
            ok = sum(1 for p in pdf.pages if (p.extract_text() or "").strip())
        if n > 0 and ok / n > 0.5:
            return "local_pdf_parser"
    except Exception:  # noqa: BLE001
        pass
    try:
        import opendataloader_pdf  # type: ignore  # noqa: F401

        return "opendataloader"
    except ImportError:
        pass
    return "grobid"