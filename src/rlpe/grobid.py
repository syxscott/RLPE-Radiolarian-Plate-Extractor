from __future__ import annotations

import logging
import re
import threading
import time
import xml
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .layout import extract_figure_number
from .types import CaptionEntity, CaptionRecord, PaperMetadata
from .utils import ensure_dir, stable_id

logger = logging.getLogger(__name__)


class PipelineCancelledError(Exception):
    """Raised when ``cancel_event`` is set during a long-running pipeline
    operation. Phase 59 (Bug 2.2): the GROBID retry loop honours
    ``cancel_event`` and raises this error instead of waiting for all
    retries to elapse.
    """


@dataclass(slots=True)
class GrobidResult:
    paper_id: str
    pdf_path: Path
    tei_path: Path | None
    tei_xml: str | None
    captions: list[CaptionRecord]
    fulltext_sections: list[dict[str, str]]
    success: bool
    error: str | None = None
    # Phase 29: ``retry_count`` records how many HTTP attempts were
    # made before either success or final failure. ``error_type`` is
    # one of the constants in ``GrobidClient._ERROR_TYPES`` (or
    # ``"none"`` on success). Pipeline code uses these to decide
    # between OD-fallback vs. visual-stub fallback.
    retry_count: int = 0
    error_type: str = "none"


class GrobidClient:
    """GROBID TEI client with retry + structured error reporting.

    Phase 29: The previous implementation made a single POST and
    silently swallowed every failure with ``except Exception``.
    Operators had no visibility into whether GROBID was down, slow,
    or rejecting the PDF, and the pipeline had no way to distinguish
    ``ConnectionRefused`` (server down) from ``HTTPError 500`` (server
    processing error) from ``Timeout`` (server too slow).

    The new client:
    * Retries up to ``max_retries`` times with exponential backoff
      capped at 30s (mirrors the pattern in
      ``llm_backends.MiniMaxM3Backend._call_api``).
    * Distinguishes error types via ``_classify_exception``.
    * Returns ``retry_count`` and ``error_type`` in ``GrobidResult``
      so the pipeline can make an informed OD-vs-visual-stub choice.
    """

    _ERROR_TYPES = (
        "none",
        "connection_refused",
        "timeout",
        "http_5xx",
        "http_4xx",
        "parse_error",
        "unknown",
    )

    def __init__(
        self,
        server_url: str = "http://localhost:8070",
        timeout: int = 300,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        cancel_event: "threading.Event | None" = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        # Phase 59 (Bug 2.2): cooperative cancellation. When set, the
        # retry loop checks the event at the top of every iteration
        # and raises ``PipelineCancelledError`` so the user doesn't
        # wait up to ``max_retries * timeout`` seconds after clicking
        # Cancel on a hung GROBID run.
        self.cancel_event = cancel_event

    def is_available(self, probe_timeout: float = 5.0) -> bool:
        """Phase 38: fast ``/api/isalive`` probe.

        Returns True if the GROBID server responds with HTTP 200 to
        its well-known health-check endpoint within ``probe_timeout``
        seconds. False on any error (timeout, connection refused, 4xx,
        5xx). Use this before calling ``process_pdf`` to fail-fast
        when GROBID is offline, rather than spending ``max_retries *
        timeout`` seconds (up to 900s by default) on a dead server.
        """
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(
                f"{self.server_url}/api/isalive",
                headers={"User-Agent": "RLPE/1.0 (+probe)"},
            )
            with urllib.request.urlopen(req, timeout=probe_timeout) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return False
        except Exception:
            return False

    @staticmethod
    def _classify_exception(exc: BaseException) -> str:
        """Map a requests/IO exception to a coarse error_type bucket.

        Used by ``process_pdf`` to surface the failure mode to the
        pipeline so it can pick the right fallback (OD vs. visual
        stub) and so operators can grep the log for the category.
        """
        # requests.ConnectionError wraps socket errors; the more
        # specific ConnectionRefusedError bubbles up in Py3.10+ as
        # ``ConnectionRefusedError(111, ...)`` inside the wrapped
        # ``__cause__`` chain. Check both forms.
        if isinstance(exc, requests.ConnectionError):
            cause = exc.__cause__ or exc.__context__
            if isinstance(cause, ConnectionRefusedError):
                return "connection_refused"
            # EOF / DNS failure on some platforms shows up as a
            # generic OSError wrapped by requests.
            if isinstance(cause, OSError) and getattr(cause, "errno", None) in {
                111,  # ECONNREFUSED
                -2,   # Name or service not known (some glibc)
            }:
                return "connection_refused"
            return "connection_refused"
        if isinstance(exc, (requests.Timeout, TimeoutError)):
            return "timeout"
        if isinstance(exc, requests.HTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", 0) or 0
            if 500 <= status < 600:
                return "http_5xx"
            if 400 <= status < 500:
                return "http_4xx"
            return "unknown"
        # XML parse error (raised by callers of parse_captions_from_tei)
        if isinstance(exc, (ET.ParseError, xml.etree.ElementTree.ParseError)):
            return "parse_error"
        return "unknown"

    def process_pdf(self, pdf_path: Path, output_dir: Path) -> GrobidResult:
        paper_id = stable_id(pdf_path)
        tei_dir = ensure_dir(output_dir / "tei")
        tei_path = tei_dir / f"{paper_id}.tei.xml"
        last_exc: BaseException | None = None
        last_error_type = "unknown"
        # Phase 29 retry loop. ``max_retries`` is the *total* number of
        # attempts; we sleep ``min(2**attempt * retry_backoff, 30)``
        # between attempts. Successful attempt returns immediately.
        for attempt in range(self.max_retries):
            # Phase 59 (Bug 2.2): honour cancel_event at the top of
            # every retry. Without this check, a 200-page paper with
            # max_retries=3 + timeout=300s runs for 15 minutes after
            # the user clicked Cancel. With it, the user sees the
            # cancellation take effect within the backoff window.
            if self.cancel_event is not None and self.cancel_event.is_set():
                logger.info(
                    "GROBID cancel_event set; aborting retry loop for %s",
                    pdf_path.name,
                )
                raise PipelineCancelledError(
                    f"GROBID retry loop cancelled for {pdf_path.name}"
                )
            try:
                with pdf_path.open("rb") as f:
                    resp = requests.post(
                        f"{self.server_url}/api/processFulltextDocument",
                        files={"input": (pdf_path.name, f, "application/pdf")},
                        data={"consolidateHeader": "1", "consolidateCitations": "1"},
                        timeout=self.timeout,
                    )
                resp.raise_for_status()
                tei_path.write_text(resp.text, encoding="utf-8")
                captions = parse_captions_from_tei(
                    resp.text, paper_id=paper_id, source_xml=str(tei_path)
                )
                sections = parse_fulltext_sections_from_tei(resp.text)
                return GrobidResult(
                    paper_id=paper_id,
                    pdf_path=pdf_path,
                    tei_path=tei_path,
                    tei_xml=resp.text,
                    captions=captions,
                    fulltext_sections=sections,
                    success=True,
                    retry_count=attempt,
                    error_type="none",
                )
            except Exception as exc:
                last_exc = exc
                last_error_type = self._classify_exception(exc)
                logger.warning(
                    "GROBID attempt %d/%d failed for %s (error_type=%s): %s",
                    attempt + 1,
                    self.max_retries,
                    pdf_path.name,
                    last_error_type,
                    exc,
                )
                # Sleep before next attempt unless this was the last.
                if attempt + 1 < self.max_retries and self.retry_backoff > 0:
                    delay = min(2**attempt * self.retry_backoff, 30.0)
                    # Phase 59 (Bug 2.2): also honour cancel_event
                    # during the backoff sleep, so cancellation
                    # takes effect within ``delay`` seconds instead
                    # of waiting for the full sleep.
                    if self.cancel_event is not None:
                        if not self.cancel_event.wait(timeout=delay):
                            # delay elapsed without cancel; fall
                            # through to next iteration's cancel check.
                            pass
                    else:
                        time.sleep(delay)
        # All retries exhausted. Return a structured failure.
        return GrobidResult(
            paper_id=paper_id,
            pdf_path=pdf_path,
            tei_path=None,
            tei_xml=None,
            captions=[],
            fulltext_sections=[],
            success=False,
            error=str(last_exc) if last_exc is not None else "GROBID failed",
            retry_count=self.max_retries,
            error_type=last_error_type,
        )


def parse_captions_from_tei(
    tei_xml: str, paper_id: str, source_xml: str | None = None
) -> list[CaptionRecord]:
    if not tei_xml.strip():
        return []
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return []

    ns = {"tei": root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""}
    captions: list[CaptionRecord] = []
    for idx, fig in enumerate(
        root.findall(".//tei:figure", ns) if ns["tei"] else root.findall(".//figure"), start=1
    ):
        fig_id = fig.attrib.get("xml:id") or fig.attrib.get("id") or f"fig{idx}"
        caption = extract_figure_caption(fig, ns)
        figure_number = (
            fig.attrib.get("n") or extract_figure_number(caption) or _figure_number_from_id(fig_id)
        )
        panel_labels = extract_panel_labels_from_caption(caption)
        entities = extract_taxon_candidates(caption)
        captions.append(
            CaptionRecord(
                paper_id=paper_id,
                figure_id=fig_id,
                caption=caption,
                entities=entities,
                figure_number=figure_number,
                panel_labels=panel_labels,
                source_xml=source_xml,
            )
        )
    return captions


def extract_figure_caption(fig: ET.Element, ns: dict[str, str]) -> str:
    parts: list[str] = []
    for tag in ("head", "figDesc", "note", "label"):
        node = fig.find(f"tei:{tag}", ns) if ns.get("tei") else fig.find(tag)
        if node is not None:
            text = " ".join(t.strip() for t in node.itertext() if t and t.strip())
            if text:
                parts.append(text)
    if not parts:
        text = " ".join(t.strip() for t in fig.itertext() if t and t.strip())
        if text:
            parts.append(text)
    return " ".join(parts).strip()


# Phase 54 audit m18 — common false-positive triggers in the
# ``extract_taxon_candidates`` regex. Capitalised+lowercase two-word
# phrases like "Cambrian Ordovician", "Atlantic Ocean", "Tethys Sea"
# match the ``[A-Z][a-zA-Z-]+\s+[a-z][a-zA-Z-]+`` shape but are
# geological periods / geography, not taxa. The association module
# already maintains a similar stoplist; mirror it here.
_TAXON_LIKE_NONTAXON_FIRST_WORDS: frozenset[str] = frozenset(
    {
        "Cambrian", "Ordovician", "Silurian", "Devonian", "Carboniferous",
        "Permian", "Triassic", "Jurassic", "Cretaceous", "Paleogene",
        "Neogene", "Paleocene", "Eocene", "Oligocene", "Miocene",
        "Pliocene", "Pleistocene", "Holocene",
        "Atlantic", "Pacific", "Indian", "Arctic", "Southern",
        "Northern", "Western", "Eastern", "Central",
        "Tethys", "Gondwana", "Laurasia", "Pangaea", "Panthalassa",
        "Upper", "Lower", "Middle", "Late", "Early",
    }
)


def extract_taxon_candidates(text: str) -> list[CaptionEntity]:
    if not text:
        return []
    pattern = re.compile(r"\b([A-Z][a-zA-Z-]+\s+(?:sp\.|spp\.|cf\.|aff\.|[a-z][a-zA-Z-]+))\b")
    out: list[CaptionEntity] = []
    for m in pattern.finditer(text):
        first_word = m.group(1).split()[0]
        if first_word in _TAXON_LIKE_NONTAXON_FIRST_WORDS:
            continue
        out.append(
            CaptionEntity(
                text=m.group(1), start=m.start(1), end=m.end(1), label="taxon", score=0.65
            )
        )
    return out


def extract_panel_labels_from_caption(text: str) -> list[str]:
    if not text:
        return []
    labels: list[str] = []
    # Bug #10 fix: require explicit wrapping or separator to avoid false
    # positives from sentence-initial capitals. Two alternatives:
    #   group 1: parenthesised/bracketed labels like (A) [1] 1)
    #   group 2: separator-style labels like A. 1: 2.
    pattern = re.compile(
        r"(?:\(|\[)([A-Z]|[0-9]{1,2})(?:\)|\])"
        r"|(?<!\w)([A-Z]|[0-9]{1,2})(?=\.|\:)\s*"
    )
    for m in pattern.finditer(text):
        label = (m.group(1) or m.group(2) or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _figure_number_from_id(fig_id: str) -> str | None:
    m = re.search(r"(\d+[A-Za-z]?)", fig_id)
    return m.group(1) if m else None


def parse_fulltext_sections_from_tei(tei_xml: str) -> list[dict[str, str]]:
    """Parse structured full text sections from TEI for geology/systematics extraction."""
    if not tei_xml.strip():
        return []
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return []

    ns = {"tei": root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""}
    sections: list[dict[str, str]] = []
    divs = (
        root.findall(".//tei:text//tei:body//tei:div", ns)
        if ns["tei"]
        else root.findall(".//text//body//div")
    )
    for idx, div in enumerate(divs, start=1):
        head = div.find("tei:head", ns) if ns["tei"] else div.find("head")
        title = (
            " ".join(t.strip() for t in head.itertext() if t and t.strip())
            if head is not None
            else f"section_{idx}"
        )

        paragraphs = div.findall("tei:p", ns) if ns["tei"] else div.findall("p")
        text = "\n".join(
            " ".join(t.strip() for t in p.itertext() if t and t.strip()) for p in paragraphs
        )
        text = text.strip()
        if not text:
            continue
        section_type = infer_section_type(title)
        sections.append(
            {"section_id": f"sec_{idx}", "title": title, "section_type": section_type, "text": text}
        )
    return sections


def infer_section_type(title: str) -> str:
    t = (title or "").lower()
    if "systematic" in t or "paleontology" in t:
        return "systematic_paleontology"
    if "geological" in t or "setting" in t or "stratigraph" in t:
        return "geological_setting"
    if "material" in t or "method" in t:
        return "materials_methods"
    # Round 20 sampling: the Danelian 2006 paper had a "References"
    # section that was being classified as ``other`` and fed into
    # geology extraction. Citation paragraphs in references mention
    # formation names / countries from OTHER papers as part of
    # bibliographic titles, which would leak into the Danelian
    # records (e.g. country=Japan, formation="Fonzaso Formation"
    # were both extracted from a Beccaro 2002 reference). Recognise
    # the references / bibliography sections explicitly so they can
    # be skipped downstream.
    if "reference" in t or "bibliograph" in t or "cited work" in t or "literature cited" in t:
        return "references"
    return "other"


def process_pdf_dir(
    pdf_dir: Path, output_dir: Path, server_url: str = "http://localhost:8070"
) -> list[GrobidResult]:
    client = GrobidClient(server_url=server_url)
    results = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        results.append(client.process_pdf(pdf_path, output_dir))
    return results


# ---------------------------------------------------------------------------
# Paper-level metadata extraction
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def parse_paper_metadata_from_tei(tei_xml: str) -> PaperMetadata:
    """Extract title, authors, DOI, journal, year, abstract, keywords from GROBID TEI.

    Returns a ``PaperMetadata`` with ``source="grobid"`` and ``confidence=0.95``
    when the header is well-formed; otherwise returns an empty record with
    ``source="none"``. This function never raises — all parse failures fall
    back to a partially-filled record.
    """
    meta = PaperMetadata(source="none")
    if not tei_xml or not tei_xml.strip():
        return meta
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return meta

    ns_raw = root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""
    ns = {"tei": ns_raw} if ns_raw else {}

    def find(path: str) -> ET.Element | None:
        return root.find(path, ns) if ns else root.find(path.replace("tei:", ""))

    # Title: <titleStmt>/<title> (level="a" preferred for article title)
    title_node = (
        find(".//tei:titleStmt/tei:title[@type='article']")
        or find(".//tei:titleStmt/tei:title")
        or find(".//tei:title")
    )
    if title_node is not None:
        title_txt = " ".join(t.strip() for t in title_node.itertext() if t and t.strip())
        if title_txt:
            meta.title = title_txt

    # Authors: <fileDesc>/<titleStmt>/<author>/<persName>  (GROBID canonical location)
    authors: list[str] = []
    author_paths = [
        ".//tei:fileDesc/tei:titleStmt/tei:author",
        ".//tei:analytic/tei:author",
        ".//tei:author",
    ]
    author_elements: list[ET.Element] = []
    for path in author_paths:
        nodes = root.findall(path, ns) if ns else root.findall(path.replace("tei:", ""))
        if nodes:
            author_elements = nodes
            break
    for author_el in author_elements:
        pers = author_el.find("tei:persName", ns) if ns else author_el.find("persName")
        if pers is None:
            # Some GROBID versions flatten the persName into <author>
            full = " ".join(t.strip() for t in author_el.itertext() if t and t.strip())
            if full:
                authors.append(full)
            continue
        forenames: list[str] = []
        for fnode in pers.findall("tei:forename", ns) if ns else pers.findall("forename"):
            txt = " ".join(t.strip() for t in fnode.itertext() if t and t.strip())
            if txt:
                forenames.append(txt)
        surname_node = pers.find("tei:surname", ns) if ns else pers.find("surname")
        surname = (
            " ".join(
                t.strip()
                for t in surname_node.itertext()
                if t and surname_node is not None and t and t.strip()
            )
            if surname_node is not None
            else ""
        )
        full = " ".join([p for p in forenames + ([surname] if surname else []) if p]).strip()
        if not full:
            full = " ".join(t.strip() for t in pers.itertext() if t and t.strip())
        if full:
            authors.append(full)
    if authors:
        meta.authors = authors

    # DOI: <idno type="DOI">
    for idno in root.findall(".//tei:idno", ns) if ns else root.findall(".//idno"):
        if idno.attrib.get("type", "").upper() == "DOI":
            txt = " ".join(t.strip() for t in idno.itertext() if t and t.strip())
            if txt:
                meta.doi = txt
                break
    # Fallback: regex over the whole TEI text
    if not meta.doi:
        m = _DOI_RE.search(tei_xml)
        if m:
            meta.doi = m.group(0).rstrip(".,;)")

    # Journal title: <titleStmt>/<title level="j"> or <monogr>/<title level="j">
    jrn_node = find(".//tei:titleStmt/tei:title[@level='j']") or find(
        ".//tei:monogr/tei:title[@level='j']"
    )
    if jrn_node is None:
        # GROBID may use the second <title> child as the journal
        titles = (
            root.findall(".//tei:titleStmt/tei:title", ns)
            if ns
            else root.findall(".//titleStmt/title")
        )
        for tn in titles:
            lvl = tn.attrib.get("level", "").lower()
            typ = tn.attrib.get("type", "").lower()
            if lvl == "j" or typ == "j":
                txt = " ".join(t.strip() for t in tn.itertext() if t and t.strip())
                if txt:
                    meta.journal = txt
                    break
    else:
        txt = " ".join(t.strip() for t in jrn_node.itertext() if t and t.strip())
        if txt:
            meta.journal = txt

    # Volume / issue / pages
    monogr = find(".//tei:monogr")
    if monogr is not None:
        for child_tag, attr in (
            ("tei:imprint/tei:biblScope[@unit='volume']", "volume"),
            ("tei:imprint/tei:biblScope[@unit='issue']", "issue"),
            ("tei:imprint/tei:biblScope[@unit='page']", "pages"),
        ):
            node = monogr.find(child_tag, ns) if ns else monogr.find(child_tag.replace("tei:", ""))
            if node is not None:
                txt = " ".join(t.strip() for t in node.itertext() if t and t.strip())
                if txt and getattr(meta, attr) is None:
                    setattr(meta, attr, txt)

    # Year: <date type="published"> or <publicationStmt>/<date>
    for date_el in (
        root.findall(".//tei:publicationStmt/tei:date", ns)
        if ns
        else root.findall(".//publicationStmt/date")
    ):
        when = date_el.attrib.get("when") or date_el.attrib.get("notBefore") or ""
        m = _YEAR_RE.search(when)
        if m:
            try:
                meta.year = int(m.group(1))
                break
            except Exception:
                pass
    if meta.year is None:
        m = _YEAR_RE.search(tei_xml)
        if m:
            try:
                meta.year = int(m.group(1))
            except Exception:
                pass

    # Abstract: <profileDesc>/<abstract>
    abs_node = find(".//tei:profileDesc/tei:abstract")
    if abs_node is not None:
        parts = [
            " ".join(t.strip() for t in p.itertext() if t and t.strip())
            for p in (abs_node.findall("tei:p", ns) if ns else abs_node.findall("p"))
        ]
        text = " ".join(p for p in parts if p).strip()
        if not text:
            text = " ".join(t.strip() for t in abs_node.itertext() if t and t.strip()).strip()
        if text:
            meta.abstract = text

    # Keywords: <profileDesc>/<textClass>/<keywords>/<term>
    for term in (
        root.findall(".//tei:profileDesc/tei:textClass/tei:keywords/tei:term", ns)
        if ns
        else root.findall(".//profileDesc/textClass/keywords/term")
    ):
        txt = " ".join(t.strip() for t in term.itertext() if t and t.strip())
        if txt and txt not in meta.keywords:
            meta.keywords.append(txt)

    # Publisher: <publicationStmt>/<publisher>
    pub_node = find(".//tei:publicationStmt/tei:publisher")
    if pub_node is not None:
        txt = " ".join(t.strip() for t in pub_node.itertext() if t and t.strip())
        if txt:
            meta.publisher = txt

    # Page count: <measure unit="page" quantity="N"> or count <pb>
    for meas in root.findall(".//tei:measure", ns) if ns else root.findall(".//measure"):
        if (meas.attrib.get("unit") or "").lower().startswith("page"):
            try:
                meta.page_count = int(float(meas.attrib.get("quantity", "0")))
                break
            except Exception:
                pass
    if meta.page_count is None:
        pbs = root.findall(".//tei:pb", ns) if ns else root.findall(".//pb")
        if pbs:
            meta.page_count = len(pbs)

    # Compute confidence: how many key fields are populated?
    filled = sum(1 for k in ("title", "doi", "abstract", "year", "journal") if getattr(meta, k))
    if filled == 0:
        meta.confidence = 0.0
        meta.source = "none"
    else:
        meta.confidence = min(0.95, 0.4 + 0.15 * filled)
        meta.source = "grobid"
    return meta
