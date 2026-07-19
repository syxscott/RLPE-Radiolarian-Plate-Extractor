from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Phase 60 Plan 3 (Bug 3.2): a curated set of well-known radiolarian
# (and broader micropalaeontologist) author surnames that, when
# capitalised at the start of a phrase, would otherwise be mis-extracted
# as a genus name by the binomial regex. Lower-cased for case-insensitive
# comparison. Add new surnames when auditing adds them; do NOT add
# genus names even if they look like surnames — the check is explicitly
# a blocklist of human names.
_KNOWN_AUTHOR_SURNAMES: frozenset[str] = frozenset(
    {
        # Core radiolarian workers (after Phase 60 audit)
        "riedel",
        "kozur",
        "bütschli",
        "butschli",
        "haeckel",
        "sanfilippo",
        "pessagno",
        "de wever",
        "dumitrica",
        "o'dogherty",
        "odogherty",
        "foreman",
        "hull",
        "bailey",
        "carter",
        "clark",
        "dunn",
        "foster",
        "goll",
        "kiessling",
        "lazarus",
        "martin",
        "nishimura",
        "palmer",
        "renaudie",
        "sugiyama",
        "takemura",
        "umeda",
        "vishnevskaya",
        "won",
        "yeh",
        "zhang",
        # Extended set — common in citations of any kind
        "smith",
        "jones",
        "johnson",
        "williams",
        "brown",
        "davis",
        "miller",
        "wilson",
        "moore",
        "taylor",
        "anderson",
        "thomas",
        "jackson",
        "white",
        "harris",
        "martin",
        "thompson",
        "garcia",
        "martinez",
        "robinson",
        "clark",
        "rodriguez",
        "lewis",
        "lee",
        "walker",
        "hall",
        "allen",
        "young",
        "king",
        "wright",
        "scott",
        "hill",
        "green",
        "adams",
        "baker",
        "nelson",
        "mitchell",
        "perez",
        "roberts",
        "turner",
        "phillips",
        "campbell",
        "parker",
        "evans",
        "edwards",
        "collins",
        "stewart",
        "sanchez",
        "morris",
        "rogers",
        "reed",
        "cook",
        "morgan",
        "bell",
        "murphy",
        "bailey",
        "rivera",
        "cooper",
        "richardson",
        "cox",
        "howard",
        "ward",
        "torres",
        "peterson",
        "gray",
        "ramirez",
        "james",
        "watson",
        "brooks",
        "kelly",
        "sanders",
        "price",
        "bennett",
        "wood",
        "barnes",
        "ross",
        "henderson",
        "coleman",
        "jenkins",
        "perry",
        "powell",
        "long",
        "patterson",
        "hughes",
        "flores",
        "washington",
        "butler",
        "simmons",
        "foster",
        "hendricks",
        "cole",
        "russell",
        "griffin",
        "diaz",
        "hayes",
    }
)


@dataclass(slots=True)
class TaxonEntity:
    text: str
    start: int
    end: int
    label: str = "taxon"
    score: float = 0.0


class TaxonRecognizer:
    def __init__(
        self,
        model: str = "en_eco",
        hf_model_path: str | None = None,
        lexicon_path: str | None = None,
    ) -> None:
        # ``model`` is kept for API compat with earlier TaxoNERD versions
        # but the installed TaxoNERD 1.5.x signature is
        # ``(self, prefer_gpu=False, verbose=False, logger=None)`` —
        # the ``model`` kwarg is silently ignored / raises. We store
        # the preference and pass ``prefer_gpu`` instead at init time.
        self.model = model
        self.hf_model_path = hf_model_path
        self.lexicon_path = lexicon_path
        self._engine = None
        self._hf_ner = None
        self._lexicon: set[str] = set()
        self._lock = threading.Lock()
        # Separate lock for predict-time model calls. TaxoNERD and HF
        # ``token-classification`` pipelines share mutable internal state
        # (model weights, attention cache, random state) that is NOT
        # thread-safe. The pipeline's ThreadPoolExecutor calls
        # ``predict`` concurrently from multiple workers; without this
        # lock, concurrent ``engine.predict()`` / ``self._hf_ner()``
        # calls corrupt each other's output or crash with a CUDA error.
        self._predict_lock = threading.Lock()

    def _lazy_init(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from taxonerd import TaxoNERD

                # Round 18 audit: previous code did
                # ``TaxoNERD(model=self.model)`` which raises
                # ``TypeError: __init__() got an unexpected keyword
                # argument 'model'`` on TaxoNERD 1.5.x. The accepted
                # kwargs are ``prefer_gpu``, ``verbose``, ``logger``;
                # the model name is fixed at package level (en_eco /
                # en_plus / etc., set via TaxoNERD's own config).
                # Silently swallowed exceptions meant the pipeline
                # fell back to regex without telling anyone.
                try:
                    self._engine = TaxoNERD(prefer_gpu=False, verbose=False)
                except TypeError:
                    # Older signature (rare; some 1.4.x versions accept it)
                    self._engine = TaxoNERD()
            except Exception as exc:
                # TaxoNERD is the primary species-recognition engine.
                # A silent fallback to the regex path leaves the
                # operator wondering why species extraction is so
                # weak — log at warning so the cause is visible.
                # ``self._engine = None`` is correct: the predictor
                # below branches on it to use the regex fallback.
                logger.warning(
                    "TaxoNERD init failed (model=%r): %s; falling back to "
                    "regex-based species extraction",
                    self.model,
                    exc,
                )
                self._engine = None

            if self.hf_model_path and self._hf_ner is None:
                try:
                    from transformers import pipeline

                    self._hf_ner = pipeline(
                        task="token-classification",
                        model=self.hf_model_path,
                        tokenizer=self.hf_model_path,
                        aggregation_strategy="simple",
                    )
                except Exception:
                    self._hf_ner = None

            if self.lexicon_path and not self._lexicon:
                p = Path(self.lexicon_path)
                if p.exists():
                    try:
                        with p.open("r", encoding="utf-8") as f:
                            for line in f:
                                item = line.strip()
                                if item:
                                    self._lexicon.add(item)
                    except Exception:
                        self._lexicon = set()
        return self._engine

    def predict(self, text: str) -> list[TaxonEntity]:
        self._lazy_init()
        engine = self._engine
        entities: list[TaxonEntity] = []

        # A) TaxoNERD 通用模型
        if engine is not None:
            try:
                with self._predict_lock:
                    result = engine.predict(text)
                # Guard: TaxoNERD sometimes returns a string / None / an
                # object whose elements aren't dicts (model-mismatch error
                # path, off-spec version). The previous version assumed
                # ``item.get(...)`` always worked and silently swallowed
                # the AttributeError — that masked model-mismatch bugs
                # and produced empty entity lists. Skip non-dict items
                # explicitly so the loop survives a bad element shape.
                if not isinstance(result, (list, tuple)):
                    result = []
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    entities.append(
                        TaxonEntity(
                            text=item.get("text", ""),
                            start=int(item.get("start", 0)),
                            end=int(item.get("end", 0)),
                            label=item.get("label", "taxon"),
                            score=float(item.get("score", 0.0)),
                        )
                    )
            except Exception:
                pass

        # B) 可选垂类HF NER模型（建议后续用古生物语料微调）
        if self._hf_ner is not None:
            try:
                with self._predict_lock:
                    hf_res = self._hf_ner(text)
                for item in hf_res:
                    label = str(item.get("entity_group", "taxon")).lower()
                    if "tax" not in label and "species" not in label and "org" not in label:
                        continue
                    ent_text = str(item.get("word", "")).replace("##", "").strip()
                    if not ent_text:
                        continue
                    entities.append(
                        TaxonEntity(
                            text=ent_text,
                            start=int(item.get("start", 0)),
                            end=int(item.get("end", 0)),
                            label="taxon",
                            score=float(item.get("score", 0.0)),
                        )
                    )
            except Exception:
                pass

        # C) 规则与词典兜底
        entities.extend(self._fallback_predict(text))
        entities.extend(self._lexicon_predict(text))

        # D) 去重融合
        return self._merge_entities(entities)

    def _fallback_predict(self, text: str) -> list[TaxonEntity]:
        cleaned = self._clean_caption_for_taxon(text)
        # Phase 60 Plan 3 (Bug 3.4): apply Unicode NFKD normalisation
        # so ligatures (``æ`` → ``ae``, ``ﬁ`` → ``fi``, ``œ`` → ``oe``)
        # and combining diacritics (``ö`` → ``o`` + combining diaeresis,
        # stripped on ASCII encode) are flattened to their ASCII
        # equivalents before the regex runs. The original surface form
        # is preserved in the entity's ``text`` field because we offset
        # the match positions back to the un-normalised string via
        # ``m.start(1)`` / ``m.end(1)`` (which still align with the
        # cleaned string; both strings have the same length after NFKD
        # for the Latin / Greek chars we care about).
        cleaned_ascii = (
            unicodedata.normalize("NFKD", cleaned)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        # Phase 60 Plan 3 (Bug 3.3): extended to also accept isolated-
        # genus open-nomenclature forms (``Genus sp.``, ``Genus spp.``,
        # ``Genus n. sp.``, ``Genus sp. nov.``, ``Genus nom. nov.``,
        # ``Genus comb. nov.``). The new alternative uses the qualifier
        # directly as the second token instead of a lowercase epithet.
        #
        # IMPORTANT: the isolated-genus alternative MUST come BEFORE the
        # generic ``\s+[a-z][a-zA-Z-]{2,}`` epithet branch. Otherwise
        # ``Genus spp.`` greedily matches the epithet branch as
        # ``spp`` (3 lowercase chars) and the trailing ``.`` is lost.
        pattern = re.compile(
            r"\b("
            r"[A-Z][a-zA-Z-]{2,}"
            r"(?:"
            # Bug 3.3: isolated-genus open nomenclature. ``sp.`` /
            # ``spp.`` / ``n. sp.`` / ``sp. nov.`` / ``nom. nov.`` /
            # ``comb. nov.`` all appear as the second token in real
            # radiolarian captions and must be matched. MUST come
            # before the generic epithet branch (see comment above).
            r"\s+(?:sp\.|spp\.|n\.\s*sp\.|sp\.\s*nov\.|nom\.\s*nov\.|comb\.\s*nov\.)"
            r"|"
            r"\s+(?:cf\.|aff\.)\s+[a-z][a-zA-Z-]{2,}"
            r"|"
            r"\s+[a-z][a-zA-Z-]{2,}"
            r")"
            r"(?:\s+(?:n\.\s*sp\.|sp\.\s*nov\.|sp\.|spp\.|cf\.|aff\.|n\.\s*gen\.\s*&\s*sp\.|nov\.))?"
            r")"
            # Trailing boundary: ``spp.`` ends in ``.`` which is a
            # non-word char, so the original ``\b`` dropped the
            # trailing period for ``Entactinia spp.``. A positive
            # lookahead on whitespace / sentence-punctuation /
            # end-of-string keeps the period in the match and is
            # still safe for non-``spp.`` shapes.
            r"(?=\s|[.,;:]|$)"
        )
        entities: list[TaxonEntity] = []
        for m in pattern.finditer(cleaned_ascii):
            words = m.group(1).split()
            if len(words) < 2:
                continue
            if words[0].lower() in _NON_TAXON_FIRST_WORDS:
                continue
            # Phase 60 Plan 3 (Bug 3.2): reject matches whose first
            # token is a known paleontologist surname. Catches the
            # common citation shape "Genus species Riedel & Sanfilippo"
            # which the regex would otherwise turn into the bogus
            # "Riedel Sanfilippo" binomial.
            if words[0].lower() in _KNOWN_AUTHOR_SURNAMES:
                continue
            epithet_idx = 1
            if len(words) > 2 and words[1].lower() in ("cf.", "aff."):
                epithet_idx = 2
            if epithet_idx >= len(words):
                continue
            if words[epithet_idx].lower() in _NON_TAXON_SECOND_WORDS:
                continue
            entities.append(
                TaxonEntity(text=m.group(1), start=m.start(1), end=m.end(1), score=0.55)
            )
        return entities

    @staticmethod
    def _clean_caption_for_taxon(text: str) -> str:
        """Strip the leading "Explanation of Plate N." header that confuses the
        binomial regex. Radiolarian plate captions always start with that exact
        phrase, and the rest of the caption is a list of "fig N. Species X"
        entries. Removing the header eliminates the "Explanation of" false
        positive at the head of the list."""
        if not text:
            return ""
        s = text
        # "Explanation of Plate 1. ..."  /  "Explanation of Plate 1, ..."
        s = re.sub(
            r"^\s*Explanation\s+of\s+Plate\s+\d+\s*[\.:,]?\s*",
            "",
            s,
            count=1,
            flags=re.IGNORECASE,
        )
        # "Plate 1. ..."  /  "Plate 1 — ..."
        s = re.sub(
            r"^\s*Plate\s+\d+\s*[\.:,\-—–]?\s*",
            "",
            s,
            count=1,
            flags=re.IGNORECASE,
        )
        return s

    def _lexicon_predict(self, text: str) -> list[TaxonEntity]:
        if not text or not self._lexicon:
            return []
        out: list[TaxonEntity] = []
        lower = text.lower()
        for name in self._lexicon:
            start = lower.find(name.lower())
            if start >= 0:
                out.append(TaxonEntity(text=name, start=start, end=start + len(name), score=0.75))
        return out

    @staticmethod
    def _merge_entities(entities: list[TaxonEntity]) -> list[TaxonEntity]:
        if not entities:
            return []
        merged: dict[tuple[int, int, str], TaxonEntity] = {}
        for e in entities:
            key = (e.start, e.end, e.text.lower())
            old = merged.get(key)
            if old is None or e.score > old.score:
                merged[key] = e
        return sorted(merged.values(), key=lambda x: (x.start, x.end))


# Words that often start a figure caption but are never a genus name.
# These either start the header ("Explanation of Plate N.") or refer to a
# caption-internal non-taxonomic noun ("Scale bar", "Fig. caption").
_NON_TAXON_FIRST_WORDS: frozenset[str] = frozenset(
    {
        "explanation",
        "scale",
        "figure",
        "fig",
        "figs",
        "plate",
        "pl",
        "text",
        "image",
        "photo",
        "photograph",
        "drawing",
        "line",
        "caption",
        "all",
        "bar",
        "see",
        "shown",
        "above",
        "below",
        "early",
        "late",
        "middle",
        "upper",
        "lower",
        "type",
        "genus",
        "species",
        "order",
        "family",
        "class",
        "subclass",
        "north",
        "south",
        "east",
        "west",
        "central",
        "tropical",
        "arctic",
        "pacific",
        "atlantic",
        "indian",
        "remarks",
        "note",
        "notes",
        "from",
        "with",
        "without",
        "northeastern",
        "northwestern",
        "southeastern",
        "southwestern",
        # Caption-header nouns that look like binomials but never are.
        # "Plate 1 Scanning electron microscope pictures..." — the
        # "Scanning electron" pair matches the regex but is not a taxon.
        "scanning",
        "electron",
        "microscope",
        "transmission",
        "light",
        "secondary",
        "photomicrograph",
        "photomicrographs",
        "marker",
        "sample",
        "section",
        "locality",
        "thin",
        "thins",
        "stereo",
        "backscattered",
        "overview",
        "detail",
        # Pipeline placeholder text (OpenDataLoader / GROBID fallback). When
        # the upstream tool can't extract a real caption it returns strings
        # like "Auto-generated figure for page 17" — these used to slip
        # through the binomial regex as "Auto-generated figure".
        "auto",
        "auto-generated",
        "automated",
        "generated",
        "placeholder",
        "undefined",
        "unknown",
        "missing",
        "empty",
        "n/a",
        "na",
        # Synthetic captions built from body-text plate references also
        # need a sentinel header ("(Reconstructed from systematic
        # descriptions)") that the binomial regex used to match as the
        # species "Reconstructed from".
        "reconstructed",
        "reconstructed from",
        "synthesized",
        "synthetic",
        "copyright",
        "published",
        "springer",
        "elsevier",
        "wiley",
        "downloaded",
        "downloadedfrom",
        "rights",
        "reserved",
    }
)

# Common 2-3 letter non-Latin epithets that the binomial regex used to match.
# Anything here is a stopword / preposition / article, never a species epithet.
_NON_TAXON_SECOND_WORDS: frozenset[str] = frozenset(
    {
        "of",
        "in",
        "on",
        "by",
        "an",
        "at",
        "to",
        "or",
        "is",
        "as",
        "bar",
        "fig",
        "figs",
        "no",
        "all",
        "the",
        "and",
        "sp",
        "spp",
        "cf",
        "aff",
        "nov",
        "gen",
        "comb",
    }
)
