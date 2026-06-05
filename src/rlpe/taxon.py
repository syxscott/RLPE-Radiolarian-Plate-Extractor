from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaxonEntity:
    text: str
    start: int
    end: int
    label: str = "taxon"
    score: float = 0.0


class TaxonRecognizer:
    def __init__(self, model: str = "en_eco", hf_model_path: str | None = None, lexicon_path: str | None = None) -> None:
        self.model = model
        self.hf_model_path = hf_model_path
        self.lexicon_path = lexicon_path
        self._engine = None
        self._hf_ner = None
        self._lexicon: set[str] = set()
        self._lock = threading.Lock()

    def _lazy_init(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from taxonerd import TaxoNERD

                self._engine = TaxoNERD(model=self.model)
            except Exception:
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
                result = engine.predict(text)
                for item in result:
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
        pattern = re.compile(
            r"\b("
            r"[A-Z][a-zA-Z-]{2,}"
            r"(?:"
            r"\s+(?:cf\.|aff\.)\s+[a-z][a-zA-Z-]{2,}"
            r"|"
            r"\s+[a-z][a-zA-Z-]{2,}"
            r")"
            r"(?:\s+(?:n\.\s*sp\.|sp\.\s*nov\.|sp\.|spp\.|cf\.|aff\.|n\.\s*gen\.\s*&\s*sp\.|nov\.))?"
            r")\b"
        )
        entities: list[TaxonEntity] = []
        for m in pattern.finditer(cleaned):
            words = m.group(1).split()
            if len(words) < 2:
                continue
            if words[0].lower() in _NON_TAXON_FIRST_WORDS:
                continue
            epithet_idx = 1
            if len(words) > 2 and words[1].lower() in ("cf.", "aff."):
                epithet_idx = 2
            if epithet_idx >= len(words):
                continue
            if words[epithet_idx].lower() in _NON_TAXON_SECOND_WORDS:
                continue
            entities.append(TaxonEntity(text=m.group(1), start=m.start(1), end=m.end(1), score=0.55))
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
_NON_TAXON_FIRST_WORDS: frozenset[str] = frozenset({
    "explanation", "scale", "figure", "fig", "figs", "plate", "pl",
    "text", "image", "photo", "photograph", "drawing", "line",
    "caption", "all", "bar", "see", "shown", "above", "below",
    "early", "late", "middle", "upper", "lower", "type", "genus",
    "species", "order", "family", "class", "subclass",
    "north", "south", "east", "west", "central",
    "tropical", "arctic", "pacific", "atlantic", "indian",
    "remarks", "note", "notes", "from", "with", "without",
    "northeastern", "northwestern", "southeastern", "southwestern",
    # Caption-header nouns that look like binomials but never are.
    # "Plate 1 Scanning electron microscope pictures..." — the
    # "Scanning electron" pair matches the regex but is not a taxon.
    "scanning", "electron", "microscope", "transmission", "light",
    "secondary", "photomicrograph", "photomicrographs", "marker",
    "sample", "section", "locality", "thin", "thins", "stereo",
    "backscattered", "overview", "detail",
})

# Common 2-3 letter non-Latin epithets that the binomial regex used to match.
# Anything here is a stopword / preposition / article, never a species epithet.
_NON_TAXON_SECOND_WORDS: frozenset[str] = frozenset({
    "of", "in", "on", "by", "an", "at", "to", "or", "is", "as",
    "bar", "fig", "figs", "no", "all", "the", "and", "or",
    "sp", "spp", "cf", "aff", "nov", "gen", "comb",
})
