from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class TaxonEntity:
    text: str
    start: int
    end: int
    label: str = "taxon"
    score: float = 0.0


class TaxonRecognizer:
    def __init__(self, model: str = "en_eco") -> None:
        self.model = model
        self._engine = None

    def _lazy_init(self):
        if self._engine is not None:
            return self._engine
        try:
            from taxonerd import TaxoNERD

            self._engine = TaxoNERD(model=self.model)
        except Exception:
            self._engine = None
        return self._engine

    def predict(self, text: str) -> list[TaxonEntity]:
        engine = self._lazy_init()
        if engine is not None:
            try:
                result = engine.predict(text)
                entities: list[TaxonEntity] = []
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
                return entities
            except Exception:
                pass
        return self._fallback_predict(text)

    def _fallback_predict(self, text: str) -> list[TaxonEntity]:
        pattern = re.compile(r"\b([A-Z][a-zA-Z-]+\s+[a-z][a-zA-Z-]+(?:\s*(?:sp\.|spp\.|cf\.|aff\.))?)\b")
        entities: list[TaxonEntity] = []
        for m in pattern.finditer(text or ""):
            entities.append(TaxonEntity(text=m.group(1), start=m.start(1), end=m.end(1), score=0.55))
        return entities
