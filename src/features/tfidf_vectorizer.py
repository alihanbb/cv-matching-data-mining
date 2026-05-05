from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def _sublinear_tf_safe(params: dict[str, Any]) -> TfidfVectorizer:
    sublinear = bool(params.pop("sublinear_tf", False))
    return TfidfVectorizer(sublinear_tf=sublinear, **params)


class TfidfFeatureBuilder:
    def __init__(self, tfidf_params: dict[str, Any] | None = None) -> None:
        p = dict(tfidf_params or {})
        ng = p.get("ngram_range")
        if isinstance(ng, list):
            p["ngram_range"] = tuple(ng)
        self._raw_params = dict(p)
        self._vectorizer = _sublinear_tf_safe(dict(p))
        self._pipe = Pipeline(
            [
                ("tfidf", self._vectorizer),
            ]
        )

    def fit(self, documents: list[str]) -> "TfidfFeatureBuilder":
        self._pipe.fit(documents)
        return self

    def transform(self, documents: list[str]):
        return self._pipe.transform(documents)

    def fit_transform(self, documents: list[str]):
        return self._pipe.fit_transform(documents)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pipe, f)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfFeatureBuilder":
        with open(path, "rb") as f:
            pipe = pickle.load(f)
        inst = cls()
        inst._pipe = pipe
        inst._vectorizer = pipe.named_steps["tfidf"]
        return inst
