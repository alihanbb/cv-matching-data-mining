from __future__ import annotations

import re
import string
from typing import Callable

from .tokenizer import tokenize

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:  # pragma: no cover
    nltk = None
    stopwords = None
    WordNetLemmatizer = None


def _ensure_nltk_resources() -> None:
    if nltk is None:
        return
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(
                f"corpora/{pkg}" if pkg in ("stopwords", "wordnet") else f"tokenizers/{pkg}"
            )
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


class TextCleaner:
    def __init__(
        self,
        *,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        language: str = "en",
        custom_stopwords: frozenset[str] | None = None,
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize and WordNetLemmatizer is not None
        self.language = language
        self._stop: set[str] = set(custom_stopwords or [])
        self._lemmatizer: Callable[[str], str] | None = None
        if remove_stopwords and stopwords is not None:
            _ensure_nltk_resources()
            try:
                self._stop |= set(stopwords.words("english" if language == "en" else language))
            except OSError:
                self._stop |= set()
        if self.lemmatize:
            _ensure_nltk_resources()
            if WordNetLemmatizer is not None:
                wnl = WordNetLemmatizer()
                self._lemmatizer = wnl.lemmatize

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        lower = text.lower()
        no_punct = lower.translate(
            str.maketrans("", "", string.punctuation.replace("#", "").replace("+", ""))
        )
        no_punct = re.sub(r"\s+", " ", no_punct).strip()
        tokens = tokenize(no_punct)
        if self.remove_stopwords and self._stop:
            tokens = [t for t in tokens if t not in self._stop]
        if self._lemmatizer:
            tokens = [self._lemmatizer(t) for t in tokens]
        return " ".join(tokens)
