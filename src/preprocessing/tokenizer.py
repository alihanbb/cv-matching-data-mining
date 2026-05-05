from __future__ import annotations

import re
from typing import Iterable


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9+#.]+", text.lower())
