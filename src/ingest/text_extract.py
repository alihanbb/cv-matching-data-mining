from __future__ import annotations

from pathlib import Path


def extract_text_from_path(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suf == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            parts.append(t)
        return "\n".join(parts).strip()
    if suf in {".docx"}:
        import docx

        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
    raise ValueError(f"Unsupported file type: {path.suffix}")
