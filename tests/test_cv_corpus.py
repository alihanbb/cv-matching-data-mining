import json
from pathlib import Path

from src.ingest.cv_corpus import load_cv_rows_from_jsonl


def test_load_cv_rows_from_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"record_id": "a1", "text": "x" * 50})
        + "\n"
        + json.dumps({"record_id": "a2", "text": "short"})
        + "\n",
        encoding="utf-8",
    )
    rows = load_cv_rows_from_jsonl(p, id_prefix="t_")
    assert len(rows) == 1
    assert rows[0]["cv_id"] == "t_a1"
    assert len(rows[0]["text"]) >= 40
