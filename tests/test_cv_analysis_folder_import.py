from __future__ import annotations

from pathlib import Path

from src.ingest.cv_analysis_folder_import import (
    build_resume_row,
    iter_corpus_files,
    merge_cv_analysis_pdf_corpus,
)

CV_ANALYSIS_TAG = "cv_analysis_pdf_corpus"


def test_iter_corpus_files_finds_txt(tmp_path: Path) -> None:
    root = tmp_path / "data"
    cat = root / "ACCOUNTANT"
    cat.mkdir(parents=True)
    f = cat / "sample.txt"
    f.write_text("x" * 50, encoding="utf-8")
    pairs = iter_corpus_files(root)
    assert len(pairs) == 1
    assert pairs[0][0] == f
    assert pairs[0][1] == cat


def test_build_resume_row_short_text_after_extract(tmp_path: Path) -> None:
    cat = tmp_path / "SALES"
    cat.mkdir()
    f = cat / "t.txt"
    f.write_text("short", encoding="utf-8")
    row = build_resume_row(f, cat, source_tag=CV_ANALYSIS_TAG)
    assert row is not None
    assert row["resume_id"].startswith("cv_pdf_sales_")


def test_merge_respects_overwrite_source_tag(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    bronze = proj / "data" / "bronze" / "resumes"
    bronze.mkdir(parents=True)
    resume_path = bronze / "resumes_bronze.jsonl"
    resume_path.write_text(
        '{"resume_id": "cv_pdf_test_a", "source": "' + CV_ANALYSIS_TAG + '", '
        '"source_file": "X/a.txt", "raw_text": "' + ("word " * 15) + '"}\n'
        '{"resume_id": "keep_me", "source": "other", "source_file": "z", "raw_text": "' + ("word " * 15) + '"}\n',
        encoding="utf-8",
    )

    corpus = tmp_path / "corpus" / "ENGINEERING"
    corpus.mkdir(parents=True)
    txt = corpus / "cv1.txt"
    txt.write_text("Professional software engineer with extensive experience. " * 3, encoding="utf-8")

    merge_cv_analysis_pdf_corpus(
        project_root=proj,
        corpus_root=tmp_path / "corpus",
        overwrite=True,
        source_tag=CV_ANALYSIS_TAG,
    )

    text = resume_path.read_text(encoding="utf-8")
    assert "keep_me" in text
    assert "cv_pdf_test_a" not in text
    assert "ENGINEERING" in text
