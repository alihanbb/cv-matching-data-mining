"""Streamlit dashboard — tabs for ranking, profiles, coverage, evaluation, debug."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.dashboard_ranking import (
    NER_SOURCE_TAGS,
    RANKING_SOURCE_TAGS,
    prepare_candidate_ranking_view,
)
from src.utils.id_normalization import normalize_cv_id, normalize_job_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPLAINED_PATH = PROJECT_ROOT / "data" / "gold" / "rankings" / "candidate_scores_explained.csv"
UNIFIED_PATH = PROJECT_ROOT / "data" / "silver" / "unified_resumes.jsonl"
JOBS_SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "cleaned_jobs.csv"
CVS_SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "cleaned_cvs.csv"
EVAL_RESULTS = PROJECT_ROOT / "data" / "gold" / "evaluation" / "evaluation_results.csv"
MODEL_COMP = PROJECT_ROOT / "data" / "gold" / "evaluation" / "model_comparison.csv"


@st.cache_data(show_spinner=False)
def load_rankings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("tfidf_score", "semantic_score", "skill_score", "experience_score", "bm25_score"):
        if col not in df.columns:
            df[col] = 0.0
    if "bm25_score" not in df.columns:
        df["bm25_score"] = 0.0
    if "final_score" not in df.columns and "score" in df.columns:
        df = df.rename(columns={"score": "final_score"})
    if "cv_id" in df.columns:
        df["cv_id"] = df["cv_id"].map(normalize_cv_id)
    if "job_id" in df.columns:
        df["job_id"] = df["job_id"].map(normalize_job_id)
    return df


@st.cache_data(show_spinner=False)
def load_silver(path: Path, id_col: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=[id_col, "text"])
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_unified_lines() -> list[dict]:
    if not UNIFIED_PATH.is_file():
        return []
    rows: list[dict] = []
    with open(UNIFIED_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _unified_doc_id(row: dict) -> str:
    """Support pipeline JSONL (cv_id) and imported/legacy rows (record_id, id, …)."""
    for key in ("cv_id", "record_id", "doc_id", "id"):
        v = row.get(key)
        if v is not None and str(v).strip():
            return str(v)
    return "unknown"


def _cv_text_for_id(cvs_silver: pd.DataFrame, cv_id: str) -> str | None:
    if cvs_silver.empty or "cv_id" not in cvs_silver.columns:
        return None
    target = normalize_cv_id(cv_id)
    row = cvs_silver[cvs_silver["cv_id"].astype(str).map(normalize_cv_id) == target]
    if row.empty or "text" not in row.columns:
        return None
    return str(row["text"].iloc[0])


def tab_ranking(rankings: pd.DataFrame, jobs_silver: pd.DataFrame, cvs_silver: pd.DataFrame) -> None:
    st.subheader("Candidate ranking")
    model_cols = [
        "final_score",
        "final_score_v2_bm25",
        "learned_fusion_score",
        "final_rerank_score",
    ]
    available = [c for c in model_cols if c in rankings.columns and rankings[c].notna().any()]
    score_col = st.selectbox("Score column", available or ["final_score"])
    include_ner_sources = st.checkbox("Include NER-only corpus sources", value=False)

    source_filtered = rankings.copy()
    if "source" in source_filtered.columns:
        source_filtered["source"] = source_filtered["source"].fillna("").astype(str).str.strip()
        source_filtered = source_filtered[source_filtered["source"].isin(set(RANKING_SOURCE_TAGS))]
        if not include_ner_sources:
            source_filtered = source_filtered[
                ~source_filtered["source"].isin(set(NER_SOURCE_TAGS))
            ]
    job_ids = sorted(source_filtered["job_id"].astype(str).unique().tolist())
    if not job_ids:
        st.warning("No ranking rows available after source filtering.")
        return

    left, right = st.columns(2)
    with left:
        job = st.selectbox("Job", job_ids)
    with right:
        topn = st.slider("Top-N", 1, 20, 5)
    block = prepare_candidate_ranking_view(
        rankings,
        job_id=job,
        score_column=score_col,
        top_n=topn,
        include_ner_sources=include_ner_sources,
        ranking_sources=RANKING_SOURCE_TAGS,
        ner_sources=NER_SOURCE_TAGS,
    )

    show_cols = [
        "rank_for_job",
        "cv_id",
        score_col,
        "tfidf_score",
        "semantic_score",
        "bm25_score",
        "skill_score",
        "experience_score",
    ]
    st.dataframe(block[[c for c in show_cols if c in block.columns]])
    jt = jobs_silver[jobs_silver["job_id"].astype(str).map(normalize_job_id) == normalize_job_id(job)]
    if not jt.empty:
        st.text_area("İlan metni", jt["text"].iloc[0], height=200, label_visibility="visible")

    st.markdown("**Aday CV metinleri** — `data/silver/cleaned_cvs.csv` ile eşleşen kayıtlar")
    if block.empty:
        st.info("No candidates found for selected filters.")
        return
    if cvs_silver.empty:
        st.warning(f"Silver CV tablosu yok veya boş: `{CVS_SILVER_PATH}`. Önce `python main.py --ingest` çalıştırın.")
    else:
        cv_ids_block = [str(x) for x in block["cv_id"].tolist()]
        missing = [cid for cid in cv_ids_block if _cv_text_for_id(cvs_silver, cid) is None]
        if missing:
            st.caption(
                "Bu sıralamadaki bazı `cv_id` değerleri silver CV dosyasında yok (ID uyuşmazlığı veya eski çalıştırma): "
                + ", ".join(missing[:8])
                + ("…" if len(missing) > 8 else "")
            )
        for _, r in block.iterrows():
            cid = str(r["cv_id"])
            rank = int(r.get("rank_for_job", 0) or 0)
            body = _cv_text_for_id(cvs_silver, cid)
            with st.expander(f"#{rank} · {cid}", expanded=False):
                if body:
                    st.text(body[:12000] + ("…" if len(body) > 12000 else ""))
                else:
                    st.info("Bu `cv_id` için metin bulunamadı.")

        pick = st.selectbox("Tek seferde tam metin", cv_ids_block, key="cv_full_pick")
        full = _cv_text_for_id(cvs_silver, pick)
        if full:
            st.text_area(f"Seçili CV: {pick}", full, height=320, key="cv_full_body")


def tab_profile(cvs_silver: pd.DataFrame) -> None:
    st.subheader("CV profile analysis")
    source = st.radio(
        "Kaynak",
        ["Silver CSV (pipeline ile aynı cv_id)", "Unified JSONL"],
        horizontal=True,
        key="profile_source",
    )

    if source.startswith("Silver"):
        if cvs_silver.empty or "cv_id" not in cvs_silver.columns:
            st.warning(f"`{CVS_SILVER_PATH}` bulunamadı veya boş. `python main.py --ingest` çalıştırın.")
            return
        ids = sorted(cvs_silver["cv_id"].astype(str).unique().tolist())
        cid = st.selectbox("cv_id", ids, key="profile_silver_cv")
        body = _cv_text_for_id(cvs_silver, cid)
        if body:
            st.text_area("CV metni (cleaned_cvs)", body, height=400, key="silver_cv_body")
        else:
            st.error("Metin okunamadı.")
        return

    unified = load_unified_lines()
    if not unified:
        st.info("Run pipeline with `silver.write_unified_resumes: true` in config to generate unified_resumes.jsonl.")
        return
    n = len(unified)
    idx = st.selectbox(
        "Document",
        range(n),
        format_func=lambda i: f"{_unified_doc_id(unified[i])}  (#{i + 1})",
    )
    row = unified[int(idx)]

    pipeline_shape = "cv_id" in row or "sections" in row or "cleaned_text" in row
    if pipeline_shape:
        st.metric("CV quality", f"{float(row.get('cv_quality_score', 0) or 0):.2f}")
        st.metric("Years (estimate)", f"{float(row.get('total_years_experience', 0) or 0):.1f}")
        detail = {k: row[k] for k in ("sections", "extracted_skills", "skill_categories") if k in row}
        if detail:
            st.json(detail)
        if row.get("raw_text") or row.get("cleaned_text"):
            st.text_area(
                "Text preview",
                str(row.get("cleaned_text") or row.get("raw_text") or "")[:6000],
                height=240,
            )
    else:
        st.caption(
            "Bu satır pipeline `cv_id` + `sections` formatında değil (ör. harici JSONL import). "
            "Özet alanlar aşağıda."
        )
        preview_cols = ("text", "language", "document_type", "source", "category", "record_id", "extraction_status")
        slim = {k: row[k] for k in preview_cols if k in row}
        if slim:
            st.json(slim)
        body = str(row.get("text") or row.get("raw_text") or "")[:8000]
        if body:
            st.text_area("Text preview", body, height=260)


def tab_coverage(rankings: pd.DataFrame) -> None:
    st.subheader("Requirement coverage")
    job = st.selectbox("Job (coverage)", sorted(rankings["job_id"].astype(str).unique()), key="cov_job")
    block = rankings[rankings["job_id"].astype(str) == job].sort_values("rank_for_job").head(10)
    cols = [
        "cv_id",
        "must_have_coverage",
        "nice_to_have_coverage",
        "matched_required_skills",
        "missing_critical_skills",
    ]
    st.dataframe(block[[c for c in cols if c in block.columns]])


def tab_eval() -> None:
    st.subheader("Evaluation metrics")
    if not EVAL_RESULTS.is_file():
        st.warning("Run `python main.py --export-eval-csv` after adding ground truth.")
        return
    st.dataframe(pd.read_csv(EVAL_RESULTS))


def tab_comparison() -> None:
    st.subheader("Model comparison (NDCG)")
    if not MODEL_COMP.is_file():
        st.warning("Missing model_comparison.csv — run --export-eval-csv.")
        return
    st.dataframe(pd.read_csv(MODEL_COMP))


def tab_debug(rankings: pd.DataFrame) -> None:
    st.subheader("Score debug")
    for col in ("final_score_raw", "score_check", "score_diff", "score_warning"):
        if col not in rankings.columns:
            rankings[col] = None
    zero_ratio = float((rankings["semantic_score"].astype(float).abs() < 1e-9).mean()) if "semantic_score" in rankings.columns else 0.0
    st.metric("Semantic score zero ratio", f"{zero_ratio:.1%}")
    st.dataframe(
        rankings[
            [c for c in ("job_id", "cv_id", "final_score_raw", "score_check", "score_diff", "score_warning", "semantic_score") if c in rankings.columns]
        ].head(50)
    )


def main() -> None:
    st.set_page_config(page_title="CV Matching Dashboard", layout="wide")
    st.title("CV–Job Matching Dashboard")
    st.caption(
        "Aday CV metinleri **Candidate Ranking** sekmesinde `data/silver/cleaned_cvs.csv` ile eşlenir. "
        "Metin eksikse `python main.py --ingest` çalıştırın."
    )

    if not EXPLAINED_PATH.is_file():
        st.error(f"Explained rankings missing: `{EXPLAINED_PATH}` — run `python main.py`.")
        st.stop()

    rankings = load_rankings(EXPLAINED_PATH)
    jobs_silver = load_silver(JOBS_SILVER_PATH, "job_id")
    cvs_silver = load_silver(CVS_SILVER_PATH, "cv_id")

    tabs = st.tabs(
        [
            "Candidate Ranking",
            "CV Profile Analysis",
            "Requirement Coverage",
            "Evaluation Metrics",
            "Model Comparison",
            "Score Debug",
        ]
    )
    with tabs[0]:
        tab_ranking(rankings, jobs_silver, cvs_silver)
    with tabs[1]:
        tab_profile(cvs_silver)
    with tabs[2]:
        tab_coverage(rankings)
    with tabs[3]:
        tab_eval()
    with tabs[4]:
        tab_comparison()
    with tabs[5]:
        tab_debug(rankings)


if __name__ == "__main__":
    main()
