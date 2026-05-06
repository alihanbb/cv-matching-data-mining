"""Streamlit dashboard for CV Matching Data Mining.

Reads the explainable rankings produced by the pipeline and lets the user:
- pick a job description,
- filter top-N candidates,
- inspect tfidf / semantic / skill / experience component scores,
- see matched vs missing skills,
- read a per-candidate explanation.

Run:
    streamlit run app/streamlit_app.py

The dashboard is read-only: it never re-runs the pipeline. Re-run the pipeline
with ``python main.py [--semantic] [--evaluate]`` to refresh data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPLAINED_PATH = PROJECT_ROOT / "data" / "gold" / "rankings" / "candidate_scores_explained.csv"
JOBS_SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "cleaned_jobs.csv"
CVS_SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "cleaned_cvs.csv"

EXPECTED_COMPONENTS = ("tfidf_score", "semantic_score", "skill_score", "experience_score")


@st.cache_data(show_spinner=False)
def load_rankings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in EXPECTED_COMPONENTS:
        if col not in df.columns:
            df[col] = 0.0
    if "final_score" not in df.columns and "score" in df.columns:
        df = df.rename(columns={"score": "final_score"})
    if "explanation" not in df.columns and "experience_note" in df.columns:
        df = df.rename(columns={"experience_note": "explanation"})
    return df


@st.cache_data(show_spinner=False)
def load_silver(path: Path, id_col: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=[id_col, "text"])
    return pd.read_csv(path)


def _format_skill_chips(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "_yok_"
    items = [s.strip() for s in value.split(";") if s.strip()]
    return ", ".join(f"`{s}`" for s in items)


def main() -> None:
    st.set_page_config(
        page_title="CV Matching Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("CV - Job Matching Dashboard")
    st.caption(
        "Multi-channel scoring (TF-IDF + semantic + skill + experience) with "
        "explainable per-candidate output."
    )

    if not EXPLAINED_PATH.is_file():
        st.error(
            f"Explained rankings not found:\n\n`{EXPLAINED_PATH}`\n\n"
            "Run the pipeline first:\n\n"
            "```\npython main.py --ingest\npython main.py --semantic\n```"
        )
        st.stop()

    rankings = load_rankings(EXPLAINED_PATH)
    jobs_silver = load_silver(JOBS_SILVER_PATH, "job_id")
    cvs_silver = load_silver(CVS_SILVER_PATH, "cv_id")

    job_ids = sorted(rankings["job_id"].astype(str).unique().tolist())
    if not job_ids:
        st.warning("No jobs found in rankings.")
        st.stop()

    with st.sidebar:
        st.header("Filters")
        selected_job = st.selectbox("Job description", job_ids)
        max_n = int(rankings[rankings["job_id"].astype(str) == selected_job].shape[0]) or 1
        top_n = st.slider("Top-N candidates", min_value=1, max_value=max_n, value=min(5, max_n))
        st.caption(
            "Final score formula:\n\n"
            "`0.35 * tfidf + 0.35 * semantic + 0.20 * skill + 0.10 * experience`"
        )

    job_block = rankings[rankings["job_id"].astype(str) == selected_job].copy()
    job_block = job_block.sort_values("rank_for_job") if "rank_for_job" in job_block.columns else job_block
    job_block = job_block.head(top_n).reset_index(drop=True)

    job_text_row = jobs_silver[jobs_silver["job_id"].astype(str) == selected_job]
    job_text = job_text_row["text"].iloc[0] if not job_text_row.empty else None

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("Job description")
        if job_text:
            st.text_area("Selected job", job_text, height=260, label_visibility="collapsed")
        else:
            st.info("Silver CV/job tables not found (run --ingest).")

        st.subheader("Average score components")
        if not job_block.empty:
            avg = job_block[list(EXPECTED_COMPONENTS) + ["final_score"]].mean().round(3)
            st.bar_chart(avg)

    with right:
        st.subheader(f"Top {top_n} candidates")
        display_cols = [
            "rank_for_job",
            "cv_id",
            "final_score",
            "tfidf_score",
            "semantic_score",
            "skill_score",
            "experience_score",
        ]
        present_cols = [c for c in display_cols if c in job_block.columns]
        st.dataframe(
            job_block[present_cols].style.format(
                {c: "{:.3f}" for c in present_cols if c not in ("rank_for_job", "cv_id")}
            ),
            width="stretch",
        )

        st.subheader("Per-candidate explanation")
        for _, row in job_block.iterrows():
            cv_id = str(row["cv_id"])
            with st.expander(
                f"#{int(row.get('rank_for_job', 0))} - {cv_id}  -  "
                f"final={row.get('final_score', 0):.3f}",
                expanded=False,
            ):
                cols = st.columns(4)
                cols[0].metric("TF-IDF", f"{row.get('tfidf_score', 0):.3f}")
                cols[1].metric("Semantic", f"{row.get('semantic_score', 0):.3f}")
                cols[2].metric("Skill", f"{row.get('skill_score', 0):.3f}")
                cols[3].metric("Experience", f"{row.get('experience_score', 0):.3f}")

                st.markdown("**Matched skills:** " + _format_skill_chips(row.get("matched_skills")))
                st.markdown("**Missing skills:** " + _format_skill_chips(row.get("missing_skills")))

                explanation = row.get("explanation")
                if isinstance(explanation, str) and explanation.strip():
                    st.markdown(f"**Explanation:** `{explanation}`")

                cv_row = cvs_silver[cvs_silver["cv_id"].astype(str) == cv_id]
                if not cv_row.empty:
                    with st.popover("Show CV text"):
                        st.write(cv_row["text"].iloc[0])

    st.divider()
    st.caption(
        "Source: data/gold/rankings/candidate_scores_explained.csv  -  "
        "Re-run the pipeline to refresh."
    )


if __name__ == "__main__":
    main()
