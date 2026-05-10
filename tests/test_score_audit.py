import pandas as pd

from src.scoring.score_audit import add_score_audit_columns


def test_score_audit_warns_on_mismatch() -> None:
    df = pd.DataFrame(
        {
            "tfidf_score": [0.5],
            "semantic_score": [0.5],
            "skill_score": [1.0],
            "experience_score": [1.0],
            "final_score": [999.0],
        }
    )
    out = add_score_audit_columns(
        df,
        {"tfidf": 0.5, "dense": 0.5, "skills": 0.0, "experience": 0.0},
        has_semantic=True,
        has_bm25=False,
    )
    assert out["score_diff"].iloc[0] > 0.5
    assert out["score_warning"].iloc[0] == "CHECK>0.01"
