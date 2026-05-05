import numpy as np

from src.scoring.fusion import fuse_scores, minmax_per_column, skill_jaccard_matrix


def test_minmax_per_column_constant_column_is_ones():
    m = np.ones((3, 2))
    out = minmax_per_column(m)
    assert out.shape == m.shape
    assert np.allclose(out[:, 0], 1.0)


def test_skill_jaccard():
    cv = [{"a", "b"}, {"c"}]
    job = [{"a", "b", "c"}, {"c"}]
    s = skill_jaccard_matrix(cv, job)
    assert s[0, 0] == 2 / 3
    assert s[1, 1] == 1.0


def test_fuse_without_dense():
    tfidf = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=float)
    skills = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    exp = np.ones_like(tfidf)
    fused, w = fuse_scores(tfidf, None, skills, exp, {"tfidf": 1, "dense": 1, "skills": 1, "experience": 1}, False)
    assert fused.shape == tfidf.shape
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["dense"] == 0.0
