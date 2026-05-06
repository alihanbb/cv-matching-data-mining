from src.extraction.skill_extractor import extract_skills


def test_extract_skills_returns_canonical_terms() -> None:
    text = "I have 5 years of Python and SQL experience and used Docker in production."
    skills = {s.lower() for s in extract_skills(text)}
    assert {"python", "sql", "docker"}.issubset(skills)


def test_extract_skills_uses_alias() -> None:
    text = "Strong ML and TF background; some k8s ops."
    skills = {s.lower() for s in extract_skills(text)}
    assert "machine learning" in skills
    assert "tensorflow" in skills
    assert "kubernetes" in skills


def test_extract_skills_empty_text() -> None:
    assert extract_skills("") == []


def test_extract_skills_no_false_positive_for_partial_words() -> None:
    text = "This Python-like syntax has nothing to do with Reactjs frameworks."
    skills = {s.lower() for s in extract_skills(text)}
    assert "python" in skills
    # `react` is in the lexicon and we want word-boundary matching to detect it,
    # while bare `node` should not appear if not present.
    assert "node" not in skills
