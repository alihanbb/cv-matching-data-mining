from src.extraction.requirements_extractor import extract_job_requirements, pair_requirement_summary, skill_score_from_coverage
from src.extraction.skills_lexicon import load_skills_lexicon
from src.utils.helpers import project_root


def test_requirement_sections_and_skill_score() -> None:
    root = project_root()
    lex = load_skills_lexicon(root / "config" / "skills.yaml")
    jd = """
    Requirements:
    Must have: Python, Docker
    Nice to have: Kubernetes
    """
    req = extract_job_requirements(jd, lex)
    assert "python" in req.must_have
    assert "docker" in req.must_have
    assert "kubernetes" in req.nice_to_have

    det = pair_requirement_summary({"python", "kubernetes"}, req)
    assert det["must_have_coverage"] == 0.5
    assert det["nice_to_have_coverage"] == 1.0
    assert skill_score_from_coverage(0.5, 1.0, True) == 0.75 * 0.5 + 0.25 * 1.0
