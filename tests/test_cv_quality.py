from src.processing.cv_sections import cv_quality_score, segment_cv


def test_segment_cv_has_skills_block() -> None:
    text = "Summary\nExperienced dev\nSkills\nPython SQL\nExperience\nCompany A"
    sec = segment_cv(text)
    assert "python" in sec.get("skills", "").lower()


def test_cv_quality_increases_with_sections() -> None:
    minimal = segment_cv("just text")
    rich = segment_cv(
        "Skills\nPython\nExperience\n3 years\nEducation\nBS CS\nProjects\nApp\nCertificates\nAWS"
    )
    assert cv_quality_score(rich, "\n".join(rich.values())) >= cv_quality_score(minimal, "just text")
