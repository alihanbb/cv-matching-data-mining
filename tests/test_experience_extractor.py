from src.extraction.experience_extractor import (
    cv_max_years,
    extract_experience_signals,
    extract_job_required_years,
)


def test_extract_cv_years_from_phrase() -> None:
    text = "6 years of experience in Python backend development."
    sig = extract_experience_signals(text)
    assert cv_max_years(sig) == 6.0


def test_extract_cv_years_from_experience_label() -> None:
    text = "Experience: 4 years in NLP."
    sig = extract_experience_signals(text)
    assert cv_max_years(sig) == 4.0


def test_extract_cv_years_handles_missing_data() -> None:
    sig = extract_experience_signals("Hello world")
    assert sig.years_mentioned == []
    assert cv_max_years(sig) == 0.0


def test_extract_job_required_years_at_least() -> None:
    assert extract_job_required_years("At least 5 years of experience required.") == 5.0


def test_extract_job_required_years_plus_pattern() -> None:
    assert extract_job_required_years("3+ years experience with Docker.") == 3.0


def test_extract_job_required_years_returns_max_when_multiple() -> None:
    text = "Minimum 3 years of Python; ideally 5+ years building APIs."
    assert extract_job_required_years(text) == 5.0


def test_extract_job_required_years_none_when_unknown() -> None:
    assert extract_job_required_years("Friendly team and remote work.") is None
