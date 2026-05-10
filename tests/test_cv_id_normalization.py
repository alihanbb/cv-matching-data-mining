import pytest

from src.utils.id_normalization import normalize_cv_id


@pytest.mark.parametrize(
    ("raw_id", "expected"),
    [
        ("corpus_corpus_vanetik_cv_014", "vanetik_cv_014"),
        ("corpus_vanetik_cv_014", "vanetik_cv_014"),
        ("vanetik_cv_014", "vanetik_cv_014"),
        ("  corpus_corpus_vanetik_cv_014  ", "vanetik_cv_014"),
    ],
)
def test_normalize_cv_id_prefix_collapsing(raw_id: str, expected: str) -> None:
    assert normalize_cv_id(raw_id) == expected

