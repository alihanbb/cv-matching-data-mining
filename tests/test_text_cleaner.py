from src.preprocessing.cleaner import TextCleaner


def test_clean_lowercases_and_strips_punctuation() -> None:
    cleaner = TextCleaner(remove_stopwords=False, lemmatize=False)
    out = cleaner.clean("Hello, World!! Python.")
    assert out == "hello world python"


def test_clean_handles_non_string_input() -> None:
    cleaner = TextCleaner(remove_stopwords=False, lemmatize=False)
    assert cleaner.clean(None) == ""  # type: ignore[arg-type]


def test_clean_collapses_whitespace() -> None:
    cleaner = TextCleaner(remove_stopwords=False, lemmatize=False)
    assert (
        cleaner.clean("  multi   spaces\tand\nnewlines  ")
        == "multi spaces and newlines"
    )
