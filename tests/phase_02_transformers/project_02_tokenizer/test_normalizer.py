from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.normalizer import (
    BasicNormalizer,
)


def test_normalizer_without_lowercase():
    normalizer = BasicNormalizer(lowercase=False)

    assert normalizer.normalize("Hello WORLD") == "Hello WORLD"


def test_normalizer_with_lowercase():
    normalizer = BasicNormalizer(lowercase=True)

    assert normalizer.normalize("Hello WORLD") == "hello world"