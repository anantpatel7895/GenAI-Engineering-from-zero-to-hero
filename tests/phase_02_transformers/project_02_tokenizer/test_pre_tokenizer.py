from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.pre_tokenizer import (
    BasicPreTokenizer,
)


def test_pre_tokenize_whitespace_and_punctuation():
    pre_tokenizer = BasicPreTokenizer()

    result = pre_tokenizer.pre_tokenize(
        "Hello, world!"
    )

    assert result == [
        "Hello",
        ",",
        "world",
        "!",
    ]