from pathlib import Path

from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.tokenizer import (
    BPETokenizer,
)


def test_load_tokenizer():

    tokenizer = BPETokenizer()

    tokenizer.load(
        Path("artifacts/tokenizer.json")
    )

    assert len(tokenizer.vocabulary) > 0
    assert len(tokenizer.merge_rules) > 0