from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.tokenizer import (
    Tokenizer,
)
from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.vocabulary import (
    Vocabulary,
)


from pathlib import Path

from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.tokenizer import (
    BPETokenizer,
)

def create_tokenizer() -> Tokenizer:
    vocabulary = Vocabulary(
        [
            "<UNK>",
            "I",
            "love",
            "AI",
        ]
    )

    return Tokenizer(vocabulary)


def test_encode_tokens():
    tokenizer = create_tokenizer()

    tokens = ["I", "love", "AI"]

    token_ids = tokenizer.encode_tokens(tokens)

    assert token_ids == [1, 2, 3]


def test_decode_ids():
    tokenizer = create_tokenizer()

    token_ids = [1, 2, 3]

    tokens = tokenizer.decode_ids(token_ids)

    assert tokens == ["I", "love", "AI"]


def test_encode_decode_round_trip():
    tokenizer = create_tokenizer()

    tokens = ["I", "love", "AI"]

    token_ids = tokenizer.encode_tokens(tokens)
    decoded_tokens = tokenizer.decode_ids(token_ids)

    assert decoded_tokens == tokens



tokenizer = BPETokenizer()

tokenizer.load(
    Path("artifacts/tokenizer.json")
)

print("Vocabulary Size")

print(len(tokenizer.vocabulary))

print()

print("Merge Rules")

for rule in tokenizer.merge_rules:
    print(rule)

from pathlib import Path

from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.tokenizer import (
    BPETokenizer,
)

tokenizer = BPETokenizer()

tokenizer.load(
    Path("artifacts/tokenizer.json")
)

tokens = tokenizer.encode_to_tokens(
    "Machine learning is powerful."
)

print(tokens)