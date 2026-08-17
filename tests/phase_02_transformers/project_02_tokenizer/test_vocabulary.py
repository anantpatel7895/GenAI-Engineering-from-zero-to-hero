from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.vocabulary import (
    Vocabulary,
)

def test_token_to_id():
    vocab = Vocabulary([
        "<PAD>",
        "<UNK>",
        "hello",
        "world",
    ])

    assert vocab.token_to_id_lookup("hello") == 2


def test_id_to_token():
    vocab = Vocabulary([
        "<PAD>",
        "<UNK>",
        "hello",
        "world",
    ])

    assert vocab.id_to_token_lookup(3) == "world"


def test_vocabulary_size():
    vocab = Vocabulary([
        "<PAD>",
        "<UNK>",
        "hello",
        "world",
    ])

    assert len(vocab) == 4