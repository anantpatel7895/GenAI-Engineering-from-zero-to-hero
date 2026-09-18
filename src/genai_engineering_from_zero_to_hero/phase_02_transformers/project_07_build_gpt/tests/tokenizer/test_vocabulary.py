from pathlib import Path

import pytest

from tokenizer.vocabulary import Vocabulary


@pytest.fixture
def vocabulary() -> Vocabulary:
    vocab = Vocabulary()
    vocab.add("hello")
    vocab.add("world")
    return vocab


def test_create_empty_vocabulary() -> None:
    vocab = Vocabulary()

    assert len(vocab) == 0


def test_add_single_token() -> None:
    vocab = Vocabulary()

    token_id = vocab.add("hello")

    assert token_id == 0
    assert len(vocab) == 1


def test_add_multiple_tokens(vocabulary: Vocabulary) -> None:
    assert len(vocabulary) == 2


def test_duplicate_token_raises(vocabulary: Vocabulary) -> None:
    with pytest.raises(ValueError):
        vocabulary.add("hello")


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("hello", 0),
        ("world", 1),
    ],
)
def test_token_to_id(
    vocabulary: Vocabulary,
    token: str,
    expected: int,
) -> None:
    assert vocabulary.token_to_id(token) == expected


def test_unknown_token_raises(vocabulary: Vocabulary) -> None:
    with pytest.raises(KeyError):
        vocabulary.token_to_id("python")


@pytest.mark.parametrize(
    ("token_id", "expected"),
    [
        (0, "hello"),
        (1, "world"),
    ],
)
def test_id_to_token(
    vocabulary: Vocabulary,
    token_id: int,
    expected: str,
) -> None:
    assert vocabulary.id_to_token(token_id) == expected


def test_unknown_id_raises(vocabulary: Vocabulary) -> None:
    with pytest.raises(KeyError):
        vocabulary.id_to_token(100)


def test_contains(vocabulary: Vocabulary) -> None:
    assert "hello" in vocabulary
    assert "python" not in vocabulary


def test_save_and_load(
    vocabulary: Vocabulary,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "vocab.json"

    vocabulary.save(file_path)

    loaded = Vocabulary.load(file_path)

    assert len(loaded) == 2
    assert loaded.token_to_id("hello") == 0
    assert loaded.id_to_token(1) == "world"