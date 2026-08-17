"""
Debug BPE training.

Run:

uv run python debug/debug_trainer.py
"""

from pathlib import Path

from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.normalizer import (
    BasicNormalizer,
)
from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.pre_tokenizer import (
    BasicPreTokenizer,
)
from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.trainer import (
    BPETrainer,
)


def load_corpus(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def to_initial_symbols(
    tokens: list[str],
) -> list[list[str]]:
    return [list(token) for token in tokens]


def main():

    corpus = load_corpus(
        Path("data/corpus.txt")
    )

    normalizer = BasicNormalizer(lowercase=True)

    pre_tokenizer = BasicPreTokenizer()

    normalized = [
        normalizer.normalize(sentence)
        for sentence in corpus
    ]

    pretokenized = [
        pre_tokenizer.pre_tokenize(sentence)
        for sentence in normalized
    ]

    symbol_corpus = [
        to_initial_symbols(tokens)
        for tokens in pretokenized
    ]

    trainer = BPETrainer(
        vocab_size=100,
    )

    final_corpus = trainer.train(
        symbol_corpus,
        verbose=True,
    )

    trainer.assign_token_ids(
        verbose=True,
    )

    trainer.save(
        Path("artifacts/tokenizer.json"),
        verbose=True,
    )

    print("=" * 80)
    print("FINAL VOCABULARY")
    print("=" * 80)

    for token, idx in trainer.token_to_id.items():
        print(f"{idx:>4} -> {token}")

    print()

    print("=" * 80)
    print("MERGE RULES")
    print("=" * 80)

    for i, pair in enumerate(
        trainer.merge_rules,
        start=1,
    ):
        print(f"{i:>3}. {pair}")

    print()

    print("=" * 80)
    print("FINAL MERGED CORPUS")
    print("=" * 80)

    for sentence in final_corpus:
        print(sentence)


if __name__ == "__main__":
    main()