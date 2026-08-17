"""
train_tokenizer.py
"""

from pathlib import Path
from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.normalizer import (
    BasicNormalizer,
)
from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.pre_tokenizer import (
    BasicPreTokenizer,
)

from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.bpe import (
    count_symbol_pairs,
)


from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.bpe import (
    count_symbol_pairs,
    get_best_pair,
    merge_pair,
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

normalizer = BasicNormalizer(lowercase=True)
pre_tokenizer = BasicPreTokenizer()

def to_initial_symbols(tokens: list[str]) -> list[list[str]]:
    """
    Convert each pre-token into a list of character symbols.
    """
    return [list(token) for token in tokens]

def main():
    corpus_path = Path("data/corpus.txt")

    corpus = load_corpus(corpus_path)

    print("=" * 60)
    print("Tokenizer Training")
    print("=" * 60)

    print(f"\nLoaded {len(corpus)} sentences:\n")

    # for sentence in corpus:
    #     print(sentence)

    normalized_corpus = [
        normalizer.normalize(sentence)
        for sentence in corpus
    ]

    pretokenized_corpus = [
        pre_tokenizer.pre_tokenize(sentence)
        for sentence in normalized_corpus
    ]

    symbol_corpus = [
        to_initial_symbols(tokens)
        for tokens in pretokenized_corpus
    ]
    print("\n" + "=" * 60)
    print("Normalization")
    print("=" * 60)

    for raw, normalized in zip(corpus, normalized_corpus):
        print(f"RAW : {raw}")
        print(f"NORM: {normalized}")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("Pre-tokenization")
    print("=" * 60)

    for sentence, tokens in zip(normalized_corpus, pretokenized_corpus):
        print(f"TEXT  : {sentence}")
        print(f"TOKENS: {tokens}")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("Initial Symbol Representation")
    print("=" * 60)

    for tokens, symbols in zip(pretokenized_corpus, symbol_corpus):
        print(f"TOKENS : {tokens}")
        print(f"SYMBOLS: {symbols}")
        print("-" * 40)

    # pair_counts = count_symbol_pairs(symbol_corpus)

    # print("\n" + "=" * 60)
    # print("Adjacent Symbol Pair Counts")
    # print("=" * 60)

    # for pair, count in pair_counts.most_common():
    #     print(f"{pair} : {count}")

    # best_pair, frequency = get_best_pair(pair_counts)

    # print("\n" + "=" * 60)
    # print("Best Pair")
    # print("=" * 60)

    # print(f"Pair      : {best_pair}")
    # print(f"Frequency : {frequency}")

    # merged_corpus = merge_pair(symbol_corpus, best_pair)

    # print("\n" + "=" * 60)
    # print("Merge Result")
    # print("=" * 60)

    # for before, after in zip(symbol_corpus, merged_corpus):

    #     print("BEFORE")

    #     print(before)

    #     print()

    #     print("AFTER")

    #     print(after)

    #     print("-" * 60)

    trainer = BPETrainer()

    trained_corpus = trainer.train(
        symbol_corpus=symbol_corpus,
        num_merges=10,
    )

    print("\n" + "=" * 60)
    print("Learned Merge Rules")
    print("=" * 60)

    for i, rule in enumerate(trainer.merge_rules, start=1):
        print(f"{i:02d}. {rule}")

    print("\n" + "=" * 60)
    print("Vocabulary")
    print("=" * 60)

    for token in sorted(trainer.vocabulary):
        print(token)


    output = Path(
        "artifacts/tokenizer.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save(output)

    print("\nTokenizer saved to")

    print(output)
    


if __name__ == "__main__":
    main()