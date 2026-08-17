from collections import Counter
from dataclasses import dataclass
from collections import Counter


Pair = tuple[str, str]

@dataclass(frozen=True)
class MergeRule:
    left: str
    right: str

    @property
    def merged(self) -> str:
        return self.left + self.right


def count_pairs(symbols: list[str]) -> Counter[Pair]:
    pairs: Counter[Pair] = Counter()

    for index in range(len(symbols) - 1):
        pair = (symbols[index], symbols[index + 1])
        pairs[pair] += 1

    return pairs

def count_corpus_pairs(
    corpus: list[list[str]],
) -> Counter[Pair]:
    pair_counts: Counter[Pair] = Counter()

    for symbols in corpus:
        pair_counts.update(count_pairs(symbols))

    return pair_counts

def get_most_frequent_pair(
    pair_counts: Counter[Pair],
) -> Pair | None:
    if not pair_counts:
        return None

    return pair_counts.most_common(1)[0][0]

def merge_pair(
    symbols: list[str],
    pair: Pair,
) -> list[str]:
    merged: list[str] = []

    index = 0

    while index < len(symbols):
        if (
            index < len(symbols) - 1
            and symbols[index] == pair[0]
            and symbols[index + 1] == pair[1]
        ):
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1

    return merged

def merge_corpus(
    corpus: list[list[str]],
    pair: Pair,
) -> list[list[str]]:
    return [
        merge_pair(symbols, pair)
        for symbols in corpus
    ]


def count_symbol_pairs(symbol_corpus: list[list[list[str]]]) -> Counter:
    """
    Count every adjacent symbol pair across the entire corpus.

    Input:

    [
        [
            ['l','o','v','e'],
            ['m','a','c','h','i','n','e']
        ],
        ...
    ]

    Output:

    Counter({
        ('l','o'): 2,
        ('o','v'): 2,
        ...
    })
    """

    pair_counts = Counter()

    # Iterate over every sentence
    for sentence in symbol_corpus:

        # Iterate over every token
        for token in sentence:

            # Skip tokens with fewer than 2 symbols
            if len(token) < 2:
                continue

            # Count adjacent pairs
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] += 1

    return pair_counts

def get_best_pair(pair_counts: Counter) -> tuple[tuple[str, str], int]:
    """
    Return the most frequent adjacent symbol pair
    along with its frequency.
    """

    if not pair_counts:
        raise ValueError("No symbol pairs found.")

    best_pair, frequency = pair_counts.most_common(1)[0]

    return best_pair, frequency

def merge_pair(
    symbol_corpus: list[list[list[str]]],
    pair_to_merge: tuple[str, str],
) -> list[list[list[str]]]:
    """
    Merge the selected symbol pair across the corpus.

    Example:

    ("i", "n")

    ["m","a","c","h","i","n","e"]

    →

    ["m","a","c","h","in","e"]
    """

    merged_symbol = "".join(pair_to_merge)

    new_corpus = []

    for sentence in symbol_corpus:

        new_sentence = []

        for token in sentence:

            merged_token = []

            i = 0

            while i < len(token):

                # Is there a next symbol?
                if (
                    i < len(token) - 1
                    and token[i] == pair_to_merge[0]
                    and token[i + 1] == pair_to_merge[1]
                ):

                    merged_token.append(merged_symbol)

                    i += 2

                else:

                    merged_token.append(token[i])

                    i += 1

            new_sentence.append(merged_token)

        new_corpus.append(new_sentence)

    return new_corpus

class BPETrainer:
    def __init__(self, num_merges: int):
        if num_merges < 0:
            raise ValueError("num_merges must be >= 0")

        self.num_merges = num_merges
        self.merge_rules: list[MergeRule] = []
        self.vocabulary: set[str] = set()

    def _build_initial_vocabulary(
        self,
        corpus: list[list[str]],
    ) -> set[str]:
        vocabulary: set[str] = set()

        for symbols in corpus:
            vocabulary.update(symbols)

        return vocabulary

    def train(
        self,
        corpus: list[list[str]],
    ) -> None:
        corpus = [
            list(symbols)
            for symbols in corpus
        ]

        self.vocabulary = self._build_initial_vocabulary(
            corpus
        )

        for _ in range(self.num_merges):
            pair_counts = count_corpus_pairs(corpus)

            best_pair = get_most_frequent_pair(
                pair_counts
            )

            if best_pair is None:
                break

            rule = MergeRule(
                left=best_pair[0],
                right=best_pair[1],
            )

            corpus = merge_corpus(
                corpus,
                best_pair,
            )

            self.merge_rules.append(rule)
            self.vocabulary.add(rule.merged)



if __name__ == "__main__":
    
    corpus = [
    ["l", "o", "w"],
    ["l", "o", "w", "e", "r"],
    ["l", "o", "w", "e", "s", "t"],
]

    trainer = BPETrainer(num_merges=3)

    trainer.train(corpus)

    for rule in trainer.merge_rules:
        print(
            rule.left,
            "+",
            rule.right,
            "→",
            rule.merged,
        )

    print(trainer.vocabulary)

