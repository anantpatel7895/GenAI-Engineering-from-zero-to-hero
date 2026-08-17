from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.bpe import (
    count_pairs, merge_pair, BPETrainer
)


def test_count_pairs():
    symbols = ["l", "o", "w", "e", "r"]

    pairs = count_pairs(symbols)

    assert pairs[("l", "o")] == 1
    assert pairs[("o", "w")] == 1
    assert pairs[("w", "e")] == 1
    assert pairs[("e", "r")] == 1

def test_merge_pair():
    symbols = ["l", "o", "w", "e", "r"]

    result = merge_pair(
        symbols,
        ("l", "o"),
    )

    assert result == ["lo", "w", "e", "r"]

def test_merge_repeated_pair():
    symbols = ["a", "b", "a", "b"]

    result = merge_pair(
        symbols,
        ("a", "b"),
    )

    assert result == ["ab", "ab"]

def test_bpe_trainer():
    corpus = [
        ["l", "o", "w"],
        ["l", "o", "w", "e", "r"],
        ["l", "o", "w", "e", "s", "t"],
    ]

    trainer = BPETrainer(num_merges=3)

    trainer.train(corpus)

    assert trainer.merge_rules == [
        ("l", "o"),
        ("lo", "w"),
        ("low", "e"),
    ]

    assert "lo" in trainer.vocabulary
    assert "low" in trainer.vocabulary
    assert "lowe" in trainer.vocabulary

def test_merge_overlapping_pair():
    symbols = ["a", "a", "a"]

    result = merge_pair(
        symbols,
        ("a", "a"),
    )

    assert result == ["aa", "a"]

def test_merge_pair_not_present():
    symbols = ["a", "b", "c"]

    result = merge_pair(
        symbols,
        ("x", "y"),
    )

    assert result == ["a", "b", "c"]

def test_count_pairs_empty():
    assert count_pairs([]) == {}