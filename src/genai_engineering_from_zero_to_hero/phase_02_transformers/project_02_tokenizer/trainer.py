"""
trainer.py

Train a Byte Pair Encoding (BPE) tokenizer from a symbol corpus.

Responsibilities:
1. Learn merge rules.
2. Build the final vocabulary.
3. Assign token IDs.
4. Save the tokenizer artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bpe import (
    count_symbol_pairs,
    get_best_pair,
    merge_pair,
)


class BPETrainer:
    """
    Learns a BPE tokenizer.

    Parameters
    ----------
    vocab_size:
        Maximum vocabulary size.
    special_tokens:
        Reserved tokens inserted at the beginning
        of the vocabulary.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        special_tokens: list[str] | None = None,
    ) -> None:

        self.vocab_size = vocab_size

        self.special_tokens = (
            special_tokens
            if special_tokens is not None
            else [
                "<unk>",
                "<pad>",
                "<bos>",
                "<eos>",
            ]
        )

        # Learned during training
        self.merge_rules: list[tuple[str, str]] = []

        self.vocabulary: list[str] = []

        self.token_to_id: dict[str, int] = {}

    # ==========================================================
    # Public API
    # ==========================================================

    def train(
        self,
        symbol_corpus: list[list[list[str]]],
        verbose: bool = False,
    ) -> list[list[list[str]]]:
        """
        Learn BPE merge rules.

        Returns
        -------
        Final merged corpus.
        """

        if verbose:
            print("=" * 80)
            print("BPE TRAINING")
            print("=" * 80)
            print()

        iteration = 1
        current_corpus = symbol_corpus

        while True:

            pair_counts = count_symbol_pairs(current_corpus)

            if not pair_counts:
                break

            best_pair, frequency = get_best_pair(pair_counts)

            if best_pair is None:
                break

            current_corpus = merge_pair(
                current_corpus,
                best_pair,
            )

            self.merge_rules.append(best_pair)

            current_vocab = self.build_vocabulary(current_corpus)

            if verbose:

                print("-" * 80)
                print(f"Iteration       : {iteration}")
                print(f"Best Pair       : {best_pair}")
                print(f"Frequency       : {frequency}")
                print(f"Vocabulary Size : {len(current_vocab)}")
                print("-" * 80)

            if (
                len(current_vocab)
                + len(self.special_tokens)
                >= self.vocab_size
            ):
                if verbose:
                    print()
                    print("Vocabulary limit reached.")
                    print()
                break

            iteration += 1

        self.vocabulary = self.build_vocabulary(current_corpus)

        self.assign_token_ids()

        if verbose:
            print("=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)
            print(f"Merge Rules Learned : {len(self.merge_rules)}")
            print(f"Vocabulary Size     : {len(self.vocabulary)}")
            print(f"Total Tokens        : {len(self.token_to_id)}")
            print()

        return current_corpus

    # ==========================================================
    # Vocabulary
    # ==========================================================

    def build_vocabulary(
        self,
        symbol_corpus: list[list[list[str]]],
    ) -> list[str]:
        """
        Build the vocabulary from the merged corpus.
        """

        vocabulary: set[str] = set()

        for sentence in symbol_corpus:

            for token in sentence:

                vocabulary.update(token)

        return sorted(vocabulary)

    # ==========================================================
    # Token IDs
    # ==========================================================

    def assign_token_ids(
        self,
        verbose: bool = False,
    ) -> None:
        """
        Assign integer IDs.
        """

        self.token_to_id.clear()

        index = 0

        if verbose:
            print("=" * 80)
            print("ASSIGN TOKEN IDS")
            print("=" * 80)

        # Special tokens first
        for token in self.special_tokens:

            self.token_to_id[token] = index

            if verbose:
                print(f"{index:>4} -> {token}")

            index += 1

        # Learned vocabulary
        for token in self.vocabulary:

            self.token_to_id[token] = index

            if verbose:
                print(f"{index:>4} -> {token}")

            index += 1

    # ==========================================================
    # Save
    # ==========================================================

    def save(
        self,
        path: Path,
        lowercase: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Save tokenizer artifact.
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifact: dict[str, Any] = {
            "version": "1.0",
            "model": "BPE",
            "special_tokens": {
                token: self.token_to_id[token]
                for token in self.special_tokens
            },
            "vocabulary": self.token_to_id,
            "merge_rules": [
                list(pair)
                for pair in self.merge_rules
            ],
            "normalizer": {
                "lowercase": lowercase,
            },
            "pre_tokenizer": {
                "type": "BasicPreTokenizer",
            },
        }

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                artifact,
                file,
                indent=4,
                ensure_ascii=False,
            )

        if verbose:

            print("=" * 80)
            print("TOKENIZER SAVED")
            print("=" * 80)
            print(f"Location : {path}")
            print(f"Vocabulary : {len(self.token_to_id)}")
            print(f"Merge Rules : {len(self.merge_rules)}")
            print()