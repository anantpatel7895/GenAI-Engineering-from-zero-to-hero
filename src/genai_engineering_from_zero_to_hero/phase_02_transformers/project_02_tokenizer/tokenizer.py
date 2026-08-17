"""
tokenizer.py

Inference-only Byte Pair Encoding (BPE) tokenizer.

Responsibilities
----------------
1. Load tokenizer artifacts.
2. Normalize text.
3. Pre-tokenize text.
4. Apply learned BPE merge rules.
5. Convert tokens to IDs.
6. Convert IDs back to tokens.
"""

import json
from __future__ import annotations

from pathlib import Path

from .normalizer import BasicNormalizer
from .pre_tokenizer import BasicPreTokenizer

from .vocabulary import Vocabulary


class Tokenizer:
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        return [
            self.vocabulary.token_to_id_lookup(token)
            for token in tokens
        ]

    def decode_ids(self, token_ids: list[int]) -> list[str]:
        return [
            self.vocabulary.id_to_token_lookup(token_id)
            for token_id in token_ids
        ]


class BPETokenizer:
    """
    BPE tokenizer used during inference.
    """

    def __init__(self) -> None:

        self.normalizer = BasicNormalizer(lowercase=True)
        self.pre_tokenizer = BasicPreTokenizer()

        self.merge_rules: list[tuple[str, str]] = []

        self.token_to_id: dict[str, int] = {}

        self.id_to_token: dict[int, str] = {}

    # ==========================================================
    # Load
    # ==========================================================

    def load(
        self,
        path: Path,
        verbose: bool = False,
    ) -> None:
        """
        Load tokenizer artifact.
        """

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            artifact = json.load(file)

        self.merge_rules = [
            tuple(pair)
            for pair in artifact["merge_rules"]
        ]

        self.token_to_id = artifact["vocabulary"]

        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

        lowercase = artifact["normalizer"].get(
            "lowercase",
            True,
        )

        self.normalizer = BasicNormalizer(
            lowercase=lowercase,
        )

        if verbose:

            print("=" * 80)
            print("TOKENIZER LOADED")
            print("=" * 80)

            print(f"Vocabulary Size : {len(self.token_to_id)}")
            print(f"Merge Rules     : {len(self.merge_rules)}")

            print()

    # ==========================================================
    # Merge Utilities
    # ==========================================================

    @staticmethod
    def merge_once(
        symbols: list[str],
        pair: tuple[str, str],
    ) -> list[str]:
        """
        Apply one merge rule.
        """

        merged = []

        i = 0

        while i < len(symbols):

            if (
                i < len(symbols) - 1
                and (symbols[i], symbols[i + 1]) == pair
            ):

                merged.append(
                    symbols[i] + symbols[i + 1]
                )

                i += 2

            else:

                merged.append(symbols[i])

                i += 1

        return merged

    def apply_merges(
        self,
        symbols: list[str],
        verbose: bool = False,
    ) -> list[str]:
        """
        Apply every learned merge rule.
        """

        current = symbols

        if verbose:
            print()
            print("Initial Symbols")
            print(current)

        for pair in self.merge_rules:

            before = current

            current = self.merge_once(
                current,
                pair,
            )

            if verbose and before != current:

                print()
                print(f"Applied Merge : {pair}")
                print(current)

        if verbose:
            print()
            print("Final Symbols")
            print(current)

        return current

    # ==========================================================
    # Encode
    # ==========================================================

    def encode_to_tokens(
        self,
        text: str,
        verbose: bool = False,
    ) -> list[str]:
        """
        Convert raw text into BPE tokens.
        """

        if verbose:

            print("=" * 80)
            print("TOKENIZER INFERENCE")
            print("=" * 80)

        normalized = self.normalizer.normalize(text)

        if verbose:

            print("\n[1] RAW TEXT")
            print("-" * 80)
            print(text)

            print("\n[2] NORMALIZED")
            print("-" * 80)
            print(normalized)

        pre_tokens = self.pre_tokenizer.pre_tokenize(
            normalized
        )

        if verbose:

            print("\n[3] PRE-TOKENS")
            print("-" * 80)
            print(pre_tokens)

        output_tokens = []

        for token in pre_tokens:

            if verbose:

                print("\n" + "=" * 80)
                print(f"PROCESS TOKEN : {token}")
                print("=" * 80)

            symbols = list(token)

            merged = self.apply_merges(
                symbols,
                verbose=verbose,
            )

            output_tokens.extend(merged)

        if verbose:

            print("\n" + "=" * 80)
            print("FINAL SUBWORD TOKENS")
            print("=" * 80)
            print(output_tokens)

        return output_tokens

    def encode(
        self,
        text: str,
        verbose: bool = False,
    ) -> list[int]:
        """
        Convert text into token IDs.
        """

        tokens = self.encode_to_tokens(
            text,
            verbose=verbose,
        )

        ids = []

        if verbose:

            print("\n" + "=" * 80)
            print("TOKEN → ID")
            print("=" * 80)

        unk_id = self.token_to_id["<unk>"]

        for token in tokens:

            token_id = self.token_to_id.get(
                token,
                unk_id,
            )

            ids.append(token_id)

            if verbose:
                print(f"{token:<20} -> {token_id}")

        if verbose:

            print("\nFinal Token IDs")
            print(ids)

        return ids

    # ==========================================================
    # Decode
    # ==========================================================

    def decode(
        self,
        ids: list[int],
        verbose: bool = False,
    ) -> str:
        """
        Convert token IDs back into text.
        """

        tokens = []

        if verbose:

            print("=" * 80)
            print("DECODING")
            print("=" * 80)

        for token_id in ids:

            token = self.id_to_token.get(
                token_id,
                "<unk>",
            )

            tokens.append(token)

            if verbose:
                print(f"{token_id:<5} -> {token}")

        text = "".join(tokens)

        if verbose:

            print("\nDecoded Text")
            print(text)

        return text