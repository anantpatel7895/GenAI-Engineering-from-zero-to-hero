"""
Vocabulary implementation.

Responsible for maintaining the mapping between
tokens and integer IDs.
"""

from __future__ import annotations

import json
from pathlib import Path


class Vocabulary:
    """Bidirectional token <-> id mapping."""

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def add(self, token: str) -> int:
        """
        Add a token to the vocabulary.

        Returns:
            Assigned token ID.

        Raises:
            ValueError: If token already exists.
        """
        if token in self._token_to_id:
            raise ValueError(f"Token already exists: {token!r}")

        token_id = len(self._token_to_id)

        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token

        return token_id

    def token_to_id(self, token: str) -> int:
        """
        Convert token to integer ID.
        """
        return self._token_to_id[token]

    def id_to_token(self, token_id: int) -> str:
        """
        Convert integer ID back to token.
        """
        return self._id_to_token[token_id]

    def save(self, path: str | Path) -> None:
        """
        Save vocabulary as JSON.
        """
        path = Path(path)

        with path.open("w", encoding="utf-8") as fp:
            json.dump(
                self._token_to_id,
                fp,
                indent=4,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """
        Load vocabulary from JSON.
        """
        path = Path(path)

        with path.open("r", encoding="utf-8") as fp:
            token_to_id: dict[str, int] = json.load(fp)

        vocab = cls()

        vocab._token_to_id = token_to_id
        vocab._id_to_token = {
            token_id: token
            for token, token_id in token_to_id.items()
        }

        return vocab

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def __len__(self) -> int:
        return len(self._token_to_id)