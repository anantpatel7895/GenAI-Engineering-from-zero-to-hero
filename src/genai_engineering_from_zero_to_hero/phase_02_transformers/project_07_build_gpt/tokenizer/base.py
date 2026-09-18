"""
Abstract tokenizer interface.

Every tokenizer implementation in this project
must follow this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTokenizer(ABC):
    """
    Abstract tokenizer interface.
    """

    @abstractmethod
    def train(self, text: str) -> None:
        """
        Train the tokenizer on a corpus.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Convert text into token IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back to text.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, directory: Path) -> None:
        """
        Save tokenizer files.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "BaseTokenizer":
        """
        Load tokenizer files.
        """
        raise NotImplementedError