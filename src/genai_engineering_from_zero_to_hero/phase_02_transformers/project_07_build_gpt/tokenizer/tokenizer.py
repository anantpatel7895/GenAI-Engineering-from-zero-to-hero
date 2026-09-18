from __future__ import annotations

from pathlib import Path

from .base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding tokenizer.

    Production implementation developed throughout Project 7.
    """

    def __init__(self):

        self._trained = False

    def train(self, text: str) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path):

        raise NotImplementedError