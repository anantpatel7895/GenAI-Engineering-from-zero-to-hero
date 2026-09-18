"""
Configuration used throughout the GPT project.

Every module receives a GPTConfig instance instead of
hard-coding model parameters.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class GPTConfig:
    """
    Configuration for a decoder-only GPT model.
    """

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    vocab_size: int = 50_257

    # --------------------------------------------------
    # Input
    # --------------------------------------------------

    context_length: int = 256

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    embedding_dim: int = 768

    num_layers: int = 12

    num_heads: int = 12

    dropout: float = 0.1

    bias: bool = True

    # --------------------------------------------------
    # Feed Forward Network
    # --------------------------------------------------

    expansion_factor: int = 4

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    batch_size: int = 8

    learning_rate: float = 3e-4

    weight_decay: float = 0.01

    epochs: int = 10

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    temperature: float = 1.0

    top_k: int = 50