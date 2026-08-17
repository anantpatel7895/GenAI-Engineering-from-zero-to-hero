"""
Compare our tokenizer with Hugging Face tokenizers.

Run

uv run python debug/compare_with_huggingface.py
"""

from pathlib import Path

from transformers import AutoTokenizer

from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.tokenizer import (
    BPETokenizer,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "bert-base-uncased"

TEST_SENTENCES = [
    "Machine learning is powerful.",
    "I love deep learning.",
    "Transformers are amazing!",
    "Artificial Intelligence is changing the world.",
]


# ---------------------------------------------------------
# Load Tokenizers
# ---------------------------------------------------------

our_tokenizer = BPETokenizer()

our_tokenizer.load(
    Path("artifacts/tokenizer.json")
)

hf_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

for text in TEST_SENTENCES:

    print("=" * 100)
    print("INPUT")
    print("=" * 100)

    print(text)

    print()

    # -----------------------------------------------------

    print("-" * 100)
    print("OUR TOKENIZER")
    print("-" * 100)

    our_tokens = our_tokenizer.encode_to_tokens(text)

    our_ids = our_tokenizer.encode(text)

    print("Tokens")

    print(our_tokens)

    print()

    print("IDs")

    print(our_ids)

    print()

    # -----------------------------------------------------

    print("-" * 100)
    print("HUGGING FACE TOKENIZER")
    print("-" * 100)

    hf_tokens = hf_tokenizer.tokenize(text)

    hf_ids = hf_tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    print("Tokens")

    print(hf_tokens)

    print()

    print("IDs")

    print(hf_ids)

    print()

    print("=" * 100)

    print()