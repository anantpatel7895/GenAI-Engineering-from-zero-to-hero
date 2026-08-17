"""
Debug tokenizer inference.

Run:

uv run python debug/debug_tokenizer.py
"""

from pathlib import Path

from genai_engineering_from_zero_to_hero.phase_02_transformers.project_02_tokenizer.tokenizer import (
    BPETokenizer,
)


def main():

    tokenizer = BPETokenizer()

    tokenizer.load(
        Path("artifacts/tokenizer.json"),
        verbose=True,
    )

    print()

    while True:

        print("=" * 80)

        text = input("Enter text (q to quit): ")

        if text.lower() in {"q", "quit", "exit"}:
            break

        print()

        tokens = tokenizer.encode_to_tokens(
            text,
            verbose=True,
        )

        print()

        ids = tokenizer.encode(
            text,
            verbose=True,
        )

        print()

        decoded = tokenizer.decode(
            ids,
            verbose=True,
        )

        print()

        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        print(f"Input   : {text}")
        print(f"Tokens  : {tokens}")
        print(f"IDs     : {ids}")
        print(f"Decoded : {decoded}")

        print()


if __name__ == "__main__":
    main()