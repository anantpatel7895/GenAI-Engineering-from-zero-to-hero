from config import GPTConfig


def main():

    config = GPTConfig()

    print("=" * 60)
    print("GPT Configuration")
    print("=" * 60)

    print(config)

    print()

    print(f"Vocabulary Size : {config.vocab_size}")
    print(f"Context Length  : {config.context_length}")
    print(f"Embedding Dim   : {config.embedding_dim}")
    print(f"Layers          : {config.num_layers}")
    print(f"Heads           : {config.num_heads}")
    print(f"Dropout         : {config.dropout}")
    print(f"Batch Size      : {config.batch_size}")


if __name__ == "__main__":
    main()