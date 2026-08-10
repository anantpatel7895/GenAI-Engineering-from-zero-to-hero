# 13_positional_encoding.py

import math
import torch

torch.manual_seed(42)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
seq_len = 6
embed_dim = 8

# -------------------------------------------------
# Step 1: Create token embeddings
# Shape: (seq_len, embed_dim)
# -------------------------------------------------
token_embeddings = torch.randn(seq_len, embed_dim)

print("=" * 60)
print("Token Embeddings")
print("=" * 60)
print(token_embeddings)

# -------------------------------------------------
# Step 2: Create sinusoidal positional encoding
# -------------------------------------------------
position = torch.arange(seq_len).unsqueeze(1).float()

# Even indices: 0, 2, 4, ...
# div_term == 1 / 10000^(2i/d), rewritten via a^x = e^(x·ln a) for numerical
# stability: e^(-ln(10000) · 2i/d). So dividing by the power in the paper's
# formula becomes multiplying by div_term below.
div_term = torch.exp(
    torch.arange(0, embed_dim, 2).float()
    * (-math.log(10000.0) / embed_dim)
)

positional_encoding = torch.zeros(seq_len, embed_dim)

# Apply sine to even dimensions
positional_encoding[:, 0::2] = torch.sin(position * div_term)

# Apply cosine to odd dimensions
positional_encoding[:, 1::2] = torch.cos(position * div_term)

print("\n" + "=" * 60)
print("Positional Encodings")
print("=" * 60)
print(positional_encoding)

# -------------------------------------------------
# Step 3: Add positional encoding
# -------------------------------------------------
input_embeddings = token_embeddings + positional_encoding

print("\n" + "=" * 60)
print("Final Input Embeddings")
print("=" * 60)
print(input_embeddings)

# -------------------------------------------------
# Step 4: Verify uniqueness
# -------------------------------------------------
print("\n" + "=" * 60)
print("Verify Unique Positional Encodings")
print("=" * 60)

unique = True

for i in range(seq_len):
    for j in range(i + 1, seq_len):
        if torch.allclose(positional_encoding[i], positional_encoding[j]):
            unique = False
            print(f"Positions {i} and {j} are identical!")

if unique:
    print("✓ Every position has a unique positional encoding.")

print("\nEncoding for each position:")
for i in range(seq_len):
    print(f"Position {i}:")
    print(positional_encoding[i])