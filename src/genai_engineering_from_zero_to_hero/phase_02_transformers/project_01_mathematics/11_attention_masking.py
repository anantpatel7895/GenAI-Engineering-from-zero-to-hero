# 11_attention_masking.py

import torch
import torch.nn.functional as F

torch.manual_seed(42)

# -----------------------------------
# Step 1: Create a score matrix
# -----------------------------------
seq_len = 5

scores = torch.randn(seq_len, seq_len)

print("Original Scores:")
print(scores)

# -----------------------------------
# Step 2: Build a causal mask
# -----------------------------------
# True where future tokens should be masked
# upper triangular matrix with diagonal=1 creates a mask for future tokens
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, dtype=torch.bool),
    diagonal=1
)

print("\nCausal Mask:")
print(causal_mask)

# -----------------------------------
# Step 3: Apply masked_fill
# -----------------------------------

masked_scores = scores.masked_fill(causal_mask, float("-inf"))

print("\nMasked Scores:")
print(masked_scores)

# -----------------------------------
# Step 4: Apply Softmax
# -----------------------------------

attention = F.softmax(masked_scores, dim=-1)

print("\nAttention Weights:")
print(attention)

# -----------------------------------
# Step 5: Verify future positions are zero
# -----------------------------------

future_weights = attention.masked_select(causal_mask)

print("\nFuture Attention Weights:")
print(future_weights)

assert torch.all(future_weights == 0), \
    "Future positions are not zero!"

print("\n✓ Verification Passed!")
print("All future positions have exactly zero attention.")