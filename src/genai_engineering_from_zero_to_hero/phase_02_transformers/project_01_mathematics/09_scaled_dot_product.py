import torch
import math

# -------------------------------------------------
# Small Query and Key Vectors
# -------------------------------------------------
query = torch.tensor([1.0, 2.0, 3.0])

keys = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 2.0, 2.0],
    [3.0, 1.0, 0.0]
])

print("=" * 70)
print("Example 1: Small Values")
print("=" * 70)

print("Query:")
print(query)

print("\nKeys:")
print(keys)

# -------------------------------------------------
# Compute Dot Products
# -------------------------------------------------
scores = torch.matmul(keys, query)

print("\nDot Product Scores:")
print(scores)

# -------------------------------------------------
# Compute Scaled Dot Products
# -------------------------------------------------
dk = query.size(0)

scaled_scores = scores / math.sqrt(dk)

print("\nScaled Dot Product Scores:")
print(scaled_scores)

# -------------------------------------------------
# Apply Softmax
# -------------------------------------------------
unscaled_probs = torch.softmax(scores, dim=0)
scaled_probs = torch.softmax(scaled_scores, dim=0)

print("\nSoftmax (Unscaled Scores)")
print(unscaled_probs)

print("\nSoftmax (Scaled Scores)")
print(scaled_probs)

# -------------------------------------------------
# Compare
# -------------------------------------------------
print("\nSum of Unscaled Probabilities:",
      unscaled_probs.sum().item())

print("Sum of Scaled Probabilities:",
      scaled_probs.sum().item())


# =================================================
# Example 2 : Larger Values
# =================================================

print("\n")
print("=" * 70)
print("Example 2: Large Values")
print("=" * 70)

large_query = torch.tensor([5., 6., 7.])

large_keys = torch.tensor([
    [4., 5., 6.],
    [5., 6., 7.],
    [3., 2., 4.]
])

# large_query = query * 20
# large_keys = keys * 20

print("Large Query:")
print(large_query)

print("\nLarge Keys:")
print(large_keys)

large_scores = torch.matmul(large_keys, large_query)

large_scaled_scores = large_scores / math.sqrt(dk)

print("\nDot Product Scores:")
print(large_scores)

print("\nScaled Scores:")
print(large_scaled_scores)

large_unscaled_probs = torch.softmax(large_scores, dim=0)

large_scaled_probs = torch.softmax(large_scaled_scores, dim=0)

print("\nSoftmax (Unscaled)")
print(large_unscaled_probs)

print("\nSoftmax (Scaled)")
print(large_scaled_probs)