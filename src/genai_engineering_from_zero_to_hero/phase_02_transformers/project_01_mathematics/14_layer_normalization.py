import torch
import torch.nn as nn

# Set seed for reproducibility
torch.manual_seed(42)

# --------------------------------------------------
# Step 1: Create Random Embedding Tensor
# --------------------------------------------------

batch_size = 2
seq_len = 3
embed_dim = 4

embedding = torch.randn(batch_size, seq_len, embed_dim)

print("=" * 60)
print("Original Embedding")
print("=" * 60)
print(embedding)

# --------------------------------------------------
# Step 2: Compute Mean Manually
# --------------------------------------------------

mean = embedding.mean(dim=-1, keepdim=True)

print("\n" + "=" * 60)
print("Mean (along embedding dimension)")
print("=" * 60)
print(mean)

# --------------------------------------------------
# Step 3: Compute Variance Manually
# --------------------------------------------------

variance = ((embedding - mean) ** 2).mean(dim=-1, keepdim=True)

print("\n" + "=" * 60)
print("Variance")
print("=" * 60)
print(variance)

# --------------------------------------------------
# Step 4: Normalize
# --------------------------------------------------

eps = 1e-5

normalized = (embedding - mean) / torch.sqrt(variance + eps)

print("\n" + "=" * 60)
print("Normalized Tensor")
print("=" * 60)
print(normalized)

# --------------------------------------------------
# Step 5: Verify Mean ≈ 0
# --------------------------------------------------

normalized_mean = normalized.mean(dim=-1)

print("\n" + "=" * 60)
print("Mean After Normalization")
print("=" * 60)
print(normalized_mean)

# --------------------------------------------------
# Step 6: Verify Variance ≈ 1
# --------------------------------------------------

normalized_var = normalized.var(dim=-1, unbiased=False)

print("\n" + "=" * 60)
print("Variance After Normalization")
print("=" * 60)
print(normalized_var)

# --------------------------------------------------
# Step 7: Learnable Gamma and Beta
# --------------------------------------------------

gamma = torch.nn.Parameter(torch.ones(embed_dim))
beta = torch.nn.Parameter(torch.zeros(embed_dim))

print("\n" + "=" * 60)
print("Gamma")
print("=" * 60)
print(gamma)

print("\n" + "=" * 60)
print("Beta")
print("=" * 60)
print(beta)

output = gamma * normalized + beta

print("\n" + "=" * 60)
print("Output After Applying Gamma and Beta")
print("=" * 60)
print(output)

# --------------------------------------------------
# Step 8: Compare with PyTorch LayerNorm
# --------------------------------------------------

layer_norm = nn.LayerNorm(embed_dim)

# Copy gamma and beta so both implementations use the same parameters
layer_norm.weight.data = gamma.data.clone()
layer_norm.bias.data = beta.data.clone()

torch_output = layer_norm(embedding)

print("\n" + "=" * 60)
print("PyTorch LayerNorm Output")
print("=" * 60)
print(torch_output)

# --------------------------------------------------
# Step 9: Difference
# --------------------------------------------------

difference = torch.abs(output - torch_output).max()

print("\n" + "=" * 60)
print("Maximum Difference")
print("=" * 60)
print(difference)