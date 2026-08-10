# 12_multi_head_attention.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
batch_size = 2
seq_len = 4
embed_dim = 8
num_heads = 2

assert embed_dim % num_heads == 0

head_dim = embed_dim // num_heads

# -------------------------------------------------
# Step 1: Create input embedding tensor X
# Shape: (batch_size, seq_len, embed_dim)
# -------------------------------------------------
X = torch.randn(batch_size, seq_len, embed_dim)

print("Input X Shape:", X.shape)

# -------------------------------------------------
# Step 2: Linear layers for Q, K, V
# -------------------------------------------------
W_q = nn.Linear(embed_dim, embed_dim)
W_k = nn.Linear(embed_dim, embed_dim)
W_v = nn.Linear(embed_dim, embed_dim)

Q = W_q(X)
K = W_k(X)
V = W_v(X)

print("\nAfter Linear Projection")
print("Q Shape:", Q.shape)
print("K Shape:", K.shape)
print("V Shape:", V.shape)

# -------------------------------------------------
# Step 3: Split into multiple heads
# (batch, seq_len, embed_dim)
#        ↓
# (batch, heads, seq_len, head_dim)
# -------------------------------------------------
Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

print("\nAfter Splitting into Heads")
print("Q Shape:", Q.shape)
print("K Shape:", K.shape)
print("V Shape:", V.shape)

# -------------------------------------------------
# Step 4: Compute Scaled Dot-Product Attention
# -------------------------------------------------
scores = torch.matmul(Q, K.transpose(-2, -1))
scores = scores / math.sqrt(head_dim)

print("\nAttention Scores Shape:", scores.shape)

attention_weights = F.softmax(scores, dim=-1)

print("Attention Weights Shape:", attention_weights.shape)

head_outputs = torch.matmul(attention_weights, V)

print("Head Outputs Shape:", head_outputs.shape)

# -------------------------------------------------
# Step 5: Concatenate Heads
# (batch, heads, seq_len, head_dim)
#        ↓
# (batch, seq_len, embed_dim)
# -------------------------------------------------
concat = (
    head_outputs
    .transpose(1, 2)
    .contiguous()
    .view(batch_size, seq_len, embed_dim)
)

print("\nConcatenated Output Shape:", concat.shape)

# -------------------------------------------------
# Step 6: Final Linear Projection
# -------------------------------------------------
output_projection = nn.Linear(embed_dim, embed_dim)

output = output_projection(concat)

print("Final Output Shape:", output.shape)

# -------------------------------------------------
# Done
# -------------------------------------------------
print("\nMulti-Head Attention completed successfully!")