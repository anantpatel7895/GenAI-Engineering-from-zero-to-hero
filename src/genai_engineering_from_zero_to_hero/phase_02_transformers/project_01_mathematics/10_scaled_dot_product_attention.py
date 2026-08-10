import torch
import math

# ---------------------------------------
# Set random seed for reproducibility
# ---------------------------------------
torch.manual_seed(42)

# ---------------------------------------
# considering a one sentence with 4 tokens, each represented by a 6-dimensional embedding vector
# Create a small embedding matrix X 
# (4 tokens, embedding dimension = 6)
# ---------------------------------------
X = torch.randn(4, 6)

print("Embedding Matrix X")
print(X)
print("Shape:", X.shape)

# ---------------------------------------
# Create random weight matrices
# ---------------------------------------
d_model = 6          # dimension of the model or embedding dimension
dk = 6               # dimension of the key/query vectors
dv = 6               # dimension of the value vectors


Wq = torch.randn(d_model, dk) # query weight matrix
Wk = torch.randn(d_model, dk) # key weight matrix
Wv = torch.randn(d_model, dv) # value weight matrix

print("\nWq Shape:", Wq.shape)
print("Wk Shape:", Wk.shape)
print("Wv Shape:", Wv.shape)

# ---------------------------------------
# Compute Q, K, V
# ---------------------------------------
Q = X @ Wq # Query vectors for each token of the sentence or query values matrix
K = X @ Wk # Key vectors
V = X @ Wv #` Value vectors`

print("\nQ Shape:", Q.shape)
print("K Shape:", K.shape)
print("V Shape:", V.shape)

# ---------------------------------------
# Compute Attention Scores
# ---------------------------------------
attention_scores = Q @ K.T

print("\nScores Shape:", attention_scores.shape)
print(attention_scores)

# ---------------------------------------
# Scale the Scores
# ---------------------------------------
scaled_scores = attention_scores / math.sqrt(dk)

print("\nScaled Scores Shape:", scaled_scores.shape)
print(scaled_scores)


# ---------------------------------------
# Apply Softmax
# ---------------------------------------
attention = torch.softmax(scaled_scores, dim=1) # 

print("\nAttention Matrix Shape:", attention.shape)
print(attention)

# ---------------------------------------
# Verify each row sums to 1
# ---------------------------------------
print("\nRow Sums:")
print(attention.sum(dim=1))

# ---------------------------------------
# Compute Final Output
# ---------------------------------------
output = attention @ V

print("\nOutput Shape:", output.shape)
print(output)


# ---------------------------------------
# Note:
# ---------------------------------------   

"""
1. The attention mechanism does not change the tensor shape. Instead, it changes the representation of each token by mixing information from the other tokens according to the attention weights.
2. Create a new representation for each token by taking a weighted average of all Value vectors.

```
New Token 1

=

0.00 × V₁

+

0.94 × V₂

+

0.02 × V₃

+

0.04 × V₄
```
3. Token 1 becomes context-aware because its new representation is influenced by the Value vectors of the tokens it attends to.
"""