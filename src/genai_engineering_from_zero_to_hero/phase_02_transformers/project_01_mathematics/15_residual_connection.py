import torch
import torch.nn as nn


# 1. Create a random embedding tensor x
# Shape: (batch_size, sequence_length, embedding_dim)
x = torch.randn(2, 4, 8)


# 2. Simulate F(x) using a linear layer
linear = nn.Linear(8, 8)

F_x = linear(x)


# 3. Residual connection
# output = x + F(x)
output = x + F_x


# 4. Print everything
print("Input (x):")
print(x)

print("\nLayer output F(x):")
print(F_x)

print("\nResidual output x + F(x):")
print(output)


print("\nTensor shapes:")
print("Input shape:          ", x.shape)
print("Layer output shape:   ", F_x.shape)
print("Residual output shape:", output.shape)


# 5. Verify that output shape matches input shape
print("\nDoes output shape match input shape?")
print(output.shape == x.shape)