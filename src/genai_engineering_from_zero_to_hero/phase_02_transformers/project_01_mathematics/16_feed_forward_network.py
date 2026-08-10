import torch
import torch.nn as nn


# Dimensions
d_model = 8
d_ff = 32


# 1. Create input tensor
# Shape: (batch_size, sequence_length, d_model)
x = torch.randn(2, 4, d_model)

print("Input shape:")
print(x.shape)


# 2. First linear layer: 8 -> 32
linear1 = nn.Linear(d_model, d_ff)

x = linear1(x)

print("\nAfter Linear 8 -> 32:")
print(x.shape)


# 3. ReLU
relu = nn.ReLU()

x = relu(x)

print("\nAfter ReLU:")
print(x.shape)


# 4. Second linear layer: 32 -> 8
linear2 = nn.Linear(d_ff, d_model)

x = linear2(x)

print("\nAfter Linear 32 -> 8:")
print(x.shape)