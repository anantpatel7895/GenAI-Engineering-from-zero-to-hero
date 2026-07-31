import torch


import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.mps.is_available())  # macOS

# scalar
scalar = torch.tensor(7)
print(scalar)

# vector
vector = torch.tensor([1, 2, 3])
print(vector)

# matrix
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(matrix)

# tensor
tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor)

def describe_tensor(tensor: torch.Tensor) -> None:
    print(f"Shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires grad: {tensor.requires_grad}")
    print(f"Number of dimensions: {tensor.ndim}")
    print(f"Size: {tensor.numel()}")

describe_tensor(scalar)
describe_tensor(vector)
describe_tensor(matrix)
describe_tensor(tensor)