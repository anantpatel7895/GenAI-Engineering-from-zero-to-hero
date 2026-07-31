import torch

matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

matrix_transpose = matrix.T

def describe_tensor(tensor: torch.Tensor) -> None:
    print(f"Shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires grad: {tensor.requires_grad}")
    print(f"Number of dimensions: {tensor.ndim}")
    print(f"Size: {tensor.numel()}")
    print(f"contiguous: {tensor.is_contiguous()}")


describe_tensor(matrix)
describe_tensor(matrix_transpose)