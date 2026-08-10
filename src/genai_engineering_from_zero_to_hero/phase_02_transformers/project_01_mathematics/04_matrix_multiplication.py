import torch

# Create matrices
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float32)      # Shape: (2, 3)

B = torch.tensor([[7, 8],
                  [9, 10],
                  [11, 12]], dtype=torch.float32)       # Shape: (3, 2)

# Print shapes
print("Shape of A:", A.shape)
print("Shape of B:", B.shape)

# Matrix multiplication using torch.matmul()
result1 = torch.matmul(A, B)
print("\nResult using torch.matmul():")
print(result1)

# Matrix multiplication using @ operator
result2 = A @ B
print("\nResult using @ operator:")
print(result2)

# Verify both results are identical
print("\nAre both results identical?",
      torch.equal(result1, result2))

# Print resulting shape
print("Shape of result:", result1.shape)

# ---------------------------------------
# Try multiplying incompatible matrices
# ---------------------------------------
C = torch.tensor([[1, 2],
                  [3, 4]])      # Shape: (2, 2)

try:
    print("\nTrying A @ C ...")
    print(A @ C)
except RuntimeError as e:
    print("Error:", e)

# ---------------------------------------
# Compare matrix multiplication with
# element-wise multiplication
# ---------------------------------------

# Element-wise multiplication requires same shape
X = torch.tensor([[1, 2],
                  [3, 4]])

Y = torch.tensor([[5, 6],
                  [7, 8]])

print("\nMatrix multiplication (X @ Y):")
print(X @ Y)

print("\nElement-wise multiplication (X * Y):")
print(X * Y)

