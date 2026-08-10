import torch

# ---------------------------------------
# Create several vectors
# ---------------------------------------
v1 = torch.tensor([3.0, 4.0])
v2 = torch.tensor([6.0, 8.0])      # Same direction as v1 (2x magnitude)
v3 = torch.tensor([1.0, -2.0, 2.0])
v4 = torch.tensor([-4.0, 1.0, 2.0])

vectors = [v1, v2, v3, v4]

# ---------------------------------------
# Compute L1 and L2 norms
# ---------------------------------------
print("L1 and L2 Norms")
print("-" * 40)

for i, v in enumerate(vectors, start=1):
    l1 = torch.norm(v, p=1)
    l2 = torch.norm(v, p=2)

    print(f"Vector v{i}: {v}")
    print(f"L1 Norm = {l1:.4f}")
    print(f"L2 Norm = {l2:.4f}")
    print()

# ---------------------------------------
# Normalize vectors (L2 normalization)
# ---------------------------------------
print("-" * 40)
print("Normalized Vectors")
print("-" * 40)

for i, v in enumerate(vectors, start=1):
    normalized = v / torch.norm(v, p=2)

    print(f"Original v{i}: {v}")
    print(f"Normalized v{i}: {normalized}")

    # Verify L2 norm equals 1
    print(f"L2 Norm after normalization = {torch.norm(normalized, p=2):.4f}")
    print()

# ---------------------------------------
# Compare vectors with same direction
# ---------------------------------------
print("-" * 40)
print("Same Direction Comparison")
print("-" * 40)

print("v1 =", v1)
print("v2 =", v2)

print("\nL2 Norm of v1:", torch.norm(v1, p=2).item())
print("L2 Norm of v2:", torch.norm(v2, p=2).item())

print("\nNormalized v1:", v1 / torch.norm(v1))
print("Normalized v2:", v2 / torch.norm(v2))

print("\nAre normalized vectors equal?",
      torch.allclose(
          v1 / torch.norm(v1),
          v2 / torch.norm(v2)
      ))