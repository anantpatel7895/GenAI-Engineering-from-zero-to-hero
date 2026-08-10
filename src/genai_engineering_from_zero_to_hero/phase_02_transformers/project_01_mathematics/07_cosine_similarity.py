import torch
import torch.nn.functional as F

# ---------------------------------------
# Function to compute cosine similarity manually
# ---------------------------------------
def manual_cosine_similarity(a, b):
    dot_product = torch.dot(a, b)
    norm_a = torch.norm(a, p=2)
    norm_b = torch.norm(b, p=2)
    return dot_product / (norm_a * norm_b)

# ---------------------------------------
# Test Cases
# ---------------------------------------

test_cases = {
    "Identical Vectors": (
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 2.0, 3.0])
    ),

    "Perpendicular Vectors": (
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0])
    ),

    "Opposite Vectors": (
        torch.tensor([2.0, -1.0, 3.0]),
        torch.tensor([-2.0, 1.0, -3.0])
    ),

    "Same Direction (Different Magnitudes)": (
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([2.0, 4.0, 6.0])
    )
}

# ---------------------------------------
# Compute Cosine Similarity
# ---------------------------------------

print("=" * 60)
print("Cosine Similarity Comparison")
print("=" * 60)

for name, (v1, v2) in test_cases.items():

    manual = manual_cosine_similarity(v1, v2)

    builtin = F.cosine_similarity(
        v1.unsqueeze(0),
        v2.unsqueeze(0),
        dim=1
    ).item()

    print(f"\n{name}")
    print("-" * 60)
    print("Vector 1:", v1)
    print("Vector 2:", v2)

    print(f"Manual Cosine Similarity : {manual.item():.4f}")
    print(f"PyTorch Cosine Similarity: {builtin:.4f}")

    print("Results Match:",
          torch.allclose(manual, torch.tensor(builtin)))