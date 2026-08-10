import torch

# ---------------------------------------
# Manual Softmax Function
# ---------------------------------------
def manual_softmax(x):
    # Numerical stability: subtract the maximum value
    x = x - torch.max(x)          
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x)

# ---------------------------------------
# Test Cases
# ---------------------------------------
test_cases = {
    "Normal Scores": torch.tensor([2.0, 1.0, 0.1]),

    "Equal Scores": torch.tensor([5.0, 5.0, 5.0]),

    "One Dominant Score": torch.tensor([10.0, 2.0, 1.0]),

    "Negative Scores": torch.tensor([-2.0, -1.0, -3.0]),

    "Large Scores": torch.tensor([1000.0, 1001.0, 1002.0])
}

print("=" * 65)
print("Softmax Comparison")
print("=" * 65)

for name, scores in test_cases.items():

    print(f"\n{name}")
    print("-" * 65)
    print("Scores:", scores)

    # ---------------------------------------
    # Manual Softmax
    # ---------------------------------------
    manual = manual_softmax(scores)

    # ---------------------------------------
    # PyTorch Softmax
    # ---------------------------------------
    pytorch = torch.softmax(scores, dim=0)

    print("\nManual Softmax")
    print(manual)

    print("\nPyTorch Softmax")
    print(pytorch)

    # Verify both results match
    print("\nResults Match:",
          torch.allclose(manual, pytorch, atol=1e-6))

    # Verify probabilities sum to 1
    print("Sum of probabilities:", manual.sum().item())