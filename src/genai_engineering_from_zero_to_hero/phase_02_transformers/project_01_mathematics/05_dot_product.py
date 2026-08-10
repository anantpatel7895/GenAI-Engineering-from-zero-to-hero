import torch

# ---------------------------------------
# Create two vectors
# ---------------------------------------
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])

print("Vector v1:", v1)
print("Vector v2:", v2)

# ---------------------------------------
# Compute dot product manually
# ---------------------------------------
manual_dot = torch.sum(v1 * v2)
print("\nManual dot product:", manual_dot.item())

# ---------------------------------------
# Compute using torch.dot()
# ---------------------------------------
torch_dot = torch.dot(v1, v2)
print("torch.dot() result:", torch_dot.item())

# ---------------------------------------
# Verify both results match
# ---------------------------------------
print("Results match:", torch.allclose(manual_dot, torch_dot))

# ---------------------------------------
# Case 1: Identical vectors
# ---------------------------------------
a = torch.tensor([2.0, 3.0, 4.0])
b = torch.tensor([2.0, 3.0, 4.0])

print("\nIdentical vectors:")
print("a =", a)
print("b =", b)
print("Dot product =", torch.dot(a, b).item())

# ---------------------------------------
# Case 2: Perpendicular vectors
# ---------------------------------------
c = torch.tensor([1.0, 0.0])
d = torch.tensor([0.0, 1.0])

print("\nPerpendicular vectors:")
print("c =", c)
print("d =", d)
print("Dot product =", torch.dot(c, d).item())

# ---------------------------------------
# Case 3: Opposite vectors
# ---------------------------------------
e = torch.tensor([2.0, -1.0, 3.0])
f = -e

print("\nOpposite vectors:")
print("e =", e)
print("f =", f)
print("Dot product =", torch.dot(e, f).item())