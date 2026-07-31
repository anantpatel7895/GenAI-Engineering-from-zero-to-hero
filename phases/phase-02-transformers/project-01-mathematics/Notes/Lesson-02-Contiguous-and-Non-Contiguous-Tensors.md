# Lesson 2 — Understanding Contiguous and Non-Contiguous Tensors in PyTorch

## Why is this important?

Understanding contiguous memory is essential for writing efficient PyTorch code and debugging Transformer implementations. Many tensor operations such as `transpose()`, `permute()`, `view()`, and `reshape()` depend on how tensor data is stored in memory.

---

# Step 1 — A Matrix in PyTorch

```python
import torch

matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

Visually, the matrix looks like this:

```text
      Column

      0   1   2

0     1   2   3
1     4   5   6
```

This is how we visualize the data, but **this is not how it is stored in memory**.

---

# Step 2 — Memory is One-Dimensional

Computer memory is simply one long sequence of bytes.

PyTorch stores the matrix as:

```text
Memory →

+----+----+----+----+----+----+
| 1  | 2  | 3  | 4  | 5  | 6  |
+----+----+----+----+----+----+
```

There are:

* No rows
* No columns

Only one continuous block of memory.

This is called **contiguous memory**.

---

# Step 3 — Shape is Metadata

The actual memory is:

```text
1 2 3 4 5 6
```

The tensor metadata contains:

```text
Shape = (2, 3)
```

PyTorch interprets the memory as:

```text
1 2 3
4 5 6
```

The memory itself has not changed.

The **shape only tells PyTorch how to interpret the memory**.

---

# Step 4 — Transpose

```python
matrix_t = matrix.T
```

The result is:

```text
1 4
2 5
3 6
```

Many beginners assume PyTorch copies all the data into a new block of memory.

It does **not**.

The underlying memory remains:

```text
+----+----+----+----+----+----+
| 1  | 2  | 3  | 4  | 5  | 6  |
+----+----+----+----+----+----+
```

No values are moved.

---

# How Does Transpose Work?

Instead of moving data, PyTorch changes how it **walks through memory**.

The tensor metadata is updated.

Original tensor:

```text
Shape  = (2, 3)
Stride = (3, 1)
```

Transposed tensor:

```text
Shape  = (3, 2)
Stride = (1, 3)
```

The memory stays the same, but the interpretation changes.

---

# Understanding Stride

For the original tensor:

```text
Stride = (3, 1)
```

Meaning:

* Moving to the next row skips **3 elements**.
* Moving to the next column skips **1 element**.

After transpose:

```text
Stride = (1, 3)
```

PyTorch now reads the same memory in a different pattern.

---

# Why is the Transposed Tensor Non-Contiguous?

Original access pattern:

```text
1 → 2 → 3 → 4 → 5 → 6
```

Sequential memory access.

Transposed access pattern:

```text
1 → 4 → 2 → 5 → 3 → 6
```

The values are read by jumping around the same memory block.

Since the access is no longer sequential:

```python
matrix_t.is_contiguous()
```

returns:

```text
False
```

---

# Bookshelf Analogy

Imagine six books arranged on a shelf.

```text
Book1 Book2 Book3 Book4 Book5 Book6
```

Original reading order:

```text
1 → 2 → 3 → 4 → 5 → 6
```

Transposed reading order:

```text
1 → 4 → 2 → 5 → 3 → 6
```

The books never moved.

Only the order in which you read them changed.

This is exactly what happens with a non-contiguous tensor.

---

# Why Doesn't PyTorch Copy the Data?

Copying large tensors would be extremely expensive.

For example, a Transformer tensor might have shape:

```text
(32, 4096, 8192)
```

This contains over **1 billion floating-point values**.

Instead of copying all those values, PyTorch changes only the tensor metadata (shape and stride), making transpose a very efficient operation.

---

# When Do We Need Contiguous Memory?

Some operations require the tensor data to be physically arranged in sequential memory.

A common example is:

```python
x.view(...)
```

If `x` is non-contiguous, PyTorch raises an error because `view()` assumes contiguous storage.

The solution is:

```python
x = x.contiguous()
```

This creates a new tensor with the same values stored sequentially in memory.

---

# Mental Model

```text
Tensor
│
├── Data (Memory)
│
└── Metadata
     ├── Shape
     ├── Stride
     └── Dtype
```

Most tensor operations such as `transpose()` modify only the **metadata**.

The underlying memory remains unchanged until an operation explicitly requires a contiguous copy.

---

# Experiment

```python
print("Original stride :", matrix.stride())
print("Transpose stride:", matrix_t.stride())

print(matrix.is_contiguous())
print(matrix_t.is_contiguous())
```

Observe:

* Shape changes.
* Stride changes.
* Memory remains the same.
* Only the metadata changes.
* The transposed tensor becomes non-contiguous.

---

# Key Takeaways

* Computer memory is always one-dimensional.
* A tensor's **shape** describes how to interpret the memory.
* A tensor's **stride** describes how to move through that memory.
* `transpose()` does not move data; it updates metadata.
* A **contiguous tensor** is stored sequentially in memory.
* A **non-contiguous tensor** accesses the same memory using a different traversal pattern.
* Some operations (such as `view()`) require contiguous memory and may need `.contiguous()` to create a new sequential copy.
