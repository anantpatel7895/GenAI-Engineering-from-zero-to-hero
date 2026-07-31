# Lesson 3 — Why `view()` Does Not Work on a Non-Contiguous Tensor

## The Core Idea

A common misconception is that `view()` fails because tensors share memory.

That is **not** the reason.

The real reason is:

> `view()` can only change a tensor's **metadata** (shape and stride). It **cannot rearrange** the underlying memory.

---

# Physical Memory vs Logical View

Every tensor has two important concepts:

## 1. Physical Memory

The actual values stored in RAM.

Example:

```text
0 1 2 3 4 5 6 7 8 9 10 11
```

This is called the **physical layout**.

---

## 2. Logical View

How PyTorch interprets the same memory using the tensor's metadata.

The metadata consists of:

* Shape
* Stride
* Dtype
* Device
* Requires_grad

Changing the metadata changes how the tensor is viewed without moving any data.

---

# Example 1 — Contiguous Tensor

```python
import torch

x = torch.arange(12)

y = x.view(3, 4)
```

Memory:

```text
0 1 2 3 4 5 6 7 8 9 10 11
```

Metadata:

```text
Shape  = (3,4)
Stride = (4,1)
```

`view()` succeeds because the tensor is stored sequentially in memory.

Only the metadata changes.

No data is copied.

---

# Example 2 — Transposed Tensor

```python
z = y.T
```

Memory remains exactly the same:

```text
0 1 2 3 4 5 6 7 8 9 10 11
```

Only the metadata changes:

```text
Shape  = (4,3)
Stride = (1,4)
```

The tensor is now **non-contiguous**.

---

# Logical Order After Transpose

Although the memory is:

```text
0 1 2 3 4 5 6 7 8 9 10 11
```

The transposed tensor is interpreted as:

```text
0 4 8
1 5 9
2 6 10
3 7 11
```

This is called the **logical order** of the tensor.

---

# Why Does `view()` Fail?

Suppose we execute:

```python
z.view(12)
```

`view()` only changes the metadata.

It does **not** change the physical memory.

If it simply changed the shape to `(12,)`, the result would read:

```text
0 1 2 3 4 5 6 7 8 9 10 11
```

However, the logical flattened order of the transposed tensor should be:

```text
0 4 8 1 5 9 2 6 10 3 7 11
```

These two orders are different.

Therefore, changing only the metadata would produce incorrect indexing.

Instead of silently returning the wrong result, PyTorch raises an error.

---

# Why Does `reshape()` Work?

`reshape()` first checks whether the tensor is contiguous.

## If the tensor is contiguous

It behaves exactly like `view()`.

No memory is copied.

## If the tensor is non-contiguous

It:

1. Allocates new memory.
2. Copies the elements into the correct logical order.
3. Returns a new contiguous tensor.

New memory becomes:

```text
0 4 8 1 5 9 2 6 10 3 7 11
```

Since the data is now stored sequentially, reshaping succeeds.

```text
reshape()

│
├── Is tensor contiguous?
│
├── Yes
│     │
│     └── Reuse the existing memory
|.         . work just like `view()`
│          • No new memory allocation
│          • Only metadata (shape/stride) changes
│
└── No
      │
      ├── Allocate a new block of memory
      ├── Copy the tensor values into the new memory
      ├── Create a new contiguous tensor
      └── Return the new tensor
```

---

# What Does `.contiguous()` Do?

Calling:

```python
z = z.contiguous()
```

creates a brand new contiguous tensor.

Original memory:

```text
0 1 2 3 4 5 6 7 8 9 10 11
```

New memory:

```text
0 4 8 1 5 9 2 6 10 3 7 11
```

Now:

```python
z.view(12)
```

works because the physical memory layout matches the logical tensor layout.

---

# Physical Order vs Logical Order

A tensor always has two different orders.

## Physical Order

How values are stored in memory.

Example:

```text
0 1 2 3 4 5 6 7 8 9 10 11
```

---

## Logical Order

How PyTorch interprets the values according to the tensor's metadata.

Example after transpose:

```text
0 4 8 1 5 9 2 6 10 3 7 11
```

When these two orders no longer match, the tensor becomes **non-contiguous**.

---

# Mental Model

```text
Tensor
│
├── Pointer to Memory
│
└── Metadata
    ├── Shape
    ├── Stride
    ├── Dtype
    ├── Device
    └── Requires_grad
```

* Multiple tensors can share the same memory.
* Each tensor has its own metadata.
* `view()` only changes metadata.
* `reshape()` changes metadata and creates a new memory block when required.

---

# Summary Table

| Operation       | Copies Data?   | Requires Contiguous Tensor?  |
| --------------- | -------------- | ---------------------------- |
| `view()`        | No             | Yes                          |
| `reshape()`     | Only if needed | No                           |
| `transpose()`   | No             | No                           |
| `.contiguous()` | Yes            | Produces a contiguous tensor |

---

# Key Takeaways

* `view()` only changes tensor metadata.
* `view()` never rearranges memory.
* A non-contiguous tensor has a logical element order that differs from its physical memory order.
* `view()` fails because changing only the shape would produce incorrect indexing.
* `reshape()` creates a new contiguous tensor when necessary.
* `.contiguous()` explicitly creates a contiguous copy of the tensor.
* Understanding contiguous memory is essential for implementing and optimizing Transformer models.

---

# Interview Question

**Q:** Why does `view()` fail on a non-contiguous tensor while `reshape()` succeeds?

**Answer:**

`view()` only modifies the tensor's metadata and assumes that the underlying memory is contiguous. A non-contiguous tensor has a logical element order defined by its strides that does not match the physical memory layout. Simply changing the shape would produce incorrect indexing, so `view()` raises an error. `reshape()` detects this situation, allocates new contiguous memory if necessary, copies the data into the correct logical order, and then returns the reshaped tensor.
