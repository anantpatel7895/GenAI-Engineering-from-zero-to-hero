
# Matrix Multiplication
- Think of matrics multiplication as a collection of dot products between rows of the first matrix and columns of the second matrix.

> This is why GPUs are so important—they compute thousands of these dot products in parallel.


# Matrix Multiplication vs Element-wise Multiplication

| Matrix Multiplication               | Element-wise Multiplication             |
| ----------------------------------- | --------------------------------------- |
| `torch.matmul(A, B)`                | `A * B`                                 |
| Inner dimensions must match         | Shapes must match (or be broadcastable) |
| Computes dot products               | Multiplies corresponding elements       |
| Changes feature representation      | Keeps element positions aligned         |
| Used in linear layers and attention | Used for masking, scaling, gating, etc. |



| Operation                   | Symbol                  | Requirement                                                 | Example Result                                  |
| --------------------------- | ----------------------- | ----------------------------------------------------------- | ----------------------------------------------- |
| Matrix multiplication       | `@` or `torch.matmul()` | Inner dimensions must match (`m×n` × `n×p`)                 | Computes dot products of rows and columns       |
| Element-wise multiplication | `*`                     | Both tensors must have the same shape (or be broadcastable) | Multiplies corresponding elements independently |
