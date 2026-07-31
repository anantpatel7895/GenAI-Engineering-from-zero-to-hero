## Lesson 1

### Q1. What is a matrix?

A matrix is a rank-2 tensor consisting of rows and columns. It is a special case of a tensor with exactly two dimensions.

---

### Q2. What is tensor rank?

Tensor rank is the number of dimensions (axes) of a tensor.

---

### Q3. What is tensor shape?

Tensor shape specifies the number of elements along each dimension.

---

### Q4. Why are embeddings represented as vectors?

An embedding represents a single entity as an ordered collection of floating-point values, so it is naturally represented as a 1-dimensional tensor (vector).

---

### Q5. Why do Transformers use 3D tensors?

Transformers process batches of sequences, where each token is represented by an embedding vector. This requires a tensor of shape `(Batch, Tokens, Embedding)`.

---

### Q6. What does `(B, T, C)` represent?

- **B**: Batch size (number of sequences)
- **T**: Number of tokens in each sequence
- **C**: Embedding dimension (features per token)