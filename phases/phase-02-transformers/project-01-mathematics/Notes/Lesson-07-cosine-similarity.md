# Cosine Similarity — Important Questions & Answers

## Q1. What is cosine similarity?

**Answer:**

Cosine similarity measures the similarity between two vectors by comparing the **angle** between them, regardless of their magnitude.

Formula:

```text
                A · B
Cosine = -------------------
          ||A|| × ||B||
```

where:

* `A · B` = Dot Product
* `||A||` = L2 Norm of A
* `||B||` = L2 Norm of B

---

## Q2. Why do we need cosine similarity?

**Answer:**

The raw dot product depends on:

* Vector magnitude
* Vector direction

Cosine similarity removes the effect of magnitude and compares only the direction of the vectors.

This makes it ideal for comparing embeddings.

---

## Q3. What is the range of cosine similarity?

**Answer:**

```text
Range = [-1, 1]
```

| Value  | Meaning                                   |
| ------ | ----------------------------------------- |
| **1**  | Same direction                            |
| **0**  | Perpendicular (no directional similarity) |
| **-1** | Opposite direction                        |

---

## Q4. Why is cosine similarity better than a raw dot product?

**Answer:**

A raw dot product increases when vector magnitudes increase, even if the direction stays the same.

Cosine similarity normalizes both vectors before comparison, so only the direction influences the result.

Example:

```text
A = [3,4]

B = [6,8]
```

Although `B` is twice as long as `A`, their cosine similarity is:

```text
1
```

because both vectors point in the same direction.

---

## Q5. What is the relationship between dot product and cosine similarity?

**Answer:**

The dot product is:

```text
A · B = ||A|| × ||B|| × cos(θ)
```

Rearranging:

```text
Cosine Similarity

      A · B
= -----------------
  ||A|| × ||B||
```

Cosine similarity is therefore the **normalized dot product**.

---

## Q6. What happens if vectors are normalized?

**Answer:**

If:

```text
||A|| = 1

||B|| = 1
```

Then:

```text
Cosine Similarity

= A · B
```

For normalized vectors:

> **Dot Product = Cosine Similarity**

This is one of the most important concepts in Machine Learning.

---

## Q7. Why are embeddings normalized?

**Answer:**

Embeddings are normalized so that similarity depends only on direction rather than vector magnitude.

Normalization removes the effect of vector length and makes semantic comparison more meaningful.

---

## Q8. Why do vector databases use cosine similarity?

**Answer:**

Vector databases compare embeddings rather than raw text.

Cosine similarity helps retrieve semantically similar documents by comparing the directions of embedding vectors.

Common vector databases include:

* FAISS
* Milvus
* Pinecone
* Weaviate
* pgvector
* Qdrant

---

## Q9. Why is cosine similarity useful in RAG?

**Answer:**

In Retrieval-Augmented Generation (RAG):

1. The user's query is converted into an embedding.
2. Documents are converted into embeddings.
3. Cosine similarity is computed between the query embedding and each document embedding.
4. The most similar documents are retrieved and provided to the LLM.

---

## Q10. Why don't Transformers use cosine similarity in self-attention?

**Answer:**

Transformers compute:

```python
scores = Q @ K.T
```

which uses **dot products**, not cosine similarity.

Reasons:

* Dot products preserve magnitude information.
* Dot products are computationally efficient.
* The scores are scaled using:

```text
√dk
```

before applying Softmax.

The attention mechanism learns useful representations without explicitly normalizing vectors.

---

## Q11. What is the difference between dot product and cosine similarity?

| Dot Product                        | Cosine Similarity                                |
| ---------------------------------- | ------------------------------------------------ |
| Depends on magnitude and direction | Depends only on direction                        |
| Can take any value                 | Always between -1 and 1                          |
| Used in Transformer attention      | Used in embedding search and semantic similarity |
| Not normalized                     | Normalized                                       |

---

## Q12. How is cosine similarity computed in PyTorch?

Manual:

```python
cosine = torch.dot(A, B) / (torch.norm(A) * torch.norm(B))
```

Built-in:

```python
import torch.nn.functional as F

cosine = F.cosine_similarity(
    A.unsqueeze(0),
    B.unsqueeze(0)
)
```

---

## Q13. When should cosine similarity be preferred?

Use cosine similarity when:

* Comparing embeddings
* Semantic search
* Vector databases
* Recommendation systems
* Clustering
* Duplicate detection
* Retrieval-Augmented Generation (RAG)

---

## Q14. What is the most important interview takeaway?

> **Cosine similarity is the normalized dot product.**

It measures the similarity between two vectors by comparing their direction while ignoring their magnitude.

---

# Summary

```text
                Dot Product
                     │
                     ▼
      Depends on Magnitude + Direction
                     │
                     ▼
          Divide by Both Vector Norms
                     │
                     ▼
             Cosine Similarity
                     │
                     ▼
          Depends Only on Direction
                     │
                     ▼
       Ideal for Comparing Embeddings
```
