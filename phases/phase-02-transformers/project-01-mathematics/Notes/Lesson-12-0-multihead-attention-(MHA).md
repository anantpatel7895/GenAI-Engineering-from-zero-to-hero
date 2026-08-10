# Lesson 12 — Multi-Head Attention (MHA)

## Introduction

Multi-Head Attention (MHA) is one of the key innovations introduced in the original Transformer paper (**Attention Is All You Need**, 2017).

Instead of computing a **single attention mechanism**, the Transformer computes **multiple attention mechanisms in parallel**.

Each attention head learns different relationships between tokens.

---

Query
"What am I looking for?"

Key
"What information do I contain?"

Value
"What information should I provide?"

---

# Why Isn't One Attention Head Enough?

Consider the sentence:

```text
The animal didn't cross the street because it was tired.
```

Different relationships exist in the same sentence.

Examples:

* "it" → "animal" (coreference)
* "cross" → "street" (semantic relationship)
* "didn't" → "cross" (grammar)

A single attention head has limited capacity to capture all of these relationships simultaneously.

Multi-Head Attention allows the model to learn multiple representations in parallel.

---

# Single Attention Head

Previously we learned:

```text
Input Embeddings
        │
        ▼
Linear Projection
        │
        ▼
Q, K, V
        │
        ▼
Scaled Dot-Product Attention
        │
        ▼
Context Representation
```

This produces **one contextual representation**.

---

# Multi-Head Attention

Instead of one attention computation:

```text
Input
  │
  ▼
Attention
  │
  ▼
Output
```

we compute several attention heads simultaneously.

```text
                  Input Embeddings
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
     Head 1           Head 2          Head N
        │                │                │
   Self-Attention   Self-Attention  Self-Attention
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                 Concatenate Outputs
                         ▼
                  Final Linear Layer
                         ▼
                    Final Output
```

Each head learns different information.

---

# Different Heads Learn Different Patterns

During training, different heads naturally specialize.

Example:

```text
Head 1
Grammar
```

```text
Head 2
Long-distance dependencies
```

```text
Head 3
Semantic similarity
```

```text
Head 4
Coreference resolution
```

No head is manually assigned a role.

The model discovers useful attention patterns automatically during training.

---

# Step 1 — Input Embeddings

Suppose:

```text
Batch Size      = 2

Sequence Length = 4

Embedding Size  = 8
```

Input tensor:

```text
X

Shape = (2, 4, 8)
```

Meaning:

```text
(Batch, Sequence Length, Embedding Dimension)
```

---

# Step 2 — Linear Projection

Three learned projection matrices are applied.

```python
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

Shapes:

```text
Q = (2,4,8)

K = (2,4,8)

V = (2,4,8)
```

---

# Step 3 — Split into Multiple Heads

Suppose:

```text
Embedding Dimension = 8

Number of Heads = 2
```

Head dimension:

```text
8 / 2 = 4
```

After reshaping:

```text
Q

(2,2,4,4)
```

```text
K

(2,2,4,4)
```

```text
V

(2,2,4,4)
```

Meaning:

```text
(Batch,
 Heads,
 Sequence Length,
 Head Dimension)
```

Each head now processes only a portion of the embedding.

---

# Step 4 — Scaled Dot-Product Attention

Every head independently computes:

```python
scores = Q @ K.transpose(-2, -1)

scores = scores / math.sqrt(d_kh)

attention = softmax(scores)

output = attention @ V
```

Each head produces its own contextual representation.

---

# Step 5 — Concatenate Heads

Suppose:

```text
Head 1 Output

(2,4,4)
```

```text
Head 2 Output

(2,4,4)
```

After concatenation:

```text
(2,4,8)
```

The original embedding dimension is restored.

---

# Step 6 — Final Linear Projection

The concatenated output passes through another learned projection matrix.

```python
output = concat_heads @ Wo
```

Purpose:

* Mix information from all heads.
* Produce the final attention output.
* Maintain the original embedding dimension.

---

# Complete Multi-Head Attention Pipeline

```text
                 Input Embeddings (X)
                         │
                 Linear Projection
                         │
                  Q, K, V Matrices
                         │
                 Split into Heads
                         │
      ┌──────────┬──────────┬──────────┐
      ▼          ▼          ▼
    Head 1     Head 2     Head N
      │          │          │
Scaled Dot-Product Attention
      │          │          │
      └──────────┴──────────┘
                 Concatenate
                      │
              Final Linear Layer
                      │
                 Final Output
```

---

# Tensor Shapes

Example:

```text
Batch Size      = 2

Sequence Length = 4

Embedding Size  = 8

Heads           = 2

Head Dimension  = 4
```

| Tensor              | Shape     |
| ------------------- | --------- |
| X                   | (2,4,8)   |
| Q                   | (2,4,8)   |
| K                   | (2,4,8)   |
| V                   | (2,4,8)   |
| Q after split       | (2,2,4,4) |
| K after split       | (2,2,4,4) |
| V after split       | (2,2,4,4) |
| Attention Scores    | (2,2,4,4) |
| Head Output         | (2,2,4,4) |
| Concatenated Output | (2,4,8)   |

---

# Why Must the Embedding Dimension Be Divisible by the Number of Heads?

Each head receives an equal portion of the embedding.

Example:

```text
Embedding = 768

Heads = 12

768 / 12 = 64
```

Each head receives a 64-dimensional embedding.

If:

```text
Embedding = 770

Heads = 12
```

then the embedding cannot be evenly divided.

Therefore:

```text
embed_dim % num_heads == 0
```

must be true in the standard Multi-Head Attention implementation.

---

# Real Model Examples

| Model       | Embedding | Heads | Head Dimension |
| ----------- | --------: | ----: | -------------: |
| GPT-2 Small |       768 |    12 |             64 |
| BERT Base   |       768 |    12 |             64 |
| Llama 3 8B  |      4096 |    32 |            128 |
| Qwen 2.5    |      3584 |    28 |            128 |

---

# Multi-Head Attention vs Single Attention

| Single Head                   | Multi-Head                          |
| ----------------------------- | ----------------------------------- |
| One attention pattern         | Multiple attention patterns         |
| One contextual representation | Multiple contextual representations |
| Lower representation capacity | Richer representations              |
| Simpler                       | More expressive                     |

---

# Engineering Perspective

Modern Transformer implementations do **not** compute each head in a Python loop.

Instead, all heads are processed simultaneously using batched tensor operations on the GPU.

Frameworks such as:

* PyTorch
* FlashAttention
* xFormers
* TensorRT-LLM
* vLLM

optimize Multi-Head Attention using highly parallel matrix operations.

---

# Beyond Multi-Head Attention

Modern LLMs introduce optimizations:

### Multi-Query Attention (MQA)

Many Query heads share one Key head and one Value head.

Benefits:

* Smaller KV Cache
* Faster inference
* Lower memory usage

---

### Grouped Query Attention (GQA)

Several Query heads share groups of Key and Value heads.

Used by:

* Llama 3
* Gemma 2
* Mistral
* Qwen

Benefits:

* Better balance between quality and efficiency.

---

# Key Takeaways

* Multi-Head Attention performs several attention computations in parallel.
* Each head learns different relationships between tokens.
* The embedding is split evenly across attention heads.
* Each head independently performs Scaled Dot-Product Attention.
* Head outputs are concatenated and projected back to the original embedding dimension.
* Multi-Head Attention greatly increases the representational power of Transformers.
* Modern LLMs often use optimized variants such as GQA and MQA for better inference performance.

---

# Interview Questions

### Q1. Why do Transformers use Multi-Head Attention?

To allow the model to learn multiple types of relationships between tokens simultaneously.

---

### Q2. Why must `embed_dim` be divisible by `num_heads`?

Because the embedding is split evenly across all attention heads.

---

### Q3. What does each attention head learn?

Each head automatically learns different attention patterns, such as grammatical, semantic, or long-range relationships.

---

### Q4. Why are head outputs concatenated?

To combine information learned by different heads into a single representation before the final projection.

---

### Q5. What is the purpose of the final projection matrix (`Wo`)?

It mixes information from all attention heads and projects the concatenated output back to the original embedding dimension.

---

### Q6. How do GQA and MQA differ from standard Multi-Head Attention?

* **MQA:** Many Query heads share a single Key and Value head.
* **GQA:** Groups of Query heads share Key and Value heads.
* Both reduce KV cache size and improve inference efficiency while preserving most of the quality of standard Multi-Head Attention.
