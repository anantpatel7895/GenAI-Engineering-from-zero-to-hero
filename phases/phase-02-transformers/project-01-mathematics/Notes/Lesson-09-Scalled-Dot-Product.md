# Lesson 09 — Scaled Dot Product

## Why is Scaled Dot Product Important?

The core computation of every Transformer is:

```python
scores = (Q @ K.T) / math.sqrt(dk)
attention = torch.softmax(scores, dim=-1)
```

Every modern Transformer model—including GPT, BERT, Llama, Gemma, Qwen, and DeepSeek—uses this operation.

Understanding **why we divide by √dk** is essential for understanding self-attention.

---

# Learning Objectives

By the end of this lesson, you should understand:

* Why raw dot products become problematic
* What `dk` represents
* Why the scaling factor is `√dk`
* How scaling affects Softmax
* Why scaling improves training stability

---

# Recall the Attention Pipeline

The attention mechanism begins by computing similarity scores:

```text
Query (Q)

×

Key (K)

↓

Q @ Kᵀ

↓

Similarity Scores
```

Without scaling:

```python
scores = Q @ K.T
```

Then:

```python
attention = torch.softmax(scores, dim=-1)
```

---

# The Problem with Raw Dot Products

Consider:

```text
Query = [5, 7, 8, 4]

Key   = [6, 8, 9, 3]
```

Dot product:

```text
5×6 + 7×8 + 8×9 + 4×3

=

170
```

Now imagine vectors with hundreds of dimensions.

Typical Transformer dimensions:

```text
64

128

256

512

768

1024
```

As the vector dimension increases, the dot product naturally becomes much larger.

---

# Why Large Scores Are a Problem

Suppose Softmax receives:

```text
[2, 1, 3]
```

Output:

```text
[0.24, 0.09, 0.67]
```

The distribution is smooth.

Now suppose Softmax receives:

```text
[200, 100, 300]
```

Output:

```text
[0, 0, 1]
```

Softmax becomes extremely confident.

Almost all probability is assigned to a single value.

This phenomenon is called **Softmax saturation**.

---

# Why Softmax Saturation is Bad

If one token receives nearly all the probability:

```text
Token A → 99.999%

Token B → 0.001%

Token C → 0%
```

then:

* Almost all other tokens are ignored.
* Gradients become very small.
* Learning slows down.
* Training becomes unstable.

---

# The Solution

Instead of using the raw dot product:

```python
scores = Q @ K.T
```

Transformers use:

```python
scores = (Q @ K.T) / math.sqrt(dk)
```

This reduces the magnitude of the attention scores before Softmax.

---

# What is `dk`?

`dk` is the **dimension of each key vector**.

Example:

Embedding dimension:

```text
768
```

Number of attention heads:

```text
12
```

Each attention head receives:

```text
768 / 12 = 64
```

Therefore:

```text
dk = 64
```

---

# Why Divide by √dk?

The `variance` of the dot product increases as the vector dimension (`dk`) increases.

Dividing by:

```text
√dk
```

keeps the variance approximately constant.

This prevents attention scores from growing too large and keeps Softmax in a stable operating range.

---

# Example

Suppose:

```text
Dot Product = 160

dk = 64
```

Scaling:

```text
160 / √64

=

160 / 8

=

20
```

Instead of giving Softmax:

```text
160
```

we provide:

```text
20
```

The resulting probability distribution is smoother and gradients remain healthier.

---

# Why Not Divide by `dk`?

A common interview question is:

> Why do we divide by `√dk` instead of `dk`?

The answer:

The variance of the dot product grows approximately **proportional to `dk`**.

Standard deviation grows as:

```text
√Variance
```

Therefore:

```text
√dk
```

is the mathematically correct scaling factor.

Dividing by `dk` would shrink the scores too aggressively.

> The original Transformer paper shows that the variance of the dot product grows roughly in proportion to dk.
> Dividing by √dk keeps the variance approximately constant as the embedding dimension increases.
> This helps maintain stable gradients during training.

---

# Visual Intuition

Without scaling:

```text
Scores

↓

[150, 120, 90]

↓

Softmax

↓

[1.00, 0.00, 0.00]
```

With scaling:

```text
Scores

↓

[18.7, 15.0, 11.2]

↓

Softmax

↓

[0.96, 0.03, 0.01]
```

The model still prefers the first token, but the probability distribution is less extreme.

---

# Complete Attention Pipeline

```text
Input Embeddings
        │
        ▼
Linear Layers
        │
        ▼
Queries (Q), Keys (K), Values (V)
        │
        ▼
Q @ Kᵀ
        │
        ▼
Similarity Scores
        │
        ▼
Divide by √dk
        │
        ▼
Softmax
        │
        ▼
Attention Weights
        │
        ▼
Attention Output = Attention Weights @ V
```

---

# Engineering Perspective

Typical values in production models:

| Model       | Embedding Size | Heads | `dk` |
| ----------- | -------------: | ----: | ---: |
| GPT-2 Small |            768 |    12 |   64 |
| Llama 3 8B  |           4096 |    32 |  128 |
| Qwen 2.5    |           3584 |    28 |  128 |

As models become larger, scaling becomes increasingly important for stable training.

---

# Key Takeaways

* Raw dot products grow with vector dimension.
* Large attention scores cause Softmax saturation.
* Saturated Softmax produces poor gradients.
* Dividing by `√dk` keeps attention scores in a healthy range.
* Every modern Transformer uses scaled dot-product attention.

---

# Interview Questions

### Q1. Why are raw dot products a problem in self-attention?

Because their magnitude increases with vector dimension, causing Softmax to become overly peaked and making training unstable.

---

### Q2. What does `dk` represent?

The dimensionality of each **key vector** within an attention head.

---

### Q3. Why do we divide by `√dk`?

To keep the variance of the attention scores approximately constant as the vector dimension increases.

---

### Q4. What happens if we remove the scaling factor?

Softmax becomes saturated, attention becomes overly confident, gradients become small, and training stability degrades.

---

### Q5. Why don't we divide by `dk` instead?

Because the standard deviation of the dot product grows with `√dk`, not `dk`. Dividing by `dk` would over-scale the scores.

---

### Q6. Where is scaled dot-product attention used?

Inside every self-attention and cross-attention layer of Transformer architectures.

---

### Q7. What is the complete attention equation?

```python
scores = (Q @ K.T) / math.sqrt(dk)
attention = torch.softmax(scores, dim=-1)
output = attention @ V
```

This is the core computation at the heart of every Transformer model.
