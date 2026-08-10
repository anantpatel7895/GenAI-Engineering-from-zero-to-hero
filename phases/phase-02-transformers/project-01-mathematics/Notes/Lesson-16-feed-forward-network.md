# Lesson 16 — Feed Forward Network (FFN / MLP)

## 1. What Is the Feed Forward Network?

The **Feed Forward Network (FFN)**, also called:

* Feed Forward Network
* Position-wise Feed Forward Network
* FFN
* MLP (Multi-Layer Perceptron)

is the second major computation inside a Transformer block.

A Transformer block mainly contains:

```text
Multi-Head Attention
        +
Feed Forward Network
```

The attention mechanism and FFN perform fundamentally different jobs.

---

# 2. Attention vs FFN

The most important distinction is:

```text
Attention

→ mixes information between tokens
```

while:

```text
FFN

→ transforms each token independently
```

### Attention

```text
Token 1 ─────┐
Token 2 ─────┼──► Attention
Token 3 ─────┘
```

Tokens can interact with each other.

### FFN

```text
Token 1 ──► FFN ──► Token 1'

Token 2 ──► FFN ──► Token 2'

Token 3 ──► FFN ──► Token 3'
```

> Each token is processed `independently`.
> each token is like a one sample and number of dimension of embedding is like feature

---

# 3. Original Transformer FFN

The original Transformer uses:

```text
FFN(x) = W₂ σ(W₁x + b₁) + b₂
```

where:

* `W₁` = first linear transformation
* `W₂` = second linear transformation
* `b₁` = first bias
* `b₂` = second bias
* `σ` = nonlinear activation function

The original Transformer uses **ReLU**.

Therefore:

```text
FFN(x)

=

W₂ ReLU(W₁x + b₁) + b₂
```

---

# 4. Basic Architecture

```text
Input
  │
  ▼
Linear Layer 1
  │
  ▼
Activation
  │
  ▼
Linear Layer 2
  │
  ▼
Output
```

The first linear layer expands the representation.

The second linear layer projects it back to the original model dimension.

---

# 5. Dimension Expansion

Suppose:

```text
d_model = 8
```

The FFN may expand it to:

```text
d_ff = 32
```

Therefore:

```text
8
│
▼
32
│
▼
8
```

The complete flow becomes:

```text
Input
(8)
 │
 ▼
Linear
8 → 32
 │
 ▼
ReLU
 │
 ▼
Linear
32 → 8
 │
 ▼
Output
(8)
```

The larger intermediate dimension gives the network greater representational capacity.

---

# 6. Why Expand the Dimension?

Suppose the Transformer representation is:

```text
d_model = 8
```

The FFN temporarily expands it:

```text
8 → 32
```

This gives the network a larger feature space in which to perform nonlinear transformations.

After the transformation:

```text
32 → 8
```

The model returns to the original dimension so that the result can participate in the residual connection.

---

# 7. Why Do We Need an Activation Function?

Consider:

```text
Linear 1
   │
Linear 2
```

Mathematically:

```text
W₂(W₁x)
```

This can be rewritten as:

```text
(W₂W₁)x
```

Therefore, two linear layers without an activation function are effectively just another linear transformation.

To learn nonlinear relationships, we need:

```text
Linear
   │
   ▼
Activation
   │
   ▼
Linear
```

The activation function introduces the required nonlinearity.

---

# 8. ReLU

The original Transformer uses **ReLU**.

Formula:

```text
ReLU(x) = max(0, x)
```

Example:

```text
Input:

[-2, -1, 0, 2, 5]

↓

ReLU

[0, 0, 0, 2, 5]
```

Negative values become zero.

---

# 9. Modern Transformer Activations

The original Transformer used:

```text
ReLU
```

Modern LLMs commonly use:

```text
GELU
SiLU / Swish
SwiGLU
```

Modern decoder-only models such as Llama-style architectures commonly use **gated FFNs such as SwiGLU** rather than the original ReLU FFN.

Therefore, distinguish between:

### Original Transformer

```text
Linear
   ↓
ReLU
   ↓
Linear
```

### Modern LLM

```text
Linear
   ↓
Gated Activation
   ↓
Linear
```

A common modern choice is:

```text
SwiGLU
```

---

# 10. Why Is It Called Position-Wise?

The original Transformer paper calls the FFN a:

> **Position-wise Feed-Forward Network**

because the same FFN is applied independently to every token position.

Suppose:

```text
X.shape = (2, 4, 8)
```

where:

```text
Batch = 2
Sequence Length = 4
Embedding Dimension = 8
```

The FFN processes:

```text
Batch 1

Token 1 ──► FFN
Token 2 ──► FFN
Token 3 ──► FFN
Token 4 ──► FFN
```

There is no interaction between Token 1 and Token 2 inside the FFN.

---

# 11. Tensor Shape Transformation

Suppose:

```text
Input:

(B, T, d_model)

=

(2, 4, 8)
```

First linear layer:

```text
(2, 4, 8)

↓

Linear(8 → 32)

↓

(2, 4, 32)
```

Activation:

```text
(2, 4, 32)

↓

ReLU

↓

(2, 4, 32)
```

Second linear layer:

```text
(2, 4, 32)

↓

Linear(32 → 8)

↓

(2, 4, 8)
```

Therefore:

```text
(2, 4, 8)
      │
      ▼
(2, 4, 32)
      │
      ▼
(2, 4, 32)
      │
      ▼
(2, 4, 8)
```

The FFN ultimately preserves `d_model`.

---

# 12. Why Must the FFN Return to d_model?

The Transformer uses a residual connection:

```text
Output = X + FFN(X)
```

Suppose:

```text
X.shape = (2, 4, 8)
```

Therefore:

```text
FFN(X).shape
=
(2, 4, 8)
```

Then:

```text
(2,4,8)

+

(2,4,8)

=

(2,4,8)
```

If the FFN returned:

```text
(2,4,32)
```

then:

```text
(2,4,8)

+

(2,4,32)
```

would be invalid.

---

# 13. Attention vs FFN

| Property                       | Attention                          | FFN                          |
| ------------------------------ | ---------------------------------- | ---------------------------- |
| Main purpose                   | Token interaction                  | Feature transformation       |
| Token interaction              | Yes                                | No                           |
| Uses Q/K/V                     | Yes                                | No                           |
| Processes tokens independently | No                                 | Yes                          |
| Nonlinear transformation       | Limited                            | Yes                          |
| Main computation               | Attention scores + weighted values | Linear → Activation → Linear |
| Input shape                    | `(B,T,d_model)`                    | `(B,T,d_model)`              |
| Output shape                   | `(B,T,d_model)`                    | `(B,T,d_model)`              |

---

# 14. Important Mental Model

A very useful way to remember the Transformer architecture is:

```text
Attention = Communication

FFN = Computation
```

Attention asks:

```text
"Which other tokens are relevant to me?"
```

FFN asks:

```text
"Given the information I now have,
how should I transform my representation?"
```

---

# 15. Transformer Block So Far

The architecture is becoming complete:

```text
                         ┌─────────────────────┐
                         │                     │
                         │   Residual Path     │
                         │                     │
                         ▼                     │
Input X ─────────────────┬─────────────────────┤
                         │                     │
                         ▼                     │
                Multi-Head Attention           │
                         │                     │
                         └──────────► ⊕ ◄──────┘
                                      │
                                      ▼
                                  LayerNorm
                                      │
                                      │
                         ┌────────────┴─────────┐
                         │                      │
                         │   Residual Path      │
                         │                      │
                         ▼                      │
                       FFN                      │
                         │                      │
                         └──────────► ⊕ ◄───────┘
                                      │
                                      ▼
                                  LayerNorm
                                      │
                                      ▼
                                    Output
```

`⊕` represents element-wise addition.

---

# 16. Hands-On Implementation

Create:

```text
16_feed_forward_network.py
```

Use:

```python
d_model = 8
d_ff = 32
```

Implement:

```python
Linear(8 → 32)
       ↓
     ReLU
       ↓
Linear(32 → 8)
```

Input:

```text
(2, 4, 8)
```

Expected shapes:

```text
Input:
(2, 4, 8)

After Linear 1:
(2, 4, 32)

After ReLU:
(2, 4, 32)

After Linear 2:
(2, 4, 8)
```

Finally verify:

```text
Output shape == Input shape
```

---

# 17. Key Takeaways

* FFN is the second major computation in a Transformer block.
* Attention mixes information between tokens.
* FFN processes each token independently.
* Original Transformer uses:

```text
Linear → ReLU → Linear
```

* The FFN expands:

```text
d_model → d_ff
```

* Then projects back:

```text
d_ff → d_model
```

* The activation function provides nonlinearity.
* Modern LLMs commonly use gated FFNs such as **SwiGLU**.
* FFN output must have the same shape as its input because of the residual connection.

---

# Interview Questions

### Q1. What is the purpose of the FFN?

To perform a nonlinear transformation of each token's representation independently.

### Q2. Does FFN mix information between tokens?

**No.**

Attention is responsible for token-to-token information mixing.

### Q3. Why does the FFN expand the dimension?

The larger intermediate dimension provides greater representational capacity for nonlinear feature transformations.

### Q4. Why is an activation function required?

Without an activation function, multiple linear layers collapse into a single linear transformation.

### Q5. What activation did the original Transformer use?

**ReLU.**

### Q6. What activations are common in modern LLMs?

**GELU, SiLU/Swish, and gated variants such as SwiGLU.**

### Q7. Why does the FFN return to `d_model`?

Because its output must be added to the residual stream:

```text
Output = X + FFN(X)
```

Therefore:

```text
Shape(X) = Shape(FFN(X))
```

---

# Engineering Insight

A Transformer block can be understood as two complementary operations:

```text
Attention
    ↓
Token Mixing
```

and:

```text
FFN
    ↓
Feature Transformation
```

This separation is fundamental to the Transformer architecture.

The attention mechanism determines **which information should be gathered**, while the FFN performs substantial nonlinear computation on the resulting token representation.

This distinction becomes particularly important when analyzing LLM performance, parameter counts, memory usage, and inference bottlenecks.
