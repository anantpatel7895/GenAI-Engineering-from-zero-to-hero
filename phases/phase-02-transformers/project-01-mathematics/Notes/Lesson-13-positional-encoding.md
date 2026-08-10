# Lesson 13 — Positional Encoding

## Introduction

The **Attention mechanism** treats the input as a collection of vectors.

It does **not** understand the order of words.

Without positional information, the Transformer cannot distinguish between:

```text id="t4xfrc"
Dog bites man
```

and

```text id="2iqhmj"
Man bites dog
```

Although both sentences contain the same words, their meanings are completely different.

To solve this problem, the Transformer introduces **Positional Encoding**.

---

# Why Do We Need Positional Encoding?

Suppose the input sentence is:

```text id="0mjlwm"
I love AI
```

Token embeddings:

```text id="b4z8qm"
I     → E₁

love  → E₂

AI    → E₃
```

Without positional information, the attention mechanism only sees:

```text id="a9z7wj"
[E₁, E₂, E₃]
```

It cannot determine:

* Which word came first.
* Which word came last.
* The relative distance between words.

To the attention mechanism, embeddings are simply vectors.

---

# The Solution

Every token receives an additional **position vector**.

Instead of feeding only the token embedding:

$$
\text{Input} = \text{Token Embedding}
$$

the Transformer uses:

$$
\text{Input} = \text{Token Embedding} + \text{Positional Encoding}
$$

Each token now contains:

* **Semantic information** (what the token means)
* **Positional information** (where the token appears)

---

# Example

Suppose:

Token embedding:

```text id="jlwmq2"
Embedding("I")

=

[0.5, 0.3, 0.7]
```

Position encoding for position **0**:

```text id="gtjlwm"
[0.0, 1.0, 0.0]
```

Final input:

```text id="jlwmx9"
[0.5, 1.3, 0.7]
```

The position information is added directly to the embedding.

---

# Original Transformer: Sinusoidal Positional Encoding

The original Transformer paper (**Attention Is All You Need**) did **not** learn positional vectors.

Instead, it generated them using sine and cosine functions.

For position $\text{pos}$ and dimension pair index $i$:

$$
PE_{(\text{pos},\,2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(\text{pos},\,2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{model}}}\right)
$$

Even dimensions use **sine**.

Odd dimensions use **cosine**.

Where:

* $\text{pos}$ = token position (0, 1, 2, ...)
* $i$ = dimension pair index
* $d_{model}$ = embedding size

---

# Step-by-Step Worked Example

## Step 1 — Input Sentence

```text
I love AI
```

| Position | Word |
| :------: | ---- |
|     0    | I    |
|     1    | love |
|     2    | AI   |

---

## Step 2 — Convert Words to Embeddings

Each word becomes a vector.

Assume embedding dimension $d_{model} = 4$ (real models use 512, 768, etc.).

| Word | Embedding              |
| ---- | ---------------------- |
| I    | `[0.2, 0.4, 0.1, 0.8]` |
| love | `[0.5, 0.6, 0.7, 0.1]` |
| AI   | `[0.9, 0.2, 0.3, 0.5]` |

These vectors carry the **meaning** of each word — but not its position.

---

## Step 3 — Determine the Divisors

With $d_{model} = 4$, the four dimensions pair up as:

$$
\begin{aligned}
\text{Dimensions } 0,\,1 \;&\rightarrow\; i = 0 \;&&\rightarrow\; 10000^{0/4} = 1 \\[4pt]
\text{Dimensions } 2,\,3 \;&\rightarrow\; i = 1 \;&&\rightarrow\; 10000^{2/4} = 100
\end{aligned}
$$

So:

$$
\begin{aligned}
\text{Dim } 0 \;&\rightarrow\; \sin\!\left(\frac{\text{pos}}{1}\right) \\[4pt]
\text{Dim } 1 \;&\rightarrow\; \cos\!\left(\frac{\text{pos}}{1}\right) \\[4pt]
\text{Dim } 2 \;&\rightarrow\; \sin\!\left(\frac{\text{pos}}{100}\right) \\[4pt]
\text{Dim } 3 \;&\rightarrow\; \cos\!\left(\frac{\text{pos}}{100}\right)
\end{aligned}
$$

---

## Step 4 — Compute Position 0

$$
\begin{aligned}
\text{Dim } 0 \;&\rightarrow\; \sin(0) &&= 0 \\[4pt]
\text{Dim } 1 \;&\rightarrow\; \cos(0) &&= 1 \\[4pt]
\text{Dim } 2 \;&\rightarrow\; \sin(0/100) &&= 0 \\[4pt]
\text{Dim } 3 \;&\rightarrow\; \cos(0/100) &&= 1
\end{aligned}
$$

Position 0 encoding:

```text
[0, 1, 0, 1]
```

---

## Step 5 — Compute Position 1

$$
\begin{aligned}
\text{Dim } 0 \;&\rightarrow\; \sin(1) &&= 0.8415 \\[4pt]
\text{Dim } 1 \;&\rightarrow\; \cos(1) &&= 0.5403 \\[4pt]
\text{Dim } 2 \;&\rightarrow\; \sin(0.01) &&\approx 0.01 \\[4pt]
\text{Dim } 3 \;&\rightarrow\; \cos(0.01) &&\approx 0.99995
\end{aligned}
$$

Position 1 encoding:

```text
[0.8415, 0.5403, 0.01, 0.99995]
```

---

## Step 6 — Compute Position 2

$$
\begin{aligned}
\text{Dim } 0 \;&\rightarrow\; \sin(2) &&= 0.9093 \\[4pt]
\text{Dim } 1 \;&\rightarrow\; \cos(2) &&= -0.4161 \\[4pt]
\text{Dim } 2 \;&\rightarrow\; \sin(0.02) &&\approx 0.02 \\[4pt]
\text{Dim } 3 \;&\rightarrow\; \cos(0.02) &&\approx 0.9998
\end{aligned}
$$

Position 2 encoding:

```text
[0.9093, -0.4161, 0.02, 0.9998]
```

---

## Step 7 — Add Embedding + Positional Encoding

The two vectors are added **element-wise**.

**Word "I" (position 0)**

```text
Embedding : [0.2, 0.4, 0.1, 0.8]

Position  : [0.0, 1.0, 0.0, 1.0]
            ─────────────────────
Result    : [0.2, 1.4, 0.1, 1.8]
```

**Word "love" (position 1)**

```text
Embedding : [0.5,    0.6,    0.7,  0.1]

Position  : [0.8415, 0.5403, 0.01, 0.99995]
            ──────────────────────────────────
Result    : [1.3415, 1.1403, 0.71, 1.09995]
```

**Word "AI" (position 2)**

```text
Embedding : [0.9,    0.2,     0.3,  0.5]

Position  : [0.9093, -0.4161, 0.02, 0.9998]
            ──────────────────────────────────
Result    : [1.8093, -0.2161, 0.32, 1.4998]
```

---

## Step 8 — Final Input to the Transformer

Instead of only embeddings, the Transformer receives:

| Word | Final Vector                        |
| ---- | ----------------------------------- |
| I    | `[0.2, 1.4, 0.1, 1.8]`              |
| love | `[1.3415, 1.1403, 0.71, 1.09995]`   |
| AI   | `[1.8093, -0.2161, 0.32, 1.4998]`   |

Now each vector contains both:

* **Word meaning** (from the embedding)
* **Word position** (from the positional encoding)

This allows self-attention to understand both **what** each word is and **where** it appears in the sequence.

If the sentence were reordered to `AI love I`, the embeddings would be the same set of vectors — but the positional encodings added to them would differ, producing different final inputs.

---

# Implementation Note — Why the Code Looks Different

The formula says **divide by a power**:

$$
\frac{\text{pos}}{10000^{2i/d}}
$$

But the PyTorch implementation **multiplies by an exponential**:

```python
div_term = torch.exp(
    torch.arange(0, embed_dim, 2).float()
    * (-math.log(10000.0) / embed_dim)
)
```

This looks different, but it is **mathematically equivalent**.

---

## The Identity

Any power can be rewritten as an exponential:

$$
a^{x} = e^{x \ln a}
$$

Applying it to the divisor:

$$
10000^{2i/d} = e^{\,\ln(10000)\,\cdot\,\frac{2i}{d}}
$$

Since the formula **divides** by this value, take the reciprocal:

$$
\frac{1}{10000^{2i/d}} = e^{-\ln(10000)\,\cdot\,\frac{2i}{d}}
$$

So dividing by the power becomes **multiplying by the negative exponential**.

---

## Mapping the Code to the Math

| Code                              | Math                                                                       |
| --------------------------------- | -------------------------------------------------------------------------- |
| `torch.arange(0, embed_dim, 2)`   | the values $2i = 0,\, 2,\, 4,\, 6,\, \dots$                                |
| `-math.log(10000.0) / embed_dim`  | $-\ln(10000) / d$                                                          |
| their product                     | $-\ln(10000) \cdot \frac{2i}{d}$                                           |
| `torch.exp( ... )`                | $e^{-\ln(10000)\cdot\frac{2i}{d}} = \dfrac{1}{10000^{2i/d}}$               |

Which is why the sine and cosine are applied as a **multiplication**:

```python
torch.sin(position * div_term)   # ==  sin(pos / 10000^(2i/d))

torch.cos(position * div_term)   # ==  cos(pos / 10000^(2i/d))
```

---

## Why Write It This Way?

* **Numerical stability** — computing $10000^{2i/d}$ directly produces very large numbers for high dimensions; working in log space avoids overflow and precision loss.
* **Vectorization** — one `exp` over a tensor computes every divisor at once, instead of looping over dimensions.
* **Convention** — this is the form used in the reference Transformer implementations, so you will see it in almost every codebase.

---

# Why Use Sine and Cosine?

The sinusoidal formulation provides several useful properties.

### 1. Every Position Is Unique

Each position generates a different encoding.

Example:

```text id="jlwmk3"
Position 0

↓

[0.00, 1.00, 0.00, 1.00]
```

```text id="jlwmk4"
Position 1

↓

[0.84, 0.54, 0.01, 0.99]
```

```text id="jlwmk5"
Position 2

↓

[0.91, -0.42, 0.02, 0.99]
```

No two positions share the same encoding.

---

### 2. Nearby Positions Have Similar Encodings

Positions close to each other produce similar vectors.

This helps the model understand local relationships.

---

### 3. Relative Distance Can Be Inferred

The smooth sinusoidal patterns allow the model to learn relationships such as:

```text id="jlwmk6"
Token B

is 3 positions after

Token A
```

without explicitly storing relative distances.

---

### 4. Generalization to Longer Sequences

Because the encoding is computed using mathematical functions, it can generate positional vectors for sequence lengths that were not seen during training.

This is one advantage over learned positional embeddings.

---

# Learned Positional Embeddings

Later models, such as **BERT** and **GPT-2**, replaced sinusoidal encodings with **learned positional embeddings**.

Instead of computing positions mathematically:

```text id="jlwmk7"
sin()

cos()
```

the model learns one embedding vector for every position during training.

Advantages:

* Simpler implementation.
* Position vectors adapt to the training data.
* Often better performance on specific tasks.

Limitation:

* Maximum sequence length is fixed during training.
* Difficult to extrapolate to longer contexts.

---

# Rotary Positional Embeddings (RoPE)

Modern LLMs such as:

* Llama
* Qwen
* Gemma
* DeepSeek
* Mistral

primarily use **Rotary Positional Embeddings (RoPE)**.

Instead of adding position vectors to the embeddings:

```text id="jlwmk8"
Embedding

+

Position
```

RoPE rotates the Query and Key vectors before computing attention.

Conceptually:

```text id="jlwmk9"
Q

↓

Rotate

↓

Q'
```

```text id="jlwmka"
K

↓

Rotate

↓

K'
```

Attention is then computed using the rotated vectors.

Advantages:

* Encodes relative positional information naturally.
* Better extrapolation to longer contexts.
* Excellent performance for long-context LLMs.
* Widely adopted in modern decoder-only architectures.

---

# Comparison of Positional Encoding Methods

| Method                             | Used In                               | Learnable | Generalizes to Longer Contexts |
| ---------------------------------- | ------------------------------------- | :-------: | :----------------------------: |
| Sinusoidal Positional Encoding     | Original Transformer                  |     ❌     |                ✅               |
| Learned Positional Embedding       | BERT, GPT-2                           |     ✅     |                ❌               |
| Rotary Positional Embedding (RoPE) | Llama, Gemma, Qwen, DeepSeek, Mistral |     ❌     |                ✅               |

---

# Complete Input Pipeline

Before entering the attention layer:

```text id="jlwmkb"
Input Tokens
      │
      ▼
Token Embedding
      │
      ▼
Add Positional Information
      │
      ▼
Final Input Embedding
      │
      ▼
Multi-Head Attention
```

The attention mechanism always receives embeddings that already contain positional information.

---

# Engineering Perspective

Modern production LLMs rarely use additive positional embeddings.

Instead:

* **Llama**
* **Gemma**
* **Qwen**
* **DeepSeek**
* **Mistral**

use **Rotary Positional Embeddings (RoPE)** because they:

* Improve long-context performance.
* Better preserve relative positions.
* Work naturally with self-attention.
* Scale well to modern context windows (32K, 128K, or more).

Understanding sinusoidal encoding remains important because it introduced the concept of positional information in the original Transformer architecture.

---

# Key Takeaways

* Self-attention has **no inherent understanding of word order**.
* Positional Encoding injects token position into the input representation.
* The original Transformer used deterministic **sinusoidal encoding**.
* BERT and GPT-2 adopted **learned positional embeddings**.
* Modern decoder-only LLMs primarily use **RoPE**.
* Without positional information, sentences containing the same words in different orders would appear identical to the attention mechanism.

---

# Interview Questions

### Q1. Why do Transformers require positional encoding?

Because self-attention processes tokens as vectors and has no built-in understanding of token order.

---

### Q2. What is the original positional encoding method?

The original Transformer uses deterministic **sinusoidal positional encoding** based on sine and cosine functions.

---

### Q3. Why were sine and cosine functions chosen?

They provide unique positional representations, preserve smooth relative relationships between positions, and generalize to longer sequences.

---

### Q4. What is the difference between sinusoidal and learned positional embeddings?

* **Sinusoidal:** Fixed mathematical function, no training required, generalizes to longer sequences.
* **Learned:** Position vectors are learned during training but are limited by the maximum training sequence length.

---

### Q5. What is RoPE?

Rotary Positional Embedding is a technique that encodes position by rotating the Query and Key vectors before computing attention, allowing the model to naturally capture relative positional information.

---

### Q6. Which positional encoding is used by modern LLMs?

Most modern decoder-only LLMs, including **Llama**, **Gemma**, **Qwen**, **DeepSeek**, and **Mistral**, use **Rotary Positional Embeddings (RoPE)**.
