# Lesson 11 — Attention Masking

## What is Attention Masking?

Attention masking is a technique used in Transformers to **prevent the model from attending to certain tokens** during the attention computation.

Masking ensures that the model only attends to valid positions.

Without masking, every token could attend to every other token.

---

# Why Do We Need Attention Masking?

Suppose the sentence is:

```text
I love deep learning today
```

When predicting the word **"learning"**, the model should **not** look at **"today"**.

That would reveal future information and make training incorrect.

Attention masking prevents this information leakage.

---

# Types of Attention Masks

There are two major types of attention masks.

```text
Attention Masks
│
├── Padding Mask
└── Causal Mask
```

---

# 1. Padding Mask

Padding masks are used when batching sequences of different lengths.

Example:

```text
Sentence A

I love AI
```

```text
Sentence B

Transformers are amazing today
```

After padding:

```text
Sentence A

I
love
AI
<PAD>
```

```text
Sentence B

Transformers
are
amazing
today
```

The `<PAD>` token contains no useful information.

The model should completely ignore padded positions.

Padding masks ensure that `<PAD>` tokens receive **zero attention**.

---

# 2. Causal Mask

Decoder-only models (GPT, Llama, Qwen, Gemma, DeepSeek, etc.) use a **causal mask**.

The causal mask prevents every token from attending to future tokens.

Example:

```text
Sentence

I   love   deep   learning
```

Attention visibility:

```text
             I   love  deep  learning

I            ✓    ✗     ✗        ✗

love         ✓    ✓     ✗        ✗

deep         ✓    ✓     ✓        ✗

learning     ✓    ✓     ✓        ✓
```

Meaning:

* **I** can attend only to itself.
* **love** can attend to **I** and **love**.
* **deep** can attend to **I**, **love**, and **deep**.
* **learning** can attend to all previous tokens and itself.

Future tokens are completely hidden.

---

# Causal Mask Matrix

For a sequence of length **5**:

```text
        T1  T2  T3  T4  T5

T1      ✓   ✗   ✗   ✗   ✗

T2      ✓   ✓   ✗   ✗   ✗

T3      ✓   ✓   ✓   ✗   ✗

T4      ✓   ✓   ✓   ✓   ✗

T5      ✓   ✓   ✓   ✓   ✓
```

Only the lower triangle is visible.

---

# Creating a Causal Mask in PyTorch

```python
import torch

mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()

print(mask)
```

Output:

```text
False  True   True   True   True
False  False  True   True   True
False  False  False  True   True
False  False  False  False  True
False  False  False  False  False
```

Interpretation:

* **False** → Keep the attention score.
* **True** → Mask the attention score.

---

# Applying the Mask

Suppose the attention score matrix is:

```text
[4, 2, 8, 1]
```

Suppose positions **3** and **4** are future tokens.

Instead of setting them to zero, we replace them with:

```text
-∞
```

Result:

```text
[4, 2, -∞, -∞]
```

PyTorch:

```python
scores = scores.masked_fill(mask, float("-inf"))
```

---

# Why Use `-∞` Instead of `0`?

Softmax computes:

```text
e^(score)
```

For masked positions:

```text
e^(-∞) = 0
```

Therefore:

```text
Softmax

↓

Probability = 0
```

Masked tokens receive **exactly zero attention**.

---

# Example

Original scores:

```text
[4, 2, 8, 1]
```

Masked scores:

```text
[4, 2, -∞, -∞]
```

Softmax:

```text
[0.88, 0.12, 0.00, 0.00]
```

The future tokens receive zero probability.

---

# Where Is Masking Applied?

The attention pipeline is:

```text
Q
│
▼
Kᵀ
│
▼
Similarity Scores
│
▼
Apply Attention Mask
│
▼
Scale by √dk
│
▼
Softmax
│
▼
Attention Weights
│
▼
Attention Weights @ V
│
▼
Output
```

> **Note:** Many implementations apply the mask before scaling or combine the operations in optimized kernels. The important point is that masking happens **before Softmax**, so masked positions receive zero probability.

---

# Training vs Inference

## Training

The entire sequence is processed in one forward pass.

The causal mask prevents information leakage from future tokens.

Example:

```text
Input

[I, love, deep, learning]

Target

[love, deep, learning, <EOS>]
```

Every next-token prediction is computed simultaneously.

---

## Inference

Tokens are generated one at a time.

Example:

```text
Step 1

Input:
[I]

Predict:
love

────────────────────────────

Step 2

Input:
[I, love]

Predict:
deep

────────────────────────────

Step 3

Input:
[I, love, deep]

Predict:
learning
```

Although generation is sequential, the model still obeys causal attention.

Modern LLMs use a **KV Cache** to reuse previously computed Keys and Values, making inference much faster.

---

# Model-wise Usage

| Model                           |          Training          |                     Inference                     |
| ------------------------------- | :------------------------: | :-----------------------------------------------: |
| Encoder-only (BERT)             |        Padding Mask        |                    Padding Mask                   |
| Decoder-only (GPT, Llama, Qwen) | Padding Mask + Causal Mask | Causal Mask (Padding Mask only if padding exists) |
| Encoder-Decoder (T5, BART)      | Padding Mask + Causal Mask |             Padding Mask + Causal Mask            |

---

# Engineering Perspective

Production implementations such as:

* PyTorch Scaled Dot Product Attention
* FlashAttention
* vLLM
* TensorRT-LLM

often use optimized boolean masks or fused attention kernels instead of explicitly writing `-∞` into the score matrix.

Mathematically, however, the effect is identical:

* Invalid positions receive **zero probability** after Softmax.
* The model never attends to masked tokens.

---

# Key Takeaways

* Attention masking prevents the model from attending to invalid positions.
* There are two main masks:

  * Padding Mask
  * Causal Mask
* Decoder-only models require causal masking.
* Masking is applied before Softmax.
* Masked scores are replaced with `-∞`.
* After Softmax, masked positions receive exactly **0** probability.
* Modern LLMs use **KV Cache** to enforce causal attention efficiently during inference.

---

# Interview Questions

### Q1. Why is attention masking required?

To prevent the model from attending to invalid tokens, such as `padding tokens` or `future tokens`.

---

### Q2. What is the difference between a Padding Mask and a Causal Mask?

* **Padding Mask** hides padded tokens.
* **Causal Mask** hides future tokens.

---

### Q3. Why do GPT models require a causal mask?

Because GPT predicts the next token and must not access future tokens during `training` or `generation`.

---

### Q4. Why do masked positions use `-∞` instead of `0`?

Because:

```text
e^(-∞) = 0
```

After Softmax, masked positions receive exactly zero attention.

---

### Q5. Where is the attention mask applied?

After computing attention scores and before applying Softmax.

---

### Q6. Is masking required during inference?

Yes.

Decoder-only models still enforce causal attention during inference, typically using a KV Cache for efficient implementation.

---

### Q7. What happens if the causal mask is removed from GPT?

The model would be able to attend to future tokens during training, causing information leakage and invalid learning.
