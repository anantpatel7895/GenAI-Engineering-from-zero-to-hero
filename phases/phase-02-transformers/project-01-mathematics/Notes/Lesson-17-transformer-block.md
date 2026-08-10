# Lesson 17 — Complete Transformer Block

## 1. What Is a Transformer Block?

A **Transformer Block** is the fundamental repeated unit of a Transformer model.

Modern models such as:

* GPT
* Llama
* Qwen
* Gemma
* Mistral
* DeepSeek

are constructed by stacking many Transformer blocks.

Conceptually:

```text
Input
  │
  ▼
Transformer Block 1
  │
  ▼
Transformer Block 2
  │
  ▼
Transformer Block 3
  │
  ▼
...
  │
  ▼
Transformer Block N
```

Understanding one Transformer block is therefore fundamental to understanding an entire Transformer-based LLM.

---

# 2. Main Components

A Transformer block contains two major computational sublayers:

```text
1. Multi-Head Self-Attention

2. Feed Forward Network
```

and supporting components:

```text
3. Residual Connections

4. Layer Normalization
```

Therefore:

```text
Transformer Block

├── Multi-Head Attention
├── Residual Connection
├── LayerNorm
├── Feed Forward Network
├── Residual Connection
└── LayerNorm
```

---

# 3. Original Transformer Architecture

The original Transformer paper used **Post-LayerNorm**.

```text
                         ┌──────────────────────┐
                         │                      │
                         │    Residual Path     │
                         │                      │
                         ▼                      │
Input X ─────────────────┬──────────────────────┤
                         │                      │
                         ▼                      │
                Multi-Head Attention            │
                         │                      │
                         └──────────► ⊕ ◄───────┘
                                      │
                                      ▼
                                  LayerNorm
                                      │
                                      ▼
                         ┌──────────────────────┐
                         │                      │
                         │    Residual Path     │
                         │                      │
                         ▼                      │
                         FFN                    │
                         │                      │
                         └──────────► ⊕ ◄───────┘
                                      │
                                      ▼
                                  LayerNorm
                                      │
                                      ▼
                                    Output
```

`⊕` represents **element-wise addition**.

---

# 4. Mathematical Form

Let:

```text
x = input
```

First sublayer:

```text
A = MultiHeadAttention(x)
```

First residual connection:

```text
R₁ = x + A
```

First normalization:

```text
H₁ = LayerNorm(R₁)
```

Feed Forward Network:

```text
F = FFN(H₁)
```

Second residual connection:

```text
R₂ = H₁ + F
```

Final normalization:

```text
Output = LayerNorm(R₂)
```

Therefore:

```text
H₁ = LayerNorm(x + Attention(x))

Output = LayerNorm(H₁ + FFN(H₁))
```

---

# 5. Tensor Shape Flow

Suppose:

```text
Batch Size = 2
Sequence Length = 4
d_model = 8
```

Input:

```text
(2, 4, 8)
```

Multi-Head Attention:

```text
(2, 4, 8)
```

First residual:

```text
(2, 4, 8)
```

First LayerNorm:

```text
(2, 4, 8)
```

FFN expansion:

```text
(2, 4, 32)
```

FFN projection:

```text
(2, 4, 8)
```

Second residual:

```text
(2, 4, 8)
```

Final LayerNorm:

```text
(2, 4, 8)
```

Complete shape flow:

```text
(2,4,8)
   │
   ▼
Attention
   │
   ▼
(2,4,8)
   │
   ▼
Residual
   │
   ▼
(2,4,8)
   │
   ▼
LayerNorm
   │
   ▼
(2,4,8)
   │
   ▼
FFN
   │
   ├──► (2,4,32)
   │
   └──► (2,4,8)
   │
   ▼
Residual
   │
   ▼
(2,4,8)
   │
   ▼
LayerNorm
   │
   ▼
(2,4,8)
```

The Transformer block preserves:

```text
Batch Size

Sequence Length

d_model
```

The FFN is the only component that temporarily expands the feature dimension.

---

# 6. Why Attention and FFN Both Exist

These two components perform fundamentally different jobs.

## Attention

Attention performs **token mixing**.

```text
Token 1 ◄────► Token 2
   ▲              ▲
   │              │
   └──── Token 3 ─┘
```

It allows tokens to communicate with each other.

Attention answers:

> **"Which other tokens are relevant to me?"**

---

## FFN

The FFN performs **feature transformation**.

```text
Token 1 ──► FFN ──► Token 1'

Token 2 ──► FFN ──► Token 2'

Token 3 ──► FFN ──► Token 3'
```

There is no token-to-token interaction inside the FFN.

FFN answers:

> **"Given what this token currently knows, how should I transform its representation?"**

---

# 7. Important Mental Model

A useful way to remember a Transformer block is:

```text
Attention = Communication

FFN = Computation
```

Attention allows tokens to exchange information.

FFN transforms the information that each token now contains.

---

# 8. Residual Connections

Each Transformer block contains two residual connections.

The fundamental equation is:

```text
Output = X + F(X)
```

where `F(X)` represents a sublayer such as:

```text
Multi-Head Attention
```

or:

```text
Feed Forward Network
```

The `⊕` operation is element-wise addition.

---

# 9. Why Residual Connections Are Important

Residual connections provide a direct path for:

* Information
* Gradients

through deep networks.

Instead of forcing every layer to completely transform the representation, each layer can learn an incremental modification:

```text
Output = Input + Learned Improvement
```

This makes deep Transformer architectures easier to optimize.

---

# 10. Layer Normalization

LayerNorm stabilizes the activations by normalizing each token independently across its embedding dimensions.

For:

```text
X.shape = (B, T, d_model)
```

LayerNorm operates across:

```text
d_model
```

For example:

```text
(2,4,8)
```

means each of the 8 tokens is normalized independently across its 8 embedding dimensions.

LayerNorm does not mix information between tokens.

---

# 11. Post-LayerNorm

The original Transformer uses:

```text
Sublayer
   │
   ▼
Residual Add
   │
   ▼
LayerNorm
```

Mathematically:

```text
LayerNorm(x + Sublayer(x))
```

For the complete block:

```text
H₁ = LayerNorm(x + Attention(x))

Output = LayerNorm(H₁ + FFN(H₁))
```

---

# 12. Pre-LayerNorm

Modern LLMs generally use a **Pre-LayerNorm** or related architecture.

The structure becomes:

```text
Input
  │
  ▼
LayerNorm / RMSNorm
  │
  ▼
Attention
  │
  ▼
⊕ ◄──── Input
  │
  ▼
LayerNorm / RMSNorm
  │
  ▼
FFN
  │
  ▼
⊕ ◄──── Residual
  │
  ▼
Output
```

Mathematically:

```text
A = Attention(Norm(x))

R₁ = x + A

F = FFN(Norm(R₁))

Output = R₁ + F
```

---

# 13. Why Did Modern LLMs Move Toward Pre-Norm?

As Transformer models became deeper, training stability became increasingly important.

Pre-Norm provides a cleaner residual pathway:

```text
Input
  │
  ├──────────────────────► ⊕
  │                         ▲
  ▼                         │
Norm → Attention ───────────┘
```

The residual stream can therefore pass through many layers without repeatedly passing through normalization.

This generally makes optimization more stable for deep Transformer architectures.

---

# 14. LayerNorm vs RMSNorm

The original architecture uses:

```text
LayerNorm
```

Modern LLMs frequently use:

```text
RMSNorm
```

RMSNorm removes the mean-centering step and normalizes based on the root mean square.

Conceptually:

```text
LayerNorm

Mean
 +
Variance
 +
Scale
 +
Shift
```

while:

```text
RMSNorm

RMS
 +
Scale
```

RMSNorm is computationally simpler and is widely used in modern LLM architectures.

---

# 15. Original Transformer vs Modern LLM

| Component               | Original Transformer | Modern LLM                            |
| ----------------------- | -------------------- | ------------------------------------- |
| Attention               | Multi-Head Attention | Multi-Head / optimized attention      |
| Normalization           | LayerNorm            | Often RMSNorm                         |
| Normalization placement | Post-Norm            | Usually Pre-Norm                      |
| FFN activation          | ReLU                 | Often gated activation such as SwiGLU |
| Position encoding       | Sinusoidal           | Often RoPE                            |
| Residual connections    | Yes                  | Yes                                   |

The fundamental architecture remains:

```text
Attention
+
Residual Stream
+
Feed Forward
```

but the individual components have evolved considerably.

---

# 16. PyTorch Engineering

A Transformer block should own its submodules.

Conceptually:

```python
class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff):

        self.attention = MultiHeadAttention(...)

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(...)

        self.norm2 = nn.LayerNorm(d_model)
```

Then `forward()` performs the computation.

Do **not** create trainable modules inside `forward()`.

Bad:

```python
def forward(self, x):

    attention = MultiHeadAttention(...)
    ffn = FeedForward(...)

    ...
```

This would create new parameters every forward pass.

Instead, create modules once in `__init__()` and reuse them.

---

# 17. Parameter Ownership

A Transformer block can be viewed as:

```text
TransformerBlock
│
├── MultiHeadAttention
│   ├── Wq
│   ├── Wk
│   ├── Wv
│   └── Wo
│
├── LayerNorm 1
│
├── FeedForward
│   ├── Linear 1
│   └── Linear 2
│
└── LayerNorm 2
```

This modular structure is important for:

* Testing
* Debugging
* Parameter counting
* Model serialization
* Distributed training
* Model inspection

---

# 18. Important Verification

A Transformer block should preserve the input shape.

For:

```text
Input = (B,T,d_model)
```

we expect:

```text
Output = (B,T,d_model)
```

Example:

```text
Input:

(2,4,8)

Output:

(2,4,8)
```

Therefore:

```python
assert x.shape == output.shape
```

should pass.

Also verify that the output contains no NaNs:

```python
assert not torch.isnan(output).any()
```

---

# 19. Important Clarification About Normalization

LayerNorm does not mean every tensor inside the Transformer has:

```text
Mean = 0
Variance = 1
```

Only the output of the LayerNorm operation has the normalized statistics.

For example:

```text
Input
  │
  ▼
Attention
  │
  ▼
Attention Output
```

The attention output does not necessarily have zero mean and unit variance.

Then:

```text
Residual Add
  │
  ▼
LayerNorm
```

LayerNorm normalizes the tensor at that point.

This distinction is important when debugging Transformer activations.

---

# 20. Complete Architecture

The original Transformer block:

```text
                         ┌──────────────────────┐
                         │                      │
                         │    Residual Path     │
                         │                      │
                         ▼                      │
Input X ─────────────────┬──────────────────────┤
                         │                      │
                         ▼                      │
                Multi-Head Attention            │
                         │                      │
                         └──────────► ⊕ ◄───────┘
                                      │
                                      ▼
                                  LayerNorm
                                      │
                                      ▼
                         ┌──────────────────────┐
                         │                      │
                         │    Residual Path     │
                         │                      │
                         ▼                      │
                         FFN                    │
                         │                      │
                         └──────────► ⊕ ◄───────┘
                                      │
                                      ▼
                                  LayerNorm
                                      │
                                      ▼
                                    Output
```

The complete computation is:

```text
H₁ = LayerNorm(x + Attention(x))

Output = LayerNorm(H₁ + FFN(H₁))
```

---

# 21. Hands-On Implementation

Create:

```text
17_transformer_block.py
```

Use:

```text
d_model = 8
n_heads = 2
d_ff = 32
```

Input:

```text
X.shape = (2,4,8)
```

Print:

```text
Input Shape

Attention Output Shape

After First Residual

After First LayerNorm

FFN Output Shape

After Second Residual

Final Output Shape
```

Expected final shape:

```text
torch.Size([2,4,8])
```

---

# 22. Interview Questions

### Q1. What are the two major sublayers of a Transformer block?

1. Multi-Head Self-Attention
2. Feed Forward Network

---

### Q2. Why are residual connections used?

To preserve information and improve gradient flow through deep networks.

---

### Q3. Why does the FFN expand and contract the dimension?

The expansion provides additional representational capacity, while the contraction returns the representation to `d_model` so it can participate in the residual connection.

---

### Q4. What is Post-LayerNorm?

```text
Sublayer
   ↓
Residual Add
   ↓
LayerNorm
```

---

### Q5. What is Pre-LayerNorm?

```text
LayerNorm
   ↓
Sublayer
   ↓
Residual Add
```

---

### Q6. Which architecture is common in modern LLMs?

Modern decoder-only LLMs generally use **Pre-Norm architectures**, frequently with RMSNorm.

---

### Q7. Does a Transformer block change sequence length?

No.

```text
(B,T,d_model)
      ↓
(B,T,d_model)
```

---

### Q8. Does the FFN mix information between tokens?

No.

The FFN is position-wise and processes each token independently.

---

### Q9. What is the main difference between Attention and FFN?

```text
Attention → Token Mixing

FFN → Feature Transformation
```

---

### Q10. Why must FFN output have the same `d_model` as its input?

Because the FFN output participates in the residual addition:

```text
X + FFN(X)
```

Therefore both tensors must have compatible shapes.

---

# 23. Key Takeaways

The most important mental model for the Transformer block is:

```text
Attention
    ↓
Communication between tokens

FFN
    ↓
Nonlinear feature transformation

Residual
    ↓
Preserve information + improve gradient flow

Normalization
    ↓
Stabilize representations
```

Together:

```text
Transformer Block

=
Attention
+
Residual
+
Normalization
+
FFN
+
Residual
+
Normalization
```

Once this block is understood, an entire Transformer model becomes conceptually much simpler:

```text
Input
  │
  ▼
Embedding + Position Information
  │
  ▼
Transformer Block × N
  │
  ▼
Output Representation
```

This is the core architectural pattern behind modern Transformer-based language models.
