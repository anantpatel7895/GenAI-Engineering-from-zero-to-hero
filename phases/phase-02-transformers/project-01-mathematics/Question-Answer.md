# Project 1 — End-to-End Transformer Computation

## From Token IDs to Transformer Block Output

This note connects the individual mathematical components we implemented throughout Project 1 into one complete computation pipeline.

```text
Token IDs
   │
   ▼
Token Embedding
   │
   ▼
Positional Information
   │
   ▼
Input Representation X
   │
   ▼
Q, K, V
   │
   ▼
QKᵀ
   │
   ▼
Scaling by √dₖ
   │
   ▼
Causal Mask
   │
   ▼
Softmax
   │
   ▼
Attention Weights
   │
   ▼
Attention Weights × V
   │
   ▼
Multi-Head Concatenation
   │
   ▼
Output Projection
   │
   ▼
Residual Connection
   │
   ▼
LayerNorm
   │
   ▼
Feed Forward Network
   │
   ▼
Residual Connection
   │
   ▼
LayerNorm
   │
   ▼
Transformer Block Output
```

---

# 1. Token IDs

A language model does not directly receive text.

Suppose the input is:

```text
I love deep learning
```

The tokenizer converts the text into token IDs:

```text
I       → 101
love    → 582
deep    → 931
learning → 1742
```

The exact IDs depend on the tokenizer vocabulary.

The model therefore receives:

```text
[101, 582, 931, 1742]
```

If the batch contains multiple sequences:

```text
X.shape = (B, T)
```

where:

```text
B = batch size
T = sequence length
```

Example:

```text
B = 2
T = 4

Token IDs shape:

(2, 4)
```

At this point, the numbers are simply vocabulary indices.

They do not yet contain useful semantic representations.

---

# 2. Token Embedding

The `embedding layer` converts each token ID into a `dense vector`.

Suppose:

```text
vocabulary size = 10,000
d_model = 512
```

The embedding table has shape:

```text
(10,000, 512)
```

Each token ID selects one row.

For example:

```text
"I"
   ↓
Embedding lookup
   ↓
[0.12, -0.43, 0.91, ..., 0.27]
```

Therefore:

```text
Token IDs

(B, T)

   ↓

Embedding

(B, T, d_model)
```

For example:

```text
(2, 4)

   ↓

(2, 4, 512)
```

Now every token has a dense vector representation.

---

# 3. What Does the Embedding Represent?

The embedding is a `learned numerical representation` of the token.

Instead of representing:

```text
"cat"
```

as an arbitrary integer:

```text
512
```

the model represents it using a vector:

```text
[
  0.21,
 -0.73,
  0.44,
  ...
]
```

The dimensions do not individually correspond to human-readable concepts.

Instead, the model learns a high-dimensional representation where relationships between vectors become useful for downstream computation.

For example, during training, representations can learn useful relationships involving:

```text
semantic similarity
syntax
context
linguistic patterns
relationships
```

Important:

> The embedding is learned during training.

It is not manually designed.

---

# 4. Positional Information

Attention by itself does not inherently understand sequence order.

Consider:

```text
The dog chased the cat
```

and:

```text
The cat chased the dog
```

The same tokens exist, but their order changes the meaning.

We therefore need positional information.

For the original Transformer, positional encoding is added to token embeddings:

```text
Input Representation
=
Token Embedding
+
Positional Encoding
```

Conceptually:

```text
Token Embedding
      +
Position Encoding
      │
      ▼
Final Input Representation
```

---

# 5. Why Add Positional Information?

Suppose:

```text
Token A
Token B
Token C
```

Without positional information, self-attention can operate on the token representations but does not inherently know:

```text
A comes before B
B comes before C
```

Positional information gives each position a representation of its location.

For sinusoidal positional encoding:

```text
PE(pos, 2i)
   = sin(pos / 10000^(2i/d_model))

PE(pos, 2i+1)
   = cos(pos / 10000^(2i/d_model))
```

The important concept is:

```text
Token identity
      +
Position information
      =
Contextual input representation
```

Modern LLMs often use other positional mechanisms such as RoPE rather than the original sinusoidal encoding.

---

# 6. Input Representation X

After combining token and positional information:

```text
X = TokenEmbedding + PositionalInformation
```

we obtain:

```text
X.shape = (B, T, d_model)
```

For our Project 1 examples:

```text
X.shape = (2, 4, 8)
```

Meaning:

```text
2 sequences
4 tokens per sequence
8-dimensional representation per token
```

This tensor becomes the input to the Transformer block.

---

# 7. Creating Q, K and V

The Transformer creates three different representations from X.

```text
Q = XWq

K = XWk

V = XWv
```

where:

```text
Wq = Query projection
Wk = Key projection
Wv = Value projection
```

These are learned parameters.

Suppose:

```text
X = (B, T, d_model)
```

then commonly:

```text
Q = (B, T, d_k)
K = (B, T, d_k)
V = (B, T, d_v)
```

For standard self-attention implementations, the dimensions are usually chosen so that:

```text
d_k = d_v
```

within each head.

---

# 8. What Do Q, K and V Mean?

A useful mental model is:

```text
Query
"What information am I looking for?"

Key
"What information do I contain?"

Value
"What information should I provide?"
```

Suppose the current token is:

```text
"deep"
```

Its query represents what information the token is looking for.

Every token has a key representing what information it can offer for matching.

The query interacts with keys:

```text
Query × Keys
```

to determine which tokens are relevant.

Then the corresponding values are aggregated.

---

# 9. Calculating QKᵀ

We calculate:

```text
Scores = QKᵀ
```

Suppose:

```text
Q = (B, T, d_k)

K = (B, T, d_k)
```

Then:

```text
Kᵀ = (B, d_k, T)
```

Therefore:

```text
QKᵀ
=
(B, T, d_k)
×
(B, d_k, T)

=
(B, T, T)
```

The resulting matrix is the **attention score matrix**.

---

# 10. What Does QKᵀ Calculate?

Each element represents a dot product:

```text
Score(i,j) = Q_i · K_j
```

Meaning:

> How compatible is the query of token `i` with the key of token `j`?

For a sequence:

```text
I     love     deep     learning
```

the attention matrix conceptually looks like:

```text
             Keys
          I   love deep learning
       ┌────────────────────────
Query I│
     love│
     deep│
learning│
```

Each row corresponds to one query token.

Each column corresponds to one key token.

---

# 11. Attention Matrix Shape

If:

```text
B = 2
T = 5
```

then:

```text
Attention Scores

(B, T, T)

=
(2, 5, 5)
```

The important property is:

```text
Attention Matrix = (Batch, Query Tokens, Key Tokens)
```

or:

```text
(B, T, T)
```

This means:

> For every query token, we calculate a score against every key token.

---

# 12. Scaling by √dₖ

Raw attention scores are:

```text
QKᵀ
```

We scale them:

```text
Scores =
QKᵀ / √dₖ
```

The complete equation is:

```text
Attention(Q,K,V)
=
softmax(
    QKᵀ / √dₖ
) V
```

---

# 13. Why Divide by √dₖ?

This is one of the most important mathematical details in attention.

Suppose the components of Q and K have roughly:

```text
mean = 0
variance = 1
```

The dot product is:

```text
q₁k₁ + q₂k₂ + ... + q_dk_d
```

As `d_k` increases, the variance of the dot product grows approximately in proportion to `d_k`.

Therefore:

```text
larger d_k
     ↓
larger raw scores
     ↓
larger logits
     ↓
softmax becomes extremely sharp
     ↓
very small gradients
```

Dividing by:

```text
√dₖ
```

controls the scale of the logits.

Important:

```text
Correct:

QKᵀ / √dₖ

Not:

QKᵀ / dₖ
```

---

# 14. Causal Mask

For autoregressive language models, a token must not see future tokens.

Suppose:

```text
I love deep learning
```

During training, the model learns:

```text
Step 1

Input:
[I]

Predict:
love
```

```text
Step 2

Input:
[I, love]

Predict:
deep
```

```text
Step 3

Input:
[I, love, deep]

Predict:
learning
```

Therefore the representation at each position must not have access to future tokens.

---

# 15. Causal Attention Pattern

For:

```text
[I, love, deep, learning]
```

the allowed attention pattern is:

```text
             I   love   deep   learning
        ┌────────────────────────────────
I       │  ✓     ✗      ✗       ✗
love    │  ✓     ✓      ✗       ✗
deep    │  ✓     ✓      ✓       ✗
learning│  ✓     ✓      ✓       ✓
```

Each token can attend to:

```text
itself
+
previous tokens
```

but not:

```text
future tokens
```

---

# 16. Why Use -∞?

Future positions are masked by setting their attention scores to:

```text
-inf
```

Before softmax:

```text
[2.1, 1.3, -inf, -inf]
```

Then:

```text
e^(-inf) = 0
```

Therefore:

```text
softmax(-inf) = 0
```

The corresponding attention weights become exactly zero.

So:

```text
Future score
     ↓
-inf
     ↓
softmax
     ↓
0 attention weight
```

This prevents future information leakage.

---

# 17. Softmax

After scaling and masking:

```text
Masked Scores
      ↓
    Softmax
      ↓
Attention Weights
```

Mathematically:

```text
softmax(x_i)
=
e^(x_i)
/
Σ e^(x_j)
```

Softmax converts arbitrary scores into a normalized distribution.

Properties:

```text
0 < probability < 1

sum(probabilities) ≈ 1
```

For example:

```text
Scores:

[2, 1, 0]
```

may produce approximately:

```text
[0.665, 0.245, 0.090]
```

The highest score gets the highest attention weight.

---

# 18. What Does Softmax Actually Do?

Suppose a query produces:

```text
Scores:

[1.0, 3.0, 2.0]
```

Softmax turns them into:

```text
Attention Weights:

[0.090, 0.665, 0.245]
```

Now the model can interpret them as:

```text
Token 1 → 9.0% attention
Token 2 → 66.5% attention
Token 3 → 24.5% attention
```

This gives us a differentiable way to decide where the query should focus.

---

# 19. Attention Weights × V

Once we have:

```text
Attention Weights
```

we use them to combine the value vectors.

The equation is:

```text
Output =
AttentionWeights × V
```

or:

```text
A = softmax(QKᵀ / √dₖ)

Output = AV
```

Suppose:

```text
Attention weights:

[0.1, 0.7, 0.2]
```

and the value vectors are:

```text
V₁
V₂
V₃
```

Then:

```text
Output =
0.1V₁
+
0.7V₂
+
0.2V₃
```

Therefore the output is a weighted combination of information from the tokens.

---

# 20. The Core Attention Equation

Everything above can be summarized as:

```text
Q = XWq

K = XWk

V = XWv
```

Then:

```text
Scores = QKᵀ / √dₖ
```

Then:

```text
A = softmax(Scores)
```

Then:

```text
Output = AV
```

Therefore:

```text
┌─────────────────────────────┐
│ Scaled Dot-Product Attention│
└─────────────────────────────┘

        QKᵀ
         │
         ▼
      / √dₖ
         │
         ▼
      Mask
         │
         ▼
      Softmax
         │
         ▼
   Attention Weights
         │
         ▼
         × V
         │
         ▼
      Output
```

---

# 21. Why Multiple Heads?

A single attention mechanism has one representation of relationships.

Multi-Head Attention allows the model to learn multiple attention patterns simultaneously.

Suppose:

```text
d_model = 512

n_heads = 8
```

Then:

```text
d_head = 512 / 8
       = 64
```

Instead of one 512-dimensional attention computation, we have:

```text
Head 1 → 64 dimensions
Head 2 → 64 dimensions
Head 3 → 64 dimensions
...
Head 8 → 64 dimensions
```

Together:

```text
8 × 64 = 512
```

---

# 22. What Can Different Heads Learn?

Different heads can learn different relationships.

Conceptually:

```text
Head 1
→ syntactic relationships

Head 2
→ nearby token relationships

Head 3
→ subject/object relationships

Head 4
→ long-range dependencies

...
```

We should not assume that every head has a clean human-interpretable role, but multiple heads provide the model with multiple representation subspaces.

---

# 23. Splitting into Heads

Initially:

```text
Q.shape = (B, T, d_model)
```

Suppose:

```text
d_model = 8
n_heads = 2
d_head = 4
```

We transform:

```text
(B,T,8)
```

into:

```text
(B,2,T,4)
```

The same applies to:

```text
K
V
```

Therefore:

```text
Q = (B,H,T,d_head)

K = (B,H,T,d_head)

V = (B,H,T,d_head)
```

Attention is then calculated independently for each head.

---

# 24. Per-Head Attention

For each head:

```text
Q_h
K_h
V_h
```

we calculate:

```text
Attention_h
=
softmax(
    Q_h K_hᵀ / √d_head
) V_h
```

The output shape per head is:

```text
(B,T,d_head)
```

With `H` heads:

```text
(B,H,T,d_head)
```

---

# 25. Multi-Head Concatenation

After all heads produce their outputs:

```text
Head 1
Head 2
...
Head H
```

we concatenate them along the embedding dimension.

For:

```text
H = 2
d_head = 4
```

we get:

```text
4 + 4 = 8
```

Therefore:

```text
(B,H,T,d_head)
```

becomes:

```text
(B,T,d_model)
```

Example:

```text
(B,2,T,4)
      ↓
(B,T,8)
```

---

# 26. Output Projection

After concatenation, the combined representation is passed through an output projection:

```text
Output = Concat(Head₁,...,Headₕ) Wo
```

where:

```text
Wo
```

is a learned weight matrix.

This allows the model to mix information from the different attention heads.

---

# 27. Residual Connection

The attention output is then combined with the original input:

```text
X + AttentionOutput
```

Conceptually:

```text
             ┌──────────────────┐
             │                  │
             │    Original X    │
             │                  │
             └────────► ⊕ ◄─────┘
                        ▲
                        │
                  Attention
```

The residual connection allows the model to preserve the existing representation while adding the learned transformation.

Think of it as:

```text
New Representation
=
Original Representation
+
Learned Modification
```

---

# 28. Why Residual Connections Matter

Without residual connections, every layer would have to transform the representation completely.

With residual connections:

```text
Output = X + F(X)
```

the layer can learn:

```text
F(X) ≈ useful modification
```

rather than rebuilding the representation from scratch.

Residual connections also provide a direct path for gradients through deep networks.

This becomes extremely important when stacking many Transformer blocks.

---

# 29. LayerNorm

After the residual connection in the original Post-Norm architecture:

```text
X + AttentionOutput
        │
        ▼
    LayerNorm
```

LayerNorm normalizes each token across its embedding dimension.

Given:

```text
X.shape = (B,T,D)
```

LayerNorm operates over:

```text
D
```

not:

```text
B
```

and not:

```text
T
```

Therefore every token is normalized independently.

---

# 30. Example of LayerNorm

Suppose one token representation is:

```text
[2, 4, 6, 8]
```

LayerNorm calculates statistics over:

```text
[2, 4, 6, 8]
```

and produces a normalized representation.

It does not combine:

```text
Token 1
```

with:

```text
Token 2
```

Therefore:

```text
Attention
→ mixes information between tokens

LayerNorm
→ normalizes features within each token
```

This distinction is fundamental.

---

# 31. Feed Forward Network

After the first normalization:

```text
LayerNorm
   ↓
FFN
```

A typical FFN performs:

```text
Linear
   ↓
Activation
   ↓
Linear
```

For our Project 1 implementation:

```text
Linear 8 → 32
   ↓
ReLU
   ↓
Linear 32 → 8
```

So:

```text
(B,T,8)
   ↓
(B,T,32)
   ↓
(B,T,32)
   ↓
(B,T,8)
```

---

# 32. Why Expand the Dimension?

The FFN temporarily expands:

```text
d_model
   ↓
d_ff
```

For example:

```text
8 → 32
```

The larger intermediate dimension gives the network greater capacity for nonlinear feature transformation.

Then it projects back:

```text
32 → 8
```

so the result can participate in the residual connection.

---

# 33. Does the FFN Mix Tokens?

No.

This is very important.

For:

```text
X.shape = (B,T,D)
```

the FFN applies the same transformation independently to each token.

Conceptually:

```text
Token 1 ──► FFN ──► Token 1'

Token 2 ──► FFN ──► Token 2'

Token 3 ──► FFN ──► Token 3'
```

There is no:

```text
Token 1 ↔ Token 2
```

interaction inside the FFN.

Therefore:

```text
Attention
    → token mixing

FFN
    → feature transformation
```

---

# 34. Second Residual Connection

After the FFN:

```text
FFN Output
     +
Previous Representation
     │
     ▼
Residual Output
```

Mathematically:

```text
R₂ = H₁ + FFN(H₁)
```

where:

```text
H₁
```

is the output of the first LayerNorm in our Post-Norm implementation.

---

# 35. Final LayerNorm

The second residual output is normalized:

```text
R₂
 │
 ▼
LayerNorm
 │
 ▼
Transformer Block Output
```

Therefore the complete Post-Norm block can be written as:

```text
H₁
=
LayerNorm(
    X + Attention(X)
)

Output
=
LayerNorm(
    H₁ + FFN(H₁)
)
```

---

# 36. Complete Transformer Block

Putting everything together:

```text
                           ┌─────────────────────────┐
                           │                         │
                           │      Residual X         │
                           │                         │
                           └────────────► ⊕ ◄────────┘
                                         ▲
                                         │
                                         │
Input X ───────► Multi-Head Attention ────┘
                                         │
                                         ▼
                                     LayerNorm
                                         │
                                         ▼
                                         H₁
                                         │
                                         ▼
                                        FFN
                                         │
                                         ▼
                           ┌─────────────────────────┐
                           │                         │
                           │      Residual H₁        │
                           │                         │
                           └────────────► ⊕ ◄────────┘
                                         ▲
                                         │
                                         │
                                         └──── FFN(H₁)
                                         │
                                         ▼
                                     LayerNorm
                                         │
                                         ▼
                                       Output
```

---

# 37. Complete Mathematical Pipeline

The entire block can be represented mathematically as:

## Step 1 — Input

```text
X
```

---

## Step 2 — QKV projections

```text
Q = XWq

K = XWk

V = XWv
```

---

## Step 3 — Attention scores

```text
S = QKᵀ
```

---

## Step 4 — Scaling

```text
S_scaled = S / √dₖ
```

---

## Step 5 — Causal masking

```text
S_masked[i,j] =
    S_scaled[i,j]    if j ≤ i
    -∞               if j > i
```

---

## Step 6 — Softmax

```text
A = softmax(S_masked)
```

---

## Step 7 — Weighted values

```text
AttentionOutput = AV
```

---

## Step 8 — Multi-head concatenation

```text
ConcatHeads
```

---

## Step 9 — Output projection

```text
MHAOutput = ConcatHeads Wo
```

---

## Step 10 — Residual

```text
R₁ = X + MHAOutput
```

---

## Step 11 — LayerNorm

```text
H₁ = LayerNorm(R₁)
```

---

## Step 12 — FFN

```text
F = W₂ σ(W₁H₁ + b₁) + b₂
```

For our implementation:

```text
σ = ReLU
```

---

## Step 13 — Second residual

```text
R₂ = H₁ + F
```

---

## Step 14 — Final normalization

```text
Output = LayerNorm(R₂)
```

---

# 38. Tensor Shape Trace

For our Project 1 configuration:

```text
B = 2
T = 4
D = 8
H = 2
d_head = 4
d_ff = 32
```

The complete shape flow is:

```text
Token IDs
    │
    ▼
(2,4)
    │
    ▼
Embedding
    │
    ▼
(2,4,8)
    │
    ▼
Q,K,V
    │
    ▼
(2,4,8)
    │
    ▼
Split Heads
    │
    ▼
(2,2,4,4)
    │
    ▼
QKᵀ
    │
    ▼
(2,2,4,4)
    │
    ▼
Scale + Mask
    │
    ▼
(2,2,4,4)
    │
    ▼
Softmax
    │
    ▼
(2,2,4,4)
    │
    ▼
× V
    │
    ▼
(2,2,4,4)
    │
    ▼
Concatenate Heads
    │
    ▼
(2,4,8)
    │
    ▼
Output Projection
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
FFN 8 → 32
    │
    ▼
(2,4,32)
    │
    ▼
ReLU
    │
    ▼
(2,4,32)
    │
    ▼
FFN 32 → 8
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
```

---

# 39. The Most Important Mental Model

The Transformer block can be understood as two major operations:

```text
┌─────────────────────────────┐
│       Communication         │
│                             │
│   Multi-Head Attention      │
│                             │
│   Tokens talk to tokens     │
└─────────────────────────────┘

              ↓

┌─────────────────────────────┐
│        Computation          │
│                             │
│      Feed Forward           │
│                             │
│   Transform token features  │
└─────────────────────────────┘
```

With residual and normalization around them:

```text
Attention
   ↓
Residual
   ↓
Normalization
   ↓
FFN
   ↓
Residual
   ↓
Normalization
```

This pattern is repeated many times.

---

# 40. From One Transformer Block to a Language Model

A single Transformer block is not yet a language model.

We stack many blocks:

```text
Input Embeddings
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
      │
      ▼
Final Representation
```

For example:

```text
GPT-style model

Embedding
    ↓
Block 1
    ↓
Block 2
    ↓
Block 3
    ↓
...
    ↓
Block N
```

Each block progressively transforms the token representations into increasingly contextual representations.

---

# 41. From Representation to Next Token

After the final Transformer block:

```text
Final Representation
        │
        ▼
LM Head
        │
        ▼
Vocabulary Logits
        │
        ▼
Softmax
        │
        ▼
Next Token Probabilities
```

Suppose the vocabulary contains:

```text
50,000 tokens
```

and:

```text
d_model = 768
```

The language-model head projects:

```text
( B,T,768 )
```

into:

```text
( B,T,50_000 )
```

Each position therefore produces a probability distribution over the entire vocabulary.

---

# 42. Autoregressive Generation

Suppose the user provides:

```text
I love
```

The model predicts:

```text
deep
```

Then the sequence becomes:

```text
I love deep
```

The model predicts:

```text
learning
```

Then:

```text
I love deep learning
```

The process repeats:

```text
Context
   ↓
Transformer
   ↓
Logits
   ↓
Sampling / Selection
   ↓
Next Token
   ↓
Append Token
   ↓
Repeat
```

This is autoregressive generation.

---

# 43. Training vs Inference

During training, we can process many positions in parallel.

Example:

```text
Input:

[I, love, deep, learning]
```

Targets:

```text
[love, deep, learning, <EOS>]
```

The causal mask prevents each position from accessing future information.

During inference, generation happens autoregressively:

```text
[I]
 ↓
predict love

[I, love]
 ↓
predict deep

[I, love, deep]
 ↓
predict learning
```

This distinction becomes extremely important when we later study:

```text
KV Cache
```

---

# 44. The Entire GPT-Style Architecture

Now we can see the complete picture:

```text
                 Token IDs
                     │
                     ▼
               Token Embedding
                     │
                     ▼
             Positional Information
                     │
                     ▼
               Input Representation
                     │
                     ▼
        ┌─────────────────────────────┐
        │ Transformer Block 1         │
        └─────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │ Transformer Block 2         │
        └─────────────────────────────┘
                     │
                     ▼
                    ...
                     │
                     ▼
        ┌─────────────────────────────┐
        │ Transformer Block N         │
        └─────────────────────────────┘
                     │
                     ▼
              Final Representation
                     │
                     ▼
                   LM Head
                     │
                     ▼
              Vocabulary Logits
                     │
                     ▼
                  Softmax
                     │
                     ▼
             Next Token Probability
                     │
                     ▼
             Sampling / Selection
                     │
                     ▼
                Next Token
```

---

# 45. Key Interview Summary

### What does embedding do?

Converts discrete token IDs into learned dense vectors.

```text
Token ID → Vector
```

---

### Why positional information?

Because the model needs information about token order.

```text
Token meaning + Position
```

---

### What are Q, K and V?

```text
Q → what am I looking for?

K → what information do I contain?

V → what information should I provide?
```

---

### What does QKᵀ do?

Computes query-key compatibility scores.

```text
QKᵀ → attention scores
```

---

### Why scale?

```text
QKᵀ / √dₖ
```

controls the magnitude of attention logits and prevents softmax from becoming unnecessarily saturated as the head dimension grows.

---

### Why causal masking?

Prevents future-token information from leaking into the current prediction.

```text
Future score → -∞ → softmax → 0
```

---

### What does softmax do?

Converts scores into normalized attention weights.

```text
weights ∈ (0,1)

Σ weights ≈ 1
```

---

### Why multiply by V?

To create a weighted combination of the information represented by the value vectors.

```text
Attention Output = Weights × V
```

---

### Why multiple heads?

To allow the model to perform attention in multiple learned representation subspaces simultaneously.

---

### Why residual connections?

To preserve information and provide a direct path for gradient flow.

```text
Output = X + F(X)
```

---

### What does LayerNorm do?

Normalizes each token's features across the embedding dimension.

```text
(B,T,D)

normalize across D
```

---

### What does FFN do?

Performs nonlinear feature transformation independently for each token.

```text
Attention → token mixing

FFN → feature transformation
```

---

# 46. Final Mental Model

If you remember only one diagram from Project 1, remember this:

```text
                 TOKEN IDs
                    │
                    ▼
                EMBEDDING
                    │
                    ▼
          + POSITION INFORMATION
                    │
                    ▼
                    X
                    │
                    ▼
        ┌──────────────────────────┐
        │   MULTI-HEAD ATTENTION   │
        │                          │
        │   Q = XWq                │
        │   K = XWk                │
        │   V = XWv                │
        │                          │
        │   QKᵀ / √dₖ              │
        │   Mask                   │
        │   Softmax                │
        │   × V                    │
        └────────────┬─────────────┘
                     │
                     ▼
                  RESIDUAL
                     │
                     ▼
                 LAYERNORM
                     │
                     ▼
                   FFN
                     │
                     ▼
                  RESIDUAL
                     │
                     ▼
                 LAYERNORM
                     │
                     ▼
             TRANSFORMER OUTPUT
```

The deepest conceptual summary is:

```text
Attention
    =
    "Which other tokens should I use?"

FFN
    =
    "How should I transform what I now know?"

Residual
    =
    "Keep the existing representation while adding useful changes."

LayerNorm
    =
    "Keep the representation numerically well-behaved."

Multi-Head Attention
    =
    "Perform several attention computations in parallel."

Stacked Transformer Blocks
    =
    "Repeatedly build richer contextual representations."
```

This is the mathematical foundation behind the architecture of modern Transformer-based language models.


# Q47. What is a matrix?

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

### Q7. Why normalize embeddings?

Normalization removes the effect of vector magnitude so that similarity depends only on direction. This makes semantic comparisons more meaningful and allows the dot product of normalized vectors to equal cosine similarity.

### Q8. What is the shape of the attention scores matrix?

The attention scores matrix has the shape `(seq_len, seq_len)`, where `seq_len` is the number of tokens in the sequence. Each entry represents the similarity between a pair of tokens.