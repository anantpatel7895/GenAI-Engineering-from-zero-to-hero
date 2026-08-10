# Lesson 08 — Softmax

## What is Softmax?

Softmax is a mathematical function that converts a vector of arbitrary real-valued scores (**logits**) into a **probability distribution**.

It ensures that:

* Every output value lies between **0 and 1**
* The sum of all output values is **1**
* Larger input scores receive larger probabilities 

---

# Why Do We Need Softmax?

Neural networks typically produce **scores (logits)**.

Example:

```text
[2, 1, 4]
```

These values indicate relative importance, but they are **not probabilities**.

Problems:

* They do not sum to 1.
* They cannot directly represent confidence or attention weights.

Softmax converts them into:

```text
[0.114, 0.042, 0.844]
```

Now:

* Every value is between **0 and 1**
* The values sum to **1**
* They can be interpreted as probabilities

---

# Softmax Formula

For a vector:

```text
x = [x₁, x₂, ..., xₙ]
```

The Softmax of element `i` is:

```text
                 e^(xᵢ)
Softmax = ---------------------
           Σ e^(xⱼ)
```

Steps:

1. Compute the exponential of every score.
2. Sum all exponentials.
3. Divide each exponential by the total.

---

# Manual Example

Scores:

```text
[2, 1, 4]
```

Exponentials:

```text
e² ≈ 7.39

e¹ ≈ 2.72

e⁴ ≈ 54.60
```

Sum:

```text
7.39 + 2.72 + 54.60

=

64.71
```

Probabilities:

```text
7.39 / 64.71 = 0.114

2.72 / 64.71 = 0.042

54.60 / 64.71 = 0.844
```

Final output:

```text
[0.114, 0.042, 0.844]
```

Verification:

```text
0.114 + 0.042 + 0.844 = 1
```

---

# Properties of Softmax

* Produces a probability distribution.
* Every output value is between **0 and 1**.
* All probabilities sum to **1**.
* Preserves the ordering of the input scores.
* Larger scores receive higher probabilities.    ## that is why we divide attention scores by √dk in Transformers.

---

# Why Does Softmax Use Exponentials?

Exponentials have two important properties:

1. They are always positive.
2. They **amplify** larger values much more than smaller values.

Example:

Input:

```text
[10, 2, 1]
```

Softmax:

```text
[0.9995, 0.0003, 0.0001]
```

The highest score dominates the probability distribution.

This allows the model to focus strongly on the most relevant token.

---

# Numerical Stability

Computing:

```text
e^1000
```

would overflow.

Instead, PyTorch computes:

```text
scores = scores - max(scores)
```

Example:

```text
[1000, 1001, 1002]

↓

Subtract 1002

↓

[-2, -1, 0]
```

Softmax becomes:

```text
e^-2
e^-1
e^0
```

which is numerically stable.

### Important Property

Subtracting the same constant from every score **does not change the Softmax output**.

Only the relative differences between scores matter.

---

# Examples

## Equal Scores

Input:

```text
[5, 5, 5]
```

Output:

```text
[0.333, 0.333, 0.333]
```

Interpretation:

The model has no preference.

---

## One Dominant Score

Input:

```text
[10, 2, 1]
```

Output:

```text
[0.9995, 0.0003, 0.0001]
```

Interpretation:

Almost all probability is assigned to the highest score.

---

## Negative Scores

Input:

```text
[-2, -1, -3]
```

Output:

```text
[0.2447, 0.6652, 0.0900]
```

Softmax works perfectly with negative numbers because exponentials are always positive.

---

## Large Scores

Input:

```text
[1000, 1001, 1002]
```

After subtracting the maximum:

```text
[-2, -1, 0]
```

The resulting probabilities are identical to what would have been produced from the original scores, but without numerical overflow.

---

# Softmax in Transformers

The attention mechanism computes:

```python
scores = (Q @ K.T) / math.sqrt(dk)
```

These scores represent how similar each query is to each key.

Softmax converts the similarity scores into attention weights:

```python
attention = torch.softmax(scores, dim=-1)
```

Example:

Similarity scores:

```text
[3.2, 1.4, 5.7]
```

After Softmax:

```text
[0.08, 0.01, 0.91]
```

Interpretation:

The model assigns **91% of its attention** to the third token.

---

# Softmax in Language Models

Suppose the model predicts logits for three tokens:

```text
Cat   = 2.1

Dog   = 5.4

Fish  = 0.3
```

After Softmax:

```text
Cat   = 0.03

Dog   = 0.95

Fish  = 0.02
```

The model predicts **Dog** because it has the highest probability.

---

# PyTorch Implementation

```python
import torch

scores = torch.tensor([2.0, 1.0, 4.0])

probabilities = torch.softmax(scores, dim=0)

print(probabilities)
```

Output:

```text
tensor([0.1142, 0.0420, 0.8438])
```

---

# Manual Implementation

```python
import torch

scores = torch.tensor([2.0, 1.0, 4.0])

scores = scores - torch.max(scores)

exp_scores = torch.exp(scores)

softmax = exp_scores / torch.sum(exp_scores)

print(softmax)
```

---

# Logits vs Probabilities

| Logits                 | Probabilities          |
| ---------------------- | ---------------------- |
| Raw model outputs      | Softmax outputs        |
| Can be any real number | Always between 0 and 1 |
| Do not sum to 1        | Always sum to 1        |
| Used before Softmax    | Used after Softmax     |

---

# Softmax Pipeline in Transformers

```text
Input Embeddings
        │
        ▼
Linear Layers
        │
        ▼
Q, K, V
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
Attention Output
```

---

# Key Takeaways

* Softmax converts logits into probabilities.
* Output values are always between **0 and 1**.
* The outputs always sum to **1**.
* Exponentials amplify larger scores.
* Subtracting the maximum score improves numerical stability without changing the result.
* Softmax transforms attention scores into attention weights.
* Every Transformer uses Softmax inside its attention mechanism.

---

# Interview Questions

### Q1. What is Softmax?

A function that converts arbitrary real-valued scores into a probability distribution.

---

### Q2. Why do we need Softmax?

To convert logits or attention scores into probabilities that sum to **1**.

---

### Q3. Why do Softmax outputs always sum to 1?

Because each exponential is divided by the sum of all exponentials.

---

### Q4. Why are exponentials used?

Exponentials:

* Ensure positive outputs.
* Amplify larger scores.
* Help the model focus on the most relevant items.

---

### Q5. Why do we subtract the maximum score before Softmax?

To prevent numerical overflow while producing the same probability distribution.

---

### Q6. Where is Softmax used in Transformers?

Softmax converts attention scores into attention weights during self-attention.

---

### Q7. What is the difference between logits and probabilities?

Logits are raw model scores.

Probabilities are the normalized outputs after applying Softmax.
