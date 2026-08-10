# Lesson 15 — Residual Connections (Skip Connections)

## Introduction

Residual Connections (also called **Skip Connections**) are one of the most important architectural innovations in deep learning.

They were first introduced in **ResNet (2015)** and later adopted by the Transformer architecture.

Without residual connections, training very deep neural networks becomes extremely difficult.

Modern models such as:

* GPT
* BERT
* Llama
* Qwen
* Gemma
* DeepSeek
* Mistral

all rely heavily on residual connections.

---

# Why Do We Need Residual Connections?

Consider a deep Transformer with many layers.

```text id="jlwmr1"
Input
   │
Layer 1
   │
Layer 2
   │
Layer 3
   │
...
   │
Layer 96
```

Each layer transforms the input representation.

As information passes through many layers:

* The original information becomes increasingly transformed.
* Gradients become weaker during backpropagation.
* Optimization becomes much harder.

Residual connections solve these problems by providing a direct path for both information and gradients.

---

# The Problem

Suppose the input embedding is:

```text id="jlwmr2"
x = [1, 2, 3]
```

A neural network layer computes:

```text id="jlwmr3"
F(x) = [0.5, -1, 2]
```

Without a residual connection:

```text id="jlwmr4"
Output = F(x)

↓

[0.5, -1, 2]
```

The original input is completely replaced.

---

# The Solution

Instead of replacing the input, we preserve it.

The Transformer computes:

```text id="jlwmr5"
Output = x + F(x)
```

Example:

```text id="jlwmr6"
Input

[1, 2, 3]

+

Layer Output

[0.5, -1, 2]

=

Residual Output

[1.5, 1, 5]
```

The output now contains:

* Original information (`x`)
* Newly learned information (`F(x)`)

---

# Visualization

Without a residual connection:

```text id="jlwmr7"
Input
   │
   ▼
Layer
   │
   ▼
Output
```

With a residual connection:

```text id="jlwmr8"
          ┌───────────────┐
          │               │
Input ────┼──────────────► +
          │               │
          ▼               │
        Layer             │
          │               │
          └───────────────┘
                  │
                  ▼
               Output
```

The input "skips" the layer and is added back to its output.

---

# Why Use Addition?

A common interview question is:

> Why do Transformers use addition instead of concatenation?

Suppose:

```text id="jlwmr9"
Embedding Dimension = 768
```

### Option 1 — Addition

```text id="7jlwmr"
768

+

768

↓

768
```

The embedding dimension remains unchanged.

---

### Option 2 — Concatenation

```text id="8jlwmr"
768

+

768

↓

1536
```

Now every subsequent layer must process 1536-dimensional vectors.

This would:

* Increase the number of parameters.
* Consume more GPU memory.
* Increase computation time.
* Break the fixed embedding dimension used throughout the Transformer.

Addition preserves the model width while allowing information from the original input to flow forward.

---

# Shape Requirement

Residual addition requires both tensors to have the same shape.

Example:

```text id="9jlwmr"
Input

Shape = (2, 4, 8)
```

```text id="0jlwmr"
Layer Output

Shape = (2, 4, 8)
```

Then:

```python id="jlwmra"
output = x + F(x)
```

is valid.

If the shapes differ:

```text id="jlwmrb"
(2,4,8)

+

(2,4,16)
```

the addition cannot be performed.

This is why every Transformer sublayer preserves the embedding dimension.

---

# Identity Mapping

Suppose a layer has not yet learned anything useful.

Ideally, it should not make the model worse.

If:

```text id="‍ജlwմrc"
F(x) = 0
```

then:

```text id="jlwmrd"
Output = x + 0

↓

Output = x
```

The layer behaves like an identity function.

As training progresses, the layer learns only the **difference** that should be added to the input.

This "difference" is called the **residual**.

---

# Why Is Learning the Residual Easier?

Instead of learning:

```text id="jlwmre"
Desired Output
```

the layer only learns:

```text id="jlwmrf"
Desired Output − Input
```

In other words:

```text id="rywjlwm"
Residual

=

F(x)
```

This simplifies optimization and allows very deep networks to train effectively.

---

# Gradient Flow

Without residual connections:

```text id="jlwmrg"
Layer 10
   │
Layer 9
   │
Layer 8
   │
Layer 7
```

Gradients become weaker as they travel backward through many layers.

This contributes to the **vanishing gradient problem**.

---

With residual connections:

```text id="ikarhi1"
Layer 10
     │
     ├──────────────► Layer 7
     │
     ▼
Normal Path
```

The shortcut provides an additional path for gradients.

This significantly improves gradient flow and stabilizes training.

---

# Residual Connections in a Transformer Block

Each Transformer block contains **two residual connections**.

### Around Multi-Head Attention

```text id="akathi2"
Input
   │
   ├───────────────┐
   │               │
   ▼               │
Multi-Head Attention
   │               │
   └───────► Add ◄─┘
            │
            ▼
        LayerNorm
```

---

### Around Feed Forward Network

```text id="akathi3"
Input
   │
   ├───────────────┐
   │               │
   ▼               │
Feed Forward Network
   │               │
   └───────► Add ◄─┘
            │
            ▼
        LayerNorm
```

Modern LLMs often use **Pre-LayerNorm**, but the residual connection itself remains the same.

---

# PyTorch Example

```python id="akathi4"
import torch

x = torch.randn(2, 4, 8)

fx = torch.randn(2, 4, 8)

output = x + fx
```

Output shape:

```text id="akathi5"
(2, 4, 8)
```

The shape remains unchanged.

---

# Engineering Perspective

Residual connections make it possible to train extremely deep models.

Without them:

* Information would gradually degrade.
* Gradients would become weaker.
* Optimization would become unstable.

Residual connections allow:

* Stable optimization.
* Better gradient flow.
* Preservation of useful information.
* Incremental refinement of representations.

This is one of the main reasons Transformers can scale to dozens or even hundreds of layers.

---

# Key Takeaways

* Residual connections add the original input back to the layer output.
* The Transformer computes:

```text id="akathi6"
Output = x + F(x)
```

* The input and layer output must have the same shape.
* Addition preserves the embedding dimension.
* Residual connections improve information flow and gradient propagation.
* Each Transformer block contains two residual connections.
* Residual connections are fundamental to modern deep neural networks.

---

# Interview Questions

### Q1. What is a residual connection?

A shortcut connection that adds the original input to the output of a neural network layer.

---

### Q2. Why are residual connections needed?

They preserve information, improve gradient flow, and make very deep networks easier to optimize.

---

### Q3. Why is the output computed as `x + F(x)`?

The model preserves the original input while learning only the residual (difference) that should be added.

---

### Q4. Why is addition preferred over concatenation?

Addition keeps the embedding dimension unchanged, whereas concatenation doubles the feature dimension and increases computation and memory requirements.

---

### Q5. Why must the tensor shapes match?

Element-wise addition requires both tensors to have identical shapes.

---

### Q6. How many residual connections are present in a Transformer block?

A standard Transformer block contains **two** residual connections:

1. Around the Multi-Head Attention sublayer.
2. Around the Feed Forward Network (MLP) sublayer.

---

### Q7. Which famous architecture first introduced residual connections?

Residual connections were introduced by **ResNet (2015)** and later became a core component of the Transformer architecture.
