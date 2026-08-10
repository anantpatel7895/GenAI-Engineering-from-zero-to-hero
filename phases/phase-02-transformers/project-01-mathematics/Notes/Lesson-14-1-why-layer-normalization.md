# Why Does the Distribution of Activations Change in Deep Neural Networks?

## Introduction

A common statement in deep learning is:

> **"As the network becomes deeper, the distribution of activations changes continuously."**

This statement does **not** refer to the input data changing.

Instead, it refers to the **intermediate feature vectors (activations)** produced by each layer.

---

# What Are Activations?

Every layer in a neural network produces an output.

For example:

```text
Input
   │
Linear Layer
   │
Activation
   │
Output
```

The output of one layer becomes the input to the next layer.

These intermediate outputs are called **activations**.

---

# Example

Suppose a Transformer has four layers.

```text
Input
   │
Layer 1
   │
Layer 2
   │
Layer 3
   │
Layer 4
```

Each layer produces a new embedding for every token.

---

# Layer 1

Suppose one token embedding is:

```text
[0.5, -0.2, 0.8, 1.1]
```

Mean:

```text
0.55
```

Variance:

```text
0.28
```

Everything looks stable.

---

# Layer 2

After matrix multiplication, attention, and feed-forward computation:

```text
[3.1, 4.7, 2.8, 5.3]
```

Mean:

```text
3.98
```

Variance:

```text
1.15
```

The values are already on a larger scale.

---

# Layer 3

```text
[18, 22, 15, 30]
```

Mean:

```text
21.25
```

Variance:

```text
31
```

The activations have grown even larger.

---

# Layer 4

```text
[120, 180, 95, 210]
```

Mean:

```text
151
```

Variance:

Very large.

---

# What Happened?

Every Transformer layer performs operations such as:

```python
x = x @ W
x = attention(x)
x = feed_forward(x)
```

Each operation transforms the activations.

As a result, the numerical distribution changes from layer to layer.

Example:

```text
Layer 1

Mean = 0.5

Std = 1
```

↓

```text
Layer 2

Mean = 4

Std = 5
```

↓

```text
Layer 3

Mean = 20

Std = 30
```

↓

```text
Layer 4

Mean = 150

Std = 250
```

The activations gradually drift to different scales.

This is what is meant by:

> **"The distribution of activations changes continuously as the network becomes deeper."**

---

# Why Is This a Problem?

Imagine Layer 8 during training.

Yesterday it received inputs like:

```text
[-1, 0.5, 1.2]
```

Today, because Layer 7's weights changed during training, it now receives:

```text
[80, 120, 150]
```

Tomorrow it receives:

```text
[-300, 20, 900]
```

Layer 8 is constantly trying to learn from inputs whose scale keeps changing.

This makes optimization much harder.

---

# Real-Life Analogies

## Analogy 1 — Learning to Drive 🚗

Imagine you are learning to drive.

Your instructor tells you:

> **"When the steering wheel is turned 10°, the car moves slightly to the left."**

So you learn how to react.

But imagine that every morning someone secretly changes the steering system.

### Monday

```text
10° steering  →  car turns slightly left
```

### Tuesday

```text
10° steering  →  car turns sharply left
```

### Wednesday

```text
10° steering  →  car barely turns
```

You are trying to learn how to drive, but the relationship between your **input** and the car's **behaviour** keeps changing.

Every time you start to build the right instinct, the rule you learned is already out of date.

That is similar to what can happen inside a neural network during training.

Layer 8 learns "when my input looks like this, produce that." But Layer 7's weights are updated at the same time, so the meaning of "input looks like this" keeps shifting underneath it.

---

## Analogy 2 — Grading Exams

Imagine you are a teacher grading exams.

### Day 1

Students score:

```text
40–60
```

### Day 2

Suddenly scores become:

```text
300–500
```

### Day 3

Now they are:

```text
-200 to 1000
```

Your grading strategy would constantly need to adjust because the scale keeps changing.

Deep neural network layers face a similar challenge when activations are not normalized.

---

# How Layer Normalization Solves This

Before LayerNorm:

```text
[120, 180, 95, 210]
```

↓

Compute:

* Mean
* Variance

↓

Normalize

↓

```text
[-0.7, 0.4, -1.2, 1.5]
```

Now every layer receives activations with approximately:

* Mean ≈ 0
* Variance ≈ 1

This creates a much more stable training process.

---

# Without LayerNorm

```text
Layer 1

Mean = 0.5
```

↓

```text
Layer 2

Mean = 4
```

↓

```text
Layer 3

Mean = 20
```

↓

```text
Layer 4

Mean = 150
```

The numerical scale keeps drifting.

---

# With LayerNorm

```text
Layer 1

↓

LayerNorm

↓

Mean ≈ 0
```

↓

```text
Layer 2

↓

LayerNorm

↓

Mean ≈ 0
```

↓

```text
Layer 3

↓

LayerNorm

↓

Mean ≈ 0
```

Each layer receives inputs with a consistent distribution, making optimization much more stable.

---

# Important Clarification

When people say:

> **"The distribution of activations changes continuously."**

they **do not** mean:

* The dataset changes.
* The input sentence changes.

They mean:

* The intermediate feature vectors produced by each layer change in their numerical properties (mean, variance, and scale) as they pass through the network and as the model's weights are updated during training.

---

# Why Is This Important for Transformers?

A Transformer block contains many operations:

```text
Input
   │
Multi-Head Attention
   │
Residual Add
   │
LayerNorm
   │
Feed Forward Network
   │
Residual Add
   │
LayerNorm
```

Without LayerNorm, the outputs from attention and the feed-forward network can grow or shrink unpredictably as they pass through many layers.

LayerNorm keeps the activations on a consistent scale, improving optimization and allowing very deep Transformer models to train successfully.

---

# Key Takeaways

* **Activations** are the intermediate outputs produced by each layer.
* As activations pass through many layers, their mean, variance, and scale naturally change.
* Large changes in activation distributions make optimization more difficult.
* LayerNorm stabilizes these activations by normalizing each token independently.
* Stable activation distributions improve gradient flow and enable deep Transformer architectures.

---

# Interview Questions

### Q1. What are activations?

Activations are the intermediate feature vectors produced by each neural network layer.

---

### Q2. What does "the distribution of activations changes continuously" mean?

It means the numerical properties (mean, variance, and scale) of the intermediate activations change as they pass through successive layers and as the model learns.

---

### Q3. Why is changing activation distribution a problem?

Because deeper layers continually receive inputs with different numerical scales, making optimization and convergence more difficult.

---

### Q4. How does LayerNorm help?

LayerNorm normalizes each token's embedding independently, keeping activations on a stable scale (approximately zero mean and unit variance), which leads to more stable and efficient training.
