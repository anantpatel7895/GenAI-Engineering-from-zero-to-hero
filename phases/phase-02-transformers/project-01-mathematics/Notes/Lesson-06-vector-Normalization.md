# Vector Norms – Interview Questions & Answers

## 1. What is a vector norm?
A **vector norm** is a measure of a vector's **length** or **magnitude**. It is denoted as:

\[
\|v\|
\]

Different norms measure the length of a vector in different ways.

---

## 2. What is the difference between the L1 norm and the L2 norm?

| L1 Norm | L2 Norm |
|----------|----------|
| Sum of absolute values | Square root of the sum of squared values |
| Formula: `Σ |xᵢ|` | Formula: `√(Σ xᵢ²)` |
| Also called **Manhattan norm** | Also called **Euclidean norm** |
| More robust to outliers | More sensitive to large values |
| Encourages sparse solutions | Produces smooth solutions |

**Example**

For the vector `[3, 4]`:

- **L1 Norm** = `|3| + |4| = 7`
- **L2 Norm** = `√(3² + 4²) = √25 = 5`

---

## 3. How do you normalize a vector?

Normalize a vector by dividing every element by its L2 norm:

\[
\hat{v} = \frac{v}{\|v\|_2}
\]

**PyTorch**

```python
normalized = v / torch.norm(v, p=2)
```

After normalization, the vector has an L2 norm of **1** and is called a **unit vector**.

---

## 4. Why are unit vectors useful?

Unit vectors represent **direction only**, independent of magnitude.

They are useful because they:

- Remove the effect of scale
- Simplify geometric computations
- Make vector comparisons easier
- Are widely used in machine learning, graphics, robotics, and physics

---

## 5. Why are embeddings often normalized?

Embeddings may have different magnitudes even when they represent similar meanings.

Normalization removes the effect of magnitude so comparisons depend only on **direction**, improving:

- Cosine similarity
- Information retrieval
- Recommendation systems
- Semantic search
- Model stability

---

## 6. How are vector norms used in cosine similarity?

Cosine similarity is computed as:

\[
\text{Cosine Similarity} =
\frac{A \cdot B}{\|A\|\|B\|}
\]

The norms normalize the vectors so that similarity depends only on the **angle** between them.

Interpretation:

- **1** → Same direction
- **0** → Perpendicular
- **−1** → Opposite direction

---

## 7. Why is the L2 norm called the Euclidean norm?

The L2 norm measures the **straight-line distance** from the origin to a point in Euclidean space.

For a 2D vector `(x, y)`:

\[
\|v\|_2 = \sqrt{x^2 + y^2}
\]

This is exactly the distance given by the **Pythagorean theorem**, which is why the L2 norm is also known as the **Euclidean norm**.
