# Kolmogorov-Arnold Networks (KANs)

## Concept and Problem Explanation

### 1. The Problem
We aim to approximate a mathematical function:
$$f(x, y) = \sin(x) + \cos(y) + xy$$

This function has:
- **Trigonometric components**: $\sin(x)$, $\cos(y)$.
- **Polynomial terms**: $xy$, a product of inputs.

---

### 2. Kolmogorov-Arnold Networks (KANs)

#### Kolmogorov-Arnold Representation Theorem
This theorem states that any multivariate continuous function $f(x_1, x_2, \ldots, x_n)$ can be decomposed into sums and compositions of **1-dimensional functions**:

$$f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^N g_i \left( \sum_{j=1}^n h_{ij}(x_j) \right)$$

For the problem $f(x, y) = \sin(x) + \cos(y) + xy$, KANs aim to:
- Learn the relationships between inputs $(x, y)$.
- Produce an interpretable, symbolic formula.

---

### 3. The Function
Let’s dissect $f(x, y) = \sin(x) + \cos(y) + xy$:

- **Input Nodes**: $x_1 = x$, $x_2 = y$.
- **Output Node**: $y = f(x_1, x_2)$.

#### Functional Form
- $\sin(x)$: A trigonometric operation applied to $x_1$.
- $\cos(y)$: A trigonometric operation applied to $x_2$.
- $xy$: A polynomial term representing the interaction between $x_1$ and $x_2$.

#### Concept Visualization

#### Inputs
- $x_1$: Raw input $x$.
- $x_2$: Raw input $y$.

#### Transformations
- $g_1(x_1) = \sin(x_1)$.
- $g_2(x_2) = \cos(x_2)$.
- $g_3(x_1, x_2) = x_1 \cdot x_2$.

#### Summation
$$f(x_1, x_2) = g_1(x_1) + g_2(x_2) + g_3(x_1, x_2)$$

#### Output
The final output node aggregates the contributions of these terms to predict $y$.


---

### 4. How KAN Approximates the Function

#### Node Structure
- Inputs $x_1$ and $x_2$ pass through **B-splines** (piecewise polynomial transformations) on edges.
- At intermediate nodes, the outputs of splines are **summed** to form composite functions.

#### Learning Process
1. **Spline Coefficients**: KAN adjusts spline coefficients to fit the dataset.
2. **Regularization**: Ensures smoothness and avoids overfitting.

#### Output
After training, the model produces a formula close to the true function:
$$f(x, y) \approx \sin(x) + \cos(y) + xy$$

#### Interpretability
Unlike traditional neural networks, the symbolic regression step in KANs allows you to extract this learned formula explicitly.

---
### Learned Symbolic Formula

After training, KAN produced the following learned symbolic formula:

$$f(x_1, x_2) = -0.5366 \cdot (x_1 - 0.931 \cdot x_2 + 0.0002)^2 - 1.555 \cdot \cos(0.9873 \cdot x_1 + 6.9914) + 2.1738$$
