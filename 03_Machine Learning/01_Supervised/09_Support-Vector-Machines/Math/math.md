# Support Vector Machine 

## 1. The Idea

SVM finds the **hyperplane** that separates two classes with the **maximum margin** — the largest possible gap between the closest points of each class (called **support vectors**).

## 2. Example Dataset

Suppose we have a simple 2D dataset with two classes, $y \in \{-1, +1\}$:

| Point | $x_1$ | $x_2$ | Label $y$ |
|-------|------|------|-----------|
| $P_1$ | 2 | 3 | $+1$ |
| $P_2$ | 3 | 3 | $+1$ |
| $P_3$ | 3 | 4 | $+1$ |
| $P_4$ | 1 | 1 | $-1$ |
| $P_5$ | 2 | 1 | $-1$ |
| $P_6$ | 1 | 0 | $-1$ |

We want to find a line (hyperplane) of the form:

$$
w_1 x_1 + w_2 x_2 + b = 0
$$

that separates the $+1$ class from the $-1$ class with maximum margin.

## 3. The SVM Optimization Problem

The general goal is to find $w$ and $b$ that satisfy:

$$
\min_{w, b} \ \frac{1}{2} \|w\|^2
$$

subject to the constraint that every point is correctly classified with margin at least 1:

$$
y_i (w \cdot x_i + b) \geq 1, \quad \forall i
$$

where $w \cdot x_i = w_1 x_{i1} + w_2 x_{i2}$.

## 4. Intuition Behind the Constraint

- For points on the $+1$ side: $w \cdot x_i + b \geq 1$
- For points on the $-1$ side: $w \cdot x_i + b \leq -1$

The **margin width** between the two boundary planes is:

$$
\text{margin} = \frac{2}{\|w\|}
$$

Maximizing the margin is the same as minimizing $\|w\|$ (or $\frac{1}{2}\|w\|^2$ for a smoother, differentiable objective).

## 5. Solving This Example

By inspection, the closest points between the two classes are:

$$
P_2 = (3, 3), \ y = +1 \qquad P_5 = (2, 1), \ y = -1
$$

These act as the **support vectors** — the points that lie exactly on the margin boundary.

A separating line roughly midway between the clusters is:

$$
x_1 + x_2 = 4
$$

which we can write as:

$$
1 \cdot x_1 + 1 \cdot x_2 - 4 = 0
$$

So $w = (1, 1)$ and $b = -4$.

## 6. Verifying the Margin Constraints

For $P_2 = (3,3)$, $y = +1$:

$$
y_i (w \cdot x_i + b) = (+1)\big[(1)(3) + (1)(3) - 4\big] = (+1)(2) = 2 \geq 1 \ 
$$

For $P_5 = (2,1)$, $y = -1$:

$$
y_i (w \cdot x_i + b) = (-1)\big[(1)(2) + (1)(1) - 4\big] = (-1)(-1) = 1 \geq 1 \ 
$$

Both support vectors sit exactly on their margin boundary (value $= 1$), which is what we want for a maximum-margin solution.

## 7. Margin Width

$$
\|w\| = \sqrt{w_1^2 + w_2^2} = \sqrt{1^2 + 1^2} = \sqrt{2}
$$

$$
\text{margin} = \frac{2}{\|w\|} = \frac{2}{\sqrt{2}} = \sqrt{2} \approx 1.41
$$

## 8. Classifying a New Point

For a new point $Q = (2.5, 3)$:

$$
f(Q) = w \cdot Q + b = (1)(2.5) + (1)(3) - 4 = 1.5
$$

Since $f(Q) > 0$:

$$
\hat{y}(Q) = \text{sign}(f(Q)) = +1
$$

## 9. General Decision Rule

$$
\hat{y}(x) = \text{sign}(w \cdot x + b)
$$

## 10. Soft-Margin SVM (Allowing Some Misclassification)

Real data is rarely perfectly separable, so we add slack variables $\xi_i \geq 0$:

$$
\min_{w, b, \xi} \ \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i
$$

subject to:

$$
y_i (w \cdot x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i
$$

Here, $C$ controls the trade-off between maximizing the margin and minimizing classification error — a large $C$ penalizes misclassified points heavily (narrower margin, fewer errors), while a small $C$ allows a wider margin with more tolerance for errors.

## 11. Kernel Trick (for Non-Linear Data)

When data isn't linearly separable, SVM maps it into a higher-dimensional space using a kernel function $K(x_i, x_j)$, avoiding explicit computation of the transformation:

$$
K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)
$$

Common kernels:

$$
\text{Linear:} \quad K(x_i, x_j) = x_i \cdot x_j
$$

$$
\text{Polynomial:} \quad K(x_i, x_j) = (x_i \cdot x_j + c)^d
$$

$$
\text{RBF (Gaussian):} \quad K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)
$$
