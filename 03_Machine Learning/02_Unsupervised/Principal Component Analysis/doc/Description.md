# Principal Component Analysis (PCA)

## 1. Intuition
PCA finds new axes (**principal components**) to describe your data — axes
that point in the directions of **maximum variance**. The first principal
component is the direction along which the data varies the most; the second
is the direction of next-most variance, *orthogonal* to the first; and so on.
Keeping only the top few components gives a lower-dimensional representation
that preserves as much of the data's spread (information) as possible.

## 2. The math
Given centered data `X` (n samples × d features, mean-subtracted):

1. **Covariance matrix**: `C = (1/(n-1)) * X^T X` (a d×d matrix — how each
   pair of features co-varies).
2. **Eigen-decomposition**: find eigenvectors `v_1, v_2, ..., v_d` and
   eigenvalues `λ_1 ≥ λ_2 ≥ ... ≥ λ_d` of `C`.
   - Each eigenvector `v_i` is a principal component (a direction).
   - Each eigenvalue `λ_i` is the variance of the data *along* that
     direction.
3. **Projection**: to reduce to `k` dimensions, project the data onto the
   top-`k` eigenvectors: `Z = X @ V_k` where `V_k` is the d×k matrix of the
   top-k eigenvectors.
4. **Reconstruction** (approximate inverse): `X_approx = Z @ V_k^T`.

Equivalently (and more numerically stable in practice), PCA can be computed
via **Singular Value Decomposition (SVD)**: `X = U Σ V^T`. The columns of
`V` are exactly the principal components, and `Σ`'s singular values relate
to the eigenvalues by `λ_i = σ_i^2 / (n-1)`. `scikit-learn` uses SVD
internally rather than eigen-decomposing the covariance matrix directly.

## 3. Explained variance
`explained_variance_ratio_i = λ_i / Σ_j λ_j` — the fraction of total
variance captured by component `i`. Plotting the **cumulative** explained
variance vs. number of components ("scree plot") is the standard way to
choose `k` (e.g. "keep enough components to explain 95% of variance").

## 4. Why center (and usually scale) the data first?
- **Centering** (subtracting the mean) is *required* — PCA is defined in
  terms of variance around the mean; skipping this corrupts the result.
- **Scaling** (standardizing to unit variance) is usually recommended when
  features are in different units — otherwise a feature measured in, say,
  kilometers will dominate the covariance matrix purely due to scale, not
  because it's actually more informative.

## 5. Algorithm (via SVD, as used in practice)
1. Center `X` (subtract column means).
2. Compute `U, Σ, V^T = SVD(X)`.
3. Principal components = columns of `V` (equivalently rows of `V^T`).
4. Projected data (scores) = `X @ V_k` (or equivalently `U_k @ Σ_k`).

## 6. Complexity
- Covariance + eigen-decomposition: `O(n*d^2 + d^3)`.
- SVD directly on X: `O(min(n*d^2, n^2*d))` — SVD-based PCA scales better
  when `d >> n` or `n >> d` (only need the top-k singular vectors, computed
  via truncated/randomized SVD for large data).

## 7. Strengths / Weaknesses
**Strengths**:
- Simple, deterministic (no randomness — unlike K-Means/t-SNE), fast, and
  has a proper inverse transform (great for compression/denoising).
- Components are mathematically interpretable (variance-maximizing
  directions) and always orthogonal to each other.

**Weaknesses**:
- Purely **linear** — can't unroll non-linear manifolds (Swiss roll,
  moons). See `Isomap` / `Locally Linear Embedding` / `t-SNE` for that.
- Maximizing *variance* doesn't always mean maximizing *class
  separability* or *usefulness for a downstream task* — components can
  mix signal and structure that isn't relevant to your actual question.
- Sensitive to feature scaling.

## 8. Basic → Pro
1. **Basic**: covariance matrix + eigen-decomposition, 2D projection,
   scree plot.
2. **Intermediate**: SVD-based PCA (more numerically stable, what
   `sklearn.decomposition.PCA` actually does), whitening (normalize
   projected components to unit variance).
3. **Pro**: **Kernel PCA** (apply the "kernel trick" to do PCA implicitly
   in a non-linear feature space — a bridge from linear PCA to non-linear
   manifold learning); **Incremental PCA** (fit without loading the whole
   dataset into memory, useful for streaming/huge data).

## 9. Code in this folder
- `code/pca_scratch.py` — PCA from scratch in NumPy via both the
  covariance-eigendecomposition route and the SVD route (and shows they
  agree), plus `inverse_transform` for reconstruction error.
- `code/pca_sklearn_demo.py` — `sklearn.decomposition.PCA` on the digits
  dataset: scree plot, 2D visualization, and image reconstruction from a
  reduced number of components (visualizing the compression/denoising
  trade-off).
