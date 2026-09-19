# Locally Linear Embedding (LLE)

## 1. Intuition
LLE assumes that, even though the data as a whole lies on a curved,
non-linear manifold, each small **local neighborhood** looks approximately
**linear/flat** (like how a small patch of the Earth's curved surface looks
flat). It:
1. Reconstructs each point as a linear (weighted) combination of its
   nearest neighbors.
2. Finds a low-dimensional embedding where those same neighbors, with the
   same weights, still reconstruct each point just as well.

Unlike Isomap (which cares about *global* geodesic distances between all
pairs), LLE only ever cares about *local* relationships — each point and
its immediate neighbors.

## 2. Algorithm
1. **Find neighbors**: for each point `x_i`, find its `k` nearest neighbors.
2. **Compute reconstruction weights**: solve for weights `W_ij` (only
   nonzero for `j` in `x_i`'s neighbor set) that minimize the local
   reconstruction error:
   ```
   min_W  Σ_i || x_i - Σ_j W_ij * x_j ||^2       subject to Σ_j W_ij = 1
   ```
   The constraint `Σ_j W_ij = 1` makes the weights invariant to
   translation, rotation, and scaling of the neighborhood — this is exactly
   what makes the weights "portable" to a new, lower-dimensional space in
   the next step. This is solved per-point via a local least-squares problem
   (a small linear system based on the local covariance of neighbor
   differences).
3. **Find the low-dimensional embedding**: find coordinates `Y` (in `k`
   dimensions) that best preserve those same reconstruction weights:
   ```
   min_Y  Σ_i || y_i - Σ_j W_ij * y_j ||^2
   ```
   This is solved by an eigen-decomposition of the matrix
   `M = (I - W)^T (I - W)` — take the eigenvectors corresponding to the
   smallest non-zero eigenvalues (the very smallest, ~0, eigenvalue
   corresponds to a trivial all-points-together solution and is discarded).

## 3. Why the "sum to 1" constraint matters
Without it, the optimization could trivially set all weights to 0 (or any
degenerate solution). The constraint forces every point to be *exactly*
reconstructed as a genuine affine (weighted-average) combination of its
neighbors — which is the property LLE explicitly preserves when moving to
the low-dimensional space.

## 4. Complexity
- Neighbor search: `O(n log n)` with spatial indexing.
- Weight computation: solving a small (`k×k`) linear system per point →
  `O(n * k^3)`.
- Embedding step (eigen-decomposition of a sparse `n×n` matrix, keeping
  only the bottom few eigenvectors): typically `O(d * n^2)` or better with
  sparse eigensolvers.

## 5. Strengths / Weaknesses
**Strengths**:
- Preserves *local* neighborhood geometry very well; conceptually simple
  (weighted local linear reconstruction).
- Doesn't require estimating global pairwise geodesic distances like
  Isomap — potentially more robust to a manifold with some noise/holes,
  since it's purely local.

**Weaknesses**:
- Doesn't preserve global structure/distances well — clusters can end up
  in an arbitrary relative arrangement to each other in the embedding.
- Sensitive to the choice of `k` (too small → noisy weights, not enough
  context; too large → local-linearity assumption breaks down).
- Struggles when the data isn't uniformly/densely sampled on the manifold
  (needs enough neighbors to reliably fit the local linear patch).

## 6. Relation to other methods
- Same overall "graph of neighbors -> eigen-decomposition" pattern as
  Isomap and Spectral Clustering, but LLE's graph encodes *local linear
  reconstruction weights* rather than distances.
- **Modified LLE**, **Hessian LLE**, and **LTSA (Local Tangent Space
  Alignment)** are more numerically robust/stable variants available in
  `sklearn.manifold.LocallyLinearEmbedding(method=...)`.

## 6b. A concrete illustration of neighbor-count sensitivity
On the classic Swiss-roll benchmark, correlating the first LLE embedding
dimension with the manifold's true intrinsic coordinate shows just how
sharply results swing with `k`: a good `k` (e.g. 8) can recover the
manifold almost perfectly (correlation ~0.99), while `k` even slightly
larger or smaller can drop that correlation close to 0 — a very concrete
demonstration of point 5's warning above. Always sweep `k` and sanity-check
the result rather than trusting a single default value.

## 7. Basic → Pro
1. **Basic**: standard LLE, k-NN + local weight solving + eigen-embedding.
2. **Intermediate**: tune `k`, observe local-linearity breakdown for very
   curved regions or too-large `k`.
3. **Pro**: Modified LLE / Hessian LLE for more numerically stable
   embeddings on noisier real-world data.

## 8. Code in this folder
- `code/lle_scratch.py` — LLE from scratch in NumPy: k-NN search, local
  weight solving (via least squares with the sum-to-one constraint), and
  the eigen-decomposition embedding step.
- `code/lle_sklearn_demo.py` — `sklearn.manifold.LocallyLinearEmbedding` on
  a Swiss roll and the S-curve dataset, with standard vs. modified LLE
  compared.
