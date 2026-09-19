# Spectral Clustering

## 1. Intuition
Spectral clustering treats data as a **graph**: nodes = data points, edge
weights = similarity between points. Instead of clustering in the original
feature space (like K-Means), it looks at the graph's structure through
**eigenvectors of its Laplacian matrix**, embeds points into a new space
built from those eigenvectors, and clusters *there* (usually with K-Means).

This lets it find clusters that are **not convex** — e.g. two concentric
rings or interleaving spirals — because "closeness" is defined by graph
connectivity, not raw Euclidean distance.

## 2. Building the similarity graph
Common choices:
- **ε-neighborhood graph**: connect points within distance ε.
- **k-nearest neighbors graph**: connect each point to its k nearest
  neighbors.
- **Fully connected graph with a Gaussian/RBF kernel**:
  `w_ij = exp(-||x_i - x_j||^2 / (2*sigma^2))` — closer points get
  near-1 weight, far points get near-0 weight.

## 3. The graph Laplacian
Given adjacency/weight matrix `W` and degree matrix `D` (diagonal, `D_ii =
sum_j W_ij`):
- **Unnormalized Laplacian**: `L = D - W`
- **Normalized (symmetric) Laplacian**: `L_sym = D^(-1/2) L D^(-1/2)`
- **Random-walk Laplacian**: `L_rw = D^(-1) L`

Key theoretical fact: the **number of zero eigenvalues of L equals the
number of connected components** of the graph. In practice (noisy, fully
connected graphs), we look at the **smallest k eigenvalues/eigenvectors**
instead — they capture the "almost separate" components.

## 4. Algorithm
1. Build a similarity graph `W` from the data.
2. Compute the (normalized) graph Laplacian `L`.
3. Compute the `k` eigenvectors of `L` corresponding to the `k` smallest
   eigenvalues. Stack them as columns → an `n x k` embedding matrix `U`
   (each row is a new, low-dimensional representation of one original point).
4. (Optionally row-normalize `U`.)
5. Run **K-Means** on the rows of `U` to get the final `k` cluster labels.

This is exactly why it's called "spectral" — it uses the **spectrum**
(eigenvalues/eigenvectors) of the Laplacian matrix.

## 5. Why does this work? (intuition for the eigenvectors)
Think of the graph as an electrical network / random walk. The Laplacian's
eigenvectors describe smooth functions over the graph. Eigenvectors with the
smallest eigenvalues are the *smoothest* functions on the graph — they vary
slowly *within* a well-connected community and vary sharply *between*
communities. Clustering on these eigenvectors effectively finds a "cut" that
separates loosely-connected groups of nodes with minimum cost — closely
related to the **min-cut / normalized-cut (Ncut)** graph partitioning
problem, which spectral clustering approximately solves.

## 6. Complexity
- Building the full similarity matrix: `O(n^2)`.
- Eigen-decomposition: `O(n^3)` naive (or much faster with sparse
  eigensolvers when using a sparse k-NN graph, e.g. ARPACK) — this is the
  main scalability bottleneck.

## 7. Strengths / Weaknesses
**Strengths**:
- Finds non-convex clusters that K-Means/GMM cannot.
- Works purely from a similarity/affinity matrix — doesn't require raw
  coordinates, so it's natural for graph data (social networks, etc.)
- Strong theoretical grounding (graph partitioning / normalized cuts).

**Weaknesses**:
- Expensive for large `n` (eigen-decomposition cost).
- Still need to choose `k` (number of clusters) for the final K-Means step.
- Sensitive to the choice of similarity graph and its parameters (ε, k in
  k-NN, sigma in RBF kernel).

## 8. Basic → Pro
1. **Basic**: RBF-kernel fully-connected graph, unnormalized Laplacian, plot
   eigenvalues to see the "gap" that suggests `k`.
2. **Intermediate**: k-NN graph (sparser, faster, usually better in
   practice) + normalized Laplacian.
3. **Pro**: connect to **Normalized Cut** image segmentation, or use
   spectral embeddings as a general non-linear dimensionality reduction
   step (see also the `Dimensionality Reduction`/`Isomap` folders — all
   are "manifold learning" methods that use a similarity graph +
   eigen-decomposition).

## 9. Code in this folder
- `code/spectral_scratch.py` — Spectral clustering from scratch in NumPy:
  RBF similarity graph, normalized Laplacian, eigen-decomposition, and
  K-Means on the embedding (reuses the from-scratch K-Means idea).
- `code/spectral_sklearn_demo.py` — `sklearn.cluster.SpectralClustering` on
  concentric circles, compared against K-Means to show the non-convex
  advantage, plus an eigenvalue-gap plot.
