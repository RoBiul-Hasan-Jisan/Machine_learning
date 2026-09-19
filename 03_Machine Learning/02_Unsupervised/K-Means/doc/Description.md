# K-Means Clustering

## 1. Intuition
K-Means partitions data into **k** clusters by iteratively:
1. Assigning each point to the nearest of `k` centroids.
2. Moving each centroid to the mean of the points assigned to it.

It "melts" data into k round blobs — it implicitly assumes clusters are
roughly spherical and similar in size (because it uses Euclidean distance to a
single center point).

## 2. Formal objective
K-Means minimizes the **within-cluster sum of squares (WCSS / inertia)**:

```
J = Σ_{i=1..k} Σ_{x in C_i} ||x - μ_i||²
```

where `μ_i` is the centroid (mean) of cluster `C_i`. This is NP-hard to solve
exactly, so we use an iterative heuristic (Lloyd's algorithm) that converges to
a local optimum.

## 3. Algorithm (Lloyd's algorithm)
1. Choose `k`. Initialize `k` centroids (randomly, or via **k-means++** for a
   smarter spread-out start).
2. **Assignment step**: for each point, assign it to the nearest centroid
   (by Euclidean distance).
3. **Update step**: recompute each centroid as the mean of points assigned to it.
4. Repeat 2–3 until assignments stop changing (or max iterations / tolerance
   reached).

This is exactly an instance of the **EM (Expectation-Maximization)** pattern:
assignment = E-step, centroid update = M-step.

## 4. Choosing k
- **Elbow method**: plot inertia vs k, look for the "elbow" where returns
  diminish.
- **Silhouette score**: pick k that maximizes average silhouette.
- Domain knowledge is often the best guide in practice.

## 5. k-means++ initialization (why it matters)
Random initialization can lead to poor local optima or empty clusters.
k-means++ picks initial centroids that are far apart from each other
(probability of picking a point ∝ squared distance to the nearest already
chosen centroid), which empirically gives faster, better convergence.

## 6. Complexity
- Time: `O(n * k * d * iterations)` (n points, d dimensions) — very scalable.
- Space: `O(n * d + k * d)`.

## 7. Strengths / Weaknesses
**Strengths**: simple, fast, scales to large n, easy to understand.
**Weaknesses**:
- Must choose `k` in advance.
- Assumes convex, similarly-sized, isotropic clusters (fails on moons, rings,
  elongated clusters — see `01_Clustering/code/clustering_overview.py`).
- Sensitive to initialization (mitigated by k-means++ / multiple restarts).
- Sensitive to feature scale → always standardize first.
- Sensitive to outliers (a single far point drags a centroid).

## 8. Variants (basic → pro)
- **Basic**: Standard Lloyd's algorithm with random init.
- **Better**: k-means++ initialization (default in scikit-learn).
- **Faster on huge data**: Mini-Batch K-Means (updates centroids using small
  random batches instead of the full dataset each iteration).
- **Robust**: K-Medoids / PAM (uses actual data points as centers and
  arbitrary distance metrics, more robust to outliers, more expensive).
- **Soft clustering**: Gaussian Mixture Models (probabilistic generalization —
  each point gets a probability of belonging to each cluster instead of a
  hard assignment).

## 9. Code in this folder
- `code/kmeans_scratch.py` — K-Means implemented from scratch in NumPy
  (random init + k-means++ init), including inertia tracking and a simple
  `elbow()` helper.
- `code/kmeans_sklearn_demo.py` — same problem solved with
  `sklearn.cluster.KMeans` / `MiniBatchKMeans`, with elbow + silhouette plots,
  to compare against the scratch implementation and validate correctness.
