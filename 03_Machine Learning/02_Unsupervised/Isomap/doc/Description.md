# Isomap (Isometric Mapping)

## 1. Intuition
Isomap is a non-linear dimensionality reduction method that tries to
preserve **geodesic distances** — distances measured *along the manifold*
the data lives on — rather than straight-line (Euclidean) distances through
the surrounding space.

Classic analogy: distance between two cities should be measured by the
shortest path *on the Earth's curved surface* (or road network), not by a
straight line drilled through the planet. PCA/MDS use the straight-line
distance; Isomap approximates the "along-the-surface" distance.

## 2. Algorithm
1. **Build a neighborhood graph**: connect each point to its `k` nearest
   neighbors (or all neighbors within radius ε), with edge weights = the
   Euclidean distance between them.
2. **Compute geodesic distances**: for every pair of points, find the
   shortest path *through this graph* (Dijkstra's algorithm or
   Floyd-Warshall). This approximates the true geodesic distance along the
   manifold — the more neighbors are connected, the better this
   approximation, as long as the graph doesn't create "shortcuts" that jump
   across the manifold's folds.
3. **Classical Multidimensional Scaling (MDS)**: given the full n×n matrix
   of (squared) geodesic distances, find a low-dimensional embedding whose
   pairwise Euclidean distances match those geodesic distances as closely
   as possible. This is solved via an eigen-decomposition of a "double
   centered" version of the squared distance matrix (very similar
   machinery to PCA — in fact, classical MDS on a Euclidean distance matrix
   recovers exactly PCA's solution).

## 3. Why the neighborhood graph matters so much
- **Too few neighbors (k too small)**: graph may become disconnected, or
  shortest paths become a poor approximation of the true geodesic (too
  "steppy").
- **Too many neighbors (k too large)**: the graph starts including
  "shortcut" edges that cut across folds of the manifold (e.g. connecting
  two points that are close in raw Euclidean space but far apart along the
  manifold, like two layers of the Swiss roll touching) — this breaks the
  whole method's assumption and gives a distorted embedding.

## 4. Complexity
- Building k-NN graph: `O(n^2)` naive, `O(n log n)` with spatial indexing.
- All-pairs shortest paths: `O(n^2 log n)` with Dijkstra from every node
  (using a priority queue), or `O(n^3)` with Floyd-Warshall.
- Eigen-decomposition step (like PCA/MDS): `O(n^3)` naive.
- Overall this makes Isomap noticeably more expensive than PCA and doesn't
  scale to very large `n`.

## 5. Strengths / Weaknesses
**Strengths**:
- Correctly "unrolls" manifolds that are globally curved but locally
  flat (Swiss roll, S-curve) — something linear PCA fundamentally cannot
  do.
- Preserves *global* geometric structure well (unlike t-SNE, which
  prioritizes local structure and can distort global distances/shapes).

**Weaknesses**:
- Expensive (`O(n^3)`-ish) — doesn't scale to large datasets.
- Sensitive to the neighborhood-graph parameter (k or ε) — a single "bad"
  shortcut edge can wreck the whole geodesic-distance estimate.
- Assumes the manifold is well-sampled (dense enough, no holes) and doesn't
  handle noisy data or disconnected manifolds well.

## 6. Relation to other methods
- Uses the same "eigen-decompose a similarity/distance-derived matrix"
  pattern as **PCA** (classical MDS) and **Spectral Clustering** (graph
  Laplacian) — Isomap can be seen as "MDS on geodesic distances instead of
  Euclidean distances."
- Contrast with **LLE**: Isomap uses *global* geodesic distances (all
  pairs); LLE only uses *local* linear reconstructions from each point's
  immediate neighbors (see `Locally Linear Embedding/doc/README.md`).

## 7. Basic → Pro
1. **Basic**: build k-NN graph, run Dijkstra for all-pairs shortest paths,
   classical MDS on the result — see it unroll a Swiss roll.
2. **Intermediate**: tune `k` and observe what happens with too few / too
   many neighbors (disconnection vs. shortcut distortion).
3. **Pro**: compare against **t-SNE/LLE** on the same manifold to build
   intuition for global-structure-preserving vs. local-structure-
   preserving embeddings.

## 8. Code in this folder
- `code/isomap_scratch.py` — Isomap from scratch in NumPy: k-NN graph
  construction, Dijkstra-based all-pairs shortest paths, and classical MDS
  via eigen-decomposition.
- `code/isomap_sklearn_demo.py` — `sklearn.manifold.Isomap` unrolling a
  Swiss roll, with a comparison across different `n_neighbors` values.
