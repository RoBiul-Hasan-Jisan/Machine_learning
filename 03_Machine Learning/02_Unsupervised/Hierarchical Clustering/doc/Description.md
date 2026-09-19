# Hierarchical Clustering

## 1. Intuition
Instead of picking `k` up front, hierarchical clustering builds a **tree of
nested clusters** (a *dendrogram*). You can then "cut" the tree at any height
to get any number of clusters — the whole hierarchy is computed once.

Two directions:
- **Agglomerative (bottom-up)** — start with every point as its own cluster,
  repeatedly merge the two closest clusters. **This is the common one.**
- **Divisive (top-down)** — start with one cluster containing everything,
  repeatedly split it. Rarely used (more expensive, needs a splitting rule).

## 2. Agglomerative algorithm
1. Start: each of the `n` points is its own cluster.
2. Compute pairwise distances between all clusters.
3. Merge the two closest clusters into one.
4. Update distances between the new cluster and all others (using a
   **linkage** rule — see below).
5. Repeat 2–4 until only one cluster remains. Record every merge and the
   distance at which it happened → this record IS the dendrogram.

## 3. Linkage criteria (how do you measure distance between *clusters*, not points?)
| Linkage | Distance between clusters A, B | Tendency |
|---|---|---|
| **Single** | min distance between any pair (a∈A, b∈B) | Can produce long "chained" clusters |
| **Complete** | max distance between any pair | Compact, roughly equal-sized clusters |
| **Average** | mean of all pairwise distances | Balance between single & complete |
| **Ward** | increase in total within-cluster variance after merging | Compact, similar-sized clusters (like K-Means but hierarchical) |

## 4. Reading a dendrogram
- The x-axis lists individual points (leaves).
- The y-axis (height) is the distance at which two sub-clusters were merged.
- **Cutting** the dendrogram horizontally at height `h` gives you the clusters
  as they existed just before any merge above `h` — the number of vertical
  lines you cross = number of clusters.
- Long vertical lines with no nearby merges = a natural place to cut (a big
  "jump" in merge distance often signals a good number of clusters).

## 5. Complexity
- Naive agglomerative clustering: `O(n^3)` time, `O(n^2)` space (must store
  and repeatedly update an n×n distance matrix) — this is the main limitation
  vs K-Means's `O(nk)`. Optimized implementations (e.g. nearest-neighbor
  chains) bring this down to `O(n^2 log n)` or `O(n^2)`.

## 6. Strengths / Weaknesses
**Strengths**:
- No need to choose `k` in advance — decide after seeing the dendrogram.
- Deterministic (no random initialization issues like K-Means).
- Captures nested/multi-scale structure.
- Any distance metric can be used (not just Euclidean).

**Weaknesses**:
- Expensive: `O(n^2)`+ — doesn't scale to very large `n`.
- Greedy merges are never undone — an early bad merge can't be fixed later.
- Sensitive to linkage choice and noise/outliers (especially single linkage).

## 7. Basic → Pro
1. **Basic**: single/complete/average linkage, small dataset, plot dendrogram.
2. **Intermediate**: Ward linkage (usually the best default for
   Euclidean, roughly spherical-ish data).
3. **Pro**: combine with a **cophenetic correlation** check (how well does
   the dendrogram preserve original pairwise distances?), or use
   **BIRCH**/**connectivity constraints** (e.g. `sklearn`'s
   `kneighbors_graph`) to scale hierarchical ideas to bigger data by
   restricting which clusters are allowed to merge.

## 8. Code in this folder
- `code/hierarchical_scratch.py` — agglomerative clustering from scratch in
  NumPy (single, complete, average, and Ward linkage), building the merge
  history needed to draw a dendrogram, plus a `cut_tree(k)` helper.
- `code/hierarchical_sklearn_demo.py` — same idea using
  `scipy.cluster.hierarchy` (real dendrogram plotting) and
  `sklearn.cluster.AgglomerativeClustering`.
