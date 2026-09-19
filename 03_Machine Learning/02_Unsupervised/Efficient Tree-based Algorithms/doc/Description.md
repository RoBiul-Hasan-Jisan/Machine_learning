# Efficient Tree-based Algorithms (for Unsupervised Learning)

## 1. Why "tree-based" shows up all over unsupervised learning
Several unsupervised methods rely on **tree data structures** purely for
*computational efficiency* — the tree isn't a model of the data's meaning
(unlike a decision tree), it's an indexing structure that turns an
expensive `O(n^2)` brute-force operation (nearest-neighbor search, range
queries, all-pairs distance computation) into something closer to
`O(n log n)`. This folder is a roundup of where those trees appear
elsewhere in this repo, plus a from-scratch look at the core structures
themselves.

## 2. KD-Trees
A **KD-tree** ("k-dimensional tree") recursively partitions space by
splitting on one coordinate axis at a time (cycling through dimensions,
splitting each region at the median of points along that axis).
- **Build**: `O(n log n)`.
- **Nearest-neighbor query**: `O(log n)` average case in low dimensions —
  but degrades toward `O(n)` as dimensionality grows (the "curse of
  dimensionality" hits geometric partitioning hard past roughly 10-20
  dimensions, because most points end up equally far from a query point
  and pruning branches stops helping).
- **Used for**: accelerating DBSCAN's/Mean-Shift's neighborhood queries,
  Isomap/LLE's k-NN graph construction (`sklearn.neighbors.NearestNeighbors`
  uses a KD-tree or Ball-tree under the hood for exactly this reason).

## 3. Ball-Trees
A **Ball-tree** partitions data into nested hyperspheres ("balls") instead
of axis-aligned boxes. More robust than KD-trees in **higher dimensions**
and with **non-Euclidean distance metrics** (it only needs a valid
distance metric to determine ball membership — doesn't require
coordinate-axis splits the way a KD-tree does).
- **Build**: `O(n log n)`.
- **Query**: similar average-case behavior to KD-trees, but generally
  degrades more gracefully in moderate-to-high dimensions.
- **Used for**: same purpose as KD-trees (nearest-neighbor acceleration)
  — `sklearn` automatically picks between KD-tree and Ball-tree based on
  data dimensionality and the requested metric.

## 4. FP-tree (Frequent Pattern tree)
Already covered in depth in `FP-Growth Algorithm/doc/Description.md` — a
**prefix tree** that compresses a transaction database so that frequent
itemset mining needs only two database scans instead of one per itemset
size (as in Apriori). This is the clearest example in this repo of a tree
used purely as a *compressed computational structure*, not a
distance-index.

## 5. Dendrograms (hierarchical clustering trees)
Covered in `Hierarchical Clustering/doc/Description.md` — technically the
*output* of agglomerative clustering rather than an internal
efficiency-accelerating structure, but still worth remembering as "a tree
that represents the mining/clustering result" alongside FP-Growth's "a
tree used purely to make the mining process itself efficient."

## 6. Why this matters in practice
Nearly every distance-based unsupervised algorithm in this repo
(K-Means's assignment step, DBSCAN's/Mean-Shift's neighborhood queries,
Isomap's/LLE's/Spectral Clustering's k-NN graph construction) has an inner
loop of "for each point, find its nearest neighbors" — naive computation
is `O(n^2)`, which becomes the real-world bottleneck long before any of
the algorithms' own "big-O" analysis in their respective docs. Swapping a
brute-force distance loop for a KD-tree/Ball-tree-backed neighbor search
(as `scikit-learn` does automatically) is often the single biggest
practical speedup available, with zero change to the algorithm's actual
output.

## 7. Basic → Pro
1. **Basic**: build a KD-tree from scratch, understand the recursive
   median-split construction, do a naive brute-force nearest-neighbor
   search alongside it to check correctness.
2. **Intermediate**: implement the KD-tree's actual nearest-neighbor query
   with branch-and-bound pruning (skip a subtree once you can prove it
   can't contain anything closer than the current best).
3. **Pro**: compare KD-tree vs Ball-tree vs brute-force query time as
   dimensionality increases, to see the curse-of-dimensionality crossover
   point where tree-based indexing stops helping.

## 8. Code in this folder
- `code/kdtree_scratch.py` — a KD-tree from scratch in Python: recursive
  construction and a branch-and-bound nearest-neighbor query, validated
  against brute-force search for correctness.
- `code/tree_efficiency_demo.py` — benchmarks brute-force vs
  `sklearn.neighbors.NearestNeighbors` (KD-tree, Ball-tree, and brute
  algorithms) as `n` and dimensionality increase, showing where tree-based
  indexing helps and where the curse of dimensionality erases the
  advantage.
