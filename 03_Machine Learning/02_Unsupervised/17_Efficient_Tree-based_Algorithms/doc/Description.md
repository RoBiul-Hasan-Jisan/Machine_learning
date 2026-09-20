# Efficient Tree-Based Algorithms


## Problem Framing: The Common Bottleneck

Several unsupervised methods in this repository rely on tree data
structures, but not for a uniform reason. It is useful to distinguish two
structurally different roles a tree can play:

- **Indexing structure for distance queries.** The tree encodes no
  information about the meaning of the data; it exists solely to convert
  an $O(n^2)$ brute-force nearest-neighbor or range query into a
  sublinear-per-query operation. KD-trees and Ball-trees (Sections 2–3)
  serve this role.
- **Compressed representation for exact computation.** The tree encodes
  the dataset itself in a form that permits an algorithm to avoid repeated
  full passes over the raw data. The FP-tree (Section 4) serves this role
  and is not a distance index at all.

A third category — trees that are the *output* of a method rather than an
internal mechanism — is addressed in Section 5 for completeness, since it
is easily confused with the first two categories.

##  KD-Trees

A KD-tree ("$k$-dimensional tree") recursively partitions $\mathbb{R}^d$ by
splitting on one coordinate axis at a time, cycling through dimensions,
and splitting each region at the median of the points along the current
axis.

**Complexity.**

| Operation | Complexity |
|---|---|
| Construction | $O(n \log n)$ |
| Nearest-neighbor query (low dimensions) | $O(\log n)$, average case |
| Nearest-neighbor query (high dimensions) | Degrades toward $O(n)$ |

**Why high-dimensional query performance degrades.** KD-tree query
efficiency relies on the ability to prune subtrees — to prove, without
searching a subtree, that it cannot contain a point closer than the
current best candidate. This pruning argument depends on there being
meaningful variation in distance across the space. As dimensionality
increases past approximately 10–20 dimensions, the curse of
dimensionality (see `Dimensionality Reduction/doc/Description.md`) causes
distances to concentrate, so that most points become nearly equidistant
from a given query point. Under this condition, the geometric argument for
pruning a subtree stops holding, and the query degrades toward exhaustive
search.

**Usage in this repository.** KD-trees accelerate the neighborhood queries
required by DBSCAN and Mean-Shift, and the $k$-nearest-neighbor graph
construction required by Isomap and LLE.
`sklearn.neighbors.NearestNeighbors` uses a KD-tree (or Ball-tree) for
exactly this purpose.

##  Ball-Trees

A Ball-tree partitions the data into nested hyperspheres ("balls") rather
than axis-aligned boxes.

**Distinguishing property.** A Ball-tree requires only a valid distance
metric to determine ball membership, whereas a KD-tree's construction
depends on splitting along coordinate axes. This makes Ball-trees
applicable to non-Euclidean metrics for which axis-aligned splits are not
well defined, and generally more robust in moderate-to-high dimensions
than KD-trees, since ball membership does not depend on the coordinate
structure that degrades under the curse of dimensionality.

**Complexity.** Construction is $O(n \log n)$, matching KD-trees. Query
performance is comparable to KD-trees on average but degrades more
gracefully as dimensionality increases.

**Usage in this repository.** Ball-trees serve the identical role as
KD-trees — accelerating nearest-neighbor queries — and
`scikit-learn` selects automatically between the two based on data
dimensionality and the requested distance metric.

##  FP-Trees: A Structurally Different Use of Trees

The FP-tree (Frequent Pattern tree), detailed in
`FP-Growth Algorithm/doc/Description.md`, is a prefix tree that compresses
an entire transaction database into a single in-memory structure, such
that frequent itemset mining requires exactly two database scans in total,
independent of the length of the itemsets discovered — in contrast to
Apriori, which requires one scan per itemset length (see
`Apriori Algorithm/doc/Description.md`, Section 6).

This is a categorically different use of a tree from Sections 2–3: the
FP-tree encodes no distance relationships and answers no nearest-neighbor
queries. It is a compressed data representation, not a spatial index. It
is included here as the clearest example in this repository of a tree
used purely to make an exact computation efficient, distinct from geometric
indexing.

##  Dendrograms: Output Structure, Not Index Structure

The dendrogram produced by agglomerative hierarchical clustering (see
`Hierarchical Clustering/doc/Description.md`) is a tree, but it is the
*output* of the clustering procedure — a representation of the nested
merge structure discovered by the algorithm — rather than an internal
structure used to accelerate computation. It is noted here only to
prevent conflation with the accelerating structures of  a
dendrogram represents a clustering result, whereas a KD-tree, Ball-tree,
or FP-tree exists to make some other computation faster or exact.

##  Practical Significance Across This Repository

Nearly every distance-based unsupervised method in this repository
contains an inner loop of the form "for each point, find its nearest
neighbors": K-Means's assignment step, DBSCAN's and Mean-Shift's
neighborhood queries, and the $k$-nearest-neighbor graph construction
underlying Isomap, LLE, and Spectral Clustering. Computed naively, this
operation is $O(n^2)$, and in practice this cost dominates the runtime of
these methods well before the algorithm-specific asymptotic behavior
documented in each method's own theory file becomes the limiting factor.

Replacing a brute-force distance loop with a KD-tree- or Ball-tree-backed
neighbor search — as `scikit-learn` performs automatically — is therefore
frequently the single largest available performance improvement for these
methods, and importantly, it changes no aspect of the algorithm's output:
it is a computational acceleration of an existing exact operation, not an
approximation of it.


##  Code Reference

| File | Contents |
|---|---|
| `code/kdtree_scratch.py` | KD-tree implemented from scratch in Python: recursive construction and a branch-and-bound nearest-neighbor query, validated against brute-force search for correctness |
| `code/tree_efficiency_demo.py` | Benchmark of brute-force search against `sklearn.neighbors.NearestNeighbors` (KD-tree, Ball-tree, and brute-force algorithms) as $n$ and dimensionality increase, illustrating where tree-based indexing helps and where the curse of dimensionality erases the advantage |