# Isomap



##  Problem Framing

Linear methods such as PCA preserve Euclidean distance in the ambient
space $\mathbb{R}^d$. When data lies on a manifold that is curved in that
ambient space — globally curved but locally flat, such as the Swiss-roll
surface — Euclidean distance between two points can substantially
understate or overstate the distance actually traversed along the
manifold's surface. The standard illustration is geographic distance
between two cities: the physically meaningful distance is the shortest
path along the Earth's curved surface, not the straight line connecting
them through the interior of the planet. Isomap is constructed to
approximate this along-the-manifold ("geodesic") distance rather than the
straight-line distance that PCA and classical MDS preserve.

##  Algorithm Specification

**Input:** dataset $X \in \mathbb{R}^{n \times d}$, neighborhood size $k$
(or radius $\varepsilon$), target dimension $m$.

**Step 1: Neighborhood graph construction.** Connect each point to its $k$
nearest neighbors (or all neighbors within radius $\varepsilon$), with edge
weight equal to the Euclidean distance between the connected points. This
graph is a discrete, local approximation to the manifold: it encodes
Euclidean distance only between points assumed to be close enough that
their straight-line and along-manifold distances coincide.

**Step 2: Geodesic distance approximation.** For every pair of points,
compute the shortest path through the neighborhood graph, using Dijkstra's
algorithm from each node or the Floyd–Warshall algorithm for all pairs
simultaneously. This graph-shortest-path distance approximates the true
geodesic distance along the manifold: as the graph's local connectivity
increases in density relative to the manifold's curvature, the
piecewise-Euclidean path along graph edges converges to the manifold's
true geodesic. This approximation is valid only insofar as every edge in
the graph connects points that are genuinely close along the manifold
(see Section 3).

**Step 3: Classical multidimensional scaling (MDS).** Given the $n \times n$
matrix $D_G$ of pairwise geodesic-distance approximations from Step 2, find
an embedding in $\mathbb{R}^m$ whose pairwise Euclidean distances match
$D_G$ as closely as possible in squared error. This is solved by
double-centering the matrix of squared distances,
$B = -\tfrac{1}{2} J D_G^{(2)} J$ where $J = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$
and $D_G^{(2)}$ denotes the elementwise square of $D_G$, and taking the
top $m$ eigenvectors of $B$, scaled by the square roots of their
corresponding eigenvalues, as the embedding coordinates.

##  Sensitivity of the Neighborhood Graph

The validity of Step 2's approximation depends entirely on the
neighborhood graph constructed in Step 1, and the choice of $k$ (or
$\varepsilon$) introduces a direct trade-off:

- **$k$ too small.** The graph may become disconnected, in which case
  shortest-path distance is undefined between components; even when
  connected, an overly sparse graph produces a coarse, "steppy"
  approximation to the true geodesic, since the shortest path is
  constrained to follow relatively few available edges.
- **$k$ too large.** The graph begins to include shortcut edges that
  connect points which are close in raw Euclidean distance but far apart
  along the manifold — for instance, connecting two layers of a Swiss
  roll that happen to lie near each other in ambient space despite being
  distant along the unrolled surface. A single such shortcut edge can
  substantially corrupt the shortest-path computation for many pairs of
  points that route through it, since Dijkstra's and Floyd–Warshall's
  algorithms will exploit any available shorter path, including an
  erroneous one. This directly violates the assumption underlying Step 2
  and produces a distorted embedding.

The neighborhood-graph parameter is therefore not a minor tuning detail
but the primary determinant of whether Isomap's core assumption — that
graph-shortest-path distance approximates geodesic distance — holds at
all.

##  Complexity Analysis

| Step | Complexity |
|---|---|
| $k$-NN graph construction | $O(n^2)$ naive; $O(n \log n)$ with spatial indexing (see `Efficient Tree-based Algorithms/doc/Description.md`) |
| All-pairs shortest paths | $O(n^2 \log n)$ using Dijkstra from every node with a priority queue; $O(n^3)$ using Floyd–Warshall |
| Classical MDS (eigendecomposition) | $O(n^3)$ naive |

The eigendecomposition and all-pairs shortest-path steps make Isomap
substantially more expensive than PCA, and the overall $O(n^3)$-class cost
limits its practical scalability to large datasets.

##  Strengths and Limitations

**Strengths.**
- Correctly recovers the intrinsic structure of manifolds that are
  globally curved but locally flat, such as the Swiss roll and S-curve —
  a class of structure that PCA's linear projection cannot represent by
  construction.
- Preserves global geometric structure, since the objective in Step 3
  matches an embedding to the full matrix of pairwise geodesic distances,
  in contrast to methods such as t-SNE that are optimized primarily for
  local neighborhood fidelity and can distort global distances and shapes.

**Limitations.**
- The $O(n^3)$-class cost identified in Section 4 limits scalability to
  large datasets.
- The method is highly sensitive to the neighborhood-graph parameter, as
  established in Section 3: a single erroneous shortcut edge can degrade
  the geodesic-distance estimate for a large fraction of point pairs.
- The method assumes the manifold is densely and uniformly sampled, with
  no substantial holes; it does not handle noisy data or disconnected
  manifolds well, since both violate the assumption that local Euclidean
  distance between neighbors approximates local geodesic distance.

##  Relation to Other Methods

**Correspondence with PCA.** Classical MDS applied to a matrix of ordinary
Euclidean distances recovers exactly the PCA solution: both methods reduce
to an eigendecomposition of a matrix derived from pairwise inner products,
and in the Euclidean case these two derivations coincide. Isomap can
therefore be understood precisely as classical MDS applied to geodesic
distances rather than Euclidean distances, with Step 3 of Section 2 being
otherwise identical to PCA's underlying computation.

**Relation to Spectral Clustering.** Isomap shares its general
computational pattern — eigendecomposition of a matrix derived from a
similarity or distance structure — with Spectral Clustering's use of the
graph Laplacian (see `Spectral Clustering/doc/Description.md`). Both
methods reduce a graph-based representation of the data to a small number
of eigenvectors, though the specific matrix decomposed and the intended
downstream use (embedding versus partitioning) differ.

**Contrast with LLE.** Isomap and Locally Linear Embedding (LLE) address
the same manifold-learning problem with opposite notions of scope: Isomap
uses global geodesic distances computed over all pairs of points, while
LLE relies only on local linear reconstructions from each point's
immediate neighbors (see `Locally Linear Embedding/doc/Description.md`).
This distinction determines which structural properties of the manifold
each method is able to preserve.

##  Code Reference

| File | Contents |
|---|---|
| `code/isomap_scratch.py` | Isomap implemented from scratch in NumPy: $k$-NN graph construction, Dijkstra-based all-pairs shortest paths, and classical MDS via eigendecomposition |
| `code/isomap_sklearn_demo.py` | `sklearn.manifold.Isomap` applied to unroll a Swiss-roll dataset, with a comparison across different values of `n_neighbors` |