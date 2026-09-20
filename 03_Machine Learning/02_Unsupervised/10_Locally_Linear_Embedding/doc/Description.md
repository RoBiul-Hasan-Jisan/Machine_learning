# Locally Linear Embedding (LLE)



##  Problem Framing

LLE is built on the assumption that although a dataset as a whole may lie
on a curved, non-linear manifold, a sufficiently small neighborhood around
any point is approximately flat — analogous to how a small patch of the
Earth's curved surface appears locally planar. Under this assumption, LLE
constructs an embedding in two stages: first, it represents each point as
a linear combination of its nearest neighbors in the original space;
second, it finds a low-dimensional configuration of points in which the
same neighbors, weighted identically, reconstruct each point equally well.

This is a fundamentally local objective. In contrast to Isomap, which
requires geodesic distance between all pairs of points to be preserved
(see `Isomap/doc/Description.md`), LLE's objective at every stage
references only a point and its immediate neighbor set.

##  Algorithm Specification

**Input:** dataset $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$, neighbor
count $k$, target dimension $m$.

**Step 1: Neighbor identification.** For each point $x_i$, identify its
$k$ nearest neighbors under a chosen distance metric.

**Step 2: Reconstruction weight computation.** Solve for weights $W_{ij}$,
nonzero only for $j$ in $x_i$'s neighbor set, that minimize the local
reconstruction error:

$$\min_{W} \sum_{i} \left\lVert x_i - \sum_j W_{ij} x_j \right\rVert^2
\quad \text{subject to} \quad \sum_j W_{ij} = 1 \ \text{for all } i$$

Because this problem decomposes over $i$, each point's weights are solved
independently as a local least-squares problem, reducing to a small linear
system built from the local covariance of the differences between $x_i$
and its neighbors.

**Step 3: Embedding via eigendecomposition.** Find coordinates
$Y = \{y_1, \dots, y_n\} \subset \mathbb{R}^m$ that preserve the same
reconstruction weights found in Step 2:

$$\min_{Y} \sum_i \left\lVert y_i - \sum_j W_{ij} y_j \right\rVert^2$$

This is solved by forming $M = (I - W)^\top (I - W)$ and computing its
eigenvectors. The eigenvectors corresponding to the smallest non-zero
eigenvalues are taken as the embedding coordinates; the eigenvector with
eigenvalue exactly (or approximately) zero corresponds to the trivial
solution of placing every point at the same location and is discarded.

##  Necessity of the Weight-Normalization Constraint

The constraint $\sum_j W_{ij} = 1$ in Step 2 is not an arbitrary
normalization; it is required to avoid a degenerate optimum and to make
the weights meaningful in the reduced-dimension space.

**Preventing degeneracy.** Without this constraint, the unconstrained
minimization of $\sum_i \lVert x_i - \sum_j W_{ij} x_j \rVert^2$ admits the
trivial solution $W_{ij} = 0$ for all $i, j$, which achieves zero
reconstruction error only if $x_i = 0$; more generally, the unconstrained
problem is ill-posed with respect to scale, since weights can be scaled
arbitrarily without a fixed reference.

**Invariance and portability.** More importantly, requiring the weights of
each point's reconstruction to sum to 1 makes the reconstruction an affine
combination of the neighbors, which is invariant to translation, rotation,
and uniform scaling of the local neighborhood. This invariance is exactly
what permits the same weights $W_{ij}$, computed in the original
$d$-dimensional space, to be reused directly as the target relationship in
Step 3's $m$-dimensional embedding: because the weights describe a
coordinate-independent affine relationship among neighbors, they remain
valid as a description of local geometry regardless of the space in which
that geometry is expressed.

##  Complexity Analysis

| Step | Complexity |
|---|---|
| Neighbor search | $O(n \log n)$ with spatial indexing (see `Efficient Tree-based Algorithms/doc/Description.md`) |
| Weight computation | $O(n \cdot k^3)$, from solving a $k \times k$ linear system per point |
| Embedding (sparse eigendecomposition, bottom eigenvectors only) | Typically $O(d \cdot n^2)$ or better with sparse eigensolvers |

## Strengths and Limitations

**Strengths.**
- Preserves local neighborhood geometry with high fidelity, following
  directly from the reconstruction objective in Step 2 being solved
  independently and exactly for each point's local neighborhood.
- Conceptually simple: the entire method is a weighted local linear
  reconstruction problem followed by a single eigendecomposition.
- Does not require estimating global pairwise geodesic distances, as
  Isomap does; because every computation is local, LLE can be more robust
  to manifolds containing noise or small holes, where a global
  shortest-path computation (as in Isomap) would be more strongly affected
  by a single problematic region.

**Limitations.**
- Global structure is not preserved: because the embedding objective in
  Step 3 references only local neighbor relationships, distinct regions of
  the manifold can be placed in an arbitrary relative arrangement in the
  embedding, with no mechanism to correct this at a global scale.
- Highly sensitive to the choice of $k$ (Section 7): too small a value
  yields noisy, poorly-conditioned local weight estimates from
  insufficient context; too large a value violates the local-linearity
  assumption that Step 1 depends on.
- Requires reasonably uniform, dense sampling of the manifold, since each
  point's local linear patch (Step 2) must be well-approximated by its
  $k$ nearest neighbors; sparse or non-uniform sampling degrades the
  reconstruction weights accordingly.

##  Relation to Other Methods

LLE shares the general computational pattern of constructing a graph over
neighboring points and extracting an embedding via eigendecomposition,
also used by Isomap and Spectral Clustering. The methods differ in what
that graph encodes: Isomap's graph encodes pairwise distances, from which
geodesic distance is approximated (see `Isomap/doc/Description.md`,
Section 2); Spectral Clustering's graph encodes pairwise similarity, from
which a graph Laplacian is derived (see
`Spectral Clustering/doc/Description.md`); LLE's graph instead encodes
local linear reconstruction weights, which is why LLE preserves local
structure without reference to any global distance quantity.

**Numerically stable variants.** Modified LLE, Hessian LLE, and Local
Tangent Space Alignment (LTSA) address numerical stability issues that can
arise in the standard LLE weight-solving step, particularly when the
number of neighbors $k$ exceeds the ambient dimensionality $d$, which
makes the local covariance matrix in Step 2 singular or ill-conditioned.
These variants are available via
`sklearn.manifold.LocallyLinearEmbedding(method=...)`.

##  Empirical Sensitivity to Neighbor Count

On the standard Swiss-roll benchmark, correlating the first LLE embedding
dimension with the manifold's true intrinsic coordinate demonstrates the
practical severity of the $k$-sensitivity described in Section 5: a
well-chosen value of $k$ (e.g., $k = 8$) can recover the manifold's
intrinsic structure with correlation near 0.99, while a value of $k$ only
slightly larger or smaller can reduce that correlation to near zero. This
is a direct empirical consequence of the trade-off in Section 5 — the
local-linearity assumption in Step 1 holds only within a narrow range of
$k$ for a given manifold — and it motivates treating $k$ as a parameter
that must be swept and validated for each dataset rather than fixed at a
default value.


##  Code Reference

| File | Contents |
|---|---|
| `code/lle_scratch.py` | LLE implemented from scratch in NumPy: $k$-NN search, local weight solving via least squares under the sum-to-one constraint, and the eigendecomposition-based embedding step |
| `code/lle_sklearn_demo.py` | `sklearn.manifold.LocallyLinearEmbedding` applied to the Swiss-roll and S-curve datasets, comparing standard LLE against modified LLE |