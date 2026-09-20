# t-SNE (t-Distributed Stochastic Neighbor Embedding)



##  Problem Framing

t-SNE is designed for a single, specific objective: producing
two- or three-dimensional visualizations in which points that are close
in the original high-dimensional space remain close in the visualization.
It makes no attempt to preserve distances between points that are far
apart, nor does it aim to preserve global structure faithfully, in
contrast to PCA (which preserves global variance) or Isomap (which
preserves global geodesic distance; see `Isomap/doc/Description.md`. This narrower objective is what makes t-SNE the standard tool
for visualizing high-dimensional embeddings — word embeddings, image
embeddings, single-cell genomics data — where the goal is a legible
picture of local cluster structure rather than a quantitatively faithful
low-dimensional coordinate system.

##  High-Dimensional Affinities

For every ordered pair of points $(i, j)$, define a conditional
probability that point $i$ would select point $j$ as its neighbor, using a
Gaussian centered at $x_i$:

$$p_{j \mid i} = \frac{\exp\!\left(-\lVert x_i - x_j \rVert^2 / 2\sigma_i^2\right)}{\sum_{k \neq i} \exp\!\left(-\lVert x_i - x_k \rVert^2 / 2\sigma_i^2\right)}$$

The bandwidth $\sigma_i$ is chosen independently for each point $i$ such
that the resulting conditional distribution $\{p_{j \mid i}\}_j$ achieves a
target perplexity — this per-point adaptivity allows the
method to use a narrower Gaussian in dense regions and a wider one in
sparse regions, so that each point's effective neighborhood size is
comparable regardless of local density.

The conditional probabilities are symmetrized to obtain a single joint
distribution over pairs:

$$p_{ij} = \frac{p_{j \mid i} + p_{i \mid j}}{2n}$$

##  Low-Dimensional Affinities

For the corresponding low-dimensional points $y_i$, define a joint
similarity using a Student-t distribution with one degree of freedom
(equivalently, a Cauchy distribution):

$$q_{ij} = \frac{\left(1 + \lVert y_i - y_j \rVert^2\right)^{-1}}{\sum_{k \neq l} \left(1 + \lVert y_k - y_l \rVert^2\right)^{-1}}$$

##  The Crowding Problem and the Role of the Student-t Distribution

The use of a heavy-tailed distribution in the low-dimensional space,
rather than a Gaussian matching Section 2's construction, is the defining
feature of t-SNE (the "t" in its name) and is a direct solution to a
specific geometric obstruction known as the crowding problem.

**The obstruction.** In a high-dimensional space, the volume available at
a given radius from a point grows rapidly with dimensionality, so a large
number of points can simultaneously be at a moderate distance from a
given point without being close to one another. When such a configuration
is mapped into two dimensions, there is no longer enough area at moderate
distance to accommodate all of these points without forcing them either
to overlap or to compress into an unnaturally small region — this
constraint is what "crowding" refers to.

**Why a heavy tail resolves it.** Under a Gaussian low-dimensional
distribution, the probability $q_{ij}$ decays very rapidly as
$\lVert y_i - y_j \rVert$ increases, so points that end up moderately far
apart in the low-dimensional map are assigned a similarity far smaller
than their corresponding $p_{ij}$, and the optimization in Section 5 is
driven to pull them closer together to reduce this mismatch — recreating
the crowding problem within the optimization itself. The Student-t
distribution's heavier tail means $q_{ij}$ decays much more slowly with
distance, so points that are moderately far apart in the low-dimensional
embedding can still receive a $q_{ij}$ comparable to their target
$p_{ij}$, without requiring them to be pulled inward. This allows
moderately dissimilar points to be placed further apart in the
low-dimensional map than a Gaussian assumption would permit, which is the
specific mechanism producing t-SNE's characteristic well-separated,
visually distinct clusters.

##  Optimization Objective

The low-dimensional coordinates $\{y_i\}$ are chosen to minimize the
Kullback–Leibler divergence between the joint distributions $P$ and $Q$:

$$\text{KL}(P \parallel Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

This is minimized by gradient descent on the positions $\{y_i\}$. Because
KL divergence is asymmetric, this objective penalizes $q_{ij}$ being small
when $p_{ij}$ is large far more heavily than the reverse — that is,
failing to represent a true high-dimensional neighbor relationship in the
low-dimensional map incurs a large penalty, while placing two points that
are not true neighbors close together in the low-dimensional map incurs a
comparatively small penalty. This asymmetry is what causes the
optimization to prioritize preserving local neighborhoods over avoiding
incidental proximity between unrelated points, consistent with the local,
visualization-oriented objective established in Section 1.

##  Perplexity

Perplexity, typically set between 5 and 50, parameterizes the target
effective neighborhood size used when solving for each $\sigma_i$ in
Section 2; it can be understood as a smooth analogue of "number of
nearest neighbors considered."

- **Low perplexity** concentrates the affinity distribution on a small
  number of very close neighbors, which can fragment genuine clusters into
  smaller sub-clusters or reveal fine local structure that may or may not
  reflect meaningful signal.
- **High perplexity** spreads the affinity distribution across a larger
  neighborhood, which tends to merge small or fine-grained clusters and
  produces a smoother, more globally averaged impression of the data.

No single perplexity value is universally correct; van der Maaten and
Hinton's original formulation reports that results are reasonably robust
across the 5–50 range, but the resulting visualization does change
appreciably within that range, which is why standard practice is to
compare multiple perplexity values rather than rely on a single run.

##  Interpretive Caveats

Because t-SNE's objective (Section 5) is defined entirely in terms of
local neighbor probabilities, several visual features of a t-SNE plot
carry no statistical meaning and should not be interpreted as reflecting
properties of the original data:

- **Apparent cluster size and density are not meaningful.** The
  optimization in Section 5 has no term constraining how much
  low-dimensional area a cluster occupies; t-SNE can expand a genuinely
  sparse region or compress a genuinely dense one, so the visual spread of
  a cluster in the plot does not indicate its true spread in the original
  space.
- **Distances between separate clusters are not reliable.** Because
  KL divergence, as noted in Section 5, penalizes a large $p_{ij}$ mapped
  to a small $q_{ij}$ far more than the reverse, the optimization has
  little incentive to preserve the relative distances between clusters
  that are already well-separated; two clusters appearing close together
  or far apart in the plot need not reflect their true relationship in
  the original space.
- **Results vary across runs.** t-SNE's optimization is stochastic, owing
  to random initialization and the non-convexity of the KL-divergence
  objective in Section 5; different runs, even with identical perplexity,
  can converge to visually different layouts unless the random seed is
  fixed.

Given these caveats, conclusions should not be drawn from a single t-SNE
plot; comparing multiple perplexity values and random seeds is standard
practice for assessing which visual features are stable and which are
artifacts of a particular run.

##  Complexity Analysis

The naive algorithm requires $O(n^2)$ time per gradient-descent iteration,
since both the high-dimensional affinities (Section 2) and low-dimensional
affinities (Section 3) involve summing over all pairs of points. This is
expensive for large $n$.

**Barnes-Hut approximation.** The default implementation in most
libraries, including `scikit-learn`, for larger datasets approximates the
contribution of distant points using a spatial tree (an application of the
Barnes-Hut algorithm originally developed for $n$-body simulation),
reducing the per-iteration cost to approximately $O(n \log n)$.

## 9. Strengths and Limitations

**Strengths.**
- Highly effective for visualizing local cluster structure in
  high-dimensional data, as established by the crowding-problem analysis
  in Section 4.
- The heavy-tailed low-dimensional distribution produces visually crisp,
  well-separated clusters, a direct consequence of the mechanism described
  in Section 4.

**Limitations.**
- Does not preserve global structure or true distances, as formalized in
  Section 7; the method's objective (Section 5) provides no basis for
  interpreting inter-cluster distances or relative cluster sizes.
- Stochastic and sensitive to the perplexity hyperparameter (Section 6),
  requiring multiple runs to assess result stability.
- Computationally expensive on very large datasets without the Barnes-Hut
  approximation (Section 8).
- Provides no simple mechanism for embedding new, out-of-sample points:
  because the embedding is the direct output of an optimization performed
  jointly over all points in a single dataset, rather than a learned
  mapping as in PCA (see `Principal Component Analysis/doc/Description.md`,
  Section 2), incorporating new points generally requires refitting on the
  combined dataset. UMAP, a related but differently motivated method not
  covered in this repository, supports out-of-sample transformation more
  naturally.


##  Code Reference

| File | Contents |
|---|---|
| `code/tsne_scratch.py` | Core t-SNE mathematics implemented from scratch in NumPy: per-point $\sigma_i$ search to achieve a target perplexity, symmetrized high-dimensional affinities, Student-t low-dimensional affinities, and gradient-descent optimization of the KL divergence, run on a small dataset given the $O(n^2)$ cost of this unoptimized version |
| `code/tsne_sklearn_demo.py` | `sklearn.manifold.TSNE` applied to the digits dataset with a perplexity sweep, illustrating how the resulting visualization changes |