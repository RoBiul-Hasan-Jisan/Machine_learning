# K-Means Clustering


##  Problem Framing

K-Means partitions a dataset into $k$ clusters by alternating between two
steps: assigning each point to its nearest of $k$ centroids, and moving
each centroid to the mean of the points currently assigned to it. Because
each cluster is represented by a single center point under Euclidean
distance, the method implicitly imposes a geometric assumption on the data
— that clusters are approximately spherical and of comparable size — which
is a direct consequence of the objective formalized in Section 2, not an
incidental limitation of the algorithm.

##  Formal Objective

K-Means seeks the partition $\{C_1, \dots, C_k\}$ that minimizes the
**within-cluster sum of squares** (WCSS, also called inertia):

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} \lVert x - \mu_i \rVert^2$$

where $\mu_i$ is the mean of the points in cluster $C_i$. This
optimization problem is NP-hard, even for the case $k = 2$; consequently,
K-Means in practice does not solve this objective exactly, but instead
applies an iterative heuristic — Lloyd's algorithm — that is guaranteed
only to converge to a local minimum of $J$.

##  Lloyd's Algorithm

**Input:** dataset $X$, number of clusters $k$.

1. Select $k$, and initialize $k$ centroids, either at random or via
   k-means++ (Section 5).
2. **Assignment step.** Assign each point to the centroid nearest to it
   under Euclidean distance.
3. **Update step.** Recompute each centroid as the mean of the points
   currently assigned to it.
4. Repeat steps 2–3 until assignments no longer change, or until a maximum
   iteration count or convergence tolerance is reached.

**Correspondence to Expectation-Maximization.** This procedure is a
specific instance of the general Expectation-Maximization pattern: the
assignment step corresponds to the E-step (computing, for fixed centroid
parameters, the most likely cluster membership for each point), and the
update step corresponds to the M-step (updating parameters — here, the
centroids — to maximize fit given the current assignments). This
correspondence is made precise in Section 7 of the Gaussian Mixture Models
theory document (`Gaussian Mixture Models/doc/Description.md`), where
K-Means is shown to be the limiting case of EM on a GMM as component
variance approaches zero.

**Guarantee and its limits.** Each of the assignment and update steps can
only decrease or maintain the value of $J$; the algorithm therefore
converges. However, because $J$ is generally non-convex in the joint space
of assignments and centroid positions, this convergence is only to a
local minimum, and the minimum reached depends on the initial centroid
positions. This is the formal justification for running Lloyd's algorithm
from multiple initializations and retaining the solution with the lowest
achieved $J$ — a standard practice (`n_init` in most implementations) that
compensates for non-convexity rather than working around an
implementation deficiency.

##  Choosing k

Because $k$ is a required input rather than a quantity the objective in
Section 2 determines on its own, it must be selected by an auxiliary
criterion:

- **Elbow method.** Plot inertia $J$ against $k$ and identify the point of
  diminishing returns. Because $J$ decreases monotonically as $k$
  increases — reaching zero at $k = n$ — this method looks for a change in
  the rate of decrease rather than a global minimum, and the resulting
  "elbow" can be ambiguous in practice.
- **Silhouette score.** Select the $k$ that maximizes the average
  silhouette coefficient across points (see
  `Clustering — Overview/doc/Description.md`, Section 5, for the formal
  definition).
- **Domain knowledge.** In practice, prior knowledge of the expected
  number of natural groupings in the data is frequently the most reliable
  guide, particularly when the elbow or silhouette criteria are ambiguous.

##  k-means++ Initialization

Random centroid initialization can converge to poor local optima or
produce degenerate (empty) clusters, since Lloyd's algorithm provides no
mechanism to recover from a poor starting configuration (Section 3).

**k-means++ procedure.** Centroids are chosen sequentially: the first is
selected uniformly at random from the data, and each subsequent centroid
is selected with probability proportional to the squared distance from
the nearest already-chosen centroid, $D(x)^2$. This biases initialization
toward centroids that are well-separated from one another.

**Why this matters.** Because the initial centroid configuration
determines which local minimum of $J$ Lloyd's algorithm converges to
(Section 3), a well-separated initialization reduces the likelihood of
converging to a poor local minimum and empirically improves both
convergence speed and final clustering quality relative to uniform random
initialization. k-means++ is the default initialization strategy in most
production implementations, including scikit-learn's.

##  Complexity Analysis

| Resource | Complexity |
|---|---|
| Time | $O(n \cdot k \cdot d \cdot t)$, where $n$ is the number of points, $d$ the dimensionality, and $t$ the number of iterations to convergence |
| Space | $O(n \cdot d + k \cdot d)$ |

This time complexity is linear in $n$, which makes K-Means substantially
more scalable to large datasets than hierarchical clustering's $O(n^2)$ or
worse cost (see `Hierarchical Clustering/doc/Description.md`, Section 5).

##  Strengths and Limitations

**Strengths.** Simple to implement, computationally efficient at scale
(Section 6), and straightforward to interpret.

**Limitations, traced to their source in the objective:**

- **Requires $k$ in advance.** The objective in Section 2 is defined only
  once $k$ is fixed; $k$ is not something the optimization determines.
- **Assumes convex, similarly-sized, isotropic clusters.** The squared-
  Euclidean term $\lVert x - \mu_i \rVert^2$ in $J$ penalizes deviation
  from a single center equally in every direction, which is equivalent to
  assuming each cluster is generated by an isotropic distribution with
  equal variance across clusters. This is why K-Means fails on non-convex
  shapes such as moons or rings, and on elongated or differently-scaled
  clusters (see `Clustering — Overview/doc/Description.md`, Section 3, and
  the runnable comparison in `01_Clustering/code/clustering_overview.py`).
- **Sensitive to initialization.** A direct consequence of $J$'s
  non-convexity (Section 3); mitigated, not eliminated, by k-means++ and
  multiple restarts.
- **Sensitive to feature scale.** Because Euclidean distance treats all
  feature dimensions identically, a feature with a larger numeric range
  dominates the distance computation regardless of its actual relevance,
  which is why standardization is a required preprocessing step rather
  than an optional one.
- **Sensitive to outliers.** Because $\mu_i$ is an arithmetic mean, a
  single distant point shifts the centroid in proportion to its distance
  from the rest of the cluster, with no bound on that influence — a
  structural property of the mean as an estimator, not specific to this
  algorithm.

##  Variants

| Variant | Modification | Purpose |
|---|---|---|
| Standard Lloyd's algorithm | Random initialization | Baseline |
| k-means++ | Distance-weighted initialization (Section 5) | Reduces sensitivity to poor local optima; default in scikit-learn |
| Mini-Batch K-Means | Centroid updates computed from small random batches rather than the full dataset each iteration | Scales to substantially larger datasets, trading some solution quality for speed |
| K-Medoids / PAM | Cluster centers constrained to be actual data points; supports arbitrary distance metrics | More robust to outliers than the arithmetic-mean centroid, at higher computational cost |
| Gaussian Mixture Models | Soft (probabilistic) cluster assignment via a generative model | Generalizes K-Means by allowing each point a probability of membership in each cluster, and by allowing non-isotropic cluster covariance (see `Gaussian Mixture Models/doc/Description.md`) |

##  Code Reference

| File | Contents |
|---|---|
| `code/kmeans_scratch.py` | K-Means implemented from scratch in NumPy, with both random and k-means++ initialization, inertia tracking, and a simple `elbow()` helper |
| `code/kmeans_sklearn_demo.py` | Equivalent problem solved with `sklearn.cluster.KMeans` and `MiniBatchKMeans`, including elbow and silhouette plots, used to validate the from-scratch implementation |